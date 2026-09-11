#!/usr/bin/env python3
"""Asynchronous OpenVLA actor-critic training on real-initialized imagination.

Real synchronized Brush frames provide every rollout/evaluation start state.
A frozen Ctrl-World imagines all subsequent observations. A temporary dense
smoke reward is evaluated once per 10 Hz OpenVLA action. Only the OpenVLA
actor-critic is optimized.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("TMPDIR", "/dev/shm")
os.environ.setdefault("RAY_DEDUP_LOGS", "0")
# CUDA visibility must be established before importing torch/deepspeed. This
# also makes direct Python invocation behave like the shell launcher.
if "--cuda-visible-devices" in sys.argv:
    _cuda_arg_index = sys.argv.index("--cuda-visible-devices")
    if _cuda_arg_index + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[_cuda_arg_index + 1]

import deepspeed
import numpy as np
import ray
import torch
import torch.distributed as distributed
import torch.nn as nn
from PIL import Image
from scipy.spatial.transform import Rotation
from torch.distributions import kl
from torch.utils.tensorboard import SummaryWriter

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
from rl.brush_real.data import (
    CAMERA_ORDER,
    NUM_FRAMES,
    NUM_HISTORY,
    PROPRIO_DIM,
    VLA_FROM_WM_ORDER,
    WM_HEIGHT,
    WM_WIDTH,
    BrushRealStartDataset,
    load_action_bounds,
    read_json,
    resolve_checkpoint,
    validate_paths,
)
from rl.brush_real.models import BrushActorCritic, BrushOpenVLAConfig
from rl.brush_real.reward import SmokeDenseReward
from rl.com_utils import find_free_port
from rl.ds_com import InferenceActorCom, TrainerActorCom
from rl.ray_debug_utils import setup_debugger
from rl.utils import process_one_obs
from rl.wm_training_utils import cosine_warmup_lr


STRIDE = NUM_FRAMES - 1


@dataclass
class Experience:
    obs: dict[str, torch.Tensor]
    action_token: np.ndarray
    advantage: float
    behaviour_logits: np.ndarray
    value_target: float


def pose_to_vla_proprio(position: np.ndarray, orientation: Rotation) -> np.ndarray:
    value = np.concatenate(
        [position, orientation.as_euler("xyz", degrees=False), [0.0, 1.0]]
    ).astype(np.float32)
    if value.shape != (PROPRIO_DIM,):
        raise AssertionError(value.shape)
    return value


def pair_compose_vla_actions(
    actions_10hz: np.ndarray,
    position: np.ndarray,
    orientation: Rotation,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, Rotation]:
    """Compose pairs of 10 Hz Euler deltas into 5 Hz Ctrl-World rotvec deltas.

    The returned proprio list has one entry after every 10 Hz low-level action,
    which is also the exact cadence at which the dense reward is called.
    """
    actions = np.asarray(actions_10hz, dtype=np.float32)
    if actions.shape != (NUM_ACTIONS_CHUNK, ACTION_DIM):
        raise ValueError(
            f"Expected {(NUM_ACTIONS_CHUNK, ACTION_DIM)} actions, got {actions.shape}"
        )
    if NUM_ACTIONS_CHUNK % 2:
        raise ValueError("The 10 Hz to 5 Hz adapter requires an even action chunk")
    current_position = np.asarray(position, dtype=np.float64).copy()
    current_orientation = orientation
    actions_5hz: list[np.ndarray] = []
    proprios_10hz: list[np.ndarray] = []
    for pair_start in range(0, NUM_ACTIONS_CHUNK, 2):
        pair_position = current_position.copy()
        pair_orientation = current_orientation
        for action in actions[pair_start : pair_start + 2]:
            current_position += np.asarray(action[:3], dtype=np.float64)
            current_orientation = (
                Rotation.from_euler("xyz", np.asarray(action[3:6], dtype=np.float64))
                * current_orientation
            )
            proprios_10hz.append(
                pose_to_vla_proprio(current_position, current_orientation)
            )
        rotation_delta = current_orientation * pair_orientation.inv()
        actions_5hz.append(
            np.concatenate(
                [
                    current_position - pair_position,
                    rotation_delta.as_rotvec(),
                    [0.0],
                ]
            ).astype(np.float32)
        )
    result = np.stack(actions_5hz)
    if result.shape != (STRIDE, ACTION_DIM):
        raise AssertionError(result.shape)
    return result, proprios_10hz, current_position, current_orientation


def tensor_to_uint8(image: torch.Tensor) -> np.ndarray:
    return (
        ((image.detach().float().cpu().clamp(-1, 1) + 1.0) * 127.5)
        .byte()
        .permute(1, 2, 0)
        .numpy()
    )


def prepare_vla_image(image: torch.Tensor) -> Image.Image:
    """Match the Brush RLDS letterbox plus training center-crop pipeline."""
    pil = Image.fromarray(tensor_to_uint8(image)).convert("RGB")
    pil.thumbnail((256, 256), resample=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (256, 256), color=(127, 127, 127))
    canvas.paste(pil, ((256 - pil.width) // 2, (256 - pil.height) // 2))
    side = int(round(256 * math.sqrt(0.9)))
    offset = (256 - side) // 2
    return canvas.crop((offset, offset, offset + side, offset + side)).resize(
        (224, 224), resample=Image.Resampling.BILINEAR
    )


def prepare_brush_obs(
    cfg: BrushOpenVLAConfig,
    processor: Any,
    observation_wm_order: torch.Tensor,
    proprio: np.ndarray,
    instruction: str,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    if tuple(observation_wm_order.shape[:2]) != (len(CAMERA_ORDER), 3):
        raise ValueError(
            f"Expected three RGB camera tensors, got {observation_wm_order.shape}"
        )
    images = [
        prepare_vla_image(observation_wm_order[index])
        for index in VLA_FROM_WM_ORDER
    ]
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    inputs = processor(prompt, images[0]).to(dtype=dtype)
    extras = [processor(prompt, image).to(dtype=dtype) for image in images[1:]]
    inputs["pixel_values"] = torch.cat(
        [inputs["pixel_values"]] + [item["pixel_values"] for item in extras], dim=1
    )
    input_ids, attention_mask, labels = process_one_obs(
        inputs["input_ids"], inputs["attention_mask"]
    )
    inputs["input_ids"] = input_ids
    inputs["attention_mask"] = attention_mask
    inputs["labels"] = labels
    inputs["proprio"] = np.asarray(proprio, dtype=np.float32)
    return inputs


@ray.remote
class StatsActor:
    def __init__(self, window_size: int):
        setup_debugger("brush_stats")
        self.window_size = int(window_size)
        self.rollout_returns = deque(maxlen=window_size)
        self.eval_returns = deque(maxlen=window_size)
        self.rollout_lengths = deque(maxlen=window_size)
        self.eval_lengths = deque(maxlen=window_size)
        self.timings = defaultdict(lambda: deque(maxlen=window_size))
        self.total_rollouts = 0
        self.total_evals = 0
        self.total_policy_samples = 0
        self.total_reward_calls = 0
        self.actor_last_active: dict[int, float] = {}

    def add_trajectory(
        self,
        trajectory_return: float,
        length: int,
        actor_id: int,
        eval_mode: bool,
        reward_calls: int,
    ) -> None:
        if eval_mode:
            self.eval_returns.append(trajectory_return)
            self.eval_lengths.append(length)
            self.total_evals += 1
        else:
            self.rollout_returns.append(trajectory_return)
            self.rollout_lengths.append(length)
            self.total_rollouts += 1
            self.total_policy_samples += length // NUM_ACTIONS_CHUNK
            self.actor_last_active[actor_id] = time.time()
        self.total_reward_calls += reward_calls

    def add_timing(self, name: str, value: float) -> None:
        self.timings[name].append(value)

    def get_stats(self) -> dict[str, Any]:
        cutoff = time.time() - 600
        return {
            "rollout_return": float(np.mean(self.rollout_returns)) if self.rollout_returns else 0.0,
            "eval_return": float(np.mean(self.eval_returns)) if self.eval_returns else 0.0,
            "rollout_length": float(np.mean(self.rollout_lengths)) if self.rollout_lengths else 0.0,
            "eval_length": float(np.mean(self.eval_lengths)) if self.eval_lengths else 0.0,
            "total_rollouts": self.total_rollouts,
            "total_evals": self.total_evals,
            "total_policy_samples": self.total_policy_samples,
            "total_reward_calls": self.total_reward_calls,
            "active_rollout_workers": sum(
                timestamp >= cutoff for timestamp in self.actor_last_active.values()
            ),
            "timings": {
                key: float(np.mean(values)) if values else 0.0
                for key, values in self.timings.items()
            },
        }


@ray.remote
class ReplayBufferActor:
    def __init__(self, capacity: int):
        setup_debugger("brush_replay")
        self.buffer = deque(maxlen=int(capacity))

    def add_batch(self, batch: list[Experience]) -> None:
        self.buffer.extend(batch)

    def size(self) -> int:
        return len(self.buffer)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return (
            [item.obs for item in batch],
            np.stack([item.action_token for item in batch]),
            np.asarray([item.advantage for item in batch], dtype=np.float32),
            np.stack([item.behaviour_logits for item in batch]),
            np.asarray([item.value_target for item in batch], dtype=np.float32),
        )


@ray.remote(num_gpus=1)
class PolicyInferenceActor(InferenceActorCom):
    def __init__(
        self,
        actor_id: int,
        cfg: BrushOpenVLAConfig,
        stats_actor,
        dtype: torch.dtype,
        inference_batch: int,
        inference_timeout_ms: int,
    ) -> None:
        super().__init__()
        setup_debugger("brush_policy_inference", actor_id)
        self.actor_id = actor_id
        self.cfg = cfg
        self.stats_actor = stats_actor
        self.model = BrushActorCritic(cfg, torch_dtype=dtype).cuda().eval()
        self.batch_size = int(inference_batch)
        self.timeout = inference_timeout_ms / 1000.0
        self.requests: list[tuple[Any, bool]] = []
        self.promises: list[asyncio.Future] = []
        self.last_process = time.time()
        self._task = asyncio.get_event_loop().create_task(self._loop())

    async def request(self, inputs: dict[str, torch.Tensor], deterministic: bool = False):
        future = asyncio.get_event_loop().create_future()
        self.requests.append((inputs, deterministic))
        self.promises.append(future)
        return await future

    async def _loop(self) -> None:
        while True:
            ready = self.requests and (
                len(self.requests) >= self.batch_size
                or time.time() - self.last_process >= self.timeout
            )
            if not ready:
                await asyncio.sleep(0.0005)
                continue
            requests, promises = self.requests, self.promises
            self.requests, self.promises = [], []
            self.last_process = time.time()
            started = time.time()
            try:
                batch = self.model.prepare_inputs_batch([item[0] for item in requests])
                with torch.inference_mode():
                    logits, values = self.model(batch)
                    _, tokens, normalized = self.model.post_process(
                        logits, [item[1] for item in requests]
                    )
                tokens = tokens.view(-1, NUM_ACTIONS_CHUNK, ACTION_DIM).cpu().numpy()
                logits_np = logits.view(
                    -1, NUM_ACTIONS_CHUNK, ACTION_DIM, logits.shape[-1]
                ).float().cpu().numpy()
                values_np = values.float().cpu().numpy()
                for index, promise in enumerate(promises):
                    actions = self.model.vla._unnormalize_actions(
                        normalized[index], self.cfg.unnorm_key
                    ).astype(np.float32)
                    # The seventh component is a dummy gripper for this fixed tool.
                    actions[:, 6] = 1.0
                    promise.set_result(
                        (normalized[index], actions, tokens[index], logits_np[index], values_np[index])
                    )
                self.stats_actor.add_timing.remote(
                    "policy_inference_seconds", time.time() - started
                )
            except Exception as exc:
                for promise in promises:
                    if not promise.done():
                        promise.set_exception(exc)
                raise

    def get_model_keys(self) -> dict[str, float]:
        return {key: float(value.abs().sum()) for key, value in self.model.state_dict().items()}

    def forward_test(self) -> None:
        return None


@ray.remote(num_gpus=1)
class FrozenCtrlWorldActor:
    def __init__(
        self,
        actor_id: int,
        stats_actor,
        args: argparse.Namespace,
        checkpoint: str,
        action_stat_path: str,
    ) -> None:
        setup_debugger("brush_ctrl_world", actor_id)
        self.actor_id = actor_id
        self.stats_actor = stats_actor
        ctrl_root = str(Path(args.ctrl_world_root).expanduser().resolve())
        if ctrl_root not in sys.path:
            sys.path.insert(0, ctrl_root)
        from ctrl_world.config import wm_args
        from ctrl_world.models.ctrl_world import CrtlWorld
        from rl.ctrl_world.ctrl_world_env_batch import CtrlWorldEnvBatch

        try:
            cfg = wm_args(task_type="replay")
        except TypeError:
            cfg = wm_args()
        cfg.svd_model_path = args.svd_model_path
        cfg.clip_model_path = args.clip_model_path
        cfg.num_cams = len(CAMERA_ORDER)
        cfg.num_history = NUM_HISTORY
        cfg.num_frames = NUM_FRAMES
        cfg.action_dim = ACTION_DIM
        cfg.height = WM_HEIGHT
        cfg.width = WM_WIDTH
        cfg.fps = 7
        cfg.text_cond = True
        cfg.frame_level_cond = True
        cfg.his_cond_zero = False

        print(f"FrozenCtrlWorldActor {actor_id}: loading {checkpoint}")
        model = CrtlWorld(cfg)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        if isinstance(state, dict) and ("model" in state or "state_dict" in state):
            state = state.get("model", state.get("state_dict"))
        state = {key.removeprefix("module."): value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
        del state
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model = model.cuda().to(torch.bfloat16).eval()
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("Ctrl-World freeze invariant failed")
        low, high = load_action_bounds(action_stat_path)

        @dataclass
        class EnvConfig:
            horizon: int = 220

        self.env = CtrlWorldEnvBatch(
            ctrl_world_model=model,
            cfg=EnvConfig(),
            torch_dtype=torch.bfloat16,
            num_cams=len(CAMERA_ORDER),
            target_height=WM_HEIGHT,
            target_width=WM_WIDTH,
            num_frames_pred=NUM_FRAMES,
            num_inference_steps=args.num_inference_steps,
            condition_low=low,
            condition_high=high,
        )
        self.model = model
        self.device = self.env.device
        self.batch_size = int(args.inference_batch)
        self.timeout = args.inference_timeout_ms / 1000.0
        self.requests = []
        self.promises = []
        self.last_process = time.time()
        self._task = asyncio.get_event_loop().create_task(self._loop())

    def get_freeze_status(self) -> dict[str, int | bool]:
        return {
            "frozen": not any(p.requires_grad for p in self.model.parameters()),
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }

    def init_latent_state(self, observations: torch.Tensor, deterministic: bool = False):
        with torch.inference_mode():
            history, current = self.env.init_latent_state(
                observations.unsqueeze(0).to(self.device), deterministic=deterministic
            )
        return history[0].cpu(), current[0].cpu()

    async def request(
        self,
        current: torch.Tensor,
        history: torch.Tensor,
        action_condition: torch.Tensor,
        instruction: str,
        seed: int,
    ):
        future = asyncio.get_event_loop().create_future()
        self.requests.append((current, history, action_condition, instruction, seed))
        self.promises.append(future)
        return await future

    async def _loop(self) -> None:
        while True:
            ready = self.requests and (
                len(self.requests) >= self.batch_size
                or time.time() - self.last_process >= self.timeout
            )
            if not ready:
                await asyncio.sleep(0.0005)
                continue
            requests, promises = self.requests, self.promises
            self.requests, self.promises = [], []
            self.last_process = time.time()
            started = time.time()
            try:
                current = torch.stack([item[0] for item in requests]).to(self.device)
                history = torch.stack([item[1] for item in requests]).to(self.device)
                actions = torch.stack([item[2] for item in requests]).to(self.device)
                instructions = [item[3] for item in requests]
                generators = []
                for item in requests:
                    generator = torch.Generator(device=self.device)
                    generator.manual_seed(int(item[4]))
                    generators.append(generator)
                with torch.inference_mode():
                    observations, latents = self.env.predict_chunk_stateless(
                        current,
                        history,
                        actions,
                        instructions,
                        output_size=(WM_HEIGHT, WM_WIDTH),
                        generator=generators,
                    )
                observations = observations.float().cpu()
                latents = latents.cpu()
                for index, promise in enumerate(promises):
                    promise.set_result((observations[index], latents[index]))
                self.stats_actor.add_timing.remote(
                    "ctrl_world_inference_seconds", time.time() - started
                )
            except Exception as exc:
                for promise in promises:
                    if not promise.done():
                        promise.set_exception(exc)
                raise


class BrushImaginationWorker:
    def __init__(
        self,
        policy,
        world,
        stats_actor,
        cfg: BrushOpenVLAConfig,
        args: argparse.Namespace,
        worker_id: int,
        split: str,
        eval_mode: bool,
        replay=None,
    ) -> None:
        self.policy = policy
        self.world = world
        self.stats = stats_actor
        self.cfg = cfg
        self.args = args
        self.worker_id = worker_id
        self.eval_mode = eval_mode
        self.replay = replay
        self.dtype = torch.bfloat16 if args.use_bf16 else torch.float32
        self.dataset = BrushRealStartDataset(
            args.real_data_root,
            args.wm_dataset_root,
            args.wm_meta_root,
            args.wm_dataset_name,
            split,
        )
        self.reward = SmokeDenseReward(args.smoke_reward_value)
        self.rng = random.Random(args.seed + 1009 * worker_id + (1_000_000 if eval_mode else 0))
        self.sample_index = worker_id

    def _next_start(self):
        if self.eval_mode:
            start = self.dataset.sample(index=self.sample_index)
            self.sample_index += max(1, self.args.num_eval_workers)
            return start
        return self.dataset.sample(rng=self.rng)

    async def imagine_one(self) -> tuple[list[tuple], float, int, float]:
        start = self._next_start()
        current_obs = start.observations[-1].clone()
        action_window = torch.from_numpy(start.frame_actions).float()
        history, current = await self.world.init_latent_state.remote(
            start.observations, self.eval_mode
        )
        position = start.pose_rotvec[:3].astype(np.float64).copy()
        orientation = Rotation.from_rotvec(start.pose_rotvec[3:6].astype(np.float64))
        segment = []
        trajectory_return = 0.0
        reward_calls = 0

        for policy_step in range(self.args.imagine_horizon // NUM_ACTIONS_CHUNK):
            proprio = pose_to_vla_proprio(position, orientation)
            inputs = prepare_brush_obs(
                self.cfg,
                self.policy_processor,
                current_obs,
                proprio,
                self.args.vla_instruction,
                self.dtype,
            )
            _, actions_10hz, tokens, logits, value = await self.policy.request.remote(
                inputs, deterministic=self.eval_mode
            )
            actions_5hz, proprios, position, orientation = pair_compose_vla_actions(
                actions_10hz, position, orientation
            )
            condition = torch.cat([action_window, torch.from_numpy(actions_5hz)], dim=0)
            if condition.shape != (NUM_HISTORY + NUM_FRAMES, ACTION_DIM):
                raise AssertionError(condition.shape)
            future_obs, future_latents = await self.world.request.remote(
                current,
                history,
                condition,
                self.args.wm_instruction or start.wm_instruction,
                self.rng.randrange(0, 2**63 - 1),
            )

            chunk_reward = 0.0
            for pair_index in range(STRIDE):
                for within_pair in range(2):
                    low_index = pair_index * 2 + within_pair
                    reward = self.reward(
                        observation=future_obs[pair_index],
                        action=actions_10hz[low_index],
                        proprio=proprios[low_index],
                        low_level_step=policy_step * NUM_ACTIONS_CHUNK + low_index,
                    )
                    chunk_reward += reward * self.args.reward_scale
                    reward_calls += 1
                history = torch.cat([history[1:], current.unsqueeze(0)], dim=0)
                current = future_latents[pair_index]
                action_window = torch.cat(
                    [action_window[1:], torch.from_numpy(actions_5hz[pair_index : pair_index + 1])],
                    dim=0,
                )
            current_obs = future_obs[-1]
            trajectory_return += chunk_reward
            if not self.eval_mode:
                segment.append((inputs, tokens, chunk_reward, logits, float(value)))

        if reward_calls != self.args.imagine_horizon:
            raise AssertionError(
                f"Dense reward called {reward_calls} times for horizon {self.args.imagine_horizon}"
            )
        bootstrap = 0.0
        if not self.eval_mode:
            final_inputs = prepare_brush_obs(
                self.cfg,
                self.policy_processor,
                current_obs,
                pose_to_vla_proprio(position, orientation),
                self.args.vla_instruction,
                self.dtype,
            )
            _, _, _, _, bootstrap_value = await self.policy.request.remote(
                final_inputs, deterministic=True
            )
            bootstrap = float(bootstrap_value)
        return segment, trajectory_return, reward_calls, bootstrap

    @property
    def policy_processor(self):
        # Processor objects do not need to be copied from the GPU actor. Loading
        # locally keeps image/token preprocessing on each CPU worker.
        if not hasattr(self, "_processor"):
            from experiments.robot.sole_utils import get_processor

            self._processor = get_processor(self.cfg)
        return self._processor

    def _finish_segment(self, segment: list[tuple], bootstrap: float) -> None:
        gae = 0.0
        advantages: list[float] = []
        targets: list[float] = []
        for index in reversed(range(len(segment))):
            value = segment[index][4]
            next_value = bootstrap if index == len(segment) - 1 else segment[index + 1][4]
            delta = segment[index][2] + self.args.gamma * next_value - value
            gae = delta + self.args.gamma * self.args.lambda_ * gae
            advantages.append(gae)
            targets.append(gae + value)
        advantages.reverse()
        targets.reverse()
        batch = [
            Experience(
                obs=item[0],
                action_token=np.asarray(item[1], dtype=np.int64),
                advantage=float(advantages[index]),
                behaviour_logits=np.asarray(item[3], dtype=np.float32),
                value_target=float(targets[index]),
            )
            for index, item in enumerate(segment)
        ]
        self.replay.add_batch.remote(batch)

    async def run_loop(self) -> None:
        while True:
            started = time.time()
            segment, trajectory_return, reward_calls, bootstrap = await self.imagine_one()
            if not self.eval_mode:
                # The configured horizon is a truncation, not a terminal state,
                # so bootstrap from the final imagined observation.
                self._finish_segment(segment, bootstrap)
            self.stats.add_trajectory.remote(
                trajectory_return,
                self.args.imagine_horizon,
                self.worker_id,
                self.eval_mode,
                reward_calls,
            )
            self.stats.add_timing.remote(
                "eval_trajectory_seconds" if self.eval_mode else "rollout_trajectory_seconds",
                time.time() - started,
            )


@ray.remote
class RolloutWorkerActor(BrushImaginationWorker):
    def __init__(self, policy, world, replay, stats, cfg, args, worker_id):
        setup_debugger("brush_rollout", worker_id)
        super().__init__(
            policy, world, stats, cfg, args, worker_id, args.train_split, False, replay
        )

    async def run(self):
        await self.run_loop()


@ray.remote
class EvaluationWorkerActor(BrushImaginationWorker):
    def __init__(self, policy, world, stats, cfg, args, worker_id):
        setup_debugger("brush_eval", worker_id)
        super().__init__(
            policy, world, stats, cfg, args, worker_id, args.eval_split, True, None
        )

    async def run(self):
        await self.run_loop()


@ray.remote(num_gpus=1)
class TrainerActor(TrainerActorCom):
    def __init__(self, rank, world_size, replay, cfg, args, dtype):
        super().__init__()
        setup_debugger("brush_trainer", rank)
        self.rank = rank
        self.world_size = world_size
        self.replay = replay
        self.cfg = cfg
        self.args = args
        self.dtype = dtype
        self.batch_size = args.train_batch_size
        self.accumulation = args.accumulation_steps
        self.super_batch = self.batch_size * self.accumulation
        self.model = None
        self.base_model = None
        self.optimizer = None
        self.next_batch = None
        self.global_step = 0

    def get_node_ip(self):
        return ray.util.get_node_ip_address()

    def get_model_keys(self):
        if self.model is None:
            return {}
        module = self.model.module if hasattr(self.model, "module") else self.model
        return {key: float(value.abs().sum()) for key, value in module.state_dict().items()}

    def setup_deepspeed_group(self, master_addr: str, master_port: int):
        os.environ.update(
            RANK=str(self.rank),
            WORLD_SIZE=str(self.world_size),
            MASTER_ADDR=master_addr,
            MASTER_PORT=str(master_port),
            LOCAL_RANK="0",
        )
        deepspeed.init_distributed(dist_backend="nccl")
        self.base_model = BrushActorCritic(self.cfg, torch_dtype=self.dtype)
        groups = self.base_model.get_parameter_groups()
        optimizer_groups = [
            {
                "params": group["params"],
                "name": group["name"],
                "lr": self.args.policy_lr if group["name"] == "policy" else self.args.value_lr,
            }
            for group in groups
        ]
        config = {
            "train_micro_batch_size_per_gpu": self.batch_size,
            "gradient_accumulation_steps": self.accumulation,
            "optimizer": {"type": "AdamW", "params": {}},
            "bf16": {"enabled": self.args.use_bf16},
            "zero_optimization": {
                "stage": 2,
                "allgather_partitions": True,
                "reduce_scatter": True,
                "overlap_comm": True,
                "contiguous_gradients": True,
            },
            "gradient_clipping": 1.0,
        }
        self.model, self.optimizer, _, _ = deepspeed.initialize(
            model=self.base_model,
            config=config,
            model_parameters=optimizer_groups,
        )
        self._fetch_task = asyncio.get_event_loop().create_task(self._fetch_loop())

    async def _fetch_loop(self):
        while True:
            try:
                if self.next_batch is not None:
                    await asyncio.sleep(0.05)
                    continue
                while await self.replay.size.remote() < self.super_batch:
                    await asyncio.sleep(1.0)
                started = time.time()
                obs, actions, advantages, old_logits, targets = await self.replay.sample.remote(
                    self.super_batch
                )
                sample_seconds = time.time() - started
                started = time.time()
                inputs = self.base_model.prepare_inputs_batch(obs)
                device = next(self.model.parameters()).device
                self.next_batch = {
                    "inputs": inputs,
                    "actions": torch.as_tensor(actions, dtype=torch.long, device=device),
                    "advantages": torch.as_tensor(advantages, dtype=torch.float32, device=device),
                    "old_logits": torch.as_tensor(old_logits, dtype=torch.float32, device=device),
                    "targets": torch.as_tensor(targets, dtype=torch.float32, device=device),
                    "sample_seconds": sample_seconds,
                    "prepare_seconds": time.time() - started,
                }
            except Exception as exc:
                print(f"Trainer {self.rank} fetch failed: {exc}", flush=True)
                await asyncio.sleep(2.0)

    async def run_training_epoch(self):
        while self.next_batch is None:
            await asyncio.sleep(0.1)
        batch, self.next_batch = self.next_batch, None
        value_lr = cosine_warmup_lr(
            self.global_step,
            self.args.value_lr,
            self.args.value_warmup_steps,
            self.args.train_iters,
        )
        policy_lr = cosine_warmup_lr(
            self.global_step,
            self.args.policy_lr,
            self.args.policy_warmup_steps,
            self.args.train_iters,
            start_step=self.args.policy_train_start_step,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = policy_lr if group["name"] == "policy" else value_lr

        advantages = batch["advantages"]
        moments = torch.stack(
            [advantages.sum(), advantages.square().sum(), advantages.new_tensor(advantages.numel())]
        )
        distributed.all_reduce(moments, op=distributed.ReduceOp.SUM)
        mean = moments[0] / moments[2].clamp_min(1)
        variance = (moments[1] / moments[2].clamp_min(1) - mean.square()).clamp_min(1e-12)
        std = variance.sqrt()

        losses = defaultdict(list)
        started = time.time()
        for update in range(self.accumulation):
            begin = update * self.batch_size
            end = begin + self.batch_size
            mini_inputs = {key: value[begin:end] for key, value in batch["inputs"].items()}
            action_logits, values = self.model(mini_inputs)
            action_logits = action_logits.view(
                -1, NUM_ACTIONS_CHUNK, ACTION_DIM, action_logits.shape[-1]
            )
            values = values.float()
            value_loss = self.args.vf_coef * (
                values - batch["targets"][begin:end]
            ).square().mean()
            if self.global_step < self.args.policy_train_start_step:
                policy_loss = value_loss.new_zeros(())
                entropy = value_loss.new_zeros(())
                entropy_loss = value_loss.new_zeros(())
                kl_loss = value_loss.new_zeros(())
                kl_value = 0.0
            else:
                distribution = torch.distributions.Categorical(logits=action_logits)
                old_distribution = torch.distributions.Categorical(
                    logits=batch["old_logits"][begin:end]
                )
                log_ratio = distribution.log_prob(batch["actions"][begin:end]) - old_distribution.log_prob(
                    batch["actions"][begin:end]
                )
                ratio = log_ratio.exp()
                normalized_advantage = (
                    (batch["advantages"][begin:end] - mean) / (std + 1e-8)
                ).unsqueeze(-1).unsqueeze(-1)
                surrogate = ratio * normalized_advantage
                if self.args.clip_mode == "ppo":
                    clipped = ratio.clamp(
                        1.0 - self.args.clip_eps, 1.0 + self.args.clip_eps
                    ) * normalized_advantage
                    policy_loss = -torch.minimum(surrogate, clipped).mean()
                elif self.args.clip_mode == "gipo":
                    weight = torch.exp(
                        -0.5 * torch.log(ratio.clamp_min(1e-9).detach()).square()
                    )
                    policy_loss = -(surrogate * weight).mean()
                elif self.args.clip_mode == "sapo":
                    tau = torch.where(
                        normalized_advantage > 0,
                        torch.ones_like(normalized_advantage),
                        torch.full_like(normalized_advantage, 2.0),
                    )
                    gate = torch.sigmoid(tau * (ratio.clamp(1e-6, 1e6) - 1.0)) * (4.0 / tau)
                    policy_loss = -(gate * normalized_advantage).mean()
                else:
                    raise ValueError(self.args.clip_mode)
                divergence = kl.kl_divergence(old_distribution, distribution)
                kl_value = float(divergence.mean())
                kl_loss = self.args.kl_coef * divergence.mean()
                entropy = distribution.entropy().mean()
                entropy_loss = -self.args.ent_coef * entropy
            total = policy_loss + value_loss + entropy_loss + kl_loss
            self.model.backward(total)
            self.model.step()
            if self.model.is_gradient_accumulation_boundary():
                self.global_step += 1
            for name, value in (
                ("total", total),
                ("policy", policy_loss),
                ("value", value_loss),
                ("entropy_loss", entropy_loss),
                ("kl_loss", kl_loss),
                ("entropy", entropy),
            ):
                losses[name].append(float(value.detach()))
            losses["kl"].append(kl_value)
        return {
            "global_step": self.global_step,
            "losses": {key: float(np.mean(value)) for key, value in losses.items()},
            "lrs": {"policy": policy_lr, "value": value_lr},
            "performance": {
                "sample_seconds": batch["sample_seconds"],
                "prepare_seconds": batch["prepare_seconds"],
                "train_seconds": time.time() - started,
            },
        }

    async def save_agent(self, checkpoint_dir: str, step: int):
        self.base_model.save_model(checkpoint_dir, epoch=step)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-visible-devices", default="4,5,6")
    parser.add_argument("--use-bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", action="store_false", dest="use_bf16")
    parser.add_argument("--real-data-root", default="/mnt/data/lcx/data/brush/data_collect")
    parser.add_argument(
        "--openvla-checkpoint",
        default=str(REPO_ROOT / "runs/brush_openvla_discrete/openvla-7b+brush_realworld+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--discrete_acts--3_cams--proprio--10hz--26000_chkpt"),
    )
    parser.add_argument("--checkpoint2", default=None)
    parser.add_argument("--unnorm-key", default="brush_realworld")
    parser.add_argument("--ctrl-world-root", default="/mnt/data/lcx3/Ctrl-World")
    parser.add_argument(
        "--world-checkpoint",
        default="/mnt/data/lcx3/Ctrl-World/model_ckpt/brush_pen_3cam_delta_5hz_20260901/exp4_bs32_4gpu_lr1e5_warmup150_cosine_fixedval_continue_seed20260902_20260909_161127",
    )
    parser.add_argument(
        "--wm-dataset-root",
        default="/mnt/data/lcx3/Ctrl-World/output_brush_pen_3cam_delta_5hz_20260901/dataset",
    )
    parser.add_argument(
        "--wm-meta-root",
        default="/mnt/data/lcx3/Ctrl-World/output_brush_pen_3cam_delta_5hz_20260901/meta",
    )
    parser.add_argument("--wm-dataset-name", default="brush_pen_3cam_delta_5hz_20260901")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--svd-model-path", default="/mnt/data/lcx/models/stabilityai/stable-video-diffusion-img2vid")
    parser.add_argument("--clip-model-path", default="/mnt/data/lcx/models/openai/clip-vit-base-patch32")
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--vla-instruction", default="brush along the outlined region on the paper")
    parser.add_argument("--wm-instruction", default="lower the brush tip onto the paper")
    parser.add_argument("--num-trainer-gpus", type=int, default=1)
    parser.add_argument("--num-inference-actors", type=int, default=1)
    parser.add_argument("--num-ctrl-inference-actors", type=int, default=1)
    parser.add_argument("--num-rollout-workers", type=int, default=10)
    parser.add_argument("--num-eval-workers", type=int, default=10)
    parser.add_argument("--inference-batch", type=int, default=8)
    parser.add_argument("--inference-timeout-ms", type=int, default=300)
    parser.add_argument("--replay-capacity", type=int, default=10_000)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=72)
    parser.add_argument("--train-iters", type=int, default=30_000)
    parser.add_argument("--imagine-horizon", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda", type=float, default=0.95, dest="lambda_")
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--kl-coef", type=float, default=0.1)
    parser.add_argument("--clip-mode", choices=("ppo", "sapo", "gipo"), default="gipo")
    parser.add_argument("--value-lr", type=float, default=1e-4)
    parser.add_argument("--policy-lr", type=float, default=1e-5)
    parser.add_argument("--value-warmup-steps", type=int, default=500)
    parser.add_argument("--policy-warmup-steps", type=int, default=500)
    parser.add_argument("--policy-train-start-step", type=int, default=0)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--smoke-reward-value", type=float, default=1.0)
    parser.add_argument("--moving-avg-window", type=int, default=1000)
    parser.add_argument("--log-interval-seconds", type=float, default=10.0)
    parser.add_argument("--ckpt-dir", default=str(REPO_ROOT / "runs/brush_real_async/checkpoints"))
    parser.add_argument("--ckpt-every-steps", type=int, default=500)
    parser.add_argument("--exp-name", default="brush_real_frozen_ctrl_world_smoke_reward")
    parser.add_argument("--object-store-memory-gb", type=float, default=64.0)
    parser.add_argument("--ray-spill-dir", default="/mnt/data/lcx3/ray_spill")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-wait", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if args.debug:
        os.environ["RAY_DEBUG"] = "1"
    if args.debug_wait:
        os.environ["RAY_DEBUG_WAIT"] = "1"
    if args.imagine_horizon <= 0 or args.imagine_horizon % NUM_ACTIONS_CHUNK:
        raise ValueError(
            f"--imagine-horizon must be a positive multiple of {NUM_ACTIONS_CHUNK}"
        )
    if args.num_inference_steps <= 0:
        raise ValueError("--num-inference-steps must be positive")
    actor_counts = {
        "num_trainer_gpus": args.num_trainer_gpus,
        "num_inference_actors": args.num_inference_actors,
        "num_ctrl_inference_actors": args.num_ctrl_inference_actors,
        "num_rollout_workers": args.num_rollout_workers,
        "num_eval_workers": args.num_eval_workers,
    }
    if any(value <= 0 for value in actor_counts.values()):
        raise ValueError(f"All actor/worker counts must be positive: {actor_counts}")
    if args.train_batch_size <= 0 or args.accumulation_steps <= 0:
        raise ValueError("Training batch size and accumulation must be positive")
    if not math.isfinite(args.smoke_reward_value):
        raise ValueError("--smoke-reward-value must be finite")
    visible = [item for item in args.cuda_visible_devices.split(",") if item.strip()]
    required = (
        args.num_trainer_gpus
        + args.num_inference_actors
        + args.num_ctrl_inference_actors
    )
    if not args.preflight_only and len(visible) < required:
        raise ValueError(f"Need {required} GPU actors, only {len(visible)} GPUs are visible")
    if args.train_batch_size * args.accumulation_steps >= args.replay_capacity:
        raise ValueError("Policy super-batch must be smaller than replay capacity")
    if args.policy_warmup_steps >= args.train_iters or args.value_warmup_steps >= args.train_iters:
        raise ValueError("Warmup steps must be smaller than train-iters")

    args.world_checkpoint = str(resolve_checkpoint(args.world_checkpoint))
    openvla = Path(args.openvla_checkpoint).expanduser().resolve()
    adapter_config = openvla / "lora_adapter/adapter_config.json"
    adapter = openvla / "lora_adapter/adapter_model.safetensors"
    stats = openvla / "dataset_statistics.json"
    validate_paths(
        [
            args.real_data_root,
            args.ctrl_world_root,
            args.world_checkpoint,
            args.wm_dataset_root,
            args.wm_meta_root,
            args.svd_model_path,
            args.clip_model_path,
            adapter_config,
            adapter,
            stats,
        ]
    )
    adapter_metadata = read_json(adapter_config)
    base_checkpoint = Path(adapter_metadata["base_model_name_or_path"]).expanduser().resolve()
    validate_paths([base_checkpoint / "config.json"])
    args.openvla_checkpoint = str(openvla)
    action_stat_path = (
        Path(args.wm_meta_root) / args.wm_dataset_name / "stat.json"
    ).resolve()
    load_action_bounds(action_stat_path)
    train_data = BrushRealStartDataset(
        args.real_data_root,
        args.wm_dataset_root,
        args.wm_meta_root,
        args.wm_dataset_name,
        args.train_split,
    )
    eval_data = BrushRealStartDataset(
        args.real_data_root,
        args.wm_dataset_root,
        args.wm_meta_root,
        args.wm_dataset_name,
        args.eval_split,
    )
    sample = train_data.sample(index=0)
    report = {
        "openvla_sft_checkpoint": args.openvla_checkpoint,
        "openvla_base_checkpoint": str(base_checkpoint),
        "ctrl_world_checkpoint": args.world_checkpoint,
        "ctrl_world_frozen": True,
        "vae_decoder_checkpoint_loaded": False,
        "learned_reward_model": False,
        "dense_reward": "SmokeDenseReward",
        "reward_calls_per_trajectory": args.imagine_horizon,
        "expected_smoke_return": (
            args.imagine_horizon * args.smoke_reward_value * args.reward_scale
        ),
        "train_data": train_data.describe(),
        "eval_data": eval_data.describe(),
        "sample": {
            "episode_id": sample.episode_id,
            "start_index": sample.start_index,
            "observations": list(sample.observations.shape),
            "actions": list(sample.frame_actions.shape),
            "pose": list(sample.pose_rotvec.shape),
        },
    }
    return str(base_checkpoint), str(action_stat_path), report


def build_cfg(args, base_checkpoint: str) -> BrushOpenVLAConfig:
    return BrushOpenVLAConfig(
        pretrained_checkpoint=base_checkpoint,
        sft_checkpoint=args.openvla_checkpoint,
        checkpoint2=args.checkpoint2,
        num_images_in_input=3,
        use_proprio=True,
        center_crop=True,
        use_lora=True,
        lora_rank=32,
        lora_dropout=0.0,
        unnorm_key=args.unnorm_key,
    )


def main(args: argparse.Namespace) -> None:
    base_checkpoint, action_stat_path, report = validate_args(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.preflight_only:
        return
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if args.use_bf16 else torch.float32
    cfg = build_cfg(args, base_checkpoint)

    ray.init(
        ignore_reinit_error=True,
        _temp_dir="/dev/shm",
        object_store_memory=int(args.object_store_memory_gb * 1024**3),
        object_spilling_directory=args.ray_spill_dir,
    )
    log_dir = REPO_ROOT / "runs/brush_real_async" / f"{int(time.time())}_{args.exp_name}"
    writer = SummaryWriter(str(log_dir))
    stats = StatsActor.remote(args.moving_avg_window)
    replay = [ReplayBufferActor.remote(args.replay_capacity) for _ in range(args.num_trainer_gpus)]
    trainers = [
        TrainerActor.remote(rank, args.num_trainer_gpus, replay[rank], cfg, args, dtype)
        for rank in range(args.num_trainer_gpus)
    ]
    policies = [
        PolicyInferenceActor.remote(
            index, cfg, stats, dtype, args.inference_batch, args.inference_timeout_ms
        )
        for index in range(args.num_inference_actors)
    ]
    worlds = [
        FrozenCtrlWorldActor.remote(index, stats, args, args.world_checkpoint, action_stat_path)
        for index in range(args.num_ctrl_inference_actors)
    ]
    rollout_workers = [
        RolloutWorkerActor.remote(
            policies[index % len(policies)],
            worlds[index % len(worlds)],
            replay[index % len(replay)],
            stats,
            cfg,
            args,
            index,
        )
        for index in range(args.num_rollout_workers)
    ]
    eval_workers = [
        EvaluationWorkerActor.remote(
            policies[index % len(policies)],
            worlds[index % len(worlds)],
            stats,
            cfg,
            args,
            index,
        )
        for index in range(args.num_eval_workers)
    ]

    train_port = find_free_port()
    trainer_addr = ray.get(trainers[0].get_node_ip.remote())
    ray.get(
        [trainer.setup_deepspeed_group.remote(trainer_addr, train_port) for trainer in trainers]
    )
    freeze_status = ray.get([world.get_freeze_status.remote() for world in worlds])
    if not all(item["frozen"] and item["trainable"] == 0 for item in freeze_status):
        raise RuntimeError(f"Ctrl-World is not frozen: {freeze_status}")

    broadcast_port = find_free_port()
    participants = [trainers[0]] + policies
    ray.get(
        [
            actor.setup_broadcast_group.remote(
                trainer_addr,
                broadcast_port,
                "broadcast_actor",
                len(participants),
                rank,
            )
            for rank, actor in enumerate(participants)
        ]
    )
    ray.get(
        [trainers[0].broadcast_weights.remote("broadcast_actor")]
        + [policy.receive_and_update_weights.remote("broadcast_actor") for policy in policies]
    )

    for worker in rollout_workers:
        worker.run.remote()
    for worker in eval_workers:
        worker.run.remote()

    minimum = args.train_batch_size * args.accumulation_steps
    while True:
        sizes = ray.get([buffer.size.remote() for buffer in replay])
        if all(size >= minimum for size in sizes):
            break
        print(f"Waiting for imagined replay: {sizes}, target={minimum}", flush=True)
        time.sleep(5)

    global_step = 0
    last_log = time.time()
    start_time = time.time()
    while global_step < args.train_iters:
        started = time.time()
        results = ray.get([trainer.run_training_epoch.remote() for trainer in trainers])
        global_step = int(results[0]["global_step"])
        sync_started = time.time()
        ray.get(
            [trainers[0].broadcast_weights.remote("broadcast_actor")]
            + [policy.receive_and_update_weights.remote("broadcast_actor") for policy in policies]
        )
        sync_seconds = time.time() - sync_started
        if global_step and global_step % args.ckpt_every_steps == 0:
            ray.get(trainers[0].save_agent.remote(args.ckpt_dir, global_step))

        if time.time() - last_log >= args.log_interval_seconds:
            stats_value = ray.get(stats.get_stats.remote())
            losses = results[0]["losses"]
            lrs = results[0]["lrs"]
            performance = results[0]["performance"]
            print(
                f"step {global_step}/{args.train_iters} | "
                f"rollout return {stats_value['rollout_return']:.3f} | "
                f"eval return {stats_value['eval_return']:.3f} | "
                f"policy/value {losses['policy']:.5f}/{losses['value']:.5f} | "
                f"elapsed {time.time() - start_time:.1f}s",
                flush=True,
            )
            for name, value in losses.items():
                writer.add_scalar(f"Loss/{name}", value, global_step)
            writer.add_scalar("Train/Learning_Rate/Policy", lrs["policy"], global_step)
            writer.add_scalar("Train/Learning_Rate/Value", lrs["value"], global_step)
            writer.add_scalar("Rollout/Average_Return", stats_value["rollout_return"], global_step)
            writer.add_scalar("Eval/Average_Return", stats_value["eval_return"], global_step)
            writer.add_scalar("System/Total_Reward_Calls", stats_value["total_reward_calls"], global_step)
            writer.add_scalar("System/Total_Policy_Samples", stats_value["total_policy_samples"], global_step)
            writer.add_scalar("System/Active_Rollout_Workers", stats_value["active_rollout_workers"], global_step)
            for name, value in performance.items():
                writer.add_scalar(f"Performance/{name}", value, global_step)
            for name, value in stats_value["timings"].items():
                writer.add_scalar(f"Performance/{name}", value, global_step)
            writer.add_scalar("Performance/sync_seconds", sync_seconds, global_step)
            writer.add_scalar("Performance/train_loop_seconds", time.time() - started, global_step)
            writer.flush()
            last_log = time.time()

    ray.get(trainers[0].save_agent.remote(args.ckpt_dir, global_step))
    writer.close()
    ray.shutdown()


if __name__ == "__main__":
    main(parse_args())
