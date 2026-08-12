#!/usr/bin/env python3
"""Collect a real VLA/LIBERO trajectory and imagine from one point with Ctrl-World.

The real trajectory is collected first.  Before ``start_index`` the world model
uses real image/action history.  Afterwards each predicted WM image is sent to
VLA, and the resulting raw VLA actions condition the next WM prediction.

Default usage:
    CUDA_VISIBLE_DEVICES=0 python tests_dsj/test_ctrl_world_wm_closed_loop_predict.py

Action-matched comparison:
    CUDA_VISIBLE_DEVICES=0 python tests_dsj/test_ctrl_world_wm_closed_loop_predict.py --action-matched
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

from libero.libero import benchmark  # noqa: E402

from ctrl_world.config import wm_args  # noqa: E402
from ctrl_world.models.ctrl_world import CrtlWorld  # noqa: E402
from ctrl_world.models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline  # noqa: E402
from experiments.robot.libero.libero_utils import (  # noqa: E402
    GenerateConfig,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
)
from experiments.robot.openvla_utils import resize_image_for_policy  # noqa: E402
from experiments.robot.robot_utils import (  # noqa: E402
    invert_gripper_action,
    normalize_gripper_action,
)
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK  # noqa: E402
from rl.actor_critic_model_discrete import ActorCritic  # noqa: E402
from rl.utils import prepare_one_obs  # noqa: E402


BENCHMARK = "libero_spatial"
VLA_CHECKPOINT = (
    "/mnt/data/lcx3/checkpoint/dsj/"
    "openvla-7b+libero_spatial_no_noops+b32+lr-0.0005+lora-r32+dropout-0.0"
    "--image_aug--parallel_dec--8_acts_chunk--discrete_acts--proprio_state--100000_chkpt"
)
#VLA_CHECKPOINT2 = "/mnt/data/lcx3/checkpoint/dsj/20251225_113851_distill_checkpoint_latest.pt"
VLA_CHECKPOINT2 =""
DEFAULT_WM_CHECKPOINT = (
    "/mnt/data/lcx3/Ctrl-World/model_ckpt/libero_spatial/2026-08-06T16-16-35_libero_spatial/best_val_loss.pt"
)
SVD_MODEL = "/mnt/data/lcx3/checkpoint/ctrl_world/svd/svd_model"
CLIP_MODEL = "/mnt/data/lcx3/checkpoint/ctrl_world/clip/clip_model"
DEFAULT_OUTPUT = "/mnt/data/lcx3/AcceRL/tests_dsj/ctrl_world_wm_closed_loop_eval"

HEIGHT = 192
WIDTH = 320
NUM_CAMS = 2
NUM_HISTORY = 6
CHUNK_FRAMES = 5
STRIDE = CHUNK_FRAMES - 1
NUM_INFERENCE_STEPS = 10
DTYPE = torch.bfloat16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, default=1)
    parser.add_argument("--initial-state-id", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=70)
    parser.add_argument("--num-chunks", type=int, default=16)
    parser.add_argument("--checkpoint", default=DEFAULT_WM_CHECKPOINT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--action-matched",
        action="store_true",
        help="Send each VLA action to both WM and LIBERO, then compare their results.",
    )
    return parser.parse_args()


def get_task(task_id: int):
    suite = benchmark.get_benchmark_dict()[BENCHMARK]()
    if not 0 <= task_id < suite.n_tasks:
        raise ValueError(f"task-id must be in [0, {suite.n_tasks - 1}]")
    return suite.get_task(task_id), suite


def load_vla(device: str, seed: int) -> Tuple[ActorCritic, GenerateConfig]:
    cfg = GenerateConfig(
        pretrained_checkpoint=VLA_CHECKPOINT,
        checkpoint2=VLA_CHECKPOINT2,
        use_l1_regression=False,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=False,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        unnorm_key=f"{BENCHMARK}_no_noops",
        task_suite_name=BENCHMARK,
        num_steps_wait=10,
        use_lora=True,
        lora_rank=32,
        lora_dropout=0.0,
        seed=seed,
    )
    actor = ActorCritic(cfg, torch_dtype=DTYPE).to(device).eval()
    if cfg.unnorm_key not in actor.vla.norm_stats:
        for key in (BENCHMARK, f"{BENCHMARK}_no_noops"):
            if key in actor.vla.norm_stats:
                cfg.unnorm_key = key
                actor.cfg.unnorm_key = key
                break
        else:
            raise KeyError(f"No VLA normalization key for {BENCHMARK}")
    return actor, cfg


def load_world_model(checkpoint: str, device: str) -> CrtlWorld:
    cfg = wm_args(task_type="replay")
    cfg.svd_model_path = SVD_MODEL
    cfg.clip_model_path = CLIP_MODEL
    cfg.ckpt_path = checkpoint
    cfg.val_model_path = checkpoint
    cfg.num_cams = NUM_CAMS
    cfg.height = HEIGHT
    cfg.width = WIDTH
    cfg.num_history = NUM_HISTORY
    cfg.num_frames = CHUNK_FRAMES
    cfg.num_inference_steps = NUM_INFERENCE_STEPS
    cfg.dtype = DTYPE

    model = CrtlWorld(cfg)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.to(device).to(DTYPE).eval()


def load_action_stats(checkpoint: str) -> Tuple[np.ndarray, np.ndarray]:
    path = Path(checkpoint).resolve().parent / "condition_stat.json"
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    low = np.asarray(data["condition_p01"], dtype=np.float32)[None]
    high = np.asarray(data["condition_p99"], dtype=np.float32)[None]
    if low.shape != (1, ACTION_DIM) or high.shape != (1, ACTION_DIM):
        raise ValueError(f"Invalid action stats in {path}")
    print(f"[ctrl-world] action stats: {path}")
    return low, high


def real_observation(raw_obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Build VLA input; get_libero_* performs the required 180-degree rotation."""
    return {
        "full_image": resize_image_for_policy(get_libero_image(raw_obs), 224),
        "wrist_image": resize_image_for_policy(get_libero_wrist_image(raw_obs), 224),
        "state": np.concatenate(
            (
                raw_obs["robot0_eef_pos"],
                quat2axisangle(raw_obs["robot0_eef_quat"]),
                raw_obs["robot0_gripper_qpos"],
            )
        ),
    }


def predicted_observation(images: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        "full_image": resize_image_for_policy(images[0], 224),
        "wrist_image": resize_image_for_policy(images[1], 224),
        "state": np.zeros(ACTION_DIM, dtype=np.float32),
    }


@torch.no_grad()
def query_vla(
    actor: ActorCritic,
    cfg: GenerateConfig,
    observation: Dict[str, np.ndarray],
    instruction: str,
) -> np.ndarray:
    inputs = prepare_one_obs(cfg, actor.processor, observation, instruction, DTYPE)
    logits, _ = actor(actor.prepare_inputs_batch([inputs]))
    _, _, normalized = actor.post_process(logits, deterministic=[True])
    actions = actor.vla._unnormalize_actions(normalized[0], cfg.unnorm_key)
    if hasattr(actions, "detach"):
        actions = actions.detach().cpu().numpy()
    return np.asarray(actions, dtype=np.float32)


def process_action_for_env(action: np.ndarray) -> np.ndarray:
    action = normalize_gripper_action(action.astype(np.float32).copy(), binarize=True)
    return invert_gripper_action(action).astype(np.float32)


def rotated_camera_frames(raw_obs: Dict[str, Any]) -> List[np.ndarray]:
    return [
        np.asarray(get_libero_image(raw_obs), dtype=np.uint8).copy(),
        np.asarray(get_libero_wrist_image(raw_obs), dtype=np.uint8).copy(),
    ]


@torch.no_grad()
def collect_real_trajectory(
    actor: ActorCritic,
    cfg: GenerateConfig,
    task: Any,
    suite: Any,
    instruction: str,
    total_frames: int,
    args: argparse.Namespace,
) -> Tuple[List[List[np.ndarray]], np.ndarray, np.ndarray, bool]:
    env, _ = get_libero_env(task, cfg.model_family, resolution=256)
    raw_obs = env.reset()
    initial_states = suite.get_task_init_states(args.task_id)
    if not 0 <= args.initial_state_id < len(initial_states):
        raise ValueError(f"initial-state-id must be in [0, {len(initial_states) - 1}]")
    raw_obs = env.set_init_state(initial_states[args.initial_state_id])
    for _ in range(10):
        raw_obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))

    frames = [rotated_camera_frames(raw_obs)]
    raw_actions: List[np.ndarray] = []
    env_actions: List[np.ndarray] = []
    queue = deque()
    done = False
    try:
        while len(frames) < total_frames:
            if done:
                frames.append([image.copy() for image in frames[-1]])
                raw_actions.append(
                    raw_actions[-1].copy()
                    if raw_actions
                    else np.zeros(ACTION_DIM, dtype=np.float32)
                )
                continue
            if not queue:
                queue.extend(query_vla(actor, cfg, real_observation(raw_obs), instruction))
            raw_action = np.asarray(queue.popleft(), dtype=np.float32)
            env_action = process_action_for_env(raw_action)
            raw_obs, _, done, _ = env.step(env_action.tolist())
            raw_actions.append(raw_action)
            env_actions.append(env_action)
            frames.append(rotated_camera_frames(raw_obs))
    finally:
        env.close()

    return (
        frames,
        np.stack(raw_actions),
        np.stack(env_actions) if env_actions else np.zeros((0, ACTION_DIM), np.float32),
        done,
    )


def resize_video(frames: Sequence[np.ndarray]) -> np.ndarray:
    tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()
    tensor = F.interpolate(tensor, size=(HEIGHT, WIDTH), mode="bilinear", align_corners=False)
    return tensor.clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()


@torch.no_grad()
def encode_frames(
    model: CrtlWorld,
    frames: Sequence[List[np.ndarray]],
    device: str,
) -> torch.Tensor:
    per_camera = []
    for camera in range(NUM_CAMS):
        video = resize_video([frame[camera] for frame in frames])
        pixels = torch.from_numpy(video).permute(0, 3, 1, 2).to(device, DTYPE)
        pixels = pixels / 255.0 * 2.0 - 1.0
        chunks = []
        for start in range(0, len(pixels), 8):
            latent = model.pipeline.vae.encode(pixels[start : start + 8]).latent_dist.sample()
            chunks.append(latent * model.pipeline.vae.config.scaling_factor)
        per_camera.append(torch.cat(chunks))
    return torch.cat(per_camera, dim=2)


@torch.no_grad()
def decode_latents(model: CrtlWorld, latents: torch.Tensor) -> List[np.ndarray]:
    latent_height = HEIGHT // 8
    videos = []
    for camera in range(NUM_CAMS):
        camera_latents = latents[
            :,
            :,
            camera * latent_height : (camera + 1) * latent_height,
            :,
        ]
        decoded = []
        for start in range(0, len(camera_latents), CHUNK_FRAMES):
            chunk = camera_latents[start : start + CHUNK_FRAMES]
            chunk = chunk / model.pipeline.vae.config.scaling_factor
            decoded.append(model.pipeline.vae.decode(chunk, num_frames=len(chunk)).sample)
        video = ((torch.cat(decoded) / 2.0 + 0.5).clamp(0, 1) * 255).byte()
        videos.append(video.permute(0, 2, 3, 1).cpu().numpy())
    return videos


def frame_aligned_real_actions(step_actions: np.ndarray, num_frames: int) -> np.ndarray:
    aligned = np.zeros((num_frames, ACTION_DIM), dtype=np.float32)
    aligned[1:] = step_actions[: num_frames - 1]
    return aligned


def frame_aligned_imagination_actions(
    current_action: np.ndarray,
    queried_actions: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(current_action, dtype=np.float32).reshape(1, ACTION_DIM),
            np.asarray(queried_actions[:STRIDE], dtype=np.float32),
        ],
        axis=0,
    )


@torch.no_grad()
def predict_chunk(
    model: CrtlWorld,
    current: torch.Tensor,
    history: torch.Tensor,
    actions: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    instruction: str,
    device: str,
) -> torch.Tensor:
    normalized = 2.0 * (actions - low) / (high - low + 1e-8) - 1.0
    normalized = np.clip(normalized, -1.0, 1.0).astype(np.float32)
    action_dtype = model.action_encoder.action_encode[0].weight.dtype
    action_tensor = torch.from_numpy(normalized).unsqueeze(0).to(device, action_dtype)
    if model.args.text_cond:
        text = model.action_encoder(
            action_tensor,
            [instruction],
            model.tokenizer,
            model.text_encoder,
        )
    else:
        text = model.action_encoder(action_tensor)

    _, latents = CtrlWorldDiffusionPipeline.__call__(
        model.pipeline,
        image=current,
        text=text,
        width=WIDTH,
        height=HEIGHT * NUM_CAMS,
        num_frames=CHUNK_FRAMES,
        history=history,
        num_inference_steps=NUM_INFERENCE_STEPS,
        decode_chunk_size=CHUNK_FRAMES,
        max_guidance_scale=1.0,
        fps=7,
        motion_bucket_id=127,
        output_type="latent",
        return_dict=False,
        frame_level_cond=True,
    )
    return latents[0]


@torch.no_grad()
def imagine(
    model: CrtlWorld,
    actor: ActorCritic,
    cfg: GenerateConfig,
    warmup_latents: torch.Tensor,
    real_actions: np.ndarray,
    instruction: str,
    low: np.ndarray,
    high: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, np.ndarray]:
    action_buffer = [
        real_actions[index : index + 1]
        for index in range(args.start_index + 1)
    ]
    latent_buffer = [
        warmup_latents[index : index + 1]
        for index in range(args.start_index + 1)
    ]
    history_lags = list(range(NUM_HISTORY, 0, -1))
    segments = []
    imagined_actions = []

    for chunk_id in range(args.num_chunks):
        history = torch.cat(
            [latent_buffer[-1 - lag] for lag in history_lags]
        ).unsqueeze(0)
        history_actions = np.concatenate(
            [action_buffer[-1 - lag] for lag in history_lags]
        )
        current = latent_buffer[-1][0]
        current_images = [
            video[0] for video in decode_latents(model, current.unsqueeze(0))
        ]
        queried = query_vla(
            actor,
            cfg,
            predicted_observation(current_images),
            instruction,
        )
        frame_actions = frame_aligned_imagination_actions(
            action_buffer[-1][0],
            queried,
        )
        condition = np.concatenate([history_actions, frame_actions])
        predicted = predict_chunk(
            model,
            current.unsqueeze(0),
            history,
            condition,
            low,
            high,
            instruction,
            args.device,
        )

        take = CHUNK_FRAMES if chunk_id == args.num_chunks - 1 else STRIDE
        segments.append(predicted[:take])
        imagined_actions.append(frame_actions[1:])
        for relative_index in range(1, CHUNK_FRAMES):
            latent_buffer.append(predicted[relative_index : relative_index + 1])
            action_buffer.append(frame_actions[relative_index : relative_index + 1])
        print(f"[imagine] chunk {chunk_id + 1}/{args.num_chunks}")

    return torch.cat(segments), np.concatenate(imagined_actions)


@torch.no_grad()
def action_matched_imagination(
    model: CrtlWorld,
    actor: ActorCritic,
    cfg: GenerateConfig,
    task: Any,
    suite: Any,
    instruction: str,
    low: np.ndarray,
    high: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, List[List[np.ndarray]], np.ndarray, np.ndarray, bool]:
    """Query VLA from WM images and apply the same actions to WM and LIBERO."""
    env, _ = get_libero_env(task, cfg.model_family, resolution=256)
    raw_obs = env.reset()
    initial_states = suite.get_task_init_states(args.task_id)
    if not 0 <= args.initial_state_id < len(initial_states):
        raise ValueError(f"initial-state-id must be in [0, {len(initial_states) - 1}]")
    raw_obs = env.set_init_state(initial_states[args.initial_state_id])
    for _ in range(10):
        raw_obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))

    warmup_frames = [rotated_camera_frames(raw_obs)]
    warmup_actions: List[np.ndarray] = []
    executed_actions: List[np.ndarray] = []
    queue = deque()
    done = False

    try:
        while len(warmup_frames) <= args.start_index:
            if not queue:
                queue.extend(query_vla(actor, cfg, real_observation(raw_obs), instruction))
            raw_action = np.asarray(queue.popleft(), dtype=np.float32)
            env_action = process_action_for_env(raw_action)
            raw_obs, _, done, _ = env.step(env_action.tolist())
            warmup_actions.append(raw_action)
            executed_actions.append(env_action)
            warmup_frames.append(rotated_camera_frames(raw_obs))
            if done and len(warmup_frames) <= args.start_index:
                raise RuntimeError("LIBERO episode ended before start-index")

        warmup_latents = encode_frames(model, warmup_frames, args.device)
        aligned_actions = frame_aligned_real_actions(
            np.stack(warmup_actions),
            len(warmup_frames),
        )
        latent_buffer = [
            warmup_latents[index : index + 1]
            for index in range(args.start_index + 1)
        ]
        action_buffer = [
            aligned_actions[index : index + 1]
            for index in range(args.start_index + 1)
        ]
        history_lags = list(range(NUM_HISTORY, 0, -1))
        segments = []
        imagined_actions = []
        libero_frames = [warmup_frames[-1]]

        for chunk_id in range(args.num_chunks):
            history = torch.cat(
                [latent_buffer[-1 - lag] for lag in history_lags]
            ).unsqueeze(0)
            history_actions = np.concatenate(
                [action_buffer[-1 - lag] for lag in history_lags]
            )
            current = latent_buffer[-1][0]
            current_images = [
                video[0] for video in decode_latents(model, current.unsqueeze(0))
            ]
            queried = query_vla(
                actor,
                cfg,
                predicted_observation(current_images),
                instruction,
            )
            frame_actions = frame_aligned_imagination_actions(
                action_buffer[-1][0],
                queried,
            )

            # The exact same raw VLA actions condition WM and advance LIBERO.
            for raw_action in frame_actions[1:]:
                env_action = process_action_for_env(raw_action)
                if not done:
                    raw_obs, _, done, _ = env.step(env_action.tolist())
                executed_actions.append(env_action)
                libero_frames.append(rotated_camera_frames(raw_obs))

            condition = np.concatenate([history_actions, frame_actions])
            predicted = predict_chunk(
                model,
                current.unsqueeze(0),
                history,
                condition,
                low,
                high,
                instruction,
                args.device,
            )
            take = CHUNK_FRAMES if chunk_id == args.num_chunks - 1 else STRIDE
            segments.append(predicted[:take])
            imagined_actions.append(frame_actions[1:])
            for relative_index in range(1, CHUNK_FRAMES):
                latent_buffer.append(predicted[relative_index : relative_index + 1])
                action_buffer.append(frame_actions[relative_index : relative_index + 1])
            print(f"[action-matched] chunk {chunk_id + 1}/{args.num_chunks}")

        predictions = torch.cat(segments)
        if len(libero_frames) != len(predictions):
            raise RuntimeError(
                f"LIBERO/prediction length mismatch: {len(libero_frames)} != {len(predictions)}"
            )
        return (
            predictions,
            libero_frames,
            np.concatenate(imagined_actions),
            np.stack(executed_actions),
            done,
        )
    finally:
        env.close()


def metrics(gt: Sequence[np.ndarray], pred: Sequence[np.ndarray]) -> Dict[str, float]:
    squared_errors = []
    absolute_errors = []
    for gt_video, pred_video in zip(gt, pred):
        error = pred_video.astype(np.float32) / 255.0 - gt_video.astype(np.float32) / 255.0
        squared_errors.append(np.mean(error**2, axis=(1, 2, 3)))
        absolute_errors.append(np.mean(np.abs(error), axis=(1, 2, 3)))
    mse = float(np.mean(np.concatenate(squared_errors)))
    mae = float(np.mean(np.concatenate(absolute_errors)))
    return {"mse": mse, "mae": mae, "psnr": float(-10.0 * np.log10(max(mse, 1e-12)))}


def label(image: np.ndarray, text: str) -> np.ndarray:
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    draw.rectangle((0, 0, 150, 18), fill=(0, 0, 0))
    draw.text((4, 3), text, fill=(255, 255, 255))
    return np.asarray(pil)


def comparison_video(gt: Sequence[np.ndarray], pred: Sequence[np.ndarray]) -> np.ndarray:
    frames = []
    for index in range(len(pred[0])):
        camera_rows = []
        for camera, (gt_video, pred_video) in enumerate(zip(gt, pred)):
            diff = np.abs(
                pred_video[index].astype(np.int16) - gt_video[index].astype(np.int16)
            )
            diff = np.clip(diff * 3, 0, 255).astype(np.uint8)
            camera_rows.append(
                np.concatenate(
                    [
                        label(gt_video[index], f"cam{camera} REAL"),
                        label(pred_video[index], f"cam{camera} IMAGINED"),
                        label(diff, f"cam{camera} DIFF x3"),
                    ],
                    axis=1,
                )
            )
        frames.append(np.concatenate(camera_rows, axis=0))
    return np.stack(frames)


def write_video(path: Path, frames: np.ndarray, fps: int = 5) -> None:
    imageio.mimsave(path, list(frames), fps=fps)


def main() -> None:
    args = parse_args()
    if args.start_index < NUM_HISTORY:
        raise ValueError(f"start-index must be >= {NUM_HISTORY}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    task, suite = get_task(args.task_id)
    instruction = task.language
    predicted_frames = STRIDE * (args.num_chunks - 1) + CHUNK_FRAMES
    total_frames = args.start_index + predicted_frames
    output = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=True)
    print(f"[task] {task.name}: {instruction}")

    print("[models] loading VLA and Ctrl-World")
    actor, vla_cfg = load_vla(args.device, args.seed)
    model = load_world_model(args.checkpoint, args.device)
    low, high = load_action_stats(args.checkpoint)

    if args.action_matched:
        print("[action-matched] WM image -> VLA -> same actions to WM and LIBERO")
        (
            pred_latents,
            selected_real,
            imagined_actions,
            env_actions,
            rollout_done,
        ) = action_matched_imagination(
            model,
            actor,
            vla_cfg,
            task,
            suite,
            instruction,
            low,
            high,
            args,
        )
    else:
        print(f"[real] collecting {total_frames} VLA/LIBERO frames")
        real_frames, step_actions, env_actions, rollout_done = collect_real_trajectory(
            actor,
            vla_cfg,
            task,
            suite,
            instruction,
            total_frames,
            args,
        )
        aligned_actions = frame_aligned_real_actions(step_actions, len(real_frames))
        warmup_latents = encode_frames(
            model,
            real_frames[: args.start_index + 1],
            args.device,
        )
        print(f"[imagine] starting from real frame {args.start_index}")
        pred_latents, imagined_actions = imagine(
            model,
            actor,
            vla_cfg,
            warmup_latents,
            aligned_actions,
            instruction,
            low,
            high,
            args,
        )
        selected_real = real_frames[
            args.start_index : args.start_index + len(pred_latents)
        ]

    pred_videos = decode_latents(model, pred_latents)
    gt_videos = [
        resize_video([frame[camera] for frame in selected_real])
        for camera in range(NUM_CAMS)
    ]

    score = metrics(gt_videos, pred_videos)
    comparison = comparison_video(gt_videos, pred_videos)
    comparison_path = output / "real_vs_imagined.mp4"
    write_video(comparison_path, comparison)
    for camera, (real, imagined) in enumerate(zip(gt_videos, pred_videos)):
        write_video(output / f"cam{camera}_real.mp4", real)
        write_video(output / f"cam{camera}_imagined.mp4", imagined)

    np.save(output / "real_env_actions.npy", env_actions)
    np.save(output / "imagined_vla_actions.npy", imagined_actions)
    summary = {
        "mode": "action-matched" if args.action_matched else "trajectory-imagination",
        "task_id": args.task_id,
        "task": task.name,
        "instruction": instruction,
        "checkpoint": args.checkpoint,
        "start_index": args.start_index,
        "num_chunks": args.num_chunks,
        "predicted_frames": len(pred_latents),
        "real_rollout_done": rollout_done,
        "metrics": score,
        "comparison_video": str(comparison_path),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"[done] {comparison_path}")
    print("[metrics]", json.dumps(score, indent=2))


if __name__ == "__main__":
    main()
