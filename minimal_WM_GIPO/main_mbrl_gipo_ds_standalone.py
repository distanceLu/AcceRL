from __future__ import annotations

import argparse
import asyncio
import os
import random
import socket
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import ray
import torch
import torch.distributed as distributed
from torch.distributions import Categorical, kl
from torch.utils.tensorboard import SummaryWriter
from ray.exceptions import GetTimeoutError

from ds_com import InferenceActorCom, TrainerActorCom

NUM_ACTIONS_CHUNK = 4
ACTION_DIM = 3
ACTION_BINS = 11


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def resolve_torch_extensions_dir(user_path: str | None = None) -> str:
    if user_path:
        path = os.path.abspath(os.path.expanduser(user_path))
    elif os.environ.get("TORCH_EXTENSIONS_DIR"):
        path = os.environ["TORCH_EXTENSIONS_DIR"]
    elif os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK):
        path = "/dev/shm/torch_extensions"
    else:
        path = os.path.join("/tmp", "torch_extensions")
    os.makedirs(path, exist_ok=True)
    return path


def prewarm_deepspeed_ops(enabled: bool) -> None:
    if not enabled:
        print("[DeepSpeed] Skipping prewarm (--no-prewarm-deepspeed-ops).")
        return
    try:
        from deepspeed.ops.op_builder import CPUAdamBuilder, FusedAdamBuilder
    except Exception as exc:
        print(f"[DeepSpeed] Skipping prewarm: cannot import op builders ({exc})")
        return
    print("[DeepSpeed] Starting prewarm for FusedAdam/CPUAdam (first run may be slow)...")
    try:
        FusedAdamBuilder().load(verbose=True)
    except Exception as exc:
        print(f"[DeepSpeed] FusedAdam prewarm failed; will fallback at runtime: {exc}")
    try:
        CPUAdamBuilder().load(verbose=True)
    except Exception as exc:
        print(f"[DeepSpeed] CPUAdam prewarm failed; will fallback at runtime: {exc}")
    print("[DeepSpeed] Prewarm stage finished.")


def sync_weights_blocking(trainer_actor, inference_actors, group_name: str, timeout_s: float = 120.0) -> None:
    # Broadcast and receive must be launched concurrently to avoid handshake deadlocks.
    broadcast_ref = trainer_actor.broadcast_weights.remote(group_name)
    receive_refs = [inf.receive_and_update_weights.remote(group_name) for inf in inference_actors]
    all_refs = [broadcast_ref] + receive_refs
    ready, pending = ray.wait(all_refs, num_returns=len(all_refs), timeout=timeout_s)
    if pending:
        raise TimeoutError(
            f"Weight sync timed out after {timeout_s}s (ready={len(ready)}, pending={len(pending)}). "
            "Likely a collective handshake mismatch."
        )
    ray.get(all_refs)


def wait_for_replay_ready(
    replay_buffer,
    min_items: int,
    rollout_refs: List,
    timeout_s: float,
    stall_timeout_s: float,
    poll_interval_s: float = 0.5,
) -> None:
    start_t = time.time()
    last_items = -1
    last_progress_t = start_t
    while True:
        if rollout_refs:
            ready, _ = ray.wait(rollout_refs, num_returns=1, timeout=0.0)
            if ready:
                ray.get(ready[0])
                raise RuntimeError("A rollout worker exited unexpectedly during replay warmup.")

        items = int(ray.get(replay_buffer.size.remote()))
        now = time.time()
        if items >= int(min_items):
            return
        if items > last_items:
            print(f"[Warmup] replay_items={items}/{min_items}")
            last_items = items
            last_progress_t = now
        elif now - last_progress_t > stall_timeout_s:
            raise TimeoutError(
                f"Replay warmup stalled for {stall_timeout_s:.1f}s at replay_items={items}/{min_items}. "
                "Check rollout/inference actors."
            )
        if now - start_t > timeout_s:
            raise TimeoutError(
                f"Replay warmup timed out after {timeout_s:.1f}s (replay_items={items}/{min_items})."
            )
        time.sleep(poll_interval_s)


class ThroughputTracker:
    def __init__(self, exp_name: str):
        self.exp_name = exp_name
        self.run_dir = os.path.join("runs", exp_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.csv_path = os.path.join(self.run_dir, "throughput_log.csv")
        self.t0 = time.time()
        self.last_t = self.t0
        self.last_steps = 0
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                import csv
                csv.writer(f).writerow(["global_step", "elapsed_s", "replay_steps", "trainer_samples", "sps"])

    def tick(self, global_step: int, replay_steps: int, trainer_samples: int):
        now = time.time()
        dt = max(now - self.last_t, 1e-6)
        dsteps = max(replay_steps - self.last_steps, 0)
        sps = float(dsteps / dt)
        self.last_t = now
        self.last_steps = replay_steps
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            import csv
            csv.writer(f).writerow([global_step, now - self.t0, replay_steps, trainer_samples, sps])
        print(f"[AcceRL Throughput] step={global_step} replay={replay_steps} trainer_samples={trainer_samples} sps={sps:.2f}")


class TrainerThroughputRecorder:
    def __init__(self, exp_name: str):
        self.run_dir = os.path.join("runs", exp_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.csv_path = os.path.join(self.run_dir, "trainer_throughput.csv")
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                import csv
                csv.writer(f).writerow(["global_step", "loss", "policy_loss", "value_loss", "train_time"])

    def record_step(self, global_step: int, metrics: Dict[str, float]):
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            import csv
            csv.writer(f).writerow(
                [global_step, metrics.get("loss", 0.0), metrics.get("policy_loss", 0.0), metrics.get("value_loss", 0.0), metrics.get("train_time", 0.0)]
            )


class FakeProcessor:
    def __call__(self, obs, task_description, torch_dtype=torch.float32):
        _ = task_description
        image = obs.get("image", np.random.randn(3, 224, 224).astype(np.float32))
        proprio = obs.get("proprio", np.zeros(7, dtype=np.float32))
        return {
            "image": torch.tensor(np.array(image, copy=True), dtype=torch_dtype),
            "proprio": torch.tensor(np.array(proprio, copy=True), dtype=torch_dtype),
        }


class FakeEnv:
    def __init__(self, task_id: int, max_steps: int = 50):
        self.task_id = int(task_id)
        self.max_steps = int(max_steps)
        self.task_description = f"wm-fake-task-{task_id}"
        self.t = 0
        self.rng = np.random.default_rng(task_id)

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        return self._obs(), {}

    def _obs(self):
        return {
            "image": self.rng.standard_normal((3, 224, 224)).astype(np.float32),
            "proprio": self.rng.standard_normal((7,)).astype(np.float32),
            "task_description": self.task_description,
        }

    def step(self, action):
        _ = action
        self.t += 1
        obs = self._obs()
        reward = float(self.rng.normal(0.05, 0.4))
        terminated = bool(self.t >= self.max_steps or self.rng.random() < 0.05)
        return obs, reward, terminated, False, {"is_success": float(reward > 0.0)}


class FakeActorCritic(torch.nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        in_dim = 3 * 224 * 224 + 7
        self.enc = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.policy = torch.nn.Linear(hidden_dim, NUM_ACTIONS_CHUNK * ACTION_DIM * ACTION_BINS)
        self.value = torch.nn.Linear(hidden_dim, 1)
        self.processor = FakeProcessor()

    def prepare_inputs_batch(self, obs_list):
        device = next(self.parameters()).device
        imgs = torch.stack([x["image"].float().reshape(-1) for x in obs_list], 0).to(device)
        props = torch.stack([x["proprio"].float().reshape(-1) for x in obs_list], 0).to(device)
        return {"image": imgs, "proprio": props}

    def forward(self, batch):
        h = self.enc(torch.cat([batch["image"], batch["proprio"]], -1))
        logits = self.policy(h).view(-1, NUM_ACTIONS_CHUNK * ACTION_DIM, ACTION_BINS)
        value = self.value(h).squeeze(-1)
        return logits, value

    def post_process(self, logits, deterministic=False):
        if isinstance(deterministic, list):
            flags = deterministic
        else:
            flags = [bool(deterministic)] * logits.shape[0]
        d = Categorical(logits=logits)
        sampled = d.sample()
        greedy = logits.argmax(-1)
        chosen = torch.stack([greedy[i] if flags[i] else sampled[i] for i in range(logits.shape[0])], 0)
        tokens = chosen.view(-1, NUM_ACTIONS_CHUNK, ACTION_DIM)
        act = tokens.float() / max(ACTION_BINS - 1, 1) * 2.0 - 1.0
        return act, tokens

    def get_parameter_groups(self):
        return [
            {"name": "policy", "params": list(self.enc.parameters()) + list(self.policy.parameters())},
            {"name": "value", "params": list(self.value.parameters())},
        ]

    def save_model(self, ckpt_dir: str, epoch: int):
        torch.save(self.state_dict(), os.path.join(ckpt_dir, f"wm_agent_{epoch}.pt"))


class FakeRewardModel(torch.nn.Module):
    def forward(self, obs):
        if isinstance(obs, dict):
            x = obs["image"]
        else:
            x = obs
        if x.ndim == 3:
            x = x.unsqueeze(0)
        b = x.shape[0]
        return torch.randn(b, 2, device=x.device)


class FakeDenoiser(torch.nn.Module):
    def forward(self, obs_hist: torch.Tensor, act_hist: torch.Tensor):
        _ = act_hist
        if obs_hist.ndim == 4:
            obs_hist = obs_hist.unsqueeze(0)
        b = obs_hist.shape[0]
        return torch.randn(b, 3, 224, 224, device=obs_hist.device)


@dataclass
class Experience:
    obs: Dict[str, torch.Tensor]
    action_token: np.ndarray
    advantage: float
    behaviour_logits: np.ndarray
    value_target: float


@ray.remote
class StatsActor:
    def __init__(self, window_size=100):
        self.rollout_ret = deque(maxlen=window_size)
        self.eval_ret = deque(maxlen=window_size)
        self.imag_ret = deque(maxlen=window_size)

    def add_episode_return(self, env_name, ep_return, step_time, ep_len, success, actor_id=None, step_num=None):
        _ = step_time, ep_len, success, actor_id, step_num
        if str(env_name).startswith("eval_"):
            self.eval_ret.append(float(ep_return))
        else:
            self.rollout_ret.append(float(ep_return))

    def add_imagine_reward(self, avg_imagine_reward, actor_id):
        _ = actor_id
        self.imag_ret.append(float(avg_imagine_reward))

    def get_stats(self):
        return {
            "_global_rollout_": {"avg_return": float(np.mean(self.rollout_ret)) if self.rollout_ret else 0.0, "avg_imagine_reward": float(np.mean(self.imag_ret)) if self.imag_ret else 0.0},
            "_global_eval_": {"avg_return": float(np.mean(self.eval_ret)) if self.eval_ret else 0.0},
        }


@ray.remote
class ReplayBufferActor:
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=int(capacity))

    def add_batch(self, batch):
        self.buffer.extend(batch)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size):
        picked = random.sample(self.buffer, int(batch_size))
        obs = [x.obs for x in picked]
        act = np.stack([x.action_token for x in picked], 0)
        adv = np.asarray([x.advantage for x in picked], np.float32)
        logits = np.stack([x.behaviour_logits for x in picked], 0)
        vt = np.asarray([x.value_target for x in picked], np.float32)
        return obs, act, adv, logits, vt


@ray.remote(num_gpus=1)
class InferenceActor(InferenceActorCom):
    def __init__(self, actor_id, cfg, stats_actor, torch_dtype, inference_batch, inference_timeout_ms):
        super().__init__()
        _ = cfg
        self.actor_id = int(actor_id)
        self.model = FakeActorCritic()
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device=target_device, dtype=torch_dtype)
        self.model.eval()
        self.batch_size = int(inference_batch)
        self.timeout_sec = float(inference_timeout_ms) / 1000.0
        self.stats_actor = stats_actor
        self.requests, self.promises = [], []
        self.last_process_time = time.time()
        loop = asyncio.get_event_loop()
        self._bg = loop.create_task(self._loop())
        self._bg.add_done_callback(self._on_bg_done)

    def _on_bg_done(self, task: asyncio.Task):
        try:
            task.result()
        except Exception as exc:
            import traceback
            print(f"[ERROR] InferenceActor {self.actor_id} background loop crashed: {exc}", flush=True)
            traceback.print_exc()

    async def request(self, inputs_t: Dict[str, torch.Tensor], deterministic=False):
        fut = asyncio.get_event_loop().create_future()
        self.requests.append((inputs_t, deterministic))
        self.promises.append(fut)
        return await fut

    async def _loop(self):
        while True:
            should = self.requests and (len(self.requests) >= self.batch_size or time.time() - self.last_process_time > self.timeout_sec)
            if not should:
                await asyncio.sleep(0.0005)
                continue
            reqs, pros = self.requests, self.promises
            self.requests, self.promises = [], []
            try:
                batch = self.model.prepare_inputs_batch([x[0] for x in reqs])
                with torch.inference_mode():
                    logits, value = self.model(batch)
                    actions, tokens = self.model.post_process(logits, deterministic=[x[1] for x in reqs])
                logits = logits.view(-1, NUM_ACTIONS_CHUNK, ACTION_DIM, ACTION_BINS).cpu().numpy()
                value = value.float().cpu().numpy()
                actions = actions.cpu().numpy()
                tokens = tokens.cpu().numpy()
                for i, p in enumerate(pros):
                    p.set_result((actions[i], actions[i], tokens[i], logits[i], value[i]))
            except Exception as exc:
                import traceback
                print(f"[ERROR] InferenceActor {self.actor_id} batch failed: {exc}", flush=True)
                traceback.print_exc()
                for p in pros:
                    if not p.done():
                        p.set_exception(exc)


@ray.remote
class RewardInferenceActor:
    def __init__(self):
        self.model = FakeRewardModel()
        self.model.cuda() if torch.cuda.is_available() else self.model.cpu()
        self.model.eval()

    def request(self, inputs):
        with torch.no_grad():
            return self.model(inputs)[0].detach().cpu()


@ray.remote
class DenoiserInferenceActor:
    def __init__(self):
        self.model = FakeDenoiser()
        self.model.cuda() if torch.cuda.is_available() else self.model.cpu()
        self.model.eval()

    def request(self, obs, act):
        with torch.no_grad():
            return self.model(obs.unsqueeze(0).float(), act.unsqueeze(0).float())[0].detach().cpu()


@ray.remote
class RolloutWorkerActor:
    def __init__(self, infer, replay, wid, stats_actor, cfg, benchmark_name, num_step_cond, imagine_horizon, torch_dtype, reward_infer, denoiser_infer, reward_scale, gamma, lambda_):
        _ = cfg, benchmark_name
        self.infer, self.replay, self.wid, self.stats_actor = infer, replay, int(wid), stats_actor
        self.num_step_cond, self.imagine_horizon = int(num_step_cond), int(imagine_horizon)
        self.reward_infer, self.denoiser_infer = reward_infer, denoiser_infer
        self.reward_scale, self.gamma, self.lambda_ = float(reward_scale), float(gamma), float(lambda_)
        self.processor = FakeProcessor()
        self.torch_dtype = torch_dtype
        self.envs = [FakeEnv(i) for i in range(10)]
        self.env = self.envs[0]
        self.task_description = self.env.task_description
        self.current_env_name = self.env.get_name() if hasattr(self.env, "get_name") else "FakeWM"
        self.episodes = deque(maxlen=100)
        self.local_buffer = []

    def _obs2inp(self, obs):
        return self.processor(obs, self.task_description, self.torch_dtype)

    def _process_traj(self, seg, bootstrap_val, is_terminal):
        rets, advs = [], []
        gae = 0.0
        lv = 0.0 if is_terminal else float(bootstrap_val)
        for i in reversed(range(len(seg))):
            _, _, rew, _, v = seg[i]
            nv = lv if i == len(seg) - 1 else float(seg[i + 1][4])
            delta = float(rew) + self.gamma * nv - float(v)
            gae = delta + self.gamma * self.lambda_ * gae
            advs.append(gae)
            rets.append(gae + float(v))
        advs.reverse()
        rets.reverse()
        batch = []
        for i, (obs, token, _, logits, _) in enumerate(seg):
            batch.append(Experience(obs=obs, action_token=np.asarray(token, np.int64), advantage=float(advs[i]), behaviour_logits=np.asarray(logits, np.float32), value_target=float(rets[i])))
        self.replay.add_batch.remote(batch)

    def get_one_episode(self):
        obs, _ = self.env.reset(seed=int(time.time() * 1000) + self.wid)
        obs_list, reward_list, done_list, act_list = [obs], [], [], []
        reward_sum, step_count = 0.0, 0
        while True:
            inp = self._obs2inp(obs)
            act_norm, action_env, _, _, _ = ray.get(self.infer.request.remote(inp, False))
            done = False
            for i in range(len(action_env)):
                nxt, r, term, trunc, info = self.env.step(action_env[i])
                obs_list.append(nxt)
                reward_list.append(float(r))
                done_list.append(bool(term or trunc))
                act_list.append(np.asarray(act_norm[i], np.float32))
                reward_sum += float(r)
                step_count += 1
                obs = nxt
                if term or trunc:
                    done = True
                    break
            if done:
                self.stats_actor.add_episode_return.remote("rollout", reward_sum, 0.0, step_count, float(info.get("is_success", 0.0)))
                self.episodes.append((obs_list, reward_list, done_list, act_list, self.task_description))
                break

    def run(self):
        step = 0
        while True:
            if step % 10 == 0 or len(self.episodes) == 0:
                self.get_one_episode()
            step += 1
            obs_list, _, _, act_hist, task_description = random.choice(list(self.episodes))
            obs_imgs = [x["image"] for x in obs_list]
            for st in range(len(obs_imgs) - self.num_step_cond):
                self.local_buffer.clear()
                obs_sub = obs_imgs[st : st + self.num_step_cond]
                obs_t = torch.stack([torch.tensor(np.array(x, copy=True), dtype=torch.float32) for x in obs_sub], 0)
                act_sub = act_hist[st : st + self.num_step_cond - 1]
                if len(act_sub) == 0:
                    continue
                act_t = torch.stack([torch.tensor(np.array(a, copy=True), dtype=torch.float32) for a in act_sub], 0)
                prev_succ = 0.0
                end = False
                for j in range(self.imagine_horizon):
                    _ = j
                    inp = {"image": obs_t[-1], "proprio": torch.zeros(7)}
                    try:
                        act_norm, action_env, action_token, logits, value = ray.get(
                            self.infer.request.remote(inp, False), timeout=30.0
                        )
                    except GetTimeoutError as exc:
                        raise TimeoutError(
                            f"RolloutWorker {self.wid} timed out waiting for InferenceActor."
                        ) from exc
                    act_norm_t = torch.tensor(np.array(act_norm, copy=True), dtype=torch.float32)
                    chunk_reward = 0.0
                    for k in range(len(action_env)):
                        single = act_norm_t[k]
                        act_t = torch.cat([act_t, single.unsqueeze(0)], 0)
                        try:
                            next_obs = ray.get(self.denoiser_infer.request.remote(obs_t, act_t), timeout=30.0)
                        except GetTimeoutError as exc:
                            raise TimeoutError(
                                f"RolloutWorker {self.wid} timed out waiting for DenoiserInferenceActor."
                            ) from exc
                        obs_t = torch.roll(obs_t, shifts=-1, dims=0)
                        obs_t[-1] = next_obs
                        act_t = act_t[1:]
                        try:
                            logits_rew = ray.get(
                                self.reward_infer.request.remote({"image": next_obs}), timeout=30.0
                            )
                        except GetTimeoutError as exc:
                            raise TimeoutError(
                                f"RolloutWorker {self.wid} timed out waiting for RewardInferenceActor."
                            ) from exc
                        probs = torch.softmax(logits_rew, dim=-1)
                        succ = float(probs[1].item())
                        end = int(logits_rew.argmax().item()) == 1
                        rew = succ - prev_succ
                        prev_succ = succ
                        chunk_reward += rew * self.reward_scale
                        if end:
                            break
                    self.local_buffer.append((inp, action_token, chunk_reward, logits, value))
                    if end:
                        break
                if self.local_buffer:
                    if end:
                        self._process_traj(self.local_buffer, 0.0, True)
                    else:
                        boot = ray.get(
                            self.infer.request.remote({"image": obs_t[-1], "proprio": torch.zeros(7)}, False),
                            timeout=30.0,
                        )[4]
                        self._process_traj(self.local_buffer, float(boot), False)
                    self.stats_actor.add_imagine_reward.remote(float(np.mean([x[2] for x in self.local_buffer])), self.wid)
                self.local_buffer.clear()


@ray.remote(num_gpus=1.0)
class TrainerActor(TrainerActorCom):
    def __init__(self, rank, world_size, replay_buffer, cfg, train_batch_size, accumulation_steps, use_bf16, torch_dtype, policy_lr, value_lr, gamma, lambda_, clip_eps, vf_coef, ent_coef, kl_coef, reward_scale, value_warmup_steps, policy_warmup_steps, policy_train_start_step, train_iters, clip_mode, sigma, recompute_value):
        super().__init__()
        _ = cfg, reward_scale, value_warmup_steps, policy_warmup_steps, clip_mode
        self.rank, self.world_size = int(rank), int(world_size)
        self.replay_buffer = replay_buffer
        self.train_batch_size = int(train_batch_size)
        self.accumulation_steps = int(accumulation_steps)
        self.super_batch_size = self.train_batch_size * self.accumulation_steps
        self.use_bf16, self.torch_dtype = bool(use_bf16), torch_dtype
        self.policy_lr, self.value_lr = float(policy_lr), float(value_lr)
        self.gamma, self.lambda_, self.clip_eps = float(gamma), float(lambda_), float(clip_eps)
        self.vf_coef, self.ent_coef, self.kl_coef = float(vf_coef), float(ent_coef), float(kl_coef)
        self.policy_train_start_step, self.train_iters = int(policy_train_start_step), int(train_iters)
        self.sigma, self.recompute_value = float(sigma), bool(recompute_value)
        self.global_step = 0
        self.total_trainer_samples_consumed = 0
        self.model = None
        self.base_model = None

    def setup_deepspeed_group(self, master_addr, master_port):
        import deepspeed
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["LOCAL_RANK"] = "0"
        if not distributed.is_initialized():
            deepspeed.init_distributed(dist_backend="nccl" if torch.cuda.is_available() else "gloo")
        model = FakeActorCritic()
        self.base_model = model
        params = model.get_parameter_groups()
        op = [{"params": x["params"], "name": x["name"], "lr": self.policy_lr if x["name"] == "policy" else self.value_lr} for x in params]
        ds_config = {
            "train_micro_batch_size_per_gpu": self.train_batch_size,
            "gradient_accumulation_steps": self.accumulation_steps,
            "optimizer": {"type": "AdamW", "params": {}},
            "bf16": {"enabled": self.use_bf16},
            "zero_optimization": {"stage": 2, "allgather_partitions": True, "allgather_bucket_size": 5e8, "reduce_scatter": True, "reduce_bucket_size": 5e8, "overlap_comm": True, "contiguous_gradients": True},
            "gradient_clipping": 1.0,
        }
        self.model, self.optimizer, _, _ = deepspeed.initialize(model=model, config=ds_config, model_parameters=op)

    def get_node_ip(self):
        return ray.util.get_node_ip_address()

    async def run_training_epoch(self):
        while ray.get(self.replay_buffer.size.remote()) < self.super_batch_size:
            await asyncio.sleep(0.1)
        t0 = time.time()
        obs_list, action_np, adv_np, logits_np, vt_np = ray.get(self.replay_buffer.sample.remote(self.super_batch_size))
        batch = self.base_model.prepare_inputs_batch(obs_list)
        act = torch.tensor(action_np, dtype=torch.long, device=self.model.device)
        adv = torch.tensor(adv_np, dtype=torch.float32, device=self.model.device)
        lo = torch.tensor(logits_np, dtype=torch.float32, device=self.model.device)
        vt = torch.tensor(vt_np, dtype=torch.float32, device=self.model.device)
        idx = torch.randperm(act.shape[0], device=self.model.device)
        batch = {k: v[idx] for k, v in batch.items()}
        act, adv, lo, vt = act[idx], adv[idx], lo[idx], vt[idx]
        am, asd = adv.mean(), adv.std(unbiased=False).clamp_min(1e-8)
        nupd = self.super_batch_size // self.train_batch_size
        losses = []
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        for u in range(nupd):
            s, e = u * self.train_batch_size, (u + 1) * self.train_batch_size
            mb = {k: v[s:e] for k, v in batch.items()}
            ma, mlo, mvt = act[s:e], lo[s:e], vt[s:e]
            nav = (adv[s:e] - am) / asd
            logits, value = self.model(mb)
            ma_flat = ma.view(ma.shape[0], -1)
            logits_flat = logits.view(logits.shape[0], -1, logits.shape[-1])
            mlo_flat = mlo.view(mlo.shape[0], -1, mlo.shape[-1])
            dnew, dold = Categorical(logits=logits_flat), Categorical(logits=mlo_flat)
            lp = dnew.log_prob(ma_flat)
            with torch.no_grad():
                lpo = dold.log_prob(ma_flat)
            ratio = torch.exp(lp - lpo)
            surr1 = ratio * nav.unsqueeze(-1)
            coeff = torch.exp(-0.5 * (torch.log(ratio.clamp_min(1e-9).detach()) / max(self.sigma, 1e-9)) ** 2)
            p_loss = torch.tensor(0.0, device=self.model.device) if self.global_step < self.policy_train_start_step else -torch.mean(surr1 * coeff)
            v_loss = self.vf_coef * torch.mean((value.float() - mvt) ** 2)
            ent = torch.mean(dnew.entropy()) if self.global_step >= self.policy_train_start_step else torch.tensor(0.0, device=self.model.device)
            kld = torch.mean(kl.kl_divergence(dold, dnew)) if self.global_step >= self.policy_train_start_step else torch.tensor(0.0, device=self.model.device)
            loss = p_loss + v_loss - self.ent_coef * ent + self.kl_coef * kld
            self.model.backward(loss / max(self.accumulation_steps, 1))
            if ((u + 1) % max(self.accumulation_steps, 1) == 0) or (u + 1 == nupd):
                self.model.step()
            self.global_step += 1
            losses.append((float(loss.item()), float(p_loss.item()), float(v_loss.item()), float(ent.item()), float(kld.item())))
        self.model.eval()
        self.total_trainer_samples_consumed += int(nupd * self.train_batch_size)
        a = np.asarray(losses, np.float32)
        return (
            float(a[:, 0].mean()), float(a[:, 1].mean()), float(a[:, 2].mean()), float((-self.ent_coef * a[:, 3]).mean()), float((self.kl_coef * a[:, 4]).mean()), {"policy": self.policy_lr, "value": self.value_lr},
            int(self.global_step), float(a[:, 3].mean()), float(a[:, 4].mean()), {"policy_train_time": time.time() - t0}, int(self.total_trainer_samples_consumed)
        )


def parse_args():
    p = argparse.ArgumentParser(description="Standalone WM GIPO with ds_com + DeepSpeed")
    p.add_argument("--exp-name", type=str, default=f"Standalone_WM_GIPO_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    p.add_argument("--train-iters", type=int, default=20)
    p.add_argument("--train-batch-size", type=int, default=8)
    p.add_argument("--accumulation-steps", type=int, default=2)
    p.add_argument("--num-rollout-workers", type=int, default=1)
    p.add_argument("--num-inference-actors", type=int, default=1)
    p.add_argument("--num-reward-actors", type=int, default=1)
    p.add_argument("--num-denoiser-actors", type=int, default=1)
    p.add_argument("--num-step-cond", type=int, default=4)
    p.add_argument("--imagine-horizon", type=int, default=8)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lambda", dest="lambda_", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--kl-coef", type=float, default=0.0)
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--policy-lr", type=float, default=3e-4)
    p.add_argument("--value-lr", type=float, default=1e-3)
    p.add_argument("--reward-scale", type=float, default=1.0)
    p.add_argument("--broadcast-group-name", type=str, default="accerl-wm-broadcast")
    p.add_argument("--broadcast-backend", type=str, default="gloo", choices=["gloo", "nccl"])
    p.add_argument("--cuda-visible-devices", type=str, default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    p.add_argument("--use-bf16", action="store_true")
    p.add_argument("--torch-extensions-dir", type=str, default="")
    p.add_argument("--prewarm-deepspeed-ops", dest="prewarm_deepspeed_ops", action="store_true")
    p.add_argument("--no-prewarm-deepspeed-ops", dest="prewarm_deepspeed_ops", action="store_false")
    p.set_defaults(prewarm_deepspeed_ops=True)
    p.add_argument("--warmup-timeout-s", type=float, default=300.0)
    p.add_argument("--warmup-stall-timeout-s", type=float, default=90.0)
    p.add_argument("--sync-timeout-s", type=float, default=120.0)
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    torch_ext_dir = resolve_torch_extensions_dir(args.torch_extensions_dir)
    os.environ["TORCH_EXTENSIONS_DIR"] = torch_ext_dir
    print(f"[Runtime] TORCH_EXTENSIONS_DIR={torch_ext_dir}")
    prewarm_deepspeed_ops(args.prewarm_deepspeed_ops)
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        runtime_env={"env_vars": {"TORCH_EXTENSIONS_DIR": torch_ext_dir}},
    )
    writer = SummaryWriter(f"runs/Standalone/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.exp_name}")
    stats_actor = StatsActor.remote(window_size=100)
    replay_buffer = ReplayBufferActor.remote(capacity=5000)
    throughput_tracker = ThroughputTracker(args.exp_name)
    trainer_recorder = TrainerThroughputRecorder(args.exp_name)
    model_dtype = torch.bfloat16 if args.use_bf16 else torch.float32

    trainer = TrainerActor.remote(0, 1, replay_buffer, None, args.train_batch_size, args.accumulation_steps, args.use_bf16, model_dtype, args.policy_lr, args.value_lr, args.gamma, args.lambda_, args.clip_eps, args.vf_coef, args.ent_coef, args.kl_coef, args.reward_scale, 0, 0, 0, args.train_iters, "gipo", args.sigma, False)
    infs = [InferenceActor.remote(i, None, stats_actor, model_dtype, 16, 2.0) for i in range(args.num_inference_actors)]
    rewards = [RewardInferenceActor.remote() for _ in range(args.num_reward_actors)]
    denoisers = [DenoiserInferenceActor.remote() for _ in range(args.num_denoiser_actors)]

    master_addr = ray.get(trainer.get_node_ip.remote())
    master_port = find_free_port()
    ray.get(trainer.setup_deepspeed_group.remote(master_addr, master_port))
    world = 1 + len(infs)
    tasks = [trainer.setup_broadcast_group.remote(master_addr, master_port, args.broadcast_group_name, world, 0, backend=args.broadcast_backend)]
    for i, inf in enumerate(infs):
        tasks.append(inf.setup_broadcast_group.remote(master_addr, master_port, args.broadcast_group_name, world, i + 1, backend=args.broadcast_backend))
    ray.get(tasks)
    sync_weights_blocking(trainer, infs, args.broadcast_group_name, timeout_s=args.sync_timeout_s)

    workers = [
        RolloutWorkerActor.remote(
            infs[i % len(infs)],
            replay_buffer,
            i,
            stats_actor,
            None,
            "fake",
            args.num_step_cond,
            args.imagine_horizon,
            model_dtype,
            rewards[i % len(rewards)],
            denoisers[i % len(denoisers)],
            args.reward_scale,
            args.gamma,
            args.lambda_,
        )
        for i in range(args.num_rollout_workers)
    ]
    rollout_run_refs = [w.run.remote() for w in workers]

    min_items = args.train_batch_size * args.accumulation_steps
    wait_for_replay_ready(
        replay_buffer=replay_buffer,
        min_items=min_items,
        rollout_refs=rollout_run_refs,
        timeout_s=float(args.warmup_timeout_s),
        stall_timeout_s=float(args.warmup_stall_timeout_s),
    )

    for _ in range(args.train_iters):
        m = ray.get(trainer.run_training_epoch.remote())
        loss, p_loss, v_loss, e_loss, kl_loss, lrs, step, ent, kl_div, perf, trainer_samples = m
        sync_weights_blocking(trainer, infs, args.broadcast_group_name, timeout_s=args.sync_timeout_s)
        stats = ray.get(stats_actor.get_stats.remote())
        throughput_tracker.tick(step, ray.get(replay_buffer.size.remote()), trainer_samples)
        trainer_recorder.record_step(step, {"loss": loss, "policy_loss": p_loss, "value_loss": v_loss, "train_time": perf.get("policy_train_time", 0.0)})
        writer.add_scalar("train/loss", loss, step)
        writer.add_scalar("train/policy_loss", p_loss, step)
        writer.add_scalar("train/value_loss", v_loss, step)
        writer.add_scalar("rollout/avg_return", stats["_global_rollout_"]["avg_return"], step)
        writer.add_scalar("rollout/avg_imagine_reward", stats["_global_rollout_"]["avg_imagine_reward"], step)
        print(f"[step {step:04d}] loss={loss:.4f} policy={p_loss:.4f} value={v_loss:.4f} imag_rew={stats['_global_rollout_']['avg_imagine_reward']:.4f}")

    writer.close()
    ray.shutdown()


if __name__ == "__main__":
    main()

