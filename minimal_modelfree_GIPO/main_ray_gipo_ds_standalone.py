from __future__ import annotations

import argparse
import asyncio
import csv
import math
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

from rl.ds_com import InferenceActorCom, TrainerActorCom

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
        print("[DeepSpeed] 跳过预热（--no-prewarm-deepspeed-ops）")
        return
    try:
        from deepspeed.ops.op_builder import CPUAdamBuilder, FusedAdamBuilder
    except Exception as exc:
        print(f"[DeepSpeed] 预热跳过：无法导入 op builder ({exc})")
        return
    print("[DeepSpeed] 开始预热编译 FusedAdam/CPUAdam（首次运行可能较慢）...")
    try:
        FusedAdamBuilder().load(verbose=True)
    except Exception as exc:
        print(f"[DeepSpeed] FusedAdam 预热失败，将在训练时按需处理: {exc}")
    try:
        CPUAdamBuilder().load(verbose=True)
    except Exception as exc:
        print(f"[DeepSpeed] CPUAdam 预热失败，将在训练时按需处理: {exc}")
    print("[DeepSpeed] 预热阶段结束。")


def sync_weights_blocking(trainer_actor, inference_actors, group_name: str, timeout_s: float = 120.0) -> None:
    # 必须并发发起 broadcast 与 receive，避免双方互等导致死锁。
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
    min_steps: int,
    rollout_refs: List,
    timeout_s: float,
    stall_timeout_s: float,
    poll_interval_s: float = 0.5,
) -> None:
    start_t = time.time()
    last_steps = -1
    last_progress_t = start_t
    while True:
        if rollout_refs:
            ready, _ = ray.wait(rollout_refs, num_returns=1, timeout=0.0)
            if ready:
                # worker.run() 本应长期运行；若返回通常表示异常退出，直接抛出详细报错
                ray.get(ready[0])
                raise RuntimeError("A rollout worker exited unexpectedly during replay warmup.")

        steps = int(ray.get(replay_buffer.total_steps.remote()))
        now = time.time()
        if steps >= int(min_steps):
            return
        if steps > last_steps:
            print(f"[Warmup] replay_steps={steps}/{min_steps}")
            last_steps = steps
            last_progress_t = now
        elif now - last_progress_t > stall_timeout_s:
            raise TimeoutError(
                f"Replay warmup stalled for {stall_timeout_s:.1f}s at replay_steps={steps}/{min_steps}. "
                "Check rollout/inference actors."
            )
        if now - start_t > timeout_s:
            raise TimeoutError(
                f"Replay warmup timed out after {timeout_s:.1f}s (replay_steps={steps}/{min_steps})."
            )
        time.sleep(poll_interval_s)


class ThroughputTracker:
    def __init__(self, exp_name: str):
        self.exp_name = exp_name
        self.run_dir = Path("runs") / exp_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / "throughput_log.csv"
        self._start = time.time()
        self._last_t = self._start
        self._last_steps = 0
        self.history = []
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["global_step", "elapsed_s", "replay_total_steps", "trainer_total_samples", "global_sps"])

    def tick(self, global_step: int, replay_total_steps: int, trainer_total_samples: int):
        now = time.time()
        dt = max(now - self._last_t, 1e-6)
        dsteps = max(replay_total_steps - self._last_steps, 0)
        sps = float(dsteps / dt)
        self._last_t = now
        self._last_steps = replay_total_steps
        self.history.append((global_step, now - self._start, replay_total_steps, trainer_total_samples, sps))
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([global_step, now - self._start, replay_total_steps, trainer_total_samples, sps])
        print(
            f"[AcceRL Throughput] step={global_step} replay={replay_total_steps} "
            f"trainer_samples={trainer_total_samples} sps={sps:.2f}"
        )


class TrainerThroughputRecorder:
    def __init__(self, exp_name: str):
        self.run_dir = Path("runs") / exp_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / "trainer_throughput.csv"
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["global_step", "loss", "policy_loss", "value_loss", "train_time_s"])

    def record_step(self, global_step: int, metrics: Dict[str, float]):
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    global_step,
                    metrics.get("loss", 0.0),
                    metrics.get("policy_loss", 0.0),
                    metrics.get("value_loss", 0.0),
                    metrics.get("train_time", 0.0),
                ]
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


class FakeUnnorm:
    def _unnormalize_actions(self, x: np.ndarray, _key: str) -> np.ndarray:
        return np.array(x, copy=True)


class FakeActorCritic(torch.nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.processor = FakeProcessor()
        self.vla = FakeUnnorm()
        img_dim = 3 * 224 * 224
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(img_dim + 7, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Tanh(),
        )
        out = NUM_ACTIONS_CHUNK * ACTION_DIM * ACTION_BINS
        self.policy_head = torch.nn.Linear(hidden_dim, out)
        self.value_head = torch.nn.Linear(hidden_dim, 1)

    def prepare_inputs_batch(self, inputs_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        images = torch.stack([x["image"].float().reshape(-1) for x in inputs_list], dim=0).to(device)
        proprios = torch.stack([x["proprio"].float().reshape(-1) for x in inputs_list], dim=0).to(device)
        return {"image": images, "proprio": proprios}

    def forward(self, inputs_batch: Dict[str, torch.Tensor]):
        feat = torch.cat([inputs_batch["image"], inputs_batch["proprio"]], dim=-1)
        h = self.encoder(feat)
        logits = self.policy_head(h).view(-1, NUM_ACTIONS_CHUNK * ACTION_DIM, ACTION_BINS)
        value = self.value_head(h).squeeze(-1)
        return logits, value

    def post_process(self, logits: torch.Tensor, deterministic=False):
        if isinstance(deterministic, list):
            det_flags = deterministic
        else:
            det_flags = [bool(deterministic)] * logits.shape[0]
        greedy = logits.argmax(dim=-1)
        dist = Categorical(logits=logits)
        sampled = dist.sample()
        chosen = []
        for i, d in enumerate(det_flags):
            chosen.append(greedy[i] if d else sampled[i])
        action_tokens = torch.stack(chosen, dim=0).view(-1, NUM_ACTIONS_CHUNK, ACTION_DIM)
        normalized_actions = action_tokens.float() / max(ACTION_BINS - 1, 1) * 2.0 - 1.0
        return None, action_tokens.view(-1, NUM_ACTIONS_CHUNK * ACTION_DIM), normalized_actions

    def get_parameter_groups(self):
        return [
            {"name": "policy", "params": list(self.encoder.parameters()) + list(self.policy_head.parameters())},
            {"name": "value", "params": list(self.value_head.parameters())},
        ]

    def save_model(self, ckpt_dir: str, epoch: int):
        torch.save(self.state_dict(), os.path.join(ckpt_dir, f"agent_extra_layers_epoch_{epoch}.pt"))


class FakeEnv:
    def __init__(self, task_id: int, max_steps: int = 50):
        self.task_id = int(task_id)
        self.max_steps = int(max_steps)
        self.task_description = f"fake-task-{task_id}"
        self.t = 0
        self._rng = np.random.default_rng(0)

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.t = 0
        return self._obs(), {}

    def _obs(self):
        return {
            "image": self._rng.standard_normal((3, 224, 224)).astype(np.float32),
            "proprio": self._rng.standard_normal((7,)).astype(np.float32),
            "task_description": self.task_description,
        }

    def step(self, action):
        _ = action
        self.t += 1
        obs = self._obs()
        reward = float(self._rng.normal(loc=0.1, scale=0.3))
        terminated = bool(self.t >= self.max_steps or self._rng.random() < 0.03)
        truncated = False
        info = {"is_success": float(reward > 0.0)}
        return obs, reward, terminated, truncated, info

    def get_name(self):
        return f"FakeTask-{self.task_id}"


def prepare_one_obs(_cfg, processor: FakeProcessor, obs: Dict, task_description: str, torch_dtype):
    return processor(obs, task_description, torch_dtype=torch_dtype)


@dataclass
class Trajectory:
    obs_list: List[Dict[str, torch.Tensor]]
    action_tokens: np.ndarray
    rewards: np.ndarray
    behaviour_logits: np.ndarray
    old_values: np.ndarray
    bootstrap_value: float
    is_terminal: bool

    @property
    def num_steps(self):
        return len(self.rewards)


@ray.remote
class StatsActor:
    def __init__(self, window_size: int = 100):
        self.window_size = int(window_size)
        self.stats = defaultdict(lambda: {"returns": deque(maxlen=window_size), "lens": deque(maxlen=window_size), "succ": deque(maxlen=window_size)})
        self.infer_lat = deque(maxlen=window_size * 20)

    def add_episode_return(self, env_name, ep_return, step_time, ep_len, success, actor_id=None, step_num=None):
        _ = step_time, actor_id, step_num
        s = self.stats[env_name]
        s["returns"].append(float(ep_return))
        s["lens"].append(int(ep_len))
        s["succ"].append(float(success))

    def add_inference_latency_batch_ms(self, lat):
        self.infer_lat.extend([float(x) for x in lat])

    def add_timing_metric(self, name, val):
        _ = name, val

    def get_stats(self):
        all_ret, all_len, all_succ = [], [], []
        for v in self.stats.values():
            all_ret += list(v["returns"])
            all_len += list(v["lens"])
            all_succ += list(v["succ"])
        return {
            "_global_": {
                "avg_return": float(np.mean(all_ret)) if all_ret else 0.0,
                "avg_ep_len": float(np.mean(all_len)) if all_len else 0.0,
                "avg_success_rate": float(np.mean(all_succ)) if all_succ else 0.0,
            },
            "_timings_": {
                "inference_p95_ms": float(np.percentile(np.asarray(self.infer_lat, dtype=np.float32), 95)) if self.infer_lat else 0.0
            },
        }


@ray.remote
class ReplayBufferActor:
    def __init__(self, capacity: int = 512):
        self.trajectories = deque(maxlen=int(capacity))

    def add_trajectory(self, traj: Trajectory):
        self.trajectories.append(traj)

    def total_steps(self):
        return int(sum(x.num_steps for x in self.trajectories))

    def sample_trajectories(self, min_steps: int):
        idx = list(range(len(self.trajectories)))
        random.shuffle(idx)
        out, total = [], 0
        for i in idx:
            out.append(self.trajectories[i])
            total += self.trajectories[i].num_steps
            if total >= min_steps:
                break
        return out


@ray.remote
class RolloutWorkerActor:
    def __init__(self, infer, replay, wid, stats_actor, cfg, benchmark_name, reward_scale, torch_dtype, rollout_local_buf):
        _ = cfg, benchmark_name
        self.infer, self.replay, self.wid, self.stats_actor = infer, replay, int(wid), stats_actor
        self.processor = FakeProcessor()
        self.envs = [FakeEnv(task_id=i) for i in range(10)]
        self.env = None
        self.current_env_idx = -1
        self.task_description = ""
        self.current_env_name = ""
        self.local_buffer = []
        self.reward_scale = float(reward_scale)
        self.torch_dtype = torch_dtype
        self.rollout_local_buf = int(rollout_local_buf)

    def _reset_and_select_env(self, seed=None):
        self.current_env_idx = random.randint(0, len(self.envs) - 1)
        self.env = self.envs[self.current_env_idx]
        obs, info = self.env.reset(seed=seed)
        self.task_description = self.env.task_description
        self.current_env_name = self.env.get_name()
        return obs, info

    def _process_traj(self, seg, bootstrap_val: float, is_terminal: bool):
        traj = Trajectory(
            obs_list=[s for s, _, _, _, _ in seg],
            action_tokens=np.stack([a for _, a, _, _, _ in seg]).astype(np.int64),
            rewards=np.asarray([r for _, _, r, _, _ in seg], dtype=np.float32),
            behaviour_logits=np.stack([l for _, _, _, l, _ in seg]).astype(np.float32),
            old_values=np.asarray([v for _, _, _, _, v in seg], dtype=np.float32),
            bootstrap_value=float(bootstrap_val),
            is_terminal=bool(is_terminal),
        )
        self.replay.add_trajectory.remote(traj)

    def run(self):
        seed = int(time.time() * 1000) + self.wid
        obs, _ = self._reset_and_select_env(seed=seed)
        reward_sum, step_count = 0.0, 0
        while True:
            inputs_t = prepare_one_obs(None, self.processor, obs, self.task_description, self.torch_dtype)
            try:
                action_env, action_token, logits, value = ray.get(
                    self.infer.request.remote(inputs_t, deterministic=False), timeout=30.0
                )
            except GetTimeoutError as exc:
                raise TimeoutError(
                    f"RolloutWorker {self.wid} timed out waiting for InferenceActor response."
                ) from exc
            chunk_reward, done = 0.0, False
            for i in range(len(action_env)):
                nxt, r, term, trunc, info = self.env.step(action_env[i])
                reward_sum += r
                chunk_reward += r * self.reward_scale
                step_count += 1
                if term or trunc:
                    done = True
                    break
            self.local_buffer.append((inputs_t, action_token, chunk_reward, logits, value))
            obs = nxt
            if done:
                self.stats_actor.add_episode_return.remote(self.current_env_name, reward_sum, 0.0, step_count, float(info.get("is_success", 0.0)))
                if self.local_buffer:
                    self._process_traj(self.local_buffer, 0.0, True)
                self.local_buffer.clear()
                reward_sum, step_count = 0.0, 0
                seed = int(time.time() * 1000) + self.wid + step_count
                obs, _ = self._reset_and_select_env(seed=seed)
            elif len(self.local_buffer) >= self.rollout_local_buf + 1:
                self._process_traj(self.local_buffer[:-1], float(self.local_buffer[-1][-1]), False)
                self.local_buffer = [self.local_buffer[-1]]


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
        self.cfg = argparse.Namespace(unnorm_key="fake")
        self.stats_actor = stats_actor
        self.batch_size = int(inference_batch)
        self.timeout_sec = float(inference_timeout_ms) / 1000.0
        self.requests, self.promises = [], []
        self.last_process_time = time.time()
        loop = asyncio.get_event_loop()
        self._bg_task = loop.create_task(self._loop())
        self._bg_task.add_done_callback(self._on_bg_task_done)

    def _on_bg_task_done(self, task: asyncio.Task):
        try:
            task.result()
        except Exception as exc:
            import traceback
            print(f"[ERROR] InferenceActor {self.actor_id} background loop crashed: {exc}", flush=True)
            traceback.print_exc()

    async def request(self, inputs_t: Dict[str, torch.Tensor], deterministic: bool = False):
        fut = asyncio.get_event_loop().create_future()
        self.requests.append((inputs_t, deterministic, time.perf_counter()))
        self.promises.append(fut)
        return await fut

    async def _loop(self):
        while True:
            should_process = self.requests and (
                len(self.requests) >= self.batch_size or time.time() - self.last_process_time > self.timeout_sec
            )
            if not should_process:
                await asyncio.sleep(0.0005)
                continue
            reqs, pros = self.requests, self.promises
            self.requests, self.promises = [], []
            self.last_process_time = time.time()
            try:
                inputs_batch = self.model.prepare_inputs_batch([r[0] for r in reqs])
                with torch.inference_mode():
                    action_logits, value = self.model(inputs_batch)
                    _, action_tokens_all, normalized_actions_all = self.model.post_process(
                        action_logits, deterministic=[r[1] for r in reqs]
                    )
                    action_tokens = action_tokens_all.view(-1, NUM_ACTIONS_CHUNK, ACTION_DIM).cpu().numpy()
                    logits = action_logits.view(-1, NUM_ACTIONS_CHUNK, ACTION_DIM, ACTION_BINS).float().cpu().numpy()
                    values = value.float().cpu().numpy()
                actions_env = [
                    self.model.vla._unnormalize_actions(
                        normalized_actions_all[i].cpu().numpy(), self.cfg.unnorm_key
                    ).astype(np.float32)
                    for i in range(normalized_actions_all.shape[0])
                ]
                now = time.perf_counter()
                lat = []
                for i, p in enumerate(pros):
                    p.set_result((actions_env[i], action_tokens[i], logits[i], values[i]))
                    lat.append((now - reqs[i][2]) * 1000.0)
                self.stats_actor.add_inference_latency_batch_ms.remote(lat)
            except Exception as exc:
                import traceback
                print(f"[ERROR] InferenceActor {self.actor_id} batch failed: {exc}", flush=True)
                traceback.print_exc()
                for p in pros:
                    if not p.done():
                        p.set_exception(exc)


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
        self.sigma = float(sigma)
        self.recompute_value = bool(recompute_value)
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
        pg = model.get_parameter_groups()
        optimizer_params = [{"params": x["params"], "name": x["name"], "lr": self.policy_lr if x["name"] == "policy" else self.value_lr} for x in pg]
        ds_config = {
            "train_micro_batch_size_per_gpu": self.train_batch_size,
            "gradient_accumulation_steps": self.accumulation_steps,
            "optimizer": {"type": "AdamW", "params": {}},
            "bf16": {"enabled": self.use_bf16},
            "zero_optimization": {"stage": 2, "allgather_partitions": True, "allgather_bucket_size": 5e8, "reduce_scatter": True, "reduce_bucket_size": 5e8, "overlap_comm": True, "contiguous_gradients": True},
            "gradient_clipping": 1.0,
        }
        self.model, self.optimizer, _, _ = deepspeed.initialize(model=model, config=ds_config, model_parameters=optimizer_params)

    def get_node_ip(self):
        return ray.util.get_node_ip_address()

    def _compute_gae(self, rewards, values, bootstrap_value, is_terminal):
        T = rewards.shape[0]
        adv = torch.zeros(T, dtype=torch.float32, device=values.device)
        ret = torch.zeros(T, dtype=torch.float32, device=values.device)
        last_value = 0.0 if is_terminal else float(bootstrap_value)
        gae = 0.0
        for i in reversed(range(T)):
            nxt = last_value if i == T - 1 else float(values[i + 1].item())
            delta = float(rewards[i].item()) + self.gamma * nxt - float(values[i].item())
            gae = delta + self.gamma * self.lambda_ * gae
            adv[i] = gae
            ret[i] = gae + float(values[i].item())
        return adv, ret

    async def run_training_epoch(self):
        while ray.get(self.replay_buffer.total_steps.remote()) < self.super_batch_size:
            await asyncio.sleep(0.1)
        t0 = time.time()
        trajs = ray.get(self.replay_buffer.sample_trajectories.remote(self.super_batch_size))
        obs_list = []
        for tr in trajs:
            obs_list.extend(tr.obs_list)
        old_values = torch.tensor(np.concatenate([t.old_values for t in trajs], axis=0), dtype=torch.float32, device=self.model.device)
        actions, beh_logits, advs, rets = [], [], [], []
        off = 0
        for tr in trajs:
            tv = old_values[off: off + tr.num_steps]
            off += tr.num_steps
            rew = torch.tensor(tr.rewards, dtype=torch.float32, device=self.model.device)
            a, r = self._compute_gae(rew, tv, tr.bootstrap_value, tr.is_terminal)
            actions.append(torch.tensor(tr.action_tokens, dtype=torch.long, device=self.model.device))
            beh_logits.append(torch.tensor(tr.behaviour_logits, dtype=torch.float32, device=self.model.device))
            advs.append(a)
            rets.append(r)
        actions = torch.cat(actions, 0)
        beh_logits = torch.cat(beh_logits, 0)
        advs = torch.cat(advs, 0)
        rets = torch.cat(rets, 0)
        idx = torch.randperm(actions.shape[0], device=self.model.device)
        obs_list = [obs_list[i] for i in idx.cpu().tolist()]
        actions, beh_logits, advs, rets = actions[idx], beh_logits[idx], advs[idx], rets[idx]
        am, asd = advs.mean(), advs.std(unbiased=False).clamp_min(1e-8)
        self.model.train()
        losses = []
        nupd = max(1, actions.shape[0] // self.train_batch_size)
        self.optimizer.zero_grad(set_to_none=True)
        for u in range(nupd):
            s, e = u * self.train_batch_size, (u + 1) * self.train_batch_size
            batch_obs = self.base_model.prepare_inputs_batch(obs_list[s:e])
            bat_a, bat_l, bat_r = actions[s:e], beh_logits[s:e], rets[s:e]
            nav = (advs[s:e] - am) / asd
            cl, cv = self.model(batch_obs)
            bat_a_flat = bat_a.view(bat_a.shape[0], -1)
            cl_flat = cl.view(cl.shape[0], -1, cl.shape[-1])
            bat_l_flat = bat_l.view(bat_l.shape[0], -1, bat_l.shape[-1])
            dnew, dold = Categorical(logits=cl_flat), Categorical(logits=bat_l_flat)
            lp = dnew.log_prob(bat_a_flat)
            with torch.no_grad():
                lpo = dold.log_prob(bat_a_flat)
            ratio = torch.exp(lp - lpo)
            surr1 = ratio * nav.unsqueeze(-1)
            coeff = torch.exp(-0.5 * (torch.log(ratio.clamp_min(1e-9).detach()) / max(self.sigma, 1e-9)) ** 2)
            p_loss = -torch.mean(surr1 * coeff)
            v_loss = self.vf_coef * torch.mean((cv.float() - bat_r) ** 2)
            ent = torch.mean(dnew.entropy())
            kl_div = torch.mean(kl.kl_divergence(dold, dnew))
            loss = p_loss + v_loss - self.ent_coef * ent + self.kl_coef * kl_div
            self.model.backward(loss / max(self.accumulation_steps, 1))
            if ((u + 1) % max(self.accumulation_steps, 1) == 0) or (u + 1 == nupd):
                self.model.step()
            self.global_step += 1
            losses.append((float(loss.item()), float(p_loss.item()), float(v_loss.item()), float(ent.item()), float(kl_div.item())))
        self.model.eval()
        self.total_trainer_samples_consumed += int(nupd * self.train_batch_size)
        arr = np.asarray(losses, dtype=np.float32)
        return (
            float(arr[:, 0].mean()), float(arr[:, 1].mean()), float(arr[:, 2].mean()), float((-self.ent_coef * arr[:, 3]).mean()),
            float((self.kl_coef * arr[:, 4]).mean()), {"policy": self.policy_lr, "value": self.value_lr},
            int(self.global_step), float(arr[:, 3].mean()), float(arr[:, 4].mean()),
            {"policy_sample_time": 0.0, "policy_prep_time": 0.0, "policy_train_time": time.time() - t0},
            int(self.total_trainer_samples_consumed)
        )


def parse_args():
    p = argparse.ArgumentParser(description="Standalone minimal GIPO with DeepSpeed and ds_com broadcast")
    p.add_argument("--exp-name", type=str, default=f"Standalone_GIPO_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    p.add_argument("--train-iters", type=int, default=20)
    p.add_argument("--num-rollout-workers", type=int, default=1)
    p.add_argument("--num-eval-workers", type=int, default=0)
    p.add_argument("--num-inference-actors", type=int, default=1)
    p.add_argument("--train-batch-size", type=int, default=16)
    p.add_argument("--accumulation-steps", type=int, default=2)
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
    p.add_argument("--rollout-local-buf", type=int, default=16)
    p.add_argument("--broadcast-group-name", type=str, default="accerl-broadcast")
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
    log_dir = f"runs/Standalone/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.exp_name}"
    writer = SummaryWriter(log_dir)
    stats_actor = StatsActor.remote(window_size=100)
    replay_buffer = ReplayBufferActor.remote(capacity=2048)
    throughput_tracker = ThroughputTracker(args.exp_name)
    trainer_recorder = TrainerThroughputRecorder(args.exp_name)
    model_dtype = torch.bfloat16 if args.use_bf16 else torch.float32

    trainer_group = [TrainerActor.remote(0, 1, replay_buffer, None, args.train_batch_size, args.accumulation_steps, args.use_bf16, model_dtype, args.policy_lr, args.value_lr, args.gamma, args.lambda_, args.clip_eps, args.vf_coef, args.ent_coef, args.kl_coef, args.reward_scale, 0, 0, 0, args.train_iters, "gipo", args.sigma, False)]
    inference_pool = [InferenceActor.remote(i, None, stats_actor, model_dtype, 16, 2.0) for i in range(args.num_inference_actors)]

    master_addr = ray.get(trainer_group[0].get_node_ip.remote())
    master_port = find_free_port()
    ray.get(trainer_group[0].setup_deepspeed_group.remote(master_addr, master_port))

    group_world_size = 1 + len(inference_pool)
    rank_tasks = [trainer_group[0].setup_broadcast_group.remote(master_addr, master_port, args.broadcast_group_name, group_world_size, 0, backend=args.broadcast_backend)]
    for i, actor in enumerate(inference_pool):
        rank_tasks.append(actor.setup_broadcast_group.remote(master_addr, master_port, args.broadcast_group_name, group_world_size, i + 1, backend=args.broadcast_backend))
    ray.get(rank_tasks)

    sync_weights_blocking(
        trainer_group[0], inference_pool, args.broadcast_group_name, timeout_s=args.sync_timeout_s
    )

    workers = [
        RolloutWorkerActor.remote(
            inference_pool[i % len(inference_pool)],
            replay_buffer,
            i,
            stats_actor,
            None,
            "fake",
            args.reward_scale,
            model_dtype,
            args.rollout_local_buf,
        )
        for i in range(args.num_rollout_workers)
    ]
    rollout_run_refs = [w.run.remote() for w in workers]

    min_steps = args.train_batch_size * args.accumulation_steps
    wait_for_replay_ready(
        replay_buffer=replay_buffer,
        min_steps=min_steps,
        rollout_refs=rollout_run_refs,
        timeout_s=float(args.warmup_timeout_s),
        stall_timeout_s=float(args.warmup_stall_timeout_s),
    )

    for _ in range(args.train_iters):
        metrics = ray.get(trainer_group[0].run_training_epoch.remote())
        loss, p_loss, v_loss, e_loss, kl_loss, lrs, global_step, ent, kl_div, perf, trainer_total_samples = metrics
        sync_weights_blocking(
            trainer_group[0], inference_pool, args.broadcast_group_name, timeout_s=args.sync_timeout_s
        )
        stats = ray.get(stats_actor.get_stats.remote())
        replay_total_steps = ray.get(replay_buffer.total_steps.remote())
        throughput_tracker.tick(global_step, replay_total_steps, trainer_total_samples)
        trainer_recorder.record_step(global_step, {"loss": loss, "policy_loss": p_loss, "value_loss": v_loss, "train_time": perf.get("policy_train_time", 0.0)})
        writer.add_scalar("train/loss", loss, global_step)
        writer.add_scalar("train/policy_loss", p_loss, global_step)
        writer.add_scalar("train/value_loss", v_loss, global_step)
        writer.add_scalar("train/entropy", ent, global_step)
        writer.add_scalar("train/kl_div", kl_div, global_step)
        writer.add_scalar("rollout/avg_return", stats["_global_"]["avg_return"], global_step)
        print(f"[step {global_step:04d}] loss={loss:.4f} policy={p_loss:.4f} value={v_loss:.4f} return={stats['_global_']['avg_return']:.3f}")

    writer.close()
    ray.shutdown()


if __name__ == "__main__":
    main()

