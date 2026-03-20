from __future__ import annotations

import argparse
import os
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.distributions import Categorical, kl_divergence

try:
    import ray
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "main_ray_gipo.py requires `ray`. Install dependencies with: pip install ray torch numpy"
    ) from exc

from fake_env import FakeEnv
from fake_model import (
    ACTION_BINS,
    ACTION_DIM,
    NUM_ACTIONS_CHUNK,
    FakeActorCritic,
    FakeModelConfig,
)


@dataclass
class Trajectory:
    obs_list: List[Dict[str, np.ndarray]]
    action_tokens: np.ndarray
    rewards: np.ndarray
    behaviour_logits: np.ndarray
    old_values: np.ndarray
    bootstrap_value: float
    is_terminal: bool

    @property
    def num_steps(self) -> int:
        return len(self.rewards)


@ray.remote
class StatsActor:
    def __init__(self, window_size: int = 100):
        self.window_size = int(window_size)
        self.stats = defaultdict(
            lambda: {
                "episode_returns": deque(maxlen=self.window_size),
                "episode_lengths": deque(maxlen=self.window_size),
                "successes": deque(maxlen=self.window_size),
                "step_rewards": deque(maxlen=self.window_size),
                "total_episodes": 0,
                "total_env_steps": 0,
            }
        )
        self.timings = defaultdict(lambda: deque(maxlen=self.window_size))
        self.inference_latencies_ms = deque(maxlen=max(self.window_size * 20, 512))

    def add_episode_return(
        self,
        env_name: str,
        ep_return: float,
        ep_length: int,
        success: float,
    ) -> None:
        record = self.stats[env_name]
        record["episode_returns"].append(float(ep_return))
        record["episode_lengths"].append(int(ep_length))
        record["successes"].append(float(success))
        record["step_rewards"].append(float(ep_return) / max(int(ep_length), 1))
        record["total_episodes"] += 1
        record["total_env_steps"] += int(ep_length)

    def add_timing_metric(self, metric_name: str, value: float) -> None:
        self.timings[metric_name].append(float(value))

    def add_inference_latency_ms(self, latency_ms: float) -> None:
        self.inference_latencies_ms.append(float(latency_ms))

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        result = {}
        all_returns = []
        all_lengths = []
        all_successes = []
        for env_name, record in self.stats.items():
            returns = list(record["episode_returns"])
            lengths = list(record["episode_lengths"])
            successes = list(record["successes"])
            step_rewards = list(record["step_rewards"])
            result[env_name] = {
                "avg_return": float(np.mean(returns)) if returns else 0.0,
                "avg_ep_len": float(np.mean(lengths)) if lengths else 0.0,
                "avg_success_rate": float(np.mean(successes)) if successes else 0.0,
                "avg_step_reward": float(np.mean(step_rewards)) if step_rewards else 0.0,
                "total_episodes": int(record["total_episodes"]),
                "total_env_steps": int(record["total_env_steps"]),
            }
            all_returns.extend(returns)
            all_lengths.extend(lengths)
            all_successes.extend(successes)

        result["_global_"] = {
            "avg_return": float(np.mean(all_returns)) if all_returns else 0.0,
            "avg_ep_len": float(np.mean(all_lengths)) if all_lengths else 0.0,
            "avg_success_rate": float(np.mean(all_successes)) if all_successes else 0.0,
            "num_envs": len(self.stats),
        }
        result["_timings_"] = {
            name: float(np.mean(values)) if values else 0.0
            for name, values in self.timings.items()
        }
        if self.inference_latencies_ms:
            result["_timings_"]["inference_p95_ms"] = float(
                np.percentile(np.asarray(self.inference_latencies_ms, dtype=np.float32), 95)
            )
        else:
            result["_timings_"]["inference_p95_ms"] = 0.0
        return result


@ray.remote
class ReplayBufferActor:
    def __init__(self, capacity: int):
        self.trajectories = deque(maxlen=int(capacity))
        self.total_ingested_steps = 0

    def add_trajectory(self, traj: Trajectory) -> None:
        self.trajectories.append(traj)
        self.total_ingested_steps += int(traj.num_steps)

    def size(self) -> int:
        return len(self.trajectories)

    def total_steps(self) -> int:
        return int(sum(traj.num_steps for traj in self.trajectories))

    def total_ingested_steps_count(self) -> int:
        return int(self.total_ingested_steps)

    def sample_trajectories(self, min_steps: int) -> List[Trajectory]:
        if not self.trajectories:
            return []
        order = list(range(len(self.trajectories)))
        random.shuffle(order)
        sampled = []
        total = 0
        for idx in order:
            traj = self.trajectories[idx]
            sampled.append(traj)
            total += traj.num_steps
            if total >= int(min_steps):
                break
        return sampled


class BaseWorkerActor:
    def __init__(
        self,
        infer,
        wid: str | int,
        stats_actor,
        num_tasks: int,
        obs_dim: int,
        action_dim: int,
        env_max_steps: int,
    ) -> None:
        self.infer = infer
        self.wid = wid
        self.stats_actor = stats_actor
        self.num_tasks = int(num_tasks)
        self.envs = [
            FakeEnv(
                task_id=task_id,
                obs_dim=obs_dim,
                action_dim=action_dim,
                max_steps=env_max_steps,
            )
            for task_id in range(self.num_tasks)
        ]
        self.env = None
        self.current_env_idx = -1
        self.current_env_name = ""
        self.task_description = ""

    def _set_current_env(self, env_idx: int, seed: Optional[int] = None):
        self.current_env_idx = int(env_idx)
        self.env = self.envs[self.current_env_idx]
        obs, info = self.env.reset(seed=seed)
        self.current_env_name = self.env.get_name()
        self.task_description = self.env.task_description
        return obs, info


@ray.remote
class RolloutWorkerActor(BaseWorkerActor):
    def __init__(
        self,
        infer,
        replay,
        wid: int,
        stats_actor,
        num_tasks: int,
        obs_dim: int,
        action_dim: int,
        env_max_steps: int,
        reward_scale: float,
        rollout_local_buf: int,
    ) -> None:
        super().__init__(infer, wid, stats_actor, num_tasks, obs_dim, action_dim, env_max_steps)
        self.replay = replay
        self.reward_scale = float(reward_scale)
        self.rollout_local_buf = int(rollout_local_buf)
        self.local_buffer = []
        self.env_failures = [deque(maxlen=100) for _ in range(self.num_tasks)]

    def _reset_and_select_env(self, seed: Optional[int] = None):
        failure_counts = np.asarray([sum(hist) for hist in self.env_failures], dtype=np.float32)
        probs = failure_counts + 1.0
        probs = probs / probs.sum()
        env_idx = int(np.random.choice(self.num_tasks, p=probs))
        return self._set_current_env(env_idx, seed=seed)

    def _process_traj(self, traj_segment, bootstrap_val: float, is_terminal: bool) -> None:
        traj = Trajectory(
            obs_list=[obs for obs, _, _, _, _ in traj_segment],
            action_tokens=np.stack([tokens for _, tokens, _, _, _ in traj_segment]).astype(np.int64),
            rewards=np.asarray([reward for _, _, reward, _, _ in traj_segment], dtype=np.float32),
            behaviour_logits=np.stack([logits for _, _, _, logits, _ in traj_segment]).astype(np.float32),
            old_values=np.asarray([value for _, _, _, _, value in traj_segment], dtype=np.float32),
            bootstrap_value=float(bootstrap_val),
            is_terminal=bool(is_terminal),
        )
        self.replay.add_trajectory.remote(traj)

    def run(self) -> None:
        seed = int(time.time() * 1000) + int(self.wid)
        obs, _ = self._reset_and_select_env(seed=seed)
        reward_sum = 0.0
        step_count_total = 0
        rollout_steps = 0
        while True:
            action_env, action_tokens, logits, value = ray.get(
                self.infer.request.remote(obs, deterministic=False)
            )
            chunk_reward = 0.0
            done = False
            next_obs = obs
            info = {}
            for chunk_idx in range(len(action_env)):
                next_obs, reward, terminated, truncated, info = self.env.step(action_env[chunk_idx])
                reward_sum += reward
                chunk_reward += reward * self.reward_scale
                step_count_total += 1
                if terminated or truncated:
                    done = True
                    break

            self.local_buffer.append((obs, action_tokens, chunk_reward, logits, value))
            rollout_steps += 1
            obs = next_obs

            if done:
                success = float(info.get("is_success", 0.0))
                self.env_failures[self.current_env_idx].append(1.0 - success)
                self.stats_actor.add_episode_return.remote(
                    self.current_env_name, reward_sum, step_count_total, success
                )
                if self.local_buffer:
                    self._process_traj(self.local_buffer, bootstrap_val=0.0, is_terminal=True)
                self.local_buffer.clear()
                reward_sum = 0.0
                step_count_total = 0
                seed = int(time.time() * 1000) + int(self.wid) + rollout_steps
                obs, _ = self._reset_and_select_env(seed=seed)
            elif len(self.local_buffer) >= self.rollout_local_buf + 1:
                bootstrap_value = self.local_buffer[-1][-1]
                self._process_traj(
                    self.local_buffer[:-1],
                    bootstrap_val=float(bootstrap_value),
                    is_terminal=False,
                )
                self.local_buffer = [self.local_buffer[-1]]


@ray.remote
class EvaluationWorkerActor(BaseWorkerActor):
    def __init__(
        self,
        infer,
        wid: str,
        stats_actor,
        num_tasks: int,
        obs_dim: int,
        action_dim: int,
        env_max_steps: int,
    ) -> None:
        super().__init__(infer, wid, stats_actor, num_tasks, obs_dim, action_dim, env_max_steps)

    def _reset_and_select_env(self, seed: Optional[int] = None):
        env_idx = (self.current_env_idx + 1) % self.num_tasks
        return self._set_current_env(env_idx, seed=seed)

    def run(self) -> None:
        seed = int(time.time() * 1000)
        obs, _ = self._reset_and_select_env(seed=seed)
        while True:
            reward_sum = 0.0
            step_count_total = 0
            done = False
            while not done:
                action_env, _, _, _ = ray.get(self.infer.request.remote(obs, deterministic=True))
                for chunk_idx in range(len(action_env)):
                    obs, reward, terminated, truncated, info = self.env.step(action_env[chunk_idx])
                    reward_sum += reward
                    step_count_total += 1
                    if terminated or truncated:
                        done = True
                        break
            success = float(info.get("is_success", 0.0))
            self.stats_actor.add_episode_return.remote(
                f"eval_{self.current_env_name}",
                reward_sum,
                step_count_total,
                success,
            )
            seed = int(time.time() * 1000) + step_count_total
            obs, _ = self._reset_and_select_env(seed=seed)


@ray.remote
class InferenceActor:
    def __init__(
        self,
        actor_id: int,
        model_config: Dict,
        stats_actor=None,
        device: str = "cpu",
    ) -> None:
        self.actor_id = int(actor_id)
        self.device = torch.device(device)
        self.model = FakeActorCritic(FakeModelConfig(**model_config)).to(self.device)
        self.model.eval()
        self.stats_actor = stats_actor

    def request(self, obs: Dict[str, np.ndarray], deterministic: bool = False):
        start = time.perf_counter()
        obs_batch = self.model.prepare_inputs_batch([obs])
        with torch.no_grad():
            logits, values = self.model(obs_batch)
            action_tokens, continuous_actions = self.model.post_process(
                logits, deterministic=deterministic
            )
        if self.stats_actor is not None:
            self.stats_actor.add_inference_latency_ms.remote((time.perf_counter() - start) * 1000.0)
        return (
            continuous_actions[0].cpu().numpy().astype(np.float32),
            action_tokens[0].cpu().numpy().astype(np.int64),
            logits[0].cpu().numpy().astype(np.float32),
            float(values[0].item()),
        )

    def get_weights(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

    def set_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state_dict)
        self.model.eval()


@ray.remote
class TrainerActor:
    def __init__(
        self,
        replay_buffer,
        model_config: Dict,
        train_batch_size: int,
        accumulation_steps: int,
        policy_lr: float,
        value_lr: float,
        gamma: float,
        lambda_: float,
        clip_eps: float,
        vf_coef: float,
        ent_coef: float,
        kl_coef: float,
        train_iters: int,
        sync_stats_with_ddp: bool = False,
        recompute_value: bool = False,
        device: str = "cpu",
    ) -> None:
        self.replay_buffer = replay_buffer
        self.device = torch.device(device)
        self.model = FakeActorCritic(FakeModelConfig(**model_config)).to(self.device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.model.encoder.parameters(), "lr": policy_lr},
                {"params": self.model.policy_head.parameters(), "lr": policy_lr},
                {"params": self.model.value_head.parameters(), "lr": value_lr},
            ]
        )
        self.train_batch_size = int(train_batch_size)
        self.accumulation_steps = int(accumulation_steps)
        self.super_batch_size = self.train_batch_size * self.accumulation_steps
        self.policy_lr = float(policy_lr)
        self.value_lr = float(value_lr)
        self.gamma = float(gamma)
        self.lambda_ = float(lambda_)
        self.clip_eps = float(clip_eps)
        self.vf_coef = float(vf_coef)
        self.ent_coef = float(ent_coef)
        self.kl_coef = float(kl_coef)
        self.train_iters = int(train_iters)
        self.sync_stats_with_ddp = bool(sync_stats_with_ddp)
        self.recompute_value = bool(recompute_value)
        self.global_step = 0
        self.total_trainer_samples_consumed = 0

    def _get_current_lr(self, current_step: int, peak_lr: float, total_steps: int) -> float:
        if total_steps <= 1:
            return peak_lr
        progress = min(max(current_step / total_steps, 0.0), 1.0)
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        return float(peak_lr * cosine_decay)

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        bootstrap_value: float,
        is_terminal: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        traj_len = rewards.shape[0]
        advantages = torch.zeros(traj_len, dtype=torch.float32, device=values.device)
        returns = torch.zeros(traj_len, dtype=torch.float32, device=values.device)
        last_value = 0.0 if is_terminal else float(bootstrap_value)
        gae = 0.0
        for idx in reversed(range(traj_len)):
            next_value = last_value if idx == traj_len - 1 else float(values[idx + 1].item())
            delta = float(rewards[idx].item()) + self.gamma * next_value - float(values[idx].item())
            gae = delta + self.gamma * self.lambda_ * gae
            advantages[idx] = gae
            returns[idx] = gae + float(values[idx].item())
        return advantages, returns

    def _global_adv_stats(self, advantages: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        total_sum = advantages.sum()
        total_sq_sum = (advantages * advantages).sum()
        total_count = torch.tensor(float(advantages.numel()), device=advantages.device)
        if self.sync_stats_with_ddp and dist.is_available() and dist.is_initialized():
            stacked = torch.stack([total_sum, total_sq_sum, total_count])
            dist.all_reduce(stacked, op=dist.ReduceOp.SUM)
            total_sum, total_sq_sum, total_count = stacked
        mean = total_sum / torch.clamp(total_count, min=1.0)
        var = torch.clamp(total_sq_sum / torch.clamp(total_count, min=1.0) - mean * mean, min=1e-12)
        std = torch.sqrt(var)
        return mean, std

    def _process_trajectories(self, trajectories: List[Trajectory]):
        all_obs = []
        for traj in trajectories:
            all_obs.extend(traj.obs_list)

        if self.recompute_value:
            values_chunks = []
            with torch.no_grad():
                for start in range(0, len(all_obs), self.train_batch_size):
                    end = start + self.train_batch_size
                    obs_batch = self.model.prepare_inputs_batch(all_obs[start:end])
                    _, value_chunk = self.model(obs_batch)
                    values_chunks.append(value_chunk.float())
            flat_values = torch.cat(values_chunks, dim=0)
        else:
            flat_values = torch.from_numpy(
                np.concatenate([traj.old_values for traj in trajectories], axis=0)
            ).to(self.device)

        action_tokens = []
        behaviour_logits = []
        advantages = []
        value_targets = []
        offset = 0
        for traj in trajectories:
            traj_len = traj.num_steps
            traj_values = flat_values[offset : offset + traj_len]
            offset += traj_len
            traj_rewards = torch.tensor(np.array(traj.rewards, copy=True), dtype=torch.float32).to(self.device)
            traj_adv, traj_ret = self._compute_gae(
                rewards=traj_rewards,
                values=traj_values,
                bootstrap_value=traj.bootstrap_value,
                is_terminal=traj.is_terminal,
            )
            action_tokens.append(torch.tensor(np.array(traj.action_tokens, copy=True), dtype=torch.long).to(self.device))
            behaviour_logits.append(torch.tensor(np.array(traj.behaviour_logits, copy=True), dtype=torch.float32).to(self.device))
            advantages.append(traj_adv)
            value_targets.append(traj_ret)

        action_tokens_t = torch.cat(action_tokens, dim=0).long()
        behaviour_logits_t = torch.cat(behaviour_logits, dim=0)
        advantages_t = torch.cat(advantages, dim=0)
        value_targets_t = torch.cat(value_targets, dim=0)
        return all_obs, action_tokens_t, advantages_t, behaviour_logits_t, value_targets_t

    def get_weights(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

    def run_training_epoch(self) -> Dict[str, float]:
        while ray.get(self.replay_buffer.total_steps.remote()) < self.super_batch_size:
            time.sleep(0.2)

        sample_start = time.time()
        trajectories = ray.get(self.replay_buffer.sample_trajectories.remote(self.super_batch_size))
        sample_time = time.time() - sample_start
        if not trajectories:
            raise RuntimeError("ReplayBuffer returned no trajectories.")

        prep_start = time.time()
        obs_list, action_tokens, advantages, behaviour_logits, value_targets = self._process_trajectories(
            trajectories
        )
        permutation = torch.randperm(action_tokens.shape[0], device=self.device)
        obs_list = [obs_list[idx] for idx in permutation.cpu().tolist()]
        action_tokens = action_tokens[permutation]
        advantages = advantages[permutation]
        behaviour_logits = behaviour_logits[permutation]
        value_targets = value_targets[permutation]
        prep_time = time.time() - prep_start

        policy_lr = self._get_current_lr(self.global_step, self.policy_lr, self.train_iters)
        value_lr = self._get_current_lr(self.global_step, self.value_lr, self.train_iters)
        self.optimizer.param_groups[0]["lr"] = policy_lr
        self.optimizer.param_groups[1]["lr"] = policy_lr
        self.optimizer.param_groups[2]["lr"] = value_lr

        adv_mean, adv_std = self._global_adv_stats(advantages)

        epoch_losses = []
        epoch_policy_losses = []
        epoch_value_losses = []
        epoch_entropy_losses = []
        epoch_kl_losses = []
        epoch_entropy = []
        epoch_kl_divs = []
        train_start = time.time()

        num_updates = max(1, action_tokens.shape[0] // self.train_batch_size)
        self.model.train()
        for update_idx in range(num_updates):
            start = update_idx * self.train_batch_size
            end = start + self.train_batch_size
            batch_obs = self.model.prepare_inputs_batch(obs_list[start:end])
            batch_action_tokens = action_tokens[start:end]
            batch_advantages = advantages[start:end]
            batch_behaviour_logits = behaviour_logits[start:end]
            batch_value_targets = value_targets[start:end]
            normalized_adv = (batch_advantages - adv_mean) / (adv_std + 1e-8)

            current_logits, current_values = self.model(batch_obs)
            current_values = current_values.float()
            dist_new = Categorical(logits=current_logits)
            dist_old = Categorical(logits=batch_behaviour_logits)

            logp = dist_new.log_prob(batch_action_tokens)
            with torch.no_grad():
                logp_old = dist_old.log_prob(batch_action_tokens)

            ratio = torch.exp(logp - logp_old)
            adv_expanded = normalized_adv.unsqueeze(-1).unsqueeze(-1)
            surr1 = ratio * adv_expanded
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_expanded

            policy_loss = -torch.mean(torch.min(surr1, surr2))
            value_loss = self.vf_coef * torch.mean((current_values - batch_value_targets) ** 2)
            entropy = torch.mean(dist_new.entropy())
            entropy_loss = -self.ent_coef * entropy
            kl_div = torch.mean(kl_divergence(dist_old, dist_new))
            kl_loss = self.kl_coef * kl_div
            loss = policy_loss + value_loss + entropy_loss + kl_loss

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            self.global_step += 1
            epoch_losses.append(float(loss.item()))
            epoch_policy_losses.append(float(policy_loss.item()))
            epoch_value_losses.append(float(value_loss.item()))
            epoch_entropy_losses.append(float(entropy_loss.item()))
            epoch_kl_losses.append(float(kl_loss.item()))
            epoch_entropy.append(float(entropy.item()))
            epoch_kl_divs.append(float(kl_div.item()))

        self.model.eval()
        self.total_trainer_samples_consumed += int(num_updates * self.train_batch_size)
        return {
            "loss": float(np.mean(epoch_losses)),
            "policy_loss": float(np.mean(epoch_policy_losses)),
            "value_loss": float(np.mean(epoch_value_losses)),
            "entropy_loss": float(np.mean(epoch_entropy_losses)),
            "kl_loss": float(np.mean(epoch_kl_losses)),
            "entropy": float(np.mean(epoch_entropy)),
            "kl_div": float(np.mean(epoch_kl_divs)),
            "policy_lr": float(policy_lr),
            "value_lr": float(value_lr),
            "global_step": int(self.global_step),
            "sample_time": float(sample_time),
            "prep_time": float(prep_time),
            "train_time": float(time.time() - train_start),
            "consumed_samples": int(self.total_trainer_samples_consumed),
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal Ray GIPO with fake env/model")
    parser.add_argument("--num-trainers", type=int, default=1)
    parser.add_argument("--num-inference-actors", type=int, default=1)
    parser.add_argument("--num-rollout-workers", type=int, default=2)
    parser.add_argument("--num-eval-workers", type=int, default=1)
    parser.add_argument("--num-tasks", type=int, default=4)
    parser.add_argument("--rollout-local-buf", type=int, default=16)
    parser.add_argument("--replay-capacity", type=int, default=512)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--train-iters", type=int, default=20)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--kl-coef", type=float, default=0.0)
    parser.add_argument("--policy-lr", type=float, default=3e-4)
    parser.add_argument("--value-lr", type=float, default=1e-3)
    parser.add_argument("--obs-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--action-dim", type=int, default=ACTION_DIM)
    parser.add_argument("--num-actions-chunk", type=int, default=NUM_ACTIONS_CHUNK)
    parser.add_argument("--action-bins", type=int, default=ACTION_BINS)
    parser.add_argument("--env-max-steps", type=int, default=50)
    parser.add_argument("--trainer-num-gpus", type=float, default=0.0)
    parser.add_argument("--inference-num-gpus", type=float, default=0.0)
    parser.add_argument("--ray-num-cpus", type=int, default=max(2, os.cpu_count() or 2))
    parser.add_argument("--stats-window-size", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--weight-sync-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-path", type=str, default="minimal_ray_gipo_ckpt.pt")
    parser.add_argument("--recompute-value", action="store_true")
    parser.add_argument("--sync-stats-with-ddp", action="store_true")
    return parser


def select_device(num_gpus: float) -> str:
    if num_gpus > 0 and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def maybe_log(step: int, train_metrics: Dict[str, float], stats: Dict[str, Dict[str, float]]) -> None:
    rollout_stats = stats.get("_global_", {})
    timing_stats = stats.get("_timings_", {})
    print(
        f"[step {step:04d}] "
        f"loss={train_metrics['loss']:.4f} "
        f"policy={train_metrics['policy_loss']:.4f} "
        f"value={train_metrics['value_loss']:.4f} "
        f"entropy={train_metrics['entropy']:.4f} "
        f"return={rollout_stats.get('avg_return', 0.0):.3f} "
        f"ep_len={rollout_stats.get('avg_ep_len', 0.0):.1f} "
        f"success={rollout_stats.get('avg_success_rate', 0.0):.2f} "
        f"infer_p95_ms={timing_stats.get('inference_p95_ms', 0.0):.2f}"
    )


def main(args) -> None:
    if args.num_trainers != 1:
        raise ValueError("This minimal standalone version currently supports only --num-trainers 1.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ray.init(
        ignore_reinit_error=True,
        num_cpus=args.ray_num_cpus,
        num_gpus=max(int(np.ceil(args.trainer_num_gpus + args.inference_num_gpus * args.num_inference_actors)), 0),
        include_dashboard=False,
    )

    model_config = {
        "obs_dim": args.obs_dim,
        "hidden_dim": args.hidden_dim,
        "num_actions_chunk": args.num_actions_chunk,
        "action_dim": args.action_dim,
        "action_bins": args.action_bins,
    }

    stats_actor = StatsActor.remote(window_size=args.stats_window_size)
    replay_buffer = ReplayBufferActor.remote(capacity=args.replay_capacity)

    trainer = TrainerActor.options(num_gpus=args.trainer_num_gpus).remote(
        replay_buffer=replay_buffer,
        model_config=model_config,
        train_batch_size=args.train_batch_size,
        accumulation_steps=args.accumulation_steps,
        policy_lr=args.policy_lr,
        value_lr=args.value_lr,
        gamma=args.gamma,
        lambda_=args.lambda_,
        clip_eps=args.clip_eps,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        kl_coef=args.kl_coef,
        train_iters=args.train_iters,
        sync_stats_with_ddp=args.sync_stats_with_ddp,
        recompute_value=args.recompute_value,
        device=select_device(args.trainer_num_gpus),
    )

    inference_actors = [
        InferenceActor.options(num_gpus=args.inference_num_gpus).remote(
            actor_id=idx,
            model_config=model_config,
            stats_actor=stats_actor,
            device=select_device(args.inference_num_gpus),
        )
        for idx in range(args.num_inference_actors)
    ]

    initial_weights = ray.get(trainer.get_weights.remote())
    ray.get([actor.set_weights.remote(initial_weights) for actor in inference_actors])

    rollout_workers = [
        RolloutWorkerActor.remote(
            infer=inference_actors[idx % len(inference_actors)],
            replay=replay_buffer,
            wid=idx,
            stats_actor=stats_actor,
            num_tasks=args.num_tasks,
            obs_dim=args.obs_dim,
            action_dim=args.action_dim,
            env_max_steps=args.env_max_steps,
            reward_scale=args.reward_scale,
            rollout_local_buf=args.rollout_local_buf,
        )
        for idx in range(args.num_rollout_workers)
    ]
    eval_workers = [
        EvaluationWorkerActor.remote(
            infer=inference_actors[idx % len(inference_actors)],
            wid=f"eval_{idx}",
            stats_actor=stats_actor,
            num_tasks=args.num_tasks,
            obs_dim=args.obs_dim,
            action_dim=args.action_dim,
            env_max_steps=args.env_max_steps,
        )
        for idx in range(args.num_eval_workers)
    ]

    for worker in rollout_workers:
        worker.run.remote()
    for worker in eval_workers:
        worker.run.remote()

    min_steps = args.train_batch_size * args.accumulation_steps
    while ray.get(replay_buffer.total_steps.remote()) < min_steps:
        time.sleep(0.2)

    for epoch in range(1, args.train_iters + 1):
        train_metrics = ray.get(trainer.run_training_epoch.remote())
        if epoch % args.weight_sync_interval == 0:
            weights = ray.get(trainer.get_weights.remote())
            ray.get([actor.set_weights.remote(weights) for actor in inference_actors])
        if epoch % args.log_interval == 0:
            stats = ray.get(stats_actor.get_stats.remote())
            maybe_log(epoch, train_metrics, stats)

    final_weights = ray.get(trainer.get_weights.remote())
    torch.save(final_weights, args.checkpoint_path)
    print(f"Saved checkpoint to {args.checkpoint_path}")
    ray.shutdown()


if __name__ == "__main__":
    parser = build_arg_parser()
    main(parser.parse_args())
