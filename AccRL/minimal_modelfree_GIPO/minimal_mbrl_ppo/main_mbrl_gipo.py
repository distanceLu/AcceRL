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
from torch.distributions import Categorical, kl_divergence

try:
    import ray
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "main_mbrl_gipo.py requires `ray`. Install dependencies with: pip install ray torch numpy"
    ) from exc

from fake_env import FakeEnv
from fake_models import (
    ACTION_BINS,
    ACTION_DIM,
    NUM_ACTIONS_CHUNK,
    FakeActorCritic,
    FakeDenoiser,
    FakeRewardModel,
    ModelConfig,
)


@dataclass
class Experience:
    obs: Dict[str, torch.Tensor]
    action_token: np.ndarray
    advantage: float
    behaviour_logits: np.ndarray
    value_target: float


@ray.remote
class StatsActor:
    def __init__(self, window_size: int = 100):
        self.window_size = int(window_size)
        self.stats = defaultdict(
            lambda: {
                "episode_returns": deque(maxlen=self.window_size),
                "episode_lengths": deque(maxlen=self.window_size),
                "successes": deque(maxlen=self.window_size),
                "total_episodes": 0,
                "total_env_steps": 0,
            }
        )
        self.imagine_rewards = deque(maxlen=self.window_size)
        self.timings = defaultdict(lambda: deque(maxlen=self.window_size))

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
        record["total_episodes"] += 1
        record["total_env_steps"] += int(ep_length)

    def add_imagine_reward(self, avg_imagine_reward: float, actor_id: int) -> None:
        _ = actor_id
        self.imagine_rewards.append(float(avg_imagine_reward))

    def add_timing_metric(self, metric_name: str, value: float) -> None:
        self.timings[metric_name].append(float(value))

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        result = {}
        rollout_returns = []
        rollout_lengths = []
        rollout_success = []
        eval_returns = []
        eval_lengths = []
        eval_success = []

        for env_name, record in self.stats.items():
            returns = list(record["episode_returns"])
            lengths = list(record["episode_lengths"])
            successes = list(record["successes"])
            result[env_name] = {
                "avg_return": float(np.mean(returns)) if returns else 0.0,
                "avg_ep_len": float(np.mean(lengths)) if lengths else 0.0,
                "avg_success_rate": float(np.mean(successes)) if successes else 0.0,
                "total_episodes": int(record["total_episodes"]),
            }
            if env_name.startswith("eval_"):
                eval_returns.extend(returns)
                eval_lengths.extend(lengths)
                eval_success.extend(successes)
            else:
                rollout_returns.extend(returns)
                rollout_lengths.extend(lengths)
                rollout_success.extend(successes)

        result["_global_rollout_"] = {
            "avg_return": float(np.mean(rollout_returns)) if rollout_returns else 0.0,
            "avg_ep_len": float(np.mean(rollout_lengths)) if rollout_lengths else 0.0,
            "avg_success_rate": float(np.mean(rollout_success)) if rollout_success else 0.0,
            "avg_imagine_reward": float(np.mean(self.imagine_rewards)) if self.imagine_rewards else 0.0,
        }
        result["_global_eval_"] = {
            "avg_return": float(np.mean(eval_returns)) if eval_returns else 0.0,
            "avg_ep_len": float(np.mean(eval_lengths)) if eval_lengths else 0.0,
            "avg_success_rate": float(np.mean(eval_success)) if eval_success else 0.0,
        }
        result["_timings_"] = {
            name: float(np.mean(values)) if values else 0.0
            for name, values in self.timings.items()
        }
        return result


@ray.remote
class ReplayBufferActor:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=int(capacity))

    def add_batch(self, batch: List[Experience]) -> None:
        self.buffer.extend(batch)

    def size(self) -> int:
        return len(self.buffer)

    def sample(self, batch_size: int):
        if len(self.buffer) < batch_size:
            raise ValueError(f"ReplayBuffer has {len(self.buffer)} items, needs {batch_size}.")
        batch = random.sample(self.buffer, int(batch_size))
        obs_list = [b.obs for b in batch]
        action_token = np.stack([b.action_token for b in batch])
        adv = np.asarray([b.advantage for b in batch], dtype=np.float32)
        logits_old = np.stack([b.behaviour_logits for b in batch])
        value_target = np.asarray([b.value_target for b in batch], dtype=np.float32)
        return obs_list, action_token, adv, logits_old, value_target


class BaseWorkerActor:
    def __init__(
        self,
        infer,
        replay,
        wid,
        stats_actor,
        num_tasks: int,
        obs_shape: tuple[int, int, int],
        action_dim: int,
        env_max_steps: int,
    ) -> None:
        self.infer = infer
        self.replay = replay
        self.wid = wid
        self.stats_actor = stats_actor
        self.num_tasks = int(num_tasks)
        self.envs = [
            FakeEnv(
                task_id=task_id,
                obs_shape=obs_shape,
                action_dim=action_dim,
                max_steps=env_max_steps,
            )
            for task_id in range(self.num_tasks)
        ]
        self.env = None
        self.current_env_idx = -1
        self.task_description = ""
        self.current_env_name = ""

    def _set_current_env(self, env_idx: int, seed: Optional[int] = None):
        self.current_env_idx = int(env_idx)
        self.env = self.envs[self.current_env_idx]
        obs, info = self.env.reset(seed=seed)
        self.task_description = self.env.task_description
        self.current_env_name = self.env.get_name()
        return obs, info


@ray.remote
class RolloutWorkerActor(BaseWorkerActor):
    def __init__(
        self,
        infer,
        replay,
        wid,
        stats_actor,
        num_tasks: int,
        obs_shape: tuple[int, int, int],
        action_dim: int,
        env_max_steps: int,
        num_step_cond: int,
        imagine_horizon: int,
        reward_infer,
        denoiser_infer,
        reward_scale: float,
        gamma: float,
        lambda_: float,
    ) -> None:
        super().__init__(infer, replay, wid, stats_actor, num_tasks, obs_shape, action_dim, env_max_steps)
        self.reward_infer = reward_infer
        self.denoiser_infer = denoiser_infer
        self.reward_scale = float(reward_scale)
        self.gamma = float(gamma)
        self.lambda_ = float(lambda_)
        self.num_step_cond = int(num_step_cond)
        self.imagine_horizon = int(imagine_horizon)
        self.env_outcome = [deque(maxlen=100) for _ in range(self.num_tasks)]
        self.episodes = deque(maxlen=100)
        self.local_buffer = []

    def _reset_and_select_env(self, seed: Optional[int] = None):
        failure_counts = np.asarray([sum(history) for history in self.env_outcome], dtype=np.float32)
        weights = failure_counts + 1.0
        probs = weights / weights.sum()
        env_idx = int(np.random.choice(self.num_tasks, p=probs))
        return self._set_current_env(env_idx, seed=seed)

    def obs2inp(self, obs: torch.Tensor | np.ndarray, task_description: str) -> Dict[str, torch.Tensor]:
        _ = task_description
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(np.array(obs, copy=True)).float()
        else:
            obs = obs.float().clone()
        return {"image": obs}

    def predict_next_obs(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return ray.get(self.denoiser_infer.request.remote(obs.float(), act.float()))

    def predict_rew_end(self, next_obs: torch.Tensor, task_description: str) -> Tuple[float, int]:
        logits = ray.get(self.reward_infer.request.remote(self.obs2inp(next_obs, task_description)))
        probs = torch.softmax(logits, dim=-1)
        succ_prob = float(probs[1].item())
        end = int(logits.argmax().item())
        return succ_prob, end

    def _process_traj(self, traj_segment, bootstrap_val: float, is_terminal: bool) -> None:
        returns = []
        advantages = []
        gae = 0.0
        last_value = 0.0 if is_terminal else float(bootstrap_val)
        for idx in reversed(range(len(traj_segment))):
            _, _, reward, _, value = traj_segment[idx]
            next_value = last_value if idx == len(traj_segment) - 1 else float(traj_segment[idx + 1][4])
            delta = float(reward) + self.gamma * next_value - float(value)
            gae = delta + self.gamma * self.lambda_ * gae
            advantages.append(gae)
            returns.append(gae + float(value))
        advantages.reverse()
        returns.reverse()

        batch = []
        for idx, (obs, action_token, _, logits, _) in enumerate(traj_segment):
            batch.append(
                Experience(
                    obs=obs,
                    action_token=np.asarray(action_token, dtype=np.int64),
                    advantage=float(advantages[idx]),
                    behaviour_logits=np.asarray(logits, dtype=np.float32),
                    value_target=float(returns[idx]),
                )
            )
        self.replay.add_batch.remote(batch)

    def get_one_episode(self):
        obs_list, reward_list, done_list, action_norm_list = [], [], [], []
        seed = int(time.time() * 1000) + int(self.wid) + os.getpid()
        obs, info = self._reset_and_select_env(seed=seed)
        obs_list.append(obs)
        reward_sum = 0.0
        step_count_total = 0
        while True:
            inputs_t = self.obs2inp(obs["image"], self.task_description)
            act_norm, action_env, _, _, _ = ray.get(
                self.infer.request.remote(inputs_t, deterministic=False)
            )
            done = False
            for step_idx in range(len(action_env)):
                next_obs, reward, terminated, truncated, info = self.env.step(action_env[step_idx])
                obs_list.append(next_obs)
                reward_list.append(float(reward))
                done_list.append(bool(terminated or truncated))
                action_norm_list.append(np.asarray(act_norm[step_idx], dtype=np.float32))
                reward_sum += float(reward)
                step_count_total += 1
                obs = next_obs
                if terminated or truncated:
                    done = True
                    break
            if done:
                success = float(info.get("is_success", 0.0))
                self.env_outcome[self.current_env_idx].append(1.0 - success)
                self.stats_actor.add_episode_return.remote(
                    self.current_env_name,
                    reward_sum,
                    step_count_total,
                    success,
                )
                self.episodes.append(
                    (obs_list, reward_list, done_list, action_norm_list, self.task_description)
                )
                break

    def run(self):
        imagine_step = 0
        while True:
            if imagine_step % 10 == 0 or len(self.episodes) == 0:
                self.get_one_episode()
            imagine_step += 1

            obs_list, _, _, act_norm_list, task_description = random.choice(list(self.episodes))
            obs_images = [obs["image"] for obs in obs_list]
            for start_idx in range(len(obs_images) - self.num_step_cond):
                self.local_buffer.clear()
                obs_list_sub = obs_images[start_idx : start_idx + self.num_step_cond]
                obs_tensor = torch.stack(
                    [torch.from_numpy(np.array(frame, copy=True)).float() for frame in obs_list_sub],
                    dim=0,
                )
                act_list_sub = act_norm_list[start_idx : start_idx + self.num_step_cond - 1]
                if len(act_list_sub) == 0:
                    continue
                act_tensor = torch.stack(
                    [torch.from_numpy(np.array(act, copy=True)).float() for act in act_list_sub],
                    dim=0,
                )
                last_succ_prob, _ = self.predict_rew_end(obs_tensor[-1], task_description)
                end = False

                for j in range(self.imagine_horizon):
                    _ = j
                    inputs_t = self.obs2inp(obs_tensor[-1], task_description)
                    act_norm, action_env, action_token, logits, value = ray.get(
                        self.infer.request.remote(inputs_t, deterministic=False)
                    )
                    act_norm_t = torch.from_numpy(np.array(act_norm, copy=True)).float()
                    chunk_reward = 0.0
                    for k in range(len(action_env)):
                        single_action = act_norm_t[k]
                        act_tensor = torch.cat([act_tensor, single_action.unsqueeze(0)], dim=0)
                        next_obs = self.predict_next_obs(obs_tensor, act_tensor)
                        obs_tensor = torch.roll(obs_tensor, shifts=-1, dims=0)
                        obs_tensor[-1] = next_obs
                        act_tensor = act_tensor[1:]
                        succ_prob, end = self.predict_rew_end(next_obs, task_description)
                        reward = succ_prob - last_succ_prob
                        last_succ_prob = succ_prob
                        chunk_reward += reward * self.reward_scale
                        if end:
                            break
                    self.local_buffer.append((inputs_t, action_token, chunk_reward, logits, value))
                    if end:
                        break

                if self.local_buffer:
                    if end:
                        self._process_traj(self.local_buffer, bootstrap_val=0.0, is_terminal=True)
                    else:
                        bootstrap_inputs = self.obs2inp(obs_tensor[-1], task_description)
                        _, _, _, _, bootstrap_val = ray.get(
                            self.infer.request.remote(bootstrap_inputs, deterministic=False)
                        )
                        self._process_traj(
                            self.local_buffer,
                            bootstrap_val=float(bootstrap_val),
                            is_terminal=False,
                        )
                    imagine_rewards = [exp[2] for exp in self.local_buffer]
                    self.stats_actor.add_imagine_reward.remote(
                        float(sum(imagine_rewards) / max(len(imagine_rewards), 1)),
                        int(self.wid),
                    )
                self.local_buffer.clear()


@ray.remote
class EvaluationWorkerActor(BaseWorkerActor):
    def __init__(
        self,
        infer,
        wid,
        stats_actor,
        num_tasks: int,
        obs_shape: tuple[int, int, int],
        action_dim: int,
        env_max_steps: int,
    ) -> None:
        super().__init__(infer, None, wid, stats_actor, num_tasks, obs_shape, action_dim, env_max_steps)

    def _reset_and_select_env(self, seed: Optional[int] = None):
        env_idx = (self.current_env_idx + 1) % self.num_tasks
        return self._set_current_env(env_idx, seed=seed)

    def run(self):
        seed = int(time.time() * 1000) + os.getpid()
        obs, info = self._reset_and_select_env(seed=seed)
        _ = info
        while True:
            reward_sum = 0.0
            step_count_total = 0
            done = False
            while not done:
                inputs_t = {"image": torch.from_numpy(np.array(obs["image"], copy=True)).float()}
                _, action_env, _, _, _ = ray.get(self.infer.request.remote(inputs_t, deterministic=True))
                for idx in range(len(action_env)):
                    obs, reward, terminated, truncated, info = self.env.step(action_env[idx])
                    reward_sum += float(reward)
                    step_count_total += 1
                    if terminated or truncated:
                        done = True
                        break
            self.stats_actor.add_episode_return.remote(
                f"eval_{self.current_env_name}",
                reward_sum,
                step_count_total,
                float(info.get("is_success", 0.0)),
            )
            seed = int(time.time() * 1000) + step_count_total
            obs, info = self._reset_and_select_env(seed=seed)
            _ = info


@ray.remote
class InferenceActor:
    def __init__(self, actor_id: int, model_config: Dict, stats_actor=None, device: str = "cpu"):
        self.actor_id = int(actor_id)
        self.model = FakeActorCritic(ModelConfig(**model_config)).to(device)
        self.model.eval()
        self.stats_actor = stats_actor
        self.device = device

    def request(self, inputs_t: Dict[str, torch.Tensor], deterministic: bool = False):
        start = time.perf_counter()
        batch = self.model.prepare_inputs_batch([inputs_t])
        with torch.no_grad():
            action_logits, values = self.model(batch)
            action_tokens, action_env = self.model.post_process(action_logits, deterministic=deterministic)
        if self.stats_actor is not None:
            self.stats_actor.add_timing_metric.remote(
                "inference/policy_ms", (time.perf_counter() - start) * 1000.0
            )
        return (
            action_env[0].cpu().numpy().astype(np.float32),
            action_env[0].cpu().numpy().astype(np.float32),
            action_tokens[0].cpu().numpy().astype(np.int64),
            action_logits[0].cpu().numpy().astype(np.float32),
            float(values[0].item()),
        )

    def get_weights(self):
        return {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

    def set_weights(self, state_dict):
        self.model.load_state_dict(state_dict)
        self.model.eval()


@ray.remote
class RewardInferenceActor:
    def __init__(self, actor_id: int, model_config: Dict, stats_actor=None, device: str = "cpu"):
        self.actor_id = int(actor_id)
        self.model = FakeRewardModel(ModelConfig(**model_config)).to(device)
        self.model.eval()
        self.stats_actor = stats_actor

    def request(self, inputs_t: Dict[str, torch.Tensor]):
        start = time.perf_counter()
        batch = self.model.prepare_inputs_batch([inputs_t])
        with torch.no_grad():
            logits = self.model(batch)[0].cpu()
        if self.stats_actor is not None:
            self.stats_actor.add_timing_metric.remote(
                "inference/reward_ms", (time.perf_counter() - start) * 1000.0
            )
        return logits


@ray.remote
class DenoiserInferenceActor:
    def __init__(self, actor_id: int, model_config: Dict, stats_actor=None, device: str = "cpu"):
        self.actor_id = int(actor_id)
        self.model = FakeDenoiser(ModelConfig(**model_config)).to(device)
        self.model.eval()
        self.stats_actor = stats_actor

    def request(self, obs: torch.Tensor, act: torch.Tensor):
        start = time.perf_counter()
        with torch.no_grad():
            next_obs = self.model(obs.unsqueeze(0), act.unsqueeze(0))[0].cpu()
        if self.stats_actor is not None:
            self.stats_actor.add_timing_metric.remote(
                "inference/denoiser_ms", (time.perf_counter() - start) * 1000.0
            )
        return next_obs


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
        sigma: float,
        train_iters: int,
        policy_train_start_step: int = 0,
        device: str = "cpu",
    ) -> None:
        self.replay_buffer = replay_buffer
        self.device = torch.device(device)
        self.model = FakeActorCritic(ModelConfig(**model_config)).to(self.device)
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
        self.sigma = float(sigma)
        self.train_iters = int(train_iters)
        self.policy_train_start_step = int(policy_train_start_step)
        self.global_step = 0
        self.optimizer = torch.optim.Adam(
            [
                {"params": list(self.model.encoder.parameters()) + list(self.model.policy_head.parameters()), "lr": self.policy_lr},
                {"params": self.model.value_head.parameters(), "lr": self.value_lr},
            ]
        )

    def _get_current_lr(self, current_step: int, peak_lr: float, total_steps: int, start_step: int = 0):
        if current_step < start_step:
            return 0.0
        effective_step = current_step - start_step
        usable_total = max(total_steps - start_step, 1)
        progress = min(max(effective_step / usable_total, 0.0), 1.0)
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        return float(peak_lr * cosine_decay)

    def get_weights(self):
        return {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

    def run_training_epoch(self) -> Dict[str, float]:
        while ray.get(self.replay_buffer.size.remote()) < self.super_batch_size:
            time.sleep(0.2)

        sample_start = time.time()
        obs_list, action_token_np, adv_np, logits_old_np, value_target_np = ray.get(
            self.replay_buffer.sample.remote(self.super_batch_size)
        )
        sample_time = time.time() - sample_start

        prep_start = time.time()
        inputs_batch = self.model.prepare_inputs_batch(obs_list)
        act_token_t = torch.tensor(action_token_np, dtype=torch.long, device=self.device)
        adv_t = torch.tensor(adv_np, dtype=torch.float32, device=self.device)
        logits_old_t = torch.tensor(logits_old_np, dtype=torch.float32, device=self.device)
        value_target_t = torch.tensor(value_target_np, dtype=torch.float32, device=self.device)
        prep_time = time.time() - prep_start

        permutation = torch.randperm(act_token_t.shape[0], device=self.device)
        inputs_batch = {k: v[permutation] for k, v in inputs_batch.items()}
        act_token_t = act_token_t[permutation]
        adv_t = adv_t[permutation]
        logits_old_t = logits_old_t[permutation]
        value_target_t = value_target_t[permutation]

        adv_mean = adv_t.mean()
        adv_std = adv_t.std(unbiased=False).clamp_min(1e-8)

        policy_lr = self._get_current_lr(
            self.global_step, self.policy_lr, self.train_iters, start_step=self.policy_train_start_step
        )
        value_lr = self._get_current_lr(self.global_step, self.value_lr, self.train_iters)
        self.optimizer.param_groups[0]["lr"] = policy_lr
        self.optimizer.param_groups[1]["lr"] = value_lr

        total_losses = []
        policy_losses = []
        value_losses = []
        entropy_losses = []
        kl_losses = []
        entropies = []
        kl_divs = []
        train_start = time.time()
        num_updates = self.super_batch_size // self.train_batch_size

        self.model.train()
        for update_idx in range(num_updates):
            start = update_idx * self.train_batch_size
            end = start + self.train_batch_size
            mini_inputs = {k: v[start:end] for k, v in inputs_batch.items()}
            mini_action = act_token_t[start:end]
            mini_adv = adv_t[start:end]
            mini_logits_old = logits_old_t[start:end]
            mini_value_target = value_target_t[start:end]
            normalized_adv = (mini_adv - adv_mean) / adv_std

            action_logits, value = self.model(mini_inputs)
            value = value.float()
            dist_new = Categorical(logits=action_logits)
            dist_old = Categorical(logits=mini_logits_old)
            logp = dist_new.log_prob(mini_action)
            with torch.no_grad():
                logp_old = dist_old.log_prob(mini_action)

            ratio = torch.exp(logp - logp_old)
            adv_expand = normalized_adv.unsqueeze(-1).unsqueeze(-1)
            surr1 = ratio * adv_expand
            # GIPO: Gaussian-smoothed Trust Region Soft Clipping
            eps = 1e-9
            sigma = max(getattr(self, "sigma", 0.5), 1e-9)
            r_detach = ratio.clamp_min(eps).detach()
            coeff = torch.exp(-0.5 * (torch.log(r_detach) / sigma) ** 2)
            surr_soft = surr1 * coeff

            value_loss = self.vf_coef * torch.mean((value - mini_value_target) ** 2)
            if self.global_step < self.policy_train_start_step:
                policy_loss = torch.tensor(0.0, device=self.device)
                entropy = torch.tensor(0.0, device=self.device)
                kl_div = torch.tensor(0.0, device=self.device)
                entropy_loss = torch.tensor(0.0, device=self.device)
                kl_loss = torch.tensor(0.0, device=self.device)
                loss = value_loss
            else:
                policy_loss = -torch.mean(surr_soft)
                entropy = torch.mean(dist_new.entropy())
                kl_div = torch.mean(kl_divergence(dist_old, dist_new))
                entropy_loss = -self.ent_coef * entropy
                kl_loss = self.kl_coef * kl_div
                loss = policy_loss + value_loss + entropy_loss + kl_loss

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.global_step += 1

            total_losses.append(float(loss.item()))
            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropy_losses.append(float(entropy_loss.item()))
            kl_losses.append(float(kl_loss.item()))
            entropies.append(float(entropy.item()))
            kl_divs.append(float(kl_div.item()))

        self.model.eval()
        return {
            "loss": float(np.mean(total_losses)),
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy_loss": float(np.mean(entropy_losses)),
            "kl_loss": float(np.mean(kl_losses)),
            "entropy": float(np.mean(entropies)),
            "kl_div": float(np.mean(kl_divs)),
            "policy_lr": float(policy_lr),
            "value_lr": float(value_lr),
            "sample_time": float(sample_time),
            "prep_time": float(prep_time),
            "train_time": float(time.time() - train_start),
            "global_step": int(self.global_step),
        }


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Minimal world-model distributed GIPO")
    parser.add_argument("--num-rollout-workers", type=int, default=2)
    parser.add_argument("--num-eval-workers", type=int, default=1)
    parser.add_argument("--num-inference-actors", type=int, default=1)
    parser.add_argument("--num-reward-inference-actors", type=int, default=1)
    parser.add_argument("--num-denoiser-inference-actors", type=int, default=1)
    parser.add_argument("--num-tasks", type=int, default=4)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--train-iters", type=int, default=20)
    parser.add_argument("--replay-capacity", type=int, default=5000)
    parser.add_argument("--num-step-cond", type=int, default=4)
    parser.add_argument("--imagine-horizon", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--kl-coef", type=float, default=0.0)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--policy-lr", type=float, default=3e-4)
    parser.add_argument("--value-lr", type=float, default=1e-3)
    parser.add_argument("--policy-train-start-step", type=int, default=0)
    parser.add_argument("--trainer-num-gpus", type=float, default=0.0)
    parser.add_argument("--inference-num-gpus", type=float, default=0.0)
    parser.add_argument("--reward-num-gpus", type=float, default=0.0)
    parser.add_argument("--denoiser-num-gpus", type=float, default=0.0)
    parser.add_argument("--ray-num-cpus", type=int, default=max(2, os.cpu_count() or 2))
    parser.add_argument("--obs-channels", type=int, default=3)
    parser.add_argument("--obs-height", type=int, default=16)
    parser.add_argument("--obs-width", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--action-dim", type=int, default=ACTION_DIM)
    parser.add_argument("--num-actions-chunk", type=int, default=NUM_ACTIONS_CHUNK)
    parser.add_argument("--action-bins", type=int, default=ACTION_BINS)
    parser.add_argument("--env-max-steps", type=int, default=50)
    parser.add_argument("--stats-window-size", type=int, default=100)
    parser.add_argument("--weight-sync-interval", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--checkpoint-path", type=str, default="minimal_mbrl_gipo_ckpt.pt")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def select_device(num_gpus: float) -> str:
    return "cuda" if num_gpus > 0 and torch.cuda.is_available() else "cpu"


def print_metrics(epoch: int, train_metrics: Dict[str, float], stats: Dict[str, Dict[str, float]]) -> None:
    rollout_stats = stats.get("_global_rollout_", {})
    eval_stats = stats.get("_global_eval_", {})
    print(
        f"[step {epoch:04d}] "
        f"loss={train_metrics['loss']:.4f} "
        f"policy={train_metrics['policy_loss']:.4f} "
        f"value={train_metrics['value_loss']:.4f} "
        f"imag_rew={rollout_stats.get('avg_imagine_reward', 0.0):.4f} "
        f"rollout_ret={rollout_stats.get('avg_return', 0.0):.4f} "
        f"eval_ret={eval_stats.get('avg_return', 0.0):.4f}"
    )


def main(args) -> None:
    if args.num_step_cond < 2:
        raise ValueError("--num-step-cond must be >= 2")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    obs_shape = (args.obs_channels, args.obs_height, args.obs_width)
    model_config = {
        "obs_shape": obs_shape,
        "hidden_dim": args.hidden_dim,
        "action_dim": args.action_dim,
        "num_actions_chunk": args.num_actions_chunk,
        "action_bins": args.action_bins,
    }

    total_requested_gpus = (
        args.trainer_num_gpus
        + args.num_inference_actors * args.inference_num_gpus
        + args.num_reward_inference_actors * args.reward_num_gpus
        + args.num_denoiser_inference_actors * args.denoiser_num_gpus
    )
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        num_cpus=args.ray_num_cpus,
        num_gpus=max(int(np.ceil(total_requested_gpus)), 0),
    )

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
        sigma=args.sigma,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        kl_coef=args.kl_coef,
        train_iters=args.train_iters,
        policy_train_start_step=args.policy_train_start_step,
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
    reward_actors = [
        RewardInferenceActor.options(num_gpus=args.reward_num_gpus).remote(
            actor_id=idx,
            model_config=model_config,
            stats_actor=stats_actor,
            device=select_device(args.reward_num_gpus),
        )
        for idx in range(args.num_reward_inference_actors)
    ]
    denoiser_actors = [
        DenoiserInferenceActor.options(num_gpus=args.denoiser_num_gpus).remote(
            actor_id=idx,
            model_config=model_config,
            stats_actor=stats_actor,
            device=select_device(args.denoiser_num_gpus),
        )
        for idx in range(args.num_denoiser_inference_actors)
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
            obs_shape=obs_shape,
            action_dim=args.action_dim,
            env_max_steps=args.env_max_steps,
            num_step_cond=args.num_step_cond,
            imagine_horizon=args.imagine_horizon,
            reward_infer=reward_actors[idx % len(reward_actors)],
            denoiser_infer=denoiser_actors[idx % len(denoiser_actors)],
            reward_scale=args.reward_scale,
            gamma=args.gamma,
            lambda_=args.lambda_,
        )
        for idx in range(args.num_rollout_workers)
    ]
    eval_workers = [
        EvaluationWorkerActor.remote(
            infer=inference_actors[idx % len(inference_actors)],
            wid=f"eval_{idx}",
            stats_actor=stats_actor,
            num_tasks=args.num_tasks,
            obs_shape=obs_shape,
            action_dim=args.action_dim,
            env_max_steps=args.env_max_steps,
        )
        for idx in range(args.num_eval_workers)
    ]

    for worker in rollout_workers:
        worker.run.remote()
    for worker in eval_workers:
        worker.run.remote()

    min_replay = args.train_batch_size * args.accumulation_steps
    while ray.get(replay_buffer.size.remote()) < min_replay:
        time.sleep(0.2)

    for epoch in range(1, args.train_iters + 1):
        train_metrics = ray.get(trainer.run_training_epoch.remote())
        if epoch % args.weight_sync_interval == 0:
            weights = ray.get(trainer.get_weights.remote())
            ray.get([actor.set_weights.remote(weights) for actor in inference_actors])
        if epoch % args.log_interval == 0:
            stats = ray.get(stats_actor.get_stats.remote())
            print_metrics(epoch, train_metrics, stats)

    final_weights = ray.get(trainer.get_weights.remote())
    torch.save(final_weights, args.checkpoint_path)
    print(f"Saved checkpoint to {args.checkpoint_path}")
    ray.shutdown()


if __name__ == "__main__":
    parser = build_arg_parser()
    main(parser.parse_args())
