import os
import argparse
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["TMPDIR"] = "/dev/shm"

import sys
import time
import random
import asyncio
import json
from collections import deque, defaultdict
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass
from pathlib import Path
import math
from hydra.utils import instantiate
from omegaconf import OmegaConf
import numpy as np
import ray
import torch
import torch.nn as nn
import torch.distributions
from torch.distributions import kl
import deepspeed
import torch.distributed as distributed
from torch.utils.tensorboard import SummaryWriter

# ---- AcceRL sys.path 设置 ----
ACCE_RL_ROOT = Path(__file__).resolve().parent.parent
if str(ACCE_RL_ROOT) not in sys.path:
    sys.path.insert(0, str(ACCE_RL_ROOT))

# OpenVLA 组件与常量
from experiments.robot.sole_utils import get_processor
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM
from experiments.robot.libero.libero_utils import GenerateConfig, TaskSuite
from rl.actor_critic_model_discrete import ActorCritic
from rl.utils import prepare_one_obs
# 训练/推理通信
from rl.ds_com import TrainerActorCom, InferenceActorCom
from rl.com_utils import find_free_port
from envs.utils import tensor_to_image, image_to_tensor, load_reward_model

# ---- Ctrl-World 相关导入（ctrl_world 已作为包安装在 merged-env 中）----
from tests_dsj.ctrl_world_env_batch import CtrlWorldEnvBatch
from ctrl_world.config import wm_args
from ctrl_world.models.ctrl_world import CrtlWorld
from rl.ray_debug_utils import setup_debugger


# ================================================================
# 0. 超参数与配置 - 命令行参数解析
# ================================================================
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='OpenVLA RL Training with Ctrl-World World Model')

    # 环境变量
    parser.add_argument('--cuda-visible-devices', type=str, default='0,1,2,3',
                        help='CUDA devices; defaults cover trainer, policy, reward and Ctrl-World actors')

    # Libero benchmark
    parser.add_argument('--benchmark', type=str, default='libero_spatial',
                        choices=['libero_spatial', 'libero_object', 'libero_goal', 'libero_10', 'libero_90'],
                        help='Libero benchmark suite (default: libero_spatial)')

    # 分布式系统参数
    parser.add_argument('--num-trainer-gpus', type=int, default=1)
    parser.add_argument('--num-inference-actors', type=int, default=1)
    parser.add_argument('--num-rollout-workers', type=int, default=20)
    parser.add_argument('--num-eval-workers', type=int, default=10)
    parser.add_argument('--rollout-local-buf', type=int, default=64)
    parser.add_argument('--inference-batch', type=int, default=8)
    parser.add_argument('--inference-timeout-ms', type=int, default=300)
    parser.add_argument('--replay-capacity', type=int, default=10000)
    parser.add_argument('--train-batch-size', type=int, default=32)
    parser.add_argument('--accumulation-steps', type=int, default=8)
    parser.add_argument('--train-iters', type=int, default=30000)

    # Checkpoint
    parser.add_argument('--ckpt-dir', type=str,
                        default='/mnt/data/lcx2/yanjieworkspace/models/finetune_rl/')
    parser.add_argument('--ckpt-every-steps', type=int, default=2000000)

    # PPO 参数
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lambda', type=float, default=0.95, dest='lambda_')
    parser.add_argument('--clip-eps', type=float, default=0.2)
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.00)
    parser.add_argument('--kl-coef', type=float, default=0.1)

    # 奖励缩放
    parser.add_argument('--reward-scale', type=float, default=1.0)

    # 学习率调度参数
    parser.add_argument('--value-lr', type=float, default=1e-4)
    parser.add_argument('--policy-lr', type=float, default=1e-5)
    parser.add_argument('--value-warmup-steps', type=int, default=500)
    parser.add_argument('--policy-warmup-steps', type=int, default=500)
    parser.add_argument('--policy-train-start-step', type=int, default=0)

    # 世界模型配置
    parser.add_argument('--imagine-horizon', type=int, default=8)
    parser.add_argument('--num-step-cond', type=int, default=7,
                        help='Real context frames; must be at least num_history + 1')
    parser.add_argument('--num-reward-inference-actors', type=int, default=1)
    parser.add_argument('--num-ctrl-inference-actors', type=int, default=1,
                        help='Number of Ctrl-World inference actors')
    parser.add_argument('--agent-config-path', type=str, default='envs/config/agent.yaml')
    parser.add_argument('--trainer-config-path', type=str, default='envs/config/trainer.yaml')

    # 日志
    parser.add_argument('--moving-avg-window', type=int, default=1000)
    parser.add_argument('--log-interval-seconds', type=int, default=10)

    # 通信组
    parser.add_argument('--broadcast-group-name', type=str, default='trainer_to_inference_broadcast')

    # OpenVLA 加载配置
    parser.add_argument('--use-bf16', action='store_true', default=True)
    parser.add_argument('--no-bf16', action='store_false', dest='use_bf16')
    parser.add_argument('--use-proprio', action='store_true', default=False)
    parser.add_argument('--num-images-in-input', type=int, default=2)
    parser.add_argument('--pretrained-checkpoint', type=str,
                        default='/mnt/data/lcx2/yanjieworkspace/models/finetune_im/openvla-7b+libero_spatial_no_noops+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--discrete_acts--proprio_state--100000_chkpt')
    parser.add_argument('--checkpoint2', type=str, default=None)

    parser.add_argument('--clip-mode', type=str, default='sapo',
                        choices=['ppo', 'sapo', 'gipo'])
    parser.add_argument('--exp-name', type=str, default=None)

    # ============ Ctrl-World 专用参数 ============
    parser.add_argument('--svd-model-path', type=str,
                        default='/mnt/data/lcx3/checkpoint/ctrl_world/svd/svd_model',
                        help='Path to SVD model')
    parser.add_argument('--clip-model-path', type=str,
                        default='/mnt/data/lcx3/checkpoint/ctrl_world/clip/clip_model',
                        help='Path to CLIP model')
    parser.add_argument('--ctrl-world-ckpt', type=str, required=True,
                        help='Path to a Ctrl-World checkpoint trained on AcceRL delta actions')
    parser.add_argument('--condition-stat-path', type=str, default=None,
                        help='condition_stat.json; defaults to checkpoint sibling')
    parser.add_argument('--num-cams', type=int, default=2,
                        help='Number of camera views for Ctrl-World')
    parser.add_argument('--num-history', type=int, default=6,
                        help='Ctrl-World latent history length')
    parser.add_argument('--num-inference-steps', type=int, default=50,
                        help='SVD denoising steps')
    parser.add_argument('--target-height', type=int, default=192)
    parser.add_argument('--target-width', type=int, default=320)
    parser.add_argument('--num-frames-pred', type=int, default=5,
                        help='Ctrl-World chunk length including current anchor (checkpoint protocol: 5)')

    # Reward Model checkpoint 路径覆盖
    parser.add_argument('--reward-checkpoint', type=str, default=None,
                        help='Reward model checkpoint path (overrides agent.yaml)')

    # ============ World Model 训练参数 ============
    # Reward Model 参数（保持与原版一致）
    parser.add_argument('--reward-batch-size', type=int, default=32)
    parser.add_argument('--reward-accumulation-steps', type=int, default=4)
    parser.add_argument('--reward-lr', type=float, default=1e-4)
    parser.add_argument('--reward-warmup-steps', type=int, default=500)

    # World Model Replay Buffer 参数
    parser.add_argument('--wm-replay-capacity', type=int, default=50000)
    parser.add_argument('--real-traj-collect-interval', type=int, default=10)

    # World Model 训练频率控制
    parser.add_argument('--reward-train-interval', type=int, default=10,
                        help='Train reward model every N actor training steps')
    # ============ 调试参数 ============
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Enable debugpy in Ray actors (each actor starts a debug server)')
    parser.add_argument('--debug-wait', action='store_true', default=False,
                        help='Actors wait for VS Code debugger to attach before proceeding')

    args = parser.parse_args()

    if args.num_cams != 2:
        raise ValueError("This AcceRL integration requires agentview + wrist (num_cams=2)")
    if args.num_images_in_input != 2:
        raise ValueError("VLA must receive both generated views (num_images_in_input=2)")
    if args.use_proprio:
        raise ValueError("Imagined proprio is not modeled; run this integration with --use-proprio disabled")
    if args.num_frames_pred != 5:
        raise ValueError("The current Ctrl-World checkpoint protocol requires num_frames_pred=5")
    if args.num_step_cond < args.num_history + 1:
        raise ValueError("num_step_cond must provide num_history previous frames plus current")
    required_gpu_actors = (
        args.num_trainer_gpus
        + args.num_inference_actors
        + args.num_reward_inference_actors
        + args.num_ctrl_inference_actors
    )
    visible_gpus = [x for x in args.cuda_visible_devices.split(",") if x.strip()]
    if len(visible_gpus) < required_gpu_actors:
        raise ValueError(
            f"{required_gpu_actors} one-GPU Ray actors requested but only "
            f"{len(visible_gpus)} CUDA devices are visible"
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    # 确保 Python 环境的 bin 目录在 PATH 中（Ray worker 需要 ninja 等工具）
    conda_bin = os.path.dirname(sys.executable)
    if conda_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{conda_bin}:{os.environ.get('PATH', '')}"

    # 设置调试环境变量（Ray worker 会继承）
    if args.debug:
        os.environ["RAY_DEBUG"] = "1"
    if args.debug_wait:
        os.environ["RAY_DEBUG_WAIT"] = "1"

    if args.exp_name is None:
        args.exp_name = f"OpenVLA_DS_{args.clip_mode}_DISCRETE_task0_ctrl_wm"

    return args


# ================================================================
# 数据结构
# ================================================================
@dataclass
class Experience:
    obs: Dict[str, torch.Tensor]
    action_token: np.ndarray
    advantage: float
    behaviour_logits: np.ndarray
    value_target: float


@dataclass
class WMExperience:
    """World Model 训练数据结构"""
    obs: np.ndarray                         # [num_steps_conditioning+1, M, C, H, W]
    act: np.ndarray                         # [num_steps_conditioning, act_dim]
    rew: float
    instruction: str


# ================================================================
# 1.5 统计模块 (StatsActor) — 与原版完全一致
# ================================================================
@ray.remote
class StatsActor:
    def __init__(self, window_size):
        setup_debugger("stats")
        self.stats = defaultdict(lambda: {
            "episode_returns": deque(maxlen=window_size),
            "step_times": deque(maxlen=window_size),
            "episode_lengths": deque(maxlen=window_size),
            "successes": deque(maxlen=window_size),
            "total_episodes_processed": 0,
            "total_env_steps": 0
        })
        self.timings = defaultdict(lambda: deque(maxlen=window_size))
        self.imagine_rewards = deque(maxlen=window_size)
        self.actor_last_active = {}
        self.active_window_seconds = 600
        self.total_samples_produced = 0
        self.total_imagine_samples = 0

    def add_episode_return(self, env_name, ep_return, step_time, ep_length, success, actor_id=None, step_num=0):
        env_stats = self.stats[env_name]
        env_stats["episode_returns"].append(ep_return)
        env_stats["step_times"].append(step_time)
        env_stats["episode_lengths"].append(ep_length)
        env_stats["successes"].append(success)
        env_stats["total_episodes_processed"] += 1
        env_stats["total_env_steps"] += ep_length
        if not env_name.startswith("eval_"):
            self.total_samples_produced += step_num
            if actor_id is not None:
                self.actor_last_active[actor_id] = time.time()

    def add_timing_metric(self, metric_name, value):
        self.timings[metric_name].append(value)

    def add_imagine_reward(self, avg_imagine_reward, actor_id, num_samples=1):
        self.imagine_rewards.append(avg_imagine_reward)
        self.total_imagine_samples += num_samples
        self.actor_last_active[actor_id] = time.time()

    def get_active_actor_count(self):
        current_time = time.time()
        cutoff = current_time - self.active_window_seconds
        return sum(1 for last_active in self.actor_last_active.values() if last_active >= cutoff)

    def get_stats(self):
        per_env_stats = {}
        all_returns, all_lengths, all_step_times = [], [], []
        total_episodes_processed = 0
        total_env_steps = 0
        eval_returns, eval_lengths, eval_step_times = [], [], []
        eval_total_episodes_processed = 0
        eval_total_env_steps = 0

        for env_name, env_data in self.stats.items():
            if not env_data["episode_returns"]:
                per_env_stats[env_name] = {
                    "avg_return": 0.0, "avg_ep_len": 0.0, "avg_success_rate": 0.0,
                    "num_episodes_in_avg": 0, "total_episodes": env_data["total_episodes_processed"]}
                continue
            per_env_stats[env_name] = {
                "avg_return": np.mean(env_data["episode_returns"]),
                "avg_ep_len": np.mean(env_data["episode_lengths"]),
                "avg_success_rate": np.mean(env_data["successes"]),
                "num_episodes_in_avg": len(env_data["episode_returns"]),
                "total_episodes": env_data["total_episodes_processed"]
            }
            if env_name.startswith("eval_"):
                eval_total_episodes_processed += env_data["total_episodes_processed"]
                eval_total_env_steps += env_data["total_env_steps"]
                eval_returns.extend(env_data["episode_returns"])
                eval_lengths.extend(env_data["episode_lengths"])
                eval_step_times.extend(env_data["step_times"])
            else:
                total_episodes_processed += env_data["total_episodes_processed"]
                total_env_steps += env_data["total_env_steps"]
                all_returns.extend(env_data["episode_returns"])
                all_lengths.extend(env_data["episode_lengths"])
                all_step_times.extend(env_data["step_times"])

        per_env_stats["_global_rollout_"] = {
            "avg_return": np.mean(all_returns) if all_returns else 0.0,
            "avg_ep_len": np.mean(all_lengths) if all_lengths else 0.0,
            "avg_step_time": np.mean(all_step_times) if all_step_times else 0.0,
            "total_episodes_processed": total_episodes_processed,
            "total_env_steps": total_env_steps,
            "total_samples_produced": self.total_samples_produced,
            "total_imagine_samples": self.total_imagine_samples,
            "active_actor_count": self.get_active_actor_count(),
            "avg_imagine_reward": np.mean(self.imagine_rewards) if self.imagine_rewards else 0.0
        }
        per_env_stats["_global_eval_"] = {
            "avg_return": np.mean(eval_returns) if eval_returns else 0.0,
            "avg_ep_len": np.mean(eval_lengths) if eval_lengths else 0.0,
            "avg_step_time": np.mean(eval_step_times) if eval_step_times else 0.0,
            "total_episodes_processed": eval_total_episodes_processed,
            "total_env_steps": eval_total_env_steps
        }
        timing_stats = {}
        for name, deq in self.timings.items():
            timing_stats[name] = np.mean(deq) if deq else 0.0
        per_env_stats["_timings_"] = timing_stats
        return per_env_stats


# ================================================================
# 2. 经验回放与 Rollout — 与原版一致
# ================================================================
@ray.remote
class ReplayBufferActor:
    def __init__(self, capacity):
        setup_debugger("replay")
        self.buffer = deque(maxlen=capacity)

    def add_batch(self, batch: List[Experience]):
        self.buffer.extend(batch)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs_list = [b.obs for b in batch]
        action_token = np.stack([b.action_token for b in batch])
        adv = np.asarray([b.advantage for b in batch], np.float32)
        logits_old = np.stack([b.behaviour_logits for b in batch])
        v_targ = np.asarray([b.value_target for b in batch], np.float32)
        return obs_list, action_token, adv, logits_old, v_targ


@ray.remote
class WMReplayBufferActor:
    def __init__(self, capacity):
        setup_debugger("wm_replay")
        self.buffer = deque(maxlen=capacity)

    def add_batch(self, batch: List[WMExperience]):
        self.buffer.extend(batch)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs = np.stack([b.obs for b in batch])
        act = np.stack([b.act for b in batch])
        rew = np.asarray([b.rew for b in batch], np.float32)
        instructions = [b.instruction for b in batch]
        return obs, act, rew, instructions


class BaseWorkerActor:
    """rollout 和 eval worker 的共享逻辑。"""
    def __init__(self, infer, replay, wid, stats_actor, cfg, benchmark_name):
        self.infer = infer
        self.replay = replay
        self.stats_actor = stats_actor
        self.cfg = cfg
        self.processor = get_processor(cfg)
        self.benchmark_name = benchmark_name
        from rl.libero_env import LiberoEnvWrapper

        self.num_tasks = 1
        print(f"BaseWorker {wid}: 正在初始化 {self.num_tasks} 个 Libero 环境...")
        self.envs = [
            LiberoEnvWrapper(
                benchmark_name=self.benchmark_name,
                task_id=i,
                image_size=224,
                render_mode="rgb_array"
            ) for i in range(self.num_tasks)]
        print(f"BaseWorker {wid}: 环境初始化完成。")

        self.env = None
        self.current_env_idx = -1
        self.wid = wid
        self.task_description = None
        self.current_env_name = None


# ================================================================
# RolloutWorkerActor — ★ 修改：接入 Ctrl-World 无状态接口
# ================================================================
@ray.remote
class RolloutWorkerActor(BaseWorkerActor):
    def __init__(self, infer, replay, wid, stats_actor, cfg, benchmark_name,
                 num_step_cond, imagine_horizon, torch_dtype,
                 reward_infer, ctrl_world_infer, reward_scale, gamma, lambda_,
                 wm_replay_buffer, real_traj_collect_interval):
        super().__init__(infer, replay, wid, stats_actor, cfg, benchmark_name)
        setup_debugger("rollout_worker", wid)
        self.env_outcome = [deque(maxlen=100) for _ in range(self.num_tasks)]
        self.local_buffer = []
        self.episodes = deque(maxlen=100)
        self.num_step_cond = num_step_cond
        self.imagine_horizon = imagine_horizon
        if self.imagine_horizon % NUM_ACTIONS_CHUNK != 0:
            raise ValueError(
                f"imagine_horizon ({self.imagine_horizon}) must be divisible by "
                f"NUM_ACTIONS_CHUNK ({NUM_ACTIONS_CHUNK})"
            )
        self.num_policy_chunks = self.imagine_horizon // NUM_ACTIONS_CHUNK
        self.torch_dtype = torch_dtype
        self.reward_infer = reward_infer
        self.ctrl_world_infer = ctrl_world_infer  # ★ 替换 denoiser_infer
        self.ctrl_world_num_history = int(
            ray.get(self.ctrl_world_infer.get_num_history.remote())
        )
        self.ctrl_world_infer_stride = int(
            ray.get(self.ctrl_world_infer.get_stride.remote())
        )
        self.reward_scale = reward_scale
        self.gamma = gamma
        self.lambda_ = lambda_
        self.wm_replay_buffer = wm_replay_buffer
        self.real_traj_collect_interval = real_traj_collect_interval

    def _reset_and_select_env(self, seed=None):
        failure_counts = np.array([sum(history) for history in self.env_outcome])
        env_weights = failure_counts + 1
        probabilities = env_weights / np.sum(env_weights)
        self.current_env_idx = np.random.choice(self.num_tasks, p=probabilities)
        self.env = self.envs[self.current_env_idx]
        obs, info = self.env.reset(seed=seed)
        self.task_description = self.env.task_description
        self.current_env_name = self.env.get_name()
        return obs, info

    def run(self):
        try:
            imagine_step = 0
            while True:
                if imagine_step % self.real_traj_collect_interval == 0:
                    self.get_one_episode()
                imagine_step += 1
                experience = random.choice(self.episodes)
                (
                    obs_list,
                    reward_list,
                    done_list,
                    action_env_list,
                    task_description,
                ) = experience
                num_imagine_samples = 0
                for i in range(len(obs_list) - self.num_step_cond):
                    obs_list_sub = obs_list[i:i + self.num_step_cond]
                    obs_tensor = torch.stack(
                        [self._obs_to_views(obs) for obs in obs_list_sub], dim=0
                    )  # [T,M,C,H,W]
                    raw_actions = action_env_list[i:i + self.num_step_cond - 1]
                    first_frame_action = (
                        np.asarray(action_env_list[i - 1], dtype=np.float32)
                        if i > 0
                        else np.zeros(ACTION_DIM, dtype=np.float32)
                    )
                    action_window = torch.cat(
                        [
                            torch.from_numpy(first_frame_action).unsqueeze(0),
                            torch.from_numpy(np.stack(raw_actions)).float(),
                        ],
                        dim=0,
                    )  # frame-aligned actions for the observation window
                    last_succ_prob = self.predict_rew_end(obs_tensor[-1], task_description)[0]

                    latent_history, current_latent = ray.get(
                        self.ctrl_world_infer.init_latent_state.remote(obs_tensor)
                    )

                    end = False
                    for j in range(self.num_policy_chunks):
                        inputs_t = self.obs2inp(obs_tensor[-1], task_description)
                        act_norm, action_env, action_token, logits, value = ray.get(
                            self.infer.request.remote(inputs_t, deterministic=False))
                        num_imagine_samples += 1
                        chunk_reward = 0.0
                        action_env = torch.from_numpy(
                            np.asarray(action_env, dtype=np.float32)
                        )
                        stride = self.ctrl_world_infer_stride
                        for group_start in range(0, len(action_env), stride):
                            future_actions = action_env[group_start:group_start + stride]
                            valid_count = len(future_actions)
                            if valid_count < stride:
                                pad_value = future_actions[-1:] if valid_count else torch.zeros(1, ACTION_DIM)
                                future_actions = torch.cat(
                                    [future_actions, pad_value.repeat(stride - valid_count, 1)],
                                    dim=0,
                                )
                            action_condition = torch.cat(
                                [
                                    action_window[-(self.ctrl_world_num_history + 1):],
                                    future_actions,
                                ],
                                dim=0,
                            )
                            future_obs, future_latents = self.predict_next_chunk(
                                current_latent,
                                latent_history,
                                action_condition,
                                task_description,
                            )
                            for k in range(valid_count):
                                nxt = future_obs[k]
                                next_latent = future_latents[k]
                                latent_history = torch.cat(
                                    [latent_history[1:], current_latent.unsqueeze(0)], dim=0
                                )
                                current_latent = next_latent
                                obs_tensor = torch.roll(obs_tensor, -1, dims=0)
                                obs_tensor[-1] = nxt
                                action_window = torch.cat(
                                    [
                                        action_window[1:],
                                        future_actions[k:k + 1],
                                    ],
                                    dim=0,
                                )
                                succ_prob, end = self.predict_rew_end(nxt, task_description)
                                rew = succ_prob - last_succ_prob
                                last_succ_prob = succ_prob
                                chunk_reward += rew * self.reward_scale
                                if end:
                                    break
                            if end:
                                break
                        self.local_buffer.append((inputs_t, action_token, chunk_reward, logits, value))
                        if end:
                            break
                    if self.local_buffer:
                        if end:
                            self._process_traj(self.local_buffer, 0.0)
                        else:
                            inputs_t = self.obs2inp(obs_tensor[-1], task_description)
                            _, _, _, _, bootstrap_val = ray.get(
                                self.infer.request.remote(inputs_t, deterministic=False))
                            self._process_traj(self.local_buffer, bootstrap_val)
                        imagine_rewards = [exp[2] for exp in self.local_buffer]
                        avg_imagine_reward = sum(imagine_rewards) / len(imagine_rewards)
                        self.stats_actor.add_imagine_reward.remote(avg_imagine_reward, self.wid, num_imagine_samples)
                    self.local_buffer.clear()
        except Exception as e:
            import traceback
            print(f"[ERROR] RolloutWorker {self.wid} run() 崩溃: {e}", flush=True)
            traceback.print_exc(); raise

    def get_one_episode(self):
        obs_list, reward_list, done_list = [], [], []
        action_env_list = []
        current_seed = int(time.time() * 1000) + self.wid + os.getpid()
        obs, info = self._reset_and_select_env(seed=current_seed)
        obs_list.append(obs)
        step_count = 0
        reward_sum, time_start, step_count_total = 0.0, time.time(), 0
        while True:
            inputs_t = prepare_one_obs(self.cfg, self.processor, obs, self.task_description, self.torch_dtype)
            act_norm, action_env, action_token, logits, value = ray.get(
                self.infer.request.remote(inputs_t, deterministic=False))
            step_count += 1
            chunk_reward, done = 0.0, False
            for i in range(len(action_env)):
                single_action = action_env[i]
                nxt, r, term, trunc, info = self.env.step(single_action)
                obs_list.append(nxt)
                reward_list.append(r)
                done_list.append(term or trunc)
                action_env_list.append(np.asarray(action_env[i], dtype=np.float32))
                reward_sum += r
                chunk_reward += r * self.reward_scale
                step_count_total += 1
                if term or trunc:
                    done = True; break
            obs = nxt
            if done:
                step_time = (time.time() - time_start) / max(step_count_total, 1)
                success = float(info.get('is_success', 0.0))
                self.env_outcome[self.current_env_idx].append(1.0 - success)
                self.stats_actor.add_episode_return.remote(
                    self.current_env_name, reward_sum, step_time,
                    step_count_total, success, actor_id=self.wid, step_num=step_count)
                self.episodes.append(
                    (
                        obs_list,
                        reward_list,
                        done_list,
                        action_env_list,
                        self.task_description,
                    )
                )
                self._process_episode_for_wm(
                    obs_list, reward_list, action_env_list, self.task_description
                )
                break

    def _obs_to_views(self, obs):
        return torch.stack(
            [
                image_to_tensor(obs["full_image"], "cpu"),
                image_to_tensor(obs["wrist_image"], "cpu"),
            ],
            dim=0,
        )

    def _process_episode_for_wm(self, obs_list, reward_list, action_env_list, task_description):
        T = len(obs_list)
        if T < self.num_step_cond + 1:
            return
        obs_images = []
        for obs_dict in obs_list:
            obs_images.append(self._obs_to_views(obs_dict).numpy())
        obs_images = np.stack(obs_images, axis=0)
        actions = np.stack(action_env_list, axis=0)
        num_valid_windows = T - self.num_step_cond
        wm_batch = []
        for window_idx in range(num_valid_windows):
            obs_start_idx = window_idx
            obs_end_idx = obs_start_idx + self.num_step_cond + 1
            act_start_idx = window_idx
            act_end_idx = act_start_idx + self.num_step_cond
            rew_idx = act_end_idx - 1
            window_obs = obs_images[obs_start_idx:obs_end_idx]
            window_act = actions[act_start_idx:act_end_idx]
            window_rew = reward_list[rew_idx]
            wm_batch.append(WMExperience(
                obs=window_obs.astype(np.float32),
                act=window_act.astype(np.float32),
                rew=float(window_rew),
                instruction=task_description,
            ))
        if wm_batch:
            self.wm_replay_buffer.add_batch.remote(wm_batch)

    def _process_traj(self, traj_segment, bootstrap_val):
        rets, advs = [], []
        gae = 0.0
        for i in reversed(range(len(traj_segment))):
            _, _, r, _, v = traj_segment[i]
            nv = bootstrap_val if i == len(traj_segment) - 1 else traj_segment[i+1][4]
            delta = r + self.gamma * nv - v
            gae = delta + self.gamma * self.lambda_ * gae
            advs.append(gae)
            rets.append(gae + v)
        advs.reverse(); rets.reverse()
        advs_np = np.array(advs, dtype=np.float32)
        batch = []
        for i, (s, a_token, _, logits, _) in enumerate(traj_segment):
            batch.append(Experience(
                obs=s,
                action_token=a_token.astype(np.int64),
                advantage=float(advs_np[i]),
                behaviour_logits=logits.astype(np.float32),
                value_target=float(rets[i]),
            ))
        self.replay.add_batch.remote(batch)

    def predict_next_chunk(
        self, current_latent, latent_history, action_condition, task_description
    ):
        future_obs, future_latents = ray.get(self.ctrl_world_infer.request.remote(
            current_latent, latent_history, action_condition.float(), task_description
        ))
        return future_obs, future_latents

    def predict_rew_end(self, next_obs, task_description):
        if next_obs.ndim != 4:
            raise ValueError(f"Expected multi-view observation, got {tuple(next_obs.shape)}")
        reward_obs = {"full_image": tensor_to_image(next_obs[0])}
        inputs = prepare_one_obs(
            self.cfg, self.processor, reward_obs, task_description, self.torch_dtype
        )
        logits = ray.get(self.reward_infer.request.remote(inputs))
        probs = torch.softmax(logits, dim=-1)
        succ_prob = probs[1].item()
        end = logits.argmax().item()
        return succ_prob, end

    def obs2inp(self, obs, task_description):
        if obs.ndim != 4 or obs.shape[0] < 2:
            raise ValueError(f"Expected [M,C,H,W] with two views, got {tuple(obs.shape)}")
        obs_for_vla = {
            "full_image": tensor_to_image(obs[0]),
            "wrist_image": tensor_to_image(obs[1]),
        }
        inputs = prepare_one_obs(self.cfg, self.processor, obs_for_vla, task_description, self.torch_dtype)
        return inputs


# ================================================================
# EvaluationWorkerActor — 与原版完全一致
# ================================================================
@ray.remote
class EvaluationWorkerActor(BaseWorkerActor):
    def __init__(self, infer, wid, stats_actor, cfg, benchmark_name, torch_dtype):
        super().__init__(infer, None, wid, stats_actor, cfg, benchmark_name)
        setup_debugger("eval_worker", self.wid)
        self.torch_dtype = torch_dtype
        print(f"EvaluationWorker {self.wid}: 环境初始化完成。")

    def _reset_and_select_env(self, seed=None):
        self.current_env_idx = (self.current_env_idx + 1) % self.num_tasks
        self.env = self.envs[self.current_env_idx]
        obs, info = self.env.reset(seed=seed)
        self.task_description = self.env.task_description
        self.current_env_name = self.env.get_name()
        return obs, info

    def run(self):
        try:
            current_seed = int(time.time() * 1000) + os.getpid() + random.randint(0, 10000)
            obs, info = self._reset_and_select_env(seed=current_seed)
            while True:
                reward_sum, time_start, step_count_total, done = 0.0, time.time(), 0, False
                while not done:
                    inputs_t = prepare_one_obs(self.cfg, self.processor, obs, self.task_description, self.torch_dtype)
                    _, action_env, _, _, _ = ray.get(self.infer.request.remote(inputs_t, deterministic=True))
                    for i in range(len(action_env)):
                        single_action = action_env[i]
                        obs, r, term, trunc, info = self.env.step(single_action)
                        reward_sum += r; step_count_total += 1
                        if term or trunc: done = True; break
                step_time = (time.time() - time_start) / max(step_count_total, 1)
                success = float(info.get('is_success', 0.0))
                self.stats_actor.add_episode_return.remote(
                    f"eval_{self.current_env_name}", reward_sum, step_time,
                    step_count_total, success, actor_id=None, step_num=step_count_total)
                current_seed = int(time.time() * 1000) + os.getpid() + random.randint(0, 10000)
                obs, info = self._reset_and_select_env(seed=current_seed)
        except Exception as e:
            import traceback
            print(f"[ERROR] EvaluationWorker {self.wid} run() 崩溃: {e}", flush=True)
            traceback.print_exc(); raise


# ================================================================
# 3. 推理器 (InferenceActor) — 与原版完全一致
# ================================================================
@ray.remote(num_gpus=1)
class InferenceActor(InferenceActorCom):
    def __init__(self, actor_id, cfg, stats_actor, torch_dtype, inference_batch, inference_timeout_ms):
        super().__init__()
        setup_debugger("inference", actor_id)
        self.actor_id = actor_id
        print(f"InferenceActor {actor_id}: 正在加载 OpenVLA ActorCritic...")
        self.model = ActorCritic(cfg, torch_dtype=torch_dtype)
        self.model.cuda()
        self.model.eval()
        self.processor = self.model.processor
        self.cfg = cfg
        self.stats_actor = stats_actor

        self.batch_size = inference_batch
        self.timeout_sec = inference_timeout_ms / 1000.0
        self.requests, self.promises = [], []
        self.last_process_time = time.time()

        loop = asyncio.get_event_loop()
        self._bg_task = loop.create_task(self._loop())
        self._bg_task.add_done_callback(self._on_bg_task_done)
        print(f"InferenceActor {self.actor_id} 初始化于 GPU: {ray.get_gpu_ids()} (批次超时: {inference_timeout_ms}ms)")

    def get_model_keys(self):
        if self.model is None:
            return {}
        sd = self.model.state_dict()
        res = {k: float(v.abs().sum().item()) for k, v in sd.items()}
        return res

    def _on_bg_task_done(self, task):
        try:
            task.result()
        except Exception as e:
            import traceback
            print(f"[ERROR] InferenceActor {self.actor_id} 后台任务异常: {e}", flush=True)
            traceback.print_exc()

    async def request(self, inputs_t, deterministic=False):
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        self.requests.append((inputs_t, deterministic))
        self.promises.append(fut)
        return await fut

    async def _loop(self):
        while True:
            should_process = self.requests and (
                len(self.requests) >= self.batch_size or
                time.time() - self.last_process_time > self.timeout_sec
            )
            if not should_process:
                await asyncio.sleep(0.0005)
                continue

            requests_to_process = self.requests
            promises_to_process = self.promises
            self.requests, self.promises = [], []
            self.last_process_time = time.time()

            inputs_list = [r[0] for r in requests_to_process]
            deterministic_flags = [r[1] for r in requests_to_process]
            t_loop_start = time.time()
            try:
                inputs_batch = self.model.prepare_inputs_batch(inputs_list)
                with torch.inference_mode():
                    action_logits, value = self.model(inputs_batch)
                    _, action_tokens_all, normalized_actions_all = self.model.post_process(
                        action_logits, deterministic=deterministic_flags)
                    action_tokens = action_tokens_all.view(
                        -1, NUM_ACTIONS_CHUNK, ACTION_DIM).cpu().numpy()
                    logits = action_logits.view(
                        -1, NUM_ACTIONS_CHUNK, ACTION_DIM, action_logits.shape[-1]
                    ).float().cpu().numpy()
                    values = value.to(torch.float32).cpu().numpy()

                actions_env = []
                for i in range(normalized_actions_all.shape[0]):
                    a_env = self.model.vla._unnormalize_actions(
                        normalized_actions_all[i], self.cfg.unnorm_key)
                    actions_env.append(a_env.astype(np.float32))

                for i in range(len(promises_to_process)):
                    promises_to_process[i].set_result((
                        normalized_actions_all[i],
                        actions_env[i],
                        action_tokens[i],
                        logits[i],
                        values[i]
                    ))
                loop_duration = time.time() - t_loop_start
                self.stats_actor.add_timing_metric.remote("Inference/loop_time_s", loop_duration)
            except Exception as e:
                import traceback
                print(f"[ERROR] InferenceActor {self.actor_id} 批处理失败: {e}", flush=True)
                traceback.print_exc()
                for p in promises_to_process:
                    if not p.done():
                        p.set_exception(e)
                raise

    def forward_test(self):
        return


# ================================================================
# RewardInferenceActor — 与原版完全一致
# ================================================================
@ray.remote(num_gpus=1)
class RewardInferenceActor(InferenceActorCom):
    def __init__(self, actor_id, stats_actor, agent_config_path, inference_batch, inference_timeout_ms, reward_checkpoint=None):
        super().__init__()
        setup_debugger("reward_inference", actor_id)
        self.actor_id = actor_id
        self.stats_actor = stats_actor
        agent_cfg = OmegaConf.load(agent_config_path)
        reward_ckpt_path = reward_checkpoint if reward_checkpoint else agent_cfg.reward_model_path
        print(f"RewardInferenceActor {actor_id}: 加载 Reward Model checkpoint from {reward_ckpt_path}")
        self.reward_model, self.rew_cfg = load_reward_model(
            model_path=reward_ckpt_path,
            device="cuda",
            pretrained_checkpoint=agent_cfg.openvla_path,
            focal_alpha=agent_cfg.reward_model.focal_alpha,
        )
        self.model = self.reward_model
        self.batch_size = inference_batch
        self.timeout_sec = inference_timeout_ms / 1000.0
        self.requests, self.promises = [], []
        self.last_process_time = time.time()

        loop = asyncio.get_event_loop()
        self._bg_task = loop.create_task(self._loop())
        self._bg_task.add_done_callback(self._on_bg_task_done)
        print(f"RewardInferenceActor {self.actor_id} 初始化于 GPU: {ray.get_gpu_ids()} (批次超时: {inference_timeout_ms}ms)")

    def _on_bg_task_done(self, task):
        try:
            task.result()
        except Exception as e:
            import traceback
            print(f"[ERROR] RewardInferenceActor {self.actor_id} 后台任务异常: {e}", flush=True)
            traceback.print_exc()

    async def request(self, inputs_t):
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        self.requests.append(inputs_t)
        self.promises.append(fut)
        return await fut

    async def _loop(self):
        while True:
            should_process = self.requests and (
                len(self.requests) >= self.batch_size or
                time.time() - self.last_process_time > self.timeout_sec
            )
            if not should_process:
                await asyncio.sleep(0.0005)
                continue

            requests_to_process = self.requests
            promises_to_process = self.promises
            self.requests, self.promises = [], []
            self.last_process_time = time.time()

            inputs_list = requests_to_process
            t_loop_start = time.time()
            try:
                inputs_batch = self.reward_model.prepare_inputs_batch(inputs_list)
                with torch.inference_mode():
                    logits = self.reward_model.forward(inputs_batch).cpu()
                for i in range(len(promises_to_process)):
                    promises_to_process[i].set_result(logits[i])
                loop_duration = time.time() - t_loop_start
                self.stats_actor.add_timing_metric.remote("Inference/reward_loop_time_s", loop_duration)
            except Exception as e:
                import traceback
                print(f"[ERROR] RewardInferenceActor {self.actor_id} 批处理失败: {e}", flush=True)
                traceback.print_exc()
                for p in promises_to_process:
                    if not p.done():
                        p.set_exception(e)
                raise


# ================================================================
# ★ CtrlWorldInferenceActor — 替换 DenoiserInferenceActor
#   使用 CtrlWorldEnvBatch 的无状态接口，Worker 端维护 latent_history
# ================================================================
@ray.remote(num_gpus=1)
class CtrlWorldInferenceActor(InferenceActorCom):
    def __init__(self, actor_id, stats_actor, args, inference_batch, inference_timeout_ms):
        super().__init__()
        setup_debugger("ctrl_world_inference", actor_id)
        self.actor_id = actor_id
        self.stats_actor = stats_actor

        print(f"CtrlWorldInferenceActor {actor_id}: 正在加载 Ctrl-World 模型...")

        # 构建 Ctrl-World 配置
        cfg = wm_args()
        cfg.svd_model_path = args.svd_model_path
        cfg.clip_model_path = args.clip_model_path
        cfg.num_cams = args.num_cams
        cfg.num_history = args.num_history
        cfg.num_frames = args.num_frames_pred
        cfg.action_dim = ACTION_DIM
        cfg.text_cond = True
        cfg.frame_level_cond = True
        cfg.his_cond_zero = False

        device = torch.device("cuda")
        ctrl_world = CrtlWorld(cfg).to(device).to(torch.bfloat16)
        ctrl_world.eval()

        if not os.path.isfile(args.ctrl_world_ckpt):
            raise FileNotFoundError(f"Ctrl-World checkpoint not found: {args.ctrl_world_ckpt}")
        print(f"CtrlWorldInferenceActor {actor_id}: 加载 checkpoint: {args.ctrl_world_ckpt}")
        ckpt = torch.load(args.ctrl_world_ckpt, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
        ctrl_world.load_state_dict(state_dict, strict=True)
        print(f"CtrlWorldInferenceActor {actor_id}: checkpoint 加载完成。")

        stat_path = args.condition_stat_path
        if stat_path is None:
            stat_path = str(Path(args.ctrl_world_ckpt).resolve().parent / "condition_stat.json")
        if not os.path.isfile(stat_path):
            raise FileNotFoundError(f"Ctrl-World condition statistics not found: {stat_path}")
        with open(stat_path, "r", encoding="utf-8") as f:
            condition_stats = json.load(f)
        if condition_stats.get("condition_mode") != "action":
            raise ValueError(
                f"Expected action condition statistics, got "
                f"{condition_stats.get('condition_mode')!r}"
            )
        if condition_stats.get("normalize_condition") != "bounds":
            raise ValueError("Ctrl-World checkpoint must use bounds-normalized conditions")
        if (
            set(condition_stats.get("skip_choices", [])) != {1}
            or int(condition_stats.get("skip_his_multiplier", -1)) != 1
            or float(condition_stats.get("skip_his_zero_prob", -1.0)) != 0.0
        ):
            raise ValueError(
                "Ctrl-World checkpoint statistics are not frame-aligned for "
                "single-step delta actions"
            )
        condition_low = torch.tensor(condition_stats["condition_p01"], dtype=torch.float32)
        condition_high = torch.tensor(condition_stats["condition_p99"], dtype=torch.float32)
        if condition_low.shape != (ACTION_DIM,) or condition_high.shape != (ACTION_DIM,):
            raise ValueError(
                f"Expected {ACTION_DIM}-D condition bounds, got "
                f"{tuple(condition_low.shape)} and {tuple(condition_high.shape)}"
            )
        if not torch.all(condition_high > condition_low):
            raise ValueError("Every condition p99 value must be greater than p01")

        # 创建 CtrlWorldEnvBatch（仅用其无状态接口和编解码工具）
        @dataclass
        class FakeCfg:
            horizon: int = 220

        self.ctrl_world_env = CtrlWorldEnvBatch(
            ctrl_world_model=ctrl_world,
            cfg=FakeCfg(horizon=220),
            reward_model=None,       # Worker 端通过 RewardInferenceActor 独立推理
            reward_cfg=None,
            processor=None,
            torch_dtype=torch.bfloat16,
            instructions=None,
            num_cams=args.num_cams,
            target_height=args.target_height,
            target_width=args.target_width,
            num_frames_pred=args.num_frames_pred,
            num_inference_steps=args.num_inference_steps,
            condition_low=condition_low,
            condition_high=condition_high,
        )

        self.model = ctrl_world  # InferenceActorCom 期望 self.model
        self.device = self.ctrl_world_env.device
        self.batch_size = inference_batch
        self.timeout_sec = inference_timeout_ms / 1000.0
        self.requests, self.promises = [], []
        self.last_process_time = time.time()

        loop = asyncio.get_event_loop()
        self._bg_task = loop.create_task(self._loop())
        self._bg_task.add_done_callback(self._on_bg_task_done)
        print(f"CtrlWorldInferenceActor {self.actor_id} 初始化于 GPU: {ray.get_gpu_ids()} (批次超时: {inference_timeout_ms}ms)")

    def _on_bg_task_done(self, task):
        try:
            task.result()
        except Exception as e:
            import traceback
            print(f"[ERROR] CtrlWorldInferenceActor {self.actor_id} 后台任务异常: {e}", flush=True)
            traceback.print_exc()

    # ★ 无状态接口：Worker 传入完整状态
    async def request(self, current_latent, latent_history, action_condition, instruction):
        """
        Args:
            current_latent: [4, latent_h*num_cams, latent_w]
            latent_history: [num_history, 4, latent_h*num_cams, latent_w] (CPU tensor)
            action_condition: [num_history+num_frames, act_dim], raw delta actions
            instruction:    str
        Returns:
            future observations/latents for frames 1: of the generated chunk
        """
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        self.requests.append((current_latent, latent_history, action_condition, instruction))
        self.promises.append(fut)
        return await fut

    def init_latent_state(self, obs_seq):
        """
        Worker 端调用：从观测序列初始化 latent_history。
        Args:
            obs_seq: [T,M,C,H,W] in [-1, 1] (CPU tensor)
        Returns:
            (latent_history, current_latent)
        """
        history, current = self.ctrl_world_env.init_latent_state(
            obs_seq.unsqueeze(0).to(self.device)
        )
        return history[0].cpu(), current[0].cpu()

    def get_num_history(self):
        return self.ctrl_world_env.num_history

    def get_stride(self):
        return self.ctrl_world_env.num_frames_pred - 1

    async def _loop(self):
        while True:
            should_process = self.requests and (
                len(self.requests) >= self.batch_size or
                time.time() - self.last_process_time > self.timeout_sec
            )
            if not should_process:
                await asyncio.sleep(0.0005)
                continue

            requests_to_process = self.requests
            promises_to_process = self.promises
            self.requests, self.promises = [], []
            self.last_process_time = time.time()

            # 拆分请求
            current_latent_list = [req[0] for req in requests_to_process]
            latent_history_list = [req[1] for req in requests_to_process]
            action_condition_list = [req[2] for req in requests_to_process]
            instructions = [req[3] for req in requests_to_process]

            # Stack into batch
            current_latent_batch = torch.stack(current_latent_list, dim=0).to(self.device)
            latent_history_batch = torch.stack(latent_history_list, dim=0).to(self.device)  # [B, num_history, 4, ...]
            action_condition_batch = torch.stack(action_condition_list, dim=0).to(self.device)

            t_loop_start = time.time()
            try:
                with torch.inference_mode():
                    future_obs, future_latents = self.ctrl_world_env.predict_chunk_stateless(
                        current_latent=current_latent_batch,
                        latent_history=latent_history_batch,
                        action_condition=action_condition_batch,
                        instructions=instructions,
                        output_size=(224, 224),
                    )
                    future_obs = future_obs.float().cpu()
                    future_latents = future_latents.cpu()

                for i, promise in enumerate(promises_to_process):
                    promise.set_result((future_obs[i], future_latents[i]))

                loop_duration = time.time() - t_loop_start
                self.stats_actor.add_timing_metric.remote("Inference/ctrl_world_loop_time_s", loop_duration)
            except Exception as e:
                import traceback
                print(f"[ERROR] CtrlWorldInferenceActor {self.actor_id} 批处理失败: {e}", flush=True)
                traceback.print_exc()
                for p in promises_to_process:
                    if not p.done():
                        p.set_exception(e)
                raise


# ================================================================
# 4. 训练器 (TrainerActor) — ★ 修改：移除 Denoiser Engine，Ctrl-World 冻结
# ================================================================
@ray.remote(num_gpus=1)
class TrainerActor(TrainerActorCom):
    def __init__(self, rank, world_size, replay_buffer, cfg, train_batch_size, accumulation_steps,
                 use_bf16, torch_dtype, policy_lr, value_lr, gamma, lambda_, clip_eps, vf_coef,
                 ent_coef, kl_coef, reward_scale, value_warmup_steps, policy_warmup_steps,
                 policy_train_start_step, train_iters, clip_mode,
                 # World Model 相关参数
                 wm_replay_buffer, agent_config_path, trainer_config_path,
                 reward_batch_size, reward_accumulation_steps, reward_lr, reward_warmup_steps,
                 reward_train_interval,
                 reward_checkpoint=None):
        super().__init__()
        setup_debugger("trainer", rank)
        self.rank = rank
        self.world_size = world_size
        self.replay_buffer = replay_buffer
        self.cfg = cfg
        self.model = None
        self.optimizer = None
        self.base_model = None
        self.data_dtype = None
        self.next_ready_batch = None
        self.data_fetching_task = None

        self.reward_checkpoint = reward_checkpoint

        self.train_batch_size = train_batch_size
        self.accumulation_steps = accumulation_steps
        self.super_batch_size = train_batch_size * accumulation_steps
        self.use_bf16 = use_bf16
        self.torch_dtype = torch_dtype
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.gamma = gamma
        self.lambda_ = lambda_
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.kl_coef = kl_coef
        self.reward_scale = reward_scale
        self.value_warmup_steps = value_warmup_steps
        self.policy_warmup_steps = policy_warmup_steps
        self.policy_train_start_step = policy_train_start_step
        self.train_iters = train_iters
        self.clip_mode = clip_mode

        self.global_step = 0

        # ============ World Model 相关 ============
        self.wm_replay_buffer = wm_replay_buffer
        self.agent_config_path = agent_config_path
        self.trainer_config_path = trainer_config_path

        # Reward Model 参数
        self.reward_batch_size = reward_batch_size
        self.reward_accumulation_steps = reward_accumulation_steps
        self.reward_super_batch_size = reward_batch_size * reward_accumulation_steps
        self.reward_lr = reward_lr
        self.reward_warmup_steps = reward_warmup_steps

        # World Model 训练频率
        self.reward_train_interval = reward_train_interval

        # World Model 独立训练步数计数器
        self.reward_step = 0

        # World Model engines (将在 setup_deepspeed_group 中初始化)
        # ★ 注意：不再有 denoiser_engine，Ctrl-World 冻结不训练
        self.reward_engine = None
        self.reward_model = None
        self.reward_cfg = None
        self.processor = None

        # WM 数据获取
        self.next_ready_wm_batch = None
        self.wm_data_fetching_task = None

        print(f"TrainerActor Rank {self.rank} 初始化于 GPU: {ray.get_gpu_ids()}")

    def get_model_keys(self):
        if self.model is None:
            print("模型尚未初始化。请先调用 setup_deepspeed_group()。")
            return {}
        module = self.model.module if hasattr(self.model, "module") else self.model
        sd = module.state_dict()
        res = {k: float(v.abs().sum().item()) for k, v in sd.items()}
        return res

    def get_node_ip(self):
        return ray.util.get_node_ip_address()

    def setup_deepspeed_group(self, master_addr, master_port):
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["LOCAL_RANK"] = "0"
        deepspeed.init_distributed(dist_backend="nccl")

        # ============ 1. 初始化 Actor Engine (bf16) ============
        print(f"Trainer {self.rank}: 正在加载 OpenVLA ActorCritic...")
        model = ActorCritic(self.cfg, torch_dtype=self.torch_dtype)
        self.base_model = model

        param_groups = self.base_model.get_parameter_groups()
        optimizer_params = [
            {"params": pg["params"], "name": pg["name"],
             "lr": self.policy_lr if pg["name"] == "policy" else self.value_lr}
            for pg in param_groups
        ]

        actor_ds_config = {
            "train_micro_batch_size_per_gpu": self.train_batch_size,
            "gradient_accumulation_steps": self.accumulation_steps,
            "optimizer": {"type": "AdamW", "params": {}},
            "bf16": {"enabled": self.use_bf16},
            "zero_optimization": {
                "stage": 2, "allgather_partitions": True, "allgather_bucket_size": 5e8,
                "reduce_scatter": True, "reduce_bucket_size": 5e8, "overlap_comm": True,
                "contiguous_gradients": True
            },
            "gradient_clipping": 1.0,
        }

        if actor_ds_config.get("bf16", {}).get("enabled", False):
            self.data_dtype = torch.bfloat16
        else:
            self.data_dtype = torch.float32

        self.model, self.optimizer, _, _ = deepspeed.initialize(
            model=model, config=actor_ds_config, model_parameters=optimizer_params
        )
        print(f"TrainerActor Rank {self.rank}: Actor DeepSpeed engine (bf16) 初始化完成。")

        n_total = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Actor 总参数量: {n_total:,}, 可训练参数量: {n_trainable:,}")

        # ============ 2. 初始化 Reward Model Engine (bf16) ============
        # ★ 注意：不再初始化 Denoiser Engine，Ctrl-World 冻结不训练
        print(f"Trainer {self.rank}: 正在加载 Reward Model...")
        from envs.utils import load_reward_model

        agent_cfg = OmegaConf.load(self.agent_config_path)
        reward_ckpt_path = self.reward_checkpoint if self.reward_checkpoint else agent_cfg.reward_model_path
        print(f"Trainer {self.rank}: 加载 Reward Model checkpoint from {reward_ckpt_path}")
        self.reward_model, self.reward_cfg = load_reward_model(
            model_path=reward_ckpt_path,
            device="cuda",
            pretrained_checkpoint=agent_cfg.openvla_path,
            focal_alpha=agent_cfg.reward_model.focal_alpha,
        )
        self.processor = get_processor(self.reward_cfg)

        reward_ds_config = {
            "train_micro_batch_size_per_gpu": self.reward_batch_size,
            "gradient_accumulation_steps": self.reward_accumulation_steps,
            "optimizer": {"type": "AdamW", "params": {"lr": self.reward_lr}},
            "bf16": {"enabled": True},
            "zero_optimization": {"stage": 2},
            "gradient_clipping": 1.0,
        }

        self.reward_engine, _, _, _ = deepspeed.initialize(
            model=self.reward_model,
            config=reward_ds_config,
            model_parameters=self.reward_model.parameters(),
        )
        print(f"TrainerActor Rank {self.rank}: Reward Model DeepSpeed engine (bf16) 初始化完成。")

        n_reward = sum(p.numel() for p in self.reward_model.parameters())
        print(f"Reward Model 总参数量: {n_reward:,}")

        print(f"TrainerActor Rank {self.rank}: 所有 DeepSpeed engines 初始化完成 (Actor + Reward Model)。")

        # 启动数据获取循环
        self.data_fetching_task = asyncio.get_event_loop().create_task(self._data_fetching_loop())
        self.wm_data_fetching_task = asyncio.get_event_loop().create_task(self._wm_data_fetching_loop())

    async def save_agent(self, ckpt_dir, step):
        os.makedirs(ckpt_dir, exist_ok=True)
        self.base_model.save_model(ckpt_dir, epoch=step)
        print(f"[Trainer {self.rank}] 已保存 checkpoint -> {ckpt_dir}/agent_lora_epoch_{step}, agent_extra_layers_epoch_{step}.pt")

    def _get_current_lr(self, current_step, peak_lr, warmup_steps, total_steps, start_step=0):
        if current_step < start_step:
            return 0.0
        effective_step = current_step - start_step
        if effective_step < warmup_steps:
            return peak_lr * (effective_step / warmup_steps)
        progress = (effective_step - warmup_steps) / (total_steps - start_step - warmup_steps)
        progress = min(progress, 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return peak_lr * cosine_decay

    def _train_policy(self, inputs_batch, act_token_t, adv_t, logits_old_t, v_targ_t, global_mean, global_std, current_lrs):
        """Policy 训练（包含 Value 网络）— 与原版完全一致"""
        epoch_losses, epoch_p_losses, epoch_v_losses, epoch_e_losses, epoch_kl_losses = [], [], [], [], []
        epoch_ent, epoch_kl_divs = [], []

        num_updates_in_epoch = self.super_batch_size // self.train_batch_size
        t_policy_train_start = time.time()

        for i in range(num_updates_in_epoch):
            start = i * self.train_batch_size
            end = start + self.train_batch_size
            mini_inputs = {k: v[start:end] for k, v in inputs_batch.items()}

            mini_act_token = act_token_t[start:end]
            mini_adv = adv_t[start:end]
            mini_logits_old = logits_old_t[start:end]
            mini_v_targ = v_targ_t[start:end]

            normalized_adv = (mini_adv - global_mean) / (global_std + 1e-8)
            action_logits, value = self.model.forward(mini_inputs)
            value = value.to(torch.float32)

            action_logits_reshape = action_logits.view(
                -1, NUM_ACTIONS_CHUNK, ACTION_DIM, action_logits.shape[-1]
            )

            value_loss = self.vf_coef * torch.mean((value - mini_v_targ) ** 2)

            if self.global_step < self.policy_train_start_step:
                loss = value_loss
                policy_loss = torch.tensor(0.0, device=loss.device)
                ent_loss = torch.tensor(0.0, device=loss.device)
                kl_loss = torch.tensor(0.0, device=loss.device)
                kl_div = 0.0
                ent = torch.tensor(0.0, device=loss.device)
            else:
                dist = torch.distributions.Categorical(logits=action_logits_reshape)
                logp = dist.log_prob(mini_act_token)

                with torch.no_grad():
                    dist_old = torch.distributions.Categorical(logits=mini_logits_old)
                    logp_old = dist_old.log_prob(mini_act_token)

                kl_div_tensor = kl.kl_divergence(dist_old, dist)
                kl_div = torch.mean(kl_div_tensor).item()
                kl_loss = self.kl_coef * torch.mean(kl_div_tensor)
                ratio = torch.exp(logp - logp_old)
                adv_unsqueezed = normalized_adv.unsqueeze(dim=-1).unsqueeze(dim=-1)
                surr1 = ratio * adv_unsqueezed
                if self.clip_mode == "gipo":
                    eps = 1e-9
                    sigma = 1.0
                    r_detach = ratio.clamp_min(eps).detach()
                    coeff = torch.exp(-0.5 * (torch.log(r_detach) / sigma) ** 2)
                    surr_soft = surr1 * coeff
                    policy_loss = -torch.mean(surr_soft)
                elif self.clip_mode == "ppo":
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_unsqueezed
                    policy_loss = -torch.mean(torch.min(surr1, surr2))
                elif self.clip_mode == "sapo":
                    tau_pos = 1.0
                    tau_neg = 2.0
                    if tau_pos <= 0 or tau_neg <= 0:
                        raise ValueError(f"tau_pos/tau_neg must be > 0, got {tau_pos}, {tau_neg}")
                    ratio_min = 1e-6
                    ratio_max = 1e6
                    r = ratio.clamp(ratio_min, ratio_max)
                    tau_pos_t = torch.full_like(adv_unsqueezed, tau_pos)
                    tau_neg_t = torch.full_like(adv_unsqueezed, tau_neg)
                    tau = torch.where(adv_unsqueezed > 0, tau_pos_t, tau_neg_t)
                    x = tau * (r - 1.0)
                    gate = torch.sigmoid(x) * (4.0 / tau)
                    surr_sapo = gate * adv_unsqueezed
                    policy_loss = -torch.mean(surr_sapo)
                else:
                    raise ValueError(f"Invalid CLIP_MODE: {self.clip_mode}")
                ent = torch.mean(dist.entropy())
                ent_loss = -self.ent_coef * ent
                loss = policy_loss + value_loss + ent_loss + kl_loss

            self.model.backward(loss)
            self.model.step()
            epoch_losses.append(loss.item())
            epoch_p_losses.append(policy_loss.item())
            epoch_v_losses.append(value_loss.item())
            epoch_e_losses.append(ent_loss.item())
            epoch_kl_losses.append(kl_loss.item())
            epoch_ent.append(ent.item())
            epoch_kl_divs.append(kl_div)
            if self.model.is_gradient_accumulation_boundary():
                self.global_step += 1

        policy_train_time = time.time() - t_policy_train_start

        return {
            'epoch_losses': epoch_losses,
            'epoch_p_losses': epoch_p_losses,
            'epoch_v_losses': epoch_v_losses,
            'epoch_e_losses': epoch_e_losses,
            'epoch_kl_losses': epoch_kl_losses,
            'epoch_ent': epoch_ent,
            'epoch_kl_divs': epoch_kl_divs,
            'policy_train_time': policy_train_time,
        }

    def _train_reward_model(self, wm_reward_inputs, wm_labels, current_lrs):
        """Reward Model 训练 (bf16) — 与原版一致"""
        should_train = (self.global_step % self.reward_train_interval == 0)
        epoch_reward_losses = []
        epoch_reward_tp, epoch_reward_tn, epoch_reward_fp, epoch_reward_fn = 0, 0, 0, 0
        reward_train_time = 0.0

        if not should_train:
            return {
                'should_train': False,
                'epoch_reward_losses': epoch_reward_losses,
                'reward_train_time': reward_train_time,
                'reward_tp': epoch_reward_tp,
                'reward_tn': epoch_reward_tn,
                'reward_fp': epoch_reward_fp,
                'reward_fn': epoch_reward_fn,
            }

        reward_lr = self._get_current_lr(self.reward_step, self.reward_lr, self.reward_warmup_steps, self.train_iters)
        for pg in self.reward_engine.optimizer.param_groups:
            pg['lr'] = reward_lr
        current_lrs['reward'] = reward_lr

        self.reward_engine.train()
        num_reward_updates = self.reward_super_batch_size // self.reward_batch_size
        t_reward_train_start = time.time()

        for i in range(num_reward_updates):
            start = i * self.reward_batch_size
            end = start + self.reward_batch_size
            mini_inputs = {k: v[start:end] for k, v in wm_reward_inputs.items()}
            mini_labels = wm_labels[start:end]

            logits = self.reward_engine.forward(mini_inputs)
            reward_loss, metrics = self.reward_engine.module.compute_loss_and_metrics(mini_inputs, mini_labels)
            self.reward_engine.backward(reward_loss)
            self.reward_engine.step()

            epoch_reward_losses.append(reward_loss.item())
            epoch_reward_tp += metrics["tp"].item()
            epoch_reward_tn += metrics["tn"].item()
            epoch_reward_fp += metrics["fp"].item()
            epoch_reward_fn += metrics["fn"].item()

        reward_train_time = time.time() - t_reward_train_start
        self.reward_step += 1

        return {
            'should_train': True,
            'epoch_reward_losses': epoch_reward_losses,
            'reward_train_time': reward_train_time,
            'reward_tp': epoch_reward_tp,
            'reward_tn': epoch_reward_tn,
            'reward_fp': epoch_reward_fp,
            'reward_fn': epoch_reward_fn,
        }

    async def _data_fetching_loop(self):
        """Actor 数据获取循环 — 与原版一致"""
        print(f"Trainer {self.rank}: 后台数据准备循环已启动 (超级批次大小: {self.super_batch_size})。")
        while True:
            try:
                if self.next_ready_batch is not None:
                    await asyncio.sleep(0.1)
                    continue

                while await self.replay_buffer.size.remote() < self.super_batch_size:
                    print(f"Trainer {self.rank} (BG): 等待 ReplayBuffer 填充至 {self.super_batch_size}...")
                    await asyncio.sleep(3)

                t_sample_start = time.time()
                obs_list, action_token_np, adv_np, logits_old_np, v_targ_np = \
                    await self.replay_buffer.sample.remote(self.super_batch_size)
                sample_time = time.time() - t_sample_start

                t_prep_start = time.time()
                inputs_batch = self.base_model.prepare_inputs_batch(obs_list)

                device = next(self.model.parameters()).device
                act_token_t = torch.tensor(action_token_np, dtype=torch.long, device=device)
                adv_t = torch.tensor(adv_np, dtype=torch.float32, device=device)
                logits_old_t = torch.tensor(logits_old_np, dtype=torch.float32, device=device)
                v_targ_t = torch.tensor(v_targ_np, dtype=torch.float32, device=device)
                prep_time = time.time() - t_prep_start

                self.next_ready_batch = {
                    'inputs_batch': inputs_batch,
                    'act_token': act_token_t,
                    'advantage': adv_t,
                    'logits_old': logits_old_t,
                    'value_target': v_targ_t,
                    'sample_time': sample_time,
                    'prep_time': prep_time
                }

            except Exception as e:
                print(f"Trainer {self.rank}: 数据采样失败: {e}。将在3秒后重试。")
                await asyncio.sleep(3)

    async def _wm_data_fetching_loop(self):
        """World Model 数据获取循环 — 只获取 Reward Model 训练数据"""
        wm_super_batch_size = self.reward_super_batch_size
        print(f"Trainer {self.rank}: WM 后台数据准备循环已启动 (超级批次大小: {wm_super_batch_size})。")

        while True:
            try:
                if self.next_ready_wm_batch is not None:
                    await asyncio.sleep(0.1)
                    continue

                while await self.wm_replay_buffer.size.remote() < wm_super_batch_size:
                    print(f"Trainer {self.rank} (BG-WM): 等待 WM ReplayBuffer 填充至 {wm_super_batch_size}...")
                    await asyncio.sleep(3)

                t_sample_start = time.time()
                obs_np, act_np, rew_np, instructions = \
                    await self.wm_replay_buffer.sample.remote(wm_super_batch_size)
                sample_time = time.time() - t_sample_start

                t_prep_start = time.time()
                device = next(self.model.parameters()).device

                # Reward Model 数据 - 需要准备 inputs_batch
                last_obs_np = obs_np[:, -1, 0]  # reward model uses primary camera [B,C,H,W]

                inputs_list = []
                for i in range(len(last_obs_np)):
                    obs_tensor = last_obs_np[i]  # [C, H, W] in [-1, 1]
                    obs_np_hwc = np.transpose(obs_tensor, (1, 2, 0))  # [H, W, C]
                    obs_img = ((obs_np_hwc + 1) / 2 * 255).astype(np.uint8)
                    obs_dict = {"full_image": obs_img}

                    inputs = prepare_one_obs(
                        self.reward_cfg,
                        self.processor,
                        obs_dict,
                        instructions[i],
                        torch.bfloat16,
                    )
                    inputs_list.append(inputs)

                reward_inputs_batch = self.reward_model.prepare_inputs_batch(inputs_list)
                labels = torch.tensor((rew_np > 0).astype(np.int64), dtype=torch.long, device=device)

                prep_time = time.time() - t_prep_start

                self.next_ready_wm_batch = {
                    'rew': torch.tensor(rew_np, dtype=torch.float32, device=device),
                    'reward_inputs_batch': reward_inputs_batch,
                    'labels': labels,
                    'instructions': instructions,
                    'sample_time': sample_time,
                    'prep_time': prep_time,
                }

            except Exception as e:
                import traceback
                print(f"Trainer {self.rank}: WM 数据采样失败: {e}。将在3秒后重试。")
                traceback.print_exc()
                await asyncio.sleep(3)

    async def run_training_epoch(self):
        # ============ 等待数据准备 ============
        if self.next_ready_batch is None:
            print(f"Trainer {self.rank}: 等待初始 Actor 超级批次...")
            while self.next_ready_batch is None:
                await asyncio.sleep(0.2)
            print(f"Trainer {self.rank}: Actor 初始数据已收到。")

        if self.next_ready_wm_batch is None:
            print(f"Trainer {self.rank}: 等待初始 WM 超级批次...")
            while self.next_ready_wm_batch is None:
                await asyncio.sleep(0.2)
            print(f"Trainer {self.rank}: WM 初始数据已收到，开始训练。")

        current_lrs = {}
        value_lr = self._get_current_lr(self.global_step, self.value_lr, self.value_warmup_steps, self.train_iters)
        policy_lr = self._get_current_lr(self.global_step, self.policy_lr, self.policy_warmup_steps, self.train_iters, start_step=self.policy_train_start_step)

        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'value':
                param_group['lr'] = value_lr; current_lrs['value'] = value_lr
            elif param_group['name'] == 'policy':
                param_group['lr'] = policy_lr; current_lrs['policy'] = policy_lr

        current_batch = self.next_ready_batch
        self.next_ready_batch = None

        inputs_batch = current_batch['inputs_batch']
        act_token_t = current_batch['act_token']
        adv_t = current_batch['advantage']
        logits_old_t = current_batch['logits_old']
        v_targ_t = current_batch['value_target']
        policy_sample_time = current_batch['sample_time']
        policy_prep_time = current_batch['prep_time']

        # 修正std 归一化
        local_sum = adv_t.sum()
        local_sq_sum = (adv_t * adv_t).sum()
        local_count = torch.tensor([adv_t.numel()], device=adv_t.device, dtype=torch.float32)

        stats_tensor = torch.stack([local_sum, local_sq_sum, local_count.squeeze(0)])
        distributed.all_reduce(stats_tensor, op=distributed.ReduceOp.SUM)

        global_sum, global_sq_sum, global_count = stats_tensor[0], stats_tensor[1], stats_tensor[2]
        global_mean = global_sum / torch.clamp(global_count, min=1.0)
        global_var = torch.clamp(global_sq_sum / torch.clamp(global_count, min=1.0) - global_mean * global_mean, min=1e-12)
        global_std = torch.sqrt(global_var)

        policy_result = self._train_policy(inputs_batch, act_token_t, adv_t, logits_old_t, v_targ_t, global_mean, global_std, current_lrs)

        # ============ 2. World Model 训练（只有 Reward Model）============
        wm_batch = self.next_ready_wm_batch
        reward_should_train = (self.global_step % self.reward_train_interval == 0)
        if reward_should_train:
            self.next_ready_wm_batch = None
        wm_reward_inputs = wm_batch['reward_inputs_batch']
        wm_labels = wm_batch['labels']
        wm_sample_time = wm_batch['sample_time']
        wm_prep_time = wm_batch['prep_time']

        # Reward Model 训练
        reward_result = self._train_reward_model(wm_reward_inputs, wm_labels, current_lrs)

        # ============ 汇总结果 ============
        avg_loss = np.mean(policy_result['epoch_losses'])
        avg_p_loss = np.mean(policy_result['epoch_p_losses'])
        avg_v_loss = np.mean(policy_result['epoch_v_losses'])
        avg_e_loss = np.mean(policy_result['epoch_e_losses'])
        avg_kl_loss = np.mean(policy_result['epoch_kl_losses'])
        avg_ent = np.mean(policy_result['epoch_ent'])
        avg_kl_div = np.mean(policy_result['epoch_kl_divs'])

        perf_metrics = {
            "policy_sample_time": policy_sample_time,
            "policy_prep_time": policy_prep_time,
            "policy_train_time": policy_result['policy_train_time'],
            "wm_sample_time": wm_sample_time,
            "wm_prep_time": wm_prep_time,
        }

        if reward_result['should_train']:
            perf_metrics["reward_train_time"] = reward_result['reward_train_time']

        wm_metrics = {
            "trained_reward": reward_result['should_train'],
            "trained_ctrl": False,  # Ctrl-World 始终冻结
        }

        if reward_result['should_train']:
            print(f"Trainer {self.rank}: Reward Model 训练结果: {reward_result}")
            wm_metrics["reward_loss"] = np.mean(reward_result['epoch_reward_losses'])
            wm_metrics["reward_tp"] = reward_result['reward_tp']
            wm_metrics["reward_tn"] = reward_result['reward_tn']
            wm_metrics["reward_fp"] = reward_result['reward_fp']
            wm_metrics["reward_fn"] = reward_result['reward_fn']

        return avg_loss, avg_p_loss, avg_v_loss, avg_e_loss, avg_kl_loss, current_lrs, self.global_step, avg_ent, avg_kl_div, perf_metrics, wm_metrics

    def broadcast_weights(self, group_name, prefix=None):
        """
        广播权重方法 — 只支持 Actor 和 Reward Model（Ctrl-World 冻结不需要广播）
        """
        if group_name.startswith("broadcast_actor"):
            engine = self.model
        elif group_name.startswith("broadcast_reward"):
            engine = self.reward_engine
        else:
            engine = self.model

        try:
            original_model = self.model
            self.model = engine
            return super().broadcast_weights(group_name, prefix=None)
        finally:
            self.model = original_model


# ================================================================
# 5. 主逻辑
# ================================================================
def build_openvla_cfg(args):
    """构建 OpenVLA 配置"""
    cfg = GenerateConfig(
        pretrained_checkpoint=args.pretrained_checkpoint,
        use_l1_regression=False,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=args.num_images_in_input,
        use_proprio=args.use_proprio,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        unnorm_key=args.benchmark + "_no_noops",
        checkpoint2=args.checkpoint2,
    )
    return cfg


def main(args):
    """主函数"""
    benchmark = args.benchmark
    torch_dtype = torch.bfloat16 if args.use_bf16 else torch.float32
    agent_config_path = Path.cwd() / args.agent_config_path
    trainer_config_path = Path.cwd() / args.trainer_config_path

    if not os.path.exists(args.pretrained_checkpoint):
        print(f"错误: OpenVLA checkpoint 路径 '{args.pretrained_checkpoint}' 不存在。")
        return
    if not os.path.isfile(args.ctrl_world_ckpt):
        raise FileNotFoundError(f"Ctrl-World checkpoint not found: {args.ctrl_world_ckpt}")
    if args.condition_stat_path is None:
        args.condition_stat_path = str(
            Path(args.ctrl_world_ckpt).resolve().parent / "condition_stat.json"
        )
    if not os.path.isfile(args.condition_stat_path):
        raise FileNotFoundError(
            f"Ctrl-World condition statistics not found: {args.condition_stat_path}"
        )

    os.environ["RAY_DEDUP_LOGS"] = "0"
    object_store_size_gb = 256
    object_store_memory_bytes = int(object_store_size_gb * 1024 * 1024 * 1024)
    print(f"正在初始化 Ray，并为对象存储分配 {object_store_size_gb} GB 内存...")
    ray.init(
        ignore_reinit_error=True,
        _temp_dir='/dev/shm',
        object_store_memory=object_store_memory_bytes,
    )
    print(f"Ray 初始化完成，对象存储分配 {object_store_size_gb} GB 内存。")
    log_dir = f"runs/Libero/{args.benchmark}/{int(time.time())}_{args.exp_name}"
    writer = SummaryWriter(log_dir)
    stats_actor = StatsActor.remote(window_size=args.moving_avg_window)
    print(f"TensorBoard 日志将保存在: {log_dir}")

    cfg = build_openvla_cfg(args)

    print("--- 步骤 1: 创建 Actors ---")
    # Actor Replay Buffer (imagination rollout 数据)
    replay_buffers = [ReplayBufferActor.remote(capacity=args.replay_capacity) for _ in range(args.num_trainer_gpus)]
    # World Model Replay Buffer (真实轨迹数据)
    wm_replay_buffers = [WMReplayBufferActor.remote(capacity=args.wm_replay_capacity) for _ in range(args.num_trainer_gpus)]

    trainer_group = [
        TrainerActor.remote(
            rank=i, world_size=args.num_trainer_gpus, replay_buffer=replay_buffers[i], cfg=cfg,
            train_batch_size=args.train_batch_size, accumulation_steps=args.accumulation_steps,
            use_bf16=args.use_bf16, torch_dtype=torch_dtype, policy_lr=args.policy_lr, value_lr=args.value_lr,
            gamma=args.gamma, lambda_=args.lambda_, clip_eps=args.clip_eps, vf_coef=args.vf_coef,
            ent_coef=args.ent_coef, kl_coef=args.kl_coef, reward_scale=args.reward_scale,
            value_warmup_steps=args.value_warmup_steps, policy_warmup_steps=args.policy_warmup_steps,
            policy_train_start_step=args.policy_train_start_step, train_iters=args.train_iters,
            clip_mode=args.clip_mode,
            # World Model 相关参数
            wm_replay_buffer=wm_replay_buffers[i],
            agent_config_path=agent_config_path,
            trainer_config_path=trainer_config_path,
            reward_batch_size=args.reward_batch_size,
            reward_accumulation_steps=args.reward_accumulation_steps,
            reward_lr=args.reward_lr,
            reward_warmup_steps=args.reward_warmup_steps,
            reward_train_interval=args.reward_train_interval,
            reward_checkpoint=args.reward_checkpoint,
        )
        for i in range(args.num_trainer_gpus)
    ]

    inference_pool = [
        InferenceActor.remote(
            actor_id=i, cfg=cfg, stats_actor=stats_actor, torch_dtype=torch_dtype,
            inference_batch=args.inference_batch, inference_timeout_ms=args.inference_timeout_ms
        ) for i in range(args.num_inference_actors)
    ]
    reward_inference_pool = [
        RewardInferenceActor.remote(
            actor_id=i, stats_actor=stats_actor,
            agent_config_path=agent_config_path,
            inference_batch=args.inference_batch,
            inference_timeout_ms=args.inference_timeout_ms,
            reward_checkpoint=args.reward_checkpoint,
        ) for i in range(args.num_reward_inference_actors)
    ]
    # ★ 替换 denoiser_inference_pool 为 ctrl_world_inference_pool
    ctrl_world_inference_pool = [
        CtrlWorldInferenceActor.remote(
            actor_id=i,
            stats_actor=stats_actor,
            args=args,
            inference_batch=args.inference_batch,
            inference_timeout_ms=args.inference_timeout_ms,
        ) for i in range(args.num_ctrl_inference_actors)
    ]
    rollout_workers = [
        RolloutWorkerActor.remote(
            inference_pool[i % args.num_inference_actors],
            replay_buffers[i % args.num_trainer_gpus],
            i,
            stats_actor,
            cfg,
            benchmark,
            args.num_step_cond,
            args.imagine_horizon,
            torch_dtype,
            reward_inference_pool[i % args.num_reward_inference_actors],
            ctrl_world_inference_pool[i % args.num_ctrl_inference_actors],  # ★ 替换
            args.reward_scale,
            args.gamma,
            args.lambda_,
            wm_replay_buffers[i % args.num_trainer_gpus],
            args.real_traj_collect_interval,
        ) for i in range(args.num_rollout_workers)
    ]
    eval_workers = [
        EvaluationWorkerActor.remote(
            inference_pool[i % args.num_inference_actors], f"eval_{i}", stats_actor, cfg, benchmark, torch_dtype
        ) for i in range(args.num_eval_workers)
    ]
    print(f"已创建 {args.num_rollout_workers} 个 Rollout workers 和 {args.num_eval_workers} 个 Evaluation workers。")

    print("\n--- 步骤 2: 建立独立的 DeepSpeed 训练组 ---")
    train_group_port = find_free_port()
    broadcast_group_port = find_free_port()
    while broadcast_group_port == train_group_port:
        broadcast_group_port = find_free_port()
    trainer_master_addr = ray.get(trainer_group[0].get_node_ip.remote())
    train_setup_tasks = [actor.setup_deepspeed_group.remote(trainer_master_addr, train_group_port) for actor in trainer_group]
    ray.get(train_setup_tasks)
    print("DeepSpeed 训练组建立完成。")

    print(f"\n--- 步骤 3: 建立共享广播组 ---")
    broadcast_master_addr = ray.get(trainer_group[0].get_node_ip.remote())

    # 3.1 Actor 广播组
    broadcast_actor_port = find_free_port()
    broadcast_actor_participants = [trainer_group[0]] + inference_pool
    broadcast_actor_world_size = len(broadcast_actor_participants)
    broadcast_setup_tasks = [
        actor.setup_broadcast_group.remote(
            master_addr=broadcast_master_addr, master_port=broadcast_actor_port,
            group_name="broadcast_actor", group_world_size=broadcast_actor_world_size,
            my_rank_in_group=rank) for rank, actor in enumerate(broadcast_actor_participants)
    ]
    ray.get(broadcast_setup_tasks)
    print("Actor 广播组建立完成。")

    # 3.2 Reward Model 广播组 (Trainer -> RewardInferenceActor)
    broadcast_reward_port = find_free_port()
    broadcast_reward_participants = [trainer_group[0]] + reward_inference_pool
    broadcast_reward_world_size = len(broadcast_reward_participants)
    broadcast_setup_tasks = [
        actor.setup_broadcast_group.remote(
            master_addr=broadcast_master_addr, master_port=broadcast_reward_port,
            group_name="broadcast_reward", group_world_size=broadcast_reward_world_size,
            my_rank_in_group=rank) for rank, actor in enumerate(broadcast_reward_participants)
    ]
    ray.get(broadcast_setup_tasks)
    print("Reward Model 广播组建立完成。")

    # ★ 注意：不再需要 Denoiser 广播组，Ctrl-World 冻结不训练不广播

    # 验证 Actor 模型键
    inf_keys = ray.get(inference_pool[0].get_model_keys.remote())
    trainer_keys = ray.get(trainer_group[0].get_model_keys.remote())
    for key in inf_keys:
        if key not in trainer_keys:
            print(f"警告: Actor 推理器中缺少训练器的键: {key}")
    for key in trainer_keys:
        if key not in inf_keys:
            print(f"警告: Actor 训练器中缺少推理器的键: {key}")

    forward_test_tasks = [inf.forward_test.remote() for inf in inference_pool]
    ray.get(forward_test_tasks)
    print("Actor 推理器前向测试完成 (广播前)。")

    # 广播 Actor 权重
    broadcast_task = trainer_group[0].broadcast_weights.remote("broadcast_actor")
    receive_tasks = [inf.receive_and_update_weights.remote("broadcast_actor") for inf in inference_pool]
    ray.get([broadcast_task] + receive_tasks)
    print("Actor 初始权重已广播。")

    # 广播 Reward Model 权重
    broadcast_task = trainer_group[0].broadcast_weights.remote("broadcast_reward")
    receive_tasks = [inf.receive_and_update_weights.remote("broadcast_reward") for inf in reward_inference_pool]
    ray.get([broadcast_task] + receive_tasks)
    print("Reward Model 初始权重已广播。")

    # ★ 注意：不需要广播 Ctrl-World 权重，CtrlWorldInferenceActor 独立加载并冻结

    forward_test_tasks = [inf.forward_test.remote() for inf in inference_pool]
    ray.get(forward_test_tasks)
    print("Actor 推理器前向测试完成 (广播后)。")

    print("\n--- 步骤 4: 启动 Rollout Workers 进行数据收集 ---")
    for w in rollout_workers:
        w.run.remote()
    for w in eval_workers:
        w.run.remote()

    print("\n--- 步骤 5: 等待远程经验池填充初始数据 ---")
    min_buffer_size_for_start = args.train_batch_size * args.accumulation_steps
    assert min_buffer_size_for_start < args.replay_capacity, "初始填充量必须小于回放池总容量"

    wm_min_buffer_size = args.reward_batch_size * args.reward_accumulation_steps
    assert wm_min_buffer_size < args.wm_replay_capacity, "WM初始填充量必须小于WM回放池总容量"

    while True:
        actor_sizes = ray.get([rb.size.remote() for rb in replay_buffers])
        wm_sizes = ray.get([rb.size.remote() for rb in wm_replay_buffers])
        actor_ready = all(size >= min_buffer_size_for_start for size in actor_sizes)
        wm_ready = all(size >= wm_min_buffer_size for size in wm_sizes)

        if actor_ready and wm_ready:
            break

        print(f"等待经验池填充... Actor (目标: {min_buffer_size_for_start}): {actor_sizes}, "
              f"WM (目标: {wm_min_buffer_size}): {wm_sizes}")
        time.sleep(5)
    print("所有远程经验池已准备好，训练器将按需获取数据。")

    print("\n--- 步骤 6: 开始主训练与同步循环 ---")
    start_time = time.time()
    last_log_time = time.time()
    last_log_global_step = 0
    global_step = 0
    while global_step < args.train_iters:
        t_train_start = time.time()
        train_tasks = [trainer.run_training_epoch.remote() for trainer in trainer_group]
        results = ray.get(train_tasks)
        _, _, _, _, _, _, global_step, _, _, _, wm_metrics = results[0]
        train_time = time.time() - t_train_start

        # ============ 权重同步 ============
        t_sync_start = time.time()

        # 同步 Actor 权重（每次都同步）
        broadcast_task = trainer_group[0].broadcast_weights.remote("broadcast_actor")
        receive_tasks = [inf.receive_and_update_weights.remote("broadcast_actor") for inf in inference_pool]
        ray.get([broadcast_task] + receive_tasks)

        # 同步 Reward Model 权重（只在训练时同步）
        print(f"trained_reward: {wm_metrics['trained_reward']}")
        if wm_metrics.get("trained_reward", False):
            broadcast_task = trainer_group[0].broadcast_weights.remote("broadcast_reward")
            receive_tasks = [inf.receive_and_update_weights.remote("broadcast_reward") for inf in reward_inference_pool]
            ray.get([broadcast_task] + receive_tasks)

        # ★ 注意：不需要同步 Ctrl-World 权重

        sync_time = time.time() - t_sync_start

        if global_step > 0 and global_step % args.ckpt_every_steps == 0:
            ray.get(trainer_group[0].save_agent.remote(args.ckpt_dir, global_step))

        current_time = time.time()
        if current_time - last_log_time > args.log_interval_seconds:
            all_stats = ray.get(stats_actor.get_stats.remote())

            elapsed_log_time = current_time - last_log_time
            steps_since_last_log = global_step - last_log_global_step
            training_speed_steps_per_sec = steps_since_last_log / elapsed_log_time if elapsed_log_time > 0 else 0.0

            timing_stats = all_stats.pop("_timings_", {})
            global_stats = all_stats.pop("_global_rollout_")
            eval_stats = all_stats.pop("_global_eval_")
            avg_return = global_stats["avg_return"]
            avg_ep_len = global_stats["avg_ep_len"]
            total_episodes = global_stats["total_episodes_processed"]
            total_env_steps = global_stats["total_env_steps"]
            avg_step_time = global_stats["avg_step_time"]

            eval_avg_return = eval_stats["avg_return"]
            eval_avg_ep_len = eval_stats["avg_ep_len"]
            eval_total_episodes = eval_stats["total_episodes_processed"]
            eval_env_steps = eval_stats["total_env_steps"]
            eval_avg_step_time = eval_stats["avg_step_time"]

            total_losses, p_losses, v_losses, e_losses, kl_losses, lrs_list, _, ents, avg_kl_divs, perf_metrics_list, wm_metrics_list = zip(*results)
            current_lrs = lrs_list[0]
            wm_metrics = wm_metrics_list[0]

            elapsed_time = current_time - start_time
            total_buffer_size = sum(ray.get([rb.size.remote() for rb in replay_buffers]))
            wm_buffer_size = sum(ray.get([rb.size.remote() for rb in wm_replay_buffers]))

            wm_loss_str = ""
            if 'reward_loss' in wm_metrics:
                wm_loss_str += f" | reward loss: {wm_metrics['reward_loss']:.4f}"

            print(f"更新步 {global_step}/{args.train_iters} | 时间: {elapsed_time:.1f}s | "
                  f"全局平均奖励: {avg_return:.2f} | 全局平均幕长: {avg_ep_len:.1f} | Eval奖励: {eval_avg_return:.2f} | "
                  f"value loss: {np.mean(v_losses):.4f}{wm_loss_str} | "
                  f"LR(V/P): {current_lrs['value']:.7f}/{current_lrs['policy']:.7f} | "
                  f"Episodes数量: {total_episodes:,} | Step平均时间: {avg_step_time:.3f}s")

            writer.add_scalar('Train/Learning_Rate/Value', current_lrs['value'], global_step)
            writer.add_scalar('Train/Learning_Rate/Policy', current_lrs['policy'], global_step)
            if 'reward' in current_lrs:
                writer.add_scalar('Train/Learning_Rate/Reward', current_lrs['reward'], global_step)

            writer.add_scalar('Loss/Total', np.mean(total_losses), global_step)
            writer.add_scalar('Loss/Policy', np.mean(p_losses), global_step)
            writer.add_scalar('Loss/Value', np.mean(v_losses), global_step)
            writer.add_scalar('Loss/Entropy', np.mean(e_losses), global_step)
            writer.add_scalar('Loss/KL', np.mean(kl_losses), global_step)

            if 'reward_loss' in wm_metrics:
                writer.add_scalar('Loss/Reward_Model', wm_metrics['reward_loss'], global_step)
            if 'reward_tp' in wm_metrics:
                writer.add_scalar('WorldModel/Reward_TP', wm_metrics['reward_tp'], global_step)
                writer.add_scalar('WorldModel/Reward_TN', wm_metrics['reward_tn'], global_step)
                writer.add_scalar('WorldModel/Reward_FP', wm_metrics['reward_fp'], global_step)
                writer.add_scalar('WorldModel/Reward_FN', wm_metrics['reward_fn'], global_step)
                pos_total = wm_metrics['reward_tp'] + wm_metrics['reward_fn']
                neg_total = wm_metrics['reward_tn'] + wm_metrics['reward_fp']
                reward_pos_acc = wm_metrics['reward_tp'] / pos_total if pos_total > 0 else 0.0
                reward_neg_acc = wm_metrics['reward_tn'] / neg_total if neg_total > 0 else 0.0
                writer.add_scalar('WorldModel/Reward_Pos_Acc', reward_pos_acc, global_step)
                writer.add_scalar('WorldModel/Reward_Neg_Acc', reward_neg_acc, global_step)

            writer.add_scalar('Metrics/Entropy', np.mean(ents), global_step)
            writer.add_scalar('Metrics/KL_Divergence', np.mean(avg_kl_divs), global_step)
            writer.add_scalar('Metrics/Training_Speed_Steps_per_Sec', training_speed_steps_per_sec, global_step)
            for metric_name, metric_value in timing_stats.items():
                writer.add_scalar(f'Performance/{metric_name}', metric_value, global_step)
            avg_policy_sample_time = np.mean([pm["policy_sample_time"] for pm in perf_metrics_list])
            avg_policy_prep_time = np.mean([pm["policy_prep_time"] for pm in perf_metrics_list])
            avg_policy_train_time = np.mean([pm["policy_train_time"] for pm in perf_metrics_list])
            avg_wm_sample_time = np.mean([pm.get("wm_sample_time", 0.0) for pm in perf_metrics_list])
            avg_wm_prep_time = np.mean([pm.get("wm_prep_time", 0.0) for pm in perf_metrics_list])
            writer.add_scalar('Performance/policy_sample_time', avg_policy_sample_time, global_step)
            writer.add_scalar('Performance/policy_prep_time', avg_policy_prep_time, global_step)
            writer.add_scalar('Performance/policy_train_time', avg_policy_train_time, global_step)
            writer.add_scalar('Performance/wm_sample_time', avg_wm_sample_time, global_step)
            writer.add_scalar('Performance/wm_prep_time', avg_wm_prep_time, global_step)

            reward_times = [pm.get("reward_train_time") for pm in perf_metrics_list if "reward_train_time" in pm]
            if reward_times:
                avg_reward_train_time = np.mean(reward_times)
                writer.add_scalar('Performance/reward_train_time', avg_reward_train_time, global_step)
            writer.add_scalar('Performance/train_time', train_time, global_step)
            writer.add_scalar('Performance/sync_time', sync_time, global_step)
            writer.add_scalar('Performance/train_time_total', time.time() - t_train_start, global_step)

            writer.add_scalar('Rollout/_Global/Average_Return', avg_return, global_step)
            writer.add_scalar('Rollout/_Global/Average_Episode_Length', avg_ep_len, global_step)
            writer.add_scalar('Rollout/_Global/Average_Imagine_Reward', global_stats.get("avg_imagine_reward", 0.0), global_step)
            writer.add_scalar('Eval/_Global/Average_Return', eval_avg_return, global_step)
            writer.add_scalar('Eval/_Global/Average_Episode_Length', eval_avg_ep_len, global_step)

            writer.add_scalar('System/Replay_Buffer_Size_Total', total_buffer_size, global_step)
            writer.add_scalar('System/WM_Replay_Buffer_Size_Total', wm_buffer_size, global_step)
            writer.add_scalar('System/Total_Episodes_Processed', total_episodes, global_step)
            writer.add_scalar('System/Total_Env_Steps', total_env_steps, global_step)
            writer.add_scalar('System/Avg_Step_Time', avg_step_time, global_step)
            writer.add_scalar('System/Eval_Total_Episodes_Processed', eval_total_episodes, global_step)
            writer.add_scalar('System/Eval_Total_Env_Steps', eval_env_steps, global_step)
            writer.add_scalar('System/Eval_Avg_Step_Time', eval_avg_step_time, global_step)
            writer.add_scalar('System/Active_Rollout_Actors', global_stats.get("active_actor_count", 0), global_step)
            writer.add_scalar('System/Total_Samples_Produced', global_stats.get("total_samples_produced", 0), global_step)
            writer.add_scalar('System/Total_Imagine_Samples', global_stats.get("total_imagine_samples", 0), global_step)

            for env_name, env_stats in all_stats.items():
                if env_name.startswith("eval_"):
                    tag_prefix = f"Eval/{env_name.replace('eval_', '')}"
                    writer.add_scalar(f'{tag_prefix}/Average_Return', env_stats['avg_return'], global_step)
                    writer.add_scalar(f'{tag_prefix}/Average_Episode_Length', env_stats['avg_ep_len'], global_step)
                    writer.add_scalar(f'{tag_prefix}/Success_Rate', env_stats['avg_success_rate'], global_step)
                    writer.add_scalar(f'{tag_prefix}/Total_Episodes', env_stats['total_episodes'], global_step)
                else:
                    tag_prefix = f"Rollout/{env_name}"
                    writer.add_scalar(f'{tag_prefix}/Average_Return', env_stats['avg_return'], global_step)
                    writer.add_scalar(f'{tag_prefix}/Average_Episode_Length', env_stats['avg_ep_len'], global_step)
                    writer.add_scalar(f'{tag_prefix}/Success_Rate', env_stats['avg_success_rate'], global_step)
                    writer.add_scalar(f'{tag_prefix}/Total_Episodes', env_stats['total_episodes'], global_step)

            last_log_time = current_time
            last_log_global_step = global_step

    print(f"\n成功完成 {args.train_iters} 次训练与同步循环！")
    writer.close()
    ray.shutdown()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 80)
    print("命令行参数:")
    print("-" * 80)
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("=" * 80)
    main(args)
