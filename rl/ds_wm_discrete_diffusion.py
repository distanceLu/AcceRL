import os
import argparse
os.environ["MUJOCO_GL"] = "osmesa"           # 强制软件渲染
os.environ["PYOPENGL_PLATFORM"] = "osmesa"   # 保险起见，给 PyOpenGL 也指明
# 设置临时文件目录，避免磁盘I/O瓶颈
os.environ["TMPDIR"] = "/dev/shm"
# 为了让 Ray 能看到所有可用的 GPU，我们在脚本开头设置。
# 注意: CUDA_VISIBLE_DEVICES 现在通过命令行参数设置
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
# 防止 transformers 库的 tokenizer 并行化警告
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import random
import asyncio
from collections import deque, defaultdict
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass
import math
from hydra.utils import instantiate
from omegaconf import OmegaConf
import numpy as np
from envs.world_model_env_batch import WorldModelEnvConfig
import ray
import torch
import torch.distributions
from torch.distributions import kl
import deepspeed
import torch.distributed as distributed 
from torch.utils.tensorboard import SummaryWriter

# OpenVLA 组件与常量
# zzq1120 单独从openvla_utils取出这两个方法
from experiments.robot.sole_utils import (
    get_processor,
)
from envs.diffusion.denoiser import load_denoiser_from_checkpoint
from envs.diffusion import DiffusionSampler

from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM
from experiments.robot.libero.libero_utils import GenerateConfig, TaskSuite
from rl.actor_critic_model_discrete import ActorCritic
from rl.utils import prepare_one_obs
# 训练/推理通信（保持接口不变）
from ds_com import TrainerActorCom, InferenceActorCom
from rl.com_utils import find_free_port
from envs.utils import tensor_to_image, image_to_tensor, load_reward_model
from pathlib import Path

# ================================================================
# 0. 超参数与配置 - 命令行参数解析
# ================================================================
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='OpenVLA RL Training with World Model')
    
    # 环境变量
    parser.add_argument('--cuda-visible-devices', type=str, default='1,2',
                        help='CUDA visible devices (default: 1,2)')
    
    # Libero benchmark
    parser.add_argument('--benchmark', type=str, default='libero_spatial',
                        choices=['libero_spatial', 'libero_object', 'libero_goal', 'libero_10', 'libero_90'],
                        help='Libero benchmark suite (default: libero_spatial)')
    
    # 分布式系统参数
    parser.add_argument('--num-trainer-gpus', type=int, default=1,
                        help='Number of trainer GPUs (default: 1)')
    parser.add_argument('--num-inference-actors', type=int, default=1,
                        help='Number of inference actors (default: 1)')
    parser.add_argument('--num-rollout-workers', type=int, default=20,
                        help='Number of rollout workers (default: 20)')
    parser.add_argument('--num-eval-workers', type=int, default=10,
                        help='Number of evaluation workers (default: 10)')
    parser.add_argument('--rollout-local-buf', type=int, default=64,
                        help='Rollout local buffer size (default: 64)')
    parser.add_argument('--inference-batch', type=int, default=8,
                        help='Inference batch size (default: 8)')
    parser.add_argument('--inference-timeout-ms', type=int, default=300,
                        help='Inference timeout in milliseconds (default: 300)')
    parser.add_argument('--replay-capacity', type=int, default=10000,
                        help='Replay buffer capacity (default: 10000)')
    parser.add_argument('--train-batch-size', type=int, default=32,
                        help='Training batch size (default: 32)')
    parser.add_argument('--accumulation-steps', type=int, default=8,
                        help='Gradient accumulation steps (default: 8)')
    parser.add_argument('--train-iters', type=int, default=30000,
                        help='Total training iterations (default: 30000)')
    
    # Checkpoint
    parser.add_argument('--ckpt-dir', type=str, default='/cpfs01/liuwei_workspace/models/finetune_rl',
                        help='Checkpoint directory (default: /cpfs01/liuwei_workspace/models/finetune_rl)')
    parser.add_argument('--ckpt-every-steps', type=int, default=2000000,
                        help='Save checkpoint every N steps (default: 2000000)')
    
    # PPO 参数
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='PPO discount factor (default: 0.99)')
    parser.add_argument('--lambda', type=float, default=0.95, dest='lambda_',
                        help='PPO GAE lambda (default: 0.95)')
    parser.add_argument('--clip-eps', type=float, default=0.2,
                        help='PPO clipping epsilon (default: 0.2)')
    parser.add_argument('--vf-coef', type=float, default=0.5,
                        help='Value function coefficient (default: 0.5)')
    parser.add_argument('--ent-coef', type=float, default=0.00,
                        help='Entropy coefficient (default: 0.00)')
    parser.add_argument('--kl-coef', type=float, default=0.1,
                        help='KL divergence coefficient (default: 0.1)')
    
    # 奖励缩放
    parser.add_argument('--reward-scale', type=float, default=1.0,
                        help='Reward scaling factor (default: 1.0)')
    
    # 学习率调度参数
    parser.add_argument('--value-lr', type=float, default=1e-4,
                        help='Value network learning rate (default: 1e-4)')
    parser.add_argument('--policy-lr', type=float, default=1e-5,
                        help='Policy network learning rate (default: 1e-5)')
    parser.add_argument('--value-warmup-steps', type=int, default=500,
                        help='Value network warmup steps (default: 500)')
    parser.add_argument('--policy-warmup-steps', type=int, default=500,
                        help='Policy network warmup steps (default: 500)')
    parser.add_argument('--policy-train-start-step', type=int, default=0,
                        help='Start training policy network at step N (default: 0)')
    
    # 世界模型配置
    parser.add_argument('--imagine-horizon', type=int, default=8,
                        help='World model imagination horizon (default: 8)')
    parser.add_argument('--num-step-cond', type=int, default=4,
                        help='Number of conditional observation steps (default: 4)')
    parser.add_argument('--num-reward-inference-actors', type=int, default=1,
                        help='Number of reward inference actors (default: 1)')
    parser.add_argument('--num-denoiser-inference-actors', type=int, default=1,
                        help='Number of denoiser inference actors (default: 1)')
    parser.add_argument('--agent-config-path', type=str, default='envs/config/agent.yaml',
                        help='Agent config path (default: envs/config/agent.yaml)')
    parser.add_argument('--trainer-config-path', type=str, default='envs/config/trainer.yaml',
                        help='Trainer config path (default: envs/config/trainer.yaml)')
    
    # 日志
    parser.add_argument('--moving-avg-window', type=int, default=1000,
                        help='Moving average window size (default: 1000)')
    parser.add_argument('--log-interval-seconds', type=int, default=10,
                        help='Log interval in seconds (default: 10)')
    
    # 通信组
    parser.add_argument('--broadcast-group-name', type=str, default='trainer_to_inference_broadcast',
                        help='Broadcast group name (default: trainer_to_inference_broadcast)')
    
    # OpenVLA 加载配置
    parser.add_argument('--use-bf16', action='store_true', default=True,
                        help='Use bfloat16 (default: True)')
    parser.add_argument('--no-bf16', action='store_false', dest='use_bf16',
                        help='Disable bfloat16')
    parser.add_argument('--pretrained-checkpoint', type=str,
                        default='/cpfs01/liuwei_workspace/models/finetune_im/openvla-7b+libero_spatial_no_noops+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--discrete_acts--proprio_state--100000_chkpt',
                        help='Pretrained checkpoint path')
    parser.add_argument('--checkpoint2', type=str,
                        default='runs/distill/20251225_113851_distill/checkpoints/checkpoint_latest.pt',
                        help='Second checkpoint path')
    
    parser.add_argument('--clip-mode', type=str, default='sapo',
                        choices=['ppo', 'sapo', 'clippo'],
                        help='Clipping mode for PPO (default: sapo)')
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Experiment name (default: auto-generated based on clip-mode)')
    
    args = parser.parse_args()
    
    # 设置 CUDA_VISIBLE_DEVICES 环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    
    # 如果没有提供 exp_name，自动生成
    if args.exp_name is None:
        args.exp_name = f"OpenVLA_DS_{args.clip_mode}_DISCRETE_task0_wm"
    
    return args

# ================================================================
# 数据结构 更新经验数据结
# ================================================================
@dataclass
class Experience:
    obs: Dict[str, torch.Tensor]            # prepare_one_obs 的结果（CPU tensors）
    action_token: np.ndarray                # 采样的离散动作 token (shape: [ACTION_DIM,])
    advantage: float
    behaviour_logits: np.ndarray            # 行为策略的 logits (shape: [ACTION_DIM, VOCAB_SIZE])
    value_target: float

# ================================================================
# 1.5. 统计模块 (StatsActor)
# ================================================================
@ray.remote
class StatsActor:
    def __init__(self, window_size):
        self.stats = defaultdict(lambda: {
            "episode_returns": deque(maxlen=window_size),
            "step_times": deque(maxlen=window_size),
            "episode_lengths": deque(maxlen=window_size),
            "successes": deque(maxlen=window_size),
            "total_episodes_processed": 0,
            "total_env_steps": 0
        })
        self.timings = defaultdict(lambda: deque(maxlen=window_size))
        self.imagine_rewards = deque(maxlen=window_size)  # 用于记录 imagination rollout 的 imagine_reward
        self.actor_last_active = {}
        self.active_window_seconds = 600
        self.total_samples_produced = 0

    def add_episode_return(
        self,
        env_name: str,
        ep_return: float,
        step_time: float,
        ep_length: int,
        success: float,
        actor_id: Optional[int] = None,
        step_num: int = 0,
    ):
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

    def add_timing_metric(self, metric_name: str, value: float):
        """记录系统性能相关的计时指标"""
        self.timings[metric_name].append(value)

    def add_imagine_reward(self, avg_imagine_reward: float, actor_id: int):
        """记录 imagination rollout 中的平均 imagine_reward"""
        self.imagine_rewards.append(avg_imagine_reward)
        self.actor_last_active[actor_id] = time.time()

    def get_active_actor_count(self) -> int:
        current_time = time.time()
        cutoff = current_time - self.active_window_seconds
        return sum(1 for last_active in self.actor_last_active.values() if last_active >= cutoff)

    def get_stats(self) -> Dict[str, Dict[str, float]]:
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
                    "avg_return": 0.0, 
                    "avg_ep_len": 0.0, 
                    "avg_success_rate": 0.0, 
                    "num_episodes_in_avg": 0, 
                    "total_episodes": env_data["total_episodes_processed"]}
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
# 2. 经验回放与 Rollout
# ================================================================
@ray.remote
class ReplayBufferActor:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add_batch(self, batch: List[Experience]):
        self.buffer.extend(batch)

    def size(self):
        return len(self.buffer)
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # obs 是 prepare_one_obs 的字典，不能 stack，保持 list 返回
        obs_list = [b.obs for b in batch]
        action_token = np.stack([b.action_token for b in batch])
        adv = np.asarray([b.advantage for b in batch], np.float32)
        logits_old = np.stack([b.behaviour_logits for b in batch])
        v_targ = np.asarray([b.value_target for b in batch], np.float32)
        return obs_list, action_token, adv, logits_old, v_targ

class BaseWorkerActor:
    """rollout 和 eval worker 的共享逻辑。"""
    def __init__(self, infer, replay, wid, stats_actor, cfg, benchmark_name):
        self.infer = infer
        self.replay = replay
        self.stats_actor = stats_actor
        self.cfg = cfg
        # 仅需 processor，Worker 不加载大模型
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

@ray.remote
class RolloutWorkerActor(BaseWorkerActor):
    def __init__(self, infer, replay, wid, stats_actor, cfg, benchmark_name, num_step_cond, imagine_horizon, torch_dtype, reward_infer, denoiser_infer, reward_scale, gamma, lambda_):
        super().__init__(infer, replay, wid, stats_actor, cfg, benchmark_name)
        self.env_outcome = [deque(maxlen=100) for _ in range(self.num_tasks)]
        self.local_buffer = []
        self.episodes = deque(maxlen=100)
        self.num_step_cond = num_step_cond
        self.imagine_horizon = imagine_horizon
        if self.imagine_horizon % NUM_ACTIONS_CHUNK != 0:
            Warning(f"imagine_horizon {self.imagine_horizon} is not divisible by NUM_ACTIONS_CHUNK {NUM_ACTIONS_CHUNK}，这会导致不足NUM_ACTIONS_CHUNK的轨迹被丢弃！")
        self.torch_dtype = torch_dtype
        self.reward_infer = reward_infer
        self.denoiser_infer = denoiser_infer
        self.reward_scale = reward_scale
        self.gamma = gamma
        self.lambda_ = lambda_

    def _reset_and_select_env(self, seed: Optional[int] = None) -> Tuple[Dict, Dict]:
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
                if imagine_step % 10 == 0:
                    self.get_one_episode()
                imagine_step += 1
                experience = random.choice(self.episodes)
                obs_list, reward_list, done_list, act_norm_list, task_description = experience
                obs_list2 = [obs['full_image'] for obs in obs_list]
                for i in range(len(obs_list2) - self.num_step_cond):
                    obs_list_sub = obs_list2[i:i+self.num_step_cond]
                    for idx, sub_obs in enumerate(obs_list_sub):
                        sub_obs = image_to_tensor(sub_obs, 'cpu')
                        obs_list_sub[idx] = sub_obs
                    act_list_sub = act_norm_list[i:i+self.num_step_cond-1]
                    obs_tensor = torch.stack(obs_list_sub, dim=0) # [num_step_cond, C, H, W]
                    # Bug fix: 将 numpy 数组列表转换为 tensor 列表后再 stack
                    if isinstance(act_list_sub[0], np.ndarray):
                        act_list_sub = [torch.from_numpy(a.copy()) for a in act_list_sub]
                    act_tensor = torch.stack(act_list_sub, dim=0) # [num_step_cond-1, act_dim]
                    last_succ_prob = self.predict_rew_end(obs_tensor[-1], task_description)[0]
                    # end = False  # Bug fix: 初始化 end 变量，避免未定义错误
                    for j in range(self.imagine_horizon):
                        inputs_t = self.obs2inp(obs_tensor[-1], task_description)
                        act_norm, action_env, action_token, logits, value = ray.get(self.infer.request.remote(inputs_t, deterministic=False))
                        if isinstance(act_norm, np.ndarray):
                            act_norm = torch.from_numpy(act_norm.copy())
                        act_norm = act_norm.float().to(act_tensor.device)
                        chunk_reward = 0.0
                        for k in range(len(action_env)):
                            # TODO denoiser暂时支持action norm作为输入动作
                            # Bug fix: 使用正确的循环变量 k 而不是 i
                            single_action = act_norm[k]
                            act_tensor = torch.cat([act_tensor, single_action.unsqueeze(0)], dim=0)
                            nxt = self.predict_next_obs(obs_tensor, act_tensor)
                            obs_tensor = torch.roll(obs_tensor, -1, dims=0)
                            act_tensor = act_tensor[1:]
                            obs_tensor[-1] = nxt
                            succ_prob, end = self.predict_rew_end(nxt, task_description)
                            rew = succ_prob - last_succ_prob
                            last_succ_prob = succ_prob
                            chunk_reward += rew * self.reward_scale
                            if end: 
                                break  # TODO 没有考虑truncated
                        self.local_buffer.append((inputs_t, action_token, chunk_reward, logits, value))
                        if end: break
                    if self.local_buffer: 
                        if end:
                            self._process_traj(self.local_buffer, 0.0)
                        else:
                            inputs_t = self.obs2inp(obs_tensor[-1], task_description)
                            _, _, _, _, bootstrap_val = ray.get(self.infer.request.remote(inputs_t, deterministic=False))
                            self._process_traj(self.local_buffer, bootstrap_val)
                        # 记录 imagine_reward 平均值
                        imagine_rewards = [exp[2] for exp in self.local_buffer]
                        avg_imagine_reward = sum(imagine_rewards) / len(imagine_rewards)
                        self.stats_actor.add_imagine_reward.remote(avg_imagine_reward, self.wid)
                    self.local_buffer.clear()
        except Exception as e:
            import traceback
            print(f"[ERROR] RolloutWorker {self.wid} run() 崩溃: {e}", flush=True)
            traceback.print_exc(); raise

    def get_one_episode(self):
        obs_list, reward_list, done_list, action_norm_list = [], [], [], []
        current_seed = int(time.time() * 1000) + self.wid + os.getpid()
        obs, info = self._reset_and_select_env(seed=current_seed)
        obs_list.append(obs)
        reward_sum, time_start, step_count_total = 0.0, time.time(), 0
        while True:
            inputs_t = prepare_one_obs(self.cfg, self.processor, obs, self.task_description, self.torch_dtype)
            act_norm, action_env, action_token, logits, value = ray.get(self.infer.request.remote(inputs_t, deterministic=False))
            chunk_reward, done = 0.0, False
            for i in range(len(action_env)):
                single_action = action_env[i]
                nxt, r, term, trunc, info = self.env.step(single_action)
                obs_list.append(nxt)
                reward_list.append(r)
                done_list.append(term or trunc)
                action_norm_list.append(act_norm[i])
                reward_sum += r
                chunk_reward += r * self.reward_scale
                step_count_total += 1
                if term or trunc: done = True; break
            # self.local_buffer.append((inputs_t, action_token, chunk_reward, logits, value))
            obs = nxt

            if done:
                step_time = (time.time() - time_start) / max(step_count_total, 1)
                success = float(info.get('is_success', 0.0))
                self.env_outcome[self.current_env_idx].append(1.0 - success)
                self.stats_actor.add_episode_return.remote(
                    self.current_env_name,
                    reward_sum,
                    step_time,
                    step_count_total,
                    success,
                    actor_id=self.wid,
                    step_num=step_count_total,
                )
                self.episodes.append((obs_list, reward_list, done_list, action_norm_list, self.task_description))
                break

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

        batch: List[Experience] = []
        for i, (s, a_token, _, logits, _) in enumerate(traj_segment):
            batch.append(
                Experience(
                    obs=s,
                    action_token=a_token.astype(np.int64), # token 是整数
                    advantage=float(advs_np[i]),
                    behaviour_logits=logits.astype(np.float32),
                    value_target=float(rets[i]),
                )
            )
        self.replay.add_batch.remote(batch)

    def predict_next_obs(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return ray.get(self.denoiser_infer.request.remote(obs.float(), act.float()))
    
    def predict_rew_end(self, next_obs: torch.Tensor, task_description: str) -> Tuple[float, int]:
        """
        Predict reward and end signal using reward_model.
        
        Args:
            next_obs: [C, H, W] - the predicted next observation in [-1, 1]
        
        Returns:
            (rew, end) where:
                rew: scalar tensor - reward as probability of class 1 (range [0, 1])
                end: scalar tensor - end signal (0 or 1, binary classification)
        """
        inputs = self.obs2inp(next_obs, task_description)
        # Forward through reward model
        logits = ray.get(self.reward_infer.request.remote(inputs))
        probs = torch.softmax(logits, dim=-1)  # [2]
        succ_prob = probs[1].item()  # scalar - probability of class 1
        end = logits.argmax().item()  # scalar - 0 or 1
        return succ_prob, end

    def obs2inp(self, obs: torch.Tensor, task_description: str) -> Dict[str, torch.Tensor]:
        # Convert tensor to uint8 HWC image
        frame_uint8 = tensor_to_image(obs)  # [H, W, C] uint8
        # Prepare input for reward model
        obs_for_vla: Dict[str, Any] = {"full_image": frame_uint8}
        inputs = prepare_one_obs(
            self.cfg, 
            self.processor, 
            obs_for_vla, 
            task_description, 
            self.torch_dtype
        )
        # if (not self.reward_cfg.use_proprio) and ("proprio" in inputs) and (inputs["proprio"] is None):
        #     inputs.pop("proprio", None)
        return inputs


@ray.remote
class EvaluationWorkerActor(BaseWorkerActor):
    def __init__(self, infer, wid, stats_actor, cfg, benchmark_name, torch_dtype):
        super().__init__(infer, None, wid, stats_actor, cfg, benchmark_name)
        self.torch_dtype = torch_dtype
        print(f"EvaluationWorker {self.wid}: 环境初始化完成。")

    def _reset_and_select_env(self, seed: Optional[int] = None) -> Tuple[Dict, Dict]:
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
                    f"eval_{self.current_env_name}",
                    reward_sum,
                    step_time,
                    step_count_total,
                    success,
                    actor_id=None,
                    step_num=step_count_total,
                )
                current_seed = int(time.time() * 1000) + os.getpid() + random.randint(0, 10000)
                obs, info = self._reset_and_select_env(seed=current_seed)
        except Exception as e: import traceback; print(f"[ERROR] EvaluationWorker {self.wid} run() 崩溃: {e}", flush=True); traceback.print_exc(); raise


# ================================================================
# 3. 推理器 (InferenceActor)
# ================================================================
@ray.remote(num_gpus=0.3)
class InferenceActor(InferenceActorCom):
    def __init__(self, actor_id, cfg, stats_actor, torch_dtype, inference_batch, inference_timeout_ms):
        super().__init__()
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
            print("模型尚未初始化。")
            return {}
        sd = self.model.state_dict()
        res = {k: float(v.abs().sum().item()) for k, v in sd.items()}
        return res

    def _on_bg_task_done(self, task: asyncio.Task):
        try:
            task.result()
        except Exception as e:
            import traceback
            print(f"[ERROR] InferenceActor {self.actor_id} 后台任务异常: {e}", flush=True)
            traceback.print_exc()

    async def request(self, inputs_t: Dict[str, torch.Tensor], deterministic: bool = False):
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
                    # 1. 前向传播获取 logits 和 value
                    action_logits, value = self.model(inputs_batch)

                    # 2. 后处理以采样动作 tokens 和对应的归一化连续动作
                    _, action_tokens_all, normalized_actions_all = self.model.post_process(action_logits, deterministic=deterministic_flags)
                    
                    # action_tokens_all 的形状是 (B, NUM_ACTIONS_CHUNK * ACTION_DIM)
                    action_tokens = action_tokens_all.view(
                        -1, NUM_ACTIONS_CHUNK, ACTION_DIM
                    ).cpu().numpy()

                    # action_logits 的形状是 (B, NUM_ACTIONS_CHUNK * ACTION_DIM, VocabSize)
                    logits = action_logits.view(
                        -1, NUM_ACTIONS_CHUNK, ACTION_DIM, action_logits.shape[-1]
                    ).float().cpu().numpy()
                    
                    values = value.to(torch.float32).cpu().numpy()

                # 将标准化动作转换为环境动作
                actions_env = []
                for i in range(normalized_actions_all.shape[0]):
                    a_env = self.model.vla._unnormalize_actions(normalized_actions_all[i], self.cfg.unnorm_key)
                    actions_env.append(a_env.astype(np.float32))

                for i in range(len(promises_to_process)):
                    promises_to_process[i].set_result((
                        normalized_actions_all[i], # 归一化的环境动作
                        actions_env[i],           # 反归一化的环境动作
                        action_tokens[i],         # 离散动作 token
                        logits[i], # 对应的 logits
                        values[i]                 # 价值估计
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
        return  # TODO 测试用，后续删除 
        import pickle
        with open("experiments/robot/libero/sample_libero_spatial_observation.pkl", "rb") as file:
            observation = pickle.load(file)
        inputs_t = prepare_one_obs(self.cfg, self.processor, observation, observation['task_description'], TORCH_DTYPE)
        inputs_batch = self.model.prepare_inputs_batch([inputs_t])
        with torch.no_grad():
            action_logits, value = self.model(inputs_batch)
        return action_logits, value
    

@ray.remote(num_gpus=0.3)
class RewardInferenceActor:
    def __init__(self, actor_id, stats_actor, agent_config_path, inference_batch, inference_timeout_ms):
        super().__init__()
        self.actor_id = actor_id
        self.stats_actor = stats_actor
        agent_cfg = OmegaConf.load(agent_config_path)
        self.reward_model, self.rew_cfg = load_reward_model(
            model_path=agent_cfg.reward_model_path,
            device="cuda",
            pretrained_checkpoint=agent_cfg.openvla_path,
            focal_alpha=agent_cfg.reward_model.focal_alpha,
        )
        self.batch_size = inference_batch
        self.timeout_sec = inference_timeout_ms / 1000.0
        self.requests, self.promises = [], []
        self.last_process_time = time.time()

        loop = asyncio.get_event_loop()
        self._bg_task = loop.create_task(self._loop())
        self._bg_task.add_done_callback(self._on_bg_task_done)
        print(f"RewardInferenceActor {self.actor_id} 初始化于 GPU: {ray.get_gpu_ids()} (批次超时: {inference_timeout_ms}ms)")

    def _on_bg_task_done(self, task: asyncio.Task):
        try:
            task.result()
        except Exception as e:
            import traceback
            print(f"[ERROR] InferenceActor {self.actor_id} 后台任务异常: {e}", flush=True)
            traceback.print_exc()

    async def request(self, inputs_t: Dict[str, torch.Tensor]):
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


@ray.remote(num_gpus=0.3)
class DenoiserInferenceActor:
    def __init__(self, actor_id, stats_actor, agent_config_path, trainer_config_path, inference_batch, inference_timeout_ms):
        super().__init__()
        self.actor_id = actor_id
        self.stats_actor = stats_actor
            # 加载 Denoiser
        denoiser, trainer_cfg, agent_cfg = load_denoiser_from_checkpoint(
            agent_config_path, trainer_config_path, torch.device("cuda")
        )
        sampler_cfg = instantiate(trainer_cfg.world_model_env.diffusion_sampler)
        env_cfg = WorldModelEnvConfig(
            horizon=trainer_cfg.world_model_env.horizon,
            num_batches_to_preload=trainer_cfg.world_model_env.num_batches_to_preload,
            diffusion_sampler=sampler_cfg,
        )
        self.sampler = DiffusionSampler(denoiser, env_cfg.diffusion_sampler)
        self.device = self.sampler.sigmas.device
        self.batch_size = inference_batch
        self.timeout_sec = inference_timeout_ms / 1000.0
        self.requests, self.promises = [], []
        self.last_process_time = time.time()

        loop = asyncio.get_event_loop()
        self._bg_task = loop.create_task(self._loop())
        self._bg_task.add_done_callback(self._on_bg_task_done)
        print(f"DenoiserInferenceActor {self.actor_id} 初始化于 GPU: {ray.get_gpu_ids()} (批次超时: {inference_timeout_ms}ms)")

    def _on_bg_task_done(self, task: asyncio.Task):
        try:
            task.result()
        except Exception as e:
            import traceback
            print(f"[ERROR] InferenceActor {self.actor_id} 后台任务异常: {e}", flush=True)
            traceback.print_exc()

    async def request(self, obs: torch.Tensor, act: torch.Tensor):
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        self.requests.append((obs, act))
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
            
            obs_list = []
            act_list = []
            for req in requests_to_process:
                obs_list.append(req[0])
                act_list.append(req[1])
            obs_batch = torch.stack(obs_list, dim=0).to(self.device)
            act_batch = torch.stack(act_list, dim=0).to(self.device)
            t_loop_start = time.time()
            try:
                with torch.inference_mode():
                    next_obs, _ = self.sampler.sample(obs_batch, act_batch)
                    next_obs = next_obs.cpu()
                for i in range(len(promises_to_process)):
                    promises_to_process[i].set_result(next_obs[i])
                loop_duration = time.time() - t_loop_start
                self.stats_actor.add_timing_metric.remote("Inference/denoiser_loop_time_s", loop_duration)
            except Exception as e:
                import traceback
                print(f"[ERROR] DenoiserInferenceActor {self.actor_id} 批处理失败: {e}", flush=True)
                traceback.print_exc()
                for p in promises_to_process:
                    if not p.done():
                        p.set_exception(e)
                raise
    

# ================================================================
# 4. 训练器 (TrainerActor)
# ================================================================
@ray.remote(num_gpus=1)
class TrainerActor(TrainerActorCom):
    def __init__(self, rank, world_size, replay_buffer, cfg, train_batch_size, accumulation_steps, 
                 use_bf16, torch_dtype, policy_lr, value_lr, gamma, lambda_, clip_eps, vf_coef, 
                 ent_coef, kl_coef, reward_scale, value_warmup_steps, policy_warmup_steps, 
                 policy_train_start_step, train_iters, clip_mode):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.replay_buffer = replay_buffer
        self.cfg = cfg
        self.model = None
        self.optimizer = None
        self.base_model = None
        self.data_dtype = None
        self.next_ready_batch: Optional[Tuple] = None
        self.data_fetching_task = None
        
        # 存储训练参数
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

        print(f"Trainer {self.rank}: 正在加载 OpenVLA ActorCritic...")
        model = ActorCritic(self.cfg, torch_dtype=self.torch_dtype)
        self.base_model = model

        # 参数分组（与之前代码一致）
        param_groups = self.base_model.get_parameter_groups()
        optimizer_params = [
            {"params": pg["params"], "name": pg["name"], "lr": self.policy_lr if pg["name"] == "policy" else self.value_lr}
            for pg in param_groups
        ]
        
        ds_config = {
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

        if ds_config.get("bf16", {}).get("enabled", False): self.data_dtype = torch.bfloat16
        else: self.data_dtype = torch.float32

        self.model, self.optimizer, _, _ = deepspeed.initialize(model=model, config=ds_config, model_parameters=optimizer_params)
        print(f"TrainerActor Rank {self.rank}: DeepSpeed 训练组 (ZeRO-2) 初始化完成。")

        self.data_fetching_task = asyncio.get_event_loop().create_task(self._data_fetching_loop())

        n_total = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数量: {n_total:,}, 可训练参数量: {n_trainable:,}")

    async def save_agent(self, ckpt_dir: str, step: int):
        """
        只在 rank-0 上调用。调用 ActorCritic 内部的 save_model
        """
        os.makedirs(ckpt_dir, exist_ok=True)
        self.base_model.save_model(ckpt_dir, epoch=step)
        print(f"[Trainer {self.rank}] 已保存 checkpoint -> {ckpt_dir}/agent_lora_epoch_{step}, agent_extra_layers_epoch_{step}.pt")

    def _get_current_lr(self, current_step: int, peak_lr: float, warmup_steps: int, total_steps: int, start_step: int = 0) -> float:
        if current_step < start_step: return 0.0
        effective_step = current_step - start_step
        if effective_step < warmup_steps: return peak_lr * (effective_step / warmup_steps)
        progress = (effective_step - warmup_steps) / (total_steps - start_step - warmup_steps)
        progress = min(progress, 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return peak_lr * cosine_decay

    async def _data_fetching_loop(self):
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
                act_token_t = torch.tensor(action_token_np, dtype=torch.long, device=device) # Tokens 是 long 类型
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

    async def run_training_epoch(self) -> Tuple[float, float, float, float, Dict[str, float], int]:
        if self.next_ready_batch is None:
            print(f"Trainer {self.rank}: 等待初始超级批次...")
            while self.next_ready_batch is None:
                await asyncio.sleep(0.2)
            print(f"Trainer {self.rank}: 初始数据已收到，开始第一个训练周期。")

        current_lrs = {}
        value_lr = self._get_current_lr(self.global_step, self.value_lr, self.value_warmup_steps, self.train_iters)
        policy_lr = self._get_current_lr(self.global_step, self.policy_lr, self.policy_warmup_steps, self.train_iters, start_step=self.policy_train_start_step)
        
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'value': param_group['lr'] = value_lr; current_lrs['value'] = value_lr
            elif param_group['name'] == 'policy': param_group['lr'] = policy_lr; current_lrs['policy'] = policy_lr

        current_batch = self.next_ready_batch
        self.next_ready_batch = None
        
        inputs_batch = current_batch['inputs_batch']
        act_token_t = current_batch['act_token']
        adv_t = current_batch['advantage']
        logits_old_t = current_batch['logits_old']
        v_targ_t = current_batch['value_target']
        policy_sample_time = current_batch['sample_time']
        policy_prep_time = current_batch['prep_time']

        # 修正std 归一化（消融1）
        # 计算本地统计量
        local_sum = adv_t.sum()
        local_sq_sum = (adv_t * adv_t).sum()
        local_count = torch.tensor([adv_t.numel()], device=adv_t.device, dtype=torch.float32)

        # 使用分布式all_reduce获取全局统计量
        stats_tensor = torch.stack([local_sum, local_sq_sum, local_count.squeeze(0)])
        distributed.all_reduce(stats_tensor, op=distributed.ReduceOp.SUM)

        global_sum, global_sq_sum, global_count = stats_tensor[0], stats_tensor[1], stats_tensor[2]
        global_mean = global_sum / torch.clamp(global_count, min=1.0)
        global_var = torch.clamp(global_sq_sum / torch.clamp(global_count, min=1.0) - global_mean * global_mean, min=1e-12)
        global_std = torch.sqrt(global_var)

        epoch_losses, epoch_p_losses, epoch_v_losses, epoch_e_losses, epoch_kl_losses = [], [], [], [], []
        epoch_ent, epoch_kl_divs = [], []   
        
        num_updates_in_epoch = self.super_batch_size // self.train_batch_size
        t_policy_train_start = time.time()
        
        for i in range(num_updates_in_epoch):
            start = i * self.train_batch_size; end = start + self.train_batch_size
            mini_inputs = {k: v[start:end] for k, v in inputs_batch.items()}
            
            mini_act_token = act_token_t[start:end]
            mini_adv = adv_t[start:end]
            mini_logits_old = logits_old_t[start:end]
            mini_v_targ = v_targ_t[start:end]
            
            # 使用全局统计量进行归一化
            normalized_adv = (mini_adv - global_mean) / (global_std + 1e-8)
            # 前向
            action_logits, value = self.model.forward(mini_inputs)
            value = value.to(torch.float32)

            action_logits_reshape = action_logits.view(
                -1, NUM_ACTIONS_CHUNK, ACTION_DIM, action_logits.shape[-1]
            )

            # 价值损失 (不变)
            value_loss = self.vf_coef * torch.mean((value - mini_v_targ) ** 2)
            
            if self.global_step < self.policy_train_start_step:
                loss = value_loss
                policy_loss = torch.tensor(0.0, device=loss.device)
                ent_loss = torch.tensor(0.0, device=loss.device)
                kl_loss = torch.tensor(0.0, device=loss.device) 
                kl_div = 0.0
                ent = torch.tensor(0.0, device=loss.device)
            else:
                # 策略与熵损失 (离散版本)
                dist = torch.distributions.Categorical(logits=action_logits_reshape)
                logp = dist.log_prob(mini_act_token) # 对动作维度求和

                with torch.no_grad():
                    dist_old = torch.distributions.Categorical(logits=mini_logits_old)
                    logp_old = dist_old.log_prob(mini_act_token)

                kl_div_tensor = kl.kl_divergence(dist_old, dist)
                kl_div = torch.mean(kl_div_tensor).item() # 作为指标
                kl_loss = self.kl_coef * torch.mean(kl_div_tensor) # 作为损失
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
                    # τ 的非对称设置：通常 τ_neg > τ_pos（负优势更“硬”一点）
                    tau_pos = 1.0
                    tau_neg = 2.0
                    if tau_pos <= 0 or tau_neg <= 0:
                        raise ValueError(f"tau_pos/tau_neg must be > 0, got {tau_pos}, {tau_neg}")

                    # 数值稳定：避免 ratio 极端导致 inf（可按需调大/关掉）
                    ratio_min = 1e-6
                    ratio_max = 1e6
                    r = ratio.clamp(ratio_min, ratio_max)

                    tau_pos_t = torch.full_like(adv_unsqueezed, tau_pos)
                    tau_neg_t = torch.full_like(adv_unsqueezed, tau_neg)
                    tau = torch.where(adv_unsqueezed > 0, tau_pos_t, tau_neg_t)

                    # gate(r) = (4/τ) * sigmoid( τ*(r-1) )
                    x = tau * (r - 1.0)
                    gate = torch.sigmoid(x) * (4.0 / tau)

                    # surrogate = gate * A   （注意：这里不再是 r*A）
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

        avg_loss = np.mean(epoch_losses)
        avg_p_loss = np.mean(epoch_p_losses)
        avg_v_loss = np.mean(epoch_v_losses)
        avg_e_loss = np.mean(epoch_e_losses)
        avg_kl_loss = np.mean(epoch_kl_losses)
        avg_ent = np.mean(epoch_ent)
        avg_kl_div = np.mean(epoch_kl_divs)

        perf_metrics = {
            "policy_sample_time": policy_sample_time,
            "policy_prep_time": policy_prep_time,
            "policy_train_time": time.time() - t_policy_train_start
        }

        return avg_loss, avg_p_loss, avg_v_loss, avg_e_loss, avg_kl_loss, current_lrs, self.global_step, avg_ent, avg_kl_div, perf_metrics

# ================================================================
# 5. 主逻辑
# ================================================================
def build_openvla_cfg(args) -> GenerateConfig:
    """
    构建 OpenVLA 配置
    Args:
        args: 解析后的命令行参数
    """
    cfg = GenerateConfig(
        pretrained_checkpoint=args.pretrained_checkpoint,
        use_l1_regression=False, # Note: ActorCritic in discrete model doesn't use this
        use_diffusion=False,
        use_film=False,
        num_images_in_input=1,
        # zzq 1124 开启 proprio 
        use_proprio=False, # Note: ActorCritic in discrete model can handle this
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        unnorm_key=args.benchmark+"_no_noops",
        checkpoint2=args.checkpoint2,
    )
    return cfg

def main(args):
    """
    主函数，接受命令行参数
    Args:
        args: 解析后的命令行参数
    """
    # 将 benchmark 字符串转换为 TaskSuite 枚举
    benchmark = args.benchmark
    torch_dtype = torch.bfloat16 if args.use_bf16 else torch.float32
    agent_config_path = Path.cwd() / args.agent_config_path
    trainer_config_path = Path.cwd() / args.trainer_config_path
    
    if not os.path.exists(args.pretrained_checkpoint):
        print(f"错误: OpenVLA checkpoint 路径 '{args.pretrained_checkpoint}' 不存在。请更新 PRETRAINED_CHECKPOINT。")
        return

    os.environ["RAY_DEDUP_LOGS"] = "0"
    object_store_size_gb = 256  # 分配的GB数，根据系统内存调整（建议256-896GB）
    object_store_memory_bytes = int(object_store_size_gb * 1024 * 1024 * 1024)
    print(f"正在初始化 Ray，并为对象存储分配 {object_store_size_gb} GB 内存...")
    ray.init(
        ignore_reinit_error=True, 
        _temp_dir='/dev/shm',
        object_store_memory=object_store_memory_bytes
    )
    print(f"Ray 初始化完成，对象存储分配 {object_store_size_gb} GB 内存。")
    log_dir = f"runs/Libero/{args.benchmark}/{int(time.time())}_{args.exp_name}"
    writer = SummaryWriter(log_dir)
    stats_actor = StatsActor.remote(window_size=args.moving_avg_window)
    print(f"TensorBoard 日志将保存在: {log_dir}")

    cfg = build_openvla_cfg(args)

    print("--- 步骤 1: 创建 Actors ---")
    replay_buffers = [ReplayBufferActor.remote(capacity=args.replay_capacity) for _ in range(args.num_trainer_gpus)]
    trainer_group = [
        TrainerActor.remote(
            rank=i, world_size=args.num_trainer_gpus, replay_buffer=replay_buffers[i], cfg=cfg,
            train_batch_size=args.train_batch_size, accumulation_steps=args.accumulation_steps,
            use_bf16=args.use_bf16, torch_dtype=torch_dtype, policy_lr=args.policy_lr, value_lr=args.value_lr,
            gamma=args.gamma, lambda_=args.lambda_, clip_eps=args.clip_eps, vf_coef=args.vf_coef,
            ent_coef=args.ent_coef, kl_coef=args.kl_coef, reward_scale=args.reward_scale,
            value_warmup_steps=args.value_warmup_steps, policy_warmup_steps=args.policy_warmup_steps,
            policy_train_start_step=args.policy_train_start_step, train_iters=args.train_iters,
            clip_mode=args.clip_mode
        )
        for i in range(args.num_trainer_gpus)
    ]
    inference_pool = [InferenceActor.remote(actor_id=i, cfg=cfg, stats_actor=stats_actor, torch_dtype=torch_dtype, inference_batch=args.inference_batch, inference_timeout_ms=args.inference_timeout_ms) for i in range(args.num_inference_actors)]
    reward_inference_pool = [
        RewardInferenceActor.remote(
            actor_id=i, 
            stats_actor=stats_actor, 
            agent_config_path=agent_config_path,
            inference_batch=args.inference_batch,
            inference_timeout_ms=args.inference_timeout_ms
        ) for i in range(args.num_reward_inference_actors)
    ]
    denoiser_inference_pool = [
        DenoiserInferenceActor.remote(
            actor_id=i,
            stats_actor=stats_actor,
            agent_config_path=agent_config_path,
            trainer_config_path=trainer_config_path,
            inference_batch=args.inference_batch,
            inference_timeout_ms=args.inference_timeout_ms
        ) for i in range(args.num_denoiser_inference_actors)
    ]
    rollout_workers = [
        RolloutWorkerActor.remote(
            inference_pool[i % args.num_inference_actors],
            replay_buffers[i % args.num_trainer_gpus], 
            i, 
            stats_actor, 
            cfg,
            benchmark,                    # benchmark_name
            args.num_step_cond,               # num_step_cond
            args.imagine_horizon,            # imagine_horizon
            torch_dtype,                # torch_dtype
            reward_inference_pool[i % args.num_reward_inference_actors],    # reward_infer
            denoiser_inference_pool[i % args.num_denoiser_inference_actors],  # denoiser_infer
            args.reward_scale,              # reward_scale
            args.gamma,                     # gamma
            args.lambda_                    # lambda_
        ) for i in range(args.num_rollout_workers)
    ]
    eval_workers = [
        EvaluationWorkerActor.remote(
            inference_pool[i % args.num_inference_actors], f"eval_{i}", stats_actor, cfg, benchmark, torch_dtype
        ) for i in range(args.num_eval_workers)
    ]
    print(f"已创建 {args.num_rollout_workers} 个 Rollout workers 和 {args.num_eval_workers} 个 Evaluation workers。")

    print("\n--- 步骤 2: 建立独立的 DeepSpeed 训练组 ---")
    # zzq 1125 通信组，使用find_free_port
    train_group_port = find_free_port()

    broadcast_group_port = find_free_port()
    while broadcast_group_port == train_group_port:
        broadcast_group_port = find_free_port()
    trainer_master_addr = ray.get(trainer_group[0].get_node_ip.remote())
    train_setup_tasks = [actor.setup_deepspeed_group.remote(trainer_master_addr, train_group_port) for actor in trainer_group]
    ray.get(train_setup_tasks)
    print("DeepSpeed 训练组建立完成。")

    print(f"\n--- 步骤 3: 建立共享广播组 ({args.broadcast_group_name}) ---")
    broadcast_participants = [trainer_group[0]] + inference_pool
    broadcast_group_world_size = len(broadcast_participants)
    broadcast_master_addr = ray.get(trainer_group[0].get_node_ip.remote())
    broadcast_setup_tasks = [
        actor.setup_broadcast_group.remote(
            master_addr=broadcast_master_addr, master_port=broadcast_group_port,
            group_name=args.broadcast_group_name, group_world_size=broadcast_group_world_size,
            my_rank_in_group=rank) for rank, actor in enumerate(broadcast_participants)
    ]
    ray.get(broadcast_setup_tasks)
    print("共享广播组建立完成。")

    inf_keys = ray.get(inference_pool[0].get_model_keys.remote())
    trainer_keys = ray.get(trainer_group[0].get_model_keys.remote())
    for key in inf_keys:
        if key not in trainer_keys:
            print(f"警告: 推理器中缺少训练器的键: {key}")
    for key in trainer_keys:
        if key not in inf_keys:
            print(f"警告: 训练器中缺少推理器的键: {key}")
    train_sig = ray.get(trainer_group[0].get_broadcast_signature.remote())
    infer_sig = ray.get(inference_pool[0].get_broadcast_signature.remote())
    # 打印前几十个，或计算哈希对比
    if len(train_sig) != len(infer_sig):
        raise RuntimeError(f"训练器与推理器的广播签名长度不匹配: {len(train_sig)} vs {len(infer_sig)}")
    for i, (a, b) in enumerate(zip(train_sig, infer_sig)):
        if a != b:
            raise RuntimeError(f"First mismatch at idx: {i}, trainer: {a}, inference: {b}")
    forward_test_tasks = [inf.forward_test.remote() for inf in inference_pool]
    ray.get(forward_test_tasks)
    print("推理器前向测试完成 (广播前)。")
    
    broadcast_task = trainer_group[0].broadcast_weights.remote(args.broadcast_group_name)
    receive_tasks = [inf.receive_and_update_weights.remote(args.broadcast_group_name) for inf in inference_pool]
    ray.get([broadcast_task] + receive_tasks)
    print("初始权重已广播到所有推理器。")

    forward_test_tasks = [inf.forward_test.remote() for inf in inference_pool]
    ray.get(forward_test_tasks)
    print("推理器前向测试完成 (广播后)。")

    print("\n--- 步骤 4: 启动 Rollout Workers 进行数据收集 ---")
    for w in rollout_workers: w.run.remote()
    for w in eval_workers: w.run.remote()

    print("\n--- 步骤 5: 等待远程经验池填充初始数据 ---")
    min_buffer_size_for_start = args.train_batch_size * args.accumulation_steps
    assert min_buffer_size_for_start < args.replay_capacity, "初始填充量必须小于回放池总容量"
    while not all(size >= min_buffer_size_for_start for size in ray.get([rb.size.remote() for rb in replay_buffers])):
        sizes = ray.get([rb.size.remote() for rb in replay_buffers])
        print(f"等待所有经验池填充初始数据 (目标: {min_buffer_size_for_start})... (当前大小: {sizes})")
        time.sleep(5)
    print("远程经验池已准备好，训练器将按需获取数据。")

    print("\n--- 步骤 6: 开始主训练与同步循环 ---")
    start_time = time.time()
    last_log_time = time.time()
    last_log_global_step = 0
    global_step = 0
    while global_step < args.train_iters:
        t_train_start = time.time()
        train_tasks = [trainer.run_training_epoch.remote() for trainer in trainer_group]
        results = ray.get(train_tasks)
        _, _, _, _, _, _, global_step, _, _, _ = results[0]
        train_time = time.time() - t_train_start

        t_sync_start = time.time()
        broadcast_task = trainer_group[0].broadcast_weights.remote(args.broadcast_group_name)
        receive_tasks = [inf.receive_and_update_weights.remote(args.broadcast_group_name) for inf in inference_pool]
        ray.get([broadcast_task] + receive_tasks)
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


            total_losses, p_losses, v_losses, e_losses, kl_losses, lrs_list, _, ents, avg_kl_divs, perf_metrics_list = zip(*results)
            current_lrs = lrs_list[0]

            elapsed_time = current_time - start_time
            total_buffer_size = sum(ray.get([rb.size.remote() for rb in replay_buffers]))

            print(f"更新步 {global_step}/{args.train_iters} | 时间: {elapsed_time:.1f}s | "
                  f"全局平均奖励: {avg_return:.2f} | 全局平均幕长: {avg_ep_len:.1f} | Eval奖励: {eval_avg_return:.2f} | "
                  f"value loss: {np.mean(v_losses):.4f} | LR(V/P): {current_lrs['value']:.7f}/{current_lrs['policy']:.7f} | "
                  f"Episodes数量: {total_episodes:,} | Step平均时间: {avg_step_time:.3f}s")

            writer.add_scalar('Train/Learning_Rate/Value', current_lrs['value'], global_step)
            writer.add_scalar('Train/Learning_Rate/Policy', current_lrs['policy'], global_step)
            writer.add_scalar('Loss/Total', np.mean(total_losses), global_step)
            writer.add_scalar('Loss/Policy', np.mean(p_losses), global_step)
            writer.add_scalar('Loss/Value', np.mean(v_losses), global_step)
            writer.add_scalar('Loss/Entropy', np.mean(e_losses), global_step)
            writer.add_scalar('Loss/KL', np.mean(kl_losses), global_step)

            writer.add_scalar('Metrics/Entropy', np.mean(ents), global_step)
            writer.add_scalar('Metrics/KL_Divergence', np.mean(avg_kl_divs), global_step)
            writer.add_scalar('Metrics/Training_Speed_Steps_per_Sec', training_speed_steps_per_sec, global_step)
            for metric_name, metric_value in timing_stats.items():
                writer.add_scalar(f'Performance/{metric_name}', metric_value, global_step)
            avg_policy_sample_time = np.mean([pm["policy_sample_time"] for pm in perf_metrics_list])
            avg_policy_prep_time = np.mean([pm["policy_prep_time"] for pm in perf_metrics_list])
            avg_policy_train_time = np.mean([pm["policy_train_time"] for pm in perf_metrics_list])
            writer.add_scalar('Performance/policy_sample_time', avg_policy_sample_time, global_step)
            writer.add_scalar('Performance/policy_prep_time', avg_policy_prep_time, global_step)
            writer.add_scalar('Performance/policy_train_time', avg_policy_train_time, global_step)
            writer.add_scalar('Performance/train_time', train_time, global_step)
            writer.add_scalar('Performance/sync_time', sync_time, global_step)
            writer.add_scalar('Performance/train_time_total', time.time() - t_train_start, global_step)

            writer.add_scalar('Rollout/_Global/Average_Return', avg_return, global_step)
            writer.add_scalar('Rollout/_Global/Average_Episode_Length', avg_ep_len, global_step)
            writer.add_scalar('Rollout/_Global/Average_Imagine_Reward', global_stats.get("avg_imagine_reward", 0.0), global_step)
            writer.add_scalar('Eval/_Global/Average_Return', eval_avg_return, global_step)
            writer.add_scalar('Eval/_Global/Average_Episode_Length', eval_avg_ep_len, global_step)

            writer.add_scalar('System/Replay_Buffer_Size_Total', total_buffer_size, global_step)
            writer.add_scalar('System/Total_Episodes_Processed', total_episodes, global_step)
            writer.add_scalar('System/Total_Env_Steps', total_env_steps, global_step)
            writer.add_scalar('System/Avg_Step_Time', avg_step_time, global_step)
            writer.add_scalar('System/Eval_Total_Episodes_Processed', eval_total_episodes, global_step)
            writer.add_scalar('System/Eval_Total_Env_Steps', eval_env_steps, global_step)
            writer.add_scalar('System/Eval_Avg_Step_Time', eval_avg_step_time, global_step)
            writer.add_scalar('System/Active_Rollout_Actors', global_stats.get("active_actor_count", 0), global_step)
            writer.add_scalar('System/Total_Samples_Produced', global_stats.get("total_samples_produced", 0), global_step)

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