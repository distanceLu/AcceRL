import os
import json
import argparse
os.environ["MUJOCO_GL"] = "osmesa"           # 强制软件渲染
os.environ["PYOPENGL_PLATFORM"] = "osmesa"   # 保险起见，给 PyOpenGL 也指明
# 设置临时文件目录，避免磁盘I/O瓶颈
os.environ["TMPDIR"] = "/dev/shm"
# 为了让 Ray 能看到所有可用的 GPU，我们在脚本开头设置。
# 注意: CUDA_VISIBLE_DEVICES 现在通过命令行参数设置
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
# 防止 transformers 库的 tokenizer 并行化警告
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
from datetime import datetime
import random
import asyncio
from collections import deque, defaultdict
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import math

import numpy as np

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

from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM
from experiments.robot.libero.libero_utils import GenerateConfig, TaskSuite
from rl.actor_critic_model_discrete import ActorCritic
from rl.utils import prepare_one_obs
# 训练/推理通信（保持接口不变）
from ds_com import TrainerActorCom, InferenceActorCom
from rl.com_utils import find_free_port

# ================================================================
# 0. 超参数与配置 - 命令行参数解析
# ================================================================
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='OpenVLA RL Training (PPO - No World Model)')
    
    # 环境变量
    parser.add_argument('--cuda-visible-devices', type=str, default='6,7',
                        help='CUDA visible devices (default: 6,7)')
    
    # Libero benchmark
    parser.add_argument('--benchmark', type=str, default='libero_spatial',
                        choices=['libero_spatial', 'libero_object', 'libero_goal', 'libero_10', 'libero_90'],
                        help='Libero benchmark suite (default: libero_spatial)')
    
    # Task IDs
    parser.add_argument('--task-ids', type=str, default='0,1,2,3,4,5,6,7,8,9',
                        help='Comma-separated list of task IDs (default: 0,1,2,3,4,5,6,7,8,9)')
    
    # 分布式系统参数
    parser.add_argument('--num-trainer-gpus', type=int, default=1,
                        help='Number of trainer GPUs (default: 1)')
    parser.add_argument('--num-inference-actors', type=int, default=1,
                        help='Number of inference actors (default: 1)')
    parser.add_argument('--num-rollout-workers', type=int, default=2,
                        help='Number of rollout workers (default: 2)')
    parser.add_argument('--num-eval-workers', type=int, default=20,
                        help='Number of evaluation workers (default: 20)')
    parser.add_argument('--rollout-local-buf', type=int, default=64,
                        help='Rollout local buffer size (default: 64)')
    parser.add_argument('--inference-batch', type=int, default=8,
                        help='Inference batch size (default: 8)')
    parser.add_argument('--inference-timeout-ms', type=int, default=300,
                        help='Inference timeout in milliseconds (default: 300)')
    parser.add_argument('--replay-capacity', type=int, default=1000,
                        help='Replay buffer capacity (default: 1000)')
    parser.add_argument('--train-batch-size', type=int, default=12,
                        help='Training batch size (default: 12)')
    parser.add_argument('--accumulation-steps', type=int, default=21,
                        help='Gradient accumulation steps (default: 21)')
    parser.add_argument('--train-iters', type=int, default=30000,
                        help='Total training iterations (default: 30000)')
    
    # Ray 对象存储
    parser.add_argument('--object-store-memory-gb', type=int, default=256,
                        help='Ray object store memory in GB (default: 256)')
    
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
    parser.add_argument('--sigma', type=float, default=0.5,
                        help='Sigma parameter for GIPO clip mode (default: 0.5)')
    
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
    parser.add_argument('--use-proprio', action='store_true', default=False,
                        help='Use proprioceptive state (default: False)')
    parser.add_argument('--num-images-in-input', type=int, default=1,
                        help='Number of images in input (default: 1)')
    parser.add_argument('--pretrained-checkpoint', type=str,
                        default='/cpfs01/liuwei_workspace/models/finetune_im/openvla-7b+libero_spatial_no_noops+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--discrete_acts--proprio_state--100000_chkpt',
                        help='Pretrained checkpoint path')
    parser.add_argument('--checkpoint2', type=str,
                        default='',
                        help='Second checkpoint path')
    
    parser.add_argument('--clip-mode', type=str, default='sapo',
                        choices=['ppo', 'sapo', 'gipo'],
                        help='Clipping mode (default: ppo)')
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Experiment name (default: auto-generated based on clip-mode)')
    
    # GAE 重计算选项
    parser.add_argument('--recompute-value', action='store_true', default=False,
                        help='Recompute value using current model before GAE calculation (default: False)')
    
    # Resume 功能
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Resume training from checkpoint directory (will load all parameters from saved args.json)')
    
    # Evaluation 参数
    parser.add_argument('--eval-deterministic', action='store_true', default=True,
                        help='Use deterministic actions during evaluation (default: True)')
    parser.add_argument('--no-eval-deterministic', action='store_false', dest='eval_deterministic',
                        help='Use stochastic actions during evaluation')
    
    args = parser.parse_args()
    
    # 如果是 resume 模式，加载保存的参数
    if args.resume_from:
        print(f"\n{'='*80}")
        print(f"Resume 模式：从 {args.resume_from} 加载训练配置")
        print(f"{'='*80}\n")
        
        saved_args_path = os.path.join(args.resume_from, "args.json")
        if not os.path.exists(saved_args_path):
            raise FileNotFoundError(f"无法找到保存的参数文件: {saved_args_path}")
        
        with open(saved_args_path, 'r', encoding='utf-8') as f:
            saved_args = json.load(f)
        
        # 检查 checkpoint 状态文件
        checkpoint_files = [f for f in os.listdir(args.resume_from) if f.startswith('trainer_state_step_')]
        if not checkpoint_files:
            raise FileNotFoundError(f"无法找到训练状态文件（trainer_state_step_*.pt）在 {args.resume_from}")
        
        # 获取最新的 checkpoint step
        resume_steps = [int(f.split('_')[-1].replace('.pt', '')) for f in checkpoint_files]
        resume_step = max(resume_steps)
        
        print(f"找到 checkpoint，resume_step = {resume_step}")
        print(f"\n加载保存的训练参数（忽略命令行参数，除了 --resume-from）：")
        print("-" * 80)
        
        # 保存 resume_from 路径
        resume_from_path = args.resume_from
        
        # 用保存的参数覆盖所有参数
        for key, value in saved_args.items():
            if hasattr(args, key):
                old_val = getattr(args, key)
                if old_val != value:
                    print(f"  {key}: {old_val} -> {value}")
                setattr(args, key, value)
        
        # 恢复 resume_from 和添加 resume_step
        args.resume_from = resume_from_path
        args.resume_step = resume_step
        
        print("-" * 80)
        print(f"✓ 参数加载完成，将从步数 {resume_step} 继续训练\n")
    
    # 设置 CUDA_VISIBLE_DEVICES 环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    
    # 如果没有提供 exp_name，自动生成
    if args.exp_name is None:
        args.exp_name = f"OpenVLA_DS_{args.clip_mode}_DISCRETE_task0_10k_buffer"
    
    return args

# ================================================================
# 数据结构
# ================================================================
@dataclass
class Trajectory:
    """完整轨迹，用于按轨迹存储和 GAE 重计算"""
    obs_list: List[Dict[str, torch.Tensor]]  # 每个时间步的 obs (prepare_one_obs 的结果)
    action_tokens: np.ndarray                 # shape: [T, NUM_ACTIONS_CHUNK, ACTION_DIM]
    rewards: np.ndarray                       # shape: [T,]
    behaviour_logits: np.ndarray              # shape: [T, NUM_ACTIONS_CHUNK, ACTION_DIM, VOCAB_SIZE]
    old_values: np.ndarray                    # shape: [T,] - RolloutWorker 收集时的 value
    bootstrap_value: float                    # 截断时的 bootstrap value
    is_terminal: bool                         # True=完整 episode，False=截断
    policy_versions: np.ndarray               # shape: [T,] - 每个样本的策略版本
    insert_steps: np.ndarray                  # shape: [T,] - 每个样本的插入时间戳
    
    @property
    def num_steps(self) -> int:
        return len(self.rewards)

@dataclass
class Experience:
    """单个样本，用于训练时的 mini-batch"""
    obs: Dict[str, torch.Tensor]            # prepare_one_obs 的结果（CPU tensors）
    action_token: np.ndarray                # 采样的离散动作 token (shape: [NUM_ACTIONS_CHUNK, ACTION_DIM])
    advantage: float
    behaviour_logits: np.ndarray            # 行为策略的 logits (shape: [NUM_ACTIONS_CHUNK, ACTION_DIM, VOCAB_SIZE])
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
            "total_env_steps": 0,
            "step_rewards": deque(maxlen=window_size),
        })
        self.timings = defaultdict(lambda: deque(maxlen=window_size))
        self.actor_last_active = {}
        self.active_window_seconds = 600
        self.total_samples_produced = 0
        self.window_size = window_size

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
        step_reward = ep_return / ep_length
        env_stats["step_rewards"].append(step_reward)
        if not env_name.startswith("eval_"):
            self.total_samples_produced += step_num
            if actor_id is not None:
                self.actor_last_active[actor_id] = time.time()

    def add_timing_metric(self, metric_name: str, value: float):
        """记录系统性能相关的计时指标"""
        self.timings[metric_name].append(value)

    def get_active_actor_count(self) -> int:
        current_time = time.time()
        cutoff = current_time - self.active_window_seconds
        return sum(1 for last_active in self.actor_last_active.values() if last_active >= cutoff)

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        per_env_stats = {}
        all_returns, all_lengths, all_step_times = [], [], []
        total_episodes_processed = 0
        total_env_steps = 0
        all_step_rewards = []
        eval_returns, eval_lengths, eval_step_times = [], [], []
        eval_total_episodes_processed = 0
        eval_total_env_steps = 0
        eval_step_rewards = []
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
                "total_episodes": env_data["total_episodes_processed"],
                "avg_step_reward": np.mean(env_data["step_rewards"])
            }
            if env_name.startswith("eval_"):
                eval_total_episodes_processed += env_data["total_episodes_processed"]
                eval_total_env_steps += env_data["total_env_steps"]
                eval_returns.extend(env_data["episode_returns"])
                eval_lengths.extend(env_data["episode_lengths"])
                eval_step_times.extend(env_data["step_times"])
                eval_step_rewards.extend(env_data["step_rewards"])
            else:
                total_episodes_processed += env_data["total_episodes_processed"]
                total_env_steps += env_data["total_env_steps"]
                all_returns.extend(env_data["episode_returns"])
                all_lengths.extend(env_data["episode_lengths"])
                all_step_times.extend(env_data["step_times"])
                all_step_rewards.extend(env_data["step_rewards"])
        per_env_stats["_global_rollout_"] = {
            "avg_return": np.mean(all_returns) if all_returns else 0.0,
            "avg_ep_len": np.mean(all_lengths) if all_lengths else 0.0,
            "avg_step_time": np.mean(all_step_times) if all_step_times else 0.0,
            "avg_step_reward": np.mean(all_step_rewards) if all_step_rewards else 0.0,
            "total_episodes_processed": total_episodes_processed,
            "total_env_steps": total_env_steps,
            "total_samples_produced": self.total_samples_produced,
            "active_actor_count": self.get_active_actor_count()
        }
        per_env_stats["_global_eval_"] = {
            "avg_return": np.mean(eval_returns) if eval_returns else 0.0,
            "avg_ep_len": np.mean(eval_lengths) if eval_lengths else 0.0,
            "avg_step_time": np.mean(eval_step_times) if eval_step_times else 0.0,
            "avg_step_reward": np.mean(eval_step_rewards) if eval_step_rewards else 0.0,
            "total_episodes_processed": eval_total_episodes_processed,
            "total_env_steps": eval_total_env_steps
        }
        timing_stats = {}
        for name, deq in self.timings.items():
            timing_stats[name] = np.mean(deq) if deq else 0.0
        per_env_stats["_timings_"] = timing_stats
        return per_env_stats
    
    def save_stats(self, save_path: str):
        """保存统计信息"""
        import pickle
        # 将 defaultdict 转换为普通 dict 以便序列化
        stats_dict = {}
        for env_name, env_data in self.stats.items():
            stats_dict[env_name] = {
                "episode_returns": list(env_data["episode_returns"]),
                "step_times": list(env_data["step_times"]),
                "episode_lengths": list(env_data["episode_lengths"]),
                "successes": list(env_data["successes"]),
                "total_episodes_processed": env_data["total_episodes_processed"],
                "total_env_steps": env_data["total_env_steps"],
                "step_rewards": list(env_data["step_rewards"]),
            }
        
        timings_dict = {name: list(deq) for name, deq in self.timings.items()}
        
        state = {
            'stats': stats_dict,
            'timings': timings_dict,
            'actor_last_active': self.actor_last_active,
            'total_samples_produced': self.total_samples_produced,
            'window_size': self.window_size
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(state, f)
        print(f"StatsActor 已保存到 {save_path}")
    
    def load_stats(self, load_path: str):
        """加载统计信息"""
        import pickle
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"无法找到 StatsActor 文件: {load_path}")
        
        with open(load_path, 'rb') as f:
            state = pickle.load(f)
        
        # 恢复 stats
        self.stats.clear()
        for env_name, env_data in state['stats'].items():
            self.stats[env_name] = {
                "episode_returns": deque(env_data["episode_returns"], maxlen=self.window_size),
                "step_times": deque(env_data["step_times"], maxlen=self.window_size),
                "episode_lengths": deque(env_data["episode_lengths"], maxlen=self.window_size),
                "successes": deque(env_data["successes"], maxlen=self.window_size),
                "total_episodes_processed": env_data["total_episodes_processed"],
                "total_env_steps": env_data["total_env_steps"],
                "step_rewards": deque(env_data["step_rewards"], maxlen=self.window_size),
            }
        
        # 恢复 timings
        self.timings.clear()
        for name, timing_list in state['timings'].items():
            self.timings[name] = deque(timing_list, maxlen=self.window_size)
        
        self.actor_last_active = state['actor_last_active']
        self.total_samples_produced = state['total_samples_produced']
        
        print(f"StatsActor 已从 {load_path} 加载")

# ================================================================
# 2. 经验回放与 Rollout
# ================================================================
@ray.remote
class ReplayBufferActor:
    """按轨迹存储的经验回放缓冲区"""
    def __init__(self, capacity: int):
        # capacity 表示最大轨迹数量
        self.trajectories: deque = deque(maxlen=capacity)
        self.insert_counter = 0  # 全局插入计数器

    def add_trajectory(self, traj: Trajectory):
        """添加一条轨迹"""
        self.trajectories.append(traj)
        self.insert_counter += 1

    def size(self) -> int:
        """返回轨迹数量"""
        return len(self.trajectories)
    
    def total_steps(self) -> int:
        """返回所有轨迹的总步数"""
        return sum(t.num_steps for t in self.trajectories)
    
    def sample_trajectories(self, min_steps: int) -> List[Trajectory]:
        """采样足够步数的轨迹
        
        Args:
            min_steps: 最小需要的步数
            
        Returns:
            采样的轨迹列表，总步数 >= min_steps
        """
        if not self.trajectories:
            return []
        
        sampled = []
        total = 0
        indices = list(range(len(self.trajectories)))
        random.shuffle(indices)
        
        for idx in indices:
            traj = self.trajectories[idx]
            sampled.append(traj)
            total += traj.num_steps
            if total >= min_steps:
                break
        
        return sampled
    
    def save_buffer(self, save_path: str):
        """保存经验回放缓冲区"""
        import pickle
        state = {
            'trajectories': list(self.trajectories),
            'insert_counter': self.insert_counter,
            'capacity': self.trajectories.maxlen
        }
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(state, f)
        print(f"ReplayBuffer 已保存到 {save_path} (轨迹数: {len(self.trajectories)}, 步数: {self.total_steps()})")
    
    def load_buffer(self, load_path: str):
        """加载经验回放缓冲区"""
        import pickle
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"无法找到 ReplayBuffer 文件: {load_path}")
        
        with open(load_path, 'rb') as f:
            state = pickle.load(f)
        
        self.trajectories = deque(state['trajectories'], maxlen=self.trajectories.maxlen)
        self.insert_counter = state['insert_counter']
        print(f"ReplayBuffer 已从 {load_path} 加载 (轨迹数: {len(self.trajectories)}, 步数: {self.total_steps()})")


class BaseWorkerActor:
    """rollout 和 eval worker 的共享逻辑。"""
    def __init__(self, infer, replay, wid, stats_actor, cfg, benchmark_name, task_ids):
        self.infer = infer
        self.replay = replay
        self.stats_actor = stats_actor
        self.cfg = cfg
        # 仅需 processor，Worker 不加载大模型
        self.processor = get_processor(cfg)
        self.benchmark_name = benchmark_name
        from rl.libero_env import LiberoEnvWrapper

        # 对外暴露 task_ids 列表
        self.task_ids = task_ids
        self.num_tasks = len(task_ids)
        print(f"BaseWorker {wid}: 正在初始化 {self.num_tasks} 个 Libero 环境，task_ids = {task_ids}...")
        self.envs = [
            LiberoEnvWrapper(
                benchmark_name=self.benchmark_name,
                task_id=task_id,
                image_size=224,
                render_mode="rgb_array"
            ) for task_id in task_ids]
        print(f"BaseWorker {wid}: 环境初始化完成。")
        
        self.env = None
        self.current_env_idx = -1
        self.wid = wid
        self.task_description = None
        self.current_env_name = None

@ray.remote
class RolloutWorkerActor(BaseWorkerActor):
    def __init__(self, infer, replay, wid, stats_actor, cfg, benchmark_name, reward_scale, torch_dtype, rollout_local_buf, task_ids):
        super().__init__(infer, replay, wid, stats_actor, cfg, benchmark_name, task_ids)
        self.env_outcome = [deque(maxlen=100) for _ in range(self.num_tasks)]
        self.local_buffer = []
        self.reward_scale = reward_scale
        self.torch_dtype = torch_dtype
        self.rollout_local_buf = rollout_local_buf

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
            current_seed = int(time.time() * 1000) + self.wid + os.getpid()
            obs, info = self._reset_and_select_env(seed=current_seed)
            reward_sum, time_start, step_count_total = 0.0, time.time(), 0
            step_count = 0
            while True:
                inputs_t = prepare_one_obs(self.cfg, self.processor, obs, self.task_description, self.torch_dtype)
                action_env, action_token, logits, value, policy_version = ray.get(self.infer.request.remote(inputs_t, deterministic=False))
                chunk_reward, done = 0.0, False
                for i in range(len(action_env)):
                    single_action = action_env[i]
                    nxt, r, term, trunc, info = self.env.step(single_action)
                    reward_sum += r
                    chunk_reward += r * self.reward_scale
                    step_count_total += 1
                    if term or trunc: done = True; break
                # 记录当前时间戳作为 insert_step
                insert_step = int(time.time() * 1000)
                self.local_buffer.append((inputs_t, action_token, chunk_reward, logits, value, policy_version, insert_step))
                obs = nxt
                step_count += 1

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
                        step_num=step_count,
                    )
                    step_count = 0
                    reward_sum = 0.0
                    if self.local_buffer: 
                        self._process_traj(self.local_buffer, bootstrap_val=0.0, is_terminal=True)
                    self.local_buffer.clear()
                    current_seed = int(time.time() * 1000) + self.wid + os.getpid()
                    obs, info = self._reset_and_select_env(seed=current_seed)
                    time_start, step_count_total = time.time(), 0
                elif len(self.local_buffer) == self.rollout_local_buf + 1:
                    _, _, _, _, bootstrap_val, _, _ = self.local_buffer[-1]
                    self._process_traj(self.local_buffer[:-1], bootstrap_val=bootstrap_val, is_terminal=False)
                    self.local_buffer = [self.local_buffer[-1]]
        except Exception as e: import traceback; print(f"[ERROR] RolloutWorker {self.wid} run() 崩溃: {e}", flush=True); traceback.print_exc(); raise

    def _process_traj(self, traj_segment, bootstrap_val: float, is_terminal: bool):
        """打包轨迹原始数据，不计算 GAE（由 Trainer 统一计算）"""
        traj = Trajectory(
            obs_list=[s for s, _, _, _, _, _, _ in traj_segment],
            action_tokens=np.stack([a for _, a, _, _, _, _, _ in traj_segment]).astype(np.int64),
            rewards=np.array([r for _, _, r, _, _, _, _ in traj_segment], dtype=np.float32),
            behaviour_logits=np.stack([l for _, _, _, l, _, _, _ in traj_segment]).astype(np.float32),
            old_values=np.array([v for _, _, _, _, v, _, _ in traj_segment], dtype=np.float32),
            bootstrap_value=float(bootstrap_val),
            is_terminal=is_terminal,
            policy_versions=np.array([pv for _, _, _, _, _, pv, _ in traj_segment], dtype=np.int64),
            insert_steps=np.array([ins for _, _, _, _, _, _, ins in traj_segment], dtype=np.int64),
        )
        self.replay.add_trajectory.remote(traj)

@ray.remote
class EvaluationWorkerActor(BaseWorkerActor):
    def __init__(self, infer, wid, stats_actor, cfg, benchmark_name, torch_dtype, task_ids, deterministic=True):
        super().__init__(infer, None, wid, stats_actor, cfg, benchmark_name, task_ids)
        self.torch_dtype = torch_dtype
        self.deterministic = deterministic
        print(f"EvaluationWorker {self.wid}: 环境初始化完成，deterministic={self.deterministic}")

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
                step_count = 0
                while not done:
                    inputs_t = prepare_one_obs(self.cfg, self.processor, obs, self.task_description, self.torch_dtype)
                    action_env, _, _, _, _ = ray.get(self.infer.request.remote(inputs_t, deterministic=self.deterministic))
                    step_count += 1
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
                    step_num=step_count,
                )
                current_seed = int(time.time() * 1000) + os.getpid() + random.randint(0, 10000)
                obs, info = self._reset_and_select_env(seed=current_seed)
        except Exception as e: import traceback; print(f"[ERROR] EvaluationWorker {self.wid} run() 崩溃: {e}", flush=True); traceback.print_exc(); raise


# ================================================================
# 3. 推理器 (InferenceActor)
# ================================================================
@ray.remote(num_gpus=1)
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
        self.policy_version = 0  # 策略版本号，每次更新权重时递增

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
    
    def receive_and_update_weights(self, group_name):
        """覆盖基类方法，接收权重后自增策略版本"""
        super().receive_and_update_weights(group_name)
        self.policy_version += 1
        if self.actor_id == 0:
            print(f"InferenceActor {self.actor_id}: 已更新到 policy_version={self.policy_version}")

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
                        actions_env[i],           # 反归一化的环境动作
                        action_tokens[i],         # 离散动作 token
                        logits[i],                # 对应的 logits
                        values[i],                # 价值估计
                        self.policy_version       # 当前策略版本
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
    

# ================================================================
# 4. 训练器 (TrainerActor)
# ================================================================
@ray.remote(num_gpus=1)
class TrainerActor(TrainerActorCom):
    def __init__(self, rank, world_size, replay_buffer, cfg, train_batch_size, accumulation_steps, 
                 use_bf16, torch_dtype, policy_lr, value_lr, gamma, lambda_, clip_eps, vf_coef, 
                 ent_coef, kl_coef, reward_scale, value_warmup_steps, policy_warmup_steps, 
                 policy_train_start_step, train_iters, clip_mode, recompute_value, sigma):
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
        self.recompute_value = recompute_value
        self.sigma = sigma
        
        self.global_step = 0
        self.policy_version = 0  # 策略版本号，每次更新后递增

        print(f"TrainerActor Rank {self.rank} 初始化于 GPU: {ray.get_gpu_ids()} (recompute_value={recompute_value})")

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

        n_total = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数量: {n_total:,}, 可训练参数量: {n_trainable:,}")

    def start_data_fetching_loop(self):
        if self.data_fetching_task is not None and not self.data_fetching_task.done():
            print(f"Trainer {self.rank}: 后台数据准备循环已在运行。")
            return False
        self.data_fetching_task = asyncio.get_event_loop().create_task(self._data_fetching_loop())
        return True

    async def save_agent(self, ckpt_dir: str, step: int):
        """
        只在 rank-0 上调用。调用 ActorCritic 内部的 save_model
        """
        os.makedirs(ckpt_dir, exist_ok=True)
        self.base_model.save_model(ckpt_dir, epoch=step)
        print(f"[Trainer {self.rank}] 已保存 checkpoint -> {ckpt_dir}/agent_lora_epoch_{step}, agent_extra_layers_epoch_{step}.pt")
    
    async def save_checkpoint(self, ckpt_dir: str, step: int):
        """保存完整的训练状态（包含优化器和训练进度）"""
        os.makedirs(ckpt_dir, exist_ok=True)
        
        # 保存模型权重
        self.base_model.save_model(ckpt_dir, epoch=step)
        
        # 保存训练器状态
        checkpoint = {
            'global_step': self.global_step,
            'policy_version': self.policy_version,
            'random_state': {
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                'numpy': np.random.get_state(),
                'python': random.getstate(),
            }
        }
        
        # DeepSpeed 优化器状态通过 model 保存
        checkpoint_path = os.path.join(ckpt_dir, f'trainer_state_step_{step}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存 DeepSpeed checkpoint（包含优化器状态）
        ds_checkpoint_path = os.path.join(ckpt_dir, f'deepspeed_step_{step}')
        self.model.save_checkpoint(ds_checkpoint_path, tag=f'step_{step}')
        
        print(f"[Trainer {self.rank}] 完整训练状态已保存:")
        print(f"  - 模型: {ckpt_dir}/agent_lora_epoch_{step}")
        print(f"  - 训练状态: {checkpoint_path}")
        print(f"  - DeepSpeed: {ds_checkpoint_path}")
    
    async def load_checkpoint(self, ckpt_dir: str, step: int):
        """恢复训练状态"""
        checkpoint_path = os.path.join(ckpt_dir, f'trainer_state_step_{step}.pt')
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"无法找到训练状态文件: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 恢复训练进度
        self.global_step = checkpoint['global_step']
        self.policy_version = checkpoint['policy_version']
        
        # 恢复随机数状态
        torch.set_rng_state(checkpoint['random_state']['torch'])
        if checkpoint['random_state']['cuda'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint['random_state']['cuda'])
        np.random.set_state(checkpoint['random_state']['numpy'])
        random.setstate(checkpoint['random_state']['python'])
        
        # 加载 DeepSpeed checkpoint（包含优化器状态）
        ds_checkpoint_path = os.path.join(ckpt_dir, f'deepspeed_step_{step}')
        _, client_state = self.model.load_checkpoint(ds_checkpoint_path, tag=f'step_{step}')
        
        print(f"[Trainer {self.rank}] 训练状态已恢复:")
        print(f"  - global_step: {self.global_step}")
        print(f"  - policy_version: {self.policy_version}")
        
        return True

    def _get_current_lr(self, current_step: int, peak_lr: float, warmup_steps: int, total_steps: int, start_step: int = 0) -> float:
        if current_step < start_step: return 0.0
        effective_step = current_step - start_step
        if effective_step < warmup_steps: return peak_lr * (effective_step / warmup_steps)
        progress = (effective_step - warmup_steps) / (total_steps - start_step - warmup_steps)
        progress = min(progress, 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return peak_lr * cosine_decay

    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, 
                      bootstrap_value: float, is_terminal: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算单条轨迹的 GAE 和 returns (GPU版本)
        
        Args:
            rewards: [T,] 每步的奖励 (GPU tensor)
            values: [T,] 每步的价值估计 (GPU tensor)
            bootstrap_value: 最后一步的 bootstrap value（截断时使用）
            is_terminal: 是否是完整 episode
            
        Returns:
            advantages: [T,] (GPU tensor)
            returns: [T,] (GPU tensor)
        """
        T = len(rewards)
        device = rewards.device
        advs = torch.zeros(T, dtype=torch.float32, device=device)
        rets = torch.zeros(T, dtype=torch.float32, device=device)
        
        # 如果是完整 episode，最后的 bootstrap 应该是 0
        last_value = 0.0 if is_terminal else bootstrap_value
        gae = 0.0
        
        for i in reversed(range(T)):
            next_v = last_value if i == T - 1 else values[i + 1].item()
            delta = rewards[i].item() + self.gamma * next_v - values[i].item()
            gae = delta + self.gamma * self.lambda_ * gae
            advs[i] = gae
            rets[i] = gae + values[i].item()
        
        return advs, rets

    def _compute_diagnostic_metrics(
        self,
        ratio: torch.Tensor,  # [B, NUM_ACTIONS_CHUNK, ACTION_DIM] or [B]
        advantage: torch.Tensor,  # [B]
        policy_version: torch.Tensor,  # [B]
        insert_step: torch.Tensor,  # [B]
        current_policy_version: int,
        current_step: int,
        clip_eps: float = 0.2,
    ) -> Dict[str, float]:
        """
        计算诊断指标：staleness、ratio 分布、ESS、PG Active/Dead、有效贡献等
        
        根据 docs/clip_metrics.md 文档实现的完整指标集合。
        """
        with torch.no_grad():
            metrics = {}
            
            # ==================== 预处理 ====================
            # ratio 可能是 [B, NUM_ACTIONS_CHUNK, ACTION_DIM]，需要展平
            ratio_flat = ratio.reshape(-1)
            
            # advantage 需要扩展到与 ratio_flat 相同的维度
            if ratio.dim() == 3:  # [B, NUM_ACTIONS_CHUNK, ACTION_DIM]
                adv_expanded = advantage.unsqueeze(1).unsqueeze(2).expand_as(ratio).reshape(-1)
            else:  # [B]
                adv_expanded = advantage
            
            # ==================== 1. Staleness（数据陈旧度）====================
            staleness_ver = current_policy_version - policy_version.float()  # Δv
            metrics['staleness_ver_mean'] = staleness_ver.mean().item()
            metrics['staleness_ver_p95'] = torch.quantile(staleness_ver, 0.95).item()
            
            # Age（步数差）- 使用 batch 内相对 age
            age_steps = insert_step.float().max() - insert_step.float()
            metrics['age_steps_mean'] = age_steps.mean().item()
            metrics['age_steps_p95'] = torch.quantile(age_steps, 0.95).item() if age_steps.numel() > 0 else 0.0
            metrics['age_steps_max'] = age_steps.max().item()
            
            # 分桶阈值（绝对）
            NEW_THRESHOLD = 2
            OLD_THRESHOLD = 10
            
            new_mask_batch = staleness_ver <= NEW_THRESHOLD  # [B]
            old_mask_batch = staleness_ver >= OLD_THRESHOLD  # [B]
            
            # 扩展到与 ratio_flat 相同的维度
            if ratio.dim() == 3:
                new_mask = new_mask_batch.unsqueeze(1).unsqueeze(2).expand_as(ratio).reshape(-1)
                old_mask = old_mask_batch.unsqueeze(1).unsqueeze(2).expand_as(ratio).reshape(-1)
            else:
                new_mask = new_mask_batch
                old_mask = old_mask_batch
            
            # A. 分桶组成（绝对阈值）
            metrics['staleness_old_frac_abs'] = old_mask_batch.float().mean().item()
            metrics['staleness_new_frac_abs'] = new_mask_batch.float().mean().item()
            
            # A2) 旧桶内的陈旧度分布
            if old_mask_batch.any():
                old_gaps = staleness_ver[old_mask_batch]
                metrics['staleness_old_gap_mean_abs'] = old_gaps.mean().item()
                metrics['staleness_old_gap_p95_abs'] = torch.quantile(old_gaps, 0.95).item()
            else:
                metrics['staleness_old_gap_mean_abs'] = 0.0
                metrics['staleness_old_gap_p95_abs'] = 0.0
            
            # B. 相对陈旧度（Relative Staleness）
            staleness_ratio = staleness_ver / max(current_policy_version, 1)
            metrics['staleness_ratio_mean'] = staleness_ratio.mean().item()
            metrics['staleness_ratio_p95'] = torch.quantile(staleness_ratio, 0.95).item()
            
            # B2) 相对阈值分桶
            NEW_RATIO_THRESHOLD = 0.05   # 落后 <= 5%
            OLD_RATIO_THRESHOLD = 0.5    # 落后 >= 50%
            
            new_mask_ratio_batch = staleness_ratio <= NEW_RATIO_THRESHOLD
            old_mask_ratio_batch = staleness_ratio >= OLD_RATIO_THRESHOLD
            
            # 扩展到与 ratio_flat 相同的维度
            if ratio.dim() == 3:
                new_mask_ratio = new_mask_ratio_batch.unsqueeze(1).unsqueeze(2).expand_as(ratio).reshape(-1)
                old_mask_ratio = old_mask_ratio_batch.unsqueeze(1).unsqueeze(2).expand_as(ratio).reshape(-1)
            else:
                new_mask_ratio = new_mask_ratio_batch
                old_mask_ratio = old_mask_ratio_batch
            
            metrics['staleness_old_frac_ratio'] = old_mask_ratio_batch.float().mean().item()
            metrics['staleness_new_frac_ratio'] = new_mask_ratio_batch.float().mean().item()
            
            # ==================== 2. Ratio / log-ratio 分布 ====================
            metrics['rho_mean'] = ratio_flat.mean().item()
            metrics['rho_p50'] = torch.median(ratio_flat).item()
            metrics['rho_p90'] = torch.quantile(ratio_flat, 0.90).item()
            metrics['rho_p99'] = torch.quantile(ratio_flat, 0.99).item()
            metrics['rho_max'] = ratio_flat.max().item()
            
            # log-ratio
            logrho = torch.log(ratio_flat.clamp(min=1e-8))
            metrics['logrho_mean'] = logrho.mean().item()
            metrics['abs_logrho_p95'] = torch.quantile(torch.abs(logrho), 0.95).item()
            
            # ==================== 3. Hard Clip 指标（PPO）====================
            if self.clip_mode == "ppo":
                # Dead gradient: (A > 0 and ρ > 1+ε) or (A < 0 and ρ < 1-ε)
                dead_mask = ((adv_expanded > 0) & (ratio_flat > (1 + clip_eps))) | \
                           ((adv_expanded < 0) & (ratio_flat < (1 - clip_eps)))
                
                metrics['pg_dead_frac'] = dead_mask.float().mean().item()
                metrics['pg_active_frac'] = 1.0 - metrics['pg_dead_frac']
                
                # 分桶统计（绝对阈值）
                if new_mask.any():
                    metrics['pg_dead_frac_new'] = dead_mask[new_mask].float().mean().item()
                    metrics['pg_active_frac_new'] = 1.0 - metrics['pg_dead_frac_new']
                if old_mask.any():
                    metrics['pg_dead_frac_old'] = dead_mask[old_mask].float().mean().item()
                    metrics['pg_active_frac_old'] = 1.0 - metrics['pg_dead_frac_old']
                
                # 分桶统计（相对阈值）
                if new_mask_ratio.any():
                    metrics['pg_dead_frac_new_ratio'] = dead_mask[new_mask_ratio].float().mean().item()
                    metrics['pg_active_frac_new_ratio'] = 1.0 - metrics['pg_dead_frac_new_ratio']
                if old_mask_ratio.any():
                    metrics['pg_dead_frac_old_ratio'] = dead_mask[old_mask_ratio].float().mean().item()
                    metrics['pg_active_frac_old_ratio'] = 1.0 - metrics['pg_dead_frac_old_ratio']
                
                # U（贡献权重）for Hard Clip
                u = ratio_flat * (~dead_mask).float()
            else:
                # ==================== 4. Soft Clip 指标 ====================
                # Outside clip (按 ratio 定义)
                outside_clip = (ratio_flat < (1 - clip_eps)) | (ratio_flat > (1 + clip_eps))
                metrics['outside_clip_frac'] = outside_clip.float().mean().item()
                
                # 分桶统计
                if new_mask.any():
                    metrics['outside_clip_frac_new'] = outside_clip[new_mask].float().mean().item()
                if old_mask.any():
                    metrics['outside_clip_frac_old'] = outside_clip[old_mask].float().mean().item()
                
                # U（贡献权重）for Soft Clip - 需要根据 clip_mode 计算
                if self.clip_mode == "sapo":
                    # SAPO: gate(r) = (4/τ) * sigmoid(τ*(r-1))
                    tau_pos = 1.0
                    tau_neg = 2.0
                    ratio_min = 1e-6
                    ratio_max = 1e6
                    r = ratio_flat.clamp(ratio_min, ratio_max)
                    
                    tau = torch.where(adv_expanded > 0, 
                                     torch.full_like(ratio_flat, tau_pos),
                                     torch.full_like(ratio_flat, tau_neg))
                    x = tau * (r - 1.0)
                    p = torch.sigmoid(x)
                    w_sapo = 4.0 * p * (1.0 - p)
                    u = w_sapo * r
                    
                    # suppressed：权重被显著抑制（< 阈值）
                    w_threshold = 1e-3
                    suppressed_mask = w_sapo < w_threshold
                    metrics['suppressed_frac'] = suppressed_mask.float().mean().item()
                    
                    if new_mask.any():
                        metrics['suppressed_frac_new'] = suppressed_mask[new_mask].float().mean().item()
                    if old_mask.any():
                        metrics['suppressed_frac_old'] = suppressed_mask[old_mask].float().mean().item()
                        
                elif self.clip_mode == "gipo":
                    # GIPO: Log-Gauss soft clip
                    eps = 1e-9
                    r = ratio_flat.clamp_min(eps).detach()
                    w_gauss = torch.exp(-0.5 * (torch.log(r) / self.sigma) ** 2)
                    u = w_gauss * r
                    
                    # suppressed
                    w_threshold = 1e-3
                    suppressed_mask = w_gauss < w_threshold
                    metrics['suppressed_frac'] = suppressed_mask.float().mean().item()
                    
                    if new_mask.any():
                        metrics['suppressed_frac_new'] = suppressed_mask[new_mask].float().mean().item()
                    if old_mask.any():
                        metrics['suppressed_frac_old'] = suppressed_mask[old_mask].float().mean().item()
                else:
                    # 其他模式：直接使用 ratio
                    u = ratio_flat
            
            # ==================== 5. 贡献权重 U 统计 ====================
            metrics['u_mean'] = u.mean().item()
            metrics['u_p50'] = torch.median(u).item()
            metrics['u_p90'] = torch.quantile(u, 0.90).item()
            metrics['u_p99'] = torch.quantile(u, 0.99).item()
            metrics['u_max'] = u.max().item()
            
            # 分桶统计
            if new_mask.any():
                u_new = u[new_mask]
                metrics['u_mean_new'] = u_new.mean().item()
                metrics['u_p90_new'] = torch.quantile(u_new, 0.90).item()
            if old_mask.any():
                u_old = u[old_mask]
                metrics['u_mean_old'] = u_old.mean().item()
                metrics['u_p90_old'] = torch.quantile(u_old, 0.90).item()
            
            # ==================== 6. NearZero_U_Frac ====================
            near_zero_threshold = 1e-3
            near_zero_mask = u < near_zero_threshold
            metrics['nearzero_u_frac'] = near_zero_mask.float().mean().item()
            
            # 分桶统计（绝对阈值）
            if new_mask.any():
                metrics['nearzero_u_frac_new'] = near_zero_mask[new_mask].float().mean().item()
            if old_mask.any():
                metrics['nearzero_u_frac_old'] = near_zero_mask[old_mask].float().mean().item()
            
            # 分桶统计（相对阈值）
            if new_mask_ratio.any():
                metrics['nearzero_u_frac_new_ratio'] = near_zero_mask[new_mask_ratio].float().mean().item()
            if old_mask_ratio.any():
                metrics['nearzero_u_frac_old_ratio'] = near_zero_mask[old_mask_ratio].float().mean().item()
            
            # ==================== 7. 数据贡献占比（Contribution Share）====================
            u_sum_all = u.sum()
            
            # C1) 基于绝对阈值的贡献占比
            if old_mask.any() and u_sum_all > 0:
                metrics['contribution_old_u_share'] = (u[old_mask].sum() / u_sum_all).item()
            else:
                metrics['contribution_old_u_share'] = 0.0
            
            if new_mask.any() and u_sum_all > 0:
                metrics['contribution_new_u_share'] = (u[new_mask].sum() / u_sum_all).item()
            else:
                metrics['contribution_new_u_share'] = 0.0
            
            # C2) 基于相对阈值的贡献占比
            if old_mask_ratio.any() and u_sum_all > 0:
                metrics['contribution_old_u_share_ratio'] = (u[old_mask_ratio].sum() / u_sum_all).item()
            else:
                metrics['contribution_old_u_share_ratio'] = 0.0
            
            if new_mask_ratio.any() and u_sum_all > 0:
                metrics['contribution_new_u_share_ratio'] = (u[new_mask_ratio].sum() / u_sum_all).item()
            else:
                metrics['contribution_new_u_share_ratio'] = 0.0
            
            # C3) 基于 |u*A| 的梯度贡献占比
            u_grad_proxy = torch.abs(u * adv_expanded)
            u_grad_sum_all = u_grad_proxy.sum()
            
            # 绝对阈值版本
            if old_mask.any() and u_grad_sum_all > 0:
                metrics['contribution_old_u_share_abs_grad_proxy'] = (u_grad_proxy[old_mask].sum() / u_grad_sum_all).item()
            else:
                metrics['contribution_old_u_share_abs_grad_proxy'] = 0.0
            
            if new_mask.any() and u_grad_sum_all > 0:
                metrics['contribution_new_u_share_abs_grad_proxy'] = (u_grad_proxy[new_mask].sum() / u_grad_sum_all).item()
            else:
                metrics['contribution_new_u_share_abs_grad_proxy'] = 0.0
            
            # 相对阈值版本
            if old_mask_ratio.any() and u_grad_sum_all > 0:
                metrics['contribution_old_u_share_abs_grad_proxy_ratio'] = (u_grad_proxy[old_mask_ratio].sum() / u_grad_sum_all).item()
            else:
                metrics['contribution_old_u_share_abs_grad_proxy_ratio'] = 0.0
            
            if new_mask_ratio.any() and u_grad_sum_all > 0:
                metrics['contribution_new_u_share_abs_grad_proxy_ratio'] = (u_grad_proxy[new_mask_ratio].sum() / u_grad_sum_all).item()
            else:
                metrics['contribution_new_u_share_abs_grad_proxy_ratio'] = 0.0
            
            # ==================== 8. ESS（有效样本量）====================
            u_sum = u.sum()
            u_sq_sum = (u * u).sum()
            ess_eff = (u_sum * u_sum) / (u_sq_sum + 1e-12)
            metrics['ess_eff'] = ess_eff.item()
            metrics['ess_eff_norm'] = (ess_eff / u.numel()).item()
            
            # 分桶统计（绝对阈值）
            if new_mask.any():
                u_new = u[new_mask]
                u_new_sum = u_new.sum()
                u_new_sq_sum = (u_new * u_new).sum()
                ess_eff_new = (u_new_sum * u_new_sum) / (u_new_sq_sum + 1e-12)
                metrics['ess_eff_norm_new'] = (ess_eff_new / u_new.numel()).item()
                metrics['ess_eff_norm_new_abs'] = (ess_eff_new / u_new.numel()).item()
            if old_mask.any():
                u_old = u[old_mask]
                u_old_sum = u_old.sum()
                u_old_sq_sum = (u_old * u_old).sum()
                ess_eff_old = (u_old_sum * u_old_sum) / (u_old_sq_sum + 1e-12)
                metrics['ess_eff_norm_old'] = (ess_eff_old / u_old.numel()).item()
                metrics['ess_eff_norm_old_abs'] = (ess_eff_old / u_old.numel()).item()
            
            # 分桶统计（相对阈值）
            if new_mask_ratio.any():
                u_new_ratio = u[new_mask_ratio]
                u_new_ratio_sum = u_new_ratio.sum()
                u_new_ratio_sq_sum = (u_new_ratio * u_new_ratio).sum()
                ess_eff_new_ratio = (u_new_ratio_sum * u_new_ratio_sum) / (u_new_ratio_sq_sum + 1e-12)
                metrics['ess_eff_norm_new_ratio'] = (ess_eff_new_ratio / u_new_ratio.numel()).item()
            
            if old_mask_ratio.any():
                u_old_ratio = u[old_mask_ratio]
                u_old_ratio_sum = u_old_ratio.sum()
                u_old_ratio_sq_sum = (u_old_ratio * u_old_ratio).sum()
                ess_eff_old_ratio = (u_old_ratio_sum * u_old_ratio_sum) / (u_old_ratio_sq_sum + 1e-12)
                metrics['ess_eff_norm_old_ratio'] = (ess_eff_old_ratio / u_old_ratio.numel()).item()
            
            return metrics

    async def _process_trajectories(self, trajectories: List[Trajectory]) -> Tuple[List, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """处理轨迹，根据 recompute_value 决定是否重新计算 value (全GPU版本)
        
        Args:
            trajectories: 采样的轨迹列表
            
        Returns:
            obs_list: 所有观测列表
            action_tokens: [N, NUM_ACTIONS_CHUNK, ACTION_DIM] GPU tensor
            advantages: [N,] GPU tensor
            behaviour_logits: [N, NUM_ACTIONS_CHUNK, ACTION_DIM, VOCAB_SIZE] GPU tensor
            value_targets: [N,] GPU tensor
            policy_versions: [N,] GPU tensor
            insert_steps: [N,] GPU tensor
        """
        # 收集所有 obs
        all_obs = []
        for traj in trajectories:
            all_obs.extend(traj.obs_list)
        
        device = next(self.model.parameters()).device
        
        if self.recompute_value:
            # 重新前向计算 value - 分批次处理，保持在 GPU
            num_obs = len(all_obs)
            all_values = []
            with torch.no_grad():
                for start_idx in range(0, num_obs, self.train_batch_size):
                    end_idx = min(start_idx + self.train_batch_size, num_obs)
                    obs_batch = all_obs[start_idx:end_idx]
                    inputs_batch = self.base_model.prepare_inputs_batch(obs_batch)
                    _, batch_values = self.model.forward(inputs_batch)
                    all_values.append(batch_values.float())
            values = torch.cat(all_values, dim=0)
        else:
            # 使用老 value，转为 GPU tensor
            values = torch.from_numpy(np.concatenate([traj.old_values for traj in trajectories])).to(device)
        
        # 按轨迹分割，计算 GAE
        all_action_tokens = []
        all_advantages = []
        all_behaviour_logits = []
        all_value_targets = []
        all_policy_versions = []
        all_insert_steps = []
        
        offset = 0
        for traj in trajectories:
            T = traj.num_steps
            traj_values = values[offset:offset + T]
            offset += T
            
            # 将 rewards 转为 GPU tensor
            traj_rewards = torch.from_numpy(traj.rewards).to(device)
            
            advs, rets = self._compute_gae(
                rewards=traj_rewards,
                values=traj_values,
                bootstrap_value=traj.bootstrap_value,
                is_terminal=traj.is_terminal
            )
            
            # 收集数据
            all_action_tokens.append(torch.from_numpy(traj.action_tokens).to(device))
            all_advantages.append(advs)
            all_behaviour_logits.append(torch.from_numpy(traj.behaviour_logits).to(device))
            all_value_targets.append(rets)
            # 每个 step 都有自己的 policy_version 和 insert_step
            all_policy_versions.append(torch.from_numpy(traj.policy_versions).to(device))
            all_insert_steps.append(torch.from_numpy(traj.insert_steps).to(device))
        
        # 拼接所有数据
        action_tokens = torch.cat(all_action_tokens, dim=0)
        advantages = torch.cat(all_advantages, dim=0)
        behaviour_logits = torch.cat(all_behaviour_logits, dim=0)
        value_targets = torch.cat(all_value_targets, dim=0)
        policy_versions = torch.cat(all_policy_versions, dim=0)
        insert_steps = torch.cat(all_insert_steps, dim=0)
        
        return all_obs, action_tokens, advantages, behaviour_logits, value_targets, policy_versions, insert_steps

    async def _data_fetching_loop(self):
        print(f"Trainer {self.rank}: 后台数据准备循环已启动 (超级批次大小: {self.super_batch_size}, recompute_value={self.recompute_value})。")
        while True:
            try:
                if self.next_ready_batch is not None:
                    await asyncio.sleep(0.1)
                    continue

                # 等待足够的数据（按总步数计算）
                while await self.replay_buffer.total_steps.remote() < self.super_batch_size:
                    total_steps = await self.replay_buffer.total_steps.remote()
                    print(f"Trainer {self.rank} (BG): 等待 ReplayBuffer 填充至 {self.super_batch_size} 步... (当前: {total_steps})")
                    await asyncio.sleep(3)

                t_sample_start = time.time()
                # 采样轨迹
                trajectories = await self.replay_buffer.sample_trajectories.remote(self.super_batch_size)
                sample_time = time.time() - t_sample_start

                t_prep_start = time.time()
                # 处理轨迹（计算 GAE，可能重新计算 value）- 全部在 GPU 上
                obs_list, action_tokens, advantages, behaviour_logits, value_targets, policy_versions, insert_steps = await self._process_trajectories(trajectories)
                
                # 打乱样本顺序 - 在 GPU 上进行
                num_samples = len(obs_list)
                indices = torch.randperm(num_samples, device=action_tokens.device)
                
                # 根据打乱的索引重排数据
                obs_list = [obs_list[i] for i in indices.cpu().tolist()]
                act_token_t = action_tokens[indices].long()
                adv_t = advantages[indices]
                logits_old_t = behaviour_logits[indices]
                v_targ_t = value_targets[indices]
                policy_ver_t = policy_versions[indices]
                insert_step_t = insert_steps[indices]
                
                # 准备 batch
                inputs_batch = self.base_model.prepare_inputs_batch(obs_list)
                
                prep_time = time.time() - t_prep_start

                self.next_ready_batch = {
                    'inputs_batch': inputs_batch,
                    'act_token': act_token_t,
                    'advantage': adv_t,
                    'logits_old': logits_old_t,
                    'value_target': v_targ_t,
                    'policy_version': policy_ver_t,
                    'insert_step': insert_step_t,
                    'sample_time': sample_time,
                    'prep_time': prep_time
                }

            except Exception as e:
                import traceback
                print(f"Trainer {self.rank}: 数据采样失败: {e}。将在3秒后重试。")
                traceback.print_exc()
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
        policy_ver_t = current_batch['policy_version']
        insert_step_t = current_batch['insert_step']
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
        epoch_explained_variance = []
        epoch_grad_norms = []
        diagnostic_metrics = {}  # 只在第一个 mini-batch 计算一次
        
        num_updates_in_epoch = self.super_batch_size // self.train_batch_size
        t_policy_train_start = time.time()
        
        for i in range(num_updates_in_epoch):
            start = i * self.train_batch_size; end = start + self.train_batch_size
            mini_inputs = {k: v[start:end] for k, v in inputs_batch.items()}
            
            mini_act_token = act_token_t[start:end]
            mini_adv = adv_t[start:end]
            mini_logits_old = logits_old_t[start:end]
            mini_v_targ = v_targ_t[start:end]
            mini_policy_ver = policy_ver_t[start:end]
            mini_insert_step = insert_step_t[start:end]
            
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
            
            # Explained Variance (在第一个 mini-batch 计算)
            if i == 0:
                with torch.no_grad():
                    value_pred = value.squeeze(-1) if value.dim() > 1 else value
                    target = mini_v_targ
                    var_target = torch.var(target, unbiased=False)
                    if var_target < 1e-12:
                        ev = 0.0
                    else:
                        ev = 1.0 - torch.var(target - value_pred, unbiased=False) / (var_target + 1e-12)
                    epoch_explained_variance.append(float(ev))
            
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
                logp = dist.log_prob(mini_act_token)

                with torch.no_grad():
                    dist_old = torch.distributions.Categorical(logits=mini_logits_old)
                    logp_old = dist_old.log_prob(mini_act_token)

                kl_div_tensor = kl.kl_divergence(dist_old, dist)
                kl_div = torch.mean(kl_div_tensor).item() # 作为指标
                kl_loss = self.kl_coef * torch.mean(kl_div_tensor) # 作为损失
                ratio = torch.exp(logp - logp_old)
                
                # ========== 计算诊断指标（只在第一个 mini-batch 时计算）==========
                if i == 0 and self.global_step >= self.policy_train_start_step:
                    diagnostic_metrics = self._compute_diagnostic_metrics(
                        ratio=ratio,
                        advantage=normalized_adv,
                        policy_version=mini_policy_ver,
                        insert_step=mini_insert_step,
                        current_policy_version=self.policy_version,
                        current_step=self.global_step,
                        clip_eps=self.clip_eps,
                    )
                
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
            
            # Gradient Norm (在第一个 mini-batch 计算)
            if i == 0 and self.global_step >= self.policy_train_start_step:
                try:
                    grad_norm = self.model.get_global_grad_norm()
                    epoch_grad_norms.append(float(grad_norm))
                except Exception:
                    pass
            
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

        # 计算 Explained Variance 和 Grad Norm 的平均值
        avg_explained_variance = np.mean(epoch_explained_variance) if epoch_explained_variance else 0.0
        avg_grad_norm = np.mean(epoch_grad_norms) if epoch_grad_norms else 0.0
        
        perf_metrics = {
            "policy_sample_time": policy_sample_time,
            "policy_prep_time": policy_prep_time,
            "policy_train_time": time.time() - t_policy_train_start,
            "explained_variance": avg_explained_variance,
            "grad_norm": avg_grad_norm,
        }
        # 合并诊断指标到 perf_metrics
        perf_metrics.update(diagnostic_metrics)
        
        # 递增策略版本
        self.policy_version += 1

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
        num_images_in_input=args.num_images_in_input,
        # zzq 1124 开启 proprio 
        use_proprio=args.use_proprio, # Note: ActorCritic in discrete model can handle this
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
    torch_dtype = torch.bfloat16 if args.use_bf16 else torch.float32
    benchmark = args.benchmark
    
    if not os.path.exists(args.pretrained_checkpoint):
        print(f"错误: OpenVLA checkpoint 路径 '{args.pretrained_checkpoint}' 不存在。请更新 PRETRAINED_CHECKPOINT。")
        return

    os.environ["RAY_DEDUP_LOGS"] = "0"
    object_store_memory_bytes = int(args.object_store_memory_gb * 1024 * 1024 * 1024)
    print(f"正在初始化 Ray，并为对象存储分配 {args.object_store_memory_gb} GB 内存...")
    ray.init(
        ignore_reinit_error=True, 
        _temp_dir='/dev/shm',
        object_store_memory=object_store_memory_bytes
    )

    log_dir = f"runs/Libero/{args.benchmark}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.exp_name}_{args.clip_mode}"
    writer = SummaryWriter(log_dir)

    # 保存命令行参数到log_dir中的json文件
    args_file = os.path.join(log_dir, "args.json")
    with open(args_file, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    print(f"命令行参数已保存到: {args_file}")
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
            clip_mode=args.clip_mode, recompute_value=args.recompute_value, sigma=args.sigma
        )
        for i in range(args.num_trainer_gpus)
    ]
    # 解析 task_ids
    task_ids = [int(tid.strip()) for tid in args.task_ids.split(',')]
    print(f"\n使用 task IDs: {task_ids}\n")
    
    inference_pool = [InferenceActor.remote(actor_id=i, cfg=cfg, stats_actor=stats_actor, torch_dtype=torch_dtype, inference_batch=args.inference_batch, inference_timeout_ms=args.inference_timeout_ms) for i in range(args.num_inference_actors)]
    rollout_workers = [
        RolloutWorkerActor.remote(
            inference_pool[i % args.num_inference_actors],
            replay_buffers[i % args.num_trainer_gpus], i, stats_actor, cfg, benchmark,
            args.reward_scale, torch_dtype, args.rollout_local_buf, task_ids
        ) for i in range(args.num_rollout_workers)
    ]
    eval_workers = [
        EvaluationWorkerActor.remote(
            inference_pool[i % args.num_inference_actors], f"eval_{i}", stats_actor, cfg, benchmark, torch_dtype, task_ids, args.eval_deterministic
        ) for i in range(args.num_eval_workers)
    ]
    print(f"已创建 {args.num_rollout_workers} 个 Rollout workers 和 {args.num_eval_workers} 个 Evaluation workers。")

    # zzq 1125 通信组，使用find_free_port
    train_group_port = find_free_port()

    broadcast_group_port = find_free_port()
    while broadcast_group_port == train_group_port:
        broadcast_group_port = find_free_port()

    print(f"\n--- 步骤 2: 预先建立共享广播组 ({args.broadcast_group_name}) ---")
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

    print("\n--- 步骤 3: 建立独立的 DeepSpeed 训练组 ---")
    trainer_master_addr = ray.get(trainer_group[0].get_node_ip.remote())
    train_setup_tasks = [actor.setup_deepspeed_group.remote(trainer_master_addr, train_group_port) for actor in trainer_group]
    ray.get(train_setup_tasks)
    print("DeepSpeed 训练组建立完成。")

    print("\n--- 步骤 3.1: NCCL 广播组小张量通信测试 ---")
    for dtype_name in ("float32", "bfloat16", "int64"):
        sanity_tasks = [
            actor.broadcast_sanity_check.remote(args.broadcast_group_name, dtype_name, 8)
            for actor in broadcast_participants
        ]
        sanity_results = ray.get(sanity_tasks)
        print(f"广播组 sanity check 通过: dtype={dtype_name}, results={sanity_results}")

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

    # ================================================================
    # Resume 逻辑：恢复训练状态
    # ================================================================
    start_global_step = 0
    if hasattr(args, 'resume_step') and args.resume_step is not None:
        print(f"\n{'='*80}")
        print(f"Resume 模式：从步数 {args.resume_step} 恢复训练")
        print(f"{'='*80}\n")
        
        # 1. 恢复 Trainer 状态（包含模型权重、优化器、随机数状态）
        print("--- Resume 步骤 1: 恢复 Trainer 状态 ---")
        load_tasks = [trainer.load_checkpoint.remote(args.resume_from, args.resume_step) 
                      for trainer in trainer_group]
        ray.get(load_tasks)
        print("✓ Trainer 状态恢复完成\n")
        
        # 2. 广播恢复的权重到推理器
        print("--- Resume 步骤 2: 同步权重到推理器 ---")
        broadcast_task = trainer_group[0].broadcast_weights.remote(args.broadcast_group_name)
        receive_tasks = [inf.receive_and_update_weights.remote(args.broadcast_group_name) 
                         for inf in inference_pool]
        ray.get([broadcast_task] + receive_tasks)
        print("✓ 权重同步完成\n")
        
        # 3. 恢复 ReplayBuffer
        print("--- Resume 步骤 3: 恢复经验回放缓冲区 ---")
        load_buffer_tasks = []
        for i, rb in enumerate(replay_buffers):
            buffer_path = os.path.join(args.resume_from, f'replay_buffer_{i}_step_{args.resume_step}.pkl')
            load_buffer_tasks.append(rb.load_buffer.remote(buffer_path))
        ray.get(load_buffer_tasks)
        print("✓ 经验回放缓冲区恢复完成\n")
        
        # 4. 恢复 Stats
        print("--- Resume 步骤 4: 恢复统计信息 ---")
        stats_path = os.path.join(args.resume_from, f'stats_step_{args.resume_step}.pkl')
        ray.get(stats_actor.load_stats.remote(stats_path))
        print("✓ 统计信息恢复完成\n")
        
        start_global_step = args.resume_step
        ray.get([trainer.start_data_fetching_loop.remote() for trainer in trainer_group])
        print(f"{'='*80}")
        print(f"Resume 完成！将从步数 {start_global_step} 继续训练到 {args.train_iters}")
        print(f"{'='*80}\n")
    else:
        print("\n--- 步骤 5: 等待远程经验池填充初始数据 ---")
        ray.get([trainer.start_data_fetching_loop.remote() for trainer in trainer_group])
        min_buffer_steps_for_start = args.train_batch_size * args.accumulation_steps
        while not all(steps >= min_buffer_steps_for_start for steps in ray.get([rb.total_steps.remote() for rb in replay_buffers])):
            total_steps_list = ray.get([rb.total_steps.remote() for rb in replay_buffers])
            traj_counts = ray.get([rb.size.remote() for rb in replay_buffers])
            print(f"等待所有经验池填充初始数据 (目标步数: {min_buffer_steps_for_start})... (当前步数: {total_steps_list}, 轨迹数: {traj_counts})")
            time.sleep(5)
        print("远程经验池已准备好，训练器将按需获取数据。")

    print("\n--- 步骤 6: 开始主训练与同步循环 ---")
    start_time = time.time()
    last_log_time = time.time()
    last_log_global_step = start_global_step
    global_step = start_global_step
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
            avg_step_reward = global_stats["avg_step_reward"]
            eval_avg_return = eval_stats["avg_return"]
            eval_avg_ep_len = eval_stats["avg_ep_len"]
            eval_total_episodes = eval_stats["total_episodes_processed"]
            eval_env_steps = eval_stats["total_env_steps"]
            eval_avg_step_time = eval_stats["avg_step_time"]
            eval_avg_step_reward = eval_stats["avg_step_reward"]

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
            
            # ========== 新增核心指标：Explained Variance 和 Gradient Norm ==========
            if 'explained_variance' in perf_metrics_list[0]:
                writer.add_scalar('Metrics/ExplainedVariance', perf_metrics_list[0]['explained_variance'], global_step)
            if 'grad_norm' in perf_metrics_list[0]:
                writer.add_scalar('Metrics/Grad_Norm', perf_metrics_list[0]['grad_norm'], global_step)
            
            # ========== 新增诊断指标（Staleness、Ratio、ESS 等）==========
            # 1. Staleness
            if 'staleness_ver_mean' in perf_metrics_list[0]:
                writer.add_scalar('Staleness/Version_Mean', perf_metrics_list[0]['staleness_ver_mean'], global_step)
                writer.add_scalar('Staleness/Version_P95', perf_metrics_list[0]['staleness_ver_p95'], global_step)
                writer.add_scalar('Staleness/Age_Steps_Mean', perf_metrics_list[0]['age_steps_mean'], global_step)
                writer.add_scalar('Staleness/Age_Steps_P95', perf_metrics_list[0]['age_steps_p95'], global_step)
                if 'age_steps_max' in perf_metrics_list[0]:
                    writer.add_scalar('Staleness/Age_Steps_Max', perf_metrics_list[0]['age_steps_max'], global_step)
                
                # A. 分桶组成（绝对阈值）
                if 'staleness_old_frac_abs' in perf_metrics_list[0]:
                    writer.add_scalar('Staleness/OldFrac_Abs', perf_metrics_list[0]['staleness_old_frac_abs'], global_step)
                    writer.add_scalar('Staleness/NewFrac_Abs', perf_metrics_list[0]['staleness_new_frac_abs'], global_step)
                if 'staleness_old_gap_mean_abs' in perf_metrics_list[0]:
                    writer.add_scalar('Staleness/OldGapMean_Abs', perf_metrics_list[0]['staleness_old_gap_mean_abs'], global_step)
                    writer.add_scalar('Staleness/OldGapP95_Abs', perf_metrics_list[0]['staleness_old_gap_p95_abs'], global_step)
                
                # B. 相对陈旧度
                if 'staleness_ratio_mean' in perf_metrics_list[0]:
                    writer.add_scalar('Staleness/RatioMean', perf_metrics_list[0]['staleness_ratio_mean'], global_step)
                    writer.add_scalar('Staleness/RatioP95', perf_metrics_list[0]['staleness_ratio_p95'], global_step)
                if 'staleness_old_frac_ratio' in perf_metrics_list[0]:
                    writer.add_scalar('Staleness/OldFrac_Ratio', perf_metrics_list[0]['staleness_old_frac_ratio'], global_step)
                    writer.add_scalar('Staleness/NewFrac_Ratio', perf_metrics_list[0]['staleness_new_frac_ratio'], global_step)
            
            # 2. Ratio / log-ratio 分布
            if 'rho_mean' in perf_metrics_list[0]:
                writer.add_scalar('Ratio/Rho_Mean', perf_metrics_list[0]['rho_mean'], global_step)
                writer.add_scalar('Ratio/Rho_P50', perf_metrics_list[0]['rho_p50'], global_step)
                writer.add_scalar('Ratio/Rho_P90', perf_metrics_list[0]['rho_p90'], global_step)
                writer.add_scalar('Ratio/Rho_P99', perf_metrics_list[0]['rho_p99'], global_step)
                writer.add_scalar('Ratio/Rho_Max', perf_metrics_list[0]['rho_max'], global_step)
                writer.add_scalar('Ratio/LogRho_Mean', perf_metrics_list[0]['logrho_mean'], global_step)
                writer.add_scalar('Ratio/AbsLogRho_P95', perf_metrics_list[0]['abs_logrho_p95'], global_step)
            
            # 3. Hard Clip 指标（PPO）
            if 'pg_active_frac' in perf_metrics_list[0]:
                writer.add_scalar('Hard/PG_Active_Frac', perf_metrics_list[0]['pg_active_frac'], global_step)
                writer.add_scalar('Hard/PG_Dead_Frac', perf_metrics_list[0]['pg_dead_frac'], global_step)
                if 'pg_active_frac_new' in perf_metrics_list[0]:
                    writer.add_scalar('Hard/PG_Active_Frac_New', perf_metrics_list[0]['pg_active_frac_new'], global_step)
                    writer.add_scalar('Hard/PG_Dead_Frac_New', perf_metrics_list[0]['pg_dead_frac_new'], global_step)
                if 'pg_active_frac_old' in perf_metrics_list[0]:
                    writer.add_scalar('Hard/PG_Active_Frac_Old', perf_metrics_list[0]['pg_active_frac_old'], global_step)
                    writer.add_scalar('Hard/PG_Dead_Frac_Old', perf_metrics_list[0]['pg_dead_frac_old'], global_step)
                # 相对阈值分桶
                if 'pg_active_frac_new_ratio' in perf_metrics_list[0]:
                    writer.add_scalar('Hard/PG_Active_Frac_New_Ratio', perf_metrics_list[0]['pg_active_frac_new_ratio'], global_step)
                    writer.add_scalar('Hard/PG_Dead_Frac_New_Ratio', perf_metrics_list[0]['pg_dead_frac_new_ratio'], global_step)
                if 'pg_active_frac_old_ratio' in perf_metrics_list[0]:
                    writer.add_scalar('Hard/PG_Active_Frac_Old_Ratio', perf_metrics_list[0]['pg_active_frac_old_ratio'], global_step)
                    writer.add_scalar('Hard/PG_Dead_Frac_Old_Ratio', perf_metrics_list[0]['pg_dead_frac_old_ratio'], global_step)
            
            # 4. Soft Clip 指标
            if 'outside_clip_frac' in perf_metrics_list[0]:
                writer.add_scalar('Soft/Outside_Clip_Frac', perf_metrics_list[0]['outside_clip_frac'], global_step)
                if 'outside_clip_frac_new' in perf_metrics_list[0]:
                    writer.add_scalar('Soft/Outside_Clip_Frac_New', perf_metrics_list[0]['outside_clip_frac_new'], global_step)
                if 'outside_clip_frac_old' in perf_metrics_list[0]:
                    writer.add_scalar('Soft/Outside_Clip_Frac_Old', perf_metrics_list[0]['outside_clip_frac_old'], global_step)
            
            if 'suppressed_frac' in perf_metrics_list[0]:
                writer.add_scalar('Soft/Suppressed_Frac', perf_metrics_list[0]['suppressed_frac'], global_step)
                if 'suppressed_frac_new' in perf_metrics_list[0]:
                    writer.add_scalar('Soft/Suppressed_Frac_New', perf_metrics_list[0]['suppressed_frac_new'], global_step)
                if 'suppressed_frac_old' in perf_metrics_list[0]:
                    writer.add_scalar('Soft/Suppressed_Frac_Old', perf_metrics_list[0]['suppressed_frac_old'], global_step)
            
            # 5. 贡献权重 U
            if 'u_mean' in perf_metrics_list[0]:
                writer.add_scalar('Contribution/U_Mean', perf_metrics_list[0]['u_mean'], global_step)
                writer.add_scalar('Contribution/U_P50', perf_metrics_list[0]['u_p50'], global_step)
                writer.add_scalar('Contribution/U_P90', perf_metrics_list[0]['u_p90'], global_step)
                writer.add_scalar('Contribution/U_P99', perf_metrics_list[0]['u_p99'], global_step)
                writer.add_scalar('Contribution/U_Max', perf_metrics_list[0]['u_max'], global_step)
                
                # 分桶统计（绝对阈值）
                if 'u_mean_new' in perf_metrics_list[0]:
                    writer.add_scalar('Contribution/U_Mean_New', perf_metrics_list[0]['u_mean_new'], global_step)
                    writer.add_scalar('Contribution/U_P90_New', perf_metrics_list[0]['u_p90_new'], global_step)
                if 'u_mean_old' in perf_metrics_list[0]:
                    writer.add_scalar('Contribution/U_Mean_Old', perf_metrics_list[0]['u_mean_old'], global_step)
                    writer.add_scalar('Contribution/U_P90_Old', perf_metrics_list[0]['u_p90_old'], global_step)
            
            # 6. 数据贡献占比
            if 'contribution_old_u_share' in perf_metrics_list[0]:
                writer.add_scalar('Contribution/OldUShare', perf_metrics_list[0]['contribution_old_u_share'], global_step)
                writer.add_scalar('Contribution/NewUShare', perf_metrics_list[0]['contribution_new_u_share'], global_step)
            if 'contribution_old_u_share_ratio' in perf_metrics_list[0]:
                writer.add_scalar('Contribution/OldUShare_Ratio', perf_metrics_list[0]['contribution_old_u_share_ratio'], global_step)
                writer.add_scalar('Contribution/NewUShare_Ratio', perf_metrics_list[0]['contribution_new_u_share_ratio'], global_step)
            
            # 基于 |u*A| 的梯度贡献占比
            if 'contribution_old_u_share_abs_grad_proxy' in perf_metrics_list[0]:
                writer.add_scalar('Contribution/OldUShare_AbsGradProxy', perf_metrics_list[0]['contribution_old_u_share_abs_grad_proxy'], global_step)
                writer.add_scalar('Contribution/NewUShare_AbsGradProxy', perf_metrics_list[0]['contribution_new_u_share_abs_grad_proxy'], global_step)
            if 'contribution_old_u_share_abs_grad_proxy_ratio' in perf_metrics_list[0]:
                writer.add_scalar('Contribution/OldUShare_AbsGradProxy_Ratio', perf_metrics_list[0]['contribution_old_u_share_abs_grad_proxy_ratio'], global_step)
                writer.add_scalar('Contribution/NewUShare_AbsGradProxy_Ratio', perf_metrics_list[0]['contribution_new_u_share_abs_grad_proxy_ratio'], global_step)
            
            # 7. NearZero_U_Frac
            if 'nearzero_u_frac' in perf_metrics_list[0]:
                writer.add_scalar('Contribution/NearZero_U_Frac', perf_metrics_list[0]['nearzero_u_frac'], global_step)
                # 分桶统计（绝对阈值）
                if 'nearzero_u_frac_new' in perf_metrics_list[0]:
                    writer.add_scalar('Contribution/NearZero_U_Frac_New', perf_metrics_list[0]['nearzero_u_frac_new'], global_step)
                if 'nearzero_u_frac_old' in perf_metrics_list[0]:
                    writer.add_scalar('Contribution/NearZero_U_Frac_Old', perf_metrics_list[0]['nearzero_u_frac_old'], global_step)
                # 分桶统计（相对阈值）
                if 'nearzero_u_frac_new_ratio' in perf_metrics_list[0]:
                    writer.add_scalar('Contribution/NearZero_U_Frac_New_Ratio', perf_metrics_list[0]['nearzero_u_frac_new_ratio'], global_step)
                if 'nearzero_u_frac_old_ratio' in perf_metrics_list[0]:
                    writer.add_scalar('Contribution/NearZero_U_Frac_Old_Ratio', perf_metrics_list[0]['nearzero_u_frac_old_ratio'], global_step)
            
            # 8. ESS（有效样本量）
            if 'ess_eff' in perf_metrics_list[0]:
                writer.add_scalar('ESS/ESS_Eff', perf_metrics_list[0]['ess_eff'], global_step)
                writer.add_scalar('ESS/ESS_Eff_Norm', perf_metrics_list[0]['ess_eff_norm'], global_step)
                
                # 分桶统计（绝对阈值）
                if 'ess_eff_norm_new' in perf_metrics_list[0]:
                    writer.add_scalar('ESS/ESS_Eff_Norm_New', perf_metrics_list[0]['ess_eff_norm_new'], global_step)
                if 'ess_eff_norm_old' in perf_metrics_list[0]:
                    writer.add_scalar('ESS/ESS_Eff_Norm_Old', perf_metrics_list[0]['ess_eff_norm_old'], global_step)
                if 'ess_eff_norm_new_abs' in perf_metrics_list[0]:
                    writer.add_scalar('ESS/ESS_Eff_Norm_New_Abs', perf_metrics_list[0]['ess_eff_norm_new_abs'], global_step)
                if 'ess_eff_norm_old_abs' in perf_metrics_list[0]:
                    writer.add_scalar('ESS/ESS_Eff_Norm_Old_Abs', perf_metrics_list[0]['ess_eff_norm_old_abs'], global_step)
                
                # 分桶统计（相对阈值）
                if 'ess_eff_norm_new_ratio' in perf_metrics_list[0]:
                    writer.add_scalar('ESS/ESS_Eff_Norm_New_Ratio', perf_metrics_list[0]['ess_eff_norm_new_ratio'], global_step)
                if 'ess_eff_norm_old_ratio' in perf_metrics_list[0]:
                    writer.add_scalar('ESS/ESS_Eff_Norm_Old_Ratio', perf_metrics_list[0]['ess_eff_norm_old_ratio'], global_step)
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
            writer.add_scalar('Rollout/_Global/Average_Step_Reward', avg_step_reward, global_step)
            writer.add_scalar('Eval/_Global/Average_Return', eval_avg_return, global_step)
            writer.add_scalar('Eval/_Global/Average_Episode_Length', eval_avg_ep_len, global_step)
            writer.add_scalar('Eval/_Global/Average_Step_Reward', eval_avg_step_reward, global_step)
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
