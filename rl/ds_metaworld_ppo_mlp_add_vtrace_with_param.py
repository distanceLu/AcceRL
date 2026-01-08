import os
# os.environ["MUJOCO_GL"] = "osmesa"           # 强制软件渲染
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"   # 保险起见，给 PyOpenGL 也指明
# 设置临时文件目录，避免磁盘I/O瓶颈
os.environ["TMPDIR"] = "/dev/shm"
# 为了让 Ray 能看到所有可用的 GPU，我们在脚本开头设置。
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
# NCCL/编译相关的兜底设置，尽量避免初始化卡死
os.environ["NCCL_SOCKET_IFNAME"] = os.environ.get("NCCL_SOCKET_IFNAME", "eth0")
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/dev/shm/torch_ext")
os.environ.setdefault("DS_BUILD_FUSED_ADAM", "0")
# 防止 transformers 库的 tokenizer 并行化警告
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import random
import asyncio
import argparse
from collections import deque, defaultdict
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass
import math
import json

import numpy as np

import ray
import torch
import torch.distributions
from torch.distributions import kl
import deepspeed
import torch.distributed as distributed
from torch.utils.tensorboard import SummaryWriter
import swanlab

try:
    import yaml  # PyYAML 解析配置
except ImportError:
    yaml = None

# MetaWorld 和 MLP Actor-Critic 组件
from rl.metaworld_env import MetaWorldWrapperDiscrete
from rl.policies.mlp_actor_critic import MLPActorCriticDiscrete
# 训练/推理通信（保持接口不变）
from ds_com import TrainerActorCom, InferenceActorCom
from rl.com_utils import find_free_port

# ================================================================
# 0. 超参数与配置 (已修改为单任务学习，并对齐 ds_meta 关键超参便于对比)
# ================================================================
# # MetaWorld 单任务设置
# METAWORLD_TASKS = ["reach-v3"]  # 单任务学习：只使用 reach-v3
# BENCHMARK = "MetaWorld_reach_v3"

# 分布式系统参数（对齐 ds_meta：2 个 Trainer GPU、更多 rollout workers 等）
NUM_TRAINER_GPUS = 1
NUM_INFERENCE_ACTORS = 1
NUM_ROLLOUT_WORKERS = 1  # 与 ds_meta 相同，便于对比采样效率
NUM_EVAL_WORKERS = 5
ROLLOUT_LOCAL_BUF = 64
INFERENCE_BATCH = 32       # 对齐 ds_meta 的推理批大小
INFERENCE_TIMEOUT_MS = 300
REPLAY_CAPACITY = 50_000
TRAIN_BATCH_SIZE = 512
ACCUMULATION_STEPS = 4
TRAIN_ITERS = 100000       # 对齐 ds_meta 的训练总步数

# Checkpoint
CKPT_DIR = f"/cpfs01/qianfy_workspace/openvla_oft_rl/models/finetune_rl_metaworld"
CKPT_EVERY_STEPS = 2000   # 每 N 个训练步保存一次

# PPO（对齐 ds_meta 的大部分超参数）
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01   # 与 ds_meta 一致，保持相似的熵正则强度
KL_COEF = 0.1

# 裁剪默认配置（可扩展）
DEFAULT_CLIP_CONFIG = {
    "clip": {},
}

# 奖励缩放
REWARD_SCALE = 0.01

# ================================================================
# 学习率调度参数（与 ds_meta 的 LR & warmup 大致对齐）
# ================================================================
VALUE_LR = 3e-5   # 对齐 ds_meta 的学习率量级
POLICY_LR = 3e-5  # 保持策略与价值网络相同量级
VALUE_WARMUP_STEPS = 1000
POLICY_WARMUP_STEPS = 1000
POLICY_TRAIN_START_STEP = 500 # 策略网络从第500个 *更新步* 开始训练

# 日志（滑动窗口长度对齐 ds_meta）
MOVING_AVG_WINDOW = 100
LOG_INTERVAL_SECONDS = 10

# 通信组
BROADCAST_GROUP_NAME = "trainer_to_inference_broadcast"

# MLP Actor-Critic 配置
STATE_DIM = 39
ACTION_DIM = 4
N_ACTION_BINS = 256
TORCH_DTYPE = torch.float32  # MLP 模型使用 float32

# ================================================================
# 随机种子配置 - 新增部分
# ================================================================
SEED = 42  # 全局固定随机种子，便于复现

def set_seed(seed: int):
    """设置所有随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    # 保证可复现性（可能会降低性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args(args: Optional[List[str]] = None):
    """
    解析命令行参数，允许动态指定关键超参与设备配置。
    仅暴露与需求对齐的选项：benchmark、CUDA_VISIBLE_DEVICES、训练/采样并行规模、batch size、随机种子与策略裁剪模式。
    """
    parser = argparse.ArgumentParser(description="MetaWorld MLP DeepSpeed PPO 训练脚本")
    parser.add_argument("--task_name", type=str, required=True, help="MetaWorld 任务名称")
    parser.add_argument("--cuda-visible-devices", dest="cuda_visible_devices", type=str,
                        default=os.environ.get("CUDA_VISIBLE_DEVICES", "6,7"),
                        help="CUDA_VISIBLE_DEVICES 环境变量设置")
    parser.add_argument("--num-trainer-gpus", dest="num_trainer_gpus", type=int,
                        default=NUM_TRAINER_GPUS, help="TrainerActor 的 GPU 数量")
    parser.add_argument("--num-rollout-workers", dest="num_rollout_workers", type=int,
                        default=NUM_ROLLOUT_WORKERS, help="Rollout worker 数量")
    parser.add_argument("--num-eval-workers", dest="num_eval_workers", type=int,
                        default=NUM_EVAL_WORKERS, help="Evaluation worker 数量")
    parser.add_argument("--train-batch-size", dest="train_batch_size", type=int,
                        default=TRAIN_BATCH_SIZE, help="训练微批大小")
    parser.add_argument("--seed", dest="seed", type=int, default=SEED, help="全局随机种子")
    parser.add_argument("--clip-mode", dest="clip_mode", type=str, default="clip",
                        help=f"策略裁剪模式（支持 clip/soft_clip，或配置文件中的自定义模式）")
    parser.add_argument("--clip-config", dest="clip_config", type=str, default=None,
                        help="裁剪配置文件路径（YAML/JSON），按 clip_mode 选择对应超参")
    return parser.parse_args(args=args)


def load_clip_config(config_path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """
    加载裁剪模式配置，支持 YAML/JSON。
    返回格式: {clip_mode: {hyperparam: value, ...}, ...}
    """
    default_config = DEFAULT_CLIP_CONFIG
    if config_path is None:
        return default_config

    try:
        with open(config_path, "r") as f:
            if yaml is not None:
                cfg = yaml.safe_load(f)
            else:
                cfg = json.load(f)
        if not isinstance(cfg, dict):
            print(f"[Warn] clip_config {config_path} 内容不是 dict，使用默认配置。")
            return default_config
        # 合并默认，允许覆盖或新增模式
        merged = {**default_config, **cfg}
        return merged
    except Exception as e:
        print(f"[Warn] 加载 clip_config 失败: {e}，回退默认配置。")
        return default_config


def select_clip_params(clip_mode: str, clip_config: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    基于 clip_mode 选择对应超参，返回该模式的完整配置 dict。
    后续新增模式时，只需在配置里添加对应键与超参。
    """
    params = clip_config.get(clip_mode, {})
    if clip_mode not in clip_config:
        print(f"[Warn] 未在配置中找到 clip_mode={clip_mode}，将使用默认/空配置。")
    return params


def debug_log(msg: str):
    """统一调试日志格式，带时间戳并立即刷新。"""
    print(f"[DEBUG][{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def compute_policy_surrogate(
    clip_mode: str,
    ratio: torch.Tensor,
    adv_unsqueezed: torch.Tensor,
    clip_params: Dict[str, Any],
) -> Tuple[torch.Tensor, float]:
    """
    根据裁剪模式计算策略损失与 clip 比例。
    clip: PPO 原版 clip
    soft_clip: 软衰减版本（参考 ds_metaworld_ppo_mlp_add_vatrace_soft_clip.py 实现）
    后续如需新增模式，可在此函数中扩展。
    """
    surr1 = ratio * adv_unsqueezed

    if clip_mode == "clip":
        ratio_clipped = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
        surr2 = ratio_clipped * adv_unsqueezed
        surr_min = torch.min(surr1, surr2)
        policy_loss = -torch.mean(surr_min)
        is_clipped = ~torch.isclose(surr_min, surr1, atol=1e-8)
        clip_ratio = is_clipped.float().mean().item()
        return policy_loss, clip_ratio

    if clip_mode == "soft_clip" or clip_mode == "soft_clip_alpha-1" or clip_mode == "soft_clip_alpha-2":
        if clip_mode == "soft_clip_alpha-1":
            soft_clip_alpha = 1
        elif clip_mode == "soft_clip_alpha-2":
            soft_clip_alpha = 2
        else:
            soft_clip_alpha = clip_params.get("soft_clip_alpha", 1)
        diff = torch.maximum(ratio, 1.0 / ratio)
        coeff = (1.0 / diff).detach()
        coeff = coeff ** soft_clip_alpha
        surr_soft = surr1 * coeff
        policy_loss = -torch.mean(surr_soft)
        clip_ratio = 0.0  # 软裁剪无硬截断
        return policy_loss, clip_ratio

    raise ValueError(f"Unsupported clip mode: {clip_mode}")
    
# ================================================================
# 数据结构 更新经验数据结
# ================================================================
@dataclass
class Experience:
    obs: np.ndarray            # 状态向量 (state_dim,)
    action_token: np.ndarray   # 采样的离散动作 token (shape: [ACTION_DIM,])
    advantage: float = 0.0
    behaviour_logits: np.ndarray | None = None  # 行为策略的 logits (shape: [ACTION_DIM, VOCAB_SIZE])
    value_target: float = 0.0
    reward: float = 0.0           # r_t (已经 * REWARD_SCALE)
    discount: float = 1.0         # gamma * (1 - done_t)
    behaviour_logp: np.ndarray = None  # log μ(a_t | s_t), shape: [ACTION_DIM]

# ================================================================
# 1.5. 统计模块 (StatsActor)
# ================================================================
@ray.remote
class StatsActor:
    def __init__(self, window_size=MOVING_AVG_WINDOW):
        self.stats = defaultdict(lambda: {
            "episode_returns": deque(maxlen=window_size),
            "step_times": deque(maxlen=window_size),
            "episode_lengths": deque(maxlen=window_size),
            "successes": deque(maxlen=window_size),
            "total_episodes_processed": 0,
            "total_env_steps": 0
        })
        self.timings = defaultdict(lambda: deque(maxlen=window_size))
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
            "active_actor_count": self.get_active_actor_count()
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
    def __init__(self, capacity=REPLAY_CAPACITY, seed=SEED):
        # 设置随机种子
        random.seed(seed + id(self) % 1000)  # 为每个buffer添加不同偏移
        np.random.seed(seed + id(self) % 1000)
        
        # 存的是 trajectory 列表，每个元素为 (traj: List[Experience], done: bool)
        self.buffer = deque(maxlen=capacity)

    def add_trajectory(self, traj: List[Experience], done: bool, last_obs: np.ndarray):
        self.buffer.append((traj, done, last_obs))

    def size(self):
        # 返回总步数，保持语义近似
        return sum(len(traj) - 1 for traj, _, _ in self.buffer)

    def sample_sequences(self, min_total_steps: int):
        """
        采样若干条轨迹，使得它们的总步数 >= min_total_steps。
        """
        current_size = self.size()
        if current_size < min_total_steps:
            raise ValueError(f"Buffer total steps {current_size} < requested {min_total_steps}")
            
        # 随机打乱整个 buffer 的索引（或者简单地随机采样直到满足条件）
        # 为了效率，我们随机选择起始点，然后连续取，或者随机取索引。
        # 鉴于 buffer 是 deque，随机访问可能慢，但在 Ray Actor 中这通常是内存对象。
        # 这里采用随机抽样索引的方式。
        
        # 注意：self.buffer 是 deque，不支持切片或随机索引。先转为 list 引用会比较快，
        # 但如果 buffer 很大，每次转 list 开销大。
        # 优化方案：random.sample 直接从 deque 采一个较大的预估数量，然后截断？
        # 不，最准确的方法是维护一个 list 或者 accept 随机访问。
        # 鉴于 REPLAY_CAPACITY=50000，转 list 很快。
        
        buffer_list = list(self.buffer)
        # 随机打乱列表 (全量 shuffle 可能慢，不如随机选索引)
        indices = list(range(len(buffer_list)))
        random.shuffle(indices)
        
        traj_batch = []
        total_steps_collected = 0
        
        for idx in indices:
            traj, done, last_obs = buffer_list[idx]
            traj_batch.append((traj, done, last_obs))
            total_steps_collected += len(traj) - 1 # 最后一步不参与vtrace计算，bootstrap
            
            if total_steps_collected >= min_total_steps:
                break
        
        # 直接返回列表，不做 Padding 和 Stack，由 Trainer 自行处理
        # 为了传输效率，我们将同一字段的数据归拢到一起
        
        batch_obs_list, batch_act_list, batch_rew_list = [], [], []
        batch_disc_list, batch_logp_list = [], []
        batch_adv_list, batch_logits_list, batch_vtarg_list = [], [], []
        batch_done_list, batch_last_obs_list = [], []

        for traj, done, last_obs in traj_batch:
            # 提取基础数据
            batch_obs_list.append(np.stack([e.obs for e in traj]))
            batch_act_list.append(np.stack([e.action_token for e in traj]))
            batch_rew_list.append(np.array([e.reward for e in traj], dtype=np.float32))
            batch_disc_list.append(np.array([e.discount for e in traj], dtype=np.float32))
            batch_logp_list.append(np.stack([e.behaviour_logp for e in traj]))  # [T, ACTION_DIM]
            
            # 提取旧字段（兼容 PPO）
            batch_adv_list.append(np.array([e.advantage for e in traj], dtype=np.float32))
            batch_logits_list.append(np.stack([e.behaviour_logits for e in traj]))
            batch_vtarg_list.append(np.array([e.value_target for e in traj], dtype=np.float32))
            
            batch_done_list.append(float(done))
            batch_last_obs_list.append(last_obs)

        return (
            batch_obs_list,      # List[np.ndarray(T, D)]
            batch_act_list,
            batch_rew_list,
            batch_disc_list,
            batch_logp_list,
            batch_adv_list,
            batch_logits_list,
            batch_vtarg_list,
            np.array(batch_done_list, dtype=np.float32), # [B]
            np.stack(batch_last_obs_list)                # [B, D]
        )

# ================================================================
# 2. 经验回放与 Rollout
# ================================================================
class BaseWorkerActor:
    """rollout 和 eval worker 的共享逻辑。"""
    def __init__(self, infer, replay, wid, stats_actor, seed=SEED):
        self.infer = infer
        self.replay = replay
        self.stats_actor = stats_actor
        self.wid = wid
        
        # 设置每个worker的随机种子
        # 如果 wid 是字符串（如 "eval_0"），需要转换为整数
        if isinstance(wid, str):
            try:
                # 尝试从字符串中提取数字部分，如 "eval_0" -> 0
                wid_num = int(wid.split('_')[-1])
            except (ValueError, IndexError):
                # 如果无法提取，使用哈希值
                wid_num = hash(wid) % 10000
        else:
            wid_num = wid
            
        self.worker_seed = seed + wid_num * 1000
        random.seed(self.worker_seed)
        np.random.seed(self.worker_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.worker_seed)

        # 单任务设置：只初始化一个环境
        self.task_name = TASK_NAME  # 固定使用第一个（也是唯一的）任务
        print(f"BaseWorker {wid}: 正在初始化单任务 MetaWorld 环境: {self.task_name}")
        
        # MetaWorldWrapperDiscrete 可能不支持 seed 参数，需要检查其构造函数
        # 如果构造函数不支持 seed，我们需要在 reset 方法中设置
        try:
            # 尝试传入 seed 参数
            self.env = MetaWorldWrapperDiscrete(env_name=self.task_name, bins=N_ACTION_BINS, seed=self.worker_seed)
        except TypeError:
            # 如果构造函数不支持 seed 参数，创建一个没有 seed 的环境
            print(f"BaseWorker {wid}: MetaWorldWrapperDiscrete 不支持 seed 参数，将在 reset 时设置")
            self.env = MetaWorldWrapperDiscrete(env_name=self.task_name, bins=N_ACTION_BINS)
            # 存储种子，在第一次 reset 时使用
            self._pending_seed = self.worker_seed
        else:
            # 如果构造函数支持 seed 参数，清空 pending_seed
            self._pending_seed = None
            
        print(f"BaseWorker {wid}: 环境初始化完成。")

        self.task_description = "test: task_description"
        self.current_env_name = self.task_name
        self.episode_counter = 0  # 用于生成每个episode的唯一种子

    def _reset_and_select_env(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """重置环境，支持传入特定种子"""
        # 单任务设置：直接重置当前环境
        # 使用基础种子 + worker_id * 1000 + episode_counter
        if seed is None:
            episode_seed = self.worker_seed + self.episode_counter * 10000
            self.episode_counter += 1
        else:
            episode_seed = seed
            
        # 如果环境在初始化时没有设置种子，在 reset 时设置
        if self._pending_seed is not None:
            # 第一次 reset 时使用 pending_seed
            obs, info = self.env.reset(seed=self._pending_seed)
            self._pending_seed = None  # 只在第一次使用
        else:
            obs, info = self.env.reset(seed=episode_seed)
            
        return obs, info


@ray.remote
class RolloutWorkerActor(BaseWorkerActor):
    def __init__(self, infer, replay, wid, stats_actor):
        # wid 应该是整数，但为了安全起见，确保它是整数
        if isinstance(wid, str) and wid.startswith("rollout_"):
            try:
                wid = int(wid.split('_')[1])
            except (ValueError, IndexError):
                pass
                
        super().__init__(infer, replay, wid, stats_actor)
        # 单任务设置：移除环境选择的历史记录
        self.local_buffer = []

    def run(self):
        try:
            # 使用确定性种子
            obs, info = self._reset_and_select_env()
            reward_sum, time_start, step_count_total = 0.0, time.time(), 0
            while True:
                # 直接使用状态向量作为模型输入
                action_env, action_token, logits, value, logp_mu = ray.get(self.infer.request.remote(obs, deterministic=False))

                # 修正: 传入 discrete token 给环境
                nxt, r, term, trunc, info = self.env.step(action_token)
                reward_sum += r
                chunk_reward = r * REWARD_SCALE
                step_count_total += 1
                done = term or trunc
                
                # 计算 discount
                step_discount = GAMMA * (0.0 if done else 1.0)

                self.local_buffer.append((obs, action_token, chunk_reward, logits, value, logp_mu, step_discount))
                obs = nxt

                if done:
                    step_time = (time.time() - time_start) / max(step_count_total, 1)
                    success = float(info['success'])
                    self.stats_actor.add_episode_return.remote(
                        self.current_env_name,
                        reward_sum,
                        step_time,
                        step_count_total,
                        success,
                        actor_id=self.wid,
                        step_num=step_count_total,
                    )
                    reward_sum = 0.0
                    if self.local_buffer: 
                        bootstrap_val = 0.0  # episode 结束时 bootstrap value 为 0
                        self._process_traj(self.local_buffer, done=True, bootstrap_val=bootstrap_val, last_obs=obs)
                    self.local_buffer.clear()
                    # 使用确定性种子重置
                    obs, info = self._reset_and_select_env()
                    time_start, step_count_total = time.time(), 0
                elif len(self.local_buffer) == ROLLOUT_LOCAL_BUF + 1:
                    last_state, _, _, _, last_value, _, _ = self.local_buffer[-1]
                    last_obs = last_state
                    bootstrap_val = last_value  # 使用最后一个状态的 value 作为 bootstrap
                    self._process_traj(self.local_buffer[:-1], done=False, bootstrap_val=bootstrap_val, last_obs=last_obs)
                    self.local_buffer = [self.local_buffer[-1]]
        except Exception as e: 
            import traceback
            print(f"[ERROR] RolloutWorker {self.wid} run() 崩溃: {e}", flush=True)
            traceback.print_exc()
            raise

    def _process_traj(self, traj_segment, done: bool, bootstrap_val: float, last_obs: np.ndarray):
        rets, advs = [], []
        gae = 0.0
        for i in reversed(range(len(traj_segment))):
            _, _, r, _, v, _, _ = traj_segment[i]
            nv = bootstrap_val if i == len(traj_segment) - 1 else traj_segment[i+1][4]
            delta = r + GAMMA * nv - v
            gae = delta + GAMMA * LAMBDA * gae
            advs.append(gae)
            rets.append(gae + v)
        advs.reverse(); rets.reverse()
        advs_np = np.array(advs, dtype=np.float32)

        traj: List[Experience] = []
        for i, (s, a_token, r_val, logits, _, logp_mu, step_discount) in enumerate(traj_segment):
            traj.append(
                Experience(
                    obs=s,  # 现在是 numpy 数组
                    action_token=a_token.astype(np.int64),
                    advantage=float(advs_np[i]),
                    behaviour_logits=logits.astype(np.float32),
                    value_target=float(rets[i]),
                    reward=float(r_val),
                    discount=float(step_discount),
                    behaviour_logp=logp_mu.astype(np.float32),  # [ACTION_DIM]
                )
            )
        self.replay.add_trajectory.remote(traj, done, last_obs)


@ray.remote
class EvaluationWorkerActor(BaseWorkerActor):
    def __init__(self, infer, wid, stats_actor):
        # 确保 wid 是整数，eval_workers 创建时传入的是字符串 "eval_{i}"
        # 我们需要将字符串转换为整数用于种子计算
        if isinstance(wid, str) and wid.startswith("eval_"):
            try:
                # 提取数字部分，如 "eval_0" -> 0
                wid_num = int(wid.split('_')[1])
                # 使用提取的数字作为 wid 传递给父类
                super().__init__(infer, None, wid_num, stats_actor)
                # 但保留原始 wid 字符串用于标识
                self.original_wid = wid
            except (ValueError, IndexError):
                # 如果提取失败，使用哈希值
                wid_num = hash(wid) % 10000
                super().__init__(infer, None, wid_num, stats_actor)
                self.original_wid = wid
        else:
            super().__init__(infer, None, wid, stats_actor)
            self.original_wid = wid
            
        print(f"EvaluationWorker {self.original_wid}: 环境初始化完成。")

    def run(self):
        try:
            # 使用确定性种子
            obs, info = self._reset_and_select_env()
            while True:
                reward_sum, time_start, step_count_total, done = 0.0, time.time(), 0, False
                while not done:
                    # 直接使用状态向量作为模型输入
                    action_env, action_token, _, _, _ = ray.get(self.infer.request.remote(obs, deterministic=True))

                    # 修正: 传入 discrete token，避免双重转换
                    obs, r, term, trunc, info = self.env.step(action_token)
                    reward_sum += r
                    step_count_total += 1
                    done = term or trunc

                step_time = (time.time() - time_start) / max(step_count_total, 1)
                success = float(info["success"])
                self.stats_actor.add_episode_return.remote(
                    f"eval_{self.current_env_name}",
                    reward_sum,
                    step_time,
                    step_count_total,
                    success,
                    actor_id=self.original_wid,  # 使用原始字符串ID
                    step_num=step_count_total,
                )
                # 使用确定性种子重置
                obs, info = self._reset_and_select_env()
        except Exception as e: 
            import traceback
            print(f"[ERROR] EvaluationWorker {self.original_wid} run() 崩溃: {e}", flush=True)
            traceback.print_exc()
            raise
# ================================================================
# 3. 推理器 (InferenceActor)
# ================================================================
@ray.remote(num_gpus=1)
class InferenceActor(InferenceActorCom):
    def __init__(self, actor_id, stats_actor, seed=SEED):
        super().__init__()
        self.actor_id = actor_id
        
        # 设置推理器的随机种子
        self.inference_seed = seed + actor_id * 100
        random.seed(self.inference_seed)
        np.random.seed(self.inference_seed)
        torch.manual_seed(self.inference_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.inference_seed)
            torch.cuda.manual_seed_all(self.inference_seed)
        
        print(f"InferenceActor {actor_id}: 正在加载 MLP ActorCritic...")
        self.model = MLPActorCriticDiscrete(
            torch_dtype=TORCH_DTYPE,
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            n_action_bins=N_ACTION_BINS
        )
        self.model.cuda()
        self.model.eval()
        self.stats_actor = stats_actor

        self.batch_size = INFERENCE_BATCH
        self.timeout_sec = INFERENCE_TIMEOUT_MS / 1000.0
        self.requests, self.promises = [], []
        self.last_process_time = time.time()

        loop = asyncio.get_event_loop()
        self._bg_task = loop.create_task(self._loop())
        self._bg_task.add_done_callback(self._on_bg_task_done)
        print(f"InferenceActor {self.actor_id} 初始化于 GPU: {ray.get_gpu_ids()} (批次超时: {INFERENCE_TIMEOUT_MS}ms)")

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

    async def request(self, obs: np.ndarray, deterministic: bool = False):
        """输入: numpy 数组状态向量，输出: 连续动作数组"""
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        self.requests.append((obs, deterministic))
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

            obs_list = [r[0] for r in requests_to_process]
            deterministic_flags = [r[1] for r in requests_to_process]
            t_loop_start = time.time()
            try:
                # 将观测列表转换为批次输入
                inputs_batch = self.model.prepare_inputs_batch(obs_list)

                with torch.inference_mode():
                    # 前向传播获取 logits 和 value
                    action_logits, value = self.model(inputs_batch)

                    # 后处理以采样动作 tokens 和对应的离散动作
                    dist, action_tokens, discrete_actions = self.model.post_process(
                        action_logits, deterministic=deterministic_flags
                    )
                    
                    # 计算行为策略 log-prob (每个动作维度的 log-prob，不求和)
                    logp_mu = dist.log_prob(action_tokens)  # [B, ACTION_DIM]

                    # 将离散动作转换为连续动作（用于环境）
                    # MetaWorld 动作空间是4维连续动作，这里使用简单的线性映射
                    actions_env = []
                    for i in range(discrete_actions.shape[0]):
                        # 将每个维度的离散动作 [0, N_ACTION_BINS-1] 映射到连续动作 [-1, 1]
                        continuous_action = -1.0 + 2.0 * discrete_actions[i] / (N_ACTION_BINS - 1)
                        actions_env.append(continuous_action.astype(np.float32))

                    # 转换为numpy数组以便返回
                    actions_env = np.array(actions_env)
                    action_tokens = action_tokens.cpu().numpy()
                    logits = action_logits.cpu().numpy()
                    values = value.cpu().numpy()
                    logp_mu_np = logp_mu.cpu().numpy()

                for i in range(len(promises_to_process)):
                    promises_to_process[i].set_result((
                        actions_env[i],           # 连续环境动作
                        action_tokens[i],         # 离散动作 token
                        logits[i],                # 对应的 logits
                        values[i],                # 价值估计
                        logp_mu_np[i]             # 行为策略 log-prob
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


# ================================================================
# 4. 训练器 (TrainerActor)
# ================================================================
@ray.remote(num_gpus=1)
class TrainerActor(TrainerActorCom):
    def __init__(self, rank, world_size, replay_buffer, seed=SEED):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.replay_buffer = replay_buffer
        
        # 设置训练器的随机种子（每个rank不同的种子）
        self.trainer_seed = seed + rank * 100
        random.seed(self.trainer_seed)
        np.random.seed(self.trainer_seed)
        torch.manual_seed(self.trainer_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.trainer_seed)
            torch.cuda.manual_seed_all(self.trainer_seed)
        
        self.model = None
        self.optimizer = None
        self.base_model = None
        self.data_dtype = None
        self.next_ready_batch: Optional[Tuple] = None
        self.data_fetching_task = None
        self.super_batch_size = TRAIN_BATCH_SIZE * ACCUMULATION_STEPS
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

    def get_parameter_counts(self):
        if self.base_model is None:
            return 0, 0
        n_total = sum(p.numel() for p in self.base_model.parameters())
        n_trainable = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        return n_total, n_trainable

    def get_node_ip(self):
        return ray.util.get_node_ip_address()

    @torch.no_grad()
    def vtrace_from_log_rhos(self, log_rhos, discounts, rewards, values, bootstrap_value,
                             clip_rho=1.0, clip_c=1.0, clip_pg_rho=1.0):
        """
        V-trace 计算函数
        
        Args:
            log_rhos: [T] - log(π(a|s) / μ(a|s))，单个序列
            discounts: [T] - γ * (1 - done)
            rewards: [T] - 奖励
            values: [T] - V(s) 当前策略的价值估计
            bootstrap_value: scalar - 序列末尾的状态价值 V(s_T)
        
        Returns:
            vs: [T] - value targets for critic
            pg_adv: [T] - policy gradient advantages
        """
        rhos = torch.exp(log_rhos)
        clipped_rhos = torch.clamp(rhos, max=clip_rho)
        cs = torch.clamp(rhos, max=clip_c)

        # values_tp1 = concat([v_{t+1} for t in 0..T-1], bootstrap_value)
        values_tp1 = torch.cat([values[1:], bootstrap_value.unsqueeze(0)], dim=0)  # [T]

        # deltas_t = clipped_rho_t * (r_t + γ_{t+1} * v_{t+1} - v_t)
        deltas = clipped_rhos * (rewards + discounts * values_tp1 - values)  # [T]

        # 从后往前计算 value targets
        acc = torch.zeros_like(bootstrap_value)  # scalar，初始化为 0
        vs_minus_v = []
        for t in reversed(range(values.shape[0])):
            acc = deltas[t] + discounts[t] * cs[t] * acc
            vs_minus_v.append(acc)
        vs_minus_v = torch.stack(list(reversed(vs_minus_v)), dim=0)  # [T]
        vs = vs_minus_v + values

        # 计算 policy gradient advantages
        vs_tp1 = torch.cat([vs[1:], bootstrap_value.unsqueeze(0)], dim=0)
        pg_rhos = torch.clamp(rhos, max=clip_pg_rho)
        pg_adv = pg_rhos * (rewards + discounts * vs_tp1 - values)

        return vs, pg_adv

    def setup_deepspeed_group(self, master_addr, master_port):
        print(f"[DS-setup][rank {self.rank}] 进入 setup_deepspeed_group")
        print(f"Trainer {self.rank}: 开始设置 DeepSpeed 环境变量...")
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["LOCAL_RANK"] = "0"
        print(f"Trainer {self.rank}: 环境变量设置完成 - RANK={self.rank}, WORLD_SIZE={self.world_size}, MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")

        print(f"Trainer {self.rank}: 初始化分布式后端 (nccl)...")
        deepspeed.init_distributed(dist_backend="nccl")
        print(f"Trainer {self.rank}: 分布式后端初始化完成")

        print(f"[DS-setup][rank {self.rank}] before create model")
        print(f"Trainer {self.rank}: 正在加载 MLP ActorCritic...")
        model = MLPActorCriticDiscrete(
            torch_dtype=TORCH_DTYPE,
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            n_action_bins=N_ACTION_BINS
        )
        self.base_model = model
        print(f"Trainer {self.rank}: MLP ActorCritic 模型创建完成")

        # 参数分组
        print(f"Trainer {self.rank}: 获取参数分组...")
        param_groups = self.base_model.get_parameter_groups()
        print(f"Trainer {self.rank}: 找到 {len(param_groups)} 个参数组")
        optimizer_params = [
            {"params": pg["params"], "name": pg["name"], "lr": POLICY_LR if pg["name"] == "policy" else VALUE_LR}
            for pg in param_groups
        ]
        print(f"Trainer {self.rank}: 参数分组和优化器参数准备完成")

        ds_config = {
            "train_micro_batch_size_per_gpu": TRAIN_BATCH_SIZE,
            "gradient_accumulation_steps": ACCUMULATION_STEPS,
            "optimizer": {"type": "AdamW", "params": {}},
            "bf16": {"enabled": False},  # MLP 模型使用 float32
            "zero_optimization": {
                "stage": 2, "allgather_partitions": True, "allgather_bucket_size": 5e8,
                "reduce_scatter": True, "reduce_bucket_size": 5e8, "overlap_comm": True,
                "contiguous_gradients": True
            },
            "gradient_clipping": 1.0,
        }
        print(f"Trainer {self.rank}: DeepSpeed 配置准备完成")

        self.data_dtype = torch.float32  # MLP 使用 float32

        print(f"[DS-setup][rank {self.rank}] before deepspeed.initialize (world_size={self.world_size}, rank={self.rank})")
        print(f"Trainer {self.rank}: 开始初始化 DeepSpeed...")
        try:
            # print(f"model: {model}")
            # print(f"ds_config: {ds_config}")
            # print(f"optimizer_params: {optimizer_params}")
            self.model, self.optimizer, _, _ = deepspeed.initialize(model=model, config=ds_config, model_parameters=optimizer_params)
            print(f"[DS-setup][rank {self.rank}] after deepspeed.initialize")
            print(f"TrainerActor Rank {self.rank}: DeepSpeed 训练组 (ZeRO-2) 初始化完成。")
        except Exception as e:
            print(f"Trainer {self.rank}: DeepSpeed 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            raise

        print(f"[DS-setup][rank {self.rank}] before create data_fetching_task")
        self.data_fetching_task = asyncio.get_event_loop().create_task(self._data_fetching_loop())
        print(f"[DS-setup][rank {self.rank}] after create data_fetching_task")

        n_total = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数量: {n_total:,}, 可训练参数量: {n_trainable:,}")
        print(f"[DS-setup][rank {self.rank}] setup_deepspeed_group done")

    async def save_agent(self, ckpt_dir: str, step: int):
        """
        只在 rank-0 上调用。调用 MLPActorCritic 内部的 save_model
        """
        os.makedirs(ckpt_dir, exist_ok=True)
        self.base_model.save_model(ckpt_dir, epoch=step)
        print(f"[Trainer {self.rank}] 已保存 checkpoint -> {ckpt_dir}/mlp_actor_critic_epoch_{step}.pt")

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

                # V-trace 采样：由于结束轨迹不损失步数，截断轨迹只损失1步
                # 采样时只需要稍微多采样一些即可（考虑到部分轨迹可能是截断的）
                # 简化逻辑：直接采样 super_batch_size，如果不足会在训练循环中动态调整
                while await self.replay_buffer.size.remote() < self.super_batch_size:
                    print(f"Trainer {self.rank} (BG): 等待 ReplayBuffer 填充至 {self.super_batch_size}...")
                    await asyncio.sleep(3)

                t_sample_start = time.time()
                # 精确采样模式：采样 super_batch_size 步
                # ReplayBuffer 会自动凑够轨迹来满足这个要求。
                (obs_seq_list, act_seq_list, rew_seq_list, disc_seq_list, logp_seq_list, 
                 adv_seq_list, logits_seq_list, vtarg_seq_list, done_seq, last_obs_seq) = \
                    await self.replay_buffer.sample_sequences.remote(self.super_batch_size)
                sample_time = time.time() - t_sample_start

                t_prep_start = time.time()
                
                # 1. Flatten for GAE PPO (Compatibility Mode)
                # 直接将所有轨迹拼接起来，形成一个巨大的散乱 batch
                obs_flat = np.concatenate(obs_seq_list, axis=0)
                act_flat = np.concatenate(act_seq_list, axis=0)
                adv_flat = np.concatenate(adv_seq_list, axis=0)
                logits_flat = np.concatenate(logits_seq_list, axis=0)
                vtarg_flat = np.concatenate(vtarg_seq_list, axis=0)

                device = next(self.model.parameters()).device
                
                # Flat tensors for current PPO
                obs_t = torch.tensor(obs_flat, dtype=torch.float32, device=device)
                act_token_t = torch.tensor(act_flat, dtype=torch.long, device=device)
                adv_t = torch.tensor(adv_flat, dtype=torch.float32, device=device)
                logits_old_t = torch.tensor(logits_flat, dtype=torch.float32, device=device)
                v_targ_t = torch.tensor(vtarg_flat, dtype=torch.float32, device=device)
                
                # 2. Sequence Tensors for V-trace (Future)
                # 目前先不处理 Padding/Stacking，因为还没用到 V-trace。
                # 如果需要用，可以在这里做 Pad + Stack，或者留给训练循环做。
                # 为了避免不必要的开销，我们暂时只保留引用。
                
                prep_time = time.time() - t_prep_start

                self.next_ready_batch = {
                    # GAE 兼容数据（flatten）
                    'obs': obs_t,
                    'act_token': act_token_t,
                    'advantage': adv_t,
                    'logits_old': logits_old_t,
                    'value_target': v_targ_t,
                    
                    # V-trace 序列数据（列表形式，每个元素是一个序列）
                    'obs_seq_list': obs_seq_list,
                    'act_seq_list': act_seq_list,
                    'rew_seq_list': rew_seq_list,
                    'disc_seq_list': disc_seq_list,
                    'logp_mu_seq_list': logp_seq_list,
                    'logits_seq_list': logits_seq_list,  # 行为策略的 logits，用于 KL 散度计算
                    'done_seq': done_seq,
                    'last_obs_seq': last_obs_seq,
                    
                    'sample_time': sample_time,
                    'prep_time': prep_time
                }

            except Exception as e:
                print(f"Trainer {self.rank}: 数据采样失败: {e}。将在3秒后重试。")
                await asyncio.sleep(3)
                import traceback
                traceback.print_exc()

    async def run_training_epoch(self) -> Tuple[float, float, float, float, float, Dict[str, float], int, float, float, float, Dict[str, float]]:
        if self.next_ready_batch is None:
            print(f"Trainer {self.rank}: 等待超级批次...")
            while self.next_ready_batch is None:
                await asyncio.sleep(0.02)
            print(f"Trainer {self.rank}: 数据已收到，开始训练。")

        current_lrs = {}
        value_lr = self._get_current_lr(self.global_step, VALUE_LR, VALUE_WARMUP_STEPS, TRAIN_ITERS)
        policy_lr = self._get_current_lr(self.global_step, POLICY_LR, POLICY_WARMUP_STEPS, TRAIN_ITERS, start_step=POLICY_TRAIN_START_STEP)

        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'value': param_group['lr'] = value_lr; current_lrs['value'] = value_lr
            elif param_group['name'] == 'policy': param_group['lr'] = policy_lr; current_lrs['policy'] = policy_lr

        current_batch = self.next_ready_batch
        self.next_ready_batch = None

        policy_sample_time = current_batch['sample_time']
        policy_prep_time = current_batch['prep_time']

        # =================================================================
        # V-trace 计算：单独处理每个序列
        # =================================================================
        device = next(self.model.parameters()).device
        
        # 获取序列数据
        obs_seq_list = current_batch['obs_seq_list']
        act_seq_list = current_batch['act_seq_list']
        rew_seq_list = current_batch['rew_seq_list']
        disc_seq_list = current_batch['disc_seq_list']
        logp_mu_seq_list = current_batch['logp_mu_seq_list']
        logits_seq_list = current_batch['logits_seq_list']  # 行为策略的 logits
        done_seq = current_batch['done_seq']
        last_obs_seq = current_batch['last_obs_seq']

        # 对每个序列单独计算 V-trace（不 padding，单独处理）
        all_obs_flat, all_act_flat, all_pg_adv_flat, all_vs_flat, all_logits_old_flat = [], [], [], [], []
        
        for seq_idx in range(len(obs_seq_list)):
            # 获取当前序列数据
            obs_seq = torch.tensor(obs_seq_list[seq_idx], dtype=torch.float32, device=device)  # [T, state_dim]
            act_seq = torch.tensor(act_seq_list[seq_idx], dtype=torch.long, device=device)    # [T, ACTION_DIM]
            rew_seq = torch.tensor(rew_seq_list[seq_idx], dtype=torch.float32, device=device)  # [T]
            disc_seq = torch.tensor(disc_seq_list[seq_idx], dtype=torch.float32, device=device) # [T]
            logp_mu_seq = torch.tensor(logp_mu_seq_list[seq_idx], dtype=torch.float32, device=device)  # [T, ACTION_DIM]
            
            T = obs_seq.shape[0]
            
            # 用当前模型计算当前策略的输出（前 T-1 步）
            action_logits, value = self.model(obs_seq[:-1])  # logits: [T-1, ACTION_DIM, n_bins], value: [T-1, 1]
            value = value.squeeze(-1)  # [T-1]
            
            # 计算 bootstrap value（最后一个状态）
            # print(f"[Debug] done_seq shape: {done_seq.shape}")
            # print(f"[Debug] done_seq: {done_seq}")

            # print(f"[Debug] seq_idx: {seq_idx}")
            # raise ValueError("Debug")
            if done_seq[seq_idx] > 0.5: # done shape: [B, ]
                bootstrap_value = torch.tensor(0.0, device=device)
            else:
                _, bootstrap_value = self.model(obs_seq[-1:])  # [1, 1]
                bootstrap_value = bootstrap_value.squeeze(-1).squeeze(0)  # scalar
            
            # 计算当前策略的 log-prob（每个动作维度）
            dist = torch.distributions.Categorical(logits=action_logits)  # batch_shape [T-1, ACTION_DIM]
            logp_pi = dist.log_prob(act_seq[:-1])  # [T-1, ACTION_DIM]
            
            # 计算 log_rhos = log π(a|s) - log μ(a|s)
            # V-trace 需要标量的 log_rhos，所以对 ACTION_DIM 求和得到联合 log-prob
            log_rhos = logp_pi.sum(dim=-1) - logp_mu_seq[:-1].sum(dim=-1)  # [T-1]
            
            # V-trace 计算
            vs, pg_adv = self.vtrace_from_log_rhos(
                log_rhos,      # [T-1]
                disc_seq[:-1], # [T-1]
                rew_seq[:-1],  # [T-1]
                value,         # [T-1]
                bootstrap_value  # scalar
            )
            
            # 收集结果（只收集前 T-1 步，因为 V-trace 需要 bootstrap）
            all_obs_flat.append(obs_seq[:-1])      # [T-1, state_dim]
            all_act_flat.append(act_seq[:-1])      # [T-1, ACTION_DIM]
            all_pg_adv_flat.append(pg_adv)         # [T-1]
            all_vs_flat.append(vs)                 # [T-1]
            # 保存行为策略的 logits，用于后续计算 KL 散度
            logits_old_seq = torch.tensor(logits_seq_list[seq_idx], dtype=torch.float32, device=device)  # [T, ACTION_DIM, n_bins]
            all_logits_old_flat.append(logits_old_seq[:-1])  # [T-1, ACTION_DIM, n_bins]
        
        # 拼接所有序列的结果
        obs_t = torch.cat(all_obs_flat, dim=0)     # [total_steps, state_dim]
        act_token_t = torch.cat(all_act_flat, dim=0)  # [total_steps, ACTION_DIM]
        pg_adv_t = torch.cat(all_pg_adv_flat, dim=0)  # [total_steps]
        vs_t = torch.cat(all_vs_flat, dim=0)       # [total_steps]
        logits_old_t = torch.cat(all_logits_old_flat, dim=0)  # [total_steps, ACTION_DIM, n_bins]
        
        # 调试信息：显示采样统计
        num_traj_sampled = len(obs_seq_list)
        total_steps_sampled = sum(len(obs_seq_list[i]) for i in range(num_traj_sampled))
        actual_usable_steps = obs_t.shape[0]  # V-trace 后实际可用步数
        steps_lost = total_steps_sampled - actual_usable_steps  # 损失的步数（应该等于轨迹数）
        if self.rank == 0:  # 只在 rank 0 打印，避免重复日志
            print(f"[V-trace采样统计] Rank {self.rank}: 采样了 {num_traj_sampled} 条轨迹，"
                  f"总步数 {total_steps_sampled}，实际可用步数 {actual_usable_steps}，"
                  f"损失步数 {steps_lost} (目标: {self.super_batch_size})")
        
        # =================================================================
        # 关键修正: 对 Flatten 后的序列数据进行 Shuffle
        # =================================================================
        current_batch_size = obs_t.shape[0]
        # 使用固定的随机种子进行shuffle以确保可复现性
        generator = torch.Generator(device=obs_t.device)
        generator.manual_seed(self.trainer_seed + self.global_step * 1000)
        indices = torch.randperm(current_batch_size, device=obs_t.device, generator=generator)
        
        obs_t = obs_t[indices]
        act_token_t = act_token_t[indices]
        pg_adv_t = pg_adv_t[indices]
        vs_t = vs_t[indices]
        logits_old_t = logits_old_t[indices]  # 也需要 shuffle logits_old_t

        # 全局 advantage 归一化（使用 V-trace 的 pg_adv）
        local_sum = pg_adv_t.sum()
        local_sq_sum = (pg_adv_t * pg_adv_t).sum()
        local_count = torch.tensor([pg_adv_t.numel()], device=pg_adv_t.device, dtype=torch.float32)

        distributed.all_reduce(local_sum, op=distributed.ReduceOp.SUM)
        distributed.all_reduce(local_sq_sum, op=distributed.ReduceOp.SUM)
        distributed.all_reduce(local_count, op=distributed.ReduceOp.SUM)

        global_mean = local_sum / torch.clamp(local_count, min=1.0)
        global_var = torch.clamp(local_sq_sum / torch.clamp(local_count, min=1.0) - global_mean * global_mean, min=1e-12)
        global_std = torch.sqrt(global_var)

        epoch_losses, epoch_p_losses, epoch_v_losses, epoch_e_losses, epoch_kl_losses = [], [], [], [], []
        epoch_ent, epoch_kl_divs, epoch_clip_ratios = [], [], []

        # Update based on actual batch size (which may vary slightly due to trajectory lengths)
        current_batch_size = obs_t.shape[0]
        
        # V-trace 验证：实际可用步数 = 采样总步数 - 轨迹数量（每条轨迹损失1步用于bootstrap）
        # 如果不足，动态调整 update 次数（但至少保证能完成一次更新）
        if current_batch_size < self.super_batch_size:
            max_possible_updates = current_batch_size // TRAIN_BATCH_SIZE
            num_updates_in_epoch = max(1, min(ACCUMULATION_STEPS, max_possible_updates))
            print(f"[Warning] Rank {self.rank}: V-trace 后实际可用步数 {current_batch_size} < 目标 {self.super_batch_size}。"
                  f"将使用 {num_updates_in_epoch} 次更新（而不是 {ACCUMULATION_STEPS} 次）。")
        else:
            # 强制固定 update 次数，防止多卡死锁
            num_updates_in_epoch = ACCUMULATION_STEPS
        
        if current_batch_size < TRAIN_BATCH_SIZE:
            print(f"[Error] Rank {self.rank}: 数据严重不足! {current_batch_size} 步不足以进行一次更新（需要至少 {TRAIN_BATCH_SIZE} 步）。")
            # 返回零损失，避免崩溃
            current_lrs = {}
            for param_group in self.optimizer.param_groups:
                if param_group['name'] == 'value': current_lrs['value'] = param_group['lr']
                elif param_group['name'] == 'policy': current_lrs['policy'] = param_group['lr']
            return 0.0, 0.0, 0.0, 0.0, 0.0, current_lrs, self.global_step, 0.0, 0.0, {}

        t_policy_train_start = time.time()

        for i in range(num_updates_in_epoch):
            start = i * TRAIN_BATCH_SIZE
            end = min(start + TRAIN_BATCH_SIZE, current_batch_size)  # 防止越界
            mini_obs = obs_t[start:end]
            mini_act_token = act_token_t[start:end]
            mini_pg_adv = pg_adv_t[start:end]
            mini_vs = vs_t[start:end]
            mini_logits_old = logits_old_t[start:end]  # [B, ACTION_DIM, n_bins]

            normalized_adv = (mini_pg_adv - global_mean) / (global_std + 1e-8)

            # 前向（重新计算，因为需要当前策略的输出）
            action_logits, value = self.model(mini_obs)
            value = value.to(torch.float32)

            # V-trace value loss: (value - vs)^2
            value_loss = VF_COEF * torch.mean((value - mini_vs) ** 2)

            if self.global_step < POLICY_TRAIN_START_STEP:
                loss = value_loss
                policy_loss = torch.tensor(0.0, device=loss.device)
                ent_loss = torch.tensor(0.0, device=loss.device)
                kl_loss = torch.tensor(0.0, device=loss.device)
                kl_div = 0.0
                ent = torch.tensor(0.0, device=loss.device)
                clip_ratio = 0.0  # 策略未开始训练时，clip ratio 为 0
            else:
                # 策略与熵损失 (离散版本)
                dist = torch.distributions.Categorical(logits=action_logits)
                logp = dist.log_prob(mini_act_token)  # [B, ACTION_DIM] - 每个动作维度的 log-prob

                with torch.no_grad():
                    dist_old = torch.distributions.Categorical(logits=mini_logits_old)
                    logp_old = dist_old.log_prob(mini_act_token)  # [B, ACTION_DIM]

                kl_div_tensor = kl.kl_divergence(dist_old, dist)  # [B, ACTION_DIM]
                kl_div = torch.mean(kl_div_tensor).item()
                kl_loss = KL_COEF * torch.mean(kl_div_tensor)

                ratio = torch.exp(logp - logp_old)  # [B, ACTION_DIM]
                adv_unsqueezed = normalized_adv.unsqueeze(dim=-1)  # [B, 1]
                policy_loss, clip_ratio = compute_policy_surrogate(
                    clip_mode=CLIP_MODE,
                    ratio=ratio,
                    adv_unsqueezed=adv_unsqueezed,
                    clip_params=CLIP_PARAMS,
                )          
                ent = torch.mean(dist.entropy())  # [B, ACTION_DIM] -> scalar (对所有元素求平均)
                ent_loss = -ENT_COEF * ent

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
            epoch_clip_ratios.append(clip_ratio)
            if self.model.is_gradient_accumulation_boundary():
                self.global_step += 1

        avg_loss = np.mean(epoch_losses)
        avg_p_loss = np.mean(epoch_p_losses)
        avg_v_loss = np.mean(epoch_v_losses)
        avg_e_loss = np.mean(epoch_e_losses)
        avg_kl_loss = np.mean(epoch_kl_losses)
        avg_ent = np.mean(epoch_ent)
        avg_kl_div = np.mean(epoch_kl_divs)
        avg_clip_ratio = np.mean(epoch_clip_ratios) if epoch_clip_ratios else 0.0

        perf_metrics = {
            "policy_sample_time": policy_sample_time,
            "policy_prep_time": policy_prep_time,
            "policy_train_time": time.time() - t_policy_train_start
        }

        return avg_loss, avg_p_loss, avg_v_loss, avg_e_loss, avg_kl_loss, current_lrs, self.global_step, avg_ent, avg_kl_div, avg_clip_ratio, perf_metrics



# ================================================================
# 5. 主逻辑
# ================================================================
def main():
    args = parse_args()
    global TASK_NAME, BENCHMARK, NUM_TRAINER_GPUS, NUM_ROLLOUT_WORKERS, NUM_EVAL_WORKERS, TRAIN_BATCH_SIZE, SEED, CLIP_MODE, CLIP_PARAMS
    TASK_NAME = args.task_name
    BENCHMARK = "MetaWorld_" + TASK_NAME.replace("-", "_")
    debug_log("进入 main()")
    debug_log(f"收到命令行参数: {args}")
    debug_log(f"----------------------------------------------------")
    debug_log(f"目前训练metaworld任务为: {TASK_NAME}")
    debug_log(f"----------------------------------------------------")
    NUM_TRAINER_GPUS = args.num_trainer_gpus
    NUM_ROLLOUT_WORKERS = args.num_rollout_workers
    NUM_EVAL_WORKERS = args.num_eval_workers
    TRAIN_BATCH_SIZE = args.train_batch_size
    SEED = args.seed
    CLIP_MODE = args.clip_mode

    # 加载裁剪配置并应用对应超参
    clip_config = load_clip_config(args.clip_config)
    CLIP_PARAMS = select_clip_params(CLIP_MODE, clip_config)

    debug_log(f"[Clip] 使用模式: {CLIP_MODE}, 配置: {CLIP_PARAMS}")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    debug_log("=== 开始 MetaWorld MLP PPO 训练脚本 ===")
    debug_log(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    debug_log(f"TORCH_DISTRIBUTED_DEBUG: {os.environ.get('TORCH_DISTRIBUTED_DEBUG', 'Not set')}")
    
    # 设置全局随机种子 - 新增
    debug_log(f"设置全局随机种子: {SEED}")
    set_seed(SEED)

    # 设置调试环境变量
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_IB_DISABLE"] = "1"  # 禁用 IB，如果网络有问题

    # 检查 GPU 状态
    if torch.cuda.is_available():
        debug_log(f"CUDA 可用，GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.set_device(i)
                debug_log(f"GPU {i}: {torch.cuda.get_device_name(i)}, 内存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
                torch.cuda.empty_cache()
            except Exception as e:
                debug_log(f"GPU {i} 检查失败: {e}")
    else:
        debug_log("CUDA 不可用")

    os.environ["RAY_DEDUP_LOGS"] = "0"
    debug_log("即将调用 ray.init(ignore_reinit_error=True, _temp_dir='/dev/shm', include_dashboard=False)")
    _t_ray_init = time.time()
    ray.init(ignore_reinit_error=True, _temp_dir='/dev/shm', include_dashboard=False)
    debug_log(f"ray.init 完成，用时 {time.time() - _t_ray_init:.2f} 秒")

    log_dir = f"runs/MetaWorld/{BENCHMARK}/MLP_DS_PPO_{int(time.time())}_seed{SEED}"
    writer = SummaryWriter(log_dir)
    stats_actor = StatsActor.remote(window_size=MOVING_AVG_WINDOW)
    debug_log(f"TensorBoard 日志将保存在: {log_dir}")

    # 初始化 SwanLab
    swanlab_config = {
        "SEED": SEED,  # 新增：记录种子
        "TASK_NAME": TASK_NAME,
        "BENCHMARK": BENCHMARK,
        "NUM_TRAINER_GPUS": NUM_TRAINER_GPUS,
        "NUM_INFERENCE_ACTORS": NUM_INFERENCE_ACTORS,
        "NUM_ROLLOUT_WORKERS": NUM_ROLLOUT_WORKERS,
        "NUM_EVAL_WORKERS": NUM_EVAL_WORKERS,
        "ROLLOUT_LOCAL_BUF": ROLLOUT_LOCAL_BUF,
        "INFERENCE_BATCH": INFERENCE_BATCH,
        "INFERENCE_TIMEOUT_MS": INFERENCE_TIMEOUT_MS,
        "REPLAY_CAPACITY": REPLAY_CAPACITY,
        "TRAIN_BATCH_SIZE": TRAIN_BATCH_SIZE,
        "ACCUMULATION_STEPS": ACCUMULATION_STEPS,
        "TRAIN_ITERS": TRAIN_ITERS,
        "CKPT_DIR": CKPT_DIR,
        "CKPT_EVERY_STEPS": CKPT_EVERY_STEPS,
        "GAMMA": GAMMA,
        "LAMBDA": LAMBDA,
        "CLIP_EPS": CLIP_EPS,
        "VF_COEF": VF_COEF,
        "ENT_COEF": ENT_COEF,
        "KL_COEF": KL_COEF,
        "CLIP_MODE": CLIP_MODE,
        "CLIP_PARAMS": CLIP_PARAMS,
        "REWARD_SCALE": REWARD_SCALE,
        "VALUE_LR": VALUE_LR,
        "POLICY_LR": POLICY_LR,
        "VALUE_WARMUP_STEPS": VALUE_WARMUP_STEPS,
        "POLICY_WARMUP_STEPS": POLICY_WARMUP_STEPS,
        "POLICY_TRAIN_START_STEP": POLICY_TRAIN_START_STEP,
        "MOVING_AVG_WINDOW": MOVING_AVG_WINDOW,
        "LOG_INTERVAL_SECONDS": LOG_INTERVAL_SECONDS,
        "STATE_DIM": STATE_DIM,
        "ACTION_DIM": ACTION_DIM,
        "N_ACTION_BINS": N_ACTION_BINS,
    }
    swanlab.init(
        project="MetaWorld-PPO-Benchmark",
        experiment_name=f"{BENCHMARK}_MLP_DS_PPO_add_vtrace_{CLIP_MODE}_{int(time.time())}_seed{SEED}",
        description=f"MetaWorld MLP DeepSpeed PPO Training with vtrace - {BENCHMARK} (Seed: {SEED})",
        config=swanlab_config,
    )
    # zzq1211 保存当前运行代码
    def save_and_log_code_to_swanlab(log_dir: str, seed: int = None):
        """
        保存当前运行的代码到指定目录并上传到SwanLab
        
        Args:
            log_dir: 日志目录路径
            seed: 随机种子，用于文件名
        """
        import os
        import shutil
        from datetime import datetime
        
        try:
            # 确保日志目录存在
            os.makedirs(log_dir, exist_ok=True)
            
            # 获取当前脚本的绝对路径
            current_script = os.path.abspath(__file__)
            script_name = os.path.basename(current_script)
            
            # 生成带时间戳和种子的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            seed_suffix = f"_seed{seed}" if seed is not None else ""
            code_txt_path = os.path.join(log_dir, f"code_{timestamp}{seed_suffix}.py")
            
            # 读取当前脚本内容
            with open(current_script, "r", encoding="utf-8") as f:
                code_content = f.read()
            
            # 写入到日志目录
            with open(code_txt_path, "w", encoding="utf-8") as f:
                f.write(code_content)
            
            print(f"✅ 已保存源代码到: {code_txt_path}")
            
            # 尝试上传到SwanLab
            try:
                # 先检查swanlab是否已初始化
                import swanlab
                if hasattr(swanlab, 'run') and swanlab.run is not None:
                    # 创建一个更美观的代码文件格式
                    code_py_path = os.path.join(log_dir, f"run_code{seed_suffix}.py")
                    with open(code_py_path, "w", encoding="utf-8") as f:
                        # 添加文件头注释
                        f.write(f"# Code snapshot for experiment (Seed: {seed if seed else 'N/A'})\n")
                        f.write(f"# Saved at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"# Original file: {script_name}\n")
                        f.write("#" * 80 + "\n\n")
                        f.write(code_content)
                    
                    # 上传到SwanLab
                    swanlab.log_artifact(
                        code_py_path,
                        name=f"source_code_seed{seed}" if seed else "source_code",
                        type="code"
                    )
                    print(f"✅ 源代码已作为artifact上传至SwanLab: {code_py_path}")
                    
                    # 可选：删除临时文件
                    os.remove(code_py_path)
                else:
                    print("⚠️  SwanLab未初始化，跳过artifact上传")
            except ImportError:
                print("⚠️  SwanLab未安装，跳过artifact上传")
            except Exception as e:
                print(f"⚠️  SwanLab上传失败: {e}")
                
        except FileNotFoundError:
            print(f"❌ 找不到脚本文件: {current_script}")
        except PermissionError:
            print(f"❌ 权限不足，无法写入目录: {log_dir}")
        except Exception as e:
            print(f"❌ 保存代码失败: {e}")
            import traceback
            traceback.print_exc()
    # ✅ 新增：保存并上传当前代码
    try:
        save_and_log_code_to_swanlab(log_dir, seed=SEED)
    except Exception as e:
        print(f"⚠️ 保存代码时出现警告: {e}")
    print("--- 步骤 1: 创建 Actors ---")
    replay_buffers = [ReplayBufferActor.remote(capacity=REPLAY_CAPACITY, seed=SEED) for _ in range(NUM_TRAINER_GPUS)]
    trainer_group = [
        TrainerActor.remote(rank=i, world_size=NUM_TRAINER_GPUS, replay_buffer=replay_buffers[i], seed=SEED)
        for i in range(NUM_TRAINER_GPUS)
    ]
    inference_pool = [InferenceActor.remote(actor_id=i, stats_actor=stats_actor, seed=SEED) for i in range(NUM_INFERENCE_ACTORS)]
    rollout_workers = [
        RolloutWorkerActor.remote(
            inference_pool[i % NUM_INFERENCE_ACTORS],
            replay_buffers[i % NUM_TRAINER_GPUS], i, stats_actor,
        ) for i in range(NUM_ROLLOUT_WORKERS)
    ]
    # eval_workers = [
    #     EvaluationWorkerActor.remote(
    #         inference_pool[i % NUM_INFERENCE_ACTORS], f"eval_{i}", stats_actor
    #     ) for i in range(NUM_EVAL_WORKERS)
    # ]
    eval_workers = [
    EvaluationWorkerActor.remote(
        inference_pool[i % NUM_INFERENCE_ACTORS], i, stats_actor  # 传递整数 i
    ) for i in range(NUM_EVAL_WORKERS)
]
    print(f"已创建 {NUM_ROLLOUT_WORKERS} 个 Rollout workers 和 {NUM_EVAL_WORKERS} 个 Evaluation workers。")

    print("\n--- 步骤 2: 建立独立的 DeepSpeed 训练组 ---")
    print("正在查找空闲端口...")
    train_group_port = find_free_port()
    print(f"训练组端口: {train_group_port}")

    broadcast_group_port = find_free_port()
    while broadcast_group_port == train_group_port:
        broadcast_group_port = find_free_port()
    print(f"广播组端口: {broadcast_group_port}")

    print("获取训练器主节点地址...")
    trainer_master_addr = ray.get(trainer_group[0].get_node_ip.remote())
    print(f"训练器主节点地址: {trainer_master_addr}")

    print("开始初始化 DeepSpeed 训练组...")
    train_setup_tasks = [actor.setup_deepspeed_group.remote(trainer_master_addr, train_group_port) for actor in trainer_group]
    print(f"创建了 {len(train_setup_tasks)} 个训练器设置任务")

    try:
        print("等待 DeepSpeed 初始化完成...")
        ray.get(train_setup_tasks, timeout=300)  # 5分钟超时
        print("DeepSpeed 训练组建立完成。")

        # 获取并记录参数量
        n_total_params, n_trainable_params = ray.get(trainer_group[0].get_parameter_counts.remote())
        print(f"模型总参数量: {n_total_params:,}, 可训练参数量: {n_trainable_params:,}")
        try:
            swanlab.config.update({
                "total_params": n_total_params,
                "trainable_params": n_trainable_params
            })
        except Exception as e:
            print(f"SwanLab config update failed: {e}")

    except Exception as e:
        print(f"DeepSpeed 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        raise

    print(f"\n--- 步骤 3: 建立共享广播组 ({BROADCAST_GROUP_NAME}) ---")
    broadcast_participants = [trainer_group[0]] + inference_pool
    broadcast_group_world_size = len(broadcast_participants)
    broadcast_master_addr = ray.get(trainer_group[0].get_node_ip.remote())
    broadcast_setup_tasks = [
        actor.setup_broadcast_group.remote(
            master_addr=broadcast_master_addr, master_port=broadcast_group_port,
            group_name=BROADCAST_GROUP_NAME, group_world_size=broadcast_group_world_size,
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
    if len(train_sig) != len(infer_sig):
        raise RuntimeError(f"训练器与推理器的广播签名长度不匹配: {len(train_sig)} vs {len(infer_sig)}")
    for i, (a, b) in enumerate(zip(train_sig, infer_sig)):
        if a != b:
            raise RuntimeError(f"First mismatch at idx: {i}, trainer: {a}, inference: {b}")
    print("广播签名验证通过。")

    broadcast_task = trainer_group[0].broadcast_weights.remote(BROADCAST_GROUP_NAME)
    receive_tasks = [inf.receive_and_update_weights.remote(BROADCAST_GROUP_NAME) for inf in inference_pool]
    ray.get([broadcast_task] + receive_tasks)
    print("初始权重已广播到所有推理器。")

    print("\n--- 步骤 4: 启动 Rollout Workers 进行数据收集 ---")
    for w in rollout_workers: w.run.remote()
    for w in eval_workers: w.run.remote()

    print("\n--- 步骤 5: 等待远程经验池填充初始数据 ---")
    min_buffer_size_for_start = TRAIN_BATCH_SIZE * ACCUMULATION_STEPS
    assert min_buffer_size_for_start < REPLAY_CAPACITY, "初始填充量必须小于回放池总容量"
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
    while global_step < TRAIN_ITERS:
        t_train_start = time.time()
        train_tasks = [trainer.run_training_epoch.remote() for trainer in trainer_group]
        results = ray.get(train_tasks)
        _, _, _, _, _, _, global_step, _, _, _, _ = results[0]
        train_time = time.time() - t_train_start

        t_sync_start = time.time()
        broadcast_task = trainer_group[0].broadcast_weights.remote(BROADCAST_GROUP_NAME)
        receive_tasks = [inf.receive_and_update_weights.remote(BROADCAST_GROUP_NAME) for inf in inference_pool]
        ray.get([broadcast_task] + receive_tasks)
        sync_time = time.time() - t_sync_start

        if global_step > 0 and global_step % CKPT_EVERY_STEPS == 0:
            ray.get(trainer_group[0].save_agent.remote(CKPT_DIR, global_step))

        current_time = time.time()
        if current_time - last_log_time > LOG_INTERVAL_SECONDS:
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

            total_losses, p_losses, v_losses, e_losses, kl_losses, lrs_list, _, ents, avg_kl_divs, clip_ratios, perf_metrics_list = zip(*results)
            current_lrs = lrs_list[0]

            elapsed_time = current_time - start_time
            total_buffer_size = sum(ray.get([rb.size.remote() for rb in replay_buffers]))

            print(f"更新步 {global_step}/{TRAIN_ITERS} | 时间: {elapsed_time:.1f}s | "
                  f"全局平均奖励: {avg_return:.2f} | 全局平均幕长: {avg_ep_len:.1f} | Eval奖励: {eval_avg_return:.2f} | "
                  f"value loss: {np.mean(v_losses):.4f} | LR(V/P): {current_lrs['value']:.7f}/{current_lrs['policy']:.7f} | "
                  f"Episodes数量: {total_episodes:,} | Step平均时间: {avg_step_time:.3f}s")

            log_metrics = {}
            log_metrics['Train/Learning_Rate/Value'] = current_lrs['value']
            log_metrics['Train/Learning_Rate/Policy'] = current_lrs['policy']
            log_metrics['Loss/Total'] = np.mean(total_losses)
            log_metrics['Loss/Policy'] = np.mean(p_losses)
            log_metrics['Loss/Value'] = np.mean(v_losses)
            log_metrics['Loss/Entropy'] = np.mean(e_losses)
            log_metrics['Loss/KL'] = np.mean(kl_losses)

            log_metrics['Metrics/Entropy'] = np.mean(ents)
            log_metrics['Metrics/KL_Divergence'] = np.mean(avg_kl_divs)
            log_metrics['Metrics/Ineffective_Data_Ratio'] = np.mean(clip_ratios)  # 无效数据比例（梯度为 0 的样本比例，在 backward 后检查 ratio 的梯度）
            log_metrics['Metrics/Training_Speed_Steps_per_Sec'] = training_speed_steps_per_sec
            
            for metric_name, metric_value in timing_stats.items():
                log_metrics[f'Performance/{metric_name}'] = metric_value

            avg_policy_sample_time = np.mean([pm["policy_sample_time"] for pm in perf_metrics_list])
            avg_policy_prep_time = np.mean([pm["policy_prep_time"] for pm in perf_metrics_list])
            avg_policy_train_time = np.mean([pm["policy_train_time"] for pm in perf_metrics_list])
            
            log_metrics['Performance/policy_sample_time'] = avg_policy_sample_time
            log_metrics['Performance/policy_prep_time'] = avg_policy_prep_time
            log_metrics['Performance/policy_train_time'] = avg_policy_train_time
            log_metrics['Performance/train_time'] = train_time
            log_metrics['Performance/sync_time'] = sync_time
            log_metrics['Performance/train_time_total'] = time.time() - t_train_start

            log_metrics['Rollout/Average_Return'] = avg_return
            log_metrics['Rollout/Average_Episode_Length'] = avg_ep_len
            log_metrics['Eval/Average_Return'] = eval_avg_return
            log_metrics['Eval/Average_Episode_Length'] = eval_avg_ep_len

            log_metrics['System/Replay_Buffer_Size_Total'] = total_buffer_size
            log_metrics['System/Total_Episodes_Processed'] = total_episodes
            log_metrics['System/Total_Env_Steps'] = total_env_steps
            log_metrics['System/Avg_Step_Time'] = avg_step_time
            log_metrics['System/Eval_Total_Episodes_Processed'] = eval_total_episodes
            log_metrics['System/Eval_Total_Env_Steps'] = eval_env_steps
            log_metrics['System/Eval_Avg_Step_Time'] = eval_avg_step_time
            log_metrics['System/Active_Rollout_Actors'] = global_stats.get("active_actor_count", 0)
            log_metrics['System/Total_Samples_Produced'] = global_stats.get("total_samples_produced", 0)

            for env_name, env_stats in all_stats.items():
                if env_name.startswith("eval_"):
                    tag_prefix = f"Eval/{env_name.replace('eval_', '')}"
                    log_metrics[f'{tag_prefix}/Average_Return'] = env_stats['avg_return']
                    log_metrics[f'{tag_prefix}/Average_Episode_Length'] = env_stats['avg_ep_len']
                    log_metrics[f'{tag_prefix}/Success_Rate'] = env_stats['avg_success_rate']
                    log_metrics[f'{tag_prefix}/Total_Episodes'] = env_stats['total_episodes']
                else:
                    tag_prefix = f"Rollout/{env_name}"
                    log_metrics[f'{tag_prefix}/Average_Return'] = env_stats['avg_return']
                    log_metrics[f'{tag_prefix}/Average_Episode_Length'] = env_stats['avg_ep_len']
                    log_metrics[f'{tag_prefix}/Success_Rate'] = env_stats['avg_success_rate']
                    log_metrics[f'{tag_prefix}/Total_Episodes'] = env_stats['total_episodes']

            for k, v in log_metrics.items():
                writer.add_scalar(k, v, global_step)
            swanlab.log(log_metrics, step=global_step)

            last_log_time = current_time
            last_log_global_step = global_step

    print(f"\n成功完成 {TRAIN_ITERS} 次训练与同步循环！")
    writer.close()
    ray.shutdown()


if __name__ == "__main__":
    main()