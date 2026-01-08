import os
os.environ["MUJOCO_GL"] = "osmesa"           # 强制软件渲染
os.environ["PYOPENGL_PLATFORM"] = "osmesa"   # 保险起见，给 PyOpenGL 也指明
# 设置临时文件目录，避免磁盘I/O瓶颈
os.environ["TMPDIR"] = "/dev/shm"
# 为了让 Ray 能看到所有可用的 GPU，我们在脚本开头设置。
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# 防止 transformers 库的 tokenizer 并行化警告
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NCCL_SOCKET_IFNAME"] = os.environ.get("NCCL_SOCKET_IFNAME", "eth0")
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/dev/shm/torch_ext")
os.environ.setdefault("DS_BUILD_FUSED_ADAM", "0")
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
    import yaml
except ImportError:
    yaml = None

# OpenVLA 组件与常量
# zzq1120 单独从openvla_utils取出这两个方法
from experiments.robot.sole_utils import (
    get_processor,
)

from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM
from experiments.robot.libero.libero_utils import GenerateConfig, TaskSuite
from rl.actor_critic_model_discrete import ActorCritic
from rl.utils import prepare_one_obs
from rl.libero_env import LiberoEnvWrapper
# 训练/推理通信（保持接口不变）
from ds_com import TrainerActorCom, InferenceActorCom
from rl.com_utils import find_free_port

# ================================================================
# 0. 超参数与配置
# ================================================================
# Libero benchmark
BENCHMARK = TaskSuite.LIBERO_OBJECT

# 分布式系统参数
NUM_TRAINER_GPUS = 2
NUM_INFERENCE_ACTORS = 1
NUM_ROLLOUT_WORKERS = 2
NUM_EVAL_WORKERS = 20
ROLLOUT_LOCAL_BUF = 64
INFERENCE_BATCH = 8
INFERENCE_TIMEOUT_MS = 300
REPLAY_CAPACITY = 10000
TRAIN_BATCH_SIZE = 12
ACCUMULATION_STEPS = 21
TRAIN_ITERS = 30000

# Checkpoint
CKPT_DIR = f"/cpfs01/liuwei_workspace/models/finetune_rl"
CKPT_EVERY_STEPS = 2000000   # 每 N 个训练步保存一次

# PPO
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
VF_COEF = 0.5
ENT_COEF = 0.00
KL_COEF = 0.1

# 奖励缩放
REWARD_SCALE = 1.0

# ================================================================
# 学习率调度参数
# ================================================================
VALUE_LR = 1e-4
POLICY_LR = 1e-5
VALUE_WARMUP_STEPS = 500
POLICY_WARMUP_STEPS = 500
POLICY_TRAIN_START_STEP = 0 # 策略网络从第500个 *更新步* 开始训练

# 日志
MOVING_AVG_WINDOW = 1000
LOG_INTERVAL_SECONDS = 10

# 通信组
BROADCAST_GROUP_NAME = "trainer_to_inference_broadcast"

# 裁剪模式配置
CLIP_MODE = "clip"
CLIP_PARAMS: Dict[str, Any] = {}

# OpenVLA 加载配置
USE_BF16: bool = True
TORCH_DTYPE = torch.bfloat16 if USE_BF16 else torch.float32
PRETRAINED_CHECKPOINT = "tmp_libero_10_last_checkpoint_Nov_15+libero_10_no_noops+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--discrete_acts--proprio_state"


def parse_args(args: Optional[List[str]] = None):
    """
    允许通过命令行覆盖关键超参，保持默认值与当前脚本一致。
    算法逻辑不变，仅用于灵活配置资源与训练规模。
    """
    parser = argparse.ArgumentParser(description="Libero OpenVLA DeepSpeed PPO 离散动作训练脚本")
    parser.add_argument("--cuda-visible-devices", dest="cuda_visible_devices", type=str,
                        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"),
                        help="CUDA_VISIBLE_DEVICES 环境变量设置")
    parser.add_argument("--num-trainer-gpus", dest="num_trainer_gpus", type=int,
                        default=NUM_TRAINER_GPUS, help="TrainerActor 的 GPU 数量")
    parser.add_argument("--num-inference-actors", dest="num_inference_actors", type=int,
                        default=NUM_INFERENCE_ACTORS, help="InferenceActor 数量")
    parser.add_argument("--num-rollout-workers", dest="num_rollout_workers", type=int,
                        default=NUM_ROLLOUT_WORKERS, help="Rollout worker 数量")
    parser.add_argument("--num-eval-workers", dest="num_eval_workers", type=int,
                        default=NUM_EVAL_WORKERS, help="Evaluation worker 数量")
    parser.add_argument("--rollout-local-buf", dest="rollout_local_buf", type=int,
                        default=ROLLOUT_LOCAL_BUF, help="Rollout worker 本地缓冲长度")
    parser.add_argument("--inference-batch", dest="inference_batch", type=int,
                        default=INFERENCE_BATCH, help="推理批大小")
    parser.add_argument("--inference-timeout-ms", dest="inference_timeout_ms", type=int,
                        default=INFERENCE_TIMEOUT_MS, help="推理批次超时时间(ms)")
    parser.add_argument("--train-batch-size", dest="train_batch_size", type=int,
                        default=TRAIN_BATCH_SIZE, help="训练微批大小")
    parser.add_argument("--accumulation-steps", dest="accumulation_steps", type=int,
                        default=ACCUMULATION_STEPS, help="梯度累积步数")
    parser.add_argument("--train-iters", dest="train_iters", type=int,
                        default=TRAIN_ITERS, help="训练迭代次数")
    parser.add_argument("--replay-capacity", dest="replay_capacity", type=int,
                        default=REPLAY_CAPACITY, help="经验回放容量")
    parser.add_argument("--benchmark", dest="benchmark", type=str,
                        default="LIBERO_SPATIAL", help="Libero benchmark 名称（TaskSuite 枚举名）")
    parser.add_argument("--clip-mode", dest="clip_mode", type=str, default=CLIP_MODE,
                        help="策略裁剪模式（clip / soft_clip / soft_clip_alpha-1 / soft_clip_alpha-2 / sapo / ce-gppo_clip / cispo 等）")
    parser.add_argument("--clip-config", dest="clip_config", type=str, default=None,
                        help="裁剪配置文件路径（YAML/JSON），按 clip_mode 选择对应超参")
    parser.add_argument("--pretrained-checkpoint", dest="pretrained_checkpoint", type=str,
                        default=PRETRAINED_CHECKPOINT, help="OpenVLA 预训练 checkpoint 路径")
    parser.add_argument("--experiment-name", dest="experiment_name", type=str,
                        default=None, help="实验名称（用于 SwanLab/TensorBoard 前缀）")
    return parser.parse_args(args=args)


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

    # ===== 新增：SAPO soft clip（sigmoid gate）=====
    if clip_mode in ("sapo_soft_clip", "sapo", "sapo_gate"):
        # τ 的非对称设置：通常 τ_neg > τ_pos（负优势更“硬”一点）
        tau_pos = float(clip_params.get("tau_pos", 1.0))
        tau_neg = float(clip_params.get("tau_neg", 2.0))
        if tau_pos <= 0 or tau_neg <= 0:
            raise ValueError(f"tau_pos/tau_neg must be > 0, got {tau_pos}, {tau_neg}")

        # 数值稳定：避免 ratio 极端导致 inf（可按需调大/关掉）
        ratio_min = float(clip_params.get("ratio_min", 1e-6))
        ratio_max = float(clip_params.get("ratio_max", 1e6))
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

        # “clip_ratio”在 SAPO 里没有硬裁剪；这里给一个可选的“饱和比例”指标
        # 当 w = 4*p*(1-p) 很小，说明 sigmoid 饱和、更新被强烈抑制（类似“被clip了”）
        # p = torch.sigmoid(x)
        # w = 4.0 * p * (1.0 - p)
        # w_thresh = float(clip_params.get("sapo_w_thresh", 0.05))
        # clip_ratio = (w < w_thresh).float().mean().item()
        clip_ratio = 0.0
        return policy_loss, 0.0 # clip_ratio

    if clip_mode == "ce-gppo_clip":
        # 对齐 CE-GPPO/GPPO repo 的 general_beta 写法：三种 case
        # low_mask: (ratio < 1-eps) & (A < 0)
        # high_mask:(ratio > 1+eps) & (A > 0)
        beta1 = float(clip_params.get("beta1", 0.75))
        beta2 = float(clip_params.get("beta2", 1.0))

        eps = float(clip_params.get("clip_eps", CLIP_EPS))
        low = 1.0 - eps
        high = 1.0 + eps

        # 只统计“会被 PPO clip 的两类 token”
        low_mask = (ratio < low) & (adv_unsqueezed < 0)
        high_mask = (ratio > high) & (adv_unsqueezed > 0)
        other_mask = ~(low_mask | high_mask)

        ratio_det = ratio.detach().clamp_min(1e-8)

        eff_ratio = torch.empty_like(ratio)
        # 关键：forward = 常数(beta*(1±eps))，backward 通过 ratio/ratio_det 保留梯度
        eff_ratio[low_mask] = beta1 * low / ratio_det[low_mask] * ratio[low_mask]
        eff_ratio[high_mask] = beta2 * high / ratio_det[high_mask] * ratio[high_mask]
        eff_ratio[other_mask] = ratio[other_mask]

        surr = eff_ratio * adv_unsqueezed
        policy_loss = -torch.mean(surr)

        # clip_ratio = (low_mask | high_mask).float().mean().item()
        clip_ratio = 0.0
        return policy_loss, clip_ratio
    # ===== 新增：CISPO（MiniMax-M1）IS-weight clip（非 PPO 的 token clip/mask）=====
    # 论文形式： r_hat = clip(r, 1-eps_low^IS, 1+eps_high^IS), 目标 ~ sg(r_hat) * A * log pi
    # 这里在只有 ratio 的实现中，用 “forward=常数，backward=ratio” 的 trick 达到同样梯度：
    # eff_ratio = (r_hat.detach() / ratio.detach()) * ratio  =>  grad ~ r_hat.detach() * A * ∇logπ
    if clip_mode in ("cispo", "cispo_clip", "is_clip", "is_weight_clip"):
        eps_is_low = clip_params.get("eps_is_low", None)   # None 表示不做下界（等价 low≈0）
        eps_is_high = float(clip_params.get("eps_is_high", clip_params.get("clip_eps", CLIP_EPS)))

        if eps_is_high < 0:
            raise ValueError(f"eps_is_high must be >= 0, got {eps_is_high}")
        if eps_is_low is not None and float(eps_is_low) < 0:
            raise ValueError(f"eps_is_low must be >= 0, got {eps_is_low}")

        # 非对称区间： [1-eps_is_low, 1+eps_is_high]
        high = 1.0 + eps_is_high
        if eps_is_low is None:
            low = 0.0  # ratio>0，基本等价于“只裁上界”
        else:
            low = 1.0 - float(eps_is_low)
            low = max(0.0, low)

        # 数值稳定（可选）
        ratio_min = float(clip_params.get("ratio_min", 1e-8))
        ratio_max = float(clip_params.get("ratio_max", 1e8))
        r = ratio.clamp(ratio_min, ratio_max)

        # r_hat: forward 里就是被裁剪的 IS 权重
        r_hat = r.clamp(low, high)

        # “forward 常数 + backward 走 ratio”的 CISPO trick
        r_det = r.detach().clamp_min(ratio_min)
        eff_ratio = (r_hat.detach() / r_det) * r

        surr = eff_ratio * adv_unsqueezed
        policy_loss = -torch.mean(surr)

        # 这里的 clip_ratio 表示“有多少样本的 IS 权重被截到了边界”（并不代表无梯度）
        is_clipped = (r < low) | (r > high)
        # clip_ratio = is_clipped.float().mean().item()
        clip_ratio = 0.0
        return policy_loss, clip_ratio
    raise ValueError(f"Unsupported clip mode: {clip_mode}")


def load_clip_config(config_path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if config_path is None:
        return {}
    try:
        with open(config_path, "r") as f:
            if yaml is not None:
                cfg = yaml.safe_load(f)
            else:
                cfg = json.load(f)
        if not isinstance(cfg, dict):
            print(f"[Warn] clip_config {config_path} 内容不是 dict，使用空配置。")
            return {}
        return cfg
    except Exception as e:
        print(f"[Warn] 加载 clip_config 失败: {e}，使用空配置。")
        return {}


def select_clip_params(clip_mode: str, clip_config: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    return clip_config.get(clip_mode, {})


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
    # 新增：用于诊断旧数据的元信息（不影响算法逻辑）
    policy_version: int = 0
    insert_step: int = 0

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
    def __init__(self, capacity=REPLAY_CAPACITY):
        self.buffer = deque(maxlen=capacity)
        print(f"[DEBUG] ReplayBufferActor: 初始化完成，容量={capacity}", flush=True)
        # 新增：记录插入步数，便于“旧数据”诊断
        self.insert_counter = 0

    def add_batch(self, batch: List[Experience]):
        old_size = len(self.buffer)
        # 为每个样本填充 insert_step（仅用于指标，不影响算法逻辑）
        for i, exp in enumerate(batch):
            try:
                exp.insert_step = self.insert_counter + i
            except Exception:
                # 容错：即便意外失败也不影响训练
                pass
        self.insert_counter += len(batch)
        self.buffer.extend(batch)
        new_size = len(self.buffer)
        print(f"[DEBUG] ReplayBufferActor: add_batch 收到 {len(batch)} 个经验，缓冲区大小: {old_size} -> {new_size}", flush=True)

    def size(self):
        current_size = len(self.buffer)
        if current_size % 10 == 0 or current_size == 0:  # 每10个或为0时打印
            print(f"[DEBUG] ReplayBufferActor: size() 被调用，当前大小={current_size}", flush=True)
        return current_size
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # obs 是 prepare_one_obs 的字典，不能 stack，保持 list 返回
        obs_list = [b.obs for b in batch]
        action_token = np.stack([b.action_token for b in batch])
        adv = np.asarray([b.advantage for b in batch], np.float32)
        logits_old = np.stack([b.behaviour_logits for b in batch])
        v_targ = np.asarray([b.value_target for b in batch], np.float32)
        insert_step = np.asarray([getattr(b, "insert_step", 0) for b in batch], np.int32)
        policy_version = np.asarray([getattr(b, "policy_version", 0) for b in batch], np.int32)
        return obs_list, action_token, adv, logits_old, v_targ, insert_step, policy_version

class BaseWorkerActor:
    """rollout 和 eval worker 的共享逻辑。"""
    def __init__(self, infer, replay, wid, stats_actor, cfg, benchmark_name=BENCHMARK):
        print(f"[DEBUG] BaseWorkerActor {wid}: __init__ 开始，infer={infer}, replay={replay}, benchmark_name={benchmark_name}", flush=True)
        self.infer = infer
        self.replay = replay
        self.stats_actor = stats_actor
        self.cfg = cfg
        # 仅需 processor，Worker 不加载大模型
        print(f"[DEBUG] BaseWorkerActor {wid}: 准备获取 processor...", flush=True)
        self.processor = get_processor(cfg)
        print(f"[DEBUG] BaseWorkerActor {wid}: processor 获取完成，时间={time.time():.2f}", flush=True)
        self.benchmark_name = benchmark_name

        self.num_tasks = 1
        print(f"[DEBUG] BaseWorkerActor {wid}: 准备初始化 {self.num_tasks} 个 Libero 环境，benchmark_name={self.benchmark_name}，时间={time.time():.2f}", flush=True)
        self.envs = []
        for i in range(self.num_tasks):
            try:
                t_env_start = time.time()
                print(f"[DEBUG] BaseWorkerActor {wid}: 【环境初始化 {i}/{self.num_tasks}】开始创建 LiberoEnvWrapper，时间={t_env_start:.2f}", flush=True)
                print(f"[DEBUG] BaseWorkerActor {wid}:   - benchmark_name: {self.benchmark_name}", flush=True)
                print(f"[DEBUG] BaseWorkerActor {wid}:   - task_id: {i}", flush=True)
                print(f"[DEBUG] BaseWorkerActor {wid}:   - image_size: 224", flush=True)
                print(f"[DEBUG] BaseWorkerActor {wid}:   - render_mode: rgb_array", flush=True)
                
                env = LiberoEnvWrapper(
                    benchmark_name=self.benchmark_name,
                    task_id=i,
                    image_size=224,
                    render_mode="rgb_array"
                )
                t_env_end = time.time()
                
                self.envs.append(env)
                print(f"[DEBUG] BaseWorkerActor {wid}: 【环境初始化 {i}/{self.num_tasks}】完成，耗时: {t_env_end - t_env_start:.2f}s", flush=True)
            except Exception as e:
                print(f"[ERROR] BaseWorkerActor {wid}: 环境 {i} 初始化失败: {e}", flush=True)
                import traceback
                traceback.print_exc()
                raise
        print(f"[DEBUG] BaseWorkerActor {wid}: 所有环境初始化完成，共 {len(self.envs)} 个环境", flush=True)
        
        self.env = None
        self.current_env_idx = -1
        self.wid = wid
        self.task_description = None
        self.current_env_name = None
        print(f"[DEBUG] BaseWorkerActor {wid}: __init__ 完成", flush=True)

@ray.remote
class RolloutWorkerActor(BaseWorkerActor):
    def __init__(self, infer, replay, wid, stats_actor, cfg, benchmark_name=BENCHMARK):
        print(f"[DEBUG] RolloutWorkerActor {wid}: __init__ 开始，infer={infer}, replay={replay}", flush=True)
        super().__init__(infer, replay, wid, stats_actor, cfg, benchmark_name)
        self.env_outcome = [deque(maxlen=100) for _ in range(self.num_tasks)]
        self.local_buffer = []
        print(f"[DEBUG] RolloutWorkerActor {wid}: __init__ 完成，self.infer={self.infer}, self.replay={self.replay}", flush=True)

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
            print(f"[DEBUG] RolloutWorker {self.wid}: ============= run() 开始执行 =============", flush=True)
            print(f"[DEBUG] RolloutWorker {self.wid}: 推理器引用: {self.infer}", flush=True)
            print(f"[DEBUG] RolloutWorker {self.wid}: 回放缓冲区引用: {self.replay}", flush=True)
            print(f"[DEBUG] RolloutWorker {self.wid}: 环境数量: {len(self.envs)}", flush=True)
            print(f"[DEBUG] RolloutWorker {self.wid}: Processor 类型: {type(self.processor)}", flush=True)
            
            current_seed = int(time.time() * 1000) + self.wid + os.getpid()
            print(f"[DEBUG] RolloutWorker {self.wid}: 【步骤1】准备重置环境，seed={current_seed}, 时间={time.time():.2f}", flush=True)
            
            t_reset_start = time.time()
            obs, info = self._reset_and_select_env(seed=current_seed)
            t_reset_end = time.time()
            
            print(f"[DEBUG] RolloutWorker {self.wid}: 【步骤1完成】环境重置完成，耗时: {t_reset_end - t_reset_start:.2f}s", flush=True)
            print(f"[DEBUG] RolloutWorker {self.wid}:   - 当前环境: {self.current_env_name}", flush=True)
            print(f"[DEBUG] RolloutWorker {self.wid}:   - 任务描述: {self.task_description[:80] if self.task_description else 'None'}...", flush=True)
            print(f"[DEBUG] RolloutWorker {self.wid}:   - obs keys: {obs.keys() if isinstance(obs, dict) else type(obs)}", flush=True)
            
            reward_sum, time_start, step_count_total = 0.0, time.time(), 0
            step_iteration = 0
            
            print(f"[DEBUG] RolloutWorker {self.wid}: 【步骤2】进入主循环，时间={time.time():.2f}", flush=True)
            
            while True:
                step_iteration += 1
                if step_iteration <= 3 or step_iteration % 10 == 0:
                    print(f"[DEBUG] RolloutWorker {self.wid}: --- 第 {step_iteration} 步开始 ---", flush=True)
                
                try:
                    if step_iteration == 1:
                        print(f"[DEBUG] RolloutWorker {self.wid}: 【步骤2.1】准备 prepare_one_obs，时间={time.time():.2f}", flush=True)
                    
                    t_prepare_start = time.time()
                    inputs_t = prepare_one_obs(self.cfg, self.processor, obs, self.task_description, TORCH_DTYPE)
                    t_prepare_end = time.time()
                    
                    if step_iteration == 1:
                        print(f"[DEBUG] RolloutWorker {self.wid}: 【步骤2.1完成】prepare_one_obs 完成，耗时: {t_prepare_end - t_prepare_start:.2f}s", flush=True)
                        print(f"[DEBUG] RolloutWorker {self.wid}:   - inputs_t keys: {inputs_t.keys()}", flush=True)
                        print(f"[DEBUG] RolloutWorker {self.wid}:   - inputs_t shapes: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in inputs_t.items()]}", flush=True)
                    
                    if step_iteration == 1:
                        print(f"[DEBUG] RolloutWorker {self.wid}: 【步骤2.2】准备发送推理请求，时间={time.time():.2f}", flush=True)
                        print(f"[DEBUG] RolloutWorker {self.wid}:   - 推理器地址: {self.infer}", flush=True)
                    
                    t_infer_req_start = time.time()
                    inference_future = self.infer.request.remote(inputs_t, deterministic=False)
                    t_infer_req_end = time.time()
                    
                    if step_iteration == 1:
                        print(f"[DEBUG] RolloutWorker {self.wid}: 【步骤2.2】推理请求已发送 (耗时: {t_infer_req_end - t_infer_req_start:.4f}s)，等待结果...", flush=True)
                    
                    t_get_start = time.time()
                    action_env, action_token, logits, value, policy_version = ray.get(inference_future)
                    t_get_end = time.time()
                    
                    if step_iteration == 1:
                        print(f"[DEBUG] RolloutWorker {self.wid}: 【步骤2.2完成】推理完成，等待耗时: {t_get_end - t_get_start:.2f}s", flush=True)
                        print(f"[DEBUG] RolloutWorker {self.wid}:   - action_env type: {type(action_env)}, len/shape: {len(action_env) if isinstance(action_env, list) else action_env.shape}", flush=True)
                        print(f"[DEBUG] RolloutWorker {self.wid}:   - value: {value}", flush=True)
                        print(f"[DEBUG] RolloutWorker {self.wid}:   - policy_version: {policy_version}", flush=True)
                except Exception as e:
                    print(f"[ERROR] RolloutWorker {self.wid}: 推理请求失败 (步骤 {step_iteration}): {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    raise
                
                if step_iteration == 1:
                    print(f"[DEBUG] RolloutWorker {self.wid}: 【步骤2.3】执行环境 step，时间={time.time():.2f}", flush=True)
                
                t_step_start = time.time()
                chunk_reward, done = 0.0, False
                for i in range(len(action_env)):
                    single_action = action_env[i]
                    nxt, r, term, trunc, info = self.env.step(single_action)
                    reward_sum += r
                    chunk_reward += r * REWARD_SCALE
                    step_count_total += 1
                    if term or trunc: done = True; break
                t_step_end = time.time()
                
                if step_iteration == 1:
                    print(f"[DEBUG] RolloutWorker {self.wid}: 【步骤2.3完成】环境 step 完成，耗时: {t_step_end - t_step_start:.4f}s", flush=True)
                    print(f"[DEBUG] RolloutWorker {self.wid}:   - chunk_reward: {chunk_reward:.4f}, done: {done}", flush=True)
                
                self.local_buffer.append((inputs_t, action_token, chunk_reward, logits, value, policy_version))
                obs = nxt
                
                if step_iteration == 1:
                    print(f"[DEBUG] RolloutWorker {self.wid}: 【步骤2.4】第1步完成，local_buffer大小: {len(self.local_buffer)}", flush=True)

                if done:
                    print(f"[DEBUG] RolloutWorker {self.wid}: Episode 结束，step_count_total={step_count_total}, reward_sum={reward_sum:.2f}, local_buffer大小={len(self.local_buffer)}", flush=True)
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
                    reward_sum = 0.0
                    if self.local_buffer:
                        print(f"[DEBUG] RolloutWorker {self.wid}: Episode结束，处理轨迹，local_buffer大小={len(self.local_buffer)}", flush=True)
                        self._process_traj(self.local_buffer, 0.0)
                    self.local_buffer.clear()
                    current_seed = int(time.time() * 1000) + self.wid + os.getpid()
                    obs, info = self._reset_and_select_env(seed=current_seed)
                    time_start, step_count_total = time.time(), 0
                    step_iteration = 0
                elif len(self.local_buffer) == ROLLOUT_LOCAL_BUF + 1:
                    print(f"[DEBUG] RolloutWorker {self.wid}: local_buffer达到上限({ROLLOUT_LOCAL_BUF + 1})，处理轨迹", flush=True)
                    _, _, _, _, bootstrap_val, _ = self.local_buffer[-1]
                    self._process_traj(self.local_buffer[:-1], bootstrap_val)
                    self.local_buffer = [self.local_buffer[-1]]
        except Exception as e: 
            import traceback
            print(f"[ERROR] RolloutWorker {self.wid} run() 崩溃: {e}", flush=True)
            traceback.print_exc()
            raise

    def _process_traj(self, traj_segment, bootstrap_val):
        print(f"[DEBUG] RolloutWorker {self.wid}: _process_traj 被调用，traj_segment长度={len(traj_segment)}, bootstrap_val={bootstrap_val}", flush=True)
        rets, advs = [], []
        gae = 0.0
        for i in reversed(range(len(traj_segment))):
            _, _, r, _, v, _ = traj_segment[i]
            nv = bootstrap_val if i == len(traj_segment) - 1 else traj_segment[i+1][4]
            delta = r + GAMMA * nv - v
            gae = delta + GAMMA * LAMBDA * gae
            advs.append(gae)
            rets.append(gae + v)
        advs.reverse(); rets.reverse()
        advs_np = np.array(advs, dtype=np.float32)

        batch: List[Experience] = []
        for i, (s, a_token, _, logits, _, policy_ver) in enumerate(traj_segment):
            batch.append(
                Experience(
                    obs=s,
                    action_token=a_token.astype(np.int64), # token 是整数
                    advantage=float(advs_np[i]),
                    behaviour_logits=logits.astype(np.float32),
                    value_target=float(rets[i]),
                    policy_version=int(policy_ver),
                )
            )
        print(f"[DEBUG] RolloutWorker {self.wid}: 准备添加 {len(batch)} 个经验到回放缓冲区，replay引用={self.replay}", flush=True)
        try:
            result = self.replay.add_batch.remote(batch)
            print(f"[DEBUG] RolloutWorker {self.wid}: add_batch.remote() 调用成功，返回future: {result}", flush=True)
            # 可选：等待一下确保调用成功
            # ray.get(result)  # 这会阻塞，但可以确保调用成功
        except Exception as e:
            print(f"[ERROR] RolloutWorker {self.wid}: add_batch.remote() 调用失败: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise

@ray.remote
class EvaluationWorkerActor(BaseWorkerActor):
    def __init__(self, infer, wid, stats_actor, cfg, benchmark_name=BENCHMARK):
        super().__init__(infer, None, wid, stats_actor, cfg, benchmark_name)
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
                    inputs_t = prepare_one_obs(self.cfg, self.processor, obs, self.task_description, TORCH_DTYPE)
                    action_env, _, _, _, _ = ray.get(self.infer.request.remote(inputs_t, deterministic=True))
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
@ray.remote(num_gpus=1)
class InferenceActor(InferenceActorCom):
    def __init__(self, actor_id, cfg, stats_actor):
        super().__init__()
        self.actor_id = actor_id
        print(f"InferenceActor {actor_id}: 正在加载 OpenVLA ActorCritic...")
        self.model = ActorCritic(cfg, torch_dtype=TORCH_DTYPE)
        self.model.cuda()
        self.model.eval()
        self.processor = self.model.processor
        self.cfg = cfg
        self.stats_actor = stats_actor
        # 跟踪推理侧的策略版本，仅用于诊断，不影响算法
        self.policy_version = 0

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

    async def request(self, inputs_t: Dict[str, torch.Tensor], deterministic: bool = False):
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        self.requests.append((inputs_t, deterministic))
        self.promises.append(fut)
        
        # 在接收到第一个请求时打印调试信息
        if len(self.requests) == 1 and self.actor_id == 0:
            print(f"[DEBUG] InferenceActor {self.actor_id}: 收到第一个推理请求，时间={time.time():.2f}", flush=True)
            print(f"[DEBUG] InferenceActor {self.actor_id}:   - inputs_t keys: {inputs_t.keys()}", flush=True)
            print(f"[DEBUG] InferenceActor {self.actor_id}:   - 当前请求队列长度: {len(self.requests)}", flush=True)
        
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
            
            # 第一次处理批次时打印详细信息
            is_first_batch = not hasattr(self, '_first_batch_processed')
            if is_first_batch:
                self._first_batch_processed = True
                print(f"[DEBUG] InferenceActor {self.actor_id}: 【第一次推理批次】开始处理，时间={time.time():.2f}", flush=True)
                print(f"[DEBUG] InferenceActor {self.actor_id}:   - 批次大小: {len(requests_to_process)}", flush=True)
                print(f"[DEBUG] InferenceActor {self.actor_id}:   - 设备: {next(self.model.parameters()).device}", flush=True)
            
            try:
                if is_first_batch:
                    print(f"[DEBUG] InferenceActor {self.actor_id}:   【步骤1】prepare_inputs_batch...", flush=True)
                
                t_prepare_batch = time.time()
                inputs_batch = self.model.prepare_inputs_batch(inputs_list)
                
                if is_first_batch:
                    print(f"[DEBUG] InferenceActor {self.actor_id}:   【步骤1完成】耗时: {time.time() - t_prepare_batch:.3f}s", flush=True)
                    print(f"[DEBUG] InferenceActor {self.actor_id}:   【步骤2】模型前向传播...", flush=True)
                
                with torch.inference_mode():
                    # 1. 前向传播获取 logits 和 value
                    t_forward = time.time()
                    action_logits, value = self.model(inputs_batch)
                    
                    if is_first_batch:
                        print(f"[DEBUG] InferenceActor {self.actor_id}:   【步骤2完成】前向传播耗时: {time.time() - t_forward:.3f}s", flush=True)
                        print(f"[DEBUG] InferenceActor {self.actor_id}:   【步骤3】后处理...", flush=True)

                    # 2. 后处理以采样动作 tokens 和对应的归一化连续动作
                    t_postprocess = time.time()
                    _, action_tokens_all, normalized_actions_all = self.model.post_process(action_logits, deterministic=deterministic_flags)
                    
                    if is_first_batch:
                        print(f"[DEBUG] InferenceActor {self.actor_id}:   【步骤3完成】后处理耗时: {time.time() - t_postprocess:.3f}s", flush=True)
                    
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

                current_policy_version = self.policy_version
                
                if is_first_batch:
                    print(f"[DEBUG] InferenceActor {self.actor_id}:   【步骤4】设置返回结果...", flush=True)
                
                for i in range(len(promises_to_process)):
                    promises_to_process[i].set_result((
                        actions_env[i],           # 反归一化的环境动作
                        action_tokens[i],         # 离散动作 token
                        logits[i], # 对应的 logits
                        values[i],                 # 价值估计
                        current_policy_version    # 策略版本号
                    ))
                loop_duration = time.time() - t_loop_start
                
                if is_first_batch:
                    print(f"[DEBUG] InferenceActor {self.actor_id}:   【第一次推理批次完成】总耗时: {loop_duration:.3f}s", flush=True)
                    print(f"[DEBUG] InferenceActor {self.actor_id}:   - 批次大小: {len(promises_to_process)}", flush=True)
                    print(f"[DEBUG] InferenceActor {self.actor_id}:   - 平均每个请求: {loop_duration/len(promises_to_process):.3f}s", flush=True)
                
                self.stats_actor.add_timing_metric.remote("Inference/loop_time_s", loop_duration)
            except Exception as e:
                import traceback
                print(f"[ERROR] InferenceActor {self.actor_id} 批处理失败: {e}", flush=True)
                traceback.print_exc()
                for p in promises_to_process:
                    if not p.done():
                        p.set_exception(e)
                raise
    
    def receive_and_update_weights(self, group_name):
        """覆盖基类方法，接收权重后自增策略版本（仅用于诊断）"""
        super().receive_and_update_weights(group_name)
        self.policy_version += 1
        if self.actor_id == 0:
            print(f"InferenceActor {self.actor_id}: 已更新到 policy_version={self.policy_version}")
    
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
    def __init__(self, rank, world_size, replay_buffer, cfg):
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
        self.super_batch_size = TRAIN_BATCH_SIZE * ACCUMULATION_STEPS
        self.global_step = 0
        # 仅用于诊断：跟踪当前策略版本
        self.policy_version = 0

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

    def broadcast_weights(self, group_name):
        """覆盖基类方法，广播后自增策略版本（只影响诊断）"""
        super().broadcast_weights(group_name)
        if self.rank == 0:
            self.policy_version += 1

    def setup_deepspeed_group(self, master_addr, master_port):
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["LOCAL_RANK"] = "0"
        deepspeed.init_distributed(dist_backend="nccl")

        print(f"Trainer {self.rank}: 正在加载 OpenVLA ActorCritic...")
        model = ActorCritic(self.cfg, torch_dtype=TORCH_DTYPE)
        self.base_model = model

        # 参数分组（与之前代码一致）
        param_groups = self.base_model.get_parameter_groups()
        optimizer_params = [
            {"params": pg["params"], "name": pg["name"], "lr": POLICY_LR if pg["name"] == "policy" else VALUE_LR}
            for pg in param_groups
        ]
        
        ds_config = {
            "train_micro_batch_size_per_gpu": TRAIN_BATCH_SIZE,
            "gradient_accumulation_steps": ACCUMULATION_STEPS,
            "optimizer": {"type": "AdamW", "params": {}},
            "bf16": {"enabled": USE_BF16},
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
                obs_list, action_token_np, adv_np, logits_old_np, v_targ_np, insert_step_np, policy_version_np = \
                    await self.replay_buffer.sample.remote(self.super_batch_size)
                sample_time = time.time() - t_sample_start

                t_prep_start = time.time()
                inputs_batch = self.base_model.prepare_inputs_batch(obs_list)

                device = next(self.model.parameters()).device
                act_token_t = torch.tensor(action_token_np, dtype=torch.long, device=device) # Tokens 是 long 类型
                adv_t = torch.tensor(adv_np, dtype=torch.float32, device=device)
                logits_old_t = torch.tensor(logits_old_np, dtype=torch.float32, device=device)
                v_targ_t = torch.tensor(v_targ_np, dtype=torch.float32, device=device)
                insert_step_t = torch.tensor(insert_step_np, dtype=torch.int32, device=device)
                policy_version_t = torch.tensor(policy_version_np, dtype=torch.int32, device=device)
                prep_time = time.time() - t_prep_start

                self.next_ready_batch = {
                    'inputs_batch': inputs_batch,
                    'act_token': act_token_t,
                    'advantage': adv_t,
                    'logits_old': logits_old_t,
                    'value_target': v_targ_t,
                    'insert_step': insert_step_t,
                    'policy_version': policy_version_t,
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
        value_lr = self._get_current_lr(self.global_step, VALUE_LR, VALUE_WARMUP_STEPS, TRAIN_ITERS)
        policy_lr = self._get_current_lr(self.global_step, POLICY_LR, POLICY_WARMUP_STEPS, TRAIN_ITERS, start_step=POLICY_TRAIN_START_STEP)
        
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
        insert_step_t = current_batch.get('insert_step', None)
        policy_version_t = current_batch.get('policy_version', None)
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
        diagnostic_metrics = {}
        
        num_updates_in_epoch = self.super_batch_size // TRAIN_BATCH_SIZE
        t_policy_train_start = time.time()
        
        for i in range(num_updates_in_epoch):
            start = i * TRAIN_BATCH_SIZE; end = start + TRAIN_BATCH_SIZE
            mini_inputs = {k: v[start:end] for k, v in inputs_batch.items()}
            
            mini_act_token = act_token_t[start:end]
            mini_adv = adv_t[start:end]
            mini_logits_old = logits_old_t[start:end]
            mini_v_targ = v_targ_t[start:end]
            mini_insert_step = insert_step_t[start:end] if insert_step_t is not None else None
            mini_policy_ver = policy_version_t[start:end] if policy_version_t is not None else None
            
            # 使用全局统计量进行归一化
            normalized_adv = (mini_adv - global_mean) / (global_std + 1e-8)
            # 前向
            action_logits, value = self.model.forward(mini_inputs)
            value = value.to(torch.float32)

            action_logits_reshape = action_logits.view(
                -1, NUM_ACTIONS_CHUNK, ACTION_DIM, action_logits.shape[-1]
            )

            # 价值损失 (不变)
            value_loss = VF_COEF * torch.mean((value - mini_v_targ) ** 2)
            
            if self.global_step < POLICY_TRAIN_START_STEP:
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
                kl_loss = KL_COEF * torch.mean(kl_div_tensor) # 作为损失

                ratio = torch.exp(logp - logp_old)
                adv_unsqueezed = normalized_adv.unsqueeze(dim=-1).unsqueeze(dim=-1)
                # 仅用于诊断旧数据指标，不影响算法逻辑
                if i == 0 and self.rank == 0 and mini_insert_step is not None and mini_policy_ver is not None:
                    try:
                        diagnostic_metrics = self._compute_diagnostic_metrics(
                            ratio=ratio,
                            advantage=normalized_adv,
                            insert_step=mini_insert_step,
                            policy_version=mini_policy_ver,
                            current_policy_version=self.policy_version,
                            current_step=self.global_step,
                            clip_eps=CLIP_EPS
                        )
                    except Exception as e:
                        print(f"[Warn] 诊断指标计算失败: {e}")
                policy_loss, clip_ratio = compute_policy_surrogate(
                    clip_mode=CLIP_MODE,
                    ratio=ratio,
                    adv_unsqueezed=adv_unsqueezed,
                    clip_params=CLIP_PARAMS,
                )
                ent = torch.mean(dist.entropy())
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
        # 合并诊断指标
        perf_metrics.update(diagnostic_metrics)

        return avg_loss, avg_p_loss, avg_v_loss, avg_e_loss, avg_kl_loss, current_lrs, self.global_step, avg_ent, avg_kl_div, perf_metrics

    def _compute_diagnostic_metrics(
        self,
        ratio: torch.Tensor,
        advantage: torch.Tensor,
        insert_step: torch.Tensor,
        policy_version: torch.Tensor,
        current_policy_version: int,
        current_step: int,
        clip_eps: float = 0.2,
    ) -> Dict[str, float]:
        """
        诊断“旧数据”相关指标，仅用于监控，不影响训练逻辑。
        指标来自 metaworld 版本，加入 staleness、贡献权重 U、ESS 等。
        """
        with torch.no_grad():
            metrics: Dict[str, float] = {}
            ratio_flat = ratio.reshape(-1)
            if ratio.dim() == 3:
                adv_expanded = advantage.unsqueeze(1).unsqueeze(2).expand_as(ratio).reshape(-1)
            else:
                adv_expanded = advantage

            # ---------------- Staleness (版本/步数) ----------------
            staleness_ver = current_policy_version - policy_version.float()
            metrics["staleness_ver_mean"] = staleness_ver.mean().item()
            metrics["staleness_ver_p95"] = torch.quantile(staleness_ver, 0.95).item()

            age_steps = insert_step.float().max() - insert_step.float()
            metrics["age_steps_mean"] = age_steps.mean().item()
            metrics["age_steps_p95"] = torch.quantile(age_steps, 0.95).item()
            metrics["age_steps_max"] = age_steps.max().item()

            # 绝对阈值分桶（版本落后）
            NEW_THRESHOLD = 2
            OLD_THRESHOLD = 10
            new_mask_batch = staleness_ver <= NEW_THRESHOLD
            old_mask_batch = staleness_ver >= OLD_THRESHOLD

            if ratio.dim() == 3:
                new_mask = new_mask_batch.unsqueeze(1).unsqueeze(2).expand_as(ratio).reshape(-1)
                old_mask = old_mask_batch.unsqueeze(1).unsqueeze(2).expand_as(ratio).reshape(-1)
            else:
                new_mask = new_mask_batch
                old_mask = old_mask_batch

            metrics["staleness_old_frac_abs"] = old_mask_batch.float().mean().item()
            metrics["staleness_new_frac_abs"] = new_mask_batch.float().mean().item()

            # ---------------- Ratio / log-ratio 分布 ----------------
            metrics["rho_mean"] = ratio_flat.mean().item()
            metrics["rho_p50"] = torch.median(ratio_flat).item()
            metrics["rho_p90"] = torch.quantile(ratio_flat, 0.90).item()
            metrics["rho_p99"] = torch.quantile(ratio_flat, 0.99).item()
            metrics["rho_max"] = ratio_flat.max().item()
            logrho = torch.log(ratio_flat.clamp(min=1e-8))
            metrics["logrho_mean"] = logrho.mean().item()
            metrics["abs_logrho_p95"] = torch.quantile(torch.abs(logrho), 0.95).item()

            # ---------------- PG Active / Dead (硬截断) ----------------
            dead_mask = ((adv_expanded > 0) & (ratio_flat > (1 + clip_eps))) | \
                        ((adv_expanded < 0) & (ratio_flat < (1 - clip_eps)))
            metrics["pg_dead_frac"] = dead_mask.float().mean().item()
            metrics["pg_active_frac"] = 1.0 - metrics["pg_dead_frac"]
            if new_mask.any():
                metrics["pg_dead_frac_new"] = dead_mask[new_mask].float().mean().item()
                metrics["pg_active_frac_new"] = 1.0 - metrics["pg_dead_frac_new"]
            if old_mask.any():
                metrics["pg_dead_frac_old"] = dead_mask[old_mask].float().mean().item()
                metrics["pg_active_frac_old"] = 1.0 - metrics["pg_dead_frac_old"]

            # ---------------- 贡献权重 U ----------------
            u = ratio_flat * (~dead_mask).float()
            metrics["u_mean"] = u.mean().item()
            metrics["u_p50"] = torch.median(u).item()
            metrics["u_p90"] = torch.quantile(u, 0.90).item()
            metrics["u_p99"] = torch.quantile(u, 0.99).item()
            metrics["u_max"] = u.max().item()
            if new_mask.any():
                u_new = u[new_mask]
                metrics["u_mean_new"] = u_new.mean().item()
                metrics["u_p90_new"] = torch.quantile(u_new, 0.90).item()
            if old_mask.any():
                u_old = u[old_mask]
                metrics["u_mean_old"] = u_old.mean().item()
                metrics["u_p90_old"] = torch.quantile(u_old, 0.90).item()

            # ---------------- ESS (基于 U) ----------------
            u_sum = u.sum()
            u_sq_sum = (u * u).sum()
            ess_eff = (u_sum * u_sum) / (u_sq_sum + 1e-12)
            metrics["ess_eff"] = ess_eff.item()
            metrics["ess_eff_norm"] = (ess_eff / u.numel()).item()
            if new_mask.any():
                u_new = u[new_mask]
                ess_new = (u_new.sum() * u_new.sum()) / (u_new.pow(2).sum() + 1e-12)
                metrics["ess_eff_norm_new"] = (ess_new / u_new.numel()).item()
            if old_mask.any():
                u_old = u[old_mask]
                ess_old = (u_old.sum() * u_old.sum()) / (u_old.pow(2).sum() + 1e-12)
                metrics["ess_eff_norm_old"] = (ess_old / u_old.numel()).item()

            # ---------------- 贡献占比 (旧/新) ----------------
            u_sum_all = u.sum()
            if u_sum_all > 0 and old_mask.any():
                metrics["contribution_old_u_share"] = (u[old_mask].sum() / u_sum_all).item()
            else:
                metrics["contribution_old_u_share"] = 0.0
            if u_sum_all > 0 and new_mask.any():
                metrics["contribution_new_u_share"] = (u[new_mask].sum() / u_sum_all).item()
            else:
                metrics["contribution_new_u_share"] = 0.0

            # ---------------- Clip 分数（兼容旧指标） ----------------
            clip_mask = (ratio_flat < (1 - clip_eps)) | (ratio_flat > (1 + clip_eps))
            metrics["clip_frac"] = clip_mask.float().mean().item()
            if new_mask.any():
                metrics["clip_frac_new"] = clip_mask[new_mask].float().mean().item()
            if old_mask.any():
                metrics["clip_frac_old"] = clip_mask[old_mask].float().mean().item()

            return metrics

# ================================================================
# 5. 主逻辑
# ================================================================
def build_openvla_cfg() -> GenerateConfig:
    cfg = GenerateConfig(
        pretrained_checkpoint=PRETRAINED_CHECKPOINT,
        use_l1_regression=False, # Note: ActorCritic in discrete model doesn't use this
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        # zzq 1124 开启 proprio 
        use_proprio=True, # Note: ActorCritic in discrete model can handle this
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        unnorm_key=BENCHMARK+"_no_noops",
    )
    return cfg

def main():
    args = parse_args()
    global BENCHMARK
    global NUM_TRAINER_GPUS, NUM_INFERENCE_ACTORS, NUM_ROLLOUT_WORKERS, NUM_EVAL_WORKERS
    global ROLLOUT_LOCAL_BUF, INFERENCE_BATCH, INFERENCE_TIMEOUT_MS
    global TRAIN_BATCH_SIZE, ACCUMULATION_STEPS, TRAIN_ITERS, REPLAY_CAPACITY
    global CLIP_MODE, CLIP_PARAMS, PRETRAINED_CHECKPOINT
    # 允许命令行覆盖关键资源与规模配置
    # benchmark 解析：优先映射 TaskSuite 枚举
    try:
        BENCHMARK = getattr(TaskSuite, args.benchmark)
    except AttributeError:
        BENCHMARK = args.benchmark
    NUM_TRAINER_GPUS = args.num_trainer_gpus
    NUM_INFERENCE_ACTORS = args.num_inference_actors
    NUM_ROLLOUT_WORKERS = args.num_rollout_workers
    NUM_EVAL_WORKERS = args.num_eval_workers
    ROLLOUT_LOCAL_BUF = args.rollout_local_buf
    INFERENCE_BATCH = args.inference_batch
    INFERENCE_TIMEOUT_MS = args.inference_timeout_ms
    TRAIN_BATCH_SIZE = args.train_batch_size
    ACCUMULATION_STEPS = args.accumulation_steps
    TRAIN_ITERS = args.train_iters
    REPLAY_CAPACITY = args.replay_capacity
    CLIP_MODE = args.clip_mode
    clip_config = load_clip_config(args.clip_config)
    CLIP_PARAMS = select_clip_params(CLIP_MODE, clip_config)
    PRETRAINED_CHECKPOINT = args.pretrained_checkpoint
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    # 显示最终配置
    print("=" * 80)
    print("🔧 训练配置")
    print("=" * 80)
    print(f"Benchmark: {BENCHMARK}")
    print(f"Pretrained Checkpoint: {PRETRAINED_CHECKPOINT}")
    print(f"Clip Mode: {CLIP_MODE}")
    print(f"CUDA Devices: {os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"Num Trainer GPUs: {NUM_TRAINER_GPUS}")
    print(f"Num Inference Actors: {NUM_INFERENCE_ACTORS}")
    print(f"Num Rollout Workers: {NUM_ROLLOUT_WORKERS}")
    print(f"Num Eval Workers: {NUM_EVAL_WORKERS}")
    print(f"Train Iters: {TRAIN_ITERS}")
    print("=" * 80)
    print()

    if not os.path.exists(PRETRAINED_CHECKPOINT):
        print(f"❌ 错误: OpenVLA checkpoint 路径 '{PRETRAINED_CHECKPOINT}' 不存在。请更新 PRETRAINED_CHECKPOINT。")
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
    print(f"ray.init 完成")
    exp_name = args.experiment_name or f"{BENCHMARK}_OpenVLA_DS_PPO_DISCRETE_{int(time.time())}"
    log_dir = f"runs/Libero/{BENCHMARK}/{exp_name}"
    writer = SummaryWriter(log_dir)
    stats_actor = StatsActor.remote(window_size=MOVING_AVG_WINDOW)
    
    # 显式打印 TensorBoard 信息
    print("=" * 80)
    print("📊 TensorBoard 日志记录")
    print("=" * 80)
    print(f"日志目录: {os.path.abspath(log_dir)}")
    print(f"\n启动 TensorBoard 命令:")
    print(f"  tensorboard --logdir={log_dir}")
    print(f"\n或查看所有实验:")
    print(f"  tensorboard --logdir=runs/Libero/{BENCHMARK}")
    print("=" * 80)
    print()
    
    # 初始化 SwanLab（带 timeout 和离线模式处理）
    swanlab_available = False
    swanlab_mode = "cloud"  # 默认云端模式
    try:
        print("尝试初始化 SwanLab（云端模式）...")
        swanlab.init(
            project="Libero-PPO-Discrete",
            experiment_name=exp_name,
            description=f"Libero DS PPO Discrete - {BENCHMARK}",
            config={
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
                "REWARD_SCALE": REWARD_SCALE,
                "VALUE_LR": VALUE_LR,
                "POLICY_LR": POLICY_LR,
                "VALUE_WARMUP_STEPS": VALUE_WARMUP_STEPS,
                "POLICY_WARMUP_STEPS": POLICY_WARMUP_STEPS,
                "POLICY_TRAIN_START_STEP": POLICY_TRAIN_START_STEP,
                "MOVING_AVG_WINDOW": MOVING_AVG_WINDOW,
                "STATE_DIM": "image+proprio",  # 仅描述
                "NUM_ACTIONS_CHUNK": NUM_ACTIONS_CHUNK,
                "ACTION_DIM": ACTION_DIM,
                "USE_BF16": USE_BF16,
                "CLIP_MODE": CLIP_MODE,
            },
        )
        swanlab_available = True
        print("✓ SwanLab 云端模式初始化成功")
    except Exception as e:
        error_msg = str(e).lower()
        if "timeout" in error_msg or "connection" in error_msg or "network" in error_msg:
            print(f"⚠ SwanLab 云端模式初始化失败 (网络问题: {e})")
            print("尝试切换到 SwanLab 离线模式...")
            try:
                swanlab.init(
                    project="Libero-PPO-Discrete",
                    experiment_name=exp_name,
                    description=f"Libero DS PPO Discrete - {BENCHMARK}",
                    mode="local",  # 离线模式
                    logdir=f"{log_dir}/swanlab",  # 指定离线日志目录
                    config={
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
                        "REWARD_SCALE": REWARD_SCALE,
                        "VALUE_LR": VALUE_LR,
                        "POLICY_LR": POLICY_LR,
                        "VALUE_WARMUP_STEPS": VALUE_WARMUP_STEPS,
                        "POLICY_WARMUP_STEPS": POLICY_WARMUP_STEPS,
                        "POLICY_TRAIN_START_STEP": POLICY_TRAIN_START_STEP,
                        "MOVING_AVG_WINDOW": MOVING_AVG_WINDOW,
                        "STATE_DIM": "image+proprio",
                        "NUM_ACTIONS_CHUNK": NUM_ACTIONS_CHUNK,
                        "ACTION_DIM": ACTION_DIM,
                        "USE_BF16": USE_BF16,
                        "CLIP_MODE": CLIP_MODE,
                    },
                )
                swanlab_available = True
                swanlab_mode = "local"
                print(f"✓ SwanLab 离线模式初始化成功，日志保存在: {log_dir}/swanlab")
            except Exception as e2:
                print(f"✗ SwanLab 离线模式也失败: {e2}")
                print("SwanLab 已禁用，仅使用 TensorBoard 记录")
                swanlab_available = False
        else:
            print(f"✗ SwanLab 初始化失败: {e}")
            print("SwanLab 已禁用，仅使用 TensorBoard 记录")
            swanlab_available = False
    
    if swanlab_available:
        print(f"SwanLab 运行模式: {swanlab_mode}")
    else:
        print("⚠️  主要日志记录工具: TensorBoard")
        print(f"   所有指标将保存到: {os.path.abspath(log_dir)}")

    cfg = build_openvla_cfg()
    
    # 测试主进程是否可以正常获取 processor 和导入环境
    print("[DEBUG] 主进程: 测试 processor 和环境导入...")
    try:
        processor = get_processor(cfg)
        print(f"[DEBUG] 主进程: ✓ processor 获取成功")
    except Exception as e:
        print(f"[ERROR] 主进程: ✗ processor 获取失败: {e}")
        import traceback
        print(f"[ERROR] 主进程: 错误信息:\n{traceback.format_exc()}")
        raise
    
    try:
        # 测试 LiberoEnvWrapper 是否已正确导入
        print(f"[DEBUG] 主进程: 测试 LiberoEnvWrapper 类: {LiberoEnvWrapper}")
        print(f"[DEBUG] 主进程: ✓ LiberoEnvWrapper 导入成功")
    except Exception as e:
        print(f"[ERROR] 主进程: ✗ LiberoEnvWrapper 导入失败: {e}")
        raise
    
    print("\n--- 步骤 1: 创建 Actors ---")
    replay_buffers = [ReplayBufferActor.remote(capacity=REPLAY_CAPACITY) for _ in range(NUM_TRAINER_GPUS)]
    trainer_group = [
        TrainerActor.remote(rank=i, world_size=NUM_TRAINER_GPUS, replay_buffer=replay_buffers[i], cfg=cfg)
        for i in range(NUM_TRAINER_GPUS)
    ]
    inference_pool = [InferenceActor.remote(actor_id=i, cfg=cfg, stats_actor=stats_actor) for i in range(NUM_INFERENCE_ACTORS)]
    print(f"[DEBUG] 主进程: 创建 RolloutWorkers，NUM_ROLLOUT_WORKERS={NUM_ROLLOUT_WORKERS}, NUM_INFERENCE_ACTORS={NUM_INFERENCE_ACTORS}, NUM_TRAINER_GPUS={NUM_TRAINER_GPUS}", flush=True)
    rollout_workers = []
    for i in range(NUM_ROLLOUT_WORKERS):
        infer_idx = i % NUM_INFERENCE_ACTORS
        replay_idx = i % NUM_TRAINER_GPUS
        print(f"[DEBUG] 主进程: 创建 RolloutWorker {i}, 使用 InferenceActor {infer_idx}, ReplayBuffer {replay_idx}, benchmark={BENCHMARK}", flush=True)
        worker = RolloutWorkerActor.remote(
            inference_pool[infer_idx],
            replay_buffers[replay_idx], i, stats_actor, cfg, BENCHMARK
        )
        rollout_workers.append(worker)
        print(f"[DEBUG] 主进程: RolloutWorker {i} 创建完成，引用={worker}", flush=True)
    
    print(f"[DEBUG] 主进程: 开始创建 {NUM_EVAL_WORKERS} 个 EvaluationWorkers，benchmark={BENCHMARK}", flush=True)
    eval_workers = []
    for i in range(NUM_EVAL_WORKERS):
        infer_idx = i % NUM_INFERENCE_ACTORS
        print(f"[DEBUG] 主进程: 创建 EvaluationWorker {i}/{NUM_EVAL_WORKERS}, 使用 InferenceActor {infer_idx}", flush=True)
        t_eval_create_start = time.time()
        worker = EvaluationWorkerActor.remote(
            inference_pool[infer_idx], f"eval_{i}", stats_actor, cfg, BENCHMARK
        )
        eval_workers.append(worker)
        t_eval_create_end = time.time()
        print(f"[DEBUG] 主进程: EvaluationWorker {i} 创建调用完成 (耗时: {t_eval_create_end - t_eval_create_start:.3f}s), 引用={worker}", flush=True)
        
        # 每创建5个workers检查一次状态
        if (i + 1) % 5 == 0 or i == NUM_EVAL_WORKERS - 1:
            print(f"[DEBUG] 主进程: 已发起创建 {i+1}/{NUM_EVAL_WORKERS} 个 EvaluationWorkers", flush=True)
    
    print(f"[DEBUG] 主进程: ✓ 已创建 {NUM_ROLLOUT_WORKERS} 个 Rollout workers 和 {NUM_EVAL_WORKERS} 个 Evaluation workers。", flush=True)
    
    # 等待一小段时间，让 workers 开始初始化
    print(f"[DEBUG] 主进程: 等待 5 秒让 workers 开始初始化...", flush=True)
    for i in range(5):
        time.sleep(1)
        if i % 2 == 0:
            print(f"[DEBUG] 主进程: 初始化等待中... {i+1}/5 秒", flush=True)
    
    print(f"[DEBUG] 主进程: ✓ Workers 创建完成，继续后续步骤", flush=True)

    print("\n--- 步骤 2: 建立独立的 DeepSpeed 训练组 ---")
    # zzq 1125 通信组，使用find_free_port
    print("正在查找空闲端口...")
    train_group_port = find_free_port(base_port=29500)
    print(f"训练组端口: {train_group_port}")

    # 使用不同的 base_port 查找第二个端口，避免冲突
    # 如果 train_group_port 在 29500-29599 范围内，使用 29600 作为 base_port
    # 否则使用 train_group_port + 100 作为 base_port
    if 29500 <= train_group_port < 29600:
        broadcast_base_port = 29600
    else:
        broadcast_base_port = train_group_port + 100
    
    broadcast_group_port = find_free_port(base_port=broadcast_base_port)
    max_retries = 10
    try_times = 0
    while broadcast_group_port == train_group_port and try_times < max_retries:
        try_times += 1
        print(f"尝试 {try_times}/{max_retries} 次，广播组端口与训练组端口相同 ({broadcast_group_port})，重新查找")
        # 每次尝试使用不同的 base_port
        broadcast_base_port = broadcast_base_port + 100
        broadcast_group_port = find_free_port(base_port=broadcast_base_port)
    
    if broadcast_group_port == train_group_port:
        raise RuntimeError(f"无法找到与训练组端口不同的广播组端口（已重试 {max_retries} 次）。训练组端口: {train_group_port}")
    
    print(f"广播组端口: {broadcast_group_port}")

    trainer_master_addr = ray.get(trainer_group[0].get_node_ip.remote())
    train_setup_tasks = [actor.setup_deepspeed_group.remote(trainer_master_addr, train_group_port) for actor in trainer_group]
    ray.get(train_setup_tasks)
    print("DeepSpeed 训练组建立完成。")

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
    # 打印前几十个，或计算哈希对比
    if len(train_sig) != len(infer_sig):
        raise RuntimeError(f"训练器与推理器的广播签名长度不匹配: {len(train_sig)} vs {len(infer_sig)}")
    for i, (a, b) in enumerate(zip(train_sig, infer_sig)):
        if a != b:
            raise RuntimeError(f"First mismatch at idx: {i}, trainer: {a}, inference: {b}")
    forward_test_tasks = [inf.forward_test.remote() for inf in inference_pool]
    ray.get(forward_test_tasks)
    print("推理器前向测试完成 (广播前)。")
    
    broadcast_task = trainer_group[0].broadcast_weights.remote(BROADCAST_GROUP_NAME)
    receive_tasks = [inf.receive_and_update_weights.remote(BROADCAST_GROUP_NAME) for inf in inference_pool]
    ray.get([broadcast_task] + receive_tasks)
    print("初始权重已广播到所有推理器。")

    forward_test_tasks = [inf.forward_test.remote() for inf in inference_pool]
    ray.get(forward_test_tasks)
    print("推理器前向测试完成 (广播后)。")

    print("\n--- 步骤 4: 启动 Rollout Workers 进行数据收集 ---")
    print(f"[DEBUG] 主进程: 准备启动 {len(rollout_workers)} 个 RolloutWorkers，时间={time.time():.2f}", flush=True)
    
    t_worker_start = time.time()
    rollout_futures = [w.run.remote() for w in rollout_workers]
    t_worker_end = time.time()
    
    print(f"[DEBUG] 主进程: RolloutWorkers 启动完成，futures数量={len(rollout_futures)}，耗时: {t_worker_end - t_worker_start:.2f}s", flush=True)
    print(f"[DEBUG] 主进程: 所有 rollout futures: {rollout_futures[:3]}..." if len(rollout_futures) > 3 else f"[DEBUG] 主进程: 所有 rollout futures: {rollout_futures}", flush=True)
    
    eval_futures = [w.run.remote() for w in eval_workers]
    print(f"[DEBUG] 主进程: EvaluationWorkers 启动完成，futures数量={len(eval_futures)}", flush=True)
    
    # 等待一小段时间，让workers有时间初始化
    print(f"[DEBUG] 主进程: 等待5秒让workers初始化...", flush=True)
    for i in range(5):
        time.sleep(1)
        print(f"[DEBUG] 主进程: 初始化等待中... {i+1}/5 秒", flush=True)
    
    # 检查回放缓冲区大小
    print(f"[DEBUG] 主进程: 检查回放缓冲区初始大小...", flush=True)
    initial_sizes = ray.get([rb.size.remote() for rb in replay_buffers])
    print(f"[DEBUG] 主进程: Workers启动后，回放缓冲区初始大小: {initial_sizes}", flush=True)
    
    # 检查是否有 worker 崩溃
    print(f"[DEBUG] 主进程: 检查 workers 状态...", flush=True)
    ready, not_ready = ray.wait(rollout_futures + eval_futures, timeout=0.1)
    if ready:
        print(f"[WARN] 主进程: 发现 {len(ready)} 个 worker 已经完成（可能崩溃），检查详情...", flush=True)
        for fut in ready:
            try:
                result = ray.get(fut)
                print(f"[WARN] 主进程: Worker 异常完成，返回值: {result}", flush=True)
            except Exception as e:
                print(f"[ERROR] 主进程: Worker 崩溃: {e}", flush=True)
                import traceback
                traceback.print_exc()
    else:
        print(f"[DEBUG] 主进程: 所有 workers 运行正常 ({len(not_ready)} 个 workers 运行中)", flush=True)

    print("\n--- 步骤 5: 等待远程经验池填充初始数据 ---")
    min_buffer_size_for_start = TRAIN_BATCH_SIZE * ACCUMULATION_STEPS
    assert min_buffer_size_for_start < REPLAY_CAPACITY, "初始填充量必须小于回放池总容量"
    print(f"[DEBUG] 主进程: 开始等待经验池填充，目标大小={min_buffer_size_for_start}, ROLLOUT_LOCAL_BUF={ROLLOUT_LOCAL_BUF}", flush=True)
    
    wait_count = 0
    max_wait_count = 120  # 最多等待 120 * 5 = 600 秒 (10分钟)
    wait_start_time = time.time()
    
    while not all(size >= min_buffer_size_for_start for size in ray.get([rb.size.remote() for rb in replay_buffers])):
        wait_count += 1
        current_wait_time = time.time() - wait_start_time
        
        sizes = ray.get([rb.size.remote() for rb in replay_buffers])
        print(f"[DEBUG] 主进程: 等待第 {wait_count} 次检查 (已等待 {current_wait_time:.1f}s) - 目标: {min_buffer_size_for_start}, 当前: {sizes}", flush=True)
        
        # 每5次检查（25秒）打印一次更详细的调试信息
        if wait_count % 5 == 0:
            print(f"[DEBUG] 主进程: ========== 详细状态检查 (第 {wait_count} 次) ==========", flush=True)
            
            # 检查 workers 是否还在运行
            ready, not_ready = ray.wait(rollout_futures + eval_futures, timeout=0.1)
            print(f"[DEBUG] 主进程: Worker 状态 - 运行中: {len(not_ready)}, 已完成/崩溃: {len(ready)}", flush=True)
            
            if ready:
                print(f"[WARN] 主进程: 检测到 {len(ready)} 个 worker 已停止，检查错误...", flush=True)
                for i, fut in enumerate(ready[:3]):  # 只检查前3个
                    try:
                        result = ray.get(fut)
                        print(f"[WARN] 主进程: Worker {i} 返回: {result}", flush=True)
                    except Exception as e:
                        print(f"[ERROR] 主进程: Worker {i} 错误: {e}", flush=True)
            
            # 尝试获取统计信息
            try:
                stats = ray.get(stats_actor.get_stats.remote())
                global_stats = stats.get('_global_rollout_', {})
                print(f"[DEBUG] 主进程: 统计信息:", flush=True)
                print(f"  - 已完成 episodes: {global_stats.get('total_episodes_processed', 0)}", flush=True)
                print(f"  - 环境步数: {global_stats.get('total_env_steps', 0)}", flush=True)
                print(f"  - 生成的样本数: {global_stats.get('total_samples_produced', 0)}", flush=True)
                print(f"  - 活跃 actors: {global_stats.get('active_actor_count', 0)}", flush=True)
            except Exception as e:
                print(f"[DEBUG] 主进程: 获取StatsActor状态失败: {e}", flush=True)
            
            print(f"[DEBUG] 主进程: ====================================", flush=True)
        
        # 超时检查
        if wait_count >= max_wait_count:
            print(f"[ERROR] 主进程: 等待超时！已等待 {current_wait_time:.1f}s ({wait_count} 次检查)", flush=True)
            print(f"[ERROR] 主进程: 当前缓冲区大小: {sizes}, 目标: {min_buffer_size_for_start}", flush=True)
            print(f"[ERROR] 主进程: 这可能是因为 workers 卡住或无法生成数据", flush=True)
            
            # 打印最后的状态
            ready, not_ready = ray.wait(rollout_futures, timeout=1)
            print(f"[ERROR] 主进程: 最终 Worker 状态 - 运行中: {len(not_ready)}, 已停止: {len(ready)}", flush=True)
            
            raise TimeoutError(f"等待经验池填充超时 ({current_wait_time:.1f}s)")
        
        time.sleep(5)
    
    final_sizes = ray.get([rb.size.remote() for rb in replay_buffers])
    total_wait_time = time.time() - wait_start_time
    print(f"[DEBUG] 主进程: ✓ 经验池填充完成！", flush=True)
    print(f"[DEBUG] 主进程:   - 最终大小: {final_sizes}", flush=True)
    print(f"[DEBUG] 主进程:   - 总等待时间: {total_wait_time:.1f}s", flush=True)
    print("远程经验池已准备好，训练器将按需获取数据。")

    print("\n--- 步骤 6: 开始主训练与同步循环 ---")
    print("=" * 80)
    print("🚀 训练开始！")
    print("=" * 80)
    print(f"📊 实时监控 TensorBoard:")
    print(f"   tensorboard --logdir={log_dir}")
    if swanlab_available:
        if swanlab_mode == "local":
            print(f"📁 SwanLab 离线日志: {log_dir}/swanlab")
        else:
            print(f"☁️  SwanLab 云端监控已启用")
    print("=" * 80)
    print()
    start_time = time.time()
    last_log_time = time.time()
    last_log_global_step = 0
    global_step = 0
    while global_step < TRAIN_ITERS:
        t_train_start = time.time()
        train_tasks = [trainer.run_training_epoch.remote() for trainer in trainer_group]
        results = ray.get(train_tasks)
        _, _, _, _, _, _, global_step, _, _, _ = results[0]
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


            total_losses, p_losses, v_losses, e_losses, kl_losses, lrs_list, _, ents, avg_kl_divs, perf_metrics_list = zip(*results)
            current_lrs = lrs_list[0]

            elapsed_time = current_time - start_time
            total_buffer_size = sum(ray.get([rb.size.remote() for rb in replay_buffers]))

            print(f"更新步 {global_step}/{TRAIN_ITERS} | 时间: {elapsed_time:.1f}s | "
                  f"全局平均奖励: {avg_return:.2f} | 全局平均幕长: {avg_ep_len:.1f} | Eval奖励: {eval_avg_return:.2f} | "
                  f"value loss: {np.mean(v_losses):.4f} | LR(V/P): {current_lrs['value']:.7f}/{current_lrs['policy']:.7f} | "
                  f"Episodes数量: {total_episodes:,} | Step平均时间: {avg_step_time:.3f}s")

            log_metrics = {
                'Train/Learning_Rate/Value': current_lrs['value'],
                'Train/Learning_Rate/Policy': current_lrs['policy'],
                'Loss/Total': float(np.mean(total_losses)),
                'Loss/Policy': float(np.mean(p_losses)),
                'Loss/Value': float(np.mean(v_losses)),
                'Loss/Entropy': float(np.mean(e_losses)),
                'Loss/KL': float(np.mean(kl_losses)),
                'Metrics/Entropy': float(np.mean(ents)),
                'Metrics/KL_Divergence': float(np.mean(avg_kl_divs)),
                'Metrics/Training_Speed_Steps_per_Sec': training_speed_steps_per_sec,
                'Performance/policy_sample_time': float(np.mean([pm["policy_sample_time"] for pm in perf_metrics_list])),
                'Performance/policy_prep_time': float(np.mean([pm["policy_prep_time"] for pm in perf_metrics_list])),
                'Performance/policy_train_time': float(np.mean([pm["policy_train_time"] for pm in perf_metrics_list])),
                'Performance/train_time': train_time,
                'Performance/sync_time': sync_time,
                'Rollout/_Global/Average_Return': avg_return,
                'Rollout/_Global/Average_Episode_Length': avg_ep_len,
                'Eval/_Global/Average_Return': eval_avg_return,
                'Eval/_Global/Average_Episode_Length': eval_avg_ep_len,
                'System/Replay_Buffer_Size_Total': total_buffer_size,
                'System/Total_Episodes_Processed': total_episodes,
                'System/Total_Env_Steps': total_env_steps,
                'System/Avg_Step_Time': avg_step_time,
                'System/Eval_Total_Episodes_Processed': eval_total_episodes,
                'System/Eval_Total_Env_Steps': eval_env_steps,
                'System/Eval_Avg_Step_Time': eval_avg_step_time,
                'System/Active_Rollout_Actors': global_stats.get("active_actor_count", 0),
                'System/Total_Samples_Produced': global_stats.get("total_samples_produced", 0),
            }

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

            # 旧数据诊断指标（如有）
            diag_keys = [
                ("age_steps_mean", "Diag/Age_Steps_Mean"),
                ("age_steps_p95", "Diag/Age_Steps_P95"),
                ("age_steps_max", "Diag/Age_Steps_Max"),
                ("staleness_old_frac_abs", "Diag/Staleness_OldFrac_Abs"),
                ("staleness_new_frac_abs", "Diag/Staleness_NewFrac_Abs"),
                ("contribution_old_u_share", "Diag/Contribution_OldUShare"),
                ("contribution_new_u_share", "Diag/Contribution_NewUShare"),
            ]
            for key, tag in diag_keys:
                if key in perf_metrics_list[0]:
                    writer.add_scalar(tag, perf_metrics_list[0][key], global_step)
            
            # 同步记录到 SwanLab（如果可用）
            if swanlab_available:
                try:
                    swanlab.log(log_metrics, step=global_step)
                except Exception as e:
                    error_msg = str(e).lower()
                    if "timeout" in error_msg or "connection" in error_msg or "network" in error_msg:
                        print(f"[Warn] SwanLab 记录失败 (网络问题): {e}")
                        print(f"✓ 数据已通过 TensorBoard 保存: {log_dir}")
                        print(f"  查看命令: tensorboard --logdir={log_dir}")
                        # 禁用后续的 SwanLab 记录，避免重复警告
                        swanlab_available = False
                    else:
                        print(f"[Warn] SwanLab 记录失败: {e}")
                        print(f"✓ 数据已通过 TensorBoard 保存: {log_dir}")

            # 新增的旧数据诊断指标（与 metaworld 版本对齐）
            extra_diag = {
                "staleness_ver_mean": "Diag/Staleness_Version_Mean",
                "staleness_ver_p95": "Diag/Staleness_Version_P95",
                "rho_mean": "Diag/Ratio_Rho_Mean",
                "rho_p50": "Diag/Ratio_Rho_P50",
                "rho_p90": "Diag/Ratio_Rho_P90",
                "rho_p99": "Diag/Ratio_Rho_P99",
                "rho_max": "Diag/Ratio_Rho_Max",
                "logrho_mean": "Diag/Ratio_LogRho_Mean",
                "abs_logrho_p95": "Diag/Ratio_AbsLogRho_P95",
                "pg_active_frac": "Diag/PG_Active_Frac",
                "pg_dead_frac": "Diag/PG_Dead_Frac",
                "pg_active_frac_new": "Diag/PG_Active_Frac_New",
                "pg_dead_frac_new": "Diag/PG_Dead_Frac_New",
                "pg_active_frac_old": "Diag/PG_Active_Frac_Old",
                "pg_dead_frac_old": "Diag/PG_Dead_Frac_Old",
                "u_mean": "Diag/U_Mean",
                "u_p90": "Diag/U_P90",
                "u_p99": "Diag/U_P99",
                "u_max": "Diag/U_Max",
                "u_mean_new": "Diag/U_Mean_New",
                "u_mean_old": "Diag/U_Mean_Old",
                "ess_eff": "Diag/ESS_Eff",
                "ess_eff_norm": "Diag/ESS_Eff_Norm",
                "ess_eff_norm_new": "Diag/ESS_Eff_Norm_New",
                "ess_eff_norm_old": "Diag/ESS_Eff_Norm_Old",
                "clip_frac": "Diag/Clip_Frac",
                "clip_frac_new": "Diag/Clip_Frac_New",
                "clip_frac_old": "Diag/Clip_Frac_Old",
            }
            for key, tag in extra_diag.items():
                if key in perf_metrics_list[0]:
                    writer.add_scalar(tag, perf_metrics_list[0][key], global_step)

            writer.add_scalar('Rollout/_Global/Average_Return', avg_return, global_step)
            writer.add_scalar('Rollout/_Global/Average_Episode_Length', avg_ep_len, global_step)
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

    print(f"\n成功完成 {TRAIN_ITERS} 次训练与同步循环！")
    print("=" * 80)
    print("✅ 训练完成！")
    print("=" * 80)
    print(f"📊 查看训练结果 (TensorBoard):")
    print(f"   tensorboard --logdir={log_dir}")
    print(f"\n日志目录: {os.path.abspath(log_dir)}")
    if swanlab_available and swanlab_mode == "local":
        print(f"\n📁 SwanLab 离线日志可手动上传:")
        print(f"   swanlab upload {log_dir}/swanlab")
    print("=" * 80)
    writer.close()
    ray.shutdown()


if __name__ == "__main__":
    main()