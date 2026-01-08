import os
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# 文件移动需要
import shutil
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
# 为每个进程设置独立的 PyTorch 扩展目录，避免多进程编译冲突
os.environ.setdefault("TORCH_EXTENSIONS_DIR", f"/dev/shm/torch_ext_{os.getpid()}")
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

# SwanLab 可选：通过环境变量控制是否启用
ENABLE_SWANLAB = os.environ.get("ENABLE_SWANLAB", "1").lower() in ("1", "true", "yes")
if ENABLE_SWANLAB:
    try:
        import swanlab
    except ImportError:
        print("[Warn] SwanLab未安装，将禁用SwanLab记录")
        ENABLE_SWANLAB = False
        swanlab = None
else:
    swanlab = None
    print("[Info] SwanLab已通过环境变量禁用，仅使用TensorBoard")

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

# 诊断绘图频率
DIAG_EVERY_STEPS = int(os.environ.get("DIAG_EVERY_STEPS", "200"))

# ================================================================
# 采样过滤配置（用于 Recency Window Ablation）
# ================================================================
# A. 最新比例窗口：只从最新 f% 数据采样
REPLAY_RECENT_FRAC = 1.0  # 取值：{0.01, 0.05, 0.1, 0.2, 0.5, 1.0}，1.0表示使用全部数据
# B. 版本差窗口：只采样 current_version - policy_version <= dv_max
REPLAY_MAX_VERSION_GAP = float('inf')  # 取值：{1, 2, 5, 10, inf}，inf表示不限制

# ================================================================
# 诊断数据导出配置（用于离线绘制机制图）
# ================================================================
DUMP_INTERVAL = 1000  # 每 N 个 update 导出一次数据
DUMP_NUM_SAMPLES = 10000  # 每次导出的样本数量
DUMP_DIR = f"{CKPT_DIR}/diagnostic_dumps"  # 导出数据保存目录

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
    # 采样过滤参数
    parser.add_argument("--replay-recent-frac", dest="replay_recent_frac", type=float, 
                        default=REPLAY_RECENT_FRAC,
                        help="最新比例窗口：只从最新 f%% 数据采样 (0.01-1.0)")
    parser.add_argument("--replay-max-version-gap", dest="replay_max_version_gap", type=float,
                        default=REPLAY_MAX_VERSION_GAP,
                        help="版本差窗口：只采样 version_gap <= dv_max 的数据 (1, 2, 5, 10, inf)")
    parser.add_argument("--train-iters", dest="train_iters", type=int,
                        default=TRAIN_ITERS, help="训练迭代次数")
    parser.add_argument("--log-backend", dest="log_backend", type=str,
                        default="both", choices=["tensorboard", "swanlab", "both"],
                        help="日志后端选择: tensorboard/swanlab/both (默认: both)")
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


def consolidate_swanlab_run_dir(base_dir: str, target_name: str):
    """
    为最新的 SwanLab run-* 创建一个指向 target_name 的符号链接，保持原始 run-* 不动，
    避免在写日志过程中移动目录导致 “Run directory does not exist”。
    - 如果 target 已存在（目录或链接），则不做处理。
    """
    try:
        if not base_dir or not os.path.isdir(base_dir):
            return
        run_dirs = [
            os.path.join(base_dir, d)
            for d in os.listdir(base_dir)
            if d.startswith("run-") and os.path.isdir(os.path.join(base_dir, d))
        ]
        if not run_dirs:
            return
        latest = max(run_dirs, key=os.path.getmtime)
        target = os.path.join(base_dir, target_name)
        if os.path.exists(target):
            return
        # 使用符号链接指向最新的 run-*，不移动原目录
        try:
            os.symlink(latest, target, target_is_directory=True)
            debug_log(f"SwanLab 目录统一: {target} -> {latest}")
        except FileExistsError:
            pass
        except OSError as e:
            debug_log(f"SwanLab 符号链接创建失败: {e}")
    except Exception as e:
        debug_log(f"SwanLab 目录合并失败: {e}")


def format_clip_params_for_path(clip_params: Dict[str, Any]) -> str:
    """
    将 clip_params 字典格式化为可用于路径的字符串。
    例如: {"soft_clip_alpha": 1.0, "tau_pos": 1.0} -> "alpha1.0_tau1.0"
    """
    if not clip_params:
        return ""
    
    parts = []
    # 按字母顺序排序，确保一致性
    for key, value in sorted(clip_params.items()):
        # 跳过空值
        if value is None:
            continue
        
        # 处理不同的值类型
        if isinstance(value, float):
            # 浮点数：去除小数点，例如 1.0 -> 1, 0.5 -> 0d5 (d表示decimal)
            if value == int(value):
                value_str = str(int(value))
            else:
                # 使用 'd' 代替小数点，'-' 用 'n' 代替（负号）
                value_str = str(value).replace(".", "d").replace("-", "n")
        elif isinstance(value, bool):
            value_str = "1" if value else "0"
        elif isinstance(value, (int, str)):
            value_str = str(value)
        else:
            # 其他类型：转换为字符串并清理
            value_str = str(value)
        
        # 清理键名：移除特殊字符，缩短常见键名
        key_clean = key.replace("_", "").replace("-", "")
        # 常见键名简化
        key_map = {
            "softclipalpha": "alpha",
            "taupos": "taup",
            "tauneg": "taun",
            "clipeps": "eps",
            "epsislow": "epsl",
            "epsishigh": "epsh",
            "bet1": "b1",
            "bet2": "b2",
        }
        key_clean = key_map.get(key_clean.lower(), key_clean)
        
        parts.append(f"{key_clean}{value_str}")
    
    return "_".join(parts) if parts else ""


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

    if clip_mode == "soft_clip" or clip_mode == "soft_clip_alpha-0" or clip_mode == "soft_clip_alpha-0-5" or clip_mode == "soft_clip_alpha-1" or clip_mode == "soft_clip_alpha-2":
        if clip_mode == "soft_clip_alpha-0":
            soft_clip_alpha = 0
        elif clip_mode == "soft_clip_alpha-0-5":
            soft_clip_alpha = 0.5
        elif clip_mode == "soft_clip_alpha-1":
            soft_clip_alpha = 1
        elif clip_mode == "soft_clip_alpha-2":
            soft_clip_alpha = 2
        else:
            soft_clip_alpha = clip_params.get("soft_clip_alpha", 1)
        diff = torch.maximum(ratio, 1.0 / ratio)
        coeff = (1.0 / diff).detach()
        # 动态 alpha：alpha * (1 + sigmoid(-A))，正优势时保持，负优势时加大抑制
        alpha_tensor = soft_clip_alpha * (1 + torch.sigmoid(-adv_unsqueezed))
        coeff = coeff ** alpha_tensor
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
        p = torch.sigmoid(x)
        w = 4.0 * p * (1.0 - p)
        w_thresh = float(clip_params.get("sapo_w_thresh", 0.05))
        clip_ratio = (w < w_thresh).float().mean().item()

        return policy_loss, clip_ratio

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

        clip_ratio = (low_mask | high_mask).float().mean().item()
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
        clip_ratio = is_clipped.float().mean().item()
        return policy_loss, clip_ratio

    # ===== 新增：Log-Gaussian soft clip =====
    if clip_mode == "log_gauss_clip":
        eps = float(clip_params.get("eps", 1e-9))
        sigma = float(clip_params.get("sigma", 1))
        r = ratio.clamp_min(eps).detach()
        coeff = torch.exp(-0.5 * (torch.log(r) / sigma) ** 2)
        surr_soft = surr1 * coeff
        policy_loss = -torch.mean(surr_soft)
        clip_ratio = 0.0
        return policy_loss, clip_ratio

    raise ValueError(f"Unsupported clip mode: {clip_mode}")

def make_policy_prob_diag_image(
    p_old: torch.Tensor,
    p_new: torch.Tensor,
    max_points: int = 20000,
    title: str = "Old vs New Policy Probability",
) -> Tuple[np.ndarray | None, float]:
    """
    绘制旧/新策略在相同动作上的概率散点图。
    x 轴: p_old = exp(logp_old)
    y 轴: p_new = exp(logp)
    颜色: |p_new - p_old|
    """
    with torch.no_grad():
        po = p_old.detach().reshape(-1)
        pn = p_new.detach().reshape(-1)
        n = po.numel()
        if n == 0:
            return None, float("nan")
        if n > max_points:
            idx = torch.randperm(n, device=po.device)[:max_points]
            po = po[idx]
            pn = pn[idx]
        po_np = po.float().cpu().numpy()
        pn_np = pn.float().cpu().numpy()

    diff = np.abs(pn_np - po_np)
    corr = float(np.corrcoef(po_np, pn_np)[0, 1]) if len(po_np) >= 2 else float("nan")

    fig = plt.figure(figsize=(6.5, 5), dpi=160)
    ax = fig.add_subplot(111)
    sc = ax.scatter(po_np, pn_np, c=diff, s=10)
    ax.plot([0, 1], [0, 1], "r--", linewidth=2, label="y=x")
    ax.set_xlabel("Old Policy Probability")
    ax.set_ylabel("New Policy Probability")
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, label="|p_new - p_old|")
    ax.legend(loc="upper left")
    ax.text(
        0.60, 0.05, f"Correlation: {corr:.6f}",
        transform=ax.transAxes,
        bbox=dict(facecolor="white", edgecolor="black")
    )
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    img = mpimg.imread(buf)
    if img.shape[-1] == 4:
        img = img[..., :3]
    img_uint8 = (img * 255.0).clip(0, 255).astype(np.uint8)
    return img_uint8, corr

def make_old_new_prob_scatter_image(
    p_old: torch.Tensor,
    p_new: torch.Tensor,
    max_points: int = 20000,
    title: str = "Old vs New Action Probability",
) -> Tuple[np.ndarray, float]:
    """
    输入: p_old/p_new 为 torch Tensor (任意 shape)，会 flatten。
    输出: (RGB uint8 图像数组, corr)
    """
    po = p_old.detach().float().reshape(-1).cpu().numpy()
    pn = p_new.detach().float().reshape(-1).cpu().numpy()

    n = min(len(po), len(pn))
    po, pn = po[:n], pn[:n]
    if n == 0:
        return None, float("nan")

    if n > max_points:
        idx = np.random.choice(n, size=max_points, replace=False)
        po, pn = po[idx], pn[idx]

    diff = np.abs(pn - po)
    corr = float(np.corrcoef(po, pn)[0, 1]) if len(po) >= 2 else float("nan")

    fig = plt.figure(figsize=(6.5, 5), dpi=160)
    ax = fig.add_subplot(111)
    sc = ax.scatter(po, pn, c=diff, s=10)
    ax.plot([0, 1], [0, 1], "r--", linewidth=2, label="y=x")
    ax.set_xlabel("Old Policy Probability")
    ax.set_ylabel("New Policy Probability")
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, label="|p_new - p_old|")
    ax.legend(loc="upper left")
    ax.text(
        0.60, 0.05, f"Correlation: {corr:.6f}",
        transform=ax.transAxes,
        bbox=dict(facecolor="white", edgecolor="black")
    )
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    img = mpimg.imread(buf)  # float [0,1], shape [H,W,4] or [H,W,3]
    if img.shape[-1] == 4:
        img = img[..., :3]
    img_uint8 = (img * 255.0).clip(0, 255).astype(np.uint8)
    return img_uint8, corr

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
    
    # ========== 新增字段：用于 staleness / ratio 诊断 ==========
    policy_version: int = 0       # 产生该样本时的策略版本号
    insert_step: int = 0          # 写入 replay 时的全局步数
    episode_id: int = 0           # 所属 episode 的 ID（可选，用于 debug）
    segment_id: int = 0           # 所属 segment 的 ID（可选，用于 debug）

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
        
        # ========== 新增：全局计数器用于 staleness 诊断 ==========
        self.insert_counter = 0  # 单调递增的插入计数器
        self.episode_counter = 0  # episode ID 计数器

    def add_trajectory(self, traj: List[Experience], done: bool, last_obs: np.ndarray):
        """
        添加轨迹到 buffer，自动填充 insert_step 和 episode_id
        """
        # 为轨迹中的每个 Experience 填充 insert_step
        current_insert_step = self.insert_counter
        current_episode_id = self.episode_counter
        
        for i, exp in enumerate(traj):
            exp.insert_step = current_insert_step + i
            if exp.episode_id == 0:  # 如果还未设置 episode_id
                exp.episode_id = current_episode_id
        
        self.buffer.append((traj, done, last_obs))
        self.insert_counter += len(traj)
        if done:  # 如果 episode 结束，增加 episode 计数器
            self.episode_counter += 1

    def size(self):
        # 返回总步数，保持语义近似
        return sum(len(traj) - 1 for traj, _, _ in self.buffer)

    def sample_sequences(
        self, 
        min_total_steps: int, 
        replay_recent_frac: float = 1.0,
        replay_max_version_gap: float = float('inf'),
        current_policy_version: int = 0
    ):
        """
        采样若干条轨迹，使得它们的总步数 >= min_total_steps。
        
        Args:
            min_total_steps: 需要采样的最小步数
            replay_recent_frac: 最新比例窗口，只从最新 f% 数据采样 (0.0-1.0)
            replay_max_version_gap: 版本差窗口，只采样 version_gap <= max_gap 的数据
            current_policy_version: 当前策略版本号（用于版本差过滤）
        """
        buffer_list = list(self.buffer)
        
        # ========== 过滤逻辑 ==========
        # A. 最新比例窗口过滤
        if replay_recent_frac < 1.0:
            # 按 insert_step 排序，取最新的 f% 数据
            # 注意：每个 trajectory 中的第一个 Experience 的 insert_step 代表该 trajectory 的插入时间
            buffer_with_insert_step = [
                (idx, traj[0].insert_step if traj else 0) 
                for idx, (traj, _, _) in enumerate(buffer_list)
            ]
            buffer_with_insert_step.sort(key=lambda x: x[1], reverse=True)  # 降序：最新的在前
            
            num_to_keep = max(1, int(len(buffer_with_insert_step) * replay_recent_frac))
            recent_indices = set(idx for idx, _ in buffer_with_insert_step[:num_to_keep])
            buffer_list = [buffer_list[idx] for idx in recent_indices]
        
        # B. 版本差窗口过滤
        if replay_max_version_gap < float('inf'):
            filtered_buffer = []
            for traj, done, last_obs in buffer_list:
                # 检查轨迹中任意样本的 policy_version 是否满足条件
                # 为了性能，我们只检查第一个样本的版本
                if traj and (current_policy_version - traj[0].policy_version) <= replay_max_version_gap:
                    filtered_buffer.append((traj, done, last_obs))
            buffer_list = filtered_buffer
        
        # 检查过滤后是否还有足够数据
        filtered_size = sum(len(traj) - 1 for traj, _, _ in buffer_list)
        if filtered_size < min_total_steps:
            print(f"[Warning] 过滤后 buffer 大小 {filtered_size} < 请求的 {min_total_steps}，"
                  f"使用所有可用数据 (recent_frac={replay_recent_frac}, "
                  f"max_version_gap={replay_max_version_gap})")
            # 继续使用过滤后的数据，即使不足
        
        if not buffer_list:
            raise ValueError(f"过滤后 buffer 为空！无法采样。")
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
        # ========== 新增：诊断字段 ==========
        batch_policy_ver_list, batch_insert_step_list = [], []

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
            
            # 提取诊断字段
            batch_policy_ver_list.append(np.array([e.policy_version for e in traj], dtype=np.int32))
            batch_insert_step_list.append(np.array([e.insert_step for e in traj], dtype=np.int32))
            
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
            np.stack(batch_last_obs_list),                # [B, D]
            batch_policy_ver_list,    # List[np.ndarray(T,)] - 策略版本
            batch_insert_step_list    # List[np.ndarray(T,)] - 插入步数
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
            debug_log(f"RolloutWorker {self.wid} run() 启动，等待环境重置")
            # 使用确定性种子
            obs, info = self._reset_and_select_env()
            reward_sum, time_start, step_count_total = 0.0, time.time(), 0
            while True:
                # 直接使用状态向量作为模型输入
                action_env, action_token, logits, value, logp_mu, policy_version = ray.get(self.infer.request.remote(obs, deterministic=False))

                # 修正: 传入 discrete token 给环境
                nxt, r, term, trunc, info = self.env.step(action_token)
                reward_sum += r
                chunk_reward = r * REWARD_SCALE
                step_count_total += 1
                done = term or trunc
                
                # 计算 discount
                step_discount = GAMMA * (0.0 if done else 1.0)

                self.local_buffer.append((obs, action_token, chunk_reward, logits, value, logp_mu, step_discount, policy_version))
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
                    last_state, _, _, _, last_value, _, _, _ = self.local_buffer[-1]
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
            _, _, r, _, v, _, _, _ = traj_segment[i]
            nv = bootstrap_val if i == len(traj_segment) - 1 else traj_segment[i+1][4]
            delta = r + GAMMA * nv - v
            gae = delta + GAMMA * LAMBDA * gae
            advs.append(gae)
            rets.append(gae + v)
        advs.reverse(); rets.reverse()
        advs_np = np.array(advs, dtype=np.float32)

        traj: List[Experience] = []
        for i, (s, a_token, r_val, logits, _, logp_mu, step_discount, policy_ver) in enumerate(traj_segment):
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
                    policy_version=int(policy_ver),  # 新增：记录策略版本
                    # insert_step 将在 ReplayBuffer.add_trajectory 中自动填充
                    # episode_id 将在 ReplayBuffer.add_trajectory 中自动填充
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
                    action_env, action_token, _, _, _, _ = ray.get(self.infer.request.remote(obs, deterministic=True))

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
        
        # ========== 新增：policy_version 跟踪 ==========
        self.policy_version = 0  # 当前策略版本号，每次接收新权重时 +1

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

                # 获取当前 policy_version（所有请求使用同一个版本）
                current_policy_version = self.policy_version
                
                for i in range(len(promises_to_process)):
                    promises_to_process[i].set_result((
                        actions_env[i],           # 连续环境动作
                        action_tokens[i],         # 离散动作 token
                        logits[i],                # 对应的 logits
                        values[i],                # 价值估计
                        logp_mu_np[i],            # 行为策略 log-prob
                        current_policy_version    # 策略版本号
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
    
    def receive_and_update_weights(self, group_name):
        """覆盖基类方法，在接收权重后增加 policy_version"""
        # 调用基类方法接收权重
        super().receive_and_update_weights(group_name)
        # 增加版本号
        self.policy_version += 1
        # 可选：打印日志
        if self.actor_id == 0:  # 只让第一个推理器打印，减少日志噪音
            print(f"InferenceActor {self.actor_id}: 已更新到 policy_version={self.policy_version}")


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
        self.diag_every = int(os.environ.get("DIAG_EVERY_STEPS", DIAG_EVERY_STEPS))
        
        self.model = None
        self.optimizer = None
        self.base_model = None
        self.data_dtype = None
        self.next_ready_batch: Optional[Tuple] = None
        self.data_fetching_task = None
        self.super_batch_size = TRAIN_BATCH_SIZE * ACCUMULATION_STEPS
        self.global_step = 0
        
        # ========== 新增：policy_version 跟踪和采样过滤参数 ==========
        self.policy_version = 0  # 与 InferenceActor 同步，每次广播权重后 +1
        self.replay_recent_frac = REPLAY_RECENT_FRAC  # 将在 main() 中设置
        self.replay_max_version_gap = REPLAY_MAX_VERSION_GAP  # 将在 main() 中设置

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
    
    def set_sampling_filters(self, replay_recent_frac: float, replay_max_version_gap: float):
        """设置采样过滤参数"""
        self.replay_recent_frac = replay_recent_frac
        self.replay_max_version_gap = replay_max_version_gap
        if self.rank == 0:
            print(f"Trainer {self.rank}: 设置采样过滤 - recent_frac={replay_recent_frac}, max_version_gap={replay_max_version_gap}")
    
    def broadcast_weights(self, group_name):
        """覆盖基类方法，在广播权重后增加 policy_version"""
        # 调用基类方法广播权重
        super().broadcast_weights(group_name)
        # 增加版本号（只有 rank 0 会广播权重）
        if self.rank == 0:
            self.policy_version += 1

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
        import time as _time
        _setup_start = _time.time()
        print(f"[DS-setup][rank {self.rank}] 进入 setup_deepspeed_group, 时间戳: {_setup_start:.3f}")
        print(f"Trainer {self.rank}: 开始设置 DeepSpeed 环境变量...")
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["LOCAL_RANK"] = "0"
        
        # 为每个进程设置独立的 PyTorch 扩展目录，避免多进程编译冲突
        # 使用 PID 和端口号区分不同任务
        unique_ext_dir = f"/dev/shm/torch_ext_{os.getpid()}_{master_port}"
        os.environ["TORCH_EXTENSIONS_DIR"] = unique_ext_dir
        os.makedirs(unique_ext_dir, exist_ok=True)
        print(f"Trainer {self.rank}: PyTorch 扩展目录设置为: {unique_ext_dir}")
        
        print(f"Trainer {self.rank}: 环境变量设置完成 - RANK={self.rank}, WORLD_SIZE={self.world_size}, MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")

        print(f"[DS-setup][rank {self.rank}] before init_distributed (backend=nccl)")
        print(f"Trainer {self.rank}: 初始化分布式后端 (nccl)...")
        deepspeed.init_distributed(dist_backend="nccl")
        print(f"[DS-setup][rank {self.rank}] after init_distributed")
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
        print(f"Trainer {self.rank}: 开始初始化 DeepSpeed... 已用时: {_time.time() - _setup_start:.2f}s")
        try:
            _ds_init_start = _time.time()
            print(f"Trainer {self.rank}: [调试] 调用 deepspeed.initialize() 开始...")
            self.model, self.optimizer, _, _ = deepspeed.initialize(model=model, config=ds_config, model_parameters=optimizer_params)
            _ds_init_time = _time.time() - _ds_init_start
            print(f"[DS-setup][rank {self.rank}] after deepspeed.initialize, 耗时: {_ds_init_time:.2f}s")
            print(f"TrainerActor Rank {self.rank}: DeepSpeed 训练组 (ZeRO-2) 初始化完成。总用时: {_time.time() - _setup_start:.2f}s")
        except Exception as e:
            print(f"Trainer {self.rank}: DeepSpeed 初始化失败: {e}")
            print(f"Trainer {self.rank}: 已用时: {_time.time() - _setup_start:.2f}s")
            import traceback
            traceback.print_exc()
            raise

        print(f"[DS-setup][rank {self.rank}] before create data_fetching_task, 已用时: {_time.time() - _setup_start:.2f}s")
        self.data_fetching_task = asyncio.get_event_loop().create_task(self._data_fetching_loop())
        print(f"[DS-setup][rank {self.rank}] after create data_fetching_task")

        n_total = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数量: {n_total:,}, 可训练参数量: {n_trainable:,}")
        _total_setup_time = _time.time() - _setup_start
        print(f"[DS-setup][rank {self.rank}] setup_deepspeed_group done, 总用时: {_total_setup_time:.2f}s")

    async def save_agent(self, ckpt_dir: str, step: int):
        """
        只在 rank-0 上调用。调用 MLPActorCritic 内部的 save_model
        """
        os.makedirs(ckpt_dir, exist_ok=True)
        self.base_model.save_model(ckpt_dir, epoch=step)
        print(f"[Trainer {self.rank}] 已保存 checkpoint -> {ckpt_dir}/mlp_actor_critic_epoch_{step}.pt")
    
    async def dump_diagnostic_data(
        self,
        obs_sample: torch.Tensor,
        act_token_sample: torch.Tensor,
        logits_old_sample: torch.Tensor,
        policy_version_sample: torch.Tensor,
        insert_step_sample: torch.Tensor,
        dump_dir: str,
        step: int
    ):
        """
        导出用于绘制机制图的诊断数据
        
        Args:
            obs_sample: 观测样本 [N, state_dim]
            act_token_sample: 动作 token [N, action_dim]
            logits_old_sample: 旧策略 logits [N, action_dim, n_bins]
            policy_version_sample: 策略版本 [N]
            insert_step_sample: 插入步数 [N]
            dump_dir: 导出目录
            step: 当前训练步数
        """
        if self.rank != 0:  # 只在 rank 0 导出
            return
        
        os.makedirs(dump_dir, exist_ok=True)
        
        with torch.no_grad():
            # 前向传播获取当前策略的 logits
            action_logits, _ = self.model(obs_sample)
            
            # 计算 log-prob
            dist_old = torch.distributions.Categorical(logits=logits_old_sample)
            logp_old = dist_old.log_prob(act_token_sample)  # [N, action_dim]
            
            dist_new = torch.distributions.Categorical(logits=action_logits)
            logp_new = dist_new.log_prob(act_token_sample)  # [N, action_dim]
            
            # 计算 ratio
            ratio = torch.exp(logp_new - logp_old)  # [N, action_dim]
            logrho = logp_new - logp_old  # [N, action_dim]
            
            # 计算 staleness
            staleness_ver = self.policy_version - policy_version_sample.float()
            # ✅ 修正：使用相对 age
            age_steps = insert_step_sample.float().max() - insert_step_sample.float()
            
            # 转换为 numpy
            dump_data = {
                'logp_old': logp_old.cpu().numpy(),  # behavior policy log-prob
                'logp_new': logp_new.cpu().numpy(),  # current policy log-prob
                'ratio': ratio.cpu().numpy(),  # importance ratio
                'logrho': logrho.cpu().numpy(),  # log importance ratio
                'staleness_ver': staleness_ver.cpu().numpy(),  # version gap
                'age_steps': age_steps.cpu().numpy(),  # age in steps
                'policy_version': policy_version_sample.cpu().numpy(),
                'insert_step': insert_step_sample.cpu().numpy(),
                'current_policy_version': self.policy_version,
                'current_step': step,
            }
            
            # 保存为 npz 格式（便于 Python 读取）
            dump_path = os.path.join(dump_dir, f"diagnostic_step_{step}.npz")
            np.savez_compressed(dump_path, **dump_data)
            print(f"[Trainer {self.rank}] 已导出诊断数据 -> {dump_path} ({len(logp_old)} samples)")

    def _compute_diagnostic_metrics(
        self,
        ratio: torch.Tensor,  # [B, ACTION_DIM] or [B]
        advantage: torch.Tensor,  # [B]
        policy_version: torch.Tensor,  # [B]
        insert_step: torch.Tensor,  # [B]
        current_policy_version: int,
        current_step: int,
        clip_mode: str,
        clip_eps: float = 0.2,
        clip_params: Dict[str, Any] = None
    ) -> Dict[str, float]:
        """
        计算诊断指标：staleness、ratio 分布、ESS、PG Active/Dead、有效贡献等
        
        ✅ 离散 Token 下的 5 个必要条件（已实现）：
        1. behavior_logp 保存的是被执行 token 的 logprob（不是全 logits）
           - 实现位置：InferenceActor._loop() L986, dist.log_prob(action_tokens)
           - 形状：[B, ACTION_DIM]，每个维度是该 token 的 log-prob
        
        2. current_logp 也是对同一个 token 的 logprob（gather）
           - 实现位置：run_training_epoch() L1700, dist.log_prob(mini_act_token)
           - 与 behavior_logp 使用相同的 action_tokens
        
        3. ratio = exp(current_logp - behavior_logp) 得到的是 [B, ACTION_DIM] 标量
           - 实现位置：run_training_epoch() L1710
           - 每个元素表示该维度该 token 的重要性权重
        
        4. dead 的 A 用的是训练 loss 用的 A（normalized_adv）
           - 实现位置：本函数中，传入的 advantage 参数是 normalized_adv
           - ⚠️ 关键修正：L1716 必须传入 normalized_adv 而不是 mini_adv
        
        5. old/new 分桶用的是绝对 version gap 阈值，不是分位数
           - 实现位置：本函数中，NEW_THRESHOLD=2, OLD_THRESHOLD=10
           - 跨 run 可比，便于论文报告
        
        新增核心指标：
        - PG_Active_Frac / PG_Dead_Frac (hard clip) - 基于 normalized_adv
        - Outside_Clip_Frac (soft clip, 按 ratio)
        - Suppressed_Frac (soft clip, 按权重阈值 w < 1e-3)
        - U_Mean, U_P50/P90/P99, U_Max（贡献权重）
        - ESS_Eff, ESS_Eff_Norm（基于有效贡献的ESS）
        - 分桶指标：_Old / _New（基于绝对阈值）
        - NearZero_U_Frac (u < 1e-3)
        """
        with torch.no_grad():
            metrics = {}
            clip_params = clip_params or {}
            
            # ==================== 预处理 ====================
            # ratio 可能是 [B, ACTION_DIM]，需要展平
            ratio_flat = ratio.reshape(-1)
            
            # advantage 需要扩展到与 ratio_flat 相同的维度
            if ratio.dim() == 2:  # [B, ACTION_DIM]
                adv_expanded = advantage.unsqueeze(1).expand_as(ratio).reshape(-1)
            else:  # [B]
                adv_expanded = advantage
            
            # ==================== 3.1 Staleness（样本"新旧程度"）====================
            staleness_ver = current_policy_version - policy_version.float()  # Δv
            metrics['staleness_ver_mean'] = staleness_ver.mean().item()
            metrics['staleness_ver_p95'] = torch.quantile(staleness_ver, 0.95).item()
            
            # ✅ 修正：Age（步数差）- 使用 batch 内相对 age，避免单位不一致
            # 原因：current_step 是更新步数，insert_step 是插入计数器，单位不同
            # 相对 age：batch 中最新样本 - 当前样本的插入步数差
            age_steps = insert_step.float().max() - insert_step.float()
            metrics['age_steps_mean'] = age_steps.mean().item()
            metrics['age_steps_p95'] = torch.quantile(age_steps, 0.95).item() if age_steps.numel() > 0 else 0.0
            metrics['age_steps_max'] = age_steps.max().item()  # batch 内最老的样本
            
            # ✅ 修正：分桶使用绝对阈值（跨 run 可比，便于论文报告）
            # 根据实际量级调整：Δv 在 1e4-2e4 范围
            NEW_THRESHOLD = 500   # Δv <= 500 视为 "新鲜"
            OLD_THRESHOLD = 10000  # Δv >= 10000 视为 "陈旧"
            
            new_mask_batch = staleness_ver <= NEW_THRESHOLD  # [B]
            old_mask_batch = staleness_ver >= OLD_THRESHOLD  # [B]
            
            # 扩展到与 ratio_flat 相同的维度
            if ratio.dim() == 2:
                new_mask = new_mask_batch.unsqueeze(1).expand_as(ratio).reshape(-1)
                old_mask = old_mask_batch.unsqueeze(1).expand_as(ratio).reshape(-1)
            else:
                new_mask = new_mask_batch
                old_mask = old_mask_batch
            
            # ==================== A. 分桶组成（Bucket Composition）====================
            # A1) 绝对分桶占比
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
            
            # ==================== B. 相对陈旧度（Relative Staleness）====================
            # B1) 相对陈旧度：Δv / max(v_current, 1)
            staleness_ratio = staleness_ver / max(current_policy_version, 1)
            metrics['staleness_ratio_mean'] = staleness_ratio.mean().item()
            metrics['staleness_ratio_p95'] = torch.quantile(staleness_ratio, 0.95).item()
            
            # B2) 相对阈值分桶（更稳定，跨算法可比）
            NEW_RATIO_THRESHOLD = 0.05   # 落后 <= 5%
            OLD_RATIO_THRESHOLD = 0.5    # 落后 >= 50%
            
            new_mask_ratio_batch = staleness_ratio <= NEW_RATIO_THRESHOLD
            old_mask_ratio_batch = staleness_ratio >= OLD_RATIO_THRESHOLD
            
            # 扩展到与 ratio_flat 相同的维度
            if ratio.dim() == 2:
                new_mask_ratio = new_mask_ratio_batch.unsqueeze(1).expand_as(ratio).reshape(-1)
                old_mask_ratio = old_mask_ratio_batch.unsqueeze(1).expand_as(ratio).reshape(-1)
            else:
                new_mask_ratio = new_mask_ratio_batch
                old_mask_ratio = old_mask_ratio_batch
            
            metrics['staleness_old_frac_ratio'] = old_mask_ratio_batch.float().mean().item()
            metrics['staleness_new_frac_ratio'] = new_mask_ratio_batch.float().mean().item()
            
            # ==================== 3.2 Ratio / log-ratio 分布（核心诊断）====================
            metrics['rho_mean'] = ratio_flat.mean().item()
            metrics['rho_p50'] = torch.median(ratio_flat).item()
            metrics['rho_p90'] = torch.quantile(ratio_flat, 0.90).item()
            metrics['rho_p99'] = torch.quantile(ratio_flat, 0.99).item()
            metrics['rho_max'] = ratio_flat.max().item()
            
            # log-ratio
            logrho = torch.log(ratio_flat.clamp(min=1e-8))
            metrics['logrho_mean'] = logrho.mean().item()
            metrics['abs_logrho_p95'] = torch.quantile(torch.abs(logrho), 0.95).item()
            
            # ==================== 新增指标 1: PG_Active_Frac / PG_Dead_Frac (hard clip) ====================
            # 或 Suppressed_Frac (soft clip)
            
            if clip_mode == "clip":  # Hard clip (PPO)
                # Dead gradient: (A > 0 and ρ > 1+ε) or (A < 0 and ρ < 1-ε)
                dead_mask = ((adv_expanded > 0) & (ratio_flat > (1 + clip_eps))) | \
                           ((adv_expanded < 0) & (ratio_flat < (1 - clip_eps)))
                
                metrics['pg_dead_frac'] = dead_mask.float().mean().item()
                metrics['pg_active_frac'] = 1.0 - metrics['pg_dead_frac']
                
                # D2) 分桶统计（绝对阈值） - hard clip 核心机制证据
                if new_mask.any():
                        metrics['pg_dead_frac_new'] = dead_mask[new_mask].float().mean().item()
                        metrics['pg_active_frac_new'] = 1.0 - metrics['pg_dead_frac_new']
                if old_mask.any():
                    metrics['pg_dead_frac_old'] = dead_mask[old_mask].float().mean().item()
                    metrics['pg_active_frac_old'] = 1.0 - metrics['pg_dead_frac_old']
                
                # D2) 分桶统计（相对阈值）
                if new_mask_ratio.any():
                    metrics['pg_dead_frac_new_ratio'] = dead_mask[new_mask_ratio].float().mean().item()
                    metrics['pg_active_frac_new_ratio'] = 1.0 - metrics['pg_dead_frac_new_ratio']
                if old_mask_ratio.any():
                    metrics['pg_dead_frac_old_ratio'] = dead_mask[old_mask_ratio].float().mean().item()
                    metrics['pg_active_frac_old_ratio'] = 1.0 - metrics['pg_dead_frac_old_ratio']
                
            else:  # Soft clip 或其他非硬裁剪模式
                # ✅ 修正：Soft clip 应该用权重阈值定义 suppressed
                # 但同时保留 outside_clip 作为辅助指标
                outside_clip = (ratio_flat < (1 - clip_eps)) | (ratio_flat > (1 + clip_eps))
                metrics['outside_clip_frac'] = outside_clip.float().mean().item()  # 重命名以区分
                
                # 分桶统计（outside_clip）
                if new_mask.any():
                    metrics['outside_clip_frac_new'] = outside_clip[new_mask].float().mean().item()
                if old_mask.any():
                    metrics['outside_clip_frac_old'] = outside_clip[old_mask].float().mean().item()
            
            # ==================== 新增指标 2: 贡献权重 U（统一定义）====================
            # U = 有效贡献强度
            
            if clip_mode == "clip":  # Hard clip
                # u_hard = ρ * (1 - dead)
                # dead 样本的贡献直接置 0
                u = ratio_flat * (~dead_mask).float()
            else:  # Soft clip
                # u_soft = w(ρ) * ρ
                # 需要计算 soft clip 的权重 w(ρ)
                
                if clip_mode in ("soft_clip", "soft_clip_alpha-1", "soft_clip_alpha-2"):
                    # w = (1/max(ρ, 1/ρ))^alpha ，并引入动态 alpha: alpha*(1+sigmoid(-A))
                    if clip_mode == "soft_clip_alpha-1":
                        base_alpha = 1.0
                    elif clip_mode == "soft_clip_alpha-2":
                        base_alpha = 2.0
                    else:
                        base_alpha = float(clip_params.get("soft_clip_alpha", 1.0))
                    alpha_tensor = base_alpha * (1 + torch.sigmoid(-adv_expanded))
                    diff = torch.maximum(ratio_flat, 1.0 / ratio_flat.clamp(min=1e-8))
                    w_soft = (1.0 / diff) ** alpha_tensor
                    u = w_soft * ratio_flat
                
                elif clip_mode in ("log_gauss_clip",):
                    # w = exp(-0.5 * (log(r+eps)/sigma)^2)
                    eps = 1e-9
                    sigma = float(clip_params.get("sigma", 1))
                    r = ratio_flat.clamp_min(eps).detach()
                    w_gauss = torch.exp(-0.5 * (torch.log(r) / sigma) ** 2)
                    w = w_gauss
                    u = w_gauss * r
                    
                elif clip_mode in ("sapo_soft_clip", "sapo", "sapo_gate"):
                    # w = gate(ρ) = (4/τ) * sigmoid(τ*(ρ-1))
                    tau_pos = float(clip_params.get("tau_pos", 1.0))
                    tau_neg = float(clip_params.get("tau_neg", 2.0))
                    # 与训练一致的（可选）clamp：尽量宽松
                    ratio_min = float(clip_params.get("ratio_min", 1e-8))
                    ratio_max = float(clip_params.get("ratio_max", 1e8))
                    r = ratio_flat.clamp(ratio_min, ratio_max)
                    # 根据 advantage 符号选择 tau
                    tau = torch.where(adv_expanded > 0, 
                                     torch.full_like(ratio_flat, tau_pos),
                                     torch.full_like(ratio_flat, tau_neg))
                    x = tau * (r - 1.0)
                    p = torch.sigmoid(x)
                    gate = p * (4.0 / tau)
                    # ✅ SAPO 的有效梯度权重：w(r) = 4 p (1-p)
                    w_sapo = 4.0 * p * (1.0 - p)
                    w = w_sapo
                    u = w_sapo * r
                    
                else:
                    # 其他模式：直接使用 ρ 作为权重
                    u = ratio_flat
            
            # U 的统计量
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
            
            # ✅ 新增：基于权重阈值的 suppressed_frac（仅 soft clip）
            if clip_mode != "clip":
                # 计算权重 w（从 u 反推）
                if clip_mode in ("soft_clip", "soft_clip_alpha-1", "soft_clip_alpha-2"):
                    w = u / ratio_flat.clamp(min=1e-8)
                elif clip_mode in ("sapo_soft_clip", "sapo", "sapo_gate"):
                    w = u / ratio_flat.clamp(min=1e-8)
                else:
                    w = u / ratio_flat.clamp(min=1e-8)
                
                # suppressed：权重被显著抑制（< 某阈值）
                w_threshold = 1e-3  # 权重 < 1e-3 视为被抑制
                suppressed_mask = w < w_threshold
                metrics['suppressed_frac'] = suppressed_mask.float().mean().item()
                
                # 分桶统计
                if new_mask.any():
                    metrics['suppressed_frac_new'] = suppressed_mask[new_mask].float().mean().item()
                if old_mask.any():
                    metrics['suppressed_frac_old'] = suppressed_mask[old_mask].float().mean().item()
            
            # ==================== 新增指标 3: ESS_Eff（基于有效贡献的ESS）====================
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
                metrics['ess_eff_norm_new_abs'] = (ess_eff_new / u_new.numel()).item()  # 保持命名一致性
            if old_mask.any():
                u_old = u[old_mask]
                u_old_sum = u_old.sum()
                u_old_sq_sum = (u_old * u_old).sum()
                ess_eff_old = (u_old_sum * u_old_sum) / (u_old_sq_sum + 1e-12)
                metrics['ess_eff_norm_old'] = (ess_eff_old / u_old.numel()).item()
                metrics['ess_eff_norm_old_abs'] = (ess_eff_old / u_old.numel()).item()  # E1) 旧桶ESS
            
            # E1) 基于相对阈值的分桶 ESS
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
            
            # ==================== 新增指标 4: NearZero_U_Frac ====================
            # 几乎没贡献的样本比例
            near_zero_threshold = 1e-3
            near_zero_mask = u < near_zero_threshold
            metrics['nearzero_u_frac'] = near_zero_mask.float().mean().item()
            
            # D1) 分桶统计（绝对阈值） - 核心机制证据
            if new_mask.any():
                metrics['nearzero_u_frac_new'] = near_zero_mask[new_mask].float().mean().item()
            if old_mask.any():
                metrics['nearzero_u_frac_old'] = near_zero_mask[old_mask].float().mean().item()
            
            # D1) 分桶统计（相对阈值）
            if new_mask_ratio.any():
                metrics['nearzero_u_frac_new_ratio'] = near_zero_mask[new_mask_ratio].float().mean().item()
            if old_mask_ratio.any():
                metrics['nearzero_u_frac_old_ratio'] = near_zero_mask[old_mask_ratio].float().mean().item()
            
            # ==================== 新增指标 5: WeightShare（数据贡献占比）====================
            # 计算不同 staleness 桶的有效贡献占比
            # WeightShareOld = sum(u_old) / sum(u_all)
            # 这个指标可以直观地看出"旧数据"对梯度更新的实际贡献比例
            u_sum_all = u.sum()
            if old_mask.any() and u_sum_all > 0:
                u_sum_old = u[old_mask].sum()
                metrics['weight_share_old'] = (u_sum_old / u_sum_all).item()
            else:
                metrics['weight_share_old'] = 0.0
            
            if new_mask.any() and u_sum_all > 0:
                u_sum_new = u[new_mask].sum()
                metrics['weight_share_new'] = (u_sum_new / u_sum_all).item()
            else:
                metrics['weight_share_new'] = 0.0
            
            # ==================== C. 旧数据贡献占比（OldUShare - 核心机制证据）====================
            # C1) 基于绝对阈值的贡献占比
            if old_mask.any() and u_sum_all > 0:
                metrics['contribution_old_u_share'] = (u[old_mask].sum() / u_sum_all).item()
            else:
                metrics['contribution_old_u_share'] = 0.0
            
            if new_mask.any() and u_sum_all > 0:
                metrics['contribution_new_u_share'] = (u[new_mask].sum() / u_sum_all).item()
            else:
                metrics['contribution_new_u_share'] = 0.0
            
            # C2) 基于相对阈值的贡献占比（更稳定）
            if old_mask_ratio.any() and u_sum_all > 0:
                metrics['contribution_old_u_share_ratio'] = (u[old_mask_ratio].sum() / u_sum_all).item()
            else:
                metrics['contribution_old_u_share_ratio'] = 0.0
            
            if new_mask_ratio.any() and u_sum_all > 0:
                metrics['contribution_new_u_share_ratio'] = (u[new_mask_ratio].sum() / u_sum_all).item()
            else:
                metrics['contribution_new_u_share_ratio'] = 0.0
            
            # C3) 基于 |u*A| 的贡献占比（更贴近实际梯度贡献）
            # OldUShare_AbsGradProxy：用 advantage 做代理，反映"对 policy gradient 的实际贡献"
            # 原因：哪怕 u 不小，如果 A 很小，它对更新也没贡献
            u_grad_proxy = torch.abs(u * adv_expanded)  # |u * A|
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
            
            # ==================== 保留原有指标：clip_frac（向后兼容）====================
            clip_mask = (ratio_flat < (1 - clip_eps)) | (ratio_flat > (1 + clip_eps))
            metrics['clip_frac'] = clip_mask.float().mean().item()
            
            if new_mask.any():
                metrics['clip_frac_new'] = clip_mask[new_mask].float().mean().item()
            if old_mask.any():
                metrics['clip_frac_old'] = clip_mask[old_mask].float().mean().item()
            
            # ==================== 保留原有指标：基础 ESS（向后兼容）====================
            w = ratio_flat
            w_sum = w.sum()
            w_sq_sum = (w * w).sum()
            ess = (w_sum * w_sum) / (w_sq_sum + 1e-8)
            metrics['ess'] = ess.item()
            metrics['ess_norm'] = (ess / w.numel()).item()
            
            return metrics

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
                # 传递采样过滤参数
                (obs_seq_list, act_seq_list, rew_seq_list, disc_seq_list, logp_seq_list, 
                 adv_seq_list, logits_seq_list, vtarg_seq_list, done_seq, last_obs_seq,
                 policy_ver_list, insert_step_list) = \
                    await self.replay_buffer.sample_sequences.remote(
                        self.super_batch_size,
                        replay_recent_frac=self.replay_recent_frac,
                        replay_max_version_gap=self.replay_max_version_gap,
                        current_policy_version=self.policy_version
                    )
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
                    
                    # ========== 新增：诊断字段 ==========
                    'policy_ver_list': policy_ver_list,  # List[np.ndarray(T,)] - 策略版本
                    'insert_step_list': insert_step_list,  # List[np.ndarray(T,)] - 插入步数
                    
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
        diag_eps = float(CLIP_PARAMS.get("clip_eps", CLIP_EPS)) if CLIP_PARAMS else CLIP_EPS
        diag_outside_clip_ratio = None
        diag_dead_grad_ratio = None
        diag_ess_norm = None
        diag_image = None
        diag_image_corr = None
        diag_every = self.diag_every
        do_diag = False
        
        # ========== 数据导出标志 ==========
        should_dump = (self.global_step % DUMP_INTERVAL == 0) and (self.global_step > 0)
        dump_sample_data = None

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
        # ========== 新增：诊断字段 ==========
        policy_ver_list = current_batch['policy_ver_list']
        insert_step_list = current_batch['insert_step_list']

        # 对每个序列单独计算 V-trace（不 padding，单独处理）
        all_obs_flat, all_act_flat, all_pg_adv_flat, all_vs_flat, all_logits_old_flat = [], [], [], [], []
        # ========== 新增：诊断字段 ==========
        all_policy_ver_flat, all_insert_step_flat = [], []
        
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
            
            # ========== 新增：收集诊断字段 ==========
            policy_ver_seq = torch.tensor(policy_ver_list[seq_idx], dtype=torch.int32, device=device)  # [T]
            insert_step_seq = torch.tensor(insert_step_list[seq_idx], dtype=torch.int32, device=device)  # [T]
            all_policy_ver_flat.append(policy_ver_seq[:-1])  # [T-1]
            all_insert_step_flat.append(insert_step_seq[:-1])  # [T-1]
        
        # 拼接所有序列的结果
        obs_t = torch.cat(all_obs_flat, dim=0)     # [total_steps, state_dim]
        act_token_t = torch.cat(all_act_flat, dim=0)  # [total_steps, ACTION_DIM]
        pg_adv_t = torch.cat(all_pg_adv_flat, dim=0)  # [total_steps]
        vs_t = torch.cat(all_vs_flat, dim=0)       # [total_steps]
        logits_old_t = torch.cat(all_logits_old_flat, dim=0)  # [total_steps, ACTION_DIM, n_bins]
        # ========== 新增：诊断字段 ==========
        policy_version_t = torch.cat(all_policy_ver_flat, dim=0)  # [total_steps]
        insert_step_t = torch.cat(all_insert_step_flat, dim=0)    # [total_steps]
        
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
        # 诊断字段也需要 shuffle
        policy_version_t = policy_version_t[indices]
        insert_step_t = insert_step_t[indices]
        
        # ========== 采样数据用于导出（如果需要）==========
        if should_dump and self.rank == 0:
            num_samples_to_dump = min(DUMP_NUM_SAMPLES, current_batch_size)
            dump_sample_data = {
                'obs': obs_t[:num_samples_to_dump].clone(),
                'act_token': act_token_t[:num_samples_to_dump].clone(),
                'logits_old': logits_old_t[:num_samples_to_dump].clone(),
                'policy_version': policy_version_t[:num_samples_to_dump].clone(),
                'insert_step': insert_step_t[:num_samples_to_dump].clone(),
            }

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
        epoch_grad_norms, epoch_ev_list = [], []
        
        # ========== 新增：诊断指标存储 ==========
        diagnostic_metrics = {}

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
            # 诊断字段
            mini_policy_ver = policy_version_t[start:end]
            mini_insert_step = insert_step_t[start:end]

            normalized_adv = (mini_pg_adv - global_mean) / (global_std + 1e-8)

            # 前向（重新计算，因为需要当前策略的输出）
            action_logits, value = self.model(mini_obs)
            value = value.to(torch.float32)

            # V-trace value loss: (value - vs)^2
            value_loss = VF_COEF * torch.mean((value - mini_vs) ** 2)

            # 解释方差（Explained Variance），衡量 value 预测对目标的解释度
            with torch.no_grad():
                value_pred = value.squeeze(-1) if value.dim() > 1 else value
                target = mini_vs
                var_target = torch.var(target, unbiased=False)
                if var_target < 1e-12:
                    ev = 0.0
                else:
                    ev = 1.0 - torch.var(target - value_pred, unbiased=False) / (var_target + 1e-12)
                epoch_ev_list.append(float(ev))

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
                
                # ========== 计算诊断指标（只在第一个 mini-batch 时计算）==========
                if i == 0 and self.global_step >= POLICY_TRAIN_START_STEP:
                    diagnostic_metrics = self._compute_diagnostic_metrics(
                        ratio=ratio,
                        advantage=normalized_adv,  # ✅ 修正：传入 normalized_adv（与 loss 计算一致）
                        policy_version=mini_policy_ver,
                        insert_step=mini_insert_step,
                        current_policy_version=self.policy_version,
                        current_step=self.global_step,
                        clip_mode=CLIP_MODE,  # 传入 clip_mode
                        clip_eps=diag_eps,
                        clip_params=CLIP_PARAMS  # 传入 clip_params
                    )
                
                if diag_outside_clip_ratio is None:
                    with torch.no_grad():
                        adv_expand = adv_unsqueezed.expand_as(ratio)
                        outside = ((ratio < 1 - diag_eps) | (ratio > 1 + diag_eps)).float().mean()
                        dead_grad = (((ratio > 1 + diag_eps) & (adv_expand > 0)) | ((ratio < 1 - diag_eps) & (adv_expand < 0))).float().mean()
                        w = ratio.clamp(1 - diag_eps, 1 + diag_eps)
                        ess = (w.sum() ** 2) / (w.pow(2).sum() + 1e-8)
                        diag_outside_clip_ratio = outside.item()
                        diag_dead_grad_ratio = dead_grad.item()
                        diag_ess_norm = (ess / w.numel()).item()
                        do_diag = (self.rank == 0) and (self.global_step >= POLICY_TRAIN_START_STEP) and (self.global_step % diag_every == 0)
                        if do_diag:
                            p_old = torch.exp(logp_old)
                            p_new = torch.exp(logp)
                            diag_image, diag_image_corr = make_policy_prob_diag_image(p_old, p_new, max_points=20000)
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
            # policy update proxy：使用全局梯度范数衡量更新强度
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
        if epoch_grad_norms:
            perf_metrics["grad_norm_mean"] = float(np.mean(epoch_grad_norms))
        if epoch_ev_list:
            perf_metrics["explained_variance_mean"] = float(np.mean(epoch_ev_list))
        perf_metrics["diag_outside_clip_ratio"] = diag_outside_clip_ratio
        perf_metrics["diag_dead_grad_ratio"] = diag_dead_grad_ratio
        perf_metrics["diag_ess_norm"] = diag_ess_norm
        perf_metrics["diag_every_steps"] = self.diag_every
        if diag_image is not None:
            perf_metrics["diag_old_new_prob_image"] = diag_image
            perf_metrics["diag_old_new_prob_corr"] = diag_image_corr
        
        # ========== 合并诊断指标 ==========
        perf_metrics.update(diagnostic_metrics)
        
        # ========== 添加导出数据标志 ==========
        perf_metrics['dump_sample_data'] = dump_sample_data

        return avg_loss, avg_p_loss, avg_v_loss, avg_e_loss, avg_kl_loss, current_lrs, self.global_step, avg_ent, avg_kl_div, avg_clip_ratio, perf_metrics



# ================================================================
# 5. 主逻辑
# ================================================================
def main():
    args = parse_args()
    global TASK_NAME, BENCHMARK, NUM_TRAINER_GPUS, NUM_ROLLOUT_WORKERS, NUM_EVAL_WORKERS, TRAIN_BATCH_SIZE, TRAIN_ITERS, SEED, CLIP_MODE, CLIP_PARAMS
    global REPLAY_RECENT_FRAC, REPLAY_MAX_VERSION_GAP, LOG_BACKEND
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
    TRAIN_ITERS = args.train_iters
    SEED = args.seed
    CLIP_MODE = args.clip_mode
    
    # 采样过滤参数
    REPLAY_RECENT_FRAC = args.replay_recent_frac
    REPLAY_MAX_VERSION_GAP = args.replay_max_version_gap
    
    # 日志后端配置
    LOG_BACKEND = args.log_backend

    # 加载裁剪配置并应用对应超参
    clip_config = load_clip_config(args.clip_config)
    CLIP_PARAMS = select_clip_params(CLIP_MODE, clip_config)

    debug_log(f"[Clip] 使用模式: {CLIP_MODE}, 配置: {CLIP_PARAMS}")
    debug_log(f"[Sampling] 采样过滤 - recent_frac={REPLAY_RECENT_FRAC}, max_version_gap={REPLAY_MAX_VERSION_GAP}")
    debug_log(f"[Training] 训练迭代次数: {TRAIN_ITERS}")
    debug_log(f"[Logging] 日志后端: {LOG_BACKEND}")
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
    try:
        debug_log(f"Ray 集群资源: {ray.cluster_resources()}")
        debug_log(f"Ray 当前可用资源: {ray.available_resources()}")
    except Exception as e:
        debug_log(f"获取 Ray 资源信息失败: {e}")

    # ========== 新的日志目录结构：按任务和clip模式组织 ==========
    # 格式化 clip 参数用于路径
    clip_params_str = format_clip_params_for_path(CLIP_PARAMS)
    if clip_params_str:
        clip_suffix = f"{CLIP_MODE}_{clip_params_str}"
    else:
        clip_suffix = CLIP_MODE
    
    # 新结构: runs/MetaWorld/{TASK_NAME}/{CLIP_MODE}/run_{timestamp}_seed{SEED}
    # 这样同一个任务的不同运行都在同一个子文件夹下
    
    # 处理 SwanLab 目录（如果外部已传入 SWANLAB_DIR，批量脚本会传入 task/swanlab_all）
    swanlab_dir_env = os.environ.get("SWANLAB_DIR", "").strip()
    if swanlab_dir_env:
        # 如果外部传入的路径与当前 TASK 不匹配，则自动纠正到当前 TASK 名
        if TASK_NAME not in swanlab_dir_env and swanlab_dir_env.rstrip("/").endswith("swanlab_all"):
            parent_dir = os.path.dirname(os.path.dirname(swanlab_dir_env))
            swanlab_base_dir = os.path.join(parent_dir, TASK_NAME, "swanlab_all")
        else:
            swanlab_base_dir = swanlab_dir_env
    else:
        swanlab_base_dir = f"runs/MetaWorld/{TASK_NAME}/{clip_suffix}"
    
    # 处理 TensorBoard 目录（如果外部已传入 TENSORBOARD_DIR，批量脚本会传入 task/tensorboard_all）
    tensorboard_dir_env = os.environ.get("TENSORBOARD_DIR", "").strip()
    if tensorboard_dir_env:
        # ✅ 修复：检查传入的路径是否与当前任务匹配，如果不匹配则纠正
        if TASK_NAME not in tensorboard_dir_env and tensorboard_dir_env.rstrip("/").endswith("tensorboard_all"):
            # 环境变量中的任务名与当前任务不匹配，使用当前任务名替换
            parent_dir = os.path.dirname(os.path.dirname(tensorboard_dir_env))
            tensorboard_base_dir = os.path.join(parent_dir, TASK_NAME, "tensorboard_all")
            debug_log(f"[修复] TensorBoard 路径任务名不匹配，从 {tensorboard_dir_env} 纠正为 {tensorboard_base_dir}")
        else:
            # 使用批量脚本指定的共享目录
            tensorboard_base_dir = tensorboard_dir_env
    else:
        # 使用默认目录（与 SwanLab 相同的结构）
        tensorboard_base_dir = f"runs/MetaWorld/{TASK_NAME}/{clip_suffix}"
    
    os.makedirs(swanlab_base_dir, exist_ok=True)
    os.makedirs(tensorboard_base_dir, exist_ok=True)
    
    # 当前运行的具体名称，带上 clip 信息和时间戳便于调试
    current_run_name = f"run_{clip_suffix}_{int(time.time())}_seed{SEED}"
    
    # SwanLab 使用 swanlab_base_dir
    task_log_base_dir = swanlab_base_dir
    
    # TensorBoard 使用独立的目录
    log_dir = os.path.join(tensorboard_base_dir, current_run_name)
    
    # 设置 SwanLab 离线日志目录（如果使用离线模式）
    use_offline_mode = os.environ.get("SWANLAB_OFFLINE", "0").lower() in ("1", "true", "yes")
    if use_offline_mode:
        # 将 SwanLab 日志保存到 task_log_base_dir（可能是外部传入的 swanlab_all）
        os.environ["SWANLAB_DIR"] = task_log_base_dir
        debug_log(f"SwanLab 离线日志将保存到: {task_log_base_dir}")
    
    # 导出任务基础目录到环境变量，供 shell 脚本使用
    os.environ["TASK_LOG_BASE_DIR"] = task_log_base_dir
    os.environ["CURRENT_RUN_NAME"] = current_run_name
    
    # 初始化 TensorBoard（始终启用）
    use_tensorboard = LOG_BACKEND in ("tensorboard", "both")
    if use_tensorboard:
        writer = SummaryWriter(log_dir)
        debug_log(f"TensorBoard 日志将保存在: {log_dir}")
    else:
        writer = None
        debug_log("TensorBoard 已禁用")
    
    stats_actor = StatsActor.remote(window_size=MOVING_AVG_WINDOW)

    # 初始化 SwanLab（根据 LOG_BACKEND 决定）
    use_swanlab = LOG_BACKEND in ("swanlab", "both")
    swanlab_available = False
    
    if use_swanlab:
        if swanlab is None:
            debug_log("[Warn] SwanLab未安装或已禁用，将仅使用TensorBoard")
            use_swanlab = False
        else:
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
            try:
                swanlab_init_kwargs = {
                    "project": "MetaWorld-PPO-Benchmark",
                    "experiment_name": f"{TASK_NAME}_{CLIP_MODE}_{current_run_name}",
                    "description": f"MetaWorld PPO Training - Task: {TASK_NAME}, Clip: {CLIP_MODE}, Seed: {SEED}",
                    "config": swanlab_config,
                }
                if use_offline_mode:
                    swanlab_init_kwargs["mode"] = "offline"
                    swanlab_init_kwargs["logdir"] = task_log_base_dir
                    debug_log(f"SwanLab 使用离线模式，日志目录: {task_log_base_dir}")
                else:
                    debug_log("SwanLab 使用在线模式（需要外网连接）")
                
                swanlab.init(**swanlab_init_kwargs)
                swanlab_available = True
                debug_log("SwanLab 初始化成功")
            except Exception as e:
                debug_log(f"SwanLab 初始化失败: {e}，将仅使用TensorBoard")
                swanlab_available = False
            else:
                # 合并/重命名 SwanLab run-* 目录，便于与脚本 run_ 命名一致
                consolidate_swanlab_run_dir(task_log_base_dir, current_run_name)
    else:
        debug_log("SwanLab 已禁用，仅使用TensorBoard")
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
            # 确保日志目录存在（若为符号链接也直接使用，不重复创建）
            if not os.path.exists(log_dir):
                try:
                    os.makedirs(log_dir, exist_ok=True)
                except FileExistsError:
                    pass
            
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
    
    # ========== 新增：记录运行信息到文件 ==========
    def save_run_info():
        """保存当前运行的信息到 .run_info 文件"""
        try:
            run_info_file = os.path.join(log_dir, ".run_info")
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            # 格式化 clip 参数为字符串
            clip_params_display = []
            if CLIP_PARAMS:
                for key, value in sorted(CLIP_PARAMS.items()):
                    if value is not None:
                        clip_params_display.append(f"{key}={value}")
            clip_params_str = ", ".join(clip_params_display) if clip_params_display else "-"
            
            with open(run_info_file, "w") as f:
                f.write(f"# 运行信息\n")
                f.write(f"TASK_NAME={TASK_NAME}\n")
                f.write(f"CLIP_MODE={CLIP_MODE}\n")
                f.write(f"CLIP_PARAMS={clip_params_str}\n")
                f.write(f"SEED={SEED}\n")
                f.write(f"START_TIME={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"TRAIN_ITERS={TRAIN_ITERS}\n")
                f.write(f"LOG_DIR={log_dir}\n")
                f.write(f"TASK_LOG_BASE_DIR={task_log_base_dir}\n")
                f.write(f"CURRENT_RUN_NAME={current_run_name}\n")
            
            debug_log(f"✅ 运行信息已保存到: {run_info_file}")
        except Exception as e:
            debug_log(f"⚠️ 保存运行信息失败: {e}")
    
    save_run_info()
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
    debug_log(f"TrainerActor 数: {len(trainer_group)}, ReplayBuffer 数: {len(replay_buffers)}, InferenceActor 数: {len(inference_pool)}")
    
    # 设置采样过滤参数
    print(f"\n--- 设置采样过滤参数 ---")
    for trainer in trainer_group:
        trainer.set_sampling_filters.remote(REPLAY_RECENT_FRAC, REPLAY_MAX_VERSION_GAP)
    print(f"采样过滤参数已设置: recent_frac={REPLAY_RECENT_FRAC}, max_version_gap={REPLAY_MAX_VERSION_GAP}")

    print("\n--- 步骤 2: 建立独立的 DeepSpeed 训练组 ---")
    print("正在查找空闲端口...")
    
    # 使用进程ID和时间戳来确保每个任务使用不同的端口范围，避免多任务并行时的端口冲突
    # 每个任务使用不同的 base_port，基于进程ID和任务标识符
    import time as _time
    _pid = os.getpid()
    _task_id = hash(f"{TASK_NAME}_{CLIP_MODE}_{SEED}") % 10000  # 基于任务参数生成唯一ID
    _port_offset = (_pid % 100) * 10 + (_task_id % 10)  # 确保不同任务使用不同端口范围
    
    # 训练组端口：29500 + offset（每个任务偏移不同）
    train_base_port = 29500 + _port_offset
    print(f"[端口选择] PID={_pid}, TaskID={_task_id}, PortOffset={_port_offset}, TrainBasePort={train_base_port}")
    
    # com_utils.find_free_port 无 base_port 参数，这里直接获取系统分配的可用端口
    train_group_port = find_free_port()
    print(f"训练组端口: {train_group_port}")

    # 广播组端口：使用不同的偏移，确保与训练组端口不同
    # 使用 30000 + offset 作为广播组的基础端口
    broadcast_base_port = 30000 + _port_offset
    print(f"[端口选择] BroadcastBasePort={broadcast_base_port}")
    
    broadcast_group_port = find_free_port()
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

    print("获取训练器主节点地址...")
    trainer_master_addr = ray.get(trainer_group[0].get_node_ip.remote())
    print(f"训练器主节点地址: {trainer_master_addr}")

    print("开始初始化 DeepSpeed 训练组...")
    train_setup_tasks = [actor.setup_deepspeed_group.remote(trainer_master_addr, train_group_port) for actor in trainer_group]
    print(f"创建了 {len(train_setup_tasks)} 个训练器设置任务")

    # 增加超时时间到 10 分钟，因为 DeepSpeed 初始化（特别是 NCCL）可能需要较长时间
    # 在批量并行运行时，资源竞争可能导致初始化更慢
    deepspeed_init_timeout = int(os.environ.get("DEEPSPEED_INIT_TIMEOUT", "600"))  # 默认 10 分钟
    print(f"DeepSpeed 初始化超时设置: {deepspeed_init_timeout} 秒")
    
    try:
        print("等待 DeepSpeed 初始化完成...")
        ray.get(train_setup_tasks, timeout=deepspeed_init_timeout)
        print("DeepSpeed 训练组建立完成。")

        # 获取并记录参数量
        n_total_params, n_trainable_params = ray.get(trainer_group[0].get_parameter_counts.remote())
        print(f"模型总参数量: {n_total_params:,}, 可训练参数量: {n_trainable_params:,}")
        if swanlab_available and swanlab is not None:
            try:
                swanlab.config.update({
                    "total_params": n_total_params,
                    "trainable_params": n_trainable_params
                })
            except Exception as e:
                debug_log(f"SwanLab config update failed: {e}")
                swanlab_available = False

    except ray.exceptions.GetTimeoutError as e:
        print(f"❌ DeepSpeed 初始化超时: {e}")
        print(f"   超时时间: {deepspeed_init_timeout} 秒")
        print(f"   可能原因:")
        print(f"   1. NCCL 初始化卡住（多任务并行时资源竞争）")
        print(f"   2. 端口冲突（多个任务使用相同端口）")
        print(f"   3. GPU 资源不足或显存不足")
        print(f"   4. 网络通信问题")
        print(f"   建议:")
        print(f"   - 增加超时时间: export DEEPSPEED_INIT_TIMEOUT=900  # 15 分钟")
        print(f"   - 减少并行任务数，避免资源竞争")
        print(f"   - 检查端口是否冲突")
        print(f"   正在检查 TrainerActor 状态...")
        # 尝试获取部分结果，看看是否有 TrainerActor 已经完成
        for i, task in enumerate(train_setup_tasks):
            try:
                result = ray.get([task], timeout=1)
                print(f"   Trainer {i} 已完成初始化")
            except Exception:
                print(f"   Trainer {i} 仍在初始化中或已失败")
        import traceback
        traceback.print_exc()
        raise
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
    debug_log("广播签名校验通过，准备推送初始权重到推理器。")

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
    for idx, w in enumerate(rollout_workers):
        debug_log(f"启动 RolloutWorker {idx}")
        w.run.remote()
    for idx, w in enumerate(eval_workers):
        debug_log(f"启动 EvaluationWorker {idx}")
        w.run.remote()

    print("\n--- 步骤 5: 等待远程经验池填充初始数据 ---")
    min_buffer_size_for_start = TRAIN_BATCH_SIZE * ACCUMULATION_STEPS
    assert min_buffer_size_for_start < REPLAY_CAPACITY, "初始填充量必须小于回放池总容量"
    while not all(size >= min_buffer_size_for_start for size in ray.get([rb.size.remote() for rb in replay_buffers])):
        sizes = ray.get([rb.size.remote() for rb in replay_buffers])
        print(f"等待所有经验池填充初始数据 (目标: {min_buffer_size_for_start})... (当前大小: {sizes})")
        time.sleep(5)
        # 额外调试：周期性打印经验池大小，便于确认是否在正常采样
        try:
            debug_log(f"[调试] 当前经验池大小: {sizes}")
        except Exception:
            pass
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
        
        # ========== 导出诊断数据（用于离线绘制机制图）==========
        if global_step > 0 and global_step % DUMP_INTERVAL == 0:
            # 检查是否有待导出的数据
            perf_metrics_from_rank0 = results[0][-1]  # 获取 rank 0 的 perf_metrics
            dump_data = perf_metrics_from_rank0.get('dump_sample_data', None)
            if dump_data is not None:
                ray.get(trainer_group[0].dump_diagnostic_data.remote(
                    obs_sample=dump_data['obs'],
                    act_token_sample=dump_data['act_token'],
                    logits_old_sample=dump_data['logits_old'],
                    policy_version_sample=dump_data['policy_version'],
                    insert_step_sample=dump_data['insert_step'],
                    dump_dir=DUMP_DIR,
                    step=global_step
                ))

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
            grad_norm_mean = perf_metrics_list[0].get("grad_norm_mean", None)
            if grad_norm_mean is not None:
                log_metrics['Metrics/Grad_Norm'] = grad_norm_mean
            ev_mean = perf_metrics_list[0].get("explained_variance_mean", None)
            if ev_mean is not None:
                log_metrics['Metrics/ExplainedVariance'] = ev_mean
            diag_outside = perf_metrics_list[0].get("diag_outside_clip_ratio", None)
            diag_dead = perf_metrics_list[0].get("diag_dead_grad_ratio", None)
            diag_ess = perf_metrics_list[0].get("diag_ess_norm", None)
            diag_every = perf_metrics_list[0].get("diag_every_steps", DIAG_EVERY_STEPS)
            if diag_outside is not None:
                log_metrics['Diag/Outside_Clip_Ratio'] = diag_outside
            if diag_dead is not None:
                log_metrics['Diag/Dead_Grad_Ratio'] = diag_dead
            if diag_ess is not None:
                log_metrics['Diag/ESS_Norm'] = diag_ess
            if diag_every is not None:
                log_metrics['Diag/Diag_Every_Steps'] = diag_every
            
            # ========== 新增：诊断指标（Staleness、Ratio、ESS 等）==========
            # 3.1 Staleness
            if 'staleness_ver_mean' in perf_metrics_list[0]:
                log_metrics['Staleness/Version_Mean'] = perf_metrics_list[0]['staleness_ver_mean']
                log_metrics['Staleness/Version_P95'] = perf_metrics_list[0]['staleness_ver_p95']
                log_metrics['Staleness/Age_Steps_Mean'] = perf_metrics_list[0]['age_steps_mean']
                log_metrics['Staleness/Age_Steps_P95'] = perf_metrics_list[0]['age_steps_p95']
                if 'age_steps_max' in perf_metrics_list[0]:
                    log_metrics['Staleness/Age_Steps_Max'] = perf_metrics_list[0]['age_steps_max']
                
                # A. 分桶组成（绝对阈值）
                if 'staleness_old_frac_abs' in perf_metrics_list[0]:
                    log_metrics['Staleness/OldFrac_Abs'] = perf_metrics_list[0]['staleness_old_frac_abs']
                    log_metrics['Staleness/NewFrac_Abs'] = perf_metrics_list[0]['staleness_new_frac_abs']
                if 'staleness_old_gap_mean_abs' in perf_metrics_list[0]:
                    log_metrics['Staleness/OldGapMean_Abs'] = perf_metrics_list[0]['staleness_old_gap_mean_abs']
                    log_metrics['Staleness/OldGapP95_Abs'] = perf_metrics_list[0]['staleness_old_gap_p95_abs']
                
                # B. 相对陈旧度
                if 'staleness_ratio_mean' in perf_metrics_list[0]:
                    log_metrics['Staleness/RatioMean'] = perf_metrics_list[0]['staleness_ratio_mean']
                    log_metrics['Staleness/RatioP95'] = perf_metrics_list[0]['staleness_ratio_p95']
                if 'staleness_old_frac_ratio' in perf_metrics_list[0]:
                    log_metrics['Staleness/OldFrac_Ratio'] = perf_metrics_list[0]['staleness_old_frac_ratio']
                    log_metrics['Staleness/NewFrac_Ratio'] = perf_metrics_list[0]['staleness_new_frac_ratio']
            
            # 3.2 Ratio / log-ratio 分布
            if 'rho_mean' in perf_metrics_list[0]:
                log_metrics['Ratio/Rho_Mean'] = perf_metrics_list[0]['rho_mean']
                log_metrics['Ratio/Rho_P50'] = perf_metrics_list[0]['rho_p50']
                log_metrics['Ratio/Rho_P90'] = perf_metrics_list[0]['rho_p90']
                log_metrics['Ratio/Rho_P99'] = perf_metrics_list[0]['rho_p99']
                log_metrics['Ratio/Rho_Max'] = perf_metrics_list[0]['rho_max']
                log_metrics['Ratio/LogRho_Mean'] = perf_metrics_list[0]['logrho_mean']
                log_metrics['Ratio/AbsLogRho_P95'] = perf_metrics_list[0]['abs_logrho_p95']
            
            # 3.3 Clip fraction（PPO，向后兼容）
            if 'clip_frac' in perf_metrics_list[0]:
                log_metrics['Clip/Frac_Total'] = perf_metrics_list[0]['clip_frac']
                if 'clip_frac_new' in perf_metrics_list[0]:
                    log_metrics['Clip/Frac_New'] = perf_metrics_list[0]['clip_frac_new']
                if 'clip_frac_old' in perf_metrics_list[0]:
                    log_metrics['Clip/Frac_Old'] = perf_metrics_list[0]['clip_frac_old']
            
            # ========== 新增核心指标 1: PG Active/Dead Frac (hard clip) 或 Suppressed Frac (soft clip) ==========
            if 'pg_active_frac' in perf_metrics_list[0]:
                # Hard clip 模式
                log_metrics['PG/Active_Frac'] = perf_metrics_list[0]['pg_active_frac']
                log_metrics['PG/Dead_Frac'] = perf_metrics_list[0]['pg_dead_frac']
                if 'pg_active_frac_new' in perf_metrics_list[0]:
                    log_metrics['PG/Active_Frac_New'] = perf_metrics_list[0]['pg_active_frac_new']
                    log_metrics['PG/Dead_Frac_New'] = perf_metrics_list[0]['pg_dead_frac_new']
                if 'pg_active_frac_old' in perf_metrics_list[0]:
                    log_metrics['PG/Active_Frac_Old'] = perf_metrics_list[0]['pg_active_frac_old']
                    log_metrics['PG/Dead_Frac_Old'] = perf_metrics_list[0]['pg_dead_frac_old']
                # 相对阈值分桶
                if 'pg_active_frac_new_ratio' in perf_metrics_list[0]:
                    log_metrics['PG/Active_Frac_New_Ratio'] = perf_metrics_list[0]['pg_active_frac_new_ratio']
                    log_metrics['PG/Dead_Frac_New_Ratio'] = perf_metrics_list[0]['pg_dead_frac_new_ratio']
                if 'pg_active_frac_old_ratio' in perf_metrics_list[0]:
                    log_metrics['PG/Active_Frac_Old_Ratio'] = perf_metrics_list[0]['pg_active_frac_old_ratio']
                    log_metrics['PG/Dead_Frac_Old_Ratio'] = perf_metrics_list[0]['pg_dead_frac_old_ratio']
            
            # ✅ Soft clip 模式：区分两种指标
            if 'outside_clip_frac' in perf_metrics_list[0]:
                # 按 ratio 定义（与 PPO 可比）
                log_metrics['Soft/Outside_Clip_Frac'] = perf_metrics_list[0]['outside_clip_frac']
                if 'outside_clip_frac_new' in perf_metrics_list[0]:
                    log_metrics['Soft/Outside_Clip_Frac_New'] = perf_metrics_list[0]['outside_clip_frac_new']
                if 'outside_clip_frac_old' in perf_metrics_list[0]:
                    log_metrics['Soft/Outside_Clip_Frac_Old'] = perf_metrics_list[0]['outside_clip_frac_old']
            
            if 'suppressed_frac' in perf_metrics_list[0]:
                # 按权重阈值定义（更贴近 soft 机制）
                log_metrics['Soft/Suppressed_Frac'] = perf_metrics_list[0]['suppressed_frac']
                if 'suppressed_frac_new' in perf_metrics_list[0]:
                    log_metrics['Soft/Suppressed_Frac_New'] = perf_metrics_list[0]['suppressed_frac_new']
                if 'suppressed_frac_old' in perf_metrics_list[0]:
                    log_metrics['Soft/Suppressed_Frac_Old'] = perf_metrics_list[0]['suppressed_frac_old']
            
            # ========== 新增核心指标 2: 贡献权重 U ==========
            if 'u_mean' in perf_metrics_list[0]:
                log_metrics['Contribution/U_Mean'] = perf_metrics_list[0]['u_mean']
                log_metrics['Contribution/U_P50'] = perf_metrics_list[0]['u_p50']
                log_metrics['Contribution/U_P90'] = perf_metrics_list[0]['u_p90']
                log_metrics['Contribution/U_P99'] = perf_metrics_list[0]['u_p99']
                log_metrics['Contribution/U_Max'] = perf_metrics_list[0]['u_max']
                
                # 分桶统计（绝对阈值）
                if 'u_mean_new' in perf_metrics_list[0]:
                    log_metrics['Contribution/U_Mean_New'] = perf_metrics_list[0]['u_mean_new']
                    log_metrics['Contribution/U_P90_New'] = perf_metrics_list[0]['u_p90_new']
                if 'u_mean_old' in perf_metrics_list[0]:
                    log_metrics['Contribution/U_Mean_Old'] = perf_metrics_list[0]['u_mean_old']
                    log_metrics['Contribution/U_P90_Old'] = perf_metrics_list[0]['u_p90_old']
            
            # C. 旧数据贡献占比（核心机制证据）
            if 'contribution_old_u_share' in perf_metrics_list[0]:
                log_metrics['Contribution/OldUShare'] = perf_metrics_list[0]['contribution_old_u_share']
                log_metrics['Contribution/NewUShare'] = perf_metrics_list[0]['contribution_new_u_share']
            if 'contribution_old_u_share_ratio' in perf_metrics_list[0]:
                log_metrics['Contribution/OldUShare_Ratio'] = perf_metrics_list[0]['contribution_old_u_share_ratio']
                log_metrics['Contribution/NewUShare_Ratio'] = perf_metrics_list[0]['contribution_new_u_share_ratio']
            
            # C3. 基于 |u*A| 的梯度贡献占比（更直接反映对更新的实际贡献）
            if 'contribution_old_u_share_abs_grad_proxy' in perf_metrics_list[0]:
                log_metrics['Contribution/OldUShare_AbsGradProxy'] = perf_metrics_list[0]['contribution_old_u_share_abs_grad_proxy']
                log_metrics['Contribution/NewUShare_AbsGradProxy'] = perf_metrics_list[0]['contribution_new_u_share_abs_grad_proxy']
            if 'contribution_old_u_share_abs_grad_proxy_ratio' in perf_metrics_list[0]:
                log_metrics['Contribution/OldUShare_AbsGradProxy_Ratio'] = perf_metrics_list[0]['contribution_old_u_share_abs_grad_proxy_ratio']
                log_metrics['Contribution/NewUShare_AbsGradProxy_Ratio'] = perf_metrics_list[0]['contribution_new_u_share_abs_grad_proxy_ratio']
            
            # ========== 新增核心指标 3: ESS_Eff（基于有效贡献）==========
            if 'ess_eff' in perf_metrics_list[0]:
                log_metrics['ESS/ESS_Eff'] = perf_metrics_list[0]['ess_eff']
                log_metrics['ESS/ESS_Eff_Norm'] = perf_metrics_list[0]['ess_eff_norm']
                
                # 分桶统计（绝对阈值）- 关键证据
                if 'ess_eff_norm_new' in perf_metrics_list[0]:
                    log_metrics['ESS/ESS_Eff_Norm_New'] = perf_metrics_list[0]['ess_eff_norm_new']
                if 'ess_eff_norm_old' in perf_metrics_list[0]:
                    log_metrics['ESS/ESS_Eff_Norm_Old'] = perf_metrics_list[0]['ess_eff_norm_old']
                if 'ess_eff_norm_new_abs' in perf_metrics_list[0]:
                    log_metrics['ESS/ESS_Eff_Norm_New_Abs'] = perf_metrics_list[0]['ess_eff_norm_new_abs']
                if 'ess_eff_norm_old_abs' in perf_metrics_list[0]:
                    log_metrics['ESS/ESS_Eff_Norm_Old_Abs'] = perf_metrics_list[0]['ess_eff_norm_old_abs']
                
                # 分桶统计（相对阈值）
                if 'ess_eff_norm_new_ratio' in perf_metrics_list[0]:
                    log_metrics['ESS/ESS_Eff_Norm_New_Ratio'] = perf_metrics_list[0]['ess_eff_norm_new_ratio']
                if 'ess_eff_norm_old_ratio' in perf_metrics_list[0]:
                    log_metrics['ESS/ESS_Eff_Norm_Old_Ratio'] = perf_metrics_list[0]['ess_eff_norm_old_ratio']
            
            # ========== 新增核心指标 4: NearZero_U_Frac ==========
            if 'nearzero_u_frac' in perf_metrics_list[0]:
                log_metrics['Contribution/NearZero_U_Frac'] = perf_metrics_list[0]['nearzero_u_frac']
                # D1) 分桶统计（绝对阈值）- 核心机制证据
                if 'nearzero_u_frac_new' in perf_metrics_list[0]:
                    log_metrics['Contribution/NearZero_U_Frac_New'] = perf_metrics_list[0]['nearzero_u_frac_new']
                if 'nearzero_u_frac_old' in perf_metrics_list[0]:
                    log_metrics['Contribution/NearZero_U_Frac_Old'] = perf_metrics_list[0]['nearzero_u_frac_old']
                # 分桶统计（相对阈值）
                if 'nearzero_u_frac_new_ratio' in perf_metrics_list[0]:
                    log_metrics['Contribution/NearZero_U_Frac_New_Ratio'] = perf_metrics_list[0]['nearzero_u_frac_new_ratio']
                if 'nearzero_u_frac_old_ratio' in perf_metrics_list[0]:
                    log_metrics['Contribution/NearZero_U_Frac_Old_Ratio'] = perf_metrics_list[0]['nearzero_u_frac_old_ratio']
            
            # ========== 新增核心指标 5: WeightShare（数据贡献占比）==========
            if 'weight_share_old' in perf_metrics_list[0]:
                log_metrics['Contribution/Weight_Share_Old'] = perf_metrics_list[0]['weight_share_old']
            if 'weight_share_new' in perf_metrics_list[0]:
                log_metrics['Contribution/Weight_Share_New'] = perf_metrics_list[0]['weight_share_new']
            
            # 3.5 ESS (Effective Sample Size，向后兼容)
            if 'ess' in perf_metrics_list[0]:
                log_metrics['ESS/ESS'] = perf_metrics_list[0]['ess']
                log_metrics['ESS/ESS_Norm'] = perf_metrics_list[0]['ess_norm']

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

            # TensorBoard记录（如果启用）
            if writer is not None:
                for k, v in log_metrics.items():
                    writer.add_scalar(k, v, global_step)
            
            # SwanLab记录（如果启用）
            if swanlab_available and swanlab is not None:
                try:
                    swanlab.log(log_metrics, step=global_step)
                except Exception as e:
                    debug_log(f"SwanLab日志记录失败: {e}，后续将禁用SwanLab")
                    swanlab_available = False
            
            # SwanLab图像记录（如果启用）
            diag_img = perf_metrics_list[0].get("diag_old_new_prob_image", None)
            if diag_img is not None and swanlab_available and swanlab is not None:
                diag_corr = perf_metrics_list[0].get("diag_old_new_prob_corr", float("nan"))
                try:
                    swanlab.log({
                        "Diag/Old_vs_New_Prob_Scatter": swanlab.Image(
                            diag_img,
                            caption=f"step={global_step}, corr={diag_corr:.4f}"
                        )
                    }, step=global_step)
                except Exception as e:
                    debug_log(f"SwanLab图像记录失败: {e}")
                    # 不设置swanlab_available=False，因为图像记录失败不影响后续标量记录

            last_log_time = current_time
            last_log_global_step = global_step

    print(f"\n成功完成 {TRAIN_ITERS} 次训练与同步循环！")
    if writer is not None:
        writer.close()
    if swanlab_available and swanlab is not None:
        try:
            swanlab.finish()
        except Exception as e:
            debug_log(f"SwanLab finish失败: {e}")
    ray.shutdown()


if __name__ == "__main__":
    main()