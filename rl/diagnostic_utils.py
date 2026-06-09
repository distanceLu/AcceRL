"""
诊断和可视化工具模块
包含训练过程中的诊断指标计算和可视化功能
"""

import torch
import numpy as np
from typing import Tuple, Dict, Optional, Any
import time


def debug_log(msg: str):
    """统一调试日志格式，带时间戳并立即刷新。"""
    print(f"[DEBUG][{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def compute_diagnostic_metrics(
    ratio: torch.Tensor,
    adv_unsqueezed: torch.Tensor,
    clip_eps: float
) -> Tuple[float, float, float]:
    """
    计算训练诊断指标
    
    Args:
        ratio: 策略比率 π_new/π_old, shape: [B, ACTION_DIM]
        adv_unsqueezed: 优势值, shape: [B, 1]
        clip_eps: PPO裁剪参数 ε
        
    Returns:
        outside_clip_ratio: 超出裁剪范围的样本比例
        dead_grad_ratio: 梯度为0的样本比例（被裁剪且会导致梯度消失的样本）
        ess_norm: 归一化的有效样本量（Effective Sample Size）
    """
    with torch.no_grad():
        # 扩展优势值以匹配ratio的形状
        adv_expand = adv_unsqueezed.expand_as(ratio)
        
        # 1. 计算超出裁剪范围的样本比例
        outside = ((ratio < 1 - clip_eps) | (ratio > 1 + clip_eps)).float().mean()
        
        # 2. 计算"死梯度"比例
        # 当 ratio > 1+ε 且 A>0 时，min会选择裁剪项，梯度为0
        # 当 ratio < 1-ε 且 A<0 时，min会选择裁剪项，梯度为0
        dead_grad = (
            ((ratio > 1 + clip_eps) & (adv_expand > 0)) | 
            ((ratio < 1 - clip_eps) & (adv_expand < 0))
        ).float().mean()
        
        # 3. 计算归一化的有效样本量（ESS）
        # ESS = (Σw)² / Σw²，这里w是裁剪后的权重
        w = ratio.clamp(1 - clip_eps, 1 + clip_eps)
        ess = (w.sum() ** 2) / (w.pow(2).sum() + 1e-8)
        ess_norm = (ess / w.numel()).item()
        
        outside_ratio = outside.item()
        dead_grad_ratio = dead_grad.item()
        
    return outside_ratio, dead_grad_ratio, ess_norm


def should_compute_diagnostics(
    global_step: int,
    policy_train_start_step: int,
    diag_every_steps: int,
    rank: int = 0
) -> bool:
    """
    判断当前步是否应该计算详细的诊断信息
    
    Args:
        global_step: 当前全局训练步数
        policy_train_start_step: 策略开始训练的步数
        diag_every_steps: 诊断频率
        rank: 当前进程的rank（只在rank=0时计算）
        
    Returns:
        是否应该计算诊断信息
    """
    return (
        rank == 0 and 
        global_step >= policy_train_start_step and 
        global_step % diag_every_steps == 0
    )


def prepare_diagnostic_data(
    ratio: torch.Tensor,
    adv_unsqueezed: torch.Tensor,
    logp_old: torch.Tensor,
    logp_new: torch.Tensor,
    clip_eps: float,
    compute_image: bool = False,
    max_points: int = 20000
) -> Dict[str, Any]:
    """
    准备完整的诊断数据，包括指标和可视化
    
    Args:
        ratio: 策略比率
        adv_unsqueezed: 优势值
        logp_old: 旧策略的log概率
        logp_new: 新策略的log概率
        clip_eps: 裁剪参数
        compute_image: 是否计算可视化图像
        max_points: 图像中最多显示的点数
        
    Returns:
        包含所有诊断信息的字典
    """
    # 计算基础指标
    outside_ratio, dead_grad_ratio, ess_norm = compute_diagnostic_metrics(
        ratio, adv_unsqueezed, clip_eps
    )
    
    diag_data = {
        "outside_clip_ratio": outside_ratio,
        "dead_grad_ratio": dead_grad_ratio,
        "ess_norm": ess_norm
    }
    
    # 如果需要，计算可视化图像
    if compute_image:
        try:
            from rl.plot_utils import make_policy_prob_diag_image
            
            with torch.no_grad():
                p_old = torch.exp(logp_old)
                p_new = torch.exp(logp_new)
                diag_image, diag_image_corr = make_policy_prob_diag_image(
                    p_old, p_new, max_points=max_points
                )
                diag_data["diag_image"] = diag_image
                diag_data["diag_corr"] = diag_image_corr
        except Exception as e:
            print(f"[Warn] 生成诊断图像失败: {e}")
            diag_data["diag_image"] = None
            diag_data["diag_corr"] = None
    
    return diag_data


def log_diagnostics_to_tensorboard(
    writer,
    diag_data: Dict[str, Any],
    global_step: int,
    prefix: str = "Diag"
):
    """
    将诊断数据记录到TensorBoard
    
    Args:
        writer: TensorBoard SummaryWriter对象
        diag_data: 诊断数据字典
        global_step: 当前全局步数
        prefix: 日志标签前缀
    """
    if diag_data.get("outside_clip_ratio") is not None:
        writer.add_scalar(
            f'{prefix}/Outside_Clip_Ratio', 
            diag_data["outside_clip_ratio"], 
            global_step
        )
    
    if diag_data.get("dead_grad_ratio") is not None:
        writer.add_scalar(
            f'{prefix}/Dead_Grad_Ratio', 
            diag_data["dead_grad_ratio"], 
            global_step
        )
    
    if diag_data.get("ess_norm") is not None:
        writer.add_scalar(
            f'{prefix}/ESS_Norm', 
            diag_data["ess_norm"], 
            global_step
        )


def log_diagnostics_to_swanlab(
    swanlab,
    diag_data: Dict[str, Any],
    global_step: int,
    prefix: str = "Diag"
):
    """
    将诊断数据记录到SwanLab
    
    Args:
        swanlab: SwanLab模块
        diag_data: 诊断数据字典
        global_step: 当前全局步数
        prefix: 日志标签前缀
    """
    log_dict = {}
    
    # 记录数值指标
    if diag_data.get("outside_clip_ratio") is not None:
        log_dict[f'{prefix}/Outside_Clip_Ratio'] = diag_data["outside_clip_ratio"]
    
    if diag_data.get("dead_grad_ratio") is not None:
        log_dict[f'{prefix}/Dead_Grad_Ratio'] = diag_data["dead_grad_ratio"]
    
    if diag_data.get("ess_norm") is not None:
        log_dict[f'{prefix}/ESS_Norm'] = diag_data["ess_norm"]
    
    # 记录图像
    if diag_data.get("diag_image") is not None:
        try:
            diag_corr = diag_data.get("diag_corr", float("nan"))
            log_dict[f"{prefix}/Old_vs_New_Prob_Scatter"] = swanlab.Image(
                diag_data["diag_image"],
                caption=f"step={global_step}, corr={diag_corr:.4f}"
            )
        except Exception as e:
            print(f"[Warn] SwanLab 图像记录失败: {e}")
    
    if log_dict:
        swanlab.log(log_dict, step=global_step)


def aggregate_performance_metrics(
    perf_metrics_list: list,
    include_diagnostics: bool = True
) -> Dict[str, float]:
    """
    聚合多个trainer的性能指标
    
    Args:
        perf_metrics_list: 每个trainer返回的性能指标列表
        include_diagnostics: 是否包含诊断指标
        
    Returns:
        聚合后的指标字典
    """
    aggregated = {}
    
    # 聚合时间相关指标
    aggregated['policy_sample_time'] = np.mean([
        pm.get("policy_sample_time", 0) for pm in perf_metrics_list
    ])
    aggregated['policy_prep_time'] = np.mean([
        pm.get("policy_prep_time", 0) for pm in perf_metrics_list
    ])
    aggregated['policy_train_time'] = np.mean([
        pm.get("policy_train_time", 0) for pm in perf_metrics_list
    ])
    
    # 聚合诊断指标（通常只从rank 0获取）
    if include_diagnostics and len(perf_metrics_list) > 0:
        first_metrics = perf_metrics_list[0]
        
        for key in ["diag_outside_clip_ratio", "diag_dead_grad_ratio", 
                    "diag_ess_norm", "diag_every_steps"]:
            if key in first_metrics and first_metrics[key] is not None:
                aggregated[key] = first_metrics[key]
        
        # 图像数据（如果存在）
        if "diag_old_new_prob_image" in first_metrics:
            aggregated["diag_old_new_prob_image"] = first_metrics["diag_old_new_prob_image"]
        if "diag_old_new_prob_corr" in first_metrics:
            aggregated["diag_old_new_prob_corr"] = first_metrics["diag_old_new_prob_corr"]
    
    return aggregated

