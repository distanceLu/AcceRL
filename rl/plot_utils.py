"""
绘图工具函数

包含用于诊断和可视化的画图函数。
"""
import io
from typing import Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def make_policy_prob_diag_image(
    p_old: torch.Tensor,
    p_new: torch.Tensor,
    max_points: int = 20000,
    title: str = "Old vs New Policy Probability",
) -> Tuple[np.ndarray | None, float]:
    """
    绘制旧/新策略在相同动作上的概率散点图。
    
    Args:
        p_old: 旧策略概率，torch.Tensor，任意shape
        p_new: 新策略概率，torch.Tensor，任意shape
        max_points: 最大绘制点数（避免图像过大）
        title: 图表标题
    
    Returns:
        img_uint8: RGB uint8 图像数组 (H, W, 3)
        corr: 旧策略和新策略概率的相关系数
        
    Note:
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
) -> Tuple[np.ndarray | None, float]:
    """
    绘制旧/新动作概率散点图。
    
    Args:
        p_old: 旧策略概率，torch.Tensor，任意shape
        p_new: 新策略概率，torch.Tensor，任意shape
        max_points: 最大绘制点数（避免图像过大）
        title: 图表标题
    
    Returns:
        img_uint8: RGB uint8 图像数组 (H, W, 3)，如果输入为空则返回 None
        corr: 旧策略和新策略概率的相关系数
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
# 机制图绘制函数（用于离线分析导出的诊断数据）
# ================================================================

def plot_mechanism_diagrams(npz_path: str, save_dir: str = None, clip_eps: float = 0.2):
    """
    从导出的 npz 文件加载数据，绘制机制分析图
    
    Args:
        npz_path: 导出的 .npz 文件路径
        save_dir: 图片保存目录（如果为 None，则显示图片）
        clip_eps: PPO clip 阈值
    
    绘制的图包括：
        1. log(μ) vs log(π) scatter plot（带 PPO clip band）
        2. log(ρ) 直方图（对比 clipped vs non-clipped）
        3. staleness vs ratio scatter plot
    """
    import os
    
    # 加载数据
    data = np.load(npz_path)
    logp_old = data['logp_old']  # [N, action_dim]
    logp_new = data['logp_new']  # [N, action_dim]
    ratio = data['ratio']  # [N, action_dim]
    logrho = data['logrho']  # [N, action_dim]
    staleness_ver = data['staleness_ver']  # [N]
    age_steps = data['age_steps']  # [N]
    current_step = int(data['current_step'])
    
    # Flatten 所有数据
    logp_old_flat = logp_old.reshape(-1)
    logp_new_flat = logp_new.reshape(-1)
    ratio_flat = ratio.reshape(-1)
    logrho_flat = logrho.reshape(-1)
    
    # 为每个样本复制 staleness（因为一个样本有多个 action dim）
    action_dim = logp_old.shape[1] if logp_old.ndim > 1 else 1
    staleness_flat = np.repeat(staleness_ver, action_dim)
    age_flat = np.repeat(age_steps, action_dim)
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # ========== 图1: log(μ) vs log(π) scatter（带 PPO clip band）==========
    ax = axes[0, 0]
    
    # 采样点（避免绘制太多点）
    max_points = 20000
    if len(logp_old_flat) > max_points:
        idx = np.random.choice(len(logp_old_flat), max_points, replace=False)
        x_plot = logp_old_flat[idx]
        y_plot = logp_new_flat[idx]
        staleness_plot = staleness_flat[idx]
    else:
        x_plot = logp_old_flat
        y_plot = logp_new_flat
        staleness_plot = staleness_flat
    
    # 按 staleness 着色
    sc = ax.scatter(x_plot, y_plot, c=staleness_plot, s=5, alpha=0.6, cmap='viridis')
    
    # 绘制 y=x 线
    lim_min = min(x_plot.min(), y_plot.min())
    lim_max = max(x_plot.max(), y_plot.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=2, label='y=x')
    
    # 绘制 PPO clip band
    x_line = np.linspace(lim_min, lim_max, 100)
    ax.plot(x_line, x_line + np.log(1 + clip_eps), 'b--', alpha=0.5, linewidth=1.5, label=f'ρ = 1+ε ({clip_eps})')
    ax.plot(x_line, x_line + np.log(1 - clip_eps), 'b--', alpha=0.5, linewidth=1.5, label=f'ρ = 1-ε')
    
    ax.set_xlabel('log μ(a|s) (Old Policy)', fontsize=12)
    ax.set_ylabel('log π(a|s) (Current Policy)', fontsize=12)
    ax.set_title(f'Log-Prob Scatter (Step {current_step})', fontsize=14)
    ax.legend(loc='upper left')
    fig.colorbar(sc, ax=ax, label='Staleness (version gap)')
    
    # ========== 图2: log(ρ) 直方图（分 clipped vs non-clipped）==========
    ax = axes[0, 1]
    
    # 计算 clip mask
    clip_mask = (ratio_flat < (1 - clip_eps)) | (ratio_flat > (1 + clip_eps))
    logrho_clipped = logrho_flat[clip_mask]
    logrho_nonclipped = logrho_flat[~clip_mask]
    
    # 绘制直方图
    bins = np.linspace(logrho_flat.min(), logrho_flat.max(), 50)
    ax.hist(logrho_nonclipped, bins=bins, alpha=0.6, label=f'Non-clipped ({(~clip_mask).sum()})', color='green')
    ax.hist(logrho_clipped, bins=bins, alpha=0.6, label=f'Clipped ({clip_mask.sum()})', color='red')
    
    # 标注 clip 边界
    ax.axvline(np.log(1 + clip_eps), color='blue', linestyle='--', linewidth=2, alpha=0.7, label=f'log(1+ε)')
    ax.axvline(np.log(1 - clip_eps), color='blue', linestyle='--', linewidth=2, alpha=0.7, label=f'log(1-ε)')
    
    ax.set_xlabel('log ρ = log(π/μ)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Log-Ratio Distribution (Clip Frac: {clip_mask.mean():.2%})', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # ========== 图3: staleness vs ratio scatter ==========
    ax = axes[1, 0]
    
    # 采样点
    if len(staleness_flat) > max_points:
        idx = np.random.choice(len(staleness_flat), max_points, replace=False)
        staleness_plot = staleness_flat[idx]
        ratio_plot = ratio_flat[idx]
        clip_mask_plot = clip_mask[idx]
    else:
        staleness_plot = staleness_flat
        ratio_plot = ratio_flat
        clip_mask_plot = clip_mask
    
    # 分别绘制 clipped 和 non-clipped 点
    ax.scatter(staleness_plot[~clip_mask_plot], ratio_plot[~clip_mask_plot], 
               c='green', s=5, alpha=0.5, label='Non-clipped')
    ax.scatter(staleness_plot[clip_mask_plot], ratio_plot[clip_mask_plot], 
               c='red', s=5, alpha=0.5, label='Clipped')
    
    # 标注 clip 边界
    ax.axhline(1 + clip_eps, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    ax.axhline(1 - clip_eps, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    ax.axhline(1.0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Staleness (version gap)', fontsize=12)
    ax.set_ylabel('Importance Ratio ρ', fontsize=12)
    ax.set_title('Staleness vs Importance Ratio', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # ========== 图4: staleness vs age scatter ==========
    ax = axes[1, 1]
    
    # 每个样本只取一个点（因为 staleness 和 age 对每个样本是一样的）
    staleness_sample = staleness_ver
    age_sample = age_steps
    
    if len(staleness_sample) > max_points:
        idx = np.random.choice(len(staleness_sample), max_points, replace=False)
        staleness_sample = staleness_sample[idx]
        age_sample = age_sample[idx]
    
    ax.scatter(age_sample, staleness_sample, s=10, alpha=0.6, color='purple')
    ax.set_xlabel('Age (steps)', fontsize=12)
    ax.set_ylabel('Staleness (version gap)', fontsize=12)
    ax.set_title('Age vs Staleness', fontsize=14)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # 保存或显示
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'mechanism_step_{current_step}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"机制图已保存到: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def batch_plot_mechanism_diagrams(dump_dir: str, save_dir: str = None, clip_eps: float = 0.2):
    """
    批量绘制所有导出的诊断数据
    
    Args:
        dump_dir: 导出数据的目录
        save_dir: 图片保存目录（如果为 None，则保存到 dump_dir/plots）
        clip_eps: PPO clip 阈值
    """
    import os
    import glob
    
    if save_dir is None:
        save_dir = os.path.join(dump_dir, 'plots')
    
    # 查找所有 .npz 文件
    npz_files = sorted(glob.glob(os.path.join(dump_dir, 'diagnostic_step_*.npz')))
    
    if not npz_files:
        print(f"未找到诊断数据文件（在 {dump_dir}）")
        return
    
    print(f"找到 {len(npz_files)} 个诊断数据文件")
    
    for npz_path in npz_files:
        print(f"正在处理: {npz_path}")
        try:
            plot_mechanism_diagrams(npz_path, save_dir=save_dir, clip_eps=clip_eps)
        except Exception as e:
            print(f"绘制失败: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n所有机制图已保存到: {save_dir}")

