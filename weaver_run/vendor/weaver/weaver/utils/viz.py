"""Visualization and video I/O utilities for WEAVER evaluation scripts."""

from __future__ import annotations

import os

import cv2
import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Font
# ---------------------------------------------------------------------------

def load_font(size: int):
    """Return a PIL ImageFont at the requested size using Pillow's built-in default."""
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 9.2
    except TypeError:
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Frame manipulation
# ---------------------------------------------------------------------------

def resize_frames(frames: np.ndarray, h: int, w: int) -> np.ndarray:
    """Bilinear resize (T, H, W, C) uint8 → (T, h, w, C) uint8."""
    t = torch.from_numpy(frames).float().permute(0, 3, 1, 2)
    t = torch.nn.functional.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
    return t.permute(0, 2, 3, 1).clamp(0, 255).byte().numpy()


def resize_to(img: np.ndarray | None, hw: tuple[int, int]) -> np.ndarray:
    """Resize a single (H, W, C) image to hw=(h, w); return a black frame if img is None."""
    if img is None:
        return np.zeros((*hw, 3), dtype=np.uint8)
    if img.shape[:2] != hw:
        img = cv2.resize(img, (hw[1], hw[0]))
    return img


# ---------------------------------------------------------------------------
# Debug annotation
# ---------------------------------------------------------------------------

def stamp_reward_on_frames(
    frames: np.ndarray,
    reward: float,
    is_best: bool,
    per_step_rewards: np.ndarray | None = None,
    per_step_values:  np.ndarray | None = None,
) -> np.ndarray:
    """Overlay accumulated reward and (optionally) per-step reward/critic on each frame.

    Args:
        frames:           (T, H, W, 3) uint8 numpy array.
        reward:           Accumulated reward scalar for header label.
        is_best:          Highlight in green if True, yellow otherwise.
        per_step_rewards: (T,) per-step reward array.
        per_step_values:  (T,) per-step critic value array.

    Returns:
        (T, H, W, 3) uint8 numpy array with text overlaid.
    """
    label = f"{'[BEST] ' if is_best else ''}R={reward:.3f}"
    color = (0, 255, 0) if is_best else (255, 255, 0)
    font    = load_font(18)
    font_sm = load_font(15)
    out = []
    for i, frame in enumerate(frames):
        img  = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        draw.text((6, 6), label, fill=(0, 0, 0), font=font)
        draw.text((5, 5), label, fill=color,     font=font)
        if per_step_rewards is not None and per_step_values is not None:
            r = float(per_step_rewards[i]) if i < len(per_step_rewards) else 0.0
            v = float(per_step_values[i])  if i < len(per_step_values)  else 0.0
            step_lbl = f"r={r:.4f}  v={v:.4f}  t={i}"
            draw.text((6, 30), step_lbl, fill=(0, 0, 0),     font=font_sm)
            draw.text((5, 29), step_lbl, fill=(255, 200, 50), font=font_sm)
        out.append(np.array(img))
    return np.stack(out)


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------

def save_video_mp4(path: str, frames: np.ndarray, fps: int) -> None:
    """Write a (T, H, W, 3) uint8 array to an MP4 file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps, codec="libx264", format="FFMPEG", quality=8)


def save_trajectory(output_dir: str, global_idx: int,
                    video_frames: np.ndarray, annotation: dict, fps: int) -> str:
    """Save a synthetic trajectory: video to videos/<idx>.mp4 and annotation to annotations/<idx>.json.

    Returns the video path.
    """
    import json
    vid_dir = os.path.join(output_dir, "videos")
    ann_dir = os.path.join(output_dir, "annotations")
    os.makedirs(vid_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)
    vid_path = os.path.join(vid_dir, f"{global_idx}.mp4")
    save_video_mp4(vid_path, video_frames, fps)
    with open(os.path.join(ann_dir, f"{global_idx}.json"), "w") as f:
        json.dump(annotation, f, indent=4)
    return vid_path


# ---------------------------------------------------------------------------
# Reward curve
# ---------------------------------------------------------------------------

def render_reward_curve(
    reward_values: list,
    current_idx: int,
    frame_width: int,
    frame_height: int = 140,
) -> np.ndarray:
    """Render the reward curve as an (frame_height, frame_width, 3) uint8 image.

    Plots the full curve in blue with a red marker at current_idx.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dpi = 100
    fig, ax = plt.subplots(figsize=(frame_width / dpi, frame_height / dpi), dpi=dpi)
    ax.plot(range(len(reward_values)), reward_values, color="#2196F3", linewidth=1.5)
    ax.axvline(current_idx, color="#F44336", linewidth=1.5, linestyle="--")
    ax.scatter([current_idx], [reward_values[current_idx]], color="#F44336", s=30, zorder=5)
    y_min, y_max = min(reward_values), max(reward_values)
    pad = max((y_max - y_min) * 0.15, 0.05)
    ax.set(ylim=(y_min - pad, y_max + pad), xlim=(0, max(len(reward_values) - 1, 1)),
           xlabel="Frame", ylabel="Reward", title="Predicted Reward")
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(pad=0.4)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plt.close(fig)
    if img.shape[1] != frame_width or img.shape[0] != frame_height:
        img = cv2.resize(img, (frame_width, frame_height))
    return img


def plot_reward_comparison(
    gt_rewards:        np.ndarray | None,
    inferred_rewards:  list,
    pred_start_frame:  int,
    traj_id:           int,
    out_path:          str,
) -> None:
    """Save a figure comparing GT reward_progress and WM-inferred reward."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))
    if gt_rewards is not None:
        ax.plot(range(len(gt_rewards)), gt_rewards, color="#4CAF50", linewidth=2,
                label="GT reward_progress", alpha=0.85)
    if inferred_rewards:
        ax.plot(range(pred_start_frame, pred_start_frame + len(inferred_rewards)),
                inferred_rewards, color="#2196F3", linewidth=2,
                label="Inferred reward (WM)", alpha=0.85)
    ax.axvline(pred_start_frame, color="#F44336", linestyle="--", linewidth=1.2,
               alpha=0.6, label=f"Pred start (frame {pred_start_frame})")
    ax.set(xlabel="Frame (video rate)", ylabel="Reward",
           title=f"Trajectory {traj_id}: GT reward_progress vs WM-inferred reward")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved reward comparison figure → {out_path}")


def save_comparison_video(
    gt_obs:       dict,
    decoded_pred: dict,
    img_keys:     list,
    out_path:     str,
    fps:          int = 5,
    reward_values: list | None = None,
) -> None:
    """Save a GT (top row) / predicted (bottom row) side-by-side video.

    If reward_values is provided, appends a reward curve row at the bottom of
    every frame.
    """
    from einops import rearrange

    T = min(gt_obs[img_keys[0]].shape[1], decoded_pred[img_keys[0]].shape[1])
    gt_row   = np.concatenate([
        rearrange(gt_obs[k][0, :T].float().cpu(), "t c h w -> t h w c").numpy()
        for k in img_keys
    ], axis=2)
    pred_row = np.concatenate([
        rearrange(decoded_pred[k][0, :T].float().cpu(), "t c h w -> t h w c").numpy()
        for k in img_keys
    ], axis=2)
    video_np = (np.concatenate([gt_row, pred_row], axis=1) * 255).clip(0, 255).astype(np.uint8)

    if reward_values:
        W = video_np.shape[2]
        frames = [np.concatenate([video_np[i],
                                   render_reward_curve(reward_values, min(i, len(reward_values) - 1), W)],
                                  axis=0)
                  for i in range(T)]
        video_np = np.stack(frames)

    save_video_mp4(out_path, video_np, fps)
    print(f"  Saved video: {out_path}")
