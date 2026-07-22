#!/usr/bin/env python3
"""Closed-loop Ctrl-World WM prediction test with VLA policy actions.

This script tests the Ctrl-World world model in a **closed-loop** setting,
matching the imagination rollout in ds_wm_discrete_diffusion.py:

Default trajectory-imagination mode:

1. Collect a complete real trajectory with VLA observing LIBERO.
2. Select start_index and use only the real prefix as WM history.
3. At each imagination chunk:
   a. Decode the current predicted latent to camera images.
   b. Query VLA from those WM images.
   c. Feed the queried unnormalised actions to Ctrl-World.
4. Compare the imagined trajectory with the original real trajectory.

No pre-collected future action trajectory is fed to the WM.
The optional action-matched-live mode also applies imagined actions to LIBERO.
"""
from __future__ import annotations
'''
example: CUDA_VISIBLE_DEVICES=0 python tests_dsj/test_ctrl_world_wm_closed_loop_predict.py --ckpt-path /mnt/data/lcx3/Ctrl-World/model_ckpt/libero_vla_delta_finetune/2026-07-21T16-40-56_libero_vla_delta_finetune/checkpoint-20000.pt --condition-stat-path /mnt/data/lcx3/Ctrl-World/model_ckpt/libero_vla_delta_finetune/2026-07-21T16-40-56_libero_vla_delta_finetune/condition_stat.json --svd-model-path /mnt/data/lcx3/checkpoint/ctrl_world/svd/svd_model --clip-model-path /mnt/data/lcx3/checkpoint/ctrl_world/clip/clip_model --num-cams 2 --height 192 --width 320 --num-history 6 --chunk-frames 5 --num-chunks 8 --stride 4 --history-stride 1 --start-index 28 --device cuda --dtype bf16
'''
import argparse
import json
import os
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

from experiments.robot.libero.libero_utils import (  # noqa: E402
    GenerateConfig,
    get_libero_dummy_action,
    get_libero_env,
)
from experiments.robot.openvla_utils import resize_image_for_policy  # noqa: E402
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK  # noqa: E402
from rl.actor_critic_model_discrete import ActorCritic  # noqa: E402
from rl.utils import prepare_one_obs  # noqa: E402

from test_ctrl_world_libero_prediction import (  # noqa: E402
    compute_metrics,
    decode_stacked_latents,
    encode_views_to_stacked_latents,
    get_task,
    load_ctrl_world_model,
    make_comparison_video,
    resize_uint8_video,
    torch_dtype,
    write_video,
)
from test_ctrl_world_libero_replay_style_autoregressive import (  # noqa: E402
    parse_history_idx,
    predict_chunk_latents,
)
from test_ctrl_world_libero_vla_replay_style_autoregressive import (  # noqa: E402
    camera_frames_from_raw_obs,
    get_initial_state,
    load_vla_actor,
    prepare_vla_observation,
    process_vla_action_for_env,
)


# ================================================================
# Data structure
# ================================================================

@dataclass
class ClosedLoopSummary:
    evaluation_mode: str
    benchmark: str
    task_id: int
    task_name: str
    instruction: str
    initial_state_id: int
    vla_pretrained_checkpoint: str
    vla_checkpoint2: Optional[str]
    vla_unnorm_key: str
    checkpoint: str
    output_dir: str
    num_history: int
    chunk_frames: int
    num_chunks: int
    stride: int
    history_stride: Optional[int]
    history_idx: Optional[List[int]]
    total_pred_frames: int
    rollout_steps: int
    rollout_done: bool
    rollout_success: bool
    num_cams: int
    height: int
    width: int
    metrics: Dict[str, Any]
    videos: Dict[str, str]


# ================================================================
# Argument parsing
# ================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Closed-loop Ctrl-World WM prediction with VLA policy actions."
    )

    # --- Task / environment ---
    parser.add_argument("--benchmark", type=str, default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--initial-state-id", type=int, default=0)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument(
        "--evaluation-mode",
        choices=["trajectory-imagination", "action-matched-live"],
        default="trajectory-imagination",
        help=(
            "trajectory-imagination first records a real VLA/LIBERO trajectory, then "
            "imagines from start-index using VLA actions queried from WM images. "
            "action-matched-live applies each imagined action to LIBERO as well."
        ),
    )

    # --- VLA policy ---
    parser.add_argument(
        "--vla-pretrained-checkpoint",
        type=str,
        default="/mnt/data/lcx3/checkpoint/dsj/openvla-7b+libero_spatial_no_noops+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--discrete_acts--proprio_state--100000_chkpt",
        help="OpenVLA/ActorCritic pretrained checkpoint directory.",
    )
    parser.add_argument("--vla-checkpoint2", type=str, default="/mnt/data/lcx3/checkpoint/dsj/20251225_113851_distill_checkpoint_latest.pt")
    parser.add_argument("--no-vla-checkpoint2", action="store_true")
    parser.add_argument("--vla-unnorm-key", type=str, default=None)
    parser.add_argument("--vla-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--vla-use-proprio", action="store_true", default=False)
    parser.add_argument("--vla-num-images-in-input", type=int, default=1)
    parser.add_argument("--vla-center-crop", action="store_true", default=True)
    parser.add_argument("--no-vla-center-crop", action="store_false", dest="vla_center_crop")
    parser.add_argument("--vla-deterministic", action="store_true", default=True)
    parser.add_argument("--vla-stochastic", action="store_false", dest="vla_deterministic")
    parser.add_argument("--vla-open-loop-steps", type=int, default=NUM_ACTIONS_CHUNK)

    # --- Ctrl-World WM ---
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="/mnt/data/lcx3/Ctrl-World/model_ckpt/libero_spatial/2026-07-03T16-55-44_libero_spatial/checkpoint-100000.pt",
    )
    parser.add_argument("--svd-model-path", type=str, default="/mnt/data/lcx3/checkpoint/ctrl_world/svd/svd_model")
    parser.add_argument("--clip-model-path", type=str, default="/mnt/data/lcx3/checkpoint/ctrl_world/clip/clip_model")
    parser.add_argument(
        "--stat-path",
        type=str,
        default="/mnt/data/lcx3/dataset/dateset_meta_info/spatial/stat.json",
        help="Legacy state-stat path kept for CLI compatibility; raw-action inference uses --condition-stat-path.",
    )
    parser.add_argument(
        "--condition-stat-path",
        type=str,
        default=None,
        help="Action condition statistics. Defaults to condition_stat.json next to --ckpt-path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/data/lcx3/AcceRL/tests_dsj/ctrl_world_wm_closed_loop_eval",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--num-cams", type=int, default=2)
    parser.add_argument("--num-history", type=int, default=6)
    parser.add_argument("--chunk-frames", type=int, default=5)
    parser.add_argument("--num-chunks", type=int, default=8)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument(
        "--history-stride",
        type=int,
        default=1,
        help=(
            "Training-aligned contiguous history stride (matches rollout_replay_traj_accerl_pt.py). "
            "When set, --history-idx is ignored."
        ),
    )
    parser.add_argument(
        "--no-history-stride",
        action="store_true",
        help="Use legacy sparse --history-idx buffering instead of --history-stride.",
    )
    parser.add_argument(
        "--history-idx",
        type=str,
        default="0,0,-8,-6,-4,-2",
        help="Legacy sparse history buffer indices; only used with --no-history-stride.",
    )
    parser.add_argument("--start-index", type=int, default=28)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument(
        "--rotate-libero-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rotate LIBERO images by 180 degrees, matching the WM training dataset.",
    )
    parser.add_argument("--keep-model-loaded-only", action="store_true")
    return parser.parse_args()


# ================================================================
# Core: closed-loop prediction helpers
# ================================================================

@torch.no_grad()
def decode_latent_to_images(
    model,
    latent: torch.Tensor,
    num_cams: int,
    height: int,
    width: int,
) -> List[np.ndarray]:
    """Decode a single latent frame to per-camera uint8 images.

    Args:
        latent: [4, latent_h*num_cams, latent_w] — single frame latent.
        num_cams: number of cameras.
        height, width: target image resolution.

    Returns:
        List of ``num_cams`` uint8 arrays, each [H, W, C].
    """
    videos = decode_stacked_latents(
        model,
        latent.unsqueeze(0),          # [1, 4, h, w]
        num_cams=num_cams,
        height=height,
        width=width,
        chunk_size=1,
    )
    return [v[0] for v in videos]      # each [H, W, C]


@torch.no_grad()
def query_vla_on_predicted_images(
    actor: ActorCritic,
    cfg: GenerateConfig,
    images: List[np.ndarray],
    instruction: str,
    vla_dtype: torch.dtype,
    deterministic: bool,
    resize_size: int = 224,
) -> np.ndarray:
    """Query the VLA policy on WM-predicted images.

    This mirrors the training code's ``infer.request(inputs_t)`` call:
    the VLA sees the predicted observation and outputs actions.

    Args:
        images: list of ``num_cams`` uint8 arrays [H, W, C].
                images[0] = agentview, images[1] = wrist (if num_cams > 1).

    Returns:
        actions_unnorm: [NUM_ACTIONS_CHUNK, ACTION_DIM] unnormalised actions.
    """
    agentview = images[0]
    wrist = images[1] if len(images) > 1 else agentview

    img = resize_image_for_policy(agentview, resize_size)
    wrist_img = resize_image_for_policy(wrist, resize_size)

    # use_proprio is False by default, so state is ignored; provide a dummy.
    observation = {
        "full_image": img,
        "wrist_image": wrist_img,
        "state": np.zeros(ACTION_DIM, dtype=np.float32),
    }
    inputs_t = prepare_one_obs(cfg, actor.processor, observation, instruction, vla_dtype)
    inputs_batch = actor.prepare_inputs_batch([inputs_t])
    action_logits, _ = actor(inputs_batch)
    _, _, normalized_actions = actor.post_process(
        action_logits, deterministic=[deterministic]
    )
    actions_unnorm = actor.vla._unnormalize_actions(
        normalized_actions[0], cfg.unnorm_key
    )
    if hasattr(actions_unnorm, "detach"):
        actions_unnorm = actions_unnorm.detach().cpu().numpy()
    return np.asarray(actions_unnorm, dtype=np.float32)  # [N, 7]


@torch.no_grad()
def query_vla_on_raw_observation(
    actor: ActorCritic,
    cfg: GenerateConfig,
    raw_obs: Dict[str, Any],
    instruction: str,
    vla_dtype: torch.dtype,
    deterministic: bool,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    observation = prepare_vla_observation(raw_obs, resize_size=224)
    inputs_t = prepare_one_obs(cfg, actor.processor, observation, instruction, vla_dtype)
    inputs_batch = actor.prepare_inputs_batch([inputs_t])
    action_logits, _ = actor(inputs_batch)
    _, _, normalized_actions = actor.post_process(
        action_logits,
        deterministic=[deterministic],
    )
    raw_actions = actor.vla._unnormalize_actions(
        normalized_actions[0],
        cfg.unnorm_key,
    )
    if hasattr(raw_actions, "detach"):
        raw_actions = raw_actions.detach().cpu().numpy()
    raw_actions = np.asarray(raw_actions, dtype=np.float32)[: cfg.num_open_loop_steps]
    env_actions = [process_vla_action_for_env(action) for action in raw_actions]
    return env_actions, [action.copy() for action in raw_actions]


def load_action_condition_stats(
    condition_stat_path: Optional[str],
    ckpt_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    candidates = []
    if condition_stat_path:
        candidates.append(Path(condition_stat_path))
    candidates.append(Path(ckpt_path).resolve().parent / "condition_stat.json")
    stat_path = next((path for path in candidates if path.is_file()), None)
    if stat_path is None:
        raise FileNotFoundError(
            "Raw-action WM inference requires action condition statistics. "
            f"Tried: {[str(path) for path in candidates]}"
        )

    with stat_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if payload.get("condition_mode") != "action":
        raise ValueError(
            f"{stat_path} has condition_mode={payload.get('condition_mode')!r}, "
            "expected 'action'."
        )
    if payload.get("compose_interval_actions") is True:
        raise ValueError(
            f"{stat_path} is for composed interval actions, but this script uses "
            "single-step unnormalised VLA actions as WM condition."
        )
    if "condition_p01" not in payload or "condition_p99" not in payload:
        raise KeyError(f"{stat_path} must contain condition_p01 and condition_p99")

    p01 = np.asarray(payload["condition_p01"], dtype=np.float32)[None, :]
    p99 = np.asarray(payload["condition_p99"], dtype=np.float32)[None, :]
    if p01.shape != (1, ACTION_DIM) or p99.shape != (1, ACTION_DIM):
        raise ValueError(
            f"Expected action condition stats shape {(1, ACTION_DIM)}, "
            f"got p01={p01.shape}, p99={p99.shape}"
        )
    print(f"[ctrl-world] action condition stats: {stat_path}")
    return p01, p99


def build_frame_aligned_future_actions(
    current_action: np.ndarray,
    actions_unnorm: np.ndarray,
    num_frames: int,
) -> np.ndarray:
    """Build actions aligned to predicted frames [current, ..., current+F-1].

    The training data stores action[t] as the action that produced frame[t].
    Therefore the current frame uses the already-known current_action, while
    VLA actions queried from the current image condition frames t+1 onward.
    """
    actions = np.asarray(actions_unnorm, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[-1] != ACTION_DIM:
        raise ValueError(f"Expected VLA actions shape [N,{ACTION_DIM}], got {actions.shape}")
    current = np.asarray(current_action, dtype=np.float32).reshape(1, ACTION_DIM)
    if num_frames == 1:
        return current
    if actions.shape[0] == 0:
        next_actions = np.repeat(current, num_frames - 1, axis=0)
    else:
        next_actions = actions[: num_frames - 1]
        if next_actions.shape[0] < num_frames - 1:
            pad = np.repeat(next_actions[-1:], num_frames - 1 - next_actions.shape[0], axis=0)
            next_actions = np.concatenate([next_actions, pad], axis=0)
    return np.concatenate([current, next_actions], axis=0).astype(np.float32)


def frame_aligned_reference_actions(
    step_unnormalized: Sequence[np.ndarray],
    num_frames: int,
) -> np.ndarray:
    """Build reference actions used only to seed the real-history prefix."""
    aligned = np.zeros((num_frames, ACTION_DIM), dtype=np.float32)
    if len(step_unnormalized) == 0:
        return aligned
    for frame_idx in range(1, num_frames):
        src = min(frame_idx - 1, len(step_unnormalized) - 1)
        aligned[frame_idx] = np.asarray(step_unnormalized[src], dtype=np.float32)
    return aligned


def _init_history_stride_buffers(
    stacked_gt_latents: torch.Tensor,
    reference_actions: np.ndarray,
    start_index: int,
    num_history: int,
    history_stride: int,
) -> Tuple[List[torch.Tensor], List[np.ndarray], int]:
    history_span = int(num_history) * int(history_stride)
    if start_index < history_span:
        raise ValueError(
            f"--start-index {start_index} must be >= num_history * history_stride "
            f"({history_span}) when using --history-stride."
        )
    latent_buffer = [stacked_gt_latents[i : i + 1] for i in range(start_index + 1)]
    action_buffer = [reference_actions[i : i + 1] for i in range(start_index + 1)]
    return latent_buffer, action_buffer, history_span


def _history_lags(num_history: int, history_stride: int) -> List[int]:
    return list(range(int(num_history) * int(history_stride), 0, -int(history_stride)))


def _predict_and_update_chunk(
    model,
    actor: ActorCritic,
    vla_cfg: GenerateConfig,
    vla_dtype: torch.dtype,
    current_latent: torch.Tensor,
    history: torch.Tensor,
    history_action: np.ndarray,
    current_action: np.ndarray,
    action_01: np.ndarray,
    action_99: np.ndarray,
    instruction: str,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, np.ndarray]:
    images = decode_latent_to_images(
        model,
        current_latent,
        num_cams=args.num_cams,
        height=args.height,
        width=args.width,
    )
    actions_unnorm = query_vla_on_predicted_images(
        actor=actor,
        cfg=vla_cfg,
        images=images,
        instruction=instruction,
        vla_dtype=vla_dtype,
        deterministic=args.vla_deterministic,
    )
    frame_actions = build_frame_aligned_future_actions(
        current_action=current_action,
        actions_unnorm=actions_unnorm,
        num_frames=args.chunk_frames,
    )
    raw_action_cond = np.concatenate([history_action, frame_actions], axis=0).astype(np.float32)
    pred_chunk = predict_chunk_latents(
        model=model,
        current_latent=current_latent.unsqueeze(0),
        history=history,
        raw_action_cond=raw_action_cond,
        state_01=action_01,
        state_99=action_99,
        instruction=instruction,
        args=args,
    )
    return pred_chunk, frame_actions


@torch.no_grad()
def closed_loop_history_stride_predict(
    model,
    actor: ActorCritic,
    vla_cfg: GenerateConfig,
    vla_dtype: torch.dtype,
    stacked_gt_latents: torch.Tensor,
    reference_actions: np.ndarray,
    action_01: np.ndarray,
    action_99: np.ndarray,
    instruction: str,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, List[int], np.ndarray]:
    """Closed-loop prediction with training-aligned contiguous history."""
    history_lags = _history_lags(args.num_history, args.history_stride)
    latent_buffer, action_buffer, _history_span = _init_history_stride_buffers(
        stacked_gt_latents=stacked_gt_latents,
        reference_actions=reference_actions,
        start_index=args.start_index,
        num_history=args.num_history,
        history_stride=args.history_stride,
    )

    pred_segments: List[torch.Tensor] = []
    output_frame_indices: List[int] = []
    interactive_action_chunks: List[np.ndarray] = []

    for chunk_id in range(args.num_chunks):
        start_id = chunk_id * args.stride
        history_action = np.concatenate([action_buffer[-1 - lag] for lag in history_lags], axis=0)
        history = torch.cat([latent_buffer[-1 - lag] for lag in history_lags], dim=0).unsqueeze(0)
        current_latent = latent_buffer[-1][0]
        current_action = action_buffer[-1][0]

        print(
            f"[closed-loop] chunk {chunk_id + 1}/{args.num_chunks}: "
            f"history_stride={args.history_stride}, history_lags={history_lags}, "
            f"start_index={args.start_index + start_id}"
        )
        pred_chunk, frame_actions = _predict_and_update_chunk(
            model=model,
            actor=actor,
            vla_cfg=vla_cfg,
            vla_dtype=vla_dtype,
            current_latent=current_latent,
            history=history,
            history_action=history_action,
            current_action=current_action,
            action_01=action_01,
            action_99=action_99,
            instruction=instruction,
            args=args,
        )

        take = args.chunk_frames if chunk_id == args.num_chunks - 1 else args.stride
        pred_segments.append(pred_chunk[:take])
        output_frame_indices.extend(range(start_id, start_id + take))
        interactive_action_chunks.append(frame_actions[1:].copy())

        for rel_idx in range(1, int(args.chunk_frames)):
            latent_buffer.append(pred_chunk[rel_idx : rel_idx + 1])
            action_buffer.append(frame_actions[rel_idx : rel_idx + 1])

    return (
        torch.cat(pred_segments, dim=0),
        output_frame_indices,
        np.concatenate(interactive_action_chunks, axis=0),
    )


@torch.no_grad()
def closed_loop_autoregressive_predict(
    model,
    actor: ActorCritic,
    vla_cfg: GenerateConfig,
    vla_dtype: torch.dtype,
    stacked_gt_latents: torch.Tensor,
    reference_actions: np.ndarray,
    action_01: np.ndarray,
    action_99: np.ndarray,
    instruction: str,
    args: argparse.Namespace,
    history_idx: Sequence[int],
) -> Tuple[torch.Tensor, List[int], np.ndarray]:
    """Closed-loop prediction with legacy sparse history_idx buffering."""
    base = args.start_index
    first_latent = stacked_gt_latents[base : base + 1]

    history_buffer_len = args.num_history * 4
    history_latents: List[torch.Tensor] = [first_latent] * history_buffer_len
    history_actions: List[np.ndarray] = [reference_actions[base : base + 1]] * history_buffer_len
    current_latent = first_latent[0]

    pred_segments: List[torch.Tensor] = []
    output_frame_indices: List[int] = []
    interactive_action_chunks: List[np.ndarray] = []

    for chunk_id in range(args.num_chunks):
        start_id = chunk_id * args.stride

        history_part = np.concatenate([history_actions[idx] for idx in history_idx], axis=0)
        history = torch.cat([history_latents[idx] for idx in history_idx], dim=0).unsqueeze(0)

        print(
            f"[closed-loop] chunk {chunk_id + 1}/{args.num_chunks}: "
            f"history_idx={list(history_idx)}, start_index={base + start_id}"
        )
        current_action = history_actions[-1][0]
        pred_chunk, frame_actions = _predict_and_update_chunk(
            model=model,
            actor=actor,
            vla_cfg=vla_cfg,
            vla_dtype=vla_dtype,
            current_latent=current_latent,
            history=history,
            history_action=history_part,
            current_action=current_action,
            action_01=action_01,
            action_99=action_99,
            instruction=instruction,
            args=args,
        )

        take = args.chunk_frames if chunk_id == args.num_chunks - 1 else args.stride
        pred_segments.append(pred_chunk[:take])
        output_frame_indices.extend(range(start_id, start_id + take))
        interactive_action_chunks.append(frame_actions[1:].copy())

        for rel_idx in range(1, int(args.chunk_frames)):
            history_latents.append(pred_chunk[rel_idx : rel_idx + 1])
            history_actions.append(frame_actions[rel_idx : rel_idx + 1])
        current_latent = pred_chunk[int(args.chunk_frames) - 1]

    return (
        torch.cat(pred_segments, dim=0),
        output_frame_indices,
        np.concatenate(interactive_action_chunks, axis=0),
    )


@torch.no_grad()
def collect_real_vla_trajectory(
    actor: ActorCritic,
    vla_cfg: GenerateConfig,
    vla_dtype: torch.dtype,
    task: Any,
    task_suite: Any,
    instruction: str,
    total_frames: int,
    args: argparse.Namespace,
) -> Tuple[List[List[np.ndarray]], np.ndarray, np.ndarray, bool, bool]:
    """Collect an independent real trajectory with VLA observing LIBERO."""
    env, _ = get_libero_env(task, vla_cfg.model_family, resolution=256)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    raw_obs = env.reset()
    initial_state = get_initial_state(task_suite, args.task_id, args.initial_state_id)
    if initial_state is not None:
        raw_obs = env.set_init_state(initial_state)
    for _ in range(args.num_steps_wait):
        raw_obs, _, _, _ = env.step(get_libero_dummy_action(vla_cfg.model_family))

    frames: List[List[np.ndarray]] = [
        camera_frames_from_raw_obs(raw_obs, rotate=args.rotate_libero_images)
    ]
    raw_actions: List[np.ndarray] = []
    env_actions: List[np.ndarray] = []
    action_queue = deque()
    done = False
    success = False

    try:
        while len(frames) < total_frames:
            if done:
                frames.append([image.copy() for image in frames[-1]])
                raw_actions.append(
                    raw_actions[-1].copy()
                    if raw_actions
                    else np.zeros(ACTION_DIM, dtype=np.float32)
                )
                continue
            if not action_queue:
                env_batch, raw_batch = query_vla_on_raw_observation(
                    actor=actor,
                    cfg=vla_cfg,
                    raw_obs=raw_obs,
                    instruction=instruction,
                    vla_dtype=vla_dtype,
                    deterministic=args.vla_deterministic,
                )
                action_queue.extend(zip(env_batch, raw_batch))

            env_action, raw_action = action_queue.popleft()
            raw_obs, _, done, info = env.step(env_action.tolist())
            success = bool(info.get("is_success", done))
            env_actions.append(np.asarray(env_action, dtype=np.float32))
            raw_actions.append(np.asarray(raw_action, dtype=np.float32))
            frames.append(
                camera_frames_from_raw_obs(
                    raw_obs,
                    rotate=args.rotate_libero_images,
                )
            )
    finally:
        env.close()

    return (
        frames,
        np.stack(raw_actions, axis=0),
        np.stack(env_actions, axis=0)
        if env_actions
        else np.zeros((0, ACTION_DIM), dtype=np.float32),
        done,
        success,
    )


@torch.no_grad()
def run_live_closed_loop(
    model,
    actor: ActorCritic,
    vla_cfg: GenerateConfig,
    vla_dtype: torch.dtype,
    task: Any,
    task_suite: Any,
    instruction: str,
    action_01: np.ndarray,
    action_99: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, List[List[np.ndarray]], np.ndarray, np.ndarray, bool, bool]:
    """Run one synchronized WM/VLA/LIBERO closed loop.

    Before start_index, VLA acts on real LIBERO observations to build real WM
    history. After start_index, VLA sees only the current WM-decoded image.
    Each online VLA action is then supplied to both WM and LIBERO, so the
    resulting simulator frames are a valid action-matched reference.
    """
    env, _ = get_libero_env(task, vla_cfg.model_family, resolution=256)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    raw_obs = env.reset()
    initial_state = get_initial_state(task_suite, args.task_id, args.initial_state_id)
    if initial_state is not None:
        raw_obs = env.set_init_state(initial_state)
    for _ in range(args.num_steps_wait):
        raw_obs, _, _, _ = env.step(get_libero_dummy_action(vla_cfg.model_family))

    warmup_frames: List[List[np.ndarray]] = [
        camera_frames_from_raw_obs(raw_obs, rotate=args.rotate_libero_images)
    ]
    warmup_raw_actions: List[np.ndarray] = []
    executed_env_actions: List[np.ndarray] = []
    warmup_queue = deque()
    done = False
    success = False

    try:
        while len(warmup_frames) <= args.start_index:
            if not warmup_queue:
                env_batch, wm_batch = query_vla_on_raw_observation(
                    actor=actor,
                    cfg=vla_cfg,
                    raw_obs=raw_obs,
                    instruction=instruction,
                    vla_dtype=vla_dtype,
                    deterministic=args.vla_deterministic,
                )
                warmup_queue.extend(zip(env_batch, wm_batch))

            env_action, raw_action = warmup_queue.popleft()
            if not done:
                raw_obs, _, done, info = env.step(env_action.tolist())
                success = bool(info.get("is_success", done))
            executed_env_actions.append(np.asarray(env_action, dtype=np.float32))
            warmup_raw_actions.append(np.asarray(raw_action, dtype=np.float32))
            warmup_frames.append(
                camera_frames_from_raw_obs(raw_obs, rotate=args.rotate_libero_images)
            )

        stacked_warmup_latents = encode_views_to_stacked_latents(
            model,
            warmup_frames,
            height=args.height,
            width=args.width,
            dtype=torch_dtype(args.dtype),
            device=args.device,
        )
        reference_actions = frame_aligned_reference_actions(
            warmup_raw_actions,
            len(warmup_frames),
        )
        history_lags = _history_lags(args.num_history, args.history_stride)
        latent_buffer, action_buffer, _ = _init_history_stride_buffers(
            stacked_gt_latents=stacked_warmup_latents,
            reference_actions=reference_actions,
            start_index=args.start_index,
            num_history=args.num_history,
            history_stride=args.history_stride,
        )

        pred_segments: List[torch.Tensor] = []
        interactive_action_chunks: List[np.ndarray] = []
        live_gt_frames: List[List[np.ndarray]] = [warmup_frames[-1]]

        for chunk_id in range(args.num_chunks):
            history_action = np.concatenate(
                [action_buffer[-1 - lag] for lag in history_lags],
                axis=0,
            )
            history = torch.cat(
                [latent_buffer[-1 - lag] for lag in history_lags],
                dim=0,
            ).unsqueeze(0)
            current_latent = latent_buffer[-1][0]
            current_action = action_buffer[-1][0]

            # This is the online interaction point: VLA sees the current WM image.
            predicted_images = decode_latent_to_images(
                model,
                current_latent,
                num_cams=args.num_cams,
                height=args.height,
                width=args.width,
            )
            queried_actions = query_vla_on_predicted_images(
                actor=actor,
                cfg=vla_cfg,
                images=predicted_images,
                instruction=instruction,
                vla_dtype=vla_dtype,
                deterministic=args.vla_deterministic,
            )
            frame_actions = build_frame_aligned_future_actions(
                current_action=current_action,
                actions_unnorm=queried_actions,
                num_frames=args.chunk_frames,
            )

            # Apply exactly the same future actions to the real simulator.
            for raw_action in frame_actions[1:]:
                env_action = process_vla_action_for_env(raw_action)
                if not done:
                    raw_obs, _, done, info = env.step(env_action.tolist())
                    success = bool(info.get("is_success", done))
                executed_env_actions.append(env_action)
                live_gt_frames.append(
                    camera_frames_from_raw_obs(
                        raw_obs,
                        rotate=args.rotate_libero_images,
                    )
                )

            raw_action_cond = np.concatenate(
                [history_action, frame_actions],
                axis=0,
            ).astype(np.float32)
            out_of_stat = np.logical_or(
                raw_action_cond < action_01,
                raw_action_cond > action_99,
            )
            clipped_fraction = float(out_of_stat.mean())
            pred_chunk = predict_chunk_latents(
                model=model,
                current_latent=current_latent.unsqueeze(0),
                history=history,
                raw_action_cond=raw_action_cond,
                state_01=action_01,
                state_99=action_99,
                instruction=instruction,
                args=args,
            )

            take = args.chunk_frames if chunk_id == args.num_chunks - 1 else args.stride
            pred_segments.append(pred_chunk[:take])
            interactive_action_chunks.append(frame_actions[1:].copy())
            for rel_idx in range(1, args.chunk_frames):
                latent_buffer.append(pred_chunk[rel_idx : rel_idx + 1])
                action_buffer.append(frame_actions[rel_idx : rel_idx + 1])

            print(
                f"[live closed-loop] chunk {chunk_id + 1}/{args.num_chunks}: "
                f"WM image -> VLA -> {args.chunk_frames - 1} shared WM/LIBERO actions; "
                f"condition clip fraction={clipped_fraction:.3f}"
            )

        pred_latents = torch.cat(pred_segments, dim=0)
        if len(live_gt_frames) != int(pred_latents.shape[0]):
            raise RuntimeError(
                f"Action-matched GT/pred length mismatch: "
                f"gt={len(live_gt_frames)}, pred={pred_latents.shape[0]}"
            )
        return (
            pred_latents,
            live_gt_frames,
            np.concatenate(interactive_action_chunks, axis=0),
            np.stack(executed_env_actions, axis=0),
            done,
            success,
        )
    finally:
        env.close()


# ================================================================
# Main
# ================================================================

def main() -> None:
    args = parse_args()
    history_stride: Optional[int] = None if args.no_history_stride else int(args.history_stride)
    history_idx: Optional[List[int]] = None
    if history_stride is not None:
        if history_stride <= 0:
            raise ValueError("--history-stride must be positive")
    else:
        raise ValueError(
            "The synchronized live closed-loop requires --history-stride; "
            "--no-history-stride is only supported by the legacy offline path."
        )
    if args.chunk_frames <= 0:
        raise ValueError("--chunk-frames must be positive")
    if args.stride != args.chunk_frames - 1:
        raise ValueError(
            "--stride must equal chunk_frames - 1 to match Ctrl-World replay inference"
        )
    if args.vla_open_loop_steps <= 0 or args.vla_open_loop_steps > NUM_ACTIONS_CHUNK:
        raise ValueError(
            f"--vla-open-loop-steps must be in [1, {NUM_ACTIONS_CHUNK}]"
        )

    total_pred_frames = args.stride * (args.num_chunks - 1) + args.chunk_frames
    total_rollout_frames = args.start_index + total_pred_frames
    out_dir = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    task, task_suite = get_task(args.benchmark, args.task_id)
    instruction = task.language
    print(f"[task] {args.benchmark} task_id={args.task_id}: {task.name}")
    print(f"[task] instruction: {instruction}")
    history_desc = (
        f"history_stride={history_stride}"
        if history_stride is not None
        else f"history_idx={history_idx}"
    )
    print(
        f"[eval] closed-loop prediction: "
        f"rollout_frames={total_rollout_frames}, pred_frames={total_pred_frames}, "
        f"chunk_frames={args.chunk_frames}, stride={args.stride}, "
        f"start_index={args.start_index}, {history_desc}"
    )

    # ---- Load VLA policy ----
    print("[vla] loading ActorCritic/OpenVLA...")
    actor, vla_cfg, vla_dtype = load_vla_actor(args)
    print(f"[vla] loaded, unnorm_key={vla_cfg.unnorm_key}")

    # ---- Load Ctrl-World WM ----
    print("[ctrl-world] loading Ctrl-World...")
    model_args = SimpleNamespace(**vars(args))
    model_args.num_frames = args.chunk_frames
    ctrl_world = load_ctrl_world_model(model_args)
    print("[models] loaded")
    if args.keep_model_loaded_only:
        return

    # ---- Load normalisation stats ----
    action_01, action_99 = load_action_condition_stats(
        args.condition_stat_path,
        args.ckpt_path,
    )

    args.history_stride = history_stride
    if args.evaluation_mode == "trajectory-imagination":
        print(
            "[trajectory] collecting real VLA/LIBERO trajectory, then imagining "
            "from start-index with VLA conditioned only on WM images"
        )
        (
            real_frames,
            real_raw_actions,
            executed_actions,
            rollout_done,
            rollout_success,
        ) = collect_real_vla_trajectory(
            actor=actor,
            vla_cfg=vla_cfg,
            vla_dtype=vla_dtype,
            task=task,
            task_suite=task_suite,
            instruction=instruction,
            total_frames=total_rollout_frames,
            args=args,
        )
        warmup_latents = encode_views_to_stacked_latents(
            ctrl_world,
            real_frames[: args.start_index + 1],
            height=args.height,
            width=args.width,
            dtype=torch_dtype(args.dtype),
            device=args.device,
        )
        reference_actions = frame_aligned_reference_actions(
            real_raw_actions,
            len(real_frames),
        )
        (
            pred_latents,
            relative_gt_indices,
            interactive_actions,
        ) = closed_loop_history_stride_predict(
            model=ctrl_world,
            actor=actor,
            vla_cfg=vla_cfg,
            vla_dtype=vla_dtype,
            stacked_gt_latents=warmup_latents,
            reference_actions=reference_actions,
            action_01=action_01,
            action_99=action_99,
            instruction=instruction,
            args=args,
        )
        gt_frames = [
            real_frames[args.start_index + relative_idx]
            for relative_idx in relative_gt_indices
        ]
    else:
        print(
            "[live] real warmup, then repeat: "
            "WM image -> VLA action -> both WM and LIBERO"
        )
        (
            pred_latents,
            gt_frames,
            interactive_actions,
            executed_actions,
            rollout_done,
            rollout_success,
        ) = run_live_closed_loop(
            model=ctrl_world,
            actor=actor,
            vla_cfg=vla_cfg,
            vla_dtype=vla_dtype,
            task=task,
            task_suite=task_suite,
            instruction=instruction,
            action_01=action_01,
            action_99=action_99,
            args=args,
        )

    actions_path = out_dir / "vla_executed_env_actions.npy"
    np.save(actions_path, executed_actions)
    interactive_actions_path = out_dir / "vla_wm_closed_loop_actions.npy"
    np.save(interactive_actions_path, interactive_actions)

    # ---- Decode predicted latents to videos ----
    pred_videos = decode_stacked_latents(
        ctrl_world,
        pred_latents,
        num_cams=args.num_cams,
        height=args.height,
        width=args.width,
        chunk_size=args.chunk_frames,
    )

    # ---- Build LIBERO reference videos for comparison ----
    gt_videos = [
        resize_uint8_video(
            [frames[cam_id] for frames in gt_frames], (args.height, args.width)
        )
        for cam_id in range(args.num_cams)
    ]

    # ---- Metrics & videos ----
    metrics = compute_metrics(gt_videos, pred_videos)
    comparison = make_comparison_video(gt_videos, pred_videos)

    video_paths: Dict[str, str] = {}
    comparison_path = out_dir / "closed_loop_comparison_gt_pred_diff.mp4"
    write_video(comparison_path, comparison, fps=args.fps)
    video_paths["comparison"] = str(comparison_path)
    for cam_id, (gt, pred) in enumerate(zip(gt_videos, pred_videos)):
        gt_path = out_dir / f"cam{cam_id}_gt.mp4"
        pred_path = out_dir / f"cam{cam_id}_closed_loop_pred.mp4"
        write_video(gt_path, gt, fps=args.fps)
        write_video(pred_path, pred, fps=args.fps)
        video_paths[f"cam{cam_id}_gt"] = str(gt_path)
        video_paths[f"cam{cam_id}_pred"] = str(pred_path)
    video_paths["vla_env_actions_npy"] = str(actions_path)
    video_paths["vla_wm_closed_loop_actions_npy"] = str(interactive_actions_path)

    # ---- Summary ----
    summary = ClosedLoopSummary(
        evaluation_mode=args.evaluation_mode,
        benchmark=args.benchmark,
        task_id=args.task_id,
        task_name=task.name,
        instruction=instruction,
        initial_state_id=args.initial_state_id,
        vla_pretrained_checkpoint=args.vla_pretrained_checkpoint,
        vla_checkpoint2=args.vla_checkpoint2 if not args.no_vla_checkpoint2 else None,
        vla_unnorm_key=vla_cfg.unnorm_key,
        checkpoint=args.ckpt_path,
        output_dir=str(out_dir),
        num_history=args.num_history,
        chunk_frames=args.chunk_frames,
        num_chunks=args.num_chunks,
        stride=args.stride,
        history_stride=history_stride,
        history_idx=history_idx,
        total_pred_frames=int(pred_latents.shape[0]),
        rollout_steps=len(executed_actions),
        rollout_done=rollout_done,
        rollout_success=rollout_success,
        num_cams=args.num_cams,
        height=args.height,
        width=args.width,
        metrics=metrics,
        videos=video_paths,
    )
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")

    print("[done] wrote:")
    print(f"  summary: {summary_path}")
    for key, value in video_paths.items():
        print(f"  {key}: {value}")
    print("[metrics]", json.dumps(metrics["overall"], indent=2))


if __name__ == "__main__":
    main()
