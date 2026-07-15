#!/usr/bin/env python3
"""Replay-style Ctrl-World autoregressive evaluation on LIBERO rollouts.

This script keeps the output/comparison behavior of
test_ctrl_world_libero_autoregressive.py, but builds Ctrl-World inputs like
ctrl_world/scripts/rollout_replay_traj.py:

* initialize a history buffer with the first frame repeated num_history * 4
  times;
* select history by replay-style history_idx, default [0,0,-8,-6,-4,-2];
* use the current chunk of LIBERO eef/gripper states as the future condition;
* after each chunk, push only the last predicted frame/state into history.
"""

from __future__ import annotations

import argparse
import json
import os
import time
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

from ctrl_world.models.ctrl_world import CrtlWorld
from ctrl_world.models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline

from test_ctrl_world_libero_prediction import (
    collect_libero_rollout,
    compute_metrics,
    decode_stacked_latents,
    encode_views_to_stacked_latents,
    get_task,
    load_ctrl_world_model,
    make_comparison_video,
    normalize_bound,
    resize_uint8_video,
    torch_dtype,
    write_video,
)


@dataclass
class ReplayStyleSummary:
    benchmark: str
    task_id: int
    task_name: str
    instruction: str
    action_source: str
    demo_hdf5: Optional[str]
    demo_id: str
    checkpoint: str
    output_dir: str
    num_history: int
    chunk_frames: int
    num_chunks: int
    stride: int
    history_idx: List[int]
    total_pred_frames: int
    num_cams: int
    height: int
    width: int
    metrics: Dict[str, Any]
    videos: Dict[str, str]


def parse_history_idx(raw: str) -> List[int]:
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("--history-idx must contain at least one integer")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay-style autoregressive Ctrl-World evaluation on LIBERO-generated frames."
    )
    parser.add_argument("--benchmark", type=str, default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--demo-hdf5", type=str, default=None)
    parser.add_argument("--demo-id", type=str, default="demo_0")
    parser.add_argument("--action-source", choices=["demo", "random", "noop"], default="demo")
    parser.add_argument("--demo-root", type=str, default="/mnt/data/lcx3/dataset/libero_spatial")
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="/mnt/data/lcx3/Ctrl-World/model_ckpt/libero_spatial/"
        "2026-07-03T16-55-44_libero_spatial/checkpoint-100000.pt",
    )
    parser.add_argument("--svd-model-path", type=str, default="/mnt/data/lcx3/checkpoint/ctrl_world/svd/svd_model")
    parser.add_argument("--clip-model-path", type=str, default="/mnt/data/lcx3/checkpoint/ctrl_world/clip/clip_model")
    parser.add_argument("--stat-path", type=str, default="/mnt/data/lcx3/dataset/dateset_meta_info/spatial/stat.json")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/data/lcx3/AcceRL/tests_dsj/ctrl_world_libero_replay_style_eval",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--num-cams", type=int, default=2)
    parser.add_argument("--num-history", type=int, default=6)
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=5,
        help="Frames predicted by each Ctrl-World call. This should match the checkpoint's num_frames.",
    )
    parser.add_argument("--num-chunks", type=int, default=8)
    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help="Replay-style chunk advance. Must be chunk_frames - 1 to match rollout_replay_traj.py.",
    )
    parser.add_argument(
        "--history-idx",
        type=str,
        default="0,0,-8,-6,-4,-2",
        help="Comma-separated history buffer indices used by rollout_replay_traj.py.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--rotate-libero-images", action="store_true")
    parser.add_argument(
        "--keep-model-loaded-only",
        action="store_true",
        help="Load Ctrl-World and exit before collecting LIBERO data.",
    )
    return parser.parse_args()


def collect_args_for_replay_style(args: argparse.Namespace, total_rollout_frames: int) -> argparse.Namespace:
    copied = SimpleNamespace(**vars(args))
    copied.start_index = 0
    copied.num_history = 0
    copied.num_frames = args.start_index + total_rollout_frames
    return copied


def pad_rows(rows: np.ndarray, target_len: int) -> np.ndarray:
    if rows.shape[0] >= target_len:
        return rows[:target_len]
    if rows.shape[0] == 0:
        raise ValueError("Cannot pad an empty state array")
    pad = np.repeat(rows[-1:], target_len - rows.shape[0], axis=0)
    return np.concatenate([rows, pad], axis=0)


def build_raw_action_cond(
    raw_states: np.ndarray,
    history_states: Sequence[np.ndarray],
    history_idx: Sequence[int],
    future_start: int,
    chunk_frames: int,
) -> np.ndarray:
    history_part = np.concatenate([history_states[idx] for idx in history_idx], axis=0)
    future_part = pad_rows(raw_states[future_start : future_start + chunk_frames], chunk_frames)
    action_cond = np.concatenate([history_part, future_part], axis=0).astype(np.float32)
    expected = len(history_idx) + chunk_frames
    if action_cond.shape != (expected, 7):
        raise ValueError(f"Expected action_cond shape {(expected, 7)}, got {action_cond.shape}")
    return action_cond


@torch.no_grad()
def predict_chunk_latents(
    model: CrtlWorld,
    current_latent: torch.Tensor,
    history: torch.Tensor,
    raw_action_cond: np.ndarray,
    state_01: np.ndarray,
    state_99: np.ndarray,
    instruction: str,
    args: argparse.Namespace,
) -> torch.Tensor:
    norm_action_cond = normalize_bound(raw_action_cond, state_01, state_99)
    act_dtype = model.action_encoder.action_encode[0].weight.dtype
    action_tensor = torch.from_numpy(norm_action_cond).unsqueeze(0).to(device=args.device, dtype=act_dtype)
    if model.args.text_cond:
        text_token = model.action_encoder(action_tensor, [instruction], model.tokenizer, model.text_encoder)
    else:
        text_token = model.action_encoder(action_tensor)

    _, pred_latents = CtrlWorldDiffusionPipeline.__call__(
        model.pipeline,
        image=current_latent,
        text=text_token,
        width=args.width,
        height=args.height * args.num_cams,
        num_frames=args.chunk_frames,
        history=history,
        num_inference_steps=args.num_inference_steps,
        decode_chunk_size=args.chunk_frames,
        max_guidance_scale=1.0,
        fps=7,
        motion_bucket_id=127,
        output_type="latent",
        return_dict=False,
        frame_level_cond=True,
    )
    return pred_latents[0]


def replay_style_autoregressive_predict(
    model: CrtlWorld,
    raw_states: np.ndarray,
    stacked_gt_latents: torch.Tensor,
    state_01: np.ndarray,
    state_99: np.ndarray,
    instruction: str,
    args: argparse.Namespace,
    history_idx: Sequence[int],
) -> Tuple[torch.Tensor, List[int]]:
    base = args.start_index
    first_latent = stacked_gt_latents[base : base + 1]
    first_state = raw_states[base : base + 1]
    if first_latent.shape[0] != 1 or first_state.shape != (1, 7):
        raise ValueError(f"Invalid first latent/state at start_index={base}")

    history_buffer_len = args.num_history * 4
    history_latents: List[torch.Tensor] = [first_latent] * history_buffer_len
    history_states: List[np.ndarray] = [first_state] * history_buffer_len

    pred_segments: List[torch.Tensor] = []
    output_frame_indices: List[int] = []
    rollover_idx = args.chunk_frames - 1

    for chunk_id in range(args.num_chunks):
        start_id = chunk_id * args.stride
        future_start = base + start_id
        raw_action_cond = build_raw_action_cond(
            raw_states=raw_states,
            history_states=history_states,
            history_idx=history_idx,
            future_start=future_start,
            chunk_frames=args.chunk_frames,
        )
        history = torch.cat([history_latents[idx] for idx in history_idx], dim=0).unsqueeze(0)
        current_latent = history_latents[-1]

        print(
            f"[ctrl-world] chunk {chunk_id + 1}/{args.num_chunks}: "
            f"future=[{future_start}:{future_start + args.chunk_frames}], "
            f"history_idx={list(history_idx)}"
        )
        pred_chunk = predict_chunk_latents(
            model=model,
            current_latent=current_latent,
            history=history,
            raw_action_cond=raw_action_cond,
            state_01=state_01,
            state_99=state_99,
            instruction=instruction,
            args=args,
        )

        take = args.chunk_frames if chunk_id == args.num_chunks - 1 else args.stride
        pred_segments.append(pred_chunk[:take])
        output_frame_indices.extend(range(start_id, start_id + take))

        history_latents.append(pred_chunk[rollover_idx : rollover_idx + 1])
        history_states.append(raw_action_cond[len(history_idx) + rollover_idx : len(history_idx) + rollover_idx + 1])

    return torch.cat(pred_segments, dim=0), output_frame_indices


def main() -> None:
    args = parse_args()
    history_idx = parse_history_idx(args.history_idx)
    if len(history_idx) != args.num_history:
        raise ValueError(f"--history-idx length {len(history_idx)} must equal --num-history {args.num_history}")
    if args.chunk_frames <= 0:
        raise ValueError("--chunk-frames must be positive")
    if args.stride != args.chunk_frames - 1:
        raise ValueError("--stride must equal chunk_frames - 1 to match Ctrl-World replay inference")

    total_rollout_frames = args.stride * (args.num_chunks - 1) + args.chunk_frames
    out_dir = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    task, _ = get_task(args.benchmark, args.task_id)
    instruction = task.language
    print(f"[task] {args.benchmark} task_id={args.task_id}: {task.name}")
    print(f"[task] instruction: {instruction}")
    print(
        f"[eval] replay-style total_frames={total_rollout_frames}, "
        f"chunk_frames={args.chunk_frames}, stride={args.stride}, history_idx={history_idx}"
    )

    model_args = SimpleNamespace(**vars(args))
    model_args.num_frames = args.chunk_frames
    print("[model] loading Ctrl-World...")
    model = load_ctrl_world_model(model_args)
    print("[model] loaded")
    if args.keep_model_loaded_only:
        return

    print("[libero] collecting GT rollout frames...")
    collect_args = collect_args_for_replay_style(args, total_rollout_frames)
    frames_by_t, raw_states, demo_path, demo_id = collect_libero_rollout(collect_args, task, instruction)
    print(f"[libero] collected {len(frames_by_t)} frames, states={raw_states.shape}")
    if demo_path is not None:
        print(f"[libero] replayed demo: {demo_path}:{demo_id}")

    with open(args.stat_path, "r", encoding="utf-8") as f:
        stat = json.load(f)
    state_01 = np.asarray(stat["state_01"], dtype=np.float32)[None, :]
    state_99 = np.asarray(stat["state_99"], dtype=np.float32)[None, :]

    dtype = torch_dtype(args.dtype)
    print("[ctrl-world] encoding GT frames...")
    stacked_gt_latents = encode_views_to_stacked_latents(
        model,
        frames_by_t,
        height=args.height,
        width=args.width,
        dtype=dtype,
        device=args.device,
    )

    print("[ctrl-world] replay-style autoregressive prediction...")
    pred_latents, relative_gt_indices = replay_style_autoregressive_predict(
        model=model,
        raw_states=raw_states,
        stacked_gt_latents=stacked_gt_latents,
        state_01=state_01,
        state_99=state_99,
        instruction=instruction,
        args=args,
        history_idx=history_idx,
    )
    pred_videos = decode_stacked_latents(
        model,
        pred_latents,
        num_cams=args.num_cams,
        height=args.height,
        width=args.width,
        chunk_size=args.chunk_frames,
    )

    gt_indices = [args.start_index + idx for idx in relative_gt_indices]
    gt_frames = [frames_by_t[min(idx, len(frames_by_t) - 1)] for idx in gt_indices]
    gt_videos = [
        resize_uint8_video([frames[cam_id] for frames in gt_frames], (args.height, args.width))
        for cam_id in range(args.num_cams)
    ]

    metrics = compute_metrics(gt_videos, pred_videos)
    comparison = make_comparison_video(gt_videos, pred_videos)

    video_paths: Dict[str, str] = {}
    comparison_path = out_dir / "replay_style_comparison_gt_pred_diff.mp4"
    write_video(comparison_path, comparison, fps=args.fps)
    video_paths["comparison"] = str(comparison_path)
    for cam_id, (gt, pred) in enumerate(zip(gt_videos, pred_videos)):
        gt_path = out_dir / f"cam{cam_id}_gt.mp4"
        pred_path = out_dir / f"cam{cam_id}_pred_replay_style.mp4"
        write_video(gt_path, gt, fps=args.fps)
        write_video(pred_path, pred, fps=args.fps)
        video_paths[f"cam{cam_id}_gt"] = str(gt_path)
        video_paths[f"cam{cam_id}_pred"] = str(pred_path)

    summary = ReplayStyleSummary(
        benchmark=args.benchmark,
        task_id=args.task_id,
        task_name=task.name,
        instruction=instruction,
        action_source=args.action_source,
        demo_hdf5=str(demo_path) if demo_path is not None else None,
        demo_id=demo_id,
        checkpoint=args.ckpt_path,
        output_dir=str(out_dir),
        num_history=args.num_history,
        chunk_frames=args.chunk_frames,
        num_chunks=args.num_chunks,
        stride=args.stride,
        history_idx=history_idx,
        total_pred_frames=int(pred_latents.shape[0]),
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
