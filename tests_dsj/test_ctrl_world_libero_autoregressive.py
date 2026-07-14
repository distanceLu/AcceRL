#!/usr/bin/env python3
"""Autoregressively compare Ctrl-World predictions against LIBERO GT video."""

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
class AutoregressiveSummary:
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
    total_pred_frames: int
    stride: int
    num_cams: int
    height: int
    width: int
    metrics: Dict[str, Any]
    videos: Dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autoregressively evaluate Ctrl-World on LIBERO-generated frames."
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
        default="/mnt/data/lcx3/AcceRL/tests_dsj/ctrl_world_libero_autoregressive_eval",
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
        help="Ctrl-World frames predicted per model call. Keep this at the checkpoint's training num_frames.",
    )
    parser.add_argument("--num-chunks", type=int, default=8)
    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help="New frames contributed by each chunk. Default chunk_frames-1 removes overlap.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--rotate-libero-images", action="store_true")
    return parser.parse_args()


def _collect_args_for_long_gt(args: argparse.Namespace, total_pred_frames: int) -> argparse.Namespace:
    copied = SimpleNamespace(**vars(args))
    copied.num_frames = total_pred_frames
    return copied


def _build_action_cond(
    norm_states: np.ndarray,
    history_state_indices: Sequence[int],
    future_start: int,
    chunk_frames: int,
) -> np.ndarray:
    history_states = norm_states[np.asarray(history_state_indices, dtype=np.int64)]
    future_states = norm_states[future_start : future_start + chunk_frames]
    if future_states.shape[0] < chunk_frames:
        pad = np.repeat(future_states[-1:], chunk_frames - future_states.shape[0], axis=0)
        future_states = np.concatenate([future_states, pad], axis=0)
    action_cond = np.concatenate([history_states, future_states], axis=0)
    expected = len(history_state_indices) + chunk_frames
    if action_cond.shape != (expected, 7):
        raise ValueError(f"Expected action_cond shape {(expected, 7)}, got {action_cond.shape}")
    return action_cond.astype(np.float32)


@torch.no_grad()
def predict_chunk_latents(
    model: CrtlWorld,
    current_latent: torch.Tensor,
    history: torch.Tensor,
    action_cond: np.ndarray,
    instruction: str,
    args: argparse.Namespace,
) -> torch.Tensor:
    act_dtype = model.action_encoder.action_encode[0].weight.dtype
    action_tensor = torch.from_numpy(action_cond).unsqueeze(0).to(device=args.device, dtype=act_dtype)
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


def autoregressive_predict(
    model: CrtlWorld,
    norm_states: np.ndarray,
    stacked_gt_latents: torch.Tensor,
    instruction: str,
    args: argparse.Namespace,
) -> torch.Tensor:
    """Return stacked predicted latents [T,4,latent_h*num_cams,latent_w]."""
    current_idx = args.start_index + args.num_history
    history = stacked_gt_latents[args.start_index:current_idx].unsqueeze(0)
    if history.shape[1] != args.num_history:
        raise ValueError(f"Expected {args.num_history} history frames, got {history.shape[1]}")

    current_latent = stacked_gt_latents[current_idx : current_idx + 1]
    pred_segments: List[torch.Tensor] = []
    history_state_indices = list(range(args.start_index, current_idx))

    for chunk_id in range(args.num_chunks):
        future_start = current_idx + chunk_id * args.stride
        action_cond = _build_action_cond(
            norm_states=norm_states,
            history_state_indices=history_state_indices,
            future_start=future_start,
            chunk_frames=args.chunk_frames,
        )
        print(
            f"[ctrl-world] chunk {chunk_id + 1}/{args.num_chunks}: "
            f"future_start={future_start}, history_state_indices={history_state_indices}"
        )
        pred_chunk = predict_chunk_latents(
            model=model,
            current_latent=current_latent,
            history=history,
            action_cond=action_cond,
            instruction=instruction,
            args=args,
        )

        take = args.chunk_frames if chunk_id == args.num_chunks - 1 else args.stride
        pred_segments.append(pred_chunk[:take])

        current_latent = pred_chunk[args.stride - 1 : args.stride]
        for i in range(args.stride):
            history = torch.cat([history[:, 1:], pred_chunk[i : i + 1].unsqueeze(0)], dim=1)
            history_state_indices = history_state_indices[1:] + [future_start + i]

    return torch.cat(pred_segments, dim=0)


def main() -> None:
    args = parse_args()
    if args.stride <= 0 or args.stride > args.chunk_frames:
        raise ValueError("--stride must be in [1, chunk_frames]")

    total_pred_frames = args.stride * (args.num_chunks - 1) + args.chunk_frames
    out_dir = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    task, _ = get_task(args.benchmark, args.task_id)
    instruction = task.language
    print(f"[task] {args.benchmark} task_id={args.task_id}: {task.name}")
    print(f"[task] instruction: {instruction}")
    print(f"[eval] total_pred_frames={total_pred_frames}, duration={total_pred_frames / args.fps:.2f}s")

    model_args = SimpleNamespace(**vars(args))
    model_args.num_frames = args.chunk_frames
    print("[model] loading Ctrl-World...")
    model = load_ctrl_world_model(model_args)
    print("[model] loaded")

    print("[libero] collecting long GT rollout frames...")
    collect_args = _collect_args_for_long_gt(args, total_pred_frames)
    frames_by_t, states, demo_path, demo_id = collect_libero_rollout(collect_args, task, instruction)
    print(f"[libero] collected {len(frames_by_t)} frames, states={states.shape}")
    if demo_path is not None:
        print(f"[libero] replayed demo: {demo_path}:{demo_id}")

    with open(args.stat_path, "r", encoding="utf-8") as f:
        stat = json.load(f)
    state_01 = np.asarray(stat["state_01"], dtype=np.float32)[None, :]
    state_99 = np.asarray(stat["state_99"], dtype=np.float32)[None, :]
    norm_states = normalize_bound(states, state_01, state_99)

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
    print("[ctrl-world] autoregressive prediction...")
    pred_latents = autoregressive_predict(model, norm_states, stacked_gt_latents, instruction, args)
    pred_videos = decode_stacked_latents(
        model,
        pred_latents,
        num_cams=args.num_cams,
        height=args.height,
        width=args.width,
        chunk_size=args.chunk_frames,
    )

    current_idx = args.start_index + args.num_history
    gt_frames = frames_by_t[current_idx : current_idx + pred_latents.shape[0]]
    gt_videos = [
        resize_uint8_video([frames[cam_id] for frames in gt_frames], (args.height, args.width))
        for cam_id in range(args.num_cams)
    ]

    metrics = compute_metrics(gt_videos, pred_videos)
    comparison = make_comparison_video(gt_videos, pred_videos)

    video_paths: Dict[str, str] = {}
    comparison_path = out_dir / "autoregressive_comparison_gt_pred_diff.mp4"
    write_video(comparison_path, comparison, fps=args.fps)
    video_paths["comparison"] = str(comparison_path)
    for cam_id, (gt, pred) in enumerate(zip(gt_videos, pred_videos)):
        gt_path = out_dir / f"cam{cam_id}_gt.mp4"
        pred_path = out_dir / f"cam{cam_id}_pred_autoregressive.mp4"
        write_video(gt_path, gt, fps=args.fps)
        write_video(pred_path, pred, fps=args.fps)
        video_paths[f"cam{cam_id}_gt"] = str(gt_path)
        video_paths[f"cam{cam_id}_pred"] = str(pred_path)

    summary = AutoregressiveSummary(
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
        total_pred_frames=int(pred_latents.shape[0]),
        stride=args.stride,
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
