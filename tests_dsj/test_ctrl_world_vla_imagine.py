#!/usr/bin/env python3
"""VLA policy + Ctrl-World world model closed-loop imagination test.

Pipeline:
1. Use VLA (ActorCritic) policy to interact with LIBERO env, collect one full episode.
2. Randomly pick a starting frame from the collected trajectory.
3. From that frame, run closed-loop imagination with Ctrl-World:
   a. Decode current predicted latent → camera images.
   b. Feed images + instruction to VLA → output delta actions.
   c. Build action conditioning directly from unnormalised VLA delta actions.
   d. Use checkpoint action statistics to normalise the action conditioning.
   e. Build visual conditioning: history_latents[history_idx] + current_latent.
   f. Call CtrlWorldDiffusionPipeline to predict next chunk of frames.
   g. Push last predicted frame & action back into history buffers.
4. Compare imagined trajectory against the GT trajectory tail.

Key references:
- rollout_interact_pi.py: history buffer management, history_idx pattern.
- test_ctrl_world_wm_closed_loop_predict.py: VLA query on predicted images,
  raw VLA action conditioning.
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

from experiments.robot.openvla_utils import resize_image_for_policy  # noqa: E402
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK  # noqa: E402
from rl.actor_critic_model_discrete import ActorCritic  # noqa: E402
from rl.utils import prepare_one_obs  # noqa: E402

# Reuse helpers from existing test files
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
    collect_vla_libero_rollout,
    load_vla_actor,
)


# ================================================================
# Data structure
# ================================================================

@dataclass
class ImagineSummary:
    benchmark: str
    task_id: int
    task_name: str
    instruction: str
    initial_state_id: int
    imagine_start_index: int
    vla_pretrained_checkpoint: str
    vla_checkpoint2: Optional[str]
    vla_unnorm_key: str
    checkpoint: str
    output_dir: str
    num_history: int
    chunk_frames: int
    num_chunks: int
    stride: int
    history_idx: List[int]
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
        description="VLA policy + Ctrl-World WM closed-loop imagination from a random trajectory point."
    )

    # --- Task / environment ---
    parser.add_argument("--benchmark", type=str, default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--initial-state-id", type=int, default=0)
    parser.add_argument("--num-steps-wait", type=int, default=10)

    # --- VLA policy ---
    parser.add_argument(
        "--vla-pretrained-checkpoint",
        type=str,
        default="/mnt/data/lcx3/checkpoint/dsj/openvla-7b+libero_spatial_no_noops+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--discrete_acts--proprio_state--100000_chkpt",
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
        default="/mnt/data/lcx3/AcceRL/tests_dsj/ctrl_world_vla_imagine_eval",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--num-cams", type=int, default=2)
    parser.add_argument("--num-history", type=int, default=6)
    parser.add_argument("--chunk-frames", type=int, default=5)
    parser.add_argument("--num-chunks", type=int, default=8)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--history-idx", type=str, default="0,0,-8,-6,-4,-2")

    # --- Imagine start point ---
    parser.add_argument(
        "--imagine-start", type=int, default=-1,
        help="Fixed start index for imagination. -1 = randomly pick from the collected trajectory.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--rotate-libero-images", action="store_true")
    parser.add_argument("--keep-model-loaded-only", action="store_true")
    return parser.parse_args()


# ================================================================
# Core: action condition helpers
# ================================================================

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


def select_future_actions(actions_unnorm: np.ndarray, num_frames: int) -> np.ndarray:
    actions = np.asarray(actions_unnorm, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[-1] != ACTION_DIM:
        raise ValueError(f"Expected VLA actions shape [N,{ACTION_DIM}], got {actions.shape}")
    if actions.shape[0] == 0:
        return np.zeros((num_frames, ACTION_DIM), dtype=np.float32)

    selected = actions[:num_frames]
    if selected.shape[0] < num_frames:
        pad = np.repeat(selected[-1:], num_frames - selected.shape[0], axis=0)
        selected = np.concatenate([selected, pad], axis=0)
    return selected.astype(np.float32)


# ================================================================
# Core: query VLA on predicted images
# ================================================================

@torch.no_grad()
def query_vla_on_predicted_images(
    actor: ActorCritic,
    cfg: Any,
    images: List[np.ndarray],
    instruction: str,
    vla_dtype: torch.dtype,
    deterministic: bool,
    resize_size: int = 224,
) -> np.ndarray:
    """Query the VLA policy on WM-predicted images.

    Mirrors the training code's infer.request(inputs_t) call:
    the VLA sees the predicted observation and outputs actions.

    Args:
        images: list of num_cams uint8 arrays [H, W, C].
                images[0] = agentview, images[1] = wrist.
    Returns:
        actions_unnorm: [NUM_ACTIONS_CHUNK, 7] unnormalised actions.
    """
    agentview = images[0]
    wrist = images[1] if len(images) > 1 else agentview

    img = resize_image_for_policy(agentview, resize_size)
    wrist_img = resize_image_for_policy(wrist, resize_size)

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
    return actions_unnorm


# ================================================================
# Core: decode latent to images
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
    Returns:
        List of num_cams uint8 arrays, each [H, W, C].
    """
    videos = decode_stacked_latents(
        model,
        latent.unsqueeze(0),
        num_cams=num_cams,
        height=height,
        width=width,
        chunk_size=1,
    )
    return [v[0] for v in videos]


# ================================================================
# Core: closed-loop imagination
# ================================================================

@torch.no_grad()
def closed_loop_imagine(
    model,
    actor: ActorCritic,
    vla_cfg: Any,
    vla_dtype: torch.dtype,
    stacked_gt_latents: torch.Tensor,
    action_01: np.ndarray,
    action_99: np.ndarray,
    instruction: str,
    args: argparse.Namespace,
    history_idx: Sequence[int],
    imagine_start: int,
) -> Tuple[torch.Tensor, List[int]]:
    """Closed-loop imagination from imagine_start.

    Flow per chunk:
    1. Decode current latent → images.
    2. Query VLA on images → unnormalised delta actions.
    3. Build action_cond: history_actions[history_idx] + future VLA actions.
    4. Build history latent: history_latents[history_idx].
    5. Call CtrlWorldDiffusionPipeline → predict next chunk.
    6. Push predicted frames before the next current frame into history buffers.
    7. Use the last predicted frame as the next current frame.

    Returns:
        pred_latents: [total_pred_frames, 4, latent_h*num_cams, latent_w]
        output_frame_indices: list of relative frame indices.
    """
    base = imagine_start
    first_latent = stacked_gt_latents[base: base + 1]     # [1, 4, h, w]

    # Initialize history buffers (same pattern as rollout_interact_pi.py)
    history_buffer_len = args.num_history * 4
    history_latents: List[torch.Tensor] = [first_latent] * history_buffer_len
    history_actions: Optional[List[np.ndarray]] = None
    current_latent = first_latent[0]

    pred_segments: List[torch.Tensor] = []
    output_frame_indices: List[int] = []
    rollover_idx = args.chunk_frames - 1

    for chunk_id in range(args.num_chunks):
        start_id = chunk_id * args.stride

        # ---- 1. Decode current latent to images ----
        images = decode_latent_to_images(
            model, current_latent,
            num_cams=args.num_cams,
            height=args.height,
            width=args.width,
        )

        # ---- 2. Query VLA on predicted images ----
        actions_unnorm = query_vla_on_predicted_images(
            actor=actor,
            cfg=vla_cfg,
            images=images,
            instruction=instruction,
            vla_dtype=vla_dtype,
            deterministic=args.vla_deterministic,
        )

        # ---- 3. Use raw VLA actions directly as future WM condition ----
        predicted_future_actions = select_future_actions(
            actions_unnorm=actions_unnorm,
            num_frames=args.chunk_frames,
        )
        if history_actions is None:
            history_actions = [predicted_future_actions[:1]] * history_buffer_len

        # ---- 4. Build action conditioning (history + predicted future) ----
        assert history_actions is not None
        history_part = np.concatenate(
            [history_actions[idx] for idx in history_idx], axis=0
        )
        raw_action_cond = np.concatenate(
            [history_part, predicted_future_actions], axis=0
        ).astype(np.float32)

        expected = len(history_idx) + args.chunk_frames
        if raw_action_cond.shape != (expected, 7):
            raise ValueError(
                f"Expected action_cond shape {(expected, 7)}, "
                f"got {raw_action_cond.shape}"
            )

        # ---- 5. Build history latent input ----
        history = torch.cat(
            [history_latents[idx] for idx in history_idx], dim=0
        ).unsqueeze(0)

        print(
            f"[imagine] chunk {chunk_id + 1}/{args.num_chunks}: "
            f"VLA → {len(actions_unnorm)} actions, "
            f"using first {args.chunk_frames} raw actions as condition, "
            f"history_idx={list(history_idx)}"
        )

        # ---- 6. Predict next chunk with Ctrl-World ----
        pred_chunk = predict_chunk_latents(
            model=model,
            current_latent=current_latent.unsqueeze(0),   # [1, 4, h, w]
            history=history,
            raw_action_cond=raw_action_cond,
            state_01=action_01,
            state_99=action_99,
            instruction=instruction,
            args=args,
        )
        # pred_chunk: [chunk_frames, 4, h, w]

        # ---- 7. Collect output ----
        take = args.chunk_frames if chunk_id == args.num_chunks - 1 else args.stride
        pred_segments.append(pred_chunk[:take])
        output_frame_indices.extend(range(start_id, start_id + take))

        # ---- 8. Update history buffers and next current latent ----
        for rel_idx in range(rollover_idx):
            history_latents.append(pred_chunk[rel_idx: rel_idx + 1])
            history_actions.append(predicted_future_actions[rel_idx: rel_idx + 1])
        current_latent = pred_chunk[rollover_idx]

    return torch.cat(pred_segments, dim=0), output_frame_indices


# ================================================================
# Main
# ================================================================

def main() -> None:
    args = parse_args()
    history_idx = parse_history_idx(args.history_idx)
    if len(history_idx) != args.num_history:
        raise ValueError(
            f"--history-idx length {len(history_idx)} must equal "
            f"--num-history {args.num_history}"
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

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    total_pred_frames = args.stride * (args.num_chunks - 1) + args.chunk_frames
    # We need enough GT frames for: a reasonable trajectory + imagination tail
    min_rollout_frames = total_pred_frames + 20  # extra margin for random start
    out_dir = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    task, task_suite = get_task(args.benchmark, args.task_id)
    instruction = task.language
    print(f"[task] {args.benchmark} task_id={args.task_id}: {task.name}")
    print(f"[task] instruction: {instruction}")

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

    # ---- Collect real episode using VLA ----
    print("[libero] rolling out with VLA actions (collecting GT trajectory)...")
    frames_by_t, _raw_states, executed_actions, rollout_done, rollout_success = (
        collect_vla_libero_rollout(
            args=args,
            task=task,
            task_suite=task_suite,
            instruction=instruction,
            actor=actor,
            vla_cfg=vla_cfg,
            vla_dtype=vla_dtype,
            total_frames=min_rollout_frames,
        )
    )
    print(
        f"[libero] collected {len(frames_by_t)} frames, "
        f"executed_actions={len(executed_actions)}, "
        f"done={rollout_done}, success={rollout_success}"
    )

    # Save executed VLA actions
    actions_path = out_dir / "vla_executed_env_actions.npy"
    np.save(
        actions_path,
        np.stack(executed_actions, axis=0)
        if executed_actions
        else np.zeros((0, ACTION_DIM), dtype=np.float32),
    )

    # ---- Pick imagination start point ----
    max_start = len(frames_by_t) - total_pred_frames
    if max_start < 1:
        max_start = 1
    if args.imagine_start >= 0:
        imagine_start = min(args.imagine_start, max_start)
    else:
        imagine_start = np.random.randint(0, max_start)
    print(
        f"[imagine] starting imagination from frame {imagine_start} "
        f"(trajectory length={len(frames_by_t)}, pred_frames={total_pred_frames})"
    )

    # ---- Load normalization stats ----
    action_01, action_99 = load_action_condition_stats(
        args.condition_stat_path,
        args.ckpt_path,
    )

    # ---- Encode GT frames to latents ----
    dtype = torch_dtype(args.dtype)
    print("[ctrl-world] encoding GT frames to latents...")
    stacked_gt_latents = encode_views_to_stacked_latents(
        ctrl_world,
        frames_by_t,
        height=args.height,
        width=args.width,
        dtype=dtype,
        device=args.device,
    )

    # ---- Closed-loop imagination ----
    print(
        "[imagine] closed-loop imagination: "
        "VLA(predicted image) → raw action condition → Ctrl-World → next frame → ..."
    )
    pred_latents, relative_gt_indices = closed_loop_imagine(
        model=ctrl_world,
        actor=actor,
        vla_cfg=vla_cfg,
        vla_dtype=vla_dtype,
        stacked_gt_latents=stacked_gt_latents,
        action_01=action_01,
        action_99=action_99,
        instruction=instruction,
        args=args,
        history_idx=history_idx,
        imagine_start=imagine_start,
    )

    # ---- Decode predicted latents to videos ----
    pred_videos = decode_stacked_latents(
        ctrl_world,
        pred_latents,
        num_cams=args.num_cams,
        height=args.height,
        width=args.width,
        chunk_size=args.chunk_frames,
    )

    # ---- Build GT videos for comparison ----
    gt_indices = [imagine_start + idx for idx in relative_gt_indices]
    gt_frames = [frames_by_t[min(idx, len(frames_by_t) - 1)] for idx in gt_indices]
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
    comparison_path = out_dir / "imagine_comparison_gt_pred_diff.mp4"
    write_video(comparison_path, comparison, fps=args.fps)
    video_paths["comparison"] = str(comparison_path)
    for cam_id, (gt, pred) in enumerate(zip(gt_videos, pred_videos)):
        gt_path = out_dir / f"cam{cam_id}_gt.mp4"
        pred_path = out_dir / f"cam{cam_id}_imagine_pred.mp4"
        write_video(gt_path, gt, fps=args.fps)
        write_video(pred_path, pred, fps=args.fps)
        video_paths[f"cam{cam_id}_gt"] = str(gt_path)
        video_paths[f"cam{cam_id}_pred"] = str(pred_path)
    video_paths["vla_env_actions_npy"] = str(actions_path)

    # ---- Summary ----
    summary = ImagineSummary(
        benchmark=args.benchmark,
        task_id=args.task_id,
        task_name=task.name,
        instruction=instruction,
        initial_state_id=args.initial_state_id,
        imagine_start_index=imagine_start,
        vla_pretrained_checkpoint=args.vla_pretrained_checkpoint,
        vla_checkpoint2=args.vla_checkpoint2 if not args.no_vla_checkpoint2 else None,
        vla_unnorm_key=vla_cfg.unnorm_key,
        checkpoint=args.ckpt_path,
        output_dir=str(out_dir),
        num_history=args.num_history,
        chunk_frames=args.chunk_frames,
        num_chunks=args.num_chunks,
        stride=args.stride,
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
