#!/usr/bin/env python3
"""VLA-driven LIBERO rollout + replay-style Ctrl-World prediction comparison.

This is the VLA-action version of
test_ctrl_world_libero_replay_style_autoregressive.py. It does not replay demo
actions from an HDF5 file. Instead, it loads an ActorCritic/OpenVLA policy,
uses that policy to act in a real LIBERO environment, records the resulting
agentview/wrist frames and eef/gripper state trajectory, then evaluates the
trained Ctrl-World checkpoint on that VLA-generated trajectory.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

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
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
)
from experiments.robot.openvla_utils import resize_image_for_policy  # noqa: E402
from experiments.robot.robot_utils import invert_gripper_action, normalize_gripper_action  # noqa: E402
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK  # noqa: E402
from rl.actor_critic_model_discrete import ActorCritic  # noqa: E402
from rl.utils import prepare_one_obs  # noqa: E402

from test_ctrl_world_libero_prediction import (  # noqa: E402
    camera_frames_from_raw_obs,
    compute_metrics,
    decode_stacked_latents,
    encode_views_to_stacked_latents,
    get_task,
    load_ctrl_world_model,
    make_comparison_video,
    resize_uint8_video,
    state7_from_raw_obs,
    torch_dtype,
    write_video,
)
from test_ctrl_world_libero_replay_style_autoregressive import (  # noqa: E402
    parse_history_idx,
    replay_style_autoregressive_predict,
)


@dataclass
class VlaReplayStyleSummary:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use a VLA policy in LIBERO, then compare Ctrl-World predictions against the VLA rollout."
    )

    parser.add_argument("--benchmark", type=str, default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--initial-state-id", type=int, default=0)
    parser.add_argument("--num-steps-wait", type=int, default=10)

    parser.add_argument(
        "--vla-pretrained-checkpoint",
        type=str,
        default="/mnt/data/lcx2/yanjieworkspace/models/finetune_im/openvla-7b+libero_spatial_no_noops+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--discrete_acts--proprio_state--100000_chkpt",
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
        default="/mnt/data/lcx3/AcceRL/tests_dsj/ctrl_world_libero_vla_replay_style_eval",
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
    parser.add_argument("--history-idx", type=str, default="0,0,-8,-6,-4,-2")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--rotate-libero-images", action="store_true")
    parser.add_argument("--keep-model-loaded-only", action="store_true")
    return parser.parse_args()


def build_vla_cfg(args: argparse.Namespace) -> GenerateConfig:
    checkpoint2 = "" if args.no_vla_checkpoint2 or not args.vla_checkpoint2 else args.vla_checkpoint2
    unnorm_key = args.vla_unnorm_key or f"{args.benchmark}_no_noops"
    return GenerateConfig(
        pretrained_checkpoint=args.vla_pretrained_checkpoint,
        checkpoint2=checkpoint2,
        use_l1_regression=False,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=args.vla_num_images_in_input,
        use_proprio=args.vla_use_proprio,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=args.vla_center_crop,
        num_open_loop_steps=args.vla_open_loop_steps,
        unnorm_key=unnorm_key,
        task_suite_name=args.benchmark,
        num_steps_wait=args.num_steps_wait,
        use_lora=True,
        lora_rank=32,
        lora_dropout=0.0,
        seed=args.seed,
    )


def resolve_vla_unnorm_key(cfg: GenerateConfig, actor: ActorCritic, benchmark_name: str) -> None:
    if cfg.unnorm_key in actor.vla.norm_stats:
        return
    alternatives = [benchmark_name, f"{benchmark_name}_no_noops"]
    for key in alternatives:
        if key in actor.vla.norm_stats:
            print(f"[vla] unnorm_key '{cfg.unnorm_key}' not found; using '{key}'")
            cfg.unnorm_key = key
            actor.cfg.unnorm_key = key
            return
    raise KeyError(f"VLA unnorm_key '{cfg.unnorm_key}' not found. Available: {list(actor.vla.norm_stats.keys())}")


def load_vla_actor(args: argparse.Namespace) -> Tuple[ActorCritic, GenerateConfig, torch.dtype]:
    vla_cfg = build_vla_cfg(args)
    vla_dtype = torch_dtype(args.vla_dtype)
    actor = ActorCritic(vla_cfg, torch_dtype=vla_dtype)
    actor.to(args.device)
    actor.eval()
    resolve_vla_unnorm_key(vla_cfg, actor, args.benchmark)
    print(f"[vla] loaded, unnorm_key={vla_cfg.unnorm_key}")
    return actor, vla_cfg, vla_dtype


def prepare_vla_observation(raw_obs: Dict[str, Any], resize_size: int) -> Dict[str, np.ndarray]:
    img = resize_image_for_policy(get_libero_image(raw_obs), resize_size)
    wrist_img = resize_image_for_policy(get_libero_wrist_image(raw_obs), resize_size)
    return {
        "full_image": img,
        "wrist_image": wrist_img,
        "state": np.concatenate(
            (
                raw_obs["robot0_eef_pos"],
                quat2axisangle(raw_obs["robot0_eef_quat"]),
                raw_obs["robot0_gripper_qpos"],
            )
        ),
    }


def process_vla_action_for_env(action: np.ndarray) -> np.ndarray:
    action = normalize_gripper_action(action.astype(np.float32).copy(), binarize=True)
    return invert_gripper_action(action).astype(np.float32)


@torch.inference_mode()
def query_vla_actions(
    actor: ActorCritic,
    cfg: GenerateConfig,
    raw_obs: Dict[str, Any],
    instruction: str,
    resize_size: int,
    vla_dtype: torch.dtype,
    deterministic: bool,
) -> List[np.ndarray]:
    observation = prepare_vla_observation(raw_obs, resize_size)
    inputs_t = prepare_one_obs(cfg, actor.processor, observation, instruction, vla_dtype)
    inputs_batch = actor.prepare_inputs_batch([inputs_t])
    action_logits, _ = actor(inputs_batch)
    _, _, normalized_actions = actor.post_process(action_logits, deterministic=[deterministic])
    actions_unnorm = actor.vla._unnormalize_actions(normalized_actions[0], cfg.unnorm_key)
    actions = [process_vla_action_for_env(action) for action in actions_unnorm]
    return actions[: cfg.num_open_loop_steps]


def get_initial_state(task_suite: Any, task_id: int, initial_state_id: int) -> Optional[np.ndarray]:
    if initial_state_id < 0:
        return None
    init_states = task_suite.get_task_init_states(task_id)
    if len(init_states) == 0:
        return None
    if initial_state_id >= len(init_states):
        raise ValueError(f"--initial-state-id {initial_state_id} out of range [0, {len(init_states) - 1}]")
    return init_states[initial_state_id]


def collect_vla_libero_rollout(
    args: argparse.Namespace,
    task: Any,
    task_suite: Any,
    instruction: str,
    actor: ActorCritic,
    vla_cfg: GenerateConfig,
    vla_dtype: torch.dtype,
    total_frames: int,
) -> Tuple[List[List[np.ndarray]], np.ndarray, List[np.ndarray], bool, bool]:
    env, _ = get_libero_env(task, vla_cfg.model_family, resolution=256)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    raw_obs = env.reset()
    initial_state = get_initial_state(task_suite, args.task_id, args.initial_state_id)
    if initial_state is not None:
        raw_obs = env.set_init_state(initial_state)

    for _ in range(args.num_steps_wait):
        raw_obs, _, _, _ = env.step(get_libero_dummy_action(vla_cfg.model_family))

    frames_by_t: List[List[np.ndarray]] = []
    states: List[np.ndarray] = []
    executed_actions: List[np.ndarray] = []
    action_queue: Deque[np.ndarray] = deque()
    resize_size = 224
    done = False
    success = False
    max_steps = max(total_frames + args.vla_open_loop_steps + args.num_steps_wait, total_frames)

    try:
        for step in range(max_steps):
            frames_by_t.append(camera_frames_from_raw_obs(raw_obs, rotate=args.rotate_libero_images))
            states.append(state7_from_raw_obs(raw_obs))
            if len(frames_by_t) >= total_frames:
                break

            if not action_queue:
                action_queue.extend(
                    query_vla_actions(
                        actor=actor,
                        cfg=vla_cfg,
                        raw_obs=raw_obs,
                        instruction=instruction,
                        resize_size=resize_size,
                        vla_dtype=vla_dtype,
                        deterministic=args.vla_deterministic,
                    )
                )
            action = action_queue.popleft()
            raw_obs, _, done, info = env.step(action.tolist())
            executed_actions.append(action)
            if done:
                success = bool(info.get("is_success", 1.0))
                print(f"[libero] environment ended at VLA step {step}; tail frames will repeat if needed.")
                break
    finally:
        env.close()

    while len(frames_by_t) < total_frames:
        frames_by_t.append([img.copy() for img in frames_by_t[-1]])
        states.append(states[-1].copy())

    return frames_by_t, np.stack(states, axis=0).astype(np.float32), executed_actions, done, success


def main() -> None:
    args = parse_args()
    history_idx = parse_history_idx(args.history_idx)
    if len(history_idx) != args.num_history:
        raise ValueError(f"--history-idx length {len(history_idx)} must equal --num-history {args.num_history}")
    if args.chunk_frames <= 0:
        raise ValueError("--chunk-frames must be positive")
    if args.stride != args.chunk_frames - 1:
        raise ValueError("--stride must equal chunk_frames - 1 to match Ctrl-World replay inference")
    if args.vla_open_loop_steps <= 0 or args.vla_open_loop_steps > NUM_ACTIONS_CHUNK:
        raise ValueError(f"--vla-open-loop-steps must be in [1, {NUM_ACTIONS_CHUNK}]")

    total_pred_frames = args.stride * (args.num_chunks - 1) + args.chunk_frames
    total_rollout_frames = args.start_index + total_pred_frames
    out_dir = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    task, task_suite = get_task(args.benchmark, args.task_id)
    instruction = task.language
    print(f"[task] {args.benchmark} task_id={args.task_id}: {task.name}")
    print(f"[task] instruction: {instruction}")
    print(
        f"[eval] VLA rollout frames={total_rollout_frames}, pred_frames={total_pred_frames}, "
        f"chunk_frames={args.chunk_frames}, stride={args.stride}, history_idx={history_idx}"
    )

    print("[vla] loading ActorCritic/OpenVLA...")
    actor, vla_cfg, vla_dtype = load_vla_actor(args)
    print("[ctrl-world] loading Ctrl-World...")
    model_args = SimpleNamespace(**vars(args))
    model_args.num_frames = args.chunk_frames
    ctrl_world = load_ctrl_world_model(model_args)
    print("[models] loaded")
    if args.keep_model_loaded_only:
        return

    print("[libero] rolling out with VLA actions...")
    frames_by_t, raw_states, executed_actions, rollout_done, rollout_success = collect_vla_libero_rollout(
        args=args,
        task=task,
        task_suite=task_suite,
        instruction=instruction,
        actor=actor,
        vla_cfg=vla_cfg,
        vla_dtype=vla_dtype,
        total_frames=total_rollout_frames,
    )
    print(
        f"[libero] collected {len(frames_by_t)} frames, states={raw_states.shape}, "
        f"executed_actions={len(executed_actions)}, done={rollout_done}, success={rollout_success}"
    )

    actions_path = out_dir / "vla_executed_env_actions.npy"
    np.save(actions_path, np.stack(executed_actions, axis=0) if executed_actions else np.zeros((0, ACTION_DIM), dtype=np.float32))

    with open(args.stat_path, "r", encoding="utf-8") as f:
        stat = json.load(f)
    state_01 = np.asarray(stat["state_01"], dtype=np.float32)[None, :]
    state_99 = np.asarray(stat["state_99"], dtype=np.float32)[None, :]

    dtype = torch_dtype(args.dtype)
    print("[ctrl-world] encoding VLA rollout GT frames...")
    stacked_gt_latents = encode_views_to_stacked_latents(
        ctrl_world,
        frames_by_t,
        height=args.height,
        width=args.width,
        dtype=dtype,
        device=args.device,
    )

    print("[ctrl-world] replay-style prediction on VLA rollout...")
    pred_latents, relative_gt_indices = replay_style_autoregressive_predict(
        model=ctrl_world,
        raw_states=raw_states,
        stacked_gt_latents=stacked_gt_latents,
        state_01=state_01,
        state_99=state_99,
        instruction=instruction,
        args=args,
        history_idx=history_idx,
    )
    pred_videos = decode_stacked_latents(
        ctrl_world,
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
    comparison_path = out_dir / "vla_replay_style_comparison_gt_pred_diff.mp4"
    write_video(comparison_path, comparison, fps=args.fps)
    video_paths["comparison"] = str(comparison_path)
    for cam_id, (gt, pred) in enumerate(zip(gt_videos, pred_videos)):
        gt_path = out_dir / f"cam{cam_id}_vla_gt.mp4"
        pred_path = out_dir / f"cam{cam_id}_ctrl_world_pred.mp4"
        write_video(gt_path, gt, fps=args.fps)
        write_video(pred_path, pred, fps=args.fps)
        video_paths[f"cam{cam_id}_gt"] = str(gt_path)
        video_paths[f"cam{cam_id}_pred"] = str(pred_path)
    video_paths["vla_env_actions_npy"] = str(actions_path)

    summary = VlaReplayStyleSummary(
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
