#!/usr/bin/env python3
"""Measure DIAMOND autoregressive LPIPS degradation from horizon 2 through H.

Each rollout starts from four real frames.  Predicted frames are fed back into
the observation history, while future actions continue to come from the real
episode.  This isolates visual compounding error from policy/action drift.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from test_diamond_world_model_metrics import (
    DEFAULT_DATASET,
    DEFAULT_DIAMOND_CHECKPOINT,
    DEFAULT_FRAMEWORK_CHECKPOINT,
    batch_ssim,
    build_denoiser,
    load_action_stats,
    prepare_actions,
    prepare_video,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_DIAMOND_CHECKPOINT)
    parser.add_argument(
        "--framework-checkpoint", type=Path, default=DEFAULT_FRAMEWORK_CHECKPOINT
    )
    parser.add_argument("--agent-config", type=Path, default=Path("envs/config/agent.yaml"))
    parser.add_argument("--trainer-config", type=Path, default=Path("envs/config/trainer.yaml"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-horizon", type=int, default=16)
    parser.add_argument(
        "--rollouts-per-episode",
        type=int,
        default=1,
        help="Number of evenly spaced rollout anchors sampled from each episode.",
    )
    parser.add_argument(
        "--episode-batch-size",
        type=int,
        default=16,
        help="Episodes loaded together; effective batch also includes rollout anchors.",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--action-source",
        choices=("normalized_continuous", "continuous"),
        default="normalized_continuous",
    )
    parser.add_argument("--normalization-key", default="libero_spatial_no_noops")
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="alex")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests_dsj/diamond_multistep_lpips.json"),
    )
    return parser.parse_args()


def select_starts(
    valid_length: int,
    history: int,
    max_horizon: int,
    count: int,
) -> List[int]:
    """Select first-target indices with room for a complete H-step rollout."""
    first = history
    last = valid_length - max_horizon
    if last < first:
        return []
    available = last - first + 1
    count = min(count, available)
    if count == 1:
        return [first + (available - 1) // 2]
    return np.linspace(first, last, num=count, dtype=np.int64).tolist()


def load_rollout_batch(
    episode_paths: List[Path],
    history: int,
    args: argparse.Namespace,
    action_stats: Mapping[str, np.ndarray] | None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    List[str],
]:
    initial_histories = []
    initial_action_histories = []
    future_actions = []
    future_targets = []
    rollout_ids = []

    for episode_path in episode_paths:
        episode: Dict[str, Any] = torch.load(
            episode_path, map_location="cpu", weights_only=False
        )
        video = prepare_video(episode)
        actions = prepare_actions(episode, args.action_source, action_stats)
        valid_length = min(len(video), len(actions))
        starts = select_starts(
            valid_length,
            history,
            args.max_horizon,
            args.rollouts_per_episode,
        )
        for start in starts:
            initial_histories.append(video[start - history : start])
            # Three actions connect the four initial history frames.
            initial_action_histories.append(actions[start - history + 1 : start])
            future_actions.append(actions[start : start + args.max_horizon])
            future_targets.append(video[start : start + args.max_horizon])
            rollout_ids.append(f"{episode_path.name}:target={start}")

    if not initial_histories:
        raise ValueError("当前 episode batch 中没有足够长的 H-step 轨迹")
    return (
        torch.stack(initial_histories),
        torch.stack(initial_action_histories),
        torch.stack(future_actions),
        torch.stack(future_targets),
        rollout_ids,
    )


def clips_to_fvd_numpy(
    clips: torch.Tensor,
    horizon: int,
    temporal_length: int = 16,
) -> np.ndarray:
    """Convert short clips to I3D input, temporally resampling to 16 frames."""
    clips = clips[:, :horizon].permute(0, 2, 1, 3, 4).float()
    clips = F.interpolate(
        clips,
        size=(temporal_length, 224, 224),
        mode="trilinear",
        align_corners=False,
    )
    return (
        clips.clamp(-1.0, 1.0)
        .add(1.0)
        .mul(127.5)
        .round()
        .byte()
        .permute(0, 2, 3, 4, 1)
        .contiguous()
        .numpy()
    )


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    if args.max_horizon < 2:
        raise ValueError("--max-horizon 必须至少为 2")
    if args.rollouts_per_episode <= 0 or args.episode_batch_size <= 0:
        raise ValueError("rollouts-per-episode 和 episode-batch-size 必须大于 0")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但 torch.cuda.is_available() 为 False")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    _, sampler, history, state_source = build_denoiser(
        args.checkpoint, args.agent_config, args.trainer_config, device
    )
    perceptual_model = lpips.LPIPS(net=args.lpips_net).to(device).eval()
    action_stats = (
        load_action_stats(args.framework_checkpoint, args.normalization_key)
        if args.action_source == "normalized_continuous"
        else None
    )

    episode_paths = sorted(args.dataset.glob("*.pt"))
    if args.max_episodes is not None:
        episode_paths = episode_paths[: args.max_episodes]
    if not episode_paths:
        raise FileNotFoundError(f"{args.dataset} 中没有 episode .pt 文件")

    lpips_sums = torch.zeros(args.max_horizon, dtype=torch.float64)
    ssim_sums = torch.zeros(args.max_horizon, dtype=torch.float64)
    squared_error_sums = torch.zeros(args.max_horizon, dtype=torch.float64)
    pixel_counts = torch.zeros(args.max_horizon, dtype=torch.long)
    horizon_counts = torch.zeros(args.max_horizon, dtype=torch.long)
    predicted_clips = []
    target_clips = []
    rollout_count = 0
    progress = tqdm(
        range(0, len(episode_paths), args.episode_batch_size),
        desc="DIAMOND multi-step LPIPS",
        unit="batch",
    )

    with torch.inference_mode():
        for batch_start in progress:
            batch_paths = episode_paths[
                batch_start : batch_start + args.episode_batch_size
            ]
            try:
                obs_buffer, act_buffer, future_actions, targets, rollout_ids = (
                    load_rollout_batch(
                        batch_paths, history, args, action_stats
                    )
                )
            except ValueError:
                continue
            obs_buffer = obs_buffer.to(device)
            act_buffer = act_buffer.to(device)
            future_actions = future_actions.to(device)
            targets = targets.to(device)
            batch_predictions = []

            for horizon_index in range(args.max_horizon):
                step_actions = future_actions[:, horizon_index]
                model_actions = torch.cat(
                    [act_buffer, step_actions.unsqueeze(1)], dim=1
                )
                predictions, _ = sampler.sample(obs_buffer, model_actions)
                predictions = predictions.clamp(-1.0, 1.0)
                distances = perceptual_model(
                    predictions, targets[:, horizon_index], normalize=False
                ).flatten()

                lpips_sums[horizon_index] += distances.double().sum().cpu()
                predictions_01 = predictions.float().add(1.0).mul(0.5)
                targets_01 = targets[:, horizon_index].float().add(1.0).mul(0.5)
                ssim_sums[horizon_index] += (
                    batch_ssim(predictions_01, targets_01).double().sum().cpu()
                )
                squared_error_sums[horizon_index] += (
                    (predictions_01 - targets_01).square().double().sum().cpu()
                )
                pixel_counts[horizon_index] += predictions_01.numel()
                horizon_counts[horizon_index] += len(distances)
                batch_predictions.append(predictions.cpu())

                obs_buffer = torch.cat(
                    [obs_buffer[:, 1:], predictions.unsqueeze(1)], dim=1
                )
                act_buffer = model_actions[:, 1:]

            predicted_clips.append(torch.stack(batch_predictions, dim=1))
            target_clips.append(targets.cpu())
            rollout_count += len(rollout_ids)
            progress.set_postfix(rollouts=rollout_count)

    if rollout_count == 0:
        raise RuntimeError("没有生成任何 multi-step rollout")

    lpips_curve = {
        str(horizon): float(
            lpips_sums[horizon - 1] / horizon_counts[horizon - 1]
        )
        for horizon in range(2, args.max_horizon + 1)
        if horizon_counts[horizon - 1] > 0
    }
    ssim_curve = {
        str(horizon): float(
            ssim_sums[horizon - 1] / horizon_counts[horizon - 1]
        )
        for horizon in range(2, args.max_horizon + 1)
        if horizon_counts[horizon - 1] > 0
    }
    psnr_curve = {
        str(horizon): float(
            -10.0
            * torch.log10(
                (
                    squared_error_sums[horizon - 1]
                    / pixel_counts[horizon - 1]
                ).clamp_min(1e-12)
            )
        )
        for horizon in range(2, args.max_horizon + 1)
        if pixel_counts[horizon - 1] > 0
    }

    # Canonical I3D requires at least 16 temporal frames.  For short-horizon
    # curves, each prefix is linearly resampled to 16 frames before extraction.
    from cdfvd import fvd

    all_predictions = torch.cat(predicted_clips, dim=0)
    all_targets = torch.cat(target_clips, dim=0)
    fvd_evaluator = fvd.cdfvd(
        "i3d",
        n_real="full",
        n_fake="full",
        device=str(device),
        seed=args.seed,
    )
    fvd_curve = {}
    for horizon in tqdm(
        range(2, args.max_horizon + 1),
        desc="I3D FVD by horizon",
        unit="horizon",
    ):
        real_videos = clips_to_fvd_numpy(all_targets, horizon)
        fake_videos = clips_to_fvd_numpy(all_predictions, horizon)
        fvd_curve[str(horizon)] = float(
            fvd_evaluator.compute_fvd(real_videos, fake_videos)
        )
        fvd_evaluator.empty_real_stats()
        fvd_evaluator.empty_fake_stats()

    return {
        "lpips_by_horizon": lpips_curve,
        "ssim_by_horizon": ssim_curve,
        "psnr_by_horizon": psnr_curve,
        "fvd_by_horizon": fvd_curve,
        "fvd_feature_extractor": "I3D-Kinetics-400",
        "fvd_temporal_resample_frames": 16,
        "max_horizon": args.max_horizon,
        "num_rollouts": rollout_count,
        "num_episodes": len(episode_paths),
        "rollouts_per_episode": args.rollouts_per_episode,
        "num_steps_conditioning": history,
        "rollout_mode": "autoregressive_predictions_with_ground_truth_actions",
        "lpips_net": args.lpips_net,
        "action_source": args.action_source,
        "checkpoint": str(args.checkpoint),
        "framework_checkpoint": str(args.framework_checkpoint),
        "loaded_world_model_state": state_source,
        "dataset": str(args.dataset),
        "seed": args.seed,
    }


def main() -> None:
    args = parse_args()
    report = evaluate(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    print(f"报告已保存到 {args.output}")


if __name__ == "__main__":
    main()
