#!/usr/bin/env python3
"""Evaluate a pretrained Ctrl-World checkpoint with the online-RL metric protocol.

The evaluator uses real AcceRL episodes from one LIBERO task.  It mirrors
``CtrlWorldInferenceActor.evaluate_prediction_metrics``:

* fixed, episode-balanced windows (eight uniformly spaced windows/episode),
* six history frames plus the current frame,
* one predicted chunk evaluated at horizons h1 through h4,
* both agent and wrist cameras,
* deterministic VAE encoding and deterministic per-window diffusion seeds,
* per-image MSE/PSNR/SSIM/LPIPS followed by sample/horizon averaging.

LIBERO task numbering is zero based.  ``--task-id 0`` is the first task.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import lpips
import numpy as np
import torch
import torch.nn.functional as F

# Direct execution sets sys.path[0] to tests_dsj rather than the repository.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ctrl_world.config import wm_args
from ctrl_world.scripts.train_wm_accerl_pt import AcceRLCtrlWorld
from prismatic.vla.constants import ACTION_DIM
from tests_dsj.ctrl_world_env_batch import CtrlWorldEnvBatch


DEFAULT_DATASET = Path("tests_dsj/dataset_episode")
DEFAULT_CHECKPOINT = Path(
    "/mnt/data/lcx3/Ctrl-World/model_ckpt/libero_spatial/"
    "2026-08-07T10-39-47_libero_spatial_newdecoder_fromlastbest/"
    "best_val_loss.pt"
)
DEFAULT_CONDITION_STATS = Path(
    "/mnt/data/lcx3/Ctrl-World/model_ckpt/libero_spatial/"
    "2026-07-21T16-40-56_libero_vla_delta_finetune/condition_stat.json"
)
DEFAULT_SVD_MODEL = Path("/mnt/data/lcx3/checkpoint/ctrl_world/svd/svd_model")
DEFAULT_CLIP_MODEL = Path("/mnt/data/lcx3/checkpoint/ctrl_world/clip/clip_model")


def batch_ssim(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """RL-identical Gaussian-window SSIM for RGB images in [0, 1]."""
    channels = predictions.shape[1]
    coordinates = torch.arange(
        window_size, device=predictions.device, dtype=predictions.dtype
    ) - (window_size - 1) / 2
    kernel_1d = torch.exp(-(coordinates.square()) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    window = kernel_2d.expand(channels, 1, window_size, window_size)
    mu_pred = F.conv2d(predictions, window, groups=channels)
    mu_target = F.conv2d(targets, window, groups=channels)
    mu_pred_sq = mu_pred.square()
    mu_target_sq = mu_target.square()
    mu_product = mu_pred * mu_target
    variance_pred = (
        F.conv2d(predictions.square(), window, groups=channels) - mu_pred_sq
    )
    variance_target = (
        F.conv2d(targets.square(), window, groups=channels) - mu_target_sq
    )
    covariance = (
        F.conv2d(predictions * targets, window, groups=channels) - mu_product
    )
    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2 * mu_product + c1) * (2 * covariance + c2)) / (
        (mu_pred_sq + mu_target_sq + c1)
        * (variance_pred + variance_target + c2)
    )
    return ssim_map.mean(dim=(1, 2, 3))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--condition-stat-path", type=Path, default=DEFAULT_CONDITION_STATS
    )
    parser.add_argument("--svd-model-path", type=Path, default=DEFAULT_SVD_MODEL)
    parser.add_argument("--clip-model-path", type=Path, default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-windows", type=int, default=64)
    parser.add_argument("--windows-per-episode", type=int, default=8)
    parser.add_argument("--max-horizon", type=int, default=4)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="alex")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests_dsj/ctrl_world_pretrained_task0_rl_metrics.json"),
    )
    return parser.parse_args()


def select_window_starts(
    episode_length: int,
    window_length: int,
    windows_per_episode: int,
) -> List[int]:
    """Match the RL holdout's linspace selection over all valid windows."""
    num_valid = episode_length - window_length + 1
    if num_valid <= 0:
        return []
    keep = min(int(windows_per_episode), num_valid)
    return np.linspace(0, num_valid - 1, num=keep, dtype=np.int64).tolist()


def load_task_windows(
    dataset: Path,
    task_id: int,
    num_history: int,
    num_frames: int,
    num_windows: int,
    windows_per_episode: int,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict[str, Any]]]:
    """Load fixed task windows as uint8 pixels plus frame-aligned raw actions."""
    episode_paths = sorted(dataset.glob(f"task{task_id}_ep*.pt"))
    if not episode_paths:
        raise FileNotFoundError(f"No task{task_id}_ep*.pt episodes under {dataset}")

    window_length = int(num_history + num_frames)
    observations: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    instructions: List[str] = []
    selected: List[Dict[str, Any]] = []

    for episode_path in episode_paths:
        episode = torch.load(episode_path, map_location="cpu", weights_only=False)
        required = ("video", "wrist_video", "actions_continuous", "mask")
        missing = [key for key in required if key not in episode]
        if missing:
            raise KeyError(f"{episode_path} is missing keys: {missing}")

        mask = np.asarray(episode["mask"]).astype(bool)
        valid = np.flatnonzero(mask)
        agent = np.asarray(episode["video"])[valid]
        wrist = np.asarray(episode["wrist_video"])[valid]
        action = np.asarray(episode["actions_continuous"], dtype=np.float32)[valid]
        valid_length = min(len(agent), len(wrist), len(action))
        agent = agent[:valid_length]
        wrist = wrist[:valid_length]
        action = action[:valid_length]

        for start in select_window_starts(
            valid_length, window_length, windows_per_episode
        ):
            end = start + window_length
            # [F,M,C,H,W], retained as uint8 until each GPU micro-batch.
            views = np.stack([agent[start:end], wrist[start:end]], axis=1)
            views = views.transpose(0, 1, 4, 2, 3)
            observations.append(np.ascontiguousarray(views, dtype=np.uint8))
            actions.append(np.ascontiguousarray(action[start:end], dtype=np.float32))
            instructions.append(str(episode.get("instruction", "")))
            selected.append(
                {
                    "episode": episode_path.name,
                    "start": int(start),
                    "end_exclusive": int(end),
                }
            )
            if len(observations) >= num_windows:
                return (
                    np.stack(observations),
                    np.stack(actions),
                    instructions,
                    selected,
                )

    if len(observations) < num_windows:
        raise ValueError(
            f"Task {task_id} supplied only {len(observations)} selected windows; "
            f"{num_windows} requested"
        )
    raise AssertionError("unreachable")


def load_model_and_env(
    args: argparse.Namespace, device: torch.device
) -> Tuple[AcceRLCtrlWorld, CtrlWorldEnvBatch]:
    cfg = wm_args()
    cfg.svd_model_path = str(args.svd_model_path)
    cfg.clip_model_path = str(args.clip_model_path)
    cfg.num_cams = 2
    cfg.num_history = 6
    cfg.num_frames = 5
    cfg.action_dim = ACTION_DIM
    cfg.text_cond = True
    cfg.frame_level_cond = True
    cfg.his_cond_zero = False

    model = AcceRLCtrlWorld(cfg).to(device).to(torch.bfloat16).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)

    with args.condition_stat_path.open("r", encoding="utf-8") as file:
        stats = json.load(file)
    condition_low = torch.tensor(stats["condition_p01"], dtype=torch.float32)
    condition_high = torch.tensor(stats["condition_p99"], dtype=torch.float32)

    @dataclass
    class EvalConfig:
        horizon: int = 220

    env = CtrlWorldEnvBatch(
        ctrl_world_model=model,
        cfg=EvalConfig(),
        torch_dtype=torch.bfloat16,
        num_cams=2,
        target_height=192,
        target_width=320,
        num_frames_pred=5,
        num_inference_steps=args.num_inference_steps,
        condition_low=condition_low,
        condition_high=condition_high,
    )
    return model, env


@torch.inference_mode()
def evaluate(
    env: CtrlWorldEnvBatch,
    observations_u8: np.ndarray,
    actions_np: np.ndarray,
    instructions: Sequence[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Metric aggregation identical to the RL Ctrl-World fixed evaluator."""
    device = env.device
    batch_size, num_frames, num_cams, channels, _, _ = observations_u8.shape
    context_len = env.num_history + 1
    available_future = num_frames - context_len
    horizon = min(args.max_horizon, available_future, env.num_frames_pred - 1)
    if horizon < 1:
        raise ValueError("No future frames available for evaluation")

    lpips_model = lpips.LPIPS(net=args.lpips_net).to(device).eval()
    samples = {
        name: {str(h): [] for h in range(1, horizon + 1)}
        for name in ("mse", "psnr", "ssim", "lpips")
    }
    view_samples = {
        view: {name: [] for name in ("mse", "psnr", "ssim", "lpips")}
        for view in range(num_cams)
    }

    for start in range(0, batch_size, args.micro_batch_size):
        end = min(batch_size, start + args.micro_batch_size)
        micro_size = end - start
        obs = torch.from_numpy(observations_u8[start:end]).float()
        obs = obs.div(127.5).sub(1.0)
        flat = obs.reshape(micro_size * num_frames * num_cams, channels, *obs.shape[-2:])
        flat = F.interpolate(
            flat,
            size=(env.target_height, env.target_width),
            mode="bilinear",
            align_corners=False,
        )
        obs = flat.reshape(
            micro_size,
            num_frames,
            num_cams,
            channels,
            env.target_height,
            env.target_width,
        )
        context = obs[:, :context_len].to(device)
        gt_future = obs[:, context_len : context_len + horizon].to(device)
        history, current = env.init_latent_state(context, deterministic=True)
        action_condition = torch.from_numpy(actions_np[start:end]).to(device)

        generators = []
        for sample_index in range(start, end):
            generator = torch.Generator(device=device)
            generator.manual_seed(args.seed + sample_index)
            generators.append(generator)

        future_obs, _ = env.predict_chunk_stateless(
            current_latent=current,
            latent_history=history,
            action_condition=action_condition,
            instructions=list(instructions[start:end]),
            output_size=(env.target_height, env.target_width),
            generator=generators,
        )
        pred_future = future_obs[:, :horizon].float()

        for horizon_index in range(horizon):
            pred = pred_future[:, horizon_index]
            target = gt_future[:, horizon_index].float()
            pred_flat = pred.reshape(
                micro_size * num_cams,
                channels,
                env.target_height,
                env.target_width,
            )
            target_flat = target.reshape_as(pred_flat)
            pred_01 = pred_flat.add(1.0).mul(0.5).clamp(0.0, 1.0)
            target_01 = target_flat.add(1.0).mul(0.5).clamp(0.0, 1.0)
            mse_view = (
                (pred_01 - target_01)
                .square()
                .mean(dim=(1, 2, 3))
                .reshape(micro_size, num_cams)
            )
            psnr_view = -10.0 * torch.log10(mse_view.clamp_min(1e-12))
            ssim_view = batch_ssim(pred_01, target_01).reshape(
                micro_size, num_cams
            )
            lpips_view = lpips_model(pred_flat, target_flat).reshape(
                micro_size, num_cams
            )
            key = str(horizon_index + 1)
            for name, values in (
                ("mse", mse_view),
                ("psnr", psnr_view),
                ("ssim", ssim_view),
                ("lpips", lpips_view),
            ):
                samples[name][key].extend(values.mean(dim=1).float().cpu().tolist())
                if horizon_index + 1 == horizon:
                    for view in range(num_cams):
                        view_samples[view][name].extend(
                            values[:, view].float().cpu().tolist()
                        )

    result: Dict[str, Any] = {
        "eval_horizon": horizon,
        "num_windows": batch_size,
        "eval_seed": args.seed,
    }
    for name in ("mse", "psnr", "ssim", "lpips"):
        all_values: List[float] = []
        result[f"{name}_by_horizon"] = {}
        result[f"{name}_std_by_horizon"] = {}
        for horizon_index in range(1, horizon + 1):
            key = str(horizon_index)
            values = np.asarray(samples[name][key], dtype=np.float64)
            all_values.extend(values.tolist())
            result[f"{name}_by_horizon"][key] = float(values.mean())
            result[f"{name}_std_by_horizon"][key] = float(values.std())
        values = np.asarray(all_values, dtype=np.float64)
        result[name] = float(values.mean())
        result[f"{name}_std"] = float(values.std())

    view_names = ("agent_view", "wrist_view")
    for view, metrics in view_samples.items():
        for name, values in metrics.items():
            array = np.asarray(values, dtype=np.float64)
            result[f"{view_names[view]}_{name}"] = float(array.mean())
            result[f"{view_names[view]}_{name}_std"] = float(array.std())
    return result


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    if args.num_windows < 1 or args.windows_per_episode < 1:
        raise ValueError("num-windows and windows-per-episode must be positive")

    observations, actions, instructions, selected = load_task_windows(
        dataset=args.dataset,
        task_id=args.task_id,
        num_history=6,
        num_frames=5,
        num_windows=args.num_windows,
        windows_per_episode=args.windows_per_episode,
    )
    _, env = load_model_and_env(args, device)
    metrics = evaluate(env, observations, actions, instructions, args)
    payload = {
        "protocol": "online_rl_fixed_holdout_equivalent",
        "task_id": args.task_id,
        "checkpoint": str(args.checkpoint),
        "condition_stat_path": str(args.condition_stat_path),
        "dataset": str(args.dataset),
        "num_inference_steps": args.num_inference_steps,
        "windows_per_episode": args.windows_per_episode,
        "selected_windows": selected,
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
