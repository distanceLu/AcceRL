#!/usr/bin/env python3
"""Evaluate the DIAMOND denoiser on generated episode data.

The evaluation is teacher-forced: four real frames and the four corresponding
actions are used to predict the next frame.  Only the DIAMOND/denoiser state
dict is loaded; the OpenVLA policy is never instantiated.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf
from safetensors import safe_open
from tqdm import tqdm

from envs.diffusion import Denoiser, DiffusionSampler


DEFAULT_DATASET = Path("tests_dsj/dataset_episode_task0")
DEFAULT_FRAMEWORK_CHECKPOINT = Path(
    "/mnt/data/lcx3/checkpoint/dsj/"
    "openvla-7b+libero_spatial_no_noops+b32+lr-0.0005+lora-r32+dropout-0.0"
    "--image_aug--parallel_dec--8_acts_chunk--discrete_acts--proprio_state--100000_chkpt"
)
DEFAULT_DIAMOND_CHECKPOINT = Path(
    "/mnt/data/lcx2/yanjieworkspace/openvla_oft_rl/runs/"
    "wm_reward_denoiser_distill_named/denoiser_smoke_test/denoiser_smoke_test.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_DIAMOND_CHECKPOINT,
        help=(
            "A standalone DIAMOND .pt checkpoint, or a framework checkpoint "
            "file/directory containing denoiser-prefixed weights."
        ),
    )
    parser.add_argument(
        "--framework-checkpoint",
        type=Path,
        default=DEFAULT_FRAMEWORK_CHECKPOINT,
        help="OpenVLA checkpoint containing dataset_statistics.json for action normalization.",
    )
    parser.add_argument("--agent-config", type=Path, default=Path("envs/config/agent.yaml"))
    parser.add_argument("--trainer-config", type=Path, default=Path("envs/config/trainer.yaml"))
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--max-predictions-per-episode",
        type=int,
        default=None,
        help="Limit predicted frames per episode; useful for a quick smoke test.",
    )
    parser.add_argument(
        "--action-source",
        choices=("normalized_continuous", "continuous"),
        default="normalized_continuous",
        help="DIAMOND in run_oft_diamond.sh is conditioned on float actions.",
    )
    parser.add_argument(
        "--normalization-key",
        default="libero_spatial_no_noops",
        help="dataset_statistics.json key used to recover normalized policy actions.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _is_tensor_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value) and all(
        isinstance(item, torch.Tensor) for item in value.values()
    )


def _tensor_mappings(value: Any, name: str = "root") -> Iterable[Tuple[str, Mapping[str, torch.Tensor]]]:
    if not isinstance(value, Mapping):
        return
    if _is_tensor_mapping(value):
        yield name, value
    for key, child in value.items():
        if isinstance(child, Mapping):
            yield from _tensor_mappings(child, f"{name}.{key}")


def _strip_denoiser_prefix(key: str) -> str | None:
    """Return a key relative to Denoiser, ignoring unrelated model weights."""
    key = key.removeprefix("module.").removeprefix("_forward_module.")
    if key.startswith("inner_model."):
        return key
    markers = (
        "world_model.denoiser.",
        "world_model.",
        "diamond.denoiser.",
        "diamond.",
        "denoiser.",
    )
    for marker in markers:
        position = key.find(marker)
        if position >= 0:
            relative = key[position + len(marker) :]
            if relative.startswith("inner_model."):
                return relative
    return None


def _extract_denoiser_state(raw: Any) -> Tuple[Dict[str, torch.Tensor], str]:
    best_state: Dict[str, torch.Tensor] = {}
    best_name = ""
    for name, candidate in _tensor_mappings(raw):
        extracted = {}
        for key, value in candidate.items():
            relative = _strip_denoiser_prefix(str(key))
            if relative is not None:
                extracted[relative] = value
        if len(extracted) > len(best_state):
            best_state, best_name = extracted, name
    if not best_state:
        raise ValueError("checkpoint 中没有发现 DIAMOND denoiser/inner_model 权重")
    return best_state, best_name


def _load_indexed_safetensors(directory: Path) -> Tuple[Dict[str, torch.Tensor], str]:
    index_path = directory / "model.safetensors.index.json"
    if not index_path.is_file():
        raise ValueError(f"{directory} 中没有 model.safetensors.index.json")
    with index_path.open("r", encoding="utf-8") as file:
        weight_map = json.load(file).get("weight_map", {})

    selected: Dict[str, Tuple[str, str]] = {}
    for full_key, shard_name in weight_map.items():
        relative = _strip_denoiser_prefix(full_key)
        if relative is not None:
            selected[full_key] = (relative, shard_name)
    if not selected:
        raise ValueError(
            f"{index_path} 不包含 world_model/diamond/denoiser/inner_model 权重键；"
            "该目录看起来是纯 OpenVLA checkpoint"
        )

    state: Dict[str, torch.Tensor] = {}
    by_shard: Dict[str, list[Tuple[str, str]]] = {}
    for full_key, (relative, shard_name) in selected.items():
        by_shard.setdefault(shard_name, []).append((full_key, relative))
    for shard_name, keys in by_shard.items():
        with safe_open(directory / shard_name, framework="pt", device="cpu") as file:
            for full_key, relative in keys:
                state[relative] = file.get_tensor(full_key)
    return state, str(index_path)


def load_denoiser_state(checkpoint: Path) -> Tuple[Dict[str, torch.Tensor], str]:
    if checkpoint.is_file():
        raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state, location = _extract_denoiser_state(raw)
        return state, f"{checkpoint}:{location}"
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint}")

    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        return _load_indexed_safetensors(checkpoint)

    candidates = sorted(
        path
        for pattern in ("*.pt", "*.pth", "*.bin")
        for path in checkpoint.glob(pattern)
        if path.is_file()
    )
    errors = []
    for path in candidates:
        try:
            raw = torch.load(path, map_location="cpu", weights_only=False)
            state, location = _extract_denoiser_state(raw)
            return state, f"{path}:{location}"
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"{path.name}: {error}")
    detail = "; ".join(errors) if errors else "没有可加载的 .pt/.pth/.bin 文件"
    raise ValueError(f"无法从目录 {checkpoint} 提取 DIAMOND 权重: {detail}")


def build_denoiser(
    checkpoint: Path,
    agent_config: Path,
    trainer_config: Path,
    device: torch.device,
) -> Tuple[Denoiser, DiffusionSampler, int, str]:
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    agent_cfg = OmegaConf.load(agent_config)
    trainer_cfg = OmegaConf.load(trainer_config)
    denoiser_cfg = instantiate(agent_cfg.denoiser)
    if denoiser_cfg.inner_model.num_actions is None:
        denoiser_cfg.inner_model.num_actions = 256

    state, state_source = load_denoiser_state(checkpoint)
    model = Denoiser(denoiser_cfg).to(device)

    float_weight_key = "inner_model.act_emb_float.0.weight"
    if float_weight_key not in state:
        raise RuntimeError(
            "DIAMOND checkpoint 缺少 float action embedding；"
            "run_oft_diamond.sh 的世界模型需要 float normalized actions"
        )
    model.inner_model._get_act_emb_float(state[float_weight_key].shape[1])

    long_weight_key = "inner_model.act_emb_long.0.weight"
    if long_weight_key in state:
        num_actions, embedding_dim = state[long_weight_key].shape
        current = model.inner_model.act_emb_long[0]
        if (current.num_embeddings, current.embedding_dim) != (num_actions, embedding_dim):
            model.inner_model.act_emb_long[0] = nn.Embedding(num_actions, embedding_dim).to(device)
            model.inner_model.cfg.num_actions = num_actions

    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"DIAMOND checkpoint 存在未识别权重: {incompatible.unexpected_keys}")
    missing_trainable = [
        key
        for key in incompatible.missing_keys
        if not key.startswith("inner_model.act_emb_long.")
        and not key.startswith("inner_model.act_emb_float.")
    ]
    if missing_trainable:
        raise RuntimeError(f"DIAMOND checkpoint 缺少模型权重: {missing_trainable}")

    model.eval()
    sampler_cfg = instantiate(trainer_cfg.world_model_env.diffusion_sampler)
    sampler = DiffusionSampler(model, sampler_cfg)
    return model, sampler, int(denoiser_cfg.inner_model.num_steps_conditioning), state_source


def load_action_stats(framework_checkpoint: Path, key: str) -> Dict[str, np.ndarray]:
    stats_path = framework_checkpoint / "dataset_statistics.json"
    if not stats_path.is_file():
        raise FileNotFoundError(f"action normalization statistics 不存在: {stats_path}")
    with stats_path.open("r", encoding="utf-8") as file:
        all_stats = json.load(file)
    if key not in all_stats:
        raise KeyError(f"{stats_path} 中没有 normalization key {key!r}")
    stats = all_stats[key]["action"]
    return {name: np.asarray(stats[name]) for name in ("q01", "q99", "mask")}


def prepare_actions(
    episode: Mapping[str, Any],
    source: str,
    action_stats: Mapping[str, np.ndarray] | None,
) -> torch.Tensor:
    actions = np.asarray(episode["actions_continuous"], dtype=np.float32)
    if source == "normalized_continuous":
        assert action_stats is not None
        low = action_stats["q01"].astype(np.float32)
        high = action_stats["q99"].astype(np.float32)
        mask = action_stats["mask"].astype(bool)
        normalized = np.where(
            mask,
            np.clip(2.0 * (actions - low) / (high - low + 1e-8) - 1.0, -1.0, 1.0),
            actions,
        )
        normalized[:, np.isclose(high, low)] = 0.0
        actions = normalized.astype(np.float32)
    return torch.from_numpy(actions)


def prepare_video(episode: Mapping[str, Any]) -> torch.Tensor:
    video = np.asarray(episode["video"])
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError(f"video 应为 [T,H,W,3]，实际为 {video.shape}")
    return torch.from_numpy(video).permute(0, 3, 1, 2).float().div(127.5).sub(1.0)


def batch_ssim(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Return standard Gaussian-window SSIM for each RGB image in [0, 1]."""
    channels = predictions.shape[1]
    coordinates = torch.arange(
        window_size, device=predictions.device, dtype=predictions.dtype
    ) - (window_size - 1) / 2
    kernel_1d = torch.exp(-(coordinates.square()) / (2 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    window = kernel_2d.expand(channels, 1, window_size, window_size)

    mu_pred = F.conv2d(predictions, window, groups=channels)
    mu_target = F.conv2d(targets, window, groups=channels)
    mu_pred_sq = mu_pred.square()
    mu_target_sq = mu_target.square()
    mu_product = mu_pred * mu_target

    variance_pred = F.conv2d(predictions.square(), window, groups=channels) - mu_pred_sq
    variance_target = F.conv2d(targets.square(), window, groups=channels) - mu_target_sq
    covariance = F.conv2d(predictions * targets, window, groups=channels) - mu_product

    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2 * mu_product + c1) * (2 * covariance + c2)) / (
        (mu_pred_sq + mu_target_sq + c1)
        * (variance_pred + variance_target + c2)
    )
    return ssim_map.mean(dim=(1, 2, 3))


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    if args.batch_size <= 0:
        raise ValueError("--batch-size 必须大于 0")
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
    action_stats = (
        load_action_stats(args.framework_checkpoint, args.normalization_key)
        if args.action_source == "normalized_continuous"
        else None
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    episode_paths = sorted(args.dataset.glob("*.pt"))
    if args.max_episodes is not None:
        episode_paths = episode_paths[: args.max_episodes]
    if not episode_paths:
        raise FileNotFoundError(f"{args.dataset} 中没有 episode .pt 文件")

    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    ssim_sum = 0.0
    pixel_count = 0
    prediction_count = 0
    evaluated_episodes = 0
    sample_time_seconds = 0.0

    progress = tqdm(episode_paths, desc="DIAMOND evaluation", unit="episode")
    with torch.inference_mode():
        for episode_path in progress:
            episode = torch.load(episode_path, map_location="cpu", weights_only=False)
            video = prepare_video(episode)
            actions = prepare_actions(episode, args.action_source, action_stats)
            valid = min(len(video), len(actions))
            target_indices = list(range(history, valid))
            if args.max_predictions_per_episode is not None:
                target_indices = target_indices[: args.max_predictions_per_episode]
            if not target_indices:
                continue

            for start in range(0, len(target_indices), args.batch_size):
                indices = target_indices[start : start + args.batch_size]
                previous_obs = torch.stack([video[index - history : index] for index in indices])
                previous_act = torch.stack(
                    [actions[index - history + 1 : index + 1] for index in indices]
                )
                targets = torch.stack([video[index] for index in indices]).to(device)
                previous_obs = previous_obs.to(device, non_blocking=True)
                previous_act = previous_act.to(device, non_blocking=True)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                sample_start = time.perf_counter()
                predictions, _ = sampler.sample(
                    previous_obs,
                    previous_act,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                sample_time_seconds += time.perf_counter() - sample_start
                predictions = predictions.clamp(-1.0, 1.0)

                # Metrics are computed in RGB [0, 1].
                predictions_01 = predictions.float().add(1.0).mul(0.5)
                targets_01 = targets.float().add(1.0).mul(0.5)
                error = predictions_01 - targets_01
                squared_error_sum += error.square().sum().item()
                absolute_error_sum += error.abs().sum().item()
                ssim_sum += batch_ssim(predictions_01, targets_01).sum().item()
                pixel_count += error.numel()
                prediction_count += len(indices)

            evaluated_episodes += 1
            running_mse = squared_error_sum / pixel_count
            running_sps = (
                prediction_count / sample_time_seconds
                if sample_time_seconds > 0
                else 0.0
            )
            progress.set_postfix(
                mse=f"{running_mse:.6f}",
                imagined_sps=f"{running_sps:.2f}",
            )

    if pixel_count == 0:
        raise RuntimeError("没有可评估的预测帧")
    mse = squared_error_sum / pixel_count
    mae = absolute_error_sum / pixel_count
    psnr = -10.0 * np.log10(max(mse, 1e-12))
    ssim = ssim_sum / prediction_count
    ms_per_imagined_step = sample_time_seconds * 1000.0 / prediction_count
    imagined_sps = (
        prediction_count / sample_time_seconds if sample_time_seconds > 0 else 0.0
    )
    gpu_mem_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else None
    )
    return {
        "mse": float(mse),
        "mae": float(mae),
        "psnr": float(psnr),
        "ssim": float(ssim),
        "ms_per_imagined_step": float(ms_per_imagined_step),
        "imagined_sps": float(imagined_sps),
        "gpu_mem_mb": float(gpu_mem_mb) if gpu_mem_mb is not None else None,
        "num_episodes": evaluated_episodes,
        "num_predictions": prediction_count,
        "num_steps_conditioning": history,
        "action_source": args.action_source,
        "checkpoint": str(args.checkpoint),
        "framework_checkpoint": str(args.framework_checkpoint),
        "loaded_world_model_state": state_source,
        "dataset": str(args.dataset),
        "seed": args.seed,
    }


def main() -> None:
    args = parse_args()
    metrics = evaluate(args)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2, ensure_ascii=False)
        print(f"指标已保存到 {args.output}")


if __name__ == "__main__":
    main()
