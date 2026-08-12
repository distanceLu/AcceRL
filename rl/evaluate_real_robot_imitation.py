"""Evaluate a three-camera real-robot imitation checkpoint against recorded actions.

The model observation contains exactly the three configured camera images plus
the fixed OpenVLA prompt tokens.  Commanded TCP poses are used only to rebuild
the continuous ground-truth action labels for offline comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from rl.train_real_robot_imitation import (
    ACTION_NAMES,
    ThreeCameraImitationDataset,
    _disable_tensorflow_gpu,
    _seed_everything,
    discover_aligned_frames,
    make_collate_fn,
    make_training_samples,
)


class EvaluationDataset(Dataset):
    """Add continuous targets and stable sample metadata to the training dataset."""

    def __init__(self, dataset: ThreeCameraImitationDataset) -> None:
        self.dataset = dataset
        self.samples = dataset.samples

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.dataset[index]
        sample = self.samples[index]
        item["continuous_target_actions"] = torch.from_numpy(sample.action_chunk.copy())
        item["sample_index"] = index
        item["session"] = sample.frame.session
        item["timestamp_us"] = sample.frame.timestamp_us
        return item


def make_evaluation_collate_fn(pad_token_id: int):
    base_collate = make_collate_fn(pad_token_id)

    def collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
        model_batch = base_collate(batch)
        model_batch["continuous_target_actions"] = torch.stack(
            [item["continuous_target_actions"] for item in batch]
        )
        model_batch["sample_indices"] = [int(item["sample_index"]) for item in batch]
        model_batch["sessions"] = [str(item["session"]) for item in batch]
        model_batch["timestamps_us"] = [int(item["timestamp_us"]) for item in batch]
        return model_batch

    return collate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a trained three-camera imitation checkpoint with recorded real-robot actions"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--agent-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--task-label", default="brush the paper surface")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dataloader-workers", type=int, default=0)
    parser.add_argument("--max-camera-delta-us", type=int, default=150_000)
    parser.add_argument("--max-state-delta-us", type=int, default=100_000)
    parser.add_argument("--max-samples", type=int, default=0, help="0 evaluates every aligned sample")
    parser.add_argument(
        "--session-regex",
        default="",
        help="Optional regular expression selecting complete sessions before samples are built",
    )
    parser.add_argument("--print-every", type=int, default=1, help="Print cumulative metrics every N batches")
    parser.add_argument(
        "--print-actions",
        action="store_true",
        help="At each print interval, show predicted, target and error values for all 8x6 actions in the batch",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-center-crop", action="store_true")
    return parser.parse_args()


def _load_checkpoint_metadata(checkpoint_dir: Path) -> dict[str, Any]:
    extra_path = checkpoint_dir / "agent_extra_layers.pt"
    lora_path = checkpoint_dir / "agent_lora" / "adapter_config.json"
    if not extra_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint metadata and extra layers: {extra_path}")
    if not lora_path.is_file():
        raise FileNotFoundError(f"Missing LoRA adapter config: {lora_path}")
    extra = torch.load(extra_path, map_location="cpu", weights_only=False)
    adapter = json.loads(lora_path.read_text(encoding="utf-8"))
    return {
        "action_dim": int(extra.get("action_dim", len(ACTION_NAMES))),
        "num_actions_chunk": int(extra.get("num_actions_chunk", 8)),
        "num_images_in_input": int(extra.get("num_images_in_input", 3)),
        "action_range_table": extra.get("action_range_table"),
        "lora_rank": int(adapter.get("r", 32)),
        "lora_dropout": float(adapter.get("lora_dropout", 0.0)),
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_ready) + "\n", encoding="utf-8")


def _format_action(action: np.ndarray) -> str:
    """Format one six-axis action using millimeters and radians."""
    action = np.asarray(action, dtype=np.float64)
    translation_mm = action[:3] * 1000.0
    rotation_rad = action[3:]
    return (
        f"dxyz_mm=({translation_mm[0]:+.4f},{translation_mm[1]:+.4f},{translation_mm[2]:+.4f}) "
        f"drot_rad=({rotation_rad[0]:+.6f},{rotation_rad[1]:+.6f},{rotation_rad[2]:+.6f})"
    )


def _summary(
    *,
    samples_seen: int,
    actions_seen: int,
    batches_seen: int,
    sum_abs_error: np.ndarray,
    sum_signed_error: np.ndarray,
    sum_squared_error: np.ndarray,
    correct_by_dimension: np.ndarray,
    total_tokens_by_dimension: np.ndarray,
    complete_actions_correct: int,
    raw_span: np.ndarray,
    active_mask: np.ndarray,
    out_of_range_counts: np.ndarray,
    checkpoint_dir: Path,
    data_root: Path,
    evaluation_scope: str,
) -> dict[str, Any]:
    denominator = max(actions_seen, 1)
    mae = sum_abs_error / denominator
    signed_bias = sum_signed_error / denominator
    rmse = np.sqrt(sum_squared_error / denominator)
    token_accuracy = np.divide(
        correct_by_dimension,
        np.maximum(total_tokens_by_dimension, 1),
        dtype=np.float64,
    )
    relative = np.full(len(ACTION_NAMES), np.nan, dtype=np.float64)
    relative[active_mask] = mae[active_mask] / raw_span[active_mask]
    active_relative = float(np.mean(relative[active_mask])) if np.any(active_mask) else None
    active_token_accuracy = (
        float(correct_by_dimension[active_mask].sum() / total_tokens_by_dimension[active_mask].sum())
        if np.any(active_mask)
        else None
    )

    dimensions: dict[str, Any] = {}
    for index, name in enumerate(ACTION_NAMES):
        dimensions[name] = {
            "active": bool(active_mask[index]),
            "range_span": float(raw_span[index]),
            "mae": float(mae[index]),
            "signed_bias": float(signed_bias[index]),
            "rmse": float(rmse[index]),
            "relative_deviation": float(relative[index]) if active_mask[index] else None,
            "relative_deviation_percent": float(relative[index] * 100.0) if active_mask[index] else None,
            "token_accuracy": float(token_accuracy[index]),
            "out_of_checkpoint_range_count": int(out_of_range_counts[index]),
        }

    return {
        "metric_definition": {
            "relative_deviation": "MAE / (checkpoint raw_max - checkpoint raw_min), per active dimension",
            "inactive_dimension": "raw range span < 1e-12; relative deviation is intentionally null",
            "mae_and_bias_units": "meters for d{x,y,z}; radians for drot_{x,y,z}",
        },
        "evaluation_scope": evaluation_scope,
        "agent_checkpoint": str(checkpoint_dir),
        "data_root": str(data_root),
        "samples_seen": int(samples_seen),
        "action_vectors_seen": int(actions_seen),
        "batches_seen": int(batches_seen),
        "active_dimension_names": [name for index, name in enumerate(ACTION_NAMES) if active_mask[index]],
        "active_mean_relative_deviation": active_relative,
        "active_mean_relative_deviation_percent": active_relative * 100.0 if active_relative is not None else None,
        "all_token_accuracy": float(correct_by_dimension.sum() / total_tokens_by_dimension.sum()),
        "active_token_accuracy": active_token_accuracy,
        "complete_6d_action_accuracy": float(complete_actions_correct / denominator),
        "dimensions": dimensions,
    }


def evaluate(args: argparse.Namespace) -> Path:
    if args.batch_size <= 0 or args.dataloader_workers < 0 or args.print_every <= 0 or args.max_samples < 0:
        raise ValueError("Batch size and print frequency must be positive; workers and max samples cannot be negative.")

    data_root = args.data_root.resolve()
    pretrained_checkpoint = args.pretrained_checkpoint.resolve()
    agent_checkpoint = args.agent_checkpoint.resolve()
    metadata = _load_checkpoint_metadata(agent_checkpoint)
    if metadata["action_dim"] != len(ACTION_NAMES):
        raise ValueError(f"Expected a six-dimensional checkpoint, got action_dim={metadata['action_dim']}.")
    if metadata["num_images_in_input"] != 3:
        raise ValueError(f"Expected a three-image checkpoint, got {metadata['num_images_in_input']} images.")
    range_table = metadata["action_range_table"]
    if range_table is None:
        raise ValueError("The checkpoint does not contain an action range table; physical deviation is undefined.")
    if tuple(range_table.get("dimensions", ())) != ACTION_NAMES:
        raise ValueError("Checkpoint range-table dimensions do not match the evaluator's action order.")

    if args.output_dir is None:
        output_dir = agent_checkpoint / "evaluations" / datetime.now().strftime("%Y%m%d_%H%M%S_relative_deviation")
    else:
        output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    sessions, skipped = discover_aligned_frames(
        data_root=data_root,
        max_camera_delta_us=args.max_camera_delta_us,
        max_state_delta_us=args.max_state_delta_us,
    )
    if args.session_regex:
        pattern = re.compile(args.session_regex)
        sessions = {name: frames for name, frames in sessions.items() if pattern.search(name)}
        if not sessions:
            raise ValueError(f"No sessions matched --session-regex={args.session_regex!r}.")
    samples = make_training_samples(sessions, metadata["num_actions_chunk"])
    if args.max_samples:
        samples = samples[: args.max_samples]
    if not samples:
        raise RuntimeError("No evaluation samples were selected.")

    _disable_tensorflow_gpu()
    _seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    cfg = SimpleNamespace(
        pretrained_checkpoint=str(pretrained_checkpoint),
        checkpoint2="",
        use_lora=True,
        lora_rank=metadata["lora_rank"],
        lora_dropout=metadata["lora_dropout"],
        use_l1_regression=False,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=3,
        use_proprio=False,
        load_in_8bit=False,
        load_in_4bit=False,
        unnorm_key="",
        center_crop=not args.no_center_crop,
        device=device,
        action_dim=metadata["action_dim"],
        num_actions_chunk=metadata["num_actions_chunk"],
    )

    from rl.actor_critic_model_discrete import ActorCritic

    model = ActorCritic(cfg, torch_dtype)
    model.safe_load_model(agent_checkpoint)
    model.eval()
    model.vla.eval()
    if model.action_range_table is None:
        raise AssertionError("safe_load_model did not restore the checkpoint action range table.")

    base_dataset = ThreeCameraImitationDataset(
        samples=samples,
        processor=model.processor,
        action_range_table=model.action_range_table,
        task_label=args.task_label,
        center_crop=not args.no_center_crop,
    )
    dataset = EvaluationDataset(base_dataset)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.dataloader_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.dataloader_workers > 0),
        collate_fn=make_evaluation_collate_fn(int(model.vla.pad_token_id)),
    )

    raw_min = np.asarray(model.action_range_table["raw_min"], dtype=np.float64)
    raw_max = np.asarray(model.action_range_table["raw_max"], dtype=np.float64)
    raw_span = raw_max - raw_min
    active_mask = raw_span >= 1e-12
    evaluation_scope = "selected sessions from recorded real-robot data; may overlap training data"
    config_record = {
        "arguments": vars(args),
        "checkpoint_metadata": metadata,
        "selected_sessions": {name: len(frames) for name, frames in sessions.items()},
        "selected_sample_count": len(samples),
        "skipped_sessions": skipped,
        "evaluation_scope_warning": evaluation_scope,
    }
    _write_json(output_dir / "evaluation_config.json", config_record)

    sum_abs_error = np.zeros(len(ACTION_NAMES), dtype=np.float64)
    sum_signed_error = np.zeros(len(ACTION_NAMES), dtype=np.float64)
    sum_squared_error = np.zeros(len(ACTION_NAMES), dtype=np.float64)
    correct_by_dimension = np.zeros(len(ACTION_NAMES), dtype=np.int64)
    total_tokens_by_dimension = np.zeros(len(ACTION_NAMES), dtype=np.int64)
    out_of_range_counts = np.zeros(len(ACTION_NAMES), dtype=np.int64)
    complete_actions_correct = 0
    samples_seen = 0
    actions_seen = 0
    batches_seen = 0

    predictions_path = output_dir / "predictions.jsonl"
    print(f"Evaluation output: {output_dir}")
    print(f"Sessions: {len(sessions)}, samples: {len(samples)}, active dimensions: "
          f"{[name for index, name in enumerate(ACTION_NAMES) if active_mask[index]]}")
    with predictions_path.open("w", encoding="utf-8") as prediction_file, torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            model_inputs = {
                key: batch[key].to(device, non_blocking=True)
                for key in ("input_ids", "attention_mask", "labels", "pixel_values")
            }
            logits, _ = model(model_inputs)
            predicted_ids = logits.argmax(dim=-1)
            target_ids = batch["action_class_ids"].to(device, non_blocking=True)
            batch_size = target_ids.shape[0]
            predicted_actions = model.token_ids_to_continuous_actions(
                predicted_ids.cpu().numpy().reshape(batch_size, model.num_actions_chunk, model.action_dim)
            ).astype(np.float64)
            target_actions = batch["continuous_target_actions"].numpy().astype(np.float64)

            error = predicted_actions - target_actions
            absolute_error = np.abs(error)
            sum_abs_error += absolute_error.sum(axis=(0, 1))
            sum_signed_error += error.sum(axis=(0, 1))
            sum_squared_error += np.square(error).sum(axis=(0, 1))
            out_of_range_counts += ((target_actions < raw_min) | (target_actions > raw_max)).sum(axis=(0, 1))

            correct = (predicted_ids == target_ids).reshape(
                batch_size, model.num_actions_chunk, model.action_dim
            )
            correct_numpy = correct.cpu().numpy()
            correct_by_dimension += correct_numpy.sum(axis=(0, 1))
            total_tokens_by_dimension += np.asarray(correct_numpy.shape[:2]).prod()
            complete_actions_correct += int(correct_numpy.all(axis=-1).sum())
            samples_seen += batch_size
            actions_seen += batch_size * model.num_actions_chunk
            batches_seen += 1

            for local_index in range(batch_size):
                sample_mae = absolute_error[local_index].mean(axis=0)
                sample_relative = np.full(len(ACTION_NAMES), np.nan, dtype=np.float64)
                sample_relative[active_mask] = sample_mae[active_mask] / raw_span[active_mask]
                row = {
                    "sample_index": batch["sample_indices"][local_index],
                    "session": batch["sessions"][local_index],
                    "timestamp_us": batch["timestamps_us"][local_index],
                    "predicted_actions": predicted_actions[local_index],
                    "target_actions": target_actions[local_index],
                    "mae_by_dimension": dict(zip(ACTION_NAMES, sample_mae.tolist())),
                    "relative_deviation_by_dimension": {
                        name: float(sample_relative[index]) if active_mask[index] else None
                        for index, name in enumerate(ACTION_NAMES)
                    },
                }
                prediction_file.write(json.dumps(row, ensure_ascii=False, default=_json_ready) + "\n")
            prediction_file.flush()

            running_summary = _summary(
                samples_seen=samples_seen,
                actions_seen=actions_seen,
                batches_seen=batches_seen,
                sum_abs_error=sum_abs_error,
                sum_signed_error=sum_signed_error,
                sum_squared_error=sum_squared_error,
                correct_by_dimension=correct_by_dimension,
                total_tokens_by_dimension=total_tokens_by_dimension,
                complete_actions_correct=complete_actions_correct,
                raw_span=raw_span,
                active_mask=active_mask,
                out_of_range_counts=out_of_range_counts,
                checkpoint_dir=agent_checkpoint,
                data_root=data_root,
                evaluation_scope=evaluation_scope,
            )
            _write_json(output_dir / "running_summary.json", running_summary)
            should_print = batch_index == 1 or batch_index % args.print_every == 0 or samples_seen == len(samples)
            if should_print:
                active_relative = running_summary["active_mean_relative_deviation_percent"]
                active_text = "n/a" if active_relative is None else f"{active_relative:.4f}%"
                first_active_name = next(
                    (name for index, name in enumerate(ACTION_NAMES) if active_mask[index]), None
                )
                dimension_text = ""
                if first_active_name is not None:
                    dimension = running_summary["dimensions"][first_active_name]
                    unit_scale = 1000.0 if first_active_name.endswith("_m") else 1.0
                    unit = "mm" if first_active_name.endswith("_m") else "rad"
                    dimension_text = (
                        f" {first_active_name}_mae={dimension['mae'] * unit_scale:.6f}{unit}"
                        f" {first_active_name}_relative={dimension['relative_deviation_percent']:.4f}%"
                    )
                print(
                    f"batch={batch_index}/{len(loader)} samples={samples_seen}/{len(samples)}"
                    f" active_relative={active_text}{dimension_text}"
                    f" active_token_accuracy={running_summary['active_token_accuracy']}",
                    flush=True,
                )
                if args.print_actions:
                    for local_index in range(batch_size):
                        print(
                            f"  action_compare sample={batch['sample_indices'][local_index]} "
                            f"session={batch['sessions'][local_index]} "
                            f"timestamp_us={batch['timestamps_us'][local_index]}",
                            flush=True,
                        )
                        for horizon in range(model.num_actions_chunk):
                            prediction = predicted_actions[local_index, horizon]
                            target = target_actions[local_index, horizon]
                            difference = prediction - target
                            print(
                                f"    horizon={horizon}: "
                                f"pred[{_format_action(prediction)}] "
                                f"true[{_format_action(target)}] "
                                f"error[{_format_action(difference)}]",
                                flush=True,
                            )

    final_summary = _summary(
        samples_seen=samples_seen,
        actions_seen=actions_seen,
        batches_seen=batches_seen,
        sum_abs_error=sum_abs_error,
        sum_signed_error=sum_signed_error,
        sum_squared_error=sum_squared_error,
        correct_by_dimension=correct_by_dimension,
        total_tokens_by_dimension=total_tokens_by_dimension,
        complete_actions_correct=complete_actions_correct,
        raw_span=raw_span,
        active_mask=active_mask,
        out_of_range_counts=out_of_range_counts,
        checkpoint_dir=agent_checkpoint,
        data_root=data_root,
        evaluation_scope=evaluation_scope,
    )
    _write_json(output_dir / "summary.json", final_summary)
    print(json.dumps(final_summary, indent=2, ensure_ascii=False, default=_json_ready), flush=True)
    print(f"Evaluation complete: {output_dir}")
    return output_dir


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
