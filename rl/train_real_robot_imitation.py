"""Image-only imitation learning for the three-camera real-robot trajectories.

Only ``camera_paper_aruco``, ``camera_pool`` and ``camera_pool1`` are loaded as
model observations.  Robot command state is used offline to construct the
six-axis supervision target; it is never returned as a model input.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

CAMERA_DIRS = ("camera_paper_aruco", "camera_pool", "camera_pool1")
ACTION_NAMES = ("dx_m", "dy_m", "dz_m", "drot_x_rad", "drot_y_rad", "drot_z_rad")
# Stable Llama/OpenVLA token constants.  Defining them here avoids importing the
# RLDS/TensorFlow stack during data-only validation.
IGNORE_INDEX = -100
ACTION_TOKEN_BEGIN_IDX = 31743
STOP_INDEX = 2
STATE_COLUMNS = (
    "command_tcp_x",
    "command_tcp_y",
    "command_tcp_z",
    "command_tcp_rx",
    "command_tcp_ry",
    "command_tcp_rz",
)


@dataclass(frozen=True)
class AlignedFrame:
    session: str
    timestamp_us: int
    image_paths: tuple[str, str, str]
    action: tuple[float, float, float, float, float, float]
    camera_delta_us: tuple[int, int]
    state_delta_us: int


@dataclass(frozen=True)
class TrainingSample:
    frame: AlignedFrame
    action_chunk: np.ndarray


def _timestamp_from_image(path: Path) -> int:
    try:
        return int(path.name.split(".", 1)[0])
    except ValueError as exc:
        raise ValueError(f"Image filename does not start with an integer timestamp: {path}") from exc


def _sorted_images(directory: Path) -> list[tuple[int, Path]]:
    paths: list[Path] = []
    for suffix in ("*.jpg", "*.jpeg", "*.png"):
        paths.extend(directory.glob(suffix))
    return sorted((_timestamp_from_image(path), path) for path in paths)


def _nearest_index(sorted_values: Sequence[int], target: int) -> int:
    if not sorted_values:
        raise ValueError("Cannot search an empty timestamp sequence.")
    pos = bisect.bisect_left(sorted_values, target)
    if pos == 0:
        return 0
    if pos == len(sorted_values):
        return len(sorted_values) - 1
    return pos if abs(sorted_values[pos] - target) < abs(sorted_values[pos - 1] - target) else pos - 1


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    """Rodrigues exponential map for an axis-angle rotation vector."""
    rotvec = np.asarray(rotvec, dtype=np.float64)
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64) + _skew(rotvec)
    axis_skew = _skew(rotvec / theta)
    return np.eye(3) + math.sin(theta) * axis_skew + (1.0 - math.cos(theta)) * (axis_skew @ axis_skew)


def matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    """SO(3) logarithm map, including stable small-angle and pi branches."""
    matrix = np.asarray(matrix, dtype=np.float64)
    cos_theta = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    theta = math.acos(cos_theta)
    vee = np.array(
        [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]],
        dtype=np.float64,
    )
    if theta < 1e-8:
        return 0.5 * vee
    if math.pi - theta < 1e-5:
        diagonal = np.maximum((np.diag(matrix) + 1.0) * 0.5, 0.0)
        axis = np.sqrt(diagonal)
        largest = int(np.argmax(axis))
        if axis[largest] < 1e-8:
            return np.zeros(3, dtype=np.float64)
        if largest == 0:
            axis[1] = math.copysign(axis[1], matrix[0, 1] + matrix[1, 0])
            axis[2] = math.copysign(axis[2], matrix[0, 2] + matrix[2, 0])
        elif largest == 1:
            axis[0] = math.copysign(axis[0], matrix[0, 1] + matrix[1, 0])
            axis[2] = math.copysign(axis[2], matrix[1, 2] + matrix[2, 1])
        else:
            axis[0] = math.copysign(axis[0], matrix[0, 2] + matrix[2, 0])
            axis[1] = math.copysign(axis[1], matrix[1, 2] + matrix[2, 1])
        axis /= np.linalg.norm(axis)
        return theta * axis
    return theta * vee / (2.0 * math.sin(theta))


def pose_delta(start_pose: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
    """Return a base-frame SE(3) delta between two commanded TCP poses.

    Translation is expressed in the robot base frame.  Rotation satisfies
    ``R_target = Exp(drot) @ R_start`` and therefore is not obtained through
    component-wise subtraction of the two absolute rotation vectors.  The
    caller first aligns commanded poses to image times and then takes this
    delta, as required by ``session_meta.json``.
    """
    translation = target_pose[:3] - start_pose[:3]
    start_rotation = rotvec_to_matrix(start_pose[3:])
    target_rotation = rotvec_to_matrix(target_pose[3:])
    rotation = matrix_to_rotvec(target_rotation @ start_rotation.T)
    return np.concatenate((translation, rotation)).astype(np.float64)


def _load_command_states(csv_path: Path) -> tuple[list[int], np.ndarray]:
    rows: list[tuple[int, np.ndarray]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = set(("timestamp", *STATE_COLUMNS)) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
        for row in reader:
            if row.get("status_valid", "1") not in ("1", "1.0", "True", "true"):
                continue
            values = np.asarray([float(row[name]) for name in STATE_COLUMNS], dtype=np.float64)
            if np.all(np.isfinite(values)):
                rows.append((int(row["timestamp"]), values))
    rows.sort(key=lambda item: item[0])
    if not rows:
        return [], np.empty((0, len(STATE_COLUMNS)), dtype=np.float64)
    return [item[0] for item in rows], np.stack([item[1] for item in rows])


def discover_aligned_frames(
    data_root: Path,
    max_camera_delta_us: int,
    max_state_delta_us: int,
) -> tuple[dict[str, list[AlignedFrame]], list[dict[str, str]]]:
    """Align the three allowed camera streams and command state by timestamp."""
    sessions: dict[str, list[AlignedFrame]] = {}
    skipped: list[dict[str, str]] = []
    candidate_sessions = sorted(path.parent for path in data_root.glob("*/*/session_meta.json"))
    if not candidate_sessions:
        candidate_sessions = sorted(path for path in data_root.glob("*/*") if path.is_dir())

    for session_dir in candidate_sessions:
        session_name = str(session_dir.relative_to(data_root))
        state_path = session_dir / "robot_state" / "robot_command_state.csv"
        missing_dirs = [name for name in CAMERA_DIRS if not (session_dir / name).is_dir()]
        if missing_dirs:
            skipped.append({"session": session_name, "reason": f"missing camera dirs: {missing_dirs}"})
            continue
        if not state_path.is_file():
            skipped.append({"session": session_name, "reason": "missing robot_command_state.csv"})
            continue

        camera_images = {name: _sorted_images(session_dir / name) for name in CAMERA_DIRS}
        if any(not camera_images[name] for name in CAMERA_DIRS):
            skipped.append({"session": session_name, "reason": "one or more allowed camera streams are empty"})
            continue
        state_timestamps, state_values = _load_command_states(state_path)
        if not state_timestamps:
            skipped.append({"session": session_name, "reason": "no valid command-state rows"})
            continue

        pool_timestamps = [item[0] for item in camera_images["camera_pool"]]
        pool1_timestamps = [item[0] for item in camera_images["camera_pool1"]]
        raw_aligned: list[tuple[int, tuple[str, str, str], tuple[int, int], int, np.ndarray]] = []
        for paper_timestamp, paper_path in camera_images["camera_paper_aruco"]:
            pool_idx = _nearest_index(pool_timestamps, paper_timestamp)
            pool1_idx = _nearest_index(pool1_timestamps, paper_timestamp)
            state_idx = _nearest_index(state_timestamps, paper_timestamp)
            pool_delta = abs(pool_timestamps[pool_idx] - paper_timestamp)
            pool1_delta = abs(pool1_timestamps[pool1_idx] - paper_timestamp)
            state_delta = abs(state_timestamps[state_idx] - paper_timestamp)
            if max(pool_delta, pool1_delta) > max_camera_delta_us or state_delta > max_state_delta_us:
                continue

            raw_aligned.append(
                (
                    paper_timestamp,
                    (
                        str(paper_path),
                        str(camera_images["camera_pool"][pool_idx][1]),
                        str(camera_images["camera_pool1"][pool1_idx][1]),
                    ),
                    (pool_delta, pool1_delta),
                    state_delta,
                    state_values[state_idx],
                )
            )

        # Supervise the command transition after each observation.  The last
        # aligned frame has no subsequent command and is intentionally omitted.
        aligned: list[AlignedFrame] = []
        for current, following in zip(raw_aligned, raw_aligned[1:]):
            paper_timestamp, image_paths, camera_deltas, state_delta, commanded_pose = current
            action = pose_delta(commanded_pose, following[-1])
            aligned.append(
                AlignedFrame(
                    session=session_name,
                    timestamp_us=paper_timestamp,
                    image_paths=image_paths,
                    action=tuple(float(value) for value in action),
                    camera_delta_us=camera_deltas,
                    state_delta_us=state_delta,
                )
            )
        if aligned:
            sessions[session_name] = aligned
        else:
            skipped.append({"session": session_name, "reason": "no frames passed timestamp alignment limits"})

    if not sessions:
        raise RuntimeError(f"No trainable sessions found below {data_root}.")
    return sessions, skipped


def make_training_samples(
    sessions: dict[str, list[AlignedFrame]], num_actions_chunk: int
) -> list[TrainingSample]:
    samples: list[TrainingSample] = []
    for frames in sessions.values():
        for start, frame in enumerate(frames):
            chunk = np.stack(
                [np.asarray(frames[min(start + offset, len(frames) - 1)].action) for offset in range(num_actions_chunk)]
            ).astype(np.float32)
            samples.append(TrainingSample(frame=frame, action_chunk=chunk))
    return samples


def build_action_range_table(
    actions: np.ndarray,
    model_output_classes: int,
    source_sample_count: int,
) -> dict[str, Any]:
    """Create independent min/max and bin tables for all six physical dimensions."""
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != len(ACTION_NAMES):
        raise ValueError(f"Expected actions with shape (N, 6), got {actions.shape}.")
    if model_output_classes < 3:
        raise ValueError("model_output_classes must be at least 3.")

    raw_min = actions.min(axis=0)
    raw_max = actions.max(axis=0)
    low = raw_min.copy()
    high = raw_max.copy()
    # A constant dimension cannot define an invertible affine map.  Expand only
    # such dimensions by a tiny, unit-aware epsilon and record the adjustment.
    adjusted_dimensions: list[str] = []
    for dim in range(actions.shape[1]):
        if high[dim] - low[dim] < 1e-12:
            center = (high[dim] + low[dim]) * 0.5
            epsilon = 1e-6
            low[dim] = center - epsilon
            high[dim] = center + epsilon
            adjusted_dimensions.append(ACTION_NAMES[dim])

    edges = np.stack([np.linspace(low[i], high[i], model_output_classes) for i in range(actions.shape[1])])
    centers = (edges[:, :-1] + edges[:, 1:]) * 0.5
    return {
        "version": 1,
        "action_representation": "base_frame_se3_delta",
        "rotation_convention": "R_next_command = Exp(drot) @ R_current_command",
        "dimensions": list(ACTION_NAMES),
        "units": ["m", "m", "m", "rad", "rad", "rad"],
        "source_sample_count": int(source_sample_count),
        "model_output_classes": int(model_output_classes),
        "usable_interval_count": int(model_output_classes - 1),
        "low": low.tolist(),
        "high": high.tolist(),
        "raw_min": raw_min.tolist(),
        "raw_max": raw_max.tolist(),
        "mean": actions.mean(axis=0).tolist(),
        "std": actions.std(axis=0).tolist(),
        "q01": np.quantile(actions, 0.01, axis=0).tolist(),
        "q99": np.quantile(actions, 0.99, axis=0).tolist(),
        "constant_dimensions_expanded": adjusted_dimensions,
        "bin_edges": edges.tolist(),
        "bin_centers": centers.tolist(),
    }


def write_action_range_table(table: dict[str, Any], run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / "action_range_table.json"
    json_path.write_text(json.dumps(table, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    edges = np.asarray(table["bin_edges"], dtype=np.float64)
    centers = np.asarray(table["bin_centers"], dtype=np.float64)
    csv_path = run_dir / "action_range_table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["bin_index"]
        for name in ACTION_NAMES:
            fieldnames.extend((f"{name}_low", f"{name}_high", f"{name}_center"))
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for bin_index in range(centers.shape[1]):
            row: dict[str, float | int] = {"bin_index": bin_index}
            for dim, name in enumerate(ACTION_NAMES):
                row[f"{name}_low"] = float(edges[dim, bin_index])
                row[f"{name}_high"] = float(edges[dim, bin_index + 1])
                row[f"{name}_center"] = float(centers[dim, bin_index])
            writer.writerow(row)


def continuous_actions_to_class_ids(
    actions: np.ndarray, table: dict[str, Any]
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float64)
    low = np.asarray(table["low"], dtype=np.float64)
    high = np.asarray(table["high"], dtype=np.float64)
    normalized = np.clip(2.0 * (actions - low) / (high - low) - 1.0, -1.0, 1.0)
    model_output_classes = int(table["model_output_classes"])
    normalized_edges = np.linspace(-1.0, 1.0, model_output_classes)
    bin_indices = np.digitize(normalized, normalized_edges) - 1
    bin_indices = np.clip(bin_indices, 0, model_output_classes - 2)
    return (model_output_classes - 1 - bin_indices).astype(np.int64)


def _center_crop(image: Image.Image, crop_scale: float = 0.9) -> Image.Image:
    width, height = image.size
    cropped_width, cropped_height = int(width * crop_scale), int(height * crop_scale)
    left = (width - cropped_width) // 2
    top = (height - cropped_height) // 2
    return image.crop((left, top, left + cropped_width, top + cropped_height))


class ThreeCameraImitationDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[TrainingSample],
        processor: Any,
        action_range_table: dict[str, Any],
        task_label: str,
        center_crop: bool,
    ) -> None:
        self.samples = list(samples)
        self.processor = processor
        self.action_range_table = action_range_table
        self.prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
        self.center_crop = center_crop
        self.action_token_count = self.samples[0].action_chunk.size

    def __len__(self) -> int:
        return len(self.samples)

    def _prepare_language(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if input_ids[-1].item() != 29871:
            input_ids = torch.cat((input_ids, input_ids.new_tensor([29871])))
            attention_mask = torch.cat((attention_mask, attention_mask.new_ones(1)))
        labels = input_ids.new_full(input_ids.shape, IGNORE_INDEX)
        placeholders = input_ids.new_ones(self.action_token_count)
        input_ids = torch.cat((input_ids, placeholders, input_ids.new_tensor([STOP_INDEX])))
        attention_mask = torch.cat((attention_mask, attention_mask.new_ones(self.action_token_count + 1)))
        action_markers = labels.new_full((self.action_token_count,), ACTION_TOKEN_BEGIN_IDX + 1)
        labels = torch.cat((labels, action_markers, labels.new_tensor([STOP_INDEX])))
        return input_ids, attention_mask, labels

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        images = []
        for image_path in sample.frame.image_paths:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                images.append(_center_crop(image) if self.center_crop else image.copy())

        processed = [self.processor(self.prompt, image) for image in images]
        pixel_values = torch.cat([item["pixel_values"] for item in processed], dim=1).squeeze(0)
        input_ids, attention_mask, labels = self._prepare_language(
            processed[0]["input_ids"].squeeze(0), processed[0]["attention_mask"].squeeze(0)
        )
        class_ids = continuous_actions_to_class_ids(sample.action_chunk, self.action_range_table)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "action_class_ids": torch.from_numpy(class_ids.reshape(-1)),
        }


def make_collate_fn(pad_token_id: int):
    def collate(batch: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_length = max(item["input_ids"].numel() for item in batch)
        input_ids, attention_masks, labels = [], [], []
        for item in batch:
            pad = max_length - item["input_ids"].numel()
            input_ids.append(F.pad(item["input_ids"], (0, pad), value=pad_token_id))
            attention_masks.append(F.pad(item["attention_mask"], (0, pad), value=0))
            labels.append(F.pad(item["labels"], (0, pad), value=IGNORE_INDEX))
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(labels),
            "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
            "action_class_ids": torch.stack([item["action_class_ids"] for item in batch]),
        }

    return collate


def _checkpoint_action_bins(pretrained_checkpoint: Path) -> int:
    config_path = pretrained_checkpoint / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing model config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return int(config.get("n_action_bins", 256))


def _write_manifest(
    run_dir: Path,
    data_root: Path,
    sessions: dict[str, list[AlignedFrame]],
    skipped: list[dict[str, str]],
    args: argparse.Namespace,
) -> None:
    manifest = {
        "data_root": str(data_root),
        "model_input_camera_dirs": list(CAMERA_DIRS),
        "excluded_camera_dirs": ["camera_3d_2d"],
        "use_proprio": False,
        "action_label_source": (
            "successive image-aligned command_tcp poses from robot_state/robot_command_state.csv "
            "(offline supervision only)"
        ),
        "session_frame_counts": {name: len(frames) for name, frames in sessions.items()},
        "aligned_frame_count": sum(len(frames) for frames in sessions.values()),
        "skipped_sessions": skipped,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (run_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Three-camera image-only real-robot imitation learning")
    parser.add_argument("--data-root", type=Path, default=Path("data_collect"))
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs/real_robot_imitation"))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--task-label", default="brush the paper surface")
    parser.add_argument("--num-actions-chunk", type=int, default=8)
    parser.add_argument("--max-camera-delta-us", type=int, default=150_000)
    parser.add_argument("--max-state-delta-us", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dataloader-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--save-freq", type=int, default=1_000)
    parser.add_argument("--log-freq", type=int, default=10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-center-crop", action="store_true")
    parser.add_argument("--use-tensorboard", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _disable_tensorflow_gpu() -> None:
    """Keep TensorFlow (imported transitively by OpenVLA) off training GPUs."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tensorflow as tf
    except ImportError:
        return
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError as exc:
        raise RuntimeError("TensorFlow initialized a GPU before it could be disabled.") from exc


def train(args: argparse.Namespace) -> Path:
    data_root = args.data_root.resolve()
    pretrained_checkpoint = args.pretrained_checkpoint.resolve()
    if args.num_actions_chunk <= 0:
        raise ValueError("--num-actions-chunk must be positive.")
    if args.batch_size <= 0 or args.grad_accumulation_steps <= 0 or args.max_steps <= 0:
        raise ValueError("Batch size, gradient accumulation steps, and max steps must be positive.")
    if args.dataloader_workers < 0 or args.log_freq <= 0:
        raise ValueError("Dataloader workers must be nonnegative and log frequency must be positive.")

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S_three_camera_image_only")
    run_dir = (args.output_root / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)

    sessions, skipped = discover_aligned_frames(
        data_root=data_root,
        max_camera_delta_us=args.max_camera_delta_us,
        max_state_delta_us=args.max_state_delta_us,
    )
    samples = make_training_samples(sessions, args.num_actions_chunk)
    all_actions = np.concatenate(
        [np.asarray([frame.action for frame in frames], dtype=np.float64) for frames in sessions.values()], axis=0
    )
    action_range_table = build_action_range_table(
        all_actions,
        model_output_classes=_checkpoint_action_bins(pretrained_checkpoint),
        source_sample_count=len(all_actions),
    )
    write_action_range_table(action_range_table, run_dir)
    _write_manifest(run_dir, data_root, sessions, skipped, args)

    print(f"Run directory: {run_dir}")
    print(f"Aligned sessions: {len(sessions)}, samples: {len(samples)}")
    print(f"Action range JSON: {run_dir / 'action_range_table.json'}")
    if args.prepare_only:
        print("Preparation-only mode: model was not loaded and no training was run.")
        return run_dir

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
        lora_rank=args.lora_rank,
        lora_dropout=args.lora_dropout,
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
        action_dim=len(ACTION_NAMES),
        num_actions_chunk=args.num_actions_chunk,
    )

    # Delayed import keeps --prepare-only lightweight and independently testable.
    from rl.actor_critic_model_discrete import ActorCritic

    model = ActorCritic(cfg, torch_dtype)
    model.set_action_range_table(action_range_table)
    if model.vla.vision_backbone.get_num_images_in_input() != 3:
        raise AssertionError("Vision backbone was not configured for exactly three images.")

    dataset = ThreeCameraImitationDataset(
        samples=samples,
        processor=model.processor,
        action_range_table=action_range_table,
        task_label=args.task_label,
        center_crop=not args.no_center_crop,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.dataloader_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.dataloader_workers > 0),
        collate_fn=make_collate_fn(int(model.vla.pad_token_id)),
    )

    policy_parameters = [parameter for parameter in model.vla.parameters() if parameter.requires_grad]
    if not policy_parameters:
        raise RuntimeError("No trainable policy parameters were found.")
    optimizer = AdamW(policy_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    metrics_path = run_dir / "train_metrics.jsonl"
    tensorboard_writer = None
    if args.use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        tensorboard_dir = run_dir / "tensorboard"
        tensorboard_writer = SummaryWriter(log_dir=str(tensorboard_dir), flush_secs=10)
        tensorboard_writer.add_text("experiment/run_directory", str(run_dir), 0)
        print(f"TensorBoard log directory: {tensorboard_dir}")

    raw_action_span = np.asarray(action_range_table["raw_max"]) - np.asarray(action_range_table["raw_min"])
    active_dimension_indices = np.flatnonzero(raw_action_span >= 1e-12).tolist()
    optimizer_step = 0
    accumulated_micro_steps = 0
    last_saved_step = -1

    while optimizer_step < args.max_steps:
        for batch in loader:
            model_inputs = {
                key: batch[key].to(device, non_blocking=True)
                for key in ("input_ids", "attention_mask", "labels", "pixel_values")
            }
            # Enforce the image-only contract at the actual forward boundary.
            if set(model_inputs) != {"input_ids", "attention_mask", "labels", "pixel_values"}:
                raise AssertionError("Unexpected model input key detected.")
            targets = batch["action_class_ids"].to(device, non_blocking=True)
            action_logits, _ = model(model_inputs)
            if action_logits.shape[:2] != targets.shape:
                raise RuntimeError(f"Logit/target shape mismatch: {action_logits.shape} versus {targets.shape}")
            loss = F.cross_entropy(action_logits.reshape(-1, model.n_action_bins), targets.reshape(-1))
            (loss / args.grad_accumulation_steps).backward()
            accumulated_micro_steps += 1

            if accumulated_micro_steps < args.grad_accumulation_steps:
                continue
            torch.nn.utils.clip_grad_norm_(policy_parameters, args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accumulated_micro_steps = 0
            optimizer_step += 1

            with torch.no_grad():
                predicted_class_ids = action_logits.argmax(dim=-1)
                correct = (predicted_class_ids == targets).reshape(
                    targets.shape[0], model.num_actions_chunk, model.action_dim
                )
                accuracy = correct.float().mean().item()
                per_dimension_accuracy = correct.float().mean(dim=(0, 1)).cpu().numpy()
                if active_dimension_indices:
                    active_accuracy = correct[:, :, active_dimension_indices].float().mean().item()
                else:
                    active_accuracy = accuracy
                complete_action_accuracy = correct.all(dim=-1).float().mean().item()

                predicted_actions = model.token_ids_to_continuous_actions(
                    predicted_class_ids.detach()
                    .cpu()
                    .numpy()
                    .reshape(targets.shape[0], model.num_actions_chunk, model.action_dim)
                )
                target_actions = model.token_ids_to_continuous_actions(
                    targets.detach()
                    .cpu()
                    .numpy()
                    .reshape(targets.shape[0], model.num_actions_chunk, model.action_dim)
                )
                per_dimension_mae = np.mean(np.abs(predicted_actions - target_actions), axis=(0, 1))
            record = {
                "step": optimizer_step,
                "loss": float(loss.detach().cpu()),
                "token_accuracy": accuracy,
                "active_token_accuracy": active_accuracy,
                "complete_action_accuracy": complete_action_accuracy,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            for dimension_index, dimension_name in enumerate(ACTION_NAMES):
                record[f"{dimension_name}_token_accuracy"] = float(per_dimension_accuracy[dimension_index])
                record[f"{dimension_name}_mae"] = float(per_dimension_mae[dimension_index])
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar("train/loss", record["loss"], optimizer_step)
                tensorboard_writer.add_scalar("accuracy/all_tokens", accuracy, optimizer_step)
                tensorboard_writer.add_scalar("accuracy/active_dimensions", active_accuracy, optimizer_step)
                tensorboard_writer.add_scalar("accuracy/complete_6d_action", complete_action_accuracy, optimizer_step)
                tensorboard_writer.add_scalar("train/learning_rate", record["learning_rate"], optimizer_step)
                for dimension_name in ACTION_NAMES:
                    tensorboard_writer.add_scalar(
                        f"accuracy_by_dimension/{dimension_name}",
                        record[f"{dimension_name}_token_accuracy"],
                        optimizer_step,
                    )
                    tensorboard_writer.add_scalar(
                        f"mae_by_dimension/{dimension_name}",
                        record[f"{dimension_name}_mae"],
                        optimizer_step,
                    )
            if optimizer_step == 1 or optimizer_step % args.log_freq == 0:
                print(
                    f"step={optimizer_step}/{args.max_steps} "
                    f"loss={record['loss']:.6f} token_accuracy={accuracy:.4f} "
                    f"active_accuracy={active_accuracy:.4f}",
                    flush=True,
                )
            if args.save_freq > 0 and optimizer_step % args.save_freq == 0:
                model.save_model(run_dir / "checkpoints", epoch=optimizer_step)
                last_saved_step = optimizer_step
            if optimizer_step >= args.max_steps:
                break

    if last_saved_step != optimizer_step:
        model.save_model(run_dir / "checkpoints", epoch=optimizer_step)
    if tensorboard_writer is not None:
        tensorboard_writer.close()
    print(f"Training complete: {run_dir}")
    return run_dir


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
