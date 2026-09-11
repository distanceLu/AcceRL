"""Read real Brush start states through the metadata used to train Ctrl-World.

The metadata contains only synchronization and pose/action information. Images
are always read from ``real_data_root``; no simulator or Libero dataset is
involved.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageOps


CAMERA_ORDER = ("camera_pool", "camera_pool1", "camera_paper_aruco")
VLA_FROM_WM_ORDER = (2, 0, 1)
ACTION_DIM = 7
PROPRIO_DIM = 8
NUM_HISTORY = 6
NUM_FRAMES = 5
WM_HEIGHT = 192
WM_WIDTH = 320


@dataclass(frozen=True)
class RealStart:
    episode_id: str
    start_index: int
    observations: torch.Tensor  # [num_history + 1, 3, 3, H, W], [-1, 1]
    frame_actions: np.ndarray  # incoming actions for the observation window
    pose_rotvec: np.ndarray  # xyz + rotation vector
    wm_instruction: str


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def resolve_checkpoint(path: str | Path) -> Path:
    """Accept either a Ctrl-World checkpoint file or a run directory."""
    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        return candidate
    if not candidate.is_dir():
        raise FileNotFoundError(f"Ctrl-World checkpoint does not exist: {candidate}")
    numbered: list[tuple[int, Path]] = []
    for item in candidate.glob("checkpoint-*.pt"):
        match = re.fullmatch(r"checkpoint-(\d+)\.pt", item.name)
        if match:
            numbered.append((int(match.group(1)), item))
    if not numbered:
        raise FileNotFoundError(f"No checkpoint-<step>.pt found in {candidate}")
    return max(numbered, key=lambda pair: pair[0])[1].resolve()


def _load_image(path: Path, height: int, width: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = ImageOps.fit(
            image,
            (width, height),
            method=Image.Resampling.BICUBIC,
            centering=(0.5, 0.5),
        )
        array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(127.5).sub_(1.0)


class BrushRealStartDataset:
    """Random-access real start points from strict synchronized 5 Hz episodes."""

    def __init__(
        self,
        real_data_root: str | Path,
        wm_dataset_root: str | Path,
        wm_meta_root: str | Path,
        dataset_name: str,
        split: str,
        *,
        num_history: int = NUM_HISTORY,
        height: int = WM_HEIGHT,
        width: int = WM_WIDTH,
    ) -> None:
        self.real_data_root = Path(real_data_root).expanduser().resolve()
        self.wm_dataset_root = Path(wm_dataset_root).expanduser().resolve()
        self.wm_meta_root = Path(wm_meta_root).expanduser().resolve()
        self.dataset_name = dataset_name
        self.split = split
        self.num_history = int(num_history)
        self.height = int(height)
        self.width = int(width)
        if not self.real_data_root.is_dir():
            raise FileNotFoundError(f"Real data root not found: {self.real_data_root}")

        meta_dir = self.wm_meta_root / dataset_name
        summary_path = meta_dir / "conversion_summary.json"
        self.action_stat_path = meta_dir / "stat.json"
        if not summary_path.is_file() or not self.action_stat_path.is_file():
            raise FileNotFoundError(
                f"Missing Ctrl-World conversion metadata under {meta_dir}"
            )
        self.summary = read_json(summary_path)
        self._validate_contract()

        annotation_dir = (
            self.wm_dataset_root / dataset_name / "annotation" / split
        )
        if not annotation_dir.is_dir():
            raise FileNotFoundError(f"Annotation split not found: {annotation_dir}")
        # The converter contract already enforces min_segment_frames >= 11.
        # Keep actor startup cheap: index filenames here and parse only sampled
        # episodes instead of reading 1,000+ annotations in every Ray worker.
        self.annotations = sorted(annotation_dir.glob("*.json"))
        if not self.annotations:
            raise RuntimeError(f"No usable real episodes in {annotation_dir}")
        first = read_json(self.annotations[0])
        if int(first.get("video_length", 0)) <= self.num_history:
            raise ValueError(f"Real episode is too short: {self.annotations[0]}")
        self._validate_source_path(first, self.annotations[0])

    def _validate_contract(self) -> None:
        expected = {
            "target_fps": 5,
            "period_us": 200_000,
            "strict_fixed_rate": True,
            "num_history": self.num_history,
            "num_frames": NUM_FRAMES,
            "num_cams": len(CAMERA_ORDER),
            "camera_order": list(CAMERA_ORDER),
            "action_dim": ACTION_DIM,
            "action_cond_key": "actions",
            "action_alignment": "frame_aligned_incoming_delta_padding_at_0",
        }
        bad = {
            key: (self.summary.get(key), value)
            for key, value in expected.items()
            if self.summary.get(key) != value
        }
        if bad:
            raise ValueError(f"Ctrl-World real-data contract mismatch: {bad}")

    def _validate_source_path(self, annotation: dict[str, Any], path: Path) -> Path:
        source = annotation.get("source", {})
        session = Path(source.get("session_path", "")).expanduser().resolve()
        try:
            session.relative_to(self.real_data_root)
        except ValueError as exc:
            raise ValueError(
                f"Annotation {path} points outside --real-data-root: {session}"
            ) from exc
        if source.get("camera_order") != list(CAMERA_ORDER):
            raise ValueError(f"Unexpected camera order in {path}")
        return session

    def __len__(self) -> int:
        return len(self.annotations)

    def describe(self) -> dict[str, Any]:
        return {
            "real_data_root": str(self.real_data_root),
            "dataset_name": self.dataset_name,
            "split": self.split,
            "episodes": len(self.annotations),
            "camera_order": list(CAMERA_ORDER),
            "frequency_hz": 5,
        }

    def sample(self, rng: random.Random | None = None, index: int | None = None) -> RealStart:
        rng = rng or random
        annotation_path = (
            self.annotations[index % len(self.annotations)]
            if index is not None
            else rng.choice(self.annotations)
        )
        annotation = read_json(annotation_path)
        length = int(annotation["video_length"])
        if index is None:
            start = rng.randint(self.num_history, length - 1)
        else:
            span = length - self.num_history
            start = self.num_history + ((index // len(self.annotations)) % span)
        return self._materialize(annotation_path, annotation, start)

    def _materialize(
        self, annotation_path: Path, annotation: dict[str, Any], start: int
    ) -> RealStart:
        length = int(annotation["video_length"])
        session = self._validate_source_path(annotation, annotation_path)
        source = annotation["source"]
        names = source.get("camera_frame_names", {})
        indices = range(start - self.num_history, start + 1)
        frames = []
        for frame_index in indices:
            views = []
            for camera in CAMERA_ORDER:
                try:
                    image_path = session / camera / names[camera][frame_index]
                except (KeyError, IndexError) as exc:
                    raise ValueError(
                        f"Incomplete camera metadata in {annotation_path} at {frame_index}"
                    ) from exc
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                views.append(_load_image(image_path, self.height, self.width))
            frames.append(torch.stack(views))

        actions = np.asarray(annotation["actions"], dtype=np.float32)
        states = np.asarray(annotation["states"], dtype=np.float32)
        if actions.shape != (length, ACTION_DIM) or states.shape != (length, 6):
            raise ValueError(
                f"Bad state/action shape in {annotation_path}: {states.shape}, {actions.shape}"
            )
        action_window = actions[start - self.num_history : start + 1].copy()
        timestamps = source.get("target_timestamps_us", [])
        selected = timestamps[start - self.num_history : start + 1]
        if len(selected) != self.num_history + 1 or any(
            b - a != 200_000 for a, b in zip(selected, selected[1:])
        ):
            raise ValueError(f"Non-5-Hz context in {annotation_path} at index {start}")
        texts = annotation.get("texts") or [self.summary.get("instruction", "")]
        return RealStart(
            episode_id=str(annotation.get("episode_id", annotation_path.stem)),
            start_index=start,
            observations=torch.stack(frames),
            frame_actions=action_window,
            pose_rotvec=states[start].copy(),
            wm_instruction=str(texts[0]),
        )


def load_action_bounds(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    stats = read_json(Path(path))
    low_values = stats.get("state_01", stats.get("condition_p01"))
    high_values = stats.get("state_99", stats.get("condition_p99"))
    low = torch.as_tensor(low_values, dtype=torch.float32)
    high = torch.as_tensor(high_values, dtype=torch.float32)
    if low.shape != (ACTION_DIM,) or high.shape != (ACTION_DIM,):
        raise ValueError(f"Expected {ACTION_DIM}-D action bounds, got {low.shape}/{high.shape}")
    # The gripper is a fixed dummy dimension in this dataset.
    varying = high > low
    if not bool(torch.all(varying[:6])):
        raise ValueError("The six SE(3) Ctrl-World action bounds must vary")
    high = torch.where(varying, high, low + 1.0)
    return low, high


def validate_paths(paths: Iterable[str | Path]) -> None:
    missing = [str(Path(path)) for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required paths: {missing}")
