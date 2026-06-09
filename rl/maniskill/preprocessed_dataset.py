"""
preprocessed_dataset.py

Fast PyTorch MapDataset that loads preprocessed .pt shard files.
Replaces the slow RLDS IterableDataset pipeline with random-access
loading from local disk. Supports image augmentation at training time.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from prismatic.vla.constants import IGNORE_INDEX


class PreprocessedVLADataset(Dataset):
    """Map-style dataset that loads pre-tokenized samples from .pt shards.

    Each shard is a list of dicts with keys:
        image:       (H, W, 3) uint8 tensor
        input_ids:   (seq_len,) long tensor
        labels:      (seq_len,) long tensor
        actions:     (chunk_size, action_dim) float32 tensor
        dataset_name: str
        proprio:     (optional) float32 tensor

    The image_transform is applied on-the-fly so that image augmentation
    (random crop, color jitter, etc.) varies each epoch.
    """

    def __init__(
        self,
        preprocessed_dir: str,
        image_transform,
        dataset_statistics: Optional[Dict] = None,
    ):
        self.preprocessed_dir = Path(preprocessed_dir)
        self.image_transform = image_transform

        meta_path = self.preprocessed_dir / "metadata.json"
        with open(meta_path) as f:
            self.metadata = json.load(f)

        stats_path = self.preprocessed_dir / "dataset_statistics.json"
        if dataset_statistics is not None:
            self.dataset_statistics = dataset_statistics
        elif stats_path.exists():
            with open(stats_path) as f:
                self.dataset_statistics = json.load(f)
        else:
            self.dataset_statistics = {}

        # Load all shards into memory (dataset is ~450MB, fits easily)
        print(f"Loading preprocessed data from {self.preprocessed_dir}...")
        shard_files = sorted(self.preprocessed_dir.glob("shard_*.pt"))
        self.samples: List[Dict[str, Any]] = []
        for sf in shard_files:
            self.samples.extend(torch.load(sf, weights_only=False))
        print(f"  Loaded {len(self.samples)} samples from {len(shard_files)} shards")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # Apply image_transform on-the-fly (includes augmentation)
        img_np = sample["image"].numpy()  # (H, W, 3) uint8
        img_pil = Image.fromarray(img_np)
        pixel_values = self.image_transform(img_pil)

        result = {
            "pixel_values": pixel_values,
            "input_ids": sample["input_ids"],
            "labels": sample["labels"],
            "actions": sample["actions"],
            "dataset_name": sample["dataset_name"],
        }

        if "proprio" in sample:
            result["proprio"] = sample["proprio"]

        if "wrist_images" in sample:
            wrist_pvs = []
            for wrist_tensor in sample["wrist_images"]:
                wrist_pil = Image.fromarray(wrist_tensor.numpy())
                wrist_pvs.append(self.image_transform(wrist_pil))
            result["pixel_values_wrist"] = torch.cat(wrist_pvs, dim=0)

        return result
