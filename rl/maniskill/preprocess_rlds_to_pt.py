"""
preprocess_rlds_to_pt.py

Reads the RLDS/TFDS dataset through the same pipeline as finetune_debug.py,
applies all deterministic transforms (decode, resize, normalize, tokenize),
and saves each sample as a .pt file for fast loading during training.

Image augmentation is NOT baked in — the raw (decoded+resized) PIL image is
saved so that augmentation can still be applied at training time.

Usage:
    python vla-scripts/preprocess_rlds_to_pt.py
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import time
import shutil
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from dataclasses import dataclass
from typing import Optional

import draccus
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
)
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset


@dataclass
class PreprocessConfig:
    vla_path: str = "/cpfs01/lcx_workspace/models/openvla-7b"
    data_root_dir: Path = Path("/data/disk1/lcx_stu4/rlds")
    dataset_name: str = "maniskill_pickcube"
    output_dir: Path = Path("/data/disk1/lcx_stu4/PickCube-v1/preprocessed_pt")
    shuffle_buffer_size: int = 1000
    num_images_in_input: int = 1
    use_proprio: bool = False
    image_aug: bool = False


@draccus.wrap()
def preprocess(cfg: PreprocessConfig) -> None:
    output_dir = Path(cfg.output_dir)
    if output_dir.exists():
        print(f"Output dir {output_dir} already exists, removing...")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    print("Loading processor (tokenizer + image processor)...")
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=False)

    print("Loading model config for image_sizes...")
    model_config = AutoConfig.from_pretrained(cfg.vla_path, trust_remote_code=False)
    resize_resolution = tuple(model_config.image_sizes)
    print(f"  image_sizes (resize_resolution): {resize_resolution}")

    action_tokenizer = ActionTokenizer(processor.tokenizer)

    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=(cfg.num_images_in_input > 1),
        use_proprio=cfg.use_proprio,
    )

    print("Building RLDS dataset (this reads TFRecords)...")
    t0 = time.perf_counter()
    train_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=resize_resolution,
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
    print(f"  RLDS dataset built in {time.perf_counter() - t0:.1f}s")
    print(f"  dataset_length = {len(train_dataset)}")

    stats_src = cfg.data_root_dir / cfg.dataset_name / "1.0.0"
    stats_files = list(stats_src.glob("dataset_statistics_*.json"))
    if stats_files:
        stats_data = json.loads(stats_files[0].read_text())
    else:
        stats_data = {}
    dataset_statistics = train_dataset.dataset_statistics

    # Save dataset_statistics so the training script can load them
    stats_out = output_dir / "dataset_statistics.json"

    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        raise TypeError(f"Cannot serialize {type(obj)}")

    with open(stats_out, "w") as f:
        json.dump(dataset_statistics, f, indent=2, default=_convert)
    print(f"  Saved dataset_statistics to {stats_out}")

    # Iterate through the RLDS dataset and save each sample
    # The RLDS tf.data pipeline repeats infinitely, so we cap at dataset_length
    max_samples = len(train_dataset)
    print(f"\nIterating RLDS dataset and saving .pt files (max {max_samples} samples)...")
    t0 = time.perf_counter()
    count = 0
    shard_size = 500
    current_shard = []
    shard_idx = 0

    for rlds_batch in train_dataset.dataset.as_numpy_iterator():
        if count >= max_samples:
            break
        # Extract raw image as uint8 numpy (before image_transform)
        img_np = rlds_batch["observation"]["image_primary"][0]  # (H, W, 3) uint8

        # Read wrist images if dual-camera
        wrist_images_np = []
        if cfg.num_images_in_input > 1:
            for k in sorted(rlds_batch["observation"].keys()):
                if "wrist" in k:
                    wrist_images_np.append(rlds_batch["observation"][k][0])
        dataset_name = rlds_batch["dataset_name"]
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        actions = rlds_batch["action"]  # (chunk_size, action_dim)
        current_action = actions[0]

        # Tokenize actions
        prompt_builder = PurePromptBuilder("openvla")
        future_actions = actions[1:]
        future_actions_string = ''.join(action_tokenizer(future_actions))
        current_action_string = action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": action_chunk_string},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = processor.tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX

        # Save the image as uint8 tensor (compact, no JPEG re-encoding)
        img_tensor = torch.from_numpy(np.ascontiguousarray(img_np))  # (H, W, 3) uint8

        sample = {
            "image": img_tensor,
            "input_ids": input_ids,
            "labels": labels,
            "actions": torch.from_numpy(np.copy(actions).astype(np.float32)),
            "dataset_name": dataset_name,
        }

        if cfg.use_proprio and "proprio" in rlds_batch["observation"]:
            sample["proprio"] = torch.from_numpy(
                np.copy(rlds_batch["observation"]["proprio"]).astype(np.float32)
            )
        if wrist_images_np:
            sample["wrist_images"] = [
                torch.from_numpy(np.ascontiguousarray(w)) for w in wrist_images_np
            ]

        current_shard.append(sample)
        count += 1

        if len(current_shard) >= shard_size:
            shard_path = output_dir / f"shard_{shard_idx:05d}.pt"
            torch.save(current_shard, shard_path)
            shard_idx += 1
            current_shard = []

        if count % 2000 == 0:
            elapsed = time.perf_counter() - t0
            rate = count / elapsed
            print(f"  {count} samples saved ({rate:.0f} samples/s)")

    # Save remaining samples
    if current_shard:
        shard_path = output_dir / f"shard_{shard_idx:05d}.pt"
        torch.save(current_shard, shard_path)
        shard_idx += 1

    elapsed = time.perf_counter() - t0
    print(f"\nDone! Saved {count} samples in {shard_idx} shards to {output_dir}")
    print(f"  Total time: {elapsed:.1f}s ({count/elapsed:.0f} samples/s)")

    # Save metadata
    meta = {
        "num_samples": count,
        "num_shards": shard_idx,
        "shard_size": shard_size,
        "dataset_name": cfg.dataset_name,
        "vla_path": cfg.vla_path,
        "resize_resolution": list(resize_resolution),
        "use_proprio": cfg.use_proprio,
        "num_images_in_input": cfg.num_images_in_input,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved to {output_dir / 'metadata.json'}")

    total_size = sum(f.stat().st_size for f in output_dir.glob("*.pt")) / 1e9
    print(f"  Total .pt size: {total_size:.2f} GB")


if __name__ == "__main__":
    preprocess()
