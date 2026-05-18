#!/usr/bin/env python
"""Smoke test: drive OpenVLA's RLDSDataset on the newly-registered maniskill_pickcube.

Verifies end-to-end:
  - Constants autodetect MANISKILL (ACTION_DIM=7, PROPRIO_DIM=8, ...)
  - OXE_DATASET_CONFIGS / OXE_STANDARDIZATION_TRANSFORMS recognize the dataset
  - TFDS files under <data_root_dir>/maniskill_pickcube/1.0.0/ can be read
  - action chunks / proprio state / resized images come out at the right shape
  - the standardization transform converts gripper to LIBERO convention (0=close, 1=open)

Note: we deliberately keep `maniskill` in sys.argv so prismatic.vla.constants
      picks MANISKILL_CONSTANTS via detect_robot_platform().
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

# Ensure MANISKILL constants are used even if this script is launched without
# that keyword in argv.
if not any("maniskill" in a.lower() for a in sys.argv):
    sys.argv.append("--maniskill")

import numpy as np  # noqa: E402

from prismatic.vla.constants import (  # noqa: E402
    ROBOT_PLATFORM,
    NUM_ACTIONS_CHUNK,
    ACTION_DIM,
    PROPRIO_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
)
from prismatic.vla.datasets import RLDSDataset  # noqa: E402

DATA_ROOT_DIR = "/data/disk1/lcx_stu4/rlds"
DATASET_NAME = "maniskill_pickcube"


class IdentityBatchTransform:
    """A no-op batch transform so we can inspect raw RLDS output."""

    def __call__(self, rlds_batch):
        return rlds_batch


def main():
    print("=" * 70)
    print(f"ROBOT_PLATFORM                    = {ROBOT_PLATFORM}")
    print(f"NUM_ACTIONS_CHUNK                 = {NUM_ACTIONS_CHUNK}")
    print(f"ACTION_DIM                        = {ACTION_DIM}")
    print(f"PROPRIO_DIM                       = {PROPRIO_DIM}")
    print(f"ACTION_PROPRIO_NORMALIZATION_TYPE = {ACTION_PROPRIO_NORMALIZATION_TYPE}")
    assert ROBOT_PLATFORM == "MANISKILL", "Expected MANISKILL platform to be auto-detected."

    print("=" * 70)
    print(f"Instantiating RLDSDataset({DATASET_NAME})...")
    ds = RLDSDataset(
        data_root_dir=DATA_ROOT_DIR,
        data_mix=DATASET_NAME,
        batch_transform=IdentityBatchTransform(),
        resize_resolution=(224, 224),
        shuffle_buffer_size=256,
        image_aug=False,
    )

    print(f"   dataset length (approx): {ds.dataset_length}")
    stats = ds.dataset_statistics[DATASET_NAME]
    print("=" * 70)
    print(f"dataset_statistics[{DATASET_NAME}]:")
    a_stats = stats["action"]
    p_stats = stats["proprio"]
    print(f"   action.mean   = {np.asarray(a_stats['mean'])}")
    print(f"   action.q01    = {np.asarray(a_stats['q01'])}")
    print(f"   action.q99    = {np.asarray(a_stats['q99'])}")
    print(f"   proprio.mean  = {np.asarray(p_stats['mean'])}")
    print(f"   proprio.q01   = {np.asarray(p_stats['q01'])}")
    print(f"   proprio.q99   = {np.asarray(p_stats['q99'])}")

    print("=" * 70)
    print("Pulling one batch...")
    it = iter(ds)
    batch = next(it)

    print("Top-level keys:", sorted(batch.keys()))
    action = batch["action"]
    print(f"   action  shape={action.shape}  dtype={action.dtype}")
    assert action.shape[-1] == ACTION_DIM, f"expected last dim {ACTION_DIM}, got {action.shape}"
    assert action.shape[-2] == NUM_ACTIONS_CHUNK, f"expected chunk {NUM_ACTIONS_CHUNK}, got {action.shape}"

    gripper = action[..., -1]
    print(f"   gripper min/max (should be in [0, 1] after transform): "
          f"{float(gripper.min()):.4f} .. {float(gripper.max()):.4f}")
    assert gripper.min() >= -1e-6 and gripper.max() <= 1.0 + 1e-6, (
        "gripper action after transform must be in [0, 1]; got "
        f"[{float(gripper.min()):.4f}, {float(gripper.max()):.4f}]"
    )

    xyz_rpy = action[..., :6]
    print(f"   xyz+rpy min/max (should be in [-1, 1] after clip): "
          f"{float(xyz_rpy.min()):.4f} .. {float(xyz_rpy.max()):.4f}")
    assert xyz_rpy.min() >= -1 - 1e-6 and xyz_rpy.max() <= 1 + 1e-6, "xyz+rpy must be clipped."

    obs = batch["observation"]
    print("   observation keys:", sorted(obs.keys()))
    if "image_primary" in obs:
        img = obs["image_primary"]
        print(f"   image_primary shape={img.shape}  dtype={img.dtype}  "
              f"min={int(img.min())}  max={int(img.max())}")
        assert img.shape[-3:] == (224, 224, 3), f"expected 224x224x3, got {img.shape}"

    if "proprio" in obs:
        proprio = obs["proprio"]
        print(f"   proprio shape={proprio.shape}  dtype={proprio.dtype}")
        assert proprio.shape[-1] == PROPRIO_DIM, (
            f"expected proprio last dim {PROPRIO_DIM}, got {proprio.shape}"
        )

    lang = batch["task"].get("language_instruction", None) if "task" in batch else None
    if lang is not None:
        if hasattr(lang, "decode"):
            lang = lang.decode()
        elif isinstance(lang, (bytes, np.bytes_)):
            lang = lang.decode()
        print(f"   language_instruction = {lang!r}")

    print("=" * 70)
    print("SMOKE TEST PASSED.")


if __name__ == "__main__":
    main()
