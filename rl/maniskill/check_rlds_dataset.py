#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Check an RLDS / TFDS dataset for OpenVLA-OFT training.

It checks:
1. TFDS dataset can be loaded.
2. Episodes contain a `steps` dataset.
3. Each step has required fields.
4. Primary image and wrist image are 224x224x3 and not black.
5. Action dimension is 7.
6. is_first / is_last / is_terminal are reasonable.
7. language_instruction exists and is non-empty.
8. proprio exists and has valid numeric values.

Usage example:

    python scripts/check_rlds_dataset.py \
        --data-dir /data/disk1/lcx_stu4/rlds \
        --dataset-name maniskill_pickcube \
        --split train \
        --num-episodes 20

If your dataset has a version:

    python rl/maniskill/check_rlds_dataset.py \
        --data-dir /mnt/data2/lcx_stu4/maniskill/demos/PegInsertionSide-v1 \
        --dataset-name maniskill_peginsertionside \
        # --split train \
        # --num-episodes 20

Optional:
    --dump-samples /tmp/rlds_check_samples
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Important for remote servers: avoid TensorFlow grabbing GPU or triggering slow CUDA/PTX init.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TFDS_DISABLE_GCS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def to_numpy(x):
    if isinstance(x, tf.Tensor):
        return x.numpy()
    return x


def decode_str(x) -> str:
    x = to_numpy(x)
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    if isinstance(x, np.ndarray):
        if x.shape == ():
            item = x.item()
            if isinstance(item, bytes):
                return item.decode("utf-8", errors="ignore")
            return str(item)
        return str(x)
    return str(x)


def get_nested(d: Dict[str, Any], path: str):
    cur = d
    for part in path.split("/"):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def has_nested(d: Dict[str, Any], path: str) -> bool:
    return get_nested(d, path) is not None


def find_first_existing(step: Dict[str, Any], candidates: List[str]) -> Tuple[Optional[str], Any]:
    for p in candidates:
        value = get_nested(step, p)
        if value is not None:
            return p, value
    return None, None


def numeric_stats(x) -> Dict[str, float]:
    arr = np.asarray(to_numpy(x))
    return {
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
    }


def is_image_valid(img, expect_h: int, expect_w: int) -> Tuple[bool, str]:
    arr = np.asarray(to_numpy(img))

    if arr.shape != (expect_h, expect_w, 3):
        return False, f"wrong image shape: {arr.shape}, expected ({expect_h}, {expect_w}, 3)"

    if not np.issubdtype(arr.dtype, np.integer) and not np.issubdtype(arr.dtype, np.floating):
        return False, f"image dtype is not numeric: {arr.dtype}"

    if not np.all(np.isfinite(arr)):
        return False, "image contains NaN/Inf"

    stats = numeric_stats(arr)

    if np.issubdtype(arr.dtype, np.integer):
        not_black = stats["max"] > stats["min"] and stats["std"] > 1.0
    else:
        not_black = stats["max"] > stats["min"] and stats["std"] > 1e-3

    if not not_black:
        return False, f"image may be black/constant: {stats}"

    return True, f"shape={arr.shape}, dtype={arr.dtype}, stats={stats}"


def is_action_valid(action, expect_dim: int) -> Tuple[bool, str]:
    arr = np.asarray(to_numpy(action))

    if arr.ndim == 0:
        return False, f"action is scalar, shape={arr.shape}"

    if arr.shape[-1] != expect_dim:
        return False, f"wrong action dim: shape={arr.shape}, expected last dim {expect_dim}"

    if not np.all(np.isfinite(arr)):
        return False, "action contains NaN/Inf"

    stats = numeric_stats(arr)

    # Not a hard failure if action is all zero in one step, but across many steps this is suspicious.
    return True, f"shape={arr.shape}, dtype={arr.dtype}, stats={stats}"


def is_proprio_valid(proprio) -> Tuple[bool, str]:
    arr = np.asarray(to_numpy(proprio))

    if arr.ndim == 0:
        return False, f"proprio is scalar, shape={arr.shape}"

    if not np.all(np.isfinite(arr)):
        return False, "proprio contains NaN/Inf"

    stats = numeric_stats(arr)
    return True, f"shape={arr.shape}, dtype={arr.dtype}, stats={stats}"


def save_image(path: str, img):
    from PIL import Image

    arr = np.asarray(to_numpy(img))

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            if arr.max() <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

    Image.fromarray(arr).save(path)


def print_pass(name: str, msg: str = ""):
    if msg:
        print(f"✅ PASS | {name}: {msg}")
    else:
        print(f"✅ PASS | {name}")


def print_fail(name: str, msg: str = ""):
    if msg:
        print(f"❌ FAIL | {name}: {msg}")
    else:
        print(f"❌ FAIL | {name}")


def print_warn(name: str, msg: str = ""):
    if msg:
        print(f"⚠️ WARN | {name}: {msg}")
    else:
        print(f"⚠️ WARN | {name}")


def flatten_spec(spec, prefix=""):
    rows = []
    if isinstance(spec, dict):
        for k, v in spec.items():
            new_prefix = f"{prefix}/{k}" if prefix else k
            rows.extend(flatten_spec(v, new_prefix))
    else:
        rows.append((prefix, spec))
    return rows


def check_episode(
    episode,
    episode_idx: int,
    args,
    counters: Dict[str, int],
):
    print("\n" + "-" * 80)
    print(f"Episode {episode_idx}")
    print("-" * 80)

    if "steps" not in episode:
        print_fail("episode has steps", "missing `steps` field")
        counters["episodes_failed"] += 1
        return

    steps_ds = episode["steps"]

    step_count = 0
    first_flags = []
    last_flags = []
    terminal_flags = []

    primary_image_stats = []
    wrist_image_stats = []
    action_values = []

    episode_failed = False

    # Candidate field names used by common OpenVLA / RLDS converters.
    primary_image_candidates = [
        "observation/image",
        "observation/image_primary",
        "observation/base_camera/rgb",
        "observation/base_rgb",
        "image",
        "image_primary",
    ]

    wrist_image_candidates = [
        "observation/wrist_image",
        "observation/image_wrist",
        "observation/hand_camera/rgb",
        "observation/hand_rgb",
        "wrist_image",
        "image_wrist",
    ]

    proprio_candidates = [
        "observation/proprio",
        "observation/state",
        "proprio",
        "state",
    ]

    action_candidates = [
        "action",
        "actions",
    ]

    language_candidates = [
        "language_instruction",
        "language_instruction_0",
        "observation/language_instruction",
    ]

    for step_idx, step in enumerate(tfds.as_numpy(steps_ds)):
        step_count += 1

        if step_idx >= args.max_steps_per_episode:
            print_warn(
                "step checking truncated",
                f"only checked first {args.max_steps_per_episode} steps in this episode",
            )
            break

        primary_path, primary_img = find_first_existing(step, primary_image_candidates)
        wrist_path, wrist_img = find_first_existing(step, wrist_image_candidates)
        proprio_path, proprio = find_first_existing(step, proprio_candidates)
        action_path, action = find_first_existing(step, action_candidates)
        language_path, language = find_first_existing(step, language_candidates)

        if primary_img is None:
            print_fail(f"step {step_idx} primary image exists", f"tried {primary_image_candidates}")
            episode_failed = True
            continue

        if wrist_img is None:
            print_fail(f"step {step_idx} wrist image exists", f"tried {wrist_image_candidates}")
            episode_failed = True
            continue

        if action is None:
            print_fail(f"step {step_idx} action exists", f"tried {action_candidates}")
            episode_failed = True
            continue

        if proprio is None:
            print_warn(f"step {step_idx} proprio exists", f"tried {proprio_candidates}")
        else:
            ok, msg = is_proprio_valid(proprio)
            if not ok:
                print_fail(f"step {step_idx} proprio valid", msg)
                episode_failed = True

        if language is None:
            print_fail(f"step {step_idx} language_instruction exists", f"tried {language_candidates}")
            episode_failed = True
        else:
            lang = decode_str(language)
            if len(lang.strip()) == 0:
                print_fail(f"step {step_idx} language_instruction non-empty", "empty string")
                episode_failed = True
            elif step_idx == 0:
                print_pass("language_instruction", repr(lang))

        ok, msg = is_image_valid(primary_img, args.expect_h, args.expect_w)
        if not ok:
            print_fail(f"step {step_idx} primary image valid", f"{primary_path}: {msg}")
            episode_failed = True
        elif step_idx == 0:
            print_pass("primary image valid", f"{primary_path}: {msg}")

        ok, msg = is_image_valid(wrist_img, args.expect_h, args.expect_w)
        if not ok:
            print_fail(f"step {step_idx} wrist image valid", f"{wrist_path}: {msg}")
            episode_failed = True
        elif step_idx == 0:
            print_pass("wrist image valid", f"{wrist_path}: {msg}")

        ok, msg = is_action_valid(action, args.expect_action_dim)
        if not ok:
            print_fail(f"step {step_idx} action valid", f"{action_path}: {msg}")
            episode_failed = True
        elif step_idx == 0:
            print_pass("action valid", f"{action_path}: {msg}")

        action_arr = np.asarray(action)
        action_values.append(action_arr.reshape(-1))

        # Flags.
        is_first = step.get("is_first", None)
        is_last = step.get("is_last", None)
        is_terminal = step.get("is_terminal", None)

        if is_first is not None:
            first_flags.append(bool(np.asarray(is_first).item()))
        if is_last is not None:
            last_flags.append(bool(np.asarray(is_last).item()))
        if is_terminal is not None:
            terminal_flags.append(bool(np.asarray(is_terminal).item()))

        # Dump first few image samples.
        if args.dump_samples and step_idx in [0, 1, 2]:
            os.makedirs(args.dump_samples, exist_ok=True)
            save_image(
                os.path.join(args.dump_samples, f"ep{episode_idx:04d}_step{step_idx:04d}_primary.png"),
                primary_img,
            )
            save_image(
                os.path.join(args.dump_samples, f"ep{episode_idx:04d}_step{step_idx:04d}_wrist.png"),
                wrist_img,
            )

    if step_count == 0:
        print_fail("episode length", "0 steps")
        episode_failed = True
    else:
        print_pass("episode length", f"{step_count} checked steps")

    # Check step flags.
    if first_flags:
        if first_flags[0] is True and sum(first_flags) == 1:
            print_pass("is_first flags", "only first step is True")
        else:
            print_fail("is_first flags", f"flags={first_flags[:20]}")
            episode_failed = True
    else:
        print_warn("is_first flags", "missing")

    if last_flags:
        if last_flags[-1] is True and sum(last_flags) == 1:
            print_pass("is_last flags", "only last checked step is True")
        else:
            print_fail("is_last flags", f"flags={last_flags[:20]}")
            episode_failed = True
    else:
        print_warn("is_last flags", "missing")

    if terminal_flags:
        # For successful demonstrations, terminal usually matches final step.
        # Some RLDS datasets use is_terminal=False when timeout happens, so this is a warning unless strict.
        if terminal_flags[-1] is True:
            print_pass("is_terminal final flag", "final checked step is terminal")
        else:
            if args.strict_terminal:
                print_fail("is_terminal final flag", f"final={terminal_flags[-1]}")
                episode_failed = True
            else:
                print_warn("is_terminal final flag", f"final={terminal_flags[-1]}")
    else:
        print_warn("is_terminal flags", "missing")

    if action_values:
        all_actions = np.concatenate(action_values, axis=0)
        action_std = float(np.std(all_actions))
        if action_std < 1e-8:
            print_warn("action variation", f"actions are nearly constant, std={action_std}")
        else:
            print_pass("action variation", f"std={action_std:.6f}")

    if episode_failed:
        counters["episodes_failed"] += 1
    else:
        counters["episodes_passed"] += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="TFDS data_dir containing the built dataset")
    parser.add_argument("--dataset-name", required=True, help="Dataset name, e.g. pick_cube_two_cam or pick_cube_two_cam/1.0.0")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--max-steps-per-episode", type=int, default=300)
    parser.add_argument("--expect-h", type=int, default=224)
    parser.add_argument("--expect-w", type=int, default=224)
    parser.add_argument("--expect-action-dim", type=int, default=7)
    parser.add_argument("--dump-samples", default=None, help="Directory to dump sample images")
    parser.add_argument("--strict-terminal", action="store_true")
    parser.add_argument("--print-spec", action="store_true")
    args = parser.parse_args()

    print("=" * 80)
    print("Checking OpenVLA-OFT RLDS dataset")
    print("=" * 80)
    print(f"data_dir:     {args.data_dir}")
    print(f"dataset_name: {args.dataset_name}")
    print(f"split:        {args.split}")
    print(f"num_episodes: {args.num_episodes}")
    print("=" * 80)

    try:
        builder = tfds.builder(args.dataset_name, data_dir=args.data_dir)
        print_pass("TFDS builder loaded", builder.name)
    except Exception as e:
        print_fail("TFDS builder loaded", repr(e))
        sys.exit(1)

    try:
        print(f"builder.info.version: {builder.info.version}")
        print(f"builder.info.splits:  {builder.info.splits}")
    except Exception as e:
        print_warn("builder info", repr(e))

    try:
        ds = builder.as_dataset(split=args.split)
        print_pass("dataset loaded", f"split={args.split}")
    except Exception as e:
        print_fail("dataset loaded", repr(e))
        sys.exit(1)

    if args.print_spec:
        print("\n[Element spec]")
        for path, spec in flatten_spec(ds.element_spec):
            print(f"{path}: {spec}")

    counters = {
        "episodes_passed": 0,
        "episodes_failed": 0,
    }

    for episode_idx, episode in enumerate(ds.take(args.num_episodes)):
        check_episode(episode, episode_idx, args, counters)

    print("\n" + "=" * 80)
    print("Final summary")
    print("=" * 80)
    print(f"episodes passed: {counters['episodes_passed']}")
    print(f"episodes failed: {counters['episodes_failed']}")

    if counters["episodes_failed"] == 0 and counters["episodes_passed"] > 0:
        print("\n✅ FINAL RESULT: PASS. This RLDS dataset looks structurally valid for OpenVLA-OFT.")
    else:
        print("\n❌ FINAL RESULT: FAIL or PARTIAL. Fix the failed checks before training OpenVLA-OFT.")


if __name__ == "__main__":
    main()