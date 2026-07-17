#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Check ManiSkill replayed trajectory quality for OpenVLA-OFT training.

It checks:
1. base_camera/rgb exists
2. hand_camera/rgb exists
3. RGB images are 224x224x3 and not black
4. action dim is 7
5. success flags if available in h5/json

Usage:
    python scripts/check_replay_dataset.py \
        --h5 /data/disk1/lcx_stu4/PickCube-v1/motionplanning/trajectory.rgbd.pd_ee_delta_pose.physx_cpu.h5

        /cpfs01/lcx_stu4_workspace/envs/why_maniskill/bin/python   rl/maniskill/check_replay_dataset.py   --h5 /mnt/data2/lcx_stu4/maniskill/demos/StackCube-v1/motionplanning_rgbd_224_two_cam/trajectory.rgbd.pd_ee_delta_pose.physx_cpu.h5   --expect-h 224   --expect-w 224   --expect-action-dim 7   --dump-keys
Optional:
    python scripts/check_replay_dataset.py \
        --h5 /path/to/replayed/trajectory.h5 \
        --json /path/to/replayed/trajectory.json \
    
        --dump-keys
"""

import argparse
import json
import os
import re
from typing import Dict, List, Tuple, Any

import h5py
import numpy as np


def collect_h5_datasets(h5_path: str) -> Dict[str, Tuple[Tuple[int, ...], str]]:
    datasets = {}

    with h5py.File(h5_path, "r") as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets[name] = (tuple(obj.shape), str(obj.dtype))

        f.visititems(visitor)

    return datasets


def find_camera_rgb_paths(datasets: Dict[str, Tuple[Tuple[int, ...], str]], camera_name: str) -> List[str]:
    paths = []
    for path in datasets.keys():
        normalized = path.replace("\\", "/")
        if f"/{camera_name}/rgb" in normalized or normalized.endswith(f"{camera_name}/rgb"):
            paths.append(path)
    return sorted(paths)


def find_action_paths(datasets: Dict[str, Tuple[Tuple[int, ...], str]]) -> List[str]:
    paths = []
    for path in datasets.keys():
        name = path.replace("\\", "/").lower()
        if name.endswith("/actions") or name.endswith("/action") or name == "actions" or name == "action":
            paths.append(path)
    return sorted(paths)


def find_success_paths(datasets: Dict[str, Tuple[Tuple[int, ...], str]]) -> List[str]:
    paths = []
    for path in datasets.keys():
        name = path.lower()
        basename = name.split("/")[-1]
        if "success" in basename or "is_success" in basename:
            paths.append(path)
    return sorted(paths)


def sample_dataset(ds: h5py.Dataset, max_frames: int = 16) -> np.ndarray:
    """
    Sample a few frames from a large dataset without loading all data.
    Assumes time dimension is usually the first dimension.
    """
    if ds.ndim == 0:
        return ds[()]

    if ds.shape[0] <= max_frames:
        return ds[...]

    indices = np.linspace(0, ds.shape[0] - 1, max_frames).astype(int)
    indices = np.unique(indices)
    return ds[indices]


def check_rgb_dataset(
    h5_path: str,
    path: str,
    expect_h: int = 224,
    expect_w: int = 224,
    max_frames: int = 16,
) -> Dict[str, Any]:
    result = {
        "path": path,
        "exists": True,
        "shape_ok": False,
        "not_black": False,
        "dtype": None,
        "shape": None,
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
        "message": "",
    }

    with h5py.File(h5_path, "r") as f:
        ds = f[path]
        result["shape"] = tuple(ds.shape)
        result["dtype"] = str(ds.dtype)

        shape = tuple(ds.shape)

        # Expected image suffix: H, W, C
        # Common shapes:
        #   [T, H, W, 3]
        #   [T, 1, H, W, 3]
        #   [H, W, 3]
        if len(shape) >= 3 and shape[-3:] == (expect_h, expect_w, 3):
            result["shape_ok"] = True
        else:
            result["message"] += f"shape mismatch, expected suffix ({expect_h}, {expect_w}, 3); "

        arr = sample_dataset(ds, max_frames=max_frames)
        arr = np.asarray(arr)

        if arr.size == 0:
            result["message"] += "empty dataset; "
            return result

        result["min"] = float(np.min(arr))
        result["max"] = float(np.max(arr))
        result["mean"] = float(np.mean(arr))
        result["std"] = float(np.std(arr))

        # For uint8 image, std should normally be much larger than 1.
        # For float image in [0, 1], std can be smaller, so use adaptive threshold.
        if np.issubdtype(arr.dtype, np.integer):
            result["not_black"] = (result["max"] > result["min"]) and (result["std"] > 1.0)
        else:
            result["not_black"] = (result["max"] > result["min"]) and (result["std"] > 1e-3)

        if not result["not_black"]:
            result["message"] += "image may be black/constant; "

    return result


def check_action_dataset(h5_path: str, path: str, expect_dim: int = 7) -> Dict[str, Any]:
    result = {
        "path": path,
        "shape": None,
        "dtype": None,
        "dim_ok": False,
        "last_dim": None,
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
        "message": "",
    }

    with h5py.File(h5_path, "r") as f:
        ds = f[path]
        result["shape"] = tuple(ds.shape)
        result["dtype"] = str(ds.dtype)

        if ds.ndim == 0:
            result["message"] = "scalar action dataset?"
            return result

        result["last_dim"] = int(ds.shape[-1])
        result["dim_ok"] = result["last_dim"] == expect_dim

        arr = sample_dataset(ds, max_frames=64)
        arr = np.asarray(arr)

        if arr.size > 0:
            result["min"] = float(np.min(arr))
            result["max"] = float(np.max(arr))
            result["mean"] = float(np.mean(arr))
            result["std"] = float(np.std(arr))

        if not result["dim_ok"]:
            result["message"] = f"action last dim is {result['last_dim']}, expected {expect_dim}"

    return result


def read_h5_success_flags(h5_path: str, success_paths: List[str]) -> List[Dict[str, Any]]:
    results = []

    with h5py.File(h5_path, "r") as f:
        for path in success_paths:
            ds = f[path]
            arr = sample_dataset(ds, max_frames=4096)
            arr = np.asarray(arr)

            if arr.size == 0:
                continue

            flat = arr.reshape(-1)
            numeric = flat.astype(np.float32)

            results.append({
                "path": path,
                "shape": tuple(ds.shape),
                "dtype": str(ds.dtype),
                "num_values": int(flat.size),
                "num_true": int(np.sum(numeric > 0.5)),
                "all_true": bool(np.all(numeric > 0.5)),
                "any_true": bool(np.any(numeric > 0.5)),
                "last_value": float(numeric[-1]),
            })

    return results


def collect_json_success(obj: Any, prefix: str = "") -> List[Tuple[str, Any]]:
    """
    Recursively find keys containing 'success' in trajectory.json.
    """
    found = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            key_path = f"{prefix}.{k}" if prefix else k
            if "success" in k.lower():
                found.append((key_path, v))
            found.extend(collect_json_success(v, key_path))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            key_path = f"{prefix}[{i}]"
            found.extend(collect_json_success(v, key_path))

    return found


def read_json_success_flags(json_path: str) -> List[Dict[str, Any]]:
    if not json_path or not os.path.exists(json_path):
        return []

    with open(json_path, "r") as f:
        data = json.load(f)

    found = collect_json_success(data)

    results = []
    for key_path, value in found:
        if isinstance(value, bool):
            results.append({
                "path": key_path,
                "type": "bool",
                "value": value,
            })
        elif isinstance(value, (int, float)):
            results.append({
                "path": key_path,
                "type": "number",
                "value": value,
            })
        elif isinstance(value, list):
            try:
                arr = np.asarray(value).astype(np.float32).reshape(-1)
                results.append({
                    "path": key_path,
                    "type": "list",
                    "num_values": int(arr.size),
                    "num_true": int(np.sum(arr > 0.5)),
                    "all_true": bool(np.all(arr > 0.5)),
                    "any_true": bool(np.any(arr > 0.5)),
                    "last_value": float(arr[-1]) if arr.size > 0 else None,
                })
            except Exception:
                results.append({
                    "path": key_path,
                    "type": "list_non_numeric",
                    "value_preview": str(value)[:200],
                })
        else:
            results.append({
                "path": key_path,
                "type": type(value).__name__,
                "value_preview": str(value)[:200],
            })

    return results


def print_status(name: str, ok: bool, message: str = ""):
    mark = "✅ PASS" if ok else "❌ FAIL"
    if message:
        print(f"{mark} | {name}: {message}")
    else:
        print(f"{mark} | {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True, help="Path to replayed trajectory.h5")
    parser.add_argument("--json", default=None, help="Path to trajectory.json. If omitted, infer from h5 dir.")
    parser.add_argument("--expect-h", type=int, default=224)
    parser.add_argument("--expect-w", type=int, default=224)
    parser.add_argument("--expect-action-dim", type=int, default=7)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--dump-keys", action="store_true")
    args = parser.parse_args()

    h5_path = args.h5
    json_path = args.json

    if json_path is None:
        candidate = os.path.join(os.path.dirname(h5_path), "trajectory.json")
        if os.path.exists(candidate):
            json_path = candidate

    if not os.path.exists(h5_path):
        raise FileNotFoundError(h5_path)

    print("=" * 80)
    print("Checking ManiSkill replay dataset")
    print(f"h5:   {h5_path}")
    print(f"json: {json_path if json_path else 'not found / not provided'}")
    print("=" * 80)

    datasets = collect_h5_datasets(h5_path)

    if args.dump_keys:
        print("\n[All H5 datasets]")
        for path, (shape, dtype) in sorted(datasets.items()):
            print(f"{path}: shape={shape}, dtype={dtype}")

    base_paths = find_camera_rgb_paths(datasets, "base_camera")
    hand_paths = find_camera_rgb_paths(datasets, "hand_camera")
    action_paths = find_action_paths(datasets)
    success_paths = find_success_paths(datasets)

    print("\n[1] Camera existence")
    print_status("base_camera/rgb exists", len(base_paths) > 0, f"found {len(base_paths)} path(s)")
    for p in base_paths:
        print(f"  - {p}: shape={datasets[p][0]}, dtype={datasets[p][1]}")

    print_status("hand_camera/rgb exists", len(hand_paths) > 0, f"found {len(hand_paths)} path(s)")
    for p in hand_paths:
        print(f"  - {p}: shape={datasets[p][0]}, dtype={datasets[p][1]}")

    print("\n[2] RGB shape and non-black check")
    rgb_results = []

    for camera_name, paths in [("base_camera", base_paths), ("hand_camera", hand_paths)]:
        if not paths:
            continue

        # Usually there is one path. If multiple episodes have separate paths, check all.
        for p in paths:
            r = check_rgb_dataset(
                h5_path,
                p,
                expect_h=args.expect_h,
                expect_w=args.expect_w,
                max_frames=args.max_frames,
            )
            rgb_results.append((camera_name, r))

            ok = r["shape_ok"] and r["not_black"]
            msg = (
                f"path={p}, shape={r['shape']}, dtype={r['dtype']}, "
                f"min={r['min']}, max={r['max']}, mean={r['mean']:.3f}, std={r['std']:.3f}"
            )
            if r["message"]:
                msg += f", warning={r['message']}"
            print_status(f"{camera_name} RGB valid", ok, msg)

    print("\n[3] Action dimension check")
    action_results = []

    if not action_paths:
        print_status("actions exists", False, "no action/actions dataset found")
    else:
        print_status("actions exists", True, f"found {len(action_paths)} path(s)")
        for p in action_paths:
            r = check_action_dataset(h5_path, p, expect_dim=args.expect_action_dim)
            action_results.append(r)

            ok = r["dim_ok"]
            msg = (
                f"path={p}, shape={r['shape']}, dtype={r['dtype']}, "
                f"last_dim={r['last_dim']}, min={r['min']}, max={r['max']}, "
                f"mean={r['mean']:.6f}, std={r['std']:.6f}"
            )
            if r["message"]:
                msg += f", warning={r['message']}"
            print_status("action dim valid", ok, msg)

    print("\n[4] Success flag check")
    h5_success = read_h5_success_flags(h5_path, success_paths)
    json_success = read_json_success_flags(json_path) if json_path else []

    success_known = bool(h5_success or json_success)
    success_ok = True

    if h5_success:
        print("[H5 success-like fields]")
        for r in h5_success:
            field_ok = bool(r["any_true"])
            success_ok = success_ok and field_ok
            print_status(
                f"success field {r['path']}",
                field_ok,
                (
                    f"shape={r['shape']}, dtype={r['dtype']}, "
                    f"num_true={r['num_true']}/{r['num_values']}, "
                    f"last_value={r['last_value']}"
                ),
            )

    if json_success:
        print("[JSON success-like fields]")
        for r in json_success:
            if r["type"] == "bool":
                field_ok = bool(r["value"])
                success_ok = success_ok and field_ok
                print_status(f"json {r['path']}", field_ok, f"value={r['value']}")
            elif r["type"] == "number":
                field_ok = float(r["value"]) > 0.5
                success_ok = success_ok and field_ok
                print_status(f"json {r['path']}", field_ok, f"value={r['value']}")
            elif r["type"] == "list":
                field_ok = bool(r["any_true"])
                success_ok = success_ok and field_ok
                print_status(
                    f"json {r['path']}",
                    field_ok,
                    (
                        f"num_true={r['num_true']}/{r['num_values']}, "
                        f"last_value={r['last_value']}"
                    ),
                )
            else:
                print(f"⚠️ WARN | json {r['path']}: cannot judge automatically, {r}")

    if not success_known:
        print(
            "⚠️ WARN | success unknown: no success-like field found in h5/json. "
            "You need to check replay logs, or explicitly save success info during replay."
        )

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    cond1 = len(base_paths) > 0 and len(hand_paths) > 0
    cond2 = bool(rgb_results) and all(r["shape_ok"] and r["not_black"] for _, r in rgb_results)
    cond3 = bool(action_results) and all(r["dim_ok"] for r in action_results)
    cond4 = success_known and success_ok

    print_status("Condition 1: base_camera and hand_camera exist", cond1)
    print_status(f"Condition 2: RGB images are {args.expect_h}x{args.expect_w}x3 and not black", cond2)
    print_status(f"Condition 3: action dim is {args.expect_action_dim}", cond3)

    if success_known:
        print_status("Condition 4: replay success is available and successful", cond4)
    else:
        print("⚠️ WARN | Condition 4: replay success cannot be judged from this h5/json")

    if cond1 and cond2 and cond3 and cond4:
        print("\n✅ FINAL RESULT: PASS. This replayed data looks ready for OpenVLA-OFT RLDS conversion.")
    elif cond1 and cond2 and cond3 and not success_known:
        print(
            "\n⚠️ FINAL RESULT: PARTIAL PASS. Image/action format is OK, "
            "but success is unknown. Check replay logs or regenerate with success recording."
        )
    else:
        print("\n❌ FINAL RESULT: FAIL. Do not use this data for OpenVLA-OFT training before fixing the failed checks.")


if __name__ == "__main__":
    main()