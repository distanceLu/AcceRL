"""TFDS/RLDS builder for DrawTriangle-v1 delta-pose demonstrations.

PandaStick emits a 6-D ``pd_ee_delta_pose`` action. OpenVLA in this repository
has a fixed 7-D action head, so the builder appends one inactive zero dimension.
"""

from __future__ import annotations

import os
import argparse
from typing import Any, Iterator, Tuple

import h5py
import numpy as np
import tensorflow_datasets as tfds
from scipy.spatial.transform import Rotation as R
from tensorflow_datasets.core.utils import gcs_utils


gcs_utils._is_gcs_disabled = True

DEFAULT_H5_PATH = (
    "/mnt/data/lcx/data/maniskill/DrawTriangle-v1/motionplanning/"
    "trajectory.rgbd.pd_ee_delta_pose.physx_cpu.h5"
)
IMG_SIZE = 224
LANGUAGE_INSTRUCTION = "draw the outlined triangle on the canvas"


def quat_wxyz_to_axisangle(quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64)
    q_xyzw = np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)
    return R.from_quat(q_xyzw).as_rotvec().astype(np.float32)


class ManiskillDrawtriangle(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "DrawTriangle-v1, PandaStick, one 224px camera, padded 7-D delta pose.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image": tfds.features.Image(
                            shape=(IMG_SIZE, IMG_SIZE, 3),
                            dtype=np.uint8,
                            encoding_format="jpeg",
                        ),
                        "state": tfds.features.Tensor(shape=(8,), dtype=np.float32),
                        "joint_state": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    }),
                    "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    "reward": tfds.features.Scalar(dtype=np.float32),
                    "discount": tfds.features.Scalar(dtype=np.float32),
                    "is_first": tfds.features.Scalar(dtype=np.bool_),
                    "is_last": tfds.features.Scalar(dtype=np.bool_),
                    "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                    "language_instruction": tfds.features.Text(),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "file_path": tfds.features.Text(),
                    "episode_id": tfds.features.Text(),
                    "success": tfds.features.Scalar(dtype=np.bool_),
                }),
            }),
            supervised_keys=None,
            homepage="https://github.com/haosulab/ManiSkill",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        h5_path = os.environ.get("MANISKILL_DRAWTRIANGLE_H5", DEFAULT_H5_PATH)
        return {"train": self._generate_examples(h5_path)}

    def _generate_examples(self, h5_path: str) -> Iterator[Tuple[str, Any]]:
        os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
        with h5py.File(h5_path, "r") as h5_file:
            traj_keys = sorted(h5_file.keys(), key=lambda name: int(name.split("_")[1]))
            for traj_key in traj_keys:
                group = h5_file[traj_key]
                actions_6d = np.asarray(group["actions"], dtype=np.float32)
                if actions_6d.ndim != 2 or actions_6d.shape[1] != 6:
                    raise ValueError(
                        f"{traj_key} has action shape {actions_6d.shape}; expected replayed "
                        "6-D pd_ee_delta_pose actions, not original pd_joint_pos data"
                    )
                num_steps = len(actions_6d)
                if num_steps == 0:
                    continue

                success = np.asarray(group["success"], dtype=bool)[:num_steps]
                episode_success = bool(success[-1])
                if not episode_success:
                    continue

                rgb = np.asarray(group["obs/sensor_data/base_camera/rgb"])[:num_steps]
                tcp_pose = np.asarray(
                    group["obs/extra/tcp_pose"], dtype=np.float32
                )[:num_steps]
                qpos = np.asarray(
                    group["obs/agent/qpos"], dtype=np.float32
                )[:num_steps, :7]

                tcp_state = np.concatenate(
                    [tcp_pose[:, :3], quat_wxyz_to_axisangle(tcp_pose[:, 3:7])], axis=-1
                )
                state = np.concatenate(
                    [tcp_state, np.zeros((num_steps, 2), dtype=np.float32)], axis=-1
                ).astype(np.float32)
                actions = np.concatenate(
                    [actions_6d, np.zeros((num_steps, 1), dtype=np.float32)], axis=-1
                )

                steps = []
                for step_idx in range(num_steps):
                    steps.append({
                        "observation": {
                            "image": np.ascontiguousarray(rgb[step_idx]),
                            "state": state[step_idx],
                            "joint_state": qpos[step_idx],
                        },
                        "action": actions[step_idx],
                        "reward": np.float32(1.0 if step_idx == num_steps - 1 else 0.0),
                        "discount": np.float32(1.0),
                        "is_first": bool(step_idx == 0),
                        "is_last": bool(step_idx == num_steps - 1),
                        "is_terminal": bool(step_idx == num_steps - 1),
                        "language_instruction": LANGUAGE_INSTRUCTION,
                    })

                yield traj_key, {
                    "steps": steps,
                    "episode_metadata": {
                        "file_path": h5_path,
                        "episode_id": traj_key,
                        "success": episode_success,
                    },
                }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the local DrawTriangle-v1 TFDS/RLDS dataset."
    )
    parser.add_argument(
        "--data-dir",
        default="/mnt/data/lcx/data/maniskill/DrawTriangle-v1/rlds",
        help="TFDS output root.",
    )
    parser.add_argument(
        "--h5-path",
        default=DEFAULT_H5_PATH,
        help="Replayed 6-D pd_ee_delta_pose trajectory H5.",
    )
    args = parser.parse_args()

    os.environ["MANISKILL_DRAWTRIANGLE_H5"] = os.path.abspath(args.h5_path)
    builder = ManiskillDrawtriangle(data_dir=os.path.abspath(args.data_dir))
    print(f"Building {builder.name}/{builder.version} from {args.h5_path}")
    print(f"TFDS output root: {args.data_dir}")
    builder.download_and_prepare()


if __name__ == "__main__":
    main()
