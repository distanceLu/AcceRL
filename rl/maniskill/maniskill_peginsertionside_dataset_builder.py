"""TFDS builder for ManiSkill PegInsertionSide-v1 demonstrations.

Input : an h5 file produced by ``mani_skill.trajectory.replay_trajectory`` with
        ``--obs-mode rgbd`` using panda_wristcam robot.
Output: an RLDS-style ``tfds`` dataset compatible with the OpenVLA RLDS loader.

Action space is ``pd_ee_delta_pose`` (7-D):
    [dx, dy, dz, drx, dry, drz, gripper]

Per-step state layout (8-D):
    [tcp_x, tcp_y, tcp_z, tcp_rx, tcp_ry, tcp_rz, finger_left, finger_right]
"""

from __future__ import annotations

import os
from typing import Any, Iterator, Tuple

import h5py
import numpy as np
import tensorflow_datasets as tfds
from scipy.spatial.transform import Rotation as R

from tensorflow_datasets.core.utils import gcs_utils

gcs_utils._is_gcs_disabled = True

DEFAULT_H5_PATH = (
    "/mnt/data2/lcx_stu4/maniskill/demos/PegInsertionSide-v1/"
    "motionplanning_rgbd_224_two_cam/trajectory.rgbd.pd_ee_delta_pose.physx_cpu.h5"
)

IMG_SIZE = 224
LANGUAGE_INSTRUCTION = "pick up the orange-white peg and insert the orange end into the box with a hole in it"


def quat_wxyz_to_axisangle(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert a batch of wxyz quaternions to axis-angle (rotvec) of shape (..., 3)."""
    q = np.asarray(quat_wxyz, dtype=np.float64)
    q_xyzw = np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)
    return R.from_quat(q_xyzw).as_rotvec().astype(np.float32)


class ManiskillPeginsertionside(tfds.core.GeneratorBasedBuilder):
    """ManiSkill PegInsertionSide-v1 imitation demos, RLDS-compatible."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release of PegInsertionSide-v1 (Panda, pd_ee_delta_pose, rgbd replay, dual camera).",
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
                            doc="Base camera RGB observation.",
                        ),
                        "wrist_image": tfds.features.Image(
                            shape=(IMG_SIZE, IMG_SIZE, 3),
                            dtype=np.uint8,
                            encoding_format="jpeg",
                            doc="Wrist (hand) camera RGB observation.",
                        ),
                        "state": tfds.features.Tensor(
                            shape=(8,),
                            dtype=np.float32,
                            doc="TCP 6D pose (xyz + axis-angle) + 2D gripper finger positions.",
                        ),
                        "joint_state": tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc="Panda arm joint angles (first 7 of qpos).",
                        ),
                    }),
                    "action": tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc="pd_ee_delta_pose action: 3 xyz-delta, 3 axis-angle-delta, 1 gripper.",
                    ),
                    "reward": tfds.features.Scalar(
                        dtype=np.float32,
                        doc="1.0 on the last step of successful episodes, else 0.0.",
                    ),
                    "discount": tfds.features.Scalar(dtype=np.float32, doc="Always 1.0."),
                    "is_first": tfds.features.Scalar(dtype=np.bool_),
                    "is_last": tfds.features.Scalar(dtype=np.bool_),
                    "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                    "language_instruction": tfds.features.Text(doc="Task language instruction."),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "file_path": tfds.features.Text(doc="Path to original h5 file."),
                    "episode_id": tfds.features.Text(doc="traj_X key in the h5 file."),
                    "success": tfds.features.Scalar(
                        dtype=np.bool_,
                        doc="Whether the final step is a successful PegInsertionSide demo.",
                    ),
                }),
            }),
            supervised_keys=None,
            homepage="https://github.com/haosulab/ManiSkill",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        h5_path = os.environ.get("MANISKILL_PEGIN_INSERTION_SIDE_H5", DEFAULT_H5_PATH)
        return {"train": self._generate_examples(h5_path=h5_path)}

    def _generate_examples(self, h5_path: str) -> Iterator[Tuple[str, Any]]:
        os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

        f = h5py.File(h5_path, "r")
        try:
            traj_keys = sorted(f.keys(), key=lambda s: int(s.split("_")[1]))
            for tk in traj_keys:
                g = f[tk]
                actions = np.asarray(g["actions"], dtype=np.float32)
                n = actions.shape[0]
                if n == 0:
                    continue

                qpos = np.asarray(g["obs/agent/qpos"], dtype=np.float32)[:n]
                tcp_pose = np.asarray(g["obs/extra/tcp_pose"], dtype=np.float32)[:n]
                rgb = np.asarray(g["obs/sensor_data/base_camera/rgb"])[:n]
                wrist_rgb = np.asarray(g["obs/sensor_data/hand_camera/rgb"])[:n]
                success_arr = np.asarray(g["success"], dtype=bool)

                tcp_xyz = tcp_pose[:, :3]
                tcp_aa = quat_wxyz_to_axisangle(tcp_pose[:, 3:7])
                gripper = qpos[:, 7:9]
                state = np.concatenate([tcp_xyz, tcp_aa, gripper], axis=-1).astype(np.float32)
                joint_state = qpos[:, :7].astype(np.float32)

                ep_success = bool(success_arr[-1])
                if not ep_success:
                    continue

                steps = []
                for t in range(n):
                    steps.append({
                        "observation": {
                            "image": np.ascontiguousarray(rgb[t]),
                            "wrist_image": np.ascontiguousarray(wrist_rgb[t]),
                            "state": state[t],
                            "joint_state": joint_state[t],
                        },
                        "action": actions[t],
                        "reward": np.float32(1.0 if (t == n - 1 and ep_success) else 0.0),
                        "discount": np.float32(1.0),
                        "is_first": bool(t == 0),
                        "is_last": bool(t == n - 1),
                        "is_terminal": bool(t == n - 1),
                        "language_instruction": LANGUAGE_INSTRUCTION,
                    })

                yield tk, {
                    "steps": steps,
                    "episode_metadata": {
                        "file_path": str(h5_path),
                        "episode_id": tk,
                        "success": ep_success,
                    },
                }
        finally:
            try:
                f.close()
            except Exception:
                pass
