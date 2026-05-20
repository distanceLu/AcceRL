"""Replay ManiSkill PickCube-v1 trajectories with panda_wristcam (dual camera)
at 224x224 resolution, converting pd_joint_pos -> pd_ee_delta_pose.

This wrapper monkey-patches gymnasium.make at the module level so that when
the official replay_trajectory code calls gym.make(), it injects robot_uids
and sensor_configs for dual-camera 224x224 rendering.

Usage:
    cd /cpfs01/lcx_stu4_workspace/openvla_oft_rl
    python scripts/replay_224_two_cam.py

    # Override source trajectory path:
    TRAJ_PATH=/path/to/trajectory.h5 python scripts/replay_224_two_cam.py
"""

import os

# Monkey-patch gymnasium.make BEFORE importing replay_trajectory
import gymnasium as gym

_original_gym_make = gym.make


def _patched_gym_make(env_id, **kwargs):
    """Intercept gym.make to inject sensor_configs and robot_uids."""
    kwargs["robot_uids"] = "panda_wristcam"
    kwargs["sensor_configs"] = {
        "base_camera": {"width": 224, "height": 224},
        "hand_camera": {"width": 224, "height": 224},
    }
    print(
        f"[replay_224_two_cam] Injecting robot_uids='panda_wristcam', "
        f"sensor_configs 224x224 into gym.make({env_id!r})"
    )
    return _original_gym_make(env_id, **kwargs)


gym.make = _patched_gym_make

# Now import replay_trajectory — it will use our patched gym.make
from mani_skill.trajectory.replay_trajectory import Args, main as replay_main


def main():
    traj_path = os.environ.get(
        "TRAJ_PATH",
        "/data/disk1/lcx_stu4/PickCube-v1/motionplanning/trajectory.h5",
    )

    args = Args(
        traj_path=traj_path,
        sim_backend="physx_cpu",
        obs_mode="rgbd",
        target_control_mode="pd_ee_delta_pose",
        save_traj=True,
        allow_failure=True,
        verbose=True,
    )

    replay_main(args)


if __name__ == "__main__":
    main()
