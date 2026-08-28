"""Replay DrawTriangle-v1 demos as 224px RGB delta end-effector actions.

The source motion-planning dataset uses ``pd_joint_pos``. The output is
written next to it as ``trajectory.rgbd.pd_ee_delta_pose.physx_cpu.h5``.
"""

import os

import gymnasium as gym


_original_gym_make = gym.make


def _patched_gym_make(env_id, **kwargs):
    kwargs["robot_uids"] = "panda_stick"
    kwargs["sensor_configs"] = {
        "base_camera": {"width": 224, "height": 224},
    }
    return _original_gym_make(env_id, **kwargs)


gym.make = _patched_gym_make

from mani_skill.trajectory.replay_trajectory import Args, main as replay_main


def main():
    traj_path = os.environ.get(
        "TRAJ_PATH",
        "/mnt/data/lcx/data/maniskill/DrawTriangle-v1/motionplanning/trajectory.h5",
    )
    replay_main(
        Args(
            traj_path=traj_path,
            sim_backend="physx_cpu",
            obs_mode="rgbd",
            target_control_mode="pd_ee_delta_pose",
            save_traj=True,
            # Numerical replay can fail for a subset of motion-planning demos;
            # save them and let the RLDS builder keep successful episodes only.
            allow_failure=True,
            verbose=True,
        )
    )


if __name__ == "__main__":
    main()
