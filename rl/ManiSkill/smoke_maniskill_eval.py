import numpy as np

import os
import numpy as np

from experiments.robot.maniskill.maniskill_utils import (
    build_maniskill_env,
    extract_maniskill_observation,
    clip_maniskill_action,
)

def main():
    # 尽量保守，先验证可运行性
    num_envs = 1
    max_steps = 5

    env = build_maniskill_env(
        task_id="PickCube-v1",
        num_envs=num_envs,
        obs_mode="rgbd",
        control_mode="pd_ee_delta_pose",
        camera_name="base_camera",
        wrist_camera_name="hand_camera",
        camera_res=224,              # 先按你当前配置测
        sim_backend="cpu",           # 稳定优先；通过后再切 gpu
        robot_uids="panda_wristcam", # 双相机关键
        max_episode_steps=100,
    )

    obs, _ = env.reset(seed=[0])

    # 检查是否能正确拿到双图
    one_obs = extract_maniskill_observation(
        obs,
        env_idx=0,
        camera_name="base_camera",
        wrist_camera_name="hand_camera",
        include_wrist_image=True,
        use_proprio=False,
    )

    print("keys:", one_obs.keys())
    print("full_image:", one_obs["full_image"].shape, one_obs["full_image"].dtype)
    print("wrist_image:", one_obs["wrist_image"].shape, one_obs["wrist_image"].dtype)

    # 随机动作冒烟
    for t in range(max_steps):
        raw_action = np.random.uniform(-1, 1, size=(num_envs, 7)).astype(np.float32)
        action = clip_maniskill_action(raw_action)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"step={t}, reward={reward}, terminated={terminated}, truncated={truncated}")

    env.close()
    print("Smoke eval passed.")

if __name__ == "__main__":
    main()