import os

# 1. 禁用物理显示器寻找，防止死锁
GPU_ID = "6"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# Prevent TensorFlow from grabbing other GPUs on import
os.environ["TF_CUDA_VISIBLE_DEVICES"] = GPU_ID

# Vulkan / SAPIEN device pinning
os.environ["VULKAN_VISIBLE_DEVICES"] = GPU_ID
os.environ["SAPIEN_VULKAN_DEVICE"] = GPU_ID
os.environ["EGL_DEVICE_ID"] = GPU_ID
os.environ["VK_ICD_FILENAMES"] = "/etc/vulkan/icd.d/nvidia_icd.json"

# import gymnasium as gym
# import mani_skill.envs

# print("开始渲染...")
# # obs_mode="rgbd" 会触发相机渲染
# env = gym.make("PickCube-v1", obs_mode="rgbd")
# print("环境初始化完成...")

# obs, _ = env.reset()
# print("渲染成功！观测数据形状:", obs["image"].keys())

# import os
# os.environ["VK_ICD_FILENAMES"] = "/etc/vulkan/icd.d/nvidia_icd.json"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
# os.environ["VULKAN_VISIBLE_DEVICES"] = "0"       # 指定 Vulkan 渲染用的 GPU
# os.environ["MUJOCO_GL"] = "osmesa"
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"


# import gymnasium as gym
# import mani_skill.envs

# # 只测试最基础的建图和视觉观测
# print("开始渲染...")
# env = gym.make("PickCube-v1", obs_mode="rgbd")
# print("环境初始化完成...")
# obs, _ = env.reset()
# print("渲染成功！观测数据形状:", obs["image"].keys())



"""
简单的 rollout 测试脚本：
- 用 Ray 启动 worker（num_gpus=0.1）
- ManiSkill 环境 + 随机动作
- 验证环境采样流程是否正常
"""
import os
# os.environ["MUJOCO_GL"] = "osmesa"
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import time
import numpy as np
import ray

from rl.maniskill.maniskill_env import ManiSkillSingleEnv


@ray.remote(num_gpus=0.1)
class RandomRolloutWorker:
    def __init__(self, wid, env_args):
        self.wid = wid
        self.env = ManiSkillSingleEnv(
            task_id=env_args["task_id"],
            camera_name=env_args["camera_name"],
            camera_res=env_args["camera_res"],
            max_episode_steps=env_args["max_episode_steps"],
            sim_backend=env_args["sim_backend"],
        )
        print(f"Worker {wid}: 环境初始化完成, GPU IDs: {ray.get_gpu_ids()}")

    def rollout(self, num_episodes=10):
        total_steps = 0
        total_reward = 0.0
        successes = 0
        t0 = time.time()

        for ep in range(num_episodes):
            obs, info = self.env.reset(seed=self.wid * 1000 + ep)
            ep_reward, done = 0.0, False
            while not done:
                action = np.random.uniform(-1, 1, size=(7,)).astype(np.float32)
                obs, r, term, trunc, info = self.env.step(action)
                ep_reward += r
                total_steps += 1
                done = term or trunc
            total_reward += ep_reward
            successes += int(info.get("is_success", 0.0) > 0.5)

        elapsed = time.time() - t0
        return {
            "worker_id": self.wid,
            "episodes": num_episodes,
            "total_steps": total_steps,
            "total_reward": total_reward,
            "successes": successes,
            "elapsed_sec": elapsed,
            "steps_per_sec": total_steps / max(elapsed, 1e-6),
        }

    def close(self):
        self.env.close()


if __name__ == "__main__":
    num_workers = 3
    num_episodes_per_worker = 20

    env_args = {
        "task_id": "PickCube-v1",
        "camera_name": "base_camera",
        "camera_res": 128,
        "max_episode_steps": 100,
        "sim_backend": "cpu",
    }
    ray.init(address="local")
    workers = [RandomRolloutWorker.remote(i, env_args) for i in range(num_workers)]

    print(f"启动 {num_workers} 个 worker，每个跑 {num_episodes_per_worker} 个 episode...")
    t_start = time.time()

    futures = [w.rollout.remote(num_episodes_per_worker) for w in workers]
    results = ray.get(futures)

    total_steps = 0
    total_successes = 0
    for r in results:
        print(f"  Worker {r['worker_id']}: {r['episodes']} eps, "
              f"{r['total_steps']} steps, "
              f"reward={r['total_reward']:.2f}, "
              f"success={r['successes']}, "
              f"{r['steps_per_sec']:.1f} steps/s")
        total_steps += r["total_steps"]
        total_successes += r["successes"]

    elapsed = time.time() - t_start
    print(f"\n总计: {total_steps} steps, {total_successes} successes, "
          f"耗时 {elapsed:.1f}s, 吞吐 {total_steps / max(elapsed, 1e-6):.1f} steps/s")

    ray.get([w.close.remote() for w in workers])
    ray.shutdown()
