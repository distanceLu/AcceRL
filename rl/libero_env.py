import sys
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from libero.libero import benchmark

try:
    sys.path.append("../../..") 
    from experiments.robot.libero.libero_utils import (
        get_libero_dummy_action,
        get_libero_env,
        get_libero_image,
        get_libero_wrist_image,
        quat2axisangle,
    )
    from experiments.robot.robot_utils import (
        invert_gripper_action,
        normalize_gripper_action,
    )
    from experiments.robot.openvla_utils import resize_image_for_policy
except ImportError as e:
    print("错误：无法导入必要的模块。")
    print("请确保此脚本位于正确的目录中，并且您的项目结构与原始代码库一致。")
    print(f"详细错误: {e}")
    sys.exit(1)


# 从评测脚本中借鉴任务最大步数定义
TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


class LiberoEnvWrapper(gym.Env):
    """
    一个将 LIBERO 任务封装为标准 Gymnasium 环境的类，方便用于强化学习训练。

    这个封装器处理了以下细节：
    - 通过任务 ID 初始化特定名称的 LIBERO 任务。
    - 根据评测脚本的逻辑，处理原始观测（图像、本体感受状态）为策略所需的格式。
    - 定义了标准的 `observation_space` 和 `action_space`。
    - 处理发送给模拟器的动作格式。
    - 管理每个 episode 的初始状态。
    """
    def __init__(
        self,
        benchmark_name: str,
        task_id: int,
        image_size: int = 224,
        model_family: str = "openvla",
        render_mode: str = "rgb_array",
        num_steps_wait: int = 10,
    ):
        """
        初始化 LIBERO 强化学习环境。

        Args:
            benchmark_name (str): LIBERO 基准测试套件的名称 (例如, "libero_spatial")。
            task_id (int): 要加载的具体任务的整数 ID。
            image_size (int): 策略输入图像应调整到的大小。
            model_family (str): 模型家族，用于确定动作处理方式（例如 'openvla'）。
            render_mode (str): 渲染模式，支持 'rgb_array'。
            num_steps_wait (int): 在 episode 开始时，等待模拟稳定的步数。
        """
        super().__init__()

        self.benchmark_name = benchmark_name
        self.task_id = task_id
        self.image_size = image_size
        self.model_family = model_family
        self.render_mode = render_mode
        self.num_steps_wait = num_steps_wait

        # 1. 初始化 LIBERO 基准和任务
        benchmark_dict = benchmark.get_benchmark_dict()
        if self.benchmark_name not in benchmark_dict:
            raise ValueError(f"基准 '{self.benchmark_name}' 不存在。可用选项: {list(benchmark_dict.keys())}")
        
        task_suite = benchmark_dict[self.benchmark_name]()
        
        # 验证 task_id 是否在有效范围内
        if not (0 <= self.task_id < task_suite.n_tasks):
            raise ValueError(
                f"Task ID {self.task_id} 超出范围。对于基准 '{self.benchmark_name}'，"
                f"有效范围是 [0, {task_suite.n_tasks - 1}]。"
            )
        
        self.task = task_suite.get_task(self.task_id)

        # 2. 初始化底层模拟环境
        self.env, self.task_description = get_libero_env(self.task, self.model_family, resolution=256)
        
        # 3. 定义动作和观测空间
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "full_image": spaces.Box(low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8),
                "wrist_image": spaces.Box(low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8),
                "state": spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32),
            }
        )

        # 4. 加载初始状态
        self.initial_states = task_suite.get_task_init_states(self.task_id)
        self.episode_idx = 0

        # 5. 设置 episode 参数
        self.max_episode_steps = TASK_MAX_STEPS.get(self.benchmark_name, 400)
        self.step_count = 0

    def _prepare_obs(self, obs: Dict) -> Dict[str, np.ndarray]:
        """将来自模拟器的原始观测数据转换为策略所需的格式。"""
        img = get_libero_image(obs)
        wrist_img = get_libero_wrist_image(obs)
        img_resized = resize_image_for_policy(img, (self.image_size, self.image_size))
        wrist_img_resized = resize_image_for_policy(wrist_img, (self.image_size, self.image_size))
        proprio_state = np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        )
        observation = {
            "full_image": img_resized,
            "wrist_image": wrist_img_resized,
            "state": proprio_state.astype(np.float32),
        }
        self.last_full_image = img 
        return observation

    def reset(self, seed: int = None, options: Dict = None) -> Tuple[Dict, Dict]:
        """重置环境到一个新的 episode。"""
        super().reset(seed=seed)
        self.env.reset()
        initial_state = self.initial_states[self.episode_idx % len(self.initial_states)]
        self.episode_idx += 1
        obs = self.env.set_init_state(initial_state)
        for _ in range(self.num_steps_wait):
            obs, _, _, _ = self.env.step(get_libero_dummy_action(self.model_family))
        self.step_count = 0
        observation = self._prepare_obs(obs)
        info = {"task_description": self.task_description, "is_success": False}
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """在环境中执行一个动作。"""
        processed_action = normalize_gripper_action(action.copy(), binarize=True)
        if self.model_family == "openvla":
            processed_action = invert_gripper_action(processed_action)
        obs, reward, done, info = self.env.step(processed_action.tolist())
        self.step_count += 1
        observation = self._prepare_obs(obs)
        terminated = bool(done)
        truncated = self.step_count >= self.max_episode_steps
        info["is_success"] = terminated
        info["task_description"] = self.task_description
        return observation, float(reward), terminated, truncated, info

    def render(self) -> np.ndarray:
        """返回当前环境的 RGB 图像。"""
        if self.render_mode == "rgb_array":
            if hasattr(self, 'last_full_image'):
                return self.last_full_image
            else:
                return np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            raise ValueError(f"不支持的渲染模式: {self.render_mode}")

    def close(self):
        """关闭环境并释放资源。"""
        self.env.close()

    def get_name(self) -> str:
        """返回当前任务的名称。"""
        return f"{self.task.name} (ID: {self.task_id})"
    

class LiberoEnvChunk(LiberoEnvWrapper):
    def __init__(
        self,
        benchmark_name: str,
        task_id: int,
        image_size: int = 224,
        model_family: str = "openvla",
        render_mode: str = "rgb_array",
        num_steps_wait: int = 10,
        chunk_num: int = 8,
    ):
        super().__init__(benchmark_name, task_id, image_size, model_family, render_mode, num_steps_wait)
        self.chunk_num = chunk_num
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7*chunk_num,), dtype=np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        action_reshape = action.reshape(self.chunk_num, -1)
        total_reward = 0.0
        for sub_act in action_reshape:
            obs, reward, terminated, truncated, info = super().step(sub_act)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, float(total_reward), terminated, truncated, info


if __name__ == '__main__':
    # --- 这是一个演示如何使用 LiberoEnvWrapper 的示例 ---
    print("正在初始化 LiberoEnvWrapper...")

    # 参数设置
    # 请确保您已经下载了 'libero_spatial' 数据集
    BENCHMARK = "libero_spatial"
    # 直接使用任务 ID 进行初始化
    # 例如，ID=3 对应 "pick_up_the_black_bowl_and_place_it_in_the_basket"
    TASK_ID = 3 

    try:
        env = LiberoEnvWrapper(
            benchmark_name=BENCHMARK,
            task_id=TASK_ID,
            image_size=224,
            render_mode="rgb_array"
        )
    except Exception as e:
        print("\n--- 初始化失败 ---")
        print(f"错误: {e}")
        print("\n请确保：")
        print("1. 您已按照 LIBERO 的说明安装了所有依赖项。")
        print("2. 您已下载了 'libero_spatial' 数据集并放置在正确的位置。")
        print("3. 当前工作目录正确，以便脚本能够找到必要的工具函数。")
        sys.exit(1)

    print("\n--- 环境信息 ---")
    print(f"任务 ID: {env.task_id}")
    print(f"任务名称: {env.task.name}")
    print(f"任务描述: {env.task_description}")
    print(f"动作空间: {env.action_space}")
    print(f"观测空间: {env.observation_space}")
    print(f"最大步数: {env.max_episode_steps}")
    print("------------------\n")

    # 重置环境
    print("正在重置环境...")
    obs, info = env.reset()
    print("环境重置成功。")

    # 运行一个 episode，执行随机动作
    terminated, truncated = False, False
    total_reward = 0.0
    step = 0
    while not (terminated or truncated):
        random_action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(random_action)
        total_reward += reward
        step += 1
        print(f"Step {step}: Reward={reward:.4f}, Terminated={terminated}, Truncated={truncated}, Success={info.get('is_success', False)}")

    print("\nEpisode 结束。")
    print(f"总步数: {step}")
    print(f"总奖励: {total_reward}")

    # 关闭环境
    env.close()
    print("环境已关闭。")