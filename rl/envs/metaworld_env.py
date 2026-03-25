from __future__ import annotations

import argparse
from enum import Enum
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


if not hasattr(gym.vector, "AutoresetMode"):
    class _CompatAutoresetMode(str, Enum):
        NEXT_STEP = "next_step"
        SAME_STEP = "same_step"
        DISABLED = "disabled"

    gym.vector.AutoresetMode = _CompatAutoresetMode


class MetaWorldWrapperDiscrete(gym.Env):
    """
    MetaWorld 单任务离散动作封装。

    - 外部输入离散 token 动作（每维 [0, bins-1]）
    - 内部映射为 MetaWorld 需要的连续动作（每维 [-1, 1]）
    - 观测直接使用底层环境的低维状态向量（通常为 39 维）
    """

    metadata = {"render_modes": ["rgb_array", None]}

    def __init__(
        self,
        env_name: str = "reach-v3",
        bins: int = 256,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        max_episode_steps: Optional[int] = None,
    ) -> None:
        super().__init__()
        if bins < 2:
            raise ValueError(f"bins 必须 >= 2，当前为 {bins}")

        self.env_name = env_name
        self.bins = int(bins)
        self._base_seed = seed
        self._episode_steps = 0

        # 导入 metaworld 以确保 gymnasium 的 Meta-World 环境被正确注册。
        import metaworld  # noqa: F401

        self.env = gym.make(
            "Meta-World/MT1",
            env_name=self.env_name,
            seed=seed,
            render_mode=render_mode,
        )

        self.max_episode_steps = int(
            max_episode_steps
            if max_episode_steps is not None
            else getattr(getattr(self.env, "spec", None), "max_episode_steps", 500)
        )

        # MetaWorld 动作空间为 4 维连续动作，这里暴露离散版动作空间。
        self.action_dim = int(np.prod(self.env.action_space.shape))
        self.action_space = spaces.MultiDiscrete([self.bins] * self.action_dim)

        # 先 reset 一次用于确定 observation_space，随后恢复计数器。
        init_obs, _ = self.env.reset(seed=seed)
        init_obs = np.asarray(init_obs, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=init_obs.shape,
            dtype=np.float32,
        )
        self._episode_steps = 0

    def _token_to_continuous(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action)
        if action.shape != (self.action_dim,):
            action = action.reshape(self.action_dim)

        # 若输入已是连续动作，则直接裁剪到合法范围。
        if np.issubdtype(action.dtype, np.floating):
            if np.all(action >= -1.01) and np.all(action <= 1.01):
                return np.clip(action, -1.0, 1.0).astype(np.float32)

        # 离散 token -> 连续动作: [0, bins-1] -> [-1, 1]
        token = np.clip(np.rint(action), 0, self.bins - 1).astype(np.float32)
        continuous = -1.0 + 2.0 * token / float(self.bins - 1)
        return continuous.astype(np.float32)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._episode_steps = 0
        obs = np.asarray(obs, dtype=np.float32)
        info = dict(info or {})
        info.setdefault("success", float(info.get("success", 0.0)))
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        continuous_action = self._token_to_continuous(action)
        obs, reward, terminated, truncated, info = self.env.step(continuous_action)
        self._episode_steps += 1

        # 与 Gymnasium 截断逻辑对齐：底层未给 truncated 时按步数兜底。
        if not truncated and self._episode_steps >= self.max_episode_steps:
            truncated = True

        obs = np.asarray(obs, dtype=np.float32)
        info = dict(info or {})
        info["success"] = float(info.get("success", 0.0))
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        if hasattr(self.env, "render"):
            return self.env.render()
        return None

    def close(self) -> None:
        if hasattr(self.env, "close"):
            self.env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="MetaWorldWrapperDiscrete 快速测试脚本")
    parser.add_argument("--env-name", type=str, default="reach-v3", help="MetaWorld 任务名，如 reach-v3")
    parser.add_argument("--bins", type=int, default=256, help="离散动作 bin 数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--episodes", type=int, default=3, help="测试 episode 数")
    parser.add_argument("--max-steps", type=int, default=300, help="每个 episode 的最大步数")
    args = parser.parse_args()

    env = MetaWorldWrapperDiscrete(
        env_name=args.env_name,
        bins=args.bins,
        seed=args.seed,
    )

    print(
        f"[MetaWorldTest] env={args.env_name} bins={args.bins} "
        f"obs_shape={env.observation_space.shape} action_dim={env.action_dim}"
    )

    try:
        for ep in range(args.episodes):
            ep_seed = args.seed + ep
            obs, info = env.reset(seed=ep_seed)
            done = False
            steps = 0
            ep_reward = 0.0
            last_info = info
            last_terminated = False
            last_truncated = False

            while not done and steps < args.max_steps:
                action_token = env.action_space.sample()
                obs, reward, terminated, truncated, last_info = env.step(action_token)
                ep_reward += reward
                steps += 1
                last_terminated = bool(terminated)
                last_truncated = bool(truncated)
                done = bool(terminated or truncated)

            print(
                f"[Episode {ep + 1}/{args.episodes}] "
                f"seed={ep_seed} steps={steps} reward={ep_reward:.4f} "
                f"success={float(last_info.get('success', 0.0)):.1f} "
                f"terminated={last_terminated} truncated={last_truncated}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
