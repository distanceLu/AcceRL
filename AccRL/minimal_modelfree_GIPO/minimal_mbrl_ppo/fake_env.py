from __future__ import annotations

import numpy as np


class FakeEnv:
    """A tiny image-based environment used for architecture study."""

    def __init__(
        self,
        task_id: int = 0,
        obs_shape: tuple[int, int, int] = (3, 16, 16),
        action_dim: int = 3,
        max_steps: int = 50,
    ) -> None:
        self.task_id = int(task_id)
        self.obs_shape = tuple(int(v) for v in obs_shape)
        self.action_dim = int(action_dim)
        self.max_steps = int(max_steps)
        self.task_description = f"fake task {self.task_id}"
        self._rng = np.random.default_rng()
        self._step = 0
        self._episode_return = 0.0
        self._latent = np.zeros(self.action_dim, dtype=np.float32)

    def get_name(self) -> str:
        return f"fake_task_{self.task_id}"

    def _make_obs(self) -> dict:
        c, h, w = self.obs_shape
        base = self._rng.normal(loc=0.0, scale=0.35, size=self.obs_shape).astype(np.float32)
        for idx in range(min(c, self.action_dim)):
            base[idx] += self._latent[idx] * 0.4
        base = np.clip(base, -1.0, 1.0)
        return {
            "image": base,
            "task_id": np.int64(self.task_id),
            "step": np.int64(self._step),
        }

    def reset(self, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        self._step = 0
        self._episode_return = 0.0
        self._latent = self._rng.normal(size=self.action_dim).astype(np.float32)
        obs = self._make_obs()
        info = {
            "task_description": self.task_description,
            "task_id": self.task_id,
            "is_success": 0.0,
        }
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.action_dim:
            raise ValueError(f"Expected action shape ({self.action_dim},), got {action.shape}")

        self._step += 1
        self._latent = 0.85 * self._latent + 0.15 * action
        reward = float(
            self._rng.normal(loc=0.15, scale=0.25) + 0.10 * np.tanh(action).mean()
        )
        self._episode_return += reward

        terminated = self._step >= self.max_steps
        truncated = False
        info = {
            "task_description": self.task_description,
            "task_id": self.task_id,
            "is_success": float(self._episode_return > 0.0) if terminated else 0.0,
            "episode_return": self._episode_return,
        }
        return self._make_obs(), reward, terminated, truncated, info
