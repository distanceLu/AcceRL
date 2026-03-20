import numpy as np


class FakeEnv:
    """A tiny stand-in for a robot environment.

    The observation is a dict with a single dense vector under the ``state`` key.
    The action is expected to be a 1D array with shape ``(action_dim,)``.
    """

    def __init__(
        self,
        task_id: int = 0,
        obs_dim: int = 16,
        action_dim: int = 3,
        max_steps: int = 50,
    ) -> None:
        self.task_id = int(task_id)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.max_steps = int(max_steps)
        self.task_description = f"fake task {self.task_id}"
        self._rng = np.random.default_rng()
        self._step_count = 0
        self._episode_return = 0.0
        self._state = np.zeros(self.obs_dim, dtype=np.float32)

    def get_name(self) -> str:
        return f"fake_task_{self.task_id}"

    def _sample_obs(self) -> dict:
        noise = self._rng.normal(loc=0.0, scale=0.2, size=self.obs_dim).astype(np.float32)
        drift = np.float32(np.sin(self._step_count / 5.0 + self.task_id))
        state = self._state + noise + drift
        return {
            "state": state.astype(np.float32),
            "task_id": np.int64(self.task_id),
            "step": np.int64(self._step_count),
        }

    def reset(self, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        self._step_count = 0
        self._episode_return = 0.0
        self._state = self._rng.normal(size=self.obs_dim).astype(np.float32)
        obs = self._sample_obs()
        info = {
            "task_id": self.task_id,
            "task_description": self.task_description,
            "is_success": 0.0,
        }
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != self.action_dim:
            raise ValueError(
                f"Expected action_dim={self.action_dim}, got shape {action.shape}"
            )

        self._step_count += 1
        reward = float(self._rng.normal(loc=0.1, scale=0.5) - 0.05 * np.square(action).mean())
        self._episode_return += reward
        self._state = 0.8 * self._state + 0.2 * np.pad(
            action,
            (0, max(0, self.obs_dim - self.action_dim)),
            mode="wrap",
        )[: self.obs_dim]

        terminated = self._step_count >= self.max_steps
        truncated = False
        success = float(self._episode_return > 0.0) if terminated else 0.0
        info = {
            "task_id": self.task_id,
            "task_description": self.task_description,
            "is_success": success,
            "episode_return": self._episode_return,
        }
        obs = self._sample_obs()
        return obs, reward, terminated, truncated, info
