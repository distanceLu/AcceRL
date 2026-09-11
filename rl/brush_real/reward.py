"""Dense reward interface for imagined Brush rollouts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SmokeDenseReward:
    """Temporary dense reward called once for every 10 Hz policy action.

    Replace :meth:`__call__` with the real task rule. Keeping this interface
    makes reward evaluation local to CPU rollout/evaluation workers; there is
    intentionally no learned reward model and no reward-model optimizer.
    """

    value: float = 1.0

    def __call__(
        self,
        *,
        observation: torch.Tensor,
        action: np.ndarray,
        proprio: np.ndarray,
        low_level_step: int,
    ) -> float:
        if observation.ndim != 4:
            raise ValueError(f"Expected [camera,channel,height,width], got {observation.shape}")
        if np.asarray(action).shape != (7,):
            raise ValueError(f"Expected a 7-D action, got {np.asarray(action).shape}")
        if np.asarray(proprio).shape != (8,):
            raise ValueError(f"Expected 8-D proprio, got {np.asarray(proprio).shape}")
        del low_level_step
        return float(self.value)
