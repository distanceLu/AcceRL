"""Pure utilities shared by online world-model training and its unit tests."""

import hashlib
import math

import numpy as np


def frame_align_transition_actions(
    transition_actions: np.ndarray, num_observations: int
) -> np.ndarray:
    """Convert T-1 transition actions to T frame-aligned actions.

    Frame 0 receives a zero/reset action. Frame ``t`` receives the action that
    produced it from frame ``t-1``, matching Ctrl-World's offline dataset.
    """
    actions = np.asarray(transition_actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"Expected transition actions [T-1,A], got {actions.shape}")
    expected = int(num_observations) - 1
    if actions.shape[0] != expected:
        raise ValueError(
            f"Expected {expected} transition actions for {num_observations} "
            f"observations, got {actions.shape[0]}"
        )
    reset_action = np.zeros((1, actions.shape[1]), dtype=np.float32)
    return np.concatenate([reset_action, actions], axis=0)


def stable_episode_fraction(episode_id: str) -> float:
    """Return a deterministic value in [0, 1) for episode-level splitting."""
    digest = hashlib.sha1(str(episode_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") / float(2**64)


def cosine_warmup_lr(
    current_step: int,
    peak_lr: float,
    warmup_steps: int,
    total_steps: int,
    start_step: int = 0,
) -> float:
    """Linear warmup followed by cosine decay over optimizer update steps."""
    current_step = int(current_step)
    warmup_steps = int(warmup_steps)
    total_steps = int(total_steps)
    start_step = int(start_step)
    if total_steps <= start_step:
        raise ValueError("total_steps must be greater than start_step")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if current_step < start_step:
        return 0.0
    effective_step = current_step - start_step
    if warmup_steps > 0 and effective_step < warmup_steps:
        return float(peak_lr) * effective_step / warmup_steps
    decay_steps = max(1, total_steps - start_step - warmup_steps)
    progress = min(max((effective_step - warmup_steps) / decay_steps, 0.0), 1.0)
    return float(peak_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
