"""Pure Python reference for the existing success-potential reward algorithm."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


def _probability(value: Any) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Success output must be a probability in [0, 1], not progress")
    return value


def success_probability(logits: Sequence[float]) -> float:
    if len(logits) != 2 or any(not math.isfinite(float(value)) for value in logits):
        raise ValueError("Success classification requires two finite logits [failure, success]")
    # Stable two-class softmax, matching the original softmax(logits)[1].
    offset = max(float(logits[0]), float(logits[1]))
    failure, success = (math.exp(float(value) - offset) for value in logits)
    return success / (failure + success)


def terminal_from_logits(logits: Sequence[float]) -> bool:
    success_probability(logits)
    # torch.argmax chooses class zero on a tie; >= 0.5 would change behavior.
    return float(logits[1]) > float(logits[0])


def delta_success_rewards(initial_probability: float, probabilities: Sequence[float],
                          reward_scale: float = 1.0) -> tuple[float, ...]:
    if not math.isfinite(float(reward_scale)):
        raise ValueError("reward_scale must be finite")
    previous = _probability(initial_probability)
    rewards = []
    for probability in probabilities:
        current = _probability(probability)
        rewards.append((current - previous) * reward_scale)
        previous = current
    return tuple(rewards)


@dataclass(frozen=True)
class RewardSequence:
    rewards: tuple[float, ...]
    terminated: tuple[bool, ...]
    success_probabilities: tuple[float, ...]

    @property
    def bootstrap_allowed(self) -> bool:
        return not any(self.terminated)


@dataclass(frozen=True)
class DeltaSuccessRewardProvider:
    reward_scale: float = 1.0
    terminal_rule: str = "argmax"
    terminal_threshold: float = 0.5

    def __post_init__(self):
        if self.terminal_rule not in ("argmax", "success_probability"):
            raise ValueError("Unknown terminal rule")
        _probability(self.terminal_threshold)
        if not math.isfinite(float(self.reward_scale)):
            raise ValueError("reward_scale must be finite")

    def from_logits(self, initial_logits: Sequence[float],
                    logits: Sequence[Sequence[float]]) -> RewardSequence:
        probabilities = tuple(success_probability(row) for row in logits)
        terminals = (tuple(terminal_from_logits(row) for row in logits) if self.terminal_rule == "argmax"
                     else tuple(value >= self.terminal_threshold for value in probabilities))
        return RewardSequence(delta_success_rewards(success_probability(initial_logits), probabilities,
                                                   self.reward_scale), terminals, probabilities)

    def from_probabilities(self, initial_probability: float, probabilities: Sequence[float],
                           terminal_predictions: Sequence[bool] | None = None) -> RewardSequence:
        probabilities = tuple(_probability(value) for value in probabilities)
        if terminal_predictions is None:
            if self.terminal_rule == "argmax":
                raise ValueError("argmax semantics require logits or explicit calibrated terminal predictions")
            terminals = tuple(value >= self.terminal_threshold for value in probabilities)
        else:
            terminals = tuple(terminal_predictions)
            if len(terminals) != len(probabilities) or any(type(value) is not bool for value in terminals):
                raise ValueError("terminal_predictions must be one boolean per future frame")
        return RewardSequence(delta_success_rewards(initial_probability, probabilities, self.reward_scale),
                              terminals, probabilities)


def build_reward_provider(config: Any = None) -> DeltaSuccessRewardProvider:
    from .contracts import WorldBackendConfig
    if not isinstance(config, WorldBackendConfig):
        config = WorldBackendConfig.from_mapping(config)
    if config.reward_semantics == "native_progress":
        raise NotImplementedError("native_progress is a distinct algorithm; no implicit conversion to success rewards")
    return DeltaSuccessRewardProvider(config.reward_scale, config.terminal_rule, config.terminal_threshold)
