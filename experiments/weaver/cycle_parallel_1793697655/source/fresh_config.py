"""Explicit, independent FRESH candidate switches; no runtime or GPU imports."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
import math


@dataclass(frozen=True)
class FreshConfig:
    protocol: str = 'weaver_fresh_v1'
    dense_reward: bool = False
    signal_telemetry: bool = False
    refresh_gae: bool = False
    replay_age_limits: bool = False
    max_policy_age: int = 16
    max_world_age: int = 3
    max_sample_uses: int = 4
    seed_balancing: bool = False
    early_seed_context: bool = False
    seed_scheduler_mode: str = 'balanced'
    seed_scheduler_seed: int = 20261028
    real_policy_replay: bool = False
    real_fraction: float = 0.25
    jit_gae_mode: str = 'off'
    advantage_norm: str = 'optimizer_window'
    advantage_ema_beta: float = 0.9
    advantage_scale_floor: float = 0.001
    jit_context_micro: int = 2
    policy_sampling: str = 'chunks'

    def __post_init__(self):
        if self.policy_sampling not in ('chunks', 'imagined_trajectories'):
            raise ValueError('Unknown Policy sampling protocol')
        if self.policy_sampling == 'imagined_trajectories' and (
                self.jit_gae_mode != 'fused' or not self.refresh_gae
                or not self.replay_age_limits or not self.real_policy_replay
                or self.real_fraction != .25 or self.max_policy_age != 16
                or self.max_world_age != 3 or self.max_sample_uses != 4):
            raise ValueError('Trajectory sampling requires fused JIT and the reviewed FRESH recipe')
        if self.jit_gae_mode not in ('off', 'trainer_separate', 'fused'):
            raise ValueError('Unknown JIT GAE mode')
        if self.advantage_norm not in ('optimizer_window', 'ema_zscore', 'ema_rms'):
            raise ValueError('Unknown advantage normalization')
        if self.jit_gae_mode == 'off' and self.advantage_norm != 'optimizer_window':
            raise ValueError('EMA requires trainer-owned GAE')
        if self.jit_gae_mode != 'off' and not self.refresh_gae:
            raise ValueError('JIT requires current-critic FRESH GAE')
        for name in ('advantage_ema_beta', 'advantage_scale_floor'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('Invalid EMA configuration')
        if not 0 <= self.advantage_ema_beta < 1 or self.advantage_scale_floor <= 0:
            raise ValueError('Invalid EMA beta/scale floor')
        if type(self.jit_context_micro) is not int or not 1 <= self.jit_context_micro <= 8:
            raise ValueError('JIT context micro must be bounded to 1..8')
        if self.protocol != 'weaver_fresh_v1':
            raise ValueError('Unsupported FRESH protocol')
        for name in ('dense_reward', 'signal_telemetry', 'refresh_gae', 'replay_age_limits',
                     'seed_balancing', 'early_seed_context', 'real_policy_replay'):
            if type(getattr(self, name)) is not bool:
                raise ValueError('FRESH %s must be an explicit boolean' % name)
        for name in ('max_policy_age', 'max_world_age', 'max_sample_uses'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError('FRESH %s must be a positive integer' % name)
        if self.seed_scheduler_mode not in ('balanced', 'random'):
            raise ValueError('FRESH seed_scheduler_mode must be balanced or random')
        if type(self.seed_scheduler_seed) is not int or self.seed_scheduler_seed < 0:
            raise ValueError('FRESH seed_scheduler_seed must be a nonnegative integer')
        if (isinstance(self.real_fraction, bool) or not isinstance(self.real_fraction, (int, float))
                or not math.isfinite(self.real_fraction) or not 0 < self.real_fraction < 1):
            raise ValueError('FRESH real_fraction must be finite and strictly between zero and one')

    @classmethod
    def from_mapping(cls, value=None):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError('FRESH configuration must be a mapping')
        unknown = set(value) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError('Unknown FRESH options: %s' % sorted(unknown))
        return cls(**dict(value))

    @classmethod
    def from_world_config(cls, world):
        if isinstance(world, Mapping):
            options = world.get('backend_options', {})
        else:
            options = getattr(world, 'backend_options', {})
        return cls.from_mapping(options.get('fresh'))

    @property
    def enabled(self):
        return any((self.dense_reward, self.signal_telemetry, self.refresh_gae,
                    self.replay_age_limits, self.seed_balancing,
                    self.early_seed_context, self.real_policy_replay))

    def to_dict(self):
        return asdict(self)

    def validate_reward_binding(self, specification):
        dense = (specification or {}).get('mode') == 'libero_discounted_success_value_v1'
        if dense != self.dense_reward:
            raise ValueError('FRESH dense_reward must match its explicit native reward specification')
        return self
