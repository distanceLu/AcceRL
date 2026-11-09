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

    def __post_init__(self):
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
