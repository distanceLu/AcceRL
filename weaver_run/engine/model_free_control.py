"""CPU-only decisions at synchronized model-free training boundaries.

Budget observations count simulator actions and reset settling. Validation scores
come only from a completed fixed set; this module never selects or saves weights.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass


def _integer(name, value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('%s must be an integer >= %d' % (name, minimum))
    return value


@dataclass(frozen=True)
class ModelFreeControlConfig:
    interval_env_steps: int = 100000
    patience: int = 3
    min_env_steps: int = 300000
    min_delta_successes: int = 1

    def __post_init__(self):
        _integer('interval_env_steps', self.interval_env_steps)
        _integer('patience', self.patience, 1)
        _integer('min_env_steps', self.min_env_steps)
        _integer('min_delta_successes', self.min_delta_successes, 1)
        if self.min_delta_successes > 50:
            raise ValueError('min_delta_successes cannot exceed the 50 fixed cases')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_args(cls, args):
        defaults = cls().to_dict()
        online_eval = _integer('num_eval_workers', getattr(args, 'num_eval_workers', 0))
        if online_eval:
            defaults['interval_env_steps'] = 0
        values = {name: getattr(args, 'fixed_validation_' + name, None) for name in defaults}
        config = cls(**{name: default if values[name] is None else values[name]
                        for name, default in defaults.items()})
        if online_eval and config.interval_env_steps:
            raise ValueError('Periodic fixed validation requires zero online Eval workers')
        return config


class ModelFreeControl:
    """Track one P0 baseline and complete scores from unique newer versions."""
    CASE_COUNT = 50

    def __init__(self, config=None):
        self.config = config if config is not None else ModelFreeControlConfig()
        if not isinstance(self.config, ModelFreeControlConfig):
            raise ValueError('config must be ModelFreeControlConfig')
        self._validations = []
        self._best_successes = None
        self._best_policy_step = None
        self._improvement_reference = None
        self._stale_count = 0
        self._target_validation = None

    @property
    def enabled(self):
        return self.config.interval_env_steps > 0

    def should_validate(self, consumed_env_steps, policy_step, force=False):
        _integer('consumed_env_steps', consumed_env_steps)
        _integer('policy_step', policy_step)
        if type(force) is not bool:
            raise ValueError('force must be boolean')
        if not self.enabled:
            return False
        if not self._validations:
            return policy_step == 0
        last = self._validations[-1]
        # Final validation may reuse the latest score without double counting.
        if policy_step <= last['policy_step']:
            return False
        if consumed_env_steps < last['consumed_env_steps']:
            raise ValueError('consumed_env_steps cannot move backwards')
        return force or consumed_env_steps - last['consumed_env_steps'] >= self.config.interval_env_steps

    def observe(self, successes, *, policy_step, consumed_env_steps, cases=50):
        if not self.enabled:
            raise ValueError('fixed validation is disabled')
        for name, value in (('successes', successes), ('policy_step', policy_step),
                            ('consumed_env_steps', consumed_env_steps), ('cases', cases)):
            _integer(name, value)
        if cases != self.CASE_COUNT or successes > cases:
            raise ValueError('fixed validation requires exactly 50 completed cases and valid successes')
        if not self._validations:
            if policy_step != 0:
                raise ValueError('fixed validation must begin with the P0 baseline')
        else:
            last = self._validations[-1]
            if policy_step <= last['policy_step']:
                raise ValueError('validation policy version must increase; duplicate results are not observations')
            if consumed_env_steps < last['consumed_env_steps']:
                raise ValueError('consumed_env_steps cannot move backwards')
        # Validate everything before changing the state. A partial/failed call
        # cannot spend patience or make a validation look completed.
        row = dict(successes=successes, cases=cases, policy_step=policy_step,
                   consumed_env_steps=consumed_env_steps)
        if self._best_successes is None or successes > self._best_successes:
            self._best_successes, self._best_policy_step = successes, policy_step
        if (self._improvement_reference is None
                or successes - self._improvement_reference >= self.config.min_delta_successes):
            self._improvement_reference = successes
            self._stale_count = 0
        else:
            self._stale_count += 1
        self._validations.append(row)
        if successes == self.CASE_COUNT and self._target_validation is None:
            self._target_validation = row
        return self.summary()

    @property
    def target_reached(self):
        return self._target_validation is not None

    @property
    def plateau_reached(self):
        return bool(self.enabled and self._validations
                    and len(self._validations) - 1 >= self.config.patience
                    and self._stale_count >= self.config.patience
                    and self._validations[-1]['consumed_env_steps'] >= self.config.min_env_steps)

    def summary(self):
        return dict(enabled=self.enabled, config=self.config.to_dict(), case_count=self.CASE_COUNT,
                    completed_validations=len(self._validations),
                    post_baseline_validations=max(0, len(self._validations) - 1),
                    best_successes=self._best_successes, best_policy_step=self._best_policy_step,
                    improvement_reference_successes=self._improvement_reference,
                    stale_count=self._stale_count, plateau_reached=self.plateau_reached,
                    target_reached=self.target_reached,
                    target_validation=deepcopy(self._target_validation),
                    baseline=deepcopy(self._validations[0]) if self._validations else None,
                    last=deepcopy(self._validations[-1]) if self._validations else None,
                    validations=deepcopy(self._validations))


def collection_budget(status):
    """Compact scheduler evidence; reservations and unknown charges cannot stop.

The scheduler's legacy ``budget_exhausted`` includes reservations, so use the
coordinator's explicit drained decision and independently check its accounting.
An uninitialized pool is unknown, never a zero-sized exhausted budget.
"""
    if not isinstance(status, dict):
        raise ValueError('pipeline status must be a mapping')
    owner = status.get('budget') or {}
    scheduler = status.get('initial_states') or {}
    initialized = bool(scheduler)
    result = dict(initialized=initialized,
        max_env_steps=scheduler.get('max_env_steps', owner.get('config', {}).get('real_env_step_budget')),
        consumed_env_steps=scheduler.get('consumed_env_steps', 0),
        reserved_env_steps=scheduler.get('reserved_env_steps', 0),
        available_env_steps=scheduler.get('available_env_steps'),
        active_real=owner.get('active_real', 0), episodes_inflight=scheduler.get('episodes_inflight', 0),
        reset_steps=owner.get('reset_steps'), paused=owner.get('paused', False),
        stopped=owner.get('stopped', False), failure=owner.get('failure'),
        conservative_unknown_env_steps=scheduler.get('conservative_unknown_env_steps', 0),
        episodes_cancelled=scheduler.get('episodes_cancelled', 0))
    nonnegative = ('max_env_steps', 'consumed_env_steps', 'reserved_env_steps', 'available_env_steps',
                   'active_real', 'episodes_inflight', 'reset_steps',
                   'conservative_unknown_env_steps', 'episodes_cancelled')
    valid = all(type(result[key]) is int and result[key] >= 0 for key in nonnegative)
    result['budget_exhausted'] = bool(initialized and valid
        and owner.get('budget_exhausted') is True and not result['failure']
        and result['max_env_steps'] > 0
        and all(result[key] == 0 for key in ('active_real', 'episodes_inflight',
            'reserved_env_steps', 'conservative_unknown_env_steps', 'episodes_cancelled'))
        and not scheduler.get('recovery_required_tokens')
        and result['available_env_steps'] <= result['reset_steps']
        and result['consumed_env_steps'] + result['available_env_steps'] == result['max_env_steps'])
    return result


def budget_exhausted(status):
    return collection_budget(status)['budget_exhausted']
