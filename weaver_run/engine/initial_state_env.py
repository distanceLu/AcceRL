"""Explicit LIBERO state selection, separate from simulator random seeding.

Reuse the existing wrapper reset/observation/settling code. The assigned index
is installed immediately before reset, so no worker-local progression chooses
the initial state. Pool and per-state hashes describe canonical state bytes.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


def _state_array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    result = np.ascontiguousarray(value)
    if result.dtype.hasobject or not np.isfinite(result).all():
        raise ValueError('Initial states must be finite numeric arrays')
    return result


def _digest(value):
    value = _state_array(value)
    header = json.dumps({'shape': list(value.shape), 'dtype': value.dtype.str}, sort_keys=True).encode()
    return hashlib.sha256(header + b'\0' + value.tobytes()).hexdigest()


def set_real_training_action_horizon(env, action_horizon):
    """Set one collector's finite task deadline without changing Eval defaults.

    The benchmark horizon is retained as provenance. Subsequent per-episode
    budget limits may be shorter than this training horizon and remain censored.
    """
    if action_horizon is None:
        return
    benchmark_horizon = int(getattr(env, '_benchmark_episode_steps',
        getattr(env, '_unbounded_episode_steps', env.max_episode_steps)))
    if type(action_horizon) is not int or not 0 < action_horizon <= benchmark_horizon:
        raise ValueError('Real training horizon must be positive and at most the benchmark horizon')
    env._benchmark_episode_steps = benchmark_horizon
    env._unbounded_episode_steps = action_horizon
    env.max_episode_steps = action_horizon


def initial_state_pool_metadata(env):
    states = _state_array(env.initial_states)
    if states.ndim != 2 or not len(states):
        raise ValueError('Expected a nonempty [initial_state, state_dimension] pool')
    result = {'count': len(states), 'shape': list(states.shape), 'sha256': _digest(states),
            'hash_semantics': 'canonical_array_shape_dtype_bytes',
            'max_action_steps': int(getattr(env, '_unbounded_episode_steps', env.max_episode_steps)),
            'reset_steps': int(env.num_steps_wait)}
    if hasattr(env, '_benchmark_episode_steps'):
        result['benchmark_action_horizon'] = int(env._benchmark_episode_steps)
    return result


def reset_with_initial_state(env, initial_state_id, env_seed, max_action_steps=None):
    if type(initial_state_id) is not int or not 0 <= initial_state_id < len(env.initial_states):
        raise ValueError('initial_state_id is outside the loaded state pool')
    if type(env_seed) is not int or not 0 <= env_seed < 2**32:
        raise ValueError('env_seed must be a uint32 integer')
    original_horizon = int(getattr(env, '_unbounded_episode_steps', env.max_episode_steps))
    if max_action_steps is None:
        max_action_steps = original_horizon
    if type(max_action_steps) is not int or not 0 < max_action_steps <= original_horizon:
        raise ValueError('Action limit must be positive and at most the original task horizon')
    env._unbounded_episode_steps = original_horizon
    env.max_episode_steps = max_action_steps
    env.env.seed(env_seed)
    env.episode_idx = initial_state_id
    observation, info = env.reset(seed=env_seed)
    info = dict(info, initial_state_id=initial_state_id, env_seed=env_seed,
                initial_state_hash=_digest(env.initial_states[initial_state_id]),
                reset_env_steps=int(env.num_steps_wait),
                budget_action_limit=max_action_steps,
                task_action_horizon=original_horizon)
    if hasattr(env, '_benchmark_episode_steps'):
        info['benchmark_action_horizon'] = int(env._benchmark_episode_steps)
    return observation, info


reset_explicit = reset_with_initial_state
