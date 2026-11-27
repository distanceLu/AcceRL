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


def _pool_content(states):
    states = _state_array(states)
    if states.ndim != 2 or not len(states):
        raise ValueError('Expected a nonempty [initial_state, state_dimension] pool')
    return {'count': len(states), 'shape': list(states.shape), 'sha256': _digest(states),
            'hash_semantics': 'canonical_array_shape_dtype_bytes'}


def initial_state_pool_metadata(env):
    result = dict(_pool_content(env.initial_states),
                  max_action_steps=int(getattr(env, '_unbounded_episode_steps', env.max_episode_steps)),
                  reset_steps=int(env.num_steps_wait))
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


def spatial_task_state(global_id):
    if type(global_id) is not int or not 0 <= global_id < 500:
        raise ValueError('Spatial task assignment must index one of 500 task/state pairs')
    return divmod(global_id, 50)


def spatial_pool_digest(names, tasks):
    if (not isinstance(names, list) or len(names) != 10 or not all(isinstance(n, str) and n for n in names)
            or len(set(names)) != 10 or not isinstance(tasks, dict) or set(tasks) != {str(i) for i in range(10)}):
        raise ValueError('Spatial pool must name all ten registered tasks')
    identity = []
    for task, name in enumerate(names):
        pool = tasks[str(task)]
        if pool['count'] != 50 or len(pool['shape']) != 2 or pool['shape'][0] != 50:
            raise ValueError(f'Spatial task {task} must have 50 initial states')
        identity.append(('libero_spatial', task, name, pool['sha256']))
    return hashlib.sha256(json.dumps(identity, separators=(',', ':')).encode()).hexdigest()


class SpatialTrainingEnvs:
    """Resolve a globally leased Spatial state, keeping one EGL scene per worker."""

    def __init__(self, first_env, suite, create_env, horizons_by_task=None):
        if first_env.benchmark_name != 'libero_spatial' or first_env.task_id != 0 or suite.n_tasks != 10:
            raise ValueError('Expected all ten LIBERO-Spatial tasks in suite order')
        self.env = first_env
        self.create_env = create_env
        self._on_scene_open = None
        first_pool = initial_state_pool_metadata(first_env)
        limits = {k: first_pool[k] for k in ('reset_steps', 'max_action_steps', 'benchmark_action_horizon')}
        if horizons_by_task is None:
            horizons_by_task = [first_pool['max_action_steps']] * suite.n_tasks
        if (len(horizons_by_task) != suite.n_tasks or any(type(h) is not int or
                not 0 < h <= first_pool['benchmark_action_horizon'] for h in horizons_by_task)):
            raise ValueError('Expected one valid action horizon per Spatial task')
        tasks = {}
        names = []
        for task in range(10):
            content = _pool_content(suite.get_task_init_states(task))
            tasks[str(task)] = dict(content, **limits)
            tasks[str(task)]['max_action_steps'] = horizons_by_task[task]
            names.append(suite.get_task(task).name)
        if tasks['0'] != first_pool or first_env.task.name != names[0]:
            raise ValueError('Simulator task or pool differs from the LIBERO-Spatial registry')
        digest = spatial_pool_digest(names, tasks)
        limits['max_action_steps'] = max(horizons_by_task)
        self.pool = dict(count=500, sha256=digest, task_suite='libero_spatial', task_names=names,
                         task_pools=tasks, hash_semantics='ordered_suite_task_name_and_pool_sha256', **limits)

    def on_scene_open(self, callback):
        """Apply a hook to the active scene and each replacement before its first reset."""
        callback(self.env)
        self._on_scene_open = callback

    def reset(self, assigned):
        global_id = assigned['initial_state_id']
        task, local = spatial_task_state(global_id)
        if task != self.env.task_id:
            self.env.close()
            env = self.create_env(task)
            try:
                set_real_training_action_horizon(env, self.pool['task_pools'][str(task)]['max_action_steps'])
                if (env.benchmark_name != self.pool['task_suite'] or env.task_id != task
                        or env.task.name != self.pool['task_names'][task]
                        or initial_state_pool_metadata(env) != self.pool['task_pools'][str(task)]):
                    raise ValueError(f'Spatial task {task} simulator or initial-state pool differs from registry')
                if self._on_scene_open is not None:
                    self._on_scene_open(env)
            except BaseException:
                env.close()
                raise
            self.env = env
        obs, info = reset_with_initial_state(self.env, local, assigned['env_seed'], assigned['max_action_steps'])
        info.update(initial_state_id=global_id, task_suite='libero_spatial', task_id=task,
                    task_initial_state_id=local)
        return obs, info
