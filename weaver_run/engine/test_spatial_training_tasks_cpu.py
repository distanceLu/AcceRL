"""Exercise the production task/state allocation, reset, and completion path without EGL."""
import os
import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from initial_state_env import initial_state_pool_metadata, set_real_training_action_horizon
from pipeline_budget import PipelineBudgetConfig
from real_collection import RealCollectionCoordinator


class Suite:
    n_tasks = 10

    def get_task_init_states(self, task):
        return np.arange(50 * (3 + task), dtype=np.float32).reshape(50, -1) + task

    def get_task(self, task):
        return SimpleNamespace(name=f'spatial-{task}')


class Env:
    benchmark_name = 'libero_spatial'
    num_steps_wait = 10

    def __init__(self, task_id, suite):
        self.task_id = task_id
        self.task = suite.get_task(task_id)
        self.task_description = f'execute task {task_id}'
        self.initial_states = suite.get_task_init_states(task_id)
        self.max_episode_steps = 220
        self.env = SimpleNamespace(seed=lambda seed: None)
        self.closed = False

    def reset(self, seed):
        if self.closed:
            raise RuntimeError('using closed scene')
        return self.initial_states[self.episode_idx], {}

    def close(self):
        self.closed = True

    def get_name(self):
        return self.task.name


class AllSpatialTests(unittest.TestCase):
    def test_rollout_worker_uses_leased_task_instruction_and_state(self):
        from initial_state_env import SpatialTrainingEnvs
        from test_pipeline_producers_cpu import make_worker
        suite = Suite()
        first = Env(0, suite)
        set_real_training_action_horizon(first, 150)
        worker, _ = make_worker([], None, None, mode='real')
        worker.spatial_tasks = SpatialTrainingEnvs(first, suite, lambda task: Env(task, suite))
        obs, info = worker._reset_and_select_env(initial_state=dict(initial_state_id=456,
                                                                    env_seed=1, max_action_steps=150))
        self.assertEqual((worker.current_env_idx, worker.current_env_name, worker.task_description),
                         (9, 'spatial-9', 'execute task 9'))
        self.assertEqual((info['initial_state_id'], info['task_initial_state_id']), (456, 6))
        np.testing.assert_array_equal(obs, suite.get_task_init_states(9)[6])

    def test_real_leases_cover_each_task_state_once_and_preserve_evidence(self):
        from initial_state_env import SpatialTrainingEnvs
        suite = Suite()
        first = Env(0, suite)
        set_real_training_action_horizon(first, 150)
        with patch.dict(os.environ, {}, clear=True):
            environments = SpatialTrainingEnvs(first, suite, lambda task: Env(task, suite))
            pool = environments.pool
            self.assertEqual((pool['count'], pool['task_suite']), (500, 'libero_spatial'))
            self.assertEqual(pool['task_pools']['0'], initial_state_pool_metadata(first))
            coordinator = RealCollectionCoordinator(PipelineBudgetConfig(model_free=True, real_workers=8,
                max_real_in_flight=8, imagination_workers=0, max_imagine_in_flight=0,
                initial_state_enabled=True, initial_state_count=500, real_training_action_horizon=150))
            with self.assertRaisesRegex(ValueError, 'Spatial'):
                coordinator.acquire_real(0, dict(pool, task_suite=None))
            seen = Counter()
            for index in range(500):
                worker = index % 8
                lease = coordinator.acquire_real(worker, pool)
                state = lease['initial_state']
                obs, info = environments.reset(state)
                task, local = divmod(state['initial_state_id'], 50)
                self.assertEqual((info['task_suite'], info['task_id'], info['task_initial_state_id']),
                                 ('libero_spatial', task, local))
                self.assertEqual(environments.env.task_id, task)
                self.assertEqual(environments.env.max_episode_steps, 150)
                np.testing.assert_array_equal(obs, suite.get_task_init_states(task)[local])
                metadata = dict(info, action_steps=3, termination_reason='success', success=True)
                if index == 0:
                    with self.assertRaisesRegex(ValueError, 'task'):
                        coordinator.finish_real_episode(lease['token'], dict(episode_id=str(index),
                            metadata=dict(metadata, task_id=(task + 1) % 10)))
                coordinator.finish_real_episode(lease['token'],dict(episode_id=str(index), metadata=metadata))
                seen[task,local] += 1
            self.assertEqual(seen, Counter((task, state) for task in range(10) for state in range(50)))
            saved = coordinator.initial_state_checkpoint()['scheduler']['terminal']
            for entry in saved.values():
                self.assertEqual(divmod(entry['initial_state_id'], 50),
                    (entry['metadata']['task_id'], entry['metadata']['task_initial_state_id']))
            self.assertEqual(coordinator.pipeline_status()['initial_states']['consumed_env_steps'], 6500)

    def test_opened_task_must_match_its_pinned_asset(self):
        from initial_state_env import SpatialTrainingEnvs
        suite = Suite()
        first = Env(0, suite)
        set_real_training_action_horizon(first, 150)
        def wrong(task):
            env = Env(task, suite)
            env.initial_states[0, 1] += 1
            return env
        environments = SpatialTrainingEnvs(first, suite, wrong)
        with self.assertRaisesRegex(ValueError, 'pool'):
            environments.reset(dict(initial_state_id=51, env_seed=123, max_action_steps=150))
        self.assertTrue(first.closed)


if __name__ == '__main__':
    unittest.main()
