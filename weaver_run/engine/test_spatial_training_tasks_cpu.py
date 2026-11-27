"""Exercise the production task/state allocation, reset, and completion path without EGL."""
import os
import unittest
from collections import Counter
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

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

    def step(self, action):
        if self.closed:
            raise RuntimeError('using closed scene')
        return self.task_id, action

    def get_name(self):
        return self.task.name


class AllSpatialTests(unittest.TestCase):
    def test_per_task_horizons_follow_shuffled_state_and_refund_only_unused_steps(self):
        from initial_state_env import SpatialTrainingEnvs
        from initial_state_scheduler import CheckpointMismatchError, InitialStateScheduler
        suite = Suite()
        first = Env(0, suite)
        set_real_training_action_horizon(first, 150)
        horizons = [150] + [220] * 9
        scenes = SpatialTrainingEnvs(first, suite, lambda task: Env(task, suite), horizons)
        cfg = PipelineBudgetConfig(model_free=True, imagination_workers=0, max_imagine_in_flight=0,
            initial_state_enabled=True, initial_state_count=500, real_env_step_budget=500,
            real_training_action_horizon=150, real_training_action_horizons_by_task=horizons)
        with patch.dict(os.environ, {}, clear=True):
            owner = RealCollectionCoordinator(cfg)
            self.assertEqual(scenes.pool['max_action_steps'], 220)
            self.assertEqual([scenes.pool['task_pools'][str(t)]['max_action_steps'] for t in range(10)], horizons)
            wrong = dict(scenes.pool, task_pools=dict(scenes.pool['task_pools']))
            wrong['task_pools']['7'] = dict(wrong['task_pools']['7'], max_action_steps=150)
            with self.assertRaisesRegex(ValueError, 'horizon'):
                owner.acquire_real('bad', wrong)
            a, b = (owner.acquire_real(worker, scenes.pool) for worker in ('a', 'b'))
            self.assertEqual(owner.acquire_real('a', scenes.pool), a)
            self.assertEqual((a['initial_state']['initial_state_id'], b['initial_state']['initial_state_id']), (373, 10))
            self.assertEqual([x['initial_state']['reserved_env_steps'] for x in (a,b)], [230, 160])
            _, success = scenes.reset(a['initial_state'])
            self.assertEqual((success['task_action_horizon'], scenes.env.max_episode_steps), (220, 220))
            owner.finish_real_episode(a['token'], dict(episode_id='late-success',
                metadata=dict(success, action_steps=180, success=True, termination_reason='success')))
            c = owner.acquire_real('c', scenes.pool)
            _, short = scenes.reset(c['initial_state'])
            self.assertEqual((short['task_id'], short['task_action_horizon'], short['budget_action_limit']), (5, 220, 140))
            owner.finish_real_episode(c['token'],dict(episode_id='budget-cut',
                metadata=dict(short, action_steps=140, success=False, termination_reason='real_env_budget')))
            _, ordinary = scenes.reset(b['initial_state'])
            self.assertEqual((ordinary['task_id'], ordinary['task_action_horizon'], ordinary['budget_action_limit']), (0,150,150))
            owner.finish_real_episode(b['token'],dict(episode_id='task-zero-timeout',
                metadata=dict(ordinary, action_steps=150, success=False, termination_reason='task_horizon')))
            journal = owner.initial_state_checkpoint()['scheduler']
            self.assertEqual(journal['counters']['consumed_env_steps'], 500)
            self.assertEqual(journal['terminal'][c['initial_state']['token']]['metadata']['task_action_horizon'], 220)
            self.assertTrue(owner.acquire_real('d', scenes.pool)['budget_exhausted'])
            restored = InitialStateScheduler.from_snapshot(journal)
            self.assertEqual(restored.status()['consumed_env_steps'], 500)
            with self.assertRaises(CheckpointMismatchError):
                InitialStateScheduler.from_snapshot(journal, action_horizons_by_state=[150] * 500)

    def test_cpu_telemetry_follows_real_scene_across_switches(self):
        from capture_hooks import _patch_envs
        from initial_state_env import SpatialTrainingEnvs
        suite = Suite()
        first = Env(0, suite)
        set_real_training_action_horizon(first, 150)
        scenes = SpatialTrainingEnvs(first, suite, lambda task: Env(task, suite))
        telemetry = Mock()
        telemetry.span.side_effect = lambda *args, **kwargs: nullcontext()
        owner = SimpleNamespace(envs=[], spatial_tasks=scenes, telemetry=telemetry)
        _patch_envs(owner)
        for global_id in (0, 51, 52, 1):
            _patch_envs(owner)
            task, state = divmod(global_id, 50)
            obs, _ = scenes.reset(dict(initial_state_id=global_id, env_seed=42, max_action_steps=150))
            np.testing.assert_equal(obs, suite.get_task_init_states(task)[state])
            self.assertEqual((task, 3), scenes.env.step(3))
        self.assertEqual(telemetry.span.call_args_list, [
            call('cpu_env_reset', cuda=False), call('cpu_env_step', cuda=False)] * 4)

    def test_cpu_telemetry_for_fixed_task_env_is_installed_once(self):
        from capture_hooks import _patch_envs
        env = Env(0, Suite())
        env.episode_idx = 0
        telemetry = Mock()
        telemetry.span.side_effect = lambda *args, **kwargs: nullcontext()
        owner = SimpleNamespace(envs=[env], telemetry=telemetry)
        _patch_envs(owner)
        _patch_envs(owner)
        env.reset(seed=2)
        self.assertEqual(env.step(7), (0, 7))
        self.assertEqual(telemetry.span.call_args_list, [
            call('cpu_env_reset', cuda=False), call('cpu_env_step', cuda=False)])

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
