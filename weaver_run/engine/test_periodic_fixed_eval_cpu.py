"""Finite validation exercises real state selection and action accounting on CPU."""
import importlib
import importlib.util
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from initial_state_env import initial_state_pool_metadata
from frozen_policy_eval import make_plan

MODULE = 'periodic_fixed_eval'


class TinyEnv:
    def __init__(self, *, success_at=3, broken_horizon=False, **kwargs):
        self.benchmark_name = 'libero_spatial'
        self.task_id = 0
        self.task_description = 'pick bowl'
        self.initial_states = np.arange(200, dtype=np.float64).reshape(50, 4)
        self.max_episode_steps, self.num_steps_wait = 220, 10
        self.env = SimpleNamespace(seed=lambda value: self.seeds.append(value))
        self.seeds, self.ids = [], []
        self.steps = self.total_steps = self.closed = 0
        self.success_at, self.broken_horizon = success_at, broken_horizon

    def reset(self, seed=None):
        self.steps = 0
        self.ids.append(self.episode_idx)
        self.episode_idx += 1
        return {'state_id': self.ids[-1]}, {'is_success': False}

    def step(self, action):
        self.steps += 1
        self.total_steps += 1
        success = self.success_at is not None and self.steps >= self.success_at
        truncated = not self.broken_horizon and self.steps >= self.max_episode_steps
        return {'state_id': self.ids[-1]}, float(success), success, truncated, {'is_success': success}

    def close(self):
        self.closed += 1


def response(version=7, count=8):
    return {'_capture_payload': (np.zeros((count, 7)), np.zeros((count, 7)),
                                np.zeros((count, 7), dtype=np.int64), None, None),
            'versions': {'policy': version}, 'request_id': 'request-test'}


class PeriodicFixedEvalTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec(MODULE), 'finite fixed evaluator is missing')
        self.api = importlib.import_module(MODULE)

    def run_eval(self, env=None, request=None, **kwargs):
        return self.api.run_fixed_evaluation(env or TinyEnv(), request or (lambda *args: response()),
                                             policy_version=7, timeout_seconds=30, **kwargs)

    def test_complete_round_uses_all_reference_cases_and_real_state_hashes(self):
        env = TinyEnv()
        calls = []
        def infer(observation, case, remaining):
            calls.append((dict(case), remaining))
            return response()
        result = self.run_eval(env, infer)
        reference = make_plan([dict(name=name, path='/'+name, sha256='a'*64, metadata={})
                               for name in ('baseline', 'final')], modes=('greedy',))
        self.assertTrue(result['complete'])
        self.assertEqual([case for case, _ in calls], reference['cases'])
        self.assertEqual(env.ids, [case['initial_state_id'] for case in reference['cases']])
        self.assertEqual(env.seeds, [case['env_seed'] for case in reference['cases']])
        self.assertEqual(result['success_count'], 50)
        self.assertEqual(result['num_cases'], 50)
        self.assertEqual(result['success_rate'], 1.0)
        self.assertEqual(result['action_steps'], 150)
        self.assertEqual(len({row['initial_state_hash'] for row in result['rows']}), 50)
        self.assertEqual({row['steps'] for row in result['rows']}, {3})
        self.assertEqual(result['policy_version'], 7)

    def test_horizon_finishes_after_220_actions_even_with_nondivisible_chunk(self):
        env = TinyEnv(success_at=None)
        result = self.run_eval(env, lambda *args: response(count=8))
        self.assertTrue(result['complete'])
        self.assertEqual(result['success_count'], 0)
        self.assertEqual(result['action_steps'], 11000)
        self.assertEqual({row['steps'] for row in result['rows']}, {220})
        self.assertEqual({row['termination'] for row in result['rows']}, {'horizon'})

    def test_wrong_or_missing_served_version_cannot_record_partial_success(self):
        for version in (8, None, True):
            calls = 0
            def infer(*args):
                nonlocal calls
                calls += 1
                return response(7 if calls == 1 else version)
            result = self.run_eval(request=infer)
            self.assertFalse(result['complete'])
            self.assertEqual(result['completed_cases'], 1)
            self.assertIsNone(result['success_rate'])
            self.assertIsNone(result['success_count'])
            self.assertIn('version', result['errors'][0]['error'].lower())

    def test_stop_between_actions_discards_incomplete_case(self):
        env = TinyEnv(success_at=None)
        result = self.run_eval(env, should_stop=lambda: env.total_steps >= 4)
        self.assertFalse(result['complete'])
        self.assertEqual(result['completed_cases'], 0)
        self.assertEqual(env.total_steps, 4)
        self.assertIsNone(result['success_rate'])
        self.assertEqual(result['errors'][0]['type'], 'InterruptedError')

    def test_deadline_after_last_case_does_not_report_complete(self):
        env = TinyEnv(success_at=1)
        result = self.run_eval(env, clock=lambda: 31.0 if env.total_steps >= 50 else 0.0)
        self.assertFalse(result['complete'])
        self.assertIsNone(result['success_count'])
        self.assertEqual(result['errors'][0]['type'], 'TimeoutError')

    def test_never_terminating_environment_is_bounded_and_rejected(self):
        env = TinyEnv(success_at=None, broken_horizon=True)
        result = self.run_eval(env)
        self.assertFalse(result['complete'])
        self.assertEqual(env.total_steps, 220)
        self.assertEqual(result['completed_cases'], 0)

    def test_pool_mismatch_is_rejected_before_policy_requests(self):
        env = TinyEnv()
        expected = dict(initial_state_pool_metadata(env), sha256='f'*64)
        result = self.run_eval(env, expected_pool=expected)
        self.assertFalse(result['complete'])
        self.assertEqual(env.total_steps, 0)
        self.assertIn('pool', result['errors'][0]['error'].lower())

    def test_protocol_mismatches_and_bad_actions_are_incomplete(self):
        for field, value in [('task_id', 1), ('benchmark_name', 'libero_goal'),
                             ('max_episode_steps', 150), ('num_steps_wait', 1)]:
            env = TinyEnv()
            setattr(env, field, value)
            with self.subTest(field=field):
                result = self.run_eval(env)
                self.assertFalse(result['complete'])
                self.assertEqual(env.total_steps, 0)
        for payload in (response(count=0), {'versions': {'policy': 7}}, response()):
            if '_capture_payload' in payload and len(payload['_capture_payload'][1]):
                payload['_capture_payload'][1][0, 0] = math.nan
            with self.subTest(payload=str(payload)[:60]):
                result = self.run_eval(request=lambda *args: payload)
                self.assertFalse(result['complete'])
                self.assertEqual(result['completed_cases'], 0)

    def test_inference_error_never_becomes_a_completed_failure_case(self):
        def infer(*args):
            raise RuntimeError('inference failed')
        result = self.run_eval(request=infer)
        self.assertFalse(result['complete'])
        self.assertEqual(result['rows'], [])
        self.assertIsNone(result['success_rate'])

    def test_worker_returns_verified_egl_receipt_and_closes_environment(self):
        env = TinyEnv()
        receipt = {'backend': 'egl', 'physics': 'cpu', 'actual': {'renderer': 'NVIDIA'},
                   'cuda_visible_devices': ''}
        requests = []
        def remote(inputs, deterministic, telemetry_meta):
            self.assertTrue(deterministic)
            self.assertEqual(telemetry_meta['workflow'], 'evaluation')
            requests.append(telemetry_meta)
            return dict(response(), request_id=telemetry_meta['request_id'])
        ray_stub = SimpleNamespace(get=lambda ref, **kwargs: ref,
                                   exceptions=SimpleNamespace(GetTimeoutError=TimeoutError))
        modules = {'ray': ray_stub,
                   'rl.libero_env': SimpleNamespace(LiberoEnvWrapper=lambda **kwargs: env),
                   'rl.utils': SimpleNamespace(prepare_one_obs=lambda *args: args[2]),
                   'experiments.robot.sole_utils': SimpleNamespace(get_processor=lambda cfg: object()),
                   'environment_rendering': SimpleNamespace(prepare_environment_renderer=lambda: 'egl',
                                                            verify_environment_renderer=lambda: receipt)}
        with patch.dict('sys.modules', modules):
            worker = self.api.PeriodicFixedEvalWorker(SimpleNamespace(request=SimpleNamespace(remote=remote)),
                                                     object(), 'libero_spatial', 'float32')
            result = worker.run(7, timeout_seconds=30)
        self.assertTrue(result['complete'])
        self.assertEqual(result['renderer'], receipt)
        self.assertEqual(env.closed, 1)
        self.assertEqual(len(requests), 50)
        self.assertEqual(len({r['request_id'] for r in requests}), 50)

    def test_worker_failure_closes_renderer_and_never_returns_success_rate(self):
        for failure in ('receipt', 'renderer', 'close', 'waiting_stop'):
            with self.subTest(failure=failure):
                env = TinyEnv()
                def close():
                    env.closed += 1
                    if failure == 'close':
                        raise RuntimeError('renderer cleanup failed')
                env.close = close
                cancelled = []
                def remote(inputs, deterministic, telemetry_meta):
                    return dict(response(), request_id=('wrong' if failure == 'receipt'
                                                        else telemetry_meta['request_id']))
                def get(ref, timeout):
                    self.assertGreater(timeout, 0)
                    self.assertLessEqual(timeout, 1)
                    if failure == 'waiting_stop':
                        worker.stop()
                        raise TimeoutError('pending request')
                    return ref
                receipt = {'backend': 'egl', 'cuda_visible_devices': '0' if failure == 'renderer' else ''}
                modules = {'ray': SimpleNamespace(get=get, cancel=lambda ref, force: cancelled.append(ref),
                                                 exceptions=SimpleNamespace(GetTimeoutError=TimeoutError)),
                           'rl.libero_env': SimpleNamespace(LiberoEnvWrapper=lambda **kwargs: env),
                           'rl.utils': SimpleNamespace(prepare_one_obs=lambda *args: args[2]),
                           'experiments.robot.sole_utils': SimpleNamespace(get_processor=lambda cfg: object()),
                           'environment_rendering': SimpleNamespace(prepare_environment_renderer=lambda: 'egl',
                                                                    verify_environment_renderer=lambda: receipt)}
                with patch.dict('sys.modules', modules):
                    worker = self.api.PeriodicFixedEvalWorker(
                        SimpleNamespace(request=SimpleNamespace(remote=remote)), object(), 'libero_spatial', 'float32')
                    result = worker.run(7, timeout_seconds=30)
                self.assertFalse(result['complete'])
                self.assertIsNone(result['success_rate'])
                self.assertIsNone(result['success_count'])
                self.assertEqual(env.closed, 1)
                self.assertTrue(worker.close())
                self.assertEqual(result['completed_cases'], 50 if failure == 'close' else 0)
                if failure == 'waiting_stop':
                    self.assertEqual(result['errors'][0]['type'], 'InterruptedError')
                    self.assertEqual(len(cancelled), 1)

    def test_stop_before_start_never_constructs_simulator(self):
        worker = self.api.PeriodicFixedEvalWorker(None, None, 'libero_spatial', None)
        worker.stop()
        result = worker.run(0)
        self.assertFalse(result['complete'])
        self.assertEqual(result['completed_cases'], 0)
        self.assertEqual(result['errors'][0]['type'], 'InterruptedError')


if __name__ == '__main__':
    unittest.main()
