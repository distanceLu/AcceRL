"""Fixed validation remains diagnostic after a perfect score."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import test_model_free_runtime_cpu as fixtures


class RuntimeTargetTests(unittest.TestCase):
    runtime = fixtures.RuntimeTests.runtime
    validation_fixture = fixtures.RuntimeTests.validation_fixture

    def result(self, **overrides):
        return dict(dict(complete=True, num_cases=50, completed_cases=50,
            policy_version=0, success_count=50, success_rate=1., protocol_sha256='a' * 64,
            initial_state_pool=dict(count=50, shape=[50, 45], sha256='b' * 64)), **overrides)

    def test_p0_full_score_resumes_and_finish_reuses_the_same_result(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                'environment_rendering.render_worker_options', return_value={}):
            runtime = self.validation_fixture(directory, self.result())
            self.assertIsNone(runtime.before_update(0))
            self.assertEqual(self.admissions, [576])
            self.assertEqual(self.pauses, [True, False])
            self.assertEqual(self.limits, [1, 2])
            self.assertEqual(len(self.killed), 1)
            latest = Path(directory) / 'fixed_validation/latest.json'
            record = latest.read_bytes()
            completion = runtime.finish(0)
            self.assertEqual(latest.read_bytes(), record)
            self.assertEqual(len(self.killed), 1)
            self.assertEqual(completion['fixed_validation']['completed_validations'], 1)
            self.assertEqual(completion['fixed_validation']['target_validation']['policy_step'], 0)

    def test_49_successes_resumes_collection_and_admits_an_update(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                'environment_rendering.render_worker_options', return_value={}):
            runtime = self.validation_fixture(directory, self.result(success_count=49, success_rate=.98))
            self.assertIsNone(runtime.before_update(0))
            self.assertEqual(self.admissions, [576])
            self.assertEqual(self.pauses, [True, False])
            self.assertFalse(runtime.control.summary().get('target_reached', True))

    def test_rollout_crossing_during_validation_keeps_collection_paused_and_skips_admission(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                'environment_rendering.render_worker_options', return_value={}):
            runtime = self.validation_fixture(directory, self.result(success_count=12, success_rate=.24))
            runtime.stats_actor.get_model_free_success = fixtures.Method(
                lambda: fixtures.rollout_snapshot(99 if runtime.last_result is not None else 98))
            self.assertEqual(runtime.before_update(0), 'rollout_success_target_reached')
            self.assertEqual(self.pauses, [True, True])
            self.assertEqual(self.admissions, [])
            self.assertEqual(runtime.control.summary()['last']['successes'], 12)

    def test_manual_stop_takes_precedence_over_a_completed_target(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                'environment_rendering.render_worker_options', return_value={}):
            runtime = self.validation_fixture(directory, self.result())
            runtime.validate(0)
            runtime.requested_stop = lambda: True
            self.assertEqual(runtime.before_update(0), 'requested_stop')
            self.assertEqual(self.admissions, [])
            self.assertEqual(self.pauses, [True, False])

    def test_manual_stop_raised_during_validation_takes_precedence_over_target(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                'environment_rendering.render_worker_options', return_value={}):
            runtime = self.validation_fixture(directory, self.result())
            runtime.requested_stop = lambda: runtime.control.summary().get('target_reached', False)
            self.assertEqual(runtime.before_update(0), 'requested_stop')
            self.assertEqual(self.admissions, [])
            self.assertEqual(self.pauses, [True, True])

    def test_worker_failure_takes_precedence_over_a_completed_target(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                'environment_rendering.render_worker_options', return_value={}):
            runtime = self.validation_fixture(directory, self.result())
            runtime.validate(0)
            runtime.worker_refs = ['failed-worker']
            with self.assertRaisesRegex(RuntimeError, 'collection worker exited'):
                runtime.before_update(0)
            self.assertEqual(self.admissions, [])

    def test_later_full_score_below_minimum_resumes_collector_and_admits(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                'environment_rendering.render_worker_options', return_value={}):
            result = self.result(success_count=49, success_rate=.98)
            runtime = self.validation_fixture(directory, result)
            runtime.validate(0)
            status = runtime.status()
            status['initial_states'].update(consumed_env_steps=100010, available_env_steps=899990,
                                            max_env_steps=1000000)
            result.update(success_count=50, success_rate=1., policy_version=1)
            self.assertIsNone(runtime.before_update(1))
            self.assertEqual(self.admissions, [576])
            self.assertEqual(self.pauses, [True, False, True, False])
            self.assertEqual(runtime.control.summary()['target_validation']['policy_step'], 1)
            runtime.finish(1)
            self.assertEqual(len(self.killed), 2)
            self.assertEqual(runtime.control.summary()['completed_validations'], 2)

    def test_incomplete_failed_or_wrong_version_full_result_cannot_stop_successfully(self):
        for changes in ({'complete': False}, {'completed_cases': 49}, {'num_cases': 49},
                        {'policy_version': 1}, {'success_count': 51}):
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as directory, patch(
                    'environment_rendering.render_worker_options', return_value={}):
                runtime = self.validation_fixture(directory, self.result(**changes))
                with self.assertRaises((RuntimeError, ValueError)):
                    runtime.before_update(0)
                self.assertFalse(runtime.control.summary().get('target_reached', True))
                self.assertEqual(runtime.control.summary()['completed_validations'], 0)
                self.assertEqual(self.admissions, [])
                self.assertEqual(self.pauses, [True, True])
                self.assertFalse((Path(directory) / 'fixed_validation/latest.json').exists())

    def test_evaluator_failure_never_records_target_or_resumes_collection(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                'environment_rendering.render_worker_options', return_value={}):
            runtime = self.validation_fixture(directory, self.result())
            original_get = runtime.ray.get

            def get(value, **kwargs):
                if isinstance(value, dict) and 'success_count' in value:
                    raise RuntimeError('evaluation failed')
                return original_get(value, **kwargs)

            runtime.ray.get = get
            with self.assertRaisesRegex(RuntimeError, 'evaluation failed'):
                runtime.before_update(0)
            self.assertFalse(runtime.control.summary().get('target_reached', True))
            self.assertEqual(self.admissions, [])
            self.assertEqual(self.pauses, [True, True])
            self.assertFalse((Path(directory) / 'fixed_validation/latest.json').exists())


if __name__ == '__main__':
    unittest.main()
