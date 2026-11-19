"""P0 completion initializes honest budget metadata without collecting an episode."""
from copy import deepcopy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from initial_state_env import initial_state_pool_metadata, set_real_training_action_horizon
from model_free_control import collection_budget
from pipeline_budget import PipelineBudgetConfig
from real_collection import RealCollectionCoordinator
import test_model_free_runtime_cpu as fixtures


class PoolInitializationTests(unittest.TestCase):
    runtime = fixtures.RuntimeTests.runtime
    validation_fixture = fixtures.RuntimeTests.validation_fixture

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        environment = patch.dict(os.environ, ACCERL_DATA_DIR=self.temp.name)
        environment.start()
        self.addCleanup(environment.stop)
        self.owner = RealCollectionCoordinator(PipelineBudgetConfig(model_free=True,
            imagination_workers=0, max_imagine_in_flight=0, initial_state_enabled=True,
            real_env_step_budget=1000000, real_training_action_horizon=150))
        env = SimpleNamespace(initial_states=np.zeros((50, 45)),
                              max_episode_steps=220, num_steps_wait=10)
        self.eval_pool = initial_state_pool_metadata(env)
        set_real_training_action_horizon(env, 150)
        self.training_pool = initial_state_pool_metadata(env)

    def initialize(self, pool):
        initialize = getattr(self.owner, 'initialize_initial_state_pool', None)
        self.assertTrue(callable(initialize), 'Metadata-only pool initialization must exist')
        return initialize(pool)

    def real_runtime(self, **overrides):
        result = dict(complete=True, completed_cases=50, num_cases=50, policy_version=0,
            success_count=50, success_rate=1., protocol_sha256='a' * 64,
            initial_state_pool=deepcopy(self.eval_pool))
        result.update(overrides)
        runtime = self.validation_fixture(self.temp.name, result)
        runtime.collector = SimpleNamespace(**{
            name: fixtures.Method(getattr(self.owner, name)) for name in
            ('pipeline_status', 'set_paused', 'initial_state_checkpoint')})
        runtime.collector.initialize_initial_state_pool = fixtures.Method(self.initialize)
        return runtime, result

    def test_metadata_initialization_preserves_zero_budget_usage_and_worker_pool_equality(self):
        before = deepcopy(self.eval_pool)
        self.initialize(self.eval_pool)
        self.assertEqual(self.eval_pool, before)
        self.assertEqual(self.owner.initial_state_checkpoint()['pool'], self.training_pool)
        status = self.owner.pipeline_status()
        budget = collection_budget(status)
        self.assertTrue(budget['initialized'])
        self.assertEqual((budget['consumed_env_steps'], budget['available_env_steps'], budget['reset_steps']),
                         (0, 1000000, 10))
        self.assertEqual((status['budget']['real_admitted'], status['initial_states']['episodes_assigned']), (0, 0))
        self.assertEqual(self.owner._next_token, 0)
        self.assertIsNone(self.owner._versions)
        self.assertEqual(self.owner.initial_state_checkpoint()['scheduler']['allocations'], [])
        self.assertTrue((Path(self.temp.name) / 'initial_state_queue.json').is_file())
        snapshot = self.owner.initial_state_checkpoint()
        self.initialize(self.eval_pool)
        self.assertEqual(self.owner.initial_state_checkpoint(), snapshot)
        lease = self.owner.acquire_real(0, self.training_pool)
        self.assertEqual(lease['token'], 'real:1')
        self.assertEqual(lease['initial_state']['max_action_steps'], 150)

    def test_pool_mismatch_still_rejects_initialization_and_worker_admission(self):
        self.initialize(self.eval_pool)
        before = self.owner.initial_state_checkpoint()
        for changes in ({'sha256': 'c' * 64}, {'shape': [50, 46]}, {'reset_steps': 11}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'different initial-state'):
                self.initialize(dict(self.eval_pool, **changes))
            self.assertEqual(self.owner.initial_state_checkpoint(), before)
            with self.assertRaisesRegex(ValueError, 'different initial-state'):
                self.owner.acquire_real(0, dict(self.training_pool, **changes))
        self.assertEqual(self.owner._next_token, 0)

    def test_real_p0_full_score_initializes_pool_and_allows_collection(self):
        with patch('environment_rendering.render_worker_options', return_value={}):
            runtime, result = self.real_runtime()
            self.assertFalse(collection_budget(self.owner.pipeline_status())['initialized'])
            self.assertIsNone(runtime.before_update(0))
            details = runtime.completion_fields()
        self.assertTrue(details['budget_status']['initialized'])
        self.assertEqual(details['budget_status']['consumed_env_steps'], 0)
        self.assertEqual(details['budget_status']['available_env_steps'], 1000000)
        self.assertEqual(details['budget_status']['reset_steps'], 10)
        self.assertEqual(self.admissions, [576])
        self.assertEqual(runtime.worker_refs, [])
        self.assertEqual(self.owner._next_token, 0)
        self.assertEqual(self.owner.pipeline_status()['budget']['real_admitted'], 0)
        self.assertFalse(self.owner.pipeline_status()['budget']['paused'])
        self.assertEqual(result['initial_state_pool']['max_action_steps'], 220)
        self.assertNotIn('benchmark_action_horizon', result['initial_state_pool'])
        self.assertEqual(self.owner.initial_state_checkpoint()['pool'], self.training_pool)

    def test_wrong_version_does_not_initialize_metadata_or_record_target(self):
        with patch('environment_rendering.render_worker_options', return_value={}):
            runtime, _ = self.real_runtime(policy_version=1)
            with self.assertRaisesRegex(RuntimeError, 'wrong-version'):
                runtime.before_update(0)
        self.assertFalse(collection_budget(self.owner.pipeline_status())['initialized'])
        self.assertFalse(runtime.control.target_reached)
        self.assertEqual(self.owner._next_token, 0)


if __name__ == '__main__':
    unittest.main()
