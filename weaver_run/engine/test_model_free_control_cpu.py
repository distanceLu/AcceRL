"""Budget drain and complete fixed-case validation drive graceful stopping."""
import importlib
import json
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pipeline_budget import PipelineBudgetConfig, LeaseExpiredError
from real_collection import RealCollectionCoordinator


POOL = dict(count=50, sha256='a' * 64, reset_steps=10, max_action_steps=5)


def completion(lease, action_steps=5, episode_id='episode'):
    assigned = lease['initial_state']
    return dict(episode_id=episode_id, metadata=dict(
        initial_state_id=assigned['initial_state_id'], env_seed=assigned['env_seed'],
        reset_env_steps=assigned['reset_steps'], initial_state_hash='b' * 64,
        termination_reason='task_horizon', policy_versions=[0],
        action_steps=action_steps, success=False))


class CollectionPauseAndDrainTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        env = patch.dict(os.environ, {'ACCERL_DATA_DIR': self.temp.name})
        env.start()
        self.addCleanup(env.stop)

    def owner(self, budget=100):
        return RealCollectionCoordinator(PipelineBudgetConfig(model_free=True,
            imagination_workers=0, max_imagine_in_flight=0,
            initial_state_enabled=True, real_env_step_budget=budget))

    def test_pause_blocks_new_leases_but_keeps_existing_lease_and_completion(self):
        owner = self.owner()
        job = owner.acquire_real(0, POOL)
        self.assertTrue(owner.set_paused(True)['budget']['paused'])
        self.assertEqual(owner.acquire_real(0, POOL), job)
        self.assertIsNone(owner.acquire_real(1, POOL))
        owner.finish_real_episode(job['token'], completion(job))
        self.assertEqual(owner.pipeline_status()['budget']['active_real'], 0)
        self.assertFalse(owner.set_paused(False)['budget']['paused'])
        self.assertIsNotNone(owner.acquire_real(1, POOL))
        with self.assertRaises(ValueError):
            owner.set_paused(1)

    def test_uninitialized_pool_and_inflight_reservation_are_not_exhaustion(self):
        owner = self.owner(budget=15)
        self.assertFalse(owner.pipeline_status()['budget']['budget_exhausted'])
        job = owner.acquire_real(0, POOL)
        status = owner.pipeline_status()
        self.assertTrue(status['initial_states']['budget_exhausted'])
        self.assertFalse(status['budget']['budget_exhausted'])
        owner.finish_real_episode(job['token'], completion(job))
        self.assertTrue(owner.pipeline_status()['budget']['budget_exhausted'])

    def test_exact_early_success_refund_changes_reserved_exhaustion_into_spendable_budget(self):
        owner = self.owner(budget=30)
        job = owner.acquire_real(0, dict(POOL, max_action_steps=20))
        self.assertFalse(owner.pipeline_status()['budget']['budget_exhausted'])
        owner.finish_real_episode(job['token'], completion(job, action_steps=1))
        status = owner.pipeline_status()
        self.assertEqual(status['initial_states']['available_env_steps'], 19)
        self.assertFalse(status['budget']['budget_exhausted'])

    def test_unused_tail_is_exhausted_only_after_last_lease_finishes(self):
        owner = self.owner(budget=25)
        job = owner.acquire_real(0, POOL)
        self.assertFalse(owner.pipeline_status()['budget']['budget_exhausted'])
        owner.finish_real_episode(job['token'], completion(job))
        status = owner.pipeline_status()
        self.assertEqual(status['initial_states']['available_env_steps'], 10)
        self.assertTrue(status['budget']['budget_exhausted'])
        self.assertEqual(status['budget']['reset_steps'], 10)
        self.assertTrue(owner.acquire_real(0, POOL)['budget_exhausted'])

    def test_failure_cancellation_is_not_successful_budget_exhaustion(self):
        owner = self.owner(budget=15)
        job = owner.acquire_real(0, POOL)
        owner.cancel_real(job['token'])
        self.assertFalse(owner.pipeline_status()['budget']['budget_exhausted'])
        self.assertFalse((owner.acquire_real(0, POOL) or {}).get('budget_exhausted', False))

    def test_status_checks_expired_leases_instead_of_waiting_forever(self):
        owner = self.owner(budget=15)
        with patch('real_collection.time.monotonic', return_value=0):
            owner.acquire_real(0, POOL)
        with patch('real_collection.time.monotonic', return_value=4000):
            with self.assertRaises(LeaseExpiredError):
                owner.pipeline_status()


class FixedValidationControlTests(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module('model_free_control')
        except ModuleNotFoundError:
            self.fail('model_free_control must implement the pure validation controller')

    def control(self, **kwargs):
        return self.module.ModelFreeControl(self.module.ModelFreeControlConfig(**kwargs))

    def test_p0_then_only_interval_and_new_version_trigger_validation(self):
        ctrl = self.control()
        self.assertTrue(ctrl.should_validate(0, 0))
        ctrl.observe(2, policy_step=0, consumed_env_steps=0)
        self.assertFalse(ctrl.should_validate(99999, 1))
        self.assertFalse(ctrl.should_validate(100000, 0))
        self.assertTrue(ctrl.should_validate(100000, 1))
        ctrl.observe(2, policy_step=1, consumed_env_steps=100020)
        self.assertFalse(ctrl.should_validate(200000, 2))
        self.assertTrue(ctrl.should_validate(200020, 2))
        self.assertFalse(ctrl.should_validate(200020, 1, force=True))
        self.assertTrue(ctrl.should_validate(100021, 2, force=True))

    def test_default_plateau_requires_three_completed_postbaseline_validations(self):
        ctrl = self.control()
        ctrl.observe(10, policy_step=0, consumed_env_steps=0)
        for step in (1, 2):
            ctrl.observe(10, policy_step=step, consumed_env_steps=step * 100000)
            self.assertFalse(ctrl.plateau_reached)
        summary = ctrl.observe(9, policy_step=3, consumed_env_steps=300000)
        self.assertTrue(ctrl.plateau_reached)
        self.assertEqual(summary['post_baseline_validations'], 3)
        self.assertEqual(summary['stale_count'], 3)
        self.assertEqual(summary['best_successes'], 10)
        self.assertEqual(summary['best_policy_step'], 0)
        self.assertEqual(summary['last']['successes'], 9)
        json.dumps(summary, allow_nan=False)

    def test_one_success_improves_and_resets_patience_without_rewinding_last_policy(self):
        ctrl = self.control()
        ctrl.observe(10, policy_step=0, consumed_env_steps=0)
        ctrl.observe(10, policy_step=1, consumed_env_steps=100000)
        ctrl.observe(11, policy_step=2, consumed_env_steps=200000)
        summary = ctrl.observe(10, policy_step=3, consumed_env_steps=300000)
        self.assertFalse(ctrl.plateau_reached)
        self.assertEqual(summary['best_successes'], 11)
        self.assertEqual(summary['stale_count'], 1)
        self.assertEqual(summary['last']['policy_step'], 3)

    def test_minimum_steps_and_configurable_delta_are_enforced(self):
        ctrl = self.control(min_delta_successes=2, patience=1)
        ctrl.observe(10, policy_step=0, consumed_env_steps=0)
        ctrl.observe(11, policy_step=1, consumed_env_steps=100000)
        self.assertFalse(ctrl.plateau_reached)
        ctrl.observe(12, policy_step=2, consumed_env_steps=200000)
        self.assertEqual(ctrl.summary()['stale_count'], 0)
        ctrl.observe(12, policy_step=3, consumed_env_steps=300000)
        self.assertTrue(ctrl.plateau_reached)

    def test_disabled_validation_cannot_be_forced_or_report_plateau(self):
        ctrl = self.control(interval_env_steps=0)
        self.assertFalse(ctrl.should_validate(1000000, 0, force=True))
        self.assertFalse(ctrl.plateau_reached)
        with self.assertRaises(ValueError):
            ctrl.observe(50, policy_step=0, consumed_env_steps=1000000)

    def test_incomplete_duplicate_or_invalid_result_never_changes_controller_state(self):
        ctrl = self.control()
        with self.assertRaises(ValueError):
            ctrl.observe(10, policy_step=1, consumed_env_steps=0)
        ctrl.observe(10, policy_step=0, consumed_env_steps=0)
        before = ctrl.summary()
        invalid = [dict(successes=1, cases=49, policy_step=1, consumed_env_steps=100000),
                   dict(successes=1, policy_step=0, consumed_env_steps=100000),
                   dict(successes=51, policy_step=1, consumed_env_steps=100000),
                   dict(successes=True, policy_step=1, consumed_env_steps=100000),
                   dict(successes=float('nan'), policy_step=1, consumed_env_steps=100000),
                   dict(successes=1, policy_step=1, consumed_env_steps=-1)]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ctrl.observe(**kwargs)
            self.assertEqual(ctrl.summary(), before)

    def test_config_rejects_noninteger_negative_and_boolean_values(self):
        for name in ('interval_env_steps', 'patience', 'min_env_steps', 'min_delta_successes'):
            for value in (-1, False, True, float('nan'), float('inf'), 1.5, '3'):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    self.control(**{name: value})
        for kwargs in ({'patience': 0}, {'min_delta_successes': 0}, {'min_delta_successes': 51}):
            with self.assertRaises(ValueError):
                self.control(**kwargs)
        config = self.module.ModelFreeControlConfig.from_args(SimpleNamespace(
            fixed_validation_interval_env_steps=200000, fixed_validation_patience=4,
            fixed_validation_min_env_steps=600000, fixed_validation_min_delta_successes=2))
        self.assertEqual(config.to_dict(), dict(interval_env_steps=200000, patience=4,
            min_env_steps=600000, min_delta_successes=2))

    def test_budget_helper_requires_owner_evidence_and_rejects_reserved_only_exhaustion(self):
        budget = self.module.collection_budget({'budget': {'config': {'real_env_step_budget': 25}}})
        self.assertFalse(budget['initialized'])
        self.assertFalse(budget['budget_exhausted'])
        states = dict(max_env_steps=25, consumed_env_steps=15, reserved_env_steps=0,
            available_env_steps=10, episodes_inflight=0, conservative_unknown_env_steps=0,
            episodes_cancelled=0)
        status = dict(budget=dict(config={'real_env_step_budget': 25}, active_real=0,
            reset_steps=10, budget_exhausted=True, failure=None, paused=False), initial_states=states)
        self.assertTrue(self.module.budget_exhausted(status))
        for changed in ({'reserved_env_steps': 10, 'episodes_inflight': 1},
                        {'conservative_unknown_env_steps': 15}, {'episodes_cancelled': 1}):
            with self.subTest(changed=changed):
                self.assertFalse(self.module.budget_exhausted(dict(status, initial_states=dict(states, **changed))))
        self.assertFalse(self.module.budget_exhausted(dict(status, budget=dict(status['budget'], failure='bad'))))

    def test_from_args_none_defaults_enable_only_when_online_eval_is_zero(self):
        args = SimpleNamespace(num_eval_workers=0, fixed_validation_interval_env_steps=None,
            fixed_validation_patience=None, fixed_validation_min_env_steps=None,
            fixed_validation_min_delta_successes=None)
        config = self.module.ModelFreeControlConfig.from_args(args)
        self.assertEqual(config.interval_env_steps, 100000)
        self.assertEqual(config.patience, 3)
        self.assertEqual(config.min_env_steps, 300000)
        self.assertEqual(config.min_delta_successes, 1)
        args.num_eval_workers = 2
        self.assertEqual(self.module.ModelFreeControlConfig.from_args(args).interval_env_steps, 0)
        args.fixed_validation_interval_env_steps = 100000
        with self.assertRaisesRegex(ValueError, 'online Eval'):
            self.module.ModelFreeControlConfig.from_args(args)
        args.fixed_validation_interval_env_steps = 0
        self.assertEqual(self.module.ModelFreeControlConfig.from_args(args).interval_env_steps, 0)


if __name__ == '__main__':
    unittest.main()
