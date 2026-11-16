"""Policy-only services, real cohorts, and simulator-budget lease behavior."""
import importlib
import os
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from fresh_config import FreshConfig
from fresh_policy_replay import FreshReplayNotReady, SegmentReplay
from pipeline_budget import PipelineBudgetConfig
from test_fresh_policy_replay_cpu import segment
from world_backends.contracts import WorldBackendConfig
from world_backends.service_plan import build_service_plan


def model_free_world(**overrides):
    return WorldBackendConfig(**dict(dict(backend='model_free', world_trainable=False,
        reward_provider='environment'), **overrides))


def collection_config(**overrides):
    return PipelineBudgetConfig(**dict(dict(model_free=True, imagination_workers=0,
        max_imagine_in_flight=0, initial_state_enabled=True, real_env_step_budget=100), **overrides))


POOL = dict(count=50, sha256='a' * 64, reset_steps=10, max_action_steps=5)


def completion(lease, episode_id='ep', action_steps=5, success=False):
    assigned = lease['initial_state']
    return dict(episode_id=episode_id, metadata=dict(
        initial_state_id=assigned['initial_state_id'], env_seed=assigned['env_seed'],
        reset_env_steps=assigned['reset_steps'], initial_state_hash='b' * 64,
        termination_reason='success' if success else 'task_horizon',
        policy_versions=[1, 2], action_steps=action_steps, success=success))


class ModelFreeContracts(unittest.TestCase):
    def test_policy_only_services_need_no_world_admission_evidence(self):
        world = model_free_world()
        world.validate_admission()
        plan = build_service_plan(world, 4).validate_runtime()
        self.assertEqual(plan.trainer_kinds, ('policy',))
        self.assertEqual(plan.inference_kinds, ('policy',))
        self.assertEqual(plan.wm_consumers, ())
        self.assertEqual(plan.expected_gpu_model_processes, 8)
        self.assertEqual(plan.expected_wm_consumers, 0)
        self.assertEqual(plan.due_kinds(5), ('policy',))
        self.assertEqual(plan.expected_optimizer_steps(10), {'policy': 10})
        self.assertEqual(plan.checkpoint_kinds, ('policy',))
        self.assertFalse(world.capabilities().provides_decoded_observations)

    def test_model_free_requires_environment_rewards_and_no_world_owner(self):
        for overrides in ({'world_trainable': True}, {'reward_provider': 'external_vlm'},
                          {'reward_semantics': 'native_progress'}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                model_free_world(**overrides)
        with self.assertRaises(ValueError):
            WorldBackendConfig(reward_provider='environment')

    def test_real_only_config_is_explicit_and_model_free_defaults_enable_learning(self):
        config = FreshConfig.from_world_config(model_free_world())
        self.assertTrue(config.real_only)
        self.assertEqual(config.real_fraction, 1.)
        for field in ('real_policy_replay', 'refresh_gae', 'replay_age_limits', 'signal_telemetry'):
            self.assertTrue(getattr(config, field))
        for field in ('dense_reward', 'seed_balancing', 'early_seed_context'):
            self.assertFalse(getattr(config, field))
        for overrides in ({'real_only': 1}, {'real_fraction': 1.},
                          {'real_only': True, 'real_fraction': .25}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                FreshConfig.from_mapping(overrides)
        for overrides in ({'real_only': False}, {'dense_reward': True}, {'seed_balancing': True},
                          {'early_seed_context': True}, {'refresh_gae': False}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                FreshConfig.from_world_config(model_free_world(backend_options={'fresh': overrides}))

    def test_only_model_free_permits_zero_imagination_and_rejects_refill(self):
        self.assertEqual(collection_config().imagination_workers, 0)
        for overrides in ({'model_free': False}, {'model_free': 1},
                          {'inventory_refill_enabled': True}, {'imagination_workers': 1},
                          {'max_imagine_in_flight': 1}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                collection_config(**overrides)


class RealOnlyReplay(unittest.TestCase):
    def test_real_cohort_refreshes_gae_and_commits_without_world_versions(self):
        config = FreshConfig.from_world_config(model_free_world())
        replay = SegmentReplay(8, config, rng=random.Random(0))
        item = segment('real', source='real', count=2, terminal=True)
        replay.add(item)
        replay.set_versions(5, 0)
        token, ids = replay.reserve(2)
        before, _ = replay.materialize(ids)
        after, records = replay.materialize(ids, {'real': [2., 3., 4.]})
        self.assertFalse(np.array_equal(before[2], after[2]))
        np.testing.assert_array_equal(before[3], after[3])
        self.assertTrue(all(row['source'] == 'real' and row['generation_world_version'] is None for row in records))
        replay.commit(token, ids)
        self.assertTrue(all(row['uses'] == 1 and row['reserved'] == 0 for row in replay.samples.values()))
        token, single = replay.reserve(1)
        replay.cancel(token)
        self.assertEqual(len(single), 1)

    def test_real_only_rejects_imagined_ingestion_and_enforces_policy_age_and_uses(self):
        config = FreshConfig.from_world_config(model_free_world(backend_options={
            'fresh': {'max_sample_uses': 1, 'max_policy_age': 1}}))
        replay = SegmentReplay(8, config)
        with self.assertRaises(ValueError):
            replay.add(segment())
        self.assertFalse(replay.samples)
        replay.add(segment('real', source='real', policy=4, count=1))
        replay.set_versions(5, 0)
        token, ids = replay.reserve(1)
        replay.commit(token, ids)
        with self.assertRaises(FreshReplayNotReady):
            replay.reserve(1)
        replay.add(segment('stale', source='real', policy=3, count=1))
        with self.assertRaises(FreshReplayNotReady):
            replay.reserve(1)


class RealCollection(unittest.TestCase):
    def setUp(self):
        # Runtime journals are scoped to each test, never the user's run output.
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        env = patch.dict(os.environ, {'ACCERL_DATA_DIR': self.temp.name})
        env.start()
        self.addCleanup(env.stop)

    def owner(self, **overrides):
        module = importlib.import_module('real_collection')
        return module.RealCollectionCoordinator(collection_config(**overrides))

    def test_four_parallel_collectors_share_budget_and_exact_success_refund(self):
        owner = self.owner()
        jobs = [owner.acquire_real(i, POOL) for i in range(4)]
        self.assertEqual(len({job['initial_state']['initial_state_id'] for job in jobs}), 4)
        self.assertEqual(owner.pipeline_status()['initial_states']['reserved_env_steps'], 60)
        self.assertIsNone(owner.acquire_real(4, POOL))
        owner.finish_real_episode(jobs[0]['token'], completion(jobs[0], action_steps=2, success=True))
        self.assertEqual(owner.pipeline_status()['initial_states']['consumed_env_steps'], 12)
        self.assertEqual(owner.pipeline_status()['initial_states']['available_env_steps'], 43)
        self.assertIsNotNone(owner.acquire_real(4, POOL))
        status = owner.pipeline_status()
        self.assertEqual(status['train_windows'], 0)
        self.assertEqual(status['budget']['active_imagine'], 0)
        self.assertEqual(status['budget']['reserved_frame_bytes'], 0)
        terminal = owner.initial_state_checkpoint()['scheduler']['terminal'][jobs[0]['initial_state']['token']]
        self.assertEqual(terminal['metadata']['policy_versions'], [1, 2])

    def test_partial_budget_cancellation_and_stop_do_not_refund_unknown_steps(self):
        owner = self.owner(real_env_step_budget=56)
        jobs = [owner.acquire_real(i, POOL) for i in range(4)]
        self.assertEqual([job['initial_state']['max_action_steps'] for job in jobs], [5, 5, 5, 1])
        self.assertIsNone(owner.acquire_real(4, POOL))
        owner.cancel_real(jobs[0]['token'])
        owner.finish_real_episode(jobs[3]['token'], completion(jobs[3], action_steps=1))
        self.assertIsNone(owner.acquire_real(4, POOL))
        stopped = owner.stop_pipeline()
        self.assertEqual(stopped['initial_states']['committed_env_steps'], 56)
        self.assertEqual(stopped['initial_states']['conservative_unknown_env_steps'], 45)
        self.assertEqual(stopped['initial_states']['episodes_inflight'], 0)
        self.assertTrue(owner.acquire_real(0, POOL)['stopped'])
        self.assertEqual(owner.stop_pipeline()['initial_states']['consumed_env_steps'], 56)

    def test_pool_assignment_and_step_mismatch_leave_lease_unsettled(self):
        owner = self.owner()
        job = owner.acquire_real(0, POOL)
        for field, value in (('env_seed', -1), ('initial_state_id', -1),
                             ('reset_env_steps', 9), ('action_steps', 6), ('success', 1)):
            bad = completion(job)
            bad['metadata'][field] = value
            with self.subTest(field=field), self.assertRaises((ValueError, RuntimeError)):
                owner.finish_real_episode(job['token'], bad)
        with self.assertRaises(ValueError):
            owner.acquire_real(1, dict(POOL, sha256='c' * 64))
        self.assertEqual(owner.pipeline_status()['initial_states']['episodes_completed'], 0)
        self.assertEqual(owner.pipeline_status()['initial_states']['reserved_env_steps'], 15)
        owner.cancel_real(job['token'])
        self.assertEqual(owner.pipeline_status()['initial_states']['conservative_unknown_env_steps'], 15)

    def test_journal_restart_preserves_ambiguous_reservations(self):
        from initial_state_scheduler import UnresolvedLeaseError
        first = self.owner()
        first.acquire_real(0, POOL)
        restarted = self.owner()
        with self.assertRaises(UnresolvedLeaseError):
            restarted.acquire_real(0, POOL)
        self.assertEqual(restarted.pipeline_status()['initial_states']['reserved_env_steps'], 15)

    def test_restarted_coordinator_counts_restored_reservations_against_real_capacity(self):
        first = self.owner()
        for worker in range(4):
            first.acquire_real(worker, POOL)
        restarted = self.owner()
        self.assertIsNone(restarted.acquire_real('new-worker', POOL))
        self.assertEqual(restarted.pipeline_status()['initial_states']['episodes_inflight'], 4)

    def test_completion_evidence_cannot_retain_arrays_in_metadata_fields(self):
        owner = self.owner()
        job = owner.acquire_real(0, POOL)
        for field, value in (('initial_state_hash', [[0] * 10] * 10),
                             ('policy_versions', [[0] * 10] * 10),
                             ('termination_reason', {'image': [0] * 100})):
            bad = completion(job)
            bad['metadata'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                owner.finish_real_episode(job['token'], bad)
        self.assertEqual(owner.pipeline_status()['initial_states']['episodes_completed'], 0)

    def test_training_horizon_applied_before_collection_and_publications_have_no_world(self):
        owner = self.owner(real_training_action_horizon=3)
        with self.assertRaises(ValueError):
            owner.acquire_real(0, POOL)
        job = owner.acquire_real(0, dict(POOL, max_action_steps=3, benchmark_action_horizon=5))
        self.assertEqual(job['initial_state']['max_action_steps'], 3)
        owner.set_fresh_replay_versions(2)
        for policy, world in ((1, 0), (2, 1), (True, 0)):
            with self.subTest(policy=policy, world=world), self.assertRaises(ValueError):
                owner.set_fresh_replay_versions(policy, world)

    def test_explicit_collection_limit_and_exhausted_budget(self):
        owner = self.owner(real_env_step_budget=11, real_collection_action_limit=2)
        job = owner.acquire_real(0, POOL)
        self.assertEqual(job['initial_state']['max_action_steps'], 1)
        owner.finish_real_episode(job['token'], completion(job, action_steps=1))
        self.assertTrue(owner.acquire_real(0, POOL)['budget_exhausted'])

    def test_frame_payload_is_rejected_before_lease_settlement(self):
        owner = self.owner()
        job = owner.acquire_real(0, POOL)
        bad = dict(completion(job), frames=np.zeros((6, 2, 3, 224, 224), np.uint8))
        with self.assertRaises(ValueError):
            owner.finish_real_episode(job['token'], bad)
        self.assertEqual(owner.pipeline_status()['initial_states']['episodes_inflight'], 1)


if __name__ == '__main__':
    unittest.main()
