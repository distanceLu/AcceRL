"""Real-only collection cap through actual allocation and collector CPU paths.

The simulated transitions and model RPCs are deterministic fakes; allocation,
reset, real collection, partial chunk masks and dense target derivation are the
production implementations. No GPU, Ray runtime or LIBERO launch is required.
"""
import json
from types import SimpleNamespace
import unittest

import numpy as np

from fresh_reward_targets import dense_reward_specification, derive_episode_reward_targets
from initial_state_env import initial_state_pool_metadata, reset_with_initial_state
from pipeline_budget import PipelineBudgetConfig
import test_fresh_policy_runtime_cpu as fresh_runtime
import test_initial_state_pipeline_cpu as initial_pipeline
import test_pipeline_producers_cpu as producers


def capped_owner(*, cap=150, budget=1000):
    return initial_pipeline.owner_for(real_collection_action_limit=cap,
        real_env_step_budget=budget, max_episode_frames=221, frame_budget=512)


class CollectionActionLimitConfigTests(unittest.TestCase):
    def test_default_preserves_original_horizon(self):
        self.assertIsNone(PipelineBudgetConfig().real_collection_action_limit)
        owner = capped_owner(cap=None)
        pool = initial_state_pool_metadata(initial_pipeline.TinyExplicitEnv(horizon=220))
        lease = owner.acquire_real(0, pool)['initial_state']
        self.assertEqual(lease['max_action_steps'], 220)
        self.assertEqual(lease['reserved_env_steps'], 230)

    def test_config_roundtrip_retains_separate_collection_limit(self):
        config = PipelineBudgetConfig(initial_state_enabled=True, real_collection_action_limit=150)
        restored = PipelineBudgetConfig.from_json(json.loads(json.dumps(config.to_dict())))
        self.assertEqual(restored, config)
        self.assertEqual(restored.real_collection_action_limit, 150)

    def test_invalid_cap_is_rejected(self):
        for cap in (False, True, 0, -1, 150., '150'):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                PipelineBudgetConfig(initial_state_enabled=True, real_collection_action_limit=cap)

    def test_cap_requires_explicit_allocation_instead_of_silent_ignore(self):
        with self.assertRaisesRegex(ValueError, 'initial_state_enabled'):
            PipelineBudgetConfig(real_collection_action_limit=150)


class CollectionActionLimitAllocationTests(unittest.TestCase):
    def test_action_cap_reserves_150_plus_reset_without_changing_task_pool(self):
        owner = capped_owner()
        env = initial_pipeline.TinyExplicitEnv(horizon=220)
        pool = initial_state_pool_metadata(env)
        lease = owner.acquire_real(0, pool)['initial_state']
        self.assertEqual(lease['max_action_steps'], 150)
        self.assertEqual(lease['reserved_env_steps'], 160)
        self.assertEqual(owner.initial_state_pool, pool)
        self.assertEqual(pool['max_action_steps'], 220)
        _, metadata = reset_with_initial_state(env, lease['initial_state_id'],
            lease['env_seed'], max_action_steps=lease['max_action_steps'])
        self.assertEqual(env.max_episode_steps, 150)
        self.assertEqual(metadata['task_action_horizon'], 220)
        self.assertEqual(initial_state_pool_metadata(env), pool)

    def test_global_budget_can_shorten_collection_below_cap(self):
        owner = capped_owner(budget=117)
        pool = initial_state_pool_metadata(initial_pipeline.TinyExplicitEnv(horizon=220))
        lease = owner.acquire_real(0, pool)['initial_state']
        self.assertEqual(lease['max_action_steps'], 107)
        self.assertEqual(lease['reserved_env_steps'], 117)
        self.assertEqual(owner.initial_state_scheduler.status()['available_env_steps'], 0)

    def test_collection_cap_never_extends_a_shorter_task(self):
        owner = capped_owner()
        pool = initial_state_pool_metadata(initial_pipeline.TinyExplicitEnv(horizon=80))
        self.assertEqual(owner.acquire_real(0, pool)['initial_state']['max_action_steps'], 80)

    def test_independent_evaluation_reset_keeps_full_220(self):
        owner = capped_owner()
        collection = initial_pipeline.TinyExplicitEnv(horizon=220)
        pool = initial_state_pool_metadata(collection)
        assigned = owner.acquire_real(0, pool)['initial_state']
        reset_with_initial_state(collection, 0, 5, max_action_steps=assigned['max_action_steps'])
        evaluation = initial_pipeline.TinyExplicitEnv(horizon=220)
        _, metadata = reset_with_initial_state(evaluation, 0, 5, max_action_steps=220)
        self.assertEqual(collection.max_episode_steps, 150)
        self.assertEqual(evaluation.max_episode_steps, 220)
        self.assertEqual(metadata['task_action_horizon'], 220)

    def test_actual_collector_consumes_only_160_steps_and_masks_unknown_rm_outcome(self):
        owner = capped_owner()
        env = initial_pipeline.TinyExplicitEnv(horizon=220)
        pool = initial_state_pool_metadata(env)
        lease = owner.acquire_real(0, pool)
        events, collected = [], []

        def finish(token, result):
            collected.append(result)
            return owner.finish_real_episode(token, result)

        wm = SimpleNamespace(finish_real_episode=producers.Remote(events, 'finish_real', finish))
        worker, _ = producers.make_worker(events, None, wm, mode='real')
        worker.envs = [env]
        worker.initial_state_enabled = True
        worker.initial_state_pool = pool
        worker.pipeline_job, worker.pipeline_lease = lease, lease['token']
        worker.ctrl_wm_frames = 3
        worker.get_one_episode()
        result = collected[0]
        self.assertEqual(env.steps, 150)
        self.assertEqual(env.total_sim_steps, 160)
        self.assertEqual(result['frames'].shape[0], 151)
        self.assertEqual(result['metadata']['termination_reason'], 'real_env_budget')
        self.assertTrue(result['metadata']['budget_censored'])
        self.assertEqual(result['metadata']['task_action_horizon'], 220)
        self.assertNotIn('submit:stats', events)
        targets = derive_episode_reward_targets(result['success'], result['terminated'],
            result['truncated'], result['metadata'], dense_reward_specification())
        self.assertEqual(targets['reward_target_metadata']['outcome'], 'censored')
        self.assertFalse(targets['reward_target_mask'].any())
        self.assertEqual(owner.pipeline_status()['initial_states']['consumed_env_steps'], 160)
        self.assertEqual(owner.pipeline_status()['initial_states']['available_env_steps'], 840)
        self.assertGreater(len(owner.buffer), 0)  # WM can learn the observed prefix.


class CollectionActionLimitLearningTests(unittest.TestCase):
    def test_150_censor_bootstraps_and_retains_six_action_last_chunk(self):
        item, _, owner, calls = fresh_runtime.RealCollectorTests().collect(
            steps=150, success=False, budget_cut=True)
        self.assertEqual(len(calls), 1)
        self.assertAlmostEqual(float(item.generation_values[-1]), .7, places=6)
        self.assertFalse(item.terminated[-1])
        self.assertTrue(item.truncated[-1])
        self.assertEqual(item.action_tokens.shape, (19, 8, 7))
        self.assertEqual(item.action_mask[-1, :, 0].tolist(), [True] * 6 + [False] * 2)
        self.assertEqual(int(item.action_mask[:, :, 0].sum()), 150)
        self.assertEqual(len(item.observations), 20)  # K+1 critic observations.
        self.assertTrue((item.action_tokens == 2).all())
        self.assertTrue((item.behaviour_logits == 0).all())
        self.assertEqual(owner.collection_metadata['termination_reason'], 'real_env_budget')

    def test_success_at_limit_remains_observed_success_without_bootstrap(self):
        item, observed, owner, calls = fresh_runtime.RealCollectorTests().collect(
            steps=150, success=True, budget_cut=True)
        self.assertEqual(calls, [])
        self.assertTrue(item.terminated[-1])
        self.assertFalse(item.truncated[-1])
        self.assertFalse(owner.collection_metadata['budget_censored'])
        terminated, truncated, success = observed[0][1]['frame_labels']
        metadata = dict(owner.collection_metadata, label_source='libero_env_step')
        targets = derive_episode_reward_targets(np.asarray(success, bool),
            np.asarray(terminated, bool), np.asarray(truncated, bool), metadata,
            dense_reward_specification())
        self.assertEqual(targets['reward_target_metadata']['outcome'], 'success')
        self.assertTrue(targets['reward_target_mask'].all())
        self.assertEqual(float(targets['reward_targets'][-1]), 1.)

    def test_full_task_horizon_remains_observed_failure(self):
        item, _, owner, calls = fresh_runtime.RealCollectorTests().collect(
            steps=220, success=False, budget_cut=False)
        self.assertEqual(calls, [])
        self.assertTrue(item.terminated[-1])
        self.assertFalse(item.truncated[-1])
        self.assertEqual(owner.collection_metadata['termination_reason'], 'task_horizon')


if __name__ == '__main__':
    unittest.main()
