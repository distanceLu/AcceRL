"""Real training deadlines, separate from earlier collection-budget censoring.

Execute production constructor, allocator, reset, collector, Replay admission,
RM targets and Policy GAE with CPU simulator/model seams. No model or GPU run.
"""
import json
from types import SimpleNamespace
import unittest

import numpy as np

from fresh_config import FreshConfig
from fresh_policy_replay import generalized_advantages
from fresh_reward_targets import dense_reward_specification, derive_episode_reward_targets
from initial_state_env import (initial_state_pool_metadata, reset_with_initial_state,
                               set_real_training_action_horizon)
from pipeline_budget import PipelineBudget, PipelineBudgetConfig
import test_fresh_policy_runtime_cpu as fresh_runtime
import test_initial_state_pipeline_cpu as initial_pipeline
import test_pipeline_producers_cpu as producers


def collect_episode(*, horizon=150, budget=1000, success_at=None, cap=None):
    owner = initial_pipeline.owner_for(real_training_action_horizon=horizon,
        real_collection_action_limit=cap, real_env_step_budget=budget,
        max_episode_frames=221, frame_budget=512)
    # Enable the production FRESH seed admission path without a model backend.
    owner.fresh = FreshConfig(seed_balancing=True)
    owner.pipeline = PipelineBudget(owner.pipeline_config, policy_chunks_per_seed=2,
                                    fresh=owner.fresh)
    events, collected = [], []
    env = initial_pipeline.TinyExplicitEnv(horizon=220, success_at=success_at)

    def finish(token, result):
        collected.append(result)
        return owner.finish_real_episode(token, result)

    wm = SimpleNamespace(finish_real_episode=producers.Remote(events, 'finish_real', finish))
    worker, _ = producers.make_worker(events, None, wm, mode='real', real_env=env,
                                     pipeline_config=owner.pipeline_config.to_dict())
    lease = owner.acquire_real(0, worker.initial_state_pool)
    worker.pipeline_job, worker.pipeline_lease = lease, lease['token']
    worker.ctrl_wm_frames = 3
    worker.get_one_episode()
    result = collected[0]
    targets = derive_episode_reward_targets(result['success'], result['terminated'],
        result['truncated'], result['metadata'], dense_reward_specification())
    return owner, env, lease, result, targets, events


class TrainingActionHorizonConfigTests(unittest.TestCase):
    def test_default_retains_existing_task_and_collection_behavior(self):
        config = PipelineBudgetConfig()
        self.assertIsNone(config.real_training_action_horizon)
        self.assertIsNone(config.real_collection_action_limit)

    def test_roundtrip_distinguishes_task_deadline_from_budget_limit(self):
        config = PipelineBudgetConfig(initial_state_enabled=True,
            real_training_action_horizon=150, real_collection_action_limit=149)
        restored = PipelineBudgetConfig.from_json(json.loads(json.dumps(config.to_dict())))
        self.assertEqual(restored, config)
        self.assertEqual(restored.real_training_action_horizon, 150)
        self.assertEqual(restored.real_collection_action_limit, 149)

    def test_invalid_horizon_is_rejected(self):
        for value in (False, True, 0, -1, 150., '150'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                PipelineBudgetConfig(initial_state_enabled=True, real_training_action_horizon=value)

    def test_explicit_initial_state_allocation_is_required(self):
        with self.assertRaisesRegex(ValueError, 'initial_state_enabled'):
            PipelineBudgetConfig(real_training_action_horizon=150)

    def test_cannot_extend_loaded_benchmark_horizon(self):
        env = initial_pipeline.TinyExplicitEnv(horizon=220)
        with self.assertRaises(ValueError):
            set_real_training_action_horizon(env, 221)
        self.assertEqual(env.max_episode_steps, 220)

    def test_imagination_constructor_does_not_construct_or_limit_an_environment(self):
        worker, _ = producers.make_worker([], None, None, mode='imagine', base_must_not_run=True,
            pipeline_config={'initial_state_enabled': True, 'real_training_action_horizon': 150})
        self.assertEqual(worker.envs, [])
        self.assertFalse(hasattr(worker, 'initial_state_pool'))


class TrainingActionHorizonCollectorTests(unittest.TestCase):
    def test_full_150_failure_has_valid_zero_rm_labels_and_failure_seeds(self):
        owner, env, lease, result, targets, events = collect_episode()
        self.assertEqual(lease['initial_state']['max_action_steps'], 150)
        self.assertEqual(lease['initial_state']['reserved_env_steps'], 160)
        self.assertEqual(owner.initial_state_pool['max_action_steps'], 150)
        self.assertEqual(owner.initial_state_pool['benchmark_action_horizon'], 220)
        self.assertEqual(env.steps, 150)
        self.assertEqual(env.total_sim_steps, 160)
        self.assertEqual(result['frames'].shape[0], 151)
        self.assertEqual(result['metadata']['task_action_horizon'], 150)
        self.assertEqual(result['metadata']['benchmark_action_horizon'], 220)
        self.assertEqual(result['metadata']['termination_reason'], 'task_horizon')
        self.assertFalse(result['metadata']['budget_censored'])
        self.assertFalse(result['terminated'].any())
        self.assertTrue(result['truncated'][-1])
        self.assertEqual(targets['reward_target_metadata']['outcome'], 'failure')
        self.assertTrue(targets['reward_target_mask'].all())
        self.assertFalse(targets['reward_targets'].any())
        self.assertIn('submit:stats', events)
        strata = owner.pipeline.stats()['fresh_seed_scheduler']
        self.assertEqual(strata['outcome_censored_episodes'], 0)
        self.assertGreater(sum(v for k, v in strata['pending'].items() if k.startswith('failure:')), 0)
        self.assertEqual(sum(v for k, v in strata['pending'].items() if k.startswith('success:')), 0)
        status = owner.initial_state_scheduler.status()
        self.assertEqual(status['consumed_env_steps'], 160)
        self.assertEqual(status['available_env_steps'], 840)

    def test_global_budget_149_is_censored_and_not_a_failure_seed(self):
        owner, env, lease, result, targets, events = collect_episode(budget=159)
        self.assertEqual(lease['initial_state']['max_action_steps'], 149)
        self.assertEqual(env.steps, 149)
        self.assertEqual(env.total_sim_steps, 159)
        self.assertEqual(result['metadata']['task_action_horizon'], 150)
        self.assertEqual(result['metadata']['termination_reason'], 'real_env_budget')
        self.assertTrue(result['metadata']['budget_censored'])
        self.assertEqual(targets['reward_target_metadata']['outcome'], 'censored')
        self.assertFalse(targets['reward_target_mask'].any())
        self.assertNotIn('submit:stats', events)
        strata = owner.pipeline.stats()['fresh_seed_scheduler']
        self.assertEqual(strata['outcome_censored_episodes'], 1)
        self.assertEqual(sum(strata['pending'].values()), 0)
        self.assertGreater(len(owner.buffer), 0)  # WM still receives observed transitions.

    def test_exact_budget_160_still_completes_formal_150_failure(self):
        owner, env, _, result, targets, _ = collect_episode(budget=160)
        self.assertEqual(env.total_sim_steps, 160)
        self.assertEqual(result['metadata']['termination_reason'], 'task_horizon')
        self.assertEqual(targets['reward_target_metadata']['outcome'], 'failure')
        self.assertEqual(owner.initial_state_scheduler.status()['available_env_steps'], 0)

    def test_success_at_150_wins_over_simultaneous_truncation(self):
        _, env, _, result, targets, events = collect_episode(success_at=150)
        self.assertEqual(env.steps, 150)
        self.assertTrue(result['terminated'][-1])
        self.assertTrue(result['truncated'][-1])
        self.assertEqual(result['metadata']['termination_reason'], 'success')
        self.assertFalse(result['metadata']['budget_censored'])
        self.assertEqual(targets['reward_target_metadata']['outcome'], 'success')
        self.assertTrue(targets['reward_target_mask'].all())
        self.assertEqual(float(targets['reward_targets'][-1]), 1.)
        self.assertAlmostEqual(float(targets['reward_targets'][0]), (.99 ** (1 / 8)) ** 150, places=6)
        self.assertIn('submit:stats', events)

    def test_separate_collection_cap_remains_a_censor_before_task_horizon(self):
        _, env, _, result, targets, _ = collect_episode(cap=149)
        self.assertEqual(env.steps, 149)
        self.assertEqual(result['metadata']['task_action_horizon'], 150)
        self.assertEqual(targets['reward_target_metadata']['outcome'], 'censored')

    def test_default_constructor_and_independent_eval_preserve_220(self):
        _, env, lease, result, targets, _ = collect_episode(horizon=None)
        self.assertEqual(lease['initial_state']['max_action_steps'], 220)
        self.assertEqual(env.steps, 220)
        self.assertEqual(result['metadata']['task_action_horizon'], 220)
        self.assertNotIn('benchmark_action_horizon', result['metadata'])
        self.assertEqual(targets['reward_target_metadata']['outcome'], 'failure')
        training = initial_pipeline.TinyExplicitEnv(horizon=220)
        evaluation = initial_pipeline.TinyExplicitEnv(horizon=220)
        set_real_training_action_horizon(training, 150)
        _, info = reset_with_initial_state(evaluation, 0, 5, max_action_steps=220)
        self.assertEqual(training.max_episode_steps, 150)
        self.assertEqual(evaluation.max_episode_steps, 220)
        self.assertEqual(info['task_action_horizon'], 220)

    def test_short_budget_reset_does_not_permanently_shorten_training_task(self):
        env = initial_pipeline.TinyExplicitEnv(horizon=220)
        set_real_training_action_horizon(env, 150)
        pool = initial_state_pool_metadata(env)
        reset_with_initial_state(env, 0, 4, max_action_steps=149)
        self.assertEqual(initial_state_pool_metadata(env), pool)
        _, info = reset_with_initial_state(env, 1, 5, max_action_steps=150)
        self.assertEqual(env.max_episode_steps, 150)
        self.assertEqual(info['task_action_horizon'], 150)
        self.assertEqual(info['benchmark_action_horizon'], 220)


class TrainingActionHorizonPolicyTests(unittest.TestCase):
    def test_failure_150_ends_without_bootstrap_and_masks_last_six_actions(self):
        item, observed, owner, calls = fresh_runtime.RealCollectorTests().collect(
            steps=150, success=False, task_horizon=150)
        self.assertEqual(calls, [])
        self.assertEqual(item.generation_values[-1], 0.)
        self.assertTrue(item.terminated[-1])
        self.assertFalse(item.truncated[-1])
        self.assertEqual(item.action_tokens.shape, (19, 8, 7))
        self.assertEqual(item.action_mask[-1, :, 0].tolist(), [True] * 6 + [False] * 2)
        self.assertEqual(int(item.action_mask[:, :, 0].sum()), 150)
        self.assertEqual(len(item.observations), 20)
        self.assertTrue((item.action_tokens == 2).all())
        self.assertTrue((item.behaviour_logits == 0).all())
        self.assertFalse(owner.collection_metadata['budget_censored'])
        env_terminal, env_truncated, success = observed[0][1]['frame_labels']
        self.assertFalse(env_terminal[-1])
        self.assertTrue(env_truncated[-1])
        self.assertFalse(success[-1])
        _, targets = generalized_advantages(item.rewards, item.generation_values,
            item.terminated, item.truncated, item.gamma, item.lambda_)
        self.assertEqual(float(targets[-1]), 0.)

    def test_budget_149_bootstraps_and_masks_last_five_actions(self):
        item, _, owner, calls = fresh_runtime.RealCollectorTests().collect(
            steps=149, success=False, budget_cut=True, task_horizon=150)
        self.assertEqual(len(calls), 1)
        self.assertAlmostEqual(float(item.generation_values[-1]), .7, places=6)
        self.assertFalse(item.terminated[-1])
        self.assertTrue(item.truncated[-1])
        self.assertEqual(item.action_mask[-1, :, 0].tolist(), [True] * 5 + [False] * 3)
        self.assertEqual(int(item.action_mask[:, :, 0].sum()), 149)
        self.assertTrue(owner.collection_metadata['budget_censored'])
        _, targets = generalized_advantages(item.rewards, item.generation_values,
            item.terminated, item.truncated, item.gamma, item.lambda_)
        self.assertAlmostEqual(float(targets[-1]), .9 * .7, places=6)

    def test_success_150_keeps_environment_reward_and_terminal_target(self):
        item, _, owner, calls = fresh_runtime.RealCollectorTests().collect(
            steps=150, success=True, task_horizon=150)
        self.assertEqual(calls, [])
        self.assertEqual(float(item.rewards[-1]), 1.)
        self.assertTrue(item.terminated[-1])
        self.assertFalse(owner.collection_metadata['budget_censored'])
        _, targets = generalized_advantages(item.rewards, item.generation_values,
            item.terminated, item.truncated, item.gamma, item.lambda_)
        self.assertEqual(float(targets[-1]), 1.)


if __name__ == '__main__':
    unittest.main()
