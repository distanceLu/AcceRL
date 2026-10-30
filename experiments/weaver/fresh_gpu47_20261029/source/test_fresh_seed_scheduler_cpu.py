"""Behavioral contracts for FRESH strata and explicitly synthetic early history."""
from collections import deque
from dataclasses import replace
import random
import unittest

import numpy as np

from episode_frame_store import EpisodeRecord, materialize_imagination_context
from fresh_seed_scheduler import BUCKETS, FreshSeedScheduler, context_seed_starts, seed_bucket
from pipeline_budget import PipelineBudget, PipelineBudgetConfig, uniform_seed_starts
from pipeline_runtime import EpisodeReplayMixin
from test_episode_frame_store_cpu import schema


def config(**changes):
    return replace(PipelineBudgetConfig(max_episode_frames=20, bytes_per_frame=10,
        max_resident_frame_bytes=2000, seeds_per_episode=12,
        policy_high_watermark=100, policy_low_watermark=20,
        max_real_in_flight=1, max_imagine_in_flight=4), **changes)


def complete(budget, name, success, frames=13, context=4):
    lease = budget.acquire_real('real')
    return budget.complete_real(lease['token'], name, frames, context, frames * 10,
                                success=success)


def record(count=5):
    frames = np.empty((count, 2, 3, 224, 224), np.uint8)
    for index in range(count):
        frames[index].fill(index + 10)
    actions = np.zeros((count, 7), np.float32)
    actions[1:] = np.arange(1, count)[:, None]
    states = np.arange(count * 8, dtype=np.float32).reshape(count, 8)
    return EpisodeRecord('observed', 'task', frames, actions, np.zeros(count), states=states)


class FreshSeedTests(unittest.TestCase):
    def test_disabled_candidate_starts_match_legacy_exhaustively(self):
        for count in range(70):
            for context in (1, 4, 32):
                self.assertEqual(context_seed_starts(count, context, 16),
                                 uniform_seed_starts(count, context, 16))
        plain = PipelineBudget(config(), 4, clock=lambda: 0)
        explicit = PipelineBudget(config(), 4, clock=lambda: 0,
                                  fresh={'seed_balancing': False, 'early_seed_context': False})
        for budget in (plain, explicit):
            complete(budget, 'success', True)
            complete(budget, 'failure', False)
        for _ in range(10):
            left, right = plain.acquire_imagine('imagine'), explicit.acquire_imagine('imagine')
            self.assertEqual(left, right)
            self.assertNotIn('seed_bucket', left)
            self.assertEqual(plain.finish_imagine(left['token'], 3),
                             explicit.finish_imagine(right['token'], 3))
            self.assertEqual(plain.stats(), explicit.stats())

    def test_six_available_strata_each_receive_a_turn_without_more_concurrency(self):
        budget = PipelineBudget(config(), 4, fresh={'seed_balancing': True,
                                                   'early_seed_context': True})
        complete(budget, 'success', True)
        complete(budget, 'failure', False)
        seen = []
        for _ in range(6):
            lease = budget.acquire_imagine('imagine')
            seen.append(lease['seed_bucket'])
            self.assertLessEqual(lease['seed_anchor'], 11)
            budget.finish_imagine(lease['token'], 4)
        self.assertEqual(seen, list(BUCKETS))
        stats = budget.stats()
        self.assertEqual(stats['policy_added'], 24)
        self.assertEqual(stats['active_imagine'], 0)
        self.assertEqual(set(stats['fresh_seed_scheduler']['admitted'].values()), {1})
        leases = [budget.acquire_imagine(str(i)) for i in range(4)]
        self.assertTrue(all(leases))
        self.assertIsNone(budget.acquire_imagine('fifth'))
        self.assertEqual(budget.stats()['reserved_policy_samples'], 16)

    def test_missing_outcome_is_rejected_before_real_lease_is_committed(self):
        budget = PipelineBudget(config(), 4, fresh={'seed_balancing': True})
        lease = budget.acquire_real('real')
        with self.assertRaisesRegex(ValueError, 'measured completed'):
            budget.complete_real(lease['token'], 'bad', 13, 4, 130)
        self.assertEqual(budget.stats()['active_real'], 1)
        self.assertEqual(budget.stats()['real_completed'], 0)

    def test_early_context_is_independent_and_keeps_candidate_and_byte_limits(self):
        budget = PipelineBudget(config(seeds_per_episode=2), 4,
                                fresh={'early_seed_context': True})
        outcome = complete(budget, 'short', False, frames=3, context=32)
        self.assertEqual(outcome['starts'], [-31, -30])
        self.assertEqual(outcome['seed_count'], 2)
        self.assertEqual(budget.stats()['pinned_frame_bytes'], 30)
        lease = budget.acquire_imagine('imagine')
        self.assertEqual(lease['seed_anchor'], 0)
        self.assertIsNone(lease['seed_bucket'])
        self.assertEqual(lease['seed_scheduler_mode'], 'legacy_round_robin')
        budget.cancel_imagine(lease['token'])

    def test_private_random_ablation_is_reproducible_and_does_not_change_global_rng(self):
        candidates = [('A', 0, 'success:early'), ('B', 1, 'failure:late')]
        state = random.getstate()
        a, b = FreshSeedScheduler('random', 93), FreshSeedScheduler('random', 93)
        self.assertEqual([a.choose(candidates) for _ in range(30)],
                         [b.choose(candidates) for _ in range(30)])
        self.assertEqual(random.getstate(), state)
        with self.assertRaises(ValueError):
            FreshSeedScheduler('value')
        with self.assertRaises(ValueError):
            seed_bucket(True, 12, 13)

    def test_cancel_and_eviction_account_by_actual_bucket(self):
        budget = PipelineBudget(config(max_pending_episodes=1), 4,
                                fresh={'seed_balancing': True, 'early_seed_context': True})
        complete(budget, 'old', True)
        lease = budget.acquire_imagine('imagine')
        budget.cancel_imagine(lease['token'])
        complete(budget, 'new', False)
        stats = budget.stats()
        strata = stats['fresh_seed_scheduler']
        self.assertEqual(sum(strata['cancelled'].values()), 1)
        self.assertEqual(sum(strata['dropped'].values()), 11)
        self.assertEqual(stats['seeds_dropped'], 11)
        self.assertEqual(stats['pinned_frame_bytes'], 130)

    def test_censored_outcome_is_not_misclassified_as_failure(self):
        budget = PipelineBudget(config(), 4, fresh={'seed_balancing': True,
                                                   'early_seed_context': True})
        lease = budget.acquire_real('real')
        outcome = budget.complete_real(lease['token'], 'censored', 13, 4, 130,
                                       success=False, outcome_censored=True)
        self.assertEqual(outcome['seed_count'], 0)
        self.assertEqual(outcome['seed_skip_reason'], 'censored_real_episode_outcome')
        self.assertIsNone(budget.acquire_imagine('imagine'))
        stats = budget.stats()['fresh_seed_scheduler']
        self.assertEqual(stats['outcome_censored_episodes'], 1)
        self.assertFalse(any(stats['pending'].values()))


class ExplicitPaddingTests(unittest.TestCase):
    def test_early_context_maps_to_real_reset_and_masks_synthetic_history(self):
        ep = record()
        context = materialize_imagination_context(ep, -2, 4, 0.05)
        self.assertEqual(context['frame_indices'], [0, 0, 0, 1])
        self.assertEqual(context['valid_mask'], [False, False, True, True])
        self.assertEqual(context['action_mask'], [False, False, False, True])
        self.assertEqual(context['dt'], [0., 0., 0., 0.05])
        self.assertEqual(context['seed_anchor'], 1)
        self.assertEqual(context['context_padding']['count'], 2)
        self.assertTrue(context['context_padding']['synthetic_history'])
        for name, values in (('frames', ep.frames), ('actions', ep.actions), ('states', ep.states)):
            np.testing.assert_array_equal(context[name], values[[0, 0, 0, 1]])
            self.assertFalse(context[name].flags.writeable)
        # Padding is a bounded projection; the canonical frame owner is intact.
        self.assertEqual(ep.frame_count, 5)
        self.assertEqual(ep.frames[0, 0, 0, 0, 0], 10)

    def test_nonpadded_context_retains_exact_observations_and_reset_action_mask(self):
        ep = record()
        context = materialize_imagination_context(ep, 0, 4, 0.05)
        self.assertNotIn('context_padding', context)
        self.assertEqual(context['valid_mask'], [True] * 4)
        self.assertEqual(context['action_mask'], [False, True, True, True])
        np.testing.assert_array_equal(context['frames'], ep.frames[:4])
        later = materialize_imagination_context(ep, 1, 3, 0.05)
        self.assertEqual(later['action_mask'], [True] * 3)
        for start, length in ((-5, 4), (2, 4), (0, 0)):
            with self.assertRaises(ValueError):
                materialize_imagination_context(ep, start, length, 0.05)


class Owner(EpisodeReplayMixin):
    def __init__(self, world, *, context=4, window=3):
        self.buffer, self.eval_buffer = [], []
        self.sampled_episode_ids = set()
        budgets = config(max_episode_frames=50,
                         bytes_per_frame=2 * 3 * 224 * 224,
                         max_resident_frame_bytes=40 * 1024 * 1024)
        self.init_pipeline(budgets, world, context, window, 4, 1)

    def add_batch(self, batch):
        self.buffer.extend(batch)


def owner_payload(*, identity='real-episode', count=6, success=True, metadata=None):
    ep = record(count)
    terminal = np.zeros(count, bool)
    terminal[-1] = True
    successes = terminal.copy() if success else np.zeros(count, bool)
    return {'episode_id': identity, 'instruction': 'task', 'frames': ep.frames,
            'actions': ep.actions, 'rewards': np.zeros(count), 'states': ep.states,
            'terminated': terminal if success else np.zeros(count, bool),
            'truncated': np.zeros(count, bool) if success else terminal,
            'success': successes,
            'metadata': metadata or {'label_source': 'libero_env_step',
                                    'action_steps': count - 1, 'budget_censored': False,
                                    'termination_reason': 'success' if success else 'task_horizon'}}


class RuntimeIntegrationTests(unittest.TestCase):
    def test_actual_owner_materializes_early_job_and_keeps_original_frames(self):
        world = schema()
        world['backend_options'] = {'fresh': {'seed_balancing': True, 'early_seed_context': True}}
        owner = Owner(world)
        payload = owner_payload(count=6)
        lease = owner.acquire_real('real')
        owner.finish_real_episode(lease['token'], payload)
        stored = owner.frame_store.get(payload['episode_id'])
        frame_bytes = stored.frames.nbytes
        job = owner.acquire_imagine('imagine')
        self.assertEqual(job['start'], -3)
        self.assertEqual(job['seed_anchor'], 0)
        self.assertEqual(job['seed_bucket'], 'success:early')
        self.assertEqual(job['valid_mask'], [False, False, False, True])
        self.assertEqual(job['action_mask'], [False] * 4)
        self.assertEqual(job['frame_indices'], [0] * 4)
        self.assertEqual(job['seed_episode_action_steps'], 5)
        self.assertFalse(job['real_anchor_terminal'])
        self.assertEqual(owner.frame_store.stats()['image_bytes'], frame_bytes)
        owner.finish_imagine(job['token'], 2)
        self.assertEqual(owner.pipeline.stats()['policy_added'], 2)

    def test_dense_targets_are_attached_to_actual_window_payload_and_cohort(self):
        from fresh_reward_targets import dense_reward_specification
        world = schema()
        world['reward_semantics'] = 'delta_native_score'
        world['backend_options'] = {'fresh': {'dense_reward': True},
                                    'native_reward_head': dense_reward_specification(.99)}
        owner = Owner(world, window=40)
        payload = owner_payload(count=42)
        lease = owner.acquire_real('real')
        owner.finish_real_episode(lease['token'], payload)
        sample = owner.buffer[-1]
        derived = sample.backend_payload
        # This is an integration assertion on the actual projected frame slice,
        # not a second implementation of dense target generation.
        np.testing.assert_array_equal(derived['success'], payload['success'][2:42])
        targets = derived['reward_targets']
        self.assertEqual(float(targets[-1]), 1.0)
        self.assertTrue(0 < float(targets[0]) < float(targets[-1]))
        # Repeated references cover bounded cohort arithmetic and spec routing;
        # they are not evidence of 1024 independent real training windows.
        owner.buffer = [sample] * 1024
        reservation = owner.reserve_windows(1024)
        cohort = owner.window_tickets[reservation['ticket_id']][1]['capture_meta']['reward_cohort']
        self.assertEqual(cohort['protocol'], 'weaver_discounted_success_value_cohort_v1')
        self.assertEqual(cohort['balance'], 'none')
        owner.release_windows(reservation['ticket_id'])


if __name__ == '__main__':
    unittest.main()
