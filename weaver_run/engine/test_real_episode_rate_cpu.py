import os
import unittest
from unittest.mock import patch

import real_collection
from test_model_free_foundation_cpu import POOL, collection_config, completion


class RealEpisodeRateTests(unittest.TestCase):
    def owner(self, rate=30, budget=300):
        return real_collection.RealCollectionCoordinator(collection_config(
            real_workers=8, max_real_in_flight=8, real_env_step_budget=budget),
            real_episodes_per_minute=rate)

    def test_serialized_arrivals_have_one_global_slot_and_keep_existing_lease(self):
        owner = self.owner()
        with patch('real_collection.time.monotonic', return_value=100.) as clock:
            first = owner.acquire_real('a', POOL)
            # Denied polls must not scan the scheduler's entire episode journal.
            with patch.object(owner.initial_state_scheduler, 'status',
                              side_effect=AssertionError('rate-limited poll scanned the episode history')):
                for worker in range(7):
                    self.assertIsNone(owner.acquire_real(worker, POOL))
                self.assertEqual(owner.acquire_real('a', POOL), first)
            status = owner.pipeline_status()
            self.assertEqual(status['initial_states']['episodes_assigned'], 1)
            self.assertEqual(status['initial_states']['reserved_env_steps'], 15)
            self.assertEqual(status['budget']['real_rate_deferred_polls'], 7)
            self.assertEqual(status['budget']['real_episodes_per_minute'], 30)
            self.assertEqual(status['budget']['real_rate_wait_seconds'], 2.)
            clock.return_value = 101.999
            self.assertIsNone(owner.acquire_real('b', POOL))
            clock.return_value = 102.
            self.assertEqual(owner.acquire_real('b', POOL)['token'], 'real:2')
            self.assertIsNone(owner.acquire_real('c', POOL))

    def test_long_pause_with_no_leases_cannot_accumulate_a_burst(self):
        owner = self.owner(rate=12)
        with patch('real_collection.time.monotonic', return_value=0.) as clock:
            first = owner.acquire_real('a', POOL)
            owner.set_paused(True)
            owner.finish_real_episode(first['token'], completion(first))
            clock.return_value = 300.
            self.assertIsNone(owner.acquire_real('b', POOL))
            owner.set_paused(False)
            self.assertEqual(owner.acquire_real('b', POOL)['token'], 'real:2')
            self.assertIsNone(owner.acquire_real('c', POOL))
            self.assertEqual(owner.pipeline_status()['budget']['real_rate_wait_seconds'], 5.)

    def test_drained_budget_and_stop_remain_visible_while_rate_limited(self):
        owner = self.owner(budget=15)
        with patch('real_collection.time.monotonic', return_value=1.):
            lease = owner.acquire_real(0, POOL)
            owner.finish_real_episode(lease['token'], completion(lease))
            self.assertTrue(owner.acquire_real(1, POOL)['budget_exhausted'])
            owner.stop_pipeline()
            self.assertTrue(owner.acquire_real(1, POOL)['stopped'])

    def test_zero_keeps_all_eight_slots_available_at_same_instant(self):
        owner = self.owner(rate=0)
        with patch('real_collection.time.monotonic', return_value=1.):
            jobs = [owner.acquire_real(i, POOL) for i in range(8)]
            self.assertEqual(len({job['token'] for job in jobs}), 8)
            self.assertEqual(owner.pipeline_status()['budget']['real_rate_deferred_polls'], 0)

    def test_rate_is_finite_and_scoped_to_mutually_exclusive_eight_worker_recipe(self):
        recipe = dict(model_free=True, real_workers=8, trajectory_jit=True,
                      real_sampling='uniform_blocks', inventory_gate=False)
        parsed = real_collection.configured_episode_rate
        self.assertEqual(parsed('12.5', **recipe), 12.5)
        for value in ('-1', 'nan', 'inf', '', 'yes', '1e-320'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parsed(value, **recipe)
        for change in (dict(model_free=False), dict(real_workers=4), dict(trajectory_jit=False),
                       dict(real_sampling='chunks'), dict(inventory_gate=True)):
            settings = dict(recipe, **change)
            self.assertEqual(parsed('0', **settings), 0.)
            with self.subTest(change=change), self.assertRaises(ValueError):
                parsed('12', **settings)

    def test_formal_rate_reaches_single_global_owner_from_real_driver(self):
        for mode in ('uniform_blocks', 'first_use_blocks'):
            with self.subTest(mode=mode):
                self.check_formal_rate_owner(mode)

    def check_formal_rate_owner(self, mode):
        import test_main_startup_cpu as startup
        context = startup.Context()
        ns = context.namespace()
        ns['os'].environ.update(ACCERL_REAL_EPISODES_PER_MINUTE='22.2', ACCERL_TRAJECTORY_JIT='1',
                                ACCERL_REAL_SAMPLING=mode)
        args = startup.arguments(validate_only=True, overrides=(
            '--model-free', '--num-eval-workers', '0', '--batch-autotune',
            '--fixed-validation-interval-env-steps', '0',
            '--pipeline-config', str(startup.HERE.parent/'recipe/pipeline_config.json')))
        original = startup.Actor.call
        def call(actor, method, args, kwargs):
            if method in ('configure_fresh', 'set_fresh_versions', 'set_fresh_replay_versions'):
                return True
            return original(actor, method, args, kwargs)
        with patch.dict(os.environ, ACCERL_MODEL_FREE_REAL_WORKERS='8'), patch.object(startup.Actor,'call', call):
            startup.execute_definition(startup.startup_ast(), ns)(args)
        owners = [actor for actor in context.actors if actor.name == 'RealCollectionCoordinator']
        self.assertEqual(len(owners), 1)
        owner = real_collection.RealCollectionCoordinator(*owners[0].positional, **owners[0].kwargs)
        pool = dict(POOL, max_action_steps=150, benchmark_action_horizon=220)
        with patch('real_collection.time.monotonic', return_value=10.) as clock:
            self.assertIsNotNone(owner.acquire_real(0, pool))
            clock.return_value = 12.7027
            self.assertIsNone(owner.acquire_real(1, pool))
            clock.return_value = 12.7028
            self.assertIsNotNone(owner.acquire_real(1, pool))


if __name__ == '__main__':
    unittest.main()
