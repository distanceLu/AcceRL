"""CPU evidence: capacity math, real replay/budget integration, RPC lifecycle.

No Ray/CUDA or GPU speedup is claimed by these deterministic tests.
"""
import asyncio
import ast
from collections import Counter
import copy
import hashlib
import json
from dataclasses import replace
from itertools import combinations
import random
from pathlib import Path
from types import SimpleNamespace
import unittest

from fresh_config import FreshConfig
from fresh_policy_replay import SegmentReplay
from inventory_lookahead import forecast, _flow, LookaheadAdmission
from pipeline_budget import PipelineBudgetConfig
from pipeline_runtime import EpisodeReplayMixin
from test_fresh_policy_replay_cpu import segment
from test_fresh_replay_pressure_cpu import observed_p30_budget, actor, add_segments, demand


def store(count=432, uses=0, policy=30, world=6):
    replay = SegmentReplay(10000, FreshConfig(replay_age_limits=True,
        real_policy_replay=True), rng=random.Random(7))
    if count:
        replay.add(segment('imag', count=count, policy=policy, world=world))
    replay.set_versions(30, 6)
    for item in replay.samples.values():
        item['uses'] = uses
    return replay


def budget(clock=None):
    value = observed_p30_budget(clock)
    value.config = replace(value.config, fresh_lookahead_updates=2)
    value._lookahead = LookaheadAdmission(2, value._clock)
    return value


def advice(revision=1, count=432, observed=0., **kw):
    return dict(revision=revision, new_chunks_needed=count, horizon=2,
        versions=[30, 6], admitted_imagined_chunks=5608, observed_monotonic=observed, **kw)


class ForecastTests(unittest.TestCase):
    def test_distinct_samples_not_just_remaining_uses(self):
        plan = forecast(store(count=108), 2)
        self.assertEqual(plan['new_chunks_needed'], 324)
        self.assertEqual(plan['feasible_exposures'], 216)

    def test_one_use_samples_and_future_reusable_samples_can_interleave(self):
        plan = forecast(store(uses=3), 2)
        self.assertEqual(plan['new_chunks_needed'], 216)
        self.assertEqual(forecast(store(), 2)['new_chunks_needed'], 0)

    def test_versions_expire_before_future_draws(self):
        self.assertEqual(forecast(store(policy=14), 2)['new_chunks_needed'], 432)
        replay = store(world=3)
        replay.set_versions(34, 6)
        self.assertEqual(forecast(replay, 2)['new_chunks_needed'], 432)

    def test_partial_publication_and_unpublished_store_never_issue_advice(self):
        replay = store()
        replay.versions = None
        self.assertFalse(forecast(replay, 2)['ready'])
        replay.set_versions(34, 7)
        self.assertEqual(forecast(replay, 2)['reason'], 'publication_in_progress')

    def test_reserved_last_uses_real_source_and_future_versions_do_not_count(self):
        replay = store(uses=3)
        for item in replay.samples.values():
            item['reserved'] = 1
        replay.add(segment('real', 'real', 30, 6, 432))
        replay.add(segment('future', policy=31, world=6, count=432))
        self.assertEqual(forecast(replay, 2)['new_chunks_needed'], 432)

    def test_forecast_is_read_only_including_random_state(self):
        replay = store()
        old_samples, old_rng = copy.deepcopy(replay.samples), replay.rng.getstate()
        old_bytes = replay.segments['imag'].behaviour_logits.tobytes()
        forecast(replay, 4)
        self.assertEqual(replay.samples, old_samples)
        self.assertEqual(replay.rng.getstate(), old_rng)
        self.assertEqual(replay.segments['imag'].behaviour_logits.tobytes(), old_bytes)
        self.assertFalse(replay.reservations)

    def test_reject_unreviewed_recipes_and_horizons(self):
        for horizon in (0, 5, True, 1.5):
            with self.subTest(horizon=horizon), self.assertRaises(ValueError):
                forecast(store(), horizon)
        for kw in ({'batch_size': 288}, {'world_interval': 1}):
            with self.assertRaises(ValueError):
                forecast(store(), 2, **kw)
        replay = store()
        replay.config = replace(replay.config, max_sample_uses=8)
        with self.assertRaises(ValueError):
            forecast(replay, 2)

    def test_grouped_flow_matches_exhaustive_individual_sample_assignments(self):
        rng = random.Random(18)
        for case in range(30):
            horizon, quota = 3, 2
            rows = [(rng.randint(1, 3), tuple(rng.choice([False, True]) for _ in range(horizon)))
                    for _ in range(rng.randint(1, 5))]
            groups = Counter(rows)
            def brute(step, remaining):
                if step == horizon:
                    return 0
                eligible = [i for i, value in enumerate(remaining) if value and rows[i][1][step]]
                best = 0
                for size in range(min(quota, len(eligible)) + 1):
                    for selected in combinations(eligible, size):
                        after = tuple(v - (i in selected) for i, v in enumerate(remaining))
                        best = max(best, size + brute(step + 1, after))
                return best
            with self.subTest(case=case):
                self.assertEqual(_flow(groups, quota, horizon, 0),
                                 brute(0, tuple(row[0] for row in rows)))

    def test_learning_files_and_off_on_profiles_match_frozen_parent(self):
        root = Path(__file__).parent
        original = json.loads((root / 'parent_provenance.json').read_text('utf8'))['sha256']
        # JIT explicitly extends Replay transport and FreshConfig. Byte identity
        # still protects unchanged reward/World/age learning files;
        # cycle controller sampling-unit changes have dedicated trajectory tests.
        # JIT-off behavior is checked against the frozen parent below.
        for name in ('fresh_reward_targets.py',
                     'examples/fresh_04_real.json', 'examples/cost_01_fixed.json',
                     'pipeline_default.json',
                     'weaver_online_engine.py', 'weaver_native_reward.py',
                     'collective_phase_gate.py'):
            with self.subTest(file=name):
                self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), original[name])
        parent = root / 'examples/execution_parent_cost_03.json'
        self.assertEqual(hashlib.sha256(parent.read_bytes()).hexdigest(), original['examples/cost_03_adaptive.json'])
        current = json.loads((root / 'examples/cost_03_adaptive.json').read_text('utf8'))
        self.assertTrue(current.pop('execution')['authoritative_production'])
        for n in range(3,9):
            self.assertEqual(current['initial_unit_costs'].pop('policy_infer:refresh:batch'+str(n)), .25)
        self.assertEqual(current, json.loads(parent.read_text('utf8')))
        off, on = [json.loads((root / 'examples' / name).read_text('utf8')) for name in
                   ('pipeline_lookahead_00_off.json', 'pipeline_lookahead_01_two_updates.json')]
        self.assertEqual(off.pop('fresh_lookahead_updates'), 0)
        self.assertEqual(on.pop('fresh_lookahead_updates'), 2)
        self.assertEqual(off, on)
        self.assertEqual(on, json.loads((root / 'pipeline_default.json').read_text('utf8')))

    def test_actual_driver_starts_and_stops_feedback_and_freezes_config(self):
        root = Path(__file__).parent
        tree = ast.parse((root / 'train.py').read_text('utf8'))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        calls = [ast.unparse(node.func) for node in ast.walk(main) if isinstance(node, ast.Call)]
        self.assertIn('replay.configure_inventory_lookahead.remote', calls)
        self.assertEqual(calls.count('replay.stop_inventory_lookahead.remote'), 2)
        runner = (root / 'runner.py').read_text('utf8')
        self.assertIn("state['pipeline_config'] = budget.to_dict()", runner)
        self.assertIn("env['ACCERL_PIPELINE_CONFIG'] = state['pipeline_config_path']", runner)


class AdmissionTests(unittest.TestCase):
    def test_preemptively_uses_earned_credit_without_reactive_shortage(self):
        value = budget(lambda: 0.)
        self.assertFalse(value.pressure_status()['pressure_active'])
        self.assertIsNone(value.acquire_imagine('before'))
        value.set_lookahead(advice(count=5))
        for worker in ('one', 'two'):
            lease = value.acquire_imagine(worker)
            self.assertIsNotNone(lease)
            value.finish_imagine(lease['token'], 4)
        self.assertIsNone(value.acquire_imagine('done'))
        self.assertEqual(value.policy_added, 5616)
        self.assertEqual(value._credits, 1720.)
        self.assertFalse(value.lookahead_status()['credit_refill_performed'])

    def test_global_inflight_credit_and_reactive_budget_are_shared(self):
        value = budget(lambda: 0.)
        value.set_lookahead(advice(count=4))
        value.set_replay_pressure(demand(imagined=4, observed=0.))
        lease = value.acquire_imagine('one')
        self.assertIsNone(value.acquire_imagine('two'))
        value.finish_imagine(lease['token'], 4)
        self.assertIsNone(value.acquire_imagine('three'))
        value.set_lookahead(advice(2, count=100))
        leases = [value.acquire_imagine(i) for i in range(4)]
        self.assertTrue(all(leases))
        self.assertIsNone(value.acquire_imagine('fifth'))
        for item in leases:
            value.cancel_imagine(item['token'])
        value._credits = 3.
        self.assertIsNone(value.acquire_imagine('unfunded'))

    def test_expiry_clear_versions_and_out_of_order_messages(self):
        clock = [0.]
        value = budget(lambda: clock[0])
        value.set_lookahead(advice())
        clock[0] = 2.
        self.assertIsNone(value.acquire_imagine('expired'))
        self.assertFalse(value.set_lookahead(advice(2))['accepted'])
        value.set_lookahead(advice(4, count=0, observed=2.))
        value.set_lookahead(advice(3, observed=2.))
        self.assertIsNone(value.acquire_imagine('delayed'))
        value.set_lookahead(advice(5, observed=2.))
        value.set_production_versions(31, 6)
        self.assertIsNone(value.acquire_imagine('published'))
        self.assertFalse(value.set_lookahead(advice(6, observed=2.))['accepted'])

    def test_completion_after_snapshot_is_not_produced_again(self):
        value = budget(lambda: 0.)
        value.set_lookahead(advice(count=4))
        lease = value.acquire_imagine('one')
        snapshot = advice(2, count=4)
        value.finish_imagine(lease['token'], 4)
        value.set_lookahead(snapshot)
        self.assertIsNone(value.acquire_imagine('duplicate'))

    def test_stop_and_runtime_owner_boundary(self):
        value = budget(lambda: 0.)
        owner = SimpleNamespace(pipeline=value, pipeline_stopped=False)
        EpisodeReplayMixin.set_fresh_lookahead(owner, advice())
        value.stop()
        self.assertFalse(value.lookahead_status()['can_spend_existing_credit'])
        self.assertEqual(value.set_lookahead(advice(2)), {'stopped': True})
        owner.pipeline_stopped = True
        self.assertEqual(EpisodeReplayMixin.set_fresh_lookahead(owner, advice(3)), {'stopped': True})

    def test_configuration_and_malformed_control_messages_fail_closed(self):
        for horizon in (-1, True, 5, 1.5):
            with self.subTest(horizon=horizon), self.assertRaises(ValueError):
                PipelineBudgetConfig(fresh_lookahead_updates=horizon)
        with self.assertRaises(ValueError):
            observed_p30_budget().set_lookahead(advice())
        for patch in ({'revision': True}, {'new_chunks_needed': 433},
                      {'new_chunks_needed': -1}, {'versions': [30, True]},
                      {'observed_monotonic': float('nan')}, {'horizon': 4},
                      {'admitted_imagined_chunks': -1}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                budget(lambda: 0.).set_lookahead({**advice(), **patch})


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_replay_pump_supplies_while_next_batch_is_already_ready(self):
        value = budget()
        replay, _ = actor(value)
        add_segments(replay, 'old-imagined', 432)
        for item in replay.fresh_store.samples.values():
            if replay.fresh_store.segments[item['segment_id']].source == 'imagined':
                item['uses'] = 3
        # Replay historical counter and budget completion counter are aligned.
        value.policy_added = replay.fresh_admitted_chunks['imagined']
        entered = asyncio.Event()
        async def submit(snapshot):
            result = value.set_lookahead(snapshot)
            entered.set()
            return result
        replay.fresh_production_budget.set_fresh_lookahead = SimpleNamespace(remote=submit)
        receipt = await replay.configure_inventory_lookahead(2, 576, 5)
        await asyncio.wait_for(entered.wait(), 1.)
        self.assertTrue(receipt['enabled'])
        self.assertIsNotNone(value.acquire_imagine('overlapped'))
        self.assertFalse(value.pressure_status()['pressure_active'])
        # The actual random Replay can already reserve the present full cohort.
        token, ids = replay.fresh_store.reserve(576)
        self.assertEqual(len(ids), 576)
        replay.fresh_store.cancel(token)
        with self.assertRaises(RuntimeError):
            await replay.configure_inventory_lookahead(2, 576, 5)
        await replay.stop_inventory_lookahead()
        self.assertEqual(value.lookahead_status()['target_new_chunks'], 0)
        self.assertTrue(replay.inventory_lookahead.task.done())

    async def test_rpc_failure_is_owned_and_visible_to_actual_sample(self):
        value = budget()
        replay, _ = actor(value)
        async def fail(snapshot):
            raise RuntimeError('test transport failure')
        replay.fresh_production_budget.set_fresh_lookahead = SimpleNamespace(remote=fail)
        await replay.configure_inventory_lookahead(2, 576, 5)
        await asyncio.wait_for(replay.inventory_lookahead.task, 1.)
        with self.assertRaisesRegex(RuntimeError, 'refusing silent fallback'):
            await replay.sample(576)
        self.assertFalse(replay.fresh_store.reservations)
        self.assertFalse(replay._fresh_feedback_pending)

    async def test_stop_during_pending_rpc_fences_late_admission(self):
        value = budget()
        replay, _ = actor(value)
        pending, release = asyncio.Event(), asyncio.Event()
        async def submit(snapshot):
            if snapshot['revision'] == 1:
                pending.set()
                await release.wait()
            return value.set_lookahead(snapshot)
        replay.fresh_production_budget.set_fresh_lookahead = SimpleNamespace(remote=submit)
        await replay.configure_inventory_lookahead(2, 576, 5)
        await asyncio.wait_for(pending.wait(), 1.)
        stop = asyncio.create_task(replay.stop_inventory_lookahead())
        await asyncio.sleep(0)
        release.set()
        await asyncio.wait_for(stop, 1.)
        self.assertIsNone(value.acquire_imagine('late-rpc'))
        self.assertFalse(replay._fresh_feedback_pending)


if __name__ == '__main__':
    unittest.main()
