"""CPU state replay of the measured P30 supply deadlock and owned-RPC cleanup.

This exercises the real PipelineBudget, SegmentReplay and ReplayBufferActor
methods, with synthetic observations and async stand-ins for Ray transport.
It is not a model, CUDA, Ray or throughput acceptance test.
"""
import asyncio
import time
from contextlib import ExitStack
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import fresh_replay_pressure as pressure
from fresh_config import FreshConfig
from fresh_policy_replay import FreshReplayNotReady
from pipeline_budget import PipelineBudget, PipelineBudgetConfig
from pipeline_runtime import EpisodeReplayMixin
from test_fresh_policy_replay_cpu import runtime_replay_class, segment, Ledger, Writer


def observed_p30_budget(clock=None):
    """Restore the measured counters, not an invented continuation/checkpoint.

    data_pipeline_status row 1007551: 5608 produced, 12960 imagined draws,
    1728 earned unused credits, paused, zero active/reserved imagination.
    Seed frame contents are synthetic; their admission still uses real code.
    """
    config = replace(PipelineBudgetConfig(), max_episode_frames=100,
                     bytes_per_frame=10, max_resident_frame_bytes=100000)
    budget = PipelineBudget(config, 4, **({'clock':clock} if clock else {}))
    budget.policy_added = 5608
    budget.policy_sampled = 12960
    budget._credits = 1728.0
    budget._paused = True
    budget.set_production_versions(30,6)
    add_episode(budget, 'initial-seeds')
    return budget


def add_episode(budget, name):
    lease = budget.acquire_real('collector')
    if lease is None:
        raise AssertionError('fixture could not acquire a bounded seed episode')
    budget.complete_real(lease['token'], name, 64, 4, 640)


def demand(revision=1, real=0, imagined=432, observed=None, baseline=5608):
    return dict(revision=revision, request_id='P31', versions=[30,6],
                observed_monotonic=time.monotonic() if observed is None else observed,
                admitted_imagined_chunks=baseline,
                missing={'real':real,'imagined':imagined})


def add_segments(replay, prefix, count, *, source='imagined', policy=30, world=6):
    for offset in range(0, count, 64):
        replay.add_segment(segment(prefix+':'+str(offset), source,
            policy=policy, world=world, count=min(64,count-offset)))


def actor(budget, *, count=1892, clear_error=False):
    async def evaluate(inputs, version):
        return dict(policy_version=version, values=[item['value']+2 for item in inputs])
    calls = []
    async def submit(snapshot):
        calls.append(snapshot)
        if clear_error and snapshot['missing'] == {'real':0,'imagined':0}:
            raise RuntimeError('clear transport failed')
        return budget.set_replay_pressure(snapshot)
    replay = runtime_replay_class()(20000, FreshConfig(
        refresh_gae=True, replay_age_limits=True, real_policy_replay=True).to_dict())
    replay.capture_ledger, replay.telemetry = Ledger(), Writer()
    critics = [SimpleNamespace(evaluate_values=SimpleNamespace(remote=evaluate)) for _ in range(4)]
    sink = SimpleNamespace(set_fresh_replay_pressure=SimpleNamespace(remote=submit))
    replay.configure_fresh(critics, sink)
    replay.set_fresh_versions(30,6)
    if count:
        add_segments(replay, 'real', count, source='real')
    replay.fresh_admitted_chunks['imagined'] = 5608  # historical admits may be evicted
    return replay, calls


def fast_timers():
    stack = ExitStack()
    for name, value in (('POLL_SECONDS',.001), ('PRESSURE_SECONDS',.01),
                        ('DIAGNOSTIC_SECONDS',.04), ('CREDIT_SHORTFALL_SECONDS',.05)):
        stack.enter_context(patch.object(pressure, name, value))
    return stack


class BudgetPressureTests(unittest.TestCase):
    def test_observed_p30_circular_wait_and_credit_only_wakeup(self):
        budget = observed_p30_budget()
        self.assertEqual(4096+12960*.25-5608, 1728)
        self.assertIsNone(budget.acquire_imagine('waiting'))
        self.assertEqual(budget.stats()['last_block_reason'], 'policy_demand')
        before = budget.stats()
        state = budget.set_replay_pressure(demand())
        self.assertFalse(state['credit_refill_performed'])
        self.assertEqual(state['available_policy_credits'], 1728)
        lease = budget.acquire_imagine('waiting')
        self.assertIsNotNone(lease)
        self.assertEqual(budget.stats()['available_policy_credits'], 1724)
        self.assertEqual(budget.stats()['reserved_policy_samples'], 4)
        self.assertEqual(budget.policy_sampled, before['policy_sampled'])
        self.assertTrue(budget.stats()['policy_paused'])
        budget.finish_imagine(lease['token'], 4)
        self.assertEqual(budget.policy_added, 5612)

    def test_completed_seed_cannot_spend_same_deficit_twice(self):
        budget = observed_p30_budget()
        budget.set_replay_pressure(demand(imagined=5))
        for index in range(2):
            lease = budget.acquire_imagine(str(index))
            self.assertIsNotNone(lease)
            budget.finish_imagine(lease['token'],4)
        self.assertIsNone(budget.acquire_imagine('third'))
        self.assertEqual(budget.policy_added, 5616)  # deficit five, one-seed tail only
        self.assertEqual(budget.pressure_status()['pressure_remaining_imagined'],0)
        self.assertEqual(budget.stats()['available_policy_credits'],1720)

    def test_reserved_seeds_and_global_inflight_limit_still_apply(self):
        budget = observed_p30_budget()
        budget.set_replay_pressure(demand(imagined=100))
        leases = [budget.acquire_imagine(str(index)) for index in range(4)]
        self.assertTrue(all(leases))
        self.assertIsNone(budget.acquire_imagine('fifth'))
        self.assertEqual(budget.stats()['last_block_reason'],'imagine_in_flight')
        for lease in leases:
            budget.cancel_imagine(lease['token'])
        self.assertEqual(budget.stats()['available_policy_credits'],1728)
        budget.set_replay_pressure(demand(2, imagined=4))
        self.assertIsNotNone(budget.acquire_imagine('one'))
        self.assertIsNone(budget.acquire_imagine('two'))

    def test_clear_expiry_and_late_messages_do_not_reopen_production(self):
        clock = [0.]
        budget = observed_p30_budget(lambda:clock[0])
        budget.set_replay_pressure(demand(observed=0))
        clock[0] = 2.01
        self.assertIsNone(budget.acquire_imagine('expired'))
        budget.set_replay_pressure(demand(3, imagined=0, observed=2.01))
        budget.set_replay_pressure(demand(2, observed=2.01))
        self.assertIsNone(budget.acquire_imagine('late'))
        self.assertEqual(budget.pressure_status()['pressure_revision'],3)
        self.assertEqual(budget.stats()['available_policy_credits'],1728)

    def test_real_only_shortage_and_unfunded_seed_do_not_mint_credit(self):
        budget = observed_p30_budget()
        budget.set_replay_pressure(demand(real=144, imagined=0))
        self.assertIsNone(budget.acquire_imagine('real-shortage'))
        budget._credits = 3
        budget.set_replay_pressure(demand(2))
        self.assertIsNone(budget.acquire_imagine('unfunded'))
        self.assertEqual(budget.stats()['available_policy_credits'],3)

    def test_strict_pressure_fields(self):
        budget = observed_p30_budget()
        for changes in ({'revision':True}, {'missing':{'real':0,'imagined':-1}},
                        {'versions':[30,True]}, {'request_id':''},
                        {'admitted_imagined_chunks':-1}, {'observed_monotonic':float('nan')}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                budget.set_replay_pressure({**demand(), **changes})

    def test_runtime_owner_propagates_stopped_instead_of_waking(self):
        owner = SimpleNamespace(pipeline=observed_p30_budget(), pipeline_stopped=True)
        self.assertEqual(EpisodeReplayMixin.set_fresh_replay_pressure(owner,demand()),{'stopped':True})
        self.assertIsNone(owner.pipeline._replay_pressure)
        owner.pipeline_stopped = False
        result = EpisodeReplayMixin.set_fresh_replay_pressure(owner,demand())
        self.assertTrue(result['pressure_can_spend_existing_credit'])

    def test_publish_clears_pressure_and_rejects_delayed_old_version(self):
        budget = observed_p30_budget()
        budget.set_replay_pressure(demand())
        budget.set_production_versions(31,6)
        self.assertIsNone(budget.acquire_imagine('after-publication'))
        result = budget.set_replay_pressure(demand(2))
        self.assertTrue(result['pressure_rejected'])
        self.assertIsNone(budget.acquire_imagine('old-version-rpc'))
        with self.assertRaises(ValueError):
            budget.set_production_versions(30,6)

    def test_expired_snapshot_is_not_renewed_on_receipt(self):
        budget = observed_p30_budget()
        result = budget.set_replay_pressure(demand(observed=time.monotonic()-3.))
        self.assertTrue(result['pressure_rejected'])
        self.assertIsNone(budget.acquire_imagine('late-snapshot'))

    def test_completion_between_snapshot_and_rpc_receipt_counts_toward_deficit(self):
        budget = observed_p30_budget()
        budget.set_replay_pressure(demand(imagined=4))
        snapshot = demand(2,imagined=4,baseline=5608)
        lease = budget.acquire_imagine('existing')
        budget.finish_imagine(lease['token'],4)
        budget.set_replay_pressure(snapshot)
        self.assertEqual(budget.pressure_status()['pressure_supplied_since_snapshot'],4)
        self.assertIsNone(budget.acquire_imagine('double-supply'))


class ActualReplayPressureTests(unittest.IsolatedAsyncioTestCase):
    async def test_feedback_timeout_and_cancel_are_bounded_without_remote_retry(self):
        owner = SimpleNamespace()
        release = asyncio.Event()
        entered = []
        async def remote():
            entered.append(True)
            await release.wait()
            return 'late acknowledgement'
        with patch.object(pressure,'RPC_TIMEOUT_SECONDS',.02):
            with self.assertRaisesRegex(RuntimeError,'feedback RPC timed out'):
                await asyncio.wait_for(pressure.bounded_feedback_rpc(owner,remote()),1.)
            self.assertEqual(len(owner._fresh_feedback_pending),1)
            task = asyncio.create_task(pressure.bounded_feedback_rpc(owner,remote()))
            await asyncio.sleep(.005)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task,1.)
        self.assertEqual(len(entered),2)
        self.assertEqual(len(owner._fresh_feedback_pending),2)
        release.set()
        await asyncio.gather(*list(owner._fresh_feedback_pending))
        await asyncio.sleep(0)
        self.assertFalse(owner._fresh_feedback_pending)

    async def test_p30_actual_sample_reaches_exact_cohort_and_current_critic(self):
        budget = observed_p30_budget()
        replay, calls = actor(budget)
        # Both exclusion reasons present in the actual last inventory.
        add_segments(replay,'world-expired',310,policy=20,world=2)
        add_segments(replay,'policy-expired',7501,source='real',policy=13,world=6)
        replay.fresh_admitted_chunks['imagined'] = 5608
        self.assertEqual(pressure.shortage(replay.fresh_store,576)['missing'],{'real':0,'imagined':432})
        with self.assertRaises(FreshReplayNotReady):
            replay.fresh_store.reserve(576)
        self.assertIsNone(budget.acquire_imagine('before-fix'))
        done = asyncio.Event()
        async def producer():
            generated = 0
            while not done.is_set():
                if not budget.stats()['pending_seed_count']:
                    add_episode(budget, 'seed-'+str(generated))
                lease = budget.acquire_imagine('cpu-transport-fixture')
                if lease:
                    replay.add_segment(segment('new-'+str(generated),policy=30,world=6,count=4))
                    budget.finish_imagine(lease['token'],4)
                    generated += 4
                await asyncio.sleep(0 if lease else .001)
            return generated
        with fast_timers():
            task = asyncio.create_task(producer())
            try:
                result = await asyncio.wait_for(replay.sample(576),10.)
            finally:
                done.set()
                generated = await task
        metadata = result[-1]
        self.assertEqual(metadata['source_counts'],{'real':144,'imagined':432})
        self.assertTrue(metadata['critic_refreshed'])
        self.assertFalse(metadata['rewards_refreshed'])
        self.assertEqual(metadata['fresh_versions'],[30,6])
        self.assertEqual(len(set(metadata['sample_ids'])),576)
        self.assertFalse(any(sid.startswith(('world-expired:','policy-expired:')) for sid in metadata['sample_ids']))
        self.assertEqual(generated,432)
        self.assertEqual(budget.stats()['available_policy_credits'],1296)
        self.assertEqual(budget.policy_sampled,12960)
        self.assertEqual(sum(x['uses'] for x in replay.fresh_store.samples.values()),0)
        self.assertEqual(calls[-1]['missing'],{'real':0,'imagined':0})
        token = metadata['fresh_reservation_token']
        for rank in range(4):
            replay.commit_fresh_use(token,metadata['sample_ids'][rank*144:(rank+1)*144])
        self.assertFalse(replay.fresh_store.reservations)
        self.assertEqual(sum(x['uses'] for x in replay.fresh_store.samples.values()),576)
        self.assertLessEqual(max(x['uses'] for x in replay.fresh_store.samples.values()),4)

    async def test_cancel_clears_advisory_and_does_not_reserve_uses(self):
        budget = observed_p30_budget()
        replay, calls = actor(budget)
        with fast_timers():
            task = asyncio.create_task(replay.sample(576))
            await asyncio.sleep(.025)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(calls[-1]['missing'],{'real':0,'imagined':0})
        self.assertFalse(replay.fresh_store.reservations)
        self.assertIsNone(budget.acquire_imagine('cancelled'))

    async def test_cancel_during_remote_acceptance_waits_then_clears(self):
        budget = observed_p30_budget()
        replay, calls = actor(budget)
        accepted = asyncio.Event()
        async def delayed(snapshot):
            if snapshot['missing']['imagined']:
                await asyncio.sleep(.02)
                result = budget.set_replay_pressure(snapshot)
                accepted.set()
                return result
            return budget.set_replay_pressure(snapshot)
        replay.fresh_production_budget.set_fresh_replay_pressure.remote = delayed
        with fast_timers():
            task = asyncio.create_task(replay.sample(576))
            await asyncio.sleep(.005)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(accepted.is_set())
        self.assertFalse(budget.pressure_status()['pressure_active'])

    async def test_clear_transport_failure_releases_new_reservation(self):
        budget = observed_p30_budget()
        replay, calls = actor(budget, clear_error=True)
        async def fill():
            await asyncio.sleep(.01)
            add_segments(replay,'supply',432)
        with fast_timers():
            task = asyncio.create_task(fill())
            with self.assertRaisesRegex(RuntimeError,'clear transport failed'):
                await replay.sample(576)
            await task
        self.assertFalse(replay.fresh_store.reservations)
        self.assertTrue(all(x['reserved']==0 for x in replay.fresh_store.samples.values()))
        self.assertTrue(all(x['uses']==0 for x in replay.fresh_store.samples.values()))

    async def test_unfunded_credit_is_bounded_even_if_real_collection_progresses(self):
        budget = observed_p30_budget()
        budget._credits = 0
        replay, calls = actor(budget)
        real_progress = []
        async def collect_real():
            for index in range(20):
                replay.add_segment(segment('real-progress-'+str(index),'real',policy=30,world=6,count=1))
                real_progress.append(index)
                await asyncio.sleep(.005)
        with fast_timers():
            task = asyncio.create_task(collect_real())
            try:
                with self.assertRaisesRegex(RuntimeError,'refusing to mint credits'):
                    await asyncio.wait_for(replay.sample(576),1.)
            finally:
                task.cancel()
                await asyncio.gather(task,return_exceptions=True)
        self.assertGreater(len(real_progress),2)
        self.assertEqual(budget.stats()['available_policy_credits'],0)
        self.assertTrue(any(event[0]=='fresh_replay_credit_shortfall' for event in replay.telemetry.events))

    async def test_wait_inventory_uses_authoritative_revision_and_is_throttled(self):
        budget = observed_p30_budget()
        replay, calls = actor(budget, count=4)
        updates = []
        update_times = []
        async def update(snapshot):
            updates.append(snapshot)
            update_times.append(pressure.time.monotonic())
        replay.cost_scheduler_enabled = True
        replay.cost_scheduler_broker = SimpleNamespace(update_inventory=SimpleNamespace(remote=update))
        with fast_timers():
            task = asyncio.create_task(replay.sample(576))
            async def enough_updates():
                while len(updates) < 3:
                    await asyncio.sleep(.001)
            await asyncio.wait_for(enough_updates(),2.)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertGreaterEqual(len(updates),3)
        self.assertLessEqual(len(updates),4)
        self.assertTrue(all(b-a >= .009 for a,b in zip(update_times,update_times[1:])))
        self.assertEqual([x['inventory_revision'] for x in updates],list(range(1,len(updates)+1)))
        self.assertTrue(all(x['inventory_versions']==[30,6] for x in updates))
        self.assertTrue(all('admitted_chunks' in x for x in updates))
        self.assertLessEqual(sum(x[0]=='fresh_replay_wait' for x in replay.telemetry.events),2)


if __name__ == '__main__':
    unittest.main()
