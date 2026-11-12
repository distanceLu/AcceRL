"""Deterministic metadata tests; no Ray, CUDA or real-model acceptance."""
import copy
import asyncio
import time
import random
import unittest
from dataclasses import replace
from types import SimpleNamespace

from fresh_config import FreshConfig
from fresh_policy_replay import SegmentReplay
from test_fresh_policy_replay_cpu import segment


def replay(count=432, uses=0, policy=20, world=3, current=(29, 5)):
    store = SegmentReplay(20000, FreshConfig(replay_age_limits=True,
        real_policy_replay=True), rng=random.Random(13))
    if count:
        store.add(segment('imag', count=count, policy=policy, world=world))
        for sample in store.samples.values():
            sample['uses'] = uses
    store.set_versions(*current)
    return store


def plan(store, *, reserved=False, now=100., lead=400., seconds=100.):
    from inventory_refill import DemandTiming, project_refill
    timing = DemandTiming()
    timing.observe('ordinary', seconds)
    timing.observe('joint', seconds*2)
    if reserved:
        timing.reserved(store.versions, now)
    return project_refill(store, timing, {'cohort_lead_seconds':lead}, now)


class ProjectionTests(unittest.TestCase):
    def test_macro_includes_bounded_post_publication_production_work(self):
        from cycle_service_controller import CycleServiceController, CycleSnapshot, InventoryBucket
        snapshot = CycleSnapshot(100., 29, 5,
            (InventoryBucket('r', 'real', 144, 29), InventoryBucket('i', 'imagined', 432, 29, 5)),
            production_refill_chunks=432,
            policy_action_observations_per_real_chunk=1.,
            policy_action_observations_per_imagined_chunk=1.25)
        demand = CycleServiceController().demands(snapshot)
        self.assertEqual(demand['minimum_new_chunks']['imagined'], 432)
        self.assertEqual(demand['policy_action_observations'], 540.)

    def test_world_age_cliff_includes_first_batch_after_publication(self):
        # W2 stock survives P29/W5, but none survives P30/W6 (P31 draw).
        value = plan(replay(world=2), reserved=True)
        self.assertEqual(value['target_versions'], [30, 6])
        self.assertEqual(value['projected_distinct'], 0)
        self.assertEqual(value['new_chunks_needed'], 432)
        self.assertEqual(value['deadline_monotonic'], 300.)

    def test_distinct_samples_cannot_be_replaced_by_four_use_credits(self):
        value = plan(replay(108, world=5))
        self.assertEqual(value['projected_remaining_uses'], 432)
        self.assertEqual(value['projected_distinct'], 108)
        self.assertEqual(value['new_chunks_needed'], 324)

    def test_four_batches_of_uses_are_not_five_batches_of_stock(self):
        value = plan(replay(current=(25, 5), world=5), lead=1000.)
        self.assertEqual(value['target_versions'], [29, 5])
        self.assertEqual(value['new_chunks_needed'], 108)
        self.assertEqual(value['forecast_draws'], 6)

    def test_do_not_produce_early_when_measured_lead_is_short(self):
        value = plan(replay(world=2), reserved=True, lead=20.)
        self.assertEqual(value['new_chunks_needed'], 0)
        self.assertEqual(value['reason'], 'outside_lead_window')

    def test_partial_publication_and_reserved_final_uses(self):
        store = replay(uses=3, world=5)
        for item in store.samples.values():
            item['reserved'] = 1
        self.assertEqual(plan(store)['new_chunks_needed'], 432)
        store.set_versions(30, 5)
        self.assertFalse(plan(store)['ready'])

    def test_projection_changes_no_samples_rewards_or_random_state(self):
        store = replay()
        before = copy.deepcopy(store.samples), store.rng.getstate()
        plan(store, reserved=True)
        self.assertEqual(store.samples, before[0])
        self.assertEqual(store.rng.getstate(), before[1])
        self.assertFalse(store.reservations)

    def test_elapsed_work_advances_deadline_without_counting_stockout(self):
        from inventory_refill import DemandTiming, project_refill
        store = replay(world=2)
        timing = DemandTiming()
        timing.observe('joint', 200.)
        timing.reserved(store.versions, 100.)
        value = project_refill(store, timing, {'cohort_lead_seconds':20.}, 285.)
        self.assertEqual(value['deadline_monotonic'], 300.)
        self.assertEqual(value['new_chunks_needed'], 432)

    def test_unused_later_work_timing_does_not_force_early_refill(self):
        from inventory_refill import DemandTiming, project_refill
        timing = DemandTiming()
        timing.observe('joint', 200.)
        store = replay(world=2)
        timing.reserved(store.versions, 100.)
        value = project_refill(store, timing, {'cohort_lead_seconds':20.}, 100.)
        self.assertEqual(value['reason'], 'outside_lead_window')
        self.assertEqual(value['new_chunks_needed'], 0)

    def test_timing_starts_after_reserve_and_rejects_skip_as_observation(self):
        from inventory_refill import DemandTiming
        timing = DemandTiming()
        timing.reserved((29, 5), 800.)  # Earlier stockout is absent.
        timing.published((30, 6), 1000.)
        self.assertEqual(timing.estimate('joint'), 200.)
        timing.reserved((30, 6), 1100.)
        timing.published((30, 6), 1400.)
        self.assertIsNone(timing.estimate('ordinary'))


def producer(now):
    from test_fresh_replay_pressure_cpu import observed_p30_budget
    value = observed_p30_budget(lambda: now[0])
    value.config = replace(value.config, inventory_refill_enabled=True,
                           seed_reserve_per_bucket=1)
    value.arm_inventory_refill()
    return value


def advice(revision=1, chunks=432, at=100., versions=(30, 6), baseline=5608):
    return dict(revision=revision, new_chunks_needed=chunks,
        versions=list(versions), observed_monotonic=at,
        admitted_imagined_chunks=baseline)


class ProductionTests(unittest.TestCase):
    def test_paused_credit_can_fund_only_accepted_bounded_demand(self):
        now = [100.]
        budget = producer(now)
        self.assertIsNone(budget.acquire_imagine('idle'))
        budget.set_refill_plan(advice(chunks=5))
        for n in range(2):
            lease = budget.acquire_imagine(str(n))
            self.assertIsNotNone(lease)
            budget.finish_imagine(lease['token'], 4)
        self.assertIsNone(budget.acquire_imagine('tail'))
        self.assertEqual(budget.policy_added, 5616)
        self.assertEqual(budget.stats()['available_policy_credits'], 1720.)
        self.assertEqual(budget.policy_sampled, 12960)

    def test_zero_plan_stops_even_a_nonpaused_credit_account(self):
        now = [100.]
        budget = producer(now)
        budget.policy_drawn(432*4)
        self.assertFalse(budget.stats()['policy_paused'])
        budget.set_refill_plan(advice(chunks=0))
        self.assertIsNone(budget.acquire_imagine('idle'))

    def test_ttl_revision_and_version_fences_and_reactive_fallback(self):
        from test_fresh_replay_pressure_cpu import demand
        now = [100.]
        budget = producer(now)
        budget.set_refill_plan(advice())
        now[0] = 106.
        self.assertIsNone(budget.acquire_imagine('expired'))
        budget.set_refill_plan(advice(revision=2, chunks=0, at=106.))
        budget.set_refill_plan(advice(revision=1, at=106.))
        self.assertIsNone(budget.acquire_imagine('old-revision'))
        budget.set_refill_plan(advice(revision=3, at=106., versions=(29, 5)))
        self.assertFalse(budget.refill_status()['active'])
        budget.set_replay_pressure(demand(observed=106.))
        self.assertIsNotNone(budget.acquire_imagine('reactive'))

    def test_four_active_leases_are_runnable_without_a_free_seed_slot(self):
        now = [100.]
        budget = producer(now)
        budget.set_refill_plan(advice())
        leases = [budget.acquire_imagine(str(i)) for i in range(4)]
        self.assertTrue(all(leases))
        state = budget.production_admission()
        self.assertFalse(state['can_start_new_seed'])
        self.assertTrue(state['has_inflight_work'])
        self.assertTrue(state['imagination_ready'])
        self.assertEqual(state['blocked_reason'], 'imagine_in_flight')
        self.assertEqual(state['reserved_policy_samples'], 16)
        now[0] = 107.
        self.assertTrue(budget.production_admission()['imagination_ready'])

    def test_renewed_snapshot_deduplicates_reserved_and_acknowledged_output(self):
        now = [100.]
        budget = producer(now)
        budget.set_refill_plan(advice(chunks=8))
        first = budget.acquire_imagine('a')
        second = budget.acquire_imagine('b')
        self.assertIsNone(budget.acquire_imagine('c'))
        budget.finish_imagine(first['token'], 4)
        budget.set_refill_plan(advice(revision=2, chunks=4, baseline=5612))
        self.assertIsNone(budget.acquire_imagine('c'))
        budget.cancel_imagine(second['token'])
        self.assertIsNotNone(budget.acquire_imagine('c'))

    def test_new_version_revokes_plan_without_revoking_active_lease(self):
        now = [100.]
        budget = producer(now)
        budget.set_refill_plan(advice())
        lease = budget.acquire_imagine('active')
        budget.set_production_versions(31, 6)
        self.assertFalse(budget.refill_status()['active'])
        self.assertIsNone(budget.acquire_imagine('late'))
        budget.finish_imagine(lease['token'], 0)
        self.assertEqual(budget.stats()['available_policy_credits'], 1728.)

    def test_unknown_or_stale_advice_never_mints_credits(self):
        now = [100.]
        budget = producer(now)
        before = budget.stats()['available_policy_credits']
        for row in (advice(chunks=433), advice(chunks=-1), advice(at=float('nan'))):
            with self.subTest(row=row), self.assertRaises(ValueError):
                budget.set_refill_plan(row)
        budget.set_refill_plan(advice(at=90.))
        self.assertFalse(budget.refill_status()['active'])
        self.assertEqual(budget.stats()['available_policy_credits'], before)

    def test_seed_reserve_yields_under_mandatory_pressure(self):
        from pipeline_budget import PipelineBudget, PipelineBudgetConfig
        from fresh_config import FreshConfig
        config = replace(PipelineBudgetConfig(), seed_reserve_per_bucket=1)
        budget = PipelineBudget(config, 4, fresh=FreshConfig(seed_balancing=True))
        for name, success in [('s', True), ('f', False)]:
            lease = budget.acquire_real('real')
            budget.complete_real(lease['token'], name, 64, 4, 1000, success=success)
        before = budget.seed_scheduler.stats({})
        self.assertIsNone(budget.drop_oldest_pending())
        self.assertEqual(budget.seed_scheduler.stats({}), before)
        self.assertEqual(budget.drop_oldest_pending(force=True), 's')
        self.assertEqual(budget.stats()['seed_reserve_forced_releases'], 1)

    def test_production_lead_is_whole_cohort_wall_time_not_one_seed_latency(self):
        from inventory_refill import ProductionTiming
        timing = ProductionTiming(4)
        for i in range(108):
            # Four workers, 27 waves of ten-second seeds, ACK at end of wave.
            started = 100.+(i//4)*10
            timing.completed(started, started+10., 4)
        self.assertEqual(timing.status()['cohort_lead_seconds'], 270.)
        self.assertEqual(timing.status()['lead_method'], 'completed_cohort_wall')

    def test_whole_cohort_does_not_count_seed_wait_twice(self):
        from inventory_refill import ProductionTiming
        timing = ProductionTiming(4)
        timing.seed_wait = 20.
        # First start includes 20 seconds waiting for the seed; later gaps are
        # already in first-demand-to-last-ACK wall time.
        for i in range(108):
            started = 80. if i == 0 else 100.+(i//4)*10
            timing.completed(started, 110.+(i//4)*10, 4)
        self.assertEqual(timing.status()['cohort_lead_seconds'], 290.)


class FeedbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_phase_inventory_waits_for_same_versions_refill_receipt(self):
        from test_fresh_replay_pressure_cpu import actor
        from test_pipeline_integration_cpu import AsyncActor
        now = [time.monotonic()]
        budget = producer(now)
        budget._clock = time.monotonic
        owner, _ = actor(budget)
        owner.configure_fresh(owner.fresh_critics, AsyncActor(budget), inventory_refill=True)
        inventory = await owner.scheduler_inventory_with_refill()
        self.assertEqual(inventory['inventory_versions'], [30, 6])
        self.assertTrue(budget.refill_status()['active'])
        self.assertEqual(budget.refill_status()['target_new_chunks'], 432)

    async def test_delayed_receipt_cannot_extend_seed_authorization(self):
        from cost_scheduler_runtime import CostAdmissionBroker
        from test_cost_scheduler_runtime_cpu import Clock, settings, cycle, SOURCE
        clock = Clock()
        budget = producer([100.])
        budget._clock = clock
        clock.advance(5.)
        row = budget.set_refill_plan(advice(at=100.))
        row['real_production_ready'] = False
        broker = CostAdmissionBroker(settings(), 'supply', SOURCE, clock=clock)
        await broker.update_cycle(cycle(policy=30, world=6, production_state_required=True))
        await broker.update_production(row)
        clock.advance(2.)
        await broker.update_cycle(cycle(policy=30, world=6, production_state_required=True))
        self.assertFalse(budget.production_admission()['can_start_new_seed'])
        self.assertNotIn('world_infer', broker.last_decision['ready_services'])

    async def test_readiness_sees_legal_reclamation_without_mutating_replay(self):
        from test_pipeline_integration_cpu import actor, ingest
        owner = actor(frame_budget=12, capacity=2)
        ingest(owner, 'train-0')
        ingest(owner, 'train-1')
        before = ([x.sample_id for x in owner.buffer], owner.pipeline.pinned_episode_ids())
        self.assertTrue(owner.production_admission()['real_production_ready'])
        self.assertEqual(before, ([x.sample_id for x in owner.buffer], owner.pipeline.pinned_episode_ids()))
        self.assertIsNotNone(owner.acquire_real('next'))

    async def test_readiness_does_not_reclaim_active_world_ticket_or_minimum(self):
        from test_pipeline_integration_cpu import actor, ingest
        owner = actor(frame_budget=12, capacity=2)
        ingest(owner, 'train-0')
        ticket = owner.reserve_windows(2)['ticket_id']
        ingest(owner, 'train-1')
        self.assertFalse(owner.production_admission()['real_production_ready'])
        self.assertIsNone(owner.acquire_real('next'))
        owner.release_windows(ticket)
        self.assertTrue(owner.production_admission()['real_production_ready'])

    async def test_window_reclamation_handles_zero_minimum_and_no_pending_replacement(self):
        from test_pipeline_integration_cpu import actor, ingest
        for minimum, replace_pending in ((0, True), (2, False)):
            with self.subTest(minimum=minimum, replace_pending=replace_pending):
                owner = actor(frame_budget=12, capacity=8,
                    minimum_train_windows=minimum, replace_old_pending=replace_pending)
                ingest(owner, 'train-0')
                ingest(owner, 'train-1')
                for _ in range(4):
                    lease = owner.acquire_imagine('i')
                    owner.finish_imagine(lease['token'], 0)
                self.assertTrue(owner.production_admission()['real_production_ready'])
                self.assertIsNotNone(owner.acquire_real('next'))

    async def test_actual_replay_actor_arms_after_fill_and_preserves_sampling(self):
        from test_fresh_replay_pressure_cpu import actor
        from test_pipeline_integration_cpu import AsyncActor
        now = [time.monotonic()]
        budget = producer(now)
        budget._clock = time.monotonic
        owner, _ = actor(budget)
        owner.configure_fresh(owner.fresh_critics, AsyncActor(budget), inventory_refill=True)
        self.assertIsNone(owner.inventory_refill.task)
        owner.fresh_store.set_versions(30, 6)
        await owner.start_inventory_refill()
        try:
            async def accepted():
                while not budget.refill_status()['active']:
                    await asyncio.sleep(.001)
            await asyncio.wait_for(accepted(), 2.)
            self.assertTrue(budget.refill_status()['active'])
            self.assertEqual(budget.refill_status()['target_new_chunks'], 432)
            self.assertIsNotNone(budget.acquire_imagine('a'))
        finally:
            await owner.stop_inventory_refill()
        self.assertTrue(owner.inventory_refill.task.done())
        self.assertEqual(budget.refill_status()['remaining_new_chunks'], 0)
        self.assertFalse(owner.fresh_store.reservations)

    async def test_timeout_keeps_one_owned_rpc_and_visible_degradation(self):
        from inventory_refill import InventoryRefillPump
        from test_fresh_policy_replay_cpu import Writer
        from unittest.mock import patch
        import fresh_replay_pressure
        entered, release = asyncio.Event(), asyncio.Event()
        store = replay(world=5)
        async def slow():
            entered.set()
            await release.wait()
            return {'cohort_lead_seconds':400.}
        owner = SimpleNamespace(fresh_store=store, fresh_admitted_chunks={'imagined':432},
            telemetry=Writer(), fresh_production_budget=SimpleNamespace(
                production_admission=SimpleNamespace(remote=slow)))
        pump = InventoryRefillPump(owner)
        with patch.object(fresh_replay_pressure, 'RPC_TIMEOUT_SECONDS', .01):
            with self.assertRaises(RuntimeError):
                await pump.tick()
            self.assertTrue(entered.is_set())
            self.assertEqual(len(owner._fresh_feedback_pending), 1)
            self.assertEqual(await pump.tick(), 'transport_pending')
            release.set()
            await asyncio.gather(*owner._fresh_feedback_pending)

    async def test_macro_uses_authoritative_admission_and_fences_stale_state(self):
        from cost_scheduler_runtime import CostAdmissionBroker
        from test_cost_scheduler_runtime_cpu import Clock, settings, cycle, SOURCE
        clock = Clock()
        broker = CostAdmissionBroker(settings(), 'supply', SOURCE, clock=clock)
        await broker.update_cycle(cycle(policy=30, world=6, production_state_required=True))
        self.assertNotIn('world_infer', broker.last_decision['ready_services'])
        row = dict(versions=[30, 6], revision=1, observed_monotonic=100., expires_at=106.,
            can_start_new_seed=False, has_inflight_work=True, imagination_ready=True,
            real_production_ready=False, refill={'remaining_new_chunks':0})
        self.assertTrue(await broker.update_production(row))
        self.assertIn('world_infer', broker.last_decision['ready_services'])
        self.assertFalse(await broker.update_production(dict(row, revision=0, imagination_ready=False)))
        clock.advance(7.)
        await broker.update_cycle(cycle(policy=30, world=6, production_state_required=True))
        self.assertNotIn('world_infer', broker.last_decision['ready_services'])
        await broker.update_cycle(cycle(policy=31, world=6, production_state_required=True))
        self.assertFalse(await broker.update_production(dict(row, revision=2, observed_monotonic=107., expires_at=113.)))

    async def test_context_reservation_and_full_publication_record_work_once(self):
        from test_fresh_replay_pressure_cpu import actor
        from test_pipeline_integration_cpu import AsyncActor
        now = [time.monotonic()]
        budget = producer(now)
        owner, _ = actor(budget)
        owner.configure_fresh(owner.fresh_critics, AsyncActor(budget), inventory_refill=True)
        owner.fresh_store = replay(current=(29, 5), world=5)
        owner.inventory_refill.timing.reserved((29, 5), now[0]-200)
        owner.set_fresh_versions(30, 5)
        self.assertIsNone(owner.inventory_refill.timing.estimate('joint'))
        owner.set_fresh_versions(30, 6)
        self.assertGreaterEqual(owner.inventory_refill.timing.estimate('joint'), 200.)

    async def test_actual_frame_pressure_can_release_soft_reserve_and_collect(self):
        from test_pipeline_integration_cpu import actor, payload
        from fresh_config import FreshConfig
        owner = actor(frame_budget=12, capacity=2, seed_reserve_per_bucket=1)
        owner.fresh = FreshConfig(seed_balancing=True)
        from fresh_seed_scheduler import FreshSeedScheduler
        owner.pipeline.seed_scheduler = FreshSeedScheduler()
        for name, success in [('train-0', True), ('train-1', False), ('train-2', False)]:
            lease = owner.acquire_real('real')
            self.assertIsNotNone(lease)
            row = payload(name)
            row['success'][:] = False
            row['success'][-1] = success
            owner.finish_real_episode(lease['token'], row)
        self.assertLessEqual(owner._resident_frame_bytes(), owner.pipeline_config.max_resident_frame_bytes)
        self.assertGreaterEqual(len(owner.buffer), owner.minimum_train_windows)
        self.assertGreater(owner.pipeline.stats()['seed_reserve_forced_releases'], 0)


if __name__ == '__main__':
    unittest.main()
