"""Review failure reproductions against real scheduling/actor method paths."""
import asyncio
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from cost_scheduler import CostModel, WeightedCostScheduler, WorkItem
from cost_scheduler_runtime import CostAdmissionBroker, load_settings, publish_after_refresh
from cycle_service_controller import (CycleServiceController, ControllerConfig,
    ResponseSample, CycleSnapshot, InventoryBucket, SERVICES)
from test_cost_scheduler_runtime_cpu import Clock, settings, independent, phase, cycle, SOURCE
from test_policy_service_dispatch_cpu import owner, Endpoint, SETTINGS
import test_scheduler_world_dispatch_cpu as world_fixture


async def until(predicate):
    for _ in range(100):
        if predicate():return
        await asyncio.sleep(.001)
    raise AssertionError('Expected asynchronous progress did not occur')


class AdmissionRegressions(unittest.IsolatedAsyncioTestCase):
    async def test_gang_makes_progress_despite_staggered_inference_backfill(self):
        b=CostAdmissionBroker(settings(),'review',SOURCE)
        active=[]
        for card in range(4,8):
            for service,prefix in [('world_infer','world'),('policy_infer.eval','policy')]:
                spec=independent(f'{prefix}:{card}:first',service,card,f'{prefix}:{card}')
                active.append((await b.acquire(spec),spec))
        group=[asyncio.create_task(b.acquire_training(r,phase())) for r in range(4)]
        await asyncio.sleep(0)
        self.assertFalse(any(t.done() for t in group))
        self.assertIsNotNone(b.snapshot()['scheduler']['resource_reservation'])
        replacements=[]
        # Free one lane on each card, immediately offering its successor.
        for index in (0,2,4,6):
            receipt,spec=active[index]
            await b.complete(receipt['work_id'],.1)
            replacements.append(asyncio.create_task(b.acquire(dict(spec,work_id=spec['work_id']+':next'))))
            await asyncio.sleep(0)
        grants=await asyncio.wait_for(asyncio.gather(*group),1)
        self.assertEqual(len({g['work_id'] for g in grants}),1)
        self.assertFalse(any(t.done() for t in replacements))
        # The four Policy executions were never revoked or globally drained.
        self.assertEqual(b.snapshot()['scheduler']['services']['policy_infer.eval']['inflight'],4)
        for r in range(4):await b.complete_training(r,grants[0]['work_id'],1)
        for receipt in await asyncio.wait_for(asyncio.gather(*replacements),1):
            await b.complete(receipt['work_id'],.1)
        for index in (1,3,5,7):await b.complete(active[index][0]['work_id'],.1)
        self.assertEqual(b.snapshot()['scheduler']['inflight'],{})

    async def test_cancelled_acquisition_releases_pending_and_racing_grants(self):
        for grant_first in (False,True):
            with self.subTest(grant_raced=grant_first):
                b=CostAdmissionBroker(settings(),'review',SOURCE)
                await b.acquire(independent('first',replica='r'))
                task=asyncio.create_task(b.acquire(independent('second',replica='r')))
                await asyncio.sleep(0)
                if grant_first:await b.complete('first',1)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):await task
                if not grant_first:await b.complete('first',1)
                self.assertEqual(b.snapshot()['scheduler']['replica_inflight'],{})
                await asyncio.wait_for(b.acquire(independent('third',replica='r')),1)
                await b.complete('third',1)
                self.assertEqual(b.snapshot()['scheduler']['services']['policy_infer.real']['completed'],2)

    async def test_cancelled_world_caller_reconciles_remote_admission(self):
        fixture=world_fixture.WorldDispatchTests();fixture.setUp()
        b,o=fixture.core,fixture.owner
        await b.acquire(independent('held','world_infer',4,'world:0'))
        task=asyncio.create_task(fixture.cls.init_rollout(o,{},0))
        await until(lambda:len(b.waiters)==2)
        task.cancel();await asyncio.sleep(0)
        await b.complete('held',.1)
        with self.assertRaises(asyncio.CancelledError):await task
        self.assertEqual(b.snapshot()['scheduler']['inflight'],{})
        self.assertEqual(fixture.calls,[])
        result=await asyncio.wait_for(fixture.cls.init_rollout(o,{},0),1)
        self.assertEqual(result['resident_handle'],'one')

    async def test_cancelled_context_reply_releases_created_resident_handle(self):
        fixture=world_fixture.WorldDispatchTests();fixture.setUp()
        entered,release=asyncio.Event(),asyncio.Event()
        async def complete(*args,**kwargs):
            answer=await fixture.core.complete(*args,**kwargs)
            entered.set();await release.wait();return answer
        fixture.broker.complete=Endpoint(complete)
        task=asyncio.create_task(fixture.cls.init_rollout(fixture.owner,{},0))
        await entered.wait();task.cancel();release.set()
        with self.assertRaises(asyncio.CancelledError):await task
        self.assertEqual([m for m,_ in fixture.calls],['init_rollout','release_rollout'])
        self.assertEqual(fixture.core.snapshot()['scheduler']['inflight'],{})

    def test_completion_refunds_survive_normal_saturation_probes(self):
        def run(probe):
            s=WeightedCostScheduler(cost_model=CostModel({'a':10,'b':1}),resource_limits={'card':1})
            for name in ('a','b'):s.register_service(name)
            for i in range(100):
                for name in ('a','b'):s.enqueue(WorkItem(f'{name}{i}',name,name,resources=('card',)))
            counts=Counter()
            for _ in range(100):
                g=s.claim_next();counts[g.service]+=1
                if probe:self.assertIsNone(s.claim_next())
                s.complete(g.work_id,1)
            return counts
        self.assertEqual(run(True),run(False))
        self.assertEqual(run(True),{'a':50,'b':50})


class ActorRegressions(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_during_refresh_registration_preserves_other_requests(self):
        o=owner();entered,release=asyncio.Event(),asyncio.Event()
        async def register(delta):
            o.cost_scheduler_broker.demand+=delta
            if delta>0:entered.set();await release.wait()
        o.cost_scheduler_broker.refresh_demand=Endpoint(register)
        request=asyncio.create_task(o.request({'value':1},telemetry_meta={
            'request_id':'other','workflow':'real_rollout'}))
        refresh=asyncio.create_task(o.scheduled_evaluate_values([{'value':1}]*3,5))
        await entered.wait();refresh.cancel();release.set()
        with self.assertRaises(asyncio.CancelledError):await refresh
        self.assertIsNone(o.fatal)
        self.assertTrue(o._policy_dispatch.refresh_pending)
        while o._policy_dispatch.pending_count:await o._scheduled_policy_step()
        self.assertEqual((await request)['request_id'],'other')
        self.assertEqual(o.cost_scheduler_broker.demand,0)
        self.assertIsNone(o.fatal)

    async def test_duplicate_rejection_does_not_corrupt_original_envelope(self):
        o=owner();meta={'request_id':'same','workflow':'real_rollout'}
        original=asyncio.create_task(o.request({'value':1},telemetry_meta=meta))
        await asyncio.sleep(0)
        with self.assertRaises(ValueError):await o.request({'value':9},telemetry_meta=meta)
        self.assertEqual(len(o.capture_pending_meta),1)
        await o._scheduled_policy_step()
        self.assertEqual((await original)['request_id'],'same')
        self.assertIsNone(o.fatal)
        self.assertEqual(len(o.capture_pending_meta),0)

    async def test_publication_drains_pin_and_new_bootstrap_uses_new_version(self):
        o=owner()
        old=asyncio.create_task(o.scheduled_evaluate_values([{'value':i} for i in range(3)],5))
        await until(lambda:o._policy_dispatch.refresh_pending and o._policy_dispatch.queues['refresh'][0].ready)
        def publish():o.capture_model_version=6;return {'version':6}
        publication=asyncio.create_task(publish_after_refresh(o,publish))
        await until(lambda:getattr(o,'_policy_publication_pending',False))
        new=asyncio.create_task(o.scheduled_evaluate_values([{'value':9}]))
        while not old.done():await o._scheduled_policy_step()
        self.assertEqual((await old)['policy_version'],5)
        self.assertEqual((await publication)['version'],6)
        while not new.done():await o._scheduled_policy_step()
        self.assertEqual((await new)['policy_version'],6)
        self.assertEqual(o.cost_scheduler_broker.demand,0)


class RealBudgetRegression(unittest.TestCase):
    def test_budget_censor_uses_actual_enabled_scheduled_critic(self):
        import test_fresh_policy_runtime_cpu as real
        original=real.actual_method
        served=owner()
        served.model.prepare_inputs_batch=lambda rows:[{'value':r['frame']} for r in rows]
        async def evaluate(rows):
            task=asyncio.create_task(served.scheduled_evaluate_values(rows))
            while not task.done():await served._scheduled_policy_step()
            return await task
        def extract(cls,name,namespace):
            fn=original(cls,name,namespace)
            if cls=='RolloutWorkerActor' and name=='get_one_episode':
                def run(worker):
                    def forbidden(*args):raise AssertionError('Legacy endpoint called')
                    worker.infer.evaluate_values=SimpleNamespace(remote=forbidden)
                    worker.infer.scheduled_evaluate_values=SimpleNamespace(remote=lambda rows:asyncio.run(evaluate(rows)))
                    return fn(worker)
                return run
            return fn
        with patch.object(real,'actual_method',extract):
            item,observed,_,_=real.RealCollectorTests().collect(steps=10,success=False,budget_cut=True)
        self.assertEqual(float(item.generation_values[-1]),12.)
        self.assertEqual(len(observed),1)
        self.assertTrue(item.truncated[-1]);self.assertFalse(item.terminated[-1])


class MacroRegressions(unittest.IsolatedAsyncioTestCase):
    async def test_actual_adaptive_profile_retains_joint_windows_before_probes(self):
        clock=Clock();events=[]
        profile=Path(__file__).parent/'examples/cost_03_adaptive.json'
        b=CostAdmissionBroker(load_settings(profile),'review',SOURCE,clock=clock,
            emit=lambda event,**kw:events.append((event,kw)))
        changes=[]
        for p in range(31):
            if p:
                clock.advance(321 if p%5==0 else 146)
                b.refresh_completed+=1353;b.action_observations_completed+=576
                weights=dict(b.weights)
                await b.publication(True)
                await b.publication(False,policy_version=p,world_version=p//5)
                self.assertEqual(b.weights,weights)
            d=await b.update_cycle(cycle(p,p//5,context='joint' if (p+1)%5==0 else 'ordinary',
                completed_context='joint' if p and p%5==0 else 'ordinary',
                admitted={'real':144*p,'imagined':432*p}))
            if d['changed']:changes.append(p)
        counts=[v['count'] for k,v in b.controller.responses.items()
            if k[1:]==('joint','world_train','world_training')]
        self.assertEqual(changes[:3],[10,20,30])
        self.assertTrue(counts and all(n>=2 for n in counts))
        self.assertTrue(any(kw['details'].get('predicted_cycle_seconds') is not None
            or kw['details'].get('baseline_cycle_seconds') is not None
            for name,kw in events if name=='cost_cycle_decision'))

    def test_shared_146_second_window_is_not_predicted_as_292(self):
        w=dict.fromkeys(SERVICES,1.)
        c=CycleServiceController(config=ControllerConfig(min_response_samples=1))
        for service,kind,units in [('policy_infer','critic_observations',1353),
                ('policy_train','policy_training',1),('world_train','world_training',1)]:
            c.observe(ResponseSample(w,service,units,146,context='joint',progress_kind=kind,observed_at=1000))
        s=CycleSnapshot(now=1000,policy_version=4,world_version=0,phase='refresh',
            context='joint',critic_observations_remaining=1353)
        self.assertEqual(c.forecast(s,w)[0]['seconds'],146.)

    async def test_current_not_enqueued_refresh_is_estimated_or_unknown(self):
        b=CostAdmissionBroker(settings(),'review',SOURCE,clock=Clock())
        await b.update_cycle(cycle(4,0,context='joint',phase='needs_batch',critic_observations_per_future_update=100))
        self.assertEqual(b.controller.demands(b.latest_cycle_snapshot)['critic_observations'],100)
        s=replace(b.latest_cycle_snapshot,critic_observations_per_future_update=None)
        self.assertIsNone(b.controller.demands(s)['critic_observations'])

    async def test_publication_updates_next_context_without_applying_weights(self):
        b=CostAdmissionBroker(settings(macro_enabled=True,allow_probe=True),'review',SOURCE,clock=Clock())
        await b.update_cycle(cycle(3,0,context='ordinary'))
        for p,expected in ((4,'joint'),(5,'ordinary')):
            weights=dict(b.weights)
            await b.publication(True);await b.publication(False,policy_version=p,world_version=p//5)
            self.assertEqual(b.latest_cycle_snapshot.context,expected)
            self.assertEqual(b.weights,weights)

    async def test_reserved_last_credit_is_removed_from_forecast_and_stale_revision_ignored(self):
        b=CostAdmissionBroker(settings(),'review',SOURCE,clock=Clock())
        inv=[dict(bucket_id=s,source=s,distinct_chunks=n,generation_policy_version=3,
            generation_world_version=0 if s=='imagined' else None,successful_uses=3,reserved_uses=0)
            for s,n in [('real',144),('imagined',432)]]
        await b.update_cycle(cycle(3,0,inventory=inv,inventory_revision=1,inventory_versions=[3,0]))
        reserved=[dict(x,reserved_uses=1) for x in inv]
        await b.update_inventory(dict(inventory=reserved,inventory_revision=2,inventory_versions=[3,0]))
        await b.refresh_demand(100)
        demand=b.controller.demands(b.latest_cycle_snapshot)
        self.assertEqual(demand['minimum_new_chunks'],{'real':144,'imagined':432})
        self.assertFalse(await b.update_inventory(dict(inventory=inv,inventory_revision=1,inventory_versions=[3,0])))
        self.assertEqual(b.controller.demands(b.latest_cycle_snapshot)['minimum_new_chunks'],{'real':144,'imagined':432})

    async def test_actual_replay_reports_reservation_commit_and_failure_cancel(self):
        from test_fresh_policy_replay_cpu import runtime_replay_class, Ledger, Writer, segment, FreshConfig
        for fail in (False,True):
            b=CostAdmissionBroker(settings(),'review',SOURCE,clock=Clock());pending=[]
            def notify(raw):
                task=asyncio.create_task(b.update_inventory(raw));pending.append(task);return task
            replay=runtime_replay_class()(8,FreshConfig(refresh_gae=True,replay_age_limits=True).to_dict())
            replay.capture_ledger=Ledger();replay.telemetry=Writer()
            async def evaluate(inputs,version):
                self.assertTrue(all(x.reserved_uses==1 for x in b.latest_inventory['inventory']))
                if fail:raise ValueError('test critic failure')
                return {'policy_version':version,'values':[0.]*len(inputs)}
            replay.configure_fresh([SimpleNamespace(scheduled_evaluate_values=Endpoint(evaluate))])
            replay.configure_cost_scheduler(SimpleNamespace(update_inventory=Endpoint(notify)),SETTINGS)
            replay.set_fresh_versions(5,1);replay.add_segment(segment())
            if fail:
                with self.assertRaises(ValueError):await replay.sample(2)
            else:
                result=await replay.sample(2)
                replay.commit_fresh_use(result[-1]['fresh_reservation_token'],result[-1]['sample_ids'])
            await asyncio.gather(*pending)
            self.assertTrue(all(x.reserved_uses==0 for x in b.latest_inventory['inventory']))
            self.assertEqual(sum(x.successful_uses*x.distinct_chunks for x in b.latest_inventory['inventory']),0 if fail else 2)


if __name__=='__main__':unittest.main()
