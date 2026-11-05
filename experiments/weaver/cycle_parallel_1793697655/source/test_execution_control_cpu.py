"""Failure/semantic regressions for actual broker and actor execution paths."""
import asyncio
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from cost_scheduler import WorkItem
from cost_scheduler_runtime import CostAdmissionBroker
from execution_control import execution_settings, production_snapshot, refresh_block
from policy_service_dispatch import PolicyServiceDispatch
from test_cost_scheduler_runtime_cpu import settings, independent, cycle, SOURCE
from test_policy_service_dispatch_cpu import owner, Endpoint, SETTINGS
from test_inventory_lookahead_cpu import budget, advice


EXEC = dict(parent_policy_budget=True, authoritative_production=True,
            refresh_lease_micros=4, refresh_lease_seconds=.8,
            refresh_trim='legacy', dependency_priority=False)


class ParentBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_classes_share_one_parent_account_and_preserve_progress(self):
        b = CostAdmissionBroker(settings(execution=EXEC), 'test', SOURCE)
        for i, cls in enumerate(('refresh', 'real', 'imagined', 'eval')):
            grant = await b.acquire(independent(str(i), 'policy_infer.'+cls))
            self.assertEqual(grant['dispatch_weight'], 1.)
            await b.complete(str(i), 1.)
        services = b.snapshot()['scheduler']['services']
        self.assertNotIn('policy_infer.refresh', services)
        self.assertEqual(services['policy_infer']['observed_cost'], 4.)
        self.assertEqual(b.refresh_completed, 1.)
        self.assertEqual(b.action_observations_completed, 2.)

    async def test_parent_share_does_not_depend_on_number_of_active_classes(self):
        for classes in [('refresh',), ('refresh','eval'), ('real','imagined','refresh','eval')]:
            b = CostAdmissionBroker(settings(execution=EXEC, card_inflight=1, max_pending=300), 'test', SOURCE)
            # Acquire into the occupied lane, then both parents are backlogged.
            await b.acquire(independent('held', 'world_infer'))
            tasks = []
            for i in range(120):
                tasks.append(asyncio.create_task(b.acquire(independent('p'+str(i),
                    'policy_infer.'+classes[i % len(classes)], replica='policy'))))
                tasks.append(asyncio.create_task(b.acquire(independent('w'+str(i),
                    'world_infer', replica='world'))))
            await asyncio.sleep(0)
            await b.complete('held', 1.)
            counts = {'policy_infer':0, 'world_infer':0}
            for i in range(120):
                await asyncio.sleep(0)
                wid, item = next(iter(b.active_items.items()))
                counts[item.service] += 1
                await b.complete(wid, 1.)
            self.assertLessEqual(abs(counts['policy_infer']-counts['world_infer']), 2)
            while b.active_items:
                wid = next(iter(b.active_items))
                await b.complete(wid, 1.)
            await asyncio.gather(*tasks)

    async def test_partial_block_refund_and_duplicate_settlement(self):
        b = CostAdmissionBroker(settings(execution=EXEC), 'test', SOURCE)
        await b.refresh_demand(10)
        spec = independent('block', 'policy_infer.refresh', units=8)
        spec['metadata'] = {'bounded_refresh_lease':True}
        grant = await b.acquire(spec)
        receipt = await b.settle_refresh('block', .4, 2)
        self.assertEqual(b.refresh_remaining, 8)
        self.assertEqual(b.refresh_completed, 2)
        self.assertEqual(receipt, await b.settle_refresh('block', .4, 2))
        self.assertEqual(b.refresh_remaining, 8)
        self.assertEqual(receipt['virtual_correction'], (.4-grant['estimated_cost'])/grant['dispatch_weight'])
        with self.assertRaises(ValueError): await b.settle_refresh('block', .5, 2)
        await b.refresh_demand(-8)

    async def test_partial_work_is_not_allowed_for_ordinary_inference(self):
        b = CostAdmissionBroker(settings(execution=EXEC), 'test', SOURCE)
        await b.acquire(independent('action', units=2))
        with self.assertRaises(ValueError): await b.complete('action', .1, completed_units=1)
        self.assertIn('action', b.active_items)
        await b.complete('action', .2, completed_units=2)

    async def test_authority_revision_and_version_must_match(self):
        b = CostAdmissionBroker(settings(execution=EXEC), 'test', SOURCE)
        await b.update_cycle(cycle())
        self.assertFalse(b.latest_cycle_snapshot.imagination_ready)
        row = dict(revision=2, versions=[0,0], real_production_ready=True, imagination_ready=False)
        await b.update_production(row)
        self.assertFalse(await b.update_production(dict(row,revision=1,imagination_ready=True)))
        await b.update_cycle(cycle())
        self.assertTrue(b.latest_cycle_snapshot.real_production_ready)
        self.assertFalse(b.latest_cycle_snapshot.imagination_ready)
        await b.publication(True)
        await b.publication(False,policy_version=1,world_version=0)
        self.assertFalse(b.latest_cycle_snapshot.real_production_ready)
        await b.update_cycle(cycle(policy=1))
        self.assertFalse(b.latest_cycle_snapshot.real_production_ready)

    async def test_paused_zero_production_is_not_fitted_as_service_speed(self):
        b = CostAdmissionBroker(settings(execution=EXEC), 'test', SOURCE)
        row = dict(revision=1, versions=[0,0], real_production_ready=False, imagination_ready=False)
        await b.update_production(row)
        await b.update_cycle(cycle())
        with patch.object(b.controller, 'observe', wraps=b.controller.observe) as observe:
            await b.update_cycle(cycle(policy=1))
            kinds = [x.args[0].progress_kind for x in observe.call_args_list]
        self.assertNotIn('imagined_chunks', kinds)
        self.assertNotIn('real_chunks', kinds)
        self.assertIn('policy_training', kinds)


class AuthorityTests(unittest.TestCase):
    def test_paused_gate_includes_lookahead_bypass_and_active_work(self):
        b = budget(lambda:0.)
        owner_ = SimpleNamespace(pipeline=b)
        self.assertFalse(production_snapshot(owner_)['imagination_can_start'])
        b.set_lookahead(advice())
        self.assertTrue(production_snapshot(owner_)['imagination_can_start'])
        leases = [b.acquire_imagine(str(i)) for i in range(4)]
        self.assertTrue(all(leases))
        state = production_snapshot(owner_)
        self.assertFalse(state['imagination_can_start'])
        self.assertTrue(state['imagination_ready'])
        self.assertEqual(state['imagination_block_reason'],'imagine_in_flight')


class CPUModel:
    def __init__(self): self.calls=[]
    def prepare_inputs_batch(self, inputs): return inputs
    def __call__(self, inputs):
        self.calls.append(len(inputs))
        return None, torch.tensor([x['value']+2 for x in inputs],dtype=torch.float32)


class BlockTests(unittest.IsolatedAsyncioTestCase):
    def make_owner(self):
        o=owner()
        o._policy_dispatch=PolicyServiceDispatch(dict(SETTINGS,execution=EXEC))
        o.model=CPUModel()
        o._scheduled_policy_step.__func__.__globals__['torch']=torch
        o._scheduled_policy_step.__func__.__globals__['np']=np
        async def settle(wid,cost,units):
            await o.cost_scheduler_broker._complete(wid,cost,completed_units=units,success=True)
            await o.cost_scheduler_broker._refresh(-units)
        o.cost_scheduler_broker.settle_refresh=Endpoint(settle)
        return o

    async def start(self,o,n):
        task=asyncio.create_task(o.scheduled_evaluate_values([{'value':i} for i in range(n)],5))
        for _ in range(40):
            if o._policy_dispatch.queues['refresh'] and o._policy_dispatch.queues['refresh'][0].ready:break
            await asyncio.sleep(0)
        return task

    async def test_actual_actor_preserves_order_and_micro_shape_with_fewer_rpcs(self):
        o=self.make_owner();task=await self.start(o,17)
        while not task.done():await o._scheduled_policy_step()
        self.assertEqual((await task)['values'],list(range(2,19)))
        self.assertEqual(o.model.calls,[2]*8+[1])
        self.assertEqual(len(o.cost_scheduler_broker.specs),3)
        self.assertEqual(o.cost_scheduler_broker.demand,0)
        self.assertFalse(o._policy_dispatch.refresh_pending)

    async def test_urgent_action_returns_unused_block_prefix(self):
        o=self.make_owner();task=await self.start(o,8)
        o._policy_dispatch.weights['refresh']=100
        fut=asyncio.get_running_loop().create_future()
        o._policy_dispatch.enqueue_action({'value':99},False,{'request_id':'urgent','workflow':'real_rollout'},fut)
        await o._scheduled_policy_step()
        self.assertEqual(o.model.calls,[2])
        self.assertEqual(o.cost_scheduler_broker.specs[0]['units'],8)
        self.assertEqual(o.cost_scheduler_broker.completions[0][2]['completed_units'],2)
        self.assertEqual(o.cost_scheduler_broker.demand,6)
        o._policy_dispatch.cancel_queued_action('urgent');fut.cancel()
        while not task.done():await o._scheduled_policy_step()
        self.assertEqual((await task)['values'],list(range(2,10)))

    async def test_nonfinite_block_fails_all_waiters_without_consumption(self):
        o=self.make_owner();task=await self.start(o,8)
        o.model=lambda inputs:(None,torch.tensor([float('nan')]*len(inputs)))
        o.model.prepare_inputs_batch=lambda inputs:inputs
        with self.assertRaises(ValueError):await o._scheduled_policy_step()
        with self.assertRaises(ValueError):await task
        await asyncio.sleep(0)
        self.assertEqual(o.cost_scheduler_broker.demand,0)
        self.assertFalse(o.cost_scheduler_broker.completions[-1][2]['success'])

    async def test_cancel_during_settlement_does_not_double_release_demand(self):
        o=self.make_owner();task=await self.start(o,12)
        entered,release=asyncio.Event(),asyncio.Event()
        async def settle(wid,cost,units):
            entered.set();await release.wait()
            await o.cost_scheduler_broker._complete(wid,cost,completed_units=units,success=True)
            await o.cost_scheduler_broker._refresh(-units)
        o.cost_scheduler_broker.settle_refresh=Endpoint(settle)
        step=asyncio.create_task(o._scheduled_policy_step());await entered.wait()
        step.cancel();release.set()
        with self.assertRaises(asyncio.CancelledError):await step
        with self.assertRaises(asyncio.CancelledError):await task
        await asyncio.sleep(0)
        self.assertEqual(o.cost_scheduler_broker.demand,0)
        self.assertEqual(len(o.cost_scheduler_broker.completions),1)

    async def test_dependency_priority_preserves_waiting_producer_service(self):
        q=PolicyServiceDispatch(dict(SETTINGS,execution=dict(EXEC,dependency_priority=True)))
        futures=[asyncio.get_running_loop().create_future() for _ in range(3)]
        job=q.enqueue_refresh([1]*12,5,futures[0]);job.ready=True
        q.enqueue_action(1,False,{'request_id':'eval','workflow':'evaluation'},futures[1])
        q.enqueue_action(1,False,{'request_id':'real','workflow':'real_rollout'},futures[2])
        q.queues['real'][0].queued_at-=2
        b=q.next_batch(5);self.assertEqual(b.request_class,'real');q.complete_batch(b,.1)
        self.assertEqual(q.next_batch(5).request_class,'refresh')
        q.fail_all(RuntimeError('cleanup'))
        for f in futures:
            if f.done():f.exception()
            else:f.cancel()


if __name__=='__main__': unittest.main()
