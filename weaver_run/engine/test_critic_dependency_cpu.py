"""Dependency pruning against the actual Replay/GAE and actor implementations."""
import asyncio
from dataclasses import replace
import math
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import critic_refresh_plan as planning
from fresh_config import FreshConfig
from fresh_policy_replay import SegmentReplay, generalized_advantages
from test_fresh_policy_replay_cpu import segment


class DependencyPlanTests(unittest.TestCase):
    def plan(self, items, selected, replicas=1, micro=2):
        self.assertTrue(hasattr(planning, 'dependency_refresh_groups'), 'Missing dependency-aware refresh planner')
        return planning.dependency_refresh_groups(items, selected, replicas, micro)

    def test_terminal_suffix_keeps_original_companion_and_skips_whole_batches(self):
        item=segment('real',source='real',count=10,terminal=True)
        plan=self.plan([item],{'real':[5,8]})[0]
        self.assertEqual(plan.positions,tuple(('real',i) for i in range(4,10)))
        self.assertEqual(plan.batch_sizes,(2,2,2))
        self.assertEqual(plan.inputs,item.observations[4:10])
        restored=plan.restore([obs['value']+2 for obs in plan.inputs])['real']
        self.assertEqual(restored.first_index,5)
        self.assertEqual(restored.values,(7.,8.,9.,10.,11.,0.))

    def test_nonterminal_bootstrap_and_original_tail_one_are_retained(self):
        item=segment('imag',count=4)
        plan=self.plan([item],{'imag':[3]})[0]
        self.assertEqual(plan.positions,(('imag',2),('imag',3),('imag',4)))
        self.assertEqual(plan.batch_sizes,(2,1))
        self.assertEqual(plan.restore([4.,5.,100.])['imag'].values,(5.,100.))

    def test_original_balanced_placement_and_cross_segment_pairs_are_preserved(self):
        items=[segment(str(i),source='real',count=k,terminal=True) for i,k in enumerate([10,2,8,4,19,3,7])]
        selected={s.segment_id:[s.count-1] for s in items}
        before=planning.balanced_refresh_groups([(s.segment_id,s.observations) for s in items],4)
        plans=self.plan(items,selected,4)
        for original,plan in zip(before,plans):
            positions=[(sid,i) for sid,obs in original for i in range(len(obs))]
            expected=[];sizes=[]
            for offset in range(0,len(positions),2):
                pair=positions[offset:offset+2]
                if any(i==selected[sid][0] for sid,i in pair):expected.extend(pair);sizes.append(len(pair))
            self.assertEqual(plan.positions,tuple(expected));self.assertEqual(plan.batch_sizes,tuple(sizes))

    def test_values_targets_masks_and_selected_order_match_full_refresh(self):
        rng=random.Random(12)
        for terminal in (False,True):
            for micro in (1,2):
                for _ in range(30):
                    n=rng.randint(1,30);item=segment('x',count=n,terminal=terminal)
                    indices=rng.sample(range(n),rng.randint(1,n))
                    ids=[item.sample_id(i) for i in indices]
                    replay=SegmentReplay(64,FreshConfig());replay.add(item);replay.set_versions(5,1)
                    full={'x':[math.sin(i) for i in range(n+1)]}
                    expected,er=replay.materialize(ids,full)
                    plan=self.plan([item],{'x':indices},micro=micro)[0]
                    sparse=plan.restore([math.sin(i) for sid,i in plan.positions])
                    actual,ar=replay.materialize(ids,sparse)
                    for j in (1,2,3,4):np.testing.assert_array_equal(actual[j],expected[j])
                    self.assertEqual(ar,er)
                    for left,right in zip(actual[0],expected[0]):np.testing.assert_array_equal(left['_fresh_action_mask'],right['_fresh_action_mask'])

    def test_bad_selection_and_response_fail_without_generation_value_fallback(self):
        item=segment('x',count=4)
        for selected in ({'x':[]},{'x':[True]},{'x':[-1]},{'x':[4]},{'x':[1,1]},{'wrong':[1]}):
            with self.assertRaises(ValueError):self.plan([item],selected)
        for replicas,micro in [(0,2),(True,2),(1,0),(1,3),(1,True)]:
            with self.assertRaises(ValueError):self.plan([item],{'x':[2]},replicas,micro)
        plan=self.plan([item],{'x':[2]})[0]
        for values in ([],[1.], [1.]* (len(plan.inputs)+1), [float('nan')]*len(plan.inputs),[True]*len(plan.inputs)):
            with self.assertRaises(ValueError):plan.restore(values)
        with self.assertRaises(ValueError):
            replace(plan,positions=(plan.positions[0],)*len(plan.positions)).restore([1.]*len(plan.inputs))
        with self.assertRaises(ValueError):
            replace(plan,positions=plan.positions[:-1]).restore([1.]*(len(plan.inputs)-1))
        sparse=plan.restore([1.]*len(plan.inputs));replay=SegmentReplay(8,FreshConfig());replay.add(item);replay.set_versions(5,1)
        with self.assertRaises(ValueError):replay.materialize([item.sample_id(0)],sparse)


class ExplicitRefreshBatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_direct_endpoint_preserves_explicit_batches(self):
        from types import MethodType
        from test_policy_service_dispatch_cpu import owner, actual_method
        o=owner();o._policy_dispatch=None
        o.evaluate_values=MethodType(actual_method('evaluate_values',o._scheduled_policy_step.__func__.__globals__),o)
        result=await o.scheduled_evaluate_values([{'value':i} for i in range(5)],5,batch_sizes=(2,1,2))
        self.assertEqual(result['values'],[2.,3.,4.,5.,6.])
        self.assertEqual(o.model.calls,[[0,1],[2],[3,4]])

    async def test_dispatcher_honors_explicit_tail_boundaries(self):
        from policy_service_dispatch import PolicyServiceDispatch
        from test_policy_service_dispatch_cpu import SETTINGS
        q=PolicyServiceDispatch(SETTINGS);f=asyncio.get_running_loop().create_future()
        self.assertIn('batch_sizes',__import__('inspect').signature(q.enqueue_refresh).parameters,'Explicit refresh boundaries are missing')
        job=q.enqueue_refresh([0,1,2,3,4],5,f,batch_sizes=(2,1,2));job.ready=True
        sizes=[]
        while not f.done():
            b=q.next_batch(5);sizes.append(b.units);q.complete_batch(b,.1);q.finish_refresh(b,list(b.inputs),5)
        self.assertEqual(sizes,[2,1,2]);self.assertEqual((await f)['values'],[0,1,2,3,4])

    async def test_actual_actor_demand_counts_only_executed_units(self):
        from test_policy_service_dispatch_cpu import owner,ActualActorTests
        o=owner()
        self.assertIn('batch_sizes',__import__('inspect').signature(o.scheduled_evaluate_values).parameters,'Actor explicit refresh boundaries are missing')
        task=asyncio.create_task(o.scheduled_evaluate_values([{'value':i} for i in range(5)],5,batch_sizes=(2,1,2)))
        await ActualActorTests().refresh_ready(o)
        while not task.done():await o._scheduled_policy_step()
        self.assertEqual((await task)['values'],[2.,3.,4.,5.,6.])
        self.assertEqual([s['units'] for s in o.cost_scheduler_broker.specs],[2,1,2])
        self.assertEqual(o.cost_scheduler_broker.demand,0)

    async def test_invalid_boundaries_do_not_enqueue_or_register_demand(self):
        from test_policy_service_dispatch_cpu import owner
        for sizes in [(2,), (True,2), (0,3), (3,), (), '3']:
            o=owner()
            with self.assertRaises(ValueError):
                await o.scheduled_evaluate_values([{'value':1}]*3,5,batch_sizes=sizes)
            self.assertFalse(o._policy_dispatch.refresh_pending)
            self.assertEqual(o.cost_scheduler_broker.demand,0)

    async def test_cancellation_during_registration_keeps_pin_and_exact_sparse_demand(self):
        from test_policy_service_dispatch_cpu import owner, Endpoint
        o=owner();gate=asyncio.Event();entered=asyncio.Event();broker=o.cost_scheduler_broker
        async def held(delta):
            broker.demand+=delta;entered.set();await gate.wait()
        broker.refresh_demand=Endpoint(held)
        task=asyncio.create_task(o.scheduled_evaluate_values([{'value':i} for i in range(5)],5,batch_sizes=(2,1,2)))
        await entered.wait();task.cancel();await asyncio.sleep(0)
        self.assertFalse(task.done())
        with self.assertRaises(RuntimeError):o._policy_dispatch.assert_publication_allowed()
        gate.set()
        with self.assertRaises(asyncio.CancelledError):await task
        while o._policy_dispatch.refresh_pending:await o._scheduled_policy_step()
        o._policy_dispatch.assert_publication_allowed()
        self.assertEqual([s['units'] for s in broker.specs],[2,1,2]);self.assertEqual(broker.demand,0)

    async def test_sparse_failure_releases_unexecuted_demand_and_failed_cost(self):
        from test_policy_service_dispatch_cpu import owner, ActualActorTests
        o=owner();task=asyncio.create_task(o.scheduled_evaluate_values([{'value':i} for i in range(5)],5,batch_sizes=(2,1,2)))
        await ActualActorTests().refresh_ready(o);await o._scheduled_policy_step()
        def broken(rows):raise ValueError('model failed')
        o.model.prepare_inputs_batch=broken
        with self.assertRaises(ValueError):await o._scheduled_policy_step()
        with self.assertRaises(ValueError):await task
        await asyncio.sleep(0)
        self.assertEqual(o.cost_scheduler_broker.demand,0)
        self.assertFalse(o._policy_dispatch.refresh_pending)
        self.assertFalse(o.cost_scheduler_broker.completions[-1][2]['success'])


class ActualPrunedReplayTests(unittest.IsolatedAsyncioTestCase):
    def replay(self, answer=None):
        from test_fresh_policy_replay_cpu import runtime_replay_class, Ledger, Writer
        replay=runtime_replay_class()(128,FreshConfig(refresh_gae=True,real_policy_replay=True,
            real_fraction=.25,signal_telemetry=True).to_dict())
        replay.capture_ledger=Ledger();replay.telemetry=Writer();calls=[]
        async def evaluate(inputs,version,*,batch_sizes=None):
            calls.append((inputs,version,batch_sizes))
            return answer if answer is not None else {'policy_version':version,'values':[o['value']+2 for o in inputs]}
        replay.configure_fresh([SimpleNamespace(scheduled_evaluate_values=SimpleNamespace(remote=evaluate))])
        replay.set_fresh_versions(5,1)
        for item in [segment('real',source='real',count=10,terminal=True),segment('imag',count=8)]:replay.add_segment(item)
        def update(snapshot):
            f=asyncio.get_running_loop().create_future();f.set_result(None);return f
        replay.configure_cost_scheduler(SimpleNamespace(update_inventory=SimpleNamespace(remote=update)),
            {'enabled':True,'refresh_micro':2,'action_micro':2,'max_pending':100,
             'class_weights':{'real':1,'imagined':1,'refresh':1,'eval':1}})
        replay.fresh_refresh_prune=True;replay.fresh_store.rng=random.Random(7)
        return replay,calls

    async def test_actual_sample_sends_original_sparse_micro_and_matches_full_targets(self):
        replay,calls=self.replay();result=await replay.sample(4);ids=result[-1]['sample_ids']
        unique=list(dict.fromkeys(replay.fresh_store.samples[sid]['segment_id'] for sid in ids))
        items=[replay.fresh_store.segments[sid] for sid in unique]
        selected={s.segment_id:[replay.fresh_store.samples[i]['index'] for i in ids if replay.fresh_store.samples[i]['segment_id']==s.segment_id] for s in items}
        expected_plan=planning.dependency_refresh_groups(items,selected,1,2)[0]
        self.assertIsNotNone(calls[0][2], 'Actual Replay did not use sparse refresh boundaries')
        self.assertEqual(tuple(calls[0][0]),expected_plan.inputs)
        self.assertEqual(calls[0][2],expected_plan.batch_sizes)
        full={s.segment_id:[o['value']+2 for o in s.observations] for s in items}
        expected,_=replay.fresh_store.materialize(ids,full)
        for j in (1,2,3,4):np.testing.assert_array_equal(result[j],expected[j])
        event=next(kw['details'] for k,kw in replay.telemetry.events if k=='fresh_critic_refresh_plan')
        self.assertTrue(event['original_micro_pairs_preserved'])
        self.assertEqual(sum(event['observations_by_replica']),len(expected_plan.inputs))
        replay.commit_fresh_use(result[-1]['fresh_reservation_token'],ids)
        self.assertFalse(replay.fresh_store.reservations)

    async def test_bad_current_response_cancels_reservation(self):
        for answer in [{'policy_version':6,'values':[1.]}, {'policy_version':5,'values':[]},
                       {'policy_version':5,'values':[float('nan')]*20}]:
            replay,_=self.replay(answer)
            with self.assertRaises((ValueError,RuntimeError)):await replay.sample(4)
            self.assertFalse(replay.fresh_store.reservations)
            self.assertTrue(all(s['reserved']==s['uses']==0 for s in replay.fresh_store.samples.values()))

    async def test_cancelled_pruned_sample_releases_reservation_without_consuming(self):
        replay,_=self.replay();entered=asyncio.Event()
        async def held(*args,**kwargs):entered.set();await asyncio.Event().wait()
        replay.fresh_critics[0].scheduled_evaluate_values.remote=held
        task=asyncio.create_task(replay.sample(4));await entered.wait();task.cancel()
        with self.assertRaises(asyncio.CancelledError):await task
        self.assertFalse(replay.fresh_store.reservations)
        self.assertTrue(all(s['reserved']==s['uses']==0 for s in replay.fresh_store.samples.values()))

    async def test_disabled_switch_keeps_full_legacy_refresh(self):
        replay,calls=self.replay();replay.fresh_refresh_prune=False
        result=await replay.sample(4)
        self.assertIsNone(calls[0][2])
        receipt=next(kw['details'] for k,kw in replay.telemetry.events if k=='fresh_critic_refresh_plan')
        self.assertTrue(receipt['all_k_plus_one_observations_preserved'])
        replay.commit_fresh_use(result[-1]['fresh_reservation_token'],result[-1]['sample_ids'])


class FeatureConfigurationTests(unittest.TestCase):
    def test_library_default_off_and_explicit_values(self):
        with patch.dict('os.environ',{},clear=True):self.assertFalse(planning.dependency_pruning_enabled())
        for value,expected in [('0',False),('1',True)]:
            with patch.dict('os.environ',{'ACCERL_CRITIC_DEPENDENCY_PRUNING':value}):
                self.assertEqual(planning.dependency_pruning_enabled(),expected)
        with patch.dict('os.environ',{'ACCERL_CRITIC_DEPENDENCY_PRUNING':'yes'}):
            with self.assertRaises(ValueError):planning.dependency_pruning_enabled()


if __name__=='__main__':unittest.main()
