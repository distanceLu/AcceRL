"""Actual asyncio broker regressions; synthetic work, no Ray/CUDA claim."""
import asyncio
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import unittest

from cost_scheduler_runtime import CostAdmissionBroker, load_settings
from cycle_feedback import CycleFeedbackConfig
from cycle_service_controller import CycleServiceController, CycleSnapshot, InventoryBucket
from test_cost_scheduler_runtime_cpu import Clock, SOURCE, settings, phase, independent


def new_settings(**kw):
    raw = settings(macro_enabled=True, allow_probe=True,
        execution={'parent_policy_budget':True,'authoritative_production':True},
        cycle_feedback={'enabled':True,'warmup_policy_updates':0,'observation_cycles':2,
                        'timeout_seconds':1000})
    raw.update(kw)
    return raw


def inventory(p, coverage=1.):
    return [dict(bucket_id='r',source='real',distinct_chunks=int(144*coverage),
                 generation_policy_version=p),
            dict(bucket_id='i',source='imagined',distinct_chunks=4*int(108*coverage),
                 generation_policy_version=p,generation_world_version=p//5,sampling_unit_chunks=4)]


class FeedbackBrokerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.clock=Clock();self.events=[]
        self.b=CostAdmissionBroker(new_settings(), 'cycle-test', SOURCE, clock=self.clock,
            emit=lambda name,**kw:self.events.append((name,kw)))

    async def step(self,p,seconds=10,coverage=2.,ready=False):
        self.clock.advance(seconds)
        await self.b.update_production(dict(revision=p,versions=[p,p//5],
            real_production_ready=ready,imagination_ready=ready))
        return await self.b.update_cycle(dict(policy_version=p,world_version=p//5,
            phase='needs_batch',context='joint' if (p+1)%5==0 else 'ordinary',
            completed_context='joint' if p and p%5==0 else 'ordinary',
            admitted_chunks={'real':p*40,'imagined':p*120},inventory=inventory(p,coverage),
            inventory_revision=p,inventory_versions=[p,p//5],critic_refresh_owner='trainer',
            policy_action_observations_per_real_chunk=1.,policy_action_observations_per_imagined_chunk=1.25))

    async def baseline(self):
        for p in range(11):result=await self.step(p)
        return result

    async def test_full_inventory_and_censored_production_still_apply_bounded_probe(self):
        result=await self.baseline()
        self.assertEqual(result['reason'],'complete_cycle_bounded_probe')
        self.assertTrue(result['changed'])
        self.assertGreater(self.b.weights['policy_train'],1.)
        self.assertEqual(self.b.weights['policy_train'],self.b.scheduler.snapshot()['services']['policy_train']['config']['weight'])
        self.assertNotIn('policy_infer.real',self.b.scheduler.snapshot()['services'])
        self.assertEqual(sum(self.b.weights.values()),4.)
        self.assertEqual(self.b.refresh_remaining,0)
        self.assertIsNone(self.b.controller._rate(self.b.latest_cycle_snapshot,
            dict.fromkeys(self.b.weights,1.),'world_infer','imagined_chunks'))
        self.assertTrue(any(name=='cost_weights_applied' for name,_ in self.events))
        self.assertIsNone(result['predicted_cycle_seconds'])

    async def test_probe_holds_two_cycles_then_accepts_measured_improvement(self):
        await self.baseline();candidate=dict(self.b.weights)
        for p in range(11,20):
            result=await self.step(p,8)
            self.assertEqual(self.b.weights,candidate)
            self.assertIsNotNone(self.b.cycle_feedback.active_probe)
        result=await self.step(20,8)
        self.assertEqual(result['reason'],'cycle_probe_accepted')
        self.assertEqual(result['probe_comparison']['candidate_seconds'],40.)
        self.assertEqual(result['probe_comparison']['baseline_seconds'],50.)
        self.assertEqual(self.b.weights,candidate)
        self.assertIsNone(self.b.cycle_feedback.active_probe)
        self.assertEqual(len(self.b.cycle_feedback.records),0)

    async def test_slower_probe_rolls_back_and_does_not_immediately_retry_same_pair(self):
        await self.baseline();first=dict(self.b.weights)
        for p in range(11,21):result=await self.step(p,12)
        self.assertEqual(result['reason'],'cycle_probe_rollback')
        self.assertEqual(self.b.weights,dict.fromkeys(self.b.weights,1.))
        for p in range(21,31):result=await self.step(p)
        self.assertTrue(result['changed']);self.assertNotEqual(self.b.weights,first)

    async def test_inventory_depletion_cannot_masquerade_as_speedup(self):
        await self.baseline()
        for p in range(11,21):result=await self.step(p,5,coverage=.2)
        self.assertFalse(result['probe_comparison']['accepted'])
        self.assertFalse(result['probe_comparison']['inventory_reserve_preserved'])

    async def test_inference_grant_keeps_dispatch_weight_after_macro_change(self):
        receipt=await self.b.acquire(independent('before','world_infer'))
        original=receipt['dispatch_weight']
        await self.baseline()
        await self.b.complete(receipt['work_id'],3.)
        completed=[v['details'] for n,v in self.events if n=='cost_work_completed'][-1]
        self.assertEqual(completed['dispatch_weight'],original)

    async def test_timeout_rolls_back_without_waiting_for_a_starved_update(self):
        await self.baseline()
        self.clock.advance(1100)
        await self.b.update_inventory(dict(inventory_revision=100,
            inventory_versions=[10,2],inventory=inventory(10,2),
            admitted_chunks={'real':400,'imagined':1200},critic_refresh_owner='trainer'))
        self.assertEqual(self.b.weights,dict.fromkeys(self.b.weights,1.))
        self.assertEqual(self.b.last_probe_result['reason'],'cycle_probe_rollback_timeout')
        self.assertFalse(self.b.last_probe_result['complete_observation_window'])

    async def test_timeout_defers_during_unpublished_training(self):
        await self.baseline();chosen=dict(self.b.weights)
        self.b.unpublished_training['policy_train']=11
        self.clock.advance(1100)
        await self.b.update_inventory(dict(inventory_revision=100,inventory_versions=[10,2],
            inventory=inventory(10,2),admitted_chunks={'real':400,'imagined':1200},critic_refresh_owner='trainer'))
        self.assertEqual(self.b.weights,chosen)
        self.assertIn('training_not_published',self.b.cycle_feedback.active_probe['timeout_blockers'])

    async def test_gap_or_duplicate_boundary_cannot_fabricate_complete_cycles(self):
        await self.step(0)
        for p in (1,2,4,5,6,7,8,9,10):await self.step(p)
        self.assertIsNone(self.b.cycle_feedback.active_probe)
        self.assertEqual(len(self.b.cycle_feedback.records),1)
        previous=deepcopy(self.b.cycle_feedback.records)
        await self.step(10)
        self.assertEqual(self.b.cycle_feedback.records,previous)

    async def test_phase_advisory_never_applies_weights(self):
        for p in range(10):await self.step(p)
        before=dict(self.b.weights)
        self.b._reconsider_phase('arbitrary',phase='train')
        self.assertEqual(self.b.weights,before)
        self.assertIsNone(self.b.cycle_feedback.active_probe)

    async def test_macro_disabled_preserves_fixed_weights(self):
        self.b.settings['macro_enabled']=False
        for p in range(21):await self.step(p)
        self.assertEqual(self.b.weights,dict.fromkeys(self.b.weights,1.))
        self.assertIsNone(self.b.cycle_feedback.active_probe)

    async def test_shortage_prioritizes_admitted_production(self):
        for p in range(10):await self.step(p)
        result=await self.step(10,coverage=.2,ready=True)
        self.assertEqual(result['reason'],'complete_cycle_bounded_probe')
        self.assertGreater(self.b.weights['world_infer'],1.)

    async def test_timeout_remainder_is_not_a_new_complete_cycle(self):
        await self.baseline();self.clock.advance(1100)
        await self.b.update_inventory(dict(inventory_revision=100,inventory_versions=[10,2],
            inventory=inventory(10,2),admitted_chunks={'real':400,'imagined':1200},critic_refresh_owner='trainer'))
        for p in range(11,16):await self.step(p)
        self.assertEqual(self.b.cycle_feedback.records,[])
        for p in range(16,21):await self.step(p)
        self.assertEqual(len(self.b.cycle_feedback.records),1)

    async def test_warmup_requires_two_additional_complete_cycles(self):
        self.b.cycle_feedback.config=CycleFeedbackConfig(enabled=True,warmup_policy_updates=20)
        for p in range(30):
            await self.step(p)
            self.assertIsNone(self.b.cycle_feedback.active_probe)
        self.assertTrue((await self.step(30))['changed'])


class ConcurrentAdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def group(self,b,spec):
        return await asyncio.wait_for(asyncio.gather(*[
            b.acquire_training(rank,deepcopy(spec)) for rank in range(4)]),1)

    async def finish(self,b,receipt):
        for rank in range(4):await b.complete_training(rank,receipt['work_id'],1.)

    async def test_two_training_groups_and_inference_make_progress_together(self):
        events=[]
        b=CostAdmissionBroker(settings(training_gangs=2,card_inflight=3), 'parallel',SOURCE,
            emit=lambda name,**kw:events.append((name,kw)))
        infer=[await b.acquire(independent('infer'+str(c),card=c)) for c in range(4,8)]
        ps=phase();ps['policy_version']=4
        policy=await self.group(b,ps)
        world=await self.group(b,phase('world_train'))
        snapshot=b.scheduler.snapshot()
        self.assertEqual(snapshot['services']['policy_train']['inflight'],1)
        self.assertEqual(snapshot['services']['world_train']['inflight'],1)
        self.assertEqual(snapshot['services']['policy_infer.real']['inflight'],4)
        pair=[v['details'] for n,v in events if n=='cost_training_execution_admitted'][-1]
        self.assertEqual(pair['coadmitted_training'],[policy[0]['work_id']])
        await self.finish(b,world[0]);await self.finish(b,policy[0])
        for r in infer:await b.complete(r['work_id'],.1)
        self.assertFalse(b.snapshot()['failed'])

    async def test_serial_ablation_still_waits_for_first_group(self):
        b=CostAdmissionBroker(settings(), 'serial',SOURCE)
        ps=phase();ps['policy_version']=4;policy=await self.group(b,ps)
        tasks=[asyncio.create_task(b.acquire_training(r,phase('world_train'))) for r in range(4)]
        await asyncio.sleep(.01);self.assertFalse(any(t.done() for t in tasks))
        await self.finish(b,policy[0])
        world=await asyncio.wait_for(asyncio.gather(*tasks),1);await self.finish(b,world[0])

    async def test_mismatched_pair_fails_new_grants_but_active_group_drains(self):
        b=CostAdmissionBroker(settings(training_gangs=2,card_inflight=3), 'parallel',SOURCE)
        policy=await self.group(b,phase())
        tasks=[b.acquire_training(r,phase('world_train')) for r in range(4)]
        result=await asyncio.wait_for(asyncio.gather(*tasks,return_exceptions=True),1)
        self.assertTrue(all(isinstance(x,Exception) for x in result))
        self.assertIsNotNone(b.failed)
        await self.finish(b,policy[0])
        self.assertEqual(b.scheduler.snapshot()['services']['policy_train']['completed'],1)

    def test_two_gangs_cannot_silently_consume_all_inference_lanes(self):
        with self.assertRaises(ValueError):
            CostAdmissionBroker(settings(training_gangs=2,card_inflight=2),'bad',SOURCE)

    async def test_publication_rejects_training_without_partial_grant(self):
        b=CostAdmissionBroker(settings(training_gangs=2,card_inflight=3),'publishing',SOURCE)
        await b.publication(True)
        results=await asyncio.gather(*[b.acquire_training(r,phase()) for r in range(4)],return_exceptions=True)
        self.assertTrue(all(isinstance(x,Exception) for x in results))
        self.assertEqual(b.scheduler.snapshot()['inflight'],{})


class ProfilesAndReadiness(unittest.TestCase):
    def test_ready_jit_training_at_needs_batch(self):
        s=CycleSnapshot(now=1,policy_version=10,world_version=2,
            inventory=tuple(InventoryBucket(**x) for x in inventory(10,2)),
            critic_observations_per_future_update=0,real_production_ready=False,imagination_ready=False)
        self.assertIn('policy_train',CycleServiceController().ready_services(s))
        self.assertNotIn('policy_train',CycleServiceController().ready_services(replace(s,inventory=())))

    def test_profiles_do_not_change_learning(self):
        root=Path(__file__).resolve().parent
        for name,gangs,feedback in [('00_parent',1,False),('00_fixed_serial',1,False),('01_feedback_serial',1,True),
                                    ('02_parallel_fixed',2,False),('03_feedback_parallel',2,True)]:
            cfg=load_settings(root/'examples'/('cycle_'+name+'.json'))
            self.assertEqual(cfg['training_gangs'],gangs)
            self.assertEqual(cfg.get('cycle_feedback',{}).get('enabled',False),feedback)
        parent=json.loads((root/'parent_trajectory_source_manifest.json').read_text())
        import hashlib
        for name in ['jit_gae_runtime.py','jit_gae_plan.py','advantage_ema.py','fresh_policy_replay.py',
                     'trajectory_sampling.py','train.py','weaver_online_engine.py',
                     'examples/trajectory_01_imagined.json','examples/pipeline_lookahead_01_two_updates.json']:
            self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(),parent['files'][name])


if __name__=='__main__':unittest.main()
