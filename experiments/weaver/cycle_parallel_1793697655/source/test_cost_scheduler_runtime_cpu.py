"""Real asyncio broker lifecycle tests with CPU-only fakes; no Ray or Torch."""
import asyncio
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cost_scheduler_runtime import (CostAdmissionBroker, TrainingAdmissionClient, replay_inventory,
    scheduled_policy_update, scheduled_world_begin, scheduled_world_end, _validate_native_boundary)
from cycle_service_controller import ResponseSample


SOURCE = 'a'*64


class Clock:
    def __init__(self):self.now=100.0
    def __call__(self):return self.now
    def advance(self,seconds):self.now+=seconds


def settings(**changes):
    value={'enabled':True,'weights':dict.fromkeys(
        ('policy_train','world_train','policy_infer','world_infer'),1.0),
        'class_weights':dict.fromkeys(('real','imagined','refresh','eval'),1.0),
        'card_inflight':2,'training_gangs':1,'max_pending':100,'macro_enabled':False,
        'allow_probe':False,'controller':{'min_response_samples':1,'cooldown_seconds':0}}
    value.update(changes)
    return value


def independent(wid,service='policy_infer.real',card=4,replica=None,units=1):
    return dict(work_id=wid,service=service,cost_key=service+':shape',units=units,
                replica=replica or ('model:'+str(card)),resources=('card:'+str(card),))


def phase(service='policy_train',sequence=0,source=SOURCE):
    world=service=='world_train'
    return dict(service=service,group_id=service+':test',sequence=sequence,
        policy_version=4 if world else sequence,world_version=sequence if world else 0,
        attestation=dict(backend='native_ddp' if world else 'policy_zero2',
            source_sha256=source,recipe_sha256='b'*64,world_size=4,
            micro_batch=4 if world else 8,accumulation_steps=64 if world else 18,
            global_batch=1024 if world else 576,
            proof_id='source_review_whole_optimizer_update_v1',boundary='optimizer_update'))


def cycle(policy=0,world=0,admitted=None,context='ordinary',**kwargs):
    d=dict(policy_version=policy,world_version=world,inventory=[],context=context,
           admitted_chunks=admitted or {'real':0,'imagined':0},
           critic_observations_per_future_update=100,
           policy_action_observations_per_real_chunk=1,
           policy_action_observations_per_imagined_chunk=1.25)
    d.update(kwargs)
    return d


class BrokerLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.clock=Clock();self.events=[]
        self.b=CostAdmissionBroker(settings(),'test-run',SOURCE,clock=self.clock,
            emit=lambda name,**details:self.events.append((name,details)))

    async def group(self,spec):
        tasks=[asyncio.create_task(self.b.acquire_training(rank,deepcopy(spec))) for rank in range(4)]
        return await asyncio.wait_for(asyncio.gather(*tasks),1)

    async def finish_group(self,receipt,costs=(1,2,3,4)):
        return [await self.b.complete_training(rank,receipt['work_id'],cost)
                for rank,cost in enumerate(costs)]

    async def test_independent_replicas_share_one_class_budget_without_gang(self):
        a,b=await asyncio.gather(self.b.acquire(independent('a',card=4)),
                                 self.b.acquire(independent('b',card=5)))
        self.assertNotEqual(a['work_id'],b['work_id'])
        self.assertEqual(self.b.snapshot()['phase_gates'],{})
        services=self.b.snapshot()['scheduler']['services']
        children=[s for s in services if s.startswith('policy_infer.')]
        self.assertAlmostEqual(sum(services[s]['config']['weight'] for s in children),1)
        self.assertEqual(services['policy_infer.real']['inflight'],2)
        await self.b.complete('a',2);await self.b.complete('b',3)
        account=self.b.snapshot()['scheduler']['services']['policy_infer.real']
        self.assertEqual(account['observed_cost'],5)
        self.assertEqual(account['completed_units'],2)

    async def test_training_needs_all_four_exact_rank_arrivals(self):
        spec=phase();tasks=[asyncio.create_task(self.b.acquire_training(r,deepcopy(spec))) for r in range(3)]
        await asyncio.sleep(0)
        self.assertFalse(any(task.done() for task in tasks))
        self.assertEqual(self.b.snapshot()['scheduler']['inflight'],{})
        tasks.append(asyncio.create_task(self.b.acquire_training(3,deepcopy(spec))))
        receipts=await asyncio.wait_for(asyncio.gather(*tasks),1)
        self.assertEqual(len({r['group_grant']['grant_id'] for r in receipts}),1)
        self.assertTrue(all(r['group_grant']['must_drain'] for r in receipts))
        result=await self.finish_group(receipts[0])
        self.assertEqual([x['group_complete'] for x in result],[False,False,False,True])
        self.assertEqual(result[-1]['actual_cost'],10)
        account=self.b.snapshot()['scheduler']['services']['policy_train']
        self.assertEqual(account['completed'],1)
        self.assertEqual(account['completed_units'],1)
        self.assertEqual(account['observed_cost'],10)
        gate=self.b.snapshot()['phase_gates'][spec['group_id']]
        self.assertEqual(gate['completed_group_phases'],1)
        self.assertEqual(gate['next_sequence'],1)
        with self.assertRaises((KeyError,ValueError)):
            await self.b.complete_training(3,receipts[0]['work_id'],4)

    async def test_training_lane_does_not_require_other_gpu_work_to_drain(self):
        infer=await self.b.acquire(independent('inflight',card=4))
        receipts=await self.group(phase())
        self.assertEqual(self.b.snapshot()['scheduler']['resource_inflight']['card:4'],2)
        during=await self.b.acquire(independent('during',card=5))
        self.assertIn('during',self.b.snapshot()['scheduler']['inflight'])
        self.assertFalse(self.b.snapshot()['global_gpu_drain'])
        world_tasks=[asyncio.create_task(self.b.acquire_training(r,phase('world_train'))) for r in range(4)]
        await asyncio.sleep(0);await asyncio.sleep(0)
        self.assertFalse(any(x.done() for x in world_tasks))
        await self.finish_group(receipts[0])
        world=await asyncio.wait_for(asyncio.gather(*world_tasks),1)
        self.assertIn(infer['work_id'],self.b.snapshot()['scheduler']['inflight'])
        await self.finish_group(world[0])
        await self.b.complete(infer['work_id'],1)
        await self.b.complete(during['work_id'],1)

    async def test_source_backend_and_card_mismatches_fail_before_dispatch(self):
        with self.assertRaises(ValueError):await self.b.acquire_training(0,phase(source='c'*64))
        bad=phase();bad['attestation']['backend']='native_ddp'
        with self.assertRaises(ValueError):await self.b.acquire_training(0,bad)
        with self.assertRaises(ValueError):await self.b.acquire(independent('wrong',card=0))
        with self.assertRaises(ValueError):await self.b.acquire(independent('wrong',service='policy_train'))
        self.assertEqual(self.b.snapshot()['scheduler']['inflight'],{})

    async def test_failure_blocks_new_admission_but_never_revokes_granted_work(self):
        receipts=await self.group(phase())
        infer=await self.b.acquire(independent('running',card=4))
        outcome=await self.b.fail_training(phase()['group_id'],'cpu test failure')
        self.assertFalse(outcome['submitted_work_revoked'])
        self.assertIn(infer['work_id'],self.b.snapshot()['scheduler']['inflight'])
        self.assertIn(receipts[0]['work_id'],self.b.snapshot()['scheduler']['inflight'])
        with self.assertRaises(RuntimeError):await self.b.acquire(independent('new',card=5))
        await self.finish_group(receipts[0])
        await self.b.complete(infer['work_id'],2)
        self.assertEqual(self.b.snapshot()['scheduler']['inflight'],{})
        self.assertIsNotNone(self.b.snapshot()['failed'])

    async def test_queue_delay_is_not_charged_as_execution_cost(self):
        first=await self.b.acquire(independent('first',card=4,replica='one'))
        queued=asyncio.create_task(self.b.acquire(independent('second',card=4,replica='one')))
        await asyncio.sleep(0)
        self.assertFalse(queued.done())
        self.clock.advance(1000)
        await self.b.complete(first['work_id'],1)
        second=await asyncio.wait_for(queued,1)
        await self.b.complete(second['work_id'],2)
        state=self.b.snapshot()['scheduler']['services']['policy_infer.real']
        self.assertEqual(state['observed_cost'],3)
        self.assertEqual(state['completed'],2)
        self.assertEqual(self.b.snapshot()['cost_provenance'],'execution_wall_proxy')

    async def test_dispatch_weight_is_preserved_for_feedback_after_weight_change(self):
        receipt=await self.b.acquire(independent('old',card=4))
        old_weight=receipt['dispatch_weight'];self.assertEqual(old_weight,0.25)
        self.b.scheduler.set_weights({'policy_infer.real':2})
        result=await self.b.complete('old',3)
        self.assertEqual(result['dispatch_weight'],old_weight)
        self.assertAlmostEqual(result['virtual_correction'],(3-receipt['estimated_cost'])/old_weight)

    async def test_failed_inference_charges_cost_but_not_successful_progress(self):
        await self.b.acquire(independent('bad',card=4,units=2))
        result=await self.b.complete('bad',3,completed_units=0,success=False)
        self.assertEqual(result['status'],'failed')
        state=self.b.snapshot()['scheduler']['services']['policy_infer.real']
        self.assertEqual(state['completed_units'],0)
        self.assertEqual(state['observed_cost'],3)

    async def test_refresh_progress_cannot_underflow(self):
        await self.b.refresh_demand(4);await self.b.refresh_demand(-2)
        self.assertEqual(self.b.snapshot()['refresh_remaining'],2)
        with self.assertRaises(ValueError):await self.b.refresh_demand(-3)
        with self.assertRaises(ValueError):await self.b.refresh_demand(True)

    async def test_bad_fourth_rank_fails_all_pending_peers_without_hanging(self):
        for field,value in [('sequence',1),('source','c'*64),('service','unsupported'),('group_id','different-group')]:
            with self.subTest(field=field):
                broker=CostAdmissionBroker(settings(),'test-run',SOURCE,clock=self.clock)
                pending=[asyncio.create_task(broker.acquire_training(r,phase())) for r in range(3)]
                await asyncio.sleep(0)
                self.assertFalse(any(task.done() for task in pending))
                bad=phase()
                if field=='source':bad['attestation']['source_sha256']=value
                else:bad[field]=value
                with self.assertRaises((RuntimeError,ValueError)):
                    await broker.acquire_training(3,bad)
                outcomes=await asyncio.wait_for(asyncio.gather(*pending,return_exceptions=True),1)
                self.assertTrue(all(isinstance(x,RuntimeError) for x in outcomes))
                self.assertIsNotNone(broker.snapshot()['failed'])
                self.assertEqual(broker.snapshot()['scheduler']['inflight'],{})

    async def test_invalid_next_phase_does_not_revoke_already_granted_phase(self):
        receipts=await self.group(phase())
        with self.assertRaises(RuntimeError):await self.b.acquire_training(0,phase(sequence=1))
        self.assertIn(receipts[0]['work_id'],self.b.snapshot()['scheduler']['inflight'])
        outcome=await self.finish_group(receipts[0])
        self.assertTrue(outcome[-1]['group_complete'])
        self.assertEqual(self.b.snapshot()['scheduler']['inflight'],{})

    async def test_refresh_cleanup_does_not_fabricate_completed_work(self):
        await self.b.refresh_demand(10)
        await self.b.refresh_demand(-10)
        self.assertEqual(self.b.refresh_completed,0)
        self.assertEqual(self.b.refresh_remaining,0)

    async def test_refresh_success_accounted_once_from_actual_service_completion(self):
        await self.b.refresh_demand(2)
        grant=await self.b.acquire(independent('refresh','policy_infer.refresh',units=2))
        await self.b.complete(grant['work_id'],1,completed_units=2)
        self.assertEqual(self.b.refresh_completed,2)
        await self.b.refresh_demand(-2)
        self.assertEqual(self.b.refresh_completed,2)
        self.assertEqual(self.b.refresh_remaining,0)
        await self.b.refresh_demand(1)
        failed=await self.b.acquire(independent('failed-refresh','policy_infer.refresh'))
        await self.b.complete(failed['work_id'],1,completed_units=0,success=False)
        await self.b.refresh_demand(-1)
        self.assertEqual(self.b.refresh_completed,2)

    async def test_all_rank_ready_reconsiders_phase_once_and_preserves_progress_window(self):
        await self.b.update_cycle(cycle())
        original=deepcopy(self.b.last_cycle)
        with patch.object(self.b.controller,'choose',wraps=self.b.controller.choose) as choose:
            pending=[asyncio.create_task(self.b.acquire_training(r,phase())) for r in range(3)]
            await asyncio.sleep(0)
            self.assertEqual(choose.call_count,0)
            pending.append(asyncio.create_task(self.b.acquire_training(3,phase())))
            receipts=await asyncio.wait_for(asyncio.gather(*pending),1)
            self.assertEqual(choose.call_count,1)
        self.assertEqual(self.b.last_decision['trigger'],'policy_train_ranks_ready')
        self.assertEqual(self.b.last_decision['phase'],'train')
        self.assertEqual(self.b.last_cycle,original)
        item=self.b.snapshot()['scheduler']['inflight'][receipts[0]['work_id']]['item']
        self.assertEqual(item['cost_key'],'policy_train:policy_zero2:micro8:gas18:global576')
        await self.finish_group(receipts[0])


class MacroRuntimeContracts(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.clock=Clock();self.b=CostAdmissionBroker(settings(),'run',SOURCE,clock=self.clock)

    async def test_progress_windows_use_wall_interval_not_request_latency(self):
        await self.b.update_cycle(cycle())
        self.clock.advance(20)
        await self.b.update_cycle(cycle(policy=1,admitted={'real':2,'imagined':4}))
        observations=self.b.controller.responses
        points={key[-1]:value['rate'] for key,value in observations.items()}
        self.assertAlmostEqual(points['policy_training'],1/20)
        self.assertAlmostEqual(points['real_chunks'],2/20)
        self.assertAlmostEqual(points['imagined_chunks'],4/20)

    async def test_context_transition_does_not_create_false_stable_measurement(self):
        await self.b.update_cycle(cycle(context='ordinary'))
        self.clock.advance(20)
        await self.b.update_cycle(cycle(policy=1,context='joint',admitted={'real':2,'imagined':4}))
        self.assertEqual(self.b.controller.responses,{})

    async def test_no_progress_is_feedback_instead_of_silent_omission(self):
        await self.b.update_cycle(cycle())
        self.clock.advance(20)
        await self.b.update_cycle(cycle())
        points={key[-1]:value['rate'] for key,value in self.b.controller.responses.items()}
        self.assertEqual(points['policy_training'],0)
        self.assertEqual(points['world_training'],0)
        self.assertEqual(points['imagined_chunks'],0)

    async def test_explicit_publication_barrier_blocks_ready_services(self):
        d=await self.b.update_cycle(cycle(publication_blocked=True,published=False))
        self.assertEqual(d['ready_services'],())
        self.assertFalse(d['changed'])

    async def test_refresh_replanning_only_on_phase_edges_with_current_remaining(self):
        await self.b.update_cycle(cycle())
        original=deepcopy(self.b.last_cycle)
        with patch.object(self.b.controller,'choose',wraps=self.b.controller.choose) as choose:
            await self.b.refresh_demand(10)
            self.assertEqual(choose.call_count,1)
            self.assertEqual(self.b.last_decision['current_critic_observations_remaining'],10)
            self.assertEqual(self.b.last_decision['phase'],'refresh')
            await self.b.refresh_demand(-2)
            await self.b.refresh_demand(-2)
            self.assertEqual(choose.call_count,1)
            await self.b.refresh_demand(-6)
            self.assertEqual(choose.call_count,2)
        self.assertEqual(self.b.last_decision['trigger'],'refresh_finished')
        self.assertEqual(self.b.last_decision['current_critic_observations_remaining'],0)
        self.assertEqual(self.b.last_cycle,original)
        self.assertEqual(self.b.controller.responses,{})

    async def test_publication_edges_replan_with_exact_new_synchronized_versions(self):
        await self.b.update_cycle(cycle())
        original=deepcopy(self.b.last_cycle)
        with patch.object(self.b.controller,'choose',wraps=self.b.controller.choose) as choose:
            await self.b.publication(True)
            self.assertEqual(choose.call_count,1)
            self.assertTrue(self.b.last_decision['publication_blocked'])
            self.assertEqual(self.b.last_decision['ready_services'],())
            await self.b.publication(True)
            self.assertEqual(choose.call_count,1)
            with self.assertRaises(ValueError):await self.b.publication(False,policy_version=5,world_version=0)
            self.assertTrue(self.b.publication_blocked)
            await self.b.publication(False,policy_version=1,world_version=0)
            self.assertEqual(choose.call_count,2)
        self.assertFalse(self.b.last_decision['publication_blocked'])
        self.assertEqual(self.b.latest_cycle_snapshot.policy_version,1)
        self.assertEqual(self.b.latest_cycle_snapshot.world_version,0)
        self.assertEqual(self.b.last_decision['phase'],'needs_batch')
        self.assertEqual(self.b.last_cycle,original)

    async def test_release_without_version_receipt_is_advisory(self):
        await self.b.update_cycle(cycle())
        await self.b.publication(True)
        await self.b.publication(False)
        self.assertEqual(self.b.last_decision['phase'],'publish')
        self.assertFalse(self.b.last_decision['changed'])
        self.assertEqual(self.b.latest_cycle_snapshot.policy_version,0)

    async def test_probe_waits_two_full_cycles_and_phase_calls_cannot_restart_it(self):
        broker=CostAdmissionBroker(settings(macro_enabled=True,allow_probe=True,probe_min_cycles=2),
                                   'run',SOURCE,clock=self.clock)
        base=deepcopy(broker.weights)
        await broker.update_cycle(cycle())
        decisions=[]
        for p in range(1,11):
            self.clock.advance(20)
            broker.refresh_completed+=100
            broker.action_observations_completed+=1000
            d=await broker.update_cycle(cycle(policy=p,world=p//5,
                context='joint' if (p+1)%5==0 else 'ordinary',
                completed_context='joint' if p%5==0 else 'ordinary',
                admitted={'real':p,'imagined':p*4}))
            decisions.append(d)
            if p<10:self.assertEqual(broker.weights,base)
        self.assertTrue(decisions[-1]['changed'])
        first_probe=deepcopy(broker.weights)
        for _ in range(5):
            self.clock.advance(60)
            await broker.refresh_demand(2);await broker.refresh_demand(-2)
            await broker.publication(True)
            await broker.publication(False,policy_version=10,world_version=2)
            self.assertEqual(broker.weights,first_probe)
        self.assertEqual(broker.snapshot()['probe_hold']['anchor_cycle'],2)
        for p in range(11,20):
            self.clock.advance(20)
            broker.refresh_completed+=100
            broker.action_observations_completed+=1000
            await broker.update_cycle(cycle(policy=p,world=p//5,
                context='joint' if (p+1)%5==0 else 'ordinary',
                completed_context='joint' if p%5==0 else 'ordinary',
                admitted={'real':p,'imagined':p*4}))
            self.assertEqual(broker.weights,first_probe)

    async def test_partial_first_cycle_does_not_count_as_complete_probe_baseline(self):
        broker=CostAdmissionBroker(settings(macro_enabled=True,allow_probe=True),'run',SOURCE,clock=self.clock)
        await broker.update_cycle(cycle(policy=4,context='joint'))
        self.assertEqual(broker.snapshot()['probe_hold']['anchor_cycle'],1)
        # Even if enough phase windows were recorded, P10 completes only one
        # wholly observed cycle after the first partial P4..5 interval.
        broker.stable_windows=10;broker.probe_contexts.add('joint')
        self.clock.advance(100)
        d=await broker.update_cycle(cycle(policy=10,world=2))
        self.assertFalse(d['changed'])

    async def test_disabled_macro_is_advisory_without_internal_weight_drift(self):
        base=dict.fromkeys(('policy_train','world_train','policy_infer','world_infer'),1.0)
        better=dict(base,world_infer=1.25,policy_train=0.75)
        service={'policy_training':'policy_train','world_training':'world_train',
                 'critic_observations':'policy_infer','policy_action_observations':'policy_infer',
                 'real_chunks':'policy_infer','imagined_chunks':'world_infer'}
        for weights in [base,better]:
            rates={'policy_training':1,'world_training':1,'critic_observations':10,
                   'policy_action_observations':10,'real_chunks':1,
                   'imagined_chunks':2 if weights==better else 1}
            for kind,rate in rates.items():
                for context in ('ordinary','joint'):
                    self.b.controller.observe(ResponseSample(weights,service[kind],rate,1,
                        context=context,progress_kind=kind,observed_at=self.clock()))
        out=await self.b.update_cycle(cycle(policy=4,world=0))
        self.assertEqual(out['reason'],'macro_disabled_advisory')
        self.assertFalse(out['changed'])
        self.assertEqual(out['weights'],base)
        self.assertEqual(out['exploration_candidate'],better)
        self.assertEqual(self.b.weights,base)
        self.assertEqual(self.b.controller.weights,base)
        self.assertIsNone(self.b.controller.last_change_at)
        self.assertEqual(self.b.snapshot()['scheduler']['services']['world_infer']['config']['weight'],1)

    def test_replay_inventory_distinct_uses_and_versions_preserved(self):
        segment=SimpleNamespace(source='imagined',versions=[{'policy':3,'world_bundle_version':0}]*2)
        store=SimpleNamespace(samples={'one':{'segment_id':'s','index':0,'uses':1,'reserved':1},
                                      'two':{'segment_id':'s','index':1,'uses':1,'reserved':1}},
                              segments={'s':segment})
        before=deepcopy(store.samples)
        out=replay_inventory(store,{'real':5,'imagined':7})
        self.assertEqual(len(out['inventory']),1)
        bucket=out['inventory'][0]
        self.assertEqual(bucket['distinct_chunks'],2)
        self.assertEqual(bucket['successful_uses'],1)
        self.assertEqual(bucket['reserved_uses'],1)
        self.assertEqual(bucket['generation_policy_version'],3)
        self.assertEqual(bucket['generation_world_version'],0)
        self.assertEqual(out['admitted_chunks'],{'real':5,'imagined':7})
        self.assertEqual(store.samples,before)


class PolicyBoundaryClientContracts(unittest.TestCase):
    def test_phase_enforces_reviewed_service_recipe_and_positive_step(self):
        client=TrainingAdmissionClient(None,{'run_id':'run'},SOURCE,ray_module=SimpleNamespace())
        for args in [('unknown',0,1,8,18),('policy_train',True,1,8,18),
                     ('policy_train',0,0,8,18),('policy_train',0,1,8,17),
                     ('world_train',0,4,4,64),('world_train',0,5,8,64)]:
            with self.subTest(args=args),self.assertRaises(ValueError):client.phase(*args)
        with self.assertRaises(ValueError):client.phase('policy_train',0,6,8,18,world_version=0)
        self.assertEqual(client.phase('world_train',0,5,4,64)['sequence'],0)

    def test_first_policy_update_is_sequence_zero_and_waits_only_own_stream(self):
        calls=[]
        class Method:
            def __init__(self,name):self.name=name
            def remote(self,*args):
                calls.append((self.name,args))
                return {'work_id':'policy-phase'} if self.name=='acquire' else None
        broker=SimpleNamespace(acquire_training=Method('acquire'),complete_training=Method('complete'),
                               fail_training=Method('fail'))
        ray=SimpleNamespace(get=lambda value:value)
        client=TrainingAdmissionClient(broker,dict(run_id='run'),SOURCE,ray_module=ray)
        owner=SimpleNamespace(cost_admission=client,rank=0,global_step=0,
                              train_batch_size=8,accumulation_steps=18)
        sync=[]
        fake_torch=SimpleNamespace(cuda=SimpleNamespace(current_stream=lambda:
                                  SimpleNamespace(synchronize=lambda:sync.append('own-stream'))))
        with patch.dict('sys.modules',{'torch':fake_torch}):
            with scheduled_policy_update(owner):pass
        self.assertEqual(sync,['own-stream','own-stream'])
        self.assertEqual([name for name,args in calls],['acquire','complete'])
        spec=calls[0][1][1]
        self.assertEqual(spec['sequence'],0)
        self.assertEqual(spec['policy_version'],0)
        self.assertEqual(spec['attestation']['global_batch'],576)


class NativeBoundaryRuntimeContracts(unittest.IsolatedAsyncioTestCase):
    def make(self):
        calls=[]
        boundary={'safe':True,'backend':'native_ddp','boundary':'optimizer_update','rank':0,
                  'world_size':4,'global_steps':0,'own_stream_complete':True,'global_gpu_drain':False}
        class Method:
            def __init__(self,name):self.name=name
            def remote(self,*args):
                async def operation():
                    calls.append((self.name,args))
                    return {'work_id':'world-stage'} if self.name=='acquire' else None
                return operation()
        broker=SimpleNamespace(acquire_training=Method('acquire'),complete_training=Method('complete'),
                               fail_training=Method('fail'))
        client=TrainingAdmissionClient(broker,{'run_id':'run'},SOURCE,ray_module=SimpleNamespace())
        worker=SimpleNamespace(call=lambda method:dict(boundary))
        owner=SimpleNamespace(rank=0,world_size=4,micro=4,gas=64,worker_client=worker,cost_admission=client)
        return owner,boundary,calls

    async def test_begin_and_end_require_exact_incremented_native_update(self):
        owner,boundary,calls=self.make()
        token=await scheduled_world_begin(owner,5)
        self.assertEqual(token['policy_step'],5)
        boundary['global_steps']=1
        await scheduled_world_end(owner,token)
        self.assertEqual([name for name,args in calls],['acquire','complete'])

    async def test_native_boundary_missing_mismatched_or_truthy_receipts_fail_closed(self):
        for key,value in [('safe',1),('backend','fsdp'),('boundary','microbatch'),('rank',1),
                          ('world_size',1),('global_steps',1),('own_stream_complete',1),
                          ('global_gpu_drain',True)]:
            with self.subTest(key=key):
                owner,boundary,calls=self.make();boundary[key]=value
                with self.assertRaises(RuntimeError):await scheduled_world_begin(owner,5)
                self.assertEqual([name for name,args in calls],['fail'])
        owner,boundary,calls=self.make();del boundary['global_steps']
        with self.assertRaises(RuntimeError):await scheduled_world_begin(owner,5)
        self.assertEqual([name for name,args in calls],['fail'])

    async def test_bad_end_receipt_never_fabricates_successful_group_completion(self):
        owner,boundary,calls=self.make();token=await scheduled_world_begin(owner,5)
        with self.assertRaises(RuntimeError):await scheduled_world_end(owner,token)
        self.assertEqual([name for name,args in calls],['acquire','fail'])


if __name__=='__main__':unittest.main()
