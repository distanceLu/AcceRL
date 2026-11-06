"""Actual scheduled actor methods with CPU-only fake model/broker/telemetry."""
import ast
import asyncio
import contextlib
from collections import deque
import math
import inspect
import pickle
from pathlib import Path
from types import MethodType, SimpleNamespace
import sys
import time
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from policy_service_dispatch import PolicyServiceDispatch, dispatch_settings, request_class
from capture_hooks import (batch_begin, batch_end, stage_begin, stage_end,
    current_request_metadata, _request_wrapper, request_begin)


SETTINGS={'enabled':True,'refresh_micro':2,'action_micro':2,'max_pending':100,
          'class_weights':{'real':1,'imagined':1,'refresh':1,'eval':1}}
TREE=ast.parse((Path(__file__).parent/'train.py').read_text('utf8'))


def actual_method(name, namespace):
    cls=next(x for x in TREE.body if isinstance(x,ast.ClassDef) and x.name=='InferenceActor')
    method=next(x for x in cls.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name==name)
    exec(compile(ast.Module(body=[method],type_ignores=[]),'actual InferenceActor.'+name,'exec'),namespace)
    return namespace[name]


class FakeSpan:
    def __enter__(self):return self
    def __exit__(self,*args):return False


class FakeTelemetry:
    def __init__(self):self.events=[]
    def event(self,*args,**kwargs):self.events.append((args,kwargs))
    def span(self,*args,**kwargs):return FakeSpan()


class FakeArray:
    def __init__(self,data):self.data=data
    def __len__(self):return len(self.data)
    def __iter__(self):return iter(self.data)
    def __getitem__(self,i):
        value=self.data[i]
        return FakeArray(value) if isinstance(value,list) else value
    def astype(self,*args):return self
    @property
    def shape(self):
        d=self.data;out=[]
        while isinstance(d,list):
            out.append(len(d));d=d[0] if d else None
        return tuple(out)


class FakeTensor:
    def __init__(self,data):self.data=data
    def detach(self):return self
    def float(self):return self
    def reshape(self,*args):return self
    def cpu(self):return self
    def numpy(self):return FakeArray(self.data)
    def view(self,*args):return self
    def to(self,*args):return self
    @property
    def shape(self):return FakeArray(self.data).shape


class FakeModel:
    def __init__(self):
        self.calls=[];self.postprocess_calls=[]
        self.vla=SimpleNamespace(_unnormalize_actions=lambda a,key:a)
    def prepare_inputs_batch(self,rows):return rows
    def __call__(self,rows):
        self.calls.append([x['value'] for x in rows])
        return FakeTensor([[[[.1,.9]],[[.1,.9]]] for _ in rows]),FakeTensor([x['value']+2 for x in rows])
    def post_process(self,logits,deterministic):
        self.postprocess_calls.append(list(deterministic));n=len(deterministic)
        return None,FakeTensor([[[7],[7]] for _ in range(n)]),FakeArray([[[.5],[.5]] for _ in range(n)])


class Endpoint:
    def __init__(self,fn):self.fn=fn
    def remote(self,*args,**kwargs):return self.fn(*args,**kwargs)


class FakeBroker:
    def __init__(self):
        self.specs=[];self.completions=[];self.demand=0
        self.acquire=Endpoint(self._acquire);self.complete=Endpoint(self._complete)
        self.refresh_demand=Endpoint(self._refresh)
    async def _acquire(self,spec):
        self.specs.append(spec);await asyncio.sleep(0);return {'work_id':spec['work_id']}
    async def _complete(self,work_id,cost,**kwargs):
        self.completions.append((work_id,cost,kwargs));await asyncio.sleep(0)
    async def _refresh(self,delta):
        self.demand+=delta
        if self.demand<0:raise AssertionError('Negative remaining critic work')
        await asyncio.sleep(0)


def owner():
    np=SimpleNamespace(float32='float32',isfinite=lambda vals:SimpleNamespace(all=lambda:all(math.isfinite(x) for x in vals)))
    ns=dict(asyncio=asyncio,time=time,torch=SimpleNamespace(inference_mode=contextlib.nullcontext,float32='float32'),
        np=np,NUM_ACTIONS_CHUNK=2,ACTION_DIM=1,batch_begin=batch_begin,batch_end=batch_end,
        stage_begin=stage_begin,stage_end=stage_end,current_request_metadata=current_request_metadata,
        _trim_cuda_cache=lambda phase:None)
    o=SimpleNamespace(_policy_dispatch=PolicyServiceDispatch(SETTINGS),capture_model_version=5,
        actor_id=2,gpu_binding={'physical_gpu':'6'},cost_scheduler_broker=FakeBroker(),
        model=FakeModel(),cfg=SimpleNamespace(unnorm_key='test'),training_batch_limit=2,
        capture_role='policy_infer',capture_pending_meta=deque(),capture_served_versions={},
        telemetry=FakeTelemetry(),requests=[],promises=[],fatal=None,
        stats_actor=SimpleNamespace(add_timing_metric=SimpleNamespace(remote=lambda *a:None)))
    def healthy():
        if o.fatal:raise RuntimeError('fatal') from o.fatal
    def fail(error):
        o.fatal=error;remaining=o._policy_dispatch.fail_all(error)
        if remaining:asyncio.create_task(o.cost_scheduler_broker.refresh_demand.remote(-remaining))
    o._check_inference_health=healthy;o._fail_pending_requests=fail
    for name in ['scheduled_evaluate_values','_scheduled_policy_step','request']:
        setattr(o,name,MethodType(actual_method(name,ns),o))
    o.request=MethodType(_request_wrapper(actual_method('request',ns)),o)
    return o


class QueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_four_classes_and_bootstrap_mapping(self):
        for flow,cls in [('real_rollout','real'),('imagined_action','imagined'),
                         ('imagined_bootstrap','imagined'),('evaluation','eval')]:
            self.assertEqual(request_class({'workflow':flow}),cls)
        with self.assertRaises(ValueError):request_class({'workflow':'mystery'})

    async def test_refresh_is_bounded_ordered_and_pins_publication(self):
        q=PolicyServiceDispatch(SETTINGS);f=asyncio.get_running_loop().create_future()
        job=q.enqueue_refresh([{'value':i} for i in range(7)],5,f);job.ready=True
        with self.assertRaises(RuntimeError):q.assert_publication_allowed()
        chunks=[]
        while not f.done():
            b=q.next_batch(5);chunks.append(b.units)
            q.complete_batch(b,.1);q.finish_refresh(b,[x['value']+2 for x in b.inputs],5)
        self.assertEqual(chunks,[2,2,2,1]);self.assertEqual((await f)['values'],list(range(2,9)))
        q.assert_publication_allowed()

    async def test_wrong_version_cannot_partially_consume_refresh(self):
        q=PolicyServiceDispatch(SETTINGS);f=asyncio.get_running_loop().create_future()
        job=q.enqueue_refresh([1,2,3],5,f);job.ready=True
        with self.assertRaises(RuntimeError):q.next_batch(6)
        self.assertEqual(job.offset,0);q.fail_all(RuntimeError('cleanup'))
        with self.assertRaises(RuntimeError):await f

    async def test_class_weights_apply_to_execution_cost_not_requests(self):
        cfg={**SETTINGS,'class_weights':{'real':4,'imagined':1,'refresh':1,'eval':1}}
        q=PolicyServiceDispatch(cfg);fs=[]
        for i in range(40):
            for flow in ['real_rollout','evaluation']:
                f=asyncio.get_running_loop().create_future();fs.append(f)
                q.enqueue_action(i,False,{'request_id':flow+str(i),'workflow':flow},f)
        counts={'real':0,'eval':0}
        for _ in range(20):
            b=q.next_batch(5);counts[b.request_class]+=b.units;q.complete_batch(b,b.estimated_cost)
        self.assertGreaterEqual(counts['real'],counts['eval']*3)
        for f in fs:f.cancel()

    async def test_cancel_only_queued_action(self):
        q=PolicyServiceDispatch(SETTINGS);f=asyncio.get_running_loop().create_future()
        q.enqueue_action(1,False,{'request_id':'r','workflow':'real_rollout'},f)
        b=q.next_batch(5);self.assertFalse(q.cancel_queued_action('r'))
        self.assertTrue(q.owns_request('r'));q.complete_batch(b,.1);self.assertFalse(q.owns_request('r'))
        f.cancel()

    async def test_fail_all_releases_refresh_demand_and_all_promises(self):
        q=PolicyServiceDispatch(SETTINGS);fs=[asyncio.get_running_loop().create_future() for _ in range(2)]
        job=q.enqueue_refresh([1,2,3],5,fs[0]);job.ready=True;job.reported_remaining=3
        q.enqueue_action(1,False,{'request_id':'a','workflow':'real_rollout'},fs[1]);q.next_batch(5)
        self.assertEqual(q.fail_all(ValueError('boom')),3);self.assertFalse(q.refresh_pending)
        self.assertIsNone(q.active)
        for f in fs:
            with self.assertRaises(ValueError):await f

    async def test_setting_validation(self):
        for change in [{'refresh_micro':3},{'action_micro':0},{'max_pending':True},
                       {'class_weights':{'real':1}},{'unknown':1},{'enabled':'yes'}]:
            with self.assertRaises(ValueError):dispatch_settings({**SETTINGS,**change})


class ActualActorTests(unittest.IsolatedAsyncioTestCase):
    async def refresh_ready(self, o):
        for _ in range(30):
            if o._policy_dispatch.queues['refresh'] and o._policy_dispatch.queues['refresh'][0].ready:
                return
            await asyncio.sleep(0)
        self.fail('Refresh registration did not complete')

    async def test_actual_refresh_yields_to_actions_and_preserves_values(self):
        o=owner();task=asyncio.create_task(o.scheduled_evaluate_values([{'value':i} for i in range(9)],5))
        await self.refresh_ready(o)
        await o._scheduled_policy_step()
        self.assertEqual(o.model.postprocess_calls,[])
        self.assertFalse(task.done());self.assertEqual(o.cost_scheduler_broker.demand,7)
        request=asyncio.create_task(o.request({'value':99},deterministic=True,
            telemetry_meta={'request_id':'real1','workflow':'real_rollout','submitted_ns':1}))
        await asyncio.sleep(0)
        await o._scheduled_policy_step()
        self.assertTrue(request.done());self.assertFalse(task.done())
        envelope=await request
        self.assertEqual(envelope['request_id'],'real1');self.assertEqual(envelope['versions'],{'policy':5})
        self.assertEqual(o.model.postprocess_calls,[[True]])
        while not task.done():await o._scheduled_policy_step()
        self.assertEqual((await task)['values'],list(range(2,11)))
        self.assertEqual(o.cost_scheduler_broker.demand,0)
        self.assertTrue(all(s['units']<=2 for s in o.cost_scheduler_broker.specs))
        self.assertEqual({s['replica'] for s in o.cost_scheduler_broker.specs},{'policy:2'})
        self.assertEqual({tuple(s['resources']) for s in o.cost_scheduler_broker.specs},{('card:6',)})
        self.assertEqual(len(o.cost_scheduler_broker.specs),len(o.cost_scheduler_broker.completions))
        self.assertEqual(sum(x[1]['actual_batch'] for x in o.telemetry.events if x[0]==('fresh_critic_refresh',)),9)

    async def test_reordered_real_and_eval_keep_exact_request_envelopes(self):
        o=owner();o._policy_dispatch.weights['real']=10
        ev=asyncio.create_task(o.request({'value':3},deterministic=True,
            telemetry_meta={'request_id':'eval_first','workflow':'evaluation','submitted_ns':1}))
        real=asyncio.create_task(o.request({'value':9},deterministic=False,
            telemetry_meta={'request_id':'real_second','workflow':'real_rollout','submitted_ns':1}))
        await asyncio.sleep(0)
        await o._scheduled_policy_step()
        self.assertTrue(real.done());self.assertFalse(ev.done())
        self.assertEqual((await real)['request_id'],'real_second')
        self.assertEqual([x['request_id'] for x in o.capture_pending_meta],['eval_first'])
        await o._scheduled_policy_step();self.assertEqual((await ev)['request_id'],'eval_first')
        self.assertEqual(len(o.capture_pending_meta),0)

    async def test_cancelled_caller_does_not_preempt_selected_action(self):
        o=owner();gate=asyncio.Event();broker=o.cost_scheduler_broker
        async def held(spec):
            broker.specs.append(spec);await gate.wait();return {'work_id':spec['work_id']}
        broker.acquire=Endpoint(held)
        request=asyncio.create_task(o.request({'value':3},telemetry_meta={
            'request_id':'submitted','workflow':'real_rollout','submitted_ns':1}))
        await asyncio.sleep(0)
        step=asyncio.create_task(o._scheduled_policy_step());await asyncio.sleep(0)
        request.cancel()
        with self.assertRaises(asyncio.CancelledError):await request
        self.assertTrue(o._policy_dispatch.owns_request('submitted'))
        self.assertEqual(len(o.capture_pending_meta),1)
        gate.set();await step
        self.assertEqual(len(broker.completions),1)
        self.assertEqual(len(o.capture_pending_meta),0)
        self.assertNotIn('submitted',o.capture_served_versions)

    async def test_cancelled_refresh_caller_keeps_pin_until_safe_completion(self):
        o=owner();task=asyncio.create_task(o.scheduled_evaluate_values([{'value':i} for i in range(5)],5))
        await asyncio.sleep(0);await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):await task
        with self.assertRaises(RuntimeError):o._policy_dispatch.assert_publication_allowed()
        while o._policy_dispatch.refresh_pending:await o._scheduled_policy_step()
        o._policy_dispatch.assert_publication_allowed()
        self.assertEqual(o.cost_scheduler_broker.demand,0)
        self.assertEqual(sum(spec['units'] for spec in o.cost_scheduler_broker.specs),5)

    async def test_refresh_failure_closes_waiters_and_charges_failed_execution(self):
        o=owner();task=asyncio.create_task(o.scheduled_evaluate_values([{'value':1}],5))
        await self.refresh_ready(o)
        def broken(rows):raise ValueError('model failed')
        o.model.prepare_inputs_batch=broken
        with self.assertRaises(ValueError):await o._scheduled_policy_step()
        with self.assertRaises(ValueError):await task
        await asyncio.sleep(0)
        self.assertEqual(o.cost_scheduler_broker.demand,0)
        self.assertFalse(o.cost_scheduler_broker.completions[-1][2]['success'])

    async def test_request_context_is_task_local(self):
        class Owner:
            capture_role='policy_infer';capture_model_version=5
            def __init__(self):self.telemetry=FakeTelemetry()
        o=Owner()
        async def request(self,expected):
            await asyncio.sleep(0)
            self.capture_served_versions[current_request_metadata()['request_id']]={'policy':5}
            return current_request_metadata()['request_id'],expected
        wrapped=_request_wrapper(request)
        a,b=await asyncio.gather(wrapped(o,'a',telemetry_meta={'request_id':'a'}),
                                 wrapped(o,'b',telemetry_meta={'request_id':'b'}))
        self.assertEqual(a['_capture_payload'],('a','a'));self.assertEqual(b['_capture_payload'],('b','b'))
        self.assertIsNone(current_request_metadata())

    async def test_dynamic_ray_wrapper_does_not_capture_unpicklable_module_state(self):
        async def original(self):return None
        wrapped=_request_wrapper(original)
        # Cloudpickle captures globals of dynamic functions. Exercise the
        # actual wrapper's captured global values rather than checking for a
        # particular helper spelling. The old direct ContextVar capture fails
        # standard pickling here; importable helpers serialize by reference.
        captured=inspect.getclosurevars(wrapped).globals
        self.assertTrue(captured)
        for name,value in captured.items():
            with self.subTest(global_name=name):
                pickle.dumps(value)

    async def test_batch_metadata_mismatch_is_atomic(self):
        o=owner();a=request_begin(o,{'request_id':'a'});b=request_begin(o,{'request_id':'b'})
        before=list(o.capture_pending_meta)
        with self.assertRaises(ValueError):batch_begin(o,[1],selected_metadata=[{**b,'workflow':'forged'}])
        self.assertEqual(before,list(o.capture_pending_meta))
        span=batch_begin(o,[1],selected_metadata=[b]);batch_end(o,span)
        self.assertEqual([x['request_id'] for x in o.capture_pending_meta],['a'])

    async def test_candidate_routes_refresh_and_guards_publication(self):
        replay=next(x for x in TREE.body if isinstance(x,ast.ClassDef) and x.name=='ReplayBufferActor')
        sample=next(x for x in replay.body if isinstance(x,ast.AsyncFunctionDef) and x.name=='sample')
        self.assertIn('scheduled_evaluate_values',ast.unparse(sample))
        apply=actual_method('apply_rank_weights',{})
        self.assertTrue(inspect.iscoroutinefunction(apply))
        self.assertIn('publish_after_refresh',apply.__code__.co_names)


if __name__=='__main__':unittest.main()
