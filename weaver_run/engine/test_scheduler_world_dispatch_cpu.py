"""Actual native bridge control methods with an asynchronous CPU-only broker."""
import asyncio
from types import SimpleNamespace
import unittest

from cost_scheduler_runtime import CostAdmissionBroker, load_settings
from weaver_online_bridge import make_weaver_online_inference


class Remote:
    def __init__(self,fn):self.fn=fn
    def remote(self,*args,**kwargs):return self.fn(*args,**kwargs)


class A:pass
class B:pass


class WorldDispatchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.events=[]
        settings={'weights':dict.fromkeys(('policy_train','world_train','policy_infer','world_infer'),1.),
            'class_weights':dict.fromkeys(('real','imagined','refresh','eval'),1.),'macro_enabled':False}
        self.core=CostAdmissionBroker(settings,'test','a'*64,emit=lambda *a,**k:self.events.append((a,k)))
        self.broker=SimpleNamespace(acquire=Remote(self.core.acquire),complete=Remote(self.core.complete))
        self.calls=[]
        def call(method,*args):
            self.calls.append((method,args))
            if method=='init_rollout':return {'backend':'weaver','version':0,'resident_handle':'one'}
            return {'released':True}
        self.owner=SimpleNamespace(actor_id=0,gpu_binding={'physical_gpu':4},capture_model_version=0,
            cost_broker=self.broker,worker_client=SimpleNamespace(call=call),_check_inference_health=lambda:None)
        self.cls=make_weaver_online_inference(A,B,{})

    async def test_context_initialization_is_accounted_without_generated_chunk_progress(self):
        result=await self.cls.init_rollout(self.owner,{'history':'fixture'},0)
        self.assertEqual(result['resident_handle'],'one')
        s=self.core.snapshot()['scheduler']
        self.assertEqual(s['services']['world_infer']['completed'],1)
        self.assertEqual(s['services']['world_infer']['completed_units'],1.)
        self.assertEqual(self.core.action_observations_completed,0)
        self.assertEqual(self.calls[0][0],'init_rollout')
        self.assertTrue(any('world_infer:init_context' in str(x) for x in self.events) or
            'world_infer:init_context' in str(s))

    async def test_release_bypasses_admission_even_when_new_work_is_failed(self):
        self.core.failed='a prior phase failed'
        result=self.cls.release_rollout(self.owner,{'resident_handle':'one'})
        self.assertTrue(result['released'])
        self.assertEqual(self.core.snapshot()['scheduler']['services']['world_infer']['dispatched'],0)

    async def test_failed_context_charges_spent_execution_but_not_successful_progress(self):
        def fail(*args):raise ValueError('stale context version')
        self.owner.worker_client.call=fail
        with self.assertRaisesRegex(ValueError,'stale context'):
            await self.cls.init_rollout(self.owner,{},0)
        s=self.core.snapshot()['scheduler']['services']['world_infer']
        self.assertEqual(s['failed'],1);self.assertEqual(s['completed'],0)
        self.assertEqual(s['completed_units'],0.)
        self.assertGreaterEqual(s['observed_cost'],0)


if __name__=='__main__':unittest.main()
