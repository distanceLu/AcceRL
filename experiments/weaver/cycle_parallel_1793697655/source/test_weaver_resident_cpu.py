"""Resident capability, memory bounds and real CPU worker-pipe integration."""
import copy
import os
from pathlib import Path
import pickle
import subprocess
import sys
import threading
import unittest
from unittest import mock
import numpy as np

from resident_cpu_fixture import FixtureOperations
from weaver_resident import ResidentRolloutOperations, ResidentStateError, retained_bytes
from weaver_subprocess import WeaverProcessClient, validate_resident_wire, send_frame, receive_frame


OWNER={'pid':123,'uid':1004,'birth':456}
ROOT=Path(__file__).resolve().parent


class ResidentRegistry(unittest.TestCase):
    def registry(self,**limits):
        return ResidentRolloutOperations(FixtureOperations(),'run',0,OWNER,**limits)
    def test_only_capability_not_latent_crosses_and_session_replaces_in_place(self):
        r=self.registry();h=r.init_rollout({'instruction':'task'})
        self.assertLess(len(pickle.dumps(h)),1024)
        self.assertNotIn('latent',h)
        size=r.bytes
        for i in range(50):
            result=r.request(h,np.zeros((8,7)),'task');h=result['next_state']
            self.assertEqual(h['_resident']['generation'],i+1)
            self.assertEqual(result['prediction_horizon'],8)
            self.assertEqual(len(r.entries),1)
            self.assertLessEqual(r.bytes,size+32)
        self.assertTrue(r.release_rollout(h)['released'])
        self.assertEqual((r.bytes,len(r.entries)),(0,0))
        self.assertFalse(r.release_rollout(h)['released'])
    def test_generation_consumed_exactly_once_and_old_capability_rejected(self):
        r=self.registry();h=r.init_rollout({});result=r.request(h,np.zeros((1,7)))
        with self.assertRaisesRegex(ResidentStateError,'generation'):
            r.request(h,np.zeros((1,7)))
        self.assertEqual(len(r.entries),1)
        self.assertTrue(r.release_rollout(result['next_state'])['released'])
    def test_forged_run_owner_actor_version_session_or_generation_rejected(self):
        for key,value in [('run_id','other'),('actor_id',1),('version',1),('session','f'*64),('generation',1),
                          ('owner',dict(OWNER,birth=457)),('owner',dict(OWNER,uid=False))]:
            r=self.registry();h=r.init_rollout({});bad=copy.deepcopy(h);bad['_resident'][key]=value
            with self.assertRaises(ResidentStateError):r.request(bad,np.zeros((1,7)))
            self.assertEqual(len(r.entries),1)
        h=copy.deepcopy(h);h['latents']=np.zeros(2)
        with self.assertRaises(ResidentStateError):r.request(h,np.zeros((1,7)))
    def test_cross_worker_same_run_actor_identity_cannot_reuse_token(self):
        a,b=self.registry(),self.registry();h=a.init_rollout({})
        with self.assertRaisesRegex(ResidentStateError,'authentication'):
            b.request(h,np.zeros((1,7)))
    def test_lru_eviction_is_explicit_and_unknown_tokens_are_not_authenticated(self):
        r=self.registry(max_entries=2);a=r.init_rollout({});b=r.init_rollout({})
        b=r.request(b,np.zeros((1,7)))['next_state'];c=r.init_rollout({})
        with self.assertRaisesRegex(ResidentStateError,'evicted'):r.request(a,np.zeros((1,7)))
        self.assertFalse(r.release_rollout(a)['released'])
        self.assertEqual(r.evictions,1)
        self.assertEqual(len(r.entries),2)
        r.release_rollout(b);r.release_rollout(c)
    def test_byte_budget_and_backing_view_count(self):
        a=np.zeros(1024,dtype=np.float32)
        self.assertEqual(retained_bytes([a[:1],a[2:3]]),a.nbytes)
        r=self.registry(max_bytes=300000);a=r.init_rollout({});r.init_rollout({})
        self.assertEqual(len(r.entries),1)
        with self.assertRaisesRegex(ResidentStateError,'evicted'):r.request(a,np.zeros((1,7)))
        with self.assertRaisesRegex(ResidentStateError,'byte budget'):self.registry(max_bytes=128).init_rollout({})
    def test_tensor_storage_count_uses_retained_storage_without_tensor_copy(self):
        class Storage:
            def nbytes(self):return 4096
            def data_ptr(self):return 7
        class Tensor:
            device='cuda:0'
            def untyped_storage(self):return Storage()
        self.assertEqual(retained_bytes([Tensor(),Tensor()]),4096)
    def test_publish_clears_all_states_and_cannot_roll_version_back(self):
        r=self.registry();h=r.init_rollout({});r.set_published_version(1)
        self.assertEqual((r.bytes,len(r.entries)),(0,0))
        with self.assertRaises(ResidentStateError):r.request(h,np.zeros((1,7)))
        n=r.init_rollout({},1);self.assertEqual(n['version'],1)
        with self.assertRaises(ResidentStateError):r.set_published_version(0)
    def test_failure_consumes_ambiguous_state_without_automatic_retry(self):
        r=self.registry();h=r.init_rollout({'instruction':'task'})
        with self.assertRaises(ValueError):r.request(h,np.zeros((1,7)),'wrong')
        with self.assertRaisesRegex(ResidentStateError,'released'):r.request(h,np.zeros((1,7)))
        self.assertEqual(r.bytes,0)
    def test_per_session_request_and_action_lifetime_are_finite(self):
        for limits in [{'max_requests':1},{'max_actions':1}]:
            r=self.registry(**limits);h=r.init_rollout({});h=r.request(h,np.zeros((1,7)))['next_state']
            with self.assertRaisesRegex(ResidentStateError,'lifetime'):r.request(h,np.zeros((1,7)))
            self.assertEqual(r.bytes,0)
    def test_oversize_chunk_cannot_enter_backend_and_growing_state_is_removed(self):
        r=self.registry();h=r.init_rollout({})
        for actions in [np.zeros((0,7)),np.zeros((9,7))]:
            with self.assertRaisesRegex(ResidentStateError,'horizon'):r.request(h,actions)
        self.assertEqual(len(r.entries),1)
        class Growing(FixtureOperations):
            def request_wire(self,*args):
                result=super().request_wire(*args)
                result['next_state']['latent']=np.zeros(200000,dtype=np.float32)
                return result
        r=ResidentRolloutOperations(Growing(),'run',0,OWNER,max_bytes=300000)
        h=r.init_rollout({})
        with self.assertRaisesRegex(ResidentStateError,'byte budget'):r.request(h,np.zeros((1,7)))
        self.assertEqual((r.bytes,len(r.entries)),(0,0))
    def test_bad_limits_and_initial_metadata_do_not_admit_objects(self):
        for limits in [{'max_entries':True},{'max_entries':129},{'max_bytes':0}]:
            with self.assertRaises(ValueError):self.registry(**limits)
        class Bad(FixtureOperations):
            def init_resident_rollout(self,*args):return {'initial_reward_score':float('nan')}
        r=ResidentRolloutOperations(Bad(),'run',0,OWNER)
        with self.assertRaises(ResidentStateError):r.init_rollout({})
        self.assertEqual(r.bytes,0)
    def test_client_preflight_rejects_cross_owner_without_closing_healthy_child(self):
        r=self.registry();h=r.init_rollout({});validate_resident_wire(h,'run',0,OWNER)
        c=WeaverProcessClient.__new__(WeaverProcessClient)
        c.lock=threading.Lock();c.closed=False;c.identity=OWNER;c.run_id='other';c.actor_id=0
        c.receipt={'resident_rollout':{'protocol':'weaver_resident_v1'}}
        with mock.patch('weaver_subprocess.same_process',return_value=True):
            with self.assertRaises(ResidentStateError):c.call('request',h,np.zeros((1,7)))
        self.assertFalse(c.closed)


class ActualCPUPrivateWorker(unittest.TestCase):
    def setUp(self):
        env=dict(os.environ,ACCERL_RUN_ID='resident_actual_cpu',WEAVER_GPU_WORKER_ID='resident_actual_cpu:0',
                 PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES='')
        self.process=subprocess.Popen([sys.executable,'-u',str(ROOT/'resident_cpu_fixture.py'),
            '--run-id','resident_actual_cpu','--rank','0','--role','inference'],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,bufsize=0)
        self.receipt=self.call('load',{'backend_options':{'optimizations':{
            'resident_rollout_state':True,'resident_max_sessions':8,'resident_max_bytes':64*1024*1024}}})
    def call(self,method,*args):
        send_frame(self.process.stdin,{'method':method,'args':args})
        result=receive_frame(self.process.stdout)
        self.assertTrue(result['ok'],result)
        return result['result']
    def tearDown(self):
        self.process.stdin.close()
        try:self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=5)
        stderr=self.process.stderr.read().decode(errors='replace')
        self.process.stdout.close();self.process.stderr.close()
        self.assertEqual(self.process.returncode,0,stderr)
    def test_real_worker_holds_array_and_only_references_cross_pipe(self):
        h=self.call('init_rollout',{'instruction':'task'},0)
        self.assertLess(len(pickle.dumps(h)),1024)
        validate_resident_wire(h,'resident_actual_cpu',0,self.receipt['worker_identity'])
        for i in range(4):h=self.call('request',h,np.zeros((8,7)),'task')['next_state']
        self.assertEqual(self.call('resident_status')['entries'],1)
        self.assertGreater(self.call('resident_status')['bytes'],250000)
        self.assertTrue(self.call('release_rollout',h)['released'])
        self.assertEqual(self.call('resident_status')['bytes'],0)
    def test_real_apply_invalidates_and_forgery_does_not_kill_worker(self):
        h=self.call('init_rollout',{},0)
        self.call('apply',{'version':1})
        reply=self.call('request',h,np.zeros((1,7)))
        self.assertEqual(reply['_weaver_recoverable_error'],'stale_rollout')
        h=self.call('init_rollout',{},1);bad=copy.deepcopy(h);bad['_resident']['session']='f'*64
        self.assertEqual(self.call('request',bad,np.zeros((1,7)))['_weaver_recoverable_error'],'stale_rollout')
        self.assertEqual(self.call('resident_status')['entries'],1)
        self.call('release_rollout',h)
    def test_legacy_default_off_keeps_raw_cpu_state_and_no_resident_receipt(self):
        env=dict(os.environ,ACCERL_RUN_ID='resident_actual_cpu',WEAVER_GPU_WORKER_ID='resident_actual_cpu:0',
                 PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES='')
        p=subprocess.Popen([sys.executable,'-u',str(ROOT/'resident_cpu_fixture.py'),
            '--run-id','resident_actual_cpu','--rank','0','--role','inference'],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,bufsize=0)
        def call(method,*args):
            send_frame(p.stdin,{'method':method,'args':args});reply=receive_frame(p.stdout)
            self.assertTrue(reply['ok'],reply);return reply['result']
        try:
            receipt=call('load',{})
            self.assertNotIn('resident_rollout',receipt)
            state=call('init_rollout',{},0)
            self.assertIn('latent',state);self.assertNotIn('_resident',state)
            self.assertFalse(call('resident_status')['resident_enabled'])
            call('apply',{'version':1})
            self.assertEqual(call('init_rollout',{},1)['version'],1)
        finally:
            p.stdin.close()
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();p.wait(timeout=5)
            err=p.stderr.read().decode(errors='replace');p.stdout.close();p.stderr.close()
            self.assertEqual(p.returncode,0,err)


if __name__=='__main__':unittest.main()
