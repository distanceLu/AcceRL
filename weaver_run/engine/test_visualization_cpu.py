"""Real recorder + production rollout tests. No model/GPU/server required."""
import ast
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

import numpy as np
import visualization as vis


class Sink:
    cfg = dict(vis.DEFAULTS, max_frames=3)
    def __init__(self): self.tasks=[]; self.errors=[]
    def offer(self, task): self.tasks.append(task); return True
    def failed(self, error): self.errors.append(str(error))


def obs(value=128):
    return dict(full_image=np.full((224,224,3),value,np.uint8),
                wrist_image=np.zeros((224,224,3),np.uint8))


class RecorderTests(unittest.TestCase):
    def test_cpu_copy_and_camera_order_no_input_mutation(self):
        raw=obs(41); frame=vis.cpu_frame(raw)
        raw['full_image'][:]=0
        self.assertEqual(frame.shape,(224,448,3))
        self.assertTrue((frame[:,:224]==41).all()); self.assertTrue((frame[:,224:]==0).all())

    def test_no_hidden_cuda_transfer(self):
        class GPU:
            device=types.SimpleNamespace(type='cuda')
            def cpu(self): raise AssertionError('must not transfer')
        with self.assertRaisesRegex(ValueError,'CPU'): vis.cpu_frame(GPU(),True)

    def test_normalized_world_pixels_not_mutated(self):
        x=np.zeros((2,3,8,8),np.float32); x[0]=-1; x[1]=1
        before=x.copy(); y=vis.cpu_frame(x,True)
        self.assertTrue(np.array_equal(x,before)); self.assertEqual(int(y[:,:224].max()),0)
        self.assertEqual(int(y[:,224:].min()),255)

    def test_cap_preserves_true_endpoint_and_metadata(self):
        writer=Sink(); clip=vis.Clip(writer,'real',0,'episode',{})
        for i in range(10): clip.add(obs(i),i)
        clip.add(obs(11),11,force=True); clip.finish(success=None,budget_censored=True)
        task=writer.tasks[-1]; self.assertEqual(task[0],'clip')
        self.assertEqual(len(task[2]),3); self.assertEqual(task[3]['samples'][-1]['step'],11)
        self.assertTrue(task[3]['frame_limit_reached']); self.assertIsNone(task[3]['success'])
        self.assertEqual(int(task[2][-1][0,0,0]),11)

    def test_recorder_does_not_consume_python_or_numpy_rng(self):
        p=random.getstate(); n=np.random.get_state()
        clip=vis.Clip(Sink(),'imagined',0,'seed',{})
        clip.add(np.zeros((2,3,4,4)),0,normalized=True); clip.finish(discarded=False)
        self.assertEqual(random.getstate(),p)
        now=np.random.get_state(); self.assertEqual(now[0],n[0]); self.assertTrue(np.array_equal(now[1],n[1])); self.assertEqual(now[2:],n[2:])

    def test_preview_errors_do_not_escape_to_training(self):
        s=Sink(); c=vis.Clip(s,'real',0,'ep',{}); c.add({'invalid':1},0); c.finish()
        self.assertTrue(c.closed); self.assertTrue(s.errors); self.assertEqual(s.tasks,[])

    def test_disabled_has_no_writer_or_frame_work(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ,ACCERL_EXPERIMENT_ROOT=d), \
             patch.object(vis,'_CONFIG',dict(vis.DEFAULTS,enabled=False)), patch.object(vis,'Writer',side_effect=AssertionError):
            self.assertIs(vis.begin('real',0,'ep'),vis.NULL)

    def test_queue_full_returns_immediately(self):
        import queue
        w=object.__new__(vis.Writer); w.jobs=queue.Queue(maxsize=1); w.dropped=0
        w.jobs.put(1); start=time.monotonic(); self.assertFalse(w.offer(('clip',)))
        self.assertLess(time.monotonic()-start,.1); self.assertEqual(w.dropped,1)

    def test_config_validated(self):
        with tempfile.TemporaryDirectory() as d:
            for value in ({'max_frames':0},{'enabled':1},{'invented':1}):
                Path(d,'visualization_config.json').write_text(json.dumps(value))
                with self.assertRaises(ValueError): vis.config(d)

    def test_disk_ring_prunes_only_owned_preview_files(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); w=object.__new__(vis.Writer); w.cfg=dict(vis.DEFAULTS,clips_per_producer=2)
            for i in range(4):
                (p/f'{i}.json').write_text('{}'); (p/f'{i}.mp4').write_bytes(b'video')
            (p/'latest.jpg').write_bytes(b'image'); (p/'unrelated.txt').write_text('keep')
            w._prune(p)
            self.assertEqual(sorted(f.name for f in p.glob('[0-9]*.json')),['2.json','3.json'])
            self.assertTrue((p/'latest.jpg').exists()); self.assertTrue((p/'unrelated.txt').exists())

    def test_cpu_module_does_not_import_torch_or_tensorflow(self):
        code='import sys;sys.path.insert(0,sys.argv[1]);import visualization;assert "torch" not in sys.modules and "tensorflow" not in sys.modules'
        subprocess.run([sys.executable,'-I','-B','-c',code,str(Path(vis.__file__).parent)],check=True,timeout=20)

    def test_production_imagination_targets_and_rpc_sequence_unchanged_with_recording(self):
        from test_integrated_semantics_cpu import ProductionImagine
        harness=ProductionImagine(); s=Sink(); clips=[]
        with patch.object(vis,'begin',return_value=vis.NULL): a,ba=harness.run_seed()
        def capture(role,worker,episode,**metadata):
            c=vis.Clip(s,role,worker,episode,metadata); clips.append(c); return c
        with patch.object(vis,'begin',side_effect=capture): b,bb=harness.run_seed()
        self.assertEqual(a,b)  # No extra inference, reward or bootstrap call.
        self.assertEqual([x.value_target for x in ba],[x.value_target for x in bb])
        self.assertEqual([x.advantage for x in ba],[x.advantage for x in bb])
        self.assertTrue(clips[0].closed)
        meta=s.tasks[-1][3]; self.assertFalse(meta['reward_is_real_success'])
        self.assertFalse(meta['discarded']); self.assertEqual(meta['samples'][-1]['step'],4)
        self.assertNotIn('success',meta)

    def test_real_reset_metadata_uses_actual_counter_not_worker_id(self):
        env=types.SimpleNamespace(initial_states=list(range(50)),episode_idx=68)
        self.assertEqual(vis.env_metadata(env,{})['initial_state_id'],17)
        self.assertEqual(vis.env_metadata(env,{'initial_state_id':2})['initial_state_id'],2)
        self.assertNotIn('initial_state_id',vis.env_metadata(object(),{}))

    def test_encoder_and_catalog_real_mp4(self):
        # ffmpeg is a shipped control requirement. No fake movie fallback.
        with tempfile.TemporaryDirectory() as d:
            w=vis.Writer(d,vis.DEFAULTS); c=vis.Clip(w,'evaluation',0,'cpu-episode',dict(success=True))
            c.add(obs(40),0); c.add(obs(80),2); c.finish(action_steps=2)
            self.assertTrue(w.flush(30)); self.assertEqual(w.errors,0,w.error)
            cat=vis.catalog(d); self.assertEqual(len(cat['clips']),1)
            row=cat['clips'][0]; self.assertEqual(row['encoded_frames'],2)
            movie=Path(d,'visualization',row['producer'],row['video'])
            self.assertIn(b'ftyp',movie.read_bytes()[:32])

    def test_actual_online_evaluation_emits_observed_versions_and_result(self):
        import contextlib, io
        tree=ast.parse(Path(vis.__file__).with_name('train.py').read_text(encoding='utf8'))
        cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='EvaluationWorkerActor')
        run=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='run')
        class Done(Exception): pass
        class Env:
            initial_states=range(50); episode_idx=19; steps=0
            def step(self,action):
                self.steps+=1
                return obs(),1.0,self.steps==2,False,{'is_success':self.steps==2}
        worker=types.SimpleNamespace(wid=1,env=Env(),task_description='fixture',current_env_name='fixture',
            cfg=None,processor=None,torch_dtype=None,infer=None,capture_last_versions={'policy':9})
        calls=[]; resets=[]; sink=Sink()
        def reset(seed):
            if resets: raise Done()
            resets.append(seed); return obs(),{}
        worker._reset_and_select_env=reset
        worker.stats_actor=types.SimpleNamespace(add_episode_return=types.SimpleNamespace(remote=lambda *a,**k:calls.append((a,k))))
        ns=dict(os=os,time=time,random=random,now_ns=time.time_ns,prepare_one_obs=lambda *a:None,
                profile_rpc=lambda *a,**k:(None,[0,0],None,None,None))
        exec(compile(ast.Module(body=[run],type_ignores=[]),'actual evaluation run','exec'),ns)
        with patch.object(vis,'begin',side_effect=lambda role,worker,episode,**m:vis.Clip(sink,role,worker,episode,m)), \
             patch.dict(os.environ,ACCERL_SEED='7'),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(Done): ns['run'](worker)
        self.assertEqual(len(calls),1); meta=sink.tasks[-1][3]
        self.assertTrue(meta['success']); self.assertEqual(meta['policy_versions'],[9])
        self.assertEqual(meta['initial_state_id'],18); self.assertEqual(meta['samples'][-1]['step'],2)

    def test_actual_frozen_evaluation_preserves_result_and_action_seed_calls(self):
        import contextlib
        import frozen_policy_eval as frozen
        class Tokens:
            def detach(self): return self
            def cpu(self): return self
            def contiguous(self): return self
            def numpy(self): return np.array([1,2],np.int32)
        class Model:
            processor=None
            vla=types.SimpleNamespace(_unnormalize_actions=lambda x,k:np.zeros((2,7),np.float32))
            def prepare_inputs_batch(self,x): return x
            def __call__(self,x): return None,None
            def post_process(self,x,deterministic): return None,Tokens(),[None]
        class Env:
            max_episode_steps=220; num_steps_wait=10; task_description='fixture'; count=0
            def step(self,action):
                self.count+=1; return obs(),0.0,False,self.count==2,{'is_success':False}
        case=dict(initial_state_id=18,env_seed=1,action_seed=7,deterministic=True)
        ckpt=dict(name='baseline',sha256='abc',metadata={'policy_step':0}); sink=Sink(); seeds=[]
        fake_torch=types.SimpleNamespace(manual_seed=lambda x:seeds.append(('cpu',x)),
            cuda=types.SimpleNamespace(manual_seed_all=lambda x:seeds.append(('cuda',x))),inference_mode=contextlib.nullcontext)
        fake_utils=types.ModuleType('rl.utils');fake_utils.prepare_one_obs=lambda *a:None
        results=[]
        for enabled in (False,True):
            factory=(lambda role,w,ep,**m:vis.Clip(sink,role,w,ep,m)) if enabled else (lambda *a,**k:vis.NULL)
            with patch.dict(sys.modules,{'torch':fake_torch,'rl.utils':fake_utils}),patch.object(vis,'begin',side_effect=factory), \
                 patch('initial_state_env.reset_with_initial_state',return_value=(obs(),{'initial_state_hash':'h','reset_env_steps':10})):
                row=frozen.run_episode(Model(),types.SimpleNamespace(unnorm_key='k'),None,Env(),case,ckpt,{'sha256':'p'},time.time()+30)
                results.append({k:v for k,v in row.items() if k not in ('started_ns','completed_ns')})
        self.assertEqual(results[0],results[1]);self.assertEqual(seeds[:2],seeds[2:])
        meta=sink.tasks[-1][3];self.assertFalse(meta['success']);self.assertEqual(meta['checkpoint'],'baseline')


if __name__=='__main__': unittest.main()
