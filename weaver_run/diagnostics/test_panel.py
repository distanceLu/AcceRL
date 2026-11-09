import json,os,re,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'weaver_run/engine'))
from .panel_common import spec,STATES,context,discounted_targets,validate_episode
from .panel_runner import ema_after,selected_batch
from .policy_panel import gradients,restore_input


def episode(n=40,success=True):
    return dict(episode_id='test',instruction='task',initial_state_id=STATES[0],success=success,
        frames=np.zeros((n+1,2,3,224,224),np.uint8),states=np.zeros((n+1,8),np.float32),
        actions=np.arange(n*7,dtype=np.float32).reshape(n,7))


class PanelTests(unittest.TestCase):
    def test_episode_requires_complete_outcome_and_endpoint(self):
        ep=episode();self.assertEqual(validate_episode(ep),40)
        with self.assertRaises(ValueError):validate_episode(dict(ep,frames=ep['frames'][:-1]))
        with self.assertRaises(ValueError):validate_episode(dict(ep,success=False))
        self.assertEqual(validate_episode(episode(220,False)),220)

    def test_context_incoming_actions_and_padding(self):
        ep=episode();cfg=dict(schema={})
        d=context(ep,0,cfg)
        self.assertEqual(d['frame_indices'],[0]*32)
        self.assertEqual(sum(d['valid_mask']),1);self.assertEqual(sum(d['action_mask']),0)
        self.assertTrue(np.all(d['actions']==0));self.assertEqual(d['context_padding']['count'],31)
        d=context(ep,31,cfg)
        np.testing.assert_array_equal(d['actions'][1:],ep['actions'][:31])
        self.assertEqual(d['frame_indices'],list(range(32)))
        with self.assertRaises(ValueError):context(ep,40,cfg)

    def test_rm_targets_are_observed_discounted_outcomes(self):
        ep=episode();v=discounted_targets(ep,[0,32,40])
        self.assertAlmostEqual(v[-1],1.);self.assertAlmostEqual(v[1],.99)
        np.testing.assert_array_equal(discounted_targets(episode(220,False),[0,220]),[0,0])
        with self.assertRaises(ValueError):discounted_targets(ep,[41])

    def test_cross_scoring_executes_native_adapter_with_tiny_cpu_model(self):
        os.environ['WEAVER_TEST_UPSTREAM']=str(ROOT/'weaver_run/vendor/weaver')
        from test_fresh_padding_backend_cpu import configuration
        from test_weaver_optimized_backend_cpu import make_backend
        from .native_panel_worker import score_episode
        cfg=configuration(cache=False);upstream,backend=make_backend(cfg)
        ep=episode();ep['actions'][:]=0;ep['frames'][:]=128
        rows=score_episode(backend,ep,cfg,0)
        self.assertEqual(len(rows),8)
        self.assertEqual({r['horizon'] for r in rows},{8,16,32})
        for r in rows:
            self.assertEqual(len(r['rm_real_predictions']),r['horizon'])
            self.assertEqual(len(r['measured_states']),r['horizon'])
            self.assertTrue(np.isfinite(r['wm']['image_mse_full']))
        # All real-image encodes must use the native [0,1] domain, not [-1,1].
        for batch in upstream.im_encoder.inputs:
            self.assertGreaterEqual(float(batch.min()),0.)
            self.assertLessEqual(float(batch.max()),1.)
            self.assertAlmostEqual(float(batch.mean()),128/255,places=5)

    def test_queue_waits_without_stopping_or_overwriting_training_pointer(self):
        from .queue_panel import submit
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);last=root/'.last.json';last.write_text('"current-training"')
            calls=[];saved={}
            def require(ok,msg):
                if not ok:raise ValueError(msg)
            def create(job,queue):
                calls.append('create');saved.update(job,id='diagnostic-job',state='queued');return dict(saved)
            cli=types.SimpleNamespace(LABEL=re.compile(r'[a-z0-9_-]+'),EXPERIMENTS=root,RUNTIME=root/'runtime',CHECKOUT=root/'code',
                PYTHON=sys.executable,QUEUE=root/'queue',require=require,lookup=lambda sid:dict(saved) if saved else None,
                queue_rows=lambda *_:[dict(id='live-training',cwd='/old-source',argv_json='["old-runner"]')],
                published=lambda *_args,**_kwargs:'a'*40,pin_worktree=lambda *_:calls.append('pin'),revision=lambda *_:'a'*40,
                cpu=lambda *_args,**_kwargs:json.dumps(dict(cuda_initialized=False)),create=create)
            with patch('shutil.disk_usage',return_value=types.SimpleNamespace(free=100*1024**3)):
                a=submit(cli,'test');b=submit(cli,'test')
            self.assertEqual(a['state'],'queued');self.assertEqual(a['waiting_behind'],['live-training'])
            self.assertEqual(b['job_id'],'diagnostic-job');self.assertEqual(calls,['pin','create'])
            self.assertEqual(last.read_text(),'"current-training"');self.assertFalse(saved['manual'])
            self.assertEqual(saved['timeout_seconds'],14520)

    def test_queue_will_not_repin_in_use_git_runtime(self):
        from .queue_panel import submit
        def require(ok,msg):
            if not ok:raise ValueError(msg)
        cli=types.SimpleNamespace(LABEL=re.compile(r'.+'),EXPERIMENTS=Path('/tmp/unused'),RUNTIME=Path('/used/runtime'),
            require=require,lookup=lambda _:None,queue_rows=lambda *_:[dict(cwd=str(Path('/used/runtime')),argv_json='[]')])
        with self.assertRaisesRegex(ValueError,'owns the Git runtime'):submit(cli,'test')

    def test_finite_protocol_and_source_selection(self):
        p=spec();self.assertEqual(len(set(p['initial_states'])),16)
        r=[dict(rewards=list(range(8)))];im=[dict(rewards=[1]*4) for _ in range(3)]
        s=selected_batch(r,im,4)
        self.assertEqual(len(s),16);self.assertEqual(len(set(s)),16)
        self.assertEqual(sum(j==0 for j,i in s),4)
        self.assertEqual({i for j,i in s if j==1},set(range(4)))
        with self.assertRaises(ValueError):selected_batch(r,im[:-1],4)

    def test_ema_requires_four_matching_ranks_at_checkpoint(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as td:
            db=Path(td)/'events.db';c=sqlite3.connect(db)
            c.execute('create table events(run_id text,kind text,details_json text)')
            row=dict(policy_version=450,ema_updates=451,ema_mean=.1,ema_variance=.25,ema_scale_floor=.001,normalization='ema_zscore',mode='fused')
            for rank in range(4):c.execute('insert into events values(?,?,?)',('run','jit_gae_optimizer_window',json.dumps(dict(row,rank=rank))))
            c.commit();m=dict(db=str(db),run_id='run',policy_step=451)
            self.assertEqual(ema_after(m)['scale'],.5)
            c.execute('delete from events where rowid=4');c.commit()
            with self.assertRaises(ValueError):ema_after(m)
            c.close()

    def test_toy_end_to_end_gradient_capture_does_not_change_weights(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__();self.vla=torch.nn.Linear(2,8*7*4);self.value_head=torch.nn.Linear(2,1)
            def prepare_inputs_batch(self,items):return torch.stack([x['x'] for x in items])
            def forward(self,x):return self.vla(x).reshape(-1,56,4),self.value_head(x.detach())
        torch.manual_seed(4);model=Toy();before={n:p.clone() for n,p in model.named_parameters()}
        def segment(source,n):
            inp=[dict(x=np.array([i*.01,1.],np.float32)) for i in range(n+1)]
            with torch.no_grad():logits,_=model(model.prepare_inputs_batch([restore_input(x,torch.float32) for x in inp[:-1]]))
            return dict(source=source,inputs=inp[:-1],endpoint=inp[-1] if source=='imagined' else None,
                terminal=source=='real',old_logits=logits.reshape(n,8,7,4).numpy(),tokens=np.zeros((n,8,7),np.int64),
                masks=np.ones((n,8,7),bool),rewards=[.1]*n)
        real=[segment('real',8)];imag=[segment('imagined',4) for _ in range(3)]
        real[0]['masks'][-1,3:]=False
        args=dict(gamma=.99,lambda_=.95,vf_coef=.5,ent_coef=0.,kl_coef=.1,clip_mode='gipo',clip_eps=.2,train_batch_size=8)
        with tempfile.TemporaryDirectory() as td:
            result=gradients(model,torch.float32,real+imag,selected_batch(real,imag,4),dict(mean=.1,scale=.5),args,'toy',Path(td))
            self.assertLess(result['first_micro_gradient_relative_error'],1e-5)
            self.assertEqual(result['source_chunks'],dict(real=4,imagined=12))
            self.assertEqual(result['geometry']['critic']['policy']['joint_norm'],0.)
            self.assertGreater(result['geometry']['critic']['value']['joint_norm'],0.)
            for n,p in model.named_parameters():torch.testing.assert_close(p,before[n]);self.assertIsNone(p.grad)


if __name__=='__main__':unittest.main()
