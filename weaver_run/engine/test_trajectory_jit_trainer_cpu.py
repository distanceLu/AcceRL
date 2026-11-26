"""Exercise the production serial preparation, LR schedule, fused optimizer and EMA together."""
import asyncio
from contextlib import ExitStack
import os
from pathlib import Path
import sys
import time
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np
import torch

from test_fresh_policy_runtime_cpu import actual_method
from test_paired_gpu_trainer_cpu import Engine
from test_batch_autotune_runtime_cpu import Cuda

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rl.wm_training_utils import cosine_warmup_lr


class TrajectoryJitTrainerTests(unittest.IsolatedAsyncioTestCase):
    async def test_normal_policy_epoch_reuses_lr_warmup_and_commits_real_updates(self):
        torch.set_num_threads(1)
        engine = Engine()
        layer = engine.module.layer
        optimizer = torch.optim.AdamW([dict(params=[layer.weight], name='policy'),
                                       dict(params=[layer.bias], name='value')])
        engine.optimizer = optimizer
        updates = []
        compile_updates = []
        used = []
        def capture(kind, **kwargs):
            if kind == 'jit_gae_optimizer_window':
                updates.append(kwargs['details'])
            if kind == 'policy_trainer_compile_first_update':
                compile_updates.append(kwargs['details'])

        owner = SimpleNamespace(model=engine, base_model=engine.module, optimizer=optimizer,
            training_role='policy', active_kinds=('policy',), data_fetch_error=None, _trajectory_jit=True,
            fresh_policy_replay=True, _serial_phase=SimpleNamespace(active=None), rank=0, global_step=0,
            _policy_slot_freed=asyncio.Event(), _batch_autotune_enabled=True,
            world_size=4, _policy_replay_ready=True, replay_buffer=SimpleNamespace(),
            batch_coordinator=SimpleNamespace(sample_policy=SimpleNamespace(remote=AsyncMock())),
            _wait_for_batch=AsyncMock(return_value=0.),
            telemetry=SimpleNamespace(event=capture, flush=lambda:None), policy_train_start_step=0,
            _record_compile_activity=True, _trainer_compile_receipt={'compiled': True},
            value_lr=.01, policy_lr=.006, value_warmup_steps=100, policy_warmup_steps=100, train_iters=500,
            cfg=SimpleNamespace(lora_dropout=0.), train_batch_size=16, super_batch_size=144,
            vf_coef=.5, clip_mode='gipo', clip_eps=.2, ent_coef=.01, kl_coef=.1,
            _verify_optimizer_update=lambda model, _, start, skipped: self.assertEqual(
                (model.global_steps-start, model.skipped_steps-skipped), (1, 0)))
        namespace = dict(os=os, time=time, now_ns=time.time_ns, torch=torch, np=np, asyncio=asyncio,
            micro_begin=lambda *args: {}, micro_end=lambda _,token,*args,**kw: used.extend(token['sample_ids']),
            _trim_cuda_cache=lambda mode:None, cosine_warmup_lr=cosine_warmup_lr)
        for method in ('_serial_fetch_cpu_batch','_serial_publish_batch','_run_policy_epoch','_train_policy','_get_current_lr'):
            setattr(owner,method,MethodType(actual_method('TrainerActor',method,namespace),owner))
        ids = [f'seg-{i}:chunk0' for i in range(144)]
        actions = np.zeros((144,2,1), np.int64)
        old = np.zeros((144,2,1,3), np.float32)
        initial = [parameter.detach().clone() for parameter in layer.parameters()]
        with ExitStack() as stack:
            stack.enter_context(patch('trainer_compile.compile_counters',
                                      side_effect=[{'unique_graphs': 4},{'unique_graphs': 7}]))
            cuda = Cuda()
            for name in ('memory_allocated','reset_peak_memory_stats','mem_get_info',
                         'max_memory_reserved','memory_reserved','max_memory_allocated'):
                stack.enter_context(patch.object(torch.cuda,name,getattr(cuda,name)))
            for version in (0,1):
                key = (version+1,'policy')
                owner._serial_phase.active = key
                owner._autobatch_next = (version+1,16)
                rows = []
                for i,sample_id in enumerate(ids):
                    segment = dict(segment_id=str(i), count=2, source='real', gamma=.99, lambda_=.9,
                        observations=[{'state':i},{'state':i+1},{}], rewards=[version*.2,.1],
                        terminated=[False,True],truncated=[False,False])
                    rows.append(dict(state=i, _jit_sample_id=sample_id, _jit_segment=segment,
                        _jit_index=0, _jit_version=version, _fresh_action_mask=np.ones((2,1),bool),
                        _jit_original_denominator=16))
                meta = dict(sample_ids=ids, fresh_versions=[version,0], jit_pending=True)
                rank_ref = asyncio.get_running_loop().create_future()
                rank_ref.set_result((rows,actions,None,old,None,meta))
                owner.batch_coordinator.sample_policy.remote.return_value = {'trajectory_jit_shard_ref':rank_ref}
                await owner._serial_publish_batch(version+1)
                result = await owner._run_policy_epoch()
                self.assertEqual(result[5],{'value':version*.01/100,'policy':version*.006/100})
                self.assertEqual((result[6],owner.advantage_ema.updates),(version+1,version+1))
                self.assertEqual(set(r['sample_id'] for r in owner._jit_receipts),set(ids))
                raw = [r['advantage'] for r in owner._jit_receipts]
                self.assertAlmostEqual(updates[-1]['raw_advantage_mean'],np.mean(raw),places=5)
                self.assertAlmostEqual(updates[-1]['raw_advantage_std'],np.std(raw),places=5)
                self.assertAlmostEqual(updates[-1]['normalization_mean'],updates[0]['raw_advantage_mean'],places=5)
                self.assertCountEqual(used,ids)
                used.clear()
                for previous, current in zip(initial,layer.parameters()):
                    self.assertEqual(torch.equal(previous,current),version == 0)
        self.assertEqual([u['initialization_prepass'] for u in updates],[True,False])
        self.assertEqual([u['context_only_states'] for u in updates],[144,144])
        self.assertEqual((engine.global_steps,engine.micro_steps,engine.skipped_steps),(2,18,0))
        self.assertEqual([(r['optimizer_step'],r['counters_delta']) for r in compile_updates],
                         [(1,{'unique_graphs':3})])
