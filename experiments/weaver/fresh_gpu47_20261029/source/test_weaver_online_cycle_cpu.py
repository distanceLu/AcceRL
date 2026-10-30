"""Real native forward/Adam/hooks through the production bridge at P1..P6.

The four CPU owners test ordered Replay use and ACK barriers. They are not DDP
or CUDA IPC; the separate finite GPU smoke covers those actual mechanisms.
"""
import copy
from pathlib import Path
import json
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from weaver_online_bridge import make_weaver_online_trainer
from weaver_online_engine import WeaverOnlineEngine
from weaver_online_smoke import MemoryWriter
from test_weaver_online_bridge_cpu import Coordinator, Method


class NativeClient:
    def __init__(self, cfg, rank, role):
        from test_weaver_backend_cpu import config, TinyUpstream, sample
        from world_backends.weaver import WeaverBackend
        self.rank = rank
        self.cfg = config()
        self.backend = WeaverBackend(self.cfg, model=TinyUpstream())
        from weaver_native_reward import install_native_reward_head
        from weaver_online_smoke import reward_specification
        self.backend.native_reward_controller=install_native_reward_head(self.backend.upstream_model,reward_specification())
        self.backend.reward_training=True
        self.backend.native_score=True
        self.backend.config['reward_provider']='world_head'
        self.backend.metadata['success_head']={'calibrated':False,'trained':False}
        self.template = sample(self.cfg,self.backend.training_window_frames)
        self.template['success'][-1]=1
        self.template['terminated'][-1]=True
        self.template['label_source']='libero_env_step'
        self.engine = WeaverOnlineEngine(self.backend,rank=rank,world_size=4,micro_batch=4,accumulation_steps=64)
        self.writer = MemoryWriter()
        self.owner = SimpleNamespace(telemetry=self.writer,rank=rank,world_size=4,training_role='ctrl',
            global_step=0,capture_versions={'ctrl':0})
        self.receipt = {'worker_identity':{'pid':1000+rank},'gpu':{'uuid':'cpu-only'}}
        self.active = None
        self.engine.publication_idle = lambda: self.active is None
        self.releases = []

    def call(self, method, *values, **kwargs):
        if method == 'setup':
            return self.receipt
        if method == 'advance':
            return self.engine.advance_policy(values[0])
        if method == 'begin_update':
            return self.engine.begin_update(*values)
        if method == 'train_micro':
            from capture_training import micro_begin,micro_end
            rows,metadata,start,end = values
            self.owner.capture_wm_metadata = metadata
            self.owner.global_step = self.engine.policy_step
            token = micro_begin(self.owner,'ctrl_train',self.engine,len(rows),start,end)
            result = self.engine.micro(rows)
            micro_end(self.owner,token,self.engine)
            return result
        if method == 'state':
            return self.engine.state()
        if method == 'export':
            from training_topology import export_cpu_weights
            payload = export_cpu_weights(self.backend.model,'ctrl',self.engine.global_steps,world_backend=self.backend)
            inventory = {key:value for key,value in payload.items() if key != 'tensors'}
            self.active = {'token':'cpu-only-%d-%d'%(self.rank,self.engine.global_steps),
                           'payload':payload,'inventory':inventory}
            return self.active
        if method == 'release':
            token,ack = values
            if self.active is None or token != self.active['token'] or ack != self.active['inventory']:
                raise RuntimeError('Wrong CPU qualification ACK')
            self.releases.append(ack)
            self.active = None
            return ack
        raise ValueError(method)


@unittest.skipIf(torch is None,'Actual native CPU Torch is required')
class WholeCycle(unittest.IsolatedAsyncioTestCase):
    async def test_four_real_native_bridges_p1_to6_sample_use_commit_ack_and_next_inference(self):
        from test_weaver_backend_cpu import TinyUpstream, sample
        from world_backends.weaver import WeaverBackend
        from training_topology import apply_cpu_weights
        cfg = json.loads((Path(__file__).parent/'examples/weaver_frozen_external.json').read_text())
        cfg['world_trainable'] = True
        cls = make_weaver_online_trainer({'_online_process_client':NativeClient})
        actors = []
        for rank in range(4):
            coordinator = Coordinator()
            coordinator.ids = ['rank%d-window%04d'%(rank,n) for n in range(256)]
            with mock.patch('capture_telemetry.Telemetry', side_effect=lambda *a,**k:MemoryWriter()):
                actor = cls(rank=rank,world_size=4,wm_replay_buffer=SimpleNamespace(size=Method(lambda:2048)),
                    batch_coordinator=coordinator,ctrl_world_args={'world_backend_config':cfg})
            actor.setup_deepspeed_group('cpu-test-only',12345)
            native = actor.worker_client
            original_reserve=coordinator.reserve
            def reserve(rank,count,consumer,*,original=original_reserve,native=native):
                result=original(rank,count,consumer)
                width=native.backend.n_history+native.backend.horizon
                start=native.backend.training_window_frames-width
                result['capture_meta']['reward_cohort']={
                    'protocol':'weaver_reward_cohort_v1','window_count':1024,'frame_slice':[start,start+width],
                    'label_source':'libero_env_step','positive_count':1024,'negative_count':1024*(width-1),
                    'positive_weight':float(width-1),'negative_weight':1.,'rm_active':True}
                return result
            coordinator.reserve_wm_stream=Method(reserve)
            def sample_micro(stream,start,end,*,coordinator=coordinator,native=native):
                rows = [dict(copy.deepcopy(native.template),sample_id=value) for value in coordinator.ids[start:end]]
                return (np.stack([row['obs'].numpy() for row in rows]),
                    np.stack([row['actions'].numpy() for row in rows]),np.zeros(4),['task']*4,
                    {'capture_meta':{'sample_ids':coordinator.ids[start:end]},'backend_samples':rows})
            coordinator.sample_wm_stream = Method(sample_micro)
            actors.append(actor)
        for policy in range(1,6):
            for actor in actors:
                result = await actor.run_training_epoch(policy)
                self.assertEqual(result['metrics']['trained_ctrl'],policy==5)
        receipts = []
        for actor in actors:
            packet = actor.export_rank_weights('ctrl')
            with self.assertRaisesRegex(RuntimeError,'publication'):
                await actor.run_training_epoch(6)
            receiver = WeaverBackend(actor.worker_client.cfg,model=TinyUpstream())
            from weaver_native_reward import install_native_reward_head
            from weaver_online_smoke import reward_specification
            receiver.native_reward_controller=install_native_reward_head(receiver.upstream_model,reward_specification())
            receiver.native_score=True
            receiver.reward_training=True
            receiver.config['reward_provider']='world_head'
            receiver.metadata['success_head']={'calibrated':False,'trained':False}
            ack = apply_cpu_weights(receiver.model,packet['payload'],'ctrl',world_backend=receiver)
            receiver.set_published_version(ack['version'])
            actor.finish_ipc_weight_publish('ctrl',packet['token'],ack)
            context = sample(actor.worker_client.cfg,receiver.required_context_frames)
            future = receiver.predict_chunk(receiver.init_rollout(context,1),torch.ones(1,7)*3,1)
            self.assertEqual(future['observations'].shape[1],1)
            self.assertTrue(torch.isfinite(future['reward_predictions']).all())
            self.assertNotIn('success_probabilities',future)
            self.assertFalse(future['terminal_predictions'].any())
            receipts.append(ack)
            await actor.run_training_epoch(6)
            client = actor.worker_client
            used = [event for event in client.writer.events if event['kind']=='replay_use']
            self.assertEqual(len(used),64)
            self.assertEqual([sid for event in used for sid in event['sample_ids']],actor.batch_coordinator.ids)
            self.assertEqual(len({event['use_id'] for event in used}),64)
            updates = [event for event in client.writer.events if event['kind'].startswith('optimizer_update')]
            self.assertEqual(len(updates),1)
            self.assertEqual(updates[0]['model_version'],1)
            self.assertEqual(client.engine.global_steps,1)
            self.assertEqual(client.engine.rm_updates,1)
            self.assertEqual(client.engine.micro_steps,64)
            self.assertEqual(client.engine.state()['distributed_world_size'],1) # actual CPU test, not pretend DDP4
            self.assertTrue(client.engine.state()['gradients_released'])
        self.assertEqual(len({ack['hash'] for ack in receipts}),1)
        self.assertEqual(len({sid for actor in actors for sid in actor.batch_coordinator.ids}),1024)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
