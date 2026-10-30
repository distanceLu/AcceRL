"""Execute actual streamed native World shell without Ray or GPU models."""
from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from weaver_online_bridge import make_weaver_online_trainer, OnlineProcessClient


class Writer:
    def __init__(self, *a, **k):
        self.events = []
    def event(self, name, **values):
        self.events.append((name, values))
    def flush(self, *a, **k):
        pass


class Method:
    def __init__(self, function):
        self.function = function
    async def remote(self, *args):
        return self.function(*args)


class Coordinator:
    def __init__(self):
        self.calls = []
        self.reserve_wm_stream = Method(self.reserve)
        self.sample_wm_stream = Method(self.sample)
        self.complete_wm_stream = Method(self.complete)
        self.ids = ['window-%04d' % index for index in range(256)]
        self.done = False
        self.corrupt = False
    def reserve(self, rank, count, consumer):
        self.calls.append(('reserve', rank, count, consumer))
        return {'stream_id': 'stream', 'capture_meta': {'reservation_id': 'r1', 'sample_ids': self.ids,
                'shard_start': 0, 'shard_end': 256,'reward_cohort':{
                    'protocol':'weaver_reward_cohort_v1','window_count':1024,'frame_slice':[30,40],
                    'label_source':'libero_env_step','positive_count':0,'negative_count':10240,
                    'positive_weight':0.,'negative_weight':1.,'rm_active':False}}}
    def sample(self, stream, start, end):
        self.calls.append(('micro', start, end))
        ids = self.ids[start:end]
        return (np.zeros((4,40,2,3,4,4), np.uint8), np.zeros((4,40,7), np.float32),
            np.zeros(4), ['task']*4,
            {'capture_meta': {'sample_ids': ['wrong']*4 if self.corrupt else ids},
             'backend_samples': [{'sample_id': value, 'states': np.zeros((40,8))} for value in ids]})
    def complete(self, stream):
        self.calls.append(('complete', stream))
        self.done = True


class Client:
    def __init__(self, config, rank, role):
        self.rank, self.role = rank, role
        self.clock = self.steps = self.micros = 0
        self.calls = []
        self.receipt = {'worker_identity': {'pid': 123 + rank}, 'gpu': {'physical_gpu': str(rank+4), 'uuid': str(rank)}}
    def call(self, method, *values, **kwargs):
        self.calls.append((method, values))
        if method == 'setup':
            return self.receipt
        if method == 'advance':
            self.clock = values[0]
        if method == 'begin_update':
            self.micros = 0
        if method == 'train_micro':
            rows, metadata, start, end = values
            assert [row['sample_id'] for row in rows] == metadata['sample_ids'][start:end]
            self.micros += 1
            commit = self.micros == 64
            self.steps += int(commit)
            return {'committed': commit, 'result': {'successful_updates': self.steps, 'loss': 1.,
                'grad_norm': .5, 'learning_rate': 5e-7, 'update_wall_seconds': .1} if commit else None}
        if method == 'state':
            return {'ctrl_steps': self.steps, 'global_steps': self.steps, 'skipped_steps':0, 'global_batch':1024,
                    'distributed_world_size':4}
        return {}


class NativeBridge(unittest.IsolatedAsyncioTestCase):
    def make(self):
        cfg = json.loads((Path(__file__).parent/'examples/weaver_frozen_external.json').read_text())
        cfg['world_trainable'] = True
        self.coordinator = Coordinator()
        cls = make_weaver_online_trainer({'_online_process_client': Client,
            'ray': SimpleNamespace(util=SimpleNamespace(get_node_ip_address=lambda: '127.0.0.1'))})
        with mock.patch('capture_telemetry.Telemetry', Writer):
            actor = cls(rank=0, world_size=4, wm_replay_buffer=SimpleNamespace(size=Method(lambda: 2048)),
                batch_coordinator=self.coordinator, ctrl_world_args={'world_backend_config':cfg},
                ctrl_batch_size=4, ctrl_accumulation_steps=64)
        actor.setup_deepspeed_group('127.0.0.1', 12345)
        return actor

    async def test_world_only_draws_and_updates_at_fifth_policy_and_streams64_micro(self):
        actor = self.make()
        for policy in range(1,6):
            result = await actor.run_training_epoch(policy)
            self.assertEqual(result['metrics']['trained_ctrl'], policy == 5)
            self.assertEqual(result['model_step'], int(policy == 5))
        self.assertEqual(self.coordinator.calls[0], ('reserve', 0, 256, 'ctrl'))
        slices = [row[1:] for row in self.coordinator.calls if row[0] == 'micro']
        self.assertEqual(slices, [(index,index+4) for index in range(0,256,4)])
        self.assertTrue(self.coordinator.done)
        self.assertEqual(actor.serial_sample_identity(5)['cohort']['sample_ids'], self.coordinator.ids)
        await actor.run_training_epoch(6)
        self.assertIsNone(actor.serial_sample_identity(6)['cohort'])
        self.assertEqual(actor.get_training_state()['engines']['ctrl']['global_steps'], 1)

    async def test_bad_stream_identity_fails_without_consumption_or_release(self):
        actor = self.make()
        self.coordinator.corrupt = True
        for policy in range(1,5):
            await actor.run_training_epoch(policy)
        with self.assertRaisesRegex(ValueError, 'identities differ'):
            await actor.run_training_epoch(5)
        self.assertFalse(self.coordinator.done)
        self.assertEqual(len([row for row in actor.worker_client.calls if row[0] == 'train_micro']), 0)

    def test_stale_inference_result_does_not_trigger_transport_process_cleanup(self):
        with mock.patch('weaver_subprocess.WeaverProcessClient.call', return_value={
            '_weaver_recoverable_error':'stale_rollout', 'message':'Stale WEAVER rollout/cache version'}):
            client = OnlineProcessClient.__new__(OnlineProcessClient)
            with self.assertRaisesRegex(ValueError, 'Stale WEAVER'):
                client.call('request', {}, np.zeros((8,7)))


if __name__ == '__main__':
    unittest.main()
