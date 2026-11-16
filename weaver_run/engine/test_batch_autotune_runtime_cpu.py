"""Exercise finite calibration adapters without Ray, CUDA or model weights."""
import asyncio
from types import SimpleNamespace
import unittest

import torch

from batch_autotune_runtime import (DeviceMemoryMeter, configure_training,
    configure_inference, refresh_values, inference_report, aggregate_sample,
    action_timer, finish_action)


class Cuda:
    def __init__(self):
        self.used, self.allocated, self.reserved = 200, 100, 120
        self.peak, self.peak_reserved = 100, 120
    def mem_get_info(self): return (1000-self.used, 1000)
    def memory_allocated(self): return self.allocated
    def memory_reserved(self): return self.reserved
    def max_memory_allocated(self): return self.peak
    def max_memory_reserved(self): return self.peak_reserved
    def reset_peak_memory_stats(self):
        self.peak, self.peak_reserved = self.allocated, self.reserved


class MemoryTests(unittest.TestCase):
    def test_counts_colocated_memory_and_peak_released_before_sample(self):
        cuda = Cuda()
        meter = DeviceMemoryMeter(cuda)
        meter.begin_micro()
        cuda.peak, cuda.peak_reserved = 180, 220
        cuda.used, cuda.reserved = 240, 150
        meter.end_micro()
        self.assertEqual(meter.report(), dict(total_bytes=1000,
            peak_used_bytes=310, activation_bytes=80))

    def test_rejects_config_drift_before_training(self):
        owner = SimpleNamespace(_batch_autotune_enabled=True, training_role='policy',
            global_step=4, _policy_joint_batch_mode=SimpleNamespace(active=None),
            world_config=SimpleNamespace(backend='model_free'), rank=0)
        with self.assertRaises(ValueError): configure_training(owner, 6, 16)
        with self.assertRaises(ValueError): configure_training(owner, 5, 32)
        self.assertEqual(configure_training(owner, 5, 16)['micro'], 16)
        owner._policy_joint_batch_mode.active = {'policy_step':5}
        with self.assertRaises(RuntimeError): configure_training(owner, 5, 8)


class ValueModel:
    def __init__(self): self.batches = []
    def prepare_inputs_batch(self, inputs):
        self.batches.append(list(inputs)); return torch.tensor(inputs, dtype=torch.float32)
    def __call__(self, prepared): return None, prepared * 2 + 1


class CriticTests(unittest.IsolatedAsyncioTestCase):
    def owner(self):
        return SimpleNamespace(actor_id=0, capture_model_version=3, _policy_dispatch=None,
            model=ValueModel(), _check_inference_health=lambda:None,
            telemetry=SimpleNamespace(event=lambda *args,**kwargs:None))

    async def test_refresh_preserves_order_version_and_yields_between_chunks(self):
        owner = self.owner()
        configure_inference(owner, 4, 4)
        ticks = []
        async def other_request():
            await asyncio.sleep(0)
            ticks.append(len(owner.model.batches))
        background = asyncio.create_task(other_request())
        result = await refresh_values(owner, list(range(9)), 3,
            torch_module=SimpleNamespace(cuda=Cuda(), inference_mode=torch.inference_mode))
        await background
        self.assertEqual(result, {'policy_version':3,'values':[1.,3.,5.,7.,9.,11.,13.,15.,17.]})
        self.assertEqual([len(x) for x in owner.model.batches], [4,4,1])
        self.assertTrue(ticks and ticks[0] < 3)
        report = inference_report(owner, 4)
        self.assertEqual(report['refresh_observations'], 9)
        self.assertFalse(owner._autobatch_refresh_active)

    async def test_rejects_version_change_and_releases_refresh_pin_on_error(self):
        owner = self.owner(); configure_inference(owner, 4, 2)
        async def change():
            owner.capture_model_version = 4
        task = asyncio.create_task(change())
        with self.assertRaisesRegex(RuntimeError, 'version'):
            await refresh_values(owner, [1,2,3,4,5], 3,
                torch_module=SimpleNamespace(cuda=Cuda(), inference_mode=torch.inference_mode))
        await task
        self.assertFalse(owner._autobatch_refresh_active)

    async def test_wrong_version_rejected_before_forward(self):
        owner=self.owner(); configure_inference(owner,4,8)
        with self.assertRaises(ValueError):
            await refresh_values(owner,[1],2)
        self.assertEqual(owner.model.batches, [])

    async def test_slow_action_crossing_report_boundary_is_not_lost_or_counted_twice(self):
        owner=self.owner(); configure_inference(owner,4,2)
        token=action_timer(owner,dict(workflow='real_rollout',submitted_ns=1))
        self.assertEqual(inference_report(owner,4)['action_latencies'],[])
        # This completion is after the snapshot but before the next config RPC.
        finish_action(token)
        owner.capture_model_version=4
        configure_inference(owner,5,4)
        self.assertEqual(len(inference_report(owner,5)['action_latencies']),1)
        self.assertEqual(inference_report(owner,5)['action_latencies'],[])


class AggregationTests(unittest.TestCase):
    def reports(self):
        trainers = [dict(rank=i,policy_step=1,micro=8,train_seconds=40+i,
            memory=dict(total_bytes=1000,peak_used_bytes=600+i,activation_bytes=100)) for i in range(4)]
        inference = [dict(rank=i,policy_step=1,refresh_micro=2,refresh_seconds=10+i,
            refresh_observations=100,action_latencies=[.1,.2],
            memory=dict(total_bytes=1000,peak_used_bytes=400,activation_bytes=20)) for i in range(4)]
        return trainers,inference

    def test_uses_slowest_rank_without_summing_parallel_elapsed_time(self):
        trainers,inference=self.reports()
        sample=aggregate_sample(1,80,trainers,inference)
        self.assertEqual(sample['train_seconds'],43)
        self.assertEqual(sample['refresh_seconds'],13)
        self.assertEqual(sample['refresh_observations'],400)
        self.assertEqual(sample['action_p95_seconds'],.2)
        self.assertEqual(sample['cards'][3]['peak_used_bytes'],603)

    def test_missing_or_mixed_rank_cannot_drive_configuration(self):
        trainers,inference=self.reports()
        with self.assertRaises(ValueError): aggregate_sample(1,80,trainers[:-1],inference)
        trainers[3]['micro']=16
        with self.assertRaises(ValueError): aggregate_sample(1,80,trainers,inference)


if __name__ == '__main__': unittest.main()
