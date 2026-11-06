"""Execute backend adapter methods with CPU fakes, not CUDA/Ray imports.

These tests check the adapters that the actual training entry points call.
They do not certify DeepSpeed/DDP communication completion on a real GPU.
"""
import asyncio
from copy import deepcopy
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

from collective_phase_gate import BoundaryAttestation, PhaseSpec
from cost_scheduler_runtime import (
    TrainingAdmissionClient, scheduled_policy_update,
    scheduled_world_begin, scheduled_world_end,
)
from weaver_online_engine import WeaverOnlineEngine


SOURCE = 'a'*64


class Stream:
    def __init__(self, events):
        self.events = events

    def synchronize(self):
        self.events.append('own_stream_completed')


class FakeCuda:
    def __init__(self, events):
        self.events = events
        self.stream = Stream(events)

    def current_stream(self, device=None):
        self.events.append(('own_stream', device))
        return self.stream

    def synchronize(self, *args):
        raise AssertionError('Adapter must not require a whole-device drain')


def validate_spec(raw):
    raw = deepcopy(raw)
    return PhaseSpec(attestation=BoundaryAttestation(**raw.pop('attestation')), **raw)


class PolicyBroker:
    """Synchronous remote seam used by the real contextmanager's ray.get."""
    def __init__(self, events):
        self.events = events
        self.raw_spec = None
        self.acquire_training = SimpleNamespace(remote=self.acquire)
        self.complete_training = SimpleNamespace(remote=self.complete)
        self.fail_training = SimpleNamespace(remote=self.fail)

    def acquire(self, rank, raw):
        validate_spec(raw)  # Mirrors the real broker's public boundary validator.
        self.raw_spec = deepcopy(raw)
        self.events.append(('admitted', rank))
        return {'work_id': 'phase-work'}

    def complete(self, rank, work_id, seconds):
        self.events.append(('complete', rank, work_id, seconds))

    def fail(self, group_id, reason):
        self.events.append(('failed', group_id, reason))


class PolicyAdapterTests(unittest.TestCase):
    def owner(self, completed_policy=0, micro=8, gas=18):
        self.events = []
        self.broker = PolicyBroker(self.events)
        self.fake_torch = SimpleNamespace(cuda=FakeCuda(self.events))
        client = TrainingAdmissionClient(self.broker, {'run_id': 'cpu-run'}, SOURCE,
            ray_module=SimpleNamespace(get=lambda value: value))
        return SimpleNamespace(cost_admission=client, rank=0,
                               global_step=completed_policy,
                               train_batch_size=micro, accumulation_steps=gas)

    def test_actual_first_policy_context_uses_sequence0_p0_w0(self):
        owner = self.owner()
        with patch.dict(sys.modules, {'torch': self.fake_torch}):
            with scheduled_policy_update(owner):
                self.events.append('optimizer_update_body')
                owner.global_step += 1
        raw = self.broker.raw_spec
        self.assertEqual((raw['sequence'], raw['policy_version'], raw['world_version']), (0, 0, 0))
        self.assertEqual(raw['attestation']['source_sha256'], SOURCE)
        self.assertEqual(raw['attestation']['backend'], 'policy_zero2')
        self.assertEqual(raw['attestation']['global_batch'], 576)
        self.assertEqual(raw['attestation']['boundary'], 'optimizer_update')
        self.assertEqual(self.events.count('own_stream_completed'), 2)
        self.assertLess(self.events.index('own_stream_completed'), self.events.index(('admitted', 0)))
        self.assertLess(self.events.index(('admitted', 0)), self.events.index('optimizer_update_body'))
        self.assertEqual(sum(isinstance(event, tuple) and event[0] == 'complete' for event in self.events), 1)

    def test_actual_fifth_and_tenth_policy_context_keep_global576(self):
        for completed in [4, 9]:
            with self.subTest(completed_policy=completed):
                owner = self.owner(completed, micro=4, gas=36)
                with patch.dict(sys.modules, {'torch': self.fake_torch}):
                    with scheduled_policy_update(owner):
                        owner.global_step += 1
                raw = self.broker.raw_spec
                self.assertEqual(raw['sequence'], completed)
                self.assertEqual(raw['policy_version'], completed)
                self.assertEqual(raw['world_version'], completed//5)
                self.assertEqual(raw['attestation']['micro_batch'], 4)
                self.assertEqual(raw['attestation']['accumulation_steps'], 36)
                self.assertEqual(raw['attestation']['global_batch'], 576)

    def test_sixth_update_restores_ordinary_micro8_gas18(self):
        owner = self.owner(5)
        with patch.dict(sys.modules, {'torch': self.fake_torch}):
            with scheduled_policy_update(owner):
                owner.global_step += 1
        raw = self.broker.raw_spec
        self.assertEqual((raw['sequence'], raw['policy_version'], raw['world_version']), (5, 5, 1))
        self.assertEqual((raw['attestation']['micro_batch'], raw['attestation']['accumulation_steps']), (8, 18))

    def test_failure_reports_group_failure_without_successful_completion(self):
        owner = self.owner()
        with patch.dict(sys.modules, {'torch': self.fake_torch}):
            with self.assertRaisesRegex(RuntimeError, 'backend failed'):
                with scheduled_policy_update(owner):
                    raise RuntimeError('backend failed')
        self.assertEqual(sum(isinstance(e, tuple) and e[0] == 'failed' for e in self.events), 1)
        self.assertFalse(any(isinstance(e, tuple) and e[0] == 'complete' for e in self.events))

    def test_disabled_adapter_does_not_import_torch_or_claim_work(self):
        owner = SimpleNamespace()
        with scheduled_policy_update(owner):
            pass


class NativeBoundaryMethodTests(unittest.TestCase):
    def engine(self, *, cuda=False):
        engine = WeaverOnlineEngine.__new__(WeaverOnlineEngine)
        self.events = []
        engine.active_update = None
        engine.micro_steps = 64
        engine.accumulation_steps = 64
        engine.publication_idle = lambda: True
        engine.device = SimpleNamespace(type='cuda' if cuda else 'cpu')
        engine.torch = SimpleNamespace(cuda=FakeCuda(self.events))
        engine.rank, engine.world_size, engine.global_steps = 0, 4, 1
        return engine

    def test_actual_native_boundary_acks_only_owner_stream(self):
        engine = self.engine(cuda=True)
        result = engine.scheduler_boundary()
        self.assertEqual((result['backend'], result['boundary']), ('native_ddp', 'optimizer_update'))
        self.assertEqual((result['rank'], result['world_size'], result['global_steps']), (0, 4, 1))
        self.assertTrue(result['own_stream_complete'])
        self.assertFalse(result['global_gpu_drain'])
        self.assertEqual(self.events[-1], 'own_stream_completed')

    def test_actual_native_boundary_rejects_live_gradients_and_publication(self):
        for mutation in ['active_update', 'micro_steps', 'publication_idle']:
            with self.subTest(mutation=mutation):
                engine = self.engine()
                setattr(engine, mutation, {'active_update': {'micro': 4}, 'micro_steps': 65,
                                          'publication_idle': lambda: False}[mutation])
                with self.assertRaises(RuntimeError):
                    engine.scheduler_boundary()


class WorldBroker:
    def __init__(self):
        self.spec = None
        self.completed = []
        self.failed = []
        self.acquire_training = SimpleNamespace(remote=self.acquire)
        self.complete_training = SimpleNamespace(remote=self.complete)
        self.fail_training = SimpleNamespace(remote=self.fail)

    async def acquire(self, rank, raw):
        validate_spec(raw)
        self.spec = deepcopy(raw)
        return {'work_id': 'world-work'}

    async def complete(self, *args):
        self.completed.append(args)

    async def fail(self, *args):
        self.failed.append(args)


class WorldAdapterTests(unittest.IsolatedAsyncioTestCase):
    def owner(self, *, world_step=0):
        broker = WorldBroker()
        receipt = {'safe': True, 'backend': 'native_ddp', 'boundary': 'optimizer_update',
                   'rank': 0, 'world_size': 4, 'global_steps': world_step,
                   'own_stream_complete': True, 'global_gpu_drain': False}
        def call(method):
            self.assertEqual(method, 'scheduler_boundary')
            return dict(receipt)
        client = TrainingAdmissionClient(broker, {'run_id': 'cpu-run'}, SOURCE,
                                        ray_module=SimpleNamespace())
        owner = SimpleNamespace(rank=0, world_size=4, micro=4, gas=64, cost_admission=client,
                                worker_client=SimpleNamespace(call=call))
        return owner, broker, receipt

    async def test_actual_world_first_and_second_joint_versions(self):
        for policy_step in [5, 10]:
            with self.subTest(policy_step=policy_step):
                owner, broker, receipt = self.owner(world_step=policy_step//5-1)
                token = await scheduled_world_begin(owner, policy_step)
                self.assertEqual(broker.spec['sequence'], policy_step//5-1)
                self.assertEqual(broker.spec['policy_version'], policy_step-1)
                self.assertEqual(broker.spec['world_version'], policy_step//5-1)
                self.assertEqual(broker.spec['attestation']['global_batch'], 1024)
                receipt['global_steps'] += 1
                await scheduled_world_end(owner, token)
                self.assertEqual(len(broker.completed), 1)

    async def test_wrong_native_boundary_receipt_cannot_admit_collective(self):
        for field, value in [('safe', False), ('backend', 'fsdp'), ('boundary', 'microbatch'),
                             ('rank', 3), ('world_size', 8), ('global_steps', 7),
                             ('own_stream_complete', False), ('global_gpu_drain', True)]:
            with self.subTest(field=field):
                owner, broker, receipt = self.owner()
                receipt[field] = value
                with self.assertRaises((ValueError, RuntimeError)):
                    await scheduled_world_begin(owner, 5)
                self.assertIsNone(broker.spec)

    async def test_wrong_native_post_update_receipt_never_commits_cost_progress(self):
        for field, value in [('safe', False), ('rank', 3), ('global_steps', 0),
                             ('own_stream_complete', False)]:
            with self.subTest(field=field):
                owner, broker, receipt = self.owner()
                token = await scheduled_world_begin(owner, 5)
                receipt['global_steps'] = 1
                receipt[field] = value
                with self.assertRaises((ValueError, RuntimeError)):
                    await scheduled_world_end(owner, token)
                self.assertFalse(broker.completed)

    async def test_actual_world_error_blocks_future_grants_without_fake_success(self):
        owner, broker, _ = self.owner()
        token = await scheduled_world_begin(owner, 5)
        await scheduled_world_end(owner, token, RuntimeError('native backward failed'))
        self.assertEqual(len(broker.failed), 1)
        self.assertFalse(broker.completed)


if __name__ == '__main__':
    unittest.main()
