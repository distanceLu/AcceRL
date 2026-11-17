"""Policy-only placement, real Replay and checkpoint contracts without GPUs."""
import asyncio
import copy
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

import fresh_replay_pressure as pressure
from fresh_config import FreshConfig
from fresh_policy_replay import SegmentReplay
from frozen_policy_eval import validate_checkpoint_payload
import parallel_checkpoint as checkpoint
from test_fresh_policy_replay_cpu import segment, Writer
import training_topology as topology


CONFIG = {'backend': 'model_free', 'world_trainable': False, 'reward_provider': 'environment'}
PLAN = dict(CONFIG, world_kind=None, world_size=4, trainer_kinds=['policy'],
            inference_kinds=['policy'], wm_consumers=[], expected_wm_consumers=0,
            gpu_model_processes_per_card=2, expected_gpu_model_processes=8,
            logical_update_ratio='1:0:0', checkpoint_kinds=['policy'], publication_kinds=['policy'])


class PolicyReplay:
    """Only the async Ray transport is replaced; coordinator sharding is real."""
    def __init__(self):
        self.calls = 0
        self.sample = SimpleNamespace(remote=self.draw)

    async def draw(self, count):
        self.calls += 1
        ids = [f'{self.calls}:{index}' for index in range(count)]
        return ids, list(range(count)), {'sample_ids': ids, 'fresh_reservation_token': 'lease',
                                        'source_counts': {'real': count, 'imagined': 0}}


class PolicyOnlyTopology(unittest.IsolatedAsyncioTestCase):
    def test_four_cards_reserve_only_policy_trainer_and_inference(self):
        bindings = [dict(index=str(i + 4), uuid=f'GPU-{i}', resource=f'card{i}') for i in range(4)]
        with patch.dict(os.environ, {'ACCERL_USE_MPS': '0'}):
            options = topology.ray_init_options(bindings, topology.SPLIT_LAYOUT, ('policy',))
            self.assertEqual(options['resources'], {'card0': 2., 'card1': 2., 'card2': 2., 'card3': 2.})
            for rank in range(4):
                for role in ('policy', 'policy_trainer'):
                    actor = topology.actor_options(bindings, role, rank, topology.SPLIT_LAYOUT, ('policy',))
                    self.assertEqual(actor['resources'], {f'card{rank}': 1.})
                    self.assertEqual(actor['runtime_env']['env_vars']['ACCERL_PHYSICAL_GPU'], str(rank + 4))
            for role in ('ctrl', 'reward', 'ctrl_trainer', 'reward_trainer'):
                with self.assertRaises(ValueError):
                    topology.actor_options(bindings, role, 0, topology.SPLIT_LAYOUT, ('policy',))
            with self.assertRaises(ValueError):
                topology.ray_init_options(bindings, topology.FOUR_LAYOUT, ('policy',))

    async def test_four_rank_real_cohort_advances_without_a_world_replay_or_credits(self):
        replay = PolicyReplay()
        coordinator = topology.GlobalBatchCoordinator(replay, None, 4, wm_consumers=(), pipeline_enabled=False)
        first = await coordinator.sample_policy(0, 2)
        ahead = asyncio.create_task(coordinator.sample_policy(0, 2))
        await asyncio.sleep(0)
        self.assertFalse(ahead.done())
        rest = [await coordinator.sample_policy(rank, 2) for rank in range(1, 4)]
        self.assertEqual([item for batch in [first] + rest for item in batch[0]],
                         ['1:0', '1:1', '1:2', '1:3', '1:4', '1:5', '1:6', '1:7'])
        self.assertEqual((await asyncio.wait_for(ahead, 1))[0], ['2:0', '2:1'])
        self.assertEqual(replay.calls, 2)
        with self.assertRaises(ValueError):
            await coordinator.sample_wm(0, 2)

    def test_empty_world_consumers_require_an_explicit_absent_world_lane(self):
        replay = PolicyReplay()
        for world, consumers, pipeline in ((replay, (), False), (None, (), True), (None, ('ctrl',), False)):
            with self.subTest(consumers=consumers, pipeline=pipeline), self.assertRaises(ValueError):
                topology.GlobalBatchCoordinator(replay, world, 4, wm_consumers=consumers,
                                                pipeline_enabled=pipeline)


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.ones(2))
        self.lm_head = torch.nn.Linear(2, 2)
        self.attn_pool = torch.nn.Linear(2, 1)
        self.value_head = torch.nn.Linear(2, 1)


def metadata():
    return {'run_id': 'model-free-cpu', 'world_backend_config': copy.deepcopy(CONFIG),
            'service_plan': copy.deepcopy(PLAN), 'args': {
                'pretrained_checkpoint': '/weights/spatial', 'checkpoint2': '/weights/distilled.pt',
                'benchmark': 'libero_spatial', 'num_images_in_input': 2,
                'use_proprio': False, 'use_bf16': True}}


class PolicyOnlyCheckpoint(unittest.TestCase):
    def setUp(self):
        self.policy = TinyPolicy()
        owner = SimpleNamespace(rank=0, world_size=4, training_role='policy', global_step=7, model=self.policy)
        self.components = {'policy': checkpoint.export_split_component(owner, 'policy')}

    def test_model_free_snapshot_roundtrips_and_is_accepted_by_fixed_policy_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = checkpoint.save_split_weights(self.components, directory,
                active_kinds=('policy',), metadata=metadata())
            self.assertEqual(manifest['steps'], {'policy': 7})
            path = Path(directory) / 'latest_weights.pt'
            payload = torch.load(path, map_location='cpu', weights_only=False)
            self.assertEqual(set(payload['models']), {'policy'})
            self.assertEqual(validate_checkpoint_payload(payload)['policy_step'], 7)
            destination = SimpleNamespace(model=TinyPolicy())
            checkpoint.apply_parallel_weights(destination, path, active_kinds=('policy',))
            for key, value in self.policy.state_dict().items():
                torch.testing.assert_close(destination.model.state_dict()[key], value)
        self.assertFalse(torch.cuda.is_initialized())

    def test_incomplete_or_contradictory_model_free_identity_cannot_overwrite_snapshot(self):
        mutations = (
            lambda m: m.pop('world_backend_config'),
            lambda m: m['world_backend_config'].update(backend='weaver'),
            lambda m: m['world_backend_config'].update(world_trainable=True),
            lambda m: m['world_backend_config'].pop('reward_provider'),
            lambda m: m['service_plan'].update(inference_kinds=['policy', 'ctrl']),
            lambda m: m['service_plan'].update(expected_gpu_model_processes=16),
            lambda m: m['service_plan'].update(wm_consumers=['ctrl']),
            lambda m: m.update(online_world={}),
        )
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'latest_weights.pt'
            target.write_bytes(b'previous checkpoint')
            for change in mutations:
                bad = metadata()
                change(bad)
                with self.subTest(change=change), self.assertRaises(ValueError):
                    checkpoint.save_split_weights(self.components, directory,
                        active_kinds=('policy',), metadata=bad)
                self.assertEqual(target.read_bytes(), b'previous checkpoint')
            bad_components = copy.deepcopy(self.components)
            bad_components['policy']['world_size'] = 2
            with self.assertRaises(ValueError):
                checkpoint.save_split_weights(bad_components, directory,
                    active_kinds=('policy',), metadata=metadata())

    def test_full_success_checkpoint_requires_the_exact_validated_policy_before_overwrite(self):
        meta = metadata()
        meta['fixed_validation'] = {'target_reached': True, 'last': {
            'policy_step': 7, 'cases': 50, 'successes': 50}}
        meta['fixed_validation']['target_validation'] = dict(meta['fixed_validation']['last'])
        with tempfile.TemporaryDirectory() as directory:
            manifest = checkpoint.save_split_weights(self.components, directory,
                reason='final', active_kinds=('policy',), metadata=meta)
            self.assertEqual(manifest['steps']['policy'], 7)
            target = Path(directory) / 'latest_weights.pt'
            saved = target.read_bytes()
            for changed in ({'policy_step': 6}, {'policy_step': 8},
                            {'successes': 49}, {'cases': 49}):
                invalid = copy.deepcopy(meta)
                invalid['fixed_validation']['last'].update(changed)
                with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, 'validated Policy'):
                    checkpoint.save_split_weights(self.components, directory,
                        reason='final', active_kinds=('policy',), metadata=invalid)
                self.assertEqual(target.read_bytes(), saved)
            invalid = copy.deepcopy(meta)
            invalid['fixed_validation']['target_validation']['policy_step'] = 6
            with self.assertRaisesRegex(ValueError, 'validated Policy'):
                checkpoint.save_split_weights(self.components, directory,
                    reason='final', active_kinds=('policy',), metadata=invalid)
            self.assertEqual(target.read_bytes(), saved)

    def test_online_weaver_still_requires_the_native_artifact_and_matching_boundary(self):
        meta = {'run_id': 'online', 'service_plan': {
            'trainer_kinds': ['policy', 'ctrl'], 'inference_kinds': ['policy', 'ctrl'],
            'reward_provider': 'world_head', 'expected_gpu_model_processes': 16}}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                checkpoint.save_split_weights(self.components, directory, active_kinds=('policy',), metadata=meta)
            meta['online_world'] = dict(protocol='weaver_online_joint_weights_v1', run_id='online',
                world_size=4, weights_only=True, tensor_count=866, reason='periodic',
                policy_step=7, successful_updates=1, model_version=1)
            result = checkpoint.save_split_weights(self.components, directory,
                active_kinds=('policy',), metadata=meta)
            self.assertEqual(result['steps'], {'policy': 7})
            meta['online_world']['policy_step'] = 5
            with self.assertRaises(ValueError):
                checkpoint.save_split_weights(self.components, directory, active_kinds=('policy',), metadata=meta)


def real_store():
    store = SegmentReplay(32, FreshConfig(real_policy_replay=True, real_only=True, real_fraction=1.))
    store.set_versions(4, 0)
    return store


class RealOnlyPressure(unittest.IsolatedAsyncioTestCase):
    def test_single_and_multi_sample_shortage_never_requests_imagined_data(self):
        store = real_store()
        store.add(segment('real', 'real', count=2))
        one = pressure.shortage(store, 1)
        self.assertEqual(one['quotas'], {'real': 1, 'imagined': 0})
        four = pressure.shortage(store, 4)
        self.assertEqual(four['quotas'], {'real': 4, 'imagined': 0})
        self.assertEqual(four['missing'], {'real': 2, 'imagined': 0})

    async def test_waits_for_real_data_without_an_imagined_producer(self):
        store = real_store()
        owner = SimpleNamespace(fresh_store=store, fresh_production_budget=None,
            fresh_admitted_chunks={'real': 0, 'imagined': 0}, telemetry=Writer())
        with patch.object(pressure, 'POLL_SECONDS', .001), patch.object(pressure, 'PRESSURE_SECONDS', .002):
            waiting = asyncio.create_task(pressure.reserve_with_production_pressure(owner, 4))
            await asyncio.sleep(.01)
            self.assertFalse(waiting.done())
            store.add(segment('collected', 'real', count=4))
            token, ids = await asyncio.wait_for(waiting, 1)
        self.assertEqual(len(ids), 4)
        events = owner.telemetry.events
        self.assertTrue(events)
        self.assertTrue(all(e[1]['details']['missing']['imagined'] == 0 for e in events))
        self.assertFalse(any(e[0] == 'fresh_replay_credit_shortfall' for e in events))
        store.cancel(token)


if __name__ == '__main__':
    unittest.main()
