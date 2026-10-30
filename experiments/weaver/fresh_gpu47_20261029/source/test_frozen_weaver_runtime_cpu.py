"""Frozen-world actual-main wiring and native CPU adapter regressions."""
from __future__ import annotations

import ast
import asyncio
from collections import Counter
import copy
import io
import json
from pathlib import Path
import types
import unittest
from unittest import mock

from test_main_startup_cpu import (Actor, Context, TREE, arguments, definition,
                                  execute_definition, startup_ast)
from pipeline_budget import PipelineBudgetConfig
from training_stop_policy import TrainingStopPolicy
from weaver_subprocess import cpu_wire, receive_frame, send_frame, WeaverRemoteOperations

ROOT = Path(__file__).resolve().parent


class FrozenMain(unittest.TestCase):
    def smoke(self):
        args = arguments()
        config = json.loads((ROOT / 'examples/weaver_frozen_external.json').read_text())
        args.world_backend_config = config
        args.ctrl_world_ckpt = None
        args.num_step_cond = 32
        args.pipeline_config = PipelineBudgetConfig(real_workers=1, max_real_in_flight=1,
            imagination_workers=4, max_imagine_in_flight=4, initial_state_enabled=True,
            real_env_step_budget=11000).to_dict()
        context = Context()
        original = Actor.call
        def call(actor, method, values, keywords):
            if method == 'get_frozen_world_state':
                return {'actor_id': actor.rank, 'backend': 'weaver', 'world_trainable': False,
                    'model_version': 0, 'trainable_parameter_count': 0,
                    'checkpoint_id': config['checkpoint_metadata']['checkpoint_id'], 'fixed_step': 3200,
                    'bundle_sha256': config['backend_options']['frozen_checkpoint_sha256'],
                    'weight_dtype': 'torch.float32', 'compute_dtype': 'torch.bfloat16'}
            if method == 'initial_state_checkpoint':
                return {'enabled': True, 'pool': {'count': 50}, 'scheduler': {'max_env_steps': 11000}}
            return original(actor, method, values, keywords)
        module = types.ModuleType('checkpoint_evaluation')
        module.preserve_initial_checkpoint = mock.Mock()
        with mock.patch.object(Actor, 'call', call), mock.patch.dict('sys.modules', {'checkpoint_evaluation': module}):
            state = execute_definition(startup_ast(), context.namespace())(args)
        return context, state

    def test_actual_main_removes_world_trainers_but_keeps_four_world_inference(self):
        context, state = self.smoke()
        self.assertEqual(set(state['training_groups']), {'policy', 'reward'})
        self.assertEqual(len(state['all_trainers']), 8)
        counts = Counter(actor.name for actor in context.actors)
        self.assertEqual(counts['TrainerActor'], 8)
        self.assertEqual(counts['WeaverWorldInferenceActor'], 4)
        self.assertEqual(counts['CtrlWorldInferenceActor'], 0)
        self.assertEqual(len([actor for actor in context.actors if actor.kind is not None]), 20)
        self.assertEqual(len([call for call in context.calls if call[2] == 'setup_deepspeed_group']), 8)
        self.assertFalse(any(call[2] == 'apply_rank_weights' and call[3][-1] == 'ctrl' for call in context.calls))
        self.assertEqual([event[1]['details']['kind'] for event in state['experiment'].events
                          if event[0] == 'weight_publication_committed'], ['policy', 'reward'])
        self.assertEqual(set(context.snapshots[0][0]), {'policy', 'reward'})
        identity = context.snapshots[0][2]['metadata']['frozen_world']
        self.assertEqual(set(identity), {'backend', 'world_trainable', 'fixed_step', 'bundle_sha256'})
        self.assertEqual(identity['fixed_step'], 3200)

    def test_actual_main_retains_one_real_four_imagination_and_only_reward_cohort(self):
        context, state = self.smoke()
        coordinator = next(actor for actor in context.actors if actor.name == 'GlobalBatchCoordinator')
        self.assertEqual(coordinator.kwargs['wm_consumers'], ('reward',))
        plan = state['service_plan']
        self.assertEqual(plan.expected_wm_consumers, 4)
        self.assertEqual(plan.logical_update_ratio, '5:0:1')
        producers = [actor for actor in context.actors if actor.name == 'RolloutWorkerActor']
        self.assertEqual(Counter(actor.kwargs['worker_mode'] for actor in producers), {'real': 1, 'imagine': 4})
        self.assertTrue(all(actor.kwargs['pipeline_config']['real_env_step_budget'] == 11000 for actor in producers))
        self.assertEqual(state['geometry']['required_context_frames'], 32)
        self.assertEqual(state['geometry']['training_window_frames'], 40)
        self.assertEqual(state['wm_min_buffer_size'], 1024)

    def test_actual_epoch_merge_accepts_frozen_world_and_policy_reward_progress(self):
        merge = execute_definition(definition(TREE, 'merge_split_epoch_results'), {})
        for step in (5, 6, 10, 15, 16):
            policy = [(0, 0, 0, 0, 0, {}, step, 0, 0, {}, {}) for _ in range(4)]
            reward = [{'kind': 'reward', 'rank': rank, 'policy_step': step, 'model_step': step // 5,
                'metrics': {'trained_ctrl': False, 'trained_reward': step % 5 == 0,
                            'ctrl_sample_unique_episodes': 3, 'ctrl_sample_fresh_episode_ratio': .5},
                'performance': {'wm_sample_time': 0, 'wm_prep_time': 0, 'wm_batch_wait_time': 0},
                'wm_cohort': {'sample_ids_sha256': str(rank)} if step % 5 == 0 else None,
                'learning_rates': {'reward': 1e-5}} for rank in range(4)]
            result = merge(policy + reward, 4, step, 5, 5, ('policy', 'reward'))
            self.assertEqual([item[6] for item in result], [step] * 4)
            self.assertTrue(all(not item[10]['trained_ctrl'] for item in result))
            self.assertTrue(all(item[10]['trained_reward'] == (step % 5 == 0) for item in result))

    def test_manual_policy_does_not_stop_at_old15_or_85_or_lr_horizon(self):
        policy = TrainingStopPolicy.from_environment({'ACCERL_RUN_UNTIL_MANUAL_STOP': '1',
            'ACCERL_RUN_UNTIL_DEADLINE': '0', 'ACCERL_STOP_ON_TARGET_85': '0'})
        for step in (15, 25, 30000, 30001):
            self.assertTrue(policy.should_continue(global_step=step, train_iters=30000,
                target_reached=True, requested_stop=False, completed_epochs=step, measure_stop_epochs=15))


class WireRegression(unittest.TestCase):
    def test_torch27_binary_and_text_uuid_forms_are_identical(self):
        import uuid
        from weaver_gpu_worker import canonical_uuid
        target = 'c43d6671-0979-b38d-9eae-4fe0a9d71c8c'
        self.assertEqual(canonical_uuid(uuid.UUID(target).bytes), 'GPU-' + target)
        self.assertEqual(canonical_uuid(target), 'GPU-' + target)
        self.assertEqual(canonical_uuid('GPU-' + target), 'GPU-' + target)

    def test_native_adapter_freezes_every_parameter_and_rejects_online_learning(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Torch unavailable in this CPU interpreter')
        from test_weaver_backend_cpu import TinyUpstream, config
        from world_backends.weaver import WeaverBackend
        cfg = config()
        cfg['world_trainable'] = False
        cfg['backend_options']['autocast_dtype'] = 'bfloat16'
        backend = WeaverBackend(cfg, model=TinyUpstream())
        self.assertTrue(all(not parameter.requires_grad for parameter in backend.model.parameters()))
        self.assertEqual(backend.dtype, torch.float32)
        self.assertEqual(backend.compute_dtype, torch.bfloat16)
        with self.assertRaisesRegex(ValueError, 'Frozen WEAVER'):
            backend.prepare_train_batch([])
        with self.assertRaisesRegex(ValueError, 'Frozen WEAVER'):
            backend.compute_losses({})
        with self.assertRaisesRegex(ValueError, 'Frozen WEAVER'):
            WeaverBackend.load('not-read', 'not-read', config=cfg, for_training=True)

    def test_numpy_frames_are_bounded_and_roundtrip(self):
        import numpy as np
        stream = io.BytesIO()
        payload = {'actions': np.arange(56, dtype=np.float32).reshape(8, 7), 'state': {'version': 0}}
        send_frame(stream, payload)
        stream.seek(0)
        actual = receive_frame(stream)
        np.testing.assert_array_equal(actual['actions'], payload['actions'])
        self.assertEqual(actual['state'], payload['state'])

    def test_partial_pipe_writes_are_completed(self):
        class PartialWriter(io.BytesIO):
            def write(self, value):
                return super().write(value[:3])
        stream = PartialWriter()
        send_frame(stream, {'value': 'large-enough-for-many-short-writes'})
        stream.seek(0)
        self.assertEqual(receive_frame(stream), {'value': 'large-enough-for-many-short-writes'})

    def test_frozen_context_requires32_frames_and_pipeline_excludes_terminal_seed(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Torch unavailable in this CPU interpreter')
        from test_weaver_backend_cpu import TinyUpstream, config, sample
        from world_backends.weaver import WeaverBackend
        from pipeline_budget import uniform_seed_starts
        cfg = config(memory=6)
        cfg['world_trainable'] = False
        cfg['schema']['t_memory'] = 5
        cfg['checkpoint_metadata']['schema'] = copy.deepcopy(cfg['schema'])
        model = TinyUpstream(memory=6)
        model._t_memory = 5
        backend = WeaverBackend(cfg, model=model)
        self.assertEqual(backend.required_context_frames, 32)
        with self.assertRaisesRegex(ValueError, 'Need >= 32'):
            backend.init_rollout(sample(cfg, 31), 0)
        state = backend.init_rollout(sample(cfg, 32), 0)
        self.assertEqual(state['latents']['states'].shape[1], 32)
        self.assertEqual(uniform_seed_starts(32, 32, 16), [])
        self.assertEqual(uniform_seed_starts(33, 32, 16), [0])

    def test_frozen_reward11_window_remains224_and_four_claims_release(self):
        import asyncio
        import numpy as np
        from episode_frame_store import EpisodeRecord
        from test_pipeline_integration_cpu import actor, payload, AsyncActor
        from training_topology import GlobalBatchCoordinator
        cfg = json.loads((ROOT / 'examples/weaver_frozen_external.json').read_text())
        raw = payload('train-frozen', 20)
        raw['states'] = np.zeros((20, 8), dtype=np.float32)
        record = EpisodeRecord(**raw, backend_config=cfg)
        window = record.window(0, 11)
        self.assertEqual(window.reward_primary.shape, (3, 224, 224))
        self.assertEqual(window.reward_primary.dtype, np.uint8)
        # A true four-Reward cohort has no absent World claimant to wait for.
        owner = actor(capacity=100, frame_budget=100, max_episode_frames=40, minimum_train_windows=4)
        owner.pipeline_world_config = cfg
        owner.context_frames = 32
        owner.window_frames = 11
        lease = owner.acquire_real(0)
        owner.finish_real_episode(lease['token'], raw)
        self.assertEqual(len(owner.buffer), 10)
        self.assertEqual(owner.acquire_imagine(0), None)
        async def consume():
            coordinator = GlobalBatchCoordinator(None, AsyncActor(owner), 4,
                wm_consumers=('reward',), pipeline_enabled=True)
            batches = await asyncio.wait_for(asyncio.gather(*[
                coordinator.sample_wm(rank, 1, 'reward') for rank in range(4)]), 2)
            self.assertEqual(len({row[-1]['capture_meta']['sample_ids'][0] for row in batches}), 4)
            self.assertEqual(owner.pipeline_status()['pending_draws'], 0)
            self.assertFalse(coordinator.claimed['wm'])
        asyncio.run(consume())

    def test_oversize_header_is_rejected_before_reading_body(self):
        import struct
        from weaver_subprocess import MAX_FRAME_BYTES
        with self.assertRaisesRegex(ValueError, 'limit'):
            receive_frame(io.BytesIO(struct.pack('!Q', MAX_FRAME_BYTES + 1)))

    def test_bf16_cpu_latents_keep_exact_values_and_cuda_never_crosses(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Torch unavailable in this CPU interpreter')
        import numpy as np
        original = torch.tensor([.1, .5, -.8], dtype=torch.bfloat16)
        result = cpu_wire({'latents': original})['latents']
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result, original.float().numpy())
        bad = types.SimpleNamespace(device='cuda:0', detach=lambda: None)
        with self.assertRaisesRegex(ValueError, 'only CPU'):
            cpu_wire(bad)

    def test_decoded_outputs_are_local_cpu_torch_but_opaque_state_stays_numpy(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Torch unavailable in this CPU interpreter')
        import numpy as np
        frame = np.zeros((8, 2, 3, 4, 4), dtype=np.float32)
        state = {'latents': np.ones((1, 32, 2), dtype=np.float32)}
        client = types.SimpleNamespace(call=lambda *a: {'observations': frame, 'predicted_states':
            np.zeros((8, 8), dtype=np.float32), 'next_state': state})
        result = WeaverRemoteOperations(client).request({}, np.zeros((8, 7)))
        self.assertIsInstance(result['observations'], torch.Tensor)
        self.assertEqual(str(result['observations'].device), 'cpu')
        # Equivalent contract required by envs.utils.tensor_to_image.
        image = result['observations'][0, 0].clone().cpu().numpy().transpose(1, 2, 0)
        self.assertEqual(image.shape, (4, 4, 3))
        self.assertIsInstance(result['next_state']['latents'], np.ndarray)


class FrozenActorBridge(unittest.IsolatedAsyncioTestCase):
    async def test_actual_actor_constructor_and_request_never_use_bridge_cuda(self):
        import numpy as np
        from weaver_inference_actor import make_weaver_inference_actor
        from capture_hooks import _placement, _receive_wrapper
        config = json.loads((ROOT / 'examples/weaver_frozen_external.json').read_text())
        events = []
        class Mixin:
            def _check_inference_health(self):
                if getattr(self, 'inference_fatal_error', None) is not None:
                    raise self.inference_fatal_error
            def _fail_pending_requests(self, error):
                self.inference_fatal_error = error
        class Communication:
            def __init__(self):
                raise AssertionError('Frozen CPU bridge initialized legacy GPU communication')
        class Client:
            def __init__(self, raw, actor_id):
                self.receipt = {'actor_id': actor_id, 'gpu': {'uuid': 'test-only'},
                    'geometry': {'required_context_frames': 32, 'training_window_frames': 40, 'horizon': 8},
                    'world_trainable': False, 'fixed_step': 3200}
            def call(self, method, *args):
                if method == 'init_rollout':
                    return {'backend': 'weaver', 'version': 0, 'instruction': 'task'}
                return {'observations': np.zeros((len(args[1]), 2, 3, 4, 4), np.float32),
                        'predicted_states': np.zeros((len(args[1]), 8), np.float32),
                        'next_state': args[0], 'served_version': 0}
        def no_cuda(*a, **k):
            raise AssertionError('CPU bridge attempted CUDA initialization')
        runtime = {'setup_debugger': lambda *a: None, 'verify_bound_gpu': no_cuda,
            '_weaver_process_client': Client,
            'batch_begin': lambda *a: {}, 'batch_end': lambda *a, **k: None,
            'stage_begin': lambda *a, **k: events.append(k) or {},
            'stage_end': lambda *a, **k: None, '_trim_cuda_cache': no_cuda}
        cls = make_weaver_inference_actor(Mixin, Communication, runtime)
        actor = cls(0, types.SimpleNamespace(add_timing_metric=types.SimpleNamespace(remote=lambda *a: None)),
                    types.SimpleNamespace(world_backend_config=config), 8, 300)
        actor.telemetry = types.SimpleNamespace(event=lambda kind, **k: events.append((kind, k)), flush=lambda: None)
        actor.capture_role, actor.capture_seed = 'ctrl_infer', 123
        try:
            state = actor.init_rollout({})
            result = await actor.request(state, np.zeros((8, 7), np.float32), 'task')
            self.assertEqual(tuple(result['observations'].shape), (8, 2, 3, 4, 4))
            self.assertTrue(actor.cpu_gpu_bridge)
            self.assertFalse(events[0]['cuda'])
            _placement(actor)
            self.assertEqual(events[-1][0], 'cpu_bridge_placement')
            with self.assertRaisesRegex(RuntimeError, 'Frozen WEAVER'):
                actor.apply_rank_weights({}, 'ctrl')
            wrapped = _receive_wrapper(cls.receive_and_update_weights)
            with self.assertRaisesRegex(RuntimeError, 'Frozen WEAVER'):
                wrapped(actor, 'broadcast_ctrl')
        finally:
            actor._bg_task.cancel()
            await asyncio.gather(actor._bg_task, return_exceptions=True)


if __name__ == '__main__':
    unittest.main()
