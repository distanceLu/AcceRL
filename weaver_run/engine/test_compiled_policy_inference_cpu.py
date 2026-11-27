import ast
import asyncio
from contextlib import suppress
from pathlib import Path
import time
import unittest
from types import SimpleNamespace
from typing import List, Tuple
from unittest.mock import Mock, patch

import numpy as np
import torch

from compiled_policy_inference import CompiledPolicyInference, static_action_outputs, unpadded_action_layout


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.))

    def forward(self, batch):
        score = batch['pixel_values'].sum(dim=1) * self.weight
        return score.unsqueeze(1), score


class FakeGraph:
    def __init__(self, model):
        self.model = model

    def replay(self, batch):
        return self.model(batch)


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.model = TinyPolicy().eval()
        self.events = []
        self.path = CompiledPolicyInference(self.model, lambda k, d: self.events.append((k, d)))
        self.batch = {'input_ids': torch.ones(1, 3, dtype=torch.long), 'pixel_values': torch.ones(1, 2),
                      'attention_mask': torch.ones(1, 3), 'labels': torch.ones(1, 3)}

    def test_cpu_eager_fallback_is_counted(self):
        with torch.inference_mode():
            self.assertEqual(self.path(self.batch, 0)[1].item(), 4.)
        self.assertEqual(sum(self.path.summary()['fallback_counts'].values()), 1)
        self.assertEqual(self.path.summary()['graph_replays'], 0)

    def test_in_place_publications_keep_identity_and_refresh_checks(self):
        policy_param = self.model.weight
        rng = torch.get_rng_state().clone()
        with patch.object(self.path, '_unsupported', return_value=None), \
             patch.object(self.path, '_capture', return_value=FakeGraph(self.model)) as capture, torch.inference_mode():
            for version in (0, 0, 1, 2, 3, 32):
                self.model.weight.fill_(version + 1)
                self.assertEqual(self.path(self.batch, version)[1].item(), 2 * (version + 1))
        capture.assert_called_once()
        self.assertIs(self.model.weight, policy_param)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(self.path.summary()['validated_versions'], [0, 1, 32])
        self.assertEqual(self.path.summary()['graph_replays'], 6)

    def test_mismatched_graph_never_serves_actions(self):
        wrong = FakeGraph(TinyPolicy().eval())
        wrong.model.weight.data.fill_(-3.)
        with patch.object(self.path, '_unsupported', return_value=None), \
             patch.object(self.path, '_capture', return_value=wrong), torch.inference_mode():
            self.assertEqual(self.path(self.batch, 0)[1].item(), 4.)
            self.assertEqual(self.path(self.batch, 0)[1].item(), 4.)
        self.assertEqual(self.path.summary()['graph_replays'], 0)
        self.assertFalse(next(d for k, d in self.events if k == 'compiled_policy_validation')['matched'])

    def test_batched_request_always_uses_original_forward(self):
        batch = {k: v.repeat(2, 1) for k, v in self.batch.items()}
        with torch.inference_mode():
            a, b = self.path(batch, 0)
        self.assertTrue(torch.equal(a, self.model(batch)[0]))
        self.assertEqual(self.path.summary()['fallback_counts'], {'batch_not_one': 1})

    def test_changed_parameter_storage_is_recaptured(self):
        with patch.object(self.path, '_unsupported', return_value=None), \
             patch.object(self.path, '_capture', side_effect=lambda b: FakeGraph(self.model)) as capture, \
             torch.inference_mode():
            self.path(self.batch, 0)
            self.model.weight = torch.nn.Parameter(torch.tensor(4.))
            self.assertEqual(self.path(self.batch, 1)[1].item(), 8.)
        self.assertEqual(capture.call_count, 2)
        self.assertIn('compiled_policy_storage_changed', [name for name, _ in self.events])

    def test_capture_unsupported_can_calibrate_partial_scope(self):
        with patch.object(self.path, '_unsupported', return_value=None), \
             patch.object(self.path, '_capture', side_effect=[RuntimeError('unsupported mask'),
                          RuntimeError('unsupported full forward'), FakeGraph(self.model)]), \
             torch.inference_mode():
            self.path(self.batch, 0)
        report = self.path.summary()
        self.assertEqual(report['capture_failures'], 2)
        self.assertEqual(report['replays_by_scope'], {'vision_projector': 1})

    def test_rng_using_candidate_is_rejected_without_consuming_action_rng(self):
        class RandomGraph(FakeGraph):
            def replay(self, batch):
                torch.rand(1)
                return super().replay(batch)

        before = torch.get_rng_state().clone()
        with patch.object(self.path, '_unsupported', return_value=None), \
             patch.object(self.path, '_capture', return_value=RandomGraph(self.model)), torch.inference_mode():
            self.assertEqual(self.path(self.batch, 0)[1].item(), 4.)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(self.path.summary()['graph_replays'], 0)

    @unittest.skipUnless(torch.cuda.is_available(), 'requires a CUDA device')
    def test_cuda_capture_replays_fresh_input_and_published_weights(self):
        free, total = torch.cuda.mem_get_info()
        if free < max(8 * 2**30, int(.15 * total)) + 2**30:
            self.skipTest('insufficient free memory for the production graph reserve')

        class TinyCudaPolicy(TinyPolicy):
            cfg = SimpleNamespace(use_proprio=False, use_film=False, use_diffusion=False, use_l1_regression=False)
            vla = SimpleNamespace(_process_action_masks=lambda labels: labels > 0)
            real_action_count = 1
            real_action_token_begin, real_ignore_index, real_stop_index = 7, -100, 2

            def real_action_vision(self, pixels):
                return pixels * self.weight

            def real_action_from_patches(self, ids, patches, attn):
                score = patches.sum(1)
                return score.unsqueeze(1), score

        model = TinyCudaPolicy().cuda().eval()
        self.batch['labels'] = torch.tensor([[-100, 8, 2]])
        self.batch['attention_mask'] = torch.ones(1, 3, dtype=torch.long)
        batch = {k: v.cuda() for k, v in self.batch.items()}
        path = CompiledPolicyInference(model, self.path.emit)
        with torch.inference_mode():
            first = path(batch, 0, original=self.batch)
            model.weight.fill_(3.)
            batch['pixel_values'].fill_(2.)
            second = path(batch, 1, original=self.batch)
            self.assertEqual((first[1].item(), second[1].item()), (4., 12.))
        self.assertEqual(path.summary()['replays_by_scope'], {'full_logits_value': 2})

    def test_action_positions_match_shifted_labels_on_admitted_cpu_layout(self):
        patches = 3
        labels = torch.tensor([[-100, -100, -100, 8, 9, 11, 2]])
        mask = labels > 7
        all_logits = torch.arange(20.).reshape(1, 10, 2)
        hidden = torch.arange(30.).reshape(1, 10, 3)
        cfg = SimpleNamespace(real_action_count=3, real_ignore_index=-100, real_action_token_begin=7, real_stop_index=2)
        source = dict(labels=labels, attention_mask=torch.ones_like(labels))
        self.assertTrue(unpadded_action_layout(source, cfg, labels.shape))
        actual_logits, value = static_action_outputs(all_logits, hidden, 3,
                                                     lambda x: x.sum(dim=(1, 2)))
        expected_h = hidden[:, patches:-1][mask[:, 1:]]
        self.assertTrue(torch.equal(actual_logits, all_logits[:, patches:-1][mask[:, 1:]].unsqueeze(0)))
        self.assertEqual(value.item(), expected_h.sum().item())
        source['labels'] = torch.tensor([[-100, -100, 8, -100, 9, 11, 2]])
        self.assertFalse(unpadded_action_layout(source, cfg, labels.shape))
        source.update(labels=labels, attention_mask=torch.tensor([[1, 1, 1, 1, 1, 1, 0]]))
        self.assertFalse(unpadded_action_layout(source, cfg, labels.shape))

    def test_production_request_loop_routes_by_workflow_and_samples_once(self):
        # Execute the two production coroutine bodies, without importing Ray or
        # loading an OpenVLA checkpoint in this CPU routing test.
        source = Path(__file__).with_name('train.py')
        actor = next(n for n in ast.parse(source.read_text(encoding='utf8')).body
                     if isinstance(n, ast.ClassDef) and n.name == 'InferenceActor')
        methods = [n for n in actor.body if getattr(n, 'name', '') in ('request', '_loop')]
        def noop(*args, **kwargs):
            pass
        ns = dict(torch=torch, np=np, asyncio=asyncio, time=time, NUM_ACTIONS_CHUNK=8, ACTION_DIM=7,
                  current_request_metadata=Mock(), batch_begin=noop, batch_end=noop,
                  stage_begin=noop, stage_end=noop, _trim_cuda_cache=noop)
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), 'exec'), ns)
        logits = torch.arange(56 * 4, dtype=torch.float32).reshape(1, 56, 4)
        raw = self.batch
        prepared = dict(raw)
        eager_logits = -logits
        model = Mock(return_value=(eager_logits, torch.tensor([5.])))
        model.prepare_inputs_batch.return_value = prepared
        model_source = source.parents[2] / 'rl' / 'actor_critic_model_discrete.py'
        ac = next(n for n in ast.parse(model_source.read_text(encoding='utf8')).body
                  if isinstance(n, ast.ClassDef) and n.name == 'ActorCritic')
        post = next(n for n in ac.body if getattr(n, 'name', '') == 'post_process')
        ns.update(List=List, Tuple=Tuple)
        exec(compile(ast.Module(body=[post], type_ignores=[]), str(model_source), 'exec'), ns)
        model.n_action_bins = 4
        model.bin_centers = np.linspace(-1., 1., 3)
        model.post_process.side_effect = lambda scores, deterministic: ns['post_process'](model, scores, deterministic)
        model.vla._unnormalize_actions.side_effect = lambda a, key: a
        optimized = Mock(return_value=(logits, torch.tensor([3.])))
        owner = SimpleNamespace(_policy_dispatch=None, requests=[], promises=[], batch_size=1,
                                model=model, compiled_actions=optimized, cfg=SimpleNamespace(unnorm_key='test'),
                                stats_actor=Mock(), _check_inference_health=noop, capture_model_version=17)

        async def serve():
            worker = asyncio.create_task(ns['_loop'](owner))
            try:
                return await asyncio.wait_for(ns['request'](owner, raw, deterministic=False), timeout=2)
            finally:
                worker.cancel()
                with suppress(asyncio.CancelledError):
                    await worker

        sample = torch.distributions.Categorical.sample
        for workflow, scores, expected_value in [('real_rollout', logits, 3.), ('evaluation', eager_logits, 5.)]:
            with self.subTest(workflow=workflow):
                model.reset_mock()
                optimized.reset_mock()
                ns['current_request_metadata'].return_value = {'workflow': workflow}
                with patch.object(torch.distributions.Categorical, 'sample', autospec=True, side_effect=sample) as sampler:
                    _, _, actions, stored_logits, value = asyncio.run(serve())
                sampler.assert_called_once()
                self.assertEqual(value, expected_value)
                self.assertTrue(np.array_equal(stored_logits, scores.reshape(8, 7, 4).numpy()))
                self.assertEqual(actions.shape, (8, 7))
                model.post_process.assert_called_once_with(scores, deterministic=[False])
                if workflow == 'real_rollout':
                    optimized.assert_called_once_with(prepared, 17, original=raw)
                    model.assert_not_called()
                else:
                    optimized.assert_not_called()
                    model.assert_called_once_with(prepared)

    def test_compiled_real_rollout_dispatches_immediately_without_timeout_or_b2_coalescing(self):
        source = Path(__file__).with_name('train.py')
        actor = next(n for n in ast.parse(source.read_text(encoding='utf8')).body
                     if isinstance(n, ast.ClassDef) and n.name == 'InferenceActor')
        methods = [n for n in actor.body if getattr(n, 'name', '') in ('request', '_loop')]
        def noop(*args, **kwargs):
            pass
        ns = dict(torch=torch, np=np, asyncio=asyncio, time=time, NUM_ACTIONS_CHUNK=8, ACTION_DIM=7,
                  current_request_metadata=Mock(return_value={'workflow': 'real_rollout'}),
                  batch_begin=noop, batch_end=noop, stage_begin=noop, stage_end=noop, _trim_cuda_cache=noop)
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), 'exec'), ns)

        raw = self.batch
        seen_batches = []
        def prepare_batch(items):
            seen_batches.append(len(items))
            return {'batch_size': len(items)}
        def run_compiled(prepared, version, *, original=None):
            bsz = prepared['batch_size']
            self.assertEqual(bsz, 1)
            self.assertIs(original, raw)
            return torch.zeros(bsz, 56, 4), torch.full((bsz,), 3.0)
        def post_process(scores, deterministic):
            bsz = scores.shape[0]
            return None, torch.zeros(bsz, 56, dtype=torch.long), np.zeros((bsz, 8, 7), dtype=np.float32)

        model = Mock(side_effect=AssertionError('eager model should not run for compiled real_rollout'))
        model.prepare_inputs_batch.side_effect = prepare_batch
        model.post_process.side_effect = post_process
        model.vla._unnormalize_actions.side_effect = lambda a, key: a
        optimized = Mock(side_effect=run_compiled)
        # Production-like settings: batch_size=8, timeout_sec=0.300, training_batch_limit=2,
        # and last_process_time set right now (so timeout_sec has NOT expired!).
        owner = SimpleNamespace(_policy_dispatch=None, requests=[], promises=[], batch_size=8,
                                timeout_sec=0.300, training_batch_limit=2, last_process_time=time.time(),
                                model=model, compiled_actions=optimized, cfg=SimpleNamespace(unnorm_key='test'),
                                stats_actor=Mock(), _check_inference_health=noop, capture_model_version=9)

        async def serve_back_to_back_and_concurrent():
            worker = asyncio.create_task(ns['_loop'](owner))
            try:
                t0 = time.monotonic()
                # 1. Single arrival right after last_process_time must not sleep 300ms.
                await asyncio.wait_for(ns['request'](owner, raw, deterministic=False), timeout=0.15)
                # 2. Second back-to-back arrival immediately after must also not sleep 300ms.
                await asyncio.wait_for(ns['request'](owner, raw, deterministic=False), timeout=0.15)
                # 3. Two simultaneous arrivals must execute as two B=1 compiled calls, not one B=2 fallback.
                r1, r2 = await asyncio.wait_for(
                    asyncio.gather(ns['request'](owner, raw, deterministic=False),
                                   ns['request'](owner, raw, deterministic=False)),
                    timeout=0.15,
                )
                return time.monotonic() - t0, r1[4], r2[4]
            finally:
                worker.cancel()
                with suppress(asyncio.CancelledError):
                    await worker

        elapsed, v1, v2 = asyncio.run(serve_back_to_back_and_concurrent())
        self.assertLess(elapsed, 0.15)
        self.assertEqual((v1, v2), (3.0, 3.0))
        self.assertEqual(seen_batches, [1, 1, 1, 1])
        self.assertEqual(optimized.call_count, 4)

    def test_eager_and_eval_respect_effective_training_batch_limit_without_waiting_for_batch8(self):
        source = Path(__file__).with_name('train.py')
        actor = next(n for n in ast.parse(source.read_text(encoding='utf8')).body
                     if isinstance(n, ast.ClassDef) and n.name == 'InferenceActor')
        methods = [n for n in actor.body if getattr(n, 'name', '') in ('request', '_loop')]
        def noop(*args, **kwargs):
            pass
        ns = dict(torch=torch, np=np, asyncio=asyncio, time=time, NUM_ACTIONS_CHUNK=8, ACTION_DIM=7,
                  current_request_metadata=Mock(return_value={'workflow': 'evaluation'}),
                  batch_begin=noop, batch_end=noop, stage_begin=noop, stage_end=noop, _trim_cuda_cache=noop)
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), 'exec'), ns)

        raw = self.batch
        seen_batches = []
        def prepare_batch(items):
            seen_batches.append(len(items))
            return {'batch_size': len(items)}
        def run_eager(prepared):
            bsz = prepared['batch_size']
            return torch.zeros(bsz, 56, 4), torch.full((bsz,), 5.0)
        def post_process(scores, deterministic):
            bsz = scores.shape[0]
            return None, torch.zeros(bsz, 56, dtype=torch.long), np.zeros((bsz, 8, 7), dtype=np.float32)

        model = Mock(side_effect=run_eager)
        model.prepare_inputs_batch.side_effect = prepare_batch
        model.post_process.side_effect = post_process
        model.vla._unnormalize_actions.side_effect = lambda a, key: a
        owner = SimpleNamespace(_policy_dispatch=None, requests=[], promises=[], batch_size=8,
                                timeout_sec=0.300, training_batch_limit=2, last_process_time=time.time(),
                                model=model, compiled_actions=None, cfg=SimpleNamespace(unnorm_key='test'),
                                stats_actor=Mock(), _check_inference_health=noop, capture_model_version=9)

        async def serve_cases():
            worker = asyncio.create_task(ns['_loop'](owner))
            try:
                # Two concurrent eager requests match training_batch_limit=2 and dispatch immediately (<300ms).
                t0 = time.monotonic()
                await asyncio.wait_for(
                    asyncio.gather(ns['request'](owner, raw, deterministic=True),
                                   ns['request'](owner, raw, deterministic=True)),
                    timeout=0.15,
                )
                t_pair = time.monotonic() - t0
                # With training_batch_limit=1 (periodic fixed validation), one request dispatches immediately.
                owner.training_batch_limit = 1
                owner.last_process_time = time.time()
                t1 = time.monotonic()
                await asyncio.wait_for(ns['request'](owner, raw, deterministic=True), timeout=0.15)
                t_single = time.monotonic() - t1
                return t_pair, t_single
            finally:
                worker.cancel()
                with suppress(asyncio.CancelledError):
                    await worker

        t_pair, t_single = asyncio.run(serve_cases())
        self.assertLess(t_pair, 0.15)
        self.assertLess(t_single, 0.15)
        self.assertEqual(seen_batches, [2, 1])


if __name__ == '__main__':
    unittest.main()
