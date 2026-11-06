"""End-to-end optimized adapter contracts on CPU, never a GPU speed claim.

The injected tiny model runs the actual pinned WEAVER.forward and original
variable-horizon wrapper. Its cached-generation alias tests adapter dispatch
only: test_weaver_cached_inference_cpu.py separately tests the real native KV
implementation and SD3 decoder. Set WEAVER_TEST_UPSTREAM explicitly to the
audited checkout before importing this module.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
import unittest

import numpy as np
import torch

if not os.environ.get('WEAVER_TEST_UPSTREAM'):
    raise RuntimeError('Set WEAVER_TEST_UPSTREAM to the audited pinned checkout')
if not (Path(os.environ['WEAVER_TEST_UPSTREAM']) / 'weaver/wm/model.py').is_file():
    raise RuntimeError('WEAVER_TEST_UPSTREAM does not contain weaver/wm/model.py')

from test_weaver_backend_cpu import Encoder, Head, TinyUpstream, config, sample
from weaver_online_engine import WeaverOnlineEngine
from world_backends.weaver import WeaverBackend
from world_backends.runtime import WeaverActorOperations
from world_backends.rollout import consume_predicted_chunk


NATIVE_SPEC = {'mode': 'libero_success_score_v1', 'link': 'sigmoid', 'loss': 'brier',
    'label_source': 'libero_env_step', 'calibrated': False, 'terminal': 'horizon_only',
    'balance': 'global_cohort_inverse_frequency', 'loss_coefficient': 1.0}


class CountingEncoder(Encoder):
    def __init__(self):
        super().__init__()
        self.encoded_images = 0
        self.decoded_images = 0
        self.inputs = []

    def forward(self, value):
        self.encoded_images += value.shape[0] * value.shape[1]
        self.inputs.append(value.detach().clone())
        return super().forward(value)

    def decode(self, value):
        self.decoded_images += value.shape[0] * value.shape[1]
        return super().decode(value)


class FrozenTaskEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('scale', torch.tensor(0.001))
        self.calls = 0

    def forward(self, texts):
        self.calls += 1
        return torch.tensor([[len(text), sum(map(ord, text))] for text in texts],
                            dtype=torch.float32) * self.scale


class StateAwareHead(Head):
    def forward(self, obs, actions, tasks):
        # Nonconstant future scores expose accidental one-step truncation.
        return (self.bias + obs['states'].mean(-1) * .01 + actions.mean(-1) * .03
                + tasks[:, :1] * .02)


class GeometryUpstream(TinyUpstream):
    """Same real geometry, tiny feature widths; deterministic frozen encoders."""
    def __init__(self):
        super().__init__(memory=6)
        self._n_horizon, self._t_memory = 8, 5
        self._history_noise_std = .05
        self.im_encoder = CountingEncoder()
        self.task_encoder = FrozenTaskEncoder()
        self.rm = StateAwareHead()
        self.cached_calls = 0

    def encode_task(self, tasks):
        if 'features' in tasks:
            return tasks['features']
        return self.task_encoder(tasks['text'])

    def sample_noise(self, values):
        return {key: torch.randn_like(value) for key, value in values.items()}

    def sample_timestep(self, batch, frames):
        return torch.rand(batch, frames)

    def interpolate(self, x1, x0, times):
        return {key: value * times.reshape(*times.shape, *([1] * (value.ndim - 2)))
                + x0[key] * (1 - times.reshape(*times.shape, *([1] * (value.ndim - 2))))
                for key, value in x1.items()}

    def compute_flow_loss(self, x1, x0, prediction, times):
        return (prediction - (x1 - x0)).square()

    def generate_latent_rollouts_cached(self, x1, actions, memory_tokens=None):
        # Dispatch fixture only, deliberately not advertised as real KV math.
        self.cached_calls += 1
        return self.generate_latent_rollouts(x1, actions, memory_tokens)


def configured(*, selected=False, cpu=False, cache=False, kv=False, decode='all', resident=False):
    cfg = config(memory=6)
    cfg.update(reward_provider='world_head', reward_semantics='delta_native_score',
               terminal_rule='horizon_only', world_trainable=True)
    cfg['schema']['t_memory'] = 5
    cfg['checkpoint_metadata']['schema'] = copy.deepcopy(cfg['schema'])
    cfg['checkpoint_metadata'].update(success_head={'calibrated': False, 'trained': False},
                                      terminal_head={'calibrated': False, 'trained': False})
    cfg['checkpoint_metadata']['load_report'].update(tensor_count=866,
        checkpoint_sha256='a' * 64, shape_mismatches=[])
    cfg['backend_options'].update(initial_checkpoint_sha256='a' * 64,
        train_reward_head=True, native_reward_head=copy.deepcopy(NATIVE_SPEC), horizon=8,
        encode_chunk_size=4, decode_chunk_size=2, autocast_dtype='bfloat16',
        optimizations={'selected_frame_encoding': selected, 'validate_on_cpu': cpu,
            'feature_cache_max_bytes': 4 * 1024 * 1024 if cache else 0,
            'kv_cache': kv, 'decode_mode': decode, 'resident_rollout_state': resident,
            'task_cache_entries': 4 if cache else 0})
    return cfg


def rows_for(cfg, frames=40, *, floating=False, dimensions=(7, 8), count=2):
    rows = []
    for index in range(count):
        row = sample(cfg, frames)
        height, width = dimensions
        pixels = (np.arange(frames * 2 * 3 * height * width, dtype=np.int64)
                  .reshape(frames, 2, 3, height, width) * 17 + index * 29) % 256
        row['obs'] = pixels.astype(np.uint8)
        if floating:
            row['obs'] = torch.from_numpy(row['obs']).float().div(255).mul(2).sub(1).numpy()
        row['states'] = (np.arange(frames * 8, dtype=np.float32).reshape(frames, 8)
                         * .013 + 2 + index * .17)
        row['actions'] = (np.arange(frames * 7, dtype=np.float32).reshape(frames, 7)
                          * .003 - .7 + index * .11)
        row['success'] = np.zeros(frames, dtype=np.float32)
        row['success'][-1] = 1
        row['success'][-4] = index % 2
        row['label_source'] = 'libero_env_step'
        row['instruction'] = 'pick up the cube' if index == 0 else 'place the bowl'
        row['episode_id'] = 'cpu-episode-%d' % index
        row['sample_id'] = row['episode_id'] + (':window7' if frames == 40 else ':seed7')
        row['frame_start'] = 7
        row['request_lineage'] = {'episode_id': row['episode_id'], 'request_id': 'q%d' % index}
        rows.append(row)
    return rows


def make_backend(cfg):
    upstream = GeometryUpstream()
    backend = WeaverBackend(cfg, model=upstream)
    return upstream, backend


def prepare(backend, rows):
    # Exercise the exact production prepare_micro entry point without creating
    # an optimizer: optimizer stepping is unrelated to this equivalence test.
    engine = WeaverOnlineEngine.__new__(WeaverOnlineEngine)
    engine.torch, engine.backend = torch, backend
    engine.micro_batch, engine.device = len(rows), torch.device('cpu')
    return engine.prepare_micro(rows)


def set_cohort(backend, rows):
    positives = sum(int(np.asarray(row['success'])[30:40].sum()) for row in rows)
    negatives = len(rows) * 10 - positives
    backend.native_reward_controller.set_cohort({'protocol': 'weaver_reward_cohort_v1',
        'window_count': len(rows), 'label_frames_per_window': 10,
        'positive_count': positives, 'negative_count': negatives,
        'global_selected_frames': positives + negatives, 'positive_weight': negatives / positives,
        'negative_weight': 1., 'rm_active': True, 'label_source': 'libero_env_step',
        'frame_slice': [30, 40]})


class OptimizedBackendCPU(unittest.TestCase):
    def assert_tree_equal(self, first, second, path='root'):
        if isinstance(first, torch.Tensor):
            self.assertIsInstance(second, torch.Tensor, path)
            self.assertEqual(first.dtype, second.dtype, path)
            torch.testing.assert_close(first, second, rtol=0, atol=0, msg=path)
        elif isinstance(first, dict):
            self.assertEqual(set(first), set(second), path)
            for key in first:
                self.assert_tree_equal(first[key], second[key], path + '.' + key)
        elif isinstance(first, (tuple, list)):
            self.assertEqual(len(first), len(second), path)
            for index, (left, right) in enumerate(zip(first, second)):
                self.assert_tree_equal(left, right, path + '[%d]' % index)
        else:
            self.assertEqual(first, second, path)

    def forward_snapshot(self, backend, prepared):
        backend.model.train()
        backend.model.zero_grad(set_to_none=True)
        before = {key: value.detach().clone() for key, value in backend.model.state_dict().items()}
        torch.manual_seed(92631)
        rng_before = torch.get_rng_state().clone()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            loss, logs = backend.model(prepared)
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        gradients = {}
        for name, parameter in backend.model.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                gradients[name] = parameter.grad.detach().clone()
            else:
                self.assertIsNone(parameter.grad, name)
        self.assertEqual(set(gradients), {'upstream.wm.gain', 'upstream.rm.bias'})
        self.assertFalse(torch.equal(rng_before, torch.get_rng_state()), 'Fixture must consume real RNG')
        self.assert_tree_equal(before, backend.model.state_dict(), 'forward mutated weights')
        return {'loss': loss.detach(), 'logs': logs, 'gradients': gradients,
                'rng': torch.get_rng_state().clone()}

    def test_full40_selected16_cold_and_warm_match_native_loss_all_gradients_and_rng(self):
        old_cfg = configured()
        rows = rows_for(old_cfg)
        old_model, old = make_backend(old_cfg)
        self.assertEqual((old.required_context_frames, old.training_window_frames), (32, 40))
        set_cohort(old, rows)
        torch.manual_seed(317)
        before = torch.get_rng_state().clone()
        reference = prepare(old, rows)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(old_model.im_encoder.encoded_images, 2 * 40 * 2)
        expected_state = (torch.from_numpy(rows[0]['states']) - 2) / 2
        torch.testing.assert_close(reference['obs']['states'][0], expected_state[30:40], rtol=0, atol=0)
        torch.testing.assert_close(reference['memory']['states'][0],
                                  expected_state[[1, 6, 11, 16, 21, 26]], rtol=0, atol=0)
        self.assertEqual(tuple(reference['actions'].shape), (2, 10, 7))
        self.assertEqual(tuple(reference['memory']['external_features'].shape), (2, 6, 16, 3))
        self.assertEqual(float(reference['gt_rewards'].sum()), 3.)
        expected = self.forward_snapshot(old, reference)
        for selected, cache in ((False, False), (True, False), (True, True)):
            with self.subTest(selected=selected, cache=cache):
                cfg = configured(selected=selected, cpu=True, cache=cache)
                model, backend = make_backend(cfg)
                set_cohort(backend, rows)
                for attempt in range(2):
                    before = torch.get_rng_state().clone()
                    prepared = prepare(backend, rows)
                    self.assertTrue(torch.equal(before, torch.get_rng_state()))
                    self.assert_tree_equal(reference, prepared)
                    self.assert_tree_equal(expected, self.forward_snapshot(backend, prepared))
                    encoded_per_pass = 2 * (16 if selected else 40) * 2
                    self.assertEqual(model.im_encoder.encoded_images,
                                     encoded_per_pass * (1 if cache else attempt + 1))
                if cache:
                    report = backend._feature_cache.report()
                    self.assertEqual((report['entries'], report['hits'], report['misses']), (64, 64, 64))
                    self.assertLessEqual(report['resident_bytes'], report['max_bytes'])

    def test_float_compatibility_has_exact_pixel_view_order_but_cannot_hit_uint8_cache(self):
        cfg = configured(selected=True, cpu=True, cache=True)
        model, backend = make_backend(cfg)
        raw = rows_for(cfg)
        reference = prepare(make_backend(configured())[1], raw)
        self.assert_tree_equal(reference, prepare(backend, raw))
        initial_report = backend._feature_cache.report()
        floating = rows_for(cfg, floating=True)
        self.assert_tree_equal(reference, prepare(backend, floating))
        self.assert_tree_equal(reference, prepare(backend, floating))
        self.assertEqual(backend._feature_cache.report(), initial_report)
        self.assertEqual(model.im_encoder.encoded_images, 3 * 64)
        # Full uint8 domain and distinct views are present before the exact old
        # /255 -> *2-1 -> +1 -> /2 -> bilinear preprocessing sequence.
        self.assertEqual(len(np.unique(raw[0]['obs'])), 256)
        self.assertFalse(torch.equal(reference['obs']['external_features'], reference['obs']['wrist_features']))

    def test_entire_unused_window_is_validated_before_any_cache_hit_or_encode(self):
        cfg = configured(selected=True, cpu=True, cache=True)
        model, backend = make_backend(cfg)
        raw = rows_for(cfg)
        prepare(backend, raw)
        encoded_before = model.im_encoder.encoded_images
        cache_before = backend._feature_cache.report()
        for field in ('obs', 'states', 'actions', 'dt'):
            with self.subTest(field=field):
                invalid = copy.deepcopy(raw)
                if field == 'obs':
                    invalid[0][field] = torch.from_numpy(invalid[0][field]).float().div(255).mul(2).sub(1).numpy()
                value = np.asarray(invalid[0][field], dtype=np.float32).copy()
                value.reshape(value.shape[0], -1)[0, 0] = np.nan  # frame 0 is not selected
                invalid[0][field] = value
                with self.assertRaises(ValueError):
                    prepare(backend, invalid)
                self.assertEqual(model.im_encoder.encoded_images, encoded_before)
                self.assertEqual(backend._feature_cache.report(), cache_before)
        for field in ('valid_mask', 'action_mask'):
            invalid = copy.deepcopy(raw)
            invalid[0][field][0] = False
            with self.subTest(field=field), self.assertRaises(ValueError):
                prepare(backend, invalid)
        invalid = copy.deepcopy(raw)
        invalid[0]['terminated'][0] = True
        with self.assertRaises(ValueError):
            prepare(backend, invalid)

    def test_warm_cache_rejects_changed_retained_frame_and_changed_encoder(self):
        cfg = configured(selected=True, cpu=True, cache=True)
        model, backend = make_backend(cfg)
        raw = rows_for(cfg)
        prepare(backend, raw)
        changed = copy.deepcopy(raw)
        changed[0]['obs'][1, 0, 0, 0, 0] ^= np.uint8(1)
        with self.assertRaisesRegex(ValueError, 'changed pixel content'):
            prepare(backend, changed)
        with torch.no_grad():
            model.im_encoder.scale.add_(.1)
        with self.assertRaisesRegex(ValueError, 'changed'):
            prepare(backend, raw)

    def test_h1_h4_h8_kv_dispatch_sparse_wire_keeps_all_state_scores_and_consumer_steps(self):
        for horizon in (1, 4, 8):
            with self.subTest(horizon=horizon):
                old_cfg = configured()
                cfg = configured(selected=True, cpu=True, cache=True, kv=True, decode='last', resident=True)
                old_model, old = make_backend(old_cfg)
                model, backend = make_backend(cfg)
                old_ops, ops = WeaverActorOperations(old, old_cfg, 4), WeaverActorOperations(backend, cfg, 4)
                seed = rows_for(cfg, frames=32, count=1)[0]
                old_seed = copy.deepcopy(seed)
                old_seed['obs'] = torch.from_numpy(seed['obs']).float().div(255).mul(2).sub(1)
                old_state = old_ops.init_rollout(old_seed)
                state = ops.init_resident_rollout(seed)
                self.assert_tree_equal(old_state, state)
                self.assertEqual(model.im_encoder.encoded_images, 64)
                self.assertEqual(model.task_encoder.calls, 1)
                repeated = ops.init_resident_rollout(seed)
                self.assert_tree_equal(state, repeated)
                self.assertEqual(model.im_encoder.encoded_images, 64)
                self.assertEqual(model.task_encoder.calls, 1)
                actions = torch.arange(horizon * 7, dtype=torch.float32).reshape(horizon, 7) * .03 - .5
                torch.manual_seed(812)
                before = torch.get_rng_state().clone()
                expected = old_ops.request(old_state, actions)
                expected_rng = torch.get_rng_state().clone()
                torch.set_rng_state(before)
                actual = ops.request_wire(state, actions)
                self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
                self.assertEqual(model.cached_calls, 1)
                self.assertEqual(old_model.cached_calls, 0)
                for name in ('predicted_states', 'reward_predictions', 'terminal_predictions',
                             'valid_mask', 'next_state', 'dt', 'request_lineage'):
                    self.assert_tree_equal(expected[name], actual[name], name)
                self.assertEqual(tuple(actual['observations'].shape), (1, 2, 3, 4, 4))
                self.assert_tree_equal(expected['observations'][-1:], actual['observations'])
                self.assertEqual(actual['observation_indices'], [horizon - 1])
                self.assertEqual(actual['prediction_horizon'], horizon)
                self.assertEqual(len(actual['predicted_states']), horizon)
                self.assertEqual(len(actual['reward_predictions']), horizon)
                self.assertEqual(actual['next_state']['predicted_steps'], horizon)
                self.assertEqual(tuple(actual['next_state']['latents']['states'].shape), (1, 32, 8))
                self.assertEqual(model.task_encoder.calls, 1)
                self.assertEqual(model.im_encoder.decoded_images, 2 * min(horizon, 2))
                self.assertEqual(old_model.im_encoder.decoded_images, 2 * horizon)
                explicit_full = backend.predict_chunk(state, actions, published_version=4,
                    requested_outputs=['all_observations'])
                self.assertEqual(tuple(explicit_full['observations'].shape), (1, horizon, 2, 3, 4, 4))
                self.assertNotIn('observation_indices', explicit_full)
                self.assertNotIn('prediction_horizon', explicit_full)
                self.assert_tree_equal(expected['observations'], explicit_full['observations'][0])
                self.assert_tree_equal(actual['next_state'], explicit_full['next_state'])
                self.assert_tree_equal(actual['reward_predictions'], explicit_full['reward_predictions'][0])
                self.assert_tree_equal(actual['predicted_states'], explicit_full['predicted_states'][0])
                consumed = consume_predicted_chunk(actual, state['initial_reward_score'],
                    reward_scale=2., reward_semantics='delta_native_score')
                expected_consumed = consume_predicted_chunk(expected, old_state['initial_reward_score'],
                    reward_scale=2., reward_semantics='delta_native_score')
                self.assert_tree_equal(expected_consumed, consumed)
                self.assertEqual(consumed[-1], horizon)
                self.assertFalse(consumed[2])
                if horizon > 1:
                    self.assertNotEqual(float(actual['reward_predictions'][0]),
                                        float(actual['reward_predictions'][-1]))
                # Cached memory, actions and every context latent enter both
                # variable-horizon paths in exactly the same order and shape.
                self.assert_tree_equal(old_model.generate_calls[-1], model.generate_calls[-1])
                ops.set_published_version(5)
                with self.assertRaisesRegex(ValueError, 'Stale'):
                    ops.request_wire(state, actions)
                fresh = ops.init_resident_rollout(seed)
                self.assertEqual(fresh['version'], 5)
                self.assertEqual(model.im_encoder.encoded_images, 64)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
