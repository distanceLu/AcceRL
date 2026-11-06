"""Bounded early-context adapter integration with the existing tiny CPU model.

This executes the adapter init/request path and pinned native horizon helper;
it is not a real-weight, CUDA, native-KV-math or learning-effect validation.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
os.environ.setdefault('WEAVER_TEST_UPSTREAM', str(HERE.parent / 'analysis/weaver_integration_20261022/upstream'))

import torch

from episode_frame_store import EpisodeRecord, materialize_imagination_context
from test_weaver_optimized_backend_cpu import configured, make_backend
from weaver_feature_cache import build_frame_keys
from world_backends.runtime import WeaverActorOperations


def configuration(*, enabled=True, cache=True):
    cfg = configured(selected=True, cpu=True, cache=cache, kv=True,
                     decode='last', resident=True)
    cfg['backend_options']['fresh'] = {'early_seed_context': enabled}
    return cfg


def measured_episode(count=40):
    frames = np.empty((count, 2, 3, 224, 224), np.uint8)
    for index in range(count):
        frames[index, 0].fill(index)
        frames[index, 1].fill(index + 17)
    actions = np.arange(count * 7, dtype=np.float32).reshape(count, 7) * .001
    actions[0] = 0.
    states = np.arange(count * 8, dtype=np.float32).reshape(count, 8) * .01 + 2.
    # The frame owner is measured CPU data; the tiny adapter's schema is an
    # explicit fixture rather than an assertion that these are real weights.
    return EpisodeRecord('padding-cpu', 'pick cube', frames, actions,
                         np.zeros(count), states=states)


def seed_context(ep, cfg, anchor):
    count = 32
    job = materialize_imagination_context(ep, anchor - count + 1, count,
                                         cfg['schema']['dt_seconds'])
    value = {'obs': job['frames'], 'states': job['states'], 'actions': job['actions'],
             'valid_mask': job['valid_mask'], 'action_mask': job['action_mask'],
             'dt': job['dt'], 'schema': copy.deepcopy(cfg['schema']),
             'instruction': ep.instruction, 'episode_id': ep.episode_id,
             'sample_id': ep.episode_id + ':seed' + str(job['frame_start']),
             'frame_start': job['frame_start'], 'frame_indices': job['frame_indices'],
             'request_lineage': {'episode_id': ep.episode_id,
                                 'seed_start': job['frame_start'],
                                 'seed_anchor': anchor,
                                 'context_start': job['context_start']}}
    if 'context_padding' in job:
        value['context_padding'] = job['context_padding']
    return value


class PaddingBackendCPU(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.ep = measured_episode()

    def test_real_adapter_init_and_two_h8_requests_accept_explicit_synthetic_context(self):
        for anchor in (0, 1, 15, 30):
            with self.subTest(anchor=anchor):
                cfg = configuration()
                model, backend = make_backend(cfg)
                ops = WeaverActorOperations(backend, cfg, 4)
                context = seed_context(self.ep, cfg, anchor)
                state = ops.init_resident_rollout(context)
                self.assertTrue(state['synthetic_history'])
                self.assertEqual(state['context_padding']['anchor_frame_index'], anchor)
                self.assertEqual(state['context_padding']['count'], 31 - anchor)
                self.assertEqual(tuple(state['latents']['states'].shape), (1, 32, 8))
                initial_encoder_count = model.im_encoder.encoded_images
                warmed = ops.init_resident_rollout(context)
                self.assertEqual(model.im_encoder.encoded_images, initial_encoder_count)
                self.assertEqual(warmed['context_padding'], state['context_padding'])
                for step in range(2):
                    result = ops.request_wire(state, torch.zeros(8, 7), self.ep.instruction)
                    self.assertEqual(result['prediction_horizon'], 8)
                    self.assertEqual(len(result['predicted_states']), 8)
                    self.assertEqual(len(result['reward_predictions']), 8)
                    self.assertTrue(np.isfinite(np.asarray(result['reward_predictions'])).all())
                    self.assertFalse(any(result['terminal_predictions']))
                    self.assertTrue(all(result['valid_mask']))
                    state = result['next_state']
                    self.assertTrue(state['synthetic_history'])
                    self.assertEqual(state['predicted_steps'], (step + 1) * 8)

    def test_padding_uses_only_real_nonnegative_frame_keys_and_reset_feature_identity(self):
        cfg = configuration()
        _, backend = make_backend(cfg)
        context = seed_context(self.ep, cfg, 1)
        backend.init_rollout(context, 0)
        keys = build_frame_keys(context, backend._feature_namespace, list(range(32)))
        self.assertEqual([views[0].frame_index for views in keys], [0] * 31 + [1])
        self.assertTrue(all(key.frame_index >= 0 for views in keys for key in views))
        self.assertEqual(len({views[0].identity for views in keys[:31]}), 1)
        self.assertEqual(len({views[0].content_sha256 for views in keys[:31]}), 1)
        self.assertNotEqual(keys[0][0].identity, keys[-1][0].identity)

    def test_corrupted_padding_is_rejected_before_encoder_or_cache_work(self):
        cfg = configuration()
        model, backend = make_backend(cfg)
        original = seed_context(self.ep, cfg, 1)
        changes = []
        bad = copy.deepcopy(original); bad['valid_mask'] = [True] * 32; changes.append(bad)
        bad = copy.deepcopy(original); bad['action_mask'][0] = True; changes.append(bad)
        bad = copy.deepcopy(original); bad['frame_indices'][0] = -1; changes.append(bad)
        bad = copy.deepcopy(original); bad.pop('frame_indices'); changes.append(bad)
        bad = copy.deepcopy(original); bad['dt'][0] = .1; changes.append(bad)
        bad = copy.deepcopy(original); bad['obs'][0, 0, 0, 0, 0] = 200; changes.append(bad)
        for context in changes:
            with self.subTest(keys=list(context)), self.assertRaises(ValueError):
                backend.init_rollout(context, 0)
            self.assertEqual(model.im_encoder.encoded_images, 0)

    def test_padding_requires_enabled_cpu_inference_admission_and_is_forbidden_in_training(self):
        enabled = configuration()
        context = seed_context(self.ep, enabled, 1)
        _, disabled = make_backend(configuration(enabled=False))
        with self.assertRaisesRegex(ValueError, 'FRESH inference-only'):
            disabled.init_rollout(context, 0)
        _, backend = make_backend(enabled)
        training = dict(context, label_source='libero_env_step')
        with self.assertRaisesRegex(ValueError, 'inference-only'):
            backend._cpu_row(training, 32, True)
        with self.assertRaisesRegex(ValueError, 'explicit CPU admission'):
            backend._raw_sample(context, 32, False)

    def test_cache_helper_cannot_invent_physical_ids_when_padding_mapping_is_missing(self):
        cfg = configuration()
        _, backend = make_backend(cfg)
        context = seed_context(self.ep, cfg, 1)
        backend.init_rollout(context, 0)
        malformed = dict(context)
        malformed.pop('frame_indices')
        with self.assertRaises(ValueError):
            build_frame_keys(malformed, backend._feature_namespace, list(range(32)))


if __name__ == '__main__':
    unittest.main()
