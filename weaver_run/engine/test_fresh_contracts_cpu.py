"""CPU-only adversarial tests for FRESH switches, padding and signal records."""
import copy
import unittest

import numpy as np

from fresh_config import FreshConfig
from fresh_context import validate_reset_padding
from fresh_signal import record_signal
from weaver_feature_cache import FeatureNamespace, build_frame_keys, validate_cpu_window


def padded():
    n, p = 32, 29
    obs = np.zeros((n, 2, 3, 2, 2), dtype=np.uint8)
    obs[p + 1:] = 17
    states = np.zeros((n, 8), dtype=np.float32)
    states[p + 1:] = 3
    actions = np.zeros((n, 7), dtype=np.float32)
    actions[p + 1:] = 1
    action_mask = np.arange(n) > p
    schema = {'domain': 'libero', 'action_schema_id': 'a7', 'state_schema_id': 's8',
        'encoder_id': 'encoder', 'normalizer_id': 'norm', 'action_dim': 7, 'state_dim': 8,
        'alignment': 'action_into_frame', 'image_keys': ['full_image', 'wrist_image'],
        'image_size': [2, 2], 'image_value_range': 'minus_one_one', 'dt_seconds': .05}
    row = {'obs': obs, 'states': states, 'actions': actions, 'schema': schema,
        'valid_mask': np.arange(n) >= p, 'action_mask': action_mask,
        'dt': np.where(action_mask, .05, 0.), 'episode_id': 'real0', 'sample_id': 'real0:seed0',
        'frame_start': 0, 'frame_indices': [max(i - p, 0) for i in range(n)],
        'context_padding': {'protocol': 'fresh_reset_left_repeat_v1', 'count': p,
            'context_start': -p, 'anchor_frame_index': n - p - 1, 'real_frame_start': 0,
            'real_frame_end_exclusive': n - p, 'synthetic_history': True}}
    return row


class Contracts(unittest.TestCase):
    def test_switches_are_off_by_default_and_independent(self):
        self.assertFalse(FreshConfig().enabled)
        for name in ('dense_reward', 'signal_telemetry', 'refresh_gae', 'replay_age_limits',
                     'seed_balancing', 'early_seed_context', 'real_policy_replay'):
            value = FreshConfig.from_mapping({name: True})
            self.assertTrue(value.enabled)
            self.assertTrue(getattr(value, name))
        for raw in ({'refresh_gae': 1}, {'max_sample_uses': True}, {'real_fraction': float('nan')},
                    {'typo': True}, {'max_world_age': 0}):
            with self.assertRaises(ValueError):
                FreshConfig.from_mapping(raw)

    def test_dense_spec_switch_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            FreshConfig(dense_reward=True).validate_reward_binding({'mode': 'libero_success_score_v1'})

    def test_only_explicit_inference_padding_is_admitted(self):
        row = padded()
        result = validate_cpu_window(row, row['schema'], 32, False, allow_context_padding=True)
        self.assertEqual(result['frame_count'], 32)
        for training, allow in ((True, True), (False, False)):
            with self.assertRaises(ValueError):
                validate_cpu_window(row, row['schema'], 32, training, allow_context_padding=allow)

    def test_padding_cannot_invent_controls_valid_frames_or_pixels(self):
        for key in ('pixels', 'state', 'action', 'valid', 'indices', 'provenance'):
            row = copy.deepcopy(padded())
            if key == 'pixels': row['obs'][0] = 25
            if key == 'state': row['states'][0] = 25
            if key == 'action': row['actions'][0] = 1
            if key == 'valid': row['valid_mask'][0] = True
            if key == 'indices': row['frame_indices'][0] = 7
            if key == 'provenance': row['context_padding']['synthetic_history'] = False
            with self.assertRaises(ValueError, msg=key):
                validate_cpu_window(row, row['schema'], 32, False, allow_context_padding=True)

    def test_padding_cache_keys_use_measured_reset_identity(self):
        row = padded()
        keys = build_frame_keys(row, FeatureNamespace('run', 'a' * 64, 'b' * 64), [0, 29, 30, 31])
        self.assertEqual([key[0].frame_index for key in keys], [0, 0, 1, 2])
        self.assertEqual(keys[0][0], keys[1][0])

    def test_small_finite_signal_records_and_no_arrays(self):
        record = record_signal(None, phase='generated', source='imagined', sample_id='actual0',
            reward=.2, advantage=-.1, phi_start=.3, phi_end=.5, rm_version=4)
        self.assertEqual(record['sample_id'], 'actual0')
        for bad in (float('nan'), np.zeros(100)):
            with self.assertRaises(ValueError):
                record_signal(None, phase='consumed', source='real', sample_id='x', reward=bad)


if __name__ == '__main__':
    unittest.main()
