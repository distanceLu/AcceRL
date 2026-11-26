from dataclasses import replace
import unittest

import numpy as np
import torch

from fresh_config import FreshConfig
from fresh_policy_replay import SegmentReplay
from jit_gae_plan import build_plan
from jit_pixel_store import resolve_shard_pixels, stage_segment_pixels
from test_fresh_policy_replay_cpu import segment


class StagedReplayPixelsTests(unittest.TestCase):
    def test_selected_order_suffix_bounds_bootstrap_and_tensor_bits(self):
        store = SegmentReplay(32, FreshConfig(refresh_gae=True, real_only=True, real_fraction=1., real_policy_replay=True))
        store.set_versions(5, 0)
        blocks, requested = {}, []

        def put(array):
            key = len(blocks)
            blocks[key] = array.copy()
            return key

        def get(keys):
            requested.extend(keys)
            return [blocks[key] for key in keys]

        original = {}
        for k, terminal in [('end', True), ('cut', False)]:
            seg = segment(k, 'real', count=5, terminal=terminal)
            observations = tuple(dict(obs, pixel_values=(torch.arange(24).reshape(2, 3, 4) +
                (100 if terminal else 500) + i).to(torch.bfloat16).transpose(1, 2))
                for i, obs in enumerate(seg.observations))
            original[k] = observations
            store.add(stage_segment_pixels(replace(seg, observations=observations), put))
        sample_ids = ['cut:chunk3', 'end:chunk4', 'end:chunk2']
        batch, _ = store.materialize(sample_ids, jit_denominators=dict.fromkeys(sample_ids, 3))
        metadata = {'sample_ids': sample_ids, 'fresh_versions': [5, 0]}
        restored, count = resolve_shard_pixels((*batch, metadata), get)
        self.assertEqual(count, 6)
        self.assertEqual(sorted(requested), [2, 3, 4, 9, 10, 11])
        self.assertIs(restored[-1], metadata)
        for a, b in zip(batch[1:], restored[1:-1]):
            np.testing.assert_array_equal(a,b)
        self.assertEqual([x['_jit_sample_id'] for x in restored[0]], sample_ids)
        contexts, _ = build_plan(restored[0], 1, 5)
        for k, contexts_for_segment in contexts.items():
            for i, obs in enumerate(contexts_for_segment['observations']):
                if obs is not None:
                    torch.testing.assert_close(obs['pixel_values'], original[k][i]['pixel_values'], rtol=0, atol=0)
        for obs in restored[0]:
            row = obs['_jit_segment']['observations'][obs['_jit_index']]
            self.assertIs(obs['pixel_values'], row['pixel_values'])

    def test_existing_inline_observations_pass_through(self):
        store = SegmentReplay(8, FreshConfig(refresh_gae=True, real_only=True, real_fraction=1., real_policy_replay=True))
        store.set_versions(5, 0)
        seg = segment('old', 'real', count=2)
        store.add(seg)
        sid = seg.sample_id(0)
        batch, _ = store.materialize([sid], jit_denominators={sid: 2})
        raw = (*batch, {'sample_ids': [sid]})
        actual, count = resolve_shard_pixels(raw, lambda _: self.fail('unexpected fetch'))
        self.assertIs(actual, raw)
        self.assertEqual(count, 0)


if __name__ == '__main__':
    unittest.main()
