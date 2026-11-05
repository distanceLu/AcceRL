"""Actual CPU shape/stride controls for the conservative layout revision.

These do not establish production CUDA equivalence or diagnose the earlier
micro1 VAE rounding difference. No GPU or remote process is used.
"""
import unittest
import torch

import test_weaver_feature_cache_cpu as features
import test_weaver_optimized_backend_cpu as backend_tests
from weaver_feature_cache import FrozenEncoderGuard, FrozenFeatureCache, encode_feature_batch, build_frame_keys


class RecordingEncoder(features.TinyFrozenEncoder):
    def __init__(self):
        super().__init__()
        self.geometry = []

    def forward(self, images):
        self.geometry.append((tuple(images.shape), tuple(images.stride()), images.detach().clone()))
        return super().forward(images)


class FixedSelectedChunkGeometry(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.encoder = RecordingEncoder()
        self.guard = FrozenEncoderGuard(self.encoder, features.namespace().encoder_sha256)
        plan = features.training_frame_selection(40, 2, 8, 6, 5)
        rows = [features.sample(episode='e%d' % b) for b in range(4)]
        self.images = torch.stack([torch.from_numpy(row['obs']) for row in rows]).float()[:, plan['selected_indices']] / 255.
        self.keys = [build_frame_keys(row, features.namespace(), plan['selected_indices']) for row in rows]

    def encode(self, cache):
        return encode_feature_batch(self.encoder, self.images, self.keys, features.CAMERAS,
                                    cache=cache, chunk_size=4, guard=self.guard)

    def test_cold_cache_preserves_uncached_batch_time_layout_and_input_values(self):
        expected = self.encode(None)
        calls = self.encoder.geometry[:]
        self.encoder.geometry.clear()
        actual = self.encode(FrozenFeatureCache(features.namespace()))
        self.assertEqual(len(calls), 8)
        for old, new in zip(calls, self.encoder.geometry):
            self.assertEqual(old[:2], new[:2])
            self.assertEqual(new[0][:2], (4, 4))
            torch.testing.assert_close(old[2], new[2], rtol=0, atol=0)
        for key in expected:
            torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)

    def test_partial_miss_reencodes_only_its_original_selected_chunk(self):
        expected = self.encode(None)
        cache = FrozenFeatureCache(features.namespace())
        omitted = (2, 1, 0)
        for b in range(4):
            for frame in range(16):
                for view, camera in enumerate(features.CAMERAS):
                    if (b, frame, view) != omitted:
                        cache.put(self.keys[b][frame][view], expected[camera + '_features'][b, frame])
        before = self.encoder.images
        self.encoder.geometry.clear()
        actual = self.encode(cache)
        self.assertEqual(len(self.encoder.geometry), 1)
        self.assertEqual(self.encoder.geometry[0][0][:2], (4, 4))
        self.assertEqual(self.encoder.images - before, 16)
        torch.testing.assert_close(self.encoder.geometry[0][2], self.images[:, :4, 0], rtol=0, atol=0)
        for key in expected:
            torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
        self.assertLessEqual(cache.resident_bytes, cache.max_bytes)

    def test_fully_warm_selected_chunks_skip_encoder_entirely(self):
        cache = FrozenFeatureCache(features.namespace())
        expected = self.encode(cache)
        before = self.encoder.images
        self.encoder.geometry.clear()
        actual = self.encode(cache)
        self.assertEqual(self.encoder.geometry, [])
        self.assertEqual(self.encoder.images, before)
        for key in expected:
            torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)


class LegacyLatentScaffold(unittest.TestCase):
    def test_main_and_memory_views_match_legacy_strides_storage_offsets_and_noise(self):
        for batch in (1, 4):
            with self.subTest(batch=batch):
                config = backend_tests.configured()
                rows = backend_tests.rows_for(config, count=batch)
                _, legacy = backend_tests.make_backend(config)
                reference = backend_tests.prepare(legacy, rows)
                optimized_config = backend_tests.configured(selected=True, cpu=True, cache=True)
                model, optimized = backend_tests.make_backend(optimized_config)
                actual = backend_tests.prepare(optimized, rows)
                for group in ('obs', 'memory'):
                    self.assertEqual(list(reference[group]), list(actual[group]))
                    for key, expected in reference[group].items():
                        value = actual[group][key]
                        torch.testing.assert_close(expected, value, rtol=0, atol=0)
                        self.assertEqual(expected.stride(), value.stride(), group + '/' + key)
                        self.assertEqual(expected.storage_offset(), value.storage_offset(), group + '/' + key)
                        self.assertEqual(expected.untyped_storage().nbytes(), value.untyped_storage().nbytes(), group + '/' + key)
                        torch.manual_seed(20261027)
                        expected_noise = torch.randn_like(expected)
                        torch.manual_seed(20261027)
                        actual_noise = torch.randn_like(value)
                        torch.testing.assert_close(expected_noise, actual_noise, rtol=0, atol=0)
                # Exactly the16 selected frames per sample/view, despite a
                # latent-only 40-slot scaffold. No unused image is encoded.
                self.assertEqual(model.im_encoder.encoded_images, batch * 16 * 2)


if __name__ == '__main__':
    unittest.main()
