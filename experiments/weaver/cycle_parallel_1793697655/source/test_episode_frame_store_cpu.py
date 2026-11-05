"""Episode retention, alignment and byte-level preprocessing tests (no GPU)."""
import gc
import importlib.util
import pickle
from pathlib import Path
import ast
from concurrent.futures import ThreadPoolExecutor
import unittest
import weakref

import numpy as np
from PIL import Image

from episode_frame_store import (EpisodeRecord, EpisodeFrameStore,
                                 encode_observations, normalize_frames, pack_episode)


HAS_TORCH = importlib.util.find_spec("torch") is not None


def schema(size=(224, 224)):
    return {"backend": "weaver", "schema": {
        "domain": "libero", "action_schema_id": "libero_eef_delta_7_v1",
        "state_schema_id": "libero_eef_pos_axisangle_gripper_v1",
        "encoder_id": "test", "normalizer_id": "test", "action_dim": 7,
        "state_dim": 8, "alignment": "action_into_frame",
        "image_keys": ["full_image", "wrist_image"], "image_size": list(size),
        "image_value_range": "minus_one_one", "dt_seconds": .05,
        "n_history": 7, "n_memory_frames": 0, "t_memory": 1}}


def record(identity="episode-0", count=15, backend=None):
    pixels = np.arange(count * 2 * 3 * 224 * 224, dtype=np.uint8).reshape(count, 2, 3, 224, 224)
    actions = np.arange(count * 7, dtype=np.float32).reshape(count, 7)
    actions[0] = 0
    rewards = np.arange(count, dtype=np.float64) * .123456789
    terminal = np.zeros(count, bool)
    terminal[-1] = True
    return EpisodeRecord(identity, "pick up the bowl", pixels, actions, rewards,
        versions={"policy": 2, "world": 3, "reward": 5},
        frame_groups=({"start": 0, "end": count, "request_id": "request-0"},),
        states=np.arange(count * 8, dtype=np.float32).reshape(count, 8),
        terminated=terminal, truncated=np.zeros(count, bool), success=terminal,
        metadata={"seed": 42, "nested": {"values": [1, 2]}}, backend_config=backend)


class EpisodeStorageTests(unittest.TestCase):
    def test_normalization_all_pixel_values_exact_float32_arithmetic(self):
        values = np.arange(256, dtype=np.uint8)
        expected = values.astype(np.float32) / np.float32(255.0)
        expected = expected * np.float32(2.0) - np.float32(1.0)
        actual = normalize_frames(values)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual((actual[0], actual[-1]), (-1., 1.))
        with self.assertRaises(ValueError):
            normalize_frames(expected)

    def test_encode_resizes_raw_images_exactly_like_original_pil_path(self):
        rng = np.random.default_rng(125)
        observations = [{"full_image": rng.integers(0, 256, (173, 319, 3), dtype=np.uint8),
                         "wrist_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)}]
        packed = encode_observations(observations)
        self.assertEqual(packed.shape, (1, 2, 3, 224, 224))
        for view, key in enumerate(("full_image", "wrist_image")):
            expected = np.array(Image.fromarray(observations[0][key]).resize((224, 224), Image.Resampling.BILINEAR))
            np.testing.assert_array_equal(packed[0, view], expected.transpose(2, 0, 1))
        observations[0]["full_image"] = observations[0]["full_image"].astype(np.float32)
        with self.assertRaisesRegex(ValueError, "uint8"):
            encode_observations(observations)

    def test_pack_preserves_action_reward_alignment_and_window_identity(self):
        observations = [{key: np.full((224, 224, 3), t, np.uint8)
                         for key in ("full_image", "wrist_image")} for t in range(7)]
        actions = np.arange(42, dtype=np.float32).reshape(6, 7) + 1
        rewards = [0., .25, 0., 0., .4, 1.]
        ep = pack_episode(observations, actions, rewards, "task", "ep")
        np.testing.assert_array_equal(ep.actions[0], np.zeros(7))
        np.testing.assert_array_equal(ep.actions[1:], actions)
        np.testing.assert_array_equal(ep.rewards, [0.] + rewards)
        for start in range(4):
            window = ep.window(start, 4)
            self.assertEqual(window.sample_id, "ep:window%d" % start)
            self.assertEqual(window.sample_index, start)
            self.assertEqual(window.rew, rewards[start + 2])
            self.assertEqual(window.instruction, "task")
            np.testing.assert_array_equal(window.act, ep.actions[start:start + 4])
            np.testing.assert_array_equal(window.obs, normalize_frames(ep.frames[start:start + 4]))
        self.assertEqual(ep.windows(8), [])

    def test_refs_share_one_episode_and_reward_projects_one_frame(self):
        ep = record(count=40)
        windows = ep.windows(11)
        self.assertEqual(len(windows), 30)
        for window in windows:
            self.assertIs(window.episode, ep)
            self.assertTrue(np.shares_memory(window.obs_uint8, ep.frames))
            self.assertTrue(np.shares_memory(window.act, ep.actions))
            self.assertTrue(np.shares_memory(window.world_observations, ep.frames))
            self.assertTrue(np.shares_memory(window.reward_primary, ep.frames))
            self.assertEqual(window.reward_primary.shape, (3, 224, 224))
            self.assertEqual(set(vars(window)), {"episode", "start", "length"})
        old_image_bytes = len(windows) * 11 * 2 * 3 * 224 * 224 * 4
        self.assertEqual(old_image_bytes / ep.image_nbytes, 33)

    def test_owned_arrays_and_nested_metadata_cannot_mutate_or_alias_inputs(self):
        frames = np.zeros((2, 2, 3, 224, 224), np.uint8)
        actions = np.zeros((2, 7), np.float32)
        versions = {"policy": 1}
        metadata = {"nested": [1, 2]}
        ep = EpisodeRecord("ep", "task", frames, actions, [0., 1.], versions=versions, metadata=metadata)
        frames[:] = 255
        actions[1] = 33
        versions["policy"] = 9
        metadata["nested"].append(3)
        self.assertFalse(ep.frames.any())
        self.assertFalse(ep.actions.any())
        self.assertEqual(ep.versions["policy"], 1)
        self.assertEqual(ep.metadata["nested"], (1, 2))
        for array in (ep.frames, ep.actions, ep.rewards, ep.window(0, 2).obs_uint8):
            with self.assertRaises(ValueError):
                array.flags.writeable = True
            with self.assertRaises(ValueError):
                array.flat[0] = 7
        mutable_version_copy = ep.window(0, 2).model_versions
        mutable_version_copy["policy"] = 99
        self.assertEqual(ep.versions["policy"], 1)
        materialized = ep.window(0, 2).obs
        materialized[:] = 9
        self.assertFalse(ep.frames.any())

    def test_pickle_restores_immutability_and_preserves_shared_record(self):
        ep = record(count=5)
        restored = pickle.loads(pickle.dumps(ep.windows(3), protocol=5))
        self.assertIs(restored[0].episode, restored[1].episode)
        np.testing.assert_array_equal(restored[0].obs, ep.window(0, 3).obs)
        with self.assertRaises(ValueError):
            restored[0].episode.frames.flags.writeable = True

    def test_eviction_releases_only_after_last_replay_and_imagination_owner(self):
        store = EpisodeFrameStore()
        ep = store.register(record())
        observer = weakref.ref(ep)
        windows = ep.windows(11)
        imagination_lease = ep
        del ep
        gc.collect()
        self.assertEqual(store.stats()["resident_episodes"], 1)
        self.assertEqual(store.stats()["resident_frames"], 15)
        windows.clear()
        gc.collect()
        self.assertIsNotNone(observer())
        del imagination_lease
        gc.collect()
        self.assertIsNone(observer())
        self.assertEqual(store.stats()["image_bytes"], 0)
        self.assertEqual(store.stats()["registered_episodes"], 1)
        with self.assertRaises(KeyError):
            store.get("episode-0")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            store.register(record())

    def test_duplicate_registration_is_atomic(self):
        store, ep = EpisodeFrameStore(), record()
        def attempt(_):
            try:
                store.register(ep)
                return True
            except ValueError:
                return False
        with ThreadPoolExecutor(max_workers=4) as pool:
            self.assertEqual(sum(pool.map(attempt, range(12))), 1)
        self.assertIs(store.get(ep.episode_id), ep)
        self.assertEqual(store.stats()["array_bytes"], ep.array_nbytes)

    def test_bounds_and_episode_boundaries_reject_invalid_windows(self):
        ep = record(count=5)
        for start, length in ((-1, 2), (0, 0), (4, 2), (0, 6), (1., 2), (False, 2)):
            with self.assertRaises(ValueError):
                ep.window(start, length)
        self.assertEqual(ep.window(np.int64(1), np.int64(2)).end, 3)

    def test_invalid_episode_rejected_before_registration(self):
        frames = np.zeros((3, 2, 3, 224, 224), np.uint8)
        good = dict(episode_id="ep", instruction="task", frames=frames,
                    actions=np.zeros((3, 7)), rewards=[0., 0., 1.])
        corruptions = [dict(frames=frames.astype(np.float32)), dict(actions=np.ones((3, 7))),
            dict(actions=np.zeros((2, 7))), dict(rewards=[1., 0., 1.]),
            dict(rewards=[0., np.nan, 1.]), dict(states=np.zeros((2, 8))),
            dict(terminated=[False, False, True]),
            dict(terminated=[False, True, True], truncated=[False]*3, success=[False]*3)]
        store = EpisodeFrameStore()
        for bad in corruptions:
            with self.assertRaises(ValueError):
                store.register(EpisodeRecord(**(good | bad)))
        self.assertEqual(store.stats()["registered_episodes"], 0)

    def test_weaver_metadata_equivalent_without_materializing_images(self):
        from types import SimpleNamespace
        from world_backends.libero import backend_window_payload
        backend = schema((112, 160))
        ep = record(count=6, backend=backend)
        # Metadata construction works without torch and without any image resize.
        for start in (0, 2):
            window = ep.window(start, 4)
            actual = window.backend_payload
            expected = backend_window_payload(SimpleNamespace(**backend),
                np.empty((6, 2, 3, 112, 160), np.float32), ep.states, ep.actions,
                ep.rewards, ep.success, ep.terminated, ep.truncated,
                start, start + 4, ep.instruction, ep.episode_id, window.sample_id)
            self.assertEqual(set(actual), set(expected))
            for key, value in expected.items():
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(actual[key], value)
                else:
                    self.assertEqual(actual[key], value)
            self.assertNotIn("obs", actual)
            self.assertNotIn("act", actual)
        self.assertFalse(ep.window(0, 4).backend_payload["action_mask"][0])
        self.assertTrue(ep.window(2, 4).backend_payload["action_mask"].all())

    @unittest.skipUnless(HAS_TORCH, "torch not installed; run this test on the remote CPU runtime")
    def test_torch_exactness_including_alternate_weaver_resolution(self):
        import torch
        import torch.nn.functional as functional
        pixels = np.arange(256, dtype=np.uint8)
        expected = (torch.from_numpy(pixels).float() / 255.0 * 2.0 - 1.0).numpy()
        np.testing.assert_array_equal(normalize_frames(pixels), expected)
        ep = record(count=5, backend=schema((96, 128)))
        # Reference computes all episode frames before slicing, as legacy code.
        all_frames = torch.from_numpy(ep.frames.copy()).float() / 255.0 * 2.0 - 1.0
        all_frames = functional.interpolate(all_frames.flatten(0, 1), size=(96, 128),
            mode="bilinear", align_corners=False).reshape(5, 2, 3, 96, 128).numpy()
        window = ep.window(1, 3)
        np.testing.assert_array_equal(window.obs, all_frames[1:4])
        np.testing.assert_array_equal(window.world_observations, all_frames[1:4])
        np.testing.assert_array_equal(window.reward_primary, all_frames[3, 0])


if __name__ == "__main__":
    unittest.main()
