"""Pixel parity, real preparation method, and eight-consumer ticket contracts."""
from __future__ import annotations

import ast
import asyncio
from pathlib import Path
import types
import unittest

import numpy as np

from replay_projection import materialize_window_shard, normalized_observations, reward_last_primary
from training_topology import GlobalBatchCoordinator

try:
    import torch
except ImportError:
    torch = None


class Window:
    def __init__(self, sample_id, value):
        self.sample_id = sample_id
        self.instruction = "task " + sample_id
        self.rew = value % 2
        self.value = value

    @property
    def world_observations(self):
        return np.full((11, 2, 3, 4, 4), self.value, np.uint8)

    @property
    def reward_primary(self):
        return np.full((3, 4, 4), self.value, np.uint8)

    @property
    def act(self):
        return np.full((11, 7), self.value, np.float32)

    @property
    def backend_payload(self):
        return {"sample_id": self.sample_id, "states": [self.value]}


class RewardOnlyWindow(Window):
    @property
    def world_observations(self):
        raise AssertionError("Reward materialized a full window")

    @property
    def act(self):
        raise AssertionError("Reward materialized unused actions")

    @property
    def backend_payload(self):
        raise AssertionError("Reward materialized backend targets")


def stats_for(records, reservation="draw1"):
    return {"unique_episodes": 1, "fresh_episode_ratio": 1.0,
            "capture_meta": {"sample_ids": [x.sample_id for x in records],
                             "reservation_id": reservation, "reserved_ns": 123}}


class TicketReplay:
    def __init__(self):
        self.calls = self.peak_pending = self.policy_credits = 0
        self.tickets = {}
        self.released = []
        self.fail_projection = self.fail_reservation = self.fail_release = False
        for name in ("reserve_windows", "project_windows", "release_windows", "policy_drawn"):
            setattr(self, name, types.SimpleNamespace(remote=getattr(self, "_" + name)))

    async def _reserve_windows(self, size):
        self.calls += 1
        if self.fail_reservation:
            raise ValueError("reservation failure")
        ticket_id = str(self.calls)
        records = [Window(ticket_id + ":" + str(i), i) for i in range(size)]
        self.tickets[ticket_id] = (records, stats_for(records, ticket_id))
        self.peak_pending = max(self.peak_pending, len(self.tickets))
        return {"ticket_id": ticket_id, "global_batch_size": size}

    async def _project_windows(self, ticket_id, rank, local_size, consumer):
        if self.fail_projection:
            raise ValueError("projection failure")
        records, stats = self.tickets[ticket_id]
        return materialize_window_shard(records, stats, rank, local_size, 4, consumer)

    async def _release_windows(self, ticket_id):
        self.released.append(ticket_id)
        if self.fail_release:
            raise ValueError("release failure")
        del self.tickets[ticket_id]

    async def _policy_drawn(self, count):
        self.policy_credits += count


class ProjectionTests(unittest.TestCase):
    def test_reward_avoids_all_unused_materialization(self):
        windows = [RewardOnlyWindow(str(i), i) for i in range(8)]
        stats = stats_for(windows)
        stats["backend_samples"] = ["global payload must be removed"] * 8
        batch = materialize_window_shard(windows, stats, 2, 2, 4, "reward")
        self.assertEqual(batch[0].shape, (2, 3, 4, 4))
        self.assertEqual(batch[0].dtype, np.uint8)
        self.assertEqual(batch[1].shape, (2, 0))
        self.assertNotIn("backend_samples", batch[-1])
        self.assertEqual(batch[-1]["capture_meta"]["sample_ids"], ["4", "5"])
        self.assertEqual(batch[-1]["capture_meta"]["shard_start"], 4)
        self.assertEqual(batch[-1]["sample_stats_scope"], "global_batch")
        self.assertEqual(stats["capture_meta"]["sample_ids"], [str(i) for i in range(8)])

    def test_world_backend_metadata_stays_rank_aligned(self):
        windows = [Window(str(i), i) for i in range(8)]
        batch = materialize_window_shard(windows, stats_for(windows), 3, 2, 4, "ctrl")
        self.assertEqual(batch[0].shape, (2, 11, 2, 3, 4, 4))
        self.assertEqual(batch[1].shape, (2, 11, 7))
        self.assertEqual([x["sample_id"] for x in batch[-1]["backend_samples"]], ["6", "7"])
        reward = materialize_window_shard(windows, stats_for(windows), 3, 2, 4, "reward")
        self.assertEqual(batch[0].nbytes // reward[0].nbytes, 22)
        np.testing.assert_array_equal(reward[0], batch[0][:, -1, 0])
        np.testing.assert_array_equal(reward[2], batch[2])
        self.assertEqual(reward[3], batch[3])
        self.assertEqual(reward[-1]["capture_meta"], batch[-1]["capture_meta"])

    def test_all_256_pixel_values_preserve_legacy_roundtrip(self):
        pixels = np.broadcast_to(np.arange(256, dtype=np.uint8).reshape(1, 1, 16, 16), (1, 3, 16, 16)).copy()
        legacy = pixels.astype(np.float32) / 255.0 * 2.0 - 1.0
        np.testing.assert_array_equal(normalized_observations(pixels), legacy)
        np.testing.assert_array_equal(reward_last_primary(pixels), legacy)
        reconstructed = ((reward_last_primary(pixels) + 1) / 2 * 255).astype(np.uint8)
        np.testing.assert_array_equal(reconstructed, ((legacy + 1) / 2 * 255).astype(np.uint8))
        self.assertGreater(np.count_nonzero(reconstructed != pixels), 0)
        self.assertIs(normalized_observations(legacy), legacy)
        with self.assertRaises(ValueError):
            reward_last_primary(np.ones((3, 4, 4), dtype=np.uint8))

    def test_actual_reward_preparation_matches_legacy_images_labels_text(self):
        # Execute the actual production method without importing Ray/models/CUDA.
        source = ast.parse(Path(__file__).with_name("train.py").read_text(encoding="utf-8"))
        method = next(n for n in ast.walk(source) if isinstance(n, ast.FunctionDef)
                      and n.name == "_prepare_reward_batch")

        class Tensor:
            def __init__(self, value): self.value = np.asarray(value)
            def float(self): return Tensor(self.value.astype(np.float32))
            def mean(self): return Tensor(self.value.mean())
            def item(self): return self.value.item()
            def numel(self): return self.value.size

        namespace = {"np": np,
                     "torch": types.SimpleNamespace(bfloat16="bf16", long="long",
                                tensor=lambda value, **kw: Tensor(value)),
                     "prepare_one_obs": lambda cfg, processor, obs, text, dtype: (obs["full_image"], text)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                     "train.py", "exec"), namespace)
        actor = types.SimpleNamespace(
            reward_engine=types.SimpleNamespace(parameters=lambda: iter([types.SimpleNamespace(device="cpu")])),
            reward_super_batch_size=2, reward_cfg=None, processor=None,
            reward_model=types.SimpleNamespace(prepare_inputs_batch=lambda items: items))
        pixels = np.broadcast_to(np.arange(256, dtype=np.uint8).reshape(1, 1, 16, 16), (2, 3, 16, 16)).copy()
        legacy = np.broadcast_to(normalized_observations(pixels)[:, None, None], (2, 11, 2, 3, 16, 16))
        method = namespace["_prepare_reward_batch"]
        old, new = [method(actor, data, np.array([0.0, 1.0], np.float32), ["a", "b"])
                    for data in (legacy, pixels)]
        np.testing.assert_array_equal(old["labels"].value, new["labels"].value)
        self.assertEqual(old["reward_pos_ratio_batch"], new["reward_pos_ratio_batch"])
        for first, second in zip(old["reward_inputs_batch"], new["reward_inputs_batch"]):
            np.testing.assert_array_equal(first[0], second[0])
            self.assertEqual(first[1], second[1])


class TicketCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    def coordinator(self, replay):
        return GlobalBatchCoordinator(None, replay, 4, ("ctrl", "reward"), pipeline_enabled=True)

    async def test_interleaved_eight_claims_identity_single_draw_and_bounded_pending(self):
        replay = TicketReplay()
        coordinator = self.coordinator(replay)
        batches = {}
        batches[("ctrl", 0)] = await coordinator.sample_wm(0, 2, "ctrl")
        early = asyncio.create_task(coordinator.sample_wm(0, 2, "ctrl"))
        await asyncio.sleep(0)
        self.assertFalse(early.done())
        self.assertEqual(replay.calls, 1)
        # Coordinator retains a lightweight ticket, never a global pixel array.
        self.assertEqual(set(coordinator.pending["wm"]), {"ticket_id", "global_batch_size"})
        for consumer, rank in [("reward", 3), ("ctrl", 2), ("reward", 0),
                               ("reward", 1), ("ctrl", 1), ("ctrl", 3), ("reward", 2)]:
            batches[(consumer, rank)] = await coordinator.sample_wm(rank, 2, consumer)
        await asyncio.wait_for(early, 1)
        self.assertEqual(replay.calls, 2)
        self.assertEqual(replay.released, ["1"])
        self.assertEqual(replay.peak_pending, 1)
        for rank in range(4):
            ctrl, reward = batches[("ctrl", rank)], batches[("reward", rank)]
            self.assertEqual(ctrl[-1]["capture_meta"], reward[-1]["capture_meta"])
            np.testing.assert_array_equal(ctrl[0][:, -1, 0], reward[0])
            self.assertEqual(ctrl[-1]["capture_meta"]["sample_ids"], [f"1:{2*rank}", f"1:{2*rank+1}"])

    async def test_projection_failure_wakes_waiters_releases_ticket_and_never_redraws(self):
        replay = TicketReplay()
        coordinator = self.coordinator(replay)
        await coordinator.sample_wm(0, 2, "ctrl")
        blocked = asyncio.create_task(coordinator.sample_wm(0, 2, "ctrl"))
        await asyncio.sleep(0)
        replay.fail_projection = True
        with self.assertRaisesRegex(ValueError, "projection failure"):
            await coordinator.sample_wm(1, 2, "reward")
        with self.assertRaisesRegex(RuntimeError, "cohort failed"):
            await asyncio.wait_for(blocked, 1)
        with self.assertRaisesRegex(RuntimeError, "cohort failed"):
            await coordinator.sample_wm(2, 2, "reward")
        self.assertEqual(replay.calls, 1)
        self.assertEqual(replay.released, ["1"])
        self.assertEqual(replay.tickets, {})

    async def test_reservation_and_release_errors_are_fatal(self):
        replay = TicketReplay()
        replay.fail_reservation = True
        coordinator = self.coordinator(replay)
        results = await asyncio.gather(*[coordinator.sample_wm(i, 2, "ctrl") for i in range(4)],
                                       return_exceptions=True)
        self.assertTrue(all(isinstance(error, Exception) for error in results))
        self.assertEqual(replay.calls, 1)
        replay = TicketReplay()
        replay.fail_release = True
        coordinator = self.coordinator(replay)
        for consumer in ("ctrl", "reward"):
            for rank in range(4):
                if consumer == "reward" and rank == 3:
                    with self.assertRaisesRegex(ValueError, "release failure"):
                        await coordinator.sample_wm(rank, 2, consumer)
                else:
                    await coordinator.sample_wm(rank, 2, consumer)
        with self.assertRaisesRegex(RuntimeError, "cohort failed"):
            await coordinator.sample_wm(0, 2, "ctrl")
        self.assertEqual(replay.released, ["1"])

    async def test_policy_budget_credits_once_per_global_draw(self):
        async def sample(size):
            ids = [str(i) for i in range(size)]
            return ids, ids, ids, ids, ids, {"sample_ids": ids, "reservation_id": "policy"}

        replay = TicketReplay()
        policy = types.SimpleNamespace(sample=types.SimpleNamespace(remote=sample))
        coordinator = GlobalBatchCoordinator(policy, replay, 4, pipeline_enabled=True)
        batches = await asyncio.gather(*[coordinator.sample_policy(i, 2) for i in range(4)])
        self.assertEqual(replay.policy_credits, 8)
        self.assertEqual([batch[0] for batch in batches], [["0", "1"], ["2", "3"], ["4", "5"], ["6", "7"]])

    async def test_corrupt_shard_identity_fails_closed(self):
        replay = TicketReplay()
        coordinator = self.coordinator(replay)
        await coordinator.sample_wm(0, 2, "ctrl")
        original = replay.project_windows.remote

        async def corrupt(*args):
            batch = await original(*args)
            batch[-1]["capture_meta"]["sample_ids"] = ["wrong-a", "wrong-b"]
            return batch

        replay.project_windows.remote = corrupt
        with self.assertRaisesRegex(ValueError, "different reserved sample"):
            await coordinator.sample_wm(0, 2, "reward")
        self.assertEqual(replay.released, ["1"])


@unittest.skipIf(torch is None, "torch unavailable; run this CPU parity test in the training environment")
class TorchPreparationTests(unittest.TestCase):
    def methods(self):
        tree = ast.parse(Path(__file__).with_name("train.py").read_text(encoding="utf-8"))
        names = {"_prepare_ctrl_batch", "_prepare_reward_batch", "_prepare_wm_batch", "_normalize_actions_bounds"}
        definitions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
        namespace = {"np": np, "torch": torch,
                     "prepare_one_obs": lambda cfg, processor, obs, text, dtype: (obs["full_image"], text)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[])),
                     "train.py", "exec"), namespace)
        return namespace

    def test_actual_ctrl_preparation_pixels_actions_rewards_exact_cpu_parity(self):
        methods = self.methods()
        engine = torch.nn.Linear(1, 1).cpu()
        actor = types.SimpleNamespace(ctrl_world_engine=engine,
                    ctrl_target_height=13, ctrl_target_width=11, ctrl_super_batch_size=2,
                    condition_low=torch.full((7,), -2.0), condition_high=torch.full((7,), 2.0))
        generator = np.random.default_rng(14)
        pixels = generator.integers(0, 256, (2, 11, 2, 3, 16, 16), dtype=np.uint8)
        # Original image_to_tensor is torch arithmetic; use this independent
        # reference rather than our NumPy conversion helper.
        old_pixels = (torch.from_numpy(pixels).float() / 255.0 * 2.0 - 1.0).numpy()
        actions = generator.uniform(-3, 3, (2, 11, 7)).astype(np.float32)
        rewards = np.array([0, 1], dtype=np.float32)
        outputs = [methods["_prepare_ctrl_batch"](actor, data, actions, rewards, ["a", "b"])
                   for data in (old_pixels, pixels)]
        for key in ("pixel", "action", "rew"):
            torch.testing.assert_close(outputs[0][key], outputs[1][key], rtol=0, atol=0)
            self.assertEqual(outputs[1][key].device.type, "cpu")
        self.assertEqual(outputs[0]["text"], outputs[1]["text"])

    def test_actual_reward_cpu_labels_and_every_pixel_value(self):
        methods = self.methods()
        actor = types.SimpleNamespace(reward_engine=torch.nn.Linear(1, 1).cpu(),
                    reward_super_batch_size=2, reward_cfg=None, processor=None,
                    reward_model=types.SimpleNamespace(prepare_inputs_batch=lambda items: items))
        pixels = np.broadcast_to(np.arange(256, dtype=np.uint8).reshape(1, 1, 16, 16), (2, 3, 16, 16)).copy()
        old_primary = (torch.from_numpy(pixels).float() / 255.0 * 2.0 - 1.0).numpy()
        legacy = np.broadcast_to(old_primary[:, None, None], (2, 11, 2, 3, 16, 16))
        outputs = [methods["_prepare_reward_batch"](actor, data, np.array([0, 1], np.float32), ["a", "b"])
                   for data in (legacy, pixels)]
        torch.testing.assert_close(outputs[0]["labels"], outputs[1]["labels"], rtol=0, atol=0)
        self.assertEqual(outputs[1]["labels"].device.type, "cpu")
        self.assertEqual(outputs[0]["reward_pos_ratio_batch"], outputs[1]["reward_pos_ratio_batch"])
        for old, new in zip(outputs[0]["reward_inputs_batch"], outputs[1]["reward_inputs_batch"]):
            np.testing.assert_array_equal(old[0], new[0])
            self.assertEqual(old[1], new[1])


if __name__ == "__main__":
    unittest.main()
