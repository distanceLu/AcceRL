"""Actual Replay actor + capture hooks + coordinator, isolated from Ray/GPU."""
import ast
import asyncio
from collections import deque
from contextlib import nullcontext
import copy
import gc
import hashlib
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np

import capture_training
from pipeline_runtime import EpisodeReplayMixin
from training_topology import GlobalBatchCoordinator
from episode_frame_store import normalize_frames


class Writer:
    run_id, actor_id = "cpu-pipeline-test", "0"
    def __init__(self, *args):
        self.events, self.sql = [], []
    def event(self, name, **kwargs): self.events.append((name, kwargs))
    def _queue(self, sql, values): self.sql.append((sql, values))
    def _queue_many(self, sql, values): self.sql.extend((sql, row) for row in values)
    def span(self, *args, **kwargs): return nullcontext()


def fraction(identity):
    # Tests select this adapter explicitly; routing itself is actual actor code.
    return 0. if identity.startswith("eval-") else 1.


def classes(hooks=True):
    tree = ast.parse(Path(__file__).with_name("train.py").read_text(encoding="utf-8"))
    names = {"WMReplayBufferActor", "ReplayBufferActor"}
    nodes = [copy.deepcopy(node) for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    for node in nodes:
        node.decorator_list = []
    namespace = dict(np=np, random=random, deque=deque, List=list, Experience=object,
                     WMExperience=object, EpisodeReplayMixin=EpisodeReplayMixin,
                     setup_debugger=lambda *a: None, stable_episode_fraction=fraction)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 "<actual Replay classes>", "exec"), namespace)
    if hooks:
        capture_training._install_replay_hooks(namespace)
    return namespace


def actor(*, hooks=True, capacity=8, frame_budget=60, max_episode_frames=6,
          minimum_train_windows=2, **budget):
    values = dict(max_episode_frames=max_episode_frames,
                  max_resident_frame_bytes=frame_budget * 2 * 3 * 224 * 224,
                  real_workers=2, imagination_workers=2, max_real_in_flight=2,
                  max_imagine_in_flight=2, seeds_per_episode=2, max_pending_episodes=16,
                  policy_high_watermark=32, policy_low_watermark=16)
    values.update(budget)
    cls = classes(hooks)["WMReplayBufferActor"]
    with patch.object(capture_training, "Telemetry", Writer):
        return cls(capacity, ctrl_eval_capacity=2, ctrl_eval_windows_per_episode=2,
                   ctrl_eval_episode_ratio=.5, pipeline_config=values,
                   world_config={"backend": "ctrl", "schema": {}}, context_frames=2,
                   window_frames=3, policy_chunks_per_seed=2,
                   minimum_train_windows=minimum_train_windows)


def payload(identity, count=6):
    frames = np.arange(count * 2 * 3 * 224 * 224, dtype=np.uint8).reshape(count, 2, 3, 224, 224)
    actions = np.arange(count * 7, dtype=np.float32).reshape(count, 7)
    actions[0] = 0
    done = np.zeros(count, bool)
    done[-1] = True
    return dict(episode_id=identity, instruction="pick bowl", frames=frames, actions=actions,
                rewards=np.arange(count, dtype=np.float64) / 10,
                terminated=done, truncated=np.zeros(count, bool), success=done,
                frame_groups=[dict(start_frame=0, end_frame=count-1, group_id=identity+":group")],
                versions={"policy": 12})


def ingest(owner, identity, worker=0, count=6):
    lease = owner.acquire_real(worker)
    if lease is None:
        raise AssertionError("Unexpected admission block: %r" % owner.pipeline_status())
    owner.finish_real_episode(lease["token"], payload(identity, count))


class AsyncMethod:
    def __init__(self, function): self.function = function
    async def remote(self, *args, **kwargs): return self.function(*args, **kwargs)


class AsyncActor:
    def __init__(self, actual): self.actual = actual
    def __getattr__(self, name): return AsyncMethod(getattr(self.actual, name))


class PipelineIntegration(unittest.TestCase):
    def test_actual_actor_hooks_reservation_projection_same_cohort(self):
        owner = actor()
        ingest(owner, "train-0")
        ticket = owner.reserve_windows(4)["ticket_id"]
        windows = owner.window_tickets[ticket][0]
        for rank in (0, 1):
            world = owner.project_windows(ticket, rank, 2, "ctrl")
            reward = owner.project_windows(ticket, rank, 2, "reward")
            selected = windows[rank*2:rank*2+2]
            np.testing.assert_array_equal(normalize_frames(world[0]), np.stack([row.obs for row in selected]))
            np.testing.assert_array_equal(normalize_frames(reward[0]), np.stack([row.obs[-1, 0] for row in selected]))
            self.assertEqual(world[-1]["capture_meta"], reward[-1]["capture_meta"])
            self.assertEqual(reward[1].shape, (2, 0))
            self.assertNotIn("backend_samples", reward[-1])
            np.testing.assert_array_equal(world[2], reward[2])
            self.assertEqual(world[0].nbytes / reward[0].nbytes, 6.)
        reserves = [event for event in owner.capture_telemetry.events if event[0] == "replay_reserve"]
        self.assertEqual(len(reserves), 1)
        self.assertEqual(set(owner.capture_ledger.resident), {window.sample_id for window in owner.buffer})
        owner.release_windows(ticket)
        self.assertEqual(owner.pipeline_status()["pending_draws"], 0)

    def test_capture_hook_sample_override_does_not_replace_ticket_path(self):
        owner = actor()
        ingest(owner, "train-0")
        # The existing hooks replace .sample; the compact path must never call it.
        owner.sample = lambda *a: (_ for _ in ()).throw(AssertionError("legacy full-window sample invoked"))
        ticket = owner.reserve_windows(4)["ticket_id"]
        result = owner.project_windows(ticket, 0, 4, "reward")
        self.assertEqual(result[0].shape, (4, 3, 224, 224))
        owner.release_windows(ticket)

    def test_capacity_eviction_keeps_reserved_pixels_until_all_consumers_release(self):
        owner = actor(capacity=4)
        ingest(owner, "train-0")
        previous = weakref.ref(owner.frame_store.get("train-0"))
        ticket = owner.reserve_windows(4)["ticket_id"]
        ingest(owner, "train-1")
        owner.stop_pipeline()  # Remove imagination pins, preserve draw and replay.
        gc.collect()
        self.assertIsNotNone(previous())
        self.assertTrue(all(row.episode_id == "train-1" for row in owner.buffer))
        self.assertFalse(any("train-0" in sid for sid in owner.capture_ledger.resident))
        result = owner.project_windows(ticket, 0, 4, "reward")
        self.assertTrue(all(sid.startswith("train-0") for sid in result[-1]["capture_meta"]["sample_ids"]))
        owner.release_windows(ticket)
        gc.collect()
        self.assertIsNone(previous())
        self.assertEqual(owner.frame_store.stats()["resident_episodes"], 1)

    def test_fixed_holdout_uses_same_episode_routing_and_retains_only_selected_refs(self):
        owner = actor()
        ingest(owner, "eval-0")
        self.assertEqual(len(owner.buffer), 0)
        self.assertEqual([row.sample_index for row in owner.eval_buffer], [0, 3])
        self.assertEqual(set(owner.capture_eval_ledger.resident), {"eval-0:window0", "eval-0:window3"})
        ingest(owner, "eval-1")
        second = weakref.ref(owner.frame_store.get("eval-1"))
        owner.stop_pipeline()
        gc.collect()
        self.assertIsNone(second())  # full holdout did not retain a second episode
        self.assertEqual(owner.frame_store.stats()["resident_episodes"], 1)
        images, actions, texts = owner.sample_eval(2)
        self.assertEqual(images.shape, (2, 3, 2, 3, 224, 224))
        self.assertEqual(actions.shape, (2, 3, 7))
        self.assertEqual(texts, ["pick bowl"] * 2)

    def test_byte_pressure_releases_pending_pins_and_evicts_with_ledger_reason(self):
        owner = actor(capacity=32, frame_budget=12)
        ingest(owner, "train-0")
        ingest(owner, "train-1")
        previous = weakref.ref(owner.frame_store.get("train-0"))
        lease = owner.acquire_real(0)
        self.assertIsNotNone(lease)
        gc.collect()
        self.assertIsNone(previous())
        status = owner.pipeline_status()
        self.assertLessEqual(status["frames"]["image_bytes"] + status["budget"]["reserved_frame_bytes"],
                             owner.pipeline_config.max_resident_frame_bytes)
        self.assertGreaterEqual(len(owner.buffer), owner.minimum_train_windows)
        evictions = [values for sql, values in owner.capture_telemetry.sql
                     if sql.startswith("UPDATE replay_samples SET resident=0")]
        self.assertEqual(len(evictions), 4)
        self.assertTrue(all(values[1] == "episode_frame_byte_budget" for values in evictions))
        owner.cancel_real(lease["token"])

    def test_invalid_or_duplicate_ingestion_is_atomic_and_lease_can_cancel(self):
        owner = actor()
        ingest(owner, "train-0")
        duplicate_lease = owner.acquire_real(0)
        before = owner.pipeline.stats()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            owner.finish_real_episode(duplicate_lease["token"], payload("train-0"))
        self.assertEqual(owner.pipeline.stats(), before)
        owner.cancel_real(duplicate_lease["token"])
        invalid_lease = owner.acquire_real(0)
        invalid = payload("train-1")
        invalid["actions"][0] = 1
        with self.assertRaisesRegex(ValueError, "Reset"):
            owner.finish_real_episode(invalid_lease["token"], invalid)
        self.assertNotIn("train-1", owner.frame_store)
        self.assertEqual(owner.pipeline.stats()["active_real"], 1)
        owner.cancel_real(invalid_lease["token"])

    def test_async_coordinator_releases_one_ticket_after_both_groups_claim(self):
        owner = actor()
        ingest(owner, "train-0")
        coordinator = GlobalBatchCoordinator(None, AsyncActor(owner), 2,
                    wm_consumers=("ctrl", "reward"), pipeline_enabled=True)
        async def work():
            first = await coordinator.sample_wm(0, 2, "reward")
            self.assertEqual(len(owner.window_tickets), 1)
            others = await asyncio.gather(coordinator.sample_wm(1, 2, "ctrl"),
                          coordinator.sample_wm(0, 2, "ctrl"), coordinator.sample_wm(1, 2, "reward"))
            self.assertEqual(first[-1]["capture_meta"], others[1][-1]["capture_meta"])
            self.assertEqual(others[0][-1]["capture_meta"], others[2][-1]["capture_meta"])
        asyncio.run(work())
        self.assertEqual(len(owner.window_tickets), 0)
        reserves = [event for event in owner.capture_telemetry.events if event[0] == "replay_reserve"]
        self.assertEqual(len(reserves), 1)

    def test_stop_signals_new_workers_and_releases_coordinator_pins(self):
        owner = actor()
        ingest(owner, "train-0")
        seed = owner.acquire_imagine(1)
        real = owner.acquire_real(0)
        self.assertIsNotNone(seed)
        self.assertIsNotNone(real)
        result = owner.stop_pipeline()
        self.assertEqual(result["budget"]["active_real"], 0)
        self.assertEqual(result["budget"]["active_imagine"], 0)
        self.assertEqual(owner.imagination_records, {})
        self.assertEqual(owner.acquire_real(0), {"stopped": True})
        self.assertEqual(owner.acquire_imagine(1), {"stopped": True})
        self.assertEqual(owner.finish_imagine(seed['token'], 2), {'stopped': True})
        self.assertTrue(owner.finish_real_episode(real['token'], payload('late-episode'))['stopped'])
        self.assertNotIn('late-episode', owner.frame_store)
        self.assertEqual(owner.cancel_real(real['token']), {'stopped': True})
        self.assertEqual(owner.cancel_imagine(seed['token']), {'stopped': True})


if __name__ == "__main__":
    unittest.main()
