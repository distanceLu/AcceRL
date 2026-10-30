"""Backend role, Replay, publication and snapshot contracts without CUDA/Ray."""
from __future__ import annotations

import asyncio
import copy
from contextlib import nullcontext
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch

import capture_training
import cuda_ipc_weights as ipc
import parallel_checkpoint as checkpoint
import training_topology as topology


class Bundle(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dynamics = torch.nn.Linear(2, 2)
        self.reward_head = torch.nn.Linear(2, 1)
        self.frozen = torch.nn.Parameter(torch.ones(2), requires_grad=False)
        self.register_buffer("normalizer", torch.ones(2))
        self.register_buffer("cache", torch.zeros(1), persistent=False)


class Backend:
    def __init__(self, model, family="online"):
        self.model, self.family = model, family

    def publication_inventory(self):
        return {"parameter_names": [name for name, p in self.model.named_parameters() if p.requires_grad],
                "buffer_names": ["normalizer"],
                "backend_identity": {"backend": "weaver", "adapter": "test-v1"},
                "bundle_metadata": {"reward": "success_terminal", "weight_family": self.family},
                "cache_version": "normalizer-v1"}

    def publication_tensors(self):
        values = dict(self.model.named_parameters()) | dict(self.model.named_buffers())
        inventory = self.publication_inventory()
        return {name: (values[name] + 7 if self.family == "ema" and name in inventory["parameter_names"]
                       else values[name])
                for name in inventory["parameter_names"] + inventory["buffer_names"]}


class Replay:
    def __init__(self, backend_payload=True):
        self.calls = 0
        self.backend_payload = backend_payload
        self.sample = types.SimpleNamespace(remote=self._sample)

    async def _sample(self, size):
        self.calls += 1
        ids = [f"{self.calls}:{i}" for i in range(size)]
        stats = {"capture_meta": {"sample_ids": ids, "reservation_id": f"r{self.calls}"},
                 "unique_episodes": size, "fresh_episode_ratio": 1.0}
        if self.backend_payload:
            stats["backend_samples"] = [{"sample_id": sample_id} for sample_id in ids]
        return ids, list(range(size)), [0] * size, ["task"] * size, stats


class RoleTopologyTest(unittest.TestCase):
    def test_default_and_integrated_resources_and_roles(self):
        bindings = [{"resource": f"gpu_{i}", "index": str(i), "uuid": f"gpu-{i}"} for i in range(4)]
        with mock.patch.dict("os.environ", {"ACCERL_USE_MPS": "0"}):
            for kinds, total in ((None, 24), (("policy", "ctrl"), 16)):
                options = topology.ray_init_options(bindings, topology.SPLIT_LAYOUT, kinds)
                self.assertEqual(sum(options["resources"].values()), total)
                for kind in topology.active_model_kinds(kinds):
                    for role in (kind, kind + "_trainer"):
                        result = topology.actor_options(bindings, role, 3, topology.SPLIT_LAYOUT, kinds)
                        self.assertEqual(result["resources"], {"gpu_3": 1.0})
            with self.assertRaises(ValueError):
                topology.actor_options(bindings, "reward", 0, topology.SPLIT_LAYOUT, ("policy", "ctrl"))
            with self.assertRaises(ValueError):
                topology.ray_init_options(bindings, topology.DUAL_LAYOUT, ("policy", "ctrl"))


class ReplayCohortTest(unittest.IsolatedAsyncioTestCase):
    async def test_integrated_four_claims_advance_and_shard_metadata(self):
        replay = Replay()
        coordinator = topology.GlobalBatchCoordinator(None, replay, 4, wm_consumers=("ctrl",))
        batches = await asyncio.wait_for(asyncio.gather(*[
            coordinator.sample_wm(rank, 2, "ctrl") for rank in range(4)]), 1)
        self.assertEqual(replay.calls, 1)
        ids = []
        for batch in batches:
            sample_ids = batch[-1]["capture_meta"]["sample_ids"]
            self.assertEqual(batch[0], sample_ids)
            self.assertEqual([x["sample_id"] for x in batch[-1]["backend_samples"]], sample_ids)
            ids.extend(sample_ids)
        self.assertEqual(len(set(ids)), 8)
        next_batch = await asyncio.wait_for(coordinator.sample_wm(0, 2, "ctrl"), 1)
        self.assertEqual(replay.calls, 2)
        self.assertTrue(next_batch[0][0].startswith("2:"))
        with self.assertRaises(ValueError):
            await coordinator.sample_wm(0, 2, "reward")

    async def test_external_requires_eight_claims_and_shares_identical_rank_slices(self):
        replay = Replay()
        coordinator = topology.GlobalBatchCoordinator(None, replay, 4, wm_consumers=("ctrl", "reward"))
        ctrl = [await coordinator.sample_wm(rank, 2, "ctrl") for rank in range(4)]
        waiting = asyncio.create_task(coordinator.sample_wm(0, 2, "ctrl"))
        await asyncio.sleep(0)
        self.assertFalse(waiting.done())
        self.assertEqual(replay.calls, 1)
        reward = [await coordinator.sample_wm(rank, 2, "reward") for rank in range(4)]
        self.assertEqual(ctrl, reward)
        await asyncio.wait_for(waiting, 1)
        self.assertEqual(replay.calls, 2)

    async def test_legacy_tuple_and_incomplete_backend_metadata(self):
        replay = Replay(backend_payload=False)
        coordinator = topology.GlobalBatchCoordinator(None, replay, 4)
        batches = [await coordinator.sample_wm(rank, 2) for rank in range(4)]
        self.assertTrue(all(len(batch) == 5 and "backend_samples" not in batch[-1] for batch in batches))
        raw = await Replay()._sample(8)
        raw[-1]["backend_samples"].pop()
        with self.assertRaises(ValueError):
            topology.shard_batch(raw, "wm", 0, 2, 4)


class CapturePayloadTest(unittest.TestCase):
    def test_backend_targets_are_reservations_only_and_defaults_keep_tuple5(self):
        class Policy:
            def __init__(self, *args): pass
            def add_batch(self, *args): pass
        class World:
            def __init__(self, *args): pass
            def add_batch(self, *args): pass
            def sample_eval(self, *args): pass
        capture_training._install_replay_hooks({"np": np, "ReplayBufferActor": Policy, "WMReplayBufferActor": World})
        samples = [types.SimpleNamespace(obs=np.zeros((1,)), act=np.zeros((1,)), rew=0,
                   instruction="task", episode_id=f"e{i}", sample_id=f"s{i}",
                   backend_payload={"sample_id": f"s{i}"}) for i in range(2)]
        ledger = mock.Mock()
        ledger.reserve.side_effect = lambda ids: {"sample_ids": ids}
        actor = object.__new__(World)
        actor.buffer, actor.sampled_episode_ids, actor.capture_ledger = samples, set(), ledger
        actor.capture_sample_calls = 0
        writer = mock.Mock()
        writer.span.side_effect = lambda *a, **kw: nullcontext()
        with mock.patch.object(capture_training, "_writer", return_value=writer), \
                mock.patch.object(capture_training.random, "sample", return_value=samples):
            result = actor.sample(2)
            self.assertEqual(len(result), 5)
            self.assertEqual([x["sample_id"] for x in result[-1]["backend_samples"]], ["s0", "s1"])
            writer.replay_used.assert_not_called()
            for sample in samples:
                sample.backend_payload = {}
            self.assertNotIn("backend_samples", actor.sample(2)[-1])


class PublicationTest(unittest.TestCase):
    def test_legacy_ctrl_does_not_publish_buffers_and_hash_is_version_independent(self):
        model = Bundle()
        first = topology.export_cpu_weights(model, "ctrl", 1)
        second = topology.export_cpu_weights(model, "ctrl", 2)
        self.assertEqual(first["hash"], second["hash"])
        self.assertNotIn("normalizer", first["tensors"])
        self.assertNotIn("publication", first)
        self.assertEqual(set(first), {"kind", "version", "hash", "bytes", "tensor_count", "tensors"})
        destination = Bundle()
        ack = topology.apply_cpu_weights(destination, first, "world")
        self.assertEqual(ack, {key: first[key] for key in ipc.INVENTORY_KEYS})

    def test_bundle_publishes_head_buffer_and_selected_ema_without_mutating_source(self):
        source, target = Bundle(), Bundle()
        before = source.dynamics.weight.detach().clone()
        src_backend, dst_backend = Backend(source, "ema"), Backend(target, "ema")
        payload = topology.export_cpu_weights(source, "world", 5, world_backend=src_backend)
        self.assertIn("reward_head.weight", payload["tensors"])
        self.assertIn("normalizer", payload["tensors"])
        self.assertEqual(payload["publication"]["bundle_version"], 5)
        ack = topology.apply_cpu_weights(target, payload, "ctrl", world_backend=dst_backend)
        self.assertEqual(ack["publication"], payload["publication"])
        torch.testing.assert_close(source.dynamics.weight, before)
        torch.testing.assert_close(target.dynamics.weight, before + 7)

    def test_missing_trainable_head_and_unknown_buffer_are_rejected(self):
        model = Bundle()
        for change in (lambda spec: spec["parameter_names"].remove("reward_head.weight"),
                       lambda spec: spec["buffer_names"].append("missing")):
            spec = Backend(model).publication_inventory()
            change(spec)
            with self.assertRaises(ValueError):
                topology.export_cpu_weights(model, "ctrl", 1, world_backend=spec)

    def test_backend_metadata_and_buffers_are_bound_to_hash_before_any_copy(self):
        source, target = Bundle(), Bundle()
        src_backend, dst_backend = Backend(source), Backend(target)
        payload = topology.export_cpu_weights(source, "ctrl", 1, world_backend=src_backend)
        before = target.dynamics.weight.detach().clone()
        for mutate in (lambda p: p["publication"].update(cache_version="other"),
                       lambda p: p["tensors"]["normalizer"].fill_(17),
                       lambda p: p["tensors"].pop("reward_head.bias")):
            bad = copy.deepcopy(payload)
            mutate(bad)
            with self.assertRaises(ValueError):
                topology.apply_cpu_weights(target, bad, "ctrl", world_backend=dst_backend)
            torch.testing.assert_close(target.dynamics.weight, before)

    def test_ipc_inventory_metadata_schema_and_ack_lifetime(self):
        model = Bundle()
        snapshot = topology.export_cpu_weights(model, "ctrl", 1, world_backend=Backend(model))
        descriptors = []
        for name, tensor in snapshot["tensors"].items():
            shape, stride = list(tensor.shape), list(tensor.stride())
            args = ["torch.Tensor", shape, stride, 0, "torch.storage.TypedStorage", str(tensor.dtype),
                    0, b"\x02c" + b"a" * 64, tensor.numel() * tensor.element_size(), 0, False,
                    b"ref", 0, b"e" * 64, True]
            descriptors.append({"name": name, "dtype": str(tensor.dtype), "size": shape,
                                "stride": stride, "args": args})
        uuid = "12345678-1234-1234-1234-123456789012"
        inventory = {key: value for key, value in snapshot.items() if key != "tensors"}
        payload = {"protocol": ipc.PROTOCOL, "kind": "ctrl", "rank": 0, "uuid": uuid,
                   "run_id": "test", "token": "a" * 32, "inventory": inventory,
                   "source": {"uid": 0, "pid": 1, "birth_ticks": 1, "hostname": "test"},
                   "runtime": {"torch": "test", "cuda": "test"}, "ready_event": b"r" * 64,
                   "tensors": descriptors}
        self.assertEqual(ipc.validate_payload(payload, "world", uuid, "test", 0), inventory)
        for change in (lambda p: p["inventory"]["publication"].update(bundle_version=2),
                       lambda p: p["inventory"]["publication"]["buffer_names"].clear(),
                       lambda p: p["inventory"].update(unrecognized="bad"),
                       lambda p: p["tensors"][0]["args"].__setitem__(7, b"bad")):
            bad = copy.deepcopy(payload)
            change(bad)
            with self.assertRaises(ValueError):
                ipc.validate_payload(bad, "ctrl", uuid, "test", 0)
        exporter = ipc.CudaIPCWeightExporter(model, "ctrl", uuid, "test", 0, world_backend=Backend(model))
        exporter._active = {"token": payload["token"], "payload": payload, "refs": [("head", model.reward_head)]}
        bad_ack = copy.deepcopy(inventory)
        bad_ack["publication"]["bundle_metadata"]["reward"] = "different"
        with self.assertRaises(RuntimeError):
            exporter.finish(payload["token"], bad_ack)
        self.assertTrue(exporter.active)
        with mock.patch.object(torch.cuda, "ipc_collect") as collect:
            self.assertEqual(exporter.finish(payload["token"], inventory), inventory)
            collect.assert_called_once()
        self.assertFalse(exporter.active)
        self.assertFalse(torch.cuda.is_initialized())


class CheckpointRolesTest(unittest.TestCase):
    def test_integrated_snapshot_contains_only_two_actual_owners(self):
        world, policy = Bundle(), torch.nn.Linear(2, 1)
        owners = {
            "policy": types.SimpleNamespace(training_role="policy", rank=0, world_size=4,
                                             global_step=5, model=policy),
            "ctrl": types.SimpleNamespace(training_role="ctrl", rank=0, world_size=4,
                global_step=5, ctrl_step=1, ctrl_world_engine=world, world_backend=Backend(world, "ema"))}
        components = {kind: checkpoint.export_split_component(owner, kind) for kind, owner in owners.items()}
        with tempfile.TemporaryDirectory() as directory:
            manifest = checkpoint.save_split_weights(components, directory, active_kinds=("policy", "ctrl"))
            self.assertEqual(manifest["steps"], {"policy": 5, "ctrl": 1})
            target = Bundle()
            trainer = types.SimpleNamespace(model=torch.nn.Linear(2, 1), ctrl_world_engine=target,
                                            world_backend=Backend(target, "ema"))
            checkpoint.apply_parallel_weights(trainer, Path(directory) / "latest_weights.pt",
                                               active_kinds=("policy", "ctrl"))
            torch.testing.assert_close(target.reward_head.weight, world.reward_head.weight + 7)
            with self.assertRaises(ValueError):
                checkpoint.save_split_weights(components, directory)
            mixed = copy.deepcopy(components)
            mixed["ctrl"]["policy_step"] = 4
            with self.assertRaises(ValueError):
                checkpoint.save_split_weights(mixed, directory, active_kinds=("policy", "ctrl"))

    def test_default_snapshot_keeps_three_roles(self):
        trainer = types.SimpleNamespace(rank=0, world_size=4, global_step=5, ctrl_step=1,
            reward_step=1, model=torch.nn.Linear(2, 1), ctrl_world_engine=Bundle(),
            reward_engine=torch.nn.Linear(2, 2))
        with tempfile.TemporaryDirectory() as directory:
            manifest = checkpoint.save_parallel_weights(trainer, directory)
            self.assertEqual(set(manifest["steps"]), {"policy", "ctrl", "reward"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
