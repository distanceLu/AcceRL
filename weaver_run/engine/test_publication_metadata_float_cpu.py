"""Actual WEAVER inventory/receiver metadata admission, tiny CPU tensors only.

The production frozen configuration and actual publication_inventory method are
used. This deliberately does not load the 866 native tensors or open CUDA IPC.
"""
from pathlib import Path
import copy
import importlib.util
import json
import math
import os
import socket
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SOURCE = HERE

sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(HERE))
import torch
import training_topology as topology
import cuda_ipc_weights as ipc
from world_backends.weaver import WeaverBackend, _config_mapping

UUID = "GPU-c43d6671-0979-b38d-9eae-4fe0a9d71c8c"
RUN_ID = "metadata_float_cpu_diagnostic_only"


def frozen_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, SOURCE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def backend():
    raw = json.loads((SOURCE / "examples/weaver_integrated_internal.json").read_text(encoding="utf8"))
    class TinyUpstream(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.wm = torch.nn.Linear(2, 2)
            self.rm = torch.nn.Linear(2, 1)
            self.critic = torch.nn.Linear(2, 1).requires_grad_(False)
    class TinyBundle(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.upstream = TinyUpstream()
            self.register_buffer("state_mean", torch.zeros(8))
            self.register_buffer("state_std", torch.ones(8))
            self.register_buffer("actions_mean", torch.zeros(7))
            self.register_buffer("actions_std", torch.ones(7))
    # Explicit CPU publication fixture; actual constructor/loader is outside scope.
    item = WeaverBackend.__new__(WeaverBackend)
    item.config = _config_mapping(raw)
    item.schema = item.config["schema"]
    item.metadata = item.config["checkpoint_metadata"]
    item.model = TinyBundle()
    item.publication_weights = item.config.get("publication_weights", "online")
    item.critic_mode = item.config.get("critic_mode", "disabled")
    item.loaded_snapshot = False
    return item


def payload_for(item):
    selected, publication = topology._publication_state(item.model, "ctrl", 0, item, for_export=True)
    descriptors = []
    for name, tensor in selected:
        size, stride = list(tensor.shape), list(tensor.stride())
        arguments = ["torch.Tensor", size, stride, 0, "torch.storage.TypedStorage",
                     str(tensor.dtype), 0, b"\x01c" + b"h" * 64,
                     tensor.numel() * tensor.element_size(), 0, False,
                     b"reference", 0, b"e" * 64, False]
        descriptors.append({"name": name, "dtype": str(tensor.dtype), "size": size,
                            "stride": stride, "args": arguments})
    inventory = {"kind": "ctrl", "version": 0,
                 "hash": topology._weight_hash(selected, "ctrl", publication),
                 "bytes": sum(t.numel() * t.element_size() for _, t in selected),
                 "tensor_count": len(selected), "publication": publication}
    return {"protocol": ipc.PROTOCOL, "run_id": RUN_ID, "kind": "ctrl", "rank": 0,
            "uuid": UUID, "token": "a" * 32, "inventory": inventory,
            "source": {"uid": 1004, "pid": 1, "birth_ticks": 1, "hostname": socket.gethostname()},
            "runtime": {"torch": str(torch.__version__), "cuda": str(torch.version.cuda)},
            "ready_event": b"r" * 64, "tensors": descriptors}


class MetadataAdmittedBeforeCUDA(RuntimeError):
    pass


class PublicationMetadataFloatCPU(unittest.TestCase):
    def test_actual_inventory_spec_state_and_hash_keep_float_and_wm_rm(self):
        item = backend()
        raw_inventory = item.publication_inventory()
        spec = topology._publication_spec(item.model, "ctrl", item)
        selected, publication = topology._publication_state(item.model, "ctrl", 0, item, for_export=True)
        for key in ("parameter_names", "buffer_names"):
            raw_inventory[key] = sorted(raw_inventory[key])
        self.assertEqual(raw_inventory, spec)
        self.assertIs(type(publication["bundle_metadata"]["native_reward_head"]["loss_coefficient"]), float)
        names = [name for name, _ in selected]
        self.assertTrue(any(name.startswith("upstream.wm.") for name in names))
        self.assertTrue(any(name.startswith("upstream.rm.") for name in names))
        self.assertFalse(any(name.startswith("upstream.critic.") for name in names))
        self.assertEqual(topology._weight_hash(selected, "ctrl", publication),
                         topology._weight_hash(selected, "ctrl", copy.deepcopy(publication)))

    def test_actual_inventory_through_ipc_fingerprint_and_payload_validation(self):
        item = backend()
        payload = payload_for(item)
        fingerprint = ipc.metadata_fingerprint(payload)
        self.assertEqual(len(fingerprint), 64)
        self.assertEqual(fingerprint, ipc.metadata_fingerprint(copy.deepcopy(payload)))
        admitted = ipc.validate_payload(payload, "ctrl", UUID, RUN_ID, 0)
        self.assertEqual(admitted, payload["inventory"])
        admitted["publication"]["bundle_metadata"]["native_reward_head"]["loss_coefficient"] = 9.0
        self.assertEqual(payload["inventory"]["publication"]["bundle_metadata"]["native_reward_head"]["loss_coefficient"], 1.0)

    def test_receiver_metadata_admitted_before_any_cuda_operation(self):
        item = backend()
        payload = payload_for(item)
        with patch.object(ipc, "_bound_device", side_effect=MetadataAdmittedBeforeCUDA("metadata passed")) as bound:
            with self.assertRaises(MetadataAdmittedBeforeCUDA):
                ipc.apply_cuda_ipc_weights(item.model, payload, "ctrl", UUID, RUN_ID, 0, world_backend=item)
        bound.assert_called_once()
        self.assertFalse(torch.cuda.is_initialized())

    def test_receiver_still_rejects_finite_but_different_head_metadata(self):
        item = backend()
        payload = payload_for(item)
        payload["inventory"]["publication"]["bundle_metadata"]["native_reward_head"]["loss_coefficient"] = 2.0
        with patch.object(ipc, "_bound_device") as bound:
            with self.assertRaisesRegex(ValueError, "destination backend bundle"):
                ipc.apply_cuda_ipc_weights(item.model, payload, "ctrl", UUID, RUN_ID, 0, world_backend=item)
        bound.assert_not_called()

    def test_both_layers_accept_only_finite_builtin_float(self):
        for value in (0.0, -0.0, 1.0, -3.25, sys.float_info.max, sys.float_info.min):
            with self.subTest(value=value):
                nested = {"a": [value, {"b": (None, True, 3, "safe")}]}
                topology._validate_publication_metadata(nested)
                self.assertTrue(ipc._primitive(nested))
                self.assertEqual(len(ipc.metadata_fingerprint(nested)), 64)

    def test_both_layers_reject_nan_inf_tensor_callable_and_objects_recursively(self):
        class FloatLike:
            def __float__(self):
                return 1.0
        class FloatSubclass(float):
            pass
        values = [float("nan"), float("inf"), -float("inf"), torch.tensor(1.0),
                  torch.tensor([1.0]), lambda: 1.0, object(), FloatLike(), FloatSubclass(1.0)]
        for value in values:
            with self.subTest(type=type(value).__name__):
                nested = {"a": [0.5, {"b": value}]}
                with self.assertRaises(ValueError):
                    topology._validate_publication_metadata(nested)
                self.assertFalse(ipc._primitive(nested))
                with self.assertRaises(ValueError):
                    ipc.metadata_fingerprint(nested)

    def test_payload_rejects_nonfinite_or_objects_before_receiver_cuda(self):
        for invalid in (float("nan"), float("inf"), torch.tensor(1.0), lambda: 1.0, object()):
            with self.subTest(type=type(invalid).__name__):
                item = backend()
                payload = payload_for(item)
                payload["inventory"]["publication"]["bundle_metadata"]["native_reward_head"]["loss_coefficient"] = invalid
                with self.assertRaises(ValueError):
                    ipc.validate_payload(payload, "ctrl", UUID, RUN_ID, 0)
                with patch.object(ipc, "_bound_device") as bound:
                    with self.assertRaises(ValueError):
                        ipc.apply_cuda_ipc_weights(item.model, payload, "ctrl", UUID, RUN_ID, 0, world_backend=item)
                bound.assert_not_called()

    def test_finite_coefficient_is_hashed_without_silent_type_normalization(self):
        one_float = {"coefficient": 1.0}
        one_int = {"coefficient": 1}
        self.assertNotEqual(ipc.metadata_fingerprint(one_float), ipc.metadata_fingerprint(one_int))
        self.assertEqual(json.loads(json.dumps(one_float)), one_float)
        self.assertIs(type(json.loads(json.dumps(one_float))["coefficient"]), float)

    def test_same_tiny_gpu_preflight_now_includes_the_production_float_metadata(self):
        from weaver_online_smoke import tiny_backend, reward_specification
        item = tiny_backend(torch, "cpu")
        cfg = json.loads((SOURCE / "examples/weaver_integrated_internal.json").read_text(encoding="utf8"))
        self.assertEqual(reward_specification(), cfg["backend_options"]["native_reward_head"])
        inventory = item.publication_inventory()
        self.assertEqual(inventory["bundle_metadata"]["native_reward_head"], reward_specification())
        spec = topology._publication_spec(item.model, "ctrl", item)
        selected, publication = topology._publication_state(item.model, "ctrl", 0, item)
        self.assertEqual(spec["bundle_metadata"]["native_reward_head"]["loss_coefficient"], 1.0)
        self.assertIs(type(publication["bundle_metadata"]["native_reward_head"]["loss_coefficient"]), float)
        self.assertEqual([name for name, _ in selected], ["rm.net.bias", "rm.net.weight", "wm.bias", "wm.weight"])
        self.assertEqual(len(ipc.metadata_fingerprint(publication)), 64)
        payload = payload_for(item)
        self.assertEqual(ipc.validate_payload(payload, "ctrl", UUID, RUN_ID, 0), payload["inventory"])
        self.assertFalse(torch.cuda.is_initialized())

    def test_existing_storage_handle_runtime_and_identity_rules_remain(self):
        item = backend()
        payload = payload_for(item)
        for uuid, run, rank in ((UUID, "wrong-run", 0), (UUID, RUN_ID, 1),
                               ("GPU-fe0edfe4-b193-1af7-6a98-cbff40214f74", RUN_ID, 0)):
            with self.assertRaises(ValueError):
                ipc.validate_payload(payload, "ctrl", uuid, run, rank)
        for prefix in (b"\x01c", b"\x02c"):
            changed = copy.deepcopy(payload)
            for descriptor in changed["tensors"]:
                descriptor["args"][7] = prefix + b"h" * 64
            ipc.validate_payload(changed, "ctrl", UUID, RUN_ID, 0)
        payload["tensors"][0]["args"][7] = b"\x03c" + b"h" * 64
        with self.assertRaises(ValueError):
            ipc.validate_payload(payload, "ctrl", UUID, RUN_ID, 0)


if __name__ == "__main__":
    unittest.main()
