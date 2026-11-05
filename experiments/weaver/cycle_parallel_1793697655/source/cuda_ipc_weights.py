"""Rank-local CUDA IPC weight transport for colocated ZeRO-2 owners.

Only control metadata crosses Ray. Inference keeps its own Parameters and copies
from IPC views; it never aliases its live model to the trainer. Both ends retain
the existing full CPU SHA256 verification, so verification still incurs D2H.
This module does not initialize CUDA at import time and never silently falls back
to CPU transport. The caller must gate training with ``assert_idle`` and keep the
exporting actor alive until all matching inference ACKs have been received.

The private PyTorch reduce/rebuild protocol is validated explicitly. Its classes
and dtype objects are represented by whitelisted strings in the wire descriptor;
no Python pickle or callable is decoded by the receiver.
"""
from __future__ import annotations

import hashlib
import copy
import inspect
import json
import math
import os
import re
import socket
import threading
import uuid as uuid_module
import weakref

from training_topology import _publication_state, _weight_hash, _weight_version


PROTOCOL = "accerl.cuda_ipc.weights.v1"
INVENTORY_KEYS = ("kind", "version", "hash", "bytes", "tensor_count")
REBUILD_PARAMETERS = (
    "tensor_cls", "tensor_size", "tensor_stride", "tensor_offset", "storage_cls",
    "dtype", "storage_device", "storage_handle", "storage_size_bytes",
    "storage_offset_bytes", "requires_grad", "ref_counter_handle",
    "ref_counter_offset", "event_handle", "event_sync_required",
)
_DTYPES = {
    "torch.bool": "bool", "torch.uint8": "uint8", "torch.int8": "int8",
    "torch.int16": "int16", "torch.int32": "int32", "torch.int64": "int64",
    "torch.float16": "float16", "torch.bfloat16": "bfloat16",
    "torch.float32": "float32", "torch.float64": "float64",
    "torch.complex64": "complex64", "torch.complex128": "complex128",
}
_RECEIVERS = weakref.WeakKeyDictionary()
_RECEIVERS_LOCK = threading.RLock()


def _kind(value):
    value = {"world": "ctrl", "wm": "ctrl"}.get(value, value)
    if value not in ("policy", "ctrl"):
        raise ValueError("CUDA IPC is enabled only for Policy and World; Reward stays CPU-Ray")
    return value


def normalize_uuid(value):
    if not isinstance(value, str):
        raise ValueError("GPU UUID must be a string")
    value = value.removeprefix("GPU-").lower()
    if re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", value) is None:
        raise ValueError("Invalid physical GPU UUID")
    return value


def _integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("Invalid IPC integer: " + label)
    return value


def process_identity(pid=None):
    """Read Linux UID and PID birth; a reused PID cannot satisfy this identity."""
    from pathlib import Path
    pid = os.getpid() if pid is None else _integer(pid, "pid", 1)
    path = Path("/proc") / str(pid)
    fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
    if fields[0] in ("Z", "X"):
        raise RuntimeError("CUDA IPC owner is not alive")
    return {"pid": pid, "uid": path.stat().st_uid, "birth_ticks": int(fields[19]),
            "hostname": socket.gethostname()}


def _runtime(torch):
    from torch.multiprocessing import reductions
    if tuple(inspect.signature(reductions.rebuild_cuda_tensor).parameters) != REBUILD_PARAMETERS:
        raise RuntimeError("Unsupported installed PyTorch CUDA IPC rebuild protocol")
    return {"torch": str(torch.__version__), "cuda": str(torch.version.cuda)}


def _bound_device(torch, expected_uuid):
    expected_uuid = normalize_uuid(expected_uuid)
    if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
        raise RuntimeError("IPC model actor must expose exactly its bound physical GPU as cuda:0")
    raw = torch.cuda.get_device_properties(0).uuid
    actual = normalize_uuid(str(uuid_module.UUID(bytes=raw))
                            if isinstance(raw, bytes) and len(raw) == 16 else str(raw))
    if actual != expected_uuid:
        raise RuntimeError("CUDA IPC physical GPU UUID differs from actor binding")
    return actual


def _primitive(value):
    if value is None or isinstance(value, (str, bytes, bool)):
        return True
    if isinstance(value, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if isinstance(value, (tuple, list)):
        return all(_primitive(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _primitive(item) for key, item in value.items())
    return False


def metadata_fingerprint(payload):
    """Canonical metadata digest for idempotency, not the model-content digest."""
    def encode(value):
        if isinstance(value, bytes):
            return {"$bytes": value.hex()}
        if isinstance(value, dict):
            return {key: encode(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [encode(item) for item in value]
        return value
    if not _primitive(payload):
        raise ValueError("IPC wire payload contains tensors, objects or callables")
    return hashlib.sha256(json.dumps(encode(payload), sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def _handle(value, label, nullable=True):
    if value is None and nullable:
        return
    if not isinstance(value, bytes) or len(value) != 64:
        raise ValueError("Invalid CUDA IPC handle: " + label)


def _storage_handle(value):
    """Accept the verified native allocator formats without changing a byte.

    PyTorch 2.7 and 2.12 ``shareIpcHandle`` prepend versions ``1`` and ``2``
    respectively, plus cudaMalloc type ``'c'``, to the 64-byte CUDA handle.
    Both envelopes remain opaque and are passed through without truncation or
    rewriting. The receiver still requires identical actual Torch/CUDA runtimes.
    Expandable-segment and unknown-version envelopes remain unsupported.
    https://github.com/pytorch/pytorch/blob/v2.7.0/c10/cuda/CUDACachingAllocator.cpp
    https://github.com/pytorch/pytorch/blob/v2.12.0/c10/cuda/CUDACachingAllocator.cpp
    """
    if value is None:
        return
    if not isinstance(value, bytes):
        raise ValueError("Invalid CUDA IPC storage handle type")
    if len(value) == 64:
        return
    if len(value) == 66 and value[:2] in (b"\x01c", b"\x02c"):
        return
    raise ValueError("Unsupported CUDA IPC storage handle format")


def validate_reduction_descriptor(item):
    """Pure CPU schema/bounds checks; never opens a GPU mapping."""
    if not isinstance(item, dict) or set(item) != {"name", "dtype", "size", "stride", "args"}:
        raise ValueError("Malformed CUDA IPC tensor descriptor")
    if not isinstance(item["name"], str) or not item["name"]:
        raise ValueError("Missing CUDA IPC tensor name")
    if item["dtype"] not in _DTYPES:
        raise ValueError("Unsupported CUDA IPC dtype")
    size, stride = item["size"], item["stride"]
    if not isinstance(size, (list, tuple)) or not isinstance(stride, (list, tuple)) or len(size) != len(stride):
        raise ValueError("Invalid CUDA IPC tensor dimensions")
    for value in (*size, *stride):
        _integer(value, "shape/stride")
    args = item["args"]
    if not isinstance(args, (tuple, list)) or len(args) != len(REBUILD_PARAMETERS):
        raise ValueError("CUDA IPC reduce tuple length differs from installed protocol")
    if args[0] != "torch.Tensor" or args[4] not in ("torch.storage.TypedStorage", "torch.UntypedStorage"):
        raise ValueError("CUDA IPC tensor/storage class is not whitelisted")
    if list(args[1]) != list(size) or list(args[2]) != list(stride) or args[5] != item["dtype"]:
        raise ValueError("CUDA IPC shape/dtype descriptor mismatch")
    for index in (3, 6, 8, 9, 12):
        _integer(args[index], REBUILD_PARAMETERS[index])
    if args[6] != 0 or args[10] is not False:
        raise ValueError("CUDA IPC must describe detached tensors on local cuda:0")
    _storage_handle(args[7])
    if args[7] is None:
        # StorageSharing.cpp leaves these fields as Py_None when data()==null.
        # Preserve the actual reduction tuple for rebuild_cuda_tensor's empty
        # storage branch; never relax the nonempty storage/event protocol.
        if (args[8] != 0 or math.prod(size) != 0 or args[3] != 0 or args[9] != 0
                or args[11] is not None or args[12] != 0 or args[13] is not None
                or args[14] is not None):
            raise ValueError("Malformed empty CUDA IPC storage descriptor")
    else:
        if not isinstance(args[14], bool):
            raise ValueError("Nonempty CUDA IPC event synchronization flag must be bool")
        _handle(args[13], "storage event", nullable=False)
    if args[11] is not None and not isinstance(args[11], (bytes, str)):
        raise ValueError("Invalid CUDA IPC reference-counter descriptor")
    if not _primitive(item):
        raise ValueError("CUDA IPC descriptor contains an object")
    return item


def _encode_reduction(torch, name, tensor):
    from torch.multiprocessing import reductions
    function, args = reductions.reduce_tensor(tensor)
    if function is not reductions.rebuild_cuda_tensor or len(args) != len(REBUILD_PARAMETERS):
        raise RuntimeError("Expected the installed PyTorch CUDA IPC reduction")
    if args[0] is not torch.Tensor:
        raise RuntimeError("IPC weight tensor subclass is unsupported")
    storage_names = {torch.storage.TypedStorage: "torch.storage.TypedStorage",
                     torch.UntypedStorage: "torch.UntypedStorage"}
    if args[4] not in storage_names:
        raise RuntimeError("IPC weight storage subclass is unsupported")
    encoded = list(args)
    encoded[0], encoded[1], encoded[2] = "torch.Tensor", list(args[1]), list(args[2])
    encoded[4], encoded[5] = storage_names[args[4]], str(args[5])
    item = {"name": name, "dtype": str(tensor.dtype), "size": list(tensor.shape),
            "stride": list(tensor.stride()), "args": encoded}
    return validate_reduction_descriptor(item)


def _decode_reduction(torch, item):
    validate_reduction_descriptor(item)
    args = list(item["args"])
    args[0], args[1], args[2] = torch.Tensor, tuple(args[1]), tuple(args[2])
    args[4] = (torch.storage.TypedStorage if args[4] == "torch.storage.TypedStorage"
               else torch.UntypedStorage)
    args[5], args[6] = getattr(torch, _DTYPES[args[5]]), 0
    element_size = torch.empty((), dtype=args[5], device="cpu").element_size()
    if math.prod(item["size"]):
        last_element = args[3] + sum((n - 1) * stride for n, stride in zip(item["size"], item["stride"]))
        if (last_element + 1) * element_size > args[8]:
            raise ValueError("CUDA IPC tensor view exceeds its advertised storage")
    return args


def validate_payload(payload, kind, expected_uuid, run_id, rank):
    """Pure CPU metadata contract; runtime identity/CUDA checks follow separately."""
    kind = _kind(kind)
    if not isinstance(payload, dict) or payload.get("protocol") != PROTOCOL:
        raise ValueError("Unknown CUDA IPC wire protocol")
    if not _primitive(payload):
        raise ValueError("CUDA IPC payload must contain only control metadata")
    if not isinstance(run_id, str) or not run_id or payload.get("run_id") != run_id:
        raise ValueError("CUDA IPC run identity differs")
    if payload.get("kind") != kind or payload.get("rank") != rank or isinstance(rank, bool):
        raise ValueError("CUDA IPC kind/rank differs from receiver")
    _integer(rank, "rank")
    _integer(payload["rank"], "source rank")
    if normalize_uuid(payload.get("uuid")) != normalize_uuid(expected_uuid):
        raise ValueError("CUDA IPC payload belongs to a different physical GPU")
    token = payload.get("token")
    if not isinstance(token, str) or re.fullmatch(r"[0-9a-f]{32}", token) is None:
        raise ValueError("Invalid CUDA IPC publication token")
    inventory = payload.get("inventory")
    if (not isinstance(inventory, dict) or set(inventory) not in
            (set(INVENTORY_KEYS), set(INVENTORY_KEYS) | {"publication"})):
        raise ValueError("Invalid CUDA IPC model inventory")
    if inventory["kind"] != kind:
        raise ValueError("CUDA IPC inventory kind differs")
    _weight_version(inventory["version"])
    for key in ("bytes", "tensor_count"):
        _integer(inventory[key], key, 1)
    if not isinstance(inventory["hash"], str) or re.fullmatch(r"[0-9a-f]{64}", inventory["hash"]) is None:
        raise ValueError("CUDA IPC inventory requires full SHA256")
    source = payload.get("source")
    if not isinstance(source, dict) or set(source) != {"uid", "pid", "birth_ticks", "hostname"}:
        raise ValueError("Invalid CUDA IPC source identity")
    for key in ("uid", "pid", "birth_ticks"):
        _integer(source[key], key, 0 if key == "uid" else 1)
    if not isinstance(source["hostname"], str) or not source["hostname"]:
        raise ValueError("Missing CUDA IPC source hostname")
    if not isinstance(payload.get("runtime"), dict) or set(payload["runtime"]) != {"torch", "cuda"}:
        raise ValueError("Missing CUDA IPC runtime identity")
    _handle(payload.get("ready_event"), "publication-ready event", nullable=False)
    tensors = payload.get("tensors")
    if not isinstance(tensors, list) or len(tensors) != inventory["tensor_count"]:
        raise ValueError("CUDA IPC tensor count differs")
    names = [validate_reduction_descriptor(item)["name"] for item in tensors]
    if names != sorted(names) or len(names) != len(set(names)):
        raise ValueError("CUDA IPC tensor names must be unique and sorted")
    if "publication" in inventory:
        publication = inventory["publication"]
        required = {"parameter_names", "buffer_names", "bundle_version"}
        if (kind != "ctrl" or not isinstance(publication, dict) or not required <= set(publication)
                or set(publication) - required - {"backend_identity", "bundle_metadata", "cache_version"}):
            raise ValueError("Invalid CUDA IPC backend publication metadata")
        if _weight_version(publication["bundle_version"]) != inventory["version"]:
            raise ValueError("CUDA IPC bundle version differs from model version")
        selected = []
        for key in ("parameter_names", "buffer_names"):
            items = publication[key]
            if (not isinstance(items, list) or any(not isinstance(name, str) or not name for name in items)
                    or items != sorted(set(items))):
                raise ValueError("Invalid CUDA IPC backend publication names")
            selected.extend(items)
        if sorted(selected) != names:
            raise ValueError("CUDA IPC backend inventory differs from published tensors")
    return copy.deepcopy(inventory)


class CudaIPCWeightExporter:
    """One outstanding publication; source storage stays alive until valid ACK."""
    def __init__(self, module, kind, uuid, run_id, rank, *, world_backend=None):
        self.module, self.kind, self.uuid = module, _kind(kind), normalize_uuid(uuid)
        self.world_backend = world_backend
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("CUDA IPC exporter requires an explicit run ID")
        self.run_id, self.rank = run_id, _integer(rank, "rank")
        self._active = None
        self._last_finished_version = -1
        self._lock = threading.RLock()

    @property
    def active(self):
        return self._active is not None

    def assert_idle(self):
        if self._active is not None:
            raise RuntimeError("Training is forbidden while an IPC publication lacks its matching ACK")

    def begin(self, version):
        import torch
        with self._lock:
            self.assert_idle()
            version = _weight_version(version)
            if version <= self._last_finished_version:
                raise ValueError("CUDA IPC publication versions must strictly increase")
            if os.environ.get("ACCERL_RUN_ID", self.run_id) != self.run_id:
                raise RuntimeError("Exporter run environment differs")
            _bound_device(torch, self.uuid)
            runtime = _runtime(torch)
            selected, publication = _publication_state(
                self.module, self.kind, version, self.world_backend, for_export=True)
            refs = []
            for name, tensor in selected:
                if tensor.device.type != "cuda" or tensor.device.index != 0 or tensor.layout != torch.strided:
                    raise RuntimeError("IPC publication requires dense, full, local CUDA tensors: " + name)
                value = tensor.detach()
                if not value.is_contiguous():
                    value = value.contiguous()
                refs.append((name, value))
            # Install the hold before reduce_tensor: a partial export failure is
            # fail-closed, because some refcounter credits may already exist.
            token = uuid_module.uuid4().hex
            self._active = {"token": token, "refs": refs, "payload": None, "event": None}
            torch.cuda.synchronize(0)
            inventory = {"kind": self.kind, "version": version,
                         "hash": _weight_hash(refs, self.kind, publication),
                         "bytes": sum(value.numel() * value.element_size() for _, value in refs),
                         "tensor_count": len(refs)}
            if publication is not None:
                inventory["publication"] = publication
            event = torch.cuda.Event(interprocess=True, enable_timing=False)
            event.record(torch.cuda.current_stream(0))
            event.synchronize()
            self._active["event"] = event
            tensors = [_encode_reduction(torch, name, value) for name, value in refs]
            payload = {"protocol": PROTOCOL, "kind": self.kind, "uuid": self.uuid,
                       "rank": self.rank, "run_id": self.run_id, "token": token,
                       "source": process_identity(), "runtime": runtime,
                       "inventory": inventory, "ready_event": event.ipc_handle(), "tensors": tensors}
            validate_payload(payload, self.kind, self.uuid, self.run_id, self.rank)
            self._active["payload"] = payload
            return payload

    def describe(self, token):
        with self._lock:
            if self._active is None or token != self._active["token"] or self._active["payload"] is None:
                raise RuntimeError("No complete matching CUDA IPC publication")
            return copy.deepcopy(self._active["payload"]["inventory"])

    def finish(self, token, ack):
        import torch
        with self._lock:
            expected = self.describe(token)
            if ack != expected:
                raise RuntimeError("CUDA IPC ACK differs; retain source references and abort training")
            self._last_finished_version = expected["version"]
            self._active = None
            torch.cuda.ipc_collect()
            return expected


def apply_cuda_ipc_weights(module, payload, kind, expected_uuid, run_id, rank, *, world_backend=None):
    """Import once, copy to independent model, verify all bytes, then ACK.

    A retry of the most recently ACKed, byte-identical descriptor returns its
    cached ACK without rebuilding IPC storage/refcounter credits a second time.
    A failed partial apply poisons this receiver; callers must fail pending work
    and stop the experiment rather than serving partially updated weights.
    """
    import torch
    kind = _kind(kind)
    inventory = validate_payload(payload, kind, expected_uuid, run_id, rank)
    selected, publication = _publication_state(module, kind, inventory["version"], world_backend)
    if inventory.get("publication") != publication:
        raise ValueError("CUDA IPC publication differs from the destination backend bundle")
    fingerprint = metadata_fingerprint(payload)
    with _RECEIVERS_LOCK:
        state = _RECEIVERS.setdefault(module, {"lock": threading.RLock(), "last": {}, "failed": False})
    with state["lock"]:
        if state["failed"]:
            raise RuntimeError("CUDA IPC receiver previously failed and cannot serve weights")
        previous = state["last"].get(kind)
        if previous and previous["token"] == payload["token"]:
            if previous["fingerprint"] != fingerprint:
                raise ValueError("Reused CUDA IPC token has changed metadata")
            return copy.deepcopy(previous["ack"])
        if previous and inventory["version"] <= previous["ack"]["version"]:
            raise ValueError("Stale CUDA IPC publication version")
        _bound_device(torch, expected_uuid)
        if payload["runtime"] != _runtime(torch):
            raise RuntimeError("CUDA IPC trainer/inference Torch or CUDA runtime differs")
        source = payload["source"]
        if source["hostname"] != socket.gethostname() or source["uid"] != os.getuid():
            raise RuntimeError("CUDA IPC requires the same host and UID")
        if process_identity(source["pid"]) != source:
            raise RuntimeError("CUDA IPC source UID/PID/birth identity has changed")
        if os.environ.get("ACCERL_RUN_ID", run_id) != run_id:
            raise RuntimeError("Inference run environment differs")
        source_environment = dict(item.split("=", 1) for item in
                                  open("/proc/%d/environ" % source["pid"], "rb").read().decode(errors="replace").split("\0")
                                  if "=" in item)
        if source_environment.get("ACCERL_RUN_ID") != run_id:
            raise RuntimeError("CUDA IPC source process run identity differs")
        source_mask = source_environment.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if len(source_mask) != 1 or normalize_uuid(source_mask[0]) != normalize_uuid(expected_uuid):
            raise RuntimeError("CUDA IPC source process physical GPU binding differs")
        if [name for name, _ in selected] != [item["name"] for item in payload["tensors"]]:
            raise ValueError("CUDA IPC keys differ from the original mutable-weight contract")
        decoded = []
        total_bytes = 0
        for (name, destination), item in zip(selected, payload["tensors"]):
            if destination.device.type != "cuda" or destination.device.index != 0:
                raise ValueError("CUDA IPC destination is not on its bound GPU: " + name)
            if list(destination.shape) != item["size"] or str(destination.dtype) != item["dtype"]:
                raise ValueError("CUDA IPC source/destination shape or dtype differs: " + name)
            total_bytes += destination.numel() * destination.element_size()
            decoded.append(_decode_reduction(torch, item))
        if total_bytes != inventory["bytes"]:
            raise ValueError("CUDA IPC inventory byte count differs")
        shared = []
        source_tensor = None
        try:
            state["failed"] = True
            # Drain old inference reads before mutating its independent model.
            torch.cuda.synchronize(0)
            ready = torch.cuda.Event.from_ipc_handle(0, payload["ready_event"])
            torch.cuda.current_stream(0).wait_event(ready)
            ready.synchronize()
            from torch.multiprocessing.reductions import rebuild_cuda_tensor
            for args in decoded:
                shared.append(rebuild_cuda_tensor(*args))
            with torch.no_grad():
                for (_, destination), source_tensor in zip(selected, shared):
                    destination.copy_(source_tensor)
            torch.cuda.synchronize(0)
            loaded_hash = _weight_hash(selected, kind, publication)
            if loaded_hash != inventory["hash"]:
                raise RuntimeError("Loaded CUDA IPC inference weights differ from source SHA256")
            ack = dict(inventory, hash=loaded_hash)
            # Only this ACK allows exporter.finish and the next optimizer step.
            state["last"][kind] = {"token": payload["token"], "fingerprint": fingerprint, "ack": ack}
            state["failed"] = False
            return copy.deepcopy(ack)
        finally:
            source_tensor = None
            shared.clear()
            # Consumers drop every imported view before the RPC returns ACK.
            # The producer retains its originals/event until finish(token,ACK).

