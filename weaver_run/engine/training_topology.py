"""Explicit single-node GPU bindings and identity-preserving global batch shards."""
from __future__ import annotations

import asyncio
import math
import os
from pathlib import Path
import subprocess
from collections.abc import Mapping


DUAL_LAYOUT = "dual_train_policy_reward"
FOUR_LAYOUT = "four_rank_six_roles"
SPLIT_LAYOUT = "four_rank_split_train"
DEFAULT_MODEL_KINDS = ("policy", "ctrl", "reward")


def active_model_kinds(active_kinds=None):
    """The legacy ctrl role owns either CTRL or an entire world backend bundle."""
    kinds = DEFAULT_MODEL_KINDS if active_kinds is None else tuple(active_kinds)
    if kinds not in (DEFAULT_MODEL_KINDS, ("policy", "ctrl"), ("policy", "reward")):
        raise ValueError("Active model kinds must be the explicit Policy/World/Reward training plan")
    return kinds


def inference_model_kinds(active_kinds=None, inference_kinds=None):
    kinds = active_model_kinds(active_kinds)
    inferred = DEFAULT_MODEL_KINDS if kinds == ("policy", "reward") else kinds
    result = inferred if inference_kinds is None else tuple(inference_kinds)
    if result != inferred:
        raise ValueError("Inference roles must match the explicit frozen/trainable World plan")
    return result


def _mps_runtime_env():
    """Explicitly propagate the selected existing MPS server to Ray workers."""
    if os.environ.get("ACCERL_USE_MPS") != "1":
        return {}
    pipe = os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
    if not pipe or not os.path.isabs(pipe):
        raise ValueError("MPS requires an explicit absolute CUDA_MPS_PIPE_DIRECTORY")
    env = {"ACCERL_USE_MPS": "1", "CUDA_MPS_PIPE_DIRECTORY": pipe}
    if os.environ.get("CUDA_MPS_LOG_DIRECTORY"):
        env["CUDA_MPS_LOG_DIRECTORY"] = os.environ["CUDA_MPS_LOG_DIRECTORY"]
    return env


def parse_gpu_ids(value):
    ids = [part.strip() for part in value.split(",") if part.strip()]
    if len(ids) != 4 or len(set(ids)) != 4 or any(not part.isdigit() for part in ids):
        raise ValueError("The fixed training layouts require four distinct physical GPU indices")
    return ids


def build_bindings(ids):
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=10,
    )
    physical = dict(line.strip().split(", ", 1) for line in result.stdout.splitlines())
    missing = set(ids) - physical.keys()
    if missing:
        raise ValueError("Unknown physical GPUs: %s" % sorted(missing))
    return [{"index": index, "uuid": physical[index], "resource": "accerl_gpu_" + index}
            for index in ids]


def ray_init_options(bindings, layout=DUAL_LAYOUT, active_kinds=None, inference_kinds=None):
    # Ray's aggregate GPU resource cannot pin a specific physical index. Reserve
    # one named resource per card and verify the CUDA mask against its UUID.
    # MPS remaps the server's physical indices, so MPS clients use UUID masks;
    # verify_bound_gpu fixes this actor's DeepSpeed NVML lookup after validation.
    # CPU actors inherit an empty mask. This cluster is local to this invocation.
    if layout not in (DUAL_LAYOUT, FOUR_LAYOUT, SPLIT_LAYOUT):
        raise ValueError("Unknown fixed GPU layout: %s" % layout)
    kinds = active_model_kinds(active_kinds)
    infer = inference_model_kinds(kinds, inference_kinds)
    if layout != SPLIT_LAYOUT and kinds != DEFAULT_MODEL_KINDS:
        raise ValueError("Optional model roles require the split trainer layout")
    capacity = float(len(kinds) + len(infer)) if layout in (FOUR_LAYOUT, SPLIT_LAYOUT) else 1.0
    import_path = {}
    if layout == SPLIT_LAYOUT:
        import_path["PYTHONPATH"] = os.pathsep.join(filter(None, (
            str(Path(__file__).resolve().parent), os.environ.get("PYTHONPATH", ""))))
    return {
        "num_gpus": 0,
        "resources": {binding["resource"]: capacity for binding in bindings},
        "runtime_env": {"env_vars": {
            **_mps_runtime_env(),
            **import_path,
            "CUDA_VISIBLE_DEVICES": "",
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
        }},
    }


def actor_options(bindings, role, rank=0, layout=DUAL_LAYOUT, active_kinds=None, inference_kinds=None):
    kinds = active_model_kinds(active_kinds)
    infer = inference_model_kinds(kinds, inference_kinds)
    if layout != SPLIT_LAYOUT and kinds != DEFAULT_MODEL_KINDS:
        raise ValueError("Optional model roles require the split trainer layout")
    if layout in (FOUR_LAYOUT, SPLIT_LAYOUT):
        allowed_roles = (tuple(kind + "_trainer" for kind in kinds) + infer
                         if layout == SPLIT_LAYOUT else ("trainer", "policy", "reward", "ctrl"))
        if role not in allowed_roles:
            raise ValueError("Unknown four-card model role: %s" % role)
        if not isinstance(rank, int) or not 0 <= rank < len(bindings):
            raise ValueError("Four-card actor rank must index its physical GPU")
        slot = rank
        amount = 3.0 if role == "trainer" else 1.0
    elif layout == DUAL_LAYOUT:
        slot = rank if role == "trainer" else {"policy_reward": 2, "ctrl": 3}[role]
        amount = 1.0
    else:
        raise ValueError("Unknown fixed GPU layout: %s" % layout)
    binding = bindings[slot]
    mps_env = _mps_runtime_env()
    return {
        "num_gpus": 0,
        "max_restarts": 0,
        "max_task_retries": 0,
        "resources": {binding["resource"]: amount},
        "runtime_env": {"env_vars": {
            **mps_env,
            "CUDA_VISIBLE_DEVICES": binding["uuid"] if mps_env else binding["index"],
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "ACCERL_EXPECTED_GPU_UUID": binding["uuid"],
            "ACCERL_PHYSICAL_GPU": binding["index"],
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
        }},
    }


def _weight_kind(kind):
    kind = {"world": "ctrl", "wm": "ctrl"}.get(kind, kind)
    if kind not in ("policy", "reward", "ctrl"):
        raise ValueError("Unknown model weight kind: %s" % kind)
    return kind


def _publication_spec(module, kind, world_backend=None):
    """Resolve an explicitly declared world bundle; never infer mutable buffers.

    The inventory names are relative to this exact model root. Declaring an
    inventory cannot omit a trainable tensor, including an integrated head.
    Backends may additionally publish frozen parameters and mutable buffers.
    """
    if world_backend is not None and kind != "ctrl":
        raise ValueError("World publication inventories apply only to the ctrl role")
    if kind != "ctrl":
        return None
    backend = world_backend if world_backend is not None else getattr(module, "world_backend", None)
    if backend is None and callable(getattr(module, "publication_inventory", None)):
        backend = module
    if backend is None:
        return None
    inventory = backend if isinstance(backend, Mapping) else backend.publication_inventory()
    if inventory is None and getattr(backend, "name", None) == "ctrl":
        return None  # Explicit CTRL compatibility adapter retains its original selector.
    if not isinstance(inventory, Mapping):
        raise ValueError("World publication inventory must be a mapping")
    required = {"parameter_names", "buffer_names"}
    allowed = required | {"backend_identity", "bundle_metadata", "cache_version"}
    if not required <= set(inventory) or set(inventory) - allowed:
        raise ValueError("World publication inventory has invalid fields")
    result = {}
    for key in ("parameter_names", "buffer_names"):
        names = inventory[key]
        if (not isinstance(names, (list, tuple)) or any(not isinstance(name, str) or not name for name in names)
                or len(names) != len(set(names))):
            raise ValueError("World publication names must be distinct nonempty strings")
        result[key] = sorted(names)
    if set(result["parameter_names"]) & set(result["buffer_names"]):
        raise ValueError("World publication parameter and buffer names overlap")
    for key in ("backend_identity", "bundle_metadata", "cache_version"):
        if key in inventory:
            result[key] = inventory[key]
    _validate_publication_metadata(result)
    # Copy nested metadata so callers cannot mutate an outstanding descriptor.
    import json
    return json.loads(json.dumps(result, sort_keys=True, separators=(",", ":")))


def _validate_publication_metadata(value):
    """Keep backend metadata on the same primitive wire subset as CUDA IPC."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _validate_publication_metadata(item)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _validate_publication_metadata(item)
        return
    raise ValueError("World publication metadata must contain only primitive values")


def _selected_weights(module, kind, world_backend=None):
    """Match the production broadcast contract, excluding frozen parameters."""
    spec = _publication_spec(module, kind, world_backend)
    parameters = dict(module.named_parameters(recurse=True))
    if spec is not None:
        buffers = dict(module.named_buffers(recurse=True))
        selected_params, selected_buffers = set(spec["parameter_names"]), set(spec["buffer_names"])
        if selected_params - parameters.keys() or selected_buffers - buffers.keys():
            raise ValueError("World publication inventory names are absent from the model")
        missing = {name for name, tensor in parameters.items() if tensor.requires_grad} - selected_params
        if missing:
            raise ValueError("World publication inventory omits trainable tensors: %s" % sorted(missing))
        weights = {name: parameters[name] for name in selected_params}
        weights.update({name: buffers[name] for name in selected_buffers})
        if not weights:
            raise ValueError("No mutable weights found for %s" % kind)
        return sorted(weights.items())
    weights = {name: tensor for name, tensor in parameters.items()
               if tensor.requires_grad}
    if kind != "ctrl":
        for name, tensor in module.named_buffers(recurse=True):
            if name in weights:
                raise ValueError("Parameter and buffer share a weight name: %s" % name)
            weights[name] = tensor
    if not weights:
        raise ValueError("No mutable weights found for %s" % kind)
    return sorted(weights.items())


def _publication_state(module, kind, version, world_backend=None, *, for_export=False):
    spec = _publication_spec(module, kind, world_backend)
    weights = _selected_weights(module, kind, spec if spec is not None else world_backend)
    publication = dict(spec, bundle_version=_weight_version(version)) if spec is not None else None
    backend = world_backend if world_backend is not None else getattr(module, "world_backend", module)
    selector = getattr(backend, "publication_tensors", None)
    if for_export and spec is not None and callable(selector):
        import torch
        sources = selector()
        if not isinstance(sources, Mapping) or set(sources) != {name for name, _ in weights}:
            raise ValueError("Backend publication tensors differ from the declared inventory")
        for name, destination in weights:
            source = sources[name]
            if (not isinstance(source, torch.Tensor) or source.shape != destination.shape
                    or source.dtype != destination.dtype or source.layout != torch.strided):
                raise ValueError("Backend publication tensor shape/dtype differs: %s" % name)
        weights = [(name, sources[name]) for name, _ in weights]
    return weights, publication


def _weight_hash(weights, kind, publication=None):
    """Hash raw CPU bytes, including bfloat16, one tensor at a time."""
    import hashlib
    import json
    import torch
    digest = hashlib.sha256()
    digest.update((kind + "\n").encode("utf-8"))
    if publication is not None:
        _validate_publication_metadata(publication)
        digest.update(json.dumps(publication, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n")
    for name, tensor in weights:
        # reshape handles scalar buffers; uint8 avoids NumPy's lack of bf16.
        cpu_tensor = tensor.detach().to(device="cpu").contiguous()
        descriptor = json.dumps([name, list(cpu_tensor.shape), str(cpu_tensor.dtype)],
                                ensure_ascii=True, separators=(",", ":"))
        digest.update(descriptor.encode("utf-8") + b"\n")
        digest.update(cpu_tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _weight_version(version):
    if isinstance(version, bool) or not isinstance(version, int) or version < 0:
        raise ValueError("Weight version must be a nonnegative optimizer update count")
    return version


def export_cpu_weights(module, kind, version, *, world_backend=None):
    """Create an immutable CPU snapshot for a same-card Ray inference actor.

    ZeRO-2 keeps full parameter replicas. Copying their mutable tensors over
    Ray avoids an NCCL broadcast communicator containing two processes on the
    same physical GPU. Legacy digests exclude version. Explicit backend bundles
    also hash their identity and bundle version; ranks at that version must match.
    """
    kind = _weight_kind(kind)
    version = _weight_version(version)
    weights, publication = _publication_state(module, kind, version, world_backend, for_export=True)
    tensors = {name: tensor.detach().to(device="cpu", copy=True).contiguous()
               for name, tensor in weights}
    return {"kind": kind, "version": version,
            **({"publication": publication} if publication is not None else {}),
            "hash": _weight_hash(sorted(tensors.items()), kind, publication),
            "bytes": sum(tensor.numel() * tensor.element_size() for tensor in tensors.values()),
            "tensor_count": len(tensors), "tensors": tensors}


def apply_cpu_weights(module, payload, kind, *, world_backend=None):
    """Validate the complete snapshot before copying, then verify loaded bytes."""
    import torch
    kind = _weight_kind(kind)
    if not isinstance(payload, dict) or payload.get("kind") != kind:
        raise ValueError("CPU weight payload kind differs from the destination model")
    version = _weight_version(payload.get("version"))
    tensors = payload.get("tensors")
    if not isinstance(tensors, dict):
        raise ValueError("CPU weight payload is missing its tensor mapping")
    selected, publication = _publication_state(module, kind, version, world_backend)
    if payload.get("publication") != publication:
        raise ValueError("CPU weight publication differs from the destination backend bundle")
    if set(tensors) != {name for name, _ in selected}:
        raise ValueError("CPU weight payload keys differ from the destination model")
    for name, destination in selected:
        source = tensors[name]
        if not isinstance(source, torch.Tensor) or source.device.type != "cpu":
            raise ValueError("CPU weight payload tensor is not on CPU: %s" % name)
        if source.layout != torch.strided:
            raise ValueError("CPU weight payload tensor must be dense: %s" % name)
        if source.shape != destination.shape or source.dtype != destination.dtype:
            raise ValueError("CPU weight payload shape or dtype differs: %s" % name)
    nbytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
    if payload.get("bytes") != nbytes or payload.get("tensor_count") != len(selected):
        raise ValueError("CPU weight payload byte or tensor count differs")
    source_hash = _weight_hash(sorted(tensors.items()), kind, publication)
    if payload.get("hash") != source_hash:
        raise ValueError("CPU weight payload hash differs from its tensor contents")
    with torch.no_grad():
        for name, destination in selected:
            destination.copy_(tensors[name])
    loaded_hash = _weight_hash(selected, kind, publication)
    if loaded_hash != source_hash:
        raise RuntimeError("Loaded inference weights differ from the source snapshot")
    return {"kind": kind, "version": version, "hash": loaded_hash,
            **({"publication": publication} if publication is not None else {}),
            "bytes": nbytes, "tensor_count": len(selected)}


def _install_mps_nvml_mapping(physical_index):
    """Keep DeepSpeed's free-memory query on this already verified physical card.

    DeepSpeed 0.19 converts CUDA_VISIBLE_DEVICES to integers. MPS needs UUIDs
    to avoid server ordinal remapping, so replace only the current process's
    accelerator instance lookup. Do not edit the installed package or daemon.
    """
    from deepspeed.accelerator import get_accelerator
    accelerator = get_accelerator()
    if not callable(getattr(accelerator, "_get_nvml_gpu_id", None)):
        raise RuntimeError("The MPS binding requires DeepSpeed's CUDA NVML mapper")
    if not isinstance(physical_index, str) or not physical_index.isdigit():
        raise RuntimeError("Missing numeric physical GPU identity for the MPS actor")
    physical_index = int(physical_index)

    def physical_nvml_id(local_index):
        if local_index != 0:
            raise RuntimeError("An MPS model actor may query only its bound CUDA device 0")
        return physical_index

    accelerator._get_nvml_gpu_id = physical_nvml_id


def verify_bound_gpu():
    import torch
    expected = os.environ.get("ACCERL_EXPECTED_GPU_UUID")
    if not expected:
        return {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")}
    if torch.cuda.device_count() != 1:
        raise RuntimeError("A bound model actor must see exactly one GPU")
    torch.cuda.set_device(0)
    actual = str(torch.cuda.get_device_properties(0).uuid)
    if actual.removeprefix("GPU-").lower() != expected.removeprefix("GPU-").lower():
        raise RuntimeError("GPU binding mismatch: expected %s, got %s" % (expected, actual))
    if os.environ.get("ACCERL_USE_MPS") == "1":
        _mps_runtime_env()
        _install_mps_nvml_mapping(os.environ.get("ACCERL_PHYSICAL_GPU"))
    binding = {"physical_gpu": os.environ["ACCERL_PHYSICAL_GPU"], "uuid": actual,
               "cuda_local_device": 0}
    print("[GPU binding] %s" % binding, flush=True)
    return binding


def successful_updates(before_steps, before_skips, engine):
    return max(0, int(engine.global_steps) - before_steps
               - (int(engine.skipped_steps) - before_skips))


def combine_wm_metrics(metrics):
    """Equal rank batches: average losses/ratios, sum confusion counts."""
    combined = dict(metrics[0])
    for key in ('ctrl_loss', 'reward_loss', 'ctrl_grad_norm', 'ctrl_clip_ratio',
                'reward_grad_norm', 'reward_clip_ratio', 'reward_pos_ratio_batch'):
        values = [item[key] for item in metrics if key in item]
        if values:
            combined[key] = sum(values) / len(values)
    for key in ('reward_tp', 'reward_tn', 'reward_fp', 'reward_fn'):
        if key in combined:
            combined[key] = sum(item[key] for item in metrics)
    return combined


def shard_metadata(metadata, start, end, rank, world_size):
    result = dict(metadata)
    result["sample_ids"] = metadata["sample_ids"][start:end]
    result.update(data_parallel_rank=rank, data_parallel_world_size=world_size,
                  global_batch_size=len(metadata["sample_ids"]), shard_start=start, shard_end=end)
    if len(result["sample_ids"]) != end - start:
        raise ValueError("Incomplete sample identity coverage for batch shard")
    return result


def shard_batch(batch, kind, rank, local_size, world_size):
    start, end = rank * local_size, (rank + 1) * local_size
    if len(batch[0]) != local_size * world_size:
        raise ValueError("Global Replay batch size differs from the distributed recipe")
    data = tuple(value[start:end] for value in batch[:-1])
    if kind == "policy":
        metadata = shard_metadata(batch[-1], start, end, rank, world_size)
    else:
        # Aggregate episode statistics describe the shared global sample;
        # identities and all actual tensors describe only this rank's shard.
        metadata = dict(batch[-1])
        metadata["capture_meta"] = shard_metadata(metadata["capture_meta"], start, end, rank, world_size)
        if "backend_samples" in metadata:
            backend_samples = metadata["backend_samples"]
            if not isinstance(backend_samples, (list, tuple)) or len(backend_samples) != len(batch[0]):
                raise ValueError("Backend samples must align with every global Replay sample")
            metadata["backend_samples"] = backend_samples[start:end]
        metadata["sample_stats_scope"] = "global_batch"
    return (*data, metadata)


class GlobalBatchCoordinator:
    """Each Replay sample is drawn once and consumed as disjoint rank shards.

    At most one unclaimed global batch per lane is held. A faster rank's next
    prefetch waits until all ranks have claimed the preceding batch. Split
    World/Reward groups claim identical shards from the same WM global draw;
    both model groups must finish claiming before either can draw another.
    """
    def __init__(self, policy_replay, wm_replay, world_size, wm_consumers=("combined",),
                 pipeline_enabled=False, world_streaming=False, world_micro_batch=4):
        if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
            raise ValueError("Distributed Replay world size must be a positive integer")
        self.wm_consumers = tuple(wm_consumers)
        if (not self.wm_consumers or len(set(self.wm_consumers)) != len(self.wm_consumers)
                or any(consumer not in ("combined", "ctrl", "reward") for consumer in self.wm_consumers)
                or ("combined" in self.wm_consumers and len(self.wm_consumers) != 1)):
            raise ValueError("WM consumers must be combined or distinct split model roles")
        self.replays = {"policy": policy_replay, "wm": wm_replay}
        self.world_size = world_size
        self.conditions = {kind: asyncio.Condition() for kind in self.replays}
        self.pending = {kind: None for kind in self.replays}
        self.claimed = {kind: set() for kind in self.replays}
        self.local_sizes = {}
        self.pipeline_enabled = bool(pipeline_enabled)
        self.failures = {kind: None for kind in self.replays}
        self.wm_identities = {}
        self.wm_reservation_id = None
        self.world_streaming = bool(world_streaming)
        self.world_micro_batch = int(world_micro_batch)
        if self.world_streaming and (not self.pipeline_enabled or self.wm_consumers not in (('ctrl', 'reward'), ('ctrl',))
                                     or self.world_size != 4 or self.world_micro_batch != 4):
            raise ValueError("Online WEAVER streaming requires four ranks, ctrl/reward consumers and micro4")
        self.wm_streams = {}
        self.wm_stream_owners = set()

    def _check_failure(self, kind):
        if self.failures[kind] is not None:
            raise RuntimeError("Distributed %s Replay cohort failed: %s" %
                               (kind, self.failures[kind])) from self.failures[kind]

    async def _release_wm_ticket(self):
        ticket = self.pending["wm"]
        # Clear before the RPC: a failed release is fatal, not a license to
        # reuse the same identities or silently draw a replacement batch.
        self.pending["wm"] = None
        if ticket is not None:
            await self.replays["wm"].release_windows.remote(ticket["ticket_id"])

    def _validate_projection(self, result, rank, local_size, consumer):
        if not isinstance(result, (tuple, list)) or len(result) != 5:
            raise ValueError("Window projection must return the five-field Replay contract")
        if any(len(value) != local_size for value in result[:-1]):
            raise ValueError("Projected fields do not match the requested rank shard")
        metadata = result[-1]["capture_meta"]
        expected = {"data_parallel_rank": rank, "data_parallel_world_size": self.world_size,
                    "global_batch_size": local_size * self.world_size,
                    "shard_start": rank * local_size, "shard_end": (rank + 1) * local_size}
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("Projected shard metadata differs from the distributed recipe")
        identity = (metadata["reservation_id"], tuple(metadata["sample_ids"]))
        if len(identity[1]) != local_size:
            raise ValueError("Projected sample identity coverage is incomplete")
        if self.wm_reservation_id is None:
            self.wm_reservation_id = identity[0]
        elif self.wm_reservation_id != identity[0]:
            raise ValueError("WM ranks received different global reservation IDs")
        previous = self.wm_identities.setdefault(rank, identity)
        if identity != previous:
            raise ValueError("World and Reward received different reserved sample identities")
        if consumer == "reward":
            if (getattr(result[0], "ndim", None) != 4 or result[0].shape[1] != 3
                    or getattr(result[1], "shape", None) != (local_size, 0)
                    or "backend_samples" in result[-1]):
                raise ValueError("Reward RPC must contain only primary images, labels, text and metadata")

    async def _sample(self, kind, rank, local_size, consumer=None):
        if (isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < self.world_size
                or isinstance(local_size, bool) or not isinstance(local_size, int) or local_size < 1):
            raise ValueError("Invalid distributed Replay shard request")
        if kind == "wm" and consumer not in self.wm_consumers:
            raise ValueError("Unknown WM Replay consumer: %s" % consumer)
        identity = (consumer, rank) if kind == "wm" else rank
        expected_claims = self.world_size * (len(self.wm_consumers) if kind == "wm" else 1)
        condition = self.conditions[kind]
        async with condition:
            while identity in self.claimed[kind]:
                self._check_failure(kind)
                await condition.wait()
            self._check_failure(kind)
            try:
                if self.pending[kind] is None:
                    global_size = local_size * self.world_size
                    if self.pipeline_enabled and kind == "wm":
                        ticket = await self.replays[kind].reserve_windows.remote(global_size)
                        self.pending[kind] = ticket
                        if ticket.get("global_batch_size") != global_size:
                            raise ValueError("Reserved window ticket has an unexpected global size")
                    else:
                        self.pending[kind] = await self.replays[kind].sample.remote(global_size)
                        if self.pipeline_enabled and kind == "policy":
                            # Credits count one global draw, not individual ranks or
                            # successful optimizer updates. Never multiply by four.
                            credits = global_size
                            metadata = self.pending[kind][-1]
                            if 'fresh_reservation_token' in metadata:
                                counts = metadata.get('source_counts', {})
                                if (set(counts) != {'real', 'imagined'} or
                                        any(type(v) is not int or v < 0 for v in counts.values()) or
                                        sum(counts.values()) != global_size):
                                    raise ValueError('FRESH production credits require exact global source counts')
                                credits = counts['imagined']
                            await self.replays["wm"].policy_drawn.remote(credits)
                    self.local_sizes[kind] = local_size
                elif self.local_sizes[kind] != local_size:
                    raise ValueError("All ranks must request equal local batches")
                if self.pipeline_enabled and kind == "wm":
                    result = await self.replays[kind].project_windows.remote(
                        self.pending[kind]["ticket_id"], rank, local_size, consumer)
                    self._validate_projection(result, rank, local_size, consumer)
                else:
                    result = shard_batch(self.pending[kind], kind, rank, local_size, self.world_size)
                self.claimed[kind].add(identity)
                if len(self.claimed[kind]) == expected_claims:
                    if self.pipeline_enabled and kind == "wm":
                        await self._release_wm_ticket()
                        self.wm_identities.clear()
                        self.wm_reservation_id = None
                    else:
                        self.pending[kind] = None
                    self.claimed[kind].clear()
                    condition.notify_all()
                return result
            except (Exception, asyncio.CancelledError) as error:
                self.failures[kind] = error
                if self.pipeline_enabled and kind == "wm" and self.pending[kind] is not None:
                    try:
                        await self._release_wm_ticket()
                    except Exception as release_error:
                        self.failures[kind] = RuntimeError("%s; ticket release also failed: %s" %
                                                          (error, release_error))
                condition.notify_all()
                raise

    async def sample_policy(self, rank, local_size):
        return await self._sample("policy", rank, local_size)

    async def sample_wm(self, rank, local_size, consumer="combined"):
        if self.world_streaming and consumer == 'ctrl':
            raise ValueError("Online WEAVER must stream micro4; full World rank materialization is disabled")
        return await self._sample("wm", rank, local_size, consumer)

    async def _fail_wm_stream(self, error):
        self.failures['wm'] = error
        if self.pending['wm'] is not None:
            try:
                await self._release_wm_ticket()
            except Exception as release_error:
                self.failures['wm'] = RuntimeError(str(error) + '; release failed: ' + str(release_error))
        self.wm_streams.clear()
        self.wm_stream_owners.clear()
        self.conditions['wm'].notify_all()

    async def reserve_wm_stream(self, rank, local_size, consumer='ctrl'):
        import uuid
        if (not self.world_streaming or consumer != 'ctrl' or type(rank) is not int
                or not 0 <= rank < self.world_size or local_size != 256):
            raise ValueError("World stream requires its explicit world4/local256 ctrl role")
        identity = (consumer, rank)
        condition = self.conditions['wm']
        async with condition:
            while identity in self.claimed['wm']:
                self._check_failure('wm')
                await condition.wait()
            self._check_failure('wm')
            try:
                if identity in self.wm_stream_owners:
                    raise RuntimeError("A World rank already owns an unfinished micro stream")
                if self.pending['wm'] is None:
                    ticket = await self.replays['wm'].reserve_windows.remote(local_size * self.world_size)
                    if ticket.get('global_batch_size') != local_size * self.world_size:
                        raise ValueError("Streaming reservation has unexpected global batch")
                    self.pending['wm'] = ticket
                    self.local_sizes['wm'] = local_size
                elif self.local_sizes['wm'] != local_size:
                    raise ValueError("World/Reward stream local sizes differ")
                ticket_id = self.pending['wm']['ticket_id']
                description = await self.replays['wm'].describe_window_shard.remote(ticket_id, rank, local_size, consumer)
                metadata = description['capture_meta']
                if (len(metadata['sample_ids']) != local_size
                        or metadata['shard_start'] != rank * local_size
                        or metadata['shard_end'] != (rank + 1) * local_size):
                    raise ValueError("Streaming rank description has invalid identifiers")
                ordered_identity = (metadata['reservation_id'], tuple(metadata['sample_ids']))
                if self.wm_reservation_id is None:
                    self.wm_reservation_id = ordered_identity[0]
                if self.wm_reservation_id != ordered_identity[0]:
                    raise ValueError("World stream and Reward use different global reservations")
                previous = self.wm_identities.setdefault(rank, ordered_identity)
                if previous != ordered_identity:
                    raise ValueError("World stream and Reward use different ordered rank shards")
                token = 'wm-stream:' + uuid.uuid4().hex
                record = {'stream_id': token, 'ticket_id': ticket_id, 'rank': rank,
                          'local_size': local_size, 'capture_meta': metadata, 'next_start': 0,
                          'unique_episodes': description.get('unique_episodes', 0),
                          'fresh_episode_ratio': description.get('fresh_episode_ratio', 0.)}
                self.wm_streams[token] = record
                self.wm_stream_owners.add(identity)
                return {key: value for key, value in record.items() if key != 'next_start'}
            except (Exception, asyncio.CancelledError) as error:
                await self._fail_wm_stream(error)
                raise

    async def sample_wm_stream(self, stream_id, start, end):
        async with self.conditions['wm']:
            self._check_failure('wm')
            try:
                record = self.wm_streams[stream_id]
                if (type(start) is not int or type(end) is not int
                        or start != record['next_start'] or end - start != self.world_micro_batch
                        or end > record['local_size']):
                    raise ValueError("World stream must fetch each original micro4 slice exactly once in order")
                result = await self.replays['wm'].project_window_microshard.remote(
                    record['ticket_id'], record['rank'], record['local_size'], start, end)
                metadata = result[-1]['capture_meta']
                if (metadata['sample_ids'] != record['capture_meta']['sample_ids'][start:end]
                        or metadata['reservation_id'] != record['capture_meta']['reservation_id']
                        or len(result[0]) != end - start):
                    raise ValueError("World stream changed the reserved identifiers")
                record['next_start'] = end
                return result
            except (Exception, asyncio.CancelledError) as error:
                await self._fail_wm_stream(error)
                raise

    async def complete_wm_stream(self, stream_id):
        async with self.conditions['wm']:
            self._check_failure('wm')
            try:
                record = self.wm_streams[stream_id]
                if record['next_start'] != record['local_size']:
                    raise RuntimeError("World cannot finish a partially projected rank stream")
                identity = ('ctrl', record['rank'])
                self.claimed['wm'].add(identity)
                self.wm_stream_owners.remove(identity)
                del self.wm_streams[stream_id]
                if len(self.claimed['wm']) == self.world_size * len(self.wm_consumers):
                    await self._release_wm_ticket()
                    self.wm_identities.clear()
                    self.wm_reservation_id = None
                    self.claimed['wm'].clear()
                    self.conditions['wm'].notify_all()
                return {'rank': record['rank'], 'reservation_id': record['capture_meta']['reservation_id'],
                        'projected_samples': record['local_size'],
                        'micro_projections': record['next_start'] // self.world_micro_batch,
                        'meaning': 'projection_complete_not_proof_of_optimizer_success'}
            except (Exception, asyncio.CancelledError) as error:
                await self._fail_wm_stream(error)
                raise
