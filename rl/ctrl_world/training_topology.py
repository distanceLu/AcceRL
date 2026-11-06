"""Explicit single-node GPU bindings and identity-preserving global batch shards."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
import subprocess


DUAL_LAYOUT = "dual_train_policy_reward"
FOUR_LAYOUT = "four_rank_six_roles"
SPLIT_LAYOUT = "four_rank_split_train"


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


def ray_init_options(bindings, layout=DUAL_LAYOUT):
    # Ray's aggregate GPU resource cannot pin a specific physical index. Reserve
    # one named resource per card and verify the CUDA mask against its UUID.
    # MPS remaps the server's physical indices, so MPS clients use UUID masks;
    # verify_bound_gpu fixes this actor's DeepSpeed NVML lookup after validation.
    # CPU actors inherit an empty mask. This cluster is local to this invocation.
    if layout not in (DUAL_LAYOUT, FOUR_LAYOUT, SPLIT_LAYOUT):
        raise ValueError("Unknown fixed GPU layout: %s" % layout)
    capacity = 6.0 if layout in (FOUR_LAYOUT, SPLIT_LAYOUT) else 1.0
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


def actor_options(bindings, role, rank=0, layout=DUAL_LAYOUT):
    if layout in (FOUR_LAYOUT, SPLIT_LAYOUT):
        allowed_roles = (("policy_trainer", "ctrl_trainer", "reward_trainer", "policy", "reward", "ctrl")
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


def _selected_weights(module, kind):
    """Match the production broadcast contract, excluding frozen parameters."""
    weights = {name: tensor for name, tensor in module.named_parameters(recurse=True)
               if tensor.requires_grad}
    if kind != "ctrl":
        for name, tensor in module.named_buffers(recurse=True):
            if name in weights:
                raise ValueError("Parameter and buffer share a weight name: %s" % name)
            weights[name] = tensor
    if not weights:
        raise ValueError("No mutable weights found for %s" % kind)
    return sorted(weights.items())


def _weight_hash(weights, kind):
    """Hash raw CPU bytes, including bfloat16, one tensor at a time."""
    import hashlib
    import json
    import torch
    digest = hashlib.sha256()
    digest.update((kind + "\n").encode("utf-8"))
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


def export_cpu_weights(module, kind, version):
    """Create an immutable CPU snapshot for a same-card Ray inference actor.

    ZeRO-2 keeps full parameter replicas. Copying their mutable tensors over
    Ray avoids an NCCL broadcast communicator containing two processes on the
    same physical GPU. The digest excludes version, so different ranks can
    verify that their replicas contain identical weights.
    """
    kind = _weight_kind(kind)
    version = _weight_version(version)
    weights = _selected_weights(module, kind)
    tensors = {name: tensor.detach().to(device="cpu", copy=True).contiguous()
               for name, tensor in weights}
    return {"kind": kind, "version": version,
            "hash": _weight_hash(sorted(tensors.items()), kind),
            "bytes": sum(tensor.numel() * tensor.element_size() for tensor in tensors.values()),
            "tensor_count": len(tensors), "tensors": tensors}


def apply_cpu_weights(module, payload, kind):
    """Validate the complete snapshot before copying, then verify loaded bytes."""
    import torch
    kind = _weight_kind(kind)
    if not isinstance(payload, dict) or payload.get("kind") != kind:
        raise ValueError("CPU weight payload kind differs from the destination model")
    version = _weight_version(payload.get("version"))
    tensors = payload.get("tensors")
    if not isinstance(tensors, dict):
        raise ValueError("CPU weight payload is missing its tensor mapping")
    selected = _selected_weights(module, kind)
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
    source_hash = _weight_hash(sorted(tensors.items()), kind)
    if payload.get("hash") != source_hash:
        raise ValueError("CPU weight payload hash differs from its tensor contents")
    with torch.no_grad():
        for name, destination in selected:
            destination.copy_(tensors[name])
    loaded_hash = _weight_hash(selected, kind)
    if loaded_hash != source_hash:
        raise RuntimeError("Loaded inference weights differ from the source snapshot")
    return {"kind": kind, "version": version, "hash": loaded_hash,
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
        metadata["sample_stats_scope"] = "global_batch"
    return (*data, metadata)


class GlobalBatchCoordinator:
    """Each Replay sample is drawn once and consumed as disjoint rank shards.

    At most one unclaimed global batch per lane is held. A faster rank's next
    prefetch waits until all ranks have claimed the preceding batch. Split
    World/Reward groups claim identical shards from the same WM global draw;
    both model groups must finish claiming before either can draw another.
    """
    def __init__(self, policy_replay, wm_replay, world_size, wm_consumers=("combined",)):
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
                await condition.wait()
            if self.pending[kind] is None:
                self.pending[kind] = await self.replays[kind].sample.remote(local_size * self.world_size)
                self.local_sizes[kind] = local_size
            elif self.local_sizes[kind] != local_size:
                raise ValueError("All ranks must request equal local batches")
            result = shard_batch(self.pending[kind], kind, rank, local_size, self.world_size)
            self.claimed[kind].add(identity)
            if len(self.claimed[kind]) == expected_claims:
                self.pending[kind] = None
                self.claimed[kind].clear()
                condition.notify_all()
            return result

    async def sample_policy(self, rank, local_size):
        return await self._sample("policy", rank, local_size)

    async def sample_wm(self, rank, local_size, consumer="combined"):
        return await self._sample("wm", rank, local_size, consumer)
