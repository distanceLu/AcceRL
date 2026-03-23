from typing import Union
import contextlib
import inspect

import deepspeed
import torch
import torch.distributed as dist
from torch.distributed import Backend


def _unwrap_module(m):
    # Unwrap DeepSpeedEngine/DDP wrappers to get the underlying nn.Module.
    return getattr(m, "module", m)


def _named_tensors_in_order(module):
    """
    Return tensors that should be broadcasted:
      - Trainable parameters (requires_grad=True)
      - Optional buffers (e.g., BN running stats)
    """
    params = sorted(
        [(n, p) for n, p in module.named_parameters(recurse=True) if p.requires_grad],
        key=lambda x: x[0],
    )
    buffers = sorted(list(module.named_buffers(recurse=True)), key=lambda x: x[0])
    return params, buffers


def init_custom_process_group(
    backend=None,
    init_method=None,
    timeout=None,
    world_size=-1,
    rank=-1,
    store=None,
    group_name=None,
    pg_options=None,
):
    from torch.distributed.distributed_c10d import (
        Backend,
        PrefixStore,
        _new_process_group_helper,
        _world,
        rendezvous,
    )

    assert (store is None) or (init_method is None), "Cannot specify both init_method and store."
    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    backend = Backend(backend) if backend else Backend("nccl")

    if timeout is None:
        from datetime import timedelta

        timeout = timedelta(minutes=30)

    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)
        store = PrefixStore(group_name, store)

    helper_sig = inspect.signature(_new_process_group_helper)
    helper_kwargs = {"group_name": group_name, "timeout": timeout}
    if "backend_options" in helper_sig.parameters:
        helper_kwargs["backend_options"] = pg_options
    elif "pg_options" in helper_sig.parameters:
        helper_kwargs["pg_options"] = pg_options

    pg, _ = _new_process_group_helper(world_size, rank, [], backend, store, **helper_kwargs)
    if _world:
        _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
    return pg


class GroupManager:
    def __init__(self):
        self._name_group_map = {}
        self._group_backend_map = {}

    def create_collective_group(self, backend, world_size, rank, master_addr: str, master_port: int, group_name):
        init_method = f"tcp://{master_addr}:{master_port}"
        pg_handle = init_custom_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
            group_name=group_name,
        )
        self._name_group_map[group_name] = pg_handle
        self._group_backend_map[group_name] = str(backend)
        return pg_handle

    def is_group_exist(self, group_name):
        return group_name in self._name_group_map

    def get_group_by_name(self, group_name):
        if not self.is_group_exist(group_name):
            print(f"Warning: communication group '{group_name}' is not initialized.")
            return None
        return self._name_group_map[group_name]

    def get_backend_by_name(self, group_name):
        return self._group_backend_map.get(group_name, "nccl")


_group_mgr = GroupManager()


def init_collective_group(
    world_size: int,
    rank: int,
    master_addr: str,
    master_port: int,
    backend: Union[str, Backend] = "nccl",
    group_name: str = "default",
):
    global _group_mgr
    if not group_name:
        raise ValueError(f"group_name '{group_name}' must be a non-empty string.")
    if _group_mgr.is_group_exist(group_name):
        return
    _group_mgr.create_collective_group(backend, world_size, rank, master_addr, master_port, group_name)


class TrainerActorCom:
    def __init__(self):
        pass

    def setup_broadcast_group(self, master_addr, master_port, group_name, group_world_size, my_rank_in_group, backend="nccl"):
        init_collective_group(
            world_size=group_world_size,
            rank=my_rank_in_group,
            master_addr=master_addr,
            master_port=master_port,
            group_name=group_name,
            backend=backend,
        )
        print(
            f"TrainerActor rank {self.rank}: joined broadcast group '{group_name}' "
            f"as rank {my_rank_in_group} (backend={backend})."
        )

    def broadcast_weights(self, group_name):
        # Called only by the trainer at src=0 (as in the main loop).
        group_handle = _group_mgr.get_group_by_name(group_name)
        assert group_handle is not None, f"Broadcast group '{group_name}' is not initialized."
        backend = _group_mgr.get_backend_by_name(group_name).lower()
        use_cpu_broadcast = "gloo" in backend

        module = _unwrap_module(self.model)
        zero_ctx = getattr(deepspeed.zero, "GatheredParameters", None)
        if zero_ctx is None:
            zero_ctx = contextlib.nullcontext

        params, buffers = _named_tensors_in_order(module)
        device = next(module.parameters()).device

        with zero_ctx(module.parameters(), modifier_rank=0):
            for _, p in params:
                b_device = "cpu" if use_cpu_broadcast else device
                t = p.detach().to(device=b_device, dtype=p.dtype).contiguous()
                dist.broadcast(t, src=0, group=group_handle)
            for _, b in buffers:
                b_device = "cpu" if use_cpu_broadcast else device
                t = b.detach().to(device=b_device, dtype=b.dtype).contiguous()
                dist.broadcast(t, src=0, group=group_handle)


class InferenceActorCom:
    def __init__(self):
        pass

    def setup_broadcast_group(self, master_addr, master_port, group_name, group_world_size, my_rank_in_group, backend="nccl"):
        init_collective_group(
            world_size=group_world_size,
            rank=my_rank_in_group,
            master_addr=master_addr,
            master_port=master_port,
            group_name=group_name,
            backend=backend,
        )
        print(
            f"InferenceActor {self.actor_id}: joined broadcast group '{group_name}' "
            f"as rank {my_rank_in_group} (backend={backend})."
        )

    def receive_and_update_weights(self, group_name):
        group_handle = _group_mgr.get_group_by_name(group_name)
        assert group_handle is not None, f"Broadcast group '{group_name}' is not initialized."
        backend = _group_mgr.get_backend_by_name(group_name).lower()
        use_cpu_broadcast = "gloo" in backend

        module = _unwrap_module(self.model)
        params, buffers = _named_tensors_in_order(module)
        device = next(module.parameters()).device

        for _, p in params:
            b_device = "cpu" if use_cpu_broadcast else device
            buf = torch.empty_like(p.data, device=b_device)
            dist.broadcast(buf, src=0, group=group_handle)
            if use_cpu_broadcast:
                p.data.copy_(buf.to(device=device, dtype=p.data.dtype))
            else:
                p.data.copy_(buf)

        for _, b in buffers:
            b_device = "cpu" if use_cpu_broadcast else device
            buf = torch.empty_like(b.data, device=b_device)
            dist.broadcast(buf, src=0, group=group_handle)
            if use_cpu_broadcast:
                b.data.copy_(buf.to(device=device, dtype=b.data.dtype))
            else:
                b.data.copy_(buf)

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
