# ================================================================
# 通信模块
# ================================================================
from typing import Union
import torch
import torch.distributed as dist
from torch.distributed import Backend
import deepspeed
import contextlib


def _unwrap_module(m):
    # DeepSpeedEngine 或 DDP 包装时取到真实 nn.Module
    return getattr(m, "module", m)


def _named_tensors_in_order(module):
    """
    只返回需要广播的张量：
      - 可训练参数 (requires_grad=True)
      - 可选的 buffers (比如 BN 的 running stats)
    """
    params = sorted(
        [(n, p) for n, p in module.named_parameters(recurse=True) if p.requires_grad],
        key=lambda x: x[0]
    )
    buffers = sorted(
        list(module.named_buffers(recurse=True)), key=lambda x: x[0]
    )
    return params, buffers


def init_custom_process_group(
    backend=None, init_method=None, timeout=None, world_size=-1, rank=-1,
    store=None, group_name=None, pg_options=None,):
    from torch.distributed.distributed_c10d import (
        Backend, PrefixStore, _new_process_group_helper, _world,
        default_pg_timeout, rendezvous)
    assert (store is None) or (init_method is None), "Cannot specify both init_method and store."
    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"
    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("nccl")
    if timeout is None:
        from datetime import timedelta
        timeout = timedelta(minutes=30)
    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)
        store = PrefixStore(group_name, store)
    pg_options_param_name = "backend_options" if str(torch.__version__) >= "2.6" else "pg_options"
    pg, _ = _new_process_group_helper(
        world_size, rank, [], backend, store, group_name=group_name,
        **{pg_options_param_name: pg_options}, timeout=timeout)
    if _world:
        _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
    return pg


class GroupManager:
    def __init__(self):
        self._name_group_map = {}
    def create_collective_group(self, backend, world_size, rank, master_addr: str, master_port: int, group_name):
        init_method = f"tcp://{master_addr}:{master_port}"
        pg_handle = init_custom_process_group(
            backend=backend, init_method=init_method, world_size=world_size, rank=rank, group_name=group_name)
        self._name_group_map[group_name] = pg_handle
        return pg_handle
    def is_group_exist(self, group_name):
        return group_name in self._name_group_map
    def get_group_by_name(self, group_name):
        if not self.is_group_exist(group_name):
            print(f"警告: 通信组 '{group_name}' 未初始化。")
            return None
        return self._name_group_map[group_name]


_group_mgr = GroupManager()


def init_collective_group(
    world_size: int, rank: int, master_addr: str, master_port: int,
    backend: Union[str, Backend] = "nccl", group_name: str = "default"):
    global _group_mgr
    if not group_name: raise ValueError(f"group_name '{group_name}' 必须是一个非空字符串。")
    if _group_mgr.is_group_exist(group_name): return
    _group_mgr.create_collective_group(backend, world_size, rank, master_addr, master_port, group_name)


def broadcast(tensor, src_rank: int = 0, group_name: str = "default"):
    group_handle = _group_mgr.get_group_by_name(group_name)
    dist.broadcast(tensor, src=src_rank, group=group_handle)


class TrainerActorCom:
    def __init__(self):
        pass
    
    def setup_broadcast_group(self, master_addr, master_port, group_name, group_world_size, my_rank_in_group):
        init_collective_group(
            world_size=group_world_size, rank=my_rank_in_group, master_addr=master_addr,
            master_port=master_port, group_name=group_name)
        print(f"TrainerActor Rank {self.rank}: 已作为 rank {my_rank_in_group} 加入广播组 '{group_name}'。")

    def broadcast_weights(self, group_name):
        # 只在 src=0 的 Trainer 调用（你的主循环里就是这样）
        group_handle = _group_mgr.get_group_by_name(group_name)
        assert group_handle is not None, f"广播组 '{group_name}' 未初始化"

        module = _unwrap_module(self.model)  # DeepSpeedEngine -> nn.Module
        # ZeRO-2 下需先 gather 完整参数到 rank0
        # 非 ZeRO / 单机也可用空上下文
        zero_ctx = getattr(deepspeed.zero, "GatheredParameters", None)
        if zero_ctx is None:
            zero_ctx = contextlib.nullcontext

        params, buffers = _named_tensors_in_order(module)
        device = next(module.parameters()).device

        with zero_ctx(module.parameters(), modifier_rank=0):
            # 广播参数
            for name, p in params:
                # p 此时在 rank0 才是完整的参数
                t = p.detach().to(device=device, dtype=p.dtype).contiguous()
                dist.broadcast(t, src=0, group=group_handle)
            # 广播缓冲区（如 BN 的 running_mean/var 等）
            for name, b in buffers:
                t = b.detach().to(device=device, dtype=b.dtype).contiguous()
                dist.broadcast(t, src=0, group=group_handle)

    def get_broadcast_signature(self):
        module = _unwrap_module(self.model)
        sig = []
        # 强烈建议：只取可训练参数；缓冲区是否包含要看你是否需要动它们
        for name, p in sorted(module.named_parameters(recurse=True), key=lambda x: x[0]):
            if p.requires_grad:
                sig.append(("param", name, tuple(p.shape), str(p.dtype)))
        # 如果一定要广播 buffer，则也要列出来
        for name, b in sorted(module.named_buffers(recurse=True), key=lambda x: x[0]):
            sig.append(("buffer", name, tuple(b.shape), str(b.dtype)))
        return sig


class InferenceActorCom:
    def __init__(self):
        pass

    def setup_broadcast_group(self, master_addr, master_port, group_name, group_world_size, my_rank_in_group):
        init_collective_group(
            world_size=group_world_size, rank=my_rank_in_group, master_addr=master_addr,
            master_port=master_port, group_name=group_name)
        print(f"InferenceActor {self.actor_id}: 已作为 rank {my_rank_in_group} 加入广播组 '{group_name}'。")

    def receive_and_update_weights(self, group_name):
        group_handle = _group_mgr.get_group_by_name(group_name)
        assert group_handle is not None, f"广播组 '{group_name}' 未初始化"

        module = _unwrap_module(self.model)
        params, buffers = _named_tensors_in_order(module)
        device = next(module.parameters()).device

        # 逐个接收并原地写入，严格对齐名字顺序
        for name, p in params:
            buf = torch.empty_like(p.data, device=device)
            dist.broadcast(buf, src=0, group=group_handle)
            p.data.copy_(buf)

        for name, b in buffers:
            buf = torch.empty_like(b.data, device=device)
            dist.broadcast(buf, src=0, group=group_handle)
            b.data.copy_(buf)

        # 同步与确认
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
            
    def get_broadcast_signature(self):
        module = _unwrap_module(self.model)
        sig = []
        # 强烈建议：只取可训练参数；缓冲区是否包含要看你是否需要动它们
        for name, p in sorted(module.named_parameters(recurse=True), key=lambda x: x[0]):
            if p.requires_grad:
                sig.append(("param", name, tuple(p.shape), str(p.dtype)))
        # 如果一定要广播 buffer，则也要列出来
        for name, b in sorted(module.named_buffers(recurse=True), key=lambda x: x[0]):
            sig.append(("buffer", name, tuple(b.shape), str(b.dtype)))
        return sig