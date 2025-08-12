# ================================================================
# 通信模块
# ================================================================
from typing import Union
import torch
import torch.distributed as dist
from torch.distributed import Backend
import deepspeed


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
        with deepspeed.zero.GatheredParameters(self.model.parameters(), modifier_rank=0):
            if self.rank == 0:
                state_dict = self.model.state_dict()
                for tensor in state_dict.values():
                    tensor_gpu = tensor.to(self.model.device)
                    broadcast(tensor_gpu, src_rank=0, group_name=group_name)


class InferenceActorCom:
    def __init__(self):
        pass

    def setup_broadcast_group(self, master_addr, master_port, group_name, group_world_size, my_rank_in_group):
        init_collective_group(
            world_size=group_world_size, rank=my_rank_in_group, master_addr=master_addr,
            master_port=master_port, group_name=group_name)
        print(f"InferenceActor {self.actor_id}: 已作为 rank {my_rank_in_group} 加入广播组 '{group_name}'。")

    def receive_and_update_weights(self, group_name):
        state_dict = self.model.state_dict()
        for key, value in state_dict.items():
            received_tensor = torch.empty_like(state_dict[key], device='cuda')
            assert received_tensor.dtype == value.dtype, f"received_tensor dtype: {received_tensor.dtype}, value dtype: {value.dtype}"
            broadcast(received_tensor, src_rank=0, group_name=group_name)
            state_dict[key] = received_tensor
        self.model.load_state_dict(state_dict)

