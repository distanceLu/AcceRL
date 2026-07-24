"""
Tiny Ray + torch.distributed weight broadcast demo.

This is the smallest useful version of the pattern used in
tests_dsj/ds_wm_discrete_ctrl.py:

    main process
        creates Ray actors
        builds a broadcast process group
        asks trainer to broadcast
        asks inference actor to receive at the same time

    TrainerActor
        owns the source model
        optionally wraps it with DeepSpeed
        calls dist.broadcast(param, src=0)

    InferenceActor
        owns the target model with the same parameter shapes
        calls dist.broadcast(empty_buffer, src=0)
        copies the received buffer into its model

Run:
    python tests_dsj/ray_deepspeed_broadcast_tiny.py

With DeepSpeed wrapping on the trainer side:
    python tests_dsj/ray_deepspeed_broadcast_tiny.py --use-deepspeed

Print each tensor broadcast:
    ACCERL_DEBUG_BROADCAST=1 python tests_dsj/ray_deepspeed_broadcast_tiny.py
"""

import argparse
import contextlib
import inspect
import os
import socket
from datetime import timedelta

import ray
import torch
import torch.distributed as dist
import torch.nn as nn


GROUP_NAME = "trainer_to_inference"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="gloo", choices=["gloo"])
    parser.add_argument("--use-deepspeed", action="store_true")
    return parser.parse_args()


def find_free_port():
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.getsockname()[1]


def make_model():
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.0)
    return model


def unwrap(model_or_engine):
    return getattr(model_or_engine, "module", model_or_engine)


def named_trainable_params(model_or_engine):
    module = unwrap(model_or_engine)
    return sorted(
        [(name, param) for name, param in module.named_parameters() if param.requires_grad],
        key=lambda item: item[0],
    )


def create_broadcast_group(rank, world_size, master_addr, master_port, backend):
    """Create a small process group used only by trainer -> inference broadcast."""
    from torch.distributed.distributed_c10d import (
        PrefixStore,
        _new_process_group_helper,
        _world,
    )

    store = dist.TCPStore(
        host_name=master_addr,
        port=master_port,
        world_size=world_size,
        is_master=(rank == 0),
        timeout=timedelta(seconds=60),
    )
    store = PrefixStore(GROUP_NAME, store)
    helper_params = inspect.signature(_new_process_group_helper).parameters
    pg_options_name = "backend_options" if "backend_options" in helper_params else "pg_options"
    process_group, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=GROUP_NAME,
        timeout=timedelta(seconds=60),
        **{pg_options_name: None},
    )
    _world.pg_group_ranks[process_group] = {i: i for i in range(world_size)}
    return process_group


def init_single_rank_deepspeed(master_addr, master_port):
    """DeepSpeed training group is separate from the broadcast group."""
    import deepspeed

    if dist.is_initialized():
        return
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    deepspeed.init_distributed(
        dist_backend="gloo",
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=0,
        world_size=1,
        auto_mpi_discovery=False,
    )


def debug(msg):
    import os

    if os.environ.get("ACCERL_DEBUG_BROADCAST") == "1":
        print(msg, flush=True)


@ray.remote
class TrainerActor:
    def __init__(self, use_deepspeed):
        self.rank = 0
        self.model = make_model()
        self.use_deepspeed = use_deepspeed
        self.broadcast_group = None

    def get_node_ip(self):
        return ray.util.get_node_ip_address()

    def setup(self, broadcast_addr, broadcast_port, ds_port, world_size, backend):
        if self.use_deepspeed:
            init_single_rank_deepspeed(broadcast_addr, ds_port)
        self.broadcast_group = create_broadcast_group(
            rank=0,
            world_size=world_size,
            master_addr=broadcast_addr,
            master_port=broadcast_port,
            backend=backend,
        )

        if self.use_deepspeed:
            import deepspeed

            ds_config = {
                "train_micro_batch_size_per_gpu": 1,
                "gradient_accumulation_steps": 1,
                "optimizer": {"type": "AdamW", "params": {"lr": 1e-3}},
                "zero_optimization": {"stage": 0},
            }
            self.model, _, _, _ = deepspeed.initialize(
                model=self.model,
                model_parameters=self.model.parameters(),
                config=ds_config,
                dist_init_required=False,
            )
        return self.read_weight()

    def fake_train_one_step(self):
        module = unwrap(self.model)
        with torch.no_grad():
            module.weight.add_(1.0)
        return self.read_weight()

    def broadcast_weights(self):
        for name, param in named_trainable_params(self.model):
            debug(f"[trainer] broadcast {name} value={param.flatten()[0].item()}")
            dist.broadcast(param.data, src=0, group=self.broadcast_group)
        return self.read_weight()

    def read_weight(self):
        module = unwrap(self.model)
        return float(module.weight.flatten()[0].item())


@ray.remote
class InferenceActor:
    def __init__(self):
        self.rank = 1
        self.model = make_model()
        self.broadcast_group = None

    def setup(self, master_addr, master_port, world_size, backend):
        self.broadcast_group = create_broadcast_group(
            rank=1,
            world_size=world_size,
            master_addr=master_addr,
            master_port=master_port,
            backend=backend,
        )
        return self.read_weight()

    def receive_weights(self):
        for name, param in named_trainable_params(self.model):
            recv_buffer = torch.empty_like(param.data)
            debug(f"[inference] before receive {name} value={param.flatten()[0].item()}")
            dist.broadcast(recv_buffer, src=0, group=self.broadcast_group)
            param.data.copy_(recv_buffer)
            debug(f"[inference] after receive {name} value={param.flatten()[0].item()}")
        return self.read_weight()

    def predict(self, x):
        with torch.no_grad():
            y = self.model(torch.tensor([[float(x)]], dtype=torch.float32))
        return float(y.item())

    def read_weight(self):
        return float(self.model.weight.flatten()[0].item())


def main():
    args = parse_args()
    ray.init(ignore_reinit_error=True, include_dashboard=False)

    trainer = TrainerActor.remote(use_deepspeed=args.use_deepspeed)
    inference = InferenceActor.remote()

    master_addr = ray.get(trainer.get_node_ip.remote())
    broadcast_port = find_free_port()
    ds_port = find_free_port()
    world_size = 2

    print("1. Build the distributed broadcast group")
    ray.get(
        [
            trainer.setup.remote(master_addr, broadcast_port, ds_port, world_size, args.backend),
            inference.setup.remote(master_addr, broadcast_port, world_size, args.backend),
        ]
    )
    print("   trainer weight:", ray.get(trainer.read_weight.remote()))
    print("   inference weight:", ray.get(inference.read_weight.remote()))

    print("2. Fake train on trainer")
    print("   trainer weight:", ray.get(trainer.fake_train_one_step.remote()))
    print("   inference prediction before sync, x=2:", ray.get(inference.predict.remote(2.0)))

    print("3. Broadcast weights trainer -> inference")
    ray.get(
        [
            trainer.broadcast_weights.remote(),
            inference.receive_weights.remote(),
        ]
    )
    print("   trainer weight:", ray.get(trainer.read_weight.remote()))
    print("   inference weight:", ray.get(inference.read_weight.remote()))
    print("   inference prediction after sync, x=2:", ray.get(inference.predict.remote(2.0)))

    ray.shutdown()


if __name__ == "__main__":
    main()
