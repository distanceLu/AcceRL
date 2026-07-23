#!/usr/bin/env python
"""Minimal Ray + torch.distributed broadcast diagnostic.

This script intentionally avoids OpenVLA/LIBERO/DeepSpeed. It checks whether the
same Ray placement + custom process group path used by training can broadcast
basic tensors between two GPU actors.
"""

import argparse
import os
import socket
import time
from datetime import timedelta
from pathlib import Path

import ray


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def parse_shape(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.replace("x", ",").split(",") if part)


@ray.remote(num_gpus=1)
class BroadcastDebugActor:
    def __init__(self, rank: int, world_size: int, backend: str, group_name: str):
        import torch

        self.rank = rank
        self.world_size = world_size
        self.backend = backend
        self.group_name = group_name
        self.group = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        print(
            f"[rank {rank}] init pid={os.getpid()} "
            f"ray_gpu_ids={ray.get_gpu_ids()} "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
            f"torch_cuda_available={torch.cuda.is_available()} "
            f"device={self.device}",
            flush=True,
        )
        if torch.cuda.is_available():
            print(
                f"[rank {rank}] cuda_device_count={torch.cuda.device_count()} "
                f"current_device={torch.cuda.current_device()} "
                f"name={torch.cuda.get_device_name(torch.cuda.current_device())}",
                flush=True,
            )

    def get_node_ip(self) -> str:
        return ray.util.get_node_ip_address()

    def setup_group(self, master_addr: str, master_port: int, timeout_s: int):
        from rl.ds_com import init_custom_process_group

        init_method = f"tcp://{master_addr}:{master_port}"
        print(
            f"[rank {self.rank}] setup_group backend={self.backend} "
            f"init_method={init_method} timeout_s={timeout_s}",
            flush=True,
        )
        self.group = init_custom_process_group(
            backend=self.backend,
            init_method=init_method,
            timeout=timedelta(seconds=timeout_s),
            world_size=self.world_size,
            rank=self.rank,
            group_name=self.group_name,
        )
        print(f"[rank {self.rank}] setup_group done", flush=True)
        return True

    def run_case(self, dtype_name: str, shape: tuple[int, ...], value: float):
        import torch
        import torch.distributed as dist

        dtype = getattr(torch, dtype_name)
        numel = 1
        for dim in shape:
            numel *= dim
        if dtype.is_floating_point:
            tensor = torch.full(shape, value if self.rank == 0 else -value, dtype=dtype, device=self.device)
        else:
            tensor = torch.full(shape, int(value) if self.rank == 0 else -int(value), dtype=dtype, device=self.device)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        print(
            f"[rank {self.rank}] BEFORE broadcast dtype={dtype} shape={tuple(tensor.shape)} "
            f"numel={numel} bytes={tensor.element_size() * tensor.numel()} "
            f"device={tensor.device} first={tensor.flatten()[0].item()}",
            flush=True,
        )
        start = time.time()
        dist.broadcast(tensor, src=0, group=self.group)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start
        first = tensor.flatten()[0].item()
        ok = first == value or first == int(value)
        print(
            f"[rank {self.rank}] AFTER broadcast dtype={dtype} shape={tuple(tensor.shape)} "
            f"elapsed={elapsed:.4f}s first={first} ok={ok}",
            flush=True,
        )
        return {"rank": self.rank, "dtype": dtype_name, "shape": shape, "elapsed": elapsed, "first": first, "ok": ok}


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug Ray/NCCL broadcast with basic tensors.")
    parser.add_argument("--cuda-visible-devices", default="6,7")
    parser.add_argument("--backend", default="nccl", choices=["nccl", "gloo"])
    parser.add_argument("--object-store-memory-gb", type=int, default=8)
    parser.add_argument("--ray-temp-dir", default="/dev/shm")
    parser.add_argument("--timeout-s", type=int, default=60)
    parser.add_argument("--ray-get-timeout-s", type=int, default=90)
    parser.add_argument("--shape", default="1024")
    parser.add_argument("--dtypes", default="float32,bfloat16,float16,int64")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    os.environ.setdefault("NCCL_DEBUG", "INFO")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")

    print("=" * 80)
    print("NCCL/Ray broadcast debug")
    print(f"cwd={Path.cwd()}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"NCCL_DEBUG={os.environ.get('NCCL_DEBUG')}")
    print(f"TORCH_NCCL_ASYNC_ERROR_HANDLING={os.environ.get('TORCH_NCCL_ASYNC_ERROR_HANDLING')}")
    print(f"TORCH_NCCL_BLOCKING_WAIT={os.environ.get('TORCH_NCCL_BLOCKING_WAIT')}")
    print("=" * 80, flush=True)

    ray.init(
        ignore_reinit_error=True,
        _temp_dir=args.ray_temp_dir,
        object_store_memory=int(args.object_store_memory_gb * 1024 * 1024 * 1024),
    )

    world_size = 2
    group_name = f"debug_broadcast_{int(time.time())}"
    actors = [
        BroadcastDebugActor.remote(rank=i, world_size=world_size, backend=args.backend, group_name=group_name)
        for i in range(world_size)
    ]
    master_addr = ray.get(actors[0].get_node_ip.remote())
    master_port = find_free_port()

    try:
        setup_refs = [actor.setup_group.remote(master_addr, master_port, args.timeout_s) for actor in actors]
        ray.get(setup_refs, timeout=args.ray_get_timeout_s)
        print("Process group setup: OK", flush=True)

        shape = parse_shape(args.shape)
        for dtype_name in [item.strip() for item in args.dtypes.split(",") if item.strip()]:
            print("-" * 80)
            print(f"Running broadcast case dtype={dtype_name} shape={shape}", flush=True)
            refs = [actor.run_case.remote(dtype_name, shape, 123.0) for actor in actors]
            results = ray.get(refs, timeout=args.ray_get_timeout_s)
            print(f"Case result: {results}", flush=True)
        print("=" * 80)
        print("All broadcast cases completed.")
    except Exception as exc:
        print("=" * 80)
        print(f"DEBUG FAILED: {type(exc).__name__}: {exc}")
        print("If this is a timeout, the last BEFORE/AFTER line above shows where NCCL stopped.")
        raise
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
