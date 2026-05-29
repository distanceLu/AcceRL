#!/usr/bin/env python3
"""Minimal two-GPU "model" communication benchmark.

The model is intentionally boring: one or more huge contiguous Parameters and
an identity forward.  That makes the measured path mostly NCCL bandwidth rather
than transformer layer overhead.

Examples:
  CUDA_VISIBLE_DEVICES=6,7 python minimal_model_nccl_bandwidth.py --size-gib 8
  CUDA_VISIBLE_DEVICES=6,7 python minimal_model_nccl_bandwidth.py --op all-reduce --size-gib 8
  NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=6,7 python minimal_model_nccl_bandwidth.py --size-gib 16
"""

from __future__ import annotations

import argparse
import os
import socket
import statistics
import time
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class BandwidthModel(nn.Module):
    """A model whose whole job is to expose large contiguous parameters."""

    def __init__(
        self,
        *,
        total_bytes: int,
        dtype: torch.dtype,
        device: torch.device,
        shards: int,
    ) -> None:
        super().__init__()
        if shards < 1:
            raise ValueError("--shards must be >= 1")

        elem_size = torch.empty((), dtype=dtype).element_size()
        total_numel = total_bytes // elem_size
        if total_numel < shards:
            raise ValueError("requested model is too small for the shard count")

        base = total_numel // shards
        extra = total_numel % shards
        self.weights = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        base + (1 if shard_idx < extra else 0),
                        device=device,
                        dtype=dtype,
                    ),
                    requires_grad=False,
                )
                for shard_idx in range(shards)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


@dataclass(frozen=True)
class BenchResult:
    seconds: float
    payload_bytes: int

    @property
    def gbps(self) -> float:
        return self.payload_bytes / self.seconds / 1e9

    @property
    def gibps(self) -> float:
        return self.payload_bytes / self.seconds / 1024**3


def find_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def gib_to_bytes(value: float) -> int:
    return int(value * 1024**3)


def model_nbytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def model_num_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def parameter_tensors(model: nn.Module) -> Iterable[torch.Tensor]:
    for parameter in model.parameters():
        yield parameter.data


def run_collective(op: str, model: nn.Module) -> None:
    if op == "broadcast":
        for tensor in parameter_tensors(model):
            dist.broadcast(tensor, src=0)
        return

    if op == "all-reduce":
        for tensor in parameter_tensors(model):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return

    raise ValueError(f"unsupported op: {op}")


def worker(local_rank: int, args: argparse.Namespace, master_port: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl", rank=local_rank, world_size=2)

    dtype = DTYPES[args.dtype]
    model = BandwidthModel(
        total_bytes=gib_to_bytes(args.size_gib),
        dtype=dtype,
        device=device,
        shards=args.shards,
    )
    param_count = model_num_parameters(model)
    payload_bytes = model_nbytes(model)
    bytes_per_parameter = payload_bytes / param_count

    torch.cuda.synchronize(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)

    if local_rank == 0:
        print(
            f"op={args.op}, dtype={args.dtype}, shards={args.shards}, "
            f"params={param_count:,}, "
            f"bytes_per_param={bytes_per_parameter:.0f}, "
            f"payload={payload_bytes / 1e9:.3f} GB "
            f"({payload_bytes / 1024**3:.3f} GiB)",
            flush=True,
        )
        print(
            f"visible_gpus={torch.cuda.device_count()}, "
            f"device0_can_access_device1="
            f"{torch.cuda.can_device_access_peer(0, 1)}",
            flush=True,
        )

    print(
        f"[rank {local_rank}] {torch.cuda.get_device_name(device)} "
        f"free_after_alloc={free_bytes / 1024**3:.2f}/"
        f"{total_bytes / 1024**3:.2f} GiB",
        flush=True,
    )

    for _ in range(args.warmup):
        run_collective(args.op, model)
    torch.cuda.synchronize(device)
    dist.barrier()

    results: list[BenchResult] = []
    for iteration in range(args.iters):
        dist.barrier()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        run_collective(args.op, model)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

        result = BenchResult(seconds=elapsed, payload_bytes=payload_bytes)
        results.append(result)

        if local_rank == 0:
            print(
                f"iter={iteration + 1:03d} "
                f"time={result.seconds:.6f}s "
                f"bandwidth={result.gbps:.2f} GB/s "
                f"({result.gibps:.2f} GiB/s)",
                flush=True,
            )

    dist.barrier()
    if local_rank == 0:
        gbps = [result.gbps for result in results]
        gibps = [result.gibps for result in results]
        print("\nSummary")
        print(
            f"  avg={statistics.mean(gbps):.2f} GB/s "
            f"({statistics.mean(gibps):.2f} GiB/s)"
        )
        print(
            f"  min={min(gbps):.2f} GB/s, "
            f"max={max(gbps):.2f} GB/s, "
            f"median={statistics.median(gbps):.2f} GB/s"
        )

    dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Saturate two-GPU NCCL communication with a minimal model."
    )
    parser.add_argument(
        "--op",
        choices=("broadcast", "all-reduce"),
        default="broadcast",
        help="broadcast matches one-way trainer-to-inference weight sync; "
        "all-reduce is useful for bidirectional collective bandwidth.",
    )
    parser.add_argument(
        "--size-gib",
        type=float,
        default=4.0,
        help="Total model parameter payload per iteration.",
    )
    parser.add_argument(
        "--dtype",
        choices=tuple(DTYPES),
        default="bfloat16",
        help="Parameter dtype. Dtype only changes element count for a fixed byte size.",
    )
    parser.add_argument(
        "--shards",
        type=int,
        default=1,
        help="Number of Parameter tensors. Use 1 for maximum contiguous bandwidth.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if torch.cuda.device_count() != 2:
        raise RuntimeError(
            "This benchmark expects exactly two visible GPUs. "
            "Set CUDA_VISIBLE_DEVICES to the two cards you want to test."
        )
    if not dist.is_nccl_available():
        raise RuntimeError("PyTorch was not built with NCCL support.")

    mp.set_start_method("spawn", force=True)
    mp.spawn(worker, args=(args, find_open_port()), nprocs=2, join=True)


if __name__ == "__main__":
    main()
