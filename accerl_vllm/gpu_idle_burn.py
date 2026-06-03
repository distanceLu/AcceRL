#!/usr/bin/env python3
"""Occupy visible GPUs with dummy memory allocation and optional compute work.

Examples:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python AcceRL/accerl_vllm/gpu_idle_burn.py
  CUDA_VISIBLE_DEVICES=0,1 python AcceRL/accerl_vllm/gpu_idle_burn.py --mem-frac 0.90
  python AcceRL/accerl_vllm/gpu_idle_burn.py --duration 600 --matmul-size 8192
  python AcceRL/accerl_vllm/gpu_idle_burn.py --memory-only --mem-frac 0.90
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import signal
import time
from typing import List

torch = None


def require_torch():
    global torch
    if torch is None:
        import torch as torch_module

        torch = torch_module
    return torch


def allocate_memory(device: torch.device, mem_frac: float, reserve_mb: int) -> List[torch.Tensor]:
    """Allocate GPU memory in chunks, backing off automatically on OOM."""
    torch = require_torch()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    target_bytes = int(total_bytes * mem_frac)
    target_bytes = min(target_bytes, max(0, free_bytes - reserve_mb * 1024 * 1024))

    chunks: List[torch.Tensor] = []
    remaining = target_bytes
    chunk_bytes = 512 * 1024 * 1024

    while remaining > 0:
        this_chunk = min(chunk_bytes, remaining)
        try:
            chunks.append(torch.empty(this_chunk, dtype=torch.uint8, device=device))
            remaining -= this_chunk
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            chunk_bytes //= 2
            if chunk_bytes < 32 * 1024 * 1024:
                break

    return chunks


def burn_gpu(local_rank: int, args: argparse.Namespace) -> None:
    torch = require_torch()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    reserved = allocate_memory(device, args.mem_frac, args.reserve_mb)
    allocated_gb = sum(t.numel() for t in reserved) / 1024**3
    stop_at = None if args.duration <= 0 else time.time() + args.duration

    if args.memory_only:
        print(
            f"[gpu {local_rank}] reserved {allocated_gb:.2f} GiB, memory-only mode",
            flush=True,
        )
        while stop_at is None or time.time() < stop_at:
            time.sleep(1)
        return

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    size = args.matmul_size
    a = torch.randn((size, size), device=device, dtype=dtype)
    b = torch.randn((size, size), device=device, dtype=dtype)

    print(
        f"[gpu {local_rank}] reserved {allocated_gb:.2f} GiB, "
        f"matmul={size}x{size}, dtype={args.dtype}",
        flush=True,
    )

    i = 0
    while stop_at is None or time.time() < stop_at:
        a = a @ b
        if i % args.sync_every == 0:
            torch.cuda.synchronize(device)
        i += 1

    torch.cuda.synchronize(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dummy GPU memory and compute burner.")
    parser.add_argument("--gpus", type=int, default=None, help="Number of visible GPUs to use.")
    parser.add_argument("--mem-frac", type=float, default=0.90, help="Fraction of each GPU memory to reserve.")
    parser.add_argument("--reserve-mb", type=int, default=1024, help="Free memory to leave for compute buffers.")
    parser.add_argument(
        "--memory-only",
        action="store_true",
        help="Only reserve GPU memory, then idle without generating sustained GPU utilization.",
    )
    parser.add_argument("--matmul-size", type=int, default=4096, help="Square matrix size for compute load.")
    parser.add_argument("--duration", type=int, default=0, help="Seconds to run. 0 means run forever.")
    parser.add_argument("--sync-every", type=int, default=10, help="Synchronize every N matmuls.")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch = require_torch()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    gpu_count = torch.cuda.device_count()
    if args.gpus is not None:
        gpu_count = min(gpu_count, args.gpus)
    if gpu_count <= 0:
        raise RuntimeError("No visible CUDA devices.")

    stop = mp.Event()

    def handle_signal(signum, frame):
        del signum, frame
        stop.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    ctx = mp.get_context("spawn")
    processes = [ctx.Process(target=burn_gpu, args=(rank, args), daemon=False) for rank in range(gpu_count)]

    for process in processes:
        process.start()

    try:
        while any(process.is_alive() for process in processes):
            if stop.is_set():
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                break
            time.sleep(1)
    finally:
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()


if __name__ == "__main__":
    main()
