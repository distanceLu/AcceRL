"""
Benchmark the TrainerActor -> InferenceActor NCCL broadcast path used by
rl/ds_libero_ppo.py, without loading OpenVLA or DeepSpeed training state.

Example:
CUDA_VISIBLE_DEVICES=3,4 \
DS_COMM_TOTAL_GIB=1.0 \
DS_COMM_NUM_TENSORS=64 \
DS_COMM_REPEATS=20 \
python rl/tests/ds_trainer_infer_comm_speed.py
"""

import json
import math
import os
import socket
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

import ray
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from AcceRL.accerl_vllm.ds_com import InferenceActorCom, TrainerActorCom  # noqa: E402


BROADCAST_GROUP_NAME = os.environ.get(
    "DS_COMM_GROUP_NAME",
    "trainer_to_inference_broadcast_benchmark",
)
BROADCAST_GROUP_PORT = int(os.environ.get("DS_COMM_GROUP_PORT", "0"))
NUM_INFERENCE_ACTORS = int(os.environ.get("DS_COMM_NUM_INFER", "1"))
TOTAL_GIB = float(os.environ.get("DS_COMM_TOTAL_GIB", "1.0"))
NUM_TENSORS = int(os.environ.get("DS_COMM_NUM_TENSORS", "64"))
DTYPE_NAME = os.environ.get("DS_COMM_DTYPE", "bfloat16")
WARMUP_REPEATS = int(os.environ.get("DS_COMM_WARMUP_REPEATS", "2"))
BENCHMARK_REPEATS = int(os.environ.get("DS_COMM_REPEATS", "10"))
RESULT_PATH = os.environ.get("DS_COMM_RESULT", "ds_trainer_infer_comm_speed.json")
RAY_TEMP_DIR = os.environ.get("DS_COMM_RAY_TEMP_DIR", "/dev/shm")


DTYPE_TABLE = {
    "float16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float": torch.float32,
}


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def dtype_from_name(dtype_name: str) -> torch.dtype:
    if dtype_name not in DTYPE_TABLE:
        valid = ", ".join(sorted(DTYPE_TABLE))
        raise ValueError(f"Unsupported DS_COMM_DTYPE={dtype_name!r}; valid: {valid}")
    return DTYPE_TABLE[dtype_name]


def dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def tensor_numels(total_gib: float, num_tensors: int, dtype: torch.dtype) -> List[int]:
    if total_gib <= 0:
        raise ValueError("DS_COMM_TOTAL_GIB must be > 0")
    if num_tensors <= 0:
        raise ValueError("DS_COMM_NUM_TENSORS must be > 0")

    total_bytes = int(total_gib * 1024**3)
    total_elems = max(total_bytes // dtype_nbytes(dtype), num_tensors)
    base = total_elems // num_tensors
    rem = total_elems % num_tensors
    return [base + (1 if i < rem else 0) for i in range(num_tensors)]


def free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def timed_ray_get(label: str, refs):
    t0 = now_ms()
    out = ray.get(refs)
    elapsed_ms = now_ms() - t0
    print(f"[TIME] {label:<32} {elapsed_ms:>10.3f} ms", flush=True)
    return out, elapsed_ms


class SyntheticModel(torch.nn.Module):
    def __init__(self, numels: Iterable[int], dtype: torch.dtype):
        super().__init__()
        params: Dict[str, torch.nn.Parameter] = {}
        for idx, numel in enumerate(numels):
            name = f"p_{idx:05d}"
            tensor = torch.empty(int(numel), device="cuda", dtype=dtype)
            torch.nn.init.uniform_(tensor, -0.01, 0.01)
            params[name] = torch.nn.Parameter(tensor, requires_grad=True)
        self.params = torch.nn.ParameterDict(params)


@ray.remote(num_gpus=1)
class BenchmarkTrainerActor(TrainerActorCom):
    def __init__(self, numels: List[int], dtype_name: str):
        super().__init__()
        self.rank = 0
        self.dtype = dtype_from_name(dtype_name)
        self.model = SyntheticModel(numels, self.dtype)
        torch.cuda.synchronize()
        print(
            f"BenchmarkTrainerActor ready on GPU {ray.get_gpu_ids()} "
            f"with {len(numels)} tensors.",
            flush=True,
        )

    def get_node_ip(self):
        return ray.util.get_node_ip_address()

    def get_free_port(self):
        return free_tcp_port()

    def checksum(self) -> float:
        total = torch.zeros((), device="cuda", dtype=torch.float32)
        for param in self.model.parameters():
            total += param.detach().float().sum()
        torch.cuda.synchronize()
        return float(total.cpu().item())


@ray.remote(num_gpus=1)
class BenchmarkInferenceActor(InferenceActorCom):
    def __init__(self, actor_id: int, numels: List[int], dtype_name: str):
        super().__init__()
        self.actor_id = actor_id
        self.dtype = dtype_from_name(dtype_name)
        self.model = SyntheticModel(numels, self.dtype)
        torch.cuda.synchronize()
        print(
            f"BenchmarkInferenceActor {actor_id} ready on GPU {ray.get_gpu_ids()} "
            f"with {len(numels)} tensors.",
            flush=True,
        )

    def checksum(self) -> float:
        total = torch.zeros((), device="cuda", dtype=torch.float32)
        for param in self.model.parameters():
            total += param.detach().float().sum()
        torch.cuda.synchronize()
        return float(total.cpu().item())


def assert_matching_signatures(trainer, inference_pool):
    trainer_sig = ray.get(trainer.get_broadcast_signature.remote())
    for idx, infer in enumerate(inference_pool):
        infer_sig = ray.get(infer.get_broadcast_signature.remote())
        if trainer_sig != infer_sig:
            raise RuntimeError(
                f"Broadcast signature mismatch for inference actor {idx}: "
                f"trainer={len(trainer_sig)} tensors, infer={len(infer_sig)} tensors"
            )
    return trainer_sig


def run_one_transfer(trainer, inference_pool, label: str):
    broadcast_task = trainer.broadcast_weights.remote(BROADCAST_GROUP_NAME)
    receive_tasks = [
        infer.receive_and_update_weights.remote(BROADCAST_GROUP_NAME)
        for infer in inference_pool
    ]
    _, elapsed_ms = timed_ray_get(label, [broadcast_task] + receive_tasks)
    return elapsed_ms


def main():
    dtype = dtype_from_name(DTYPE_NAME)
    numels = tensor_numels(TOTAL_GIB, NUM_TENSORS, dtype)
    total_bytes = sum(numels) * dtype_nbytes(dtype)
    total_gib = total_bytes / 1024**3

    print(
        "[INFO] benchmark config: "
        f"num_infer={NUM_INFERENCE_ACTORS}, tensors={NUM_TENSORS}, "
        f"dtype={DTYPE_NAME}, payload={total_gib:.4f} GiB, "
        f"warmup={WARMUP_REPEATS}, repeats={BENCHMARK_REPEATS}",
        flush=True,
    )

    ray.init(ignore_reinit_error=True, _temp_dir=RAY_TEMP_DIR)

    trainer = BenchmarkTrainerActor.remote(numels, DTYPE_NAME)
    inference_pool = [
        BenchmarkInferenceActor.remote(i, numels, DTYPE_NAME)
        for i in range(NUM_INFERENCE_ACTORS)
    ]

    master_addr = ray.get(trainer.get_node_ip.remote())
    master_port = BROADCAST_GROUP_PORT or ray.get(trainer.get_free_port.remote())
    participants = [trainer] + inference_pool
    group_world_size = len(participants)
    print(
        f"[INFO] setup broadcast group {BROADCAST_GROUP_NAME!r}: "
        f"master={master_addr}:{master_port}, world_size={group_world_size}",
        flush=True,
    )

    setup_tasks = [
        actor.setup_broadcast_group.remote(
            master_addr=master_addr,
            master_port=master_port,
            group_name=BROADCAST_GROUP_NAME,
            group_world_size=group_world_size,
            my_rank_in_group=rank,
        )
        for rank, actor in enumerate(participants)
    ]
    _, setup_ms = timed_ray_get("setup broadcast group", setup_tasks)

    signature = assert_matching_signatures(trainer, inference_pool)
    print(f"[INFO] signatures match: {len(signature)} tensors", flush=True)

    for idx in range(WARMUP_REPEATS):
        run_one_transfer(trainer, inference_pool, f"warmup {idx + 1}/{WARMUP_REPEATS}")

    results = []
    for idx in range(BENCHMARK_REPEATS):
        elapsed_ms = run_one_transfer(
            trainer,
            inference_pool,
            f"broadcast {idx + 1}/{BENCHMARK_REPEATS}",
        )
        elapsed_sec = elapsed_ms / 1000.0
        per_receiver_gib_s = total_gib / elapsed_sec if elapsed_sec > 0 else math.inf
        aggregate_gib_s = (
            total_gib * NUM_INFERENCE_ACTORS / elapsed_sec
            if elapsed_sec > 0
            else math.inf
        )
        result = {
            "repeat": idx + 1,
            "ms": elapsed_ms,
            "payload_GiB": total_gib,
            "per_receiver_GiB_per_s": per_receiver_gib_s,
            "aggregate_GiB_per_s": aggregate_gib_s,
        }
        results.append(result)
        print(
            f"[RESULT] repeat={idx + 1}, time={elapsed_ms:.3f} ms, "
            f"per_receiver={per_receiver_gib_s:.4f} GiB/s, "
            f"aggregate={aggregate_gib_s:.4f} GiB/s",
            flush=True,
        )

    trainer_checksum = ray.get(trainer.checksum.remote())
    infer_checksums = ray.get([infer.checksum.remote() for infer in inference_pool])
    max_checksum_delta = max(
        [abs(trainer_checksum - infer_checksum) for infer_checksum in infer_checksums],
        default=0.0,
    )

    times = [r["ms"] for r in results]
    per_receiver_speeds = [r["per_receiver_GiB_per_s"] for r in results]
    aggregate_speeds = [r["aggregate_GiB_per_s"] for r in results]
    summary = {
        "group_name": BROADCAST_GROUP_NAME,
        "num_inference_actors": NUM_INFERENCE_ACTORS,
        "num_tensors": NUM_TENSORS,
        "dtype": DTYPE_NAME,
        "payload_bytes": total_bytes,
        "payload_GiB": total_gib,
        "warmup_repeats": WARMUP_REPEATS,
        "repeats": BENCHMARK_REPEATS,
        "setup_ms": setup_ms,
        "broadcast_ms_avg": statistics.mean(times) if times else 0.0,
        "broadcast_ms_median": statistics.median(times) if times else 0.0,
        "per_receiver_GiB_per_s_avg": (
            statistics.mean(per_receiver_speeds) if per_receiver_speeds else 0.0
        ),
        "per_receiver_GiB_per_s_median": (
            statistics.median(per_receiver_speeds) if per_receiver_speeds else 0.0
        ),
        "aggregate_GiB_per_s_avg": (
            statistics.mean(aggregate_speeds) if aggregate_speeds else 0.0
        ),
        "aggregate_GiB_per_s_median": (
            statistics.median(aggregate_speeds) if aggregate_speeds else 0.0
        ),
        "trainer_checksum": trainer_checksum,
        "inference_checksums": infer_checksums,
        "max_checksum_delta": max_checksum_delta,
        "results": results,
    }

    print(
        f"[SUMMARY] avg={summary['per_receiver_GiB_per_s_avg']:.4f} GiB/s "
        f"per receiver, aggregate={summary['aggregate_GiB_per_s_avg']:.4f} GiB/s, "
        f"avg_time={summary['broadcast_ms_avg']:.3f} ms, "
        f"checksum_delta={max_checksum_delta:.6f}",
        flush=True,
    )

    # with open(RESULT_PATH, "w") as f:
    #     json.dump(summary, f, indent=2)
    # print(f"[INFO] wrote benchmark result to {RESULT_PATH}", flush=True)

    ray.shutdown()


if __name__ == "__main__":
    main()
