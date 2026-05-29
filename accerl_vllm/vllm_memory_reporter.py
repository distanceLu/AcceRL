import os

import torch


class CudaMemoryReporter:
    """vLLM worker extension for reporting CUDA memory from the worker process."""

    def report_cuda_memory(self, label: str, empty_cache: bool = False):
        if empty_cache:
            torch.cuda.empty_cache()
        torch.cuda.synchronize()

        device = torch.cuda.current_device()
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        stats = {
            "label": label,
            "pid": os.getpid(),
            "device": device,
            "allocated_gib": torch.cuda.memory_allocated(device) / 1024**3,
            "reserved_gib": torch.cuda.memory_reserved(device) / 1024**3,
            "max_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "max_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
            "driver_used_gib": (total_bytes - free_bytes) / 1024**3,
            "driver_total_gib": total_bytes / 1024**3,
        }
        print(
            "[vLLM worker CUDA memory] "
            f"{label} pid={stats['pid']} cuda:{device} "
            f"allocated={stats['allocated_gib']:.2f}GiB "
            f"reserved={stats['reserved_gib']:.2f}GiB "
            f"max_reserved={stats['max_reserved_gib']:.2f}GiB "
            f"driver_used={stats['driver_used_gib']:.2f}/"
            f"{stats['driver_total_gib']:.2f}GiB",
            flush=True,
        )
        return stats
