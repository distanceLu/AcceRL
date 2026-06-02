#!/usr/bin/env python3
import subprocess
import time
import re
from collections import defaultdict

INTERVAL = 1.0
GPU_IDS = {0, 1, 2, 3}

pattern_gpu = re.compile(r"GPU\s+(\d+):")
pattern_data = re.compile(r"Link\s+(\d+):\s+Data\s+(Tx|Rx):\s+(\d+)\s+KiB")

def read_nvlink_counters():
    counters = {}

    for gpu_id in sorted(GPU_IDS):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "nvlink", "-gt", "d", "-i", str(gpu_id)],
                text=True,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as e:
            print(f"读取 GPU {gpu_id} 失败：")
            print(e.output)
            continue

        current_gpu = gpu_id

        for line in out.splitlines():
            m_data = pattern_data.search(line)
            if m_data:
                link = int(m_data.group(1))
                direction = m_data.group(2)
                value_kib = int(m_data.group(3))
                counters[(current_gpu, link, direction)] = value_kib

    return counters

prev = read_nvlink_counters()
prev_time = time.time()

while True:
    time.sleep(INTERVAL)

    now = read_nvlink_counters()
    now_time = time.time()
    dt = now_time - prev_time

    per_gpu = defaultdict(lambda: {"Tx": 0.0, "Rx": 0.0})

    for key, now_value in now.items():
        if key not in prev:
            continue

        gpu, link, direction = key
        delta_kib = now_value - prev[key]

        if delta_kib < 0:
            continue

        gib_per_s = delta_kib / 1024 / 1024 / dt
        per_gpu[gpu][direction] += gib_per_s

    print("=" * 70)
    print(time.strftime("%Y-%m-%d %H:%M:%S"))

    for gpu in sorted(GPU_IDS):
        tx = per_gpu[gpu]["Tx"]
        rx = per_gpu[gpu]["Rx"]
        total = tx + rx
        print(f"GPU {gpu}: Tx {tx:8.2f} GiB/s | Rx {rx:8.2f} GiB/s | Total {total:8.2f} GiB/s")

    prev = now
    prev_time = now_time