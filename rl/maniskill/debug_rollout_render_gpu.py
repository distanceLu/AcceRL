import os
import time
import argparse
import subprocess
from types import SimpleNamespace

parser = argparse.ArgumentParser(
    description="Probe which physical GPUs BaseWorkerActor / ManiSkill touches under Ray."
)
parser.add_argument("--cuda-visible-devices", default="5,6,7")
parser.add_argument("--num-workers", type=int, default=6)
parser.add_argument("--sim-backend", default="cpu")
parser.add_argument("--pretrained-checkpoint", required=True)
parser.add_argument(
    "--vulkan-visible-devices",
    type=str,
    default=None,
    help="Set VULKAN_VISIBLE_DEVICES on driver and in Ray worker runtime_env (e.g. 4 or 4,5).",
)
parser.add_argument(
    "--sapien-vulkan-device",
    type=str,
    default=None,
    help="Set SAPIEN_VULKAN_DEVICE on driver and in worker runtime_env (physical index string).",
)
parser.add_argument(
    "--egl-device-id",
    type=str,
    default=None,
    help="Set EGL_DEVICE_ID on driver and in worker runtime_env.",
)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["VK_ICD_FILENAMES"] = "/etc/vulkan/icd.d/nvidia_icd.json"
if args.vulkan_visible_devices:
    os.environ["VULKAN_VISIBLE_DEVICES"] = args.vulkan_visible_devices
if args.sapien_vulkan_device:
    os.environ["SAPIEN_VULKAN_DEVICE"] = args.sapien_vulkan_device
if args.egl_device_id:
    os.environ["EGL_DEVICE_ID"] = args.egl_device_id

import ray
import numpy as np

from rl.maniskill.ds_maniskill_ppo_discrete import BaseWorkerActor, build_openvla_cfg

# import ds_maniskill_ppo_discrete 后会被顶层 CUDA_VISIBLE_DEVICES 覆盖，
# 所以这里再设一次，保证 ray.init 前 driver 环境是你传的值。
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices


def query_pid_gpus(pid: int):
    try:
        import pynvml

        pynvml.nvmlInit()
        hits = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()

            procs = []
            for getter in (
                pynvml.nvmlDeviceGetComputeRunningProcesses,
                getattr(pynvml, "nvmlDeviceGetGraphicsRunningProcesses", None),
            ):
                if getter is None:
                    continue
                try:
                    procs.extend(getter(h))
                except Exception:
                    pass

            for p in procs:
                if int(p.pid) == int(pid):
                    used = getattr(p, "usedGpuMemory", 0)
                    hits.append({"gpu_index": i, "name": name, "used_mb": used / 1024 / 1024})
                    break

        return hits
    except Exception as e:
        return [{"error": repr(e)}]


@ray.remote(num_gpus=0.01)
class DebugRolloutWorker(BaseWorkerActor):
    def __init__(self, wid, cfg, env_args):
        self.pid = os.getpid()

        print(
            f"[debug rollout {wid}] before BaseWorkerActor init: "
            f"pid={self.pid}, "
            f"ray_gpu_ids={ray.get_gpu_ids()}, "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
            f"VULKAN_VISIBLE_DEVICES={os.environ.get('VULKAN_VISIBLE_DEVICES')}, "
            f"SAPIEN_VULKAN_DEVICE={os.environ.get('SAPIEN_VULKAN_DEVICE')}",
            flush=True,
        )

        super().__init__(
            infer=None,
            replay=None,
            wid=wid,
            stats_actor=None,
            cfg=cfg,
            env_args=env_args,
        )

    def probe(self, warmup_steps=3):
        obs, info = self.env.reset(seed=100000 + int(self.wid))

        for _ in range(warmup_steps):
            action = self.env.env.action_space.sample()
            self.env.step(action)
            time.sleep(0.3)

        return {
            "wid": self.wid,
            "pid": self.pid,
            "ray_gpu_ids": ray.get_gpu_ids(),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "VULKAN_VISIBLE_DEVICES": os.environ.get("VULKAN_VISIBLE_DEVICES"),
            "SAPIEN_VULKAN_DEVICE": os.environ.get("SAPIEN_VULKAN_DEVICE"),
            "EGL_DEVICE_ID": os.environ.get("EGL_DEVICE_ID"),
            "VK_ICD_FILENAMES": os.environ.get("VK_ICD_FILENAMES"),
            "gpu_hits": query_pid_gpus(self.pid),
        }

    def close(self):
        self.env.close()


if __name__ == "__main__":
    subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name",
            "--format=csv,noheader",
        ],
        check=False,
    )

    cfg_args = SimpleNamespace(
        pretrained_checkpoint=args.pretrained_checkpoint,
        num_images_in_input=2,
        use_proprio=False,
        unnorm_key="maniskill_pickcube",
        checkpoint2="",
    )
    cfg = build_openvla_cfg(cfg_args)

    env_args = {
        "maniskill_task": "PickCube-v1",
        "camera_name": "base_camera",
        "camera_res": 224,
        "max_episode_steps": 20,
        "language_instruction": "pick up the red cube and place it at the green target",
        "sim_backend": args.sim_backend,
        "wrist_camera_name": "hand_camera",
        "robot_uids": "panda_wristcam",
    }

    # Vulkan/SAPIEN/EGL: inject at worker process start via runtime_env (before worker imports).
    # Do not set CUDA_VISIBLE_DEVICES here — Ray assigns it per GPU actor.
    worker_env = {
        "VK_ICD_FILENAMES": os.environ.get(
            "VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json"
        ),
    }
    if args.vulkan_visible_devices:
        worker_env["VULKAN_VISIBLE_DEVICES"] = args.vulkan_visible_devices
    if args.sapien_vulkan_device:
        worker_env["SAPIEN_VULKAN_DEVICE"] = args.sapien_vulkan_device
    if args.egl_device_id:
        worker_env["EGL_DEVICE_ID"] = args.egl_device_id

    ray.init(
        ignore_reinit_error=True,
        _temp_dir="/dev/shm",
        runtime_env={"env_vars": worker_env},
    )

    workers = [
        DebugRolloutWorker.remote(i, cfg, env_args)
        for i in range(args.num_workers)
    ]

    results = ray.get([w.probe.remote() for w in workers])

    print("\n=== Debug workers using training BaseWorkerActor init ===")
    for r in results:
        print(r)

    ray.get([w.close.remote() for w in workers])
    ray.shutdown()