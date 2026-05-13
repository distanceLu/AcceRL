import os
import time
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--cuda-visible-devices", default="5,6,7")
parser.add_argument("--vulkan-visible-devices", default=None)
parser.add_argument("--render-backend", default="sapien_cuda:0")
parser.add_argument("--num-workers", type=int, default=6)
parser.add_argument("--sim-backend", default="cpu")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["VK_ICD_FILENAMES"] = "/etc/vulkan/icd.d/nvidia_icd.json"
# os.environ["VK_DRIVER_FILES"] = "/etc/vulkan/icd.d/nvidia_icd.json"

if args.vulkan_visible_devices is not None:
    os.environ["VULKAN_VISIBLE_DEVICES"] = args.vulkan_visible_devices
    os.environ["SAPIEN_VULKAN_DEVICE"] = args.vulkan_visible_devices
    os.environ["EGL_DEVICE_ID"] = args.vulkan_visible_devices

import ray
import numpy as np


# def query_pid_gpus(pid: int):
#     try:
#         import pynvml

#         pynvml.nvmlInit()
#         hits = []
#         for i in range(pynvml.nvmlDeviceGetCount()):
#             h = pynvml.nvmlDeviceGetHandleByIndex(i)
#             name = pynvml.nvmlDeviceGetName(h)
#             if isinstance(name, bytes):
#                 name = name.decode()

#             procs = []
#             for getter in (
#                 pynvml.nvmlDeviceGetComputeRunningProcesses,
#                 getattr(pynvml, "nvmlDeviceGetGraphicsRunningProcesses", None),
#             ):
#                 if getter is None:
#                     continue
#                 try:
#                     procs.extend(getter(h))
#                 except Exception:
#                     pass

#             for p in procs:
#                 if int(p.pid) == int(pid):
#                     used = getattr(p, "usedGpuMemory", 0)
#                     hits.append({"gpu_index": i, "name": name, "used_mb": used / 1024 / 1024})
#                     break
#         return hits
#     except Exception as e:
#         return [{"error": repr(e)}]

def dump_gpu_mapping():
    import subprocess
    import os

    print("=== env ===", flush=True)
    for k in [
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "VULKAN_VISIBLE_DEVICES",
        "SAPIEN_VULKAN_DEVICE",
        "EGL_DEVICE_ID",
        "VK_ICD_FILENAMES",
        "VK_DRIVER_FILES",
        "DISPLAY",
        "__GLX_VENDOR_LIBRARY_NAME",
        "__NV_PRIME_RENDER_OFFLOAD",
    ]:
        print(f"{k}={os.environ.get(k)}", flush=True)

    print("=== nvidia-smi topo ===", flush=True)
    subprocess.run("nvidia-smi topo -m", shell=True, check=False)

    print("=== vulkaninfo summary ===", flush=True)
    subprocess.run("vulkaninfo --summary | sed -n '/Devices:/,$p'", shell=True, check=False)

def print_gpu_hits(tag, pid):
    print(f"\n=== GPU hits: {tag} ===", flush=True)
    print(query_pid_gpus(pid), flush=True)


def query_pid_gpus(pid: int):
    try:
        import pynvml

        pynvml.nvmlInit()
        hits = []

        getters = [
            ("compute", pynvml.nvmlDeviceGetComputeRunningProcesses),
            ("graphics", getattr(pynvml, "nvmlDeviceGetGraphicsRunningProcesses", None)),
        ]

        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()

            for proc_type, getter in getters:
                if getter is None:
                    continue

                try:
                    procs = getter(h)
                except Exception:
                    continue

                for p in procs:
                    if int(p.pid) == int(pid):
                        used = getattr(p, "usedGpuMemory", 0)
                        hits.append({
                            "gpu_index": i,
                            "name": name,
                            "type": proc_type,
                            "used_mb": used / 1024 / 1024,
                        })

        return hits
    except Exception as e:
        return [{"error": repr(e)}]



@ray.remote(num_gpus=0.01)
class RenderProbeWorker:
    def __init__(self, wid, render_backend, sim_backend):
        self.wid = wid
        self.pid = os.getpid()

        print(
            f"[worker {wid}] pid={self.pid} "
            f"ray_gpu_ids={ray.get_gpu_ids()} "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
            f"VULKAN_VISIBLE_DEVICES={os.environ.get('VULKAN_VISIBLE_DEVICES')} "
            f"SAPIEN_VULKAN_DEVICE={os.environ.get('SAPIEN_VULKAN_DEVICE')} "
            f"render_backend={render_backend}",
            flush=True,
        )


        from rl.maniskill_env import ManiSkillSingleEnv


        self.env = ManiSkillSingleEnv(
            task_id="PickCube-v1",
            camera_name="base_camera",
            camera_res=224,
            max_episode_steps=20,
            use_proprio=False,
            sim_backend=sim_backend,
            render_backend=render_backend,
            wrist_camera_name="hand_camera",
            robot_uids="panda_wristcam",
        )
        print_gpu_hits("after ManiSkillSingleEnv construction", self.pid)

    def probe(self):

        obs, info = self.env.reset(seed=1000 + self.wid)


        for i in range(3):
            action = np.random.uniform(-1, 1, size=(7,)).astype(np.float32)
            self.env.step(action)
            time.sleep(0.2)

        gpu_hits = query_pid_gpus(self.pid)

        return {
            "wid": self.wid,
            "pid": self.pid,
            "ray_gpu_ids": ray.get_gpu_ids(),
            "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "vulkan_visible": os.environ.get("VULKAN_VISIBLE_DEVICES"),
            "sapien_vulkan_device": os.environ.get("SAPIEN_VULKAN_DEVICE"),
            "gpu_hits": gpu_hits,
        }

    def close(self):
        self.env.close()


if __name__ == "__main__":
    # print("nvidia-smi:")
    # subprocess.run(
    #     [
    #         "nvidia-smi",
    #         "--query-gpu=index,uuid,pci.bus_id,name",
    #         "--format=csv,noheader",
    #     ],
    #     check=False,
    # )

    ray.init(ignore_reinit_error=True, _temp_dir="/dev/shm")

    workers = [
        RenderProbeWorker.remote(i, args.render_backend, args.sim_backend)
        for i in range(args.num_workers)
    ]

    results = ray.get([w.probe.remote() for w in workers])

    print("\n=== Worker render GPU probe ===")
    for r in results:
        print(r)

    ray.get([w.close.remote() for w in workers])
    ray.shutdown()