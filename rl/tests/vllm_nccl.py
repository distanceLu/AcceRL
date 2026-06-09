# import os

# # 建议放在最顶部，避免 awex_adapter 插件报 megatron 缺失
# os.environ["VLLM_PLUGINS"] = ""

# import ray
# from ray.util.placement_group import placement_group
# from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
# import torch
# from transformers import AutoModelForImageTextToText
# # from transformers import AutoModelForCausalLM

# from vllm import LLM, SamplingParams
# from vllm.config import WeightTransferConfig
# from vllm.distributed.weight_transfer.nccl_engine import (
#     NCCLTrainerSendWeightsArgs,
#     NCCLWeightTransferEngine,
# )
# from vllm.utils.network_utils import get_ip, get_open_port


# MODEL_PATH = "/cpfs01/luck_hub_cache/huggingface/hub/Qwen3-VL-8B-Instruct"

# class vllmInfer(LLM):
#     def __init__(self, *args, **kwargs):
#         os.environ["VLLM_RAY_BUNDLE_INDICES"] = "0"
#         super().__init__(*args, **kwargs)


# @ray.remote(num_gpus=1)
# class TrainerModel:
#     def __init__(self, model_path: str):
#         self.model = AutoModelForImageTextToText.from_pretrained(
#             model_path,
#             torch_dtype=torch.bfloat16,
#             local_files_only=True,
#             trust_remote_code=True,
#         ).to("cuda:0")

#         self.port = get_open_port()
#         self.master_address = get_ip()
    
#     def get_master_address_and_port(self):
#         return self.master_address, self.port

#     def get_weight_metadata(self):
#         names, dtype_names, shapes = [], [], []
#         for name, p in self.model.named_parameters():
#             names.append(name)
#             dtype_names.append(str(p.dtype).split(".")[-1])
#             shapes.append(list(p.shape))
        
#         return names, dtype_names, shapes
    
#     def init_weight_transfer_group(self, world_size):
#         self.model_update_group = NCCLWeightTransferEngine.trainer_init(
#             dict(
#                 master_address=self.master_address,
#                 master_port=self.port,
#                 world_size=world_size,
#             ),
#         )
    
#     def broadcast_weights(self, packed: bool=True):
#         trainer_args = NCCLTrainerSendWeightsArgs(
#             group=self.model_update_group,
#             packed=packed,
#         )
#         NCCLWeightTransferEngine.trainer_send_weights(
#             iterator=self.model.named_parameters(),
#             trainer_args=trainer_args,
#         )

# ray.init(address="local")

# trainer_model = TrainerModel.remote(MODEL_PATH)

# pg_inference = placement_group([{"GPU": 1, "CPU":0}])
# ray.get(pg_inference.ready())
# scheduling_inference = PlacementGroupSchedulingStrategy(
#     placement_group=pg_inference,
#     placement_group_capture_child_tasks=True,
#     placement_group_bundle_index=0,
# )

# vllm_infer = ray.remote(
#     num_cpus=0,
#     num_gpus=0,
#     scheduling_strategy=scheduling_inference,
# )(vllmInfer).remote(
#     model=MODEL_PATH,
#     enforce_eager=True,
#     tensor_parallel_size=1,
#     data_parallel_size=1,
#     distributed_executor_backend="ray",
#     weight_transfer_config=WeightTransferConfig(backend="nccl"),
#     load_format="dummy",
#     trust_remote_code=True,
#     dtype="bfloat16",

#     # 显存保护：先别用 8192，验证跑通后再调大
#     max_model_len=1024,

#     # 核心：别让 vLLM 预留 85% 显存
#     # 如果这张卡还跑着 TrainerModel，建议 0.30 ~ 0.45
#     gpu_memory_utilization=0.30,

#     # 控制单次调度的 token 总量，避免 prefill 吃太多 KV cache
#     max_num_batched_tokens=1024,

#     # 控制并发请求数，你这里只有 4 个 prompt，设 4 就够了
#     max_num_seqs=4,

#     limit_mm_per_prompt={"image": 0, "video": 0},
# )

# # vllm文字prompts
# prompts = [
#     "Hello, my name is",
#     "The president of the United States is",
#     "The capital of France is",
#     "The future of AI is",
# ]

# sampling_params = SamplingParams(temperature=0)

# outputs = ray.get(vllm_infer.generate.remote(prompts, sampling_params))

# print("-" * 50)
# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
#     print("-" * 50)    

# ray.get(vllm_infer.sleep.remote(level=0))

# master_address, master_port = ray.get(trainer_model.get_master_address_and_port.remote())

# world_size = ray.get(vllm_infer.get_world_size.remote()) + 1
# inference_handle = vllm_infer.init_weight_transfer_engine.remote(
#     dict(
#         init_info=dict(
#             master_address=master_address,
#             master_port=master_port,
#             rank_offset=1,
#             world_size=world_size,
#         )
#     )
# )

# train_handle = trainer_model.init_weight_transfer_group.remote(world_size)
# ray.get([train_handle, inference_handle])

# names, dtype_names, shapes = ray.get(trainer_model.get_weight_metadata.remote())

# inference_handle = vllm_infer.update_weights.remote(
#     dict(
#         update_info=dict(
#             names=names,
#             dtype_names=dtype_names,
#             shapes=shapes,
#             packed=True,
#         )
#     )
# )

# trainer_handle = trainer_model.broadcast_weights.remote(packed=True)
# ray.get([trainer_handle, inference_handle])

# ray.get(vllm_infer.wake_up.remote(tags=["scheduling"]))

# outputs_updated = ray.get(vllm_infer.generate.remote(prompts, sampling_params))

# print("-" * 50)
# for output in outputs_updated:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
#     print("-" * 50)


import os

# 建议放在最顶部，避免 awex_adapter 插件报 megatron 缺失
os.environ["VLLM_PLUGINS"] = ""

import json
import math
import time
from typing import Any, Dict, List, Tuple

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import torch
from transformers import AutoModelForImageTextToText

from vllm import LLM, SamplingParams
from vllm.config import WeightTransferConfig
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerSendWeightsArgs,
    NCCLWeightTransferEngine,
)
from vllm.utils.network_utils import get_ip, get_open_port


MODEL_PATH = "/cpfs01/luck_hub_cache/huggingface/hub/Qwen3-VL-8B-Instruct"
PACKED = True


def cuda_sync() -> None:
    """NCCL/CUDA 操作通常是异步的；计时前后必须同步。"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def elapsed_ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def metric(step: str, ms: float, **extra: Any) -> Dict[str, Any]:
    out = {"step": step, "ms": ms}
    out.update(extra)
    return out


def print_metric(m: Dict[str, Any]) -> None:
    extras = []
    for k, v in m.items():
        if k in {"step", "ms"}:
            continue
        if isinstance(v, float):
            extras.append(f"{k}={v:.4f}")
        else:
            extras.append(f"{k}={v}")
    suffix = " | " + ", ".join(extras) if extras else ""
    print(f"[TIME] {m['step']:<42} {m['ms']:>12.3f} ms{suffix}", flush=True)


def timed_ray_get(step: str, refs):
    t0 = time.perf_counter()
    result = ray.get(refs)
    m = metric(step, elapsed_ms(t0))
    print_metric(m)
    return result, m


def dtype_nbytes(dtype_name: str) -> int:
    table = {
        "bool": 1,
        "uint8": 1,
        "int8": 1,
        "float16": 2,
        "half": 2,
        "bfloat16": 2,
        "float32": 4,
        "float": 4,
        "int32": 4,
        "float64": 8,
        "double": 8,
        "int64": 8,
        "long": 8,
    }
    return table.get(dtype_name, 0)


def estimate_weight_bytes(dtype_names: List[str], shapes: List[List[int]]) -> int:
    total = 0
    for dtype_name, shape in zip(dtype_names, shapes):
        nb = dtype_nbytes(dtype_name)
        if nb == 0:
            continue
        total += math.prod(shape) * nb
    return total


class vllmInfer(LLM):
    def __init__(self, *args, **kwargs):
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = "0"
        t0 = time.perf_counter()
        super().__init__(*args, **kwargs)
        cuda_sync()
        self.init_ms = elapsed_ms(t0)

    def ping(self):
        return metric("vllm actor __init__", self.init_ms)

    def timed_generate(self, prompts, sampling_params):
        cuda_sync()
        t0 = time.perf_counter()
        outputs = self.generate(prompts, sampling_params)
        cuda_sync()
        return outputs, metric("vllm generate", elapsed_ms(t0), num_prompts=len(prompts))

    def timed_sleep(self, level: int = 0):
        cuda_sync()
        t0 = time.perf_counter()
        ret = self.sleep(level=level)
        cuda_sync()
        return metric("vllm sleep", elapsed_ms(t0), level=level, ret=str(ret))

    def timed_init_weight_transfer_engine(self, config: Dict[str, Any]):
        cuda_sync()
        t0 = time.perf_counter()
        ret = self.init_weight_transfer_engine(config)
        cuda_sync()
        return metric("infer init_weight_transfer_engine", elapsed_ms(t0), ret=str(ret))

    def timed_update_weights(self, config: Dict[str, Any]):
        # 接收端：等待 trainer_send_weights 发起并完成 NCCL 传输。
        cuda_sync()
        t0 = time.perf_counter()
        ret = self.update_weights(config)
        cuda_sync()
        return metric("infer update_weights recv", elapsed_ms(t0), ret=str(ret))

    def timed_wake_up(self, tags=None):
        cuda_sync()
        t0 = time.perf_counter()
        ret = self.wake_up(tags=tags)
        cuda_sync()
        return metric("vllm wake_up", elapsed_ms(t0), tags=tags, ret=str(ret))


@ray.remote(num_gpus=1)
class TrainerModel:
    def __init__(self, model_path: str):
        t0 = time.perf_counter()
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            trust_remote_code=True,
        ).to("cuda:0")
        cuda_sync()
        self.init_metric = metric("trainer model load", elapsed_ms(t0))

        self.port = get_open_port()
        self.master_address = get_ip()

    def ping(self):
        return self.init_metric

    def get_master_address_and_port(self):
        return self.master_address, self.port

    def timed_get_weight_metadata(self):
        t0 = time.perf_counter()
        names, dtype_names, shapes = [], [], []
        for name, p in self.model.named_parameters():
            names.append(name)
            dtype_names.append(str(p.dtype).split(".")[-1])
            shapes.append(list(p.shape))
        total_bytes = estimate_weight_bytes(dtype_names, shapes)
        m = metric(
            "trainer get_weight_metadata",
            elapsed_ms(t0),
            n_params=len(names),
            GiB=total_bytes / (1024**3),
        )
        return names, dtype_names, shapes, m

    def timed_init_weight_transfer_group(self, world_size: int):
        cuda_sync()
        t0 = time.perf_counter()
        self.model_update_group = NCCLWeightTransferEngine.trainer_init(
            dict(
                master_address=self.master_address,
                master_port=self.port,
                world_size=world_size,
            ),
        )
        cuda_sync()
        return metric("trainer init_weight_transfer_group", elapsed_ms(t0), world_size=world_size)

    def timed_broadcast_weights(self, packed: bool = True):
        # 发送端：这里的时间包含遍历参数、打包/拷贝，以及 NCCL send/broadcast 的完成等待。
        trainer_args = NCCLTrainerSendWeightsArgs(
            group=self.model_update_group,
            packed=packed,
        )
        cuda_sync()
        t0 = time.perf_counter()
        NCCLWeightTransferEngine.trainer_send_weights(
            iterator=self.model.named_parameters(),
            trainer_args=trainer_args,
        )
        cuda_sync()
        return metric("trainer broadcast_weights send", elapsed_ms(t0), packed=packed)


ray.init(address="local")

all_metrics: List[Dict[str, Any]] = []

# 1. Trainer actor/model load
trainer_model = TrainerModel.remote(MODEL_PATH)
trainer_init, m = timed_ray_get("driver wait trainer actor ready", trainer_model.ping.remote())
print_metric(trainer_init)
all_metrics.extend([m, trainer_init])

# 2. Placement group
pg_inference = placement_group([{"GPU": 1, "CPU": 0}])
_, m = timed_ray_get("placement_group ready", pg_inference.ready())
all_metrics.append(m)

scheduling_inference = PlacementGroupSchedulingStrategy(
    placement_group=pg_inference,
    placement_group_capture_child_tasks=True,
    placement_group_bundle_index=0,
)

# 3. vLLM actor/init
vllm_infer = ray.remote(
    num_cpus=0,
    num_gpus=0,
    scheduling_strategy=scheduling_inference,
)(vllmInfer).remote(
    model=MODEL_PATH,
    enforce_eager=True,
    tensor_parallel_size=1,
    data_parallel_size=1,
    distributed_executor_backend="ray",
    weight_transfer_config=WeightTransferConfig(backend="nccl"),
    load_format="dummy",
    trust_remote_code=True,
    dtype="bfloat16",
    max_model_len=1024,
    gpu_memory_utilization=0.30,
    max_num_batched_tokens=1024,
    max_num_seqs=4,
    limit_mm_per_prompt={"image": 0, "video": 0},
)

vllm_init, m = timed_ray_get("driver wait vllm actor ready", vllm_infer.ping.remote())
print_metric(vllm_init)
all_metrics.extend([m, vllm_init])

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0)

# 4. Baseline generate
(outputs, gen_m), m = timed_ray_get(
    "driver wait baseline generate",
    vllm_infer.timed_generate.remote(prompts, sampling_params),
)
print_metric(gen_m)
all_metrics.extend([m, gen_m])

print("-" * 50)
for output in outputs:
    print(f"Prompt: {output.prompt!r}\nGenerated text: {output.outputs[0].text!r}")
    print("-" * 50)

# 5. vLLM sleep
sleep_m, m = timed_ray_get("driver wait vllm sleep", vllm_infer.timed_sleep.remote(level=0))
print_metric(sleep_m)
all_metrics.extend([m, sleep_m])

# 6. Address/world_size
(master_address, master_port), m = timed_ray_get(
    "driver get trainer address", trainer_model.get_master_address_and_port.remote()
)
all_metrics.append(m)

vllm_world_size, m = timed_ray_get("driver get vllm world_size", vllm_infer.get_world_size.remote())
world_size = vllm_world_size + 1
all_metrics.append(m)
print(f"[INFO] master={master_address}:{master_port}, vllm_world_size={vllm_world_size}, total_world_size={world_size}", flush=True)

# 7. NCCL group init: receiver and sender concurrently
infer_init_ref = vllm_infer.timed_init_weight_transfer_engine.remote(
    dict(
        init_info=dict(
            master_address=master_address,
            master_port=master_port,
            rank_offset=1,
            world_size=world_size,
        )
    )
)
trainer_init_ref = trainer_model.timed_init_weight_transfer_group.remote(world_size)
(group_init_metrics, group_driver_m) = timed_ray_get(
    "driver wait nccl group init both",
    [trainer_init_ref, infer_init_ref],
)
for mm in group_init_metrics:
    print_metric(mm)
all_metrics.append(group_driver_m)
all_metrics.extend(group_init_metrics)

# 8. Metadata
(names, dtype_names, shapes, metadata_m), m = timed_ray_get(
    "driver wait metadata", trainer_model.timed_get_weight_metadata.remote()
)
print_metric(metadata_m)
all_metrics.extend([m, metadata_m])

total_bytes = estimate_weight_bytes(dtype_names, shapes)
total_gib = total_bytes / (1024**3)
print(f"[INFO] transfer tensor metadata: n_tensors={len(names)}, estimated_weight_size={total_gib:.4f} GiB", flush=True)

# 9. NCCL weight transfer: receiver first, sender second, wait together
infer_update_ref = vllm_infer.timed_update_weights.remote(
    dict(
        update_info=dict(
            names=names,
            dtype_names=dtype_names,
            shapes=shapes,
            packed=PACKED,
        )
    )
)
trainer_bcast_ref = trainer_model.timed_broadcast_weights.remote(packed=PACKED)

(transfer_metrics, transfer_driver_m) = timed_ray_get(
    "driver wait weight transfer both",
    [trainer_bcast_ref, infer_update_ref],
)
trainer_send_m, infer_recv_m = transfer_metrics
for mm in transfer_metrics:
    print_metric(mm)

transfer_sec = transfer_driver_m["ms"] / 1000.0
if transfer_sec > 0 and total_bytes > 0:
    transfer_driver_m["GiB"] = total_gib
    transfer_driver_m["GiB_per_s"] = total_gib / transfer_sec
    print_metric(transfer_driver_m)

all_metrics.append(transfer_driver_m)
all_metrics.extend(transfer_metrics)

# 10. wake_up
wake_m, m = timed_ray_get("driver wait vllm wake_up", vllm_infer.timed_wake_up.remote(tags=["scheduling"]))
print_metric(wake_m)
all_metrics.extend([m, wake_m])

# 11. Generate after update
(outputs_updated, gen2_m), m = timed_ray_get(
    "driver wait updated generate",
    vllm_infer.timed_generate.remote(prompts, sampling_params),
)
print_metric(gen2_m)
all_metrics.extend([m, gen2_m])

print("-" * 50)
for output in outputs_updated:
    print(f"Prompt: {output.prompt!r}\nGenerated text: {output.outputs[0].text!r}")
    print("-" * 50)

# 12. JSON summary，方便后续 grep / 画图
summary_path = "nccl_step_benchmark_metrics.json"
with open(summary_path, "w") as f:
    json.dump(all_metrics, f, indent=2, ensure_ascii=False)
print(f"[INFO] wrote metrics to {summary_path}", flush=True)
