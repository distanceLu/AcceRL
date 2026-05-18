import os

# 建议放在最顶部，避免 awex_adapter 插件报 megatron 缺失
os.environ["VLLM_PLUGINS"] = ""

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import torch
from transformers import AutoModelForImageTextToText
# from transformers import AutoModelForCausalLM

from vllm import LLM, SamplingParams
from vllm.config import WeightTransferConfig
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerSendWeightsArgs,
    NCCLWeightTransferEngine,
)
from vllm.utils.network_utils import get_ip, get_open_port


MODEL_PATH = "/cpfs01/luck_hub_cache/huggingface/hub/Qwen3-VL-8B-Instruct"

class vllmInfer(LLM):
    def __init__(self, *args, **kwargs):
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = "0"
        super().__init__(*args, **kwargs)


@ray.remote(num_gpus=1)
class TrainerModel:
    def __init__(self, model_path: str):
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            trust_remote_code=True,
        ).to("cuda:0")

        self.port = get_open_port()
        self.master_address = get_ip()
    
    def get_master_address_and_port(self):
        return self.master_address, self.port

    def get_weight_metadata(self):
        names, dtype_names, shapes = [], [], []
        for name, p in self.model.named_parameters():
            names.append(name)
            dtype_names.append(str(p.dtype).split(".")[-1])
            shapes.append(list(p.shape))
        
        return names, dtype_names, shapes
    
    def init_weight_transfer_group(self, world_size):
        self.model_update_group = NCCLWeightTransferEngine.trainer_init(
            dict(
                master_address=self.master_address,
                master_port=self.port,
                world_size=world_size,
            ),
        )
    
    def broadcast_weights(self, packed: bool=True):
        trainer_args = NCCLTrainerSendWeightsArgs(
            group=self.model_update_group,
            packed=packed,
        )
        NCCLWeightTransferEngine.trainer_send_weights(
            iterator=self.model.named_parameters(),
            trainer_args=trainer_args,
        )

ray.init(address="local")

trainer_model = TrainerModel.remote(MODEL_PATH)

pg_inference = placement_group([{"GPU": 1, "CPU":0}])
ray.get(pg_inference.ready())
scheduling_inference = PlacementGroupSchedulingStrategy(
    placement_group=pg_inference,
    placement_group_capture_child_tasks=True,
    placement_group_bundle_index=0,
)

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

    # 显存保护：先别用 8192，验证跑通后再调大
    max_model_len=1024,

    # 核心：别让 vLLM 预留 85% 显存
    # 如果这张卡还跑着 TrainerModel，建议 0.30 ~ 0.45
    gpu_memory_utilization=0.30,
    
    # 控制单次调度的 token 总量，避免 prefill 吃太多 KV cache
    max_num_batched_tokens=1024,

    # 控制并发请求数，你这里只有 4 个 prompt，设 4 就够了
    max_num_seqs=4,

    limit_mm_per_prompt={"image": 0, "video": 0},
)

# vllm文字prompts
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0)

outputs = ray.get(vllm_infer.generate.remote(prompts, sampling_params))

print("-" * 50)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
    print("-" * 50)    

ray.get(vllm_infer.sleep.remote(level=0))

master_address, master_port = ray.get(trainer_model.get_master_address_and_port.remote())

world_size = ray.get(vllm_infer.get_world_size.remote()) + 1
inference_handle = vllm_infer.init_weight_transfer_engine.remote(
    dict(
        init_info=dict(
            master_address=master_address,
            master_port=master_port,
            rank_offset=1,
            world_size=world_size,
        )
    )
)

train_handle = trainer_model.init_weight_transfer_group.remote(world_size)
ray.get([train_handle, inference_handle])

names, dtype_names, shapes = ray.get(trainer_model.get_weight_metadata.remote())

inference_handle = vllm_infer.update_weights.remote(
    dict(
        update_info=dict(
            names=names,
            dtype_names=dtype_names,
            shapes=shapes,
            packed=True,
        )
    )
)

trainer_handle = trainer_model.broadcast_weights.remote(packed=True)
ray.get([trainer_handle, inference_handle])

ray.get(vllm_infer.wake_up.remote(tags=["scheduling"]))

outputs_updated = ray.get(vllm_infer.generate.remote(prompts, sampling_params))

print("-" * 50)
for output in outputs_updated:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
    print("-" * 50)

