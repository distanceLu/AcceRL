"""HF_MODEL_ID=Alibaba-DAMO-Academy/RynnBrain-8B CUDA_VISIBLE_DEVICES=6,7 python vllm_nccl.py"""


import asyncio
import math
import time
import uuid
from dataclasses import asdict

import ray
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import vllm
from vllm import SamplingParams
from vllm.config import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
)
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerSendWeightsArgs,
    NCCLWeightTransferEngine,
    NCCLWeightTransferInitInfo,
    NCCLWeightTransferUpdateInfo,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_ip, get_open_port
from vllm.v1.executor import Executor

MODEL_NAME_V1 = "Qwen/Qwen3-1.7B-Base"
MODEL_NAME_V2 = "Qwen/Qwen3-1.7B"
PAUSE_TOKEN_THRESHOLD = 10
ATTN_BACKEND = "TRITON_ATTN" if current_platform.is_rocm() else "FLASH_ATTN"
WORKER_EXTENSION_CLS = "vllm_memory_reporter.CudaMemoryReporter"


class MyLLM(vllm.AsyncLLMEngine):
    """Configure the vLLM worker for Ray placement group execution."""

    def __init__(self, **kwargs):
        engine_args = vllm.AsyncEngineArgs(**kwargs)
        vllm_config = engine_args.create_engine_config()
        executor_class = Executor.get_class(vllm_config)
        super().__init__(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_requests=engine_args.enable_log_requests,
            log_stats=not engine_args.disable_log_stats,
        )
        self._generation_paused = False
        self._request_pause_flag = False
        

    async def do_generate(
        self, prompt_token_ids: list[int], sampling_params: vllm.SamplingParams
    ) -> tuple[vllm.RequestOutput, int]:
        """Generate a single request, setting the request pause flag once the
        token count reaches the threshold.

        Returns (output, pause_token_index). pause_token_index is the number
        of tokens generated before the weight change, or -1 if no pause.
        """
        pause_token_index = -1
        async for request_output in self.generate(
            {"prompt_token_ids": prompt_token_ids},
            sampling_params,
            request_id=str(uuid.uuid4()),
        ):
            output = request_output
            cur_token_count = len(output.outputs[0].token_ids)
            if (
                cur_token_count >= PAUSE_TOKEN_THRESHOLD
                and not self._request_pause_flag
            ):
                self._request_pause_flag = True
            if self._generation_paused and pause_token_index == -1:
                # pause_generation(mode="keep") may drain one in-flight decode
                # step before the pause is observed by this stream. That token
                # is already part of the pre-update KV context, so the split
                # point must include the current streamed token count.
                pause_token_index = cur_token_count
        return output, pause_token_index

    async def do_generate_no_pause(
        self, prompt_token_ids: list[int], sampling_params: vllm.SamplingParams
    ) -> vllm.RequestOutput:
        """Generate a single request without touching the pause bookkeeping."""
        output = None
        async for request_output in self.generate(
            {"prompt_token_ids": prompt_token_ids},
            sampling_params,
            request_id=str(uuid.uuid4()),
        ):
            output = request_output
        assert output is not None
        return output

    async def report_cuda_memory(self, label: str, empty_cache: bool = False):
        """Return CUDA memory stats from the actual vLLM GPU worker."""
        stats_list = await self.collective_rpc(
            "report_cuda_memory",
            kwargs={"label": label, "empty_cache": empty_cache},
        )
        for stats in stats_list:
            print(
                "[vLLM worker CUDA memory returned] "
                f"{stats['label']} pid={stats['pid']} cuda:{stats['device']} "
                f"allocated={stats['allocated_gib']:.2f}GiB "
                f"reserved={stats['reserved_gib']:.2f}GiB "
                f"max_reserved={stats['max_reserved_gib']:.2f}GiB "
                f"driver_used={stats['driver_used_gib']:.2f}/"
                f"{stats['driver_total_gib']:.2f}GiB",
                flush=True,
            )
        return stats_list

    async def pause_after_n_tokens(self):
        """Wait for any request to set the pause flag, then pause."""
        while not self._request_pause_flag:
            await asyncio.sleep(0)
        await super().pause_generation(mode="keep")
        await asyncio.sleep(5)
        self._generation_paused = True


@ray.remote(num_gpus=1)
class TrainModel:
    """Ray actor that wraps the training model on a dedicated GPU."""

    def __init__(self, model_name: str):
        from vllm.model_executor.layers.batch_invariant import (
            init_batch_invariance,
        )

        # need to init all env vars for batch invariance which affect nccl ops
        init_batch_invariance()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16
        ).to("cuda:0")
        self.port = get_open_port()
        self.master_address = get_ip()

    def get_master_address_and_port(self):
        return self.master_address, self.port

    def get_weight_metadata(self):
        """Return weight names, dtypes, and shapes for weight transfer."""
        names = []
        dtype_names = []
        shapes = []
        for name, p in self.model.named_parameters():
            names.append(name)
            dtype_names.append(str(p.dtype).split(".")[-1])
            shapes.append(list(p.shape))
        return names, dtype_names, shapes

    def init_weight_transfer_group(self, world_size):
        """Initialize the NCCL process group for weight transfer."""
        self.model_update_group = NCCLWeightTransferEngine.trainer_init(
            dict(
                master_address=self.master_address,
                master_port=self.port,
                world_size=world_size,
            ),
        )

    def broadcast_weights(self, packed: bool = True):
        """Broadcast weights to the inference engine."""
        trainer_args = NCCLTrainerSendWeightsArgs(
            group=self.model_update_group,
            packed=packed,
        )
        NCCLWeightTransferEngine.trainer_send_weights(
            iterator=self.model.named_parameters(),
            trainer_args=trainer_args,
        )

    @torch.inference_mode()
    def generate(self, token_ids: list[int], max_new_tokens: int) -> list[int]:
        """Greedy-decode max_new_tokens from the given context."""
        input_ids = torch.tensor([token_ids], device="cuda:0")
        output = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        new_token_ids = output[0, len(token_ids) :].tolist()
        return new_token_ids


# Build platform-specific env vars for Ray
ray_env_vars = {
    # Prevent Ray from setting CUDA_VISIBLE_DEVICES
    "RAY_EXPERIMENTAL_NOSET_CUDA_ENV_VAR": "1",
}

if current_platform.is_rocm():
    # For ROCm, BATCH_INVARIANT vllm is not supported
    ray_env_vars["VLLM_ROCM_USE_SKINNY_GEMM"] = "0"
else:
    # Enable batch invariance for deterministic outputs on NVIDIA
    ray_env_vars["VLLM_BATCH_INVARIANT"] = "1"

ray.init(runtime_env={"env_vars": ray_env_vars})

# Launch the training model actor. Ray's resource scheduler will allocate
# 1 GPU (via num_gpus=1 in the decorator), ensuring pg_inference gets different GPUs.
train_model = TrainModel.remote(MODEL_NAME_V2)


rocm_determinism_kwargs = {}
if current_platform.is_rocm():
    # ROCm: To minimize non-determinism, we set fixed seed, no prefix caching, and
    # sequential request processing (max_num_seqs=1).
    rocm_determinism_kwargs = {
        "seed": 0,
        "enable_prefix_caching": False,
        "max_num_seqs": 1,
    }

# Build platform-specific LLM kwargs
llm_kwargs = dict(
    model=MODEL_NAME_V1,
    enforce_eager=True,
    max_model_len=8192,
    distributed_executor_backend="ray",
    attention_backend=ATTN_BACKEND,
    gpu_memory_utilization=0.3,
    weight_transfer_config=WeightTransferConfig(backend="nccl"),
    worker_extension_cls=WORKER_EXTENSION_CLS,
)
llm_kwargs.update(rocm_determinism_kwargs)

# Launch the vLLM inference engine.
# With data_parallel_backend="ray", vLLM's CoreEngineActorManager creates
# its own placement groups internally for each DP rank, so we must NOT
# create an outer placement group (it would reserve GPUs and hide them
# from the internal DP resource check).
llm = ray.remote(
    num_cpus=0,
    num_gpus=0,
)(MyLLM).remote(**llm_kwargs)

PROMPTS = [
    "The president of the United States is",
    "The capital of France is",
    "The largest ocean on Earth is",
    "The speed of light in a vacuum is",
    "The chemical formula for water is",
    "The tallest mountain in the world is",
    "The first person to walk on the moon was",
    "The Great Wall of China was built to",
    "Photosynthesis is the process by which",
    "The theory of general relativity was proposed by",
    "The boiling point of water at sea level is",
    "The largest planet in our solar system is",
    "DNA stands for deoxyribonucleic acid and it",
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_V1)
batch_prompt_token_ids = [
    tokenizer.encode(prompt, add_special_tokens=False) for prompt in PROMPTS
]


# Set up the communication channel between the training process and the
# inference engine.
master_address, master_port = ray.get(train_model.get_master_address_and_port.remote())

world_size = 2  # 1 trainer + 1 inference worker
inference_handle = llm.init_weight_transfer_engine.remote(
    WeightTransferInitRequest(
        init_info=asdict(
            NCCLWeightTransferInitInfo(
                master_address=master_address,
                master_port=master_port,
                rank_offset=1,
                world_size=world_size,
            )
        )
    )
)

# Initialize weight transfer group on both the training actor and inference engine
train_handle = train_model.init_weight_transfer_group.remote(world_size)
ray.get([train_handle, inference_handle])


N_NEW_TOKENS = 100
NUM_WEIGHT_TRANSFERS = 20
WEIGHT_TRANSFER_INTERVAL_SECONDS = 10.0
RUN_TRAINER_GENERATE_PROBE = True
TRAINER_GENERATE_EVERY_N_TRANSFERS = 1
TRAINER_GENERATE_PROMPT_INDEX = 0
TRAINER_GENERATE_NEW_TOKENS = 256

# Collect weight metadata once
names, dtype_names, shapes = ray.get(train_model.get_weight_metadata.remote())
transfer_bytes = sum(
    math.prod(shape) * torch.empty((), dtype=getattr(torch, dtype_name)).element_size()
    for dtype_name, shape in zip(dtype_names, shapes)
)
transfer_gb = transfer_bytes / 1e9
transfer_gib = transfer_bytes / 1024**3
print(
    f"Weight transfer payload: {transfer_gb:.3f} GB "
    f"({transfer_gib:.3f} GiB) across {len(names)} tensors"
)
transfer_timings = []

# ── Phase 1: concurrent requests with weight sync ───────────────────
print(f"\n{'=' * 50}")
print(f"Prompts ({len(PROMPTS)}):")
for p in PROMPTS:
    print(f"  - {p!r}")
print(f"{'=' * 50}")

sampling_params = SamplingParams(
    temperature=0, max_tokens=PAUSE_TOKEN_THRESHOLD + N_NEW_TOKENS
)

gen_futures = [
    llm.do_generate.remote(ptids, sampling_params) for ptids in batch_prompt_token_ids
]

ray.get(llm.pause_after_n_tokens.remote())
ray.get(llm.report_cuda_memory.remote("after pause"))

for transfer_idx in range(NUM_WEIGHT_TRANSFERS):
    print(
        f"\nWeight transfer {transfer_idx + 1}/{NUM_WEIGHT_TRANSFERS} "
        f"(NCCL, packed=True)"
    )
    transfer_start = time.perf_counter()

    ray.get(llm.report_cuda_memory.remote(f"transfer {transfer_idx + 1} before"))
    start_begin = time.perf_counter()
    ray.get(llm.start_weight_update.remote(is_checkpoint_format=True))
    start_seconds = time.perf_counter() - start_begin
    ray.get(llm.report_cuda_memory.remote(f"transfer {transfer_idx + 1} after start"))

    update_begin = time.perf_counter()
    inference_handle = llm.update_weights.remote(
        WeightTransferUpdateRequest(
            update_info=asdict(
                NCCLWeightTransferUpdateInfo(
                    names=names,
                    dtype_names=dtype_names,
                    shapes=shapes,
                    packed=True,
                )
            )
        )
    )
    train_handle = train_model.broadcast_weights.remote(packed=True)
    
    ray.get([train_handle, inference_handle])
    
    update_seconds = time.perf_counter() - update_begin
    ray.get(llm.report_cuda_memory.remote(f"transfer {transfer_idx + 1} after update"))

    finish_begin = time.perf_counter()
    ray.get(llm.finish_weight_update.remote())
    finish_seconds = time.perf_counter() - finish_begin
    ray.get(llm.report_cuda_memory.remote(f"transfer {transfer_idx + 1} after finish"))

    total_with_diagnostics_seconds = time.perf_counter() - transfer_start
    core_total_seconds = start_seconds + update_seconds + finish_seconds
    effective_gbps = transfer_gb / update_seconds
    effective_gibps = transfer_gib / update_seconds
    transfer_timings.append(
        {
            "transfer": transfer_idx + 1,
            "start_seconds": start_seconds,
            "update_seconds": update_seconds,
            "finish_seconds": finish_seconds,
            "core_total_seconds": core_total_seconds,
            "total_with_diagnostics_seconds": total_with_diagnostics_seconds,
            "effective_gbps": effective_gbps,
            "effective_gibps": effective_gibps,
        }
    )
    print(
        f"  Timing {transfer_idx + 1}/{NUM_WEIGHT_TRANSFERS}: "
        f"start={start_seconds:.3f}s, "
        f"update+broadcast={update_seconds:.3f}s, "
        f"effective={effective_gbps:.2f} GB/s "
        f"({effective_gibps:.2f} GiB/s), "
        f"finish={finish_seconds:.3f}s, "
        f"core_total={core_total_seconds:.3f}s, "
        f"with_diagnostics={total_with_diagnostics_seconds:.3f}s",
        flush=True,
    )

    if transfer_idx + 1 < NUM_WEIGHT_TRANSFERS:
        interval_start = time.perf_counter()
        if (
            RUN_TRAINER_GENERATE_PROBE
            and (transfer_idx + 1) % TRAINER_GENERATE_EVERY_N_TRANSFERS == 0
        ):
            trainer_generate_begin = time.perf_counter()
            trainer_generated_token_ids = ray.get(
                train_model.generate.remote(
                    batch_prompt_token_ids[TRAINER_GENERATE_PROMPT_INDEX],
                    TRAINER_GENERATE_NEW_TOKENS,
                )
            )
            trainer_generate_seconds = time.perf_counter() - trainer_generate_begin
            trainer_toks_per_s = (
                len(trainer_generated_token_ids) / trainer_generate_seconds
            )
            trainer_preview = tokenizer.decode(trainer_generated_token_ids[:32])
            print(
                f"  Trainer generate probe: "
                f"tokens={len(trainer_generated_token_ids)}, "
                f"time={trainer_generate_seconds:.3f}s, "
                f"throughput={trainer_toks_per_s:.1f} tok/s, "
                f"prompt={PROMPTS[TRAINER_GENERATE_PROMPT_INDEX]!r}, "
                f"preview={trainer_preview!r}",
                flush=True,
            )
        remaining_sleep = WEIGHT_TRANSFER_INTERVAL_SECONDS - (
            time.perf_counter() - interval_start
        )
        if remaining_sleep > 0:
            time.sleep(remaining_sleep)
        ray.get(llm.report_cuda_memory.remote(f"transfer {transfer_idx + 1} after sleep"))

if transfer_timings:
    update_times = [t["update_seconds"] for t in transfer_timings]
    core_total_times = [t["core_total_seconds"] for t in transfer_timings]
    diagnostic_total_times = [
        t["total_with_diagnostics_seconds"] for t in transfer_timings
    ]
    effective_gbps_values = [t["effective_gbps"] for t in transfer_timings]
    effective_gibps_values = [t["effective_gibps"] for t in transfer_timings]
    print(f"\n{'=' * 50}")
    print("Weight transfer timing summary")
    print(f"  payload:          {transfer_gb:.3f} GB ({transfer_gib:.3f} GiB)")
    print(
        f"  update+broadcast: "
        f"avg={sum(update_times) / len(update_times):.3f}s, "
        f"min={min(update_times):.3f}s, max={max(update_times):.3f}s"
    )
    print(
        f"  effective rate:   "
        f"avg={sum(effective_gbps_values) / len(effective_gbps_values):.2f} GB/s, "
        f"min={min(effective_gbps_values):.2f} GB/s, "
        f"max={max(effective_gbps_values):.2f} GB/s"
    )
    print(
        f"                    "
        f"avg={sum(effective_gibps_values) / len(effective_gibps_values):.2f} GiB/s, "
        f"min={min(effective_gibps_values):.2f} GiB/s, "
        f"max={max(effective_gibps_values):.2f} GiB/s"
    )
    print(
        f"  core total:       "
        f"avg={sum(core_total_times) / len(core_total_times):.3f}s, "
        f"min={min(core_total_times):.3f}s, max={max(core_total_times):.3f}s"
    )
    print(
        f"  with diagnostics: "
        f"avg={sum(diagnostic_total_times) / len(diagnostic_total_times):.3f}s, "
        f"min={min(diagnostic_total_times):.3f}s, "
        f"max={max(diagnostic_total_times):.3f}s"
    )
    print(f"{'=' * 50}")

ray.get(llm.resume_generation.remote())
results = ray.get(gen_futures)

for i, (output, pause_idx) in enumerate(results):
    all_token_ids = list(output.outputs[0].token_ids)
    before_text = tokenizer.decode(all_token_ids[:pause_idx])
    after_text = tokenizer.decode(all_token_ids[pause_idx:])
    print(f"\n  Request {i} ({PROMPTS[i]!r}):")
    print(f"    Old weights ({pause_idx} tokens): {before_text!r}")
    n_after = len(all_token_ids) - pause_idx
    print(f"    New weights ({n_after} tokens): {after_text!r}")

# ── Phase 2: validate with a fresh V2 vLLM instance ────────────────
# The resumed requests above intentionally keep their pre-update KV cache.
# Their continuation is therefore a mixed state: V1-computed KV cache plus V2
# weights. A fresh V2 instance that recomputes the whole prompt would not be
# expected to match that continuation exactly. To validate the NCCL weight
# sync itself, compare new requests served by the already-updated engine
# against a fresh V2 engine.
ray.get(llm.reset_prefix_cache.remote(reset_running_requests=False))

validation_sampling_params = SamplingParams(temperature=0, max_tokens=N_NEW_TOKENS)
updated_vllm_futures = [
    llm.do_generate_no_pause.remote(ptids, validation_sampling_params)
    for ptids in batch_prompt_token_ids
]
updated_vllm_results = ray.get(updated_vllm_futures)

# This validation relies on batch-invariant (deterministic) generation to
# compare outputs from the weight-synced engine against a fresh V2 instance.
# On NVIDIA, batch invariance is fully supported, so we require 100% exact
# token match. On ROCm, batch invariance is not yet fully implemented
# (see https://github.com/vllm-project/vllm/issues/27433 and
# https://github.com/vllm-project/vllm/issues/33123), so residual
# non-determinism (e.g. GEMM accumulation order, missing kernel overrides)
# can cause single-token divergences that don't indicate a weight-sync
# failure. We relax the pass rate to 90% on ROCm to accommodate this; a
# real regression (broken weight transfer) would cause ~0% pass rate, not 90%+.
MIN_PASS_RATE = 1.0 if not current_platform.is_rocm() else 0.9

print(f"\n{'=' * 50}")
print("VALIDATION: comparing updated vLLM new requests with fresh V2 instance")
if current_platform.is_rocm():
    print(f"  (ROCm mode: requiring >= {MIN_PASS_RATE:.0%} exact match rate)")
print(f"{'=' * 50}")

ray.get(llm.shutdown.remote())
ray.kill(llm)
ray.kill(train_model)

llm_v2_kwargs = dict(
    model=MODEL_NAME_V2,
    enforce_eager=True,
    max_model_len=8192,
    gpu_memory_utilization=0.3,
    distributed_executor_backend="ray",
    attention_backend=ATTN_BACKEND,
    worker_extension_cls=WORKER_EXTENSION_CLS,
)
llm_v2_kwargs.update(rocm_determinism_kwargs)

llm_v2 = ray.remote(
    num_cpus=0,
    num_gpus=0,
)(MyLLM).remote(**llm_v2_kwargs)

val_futures = [
    llm_v2.do_generate_no_pause.remote(ptids, validation_sampling_params)
    for ptids in batch_prompt_token_ids
]
val_results = ray.get(val_futures)

num_pass = 0
num_total = len(updated_vllm_results)
for i, (output, val_output) in enumerate(zip(updated_vllm_results, val_results)):
    expected = list(output.outputs[0].token_ids)
    actual = list(val_output.outputs[0].token_ids)
    match = actual == expected

    if match:
        num_pass += 1
        print(f"  [PASS] {PROMPTS[i]!r}")
    else:
        print(f"  [FAIL] {PROMPTS[i]!r}")
        print(f"         updated vLLM: {tokenizer.decode(expected)!r}")
        print(f"         V2 vLLM:           {tokenizer.decode(actual)!r}")
        for j, (e, a) in enumerate(zip(expected, actual)):
            if e != a:
                print(
                    f"         first divergence at output token {j}: "
                    f"expected {e} ({tokenizer.decode([e])!r}) vs "
                    f"actual {a} ({tokenizer.decode([a])!r})"
                )
                break

ray.get(llm_v2.shutdown.remote())
ray.kill(llm_v2)

pass_rate = num_pass / num_total
print(f"\n  Result: {num_pass}/{num_total} prompts passed ({pass_rate:.0%})")
print(f"  Required: >= {MIN_PASS_RATE:.0%}")

assert pass_rate >= MIN_PASS_RATE, (
    f"Validation pass rate {pass_rate:.0%} ({num_pass}/{num_total}) "
    f"is below the required {MIN_PASS_RATE:.0%} threshold. "
    f"See failures above for details."
)
print("=" * 50)
