# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FSDP2 training smoke pipeline with optional vLLM weight-transfer demo.

8-GPU layout:
  Training  - 4 GPUs, PyTorch FSDP2 (fully_shard)
  Optional inference demo - 4 GPUs, vLLM AsyncLLMEngine with EP+DP

By default this script launches Ray actors that form a NCCL process group and
run a dummy response-only SFT loop, matching the local_trainer.py FSDP path.
The existing FSDP -> vLLM NCCL weight-transfer helpers are kept for follow-up
integration but are not called by the default training entrypoint.

Assumes a single-node cluster with 8 GPUs.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import socket
import time
import uuid
from dataclasses import asdict
from typing import Dict, Iterable, List, Tuple

import ray
import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
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
from vllm.v1.executor import Executor

MODEL_NAME = "/mnt/data/lcx4/hf_cache/Qwen1.5-MoE-A2.7B-Chat"

FSDP_WORLD_SIZE = 4
INFERENCE_TP_SIZE = 1
INFERENCE_DP_SIZE = 4
TRANSFER_BENCH_ITERS = 20

def get_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def find_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


DUMMY_CHAT_EXAMPLES = [
    {
        "user": "什么是 MoE 模型？",
        "assistant": "MoE 模型会把 token 路由到不同专家网络中处理，从而在控制计算量的同时扩大参数规模。",
    },
    {
        "user": "用一句话解释强化学习。",
        "assistant": "强化学习是让智能体通过奖励信号学习如何在环境中做决策的方法。",
    },
    {
        "user": "给我一个 Python 列表推导式例子。",
        "assistant": "例如 squares = [x * x for x in range(5)]，它会得到 0 到 4 的平方。",
    },
    {
        "user": "What is the capital of France?",
        "assistant": "The capital of France is Paris.",
    },
    {
        "user": "Summarize local model training in one sentence.",
        "assistant": "Local model training loads weights, prepares batches, runs forward and backward passes, updates parameters, and saves a checkpoint.",
    },
    {
        "user": "解释一下 attention mask 的作用。",
        "assistant": "attention mask 用来告诉模型哪些 token 是有效输入，哪些 token 是 padding，应当被忽略。",
    },
    {
        "user": "什么是 checkpoint?",
        "assistant": "checkpoint 是训练过程中保存下来的模型权重和 tokenizer 文件，可用于恢复训练或推理。",
    },
    {
        "user": "Give a tiny JSON example.",
        "assistant": '{"name": "vllm-fsdp-trainer", "status": "ok"}',
    },
]


class EncodedExample:
    def __init__(
        self,
        input_ids: List[int],
        attention_mask: List[int],
        labels: List[int],
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels


class DummyChatDataset:
    """Small response-only SFT dataset built entirely in memory."""

    def __init__(self, tokenizer, max_length: int, repeat: int):
        self.examples = []
        for _ in range(repeat):
            for item in DUMMY_CHAT_EXAMPLES:
                self.examples.append(
                    encode_chat_example(
                        tokenizer=tokenizer,
                        user=item["user"],
                        assistant=item["assistant"],
                        max_length=max_length,
                    )
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> EncodedExample:
        return self.examples[index]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_dtype(dtype_name: str):
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def build_prompt(tokenizer, user: str) -> str:
    messages = [{"role": "user", "content": user}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"User: {user}\nAssistant: "


def build_full_text(tokenizer, user: str, assistant: str) -> str:
    messages = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    return f"User: {user}\nAssistant: {assistant}"


def encode_chat_example(
    tokenizer,
    user: str,
    assistant: str,
    max_length: int,
) -> EncodedExample:
    prompt_text = build_prompt(tokenizer, user)
    full_text = build_full_text(tokenizer, user, assistant)

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    labels = list(input_ids)

    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len
    if all(label == -100 for label in labels) and labels:
        labels[-1] = input_ids[-1]

    return EncodedExample(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )


def make_collate_fn(tokenizer):
    pad_token_id = tokenizer.pad_token_id

    def collate(examples: List[EncodedExample]) -> Dict[str, torch.Tensor]:
        max_len = max(len(example.input_ids) for example in examples)
        input_ids = []
        attention_mask = []
        labels = []

        for example in examples:
            pad_len = max_len - len(example.input_ids)
            input_ids.append(example.input_ids + [pad_token_id] * pad_len)
            attention_mask.append(example.attention_mask + [0] * pad_len)
            labels.append(example.labels + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def configure_trainable_parameters(model, train_mode: str) -> None:
    if train_mode == "full":
        for param in model.parameters():
            param.requires_grad = True
        return

    for param in model.parameters():
        param.requires_grad = False

    if train_mode == "lm_head":
        target_keywords = ("lm_head",)
    elif train_mode == "last_layer":
        num_layers = len(getattr(model.model, "layers"))
        target_keywords = (f"model.layers.{num_layers - 1}.", "lm_head")
    else:
        raise ValueError(f"Unsupported train mode: {train_mode}")

    for name, param in model.named_parameters():
        if any(keyword in name for keyword in target_keywords):
            param.requires_grad = True


def iter_trainable_parameters(model) -> Iterable:
    return (param for param in model.parameters() if param.requires_grad)


def count_parameters(model) -> Tuple[int, int]:
    total = 0
    trainable = 0
    for param in model.parameters():
        numel = param.numel()
        total += numel
        if param.requires_grad:
            trainable += numel
    return total, trainable


def log_parameter_count(model, train_mode: str, rank: int = 0):
    total_params, trainable_params = count_parameters(model)
    trainable_parameter_list = list(iter_trainable_parameters(model))
    if not trainable_parameter_list:
        raise RuntimeError(f"No trainable parameters found for mode: {train_mode}")

    if rank == 0:
        print(
            "[train] Parameter count: "
            f"trainable={trainable_params:,} / total={total_params:,} "
            f"({trainable_params / total_params:.4%})"
        )
    return trainable_parameter_list


def move_batch_to_device(batch: Dict, device) -> Dict:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def build_tokenizer(args: argparse.Namespace, log: bool = True):
    if log:
        print(f"[init] Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define either pad_token or eos_token.")
    return tokenizer


def build_model(args: argparse.Namespace, device, torch_dtype, log: bool = True):
    if log:
        print(
            f"[init] Loading model from {args.model_path} "
            f"(device={device}, dtype={torch_dtype})"
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        local_files_only=True,
        trust_remote_code=args.trust_remote_code,
    )
    model.to(device)
    model.train()
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model


def build_dataset(args: argparse.Namespace, tokenizer) -> DummyChatDataset:
    return DummyChatDataset(
        tokenizer=tokenizer,
        max_length=args.max_length,
        repeat=args.dataset_repeat,
    )


def build_dataloader(
    args: argparse.Namespace,
    tokenizer,
    dataset,
    sampler=None,
    shuffle: bool = True,
):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        collate_fn=make_collate_fn(tokenizer),
    )


def run_training_loop(
    model,
    dataloader,
    optimizer,
    trainable_parameter_list,
    device,
    args: argparse.Namespace,
    rank: int = 0,
    sampler=None,
) -> Dict[str, float]:
    if rank == 0:
        print(
            f"[train] Starting training: max_steps={args.max_steps}, "
            f"batch_size={args.batch_size}, grad_accum_steps={args.grad_accum_steps}"
        )

    step = 0
    epoch = 0
    last_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    while step < args.max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)

        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum_steps
            loss.backward()

            should_step = (step + 1) % args.grad_accum_steps == 0
            if should_step:
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameter_list,
                    max_norm=1.0,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            step += 1
            last_loss = loss.item() * args.grad_accum_steps
            if rank == 0 and step % args.log_every == 0:
                print(f"[train] step={step} loss={last_loss:.6f}")

            if step >= args.max_steps:
                break

        epoch += 1

    return {"rank": rank, "steps": step, "last_loss": last_loss}


def iter_vllm_loadable_weights(name: str, tensor: torch.Tensor):
    """Yield checkpoint-style weights accepted by vLLM's Qwen2-MoE loader.

    Recent Transformers stores routed expert weights as fused 3D parameters:
    ``experts.gate_up_proj`` and ``experts.down_proj``. vLLM's checkpoint
    loader expects the original per-expert HF names and performs its own
    loading into FusedMoE kernel parameters, so we split them before transfer.
    """
    if name.endswith(".mlp.experts.gate_up_proj"):
        prefix = name.removesuffix(".gate_up_proj")
        gate_proj, up_proj = tensor.chunk(2, dim=1)
        for expert_idx in range(tensor.shape[0]):
            yield f"{prefix}.{expert_idx}.gate_proj.weight", gate_proj[expert_idx]
            yield f"{prefix}.{expert_idx}.up_proj.weight", up_proj[expert_idx]
    elif name.endswith(".mlp.experts.down_proj"):
        prefix = name.removesuffix(".down_proj")
        for expert_idx in range(tensor.shape[0]):
            yield f"{prefix}.{expert_idx}.down_proj.weight", tensor[expert_idx]
    else:
        yield name, tensor


def get_vllm_weight_metadata(named_parameters):
    """Return names, dtypes, and shapes matching iter_vllm_loadable_weights."""
    names = []
    dtype_names = []
    shapes = []
    for name, param in named_parameters:
        for load_name, load_tensor in iter_vllm_loadable_weights(name, param):
            names.append(load_name)
            dtype_names.append(str(load_tensor.dtype).split(".")[-1])
            shapes.append(list(load_tensor.shape))
    return names, dtype_names, shapes


def dtype_nbytes(dtype_name: str) -> int:
    """Return bytes per element for dtype names emitted by get_vllm_weight_metadata."""
    return {
        "float64": 8,
        "double": 8,
        "float32": 4,
        "float": 4,
        "bfloat16": 2,
        "float16": 2,
        "half": 2,
        "int64": 8,
        "long": 8,
        "int32": 4,
        "int": 4,
        "int16": 2,
        "short": 2,
        "int8": 1,
        "uint8": 1,
        "bool": 1,
    }[dtype_name]


def numel_from_shape(shape):
    numel = 1
    for dim in shape:
        numel *= dim
    return numel


class FSDPTrainWorker:
    """
    One FSDP2 training worker per GPU.  Four of these form the FSDP group.
    Rank 0 additionally handles weight transfer to the vLLM engine.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        rank: int,
        fsdp_world_size: int,
        fsdp_master_addr: str,
        fsdp_master_port: int,
    ):
        self.args = args
        self.rank = rank
        self.fsdp_world_size = fsdp_world_size

        os.environ["MASTER_ADDR"] = fsdp_master_addr
        os.environ["MASTER_PORT"] = str(fsdp_master_port)

        dist.init_process_group(backend="nccl", rank=rank, world_size=fsdp_world_size)
        if hasattr(torch, "accelerator"):
            torch.accelerator.set_device_index(0)
        else:
            torch.cuda.set_device(0)
        self.device = torch.device("cuda:0")

        set_seed(args.seed + rank)

        self.tokenizer = build_tokenizer(args, log=rank == 0)
        torch_dtype = pick_dtype(args.dtype)
        model = build_model(args, self.device, torch_dtype, log=rank == 0)
        configure_trainable_parameters(model, args.train_mode)
        log_parameter_count(model, args.train_mode, rank=rank)

        named_parameters = list(model.named_parameters())
        self.train_param_names = [n for n, _ in named_parameters]
        (
            self.weight_names,
            self.weight_dtype_names,
            self.weight_shapes,
        ) = get_vllm_weight_metadata(named_parameters)

        for layer in model.model.layers:
            fully_shard(layer)
        fully_shard(model)

        self.model = model
        self.trainable_parameter_list = list(iter_trainable_parameters(self.model))
        if not self.trainable_parameter_list:
            raise RuntimeError(f"No trainable parameters found for mode: {args.train_mode}")

        dataset = build_dataset(args, self.tokenizer)
        self.sampler = DistributedSampler(
            dataset,
            num_replicas=fsdp_world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=False,
        )
        self.dataloader = build_dataloader(
            args,
            self.tokenizer,
            dataset,
            sampler=self.sampler,
            shuffle=False,
        )
        self.optimizer = torch.optim.AdamW(
            self.trainable_parameter_list,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        self.transfer_port = None
        self.transfer_master_address = None
        self.model_update_group = None
        print(f"[rank {rank}] FSDP worker ready.")

    def get_rank(self):
        return self.rank

    def train(self) -> Dict[str, float]:
        try:
            summary = run_training_loop(
                model=self.model,
                dataloader=self.dataloader,
                optimizer=self.optimizer,
                trainable_parameter_list=self.trainable_parameter_list,
                device=self.device,
                args=self.args,
                rank=self.rank,
                sampler=self.sampler,
            )
            dist.barrier()
            if self.rank == 0:
                print("[done] Ray FSDP trainer smoke test finished.")
            return summary
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()

    # ---- weight-transfer setup (rank 0 only) ----

    def setup_transfer_endpoint(self):
        """Create the NCCL rendezvous endpoint for weight transfer."""
        assert self.rank == 0
        self.transfer_port = find_open_port()
        self.transfer_master_address = get_local_ip()
        return self.transfer_master_address, self.transfer_port

    def init_weight_transfer_group(self, transfer_world_size: int):
        """Join the weight-transfer NCCL group as rank 0 (the source)."""
        assert self.rank == 0
        self.model_update_group = NCCLWeightTransferEngine.trainer_init(
            dict(
                master_address=self.transfer_master_address,
                master_port=self.transfer_port,
                world_size=transfer_world_size,
            ),
        )

    def get_weight_metadata(self):
        """Return weight names, dtypes, and shapes captured before FSDP wrapping."""
        return self.weight_names, self.weight_dtype_names, self.weight_shapes

    # ---- collective ops (ALL FSDP ranks must call concurrently) ----

    def gather_and_broadcast_weights(self, packed: bool = True):
        """
        All-gather full parameters and broadcast them to vLLM.
        Only rank 0 performs the actual NCCL broadcast; others just
        participate in the FSDP all-gather.

        full_tensor() is a collective — all FSDP ranks must call it
        for each parameter in the same order.  Rank 0 additionally
        feeds each gathered tensor to the weight-transfer engine.
        """
        if self.rank == 0:
            def _full_param_iter():
                params_by_name = dict(self.model.named_parameters())
                for name in self.train_param_names:
                    full_param = params_by_name[name].full_tensor().detach()
                    yield from iter_vllm_loadable_weights(name, full_param)

            trainer_args = NCCLTrainerSendWeightsArgs(
                group=self.model_update_group,
                packed=packed,
            )
            NCCLWeightTransferEngine.trainer_send_weights(
                iterator=_full_param_iter(),
                trainer_args=trainer_args,
            )
        else:
            params_by_name = dict(self.model.named_parameters())
            for name in self.train_param_names:
                params_by_name[name].full_tensor()


def create_async_engine(**kwargs):
    """Create an AsyncLLMEngine directly (no subclass needed)."""
    engine_args = vllm.AsyncEngineArgs(**kwargs)
    vllm_config = engine_args.create_engine_config()
    executor_class = Executor.get_class(vllm_config)
    return vllm.AsyncLLMEngine(
        vllm_config=vllm_config,
        executor_class=executor_class,
        log_requests=engine_args.enable_log_requests,
        log_stats=not engine_args.disable_log_stats,
    )


async def generate_batch(engine, prompts, sampling_params):
    """Generate completions for a batch of prompts."""

    async def gen_one(prompt):
        output = None
        async for request_output in engine.generate(
            {"prompt": prompt},
            sampling_params,
            request_id=str(uuid.uuid4()),
        ):
            output = request_output
        return output

    return await asyncio.gather(*[gen_one(p) for p in prompts])


async def run_weight_sync_demo(args: argparse.Namespace):
    if args.ray_address:
        ray.init(address=args.ray_address)
    else:
        ray.init()

    # Use local/shared model weights directly.
    local_model_path = args.model_path
    print(f"[init] Loading local model from {local_model_path}")

    # FSDP rendezvous address (single-node)
    fsdp_master_addr = args.fsdp_master_addr or get_local_ip()
    fsdp_master_port = args.fsdp_master_port or find_open_port()

    # Launch 4 FSDP training workers.
    # Ray allocates 1 GPU per worker; AsyncLLMEngine's internal DP
    # placement groups will land on the remaining 4 GPUs.
    remote_worker = ray.remote(num_gpus=1)(FSDPTrainWorker)
    fsdp_workers = [
        remote_worker.remote(
            args,
            rank,
            args.fsdp_world_size,
            fsdp_master_addr,
            fsdp_master_port,
        )
        for rank in range(args.fsdp_world_size)
    ]
    ray.get([w.get_rank.remote() for w in fsdp_workers])
    print(f"[init] {args.fsdp_world_size} FSDP training workers ready.")

    # Launch vLLM with expert parallelism + data parallelism.
    # AsyncLLMEngine with data_parallel_backend="ray" creates its own
    # placement groups internally — no manual placement group needed.
    print("[engine] Creating AsyncLLMEngine...")
    engine = create_async_engine(
        model=local_model_path,
        enforce_eager=True,
        tensor_parallel_size=INFERENCE_TP_SIZE,
        data_parallel_size=INFERENCE_DP_SIZE,
        enable_expert_parallel=True,
        distributed_executor_backend="ray",
        data_parallel_backend="ray",
        weight_transfer_config=WeightTransferConfig(backend="nccl"),
        load_format="dummy",
        gpu_memory_utilization=0.3,
    )
    print("[engine] AsyncLLMEngine created.")

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0)

    # Generate with dummy weights — expect gibberish.
    print("[generate] Starting generation with dummy weights...")
    outputs = await generate_batch(engine, prompts, sampling_params)
    print("[generate] Generation complete.")

    print("-" * 60)
    print("BEFORE weight sync (dummy weights):")
    print("-" * 60)
    for output in outputs:
        print(f"Prompt: {output.prompt!r}")
        print(f"Generated: {output.outputs[0].text!r}")
        print("-" * 60)

    # --- Weight-transfer setup ---
    print("[transfer] Setting up weight-transfer endpoint...")
    transfer_addr, transfer_port = ray.get(
        fsdp_workers[0].setup_transfer_endpoint.remote()
    )
    print(f"[transfer] Endpoint ready at {transfer_addr}:{transfer_port}")

    transfer_world_size = INFERENCE_TP_SIZE * INFERENCE_DP_SIZE + 1
    print(
        f"[transfer] World size: {transfer_world_size} "
        f"(1 trainer + {INFERENCE_TP_SIZE * INFERENCE_DP_SIZE} vLLM workers)"
    )

    print("[transfer] Initializing NCCL groups...")
    train_handle = fsdp_workers[0].init_weight_transfer_group.remote(
        transfer_world_size
    )
    await engine.init_weight_transfer_engine(
        WeightTransferInitRequest(
            init_info=asdict(
                NCCLWeightTransferInitInfo(
                    master_address=transfer_addr,
                    master_port=transfer_port,
                    rank_offset=1,
                    world_size=transfer_world_size,
                )
            )
        )
    )
    ray.get(train_handle)
    print("[transfer] NCCL groups initialized.")

    # --- Pause, transfer weights, resume ---
    print("[sync] Pausing generation...")
    await engine.pause_generation(mode="abort")
    print("[sync] Generation paused.")

    names, dtype_names, shapes = ray.get(fsdp_workers[0].get_weight_metadata.remote())
    print(f"[sync] Got metadata for {len(names)} parameters.")

    total_weight_bytes = sum(
        numel_from_shape(shape) * dtype_nbytes(dtype_name)
        for dtype_name, shape in zip(dtype_names, shapes)
    )
    model_gib = total_weight_bytes / 1024**3
    infer_payload_gib = model_gib * (transfer_world_size - 1)
    print(
        f"[bench] One logical model payload: {model_gib:.3f} GiB; "
        f"aggregate infer payload: {infer_payload_gib:.3f} GiB "
        f"across {transfer_world_size - 1} infer ranks."
    )

    bench_times = []
    for i in range(TRANSFER_BENCH_ITERS):
        await engine.start_weight_update()
        print(
            f"[bench] Iteration {i + 1}/{TRANSFER_BENCH_ITERS}: "
            "broadcasting weights from FSDP → vLLM..."
        )
        t0 = time.perf_counter()
        broadcast_handles = [
            w.gather_and_broadcast_weights.remote(packed=True) for w in fsdp_workers
        ]
        await engine.update_weights(
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
        ray.get(broadcast_handles)
        await engine.finish_weight_update()
        elapsed = time.perf_counter() - t0
        bench_times.append(elapsed)
        print(
            f"[bench] Iteration {i + 1}/{TRANSFER_BENCH_ITERS}: "
            f"{elapsed:.3f}s, model-sync throughput={model_gib / elapsed:.3f} GiB/s, "
            f"aggregate-infer throughput={infer_payload_gib / elapsed:.3f} GiB/s"
        )

    avg_time = sum(bench_times) / len(bench_times)
    print("[sync] Weight broadcast benchmark complete.")
    print(
        f"[bench] Average over {TRANSFER_BENCH_ITERS} iterations: "
        f"{avg_time:.3f}s, model-sync throughput={model_gib / avg_time:.3f} GiB/s, "
        f"aggregate-infer throughput={infer_payload_gib / avg_time:.3f} GiB/s"
    )

    print("[sync] Resuming generation...")
    await engine.resume_generation()
    print("[sync] Generation resumed.")

    # Generate with synced weights — expect sensible output.
    print("[generate] Starting generation with synced weights...")
    outputs_updated = await generate_batch(engine, prompts, sampling_params)
    print("[generate] Generation complete.")

    print("-" * 60)
    print("AFTER weight sync (real weights):")
    print("-" * 60)
    for output in outputs_updated:
        print(f"Prompt: {output.prompt!r}")
        print(f"Generated: {output.outputs[0].text!r}")
        print("-" * 60)


def run_fsdp_training(args: argparse.Namespace) -> None:
    if args.save_checkpoint:
        print("[save] Ray FSDP mode does not support --save-checkpoint yet; skipping.")

    fsdp_master_addr = args.fsdp_master_addr or get_local_ip()
    fsdp_master_port = args.fsdp_master_port or find_open_port()

    if args.ray_address:
        ray.init(address=args.ray_address)
    else:
        ray.init()

    try:
        remote_worker = ray.remote(num_gpus=1)(FSDPTrainWorker)
        workers = [
            remote_worker.remote(
                args,
                rank,
                args.fsdp_world_size,
                fsdp_master_addr,
                fsdp_master_port,
            )
            for rank in range(args.fsdp_world_size)
        ]
        ray.get([worker.get_rank.remote() for worker in workers])
        print(f"[init] {args.fsdp_world_size} Ray FSDP training workers ready.")

        summaries = ray.get([worker.train.remote() for worker in workers])
        rank0_summary = next(item for item in summaries if item["rank"] == 0)
        print(
            "[done] rank0 summary: "
            f"steps={rank0_summary['steps']} "
            f"last_loss={rank0_summary['last_loss']:.6f}"
        )
    finally:
        ray.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Ray FSDP2 local training smoke test."
    )
    parser.add_argument("--model-path", default=MODEL_NAME)
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "bfloat16", "float16", "float32"),
    )
    parser.add_argument(
        "--train-mode",
        default="lm_head",
        choices=("lm_head", "last_layer", "full"),
        help="Default lm_head mode is intended to validate the training loop.",
    )
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--dataset-repeat", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--fsdp-world-size", type=int, default=FSDP_WORLD_SIZE)
    parser.add_argument("--fsdp-master-addr", default=None)
    parser.add_argument("--fsdp-master-port", type=int, default=None)
    parser.add_argument(
        "--ray-address",
        default=None,
        help="Optional Ray cluster address. Defaults to local ray.init().",
    )
    parser.add_argument(
        "--save-checkpoint",
        action="store_true",
        help="Accepted for CLI compatibility; FSDP checkpoint save is skipped.",
    )
    parser.add_argument(
        "--run-weight-sync-demo",
        action="store_true",
        help="Run the original FSDP-to-vLLM NCCL weight-transfer demo instead.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.dataset_repeat < 1:
        raise ValueError("--dataset-repeat must be >= 1")
    if args.max_length < 1:
        raise ValueError("--max-length must be >= 1")
    if args.log_every < 1:
        raise ValueError("--log-every must be >= 1")
    if args.fsdp_world_size < 1:
        raise ValueError("--fsdp-world-size must be >= 1")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.run_weight_sync_demo:
        asyncio.run(run_weight_sync_demo(args))
    else:
        run_fsdp_training(args)


if __name__ == "__main__":
    main()
