# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal FSDP trainer + vLLM interruptible inference sync demo.

8-GPU layout:
  Training  - 4 GPUs, PyTorch FSDP2 (fully_shard)
  Inference - 4 GPUs, vLLM AsyncLLMEngine with EP+DP

This script launches Ray FSDP trainer workers and a vLLM inference engine.
The trainer runs in short optimizer-step segments; at each sync boundary,
generation is paused/aborted, in-flight requests are drained, FSDP weights are
sent to vLLM over NCCL, and interruptible inference resumes with resubmitted
requests.

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
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Literal, Tuple

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


def validate_weight_scope(scope: str) -> None:
    if scope not in {"all", "trainable"}:
        raise ValueError(f"Unsupported weight scope: {scope!r}")


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
        self.all_param_names = [name for name, _ in named_parameters]
        self.trainable_param_names = [
            name for name, param in named_parameters if param.requires_grad
        ]
        self.param_names_by_scope = {
            "all": self.all_param_names,
            "trainable": self.trainable_param_names,
        }
        self.weight_metadata_by_scope = {
            "all": get_vllm_weight_metadata(named_parameters),
            "trainable": get_vllm_weight_metadata(
                [
                    (name, param)
                    for name, param in named_parameters
                    if param.requires_grad
                ]
            ),
        }

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
        self.optimizer.zero_grad(set_to_none=True)

        self.train_epoch = 0
        self.train_micro_step = 0
        self.optimizer_step = 0
        self.last_loss = 0.0
        self._dataloader_iter = None

        self.transfer_port = None
        self.transfer_master_address = None
        self.model_update_group = None
        print(f"[rank {rank}] FSDP worker ready.")

    def get_rank(self):
        return self.rank

    def close(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def _next_training_batch(self):
        while True:
            if self._dataloader_iter is None:
                self.sampler.set_epoch(self.train_epoch)
                self._dataloader_iter = iter(self.dataloader)
            try:
                return next(self._dataloader_iter)
            except StopIteration:
                self.train_epoch += 1
                self._dataloader_iter = None

    def train_until_next_sync(self, num_optimizer_steps: int = 100) -> Dict[str, float]:
        """
        Continue the persistent training loop until this worker finishes the
        requested number of optimizer steps, or reaches args.max_steps.

        args.max_steps is interpreted as optimizer steps.
        """
        if num_optimizer_steps < 1:
            raise ValueError("num_optimizer_steps must be >= 1")

        start_optimizer_step = self.optimizer_step
        target_optimizer_step = min(
            self.optimizer_step + num_optimizer_steps,
            self.args.max_steps,
        )

        while self.optimizer_step < target_optimizer_step:
            batch = move_batch_to_device(self._next_training_batch(), self.device)
            outputs = self.model(**batch)
            loss = outputs.loss / self.args.grad_accum_steps
            loss.backward()

            self.train_micro_step += 1
            self.last_loss = loss.item() * self.args.grad_accum_steps
            should_step = self.train_micro_step % self.args.grad_accum_steps == 0
            if not should_step:
                continue

            torch.nn.utils.clip_grad_norm_(
                self.trainable_parameter_list,
                max_norm=1.0,
            )
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.optimizer_step += 1

            if self.rank == 0 and self.optimizer_step % self.args.log_every == 0:
                print(
                    "[train] "
                    f"optimizer_step={self.optimizer_step} "
                    f"micro_step={self.train_micro_step} "
                    f"loss={self.last_loss:.6f}"
                )

        dist.barrier()
        optimizer_steps_run = self.optimizer_step - start_optimizer_step
        return {
            "rank": self.rank,
            "optimizer_steps_run": optimizer_steps_run,
            "optimizer_step": self.optimizer_step,
            "micro_step": self.train_micro_step,
            "reached_max_steps": self.optimizer_step >= self.args.max_steps,
            "last_loss": self.last_loss,
        }

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

    def get_weight_metadata(self, scope: str = "all"):
        """Return scoped weight names, dtypes, and shapes from pre-FSDP params."""
        validate_weight_scope(scope)
        return self.weight_metadata_by_scope[scope]

    # ---- collective ops (ALL FSDP ranks must call concurrently) ----

    def gather_and_broadcast_weights(self, scope: str = "all", packed: bool = True):
        """
        All-gather scoped full parameters and broadcast them to vLLM.
        Only rank 0 performs the actual NCCL broadcast; others just
        participate in the FSDP all-gather.

        full_tensor() is a collective — all FSDP ranks must call it
        for each parameter in the same order.  Rank 0 additionally
        feeds each gathered tensor to the weight-transfer engine.
        """
        validate_weight_scope(scope)
        param_names = self.param_names_by_scope[scope]
        if self.rank == 0:
            def _full_param_iter():
                params_by_name = dict(self.model.named_parameters())
                for name in param_names:
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
            for name in param_names:
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


@dataclass
class OnlineGenerationState:
    """Token-level state for one request across weight-update interruptions."""

    index: int
    prompt: str
    input_ids: List[int]
    requested_max_tokens: int
    output_tokens: List[int] = field(default_factory=list)
    output_versions: List[int] = field(default_factory=list)
    stop_reason: Literal["length", "stop", "tool_calls", "abort"] | None = None
    completed: bool = False
    attempts: int = 0

    @property
    def remaining_max_tokens(self) -> int:
        return max(0, self.requested_max_tokens - len(self.output_tokens))

    @property
    def restart_prompt_token_ids(self) -> List[int]:
        return self.input_ids + self.output_tokens


@dataclass
class RepeatingInferenceStats:
    total_requests: int = 0
    total_tokens: int = 0
    in_flight_concurrency: int = 0
    last_completed_states: List[OnlineGenerationState] = field(default_factory=list)


def _tokens_from_output(request_output) -> List[int]:
    if not getattr(request_output, "outputs", None):
        return []
    return list(getattr(request_output.outputs[0], "token_ids", []) or [])


def _finish_reason_from_output(request_output):
    if not getattr(request_output, "outputs", None):
        return None
    return getattr(request_output.outputs[0], "finish_reason", None)


def _normalize_stop_reason(stop_reason) -> Literal["length", "stop", "tool_calls", "abort"]:
    if stop_reason in ("length", "stop", "tool_calls", "abort"):
        return stop_reason
    if stop_reason in ("eos", "stop_token", "stop_sequence"):
        return "stop"
    return "abort"


class InterruptibleGenerationRunner:
    """Run vLLM requests that survive abort-based weight-update pauses."""

    def __init__(
        self,
        engine,
        temperature: float = 0.0,
        max_resubmit_retries: int = 200,
    ):
        self.engine = engine
        self.temperature = temperature
        self.max_resubmit_retries = max_resubmit_retries
        self.version = 0
        self.paused = asyncio.Event()
        self.paused.clear()
        self._active_attempts = 0
        self._active_changed = asyncio.Condition()

    # 新的engine.generate() attempt开始 +1
    async def _increment_active_attempts(self) -> None:
        async with self._active_changed:
            self._active_attempts += 1
            self._active_changed.notify_all()

    # 一个engine.generate() attempt 结束 -1
    async def _decrement_active_attempts(self) -> None:
        async with self._active_changed:
            self._active_attempts -= 1
            self._active_changed.notify_all()

    # 等待正在跑的 generate attempt 都结束,可能是正常结束，也可能是被abort打断
    async def wait_for_idle(self) -> None:
        async with self._active_changed:
            await self._active_changed.wait_for(lambda: self._active_attempts == 0)

    async def generate(self, state: OnlineGenerationState) -> OnlineGenerationState:
        for attempt in range(1, self.max_resubmit_retries + 1):
            # 如果当前正在weight update的attempt还没结束，就等着，不要开始新的generate attempt
            while self.paused.is_set():
                await asyncio.sleep(0)

            remaining = state.remaining_max_tokens
            if remaining <= 0:
                state.stop_reason = "length"
                state.completed = True
                return state

            state.attempts = attempt
            attempt_version = self.version
            sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=remaining,
            )
            request_id = (
                f"online-sync-{state.index}-v{attempt_version}-"
                f"try{attempt}-{uuid.uuid4()}"
            )
            final_output = None
            request_finished = False

            await self._increment_active_attempts()
            try:
                # 调用vllm生成接口，拿到输出后更新state，如果生成过程中被weight update打断了，engine.generate()会抛出异常，直接进入finally块结束这个attempt
                async for request_output in self.engine.generate(
                    {"prompt_token_ids": state.restart_prompt_token_ids},
                    sampling_params,
                    request_id=request_id,
                ):
                    final_output = request_output
                    request_finished = bool(
                        getattr(request_output, "finished", False)
                    )
            finally:
                await self._decrement_active_attempts()

            if final_output is None:
                state.stop_reason = "abort"
                continue

            attempt_tokens = _tokens_from_output(final_output)[:remaining]
            if attempt_tokens:
                state.output_tokens.extend(attempt_tokens)
                state.output_versions.extend(
                    [attempt_version] * len(attempt_tokens)
                )

            stop_reason = _normalize_stop_reason(
                _finish_reason_from_output(final_output)
            )
            if len(state.output_tokens) >= state.requested_max_tokens:
                stop_reason = "length"

            state.stop_reason = stop_reason
            if stop_reason in ("stop", "tool_calls", "length"):
                state.completed = True
                return state

            if not request_finished or stop_reason == "abort":
                await asyncio.sleep(0)
                continue

            state.completed = True
            return state

        state.stop_reason = (
            "length" if state.remaining_max_tokens <= 0 else "abort"
        )
        state.completed = state.stop_reason == "length"
        print(
            "[generate] Request "
            f"{state.index} reached max_resubmit_retries="
            f"{self.max_resubmit_retries}; keeping partial output."
        )
        return state


async def run_repeating_inference(
    runner: InterruptibleGenerationRunner,
    tokenizer,
    prompts: List[str],
    infer_max_tokens: int,
    infer_concurrency: int,
    stop_after_current_cycle: asyncio.Event,
    stats: RepeatingInferenceStats,
) -> RepeatingInferenceStats:
    """Continuously keep a fixed number of inference requests in flight."""
    if not prompts:
        raise ValueError("prompts must not be empty")
    if infer_concurrency < 1:
        raise ValueError("infer_concurrency must be >= 1")

    next_request_index = 0
    next_prompt_index = 0
    stats_lock = asyncio.Lock()
    recent_state_limit = max(len(prompts), infer_concurrency)
    stats.in_flight_concurrency = infer_concurrency

    print(
        f"[infer] Queue started: concurrency={infer_concurrency} "
        f"prompt_count={len(prompts)} infer_max_tokens={infer_max_tokens} "
        f"weight_version={runner.version}"
    )

    async def make_next_state() -> OnlineGenerationState | None:
        nonlocal next_request_index, next_prompt_index
        async with stats_lock:
            if stop_after_current_cycle.is_set():
                return None
            prompt = prompts[next_prompt_index]
            state = OnlineGenerationState(
                index=next_request_index,
                prompt=prompt,
                input_ids=tokenizer.encode(prompt),
                requested_max_tokens=infer_max_tokens,
            )
            next_request_index += 1
            next_prompt_index = (next_prompt_index + 1) % len(prompts)
            return state

    async def record_completed_state(
        worker_id: int,
        state: OnlineGenerationState,
    ) -> None:
        async with stats_lock:
            stats.total_requests += 1
            stats.total_tokens += len(state.output_tokens)
            stats.last_completed_states.append(state)
            if len(stats.last_completed_states) > recent_state_limit:
                stats.last_completed_states = stats.last_completed_states[
                    -recent_state_limit:
                ]

            if state.output_versions:
                version_range = (
                    f"{min(state.output_versions)}-{max(state.output_versions)}"
                )
            else:
                version_range = "none"

            if stats.total_requests % infer_concurrency == 0:
                print(
                    "[infer] Queue progress: "
                    f"completed_requests={stats.total_requests} "
                    f"total_tokens={stats.total_tokens} "
                    f"latest_worker={worker_id} "
                    f"latest_request={state.index} "
                    f"latest_tokens={len(state.output_tokens)} "
                    f"latest_versions={version_range}"
                )

    async def infer_worker(worker_id: int) -> None:
        while True:
            state = await make_next_state()
            if state is None:
                return
            completed_state = await runner.generate(state)
            await record_completed_state(worker_id, completed_state)

    worker_tasks = [
        asyncio.create_task(infer_worker(worker_id))
        for worker_id in range(infer_concurrency)
    ]
    try:
        await asyncio.gather(*worker_tasks)
    finally:
        for task in worker_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*worker_tasks, return_exceptions=True)

    print("[infer] Stop requested; no new inference requests will be launched.")
    return stats


def summarize_weight_payload(dtype_names: List[str], shapes: List[List[int]]) -> float:
    total_weight_bytes = sum(
        numel_from_shape(shape) * dtype_nbytes(dtype_name)
        for dtype_name, shape in zip(dtype_names, shapes)
    )
    return total_weight_bytes / 1024**3


async def sync_weights_to_vllm(
    engine,
    fsdp_workers,
    scope: str,
    transfer_world_size: int,
    packed: bool = True,
):
    validate_weight_scope(scope)
    names, dtype_names, shapes = ray.get(
        fsdp_workers[0].get_weight_metadata.remote(scope)
    )
    model_gib = summarize_weight_payload(dtype_names, shapes)
    infer_payload_gib = model_gib * (transfer_world_size - 1)
    print(
        f"[sync] {scope} metadata: tensors={len(names)}, "
        f"logical_payload={model_gib:.3f} GiB, "
        f"aggregate_infer_payload={infer_payload_gib:.3f} GiB"
    )

    await engine.start_weight_update()
    t0 = time.perf_counter()
    broadcast_handles = [
        worker.gather_and_broadcast_weights.remote(scope=scope, packed=packed)
        for worker in fsdp_workers
    ]
    await engine.update_weights(
        WeightTransferUpdateRequest(
            update_info=asdict(
                NCCLWeightTransferUpdateInfo(
                    names=names,
                    dtype_names=dtype_names,
                    shapes=shapes,
                    packed=packed,
                )
            )
        )
    )
    ray.get(broadcast_handles)
    await engine.finish_weight_update()
    elapsed = time.perf_counter() - t0
    print(
        f"[sync] {scope} weight update complete: {elapsed:.3f}s, "
        f"model-sync throughput={model_gib / elapsed:.3f} GiB/s, "
        f"aggregate-infer throughput={infer_payload_gib / elapsed:.3f} GiB/s"
    )
    return elapsed


async def shutdown_vllm_engine(engine) -> None:
    """Shut down vLLM workers before Ray is torn down."""
    if engine is None:
        return

    shutdown = getattr(engine, "shutdown", None)
    if shutdown is None:
        return

    try:
        result = shutdown()
        if asyncio.iscoroutine(result) or isinstance(result, asyncio.Future):
            await result
        print("[cleanup] vLLM engine shut down.")
    except Exception as exc:
        print(f"[cleanup] Ignoring vLLM engine shutdown error: {exc!r}")


async def run_weight_sync_demo(args: argparse.Namespace):
    if args.ray_address:
        ray.init(address=args.ray_address)
    else:
        ray.init()

    fsdp_workers = []
    inference_loop_task = None
    engine = None
    try:
        # Use local/shared model weights directly.
        local_model_path = args.model_path
        print(f"[init] Loading local model from {local_model_path}")

        # FSDP rendezvous address (single-node)
        fsdp_master_addr = args.fsdp_master_addr or get_local_ip()
        fsdp_master_port = args.fsdp_master_port or find_open_port()

        # Launch FSDP training workers. Ray allocates 1 GPU per worker; vLLM's
        # internal DP placement groups will land on the remaining GPUs.
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

        print(
            "[engine] Creating AsyncLLMEngine with dummy weights "
            # 调度器同一时刻最多允许多少条 sequence 处于 active/running 状态。
            f"(max_num_seqs={args.vllm_max_num_seqs}, "
            # 一次调度最大的toekn总量
            f"max_num_batched_tokens={args.vllm_max_num_batched_tokens})..."
        )
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
            max_num_seqs=args.vllm_max_num_seqs,
            max_num_batched_tokens=args.vllm_max_num_batched_tokens,
        )
        print("[engine] AsyncLLMEngine created.")

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

        print("[sync] Initial full sync from FSDP to vLLM...")
        await engine.pause_generation(mode="abort", clear_cache=True)
        await sync_weights_to_vllm(
            engine=engine,
            fsdp_workers=fsdp_workers,
            scope="all",
            transfer_world_size=transfer_world_size,
            packed=True,
        )
        await engine.resume_generation()
        print("[sync] Initial full sync complete; generation can start.")

        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        tokenizer = engine.get_tokenizer()
        runner = InterruptibleGenerationRunner(engine, temperature=0.0)
        stop_inference = asyncio.Event()
        inference_stats = RepeatingInferenceStats()
        inference_loop_task = asyncio.create_task(
            run_repeating_inference(
                runner=runner,
                tokenizer=tokenizer,
                prompts=prompts,
                infer_max_tokens=args.infer_max_tokens,
                infer_concurrency=args.infer_concurrency,
                stop_after_current_cycle=stop_inference,
                stats=inference_stats,
            )
        )
        print(
            "[infer] Started repeating inference while trainer runs; "
            f"infer_max_tokens={args.infer_max_tokens} "
            f"infer_concurrency={args.infer_concurrency}."
        )

        sync_rounds = 0
        training_reached_max = False
        while not training_reached_max:
            print(
                "[train] Launching trainer segment: "
                f"sync_every_optimizer_steps={args.sync_every_optimizer_steps}"
            )
            train_handles = [
                worker.train_until_next_sync.remote(
                    args.sync_every_optimizer_steps
                )
                for worker in fsdp_workers
            ]
            train_future = asyncio.create_task(
                asyncio.to_thread(ray.get, train_handles)
            )

            done, pending = await asyncio.wait(
                {inference_loop_task, train_future},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if inference_loop_task in done and not stop_inference.is_set():
                train_future.cancel()
                for task in pending:
                    task.cancel()
                raise RuntimeError(
                    "Repeating inference loop exited before trainer finished."
                )

            summaries = train_future.result()
            rank0_summary = next(item for item in summaries if item["rank"] == 0)
            training_reached_max = bool(rank0_summary["reached_max_steps"])
            if rank0_summary["optimizer_steps_run"] <= 0:
                print("[train] No optimizer steps left; stopping sync loop.")
                break

            print(
                "[train] Trainer segment complete: "
                f"optimizer_step={rank0_summary['optimizer_step']} "
                f"steps_run={rank0_summary['optimizer_steps_run']} "
                f"last_loss={rank0_summary['last_loss']:.6f}"
            )

            if (
                args.max_sync_rounds is not None
                and sync_rounds >= args.max_sync_rounds
            ):
                print(
                    "[sync] max_sync_rounds reached; letting inference finish "
                    "without more trainable updates."
                )
                break

            sync_rounds += 1
            print(
                f"[sync] Round {sync_rounds}: pausing generation for "
                "trainable-only weight update..."
            )
            runner.paused.set()
            await engine.pause_generation(mode="abort", clear_cache=True)
            await runner.wait_for_idle()
            print("[sync] Generation paused and in-flight attempts drained.")

            await sync_weights_to_vllm(
                engine=engine,
                fsdp_workers=fsdp_workers,
                scope="trainable",
                transfer_world_size=transfer_world_size,
                packed=True,
            )
            runner.version += 1
            await engine.resume_generation()
            runner.paused.clear()
            print(
                f"[sync] Round {sync_rounds}: resumed generation with "
                f"weight version {runner.version}."
            )

        print(
            "[infer] Trainer finished or sync loop stopped; requesting "
            "inference drain after the last completed communication."
        )
        stop_inference.set()
        if runner.paused.is_set():
            await engine.resume_generation()
            runner.paused.clear()
        inference_stats = await inference_loop_task
    finally:
        if inference_loop_task is not None and not inference_loop_task.done():
            inference_loop_task.cancel()
            await asyncio.gather(inference_loop_task, return_exceptions=True)
        await shutdown_vllm_engine(engine)
        if fsdp_workers:
            try:
                ray.get([worker.close.remote() for worker in fsdp_workers])
            except Exception as exc:
                print(f"[cleanup] Ignoring FSDP worker close error: {exc!r}")
        ray.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a minimal Ray FSDP trainer + vLLM interruptible inference "
            "NCCL weight-sync demo."
        )
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
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum optimizer steps to run before stopping the demo.",
    )
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
        "--sync-every-optimizer-steps",
        type=int,
        default=10,
        help="Sync trainable weights after this many optimizer steps.",
    )
    parser.add_argument(
        "--infer-max-tokens",
        type=int,
        default=256,
        help="Maximum generated tokens per prompt in the weight-sync demo.",
    )
    # 并发用户级别的推理请求数量，vLLM会在这个基础上根据max_num_seqs和max_num_batched_tokens来调度实际的生成请求
    parser.add_argument(
        "--infer-concurrency",
        type=int,
        default=2048,
        help="Number of concurrent user-level inference requests in the demo.",
    )
    # vllm 最多同时调度多少条sequence，也就是最多256个active请求
    parser.add_argument(
        "--vllm-max-num-seqs",
        type=int,
        default=256,
        help="Maximum number of sequences vLLM may schedule concurrently.",
    )
    # vllm 最多同时调度多少个batched tokens，也就是最多16384个batched tokens
    parser.add_argument(
        "--vllm-max-num-batched-tokens",
        type=int,
        default=16384,
        help="Maximum number of batched tokens vLLM may schedule.",
    )
    # 可选的最大同步轮数，超过这个轮数后即使训练还没结束也停止同步，让推理继续跑下去，适合验证推理在不同版本权重下的表现差异
    parser.add_argument(
        "--max-sync-rounds",
        type=int,
        default=None,
        help="Optional maximum number of trainable-only sync rounds in the demo.",
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
    if args.sync_every_optimizer_steps < 1:
        raise ValueError("--sync-every-optimizer-steps must be >= 1")
    if args.infer_max_tokens < 1:
        raise ValueError("--infer-max-tokens must be >= 1")
    if args.infer_concurrency < 1:
        raise ValueError("--infer-concurrency must be >= 1")
    if args.vllm_max_num_seqs < 1:
        raise ValueError("--vllm-max-num-seqs must be >= 1")
    if args.vllm_max_num_batched_tokens < 1:
        raise ValueError("--vllm-max-num-batched-tokens must be >= 1")
    if args.max_sync_rounds is not None and args.max_sync_rounds < 0:
        raise ValueError("--max-sync-rounds must be >= 0 when set")


def main() -> None:
    args = parse_args()
    validate_args(args)
    asyncio.run(run_weight_sync_demo(args))


if __name__ == "__main__":
    main()
