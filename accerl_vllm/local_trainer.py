# SPDX-License-Identifier: Apache-2.0
"""Local trainer smoke-test for a local chat model.

By default this script runs the original single-process smoke test.  With
``--use-fsdp`` it launches Ray actors that form a NCCL process group and run
the same dummy SFT loop under PyTorch FSDP2.

Default behavior is conservative: only ``lm_head`` is trainable.  That makes it
useful for validating the local training lifecycle before moving to full SFT.
"""

from __future__ import annotations

import argparse
import os
import random
import socket
from typing import Dict, Iterable, List, Tuple


DEFAULT_MODEL_PATH = "/mnt/data/lcx4/hf_cache/Qwen1.5-MoE-A2.7B-Chat"
DEFAULT_OUTPUT_DIR = "/mnt/data/lcx4/outputs/local_trainer_smoke"


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
        "assistant": '{"name": "local-trainer", "status": "ok"}',
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
        self.examples = []  # type: List[EncodedExample]
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


def pick_device(device_name: str):
    if device_name != "auto":
        return torch.device(device_name)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def move_batch_to_device(batch: Dict, device) -> Dict:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def ensure_training_imports(import_distributed: bool = False) -> None:
    global torch, DataLoader, AutoModelForCausalLM, AutoTokenizer

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if import_distributed:
        global DistributedSampler

        from torch.utils.data.distributed import DistributedSampler


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


class FSDPTrainWorker:
    """
    One Ray actor per GPU. Actors form one FSDP2 process group and run training.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        rank: int,
        fsdp_world_size: int,
        fsdp_master_addr: str,
        fsdp_master_port: int,
    ):
        ensure_training_imports(import_distributed=True)

        import torch.distributed as dist
        from torch.distributed.fsdp import fully_shard

        self.args = args
        self.rank = rank
        self.fsdp_world_size = fsdp_world_size
        self.dist = dist

        os.environ["MASTER_ADDR"] = fsdp_master_addr
        os.environ["MASTER_PORT"] = str(fsdp_master_port)

        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=fsdp_world_size,
        )
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

        print(f"[rank {rank}] FSDP worker ready.")

    def get_rank(self) -> int:
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
            self.dist.barrier()
            if self.rank == 0:
                print("[done] Ray FSDP trainer smoke test finished.")
            return summary
        finally:
            if self.dist.is_initialized():
                self.dist.destroy_process_group()


def run_single_process(args: argparse.Namespace) -> None:
    ensure_training_imports()

    set_seed(args.seed)

    device = pick_device(args.device)
    torch_dtype = pick_dtype(args.dtype)

    tokenizer = build_tokenizer(args)
    model = build_model(args, device, torch_dtype)

    configure_trainable_parameters(model, args.train_mode)
    trainable_parameter_list = log_parameter_count(model, args.train_mode)

    dataset = build_dataset(args, tokenizer)
    dataloader = build_dataloader(args, tokenizer, dataset, shuffle=True)

    optimizer = torch.optim.AdamW(
        trainable_parameter_list,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    run_training_loop(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        trainable_parameter_list=trainable_parameter_list,
        device=device,
        args=args,
    )

    if args.save_checkpoint:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[save] Saving checkpoint to {args.output_dir}")
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    else:
        print("[save] Skipped checkpoint save; trainer path validation only.")

    print("[done] Local trainer smoke test finished.")


def run_fsdp(args: argparse.Namespace) -> None:
    if args.save_checkpoint:
        print("[save] Ray FSDP mode does not support --save-checkpoint yet; skipping.")

    import ray

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
            f"steps={rank0_summary['steps']} last_loss={rank0_summary['last_loss']:.6f}"
        )
    finally:
        ray.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local training smoke test, optionally with Ray FSDP."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Only used when --save-checkpoint is set.",
    )
    parser.add_argument("--device", default="cuda:0", help="auto, cpu, cuda, cuda:0, ...")
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
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--dataset-repeat", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--use-fsdp", action="store_true")
    parser.add_argument("--fsdp-world-size", type=int, default=4)
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
        help="Optionally save model/tokenizer after the smoke test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.fsdp_world_size < 1:
        raise ValueError("--fsdp-world-size must be >= 1")

    if args.use_fsdp:
        run_fsdp(args)
    else:
        run_single_process(args)


if __name__ == "__main__":
    main()
