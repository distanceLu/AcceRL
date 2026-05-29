# SPDX-License-Identifier: Apache-2.0
"""Local single-process trainer smoke-test for a local chat model.

This script intentionally avoids Ray, torch.distributed, FSDP, vLLM, and NCCL.
It loads a local Hugging Face CausalLM model, builds a tiny in-memory dummy
dataset, and runs a few optimizer steps to validate the trainer path.

Default behavior is conservative: only ``lm_head`` is trainable.  That makes it
useful for validating the local training lifecycle before moving to full SFT.
"""

import argparse
import os
import random
from typing import Dict, Iterable, List, Tuple


DEFAULT_MODEL_PATH = "/mnt/data/lcx4/why_workspace/hf_cache/Qwen1.5-MoE-A2.7B-Chat"
DEFAULT_OUTPUT_DIR = "/mnt/data/lcx4/why_workspace/outputs/local_trainer_smoke"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local single-process training smoke test."
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--dataset-repeat", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
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

    global torch, DataLoader, AutoModelForCausalLM, AutoTokenizer
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_seed(args.seed)

    device = pick_device(args.device)
    torch_dtype = pick_dtype(args.dtype)

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

    configure_trainable_parameters(model, args.train_mode)
    total_params, trainable_params = count_parameters(model)
    trainable_parameter_list = list(iter_trainable_parameters(model))
    if not trainable_parameter_list:
        raise RuntimeError(f"No trainable parameters found for mode: {args.train_mode}")
    print(
        "[train] Parameter count: "
        f"trainable={trainable_params:,} / total={total_params:,} "
        f"({trainable_params / total_params:.4%})"
    )

    dataset = DummyChatDataset(
        tokenizer=tokenizer,
        max_length=args.max_length,
        repeat=args.dataset_repeat,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer),
    )

    optimizer = torch.optim.AdamW(
        trainable_parameter_list,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print(
        f"[train] Starting local training: max_steps={args.max_steps}, "
        f"batch_size={args.batch_size}, grad_accum_steps={args.grad_accum_steps}"
    )

    step = 0
    optimizer.zero_grad(set_to_none=True)
    while step < args.max_steps:
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
            if step % args.log_every == 0:
                print(f"[train] step={step} loss={loss.item() * args.grad_accum_steps:.6f}")

            if step >= args.max_steps:
                break

    if args.save_checkpoint:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[save] Saving checkpoint to {args.output_dir}")
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    else:
        print("[save] Skipped checkpoint save; trainer path validation only.")

    print("[done] Local trainer smoke test finished.")


if __name__ == "__main__":
    main()
