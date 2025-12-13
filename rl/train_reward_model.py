import argparse
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from experiments.robot.libero.libero_utils import GenerateConfig
from experiments.robot.openvla_utils import get_processor
from rl.models.reward_model import RewardModel
from rl.utils import prepare_one_obs
import numpy as np


class RewardFrameDataset(Dataset):
    """
    Treat each frame in a saved sample as an independent training example.
    Sample file format follows generate_actor_critic_discrete_data.py output.
    """

    def __init__(self, data_dirs: List[str], cfg: Any, processor: Any, torch_dtype: torch.dtype):
        self.cfg = cfg
        self.processor = processor
        self.torch_dtype = torch_dtype

        self.items: List[Tuple[str, int, int]] = []  # (file_path, frame_idx, reward_bool)
        for data_dir in data_dirs:
            if not os.path.isdir(data_dir):
                continue
            for fname in os.listdir(data_dir):
                if not fname.endswith(".pt"):
                    continue
                fpath = os.path.join(data_dir, fname)
                sample = torch.load(fpath)
                # video = sample["video"]  # (T, H, W, 3)
                mask = sample["mask"]    # (T,)
                last_rew = sample["reward"]
                assert np.all(mask == 1) 
                # frag_rew = np.zeros_like(mask)
                # frag_rew[-1] = last_rew
                # reward_bool = 1 if float(sample["reward"]) > 0 else 0
                # for idx, valid in enumerate(mask):
                #     if bool(valid):
                #         self.items.append((fpath, idx, frag_rew[idx]))
                self.items.append((fpath, len(mask)-1, last_rew))

        if len(self.items) == 0:
            raise ValueError(f"No frames found in {data_dirs}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        fpath, frame_idx, rew = self.items[index]
        sample = torch.load(fpath)
        frame = sample["video"][frame_idx]  # HWC uint8
        instruction = sample["instruction"]

        obs = {"full_image": frame}
        inputs = prepare_one_obs(self.cfg, self.processor, obs, instruction, self.torch_dtype)
        # Remove proprio if unused (avoid None in collate)
        if not self.cfg.use_proprio and ("proprio" in inputs) and (inputs["proprio"] is None):
            inputs.pop("proprio", None)
        return inputs, torch.tensor(rew, dtype=torch.long)


def _simple_pad_batch(pad_token_id, inputs_list):
    # inputs_list elements have: input_ids, attention_mask, labels, pixel_values
    max_len = max(it["input_ids"].size(1) for it in inputs_list)
    padded = []
    for it in inputs_list:
        # skip None entries (e.g., proprio when unused)
        it = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in it.items() if v is not None}
        cur_len = it["input_ids"].size(1)
        if cur_len < max_len:
            pad_amt = max_len - cur_len
            bsz = it["input_ids"].size(0)
            pad_ids = it["input_ids"].new_full((bsz, pad_amt), pad_token_id)
            pad_mask = it["attention_mask"].new_zeros((bsz, pad_amt))
            pad_labels = it["labels"].new_full((bsz, pad_amt), -100)
            it["input_ids"] = torch.cat([it["input_ids"], pad_ids], dim=1)
            it["attention_mask"] = torch.cat([it["attention_mask"], pad_mask], dim=1)
            it["labels"] = torch.cat([it["labels"], pad_labels], dim=1)
        padded.append(it)

    batch = {}
    keys = padded[0].keys()
    for k in keys:
        tensors = [it[k] for it in padded]
        batch[k] = torch.cat(tensors, dim=0)
    return batch


def make_collate_fn(pad_token_id):
    def _fn(batch):
        inputs_list, labels = zip(*batch)
        batch_inputs = _simple_pad_batch(pad_token_id, list(inputs_list))
        batch_labels = torch.stack(labels)
        return batch_inputs, batch_labels
    return _fn


def build_cfg(device: str, pretrained_checkpoint: str) -> Any:
    return GenerateConfig(
        pretrained_checkpoint=pretrained_checkpoint,
        use_l1_regression=False,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=1,
        use_proprio=False,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        num_open_loop_steps=1,
        unnorm_key="libero_spatial_no_noops",
        device=torch.device(device),
        lora_rank=0,
    )


def train_one_epoch(model, dataloader, optimizer, device, grad_accum, writer, global_step):
    model.train()
    total_loss = 0.0
    tp = tn = fp = fn = 0
    optimizer.zero_grad()

    # accumulators for one grad-accum window
    accum_loss = 0.0
    accum_tp = accum_tn = accum_fp = accum_fn = 0
    accum_count = 0

    for step, (batch_inputs, labels) in enumerate(tqdm(dataloader, desc="train", leave=False)):
        # Move tensors
        for k, v in batch_inputs.items():
            if isinstance(v, torch.Tensor):
                batch_inputs[k] = v.to(device)
        labels = labels.to(device)

        loss, metrics = model.compute_loss_and_metrics(batch_inputs, labels)
        loss = loss / grad_accum
        loss.backward()

        total_loss += metrics["loss"].item()
        tp += metrics["tp"].item()
        tn += metrics["tn"].item()
        fp += metrics["fp"].item()
        fn += metrics["fn"].item()

        # accumulate for logging at optimizer step
        accum_loss += metrics["loss"].item()
        accum_tp += metrics["tp"].item()
        accum_tn += metrics["tn"].item()
        accum_fp += metrics["fp"].item()
        accum_fn += metrics["fn"].item()
        accum_count += 1

        if (step + 1) % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad()
            # log averages for this accumulation window at optimizer step
            avg_loss = accum_loss / max(accum_count, 1)
            writer.add_scalar("train/loss", avg_loss, global_step)
            # log absolute counts (no averaging)
            writer.add_scalar("train/tp", accum_tp, global_step)
            writer.add_scalar("train/tn", accum_tn, global_step)
            writer.add_scalar("train/fp", accum_fp, global_step)
            writer.add_scalar("train/fn", accum_fn, global_step)
            # per-window positive / negative accuracy
            pos_den = accum_tp + accum_fn
            neg_den = accum_tn + accum_fp
            pos_acc = float(accum_tp) / pos_den if pos_den > 0 else 0.0
            neg_acc = float(accum_tn) / neg_den if neg_den > 0 else 0.0
            writer.add_scalar("train/pos_acc", pos_acc, global_step)
            writer.add_scalar("train/neg_acc", neg_acc, global_step)
            global_step += 1
            accum_loss = accum_tp = accum_tn = accum_fp = accum_fn = 0
            accum_count = 0

    stats = {
        "loss": total_loss / len(dataloader),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
    # epoch-level positive / negative accuracy
    pos_den = tp + fn
    neg_den = tn + fp
    stats["pos_acc"] = float(tp) / pos_den if pos_den > 0 else 0.0
    stats["neg_acc"] = float(tn) / neg_den if neg_den > 0 else 0.0
    return stats, global_step


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    tp = tn = fp = fn = 0

    for batch_inputs, labels in tqdm(dataloader, desc="eval", leave=False):
        for k, v in batch_inputs.items():
            if isinstance(v, torch.Tensor):
                batch_inputs[k] = v.to(device)
        labels = labels.to(device)
        loss, metrics = model.compute_loss_and_metrics(batch_inputs, labels)
        total_loss += metrics["loss"].item()
        tp += metrics["tp"].item()
        tn += metrics["tn"].item()
        fp += metrics["fp"].item()
        fn += metrics["fn"].item()

    stats = {
        "loss": total_loss / len(dataloader),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
    pos_den = tp + fn
    neg_den = tn + fp
    stats["pos_acc"] = float(tp) / pos_den if pos_den > 0 else 0.0
    stats["neg_acc"] = float(tn) / neg_den if neg_den > 0 else 0.0
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dirs", nargs="+", default=["/cpfs01/lcx_workspace/Open-Sora/debug/spatial_ep300"])
    parser.add_argument("--test_dirs", nargs="+", default=["/cpfs01/lcx_workspace/Open-Sora/debug/spatial_ep30"])
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="runs/reward_model")
    parser.add_argument("--exp_name", type=str, default="reward_model")
    parser.add_argument("--focal_alpha", type=float, default=0.9)
    parser.add_argument(
        "--pretrained_checkpoint",
        type=str,
        default="/cpfs01/liuwei_workspace/models/finetune_im/openvla-7b+libero_spatial_no_noops+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--discrete_acts--proprio_state--100000_chkpt",
    )
    args = parser.parse_args()

    torch_dtype = torch.bfloat16
    cfg = build_cfg(args.device, args.pretrained_checkpoint)
    processor = get_processor(cfg)
    train_ds = RewardFrameDataset(args.train_dirs, cfg, processor, torch_dtype)

    # Model & processor
    model = RewardModel(cfg, torch_dtype, keep_num=4, focal_alpha=args.focal_alpha)
    # processor = model.processor

    val_ds = RewardFrameDataset(args.test_dirs, cfg, processor, torch_dtype)

    collate = make_collate_fn(model.vla.pad_token_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate,
    )

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=Path(args.output_dir) / f"{timestamp}_{args.exp_name}")
    best_val = float("inf")
    global_step = 0

    def _log_stats(prefix: str, stats: Dict[str, float], step: int):
        for k, v in stats.items():
            writer.add_scalar(f"{prefix}/{k}", v, step)

    for epoch in range(1, args.epochs + 1):
        train_stats, global_step = train_one_epoch(model, train_loader, optimizer, model.device, args.grad_accum, writer, global_step)
        val_stats = evaluate(model, val_loader, model.device)

        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train": train_stats,
                    "val": val_stats,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

        # epoch-level stats (logged at current global_step)
        _log_stats("eval", val_stats, global_step)

        # Save checkpoint
        ckpt_path = Path(args.output_dir) / f"epoch_{epoch}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "cfg": cfg.__dict__,
                "epoch": epoch,
                "val_loss": val_stats["loss"],
            },
            ckpt_path,
        )

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "cfg": cfg.__dict__,
                    "epoch": epoch,
                    "val_loss": val_stats["loss"],
                },
                Path(args.output_dir) / "best.pt",
            )

    writer.close()


if __name__ == "__main__":
    main()

