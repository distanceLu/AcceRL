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
import numpy as np

from experiments.robot.libero.libero_utils import GenerateConfig
from experiments.robot.openvla_utils import get_processor
from rl.models.reward_model import RewardModel
from rl.models.utils import (
    RewardFrameDataset,
    compute_pr_auc,
    make_collate_fn,
)


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


def train_one_epoch(model, dataloader, optimizer, device, grad_accum, writer, global_step, clip_grad_norm):
    model.train()
    total_loss = 0.0
    tp = tn = fp = fn = 0
    optimizer.zero_grad()

    # accumulators for one grad-accum window
    accum_loss = 0.0
    accum_tp = accum_tn = accum_fp = accum_fn = 0
    accum_count = 0

    # Collect all logits and labels for PR-AUC
    all_logits = []
    all_labels = []

    for step, (batch_inputs, labels) in enumerate(tqdm(dataloader, desc="train", leave=False)):
        # Move tensors
        for k, v in batch_inputs.items():
            if isinstance(v, torch.Tensor):
                batch_inputs[k] = v.to(device)
        labels = labels.to(device)

        logits = model.forward(batch_inputs)
        loss, metrics = model.compute_loss_and_metrics(batch_inputs, labels)
        loss = loss / grad_accum
        loss.backward()

        # Collect for PR-AUC
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

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
            if clip_grad_norm is not None and clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            # log averages for this accumulation window at optimizer step
            avg_loss = accum_loss / max(accum_count, 1)
            writer.add_scalar("train/loss", avg_loss, global_step)
            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("train/lr", current_lr, global_step)
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

    # Compute PR-AUC for entire epoch
    all_logits_tensor = torch.cat(all_logits, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)
    # Get positive class probabilities
    probs = torch.softmax(all_logits_tensor.float(), dim=-1)[:, 1].numpy()
    labels_np = all_labels_tensor.float().numpy()
    pr_auc = compute_pr_auc(labels_np, probs)

    stats = {
        "loss": total_loss / len(dataloader),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "pr_auc": pr_auc,
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

    # Collect all logits and labels for PR-AUC
    all_logits = []
    all_labels = []

    for batch_inputs, labels in tqdm(dataloader, desc="eval", leave=False):
        for k, v in batch_inputs.items():
            if isinstance(v, torch.Tensor):
                batch_inputs[k] = v.to(device)
        labels = labels.to(device)
        logits = model.forward(batch_inputs)
        loss, metrics = model.compute_loss_and_metrics(batch_inputs, labels)
        
        # Collect for PR-AUC
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

        total_loss += metrics["loss"].item()
        tp += metrics["tp"].item()
        tn += metrics["tn"].item()
        fp += metrics["fp"].item()
        fn += metrics["fn"].item()

    # Compute PR-AUC
    all_logits_tensor = torch.cat(all_logits, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)
    # Get positive class probabilities
    probs = torch.softmax(all_logits_tensor.float(), dim=-1)[:, 1].numpy()
    labels_np = all_labels_tensor.float().numpy()
    pr_auc = compute_pr_auc(labels_np, probs)

    stats = {
        "loss": total_loss / len(dataloader),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "pr_auc": pr_auc,
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
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_log_dir = Path(args.output_dir) / f"{timestamp}_{args.exp_name}"
    writer = SummaryWriter(log_dir=tb_log_dir)
    best_pr_auc = -1.0  # Track best PR-AUC instead of loss
    global_step = 0

    def _log_stats(prefix: str, stats: Dict[str, float], step: int):
        for k, v in stats.items():
            writer.add_scalar(f"{prefix}/{k}", v, step)

    for epoch in range(1, args.epochs + 1):
        train_stats, global_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            model.device,
            args.grad_accum,
            writer,
            global_step,
            args.clip_grad_norm,
        )
        val_stats = evaluate(model, val_loader, model.device)
        scheduler.step()

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
        _log_stats("train_epoch", train_stats, global_step)
        _log_stats("eval", val_stats, global_step)

        # Save best model based on PR-AUC (save to tensorboard directory)
        if val_stats["pr_auc"] > best_pr_auc:
            best_pr_auc = val_stats["pr_auc"]
            ckpt_path = tb_log_dir / "best_model.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "cfg": cfg.__dict__,
                    "epoch": epoch,
                    "val_loss": val_stats["loss"],
                    "val_pr_auc": val_stats["pr_auc"],
                },
                ckpt_path,
            )
            print(f"Saved best model (PR-AUC: {best_pr_auc:.4f}) to {ckpt_path}")

    writer.close()


if __name__ == "__main__":
    main()

