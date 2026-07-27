"""Evaluate the reward model on saved LIBERO episode trajectories."""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.robot.libero.libero_utils import GenerateConfig
from experiments.robot.openvla_utils import get_processor
from rl.models.reward_model import RewardModel
from rl.models.utils import RewardFrameDataset, make_collate_fn


DEFAULT_DATA_DIR = "tests_dsj/dataset_episode"
DEFAULT_REWARD_CHECKPOINT = "/mnt/data/lcx3/checkpoint/reward/reward.pt"
DEFAULT_PRETRAINED_CHECKPOINT = (
    "/mnt/data/lcx3/checkpoint/dsj/"
    "openvla-7b+libero_spatial_no_noops+b32+lr-0.0005+lora-r32+dropout-0.0"
    "--image_aug--parallel_dec--8_acts_chunk--discrete_acts--proprio_state"
    "--100000_chkpt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a binary reward model. Each trajectory contributes its "
            "last frame and episode reward, matching RewardFrameDataset training."
        )
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--reward-checkpoint", default=DEFAULT_REWARD_CHECKPOINT)
    parser.add_argument(
        "--pretrained-checkpoint", default=DEFAULT_PRETRAINED_CHECKPOINT
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Positive-class probability threshold used by F1 and accuracy.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path at which to save the reported metrics.",
    )
    return parser.parse_args()


def compute_auroc(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    """Compute binary AUROC, grouping tied scores at the same threshold."""
    labels = labels.astype(np.int64)
    positive_count = int(labels.sum())
    negative_count = int(labels.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        return None

    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = scores[order]

    threshold_ends = np.r_[
        np.flatnonzero(np.diff(sorted_scores) != 0), sorted_scores.size - 1
    ]
    true_positives = np.cumsum(sorted_labels)[threshold_ends]
    false_positives = (
        np.arange(1, sorted_labels.size + 1)[threshold_ends] - true_positives
    )

    true_positive_rate = np.r_[0.0, true_positives / positive_count]
    false_positive_rate = np.r_[0.0, false_positives / negative_count]
    return float(np.trapz(true_positive_rate, false_positive_rate))


def compute_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> Dict[str, object]:
    predictions = (probabilities >= threshold).astype(np.int64)
    labels = labels.astype(np.int64)

    tp = int(np.sum((predictions == 1) & (labels == 1)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))

    accuracy = float((tp + tn) / labels.size)
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0

    return {
        "num_trajectories": int(labels.size),
        "num_positive": int(labels.sum()),
        "num_negative": int(labels.size - labels.sum()),
        "threshold": threshold,
        "auroc": compute_auroc(labels, probabilities),
        "f1": f1,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def build_config(args: argparse.Namespace) -> GenerateConfig:
    return GenerateConfig(
        pretrained_checkpoint=args.pretrained_checkpoint,
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
        device=torch.device(args.device),
        lora_rank=0,
    )


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")

    data_dir = Path(args.data_dir)
    checkpoint_path = Path(args.reward_checkpoint)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Trajectory directory does not exist: {data_dir}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Reward checkpoint does not exist: {checkpoint_path}")

    torch_dtype = torch.bfloat16
    config = build_config(args)
    processor = get_processor(config)
    dataset = RewardFrameDataset(
        str(data_dir), config, processor, torch_dtype
    )

    model = RewardModel(config, torch_dtype, keep_num=4, focal_alpha=0.9)
    checkpoint = torch.load(
        checkpoint_path, map_location=model.device, weights_only=False
    )
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=model.device.type == "cuda",
        collate_fn=make_collate_fn(model.vla.pad_token_id),
    )

    probabilities = []
    labels = []
    with torch.inference_mode():
        for batch_inputs, batch_labels in tqdm(dataloader, desc="Evaluating"):
            batch_inputs = {
                key: value.to(model.device) if isinstance(value, torch.Tensor) else value
                for key, value in batch_inputs.items()
            }
            logits = model(batch_inputs)
            probabilities.append(
                torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
            )
            labels.append(batch_labels.numpy())

    metrics = compute_metrics(
        np.concatenate(labels),
        np.concatenate(probabilities),
        args.threshold,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if metrics["auroc"] is None:
        print(
            "Warning: AUROC is undefined because the dataset contains only one class."
        )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Metrics saved to {output_path}")


if __name__ == "__main__":
    main()
