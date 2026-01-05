from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.utils import get_vla, forward_vla, prepare_inputs_batch
from experiments.robot.openvla_utils import get_processor


class RewardModel(nn.Module):
    """
    Reward-only classifier using OpenVLA vision backbone.

    - Keeps vision_backbone intact.
    - Truncates language model to the first `keep_num` layers (not used in forward).
    - Uses a simple pooled head -> 2-class logits (reward 0/1).
    """

    def __init__(self, cfg: Any, torch_dtype: torch.dtype, keep_num: int = 4, focal_alpha: float = 0.9):
        super().__init__()
        self.cfg = cfg
        self.model_dtype = torch_dtype

        # Build VLA (includes processor + vision_backbone)
        self.vla = get_vla(cfg, torch_dtype)
        self.device = self.vla.device
        # Expose processor like actor_critic_model
        self.processor = get_processor(cfg)

        self.vla.language_model.model.layers = self.vla.language_model.model.layers[:keep_num]

        # Simple reward head on pooled vision features
        hidden_size = self.vla.llm_dim
        self.reward_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 2),
        )

        # Focal loss params
        self.focal_alpha = focal_alpha
        self.focal_gamma = 2.0

        self.to(self.device).to(dtype=self.model_dtype)

    def _forward_vla(self, batch: Dict[str, torch.Tensor]):
        return forward_vla(self, batch)

    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Binary focal loss on logits; targets are bool/0/1."""
        targets = targets.long()
        logp = F.log_softmax(logits.to(torch.float32), dim=-1)
        p = logp.exp()
        pt = p.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)      # p_t
        logpt = logp.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)

        # class-balanced alpha: pos -> alpha, neg -> (1 - alpha)
        alpha_t = torch.where(targets == 1, self.focal_alpha, 1.0 - self.focal_alpha)

        focal_weight = alpha_t * (1 - pt) ** self.focal_gamma
        loss = -focal_weight * logpt
        return loss.mean()

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Expects batch produced by `prepare_inputs_batch` with keys:
            input_ids, attention_mask, pixel_values, labels (ignored)
        Returns:
            logits: (B, 2)
        """
        output = self._forward_vla(batch)
        hid_state = output.hidden_states[-1]  # (B, seq_len, D)
        # Following world_model_discrete pattern: use token at position 1
        pooled = hid_state[:, 1]
        logits = self.reward_head(pooled)
        return logits.float()

    def compute_loss_and_metrics(self, batch: Dict[str, torch.Tensor], labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        logits = self.forward(batch)
        loss = self.focal_loss(logits, labels)

        preds = logits.argmax(dim=-1)
        labels_bool = labels.bool()
        preds_bool = preds.bool()

        tp = torch.sum((preds_bool == True) & (labels_bool == True)).item()
        tn = torch.sum((preds_bool == False) & (labels_bool == False)).item()
        fp = torch.sum((preds_bool == True) & (labels_bool == False)).item()
        fn = torch.sum((preds_bool == False) & (labels_bool == True)).item()

        metrics = {
            "loss": loss.detach(),
            "tp": torch.tensor(tp, device=logits.device),
            "tn": torch.tensor(tn, device=logits.device),
            "fp": torch.tensor(fp, device=logits.device),
            "fn": torch.tensor(fn, device=logits.device),
        }
        return loss, metrics

    def load_checkpoint(self, load_path: str):
        """Load model state dict from file."""
        state_dict = torch.load(load_path, map_location=self.device)
        self.load_state_dict(state_dict['model'], strict=True)  # TODO

    def get_norm_stats(self):
        return self.vla.norm_stats[self.cfg.unnorm_key]["proprio"]

    def prepare_inputs_batch(self, inp, max_len=None):
        return prepare_inputs_batch(self, inp, max_len)



if __name__ == "__main__":
    import json
    from pathlib import Path
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from experiments.robot.libero.libero_utils import GenerateConfig
    from experiments.robot.openvla_utils import get_processor
    from rl.models.utils import RewardFrameDataset, compute_pr_auc, make_collate_fn

    # Configuration
    model_path = "/cpfs01/lcx_workspace/openvla-oft/runs/reward_model/20251217_112319_spatial_1task/best.pt"
    test_data_dir = "/cpfs01/lcx_workspace/Open-Sora/debug/spatial_1task_ep30"
    device = "cuda:0"
    batch_size = 32
    pretrained_checkpoint = "/cpfs01/liuwei_workspace/models/finetune_im/openvla-7b+libero_spatial_no_noops+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--discrete_acts--proprio_state--100000_chkpt"

    print("=" * 80)
    print("Loading model and test data...")
    print("=" * 80)

    # Build config
    torch_dtype = torch.bfloat16
    cfg = GenerateConfig(
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

    # Load model
    model = RewardModel(cfg, torch_dtype, keep_num=4, focal_alpha=0.9)
    model.load_checkpoint(model_path)
    model.eval()
    print(f"✓ Model loaded from {model_path}")

    # Load test data
    processor = get_processor(cfg)
    test_ds = RewardFrameDataset(test_data_dir, cfg, processor, torch_dtype)
    collate = make_collate_fn(model.vla.pad_token_id)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate,
    )
    print(f"✓ Test dataset loaded: {len(test_ds)} samples")

    # Evaluate
    print("\n" + "=" * 80)
    print("Evaluating model...")
    print("=" * 80)

    total_loss = 0.0
    tp = tn = fp = fn = 0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch_inputs, labels in tqdm(test_loader, desc="Evaluating"):
            for k, v in batch_inputs.items():
                if isinstance(v, torch.Tensor):
                    batch_inputs[k] = v.to(model.device)
            labels = labels.to(model.device)
            logits = model.forward(batch_inputs)
            loss, metrics = model.compute_loss_and_metrics(batch_inputs, labels)

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
    probs = torch.softmax(all_logits_tensor.float(), dim=-1)[:, 1].numpy()
    labels_np = all_labels_tensor.float().numpy()
    pr_auc = compute_pr_auc(labels_np, probs)

    # Compute accuracies
    pos_den = tp + fn
    neg_den = tn + fp
    pos_acc = float(tp) / pos_den if pos_den > 0 else 0.0
    neg_acc = float(tn) / neg_den if neg_den > 0 else 0.0
    overall_acc = float(tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    # Print results
    print("\n" + "=" * 80)
    print("Evaluation Results:")
    print("=" * 80)
    print(f"Loss: {total_loss / len(test_loader):.6f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives (TP):  {tp}")
    print(f"  True Negatives (TN):  {tn}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"\nAccuracy Metrics:")
    print(f"  Overall Accuracy:    {overall_acc:.4%}")
    print(f"  Positive Accuracy:    {pos_acc:.4%} (TP / (TP + FN))")
    print(f"  Negative Accuracy:    {neg_acc:.4%} (TN / (TN + FP))")
    print(f"\nPR-AUC: {pr_auc:.6f}")
    print("=" * 80)

    # Save results to JSON
    results = {
        "loss": total_loss / len(test_loader),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "overall_acc": overall_acc,
        "pos_acc": pos_acc,
        "neg_acc": neg_acc,
        "pr_auc": pr_auc,
    }
    results_path = Path(model_path).parent / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {results_path}")



