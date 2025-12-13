from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.utils import get_vla, forward_vla
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
        ).to(self.device).to(dtype=self.model_dtype)

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
        return logits

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



