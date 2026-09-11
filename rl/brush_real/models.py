"""Brush-specific OpenVLA actor-critic initialization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model

from experiments.robot.sole_utils import get_processor
from experiments.robot.openvla_utils import load_component_state_dict
from prismatic.models.projectors import ProprioProjector
from prismatic.vla.constants import PROPRIO_DIM
from rl.actor_critic_model_discrete import ActorCritic
from rl.utils import get_vla


@dataclass
class BrushOpenVLAConfig:
    pretrained_checkpoint: Union[str, Path]
    sft_checkpoint: Union[str, Path]
    checkpoint2: Union[str, Path, None] = None
    use_l1_regression: bool = False
    use_diffusion: bool = False
    use_film: bool = False
    num_images_in_input: int = 3
    use_proprio: bool = True
    center_crop: bool = True
    num_open_loop_steps: int = 8
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    unnorm_key: str = "brush_realworld"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    device: torch.device = torch.device("cuda")


def _find_single_projector(checkpoint: Path) -> Path:
    matches = sorted(checkpoint.glob("proprio_projector--*_checkpoint.pt"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one proprio projector in {checkpoint}, found {matches}"
        )
    return matches[0]


class BrushActorCritic(ActorCritic):
    """ActorCritic initialized by merging the supplied Brush SFT adapter.

    The generic ActorCritic expects a merged OpenVLA checkpoint. The supplied
    Brush checkpoint is adapter-only, so it must be merged before the action
    vocabulary head is reduced and a fresh trainable RL LoRA is attached.
    """

    def __init__(self, cfg: BrushOpenVLAConfig, torch_dtype: torch.dtype):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.model_dtype = torch_dtype
        sft_checkpoint = Path(cfg.sft_checkpoint).expanduser().resolve()
        adapter_dir = sft_checkpoint / "lora_adapter"

        base = get_vla(cfg, torch_dtype)
        self.device = base.device
        print(f"Loading Brush SFT LoRA from {adapter_dir}")
        self.vla = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
        self.vla = self.vla.to(device=self.device, dtype=torch_dtype)
        with (sft_checkpoint / "dataset_statistics.json").open("r", encoding="utf-8") as stream:
            self.vla.norm_stats = json.load(stream)
        if cfg.unnorm_key not in self.vla.norm_stats:
            raise KeyError(f"{cfg.unnorm_key!r} missing from Brush dataset statistics")

        self.vocab_size = (
            self.vla.config.text_config.vocab_size - self.vla.config.pad_to_multiple_of
        )
        self.n_action_bins = self.vla.config.n_action_bins
        self.action_vocab_start = self.vocab_size - self.n_action_bins
        original_head = self.vla.language_model.lm_head
        with torch.no_grad():
            action_weight = original_head.weight[
                self.action_vocab_start : self.vocab_size
            ].clone()
            action_bias = (
                original_head.bias[self.action_vocab_start : self.vocab_size].clone()
                if original_head.bias is not None
                else None
            )
        slim_head = nn.Linear(
            original_head.in_features,
            self.n_action_bins,
            bias=action_bias is not None,
        ).to(device=self.device, dtype=torch_dtype)
        with torch.no_grad():
            slim_head.weight.copy_(action_weight)
            if action_bias is not None:
                slim_head.bias.copy_(action_bias)
        self.vla.language_model.lm_head = slim_head

        for parameter in self.vla.parameters():
            parameter.requires_grad = False
        if cfg.use_lora:
            self.vla = get_peft_model(
                self.vla,
                LoraConfig(
                    r=cfg.lora_rank,
                    lora_alpha=min(cfg.lora_rank, 16),
                    lora_dropout=cfg.lora_dropout,
                    target_modules="all-linear",
                    init_lora_weights="gaussian",
                ),
            )
        for parameter in self.vla.language_model.lm_head.parameters():
            parameter.requires_grad = True

        import numpy as np

        self.bins = np.linspace(-1, 1, self.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.processor = get_processor(cfg)
        self.proprio_projector = ProprioProjector(
            llm_dim=self.vla.llm_dim, proprio_dim=PROPRIO_DIM
        )
        projector_path = _find_single_projector(sft_checkpoint)
        self.proprio_projector.load_state_dict(
            load_component_state_dict(str(projector_path)), strict=True
        )
        self.proprio_projector = self.proprio_projector.to(
            device=self.device, dtype=torch_dtype
        )
        self.attn_pool = nn.Sequential(nn.Linear(self.vla.llm_dim, 1))
        self.value_head = nn.Sequential(
            nn.LayerNorm(self.vla.llm_dim),
            nn.Linear(self.vla.llm_dim, self.vla.llm_dim),
            nn.ReLU(),
            nn.Linear(self.vla.llm_dim, 1),
        )
        self.to(device=self.device, dtype=torch_dtype)
        if cfg.checkpoint2:
            self.load_checkpoint2(cfg.checkpoint2)

