import filecmp
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import json_numpy
import numpy as np
import requests
import tensorflow as tf
import torch
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

# Apply JSON numpy patch for serialization
json_numpy.patch()

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import DiffusionActionHead, L1RegressionActionHead
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import NoisyActionProjector, ProprioProjector
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
)
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

## 单独创建该文件，将这两个utils方法复制出来，进行改造
from experiments.robot.openvla_utils import model_is_on_hf_hub, find_checkpoint_file, load_component_state_dict

def get_processor(cfg: Any) -> AutoProcessor:
    """
    Get the VLA model's Hugging Face processor.

    Args:
        cfg: Configuration object with model parameters

    Returns:
        AutoProcessor: The model's processor
    """
    return AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)

def get_proprio_projector(cfg: Any, llm_dim: int, proprio_dim: int, device, dtype = torch.bfloat16) -> ProprioProjector:
    """
    Get proprioception projector for the VLA model.


    Args:
        cfg: Configuration object with model parameters
        llm_dim: Dimension of the language model
        proprio_dim: Dimension of proprioception data


    Returns:
        ProprioProjector: The initialized proprio projector
    """
    # Initialize projector and move to device
    proprio_projector = ProprioProjector(
        llm_dim=llm_dim,
        proprio_dim=proprio_dim,
    ).to(device)
    proprio_projector = proprio_projector.to(dtype).to(device) #bfloat16
    proprio_projector.eval()

    # Find and load checkpoint (may be on Hugging Face Hub or stored locally)
    if model_is_on_hf_hub(cfg.pretrained_checkpoint):
        model_path_to_proprio_projector_name = {
            "moojink/openvla-7b-oft-finetuned-libero-spatial": "proprio_projector--150000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-object": "proprio_projector--150000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-goal": "proprio_projector--50000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-10": "proprio_projector--150000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10": "proprio_projector--300000_checkpoint.pt",
        }
        if cfg.pretrained_checkpoint not in model_path_to_proprio_projector_name.keys():
            raise ValueError("Unsupported HF Hub pretrained checkpoint found!")
        # Download proprio projector directly from HF Hub
        proprio_projector_path = hf_hub_download(
            repo_id=cfg.pretrained_checkpoint, filename=model_path_to_proprio_projector_name[cfg.pretrained_checkpoint]
        )
        state_dict = load_component_state_dict(proprio_projector_path)
        proprio_projector.load_state_dict(state_dict)
    else:
        try:
            checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "proprio_projector")
            state_dict = load_component_state_dict(checkpoint_path)
            proprio_projector.load_state_dict(state_dict)
        except:
            print(f"{cfg.pretrained_checkpoint}中未发现proprio_projector!")
    return proprio_projector



def get_action_head(cfg: Any, llm_dim: int, device, dtype = torch.bfloat16) -> Union[L1RegressionActionHead, DiffusionActionHead]:
    """
    Get action head for continuous value prediction.

    Args:
        cfg: Configuration object with model parameters
        llm_dim: Dimension of the language model

    Returns:
        Union[L1RegressionActionHead, DiffusionActionHead]: The initialized action head

    Raises:
        AssertionError: If both L1 regression and diffusion are specified
    """
    assert not (cfg.use_l1_regression and cfg.use_diffusion), "Cannot use both L1 regression and diffusion action head!"

    # Initialize appropriate action head based on configuration
    if cfg.use_l1_regression:
        action_head = L1RegressionActionHead(input_dim=llm_dim, hidden_dim=llm_dim, action_dim=ACTION_DIM)
    elif cfg.use_diffusion:
        action_head = DiffusionActionHead(
            input_dim=llm_dim, hidden_dim=llm_dim, action_dim=ACTION_DIM, num_diffusion_steps_train=cfg.num_diffusion_steps_train
        )
        # Set number of diffusion steps for inference
        action_head.noise_scheduler.set_timesteps(cfg.num_diffusion_steps_inference)
    else:
        raise ValueError("Either use_l1_regression or use_diffusion must be True")

    action_head = action_head.to(dtype).to(device) 
    action_head.eval()

    # Find and load checkpoint (may be on Hugging Face Hub or stored locally)
    if model_is_on_hf_hub(cfg.pretrained_checkpoint):
        model_path_to_action_head_name = {
            "moojink/openvla-7b-oft-finetuned-libero-spatial": "action_head--150000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-object": "action_head--150000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-goal": "action_head--50000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-10": "action_head--150000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10": "action_head--300000_checkpoint.pt",
        }
        if cfg.pretrained_checkpoint not in model_path_to_action_head_name.keys():
            raise ValueError("Unsupported HF Hub pretrained checkpoint found!")
        # Download proprio projector directly from HF Hub
        action_head_path = hf_hub_download(
            repo_id=cfg.pretrained_checkpoint, filename=model_path_to_action_head_name[cfg.pretrained_checkpoint]
        )
        state_dict = load_component_state_dict(action_head_path)
        action_head.load_state_dict(state_dict)
    else:
        checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "action_head")
        state_dict = load_component_state_dict(checkpoint_path)
        action_head.load_state_dict(state_dict)

    return action_head
