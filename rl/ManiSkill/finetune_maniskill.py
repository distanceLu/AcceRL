"""
finetune_noddp.py

Fine-tunes OpenVLA via LoRA (No DDP version for easier debugging).
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2" 
os.environ["VULKAN_VISIBLE_DEVICES"] = "1" 

# 必须在任何 TFDS/RLDS 相关东西之前
# 让tf不使用gpu

import tensorflow as tf
# 先隐藏
tf.config.set_visible_devices([], "GPU")
# 再检查：这里应该是 []
print("TF visible GPUs:", [d for d in tf.config.get_visible_devices() if d.device_type == "GPU"])
# 或者检查 logical（这里也应该是 []）
print("TF logical GPUs:", tf.config.list_logical_devices("GPU"))


os.environ["WANDB_MODE"] = "disabled"
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Type
from contextlib import nullcontext

import draccus
import torch
import torch.nn as nn
import tqdm
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb

from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map,
)

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import DiffusionActionHead, L1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import (
    NoisyActionProjector,
    ProprioProjector,
)
from prismatic.training.train_utils import (
    compute_actions_l1_loss,
    compute_token_accuracy,
    get_current_action_mask,
    get_next_actions_mask,
)
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
)
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "/cpfs01/liuwei_workspace/models/finetune_im/openvla-7b+libero_spatial_no_noops+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--discrete_acts--proprio_state--100000_chkpt"             # Path to OpenVLA model (on HuggingFace Hub or stored locally)

    # Dataset
    # ManiSkill PickCube-v1 (Panda, pd_ee_delta_pose) RLDS build produced by
    # rlds_dataset_builder/maniskill_pickcube.  The directory layout is
    #   <data_root_dir>/<dataset_name>/<version>/...
    # so data_root_dir is the TFDS root, not the <dataset>/<version> subdir.
    
    # 160文件位置
    data_root_dir: Path = Path("/data/disk1/lcx_stu4/rlds")
    # 149文件位置
    #data_root_dir: Path = Path("/mnt/data2/lcx_stu4/maniskill/demos/rlds")      # Directory containing RLDS datasets
    dataset_name: str = "maniskill_pickcube"    # Name of fine-tuning dataset (e.g., `aloha_scoop_x_into_bowl`)
    run_root_dir: Path = Path("runs/imitation")                # Path to directory to store logs & checkpoints
    shuffle_buffer_size: int = 100_000               # Dataloader shuffle buffer size (can reduce if OOM errors occur)

    # Algorithm and architecture
    use_l1_regression: bool = False                   # If True, trains continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, trains continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = False                        # If True, includes robot proprioceptive state in input

    # Training configuration
    batch_size: int = 2                              # Batch size per device
    learning_rate: float = 5e-4                      # Learning rate
    lr_warmup_steps: int = 0                         # Number of steps to warm up learning rate (from 10% to 100%)
    num_steps_before_decay: int = 100_000            # Number of steps before LR decays by 10x
    grad_accumulation_steps: int = 8                 # Number of gradient accumulation steps
    max_steps: int = 200_000                         # Max number of training steps
    use_val_set: bool = False                        # If True, uses validation set and log validation metrics
    val_freq: int = 10_000                           # (When `use_val_set==True`) Validation set logging frequency in steps
    val_time_limit: int = 180                        # (When `use_val_set==True`) Time limit for computing validation metrics
    save_freq: int = 10                          # Checkpoint saving frequency in steps
    save_latest_checkpoint_only: bool = True        # If True, saves only 1 checkpoint, overwriting latest checkpoint
                                                     #   (If False, saves all checkpoints)
    enable_success_rate_checkpoints: bool = True    # If True, save milestone checkpoints when success rate crosses thresholds
    success_rate_save_thresholds: str = "0.6,0.7,0.8"  # Comma-separated success-rate milestones in [0, 1]
    success_rate_metric_name: str = "avg_success_rate"  # Primary eval metric name to monitor for milestone saves
    success_rate_metric_aliases: str = "avg_success_rate,global_success_rate,eval_success_rate"  # Fallback metric names
    success_rate_checkpoint_root_dir: Path = Path("/cpfs01/lcx_stu4_workspace/openvla_oft_rl/rl/policy_cpt")  # Root directory for milestone checkpoints

    # Real-environment LIBERO evaluation
    use_libero_env_eval: bool = False               # If True, run LIBERO rollout evaluation during training
    libero_eval_freq: int = 10_000                  # Frequency (in optimizer steps) for LIBERO rollout evaluation
    libero_eval_task_suite_name: str = "libero_spatial"  # LIBERO task suite to evaluate
    libero_eval_num_trials_per_task: int = 50       # Number of evaluation episodes per task
    libero_eval_max_tasks: Optional[int] = None     # Optional cap on number of tasks to evaluate
    libero_eval_num_workers: int = 10               # Number of environment worker processes for LIBERO evaluation
    libero_eval_num_steps_wait: int = 10            # Initial no-op steps to stabilize the scene
    libero_eval_num_open_loop_steps: int = NUM_ACTIONS_CHUNK  # Actions executed before re-querying the policy
    libero_eval_initial_states_path: str = "DEFAULT"  # DEFAULT or a path to initial states JSON
    libero_eval_env_img_res: int = 256              # Environment render resolution for LIBERO eval
    libero_eval_center_crop: bool = True            # Match eval preprocessing to image-aug training checkpoints
    libero_eval_unnorm_key: Optional[str] = None    # Optional override for action un-normalization stats key
    libero_eval_seed: int = 7                       # Random seed for LIBERO eval reproducibility

    # Real-environment ManiSkill evaluation (single-task PickCube-v1 by default)
    use_maniskill_env_eval: bool = False             # If True, run ManiSkill rollout evaluation during training
    maniskill_eval_freq: int = 200                   # Frequency (in optimizer steps) for ManiSkill rollout evaluation
    maniskill_eval_task_id: str = "PickCube-v1"      # Gym ID of the ManiSkill task to evaluate
    maniskill_eval_obs_mode: str = "rgbd"            # ManiSkill observation mode (must include sensor_data RGB)
    maniskill_eval_control_mode: str = "pd_ee_delta_pose"  # Action space matching the training data
    maniskill_eval_camera_name: str = "base_camera"  # Primary camera key under obs["sensor_data"][...] for policy image
    maniskill_eval_wrist_camera_name: str = "hand_camera"  # Wrist camera key under obs["sensor_data"][...]
    maniskill_eval_robot_uids: str = "panda_wristcam"  # Robot UID (use panda_wristcam for dual-camera eval)
    maniskill_eval_camera_res: int = 224             # Force camera resolution to match training distribution
    maniskill_eval_num_episodes: int = 50            # Total number of evaluation episodes
    maniskill_eval_num_envs: int = 10                # Number of vectorized envs run per batch
    maniskill_eval_max_steps: int = 200              # Per-episode env step budget
    maniskill_eval_num_open_loop_steps: int = NUM_ACTIONS_CHUNK  # Actions executed before re-querying the policy
    maniskill_eval_seed: int = 0                     # Base seed; per-env seed = base + ep_idx + i
    maniskill_eval_unnorm_key: Optional[str] = None  # Optional override for action un-normalization stats key
    maniskill_eval_language_instruction: str = "pick up the red cube and place it at the green target"  # Must match the RLDS builder
    maniskill_eval_sim_backend: str = "auto"         # ManiSkill sim backend: "auto" / "gpu" / "cpu"

    resume: bool = False                             # If True, resumes from checkpoint
    resume_step: Optional[int] = None                # (When `resume==True`) Step number that we are resuming from
    image_aug: bool = True                           # If True, trains with image augmentations (HIGHLY RECOMMENDED)
    diffusion_sample_freq: int = 50                  # (When `use_diffusion==True`) Frequency for sampling in steps

    # LoRA
    use_lora: bool = True                            # If True, uses LoRA fine-tuning
    lora_rank: int = 32                              # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                        # Dropout applied to LoRA weights
    merge_lora_during_training: bool = True          # If True, merges LoRA weights and saves result during training

    # Logging
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    run_id_override: Optional[str] = None            # Optional string to override the run ID with
    wandb_log_freq: int = 10                         # WandB logging frequency in steps

    # fmt: on


def remove_ddp_in_checkpoint(state_dict) -> dict:
    """
    Remove 'module.' prefix that may exist in checkpoints saved under DDP.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def get_run_id(cfg) -> str:
    if cfg.run_id_override is not None:
        run_id = cfg.run_id_override
    elif cfg.resume:
        run_id = cfg.vla_path.split("/")[-1]
        if "chkpt" in run_id.split("--")[-1]:
            run_id = "--".join(run_id.split("--")[:-1])
    else:
        run_id = (
            f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
            f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
            f"+lr-{cfg.learning_rate}"
        )
        if cfg.use_lora:
            run_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
        if cfg.image_aug:
            run_id += "--image_aug"
        if cfg.run_id_note is not None:
            run_id += f"--{cfg.run_id_note}"
    return run_id


def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)


def count_parameters(module: nn.Module, name: str) -> None:
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {num_params}")


def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: FinetuneConfig,
    device: torch.device,
    module_args: dict,
    to_bf16: bool = False,
) -> nn.Module:
    module = module_class(**module_args)
    count_parameters(module, module_name)

    if cfg.resume:
        state_dict = load_checkpoint(module_name, cfg.vla_path, cfg.resume_step, device="cpu")
        module.load_state_dict(state_dict)

    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device)

    return module


def _autocast_ctx(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def run_forward_pass(
    vla,
    action_head,
    noisy_action_projector,
    proprio_projector,
    batch,
    action_tokenizer,
    device: torch.device,
    use_l1_regression,
    use_diffusion,
    use_proprio,
    use_film,
    num_patches,
    compute_diffusion_l1=False,
    num_diffusion_steps_train=None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute model forward pass and metrics.
    """
    metrics = {}

    # Ground-truth actions
    ground_truth_actions = batch["actions"].to(device).to(torch.bfloat16)

    # [Only for diffusion] Noisy actions
    if use_diffusion:
        noisy_dict = action_head.sample_noisy_actions(ground_truth_actions)
        noise, noisy_actions, diffusion_timestep_embeddings = (
            noisy_dict["noise"],
            noisy_dict["noisy_actions"],
            noisy_dict["diffusion_timestep_embeddings"],
        )
    else:
        noise, noisy_actions, diffusion_timestep_embeddings = None, None, None

    # VLA forward pass
    with _autocast_ctx(device):
        output: CausalLMOutputWithPast = vla(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device),
            labels=batch["labels"],  # HF内部会处理 dtype/cast
            output_hidden_states=True,
            proprio=batch["proprio"] if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            noisy_actions=noisy_actions if use_diffusion else None,
            noisy_action_projector=noisy_action_projector if use_diffusion else None,
            diffusion_timestep_embeddings=diffusion_timestep_embeddings if use_diffusion else None,
            use_film=use_film,
        )

    # Get masks for logging
    ground_truth_token_ids = batch["labels"][:, 1:].to(device)
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

    # Discrete (next-token) vs continuous (L1/diffusion)
    if not (use_l1_regression or use_diffusion):
        loss = output.loss
        predicted_token_ids = output.logits[:, num_patches:-1].argmax(dim=2)
        curr_action_accuracy = compute_token_accuracy(
            predicted_token_ids, ground_truth_token_ids, mask=current_action_mask
        )
        curr_action_l1_loss = compute_actions_l1_loss(
            action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=current_action_mask
        )
        next_actions_accuracy = compute_token_accuracy(
            predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask
        )
        next_actions_l1_loss = compute_actions_l1_loss(
            action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask
        )
        metrics.update(
            {
                "loss_value": loss.item(),
                "curr_action_accuracy": curr_action_accuracy.item(),
                "curr_action_l1_loss": curr_action_l1_loss.item(),
                "next_actions_accuracy": next_actions_accuracy.item(),
                "next_actions_l1_loss": next_actions_l1_loss.item(),
            }
        )
    else:
        # Continuous action head path
        last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
        text_hidden_states = last_hidden_states[:, num_patches:-1]
        batch_size = batch["input_ids"].shape[0]
        actions_hidden_states = (
            text_hidden_states[current_action_mask | next_actions_mask]
            .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
            .to(torch.bfloat16)
        )  # (B, act_chunk_len, D)

        if use_l1_regression:
            predicted_actions = action_head.predict_action(actions_hidden_states)
            loss = torch.nn.L1Loss()(ground_truth_actions, predicted_actions)

        if use_diffusion:
            noise_pred = action_head.predict_noise(actions_hidden_states)
            noise_pred = noise_pred.reshape(noise.shape)
            loss = nn.functional.mse_loss(noise_pred, noise, reduction="mean")

            if compute_diffusion_l1:
                with torch.no_grad():
                    predicted_actions = run_diffusion_sampling(
                        vla=vla,
                        action_head=action_head,
                        noisy_action_projector=noisy_action_projector,
                        proprio_projector=proprio_projector,
                        batch=batch,
                        batch_size=batch_size,
                        num_patches=num_patches,
                        actions_shape=ground_truth_actions.shape,
                        device=device,
                        current_action_mask=current_action_mask,
                        next_actions_mask=next_actions_mask,
                        use_proprio=use_proprio,
                        use_film=use_film,
                    )

        metrics.update({"loss_value": loss.item()})

        should_log_l1_loss = not use_diffusion or (use_diffusion and compute_diffusion_l1)
        if should_log_l1_loss:
            ground_truth_curr_action = ground_truth_actions[:, 0]
            predicted_curr_action = predicted_actions[:, 0]
            ground_truth_next_actions = ground_truth_actions[:, 1:]
            predicted_next_actions = predicted_actions[:, 1:]
            curr_action_l1_loss = torch.nn.L1Loss()(ground_truth_curr_action, predicted_curr_action)
            next_actions_l1_loss = torch.nn.L1Loss()(ground_truth_next_actions, predicted_next_actions)
            metrics.update(
                {
                    "curr_action_l1_loss": curr_action_l1_loss.item(),
                    "next_actions_l1_loss": next_actions_l1_loss.item(),
                }
            )

    return loss, metrics


def run_diffusion_sampling(
    vla,
    action_head,
    noisy_action_projector,
    proprio_projector,
    batch,
    batch_size,
    num_patches,
    actions_shape,
    device: torch.device,
    current_action_mask,
    next_actions_mask,
    use_proprio,
    use_film,
) -> torch.Tensor:
    """
    Reverse diffusion to generate actions.
    """
    noise = torch.randn(
        size=(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM),
        device=device,
        dtype=torch.bfloat16,
    )

    action_head.noise_scheduler.set_timesteps(action_head.num_diffusion_steps_train)

    curr_noisy_actions = noise
    for t in action_head.noise_scheduler.timesteps:
        timesteps = torch.Tensor([t]).repeat(batch_size).to(device)
        diffusion_timestep_embeddings = action_head.time_encoder(timesteps).to(curr_noisy_actions.dtype)
        diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)

        with _autocast_ctx(device):
            output = vla(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device),
                labels=batch["labels"],
                output_hidden_states=True,
                proprio=batch["proprio"] if use_proprio else None,
                proprio_projector=proprio_projector if use_proprio else None,
                noisy_actions=curr_noisy_actions,
                noisy_action_projector=noisy_action_projector,
                diffusion_timestep_embeddings=diffusion_timestep_embeddings,
                use_film=use_film,
            )
            last_hidden_states = output.hidden_states[-1]
            text_hidden_states = last_hidden_states[:, num_patches:-1]
            actions_hidden_states = text_hidden_states[current_action_mask | next_actions_mask].reshape(
                batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1
            ).to(torch.bfloat16)
            noise_pred = action_head.predict_noise(actions_hidden_states)

        curr_noisy_actions = action_head.noise_scheduler.step(noise_pred, t, curr_noisy_actions).prev_sample

    return curr_noisy_actions.reshape(actions_shape)


def compute_smoothened_metrics(metrics_deques) -> dict:
    smoothened_metrics = {}
    for name, dq in metrics_deques.items():
        if dq and len(dq) > 0:
            smoothened_metrics[name] = sum(dq) / len(dq)
    return smoothened_metrics


def log_metrics_to_wandb(metrics, prefix, step, wandb_entity) -> None:
    log_dict = {}
    for name, value in metrics.items():
        if name == "loss_value":
            log_dict[f"{prefix}/Loss"] = value
        else:
            log_dict[f"{prefix}/{name.replace('_', ' ').title()}"] = value
    wandb_entity.log(log_dict, step=step)


def parse_success_rate_thresholds(thresholds: str) -> Tuple[float, ...]:
    parsed_thresholds = set()
    for raw_value in thresholds.split(","):
        raw_value = raw_value.strip()
        if not raw_value:
            continue
        threshold = float(raw_value)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Invalid success-rate milestone `{threshold}`; expected a value in [0, 1].")
        parsed_thresholds.add(threshold)
    return tuple(sorted(parsed_thresholds))


def resolve_metric_value(metrics: Dict[str, float], primary_name: str, aliases: str) -> Tuple[Optional[str], Optional[float]]:
    candidate_names = [primary_name]
    candidate_names.extend(alias.strip() for alias in aliases.split(",") if alias.strip())

    metrics_lower_map = {metric_name.lower(): metric_name for metric_name in metrics}
    for candidate_name in candidate_names:
        if candidate_name in metrics:
            return candidate_name, float(metrics[candidate_name])

        matched_metric_name = metrics_lower_map.get(candidate_name.lower())
        if matched_metric_name is not None:
            return matched_metric_name, float(metrics[matched_metric_name])

    return None, None


def get_vla_norm_stats(
    vla,
    preferred_norm_stats: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    candidate_modules = [vla]
    for attr_name in ("model", "base_model"):
        maybe_module = getattr(vla, attr_name, None)
        if maybe_module is not None:
            candidate_modules.append(maybe_module)
            nested_model = getattr(maybe_module, "model", None)
            if nested_model is not None:
                candidate_modules.append(nested_model)

    norm_stats = preferred_norm_stats
    if norm_stats is None:
        for module in candidate_modules:
            module_norm_stats = getattr(module, "norm_stats", None)
            if module_norm_stats:
                norm_stats = module_norm_stats
                break

    if not norm_stats:
        raise ValueError("Current model does not expose `norm_stats`, so LIBERO evaluation cannot resolve `unnorm_key`.")

    for module in candidate_modules:
        try:
            module.norm_stats = norm_stats
        except Exception:
            pass

    return norm_stats


def resolve_libero_eval_unnorm_key(
    cfg,
    vla,
    norm_stats: Optional[Dict[str, Dict]] = None,
    preferred_norm_stats: Optional[Dict[str, Dict]] = None,
) -> str:
    if cfg.libero_eval_unnorm_key is not None:
        return cfg.libero_eval_unnorm_key

    if norm_stats is None:
        norm_stats = get_vla_norm_stats(vla, preferred_norm_stats=preferred_norm_stats)

    candidate_keys = [cfg.libero_eval_task_suite_name, f"{cfg.libero_eval_task_suite_name}_no_noops", cfg.dataset_name]
    for candidate_key in candidate_keys:
        if candidate_key in norm_stats:
            return candidate_key

    raise ValueError(
        f"Could not resolve a LIBERO eval unnorm key for task suite `{cfg.libero_eval_task_suite_name}`. "
        f"Available keys: {sorted(norm_stats.keys())}"
    )


def resolve_maniskill_eval_unnorm_key(
    cfg,
    vla,
    norm_stats: Optional[Dict[str, Dict]] = None,
    preferred_norm_stats: Optional[Dict[str, Dict]] = None,
) -> str:
    if cfg.maniskill_eval_unnorm_key is not None:
        return cfg.maniskill_eval_unnorm_key

    if norm_stats is None:
        norm_stats = get_vla_norm_stats(vla, preferred_norm_stats=preferred_norm_stats)

    candidate_keys = [cfg.dataset_name, "maniskill_pickcube"]
    for candidate_key in candidate_keys:
        if candidate_key and candidate_key in norm_stats:
            return candidate_key

    raise ValueError(
        f"Could not resolve a ManiSkill eval unnorm key (tried `{cfg.dataset_name}`, `maniskill_pickcube`). "
        f"Available keys: {sorted(norm_stats.keys())}"
    )


def build_maniskill_eval_cfg(cfg, vla, proprio_projector, preferred_norm_stats=None) -> SimpleNamespace:
    """Build the SimpleNamespace consumed by ``get_vla_action_batch`` / ``get_action``."""
    norm_stats = get_vla_norm_stats(vla, preferred_norm_stats=preferred_norm_stats)
    unnorm_key = resolve_maniskill_eval_unnorm_key(
        cfg,
        vla,
        norm_stats=norm_stats,
        preferred_norm_stats=preferred_norm_stats,
    )
    use_proprio = cfg.use_proprio and proprio_projector is not None
    if use_proprio and "proprio" not in norm_stats.get(unnorm_key, {}):
        print(
            f"Warning: norm_stats for `{unnorm_key}` does not include proprio statistics. "
            "Disabling proprio for ManiSkill evaluation."
        )
        use_proprio = False

    return SimpleNamespace(
        model_family="openvla",
        center_crop=cfg.libero_eval_center_crop,
        num_images_in_input=cfg.num_images_in_input,
        use_proprio=use_proprio,
        unnorm_key=unnorm_key,
    )


def build_libero_eval_cfg(cfg, vla, proprio_projector, preferred_norm_stats=None) -> SimpleNamespace:
    norm_stats = get_vla_norm_stats(vla, preferred_norm_stats=preferred_norm_stats)
    unnorm_key = resolve_libero_eval_unnorm_key(
        cfg,
        vla,
        norm_stats=norm_stats,
        preferred_norm_stats=preferred_norm_stats,
    )
    use_proprio = cfg.use_proprio and proprio_projector is not None
    if use_proprio and "proprio" not in norm_stats.get(unnorm_key, {}):
        print(
            f"Warning: norm_stats for `{unnorm_key}` does not include proprio statistics. "
            "Disabling proprio for LIBERO evaluation."
        )
        use_proprio = False

    return SimpleNamespace(
        model_family="openvla",
        center_crop=cfg.libero_eval_center_crop,
        num_images_in_input=cfg.num_images_in_input,
        use_proprio=use_proprio,
        unnorm_key=unnorm_key,
    )


def _get_libero_task_max_steps(task_suite_name: str) -> int:
    task_max_steps = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    if task_suite_name not in task_max_steps:
        raise ValueError(f"Unsupported LIBERO task suite `{task_suite_name}` for training-time evaluation.")
    return task_max_steps[task_suite_name]


def _get_libero_eval_task_ids(cfg):
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.libero_eval_task_suite_name]()
    num_tasks = task_suite.n_tasks
    if cfg.libero_eval_max_tasks is not None and cfg.libero_eval_max_tasks > 0:
        num_tasks = min(num_tasks, cfg.libero_eval_max_tasks)
    return list(range(num_tasks))


def _safe_conn_send(conn, payload) -> bool:
    try:
        conn.send(payload)
        return True
    except (BrokenPipeError, EOFError, OSError):
        return False


def get_vla_action_batch(
    eval_cfg,
    vla,
    processor,
    observations,
    task_descriptions,
    proprio_projector=None,
    use_film=False,
):
    """
    Batched VLA inference for discrete action prediction.
    Processes multiple observations in a single GPU forward pass for much
    higher throughput than sequential single-observation calls.
    """
    import numpy as np
    from experiments.robot.openvla_utils import prepare_images_for_vla, normalize_proprio

    B = len(observations)
    if B == 0:
        return []

    device = next(vla.parameters()).device
    dtype = torch.bfloat16

    with torch.inference_mode():
        all_input_ids = []
        all_attention_masks = []
        all_pixel_values = []
        all_proprios = []

        for obs, task_label in zip(observations, task_descriptions):
            images = [obs["full_image"]]
            if eval_cfg.num_images_in_input > 1:
                images.extend([obs[k] for k in obs.keys() if "wrist" in k])
            images = prepare_images_for_vla(images, eval_cfg)
            primary = images.pop(0)

            prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
            inputs = processor(prompt, primary)

            if images:
                wrist_pvs = [processor(prompt, img)["pixel_values"] for img in images]
                inputs["pixel_values"] = torch.cat(
                    [inputs["pixel_values"]] + wrist_pvs, dim=1
                )

            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat(
                    [input_ids, torch.tensor([[29871]], dtype=input_ids.dtype)], dim=1
                )
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype)], dim=1
                )

            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            all_pixel_values.append(inputs["pixel_values"])

            if eval_cfg.use_proprio and proprio_projector is not None:
                proprio_norm_stats = vla.norm_stats[eval_cfg.unnorm_key]["proprio"]
                normalized = normalize_proprio(obs["state"], proprio_norm_stats)
                all_proprios.append(torch.tensor(normalized, dtype=dtype))
            else:
                all_proprios.append(None)

        # Right-pad all prompts to equal length so action token positions align
        max_len = max(ids.shape[1] for ids in all_input_ids)
        pad_token_id = processor.tokenizer.pad_token_id or 0

        padded_ids = []
        padded_masks = []
        for i in range(B):
            ids = all_input_ids[i]
            mask = all_attention_masks[i]
            pad_len = max_len - ids.shape[1]
            if pad_len > 0:
                ids = torch.cat(
                    [ids, torch.full((1, pad_len), pad_token_id, dtype=ids.dtype)], dim=1
                )
                mask = torch.cat(
                    [mask, torch.zeros((1, pad_len), dtype=mask.dtype)], dim=1
                )
            padded_ids.append(ids)
            padded_masks.append(mask)

        batch_input_ids = torch.cat(padded_ids, dim=0).to(device)
        batch_attention_mask = torch.cat(padded_masks, dim=0).to(device)
        batch_pixel_values = torch.cat(all_pixel_values, dim=0).to(device, dtype=dtype)

        NUM_PROMPT_TOKENS = batch_input_ids.shape[-1] - 1

        # Append placeholder action tokens + stop token (mirrors _prepare_input_for_action_prediction)
        placeholder = torch.ones(
            (B, ACTION_DIM * NUM_ACTIONS_CHUNK), device=device, dtype=batch_input_ids.dtype
        )
        batch_input_ids = torch.cat([batch_input_ids, placeholder], dim=-1)
        stop = torch.full((B, 1), STOP_INDEX, device=device, dtype=batch_input_ids.dtype)
        batch_input_ids = torch.cat([batch_input_ids, stop], dim=-1)

        mask_ext = torch.ones(
            (B, batch_input_ids.shape[-1] - batch_attention_mask.shape[-1]),
            device=device, dtype=batch_attention_mask.dtype,
        )
        batch_attention_mask = torch.cat([batch_attention_mask, mask_ext], dim=-1)

        # Build labels (mirrors _prepare_labels_for_action_prediction)
        labels = torch.full((B, max_len), IGNORE_INDEX, device=device, dtype=batch_input_ids.dtype)
        ARBIT = ACTION_TOKEN_BEGIN_IDX + 1
        labels_action = torch.full(
            (B, ACTION_DIM * NUM_ACTIONS_CHUNK), ARBIT, device=device, dtype=labels.dtype
        )
        labels_stop = torch.full((B, 1), STOP_INDEX, device=device, dtype=labels.dtype)
        labels = torch.cat([labels, labels_action, labels_stop], dim=-1)

        input_embeddings = vla.get_input_embeddings()(batch_input_ids)
        all_actions_mask = vla._process_action_masks(labels)

        language_embeddings = input_embeddings[~all_actions_mask].reshape(
            B, -1, input_embeddings.shape[2]
        )
        input_embeddings = input_embeddings * ~all_actions_mask.unsqueeze(-1)

        projected_patch_embeddings = vla._process_vision_features(
            batch_pixel_values, language_embeddings, use_film
        )

        use_proprio = eval_cfg.use_proprio and proprio_projector is not None and all_proprios[0] is not None
        if use_proprio:
            batch_proprio = torch.stack(all_proprios).to(device, dtype=dtype)
            projected_patch_embeddings = vla._process_proprio_features(
                projected_patch_embeddings, batch_proprio, proprio_projector
            )

        NUM_PATCHES = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input()
        if use_proprio:
            NUM_PATCHES += 1

        multimodal_embeddings, multimodal_attention_mask = vla._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, batch_attention_mask
        )

        output = vla.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        action_start = NUM_PATCHES + NUM_PROMPT_TOKENS
        action_end = action_start + ACTION_DIM * NUM_ACTIONS_CHUNK
        predicted_token_ids = output.logits[:, action_start:action_end].argmax(dim=2).cpu().numpy()

        discretized = vla.vocab_size - predicted_token_ids
        discretized = np.clip(discretized - 1, a_min=0, a_max=vla.bin_centers.shape[0] - 1)

        all_actions = []
        for b in range(B):
            normalized = vla.bin_centers[discretized[b]].reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
            actions = vla._unnormalize_actions(normalized, eval_cfg.unnorm_key)
            all_actions.append([actions[i] for i in range(len(actions))])

        return all_actions


def _run_libero_task_slice(task_ids, cfg, eval_cfg, resize_size, conn):
    import json
    import numpy as np

    from experiments.robot.libero.libero_utils import (
        get_libero_dummy_action,
        get_libero_env,
        get_libero_image,
        get_libero_wrist_image,
        quat2axisangle,
    )
    from experiments.robot.openvla_utils import resize_image_for_policy
    from experiments.robot.robot_utils import normalize_gripper_action, invert_gripper_action
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.libero_eval_task_suite_name]()
    task_max_steps = _get_libero_task_max_steps(cfg.libero_eval_task_suite_name)

    total_successes = 0
    total_episodes = 0
    per_task_metrics = {}

    try:
        for task_id in task_ids:
            task = task_suite.get_task(task_id)
            env, task_description = get_libero_env(task, "openvla", resolution=cfg.libero_eval_env_img_res)
            initial_states = task_suite.get_task_init_states(task_id)
            custom_initial_states = None
            if cfg.libero_eval_initial_states_path != "DEFAULT":
                with open(cfg.libero_eval_initial_states_path, "r") as file:
                    custom_initial_states = json.load(file)

            task_successes = 0
            task_episodes = 0
            max_trials = min(cfg.libero_eval_num_trials_per_task, len(initial_states))

            try:
                for episode_idx in range(max_trials):
                    if custom_initial_states is None:
                        initial_state = initial_states[episode_idx]
                    else:
                        task_key = task_description.replace(" ", "_")
                        episode_key = f"demo_{episode_idx}"
                        episode_info = custom_initial_states[task_key][episode_key]
                        if not episode_info["success"]:
                            continue
                        initial_state = np.array(episode_info["initial_state"])

                    env.reset()
                    obs = env.set_init_state(initial_state)
                    action_queue = deque(maxlen=cfg.libero_eval_num_open_loop_steps)
                    success = False
                    t = 0

                    while t < task_max_steps + cfg.libero_eval_num_steps_wait:
                        if t < cfg.libero_eval_num_steps_wait:
                            obs, _, done, _ = env.step(get_libero_dummy_action("openvla"))
                            t += 1
                            if done:
                                success = True
                                break
                            continue

                        if len(action_queue) == 0:
                            primary_image = get_libero_image(obs)
                            wrist_image = get_libero_wrist_image(obs)
                            observation = {
                                "full_image": resize_image_for_policy(primary_image, resize_size),
                                "wrist_image": resize_image_for_policy(wrist_image, resize_size),
                                "state": np.concatenate(
                                    (
                                        obs["robot0_eef_pos"],
                                        quat2axisangle(obs["robot0_eef_quat"]),
                                        obs["robot0_gripper_qpos"],
                                    )
                                ),
                            }
                            if not _safe_conn_send(
                                conn,
                                {
                                    "type": "action_request",
                                    "task_description": task_description,
                                    "observation": observation,
                                },
                            ):
                                return
                            try:
                                actions = conn.recv()
                            except (EOFError, BrokenPipeError, OSError):
                                print("LIBERO eval worker: parent connection lost, exiting.", flush=True)
                                return
                            action_queue.extend(actions)

                        action = np.asarray(action_queue.popleft()).copy()
                        action = normalize_gripper_action(action, binarize=True)
                        action = invert_gripper_action(action)
                        obs, _, done, _ = env.step(action.tolist())
                        t += 1
                        if done:
                            success = True
                            break

                    task_episodes += 1
                    total_episodes += 1
                    if success:
                        task_successes += 1
                        total_successes += 1
            finally:
                try:
                    env.close()
                except Exception:
                    pass

            task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0.0
            per_task_metrics[task_description] = {
                "episodes": task_episodes,
                "successes": task_successes,
                "success_rate": task_success_rate,
            }

        _safe_conn_send(
            conn,
            {
                "type": "result",
                "total_successes": total_successes,
                "total_episodes": total_episodes,
                "per_task_metrics": per_task_metrics,
            },
        )
    except Exception as exc:
        import traceback
        import sys

        tb_str = traceback.format_exc()
        print(f"LIBERO eval worker exception:\n{tb_str}", file=sys.stderr, flush=True)
        _safe_conn_send(
            conn,
            {
                "type": "error",
                "error": repr(exc),
                "traceback": tb_str,
            },
        )
    finally:
        conn.close()


def _run_libero_real_eval_parallel(
    cfg,
    eval_cfg,
    resize_size,
    vla,
    processor,
    action_head,
    proprio_projector,
    noisy_action_projector,
    log_step,
):
    import numpy as np

    from experiments.robot.robot_utils import get_action

    task_ids = _get_libero_eval_task_ids(cfg)
    if cfg.libero_eval_num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(
            f"Warning: libero_eval_num_open_loop_steps ({cfg.libero_eval_num_open_loop_steps}) does not match "
            f"NUM_ACTIONS_CHUNK ({NUM_ACTIONS_CHUNK})."
        )

    num_workers = max(1, min(cfg.libero_eval_num_workers, len(task_ids)))
    task_slices = [task_ids[i::num_workers] for i in range(num_workers)]
    task_slices = [task_slice for task_slice in task_slices if task_slice]
    ctx = mp.get_context("spawn")
    processes = []
    parent_conns = []
    total_successes = 0
    total_episodes = 0
    per_task_metrics = {}

    for task_slice in task_slices:
        parent_conn, child_conn = ctx.Pipe()
        process = ctx.Process(
            target=_run_libero_task_slice,
            args=(task_slice, cfg, eval_cfg, resize_size, child_conn),
        )
        process.start()
        child_conn.close()
        processes.append(process)
        parent_conns.append(parent_conn)

    use_batched_inference = action_head is None
    if use_batched_inference:
        print(f"LIBERO eval: using batched inference with {num_workers} workers")

    try:
        active_conns = list(parent_conns)
        failed_workers = []
        while active_conns:
            ready_conns = mp.connection.wait(active_conns, timeout=2.0)

            action_requests = []
            for conn in ready_conns:
                try:
                    message = conn.recv()
                except EOFError:
                    active_conns.remove(conn)
                    conn.close()
                    failed_workers.append(conn)
                    print("Warning: LIBERO eval worker closed unexpectedly (EOFError). Check worker stderr for details.")
                    continue
                message_type = message["type"]
                if message_type == "action_request":
                    action_requests.append((conn, message))
                elif message_type == "result":
                    total_successes += message["total_successes"]
                    total_episodes += message["total_episodes"]
                    per_task_metrics.update(message["per_task_metrics"])
                    active_conns.remove(conn)
                    conn.close()
                elif message_type == "error":
                    active_conns.remove(conn)
                    conn.close()
                    failed_workers.append(conn)
                    print(
                        "LIBERO eval worker failed with error:\n"
                        f"{message['error']}\n{message['traceback']}"
                    )
                else:
                    raise ValueError(f"Unknown worker message type: {message_type}")

            if action_requests:
                if use_batched_inference:
                    observations = [msg["observation"] for _, msg in action_requests]
                    task_descriptions = [msg["task_description"] for _, msg in action_requests]
                    batch_results = get_vla_action_batch(
                        eval_cfg=eval_cfg,
                        vla=vla,
                        processor=processor,
                        observations=observations,
                        task_descriptions=task_descriptions,
                        proprio_projector=proprio_projector,
                        use_film=cfg.use_film,
                    )
                    for (conn, _), actions in zip(action_requests, batch_results):
                        conn.send(actions)
                else:
                    for conn, msg in action_requests:
                        actions = get_action(
                            cfg=eval_cfg,
                            model=vla,
                            obs=msg["observation"],
                            task_label=msg["task_description"],
                            processor=processor,
                            action_head=action_head,
                            proprio_projector=proprio_projector,
                            noisy_action_projector=noisy_action_projector,
                            use_film=cfg.use_film,
                        )
                        conn.send(actions)

            for idx, proc in enumerate(processes):
                if not proc.is_alive() and proc.exitcode and proc.exitcode != 0:
                    if idx < len(parent_conns) and parent_conns[idx] in active_conns:
                        print(f"Warning: LIBERO eval worker PID {proc.pid} exited with code {proc.exitcode}")
                        try:
                            parent_conns[idx].close()
                        except Exception:
                            pass
                        active_conns = [c for c in active_conns if c is not parent_conns[idx]]
            if not active_conns:
                break
    finally:
        for conn in parent_conns:
            try:
                conn.close()
            except Exception:
                pass
        for process in processes:
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)

    for proc in processes:
        if proc.exitcode and proc.exitcode != 0:
            print(f"Warning: LIBERO eval worker PID {proc.pid} exited with code {proc.exitcode}")

    return total_successes, total_episodes, per_task_metrics


def _run_libero_real_eval_sequential(
    cfg,
    eval_cfg,
    resize_size,
    vla,
    processor,
    action_head,
    proprio_projector,
    noisy_action_projector,
    log_step,
):
    import json
    import numpy as np
    from libero.libero import benchmark

    from experiments.robot.libero.libero_utils import (
        get_libero_dummy_action,
        get_libero_env,
        get_libero_image,
        get_libero_wrist_image,
        quat2axisangle,
    )
    from experiments.robot.openvla_utils import resize_image_for_policy
    from experiments.robot.robot_utils import get_action, invert_gripper_action, normalize_gripper_action

    task_max_steps = _get_libero_task_max_steps(cfg.libero_eval_task_suite_name)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.libero_eval_task_suite_name]()
    num_tasks = task_suite.n_tasks
    if cfg.libero_eval_num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(
            f"Warning: libero_eval_num_open_loop_steps ({cfg.libero_eval_num_open_loop_steps}) does not match "
            f"NUM_ACTIONS_CHUNK ({NUM_ACTIONS_CHUNK})."
        )
    if cfg.libero_eval_max_tasks is not None and cfg.libero_eval_max_tasks > 0:
        num_tasks = min(num_tasks, cfg.libero_eval_max_tasks)

    per_task_metrics = {}
    total_successes = 0
    total_episodes = 0

    for task_id in tqdm.tqdm(range(num_tasks), desc=f"LIBERO eval @ {log_step}", leave=False):
        task = task_suite.get_task(task_id)
        env, task_description = get_libero_env(task, "openvla", resolution=cfg.libero_eval_env_img_res)
        initial_states = task_suite.get_task_init_states(task_id)
        custom_initial_states = None
        if cfg.libero_eval_initial_states_path != "DEFAULT":
            with open(cfg.libero_eval_initial_states_path, "r") as file:
                custom_initial_states = json.load(file)

        task_successes = 0
        task_episodes = 0
        max_trials = min(cfg.libero_eval_num_trials_per_task, len(initial_states))

        try:
            for episode_idx in range(max_trials):
                if custom_initial_states is None:
                    initial_state = initial_states[episode_idx]
                else:
                    task_key = task_description.replace(" ", "_")
                    episode_key = f"demo_{episode_idx}"
                    episode_info = custom_initial_states[task_key][episode_key]
                    if not episode_info["success"]:
                        continue
                    initial_state = np.array(episode_info["initial_state"])

                env.reset()
                obs = env.set_init_state(initial_state)
                action_queue = deque(maxlen=cfg.libero_eval_num_open_loop_steps)
                success = False
                t = 0

                try:
                    while t < task_max_steps + cfg.libero_eval_num_steps_wait:
                        if t < cfg.libero_eval_num_steps_wait:
                            obs, _, done, _ = env.step(get_libero_dummy_action("openvla"))
                            t += 1
                            continue

                        if len(action_queue) == 0:
                            primary_image = get_libero_image(obs)
                            wrist_image = get_libero_wrist_image(obs)
                            observation = {
                                "full_image": resize_image_for_policy(primary_image, resize_size),
                                "wrist_image": resize_image_for_policy(wrist_image, resize_size),
                                "state": np.concatenate(
                                    (
                                        obs["robot0_eef_pos"],
                                        quat2axisangle(obs["robot0_eef_quat"]),
                                        obs["robot0_gripper_qpos"],
                                    )
                                ),
                            }
                            actions = get_action(
                                cfg=eval_cfg,
                                model=vla,
                                obs=observation,
                                task_label=task_description,
                                processor=processor,
                                action_head=action_head,
                                proprio_projector=proprio_projector,
                                noisy_action_projector=noisy_action_projector,
                                use_film=cfg.use_film,
                            )
                            action_queue.extend(actions)

                        action = action_queue.popleft()
                        action = normalize_gripper_action(action, binarize=True)
                        action = invert_gripper_action(action)

                        obs, _, done, _ = env.step(action.tolist())
                        if done:
                            success = True
                            break
                        t += 1
                except Exception as e:
                    import traceback
                    print(f"LIBERO eval episode error (task={task_id}, ep={episode_idx}): {e}")
                    traceback.print_exc()

                task_episodes += 1
                total_episodes += 1
                if success:
                    task_successes += 1
                    total_successes += 1
        finally:
            try:
                env.close()
            except Exception:
                pass

        task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0.0
        per_task_metrics[task_description] = {
            "episodes": task_episodes,
            "successes": task_successes,
            "success_rate": task_success_rate,
        }

    return total_successes, total_episodes, per_task_metrics


def run_libero_real_eval(
    cfg,
    vla,
    processor,
    action_head,
    proprio_projector,
    noisy_action_projector,
    train_dataset_statistics,
    log_step,
    run_dir,
    writer: Optional[SummaryWriter] = None,
) -> Dict[str, float]:
    from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere

    module_modes = []
    for module in (vla, action_head, proprio_projector, noisy_action_projector):
        if module is not None:
            module_modes.append((module, module.training))
            module.eval()

    eval_log_dir = run_dir / "libero_eval"
    os.makedirs(eval_log_dir, exist_ok=True)
    log_path = eval_log_dir / f"step_{log_step}.log"

    try:
        set_seed_everywhere(cfg.libero_eval_seed)
        eval_cfg = build_libero_eval_cfg(
            cfg,
            vla,
            proprio_projector,
            preferred_norm_stats=train_dataset_statistics,
        )
        resize_size = get_image_resize_size(eval_cfg)

        if cfg.libero_eval_num_workers > 1:
            total_successes, total_episodes, per_task_metrics = _run_libero_real_eval_parallel(
                cfg=cfg,
                eval_cfg=eval_cfg,
                resize_size=resize_size,
                vla=vla,
                processor=processor,
                action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                log_step=log_step,
            )
        else:
            total_successes, total_episodes, per_task_metrics = _run_libero_real_eval_sequential(
                cfg=cfg,
                eval_cfg=eval_cfg,
                resize_size=resize_size,
                vla=vla,
                processor=processor,
                action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                log_step=log_step,
            )

        with open(log_path, "w") as log_file:
            log_file.write(f"step={log_step}\n")
            log_file.write(f"task_suite={cfg.libero_eval_task_suite_name}\n")
            log_file.write(f"num_trials_per_task={cfg.libero_eval_num_trials_per_task}\n")
            log_file.write(f"num_workers={cfg.libero_eval_num_workers}\n")
            for task_description, task_stats in per_task_metrics.items():
                log_file.write(
                    f"task={task_description} episodes={task_stats['episodes']} successes={task_stats['successes']} "
                    f"success_rate={task_stats['success_rate']:.4f}\n"
                )

        success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
        avg_success_rate = (
            sum(task_stats["success_rate"] for task_stats in per_task_metrics.values()) / float(len(per_task_metrics))
            if per_task_metrics
            else 0.0
        )
        eval_metrics = {
            "success_rate": success_rate,
            "avg_success_rate": avg_success_rate,
            "global_success_rate": success_rate,
            "libero_total_episodes": float(total_episodes),
            "libero_total_successes": float(total_successes),
            "libero_num_tasks": float(len(per_task_metrics)),
        }

        if writer is not None:
            writer.add_scalar("libero_eval/success_rate", success_rate, log_step)
            writer.add_scalar("libero_eval/avg_success_rate", avg_success_rate, log_step)
            writer.add_scalar("libero_eval/total_episodes", total_episodes, log_step)
            writer.add_scalar("libero_eval/total_successes", total_successes, log_step)
            for task_description, task_stats in per_task_metrics.items():
                sanitized_task_description = task_description.replace(" ", "_").replace("/", "_")
                writer.add_scalar(f"libero_eval/tasks/{sanitized_task_description}", task_stats["success_rate"], log_step)

        wandb.log(
            {
                "LIBERO Eval/Success Rate": success_rate,
                "LIBERO Eval/Average Task Success Rate": avg_success_rate,
                "LIBERO Eval/Total Episodes": total_episodes,
                "LIBERO Eval/Total Successes": total_successes,
            },
            step=log_step,
        )
        print(
            f"LIBERO real-env eval at step {log_step}: avg_success_rate={avg_success_rate:.4f}, "
            f"global_success_rate={success_rate:.4f}, successes={total_successes}/{total_episodes}, "
            f"workers={cfg.libero_eval_num_workers}"
        )
        return eval_metrics
    finally:
        for module, was_training in module_modes:
            module.train(was_training)


def run_maniskill_real_eval(
    cfg,
    vla,
    processor,
    action_head,
    proprio_projector,
    noisy_action_projector,
    train_dataset_statistics,
    log_step,
    run_dir,
    writer: Optional[SummaryWriter] = None,
) -> Dict[str, float]:
    """ManiSkill (PickCube-v1 by default) rollout evaluator.

    Mirrors ``run_libero_real_eval`` so that the returned metric dict slots into
    ``maybe_save_success_rate_checkpoints`` without any code changes downstream.
    """
    import numpy as np

    from experiments.robot.maniskill.maniskill_utils import (
        build_maniskill_env,
        clip_maniskill_action,
        extract_done_mask,
        extract_maniskill_observation,
        extract_success_mask,
        seeds_for_batch,
    )
    from experiments.robot.robot_utils import set_seed_everywhere

    module_modes = []
    for module in (vla, action_head, proprio_projector, noisy_action_projector):
        if module is not None:
            module_modes.append((module, module.training))
            module.eval()

    eval_log_dir = run_dir / "maniskill_eval"
    os.makedirs(eval_log_dir, exist_ok=True)
    log_path = eval_log_dir / f"step_{log_step}.log"

    try:
        set_seed_everywhere(cfg.maniskill_eval_seed)

        eval_cfg = build_maniskill_eval_cfg(
            cfg,
            vla,
            proprio_projector,
            preferred_norm_stats=train_dataset_statistics,
        )

        if action_head is not None:
            print(
                "Warning: ManiSkill eval currently uses the discrete-action batched path "
                "(`get_vla_action_batch`); a non-None `action_head` is being ignored."
            )

        total_episodes_target = int(cfg.maniskill_eval_num_episodes)
        if total_episodes_target <= 0:
            raise ValueError(f"maniskill_eval_num_episodes must be positive, got {total_episodes_target}")
        num_envs = int(min(cfg.maniskill_eval_num_envs, total_episodes_target))
        if num_envs <= 0:
            raise ValueError(f"maniskill_eval_num_envs must be positive, got {cfg.maniskill_eval_num_envs}")

        env = build_maniskill_env(
            task_id=cfg.maniskill_eval_task_id,
            num_envs=num_envs,
            obs_mode=cfg.maniskill_eval_obs_mode,
            control_mode=cfg.maniskill_eval_control_mode,
            camera_name=cfg.maniskill_eval_camera_name,
            wrist_camera_name=(
                cfg.maniskill_eval_wrist_camera_name
                if eval_cfg.num_images_in_input > 1
                else None
            ),
            camera_res=cfg.maniskill_eval_camera_res,
            max_episode_steps=cfg.maniskill_eval_max_steps,
            sim_backend=cfg.maniskill_eval_sim_backend,
            robot_uids=(
                cfg.maniskill_eval_robot_uids
                if eval_cfg.num_images_in_input > 1
                else None
            ),
        )
        if cfg.maniskill_eval_num_open_loop_steps != NUM_ACTIONS_CHUNK:
            print(
                f"Warning: maniskill_eval_num_open_loop_steps ({cfg.maniskill_eval_num_open_loop_steps}) "
                f"does not match NUM_ACTIONS_CHUNK ({NUM_ACTIONS_CHUNK})."
            )

        total_successes = 0
        total_episodes = 0
        per_batch_metrics = []
        ep_idx = 0

        try:
            while ep_idx < total_episodes_target:
                this_batch = min(num_envs, total_episodes_target - ep_idx)
                # The vectorized env was built with `num_envs` envs; if the
                # tail batch is smaller we still step all envs but only count
                # the first `this_batch` for stats.
                batch_seeds = list(seeds_for_batch(cfg.maniskill_eval_seed, ep_idx, num_envs))

                obs, _ = env.reset(seed=batch_seeds)

                succeeded = np.zeros(num_envs, dtype=bool)
                finished = np.zeros(num_envs, dtype=bool)
                action_queues = [deque(maxlen=cfg.maniskill_eval_num_open_loop_steps) for _ in range(num_envs)]

                for _ in range(cfg.maniskill_eval_max_steps):
                    if all(len(q) == 0 for q in action_queues):
                        observations = [
                            extract_maniskill_observation(
                                obs,
                                env_idx=i,
                                camera_name=cfg.maniskill_eval_camera_name,
                                use_proprio=eval_cfg.use_proprio,
                                wrist_camera_name=cfg.maniskill_eval_wrist_camera_name,
                                include_wrist_image=eval_cfg.num_images_in_input > 1,
                            )
                            for i in range(num_envs)
                        ]
                        task_descs = [cfg.maniskill_eval_language_instruction] * num_envs
                        batch_actions = get_vla_action_batch(
                            eval_cfg=eval_cfg,
                            vla=vla,
                            processor=processor,
                            observations=observations,
                            task_descriptions=task_descs,
                            proprio_projector=proprio_projector if eval_cfg.use_proprio else None,
                            use_film=cfg.use_film,
                        )
                        for i in range(num_envs):
                            action_queues[i].extend(batch_actions[i])

                    step_action = np.stack(
                        [np.asarray(action_queues[i].popleft(), dtype=np.float32) for i in range(num_envs)],
                        axis=0,
                    )
                    step_action = clip_maniskill_action(step_action)

                    obs, _reward, terminated, truncated, info = env.step(step_action)

                    succ_mask = extract_success_mask(info, num_envs)
                    new_success = succ_mask & ~finished
                    succeeded |= new_success
                    done_mask = extract_done_mask(terminated, truncated, num_envs)
                    finished |= new_success | done_mask

                    if finished.all():
                        break

                batch_successes = int(succeeded[:this_batch].sum())
                total_successes += batch_successes
                total_episodes += this_batch
                per_batch_metrics.append(
                    {
                        "ep_start": ep_idx,
                        "ep_end": ep_idx + this_batch,
                        "successes": batch_successes,
                        "episodes": this_batch,
                        "seeds": batch_seeds[:this_batch],
                    }
                )
                ep_idx += this_batch
        finally:
            try:
                env.close()
            except Exception:
                pass

        success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
        avg_success_rate = success_rate  # single-task eval: per-task average == global

        with open(log_path, "w") as log_file:
            log_file.write(f"step={log_step}\n")
            log_file.write(f"task_id={cfg.maniskill_eval_task_id}\n")
            log_file.write(f"control_mode={cfg.maniskill_eval_control_mode}\n")
            log_file.write(f"obs_mode={cfg.maniskill_eval_obs_mode}\n")
            log_file.write(f"camera={cfg.maniskill_eval_camera_name}@{cfg.maniskill_eval_camera_res}\n")
            log_file.write(f"num_envs={num_envs}\n")
            log_file.write(f"num_episodes={total_episodes}\n")
            log_file.write(f"max_steps={cfg.maniskill_eval_max_steps}\n")
            log_file.write(f"open_loop_steps={cfg.maniskill_eval_num_open_loop_steps}\n")
            log_file.write(f"successes={total_successes}\n")
            log_file.write(f"success_rate={success_rate:.4f}\n")
            for batch_stats in per_batch_metrics:
                log_file.write(
                    "batch ep=[{ep_start},{ep_end}) successes={successes}/{episodes} "
                    "seeds={seeds}\n".format(**batch_stats)
                )

        eval_metrics = {
            "success_rate": success_rate,
            "avg_success_rate": avg_success_rate,
            "global_success_rate": success_rate,
            "maniskill_total_episodes": float(total_episodes),
            "maniskill_total_successes": float(total_successes),
            "maniskill_num_envs": float(num_envs),
        }

        if writer is not None:
            writer.add_scalar("maniskill_eval/success_rate", success_rate, log_step)
            writer.add_scalar("maniskill_eval/avg_success_rate", avg_success_rate, log_step)
            writer.add_scalar("maniskill_eval/total_episodes", total_episodes, log_step)
            writer.add_scalar("maniskill_eval/total_successes", total_successes, log_step)

        wandb.log(
            {
                "ManiSkill Eval/Success Rate": success_rate,
                "ManiSkill Eval/Average Task Success Rate": avg_success_rate,
                "ManiSkill Eval/Total Episodes": total_episodes,
                "ManiSkill Eval/Total Successes": total_successes,
            },
            step=log_step,
        )
        print(
            f"ManiSkill real-env eval at step {log_step}: success_rate={success_rate:.4f}, "
            f"successes={total_successes}/{total_episodes}, "
            f"task={cfg.maniskill_eval_task_id}, num_envs={num_envs}"
        )
        return eval_metrics
    finally:
        for module, was_training in module_modes:
            module.train(was_training)


def _ensure_cached_module_files_writable(obj) -> None:
    """
    HuggingFace caches dynamic module files (e.g. processing_prismatic.py) from the
    model directory. If the source files are read-only, the cached copies inherit those
    permissions. When processor.save_pretrained() is called, custom_object_save() uses
    shutil.copy which preserves permissions.  If multiple auto-mapped classes share the
    same module file, the second copy attempt fails with PermissionError because the
    destination was already written as read-only by the first copy.

    This helper makes the cached source files writable so that shutil.copy produces
    writable destinations, preventing the PermissionError cascade.
    """
    import sys as _sys
    import stat

    seen = set()
    objects_to_check = [obj]
    if hasattr(obj, "attributes"):
        for attr_name in getattr(obj, "attributes", []):
            attr = getattr(obj, attr_name, None)
            if attr is not None:
                objects_to_check.append(attr)
    if hasattr(obj, "config"):
        objects_to_check.append(obj.config)

    for o in objects_to_check:
        module_name = type(o).__module__
        module = _sys.modules.get(module_name)
        if module is None or not hasattr(module, "__file__") or module.__file__ is None:
            continue
        fpath = Path(module.__file__)
        if fpath in seen or not fpath.exists():
            continue
        seen.add(fpath)
        current_mode = fpath.stat().st_mode
        if not (current_mode & stat.S_IWUSR):
            try:
                fpath.chmod(current_mode | stat.S_IWUSR | stat.S_IWGRP)
            except OSError:
                pass


def save_training_checkpoint(
    cfg,
    run_dir,
    log_step,
    vla,
    processor,
    proprio_projector,
    noisy_action_projector,
    action_head,
    train_dataset,
    is_main_process: bool,
) -> None:
    """
    Save all training checkpoints including model components, LoRA adapter, and dataset statistics.
    """
    if cfg.save_latest_checkpoint_only:
        checkpoint_dir = run_dir
        checkpoint_name_suffix = "latest_checkpoint.pt"
    else:
        checkpoint_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
        checkpoint_name_suffix = f"{log_step}_checkpoint.pt"

    adapter_dir = checkpoint_dir / "lora_adapter"

    if is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(adapter_dir, exist_ok=True)
        save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
        print(f"Saving Model Checkpoint for Step {log_step}")

        # Save processor and LoRA adapter
        processor.save_pretrained(checkpoint_dir)
        # Save LoRA adapter weights
        vla.save_pretrained(adapter_dir)

        # Save other components
        if cfg.use_proprio and proprio_projector is not None:
            torch.save(proprio_projector.state_dict(), checkpoint_dir / f"proprio_projector--{checkpoint_name_suffix}")

        if cfg.use_diffusion and noisy_action_projector is not None:
            torch.save(
                noisy_action_projector.state_dict(), checkpoint_dir / f"noisy_action_projector--{checkpoint_name_suffix}"
            )

        if (cfg.use_l1_regression or cfg.use_diffusion) and action_head is not None:
            torch.save(action_head.state_dict(), checkpoint_dir / f"action_head--{checkpoint_name_suffix}")

        if cfg.use_film:
            # Save the entire vision backbone (not just FiLM components)
            base_model = vla.model if hasattr(vla, "model") else vla
            torch.save(
                base_model.vision_backbone.state_dict(), checkpoint_dir / f"vision_backbone--{checkpoint_name_suffix}"
            )

    # Merge LoRA weights into base model and save resulting model checkpoint
    if cfg.use_lora and cfg.merge_lora_during_training and is_main_process:
        base_vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
        )
        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
        merged_vla = merged_vla.merge_and_unload()
        merged_path = checkpoint_dir / "merged_model"
        os.makedirs(merged_path, exist_ok=True, mode=0o755)
        merged_vla.save_pretrained(merged_path)
        print(f"Saved merged model for Step {log_step} at: {merged_path}")


def run_validation(
    vla,
    action_head,
    noisy_action_projector,
    proprio_projector,
    val_dataloader,
    action_tokenizer,
    device: torch.device,
    cfg,
    num_patches,
    log_step,
    is_main_process: bool,
    val_time_limit: int,
    writer: Optional[SummaryWriter] = None,
) -> None:
    """
    Compute validation set metrics for logging.
    """
    val_start_time = time.time()
    vla.eval()
    val_batches_count = 0
    all_val_metrics = []

    with torch.no_grad():
        for batch in val_dataloader:
            _, metrics = run_forward_pass(
                vla=vla,
                action_head=action_head,
                noisy_action_projector=noisy_action_projector,
                proprio_projector=proprio_projector,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device=device,
                use_l1_regression=cfg.use_l1_regression,
                use_diffusion=cfg.use_diffusion,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=num_patches,
                compute_diffusion_l1=True,
                num_diffusion_steps_train=cfg.num_diffusion_steps_train if cfg.use_diffusion else None,
            )

            metrics["loss"] = metrics["loss_value"]
            all_val_metrics.append(metrics)
            val_batches_count += 1

            if time.time() - val_start_time > val_time_limit:
                break

    avg_val_metrics = {}
    if all_val_metrics:
        for metric_name in all_val_metrics[0].keys():
            values = [m[metric_name] for m in all_val_metrics if metric_name in m]
            if values:
                avg_val_metrics[metric_name] = sum(values) / len(values)

    avg_val_metrics["val_batches_count"] = val_batches_count

    if is_main_process:
        log_metrics_to_wandb(avg_val_metrics, "VLA Val", log_step, wandb)
        # Log to TensorBoard
        if writer is not None:
            for metric_name, value in avg_val_metrics.items():
                writer.add_scalar(f"val/{metric_name}", value, log_step)


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    """
    Fine-tune base VLA on demonstration dataset via LoRA (No DDP).
    """
    assert cfg.use_lora, "Only LoRA fine-tuning is supported. Please set --use_lora=True!"
    assert not (cfg.use_l1_regression and cfg.use_diffusion), (
        "Cannot do both L1 regression and diffusion. Please pick one of them!"
    )

    # Trim trailing slash
    cfg.vla_path = cfg.vla_path.rstrip("/")
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # Run ID and dirs
    run_id = get_run_id(cfg)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.run_root_dir / f"{timestamp}_{run_id}"
    os.makedirs(run_dir, exist_ok=True)

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
    is_main_process = True  # 单进程调试

    # W&B
    if is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{run_id}")

    writer = SummaryWriter(log_dir=run_dir) if is_main_process else None
    print(f"TensorBoard logs will be saved to: {run_dir}")
    # Constants
    print(
        "Detected constants:\n"
        f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n"
        f"\tACTION_DIM: {ACTION_DIM}\n"
        f"\tPROPRIO_DIM: {PROPRIO_DIM}\n"
        f"\tACTION_PROPRIO_NORMALIZATION_TYPE: {ACTION_PROPRIO_NORMALIZATION_TYPE}"
    )

    # Model path (HF hub/local)
    if model_is_on_hf_hub(cfg.vla_path):
        vla_download_path = snapshot_download(repo_id=cfg.vla_path)
        cfg.vla_path = vla_download_path
    else:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Update config.json and sync model files (single process)
    # if is_main_process:
        # update_auto_map(cfg.vla_path)
        # check_model_logic_mismatch(cfg.vla_path)

    # Load processor & model
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)

    # Set number of images in VLA input (before LoRA wrapping)
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    # LoRA
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # FiLM
    if cfg.use_film:
        # 注意：LoRA 包裹时需通过 vla.model 访问底层模型
        base_model = vla.model if hasattr(vla, "model") else vla
        count_parameters(base_model.vision_backbone, "vla.vision_backbone (original)")
        base_model.vision_backbone = FiLMedPrismaticVisionBackbone(
            vision_backbone=base_model.vision_backbone,
            llm_dim=base_model.llm_dim,
        )
        count_parameters(base_model.vision_backbone, "vla.vision_backbone (post-wrap)")
        if cfg.resume:
            state_dict = load_checkpoint("vision_backbone", cfg.vla_path, cfg.resume_step)
            base_model.vision_backbone.load_state_dict(state_dict)
        base_model.vision_backbone = base_model.vision_backbone.to(device)

    # If applicable, instantiate projectors / heads
    base_model = vla.model if hasattr(vla, "model") else vla
    proprio_projector = None
    noisy_action_projector = None
    action_head = None

    if cfg.use_proprio:
        proprio_projector = init_module(
            ProprioProjector,
            "proprio_projector",
            cfg,
            device,
            {"llm_dim": base_model.llm_dim, "proprio_dim": PROPRIO_DIM},
        )

    if cfg.use_l1_regression:
        action_head = init_module(
            L1RegressionActionHead,
            "action_head",
            cfg,
            device,
            {"input_dim": base_model.llm_dim, "hidden_dim": base_model.llm_dim, "action_dim": ACTION_DIM},
            to_bf16=(device.type == "cuda"),
        )

    if cfg.use_diffusion:
        action_head = init_module(
            DiffusionActionHead,
            "action_head",
            cfg,
            device,
            {
                "input_dim": base_model.llm_dim,
                "hidden_dim": base_model.llm_dim,
                "action_dim": ACTION_DIM,
                "num_diffusion_steps_train": cfg.num_diffusion_steps_train,
            },
            to_bf16=(device.type == "cuda"),
        )
        noisy_action_projector = init_module(
            NoisyActionProjector, "noisy_action_projector", cfg, device, {"llm_dim": base_model.llm_dim}
        )

    # Number of vision patches
    backbone = base_model.vision_backbone
    NUM_PATCHES = backbone.get_num_patches() * backbone.get_num_images_in_input()
    if cfg.use_proprio:
        NUM_PATCHES += 1
    if cfg.use_diffusion:
        NUM_PATCHES += 1

    # Optimizer
    trainable_params = [p for p in vla.parameters() if p.requires_grad]
    if cfg.use_l1_regression or cfg.use_diffusion:
        trainable_params += [p for p in action_head.parameters() if p.requires_grad]
    if cfg.use_diffusion:
        trainable_params += [p for p in (noisy_action_projector.parameters()) if p.requires_grad]
    if cfg.use_proprio:
        trainable_params += [p for p in (proprio_projector.parameters()) if p.requires_grad]
    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # LR scheduler
    original_lr = optimizer.param_groups[0]["lr"]
    scheduler = MultiStepLR(
        optimizer,
        milestones=[cfg.num_steps_before_decay],
        gamma=0.1,
    )

    # Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Dataset(s)
    use_wrist_image = cfg.num_images_in_input > 1
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio,
    )
    train_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(base_model.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
    if cfg.use_val_set:
        val_dataset = RLDSDataset(
            cfg.data_root_dir,
            cfg.dataset_name,
            batch_transform,
            resize_resolution=tuple(base_model.config.image_sizes),
            shuffle_buffer_size=cfg.shuffle_buffer_size // 10,
            image_aug=cfg.image_aug,
            train=False,
        )

    if is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    # DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # RLDS 自带并行，调试时 0 更好断点
        pin_memory=(device.type == "cuda"),
    )
    if cfg.use_val_set:
        val_batch_size = cfg.batch_size
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

    # Recent metrics deques
    recent_metrics = {
        "loss_value": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
    }

    # Training loop
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            compute_diffusion_l1 = cfg.use_diffusion and batch_idx % cfg.diffusion_sample_freq == 0
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(vla.device)
            loss, metrics = run_forward_pass(
                vla=vla,
                action_head=action_head,
                noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                proprio_projector=proprio_projector if cfg.use_proprio else None,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device=device,
                use_l1_regression=cfg.use_l1_regression,
                use_diffusion=cfg.use_diffusion,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=NUM_PATCHES,
                compute_diffusion_l1=compute_diffusion_l1,
                num_diffusion_steps_train=cfg.num_diffusion_steps_train if cfg.use_diffusion else None,
            )

            normalized_loss = loss / cfg.grad_accumulation_steps
            normalized_loss.backward()

            for metric_name, value in metrics.items():
                if metric_name in recent_metrics:
                    recent_metrics[metric_name].append(value)

            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
            smoothened_metrics = compute_smoothened_metrics(recent_metrics)
            log_step = gradient_step_idx if not cfg.resume else cfg.resume_step + gradient_step_idx

            if is_main_process and log_step % cfg.wandb_log_freq == 0:
                log_metrics_to_wandb(smoothened_metrics, "VLA Train", log_step, wandb)
                # Log to TensorBoard
                if writer is not None:
                    for metric_name, value in smoothened_metrics.items():
                        writer.add_scalar(f"train/{metric_name}", value, log_step)

            # LR warmup (optional)
            if cfg.lr_warmup_steps > 0:
                lr_progress = min((gradient_step_idx + 1) / cfg.lr_warmup_steps, 1.0)
                current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                for pg in optimizer.param_groups:
                    pg["lr"] = current_lr

            if is_main_process and log_step % cfg.wandb_log_freq == 0:
                wandb.log({"VLA Train/Learning Rate": scheduler.get_last_lr()[0]}, step=log_step)
                # Log learning rate to TensorBoard
                if writer is not None:
                    writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], log_step)

            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress.update()

            # Save checkpoint
            if gradient_step_idx > 0 and log_step % cfg.save_freq == 0:
                save_training_checkpoint(
                    cfg=cfg,
                    run_dir=run_dir,
                    log_step=log_step,
                    vla=vla,
                    processor=processor,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                    action_head=action_head if (cfg.use_l1_regression or cfg.use_diffusion) else None,
                    train_dataset=train_dataset,
                    is_main_process=is_main_process,
                )

            # Validation
            if cfg.use_val_set and log_step > 0 and log_step % cfg.val_freq == 0:
                run_validation(
                    vla=vla,
                    action_head=action_head,
                    noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    val_dataloader=val_dataloader,
                    action_tokenizer=action_tokenizer,
                    device=device,
                    cfg=cfg,
                    num_patches=NUM_PATCHES,
                    log_step=log_step,
                    is_main_process=is_main_process,
                    val_time_limit=cfg.val_time_limit,
                    writer=writer,
                )
                vla.train()

            if log_step == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break
    
    # Close TensorBoard writer
    if writer is not None:
        writer.close()
        print(f"TensorBoard logs saved to: {run_dir}")


if __name__ == "__main__":
    finetune()