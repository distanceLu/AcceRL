"""
finetune.py (Debug Version)

Fine-tunes OpenVLA via LoRA.
This version is modified to run on a single GPU for easier debugging by removing all
Distributed Data Parallel (DDP) wrappers and related logic.
"""

import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Type

import draccus
import torch
# import torch.distributed as dist # [Debug] Removed
import torch.nn as nn
import tqdm
# from accelerate import PartialState # [Debug] Removed
from huggingface_hub import HfApi, snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
# from torch.nn.parallel import DistributedDataParallel as DDP # [Debug] Removed
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from torch.utils.tensorboard import SummaryWriter

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
    vla_path: str = "/cpfs01/lcx_workspace/models/openvla-7b-oft-finetuned-libero-spatial-object-goal-10/"             # openvla/openvla-7b Path to OpenVLA model (on HuggingFace Hub or stored locally)
    pretrained_checkpoint = "/cpfs01/lcx_workspace/models/openvla-7b-oft-finetuned-libero-spatial-object-goal-10/"
    load_in_8bit:bool=False
    load_in_4bit:bool=False
    # Dataset
    data_root_dir: Path = Path("datasets/rlds")      # Directory containing RLDS datasets
    dataset_name: str = "aloha_scoop_x_into_bowl"    # Name of fine-tuning dataset (e.g., `aloha_scoop_x_into_bowl`)
    run_root_dir: Path = Path("runs")                # Path to directory to store logs & checkpoints
    shuffle_buffer_size: int = 100_000               # Dataloader shuffle buffer size (can reduce if OOM errors occur)

    # Algorithm and architecture
    use_l1_regression: bool = True                   # If True, trains continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, trains continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 1                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = False                        # If True, includes robot proprioceptive state in input

    # Training configuration
    batch_size: int = 8                              # Batch size per device (total batch size = batch_size * num GPUs)
    learning_rate: float = 5e-4                      # Learning rate
    lr_warmup_steps: int = 0                         # Number of steps to warm up learning rate (from 10% to 100%)
    num_steps_before_decay: int = 100_000            # Number of steps before LR decays by 10x
    grad_accumulation_steps: int = 1                 # Number of gradient accumulation steps
    max_steps: int = 200_000                         # Max number of training steps
    use_val_set: bool = False                        # If True, uses validation set and log validation metrics
    val_freq: int = 10_000                           # (When `use_val_set==True`) Validation set logging frequency in steps
    val_time_limit: int = 180                        # (When `use_val_set==True`) Time limit for computing validation metrics
    save_freq: int = 10_000                          # Checkpoint saving frequency in steps
    save_latest_checkpoint_only: bool = False        # If True, saves only 1 checkpoint, overwriting latest checkpoint
                                                     #   (If False, saves all checkpoints)
    resume: bool = False                             # If True, resumes from checkpoint
    resume_step: Optional[int] = None                # (When `resume==True`) Step number that we are resuming from
    image_aug: bool = True                           # If True, trains with image augmentations (HIGHLY RECOMMENDED)
    diffusion_sample_freq: int = 50                  # (When `use_diffusion==True`) Frequency for sampling in steps

    # LoRA
    use_lora: bool = True                            # If True, uses LoRA fine-tuning
    lora_rank: int = 32                              # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                        # Dropout applied to LoRA weights
    merge_lora_during_training: bool = True          # If True, merges LoRA weights and saves result during training
                                                     #   Note: Merging can be very slow on some machines. If so, set to
                                                     #         False and merge final checkpoint offline!

    # Logging
    tensorboard_logs: str = "tensorboard_logs"       # Directory to store TensorBoard logs
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    run_id_override: Optional[str] = None            # Optional string to override the run ID with
    wandb_log_freq: int = 10                         # WandB logging frequency in steps

    # fmt: on


def remove_ddp_in_checkpoint(state_dict) -> dict:
    """
    Removes the 'module.' prefix from parameter names in a PyTorch model state dictionary that was saved using
    DistributedDataParallel (DDP). This is kept for loading DDP-trained checkpoints in a non-DDP environment.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def get_run_id(cfg) -> str:
    """
    Generates or retrieves an identifier string for an experiment run.
    """
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
    """
    Loads a checkpoint for a given module.
    """
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    # The checkpoint might have been saved with DDP, so we strip the 'module.' prefix
    return remove_ddp_in_checkpoint(state_dict)


# [Debug] Removed wrap_ddp function as it's no longer needed.


def count_parameters(module: nn.Module, name: str) -> None:
    """
    Counts and prints the number of trainable parameters in a module.
    """
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {num_params}")


def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: FinetuneConfig,
    device: str, # [Debug] Changed device_id to device string
    module_args: dict,
    to_bf16: bool = False,
    find_unused_params: bool = False, # [Debug] This arg is no longer used
) -> nn.Module: # [Debug] Changed return type from DDP to nn.Module
    """
    Initializes a module, optionally loads checkpoint, and moves to device.
    (DDP wrapper removed for debugging).
    """
    module = module_class(**module_args)
    count_parameters(module, module_name)

    if cfg.resume:
        state_dict = load_checkpoint(module_name, cfg.vla_path, cfg.resume_step)
        module.load_state_dict(state_dict)

    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device)

    # [Debug] DDP wrapping is removed.
    return module


def run_forward_pass(
    actor,
    vla,
    action_head,
    noisy_action_projector,
    proprio_projector,
    batch,
    action_tokenizer,
    device: str, # [Debug] Changed device_id to device
    use_l1_regression,
    use_diffusion,
    use_proprio,
    use_film,
    num_patches,
    compute_diffusion_l1=False,
    num_diffusion_steps_train=None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute model forward pass and metrics for both training and validation.
    """
    metrics = {}

    # Get ground-truth action labels
    ground_truth_actions = batch["actions"].to(device).to(torch.bfloat16)

    # [Only for diffusion] Sample noisy actions used as input for noise predictor network
    if use_diffusion:
        # [Debug] Removed .module from action_head
        noisy_dict = action_head.sample_noisy_actions(ground_truth_actions)
        noise, noisy_actions, diffusion_timestep_embeddings = (
            noisy_dict["noise"],
            noisy_dict["noisy_actions"],
            noisy_dict["diffusion_timestep_embeddings"],
        )
    else:
        noise, noisy_actions, diffusion_timestep_embeddings = None, None, None

    # VLA forward pass
    # [Debug] Commented out original forward pass which might be useful for reference
    # with torch.autocast("cuda", dtype=torch.bfloat16):
    #     output: CausalLMOutputWithPast = vla(
    #         input_ids=batch["input_ids"].to(device),
    #         attention_mask=batch["attention_mask"].to(device),
    #         pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device),
    #         labels=batch["labels"],
    #         output_hidden_states=True,
    #         proprio=batch["proprio"] if use_proprio else None,
    #         proprio_projector=proprio_projector if use_proprio else None,
    #         noisy_actions=noisy_actions if use_diffusion else None,
    #         noisy_action_projector=noisy_action_projector if use_diffusion else None,
    #         diffusion_timestep_embeddings=diffusion_timestep_embeddings if use_diffusion else None,
    #         use_film=use_film,
    #     )

    # Get action masks needed for logging
    ground_truth_token_ids = batch["labels"][:, 1:].to(device)
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

    # Compute metrics for discrete action representation (next-token prediction)
    if not (use_l1_regression or use_diffusion):
        # This part requires the output from the VLA forward pass.
        # If you are not using l1_regression or diffusion, you will need to uncomment
        # the VLA forward pass above.
        raise NotImplementedError("Discrete action prediction requires the VLA forward pass to be active.")
        loss = output.loss
        predicted_token_ids = output.logits[:, num_patches:-1].argmax(dim=2)
        # ... (rest of the original logic)

    # Compute metrics for continuous action representations (L1 regression | diffusion)
    else:
        action_all, mu_all, log_std_all, value, dist = actor.forward(batch)
        log_probs = dist.log_prob(ground_truth_actions)
        nll_loss = -log_probs.mean()
        l1_loss_sample = torch.nn.L1Loss()(ground_truth_actions, action_all).detach()
        l1_loss_mu = torch.nn.L1Loss()(ground_truth_actions, mu_all).detach()
        
        # [Debug] The original logic for getting hidden states and calling action heads is now
        # encapsulated within the actor.forward() call. The following code is kept for reference.

        # if use_l1_regression:
        #     predicted_actions = action_head.predict_action(actions_hidden_states)
        #     loss = torch.nn.L1Loss()(ground_truth_actions, predicted_actions)

        # if use_diffusion:
        #     noise_pred = action_head.predict_noise(actions_hidden_states)
        #     noise_pred = noise_pred.reshape(noise.shape)
        #     loss = nn.functional.mse_loss(noise_pred, noise, reduction="mean")
        #     if compute_diffusion_l1:
        #          # ... (sampling logic)

        metrics.update(
            {
                "loss_value": nll_loss.item(),  # Detached value for logging
            }
        )

    # Return both the loss tensor (with gradients) and the metrics dictionary (with detached values)
    return nll_loss, metrics


def run_diffusion_sampling(
    vla,
    action_head,
    noisy_action_projector,
    proprio_projector,
    batch,
    batch_size,
    num_patches,
    actions_shape,
    device: str, # [Debug] changed device_id to device
    current_action_mask,
    next_actions_mask,
    use_proprio,
    use_film,
) -> torch.Tensor:
    """
    Run diffusion sampling (reverse diffusion) to generate actions.
    """
    noise = torch.randn(
        size=(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM),
        device=device,
        dtype=torch.bfloat16,
    )

    # [Debug] Removed .module access
    action_head.noise_scheduler.set_timesteps(action_head.num_diffusion_steps_train)

    curr_noisy_actions = noise
    # [Debug] Removed .module access
    for t in action_head.noise_scheduler.timesteps:
        timesteps = torch.Tensor([t]).repeat(batch_size).to(device)
        diffusion_timestep_embeddings = (
            # [Debug] Removed .module access
            action_head.time_encoder(timesteps).to(curr_noisy_actions.dtype).to(curr_noisy_actions.device)
        )
        diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)

        with torch.autocast("cuda", dtype=torch.bfloat16):
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
            )
            actions_hidden_states = actions_hidden_states.to(torch.bfloat16)
            # [Debug] Removed .module access
            noise_pred = action_head.predict_noise(actions_hidden_states)

        # [Debug] Removed .module access
        curr_noisy_actions = action_head.noise_scheduler.step(noise_pred, t, curr_noisy_actions).prev_sample

    return curr_noisy_actions.reshape(actions_shape)


def compute_smoothened_metrics(metrics_deques) -> dict:
    """
    Compute smoothened metrics from recent deques.
    """
    smoothened_metrics = {}
    for name, deque in metrics_deques.items():
        if deque and len(deque) > 0:
            smoothened_metrics[name] = sum(deque) / len(deque)
    return smoothened_metrics


def log_metrics_to_wandb(metrics, prefix, step, wandb_entity) -> None:
    """
    Log metrics to Weights & Biases.
    """
    log_dict = {}
    for name, value in metrics.items():
        if name == "loss_value":
            log_dict[f"{prefix}/Loss"] = value
        else:
            log_dict[f"{prefix}/{name.replace('_', ' ').title()}"] = value
    wandb_entity.log(log_dict, step=step)


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
    # [Debug] distributed_state removed as it's no longer needed
) -> None:
    """
    Save all training checkpoints.
    """
    if cfg.save_latest_checkpoint_only:
        checkpoint_dir = run_dir
        checkpoint_name_suffix = "latest_checkpoint.pt"
    else:
        checkpoint_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
        checkpoint_name_suffix = f"{log_step}_checkpoint.pt"

    adapter_dir = checkpoint_dir / "lora_adapter"

    # [Debug] Removed 'is_main_process' check
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(adapter_dir, exist_ok=True)
    save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
    print(f"Saving Model Checkpoint for Step {log_step}")

    # [Debug] Removed dist.barrier()

    # [Debug] Removed 'is_main_process' check
    processor.save_pretrained(checkpoint_dir)
    # [Debug] Removed .module access
    vla.save_pretrained(adapter_dir)

    if cfg.use_proprio and proprio_projector is not None:
        torch.save(proprio_projector.state_dict(), checkpoint_dir / f"proprio_projector--{checkpoint_name_suffix}")

    if cfg.use_diffusion and noisy_action_projector is not None:
        torch.save(
            noisy_action_projector.state_dict(), checkpoint_dir / f"noisy_action_projector--{checkpoint_name_suffix}"
        )

    if (cfg.use_l1_regression or cfg.use_diffusion) and action_head is not None:
        torch.save(action_head.state_dict(), checkpoint_dir / f"action_head--{checkpoint_name_suffix}")

    if cfg.use_film:
        # [Debug] Removed .module access
        torch.save(
            vla.vision_backbone.state_dict(), checkpoint_dir / f"vision_backbone--{checkpoint_name_suffix}"
        )

    # [Debug] Removed dist.barrier()

    if cfg.use_lora and cfg.merge_lora_during_training:
        base_vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
        )
        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
        merged_vla = merged_vla.merge_and_unload()

        # [Debug] Removed 'is_main_process' check
        merged_vla.save_pretrained(checkpoint_dir)
        print(f"Saved merged model for Step {log_step} at: {checkpoint_dir}")

        # [Debug] Removed dist.barrier()


def run_validation(
    vla,
    action_head,
    noisy_action_projector,
    proprio_projector,
    val_dataloader,
    action_tokenizer,
    device: str, # [Debug] Changed device_id to device
    cfg,
    num_patches,
    log_step,
    val_time_limit,
    tb_writer,
    # [Debug] distributed_state removed
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
    for metric_name in all_val_metrics[0].keys():
        values = [metrics[metric_name] for metrics in all_val_metrics if metric_name in metrics]
        if values:
            avg_val_metrics[metric_name] = sum(values) / len(values)

    avg_val_metrics["val_batches_count"] = val_batches_count

    # [Debug] Removed 'is_main_process' check
    log_metrics_to_wandb(avg_val_metrics, "VLA Val", log_step, wandb)
    for name, value in avg_val_metrics.items():
        tb_writer.add_scalar(f"VLA Val/{name}", value, global_step=log_step)

@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    """
    Fine-tunes base VLA on demonstration dataset via LoRA (Single-GPU Debug Version).
    """
    assert cfg.use_lora, "Only LoRA fine-tuning is supported. Please set --use_lora=True!"
    assert not (cfg.use_l1_regression and cfg.use_diffusion), (
        "Cannot do both L1 regression and diffusion. Please pick one of them!"
    )

    cfg.vla_path = cfg.vla_path.rstrip("/")
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    run_id = get_run_id(cfg)
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)

    # [Debug] GPU setup modified for single-GPU debugging
    device_id = 0  # Hardcode to GPU 0, or change as needed
    device = f"cuda:{device_id}"
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    # [Debug] Initialize logging directly, as we are always on the main process.
    wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{run_id}")
    tensorboard_log_dir = cfg.tensorboard_logs
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=tensorboard_log_dir)

    print(
        "Detected constants:\n"
        f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n"
        f"\tACTION_DIM: {ACTION_DIM}\n"
        f"\tPROPRIO_DIM: {PROPRIO_DIM}\n"
        f"\tACTION_PROPRIO_NORMALIZATION_TYPE: {ACTION_PROPRIO_NORMALIZATION_TYPE}"
    )

    # [Debug] Removed Hugging Face model download and file sync logic that was tied to distributed setup.
    # Assuming the model path is correct and files are in place.

    from rl.actor_critic_model import ActorCritic
    from rl.utils import prepare_one_obs, check_unnorm_key

    USE_BF16: bool = True
    TORCH_DTYPE = torch.bfloat16 if USE_BF16 else torch.float32

    actor = ActorCritic(cfg, TORCH_DTYPE)
    actor.get_parameter_groups()

    processor = actor.processor
    vla = actor.vla.to(device)

    # LoRA setup
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

    # FiLM setup
    if cfg.use_film:
        count_parameters(vla.vision_backbone, "vla.vision_backbone (original)")
        # [Debug] `vla.model` is correct for PeftModel, no change needed here.
        vla.model.vision_backbone = FiLMedPrismaticVisionBackbone(
            vision_backbone=vla.model.vision_backbone,
            llm_dim=vla.llm_dim,
        )
        count_parameters(vla.vision_backbone, "vla.vision_backbone (post-wrap)")
        if cfg.resume:
            state_dict = load_checkpoint("vision_backbone", cfg.vla_path, cfg.resume_step)
            vla.model.vision_backbone.load_state_dict(state_dict)
        vla.model.vision_backbone = vla.model.vision_backbone.to(device)

    # [Debug] DDP wrapping is removed. The model is already on the correct device.
    # vla = wrap_ddp(vla, device_id, find_unused=True)

    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = actor.proprio_projector.to(device)

    action_head = None
    if cfg.use_l1_regression:
        action_head = actor.action_head.to(device)
    
    noisy_action_projector = None
    if cfg.use_diffusion:
        action_head = init_module(
            DiffusionActionHead,
            "action_head",
            cfg,
            device,
            {
                "input_dim": vla.llm_dim,
                "hidden_dim": vla.llm_dim,
                "action_dim": ACTION_DIM,
                "num_diffusion_steps_train": cfg.num_diffusion_steps_train,
            },
            to_bf16=True,
        )
        noisy_action_projector = init_module(
            NoisyActionProjector, "noisy_action_projector", cfg, device, {"llm_dim": vla.llm_dim}
        )
    
    # [Debug] Removed .module access
    NUM_PATCHES = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input()
    if cfg.use_proprio:
        NUM_PATCHES += 1
    if cfg.use_diffusion:
        NUM_PATCHES += 1

    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    if action_head:
        trainable_params += [param for param in action_head.parameters() if param.requires_grad]
    if noisy_action_projector:
        trainable_params += [param for param in noisy_action_projector.parameters() if param.requires_grad]
    if proprio_projector:
        trainable_params += [param for param in proprio_projector.parameters() if param.requires_grad]
    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    original_lr = optimizer.param_groups[0]["lr"]
    scheduler = MultiStepLR(
        optimizer, milestones=[cfg.num_steps_before_decay], gamma=0.1
    )
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    use_wrist_image = cfg.num_images_in_input > 1
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio,
    )
    # [Debug] Removed .module access
    train_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
    if cfg.use_val_set:
        # [Debug] Removed .module access
        val_dataset = RLDSDataset(
            cfg.data_root_dir,
            cfg.dataset_name,
            batch_transform,
            resize_resolution=tuple(vla.config.image_sizes),
            shuffle_buffer_size=cfg.shuffle_buffer_size // 10,
            image_aug=cfg.image_aug,
            train=False,
        )

    # [Debug] Removed 'is_main_process' check
    save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, sampler=None, collate_fn=collator, num_workers=0
    )
    if cfg.use_val_set:
        val_dataloader = DataLoader(
            val_dataset, batch_size=cfg.batch_size, sampler=None, collate_fn=collator, num_workers=0
        )

    recent_metrics = {
        "loss_value": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
    }

    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            compute_diffusion_l1 = cfg.use_diffusion and batch_idx % cfg.diffusion_sample_freq == 0
            loss, metrics = run_forward_pass(
                actor,
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
            # [Debug] Removed 'is_main_process' check for logging
            if log_step % cfg.wandb_log_freq == 0:
                log_metrics_to_wandb(smoothened_metrics, "VLA Train", log_step, wandb)
                for name, value in smoothened_metrics.items():
                    tb_writer.add_scalar(f"Train/{name}", value, global_step=log_step)

            if cfg.lr_warmup_steps > 0:
                lr_progress = min((gradient_step_idx + 1) / cfg.lr_warmup_steps, 1.0)
                current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr
            
            # [Debug] Removed 'is_main_process' check for logging LR
            if gradient_step_idx % cfg.wandb_log_freq == 0:
                current_lr = scheduler.get_last_lr()[0]
                wandb.log({"VLA Train/Learning Rate": current_lr}, step=log_step)
                tb_writer.add_scalar("VLA Train/Learning Rate", current_lr, global_step=log_step)

            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress.update()

            if gradient_step_idx > 0 and log_step % cfg.save_freq == 0:
                save_training_checkpoint(
                    cfg=cfg,
                    run_dir=run_dir,
                    log_step=log_step,
                    vla=vla,
                    processor=processor,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    action_head=action_head,
                    train_dataset=train_dataset,
                )

            if cfg.use_val_set and log_step > 0 and log_step % cfg.val_freq == 0:
                run_validation(
                    vla=vla,
                    action_head=action_head,
                    noisy_action_projector=noisy_action_projector,
                    proprio_projector=proprio_projector,
                    val_dataloader=val_dataloader,
                    action_tokenizer=action_tokenizer,
                    device=device,
                    cfg=cfg,
                    num_patches=NUM_PATCHES,
                    log_step=log_step,
                    val_time_limit=cfg.val_time_limit,
                    tb_writer=tb_writer,
                )
                vla.train()

            if log_step >= cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break
        
    tb_writer.close()

if __name__ == "__main__":
    finetune()