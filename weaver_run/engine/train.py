import os
import argparse
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["TMPDIR"] = "/dev/shm"

import sys
import time
os.environ.setdefault("ACCERL_TELEMETRY_DB", "/mnt/data/lcx1/dong/need_data/collection.sqlite")
os.environ.setdefault("ACCERL_RUN_ID", f"manual_{os.getpid()}_{time.time_ns()}")
import random
import asyncio
import json
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass, field
from pathlib import Path
import math
from hydra.utils import instantiate
from omegaconf import OmegaConf
import numpy as np
import ray
import torch
from capture_telemetry import Telemetry, now_ns, new_id
from capture_hooks import (install_hooks, profile_rpc, batch_begin, batch_end, stage_begin, stage_end,
                           lineage_begin, lineage_end, lineage_sample, lineage_frame_parent,
                           current_request_metadata)
from capture_training import install_training_hooks, micro_begin, micro_end
from fresh_config import FreshConfig
from fresh_policy_replay import PolicySegment, SegmentReplay, FreshReplayNotReady
from fresh_signal import record_signal
import torch.nn as nn
import torch.distributions
from torch.distributions import kl
import deepspeed
from fsdp_engine import PolicyFSDPEngine
from cuda_ipc_weights import CudaIPCWeightExporter, apply_cuda_ipc_weights
from ray_memory_config import validate_object_store_budget
from reward_features import install_reward_features_only
from memory_policy import install_reward_checkpointing
from training_topology import (
    DUAL_LAYOUT, FOUR_LAYOUT, SPLIT_LAYOUT, parse_gpu_ids, build_bindings, ray_init_options, actor_options,
    verify_bound_gpu, successful_updates, GlobalBatchCoordinator, combine_wm_metrics,
    export_cpu_weights, apply_cpu_weights,
)
from shared_inference import PolicyRewardInferenceActor, RewardInferenceHandle
from parallel_checkpoint import stop_requested, save_parallel_weights, export_split_component, save_split_weights
from evaluation_threshold import ThresholdTracker
from nsight_capture import NsightCaptureWindows
from observed_cpu_prefetch import ObservedCpuReplayPrefetch
from pipeline_runtime import EpisodeReplayMixin, run_producer
from pipeline_budget import PipelineBudgetConfig
from world_backends.runtime import context_geometry
from policy_joint_batch_mode import PolicyJointBatchMode
from training_stop_policy import TrainingStopPolicy, EvaluationMilestones
from split_serial import (CpuReplayPrefetch, SerialPhase, epoch_groups, finish_preparation,
                          serial_enabled, validate_sample_group, validate_shared_wm_samples)
import torch.distributed as distributed
from torch.utils.tensorboard import SummaryWriter

# ---- AcceRL sys.path 设置 ----
ACCE_RL_ROOT = Path(os.environ.get("ACCERL_REPO_ROOT", Path(__file__).resolve().parents[2]))
if str(ACCE_RL_ROOT) not in sys.path:
    sys.path.insert(0, str(ACCE_RL_ROOT))

# OpenVLA 组件与常量
from experiments.robot.sole_utils import get_processor
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM
from experiments.robot.libero.libero_utils import GenerateConfig, TaskSuite
from rl.actor_critic_model_discrete import ActorCritic
from rl.utils import prepare_one_obs
# 训练/推理通信
from rl.ds_com import TrainerActorCom, InferenceActorCom
from rl.com_utils import find_free_port
from envs.utils import tensor_to_image, image_to_tensor, load_reward_model

# ---- Ctrl-World 相关导入（ctrl_world 已作为包安装在 merged-env 中）----
# The selected backend owns its dependencies. A WEAVER candidate does not
# require the CTRL package merely to import the orchestration module.
try:
    from ctrl_world_env_batch import CtrlWorldEnvBatch
    from ctrl_world.config import wm_args
    from ctrl_world.models.ctrl_world import CrtlWorld
except ModuleNotFoundError as error:
    _ctrl_import_error = error
    CtrlWorldEnvBatch = wm_args = None
    CrtlWorld = nn.Module
from world_backends import WorldBackendConfig, build_service_plan
from world_backends.runtime import load_backend_config, CtrlBackendAdapter
from world_backends.libero import validate_online_schema, environment_dt, backend_window_payload
from weaver_inference_actor import make_weaver_inference_actor
from weaver_online_bridge import make_weaver_online_trainer, make_weaver_online_inference
from world_backend_runtime import isolated_online_weaver, save_online_split_snapshot, validate_online_runtime_arguments
from rl.ray_debug_utils import setup_debugger
from rl.ds_com import _unwrap_module
from rl.wm_training_utils import (
    cosine_warmup_lr,
    frame_align_transition_actions,
    stable_episode_fraction,
)
import einops
import torch.nn.functional as F
import torch.distributed as dist


def batch_ssim(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Gaussian-window SSIM for each RGB image in [0, 1]. Returns [N]."""
    channels = predictions.shape[1]
    coordinates = torch.arange(
        window_size, device=predictions.device, dtype=predictions.dtype
    ) - (window_size - 1) / 2
    kernel_1d = torch.exp(-(coordinates.square()) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    window = kernel_2d.expand(channels, 1, window_size, window_size)
    mu_pred = F.conv2d(predictions, window, groups=channels)
    mu_target = F.conv2d(targets, window, groups=channels)
    mu_pred_sq = mu_pred.square()
    mu_target_sq = mu_target.square()
    mu_product = mu_pred * mu_target
    variance_pred = F.conv2d(predictions.square(), window, groups=channels) - mu_pred_sq
    variance_target = F.conv2d(targets.square(), window, groups=channels) - mu_target_sq
    covariance = F.conv2d(predictions * targets, window, groups=channels) - mu_product
    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2 * mu_product + c1) * (2 * covariance + c2)) / (
        (mu_pred_sq + mu_target_sq + c1) * (variance_pred + variance_target + c2)
    )
    return ssim_map.mean(dim=(1, 2, 3))


def _normalize_actions_bounds(actions: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    return (2.0 * (actions - low) / (high - low + 1e-8) - 1.0).clamp(-1.0, 1.0)


def _load_vae_decoder_checkpoint(model: CrtlWorld, checkpoint_path: Optional[str]) -> None:
    if checkpoint_path is None:
        return
    print(f"Loading VAE decoder checkpoint from {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "decoder" not in payload:
        raise ValueError(f"{checkpoint_path} is not a decoder checkpoint with key 'decoder'.")
    model.vae.decoder.load_state_dict(payload["decoder"], strict=True)


class AcceRLCtrlWorld(CrtlWorld):
    """Multi-camera pixel encoder matching Ctrl-World AcceRL offline training."""

    def encode_pixels_to_latents(self, pixels: torch.Tensor) -> torch.Tensor:
        target_dtype = next(self.unet.parameters()).dtype
        if pixels.ndim == 5:
            return super().encode_pixels_to_latents(pixels).to(dtype=target_dtype)
        if pixels.ndim != 6:
            raise ValueError(
                "Expected pixels shaped [B,F,C,H,W] or [B,F,M,C,H,W], "
                f"got {tuple(pixels.shape)}"
            )
        batch_size, num_frames, num_cams, channels, height, width = pixels.shape
        flat_views = pixels.reshape(
            batch_size * num_frames * num_cams, 1, channels, height, width
        )
        flat_latents = super().encode_pixels_to_latents(flat_views)[:, 0]
        view_latents = flat_latents.reshape(
            batch_size, num_frames, num_cams, *flat_latents.shape[1:]
        )
        return einops.rearrange(view_latents, "b f m c h w -> b f c (m h) w").to(dtype=target_dtype)

    def forward(self, batch):
        """Ctrl-World forward with dtype kept consistent for bf16 training."""
        if "latent" in batch:
            latents = batch["latent"]
        else:
            latents = self.encode_pixels_to_latents(batch["pixel"])

        texts = batch["text"]
        dtype = next(self.unet.parameters()).dtype
        device = next(self.unet.parameters()).device
        p_mean = 0.7
        p_std = 1.6
        noise_aug_strength = 0.0

        num_history = self.args.num_history
        latents = latents.to(device=device, dtype=dtype)

        current_img = latents[:, num_history : (num_history + 1)][:, 0]
        bsz, num_frames = latents.shape[:2]
        # Keep stochastic schedules/loss weights in fp32 even when the UNet is
        # bf16; this matches the offline trainer and avoids quantizing EDM sigma.
        sigma_img = torch.rand([bsz, 1, 1, 1], device=device, dtype=torch.float32) * 0.2
        c_in_img = 1 / (sigma_img**2 + 1) ** 0.5
        current_img = c_in_img.to(dtype) * (
            current_img + torch.randn_like(current_img) * sigma_img.to(dtype)
        )
        condition_latent = einops.repeat(current_img, "b c h w -> b f c h w", f=num_frames)
        if self.args.his_cond_zero:
            condition_latent[:, :num_history] = 0.0

        action = batch["action"].to(device=device, dtype=dtype)
        action_hidden = self.action_encoder(
            action,
            texts,
            self.tokenizer,
            self.text_encoder,
            frame_level_cond=self.args.frame_level_cond,
        ).to(dtype=dtype)

        uncond_hidden_states = torch.zeros_like(action_hidden)
        text_mask = (torch.rand(action_hidden.shape[0], device=device) > 0.05).unsqueeze(1).unsqueeze(2)
        action_hidden = action_hidden * text_mask + uncond_hidden_states * (~text_mask)

        rnd_normal = torch.randn([bsz, 1, 1, 1, 1], device=device, dtype=torch.float32)
        sigma = (rnd_normal * p_std + p_mean).exp()
        c_skip = 1 / (sigma**2 + 1)
        c_out = -sigma / (sigma**2 + 1) ** 0.5
        c_in = 1 / (sigma**2 + 1) ** 0.5
        c_noise = (sigma.log() / 4).reshape([bsz]).to(dtype=dtype)
        loss_weight = (sigma**2 + 1) / sigma**2
        noisy_latents = latents + torch.randn_like(latents) * sigma.to(dtype)

        sigma_h = torch.randn(
            [bsz, num_history, 1, 1, 1], device=device, dtype=torch.float32
        ) * 0.3
        history = latents[:, :num_history]
        sigma_h_model = sigma_h.to(dtype)
        noisy_history = 1 / (sigma_h_model**2 + 1) ** 0.5 * (
            history + sigma_h_model * torch.randn_like(history)
        )
        input_latents = torch.cat(
            [noisy_history, c_in.to(dtype) * noisy_latents[:, num_history:]], dim=1
        )

        input_latents = torch.cat(
            [input_latents, condition_latent / self.vae.config.scaling_factor],
            dim=2,
        ).to(dtype=dtype)

        added_time_ids = self.pipeline._get_add_time_ids(
            self.args.fps,
            self.args.motion_bucket_id,
            noise_aug_strength,
            action_hidden.dtype,
            bsz,
            1,
            False,
        ).to(device=device, dtype=dtype)

        model_pred = self.unet(
            input_latents,
            c_noise,
            encoder_hidden_states=action_hidden,
            added_time_ids=added_time_ids,
            frame_level_cond=self.args.frame_level_cond,
        ).sample
        predict_x0 = c_out.to(dtype) * model_pred + c_skip.to(dtype) * noisy_latents
        prediction_error = (
            predict_x0[:, num_history:].float() - latents[:, num_history:].float()
        )
        loss = (prediction_error.square() * loss_weight).mean()

        return loss, torch.tensor(0.0, device=device, dtype=dtype)


# ================================================================
# 0. 超参数与配置 - 命令行参数解析
# ================================================================
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='OpenVLA RL Training with Ctrl-World World Model')

    # 环境变量
    parser.add_argument('--cuda-visible-devices', type=str, default='0,1,2,3',
                        help='Physical CUDA devices; this overnight candidate is restricted to GPU0-3')

    parser.add_argument('--gpu-layout', choices=['dedicated', DUAL_LAYOUT, FOUR_LAYOUT, SPLIT_LAYOUT], default='dedicated',
                        help='dual_train_policy_reward: two training ranks, shared Policy/Reward, dedicated World')
    parser.add_argument('--validate-only', action='store_true',
                        help='Load all real models, verify bindings and initial broadcasts, then exit without collecting/training')

    # Libero benchmark
    parser.add_argument('--benchmark', type=str, default='libero_spatial',
                        choices=['libero_spatial', 'libero_object', 'libero_goal', 'libero_10', 'libero_90'],
                        help='Libero benchmark suite (default: libero_spatial)')

    # 分布式系统参数
    parser.add_argument('--num-trainer-gpus', type=int, default=1)
    parser.add_argument('--num-inference-actors', type=int, default=1)
    parser.add_argument('--num-rollout-workers', type=int, default=20)
    parser.add_argument("--pipeline-config", default=os.environ.get("ACCERL_PIPELINE_CONFIG"))
    parser.add_argument('--num-eval-workers', type=int, default=10)
    parser.add_argument('--rollout-local-buf', type=int, default=64)
    parser.add_argument('--inference-batch', type=int, default=8)
    parser.add_argument('--inference-timeout-ms', type=int, default=300)
    parser.add_argument('--replay-capacity', type=int, default=10000)
    parser.add_argument('--train-batch-size', type=int, default=32)
    parser.add_argument('--accumulation-steps', type=int, default=8)
    parser.add_argument('--train-iters', type=int, default=30000)

    # Checkpoint
    parser.add_argument('--ckpt-dir', type=str,
                        default='/mnt/data/lcx2/yanjieworkspace/models/finetune_rl/')
    parser.add_argument('--ckpt-every-steps', type=int, default=2000000)

    # PPO 参数
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lambda', type=float, default=0.95, dest='lambda_')
    parser.add_argument('--clip-eps', type=float, default=0.2)
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.00)
    parser.add_argument('--kl-coef', type=float, default=0.1)

    # 奖励缩放
    parser.add_argument('--reward-scale', type=float, default=1.0)

    # 学习率调度参数
    parser.add_argument('--value-lr', type=float, default=1e-4)
    parser.add_argument('--policy-lr', type=float, default=1e-5)
    parser.add_argument('--value-warmup-steps', type=int, default=500)
    parser.add_argument('--policy-warmup-steps', type=int, default=500)
    parser.add_argument('--policy-train-start-step', type=int, default=0)

    # 世界模型配置
    parser.add_argument('--imagine-horizon', type=int, default=8)
    parser.add_argument('--num-step-cond', type=int, default=7,
                        help='Real context frames; must be at least num_history + 1')
    parser.add_argument('--num-reward-inference-actors', type=int, default=1)
    parser.add_argument('--num-ctrl-inference-actors', type=int, default=1,
                        help='Number of Ctrl-World inference actors')
    parser.add_argument('--agent-config-path', type=str, default='envs/config/agent.yaml')
    parser.add_argument('--trainer-config-path', type=str, default='envs/config/trainer.yaml')

    # 日志
    parser.add_argument('--moving-avg-window', type=int, default=1000)
    parser.add_argument('--log-interval-seconds', type=int, default=10)

    # 通信组
    parser.add_argument('--broadcast-group-name', type=str, default='trainer_to_inference_broadcast')

    # OpenVLA 加载配置
    parser.add_argument('--use-bf16', action='store_true', default=True)
    parser.add_argument('--no-bf16', action='store_false', dest='use_bf16')
    parser.add_argument('--use-proprio', action='store_true', default=False)
    parser.add_argument('--num-images-in-input', type=int, default=2)
    parser.add_argument('--pretrained-checkpoint', type=str,
                        default='/mnt/data/lcx2/yanjieworkspace/models/finetune_im/openvla-7b+libero_spatial_no_noops+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--discrete_acts--proprio_state--100000_chkpt')
    parser.add_argument('--checkpoint2', type=str, default=None)

    parser.add_argument('--clip-mode', type=str, default='sapo',
                        choices=['ppo', 'sapo', 'gipo'])
    parser.add_argument('--exp-name', type=str, default=None)

    # ============ Ctrl-World 专用参数 ============
    parser.add_argument('--svd-model-path', type=str,
                        default='/mnt/data/lcx3/checkpoint/ctrl_world/svd/svd_model',
                        help='Path to SVD model')
    parser.add_argument('--clip-model-path', type=str,
                        default='/mnt/data/lcx3/checkpoint/ctrl_world/clip/clip_model',
                        help='Path to CLIP model')
    parser.add_argument('--world-backend-config', type=str, default=None,
                        help='Explicit pluggable world/reward configuration JSON')
    parser.add_argument('--ctrl-world-ckpt', type=str, default=None,
                        help='Path to a Ctrl-World checkpoint trained on AcceRL delta actions')
    parser.add_argument('--vae-decoder-checkpoint', type=str, default=None,
                        help='Optional fine-tuned VAE decoder loaded after the Ctrl-World checkpoint')
    parser.add_argument('--condition-stat-path', type=str, default=None,
                        help='condition_stat.json; defaults to checkpoint sibling')
    parser.add_argument('--num-cams', type=int, default=2,
                        help='Number of camera views for Ctrl-World')
    parser.add_argument('--num-history', type=int, default=6,
                        help='Ctrl-World latent history length')
    parser.add_argument('--num-inference-steps', type=int, default=50,
                        help='SVD denoising steps')
    parser.add_argument('--target-height', type=int, default=192)
    parser.add_argument('--target-width', type=int, default=320)
    parser.add_argument('--num-frames-pred', type=int, default=5,
                        help='Ctrl-World chunk length including current anchor (checkpoint protocol: 5)')

    # Reward Model checkpoint 路径覆盖
    parser.add_argument('--reward-checkpoint', type=str, default=None,
                        help='Reward model checkpoint path (overrides agent.yaml)')

    # ============ World Model 训练参数 ============
    # Reward Model 参数（保持与原版一致）
    parser.add_argument('--reward-batch-size', type=int, default=32)
    parser.add_argument('--reward-accumulation-steps', type=int, default=4)
    parser.add_argument('--reward-lr', type=float, default=1e-4)
    parser.add_argument('--reward-warmup-steps', type=int, default=500)

    # World Model Replay Buffer 参数
    parser.add_argument('--wm-replay-capacity', type=int, default=50000)
    parser.add_argument('--real-traj-collect-interval', type=int, default=10)

    # World Model 训练频率控制
    parser.add_argument('--reward-train-interval', type=int, default=10,
                        help='Train reward model every N actor training steps')

    # Ctrl-World online training (mirrors Diamond denoiser train/broadcast)
    parser.add_argument('--ctrl-batch-size', type=int, default=1,
                        help='Ctrl-World micro-batch size (keep small; SVD UNet is heavy)')
    parser.add_argument('--ctrl-accumulation-steps', type=int, default=8,
                        help='Ctrl-World gradient accumulation steps')
    parser.add_argument('--ctrl-lr', type=float, default=1e-5,
                        help='Ctrl-World peak learning rate')
    parser.add_argument('--ctrl-warmup-steps', type=int, default=500)
    parser.add_argument('--ctrl-train-interval', type=int, default=10,
                        help='Train Ctrl-World every N actor training steps')
    # Ctrl-World prediction quality eval (LPIPS / PSNR / SSIM -> TensorBoard)
    parser.add_argument('--ctrl-eval-interval', type=int, default=100,
                        help='Evaluate Ctrl-World prediction metrics every N steps (0 disables)')
    parser.add_argument('--ctrl-eval-batch-size', type=int, default=4,
                        help='Number of fixed held-out WM windows used per eval')
    parser.add_argument('--ctrl-eval-micro-batch-size', type=int, default=4,
                        help='Inference micro-batch for fixed Ctrl-World evaluation')
    parser.add_argument('--ctrl-eval-seed', type=int, default=12345,
                        help='Fixed VAE/diffusion seed for reproducible Ctrl-World eval')
    parser.add_argument('--ctrl-eval-episode-ratio', type=float, default=0.2,
                        help='Deterministic episode-level fraction reserved from WM training')
    parser.add_argument('--ctrl-eval-windows-per-episode', type=int, default=8,
                        help='Maximum fixed eval windows retained from one held-out episode')
    parser.add_argument('--ctrl-eval-max-horizon', type=int, default=4,
                        help='Max future frames for teacher-forced Ctrl-World metrics')
    parser.add_argument('--ctrl-eval-lpips-net', type=str, default='alex',
                        choices=['alex', 'vgg', 'squeeze'],
                        help='LPIPS backbone used for Ctrl-World eval')
    parser.add_argument('--ctrl-ema-decay', type=float, default=0.99,
                        help='EMA decay applied by Ctrl inference actors when receiving train weights')
    # ============ 调试参数 ============
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Enable debugpy in Ray actors (each actor starts a debug server)')
    parser.add_argument('--debug-wait', action='store_true', default=False,
                        help='Actors wait for VS Code debugger to attach before proceeding')

    args = parser.parse_args()
    world_config = load_backend_config(args)
    world_config.validate_admission()
    validate_online_schema(world_config, ACTION_DIM)
    if world_config.backend == "weaver":
        if args.gpu_layout != SPLIT_LAYOUT or serial_enabled(args.gpu_layout, environment=os.environ):
            raise ValueError("WEAVER v1 requires concurrent split owners on four GPUs")
        if args.ctrl_train_interval != 5 or (world_config.reward_provider == "external_vlm" and args.reward_train_interval != 5):
            raise ValueError("WEAVER v1 preserves the 5:1 world update schedule")
        args.num_reward_inference_actors = 0 if world_config.reward_provider == "world_head" else 4
        schema = world_config.schema
        args.num_step_cond = schema["n_history"] + schema["n_memory_frames"] * schema["t_memory"]
        if float(args.reward_scale) != float(world_config.reward_scale):
            raise ValueError("CLI reward_scale and backend reward_scale must agree")
    elif not args.ctrl_world_ckpt:
        raise ValueError("CTRL requires --ctrl-world-ckpt")

    if args.num_cams != 2:
        raise ValueError("This AcceRL integration requires agentview + wrist (num_cams=2)")
    if args.num_images_in_input != 2:
        raise ValueError("VLA must receive both generated views (num_images_in_input=2)")
    if args.use_proprio:
        raise ValueError("Imagined proprio is not modeled; run this integration with --use-proprio disabled")
    if world_config.backend == "ctrl" and args.num_frames_pred != 5:
        raise ValueError("The current Ctrl-World checkpoint protocol requires num_frames_pred=5")
    if world_config.backend == "ctrl" and args.num_step_cond < args.num_history + 1:
        raise ValueError("num_step_cond must provide num_history previous frames plus current")
    visible_gpus = [x.strip() for x in args.cuda_visible_devices.split(",") if x.strip()]
    if (not visible_gpus or len(visible_gpus) != len(set(visible_gpus))
            or visible_gpus != os.environ.get('ACCERL_ALLOCATED_GPU_IDS', '4,5,6,7').split(',')):
        raise ValueError("This overnight candidate is restricted to the four allocated physical GPUs")
    if args.gpu_layout == SPLIT_LAYOUT and os.environ.get('ACCERL_USE_MPS') != '1':
        raise ValueError('The concurrent per-card service layout requires the existing MPS service')
    if args.gpu_layout in (DUAL_LAYOUT, FOUR_LAYOUT, SPLIT_LAYOUT):
        visible_gpus = parse_gpu_ids(args.cuda_visible_devices)
        expected = (2, 1, 1, 1) if args.gpu_layout == DUAL_LAYOUT else (4, 4, 0 if world_config.reward_provider == "world_head" else 4, 4)
        if (args.num_trainer_gpus, args.num_inference_actors,
                args.num_reward_inference_actors, args.num_ctrl_inference_actors) != expected:
            raise ValueError(f"{args.gpu_layout} requires actor counts {expected}")
        if args.gpu_layout in (FOUR_LAYOUT, SPLIT_LAYOUT):
            batches = (args.train_batch_size * args.accumulation_steps * 4,
                       args.ctrl_batch_size * args.ctrl_accumulation_steps * 4,
                       args.reward_batch_size * args.reward_accumulation_steps * 4)
            active_batches = batches if world_config.reward_provider == "external_vlm" else batches[:2]
            expected_batches = (576, 1024, 1024) if world_config.reward_provider == "external_vlm" else (576, 1024)
            if active_batches != expected_batches or args.ctrl_ema_decay != 0:
                raise ValueError("Four-rank experiment preserves active batches " + str(expected_batches) + " and Ctrl EMA=0")
        if args.num_rollout_workers < 1:
            raise ValueError("At least one rollout worker is required")
    else:
        required_gpu_actors = (args.num_trainer_gpus + args.num_inference_actors
                               + args.num_reward_inference_actors + args.num_ctrl_inference_actors)
        if len(visible_gpus) < required_gpu_actors:
            raise ValueError(f"{required_gpu_actors} GPU actors requested, only {len(visible_gpus)} visible")
    batch_fields = ('train_batch_size', 'accumulation_steps', 'ctrl_batch_size', 'ctrl_accumulation_steps')
    if world_config.reward_provider == "external_vlm":
        batch_fields += ('reward_batch_size', 'reward_accumulation_steps')
    for name in batch_fields:
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive; batch/accumulation are per rank")

    if os.environ.get("ACCERL_USE_MPS") == "1":
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            binding["uuid"] for binding in build_bindings(parse_gpu_ids(args.cuda_visible_devices)))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    # 确保 Python 环境的 bin 目录在 PATH 中（Ray worker 需要 ninja 等工具）
    conda_bin = os.path.dirname(sys.executable)
    if conda_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{conda_bin}:{os.environ.get('PATH', '')}"

    # 设置调试环境变量（Ray worker 会继承）
    if args.debug:
        os.environ["RAY_DEBUG"] = "1"
    if args.debug_wait:
        os.environ["RAY_DEBUG_WAIT"] = "1"

    if args.exp_name is None:
        args.exp_name = f"OpenVLA_DS_{args.clip_mode}_DISCRETE_task0_ctrl_train_wm"

    return args


# ================================================================
# 数据结构
# ================================================================
@dataclass
class Experience:
    obs: Dict[str, torch.Tensor]
    action_token: np.ndarray
    advantage: float
    behaviour_logits: np.ndarray
    value_target: float
    sample_id: str = field(default_factory=lambda: new_id("policy"))
    episode_id: str = ""
    sample_index: int = -1
    model_versions: dict = field(default_factory=dict)


@dataclass
class WMExperience:
    """World Model 训练数据结构"""
    obs: np.ndarray                         # [num_steps_conditioning+1, M, C, H, W]
    act: np.ndarray                         # [num_steps_conditioning, act_dim]
    rew: float
    instruction: str
    episode_id: str
    sample_id: str = ""
    sample_index: int = -1
    model_versions: dict = field(default_factory=dict)
    backend_payload: dict = field(default_factory=dict)


# ================================================================
# 1.5 统计模块 (StatsActor) — 与原版完全一致
# ================================================================
@ray.remote
class StatsActor:
    def __init__(self, window_size):
        setup_debugger("stats")
        self.stats = defaultdict(lambda: {
            "episode_returns": deque(maxlen=window_size),
            "step_times": deque(maxlen=window_size),
            "episode_lengths": deque(maxlen=window_size),
            "successes": deque(maxlen=window_size),
            "total_episodes_processed": 0,
            "total_env_steps": 0
        })
        self.timings = defaultdict(lambda: deque(maxlen=window_size))
        self.imagine_rewards = deque(maxlen=window_size)
        self.actor_last_active = {}
        self.active_window_seconds = 600
        self.total_samples_produced = 0
        self.total_imagine_samples = 0
        self.last_eval_completed_ns = 0
        self.last_eval_versions = None

    def add_episode_return(self, env_name, ep_return, step_time, ep_length, success,
                           actor_id=None, step_num=0, completed_ns=None, evaluation_versions=None):
        env_stats = self.stats[env_name]
        env_stats["episode_returns"].append(ep_return)
        env_stats["step_times"].append(step_time)
        env_stats["episode_lengths"].append(ep_length)
        env_stats["successes"].append(success)
        env_stats["total_episodes_processed"] += 1
        env_stats["total_env_steps"] += ep_length
        if env_name.startswith("eval_"):
            done_ns = int(completed_ns if completed_ns is not None else now_ns())
            if done_ns > self.last_eval_completed_ns:
                self.last_eval_completed_ns = done_ns
                self.last_eval_versions = evaluation_versions
            telemetry = getattr(self, "telemetry", None)
            if telemetry is not None:
                telemetry.event("eval_episode_completed", ts_ns=done_ns,
                                details={"env_name": env_name, "success": float(success),
                                         "episode_return": float(ep_return),
                                         "policy_versions": evaluation_versions})
        if not env_name.startswith("eval_"):
            self.total_samples_produced += step_num
            if actor_id is not None:
                self.actor_last_active[actor_id] = time.time()

    def add_timing_metric(self, metric_name, value):
        self.timings[metric_name].append(value)

    def add_imagine_reward(self, avg_imagine_reward, actor_id, num_samples=1):
        self.imagine_rewards.append(avg_imagine_reward)
        self.total_imagine_samples += num_samples
        self.actor_last_active[actor_id] = time.time()

    def get_active_actor_count(self):
        current_time = time.time()
        cutoff = current_time - self.active_window_seconds
        return sum(1 for last_active in self.actor_last_active.values() if last_active >= cutoff)

    def get_stats(self):
        per_env_stats = {}
        all_returns, all_lengths, all_step_times = [], [], []
        total_episodes_processed = 0
        total_env_steps = 0
        eval_returns, eval_lengths, eval_step_times = [], [], []
        eval_total_episodes_processed = 0
        eval_total_env_steps = 0

        for env_name, env_data in self.stats.items():
            if not env_data["episode_returns"]:
                per_env_stats[env_name] = {
                    "avg_return": 0.0, "avg_ep_len": 0.0, "avg_success_rate": 0.0,
                    "num_episodes_in_avg": 0, "total_episodes": env_data["total_episodes_processed"]}
                continue
            per_env_stats[env_name] = {
                "avg_return": np.mean(env_data["episode_returns"]),
                "avg_ep_len": np.mean(env_data["episode_lengths"]),
                "avg_success_rate": np.mean(env_data["successes"]),
                "num_episodes_in_avg": len(env_data["episode_returns"]),
                "total_episodes": env_data["total_episodes_processed"]
            }
            if env_name.startswith("eval_"):
                eval_total_episodes_processed += env_data["total_episodes_processed"]
                eval_total_env_steps += env_data["total_env_steps"]
                eval_returns.extend(env_data["episode_returns"])
                eval_lengths.extend(env_data["episode_lengths"])
                eval_step_times.extend(env_data["step_times"])
            else:
                total_episodes_processed += env_data["total_episodes_processed"]
                total_env_steps += env_data["total_env_steps"]
                all_returns.extend(env_data["episode_returns"])
                all_lengths.extend(env_data["episode_lengths"])
                all_step_times.extend(env_data["step_times"])

        per_env_stats["_global_rollout_"] = {
            "avg_return": np.mean(all_returns) if all_returns else 0.0,
            "avg_ep_len": np.mean(all_lengths) if all_lengths else 0.0,
            "avg_step_time": np.mean(all_step_times) if all_step_times else 0.0,
            "total_episodes_processed": total_episodes_processed,
            "total_env_steps": total_env_steps,
            "total_samples_produced": self.total_samples_produced,
            "total_imagine_samples": self.total_imagine_samples,
            "active_actor_count": self.get_active_actor_count(),
            "avg_imagine_reward": np.mean(self.imagine_rewards) if self.imagine_rewards else 0.0
        }
        per_env_stats["_global_eval_"] = {
            "avg_return": np.mean(eval_returns) if eval_returns else 0.0,
            "avg_ep_len": np.mean(eval_lengths) if eval_lengths else 0.0,
            "avg_step_time": np.mean(eval_step_times) if eval_step_times else 0.0,
            "total_episodes_processed": eval_total_episodes_processed,
            "total_env_steps": eval_total_env_steps,
            "window_episodes": len(eval_returns),
            "latest_episode_completed_ns": self.last_eval_completed_ns,
            "latest_episode_policy_versions": self.last_eval_versions,
        }
        timing_stats = {}
        for name, deq in self.timings.items():
            timing_stats[name] = np.mean(deq) if deq else 0.0
        per_env_stats["_timings_"] = timing_stats
        per_env_stats["_global_eval_"]["snapshot_completed_ns"] = now_ns()
        return per_env_stats


# ================================================================
# 2. 经验回放与 Rollout — 与原版一致
# ================================================================
@ray.remote
class ReplayBufferActor:
    def __init__(self, capacity, fresh_config=None):
        setup_debugger("replay")
        self.buffer = deque(maxlen=capacity)
        self.fresh_config = FreshConfig.from_mapping(fresh_config)
        self.fresh_enabled = any((self.fresh_config.refresh_gae,
            self.fresh_config.replay_age_limits, self.fresh_config.real_policy_replay))
        self.fresh_store = SegmentReplay(capacity, self.fresh_config) if self.fresh_enabled else None
        self.fresh_critics = []
        self.fresh_records = {}
        self.fresh_admitted_chunks = {'real':0, 'imagined':0}
        from critic_refresh_plan import dependency_pruning_enabled
        self.fresh_refresh_prune = dependency_pruning_enabled()
        self.fresh_refresh_micro = 2

    def configure_fresh(self, critics, production_budget=None):
        if not self.fresh_enabled:
            return False
        if not critics:
            raise ValueError('FRESH needs the published current Policy critic owners')
        self.fresh_critics = list(critics)
        self.fresh_production_budget = production_budget
        self.fresh_store.on_add = self.capture_ledger.add
        self.fresh_store.on_evict = self.capture_ledger.evict
        return True

    def configure_cost_scheduler(self, broker, settings):
        from policy_service_dispatch import dispatch_settings
        settings = dispatch_settings(settings)
        self.cost_scheduler_enabled = settings['enabled']
        self.cost_scheduler_broker = broker
        self.fresh_refresh_micro = settings['refresh_micro']
        self.cost_admitted_chunks={'real':0,'imagined':0}
        return {'enabled':self.cost_scheduler_enabled}

    def set_fresh_versions(self, policy, world):
        if self.fresh_enabled:
            self.fresh_store.set_versions(policy, world)
            if getattr(self,'cost_scheduler_enabled',False):self._notify_scheduler_inventory()

    def add_segment(self, segment):
        if not self.fresh_enabled or not self.fresh_critics:
            raise RuntimeError('FRESH ordered segment Replay is not configured')
        self.fresh_store.add(segment)
        self.fresh_admitted_chunks[segment.source] += segment.count
        if hasattr(self,'cost_admitted_chunks'):
            self.cost_admitted_chunks[segment.source] += segment.count
        self.telemetry.event('fresh_segment_added', actual_batch=segment.count,
            details={'segment_id':segment.segment_id,'source':segment.source,
                'reward_source':segment.reward_source,'ordered_observations':segment.count+1})

    def scheduler_inventory(self):
        from cost_scheduler_runtime import replay_inventory
        self.cost_inventory_revision=getattr(self,'cost_inventory_revision',0)+1
        return dict(replay_inventory(self.fresh_store,getattr(self,'cost_admitted_chunks',{})),
            inventory_revision=self.cost_inventory_revision,
            inventory_versions=list(self.fresh_store.versions))

    def _notify_scheduler_inventory(self):
        if getattr(self,'cost_scheduler_enabled',False):
            self.cost_scheduler_broker.update_inventory.remote(self.scheduler_inventory())

    def commit_fresh_use(self, token, sample_ids):
        self.fresh_store.commit(token, sample_ids)
        records = self.fresh_records[token]
        for sample_id in sample_ids:
            record = records.pop(sample_id)
            if self.fresh_config.signal_telemetry:
                record_signal(self.telemetry, phase='consumed', **record,
                    consumption_policy_version=self.fresh_store.versions[0],
                    consumption_world_version=self.fresh_store.versions[1])
        if not records:
            del self.fresh_records[token]
        if getattr(self,'cost_scheduler_enabled',False):self._notify_scheduler_inventory()
        return True

    def add_batch(self, batch: List[Experience]):
        if self.fresh_enabled:
            raise RuntimeError('FRESH Replay accepts complete ordered segments, never frozen targets')
        self.buffer.extend(batch)

    def size(self):
        if self.fresh_enabled:
            return len(self.fresh_store.samples)
        return len(self.buffer)

    async def sample(self, batch_size):
        if self.fresh_enabled:
            from fresh_replay_pressure import reserve_with_production_pressure
            token, sample_ids = await reserve_with_production_pressure(self, batch_size)
            try:
                if getattr(self,'cost_scheduler_enabled',False):
                    await self.cost_scheduler_broker.update_inventory.remote(self.scheduler_inventory())
                refreshed = None
                if self.fresh_config.refresh_gae:
                    from critic_refresh_plan import (balanced_refresh_groups, refresh_plan_receipt,
                        dependency_refresh_groups, dependency_refresh_receipt)
                    unique = list(dict.fromkeys(self.fresh_store.samples[sid]['segment_id'] for sid in sample_ids))
                    # The reviewed micro geometry is known only for the enabled
                    # cooperative dispatcher. Other endpoints keep full refresh.
                    prune = (getattr(self, 'cost_scheduler_enabled', False) and
                             self.fresh_refresh_prune)
                    if prune:
                        selected = {sid:[] for sid in unique}
                        for sample_id in sample_ids:
                            sample = self.fresh_store.samples[sample_id]
                            selected[sample['segment_id']].append(sample['index'])
                        groups = dependency_refresh_groups(
                            [self.fresh_store.segments[sid] for sid in unique], selected,
                            len(self.fresh_critics), getattr(self, 'fresh_refresh_micro', 2))
                        receipt = dependency_refresh_receipt(groups)
                    else:
                        groups = balanced_refresh_groups(
                            [(sid, self.fresh_store.segments[sid].observations) for sid in unique],
                            len(self.fresh_critics))
                        receipt = refresh_plan_receipt(groups)
                    version = self.fresh_store.versions[0]
                    self.telemetry.event('fresh_critic_refresh_plan', details={
                        **receipt, 'policy_version':version,
                        'reserved_samples':len(sample_ids)})
                    calls, maps = [], []
                    for critic, group in zip(self.fresh_critics, groups):
                        if prune:
                            if not group.inputs:
                                continue
                            calls.append(critic.scheduled_evaluate_values.remote(
                                group.inputs, version, batch_sizes=group.batch_sizes))
                            maps.append(group)
                            continue
                        if not group:
                            continue
                        inputs = [obs for _, observations in group for obs in observations]
                        calls.append(critic.scheduled_evaluate_values.remote(inputs, version)
                                     if getattr(self, 'cost_scheduler_enabled', False)
                                     else critic.evaluate_values.remote(inputs, version))
                        maps.append([(sid, len(observations)) for sid, observations in group])
                    answers = await asyncio.gather(*calls)
                    refreshed = {}
                    for answer, group in zip(answers, maps):
                        if answer['policy_version'] != version:
                            raise RuntimeError('FRESH critic cohort mixed published Policy versions')
                        if prune:
                            refreshed.update(group.restore(answer['values']))
                            continue
                        offset = 0
                        for segment_id, count in group:
                            refreshed[segment_id] = answer['values'][offset:offset+count]
                            offset += count
                        if offset != len(answer['values']):
                            raise RuntimeError('FRESH critic response differs from ordered K+1 observations')
                result, records = self.fresh_store.materialize(sample_ids, refreshed)
                ledger = self.capture_ledger.reserve(sample_ids)
                metadata = dict(ledger, fresh_reservation_token=token, fresh_versions=list(self.fresh_store.versions),
                    source_counts={source:sum(record['source']==source for record in records)
                                   for source in ('real','imagined')},
                    rewards_refreshed=False, critic_refreshed=bool(refreshed is not None))
                if self.fresh_config.signal_telemetry:
                    for record in records:
                        record_signal(self.telemetry, phase='refreshed', **record)
                self.fresh_records[token] = {record['sample_id']:record for record in records}
                return (*result, metadata)
            except BaseException:
                self.fresh_store.cancel(token)
                if getattr(self,'cost_scheduler_enabled',False):self._notify_scheduler_inventory()
                raise
        batch = random.sample(self.buffer, batch_size)
        obs_list = [b.obs for b in batch]
        action_token = np.stack([b.action_token for b in batch])
        adv = np.asarray([b.advantage for b in batch], np.float32)
        logits_old = np.stack([b.behaviour_logits for b in batch])
        v_targ = np.asarray([b.value_target for b in batch], np.float32)
        return obs_list, action_token, adv, logits_old, v_targ


@ray.remote
class WMReplayBufferActor(EpisodeReplayMixin):
    def __init__(
        self,
        capacity,
        ctrl_eval_capacity=64,
        ctrl_eval_episode_ratio=0.2,
        ctrl_eval_windows_per_episode=8,
        pipeline_config=None, world_config=None, context_frames=7, window_frames=11,
        policy_chunks_per_seed=4, minimum_train_windows=1024,
    ):
        setup_debugger("wm_replay")
        self.buffer = deque(maxlen=capacity)
        self.eval_buffer = []
        self.ctrl_eval_capacity = int(ctrl_eval_capacity)
        self.ctrl_eval_episode_ratio = float(ctrl_eval_episode_ratio)
        self.ctrl_eval_windows_per_episode = int(ctrl_eval_windows_per_episode)
        self.train_episode_ids = set()
        self.eval_episode_ids = set()
        self.sampled_episode_ids = set()
        self.init_pipeline(pipeline_config, world_config, context_frames, window_frames,
                           policy_chunks_per_seed, minimum_train_windows)

    def _is_eval_episode(self, episode_id):
        episode_id = str(episode_id)
        if episode_id in self.eval_episode_ids:
            return True
        if episode_id in self.train_episode_ids:
            return False
        is_eval = stable_episode_fraction(episode_id) < self.ctrl_eval_episode_ratio
        target = self.eval_episode_ids if is_eval else self.train_episode_ids
        target.add(episode_id)
        return is_eval

    def add_batch(self, batch: List[WMExperience]):
        if not batch:
            return
        episode_ids = {str(item.episode_id) for item in batch}
        if len(episode_ids) != 1:
            raise ValueError("add_batch expects windows from exactly one episode")
        if not self._is_eval_episode(next(iter(episode_ids))):
            self.buffer.extend(batch)
            return
        if len(self.eval_buffer) >= self.ctrl_eval_capacity:
            return
        keep = min(
            self.ctrl_eval_windows_per_episode,
            self.ctrl_eval_capacity - len(self.eval_buffer),
            len(batch),
        )
        if keep <= 0:
            return
        indices = np.linspace(0, len(batch) - 1, num=keep, dtype=np.int64)
        self.eval_buffer.extend(batch[int(i)] for i in indices)

    def size(self):
        return len(self.buffer)

    def eval_size(self):
        return len(self.eval_buffer)

    def episode_stats(self):
        return {
            "train": len(self.train_episode_ids),
            "eval": len(self.eval_episode_ids),
        }

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs = np.stack([b.obs for b in batch])
        act = np.stack([b.act for b in batch])
        rew = np.asarray([b.rew for b in batch], np.float32)
        instructions = [b.instruction for b in batch]
        sampled_ids = {str(b.episode_id) for b in batch}
        new_ids = sampled_ids - self.sampled_episode_ids
        self.sampled_episode_ids.update(sampled_ids)
        sample_stats = {
            "unique_episodes": len(sampled_ids),
            "fresh_episode_ratio": len(new_ids) / max(len(sampled_ids), 1),
        }
        return obs, act, rew, instructions, sample_stats

    def sample_eval(self, batch_size):
        if len(self.eval_buffer) < batch_size:
            raise RuntimeError(
                f"Fixed WM eval set has {len(self.eval_buffer)} windows, needs {batch_size}"
            )
        batch = self.eval_buffer[:batch_size]
        obs = np.stack([b.obs for b in batch])
        act = np.stack([b.act for b in batch])
        instructions = [b.instruction for b in batch]
        return obs, act, instructions

    def sample_eval_backend(self, batch_size):
        if len(self.eval_buffer) < batch_size:
            raise RuntimeError("Insufficient held-out backend windows")
        batch = self.eval_buffer[:batch_size]
        if any(not sample.backend_payload for sample in batch):
            raise ValueError("Held-out windows do not contain measured backend state/labels")
        return [dict(sample.backend_payload, obs=sample.obs, actions=sample.act,
                     instruction=sample.instruction) for sample in batch]


class BaseWorkerActor:
    """rollout 和 eval worker 的共享逻辑。"""
    def __init__(self, infer, replay, wid, stats_actor, cfg, benchmark_name):
        self.infer = infer
        self.replay = replay
        self.stats_actor = stats_actor
        self.cfg = cfg
        self.processor = get_processor(cfg)
        self.benchmark_name = benchmark_name
        from rl.libero_env import LiberoEnvWrapper

        self.num_tasks = 1
        print(f"BaseWorker {wid}: 正在初始化 {self.num_tasks} 个 Libero 环境...")
        self.envs = [
            LiberoEnvWrapper(
                benchmark_name=self.benchmark_name,
                task_id=i,
                image_size=224,
                render_mode="rgb_array"
            ) for i in range(self.num_tasks)]
        print(f"BaseWorker {wid}: 环境初始化完成。")

        self.env = None
        self.current_env_idx = -1
        self.wid = wid
        self.task_description = None
        self.current_env_name = None


# ================================================================
# RolloutWorkerActor — ★ 修改：接入 Ctrl-World 无状态接口
# ================================================================
@ray.remote
class RolloutWorkerActor(BaseWorkerActor):
    def __init__(self, infer, replay, wid, stats_actor, cfg, benchmark_name,
                 num_step_cond, imagine_horizon, torch_dtype,
                 reward_infer, ctrl_world_infer, reward_scale, gamma, lambda_,
                 wm_replay_buffer, real_traj_collect_interval, world_config=None,
                 worker_mode="real", pipeline_config=None):
        self.worker_mode = worker_mode
        if worker_mode not in ("real", "imagine"):
            raise ValueError("Unknown producer mode")
        pipeline = PipelineBudgetConfig.from_json(pipeline_config)
        self.initial_state_enabled = pipeline.initial_state_enabled
        self.pipeline_poll_seconds = pipeline.poll_seconds
        self.pipeline_max_episode_frames = pipeline.max_episode_frames
        if worker_mode == "real":
            super().__init__(infer, replay, wid, stats_actor, cfg, benchmark_name)
            if pipeline.real_training_action_horizon is not None:
                from initial_state_env import set_real_training_action_horizon
                for env in self.envs:
                    set_real_training_action_horizon(env, pipeline.real_training_action_horizon)
            if any(env.max_episode_steps + 1 > self.pipeline_max_episode_frames for env in self.envs):
                raise ValueError("Episode frame budget is smaller than the environment horizon")
            if self.initial_state_enabled:
                from initial_state_env import initial_state_pool_metadata
                if len(self.envs) != 1:
                    raise ValueError('Initial-state allocator requires one explicitly identified task')
                self.initial_state_pool = initial_state_pool_metadata(self.envs[0])
        else:
            # Imagination workers own no simulator and no GPU model.
            self.infer, self.replay, self.wid = infer, replay, wid
            self.stats_actor, self.cfg = stats_actor, cfg
            self.processor = get_processor(cfg)
            self.num_tasks, self.envs = 1, []
        setup_debugger("rollout_worker", wid)
        self.env_outcome = [deque(maxlen=100) for _ in range(self.num_tasks)]
        self.local_buffer = []
        # Episode storage belongs to WM Replay; workers retain one in-flight episode only.
        self.num_step_cond = num_step_cond
        self.imagine_horizon = imagine_horizon
        if self.imagine_horizon % NUM_ACTIONS_CHUNK != 0:
            raise ValueError(
                f"imagine_horizon ({self.imagine_horizon}) must be divisible by "
                f"NUM_ACTIONS_CHUNK ({NUM_ACTIONS_CHUNK})"
            )
        self.num_policy_chunks = self.imagine_horizon // NUM_ACTIONS_CHUNK
        self.torch_dtype = torch_dtype
        self.reward_infer = reward_infer
        self.ctrl_world_infer = ctrl_world_infer  # Legacy role name retained in telemetry
        self.world_config = WorldBackendConfig.from_mapping(world_config)
        self.fresh_config = FreshConfig.from_world_config(self.world_config)
        if self.world_config.backend == "weaver":
            geometry = profile_rpc(self, 'ctrl_infer', self.ctrl_world_infer, 'get_context_geometry')
            self.ctrl_world_num_history = geometry["required_context_frames"]
            self.ctrl_world_infer_stride = geometry["horizon"]
            self.ctrl_wm_frames = (geometry["training_window_frames"] if self.world_config.world_trainable
                                   else int(self.world_config.backend_options.get("external_reward_window_frames", 11)))
            if self.num_step_cond != geometry["required_context_frames"]:
                raise ValueError("Collector context differs from the loaded backend geometry")
        else:
            self.ctrl_world_num_history = int(
                profile_rpc(self, 'ctrl_infer', self.ctrl_world_infer, 'get_num_history')
            )
            self.ctrl_world_infer_stride = int(
                profile_rpc(self, 'ctrl_infer', self.ctrl_world_infer, 'get_stride')
            )
            # Offline Ctrl-World windows are [num_history + num_frames] frames.
            self.ctrl_wm_frames = self.ctrl_world_num_history + self.ctrl_world_infer_stride + 1
        self.reward_scale = reward_scale
        self.gamma = gamma
        self.lambda_ = lambda_
        self.wm_replay_buffer = wm_replay_buffer
        self.real_traj_collect_interval = real_traj_collect_interval

    def _reset_and_select_env(self, seed=None, initial_state=None):
        failure_counts = np.array([sum(history) for history in self.env_outcome])
        env_weights = failure_counts + 1
        probabilities = env_weights / np.sum(env_weights)
        self.current_env_idx = np.random.choice(self.num_tasks, p=probabilities)
        self.env = self.envs[self.current_env_idx]
        if initial_state is None:
            obs, info = self.env.reset(seed=seed)
        else:
            from initial_state_env import reset_with_initial_state
            obs, info = reset_with_initial_state(self.env, initial_state['initial_state_id'],
                initial_state['env_seed'], initial_state['max_action_steps'])
        self.task_description = self.env.task_description
        self.current_env_name = self.env.get_name()
        return obs, info

    def run(self):
        return run_producer(self)

    def get_one_episode(self):
        import visualization as visual_preview
        obs_list, reward_list, done_list = [], [], []
        action_env_list = []
        terminated_frames, truncated_frames, success_frames = [False], [False], [False]
        self.capture_episode_count = getattr(self,"capture_episode_count",0)+1
        current_seed = (int(os.environ["ACCERL_SEED"])+self.wid*1000000+self.capture_episode_count) if os.environ.get("ACCERL_SEED") else int(time.time()*1000)+self.wid+os.getpid()
        self.capture_episode_id = f"worker{self.wid}:seed{current_seed}"
        assigned = self.pipeline_job.get('initial_state') if self.initial_state_enabled else None
        if self.initial_state_enabled and assigned is None:
            raise RuntimeError('Real collection started without its explicit initial-state assignment')
        if assigned is not None:
            current_seed = assigned['env_seed']
            self.capture_episode_id = f"real:allocation{assigned['allocation_id']}"
        obs, info = self._reset_and_select_env(seed=current_seed, initial_state=assigned)
        self.collection_metadata = dict(info) if assigned is not None else {}
        self.collection_policy_versions = set()
        self.collection_started_ns = time.time_ns()
        preview = visual_preview.begin('real', self.wid, self.capture_episode_id,
            task=self.task_description, deterministic=False,
            **visual_preview.env_metadata(self.env, info))
        preview.add(obs, 0)
        if self.world_config.backend == "weaver":
            if not math.isclose(environment_dt(self.env), self.world_config.schema["dt_seconds"], rel_tol=1e-6):
                raise ValueError("Live environment timestep differs from the WEAVER checkpoint schema")
        self.capture_real_frame_groups = []
        real_prefix = ()
        obs_list.append(obs)
        step_count = 0
        reward_sum, time_start, step_count_total = 0.0, time.time(), 0
        fresh_real_segment = []
        while True:
            frame_before = step_count_total
            lineage_begin(self, "real_action_chunk", real_prefix,
                          episode_id=self.capture_episode_id, start_frame=frame_before)
            inputs_t = prepare_one_obs(self.cfg, self.processor, obs, self.task_description, self.torch_dtype)
            act_norm, action_env, action_token, logits, value = profile_rpc(self, 'policy_infer', self.infer, 'request', inputs_t, deterministic=False, _capture_context={'workflow': 'real_rollout'})
            version = getattr(self, 'capture_last_versions', {}).get('policy')
            if version is not None:
                self.collection_policy_versions.add(int(version))
            step_count += 1
            chunk_reward, done = 0.0, False
            for i in range(len(action_env)):
                single_action = action_env[i]
                nxt, r, term, trunc, info = self.env.step(single_action)
                obs_list.append(nxt)
                if len(obs_list) > self.pipeline_max_episode_frames:
                    raise RuntimeError("Environment exceeded its reserved episode frame budget")
                reward_list.append(r)
                done_list.append(term or trunc)
                terminated_frames.append(bool(term))
                truncated_frames.append(bool(trunc))
                if self.world_config.backend == "weaver" and "is_success" not in info:
                    raise ValueError("WEAVER labels require per-frame environment is_success")
                success_frames.append(bool(info.get("is_success", False)))
                action_env_list.append(np.asarray(action_env[i], dtype=np.float32))
                reward_sum += r
                chunk_reward += r * self.reward_scale
                step_count_total += 1
                preview.add(nxt, step_count_total, force=bool(term or trunc),
                            policy_version=version, reward=float(r))
                if term or trunc:
                    done = True; break
            real_group = lineage_end(self)
            self.capture_real_frame_groups.append({"group_id": real_group,
                "start_frame": frame_before, "end_frame": step_count_total})
            if self.fresh_config.real_policy_replay:
                if version is None:
                    raise RuntimeError('Real Policy Replay requires actual served behaviour version')
                fresh_real_segment.append((inputs_t, np.asarray(action_token).copy(),
                    float(bool(info.get('is_success', False))) * self.reward_scale,
                    np.asarray(logits).copy(), value,
                    {'policy':int(version),'lineage_group_id':real_group,
                     'executed_actions':step_count_total-frame_before,
                     'phi_before':None,'phi_after':None}))
            real_prefix = (real_group,)
            obs = nxt
            if done:
                step_time = (time.time() - time_start) / max(step_count_total, 1)
                success = float(info.get('is_success', 0.0))
                budget_cut = bool(assigned is not None and trunc and not term and
                    assigned['max_action_steps'] < self.collection_metadata['task_action_horizon'])
                preview.finish(success=None if budget_cut else bool(success), budget_censored=budget_cut,
                    terminated=bool(term), truncated=bool(trunc), action_steps=step_count_total,
                    episode_return=float(reward_sum), policy_versions=sorted(self.collection_policy_versions))
                if assigned is not None:
                    self.collection_metadata.update(
                        termination_reason='success' if term else 'real_env_budget' if budget_cut else 'task_horizon',
                        budget_censored=budget_cut, action_steps=step_count_total,
                        started_ns=self.collection_started_ns, completed_ns=time.time_ns(),
                        policy_versions=sorted(self.collection_policy_versions), deterministic=False,
                        allocation_id=assigned['allocation_id'], allocation_cycle=assigned['cycle'])
                self.env_outcome[self.current_env_idx].append(1.0 - success)
                if not budget_cut:
                    self.stats_actor.add_episode_return.remote(
                        self.current_env_name, reward_sum, step_time,
                        step_count_total, success, actor_id=self.wid, step_num=step_count)
                if self.fresh_config.real_policy_replay:
                    final_inputs = prepare_one_obs(self.cfg, self.processor, nxt, self.task_description, self.torch_dtype)
                    # LIBERO's complete finite task horizon is an observed
                    # failed task, even though the simulator reports truncated.
                    # Only an earlier collection-budget censoring has unknown
                    # future outcome and may bootstrap beyond the recorded data.
                    learning_terminal = bool(term or (trunc and not budget_cut))
                    learning_truncated = bool(trunc and budget_cut)
                    fresh_real_segment[-1][5].update(env_terminated=bool(term),env_truncated=bool(trunc),
                        learning_terminal=learning_terminal,learning_truncated=learning_truncated,
                        termination_reason=('success' if success else 'real_env_budget' if budget_cut
                                            else 'task_horizon' if trunc else 'environment_terminated'))
                    if learning_terminal:
                        endpoint_value = 0.
                    else:
                        endpoint = ray.get(self.infer.scheduled_evaluate_values.remote([final_inputs]))
                        endpoint_value = endpoint['values'][0]
                    self._process_traj(fresh_real_segment, endpoint_value, bootstrap_obs=final_inputs,
                        terminated=learning_terminal,truncated=learning_truncated,source='real')
                self._process_episode_for_wm(
                    obs_list,
                    reward_list,
                    action_env_list,
                    self.task_description,
                    success,
                    episode_id=self.capture_episode_id,
                    frame_labels=(terminated_frames, truncated_frames, success_frames),
                )
                break

    def _obs_to_views(self, obs):
        return torch.stack(
            [
                image_to_tensor(obs["full_image"], "cpu"),
                image_to_tensor(obs["wrist_image"], "cpu"),
            ],
            dim=0,
        )

    def _process_episode_for_wm(self, obs_list, reward_list, action_env_list,
                              task_description, success=None, episode_id=None, frame_labels=None):
        from episode_frame_store import encode_observations
        frames = encode_observations(obs_list)
        actions = frame_align_transition_actions(np.asarray(action_env_list, dtype=np.float32), len(frames))
        states = None
        if self.world_config.backend == "weaver":
            if frame_labels is None or any("state" not in obs for obs in obs_list):
                raise ValueError("WEAVER requires measured per-frame states and labels")
            states = np.stack([np.asarray(obs['state'], dtype=np.float32) for obs in obs_list])
        if frame_labels is None:
            raise ValueError("Episode frame labels must be supplied by actual environment steps")
        for start in range(max(0, len(frames) - self.ctrl_wm_frames + 1)):
            sample_id = f"{episode_id}:window{start}"
            parents = lineage_frame_parent(self.capture_real_frame_groups, start + self.ctrl_wm_frames - 1)
            lineage_sample(self, sample_id, parents, replay="wm", episode_id=str(episode_id),
                sample_index=start, observation_start=start,
                observation_end_exclusive=start + self.ctrl_wm_frames,
                model_versions_field="legacy_last_seen_not_window_version_truth",
                consumers=(["reward_train_or_wm_eval_holdout"] if not self.world_config.world_trainable
                           else (["ctrl_train", "reward_train_or_wm_eval_holdout"]
                                 if self.world_config.reward_provider == "external_vlm" else ["ctrl_train_or_wm_eval_holdout"])))
        payload = dict(episode_id=str(episode_id), instruction=task_description, frames=frames,
            actions=actions, rewards=np.asarray([0.0] + list(reward_list), dtype=np.float64),
            states=states, terminated=np.asarray(frame_labels[0], dtype=bool),
            truncated=np.asarray(frame_labels[1], dtype=bool), success=np.asarray(frame_labels[2], dtype=bool),
            versions=dict(getattr(self, 'capture_last_versions', {})),
            frame_groups=list(self.capture_real_frame_groups))
        if self.initial_state_enabled:
            payload['metadata'] = self.collection_metadata
        payload['metadata']=dict(payload.get('metadata',{}),label_source='libero_env_step')
        # Await ingestion: pending RPCs cannot become an unbounded second buffer.
        return ray.get(self.wm_replay_buffer.finish_real_episode.remote(self.pipeline_lease, payload))

    def _process_traj(self, traj_segment, bootstrap_val, *, bootstrap_obs=None,
                      terminated=False, truncated=True, source='imagined', seed_anchor=None):
        fresh = self.fresh_config
        segmented = any((fresh.refresh_gae, fresh.replay_age_limits, fresh.real_policy_replay))
        if segmented:
            if bootstrap_obs is None:
                raise ValueError('FRESH trajectories require their actual K+1 endpoint observation')
            count = len(traj_segment)
            masks = np.ones((count, NUM_ACTIONS_CHUNK, ACTION_DIM), dtype=bool)
            for index, row in enumerate(traj_segment):
                executed = int(row[5].get('executed_actions', NUM_ACTIONS_CHUNK))
                if not 1 <= executed <= NUM_ACTIONS_CHUNK:
                    raise ValueError('FRESH actual executed chunk length is invalid')
                masks[index, executed:] = False
            versions = [dict(row[5]) for row in traj_segment]
            phi = [versions[0].get('phi_before')] + [version.get('phi_after') for version in versions]
            segment = PolicySegment.create(segment_id=new_id(source+'-segment'),
                episode_id=self.capture_episode_id, source=source,
                observations=[row[0] for row in traj_segment]+[bootstrap_obs],
                action_tokens=np.stack([row[1] for row in traj_segment]),
                behaviour_logits=np.stack([row[3] for row in traj_segment]),
                rewards=[row[2] for row in traj_segment], generation_values=[float(np.asarray(row[4]).reshape(-1)[0])
                    for row in traj_segment]+[float(np.asarray(bootstrap_val).reshape(-1)[0])],
                terminated=[False]*(count-1)+[bool(terminated)],
                truncated=[False]*(count-1)+[bool(truncated)],
                versions=versions, gamma=self.gamma, lambda_=self.lambda_, action_mask=masks,
                phi=phi, seed_anchor=seed_anchor,
                reward_source='environment_success' if source=='real' else 'generation_time_native_score_delta')
            if fresh.signal_telemetry:
                from fresh_policy_replay import generalized_advantages
                adv, targets = generalized_advantages(segment.rewards, segment.generation_values,
                    segment.terminated, segment.truncated, segment.gamma, segment.lambda_)
                for index, version in enumerate(versions):
                    record_signal(self.telemetry, phase='generated', source=source,
                        sample_id=segment.sample_id(index), segment_id=segment.segment_id,
                        reward=float(segment.rewards[index]), advantage=float(adv[index]),
                        value_target=float(targets[index]), phi_before=phi[index],phi_after=phi[index+1],
                        generation_policy_version=version['policy'],
                        generation_world_version=version.get('world_bundle_version'),
                        generation_rm_version=version.get('world_reward_bundle_version'),
                        reward_source=segment.reward_source,seed_anchor=seed_anchor,
                        learning_terminal=bool(segment.terminated[index]),learning_truncated=bool(segment.truncated[index]),
                        env_terminated=version.get('env_terminated'),env_truncated=version.get('env_truncated'),
                        termination_reason=version.get('termination_reason'),
                        executed_actions=int(masks[index,:,0].sum()))
            for index, version in enumerate(versions):
                lineage_sample(self,segment.sample_id(index),(version['lineage_group_id'],),
                    replay='policy',episode_id=self.capture_episode_id,sample_index=index,
                    segment_id=segment.segment_id,source=source,consumers=['policy_train'])
            ray.get(self.replay.add_segment.remote(segment))
            if source == 'imagined':
                self.pipeline_generated_samples = getattr(self,'pipeline_generated_samples',0) + segment.count
            return
        rets, advs = [], []
        gae = 0.0
        for i in reversed(range(len(traj_segment))):
            _, _, r, _, v, _ = traj_segment[i]
            nv = bootstrap_val if i == len(traj_segment) - 1 else traj_segment[i+1][4]
            delta = r + self.gamma * nv - v
            gae = delta + self.gamma * self.lambda_ * gae
            advs.append(gae)
            rets.append(gae + v)
        advs.reverse(); rets.reverse()
        advs_np = np.array(advs, dtype=np.float32)
        # GAE/return targets depend on later rewards/values and bootstrap as well
        # as the prefix that produced each imagined observation. Record a DAG,
        # not a quadratic expansion of every earlier request into every sample.
        suffix = tuple(getattr(self, "capture_bootstrap_lineage", ()))
        target_groups = [None] * len(traj_segment)
        for index in reversed(range(len(traj_segment))):
            lineage_begin(self, "gae_target", (traj_segment[index][5]["lineage_group_id"], *suffix),
                          trajectory_index=index, target="advantage_and_value_return")
            target_groups[index] = lineage_end(self)
            suffix = (target_groups[index],)
        batch = []
        for i, (s, a_token, _, logits, _, capture_versions) in enumerate(traj_segment):
            batch.append(Experience(
                obs=s,
                action_token=a_token.astype(np.int64),
                advantage=float(advs_np[i]),
                behaviour_logits=logits.astype(np.float32),
                value_target=float(rets[i]),
                episode_id=str(getattr(self,"capture_episode_id","")),
                sample_index=i,
                model_versions=capture_versions,
            ))
            lineage_sample(self, batch[-1].sample_id,
                (capture_versions["lineage_group_id"], target_groups[i]), replay="policy",
                episode_id=batch[-1].episode_id, sample_index=i,
                consumers=["policy_train"], includes_bootstrap=bool(getattr(self, "capture_bootstrap_lineage", ())))
            if fresh.signal_telemetry:
                record_signal(self.telemetry,phase='generated',source=source,sample_id=batch[-1].sample_id,
                    reward=float(traj_segment[i][2]),advantage=float(advs_np[i]),value_target=float(rets[i]),
                    phi_before=capture_versions.get('phi_before'),phi_after=capture_versions.get('phi_after'),
                    generation_policy_version=capture_versions.get('policy'),
                    generation_world_version=capture_versions.get('world_bundle_version'),
                    generation_rm_version=capture_versions.get('world_reward_bundle_version'),seed_anchor=seed_anchor,
                    reward_source='generation_time_native_score_delta' if source=='imagined' else 'environment_success')
        ray.get(self.replay.add_batch.remote(batch))
        self.pipeline_generated_samples = getattr(self, "pipeline_generated_samples", 0) + len(batch)

    def predict_next_chunk(
        self, current_latent, latent_history, action_condition, task_description
    ):
        future_obs, future_latents = profile_rpc(self, 'ctrl_infer', self.ctrl_world_infer, 'request', current_latent, latent_history, action_condition.float(), task_description, _capture_context={'workflow': 'imagined_generation'})
        return future_obs, future_latents

    def predict_rew_end(self, next_obs, task_description):
        if next_obs.ndim != 4:
            raise ValueError(f"Expected multi-view observation, got {tuple(next_obs.shape)}")
        reward_obs = {"full_image": tensor_to_image(next_obs[0])}
        inputs = prepare_one_obs(
            self.cfg, self.processor, reward_obs, task_description, self.torch_dtype
        )
        logits = profile_rpc(self, 'reward_infer', self.reward_infer, 'request', inputs, _capture_context={'workflow': 'imagined_generation'})
        probs = torch.softmax(logits, dim=-1)
        succ_prob = probs[1].item()
        end = logits.argmax().item()
        return succ_prob, end

    def obs2inp(self, obs, task_description):
        if obs.ndim != 4 or obs.shape[0] < 2:
            raise ValueError(f"Expected [M,C,H,W] with two views, got {tuple(obs.shape)}")
        obs_for_vla = {
            "full_image": tensor_to_image(obs[0]),
            "wrist_image": tensor_to_image(obs[1]),
        }
        inputs = prepare_one_obs(self.cfg, self.processor, obs_for_vla, task_description, self.torch_dtype)
        return inputs


# ================================================================
# EvaluationWorkerActor — 与原版完全一致
# ================================================================
@ray.remote
class EvaluationWorkerActor(BaseWorkerActor):
    def __init__(self, infer, wid, stats_actor, cfg, benchmark_name, torch_dtype):
        super().__init__(infer, None, wid, stats_actor, cfg, benchmark_name)
        setup_debugger("eval_worker", self.wid)
        self.torch_dtype = torch_dtype
        print(f"EvaluationWorker {self.wid}: 环境初始化完成。")

    def _reset_and_select_env(self, seed=None):
        self.current_env_idx = (self.current_env_idx + 1) % self.num_tasks
        self.env = self.envs[self.current_env_idx]
        obs, info = self.env.reset(seed=seed)
        self.task_description = self.env.task_description
        self.current_env_name = self.env.get_name()
        return obs, info

    def run(self):
        import visualization as visual_preview
        try:
            self.capture_episode_count = getattr(self,"capture_episode_count",0)+1
            current_seed = (int(os.environ["ACCERL_SEED"])+50000000+self.wid*1000000+self.capture_episode_count) if os.environ.get("ACCERL_SEED") else int(time.time()*1000)+os.getpid()+random.randint(0,10000)
            self.capture_episode_id = f"eval{self.wid}:seed{current_seed}"
            obs, info = self._reset_and_select_env(seed=current_seed)
            while True:
                reward_sum, time_start, step_count_total, done = 0.0, time.time(), 0, False
                episode_policy_versions = set()
                preview = visual_preview.begin('evaluation', self.wid, self.capture_episode_id,
                    task=self.task_description, deterministic=True, evaluation_kind='online_cross_version',
                    **visual_preview.env_metadata(self.env, info))
                preview.add(obs, 0)
                while not done:
                    inputs_t = prepare_one_obs(self.cfg, self.processor, obs, self.task_description, self.torch_dtype)
                    _, action_env, _, _, _ = profile_rpc(self, 'policy_infer', self.infer, 'request', inputs_t, deterministic=True, _capture_context={'workflow': 'evaluation'})
                    episode_policy_versions.add(int(self.capture_last_versions.get("policy", -1)))
                    for i in range(len(action_env)):
                        single_action = action_env[i]
                        obs, r, term, trunc, info = self.env.step(single_action)
                        reward_sum += r; step_count_total += 1
                        preview.add(obs, step_count_total, force=bool(term or trunc),
                            policy_version=int(self.capture_last_versions.get('policy', -1)), reward=float(r))
                        if term or trunc: done = True; break
                completed_ns = now_ns()
                step_time = (time.time() - time_start) / max(step_count_total, 1)
                success = float(info.get('is_success', 0.0))
                preview.finish(success=bool(success), terminated=bool(term), truncated=bool(trunc),
                    action_steps=step_count_total, episode_return=float(reward_sum),
                    policy_versions=sorted(episode_policy_versions))
                self.stats_actor.add_episode_return.remote(
                    f"eval_{self.current_env_name}", reward_sum, step_time,
                    step_count_total, success, actor_id=None, step_num=step_count_total,
                    completed_ns=completed_ns, evaluation_versions=sorted(episode_policy_versions))
                self.capture_episode_count = getattr(self,"capture_episode_count",0)+1
                current_seed = (int(os.environ["ACCERL_SEED"])+50000000+self.wid*1000000+self.capture_episode_count) if os.environ.get("ACCERL_SEED") else int(time.time()*1000)+os.getpid()+random.randint(0,10000)
                self.capture_episode_id = f"eval{self.wid}:seed{current_seed}"
                obs, info = self._reset_and_select_env(seed=current_seed)
        except Exception as e:
            import traceback
            print(f"[ERROR] EvaluationWorker {self.wid} run() 崩溃: {e}", flush=True)
            traceback.print_exc(); raise


# ================================================================
# 3. 推理器 (InferenceActor) — 与原版完全一致
# ================================================================
class ColocatedInferenceMixin:
    """Bounded queues and CPU weight transport for same-card model roles."""
    def _fail_pending_requests(self, error):
        self.inference_fatal_error = error
        for promise in self.promises:
            if not promise.done():
                promise.set_exception(error)
        self.requests, self.promises = [], []

    def _check_inference_health(self):
        error = getattr(self, "inference_fatal_error", None)
        if error is not None:
            raise RuntimeError("Inference background loop failed") from error

    def set_training_batch_limit(self, maximum):
        maximum = int(maximum)
        if maximum < 1:
            raise ValueError("Inference batch limit must be positive")
        self.training_batch_limit = maximum
        return maximum

    def apply_rank_weights(self, payload, kind):
        if kind == "ctrl" and self.ema_decay != 0:
            raise ValueError("Same-card Ctrl transport requires the unchanged EMA=0 recipe")
        ipc = kind in ("policy", "ctrl")
        version = payload["inventory"]["version"] if ipc else payload["version"]
        try:
            with self.telemetry.span("broadcast_receive", cuda=True,
                                     model_version=version,
                                     transport="cuda_ipc" if ipc else "cpu_ray", kind=kind):
                if ipc:
                    result = apply_cuda_ipc_weights(
                        self.model, payload, kind, self.gpu_binding["uuid"],
                        os.environ["ACCERL_RUN_ID"], int(self.actor_id),
                        world_backend=getattr(self, "world_backend", None) if kind == "ctrl" else None)
                else:
                    result = apply_cpu_weights(self.model, payload, kind)
        except BaseException as error:
            # A partial weight copy must never be served as a valid old version.
            self._fail_pending_requests(error)
            raise
        self.capture_model_version = int(result["version"])
        self.model_version = self.capture_model_version
        if kind == "ctrl" and getattr(self, "world_backend", None) is not None:
            self.world_backend.set_published_version(self.model_version)
        if kind == "ctrl":
            self.ema_initialized = True
            self.ema_updates += 1
        self.model.eval()
        self.telemetry.event("model_version", model_version=self.capture_model_version,
                             details={"meaning": "successful_source_optimizer_updates", **result})
        _trim_cuda_cache("idle")
        return result


def _trim_cuda_cache(phase):
    if os.environ.get("ACCERL_TRIM_" + phase.upper() + "_CACHE") == "1":
        torch.cuda.empty_cache()


@ray.remote(num_gpus=1)
class InferenceActor(ColocatedInferenceMixin, InferenceActorCom):
    def __init__(self, actor_id, cfg, stats_actor, torch_dtype, inference_batch, inference_timeout_ms):
        super().__init__()
        setup_debugger("inference", actor_id)
        self.gpu_binding = verify_bound_gpu()
        self.actor_id = actor_id
        print(f"InferenceActor {actor_id}: 正在加载 OpenVLA ActorCritic...")
        self.model = ActorCritic(cfg, torch_dtype=torch_dtype)
        self.model.cuda()
        self.model.eval()
        self.processor = self.model.processor
        self.cfg = cfg
        self.stats_actor = stats_actor

        self.batch_size = inference_batch
        self.timeout_sec = inference_timeout_ms / 1000.0
        self.requests, self.promises = [], []
        self._policy_dispatch = None
        self.last_process_time = time.time()

        loop = asyncio.get_event_loop()
        self._bg_task = loop.create_task(self._loop())
        self._bg_task.add_done_callback(self._on_bg_task_done)
        print(f"InferenceActor {self.actor_id} 初始化于 GPU: {ray.get_gpu_ids()} (批次超时: {inference_timeout_ms}ms)")

    def get_model_keys(self):
        if self.model is None:
            return {}
        sd = self.model.state_dict()
        res = {k: float(v.abs().sum().item()) for k, v in sd.items()}
        return res

    def _on_bg_task_done(self, task):
        try:
            task.result()
        except Exception as e:
            import traceback
            print(f"[ERROR] InferenceActor {self.actor_id} 后台任务异常: {e}", flush=True)
            traceback.print_exc()
            self._fail_pending_requests(e)

    async def request(self, inputs_t, deterministic=False):
        self._check_inference_health()
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        if self._policy_dispatch is not None:
            metadata = current_request_metadata()
            request_id = self._policy_dispatch.enqueue_action(inputs_t, deterministic, metadata, fut)
            try:
                return await fut
            except asyncio.CancelledError:
                self._policy_dispatch.cancel_queued_action(request_id)
                raise
        self.requests.append((inputs_t, deterministic))
        self.promises.append(fut)
        return await fut

    def evaluate_values(self, inputs, expected_version=None, *, batch_sizes=None):
        """Bounded critic-only forward; never sample actions or advance RNG."""
        self._check_inference_health()
        if getattr(self, '_policy_dispatch', None) is not None:
            raise RuntimeError('Scheduled critics must use the cooperative async value endpoint')
        expected_version = self.capture_model_version if expected_version is None else expected_version
        if (type(expected_version) is not int or expected_version != self.capture_model_version
                or not isinstance(inputs, (list, tuple)) or not 1 <= len(inputs) <= 4096):
            raise ValueError('FRESH critic refresh needs a bounded batch and exact published version')
        limit = min(2, max(1, int(getattr(self, 'training_batch_limit', 2))))
        from policy_service_dispatch import refresh_batch_sizes
        sizes = refresh_batch_sizes(len(inputs), limit, batch_sizes)
        values = []
        with torch.inference_mode():
            start = 0
            for count in sizes:
                batch = self.model.prepare_inputs_batch(inputs[start:start+count])
                _, value = self.model(batch)
                current = value.detach().float().reshape(-1).cpu().numpy()
                if len(current) != count or not np.isfinite(current).all():
                    raise ValueError('FRESH critic returned invalid per-observation values')
                values.extend(float(number) for number in current)
                start += count
        if self.capture_model_version != expected_version:
            raise RuntimeError('FRESH critic changed version during ordered segment evaluation')
        self.telemetry.event('fresh_critic_refresh', actual_batch=len(values), model_version=expected_version,
            details={'action_sampling':False,'maximum_micro_batch':limit})
        return {'policy_version':expected_version,'values':values}

    def configure_cost_scheduler(self, broker, settings):
        from policy_service_dispatch import PolicyServiceDispatch, dispatch_settings
        settings = dispatch_settings(settings)
        if self.requests or self.promises or (self._policy_dispatch is not None and
                (self._policy_dispatch.pending_count or self._policy_dispatch.active is not None)):
            raise RuntimeError('Policy scheduler configuration requires an idle actor queue')
        self.cost_scheduler_broker = broker
        self._policy_dispatch = PolicyServiceDispatch(settings) if settings['enabled'] else None
        return {'enabled':settings['enabled'],'classes':['real','imagined','refresh','eval'],
                'shared_model':True,'maximum_refresh_micro':settings['refresh_micro']}

    async def apply_rank_weights(self, payload, kind):
        dispatcher = getattr(self, '_policy_dispatch', None)
        if kind == 'policy' and dispatcher is not None:
            from cost_scheduler_runtime import publish_after_refresh
            apply = super().apply_rank_weights
            return await publish_after_refresh(self, lambda: apply(payload, kind))
        return super().apply_rank_weights(payload, kind)

    async def _release_refresh_demand(self, remaining):
        if remaining:
            await self.cost_scheduler_broker.refresh_demand.remote(-remaining)

    def _fail_pending_requests(self, error):
        dispatcher = getattr(self, '_policy_dispatch', None)
        if dispatcher is not None:
            remaining = dispatcher.fail_all(error)
            if remaining:
                asyncio.get_event_loop().create_task(self._release_refresh_demand(remaining))
        super()._fail_pending_requests(error)

    async def scheduled_evaluate_values(self, inputs, expected_version=None, *, batch_sizes=None):
        self._check_inference_health()
        while getattr(self, '_policy_publication_pending', False):
            await asyncio.sleep(.0005)
            self._check_inference_health()
        if self._policy_dispatch is None:
            if batch_sizes is None:
                return self.evaluate_values(inputs, expected_version)
            return self.evaluate_values(inputs, expected_version, batch_sizes=batch_sizes)
        version = self.capture_model_version if expected_version is None else expected_version
        if type(version) is not int or version != self.capture_model_version:
            raise ValueError('Scheduled critic requires the exact published Policy version')
        future = asyncio.get_running_loop().create_future()
        job = self._policy_dispatch.enqueue_refresh(inputs, version, future, batch_sizes=batch_sizes)
        from cost_scheduler_runtime import finish_owned_rpc
        registration = asyncio.ensure_future(
            self.cost_scheduler_broker.refresh_demand.remote(len(job.inputs)))
        try:
            await asyncio.shield(registration)
            job.reported_remaining = len(job.inputs)
            job.ready = True
        except asyncio.CancelledError:
            # Registration may already have changed the remote counter. Keep
            # the version pin and let the bounded job finish for this caller.
            try:
                await finish_owned_rpc(registration)
            except BaseException as error:
                self._fail_pending_requests(error)
                if future.done() and not future.cancelled():future.exception()
                raise
            job.reported_remaining = len(job.inputs)
            job.ready = True
            future.add_done_callback(lambda done: None if done.cancelled() else done.exception())
            raise
        except BaseException as error:
            self._fail_pending_requests(error)
            if future.done() and not future.cancelled():
                future.exception()
            raise
        self.telemetry.event('fresh_critic_refresh_requested',actual_batch=len(job.inputs),
            model_version=version,details={'scheduler_job_id':job.job_id,'maximum_micro_batch':
                self._policy_dispatch.settings['refresh_micro'],'ordered_current_version':True})
        # A caller cancellation does not preempt a selected micro or lose its
        # version pin. The bounded background job continues to its safe end.
        return await asyncio.shield(future)

    async def _scheduled_policy_step(self):
        self._check_inference_health()
        dispatcher = self._policy_dispatch
        batch = dispatcher.next_batch(self.capture_model_version,
            action_limit=min(2,int(getattr(self,'training_batch_limit',2))))
        if batch is None:
            await asyncio.sleep(.0005)
            return
        spec = {'work_id':batch.batch_id,'service':'policy_infer.'+batch.request_class,
            'cost_key':'policy_infer:'+batch.request_class+':batch'+str(batch.units),
            'units':batch.units,'replica':'policy:'+str(self.actor_id),
            'resources':('card:'+str(self.gpu_binding['physical_gpu']),),
            'metadata':{'policy_version_at_submission':self.capture_model_version,
                        'request_class':batch.request_class,'refresh_job_id':None if
                            batch.refresh_job is None else batch.refresh_job.job_id}}
        acquired = False
        completed = False
        capture_batch = None
        started = None
        try:
            from cost_scheduler_runtime import acquire_inference
            await acquire_inference(self.cost_scheduler_broker, spec)
            acquired = True
            started = time.perf_counter()
            if batch.refresh_job is not None:
                job = batch.refresh_job
                if self.capture_model_version != job.version:
                    raise RuntimeError('Policy version changed before scheduled critic execution')
                with torch.inference_mode():
                    prepared = self.model.prepare_inputs_batch(list(batch.inputs))
                    _, value = self.model(prepared)
                    values = value.detach().float().reshape(-1).cpu().numpy()
                if len(values) != batch.units or not np.isfinite(values).all():
                    raise ValueError('Scheduled critic returned invalid values')
                result_values = [float(x) for x in values]
            else:
                requests = [(x.inputs,x.deterministic) for x in batch.actions]
                promises = [x.future for x in batch.actions]
                capture_batch = batch_begin(self,requests,promises,selected_metadata=batch.metadata)
                capture_stage = stage_begin(self,'input_prepare',actual_batch=batch.units)
                prepared = self.model.prepare_inputs_batch(list(batch.inputs))
                stage_end(self,capture_stage)
                capture_core = stage_begin(self,'gpu_forward_and_copy',actual_batch=batch.units,cuda=True)
                with torch.inference_mode():
                    action_logits,value = self.model(prepared)
                    _,action_tokens_all,normalized_actions_all = self.model.post_process(
                        action_logits,deterministic=[x.deterministic for x in batch.actions])
                    tokens = action_tokens_all.view(-1,NUM_ACTIONS_CHUNK,ACTION_DIM).cpu().numpy()
                    logits = action_logits.view(-1,NUM_ACTIONS_CHUNK,ACTION_DIM,action_logits.shape[-1]).float().cpu().numpy()
                    values = value.to(torch.float32).cpu().numpy()
                stage_end(self,capture_core)
                results = []
                for i in range(batch.units):
                    action_env = self.model.vla._unnormalize_actions(normalized_actions_all[i],self.cfg.unnorm_key).astype(np.float32)
                    results.append((normalized_actions_all[i],action_env,tokens[i],logits[i],values[i]))
                batch_end(self,capture_batch)
            elapsed = time.perf_counter()-started
            await self.cost_scheduler_broker.complete.remote(batch.batch_id,elapsed,
                completed_units=batch.units,success=True)
            completed = True
            dispatcher.complete_batch(batch,elapsed)
            if batch.refresh_job is not None:
                job.reported_remaining -= batch.units
                await self.cost_scheduler_broker.refresh_demand.remote(-batch.units)
                if dispatcher.finish_refresh(batch,result_values,self.capture_model_version):
                    self.telemetry.event('fresh_critic_refresh',actual_batch=len(job.inputs),
                        model_version=job.version,details={'action_sampling':False,
                            'maximum_micro_batch':dispatcher.settings['refresh_micro'],
                            'cooperatively_scheduled':True,'scheduler_job_id':job.job_id})
            else:
                for action,result in zip(batch.actions,results):
                    if not action.future.done():
                        action.future.set_result(result)
                    else:
                        self.capture_served_versions.pop(action.metadata['request_id'],None)
                self.stats_actor.add_timing_metric.remote('Inference/loop_time_s',elapsed)
            _trim_cuda_cache('idle')
            # Only this bounded micro is complete; other already-submitted GPU
            # work is not drained and independent replicas need not form a gang.
            await asyncio.sleep(0)
        except BaseException as error:
            batch_end(self,capture_batch,error=error)
            if acquired and not completed:
                cost = 0.0 if started is None else max(0.,time.perf_counter()-started)
                try:
                    await self.cost_scheduler_broker.complete.remote(batch.batch_id,cost,
                        completed_units=0,success=False)
                except BaseException:
                    pass
            self._fail_pending_requests(error)
            raise

    async def _loop(self):
        while True:
            if self._policy_dispatch is not None:
                await self._scheduled_policy_step()
                continue
            should_process = self.requests and (
                len(self.requests) >= self.batch_size or
                time.time() - self.last_process_time > self.timeout_sec
            )
            if not should_process:
                await asyncio.sleep(0.0005)
                continue

            count = min(len(self.requests), getattr(self, "training_batch_limit", len(self.requests)))
            requests_to_process, self.requests = self.requests[:count], self.requests[count:]
            promises_to_process, self.promises = self.promises[:count], self.promises[count:]
            capture_batch = batch_begin(self, requests_to_process, promises_to_process)
            self.last_process_time = time.time()

            inputs_list = [r[0] for r in requests_to_process]
            deterministic_flags = [r[1] for r in requests_to_process]
            t_loop_start = time.time()
            try:
                capture_stage = stage_begin(self, "input_prepare", actual_batch=len(inputs_list))
                inputs_batch = self.model.prepare_inputs_batch(inputs_list)
                stage_end(self, capture_stage)
                capture_core = stage_begin(self, "gpu_forward_and_copy", actual_batch=len(inputs_list), cuda=True)
                with torch.inference_mode():
                    action_logits, value = self.model(inputs_batch)
                    _, action_tokens_all, normalized_actions_all = self.model.post_process(
                        action_logits, deterministic=deterministic_flags)
                    action_tokens = action_tokens_all.view(
                        -1, NUM_ACTIONS_CHUNK, ACTION_DIM).cpu().numpy()
                    logits = action_logits.view(
                        -1, NUM_ACTIONS_CHUNK, ACTION_DIM, action_logits.shape[-1]
                    ).float().cpu().numpy()
                    values = value.to(torch.float32).cpu().numpy()
                stage_end(self, capture_core)

                actions_env = []
                for i in range(normalized_actions_all.shape[0]):
                    a_env = self.model.vla._unnormalize_actions(
                        normalized_actions_all[i], self.cfg.unnorm_key)
                    actions_env.append(a_env.astype(np.float32))

                for i in range(len(promises_to_process)):
                    promises_to_process[i].set_result((
                        normalized_actions_all[i],
                        actions_env[i],
                        action_tokens[i],
                        logits[i],
                        values[i]
                    ))
                batch_end(self, capture_batch)
                _trim_cuda_cache("idle")
                loop_duration = time.time() - t_loop_start
                self.stats_actor.add_timing_metric.remote("Inference/loop_time_s", loop_duration)
            except Exception as e:
                import traceback
                print(f"[ERROR] InferenceActor {self.actor_id} 批处理失败: {e}", flush=True)
                traceback.print_exc()
                batch_end(self, capture_batch, error=e)
                for p in promises_to_process:
                    if not p.done():
                        p.set_exception(e)
                raise

    def forward_test(self):
        return


# ================================================================
# RewardInferenceActor — 与原版完全一致
# ================================================================
@ray.remote(num_gpus=1)
class RewardInferenceActor(ColocatedInferenceMixin, InferenceActorCom):
    def __init__(self, actor_id, stats_actor, agent_config_path, inference_batch, inference_timeout_ms, reward_checkpoint=None):
        super().__init__()
        setup_debugger("reward_inference", actor_id)
        self.gpu_binding = verify_bound_gpu()
        self.actor_id = actor_id
        self.stats_actor = stats_actor
        agent_cfg = OmegaConf.load(agent_config_path)
        reward_ckpt_path = reward_checkpoint if reward_checkpoint else agent_cfg.reward_model_path
        print(f"RewardInferenceActor {actor_id}: 加载 Reward Model checkpoint from {reward_ckpt_path}")
        self.reward_model, self.rew_cfg = load_reward_model(
            model_path=reward_ckpt_path,
            device="cuda",
            pretrained_checkpoint=agent_cfg.openvla_path,
            focal_alpha=agent_cfg.reward_model.focal_alpha,
        )
        if os.environ.get("ACCERL_REWARD_FEATURES_ONLY") == "1":
            info = install_reward_features_only(self.reward_model)
            print(f"[Reward features-only inference] actor={actor_id} metadata={info}", flush=True)
        self.processor = get_processor(self.rew_cfg)
        self.model = self.reward_model
        self.batch_size = inference_batch
        self.timeout_sec = inference_timeout_ms / 1000.0
        self.requests, self.promises = [], []
        self.last_process_time = time.time()

        loop = asyncio.get_event_loop()
        self._bg_task = loop.create_task(self._loop())
        self._bg_task.add_done_callback(self._on_bg_task_done)
        print(f"RewardInferenceActor {self.actor_id} 初始化于 GPU: {ray.get_gpu_ids()} (批次超时: {inference_timeout_ms}ms)")

    def _on_bg_task_done(self, task):
        try:
            task.result()
        except Exception as e:
            import traceback
            print(f"[ERROR] RewardInferenceActor {self.actor_id} 后台任务异常: {e}", flush=True)
            traceback.print_exc()
            self._fail_pending_requests(e)

    async def request(self, inputs_t):
        self._check_inference_health()
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        self.requests.append(inputs_t)
        self.promises.append(fut)
        return await fut

    @torch.no_grad()
    def evaluate_fixed_holdout(self, obs_np, labels_np, instructions):
        """Evaluate unique terminal episodes never used for reward training."""
        labels = np.asarray(labels_np, dtype=np.int64)
        probs = []
        for start in range(0, len(labels), self.batch_size):
            end = min(len(labels), start + self.batch_size)
            inputs_list = []
            for i in range(start, end):
                obs_hwc = np.transpose(obs_np[i], (1, 2, 0))
                obs_img = ((obs_hwc + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                inputs_list.append(
                    prepare_one_obs(
                        self.rew_cfg,
                        self.processor,
                        {"full_image": obs_img},
                        instructions[i],
                        torch.bfloat16,
                    )
                )
            inputs_batch = self.reward_model.prepare_inputs_batch(inputs_list)
            logits = self.reward_model.forward(inputs_batch)
            probs.extend(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist())
        probs = np.asarray(probs, dtype=np.float64)
        preds = probs >= 0.5
        pos = labels == 1
        neg = labels == 0
        pos_acc = float(np.mean(preds[pos] == 1)) if np.any(pos) else float("nan")
        neg_acc = float(np.mean(preds[neg] == 0)) if np.any(neg) else float("nan")
        pos_probs, neg_probs = probs[pos], probs[neg]
        if len(pos_probs) and len(neg_probs):
            comparisons = pos_probs[:, None] - neg_probs[None, :]
            roc_auc = float(
                np.mean(comparisons > 0) + 0.5 * np.mean(comparisons == 0)
            )
        else:
            roc_auc = float("nan")
        eps = 1e-7
        log_loss = float(
            -np.mean(
                labels * np.log(np.clip(probs, eps, 1.0 - eps))
                + (1 - labels) * np.log(np.clip(1.0 - probs, eps, 1.0 - eps))
            )
        )
        return {
            "balanced_acc": float(np.nanmean([pos_acc, neg_acc])),
            "pos_acc": pos_acc,
            "neg_acc": neg_acc,
            "roc_auc": roc_auc,
            "brier": float(np.mean((probs - labels) ** 2)),
            "log_loss": log_loss,
            "num_samples": float(len(labels)),
            "num_pos": float(np.sum(pos)),
            "num_neg": float(np.sum(neg)),
        }

    async def _loop(self):
        while True:
            should_process = self.requests and (
                len(self.requests) >= self.batch_size or
                time.time() - self.last_process_time > self.timeout_sec
            )
            if not should_process:
                await asyncio.sleep(0.0005)
                continue

            count = min(len(self.requests), getattr(self, "training_batch_limit", len(self.requests)))
            requests_to_process, self.requests = self.requests[:count], self.requests[count:]
            promises_to_process, self.promises = self.promises[:count], self.promises[count:]
            capture_batch = batch_begin(self, requests_to_process, promises_to_process)
            self.last_process_time = time.time()

            inputs_list = requests_to_process
            t_loop_start = time.time()
            try:
                capture_stage = stage_begin(self, "input_prepare", actual_batch=len(inputs_list))
                inputs_batch = self.reward_model.prepare_inputs_batch(inputs_list)
                stage_end(self, capture_stage)
                capture_core = stage_begin(self, "gpu_forward_and_copy", actual_batch=len(inputs_list), cuda=True)
                with torch.inference_mode():
                    logits = self.reward_model.forward(inputs_batch).cpu()
                stage_end(self, capture_core)
                for i in range(len(promises_to_process)):
                    promises_to_process[i].set_result(logits[i])
                batch_end(self, capture_batch)
                _trim_cuda_cache("idle")
                loop_duration = time.time() - t_loop_start
                self.stats_actor.add_timing_metric.remote("Inference/reward_loop_time_s", loop_duration)
            except Exception as e:
                import traceback
                print(f"[ERROR] RewardInferenceActor {self.actor_id} 批处理失败: {e}", flush=True)
                traceback.print_exc()
                batch_end(self, capture_batch, error=e)
                for p in promises_to_process:
                    if not p.done():
                        p.set_exception(e)
                raise


# ================================================================
# ★ CtrlWorldInferenceActor — 可接收 Trainer 广播的可训练权重
#   使用 CtrlWorldEnvBatch 的无状态接口，Worker 端维护 latent_history
# ================================================================
@ray.remote(num_gpus=1)
class CtrlWorldInferenceActor(ColocatedInferenceMixin, InferenceActorCom):
    def __init__(self, actor_id, stats_actor, args, inference_batch, inference_timeout_ms):
        super().__init__()
        setup_debugger("ctrl_world_inference", actor_id)
        self.gpu_binding = verify_bound_gpu()
        self.actor_id = actor_id
        self.stats_actor = stats_actor
        self.ema_decay = float(args.ctrl_ema_decay)
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError(f"ctrl_ema_decay must be in [0,1), got {self.ema_decay}")
        self.ema_initialized = False
        self.ema_updates = 0

        print(f"CtrlWorldInferenceActor {actor_id}: 正在加载 Ctrl-World 模型...")

        # 构建 Ctrl-World 配置
        cfg = wm_args()
        cfg.svd_model_path = args.svd_model_path
        cfg.clip_model_path = args.clip_model_path
        cfg.num_cams = args.num_cams
        cfg.num_history = args.num_history
        cfg.num_frames = args.num_frames_pred
        cfg.action_dim = ACTION_DIM
        cfg.text_cond = True
        cfg.frame_level_cond = True
        cfg.his_cond_zero = False

        device = torch.device("cuda")
        ctrl_world = AcceRLCtrlWorld(cfg).to(device).to(torch.bfloat16)
        ctrl_world.eval()

        if not os.path.isfile(args.ctrl_world_ckpt):
            raise FileNotFoundError(f"Ctrl-World checkpoint not found: {args.ctrl_world_ckpt}")
        print(f"CtrlWorldInferenceActor {actor_id}: 加载 checkpoint: {args.ctrl_world_ckpt}")
        ckpt = torch.load(args.ctrl_world_ckpt, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
        ctrl_world.load_state_dict(state_dict, strict=True)
        _load_vae_decoder_checkpoint(ctrl_world, args.vae_decoder_checkpoint)
        print(f"CtrlWorldInferenceActor {actor_id}: checkpoint 加载完成。")

        stat_path = args.condition_stat_path
        if stat_path is None:
            stat_path = str(Path(args.ctrl_world_ckpt).resolve().parent / "condition_stat.json")
        if not os.path.isfile(stat_path):
            raise FileNotFoundError(f"Ctrl-World condition statistics not found: {stat_path}")
        with open(stat_path, "r", encoding="utf-8") as f:
            condition_stats = json.load(f)
        if condition_stats.get("condition_mode") != "action":
            raise ValueError(
                f"Expected action condition statistics, got "
                f"{condition_stats.get('condition_mode')!r}"
            )
        if condition_stats.get("normalize_condition") != "bounds":
            raise ValueError("Ctrl-World checkpoint must use bounds-normalized conditions")
        if (
            set(condition_stats.get("skip_choices", [])) != {1}
            or int(condition_stats.get("skip_his_multiplier", -1)) != 1
            or float(condition_stats.get("skip_his_zero_prob", -1.0)) != 0.0
        ):
            raise ValueError(
                "Ctrl-World checkpoint statistics are not frame-aligned for "
                "single-step delta actions"
            )
        condition_low = torch.tensor(condition_stats["condition_p01"], dtype=torch.float32)
        condition_high = torch.tensor(condition_stats["condition_p99"], dtype=torch.float32)
        if condition_low.shape != (ACTION_DIM,) or condition_high.shape != (ACTION_DIM,):
            raise ValueError(
                f"Expected {ACTION_DIM}-D condition bounds, got "
                f"{tuple(condition_low.shape)} and {tuple(condition_high.shape)}"
            )
        if not torch.all(condition_high > condition_low):
            raise ValueError("Every condition p99 value must be greater than p01")

        # 创建 CtrlWorldEnvBatch（仅用其无状态接口和编解码工具）
        @dataclass
        class FakeCfg:
            horizon: int = 220

        self.ctrl_world_env = CtrlWorldEnvBatch(
            ctrl_world_model=ctrl_world,
            cfg=FakeCfg(horizon=220),
            reward_model=None,       # Worker 端通过 RewardInferenceActor 独立推理
            reward_cfg=None,
            processor=None,
            torch_dtype=torch.bfloat16,
            instructions=None,
            num_cams=args.num_cams,
            target_height=args.target_height,
            target_width=args.target_width,
            num_frames_pred=args.num_frames_pred,
            num_inference_steps=args.num_inference_steps,
            condition_low=condition_low,
            condition_high=condition_high,
        )

        self.model = ctrl_world  # InferenceActorCom 期望 self.model
        self.world_backend = CtrlBackendAdapter(model=ctrl_world, env=self.ctrl_world_env)
        self.device = self.ctrl_world_env.device
        self.batch_size = inference_batch
        self.timeout_sec = inference_timeout_ms / 1000.0
        self.requests, self.promises = [], []
        self.last_process_time = time.time()

        loop = asyncio.get_event_loop()
        self._bg_task = loop.create_task(self._loop())
        self._bg_task.add_done_callback(self._on_bg_task_done)
        print(f"CtrlWorldInferenceActor {self.actor_id} 初始化于 GPU: {ray.get_gpu_ids()} (批次超时: {inference_timeout_ms}ms)")

    def receive_and_update_weights(self, group_name):
        """Only sync trainable Ctrl-World params; skip frozen VAE/CLIP buffers."""
        if not group_name.startswith("broadcast_ctrl"):
            return super().receive_and_update_weights(group_name)

        # Resolve the manager from rl.ds_com at call time. The inherited
        # setup_broadcast_group() writes to that module singleton inside each
        # Ray actor process; a copied script-level import can point at a
        # different, empty GroupManager.
        from rl.ds_com import _group_mgr as group_mgr

        group_handle = group_mgr.get_group_by_name(group_name)
        assert group_handle is not None, f"广播组 '{group_name}' 未初始化"
        module = _unwrap_module(self.model)
        params = sorted(
            [(n, p) for n, p in module.named_parameters(recurse=True) if p.requires_grad],
            key=lambda x: x[0],
        )
        device = next(module.parameters()).device
        for _, p in params:
            buf = torch.empty_like(p.data, device=device)
            dist.broadcast(buf, src=0, group=group_handle)
            if self.ema_initialized and self.ema_decay > 0.0:
                p.data.mul_(self.ema_decay).add_(buf, alpha=1.0 - self.ema_decay)
            else:
                p.data.copy_(buf)
        self.ema_initialized = True
        self.ema_updates += 1
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        self.model.eval()
        print(
            f"CtrlWorldInferenceActor {self.actor_id}: "
            f"received {len(params)} trainable tensors ({group_name}), "
            f"EMA decay={self.ema_decay}, updates={self.ema_updates}"
        )

    def _on_bg_task_done(self, task):
        try:
            task.result()
        except Exception as e:
            import traceback
            print(f"[ERROR] CtrlWorldInferenceActor {self.actor_id} 后台任务异常: {e}", flush=True)
            traceback.print_exc()
            self._fail_pending_requests(e)

    # ★ 无状态接口：Worker 传入完整状态
    async def request(self, current_latent, latent_history, action_condition, instruction):
        """
        Args:
            current_latent: [4, latent_h*num_cams, latent_w]
            latent_history: [num_history, 4, latent_h*num_cams, latent_w] (CPU tensor)
            action_condition: [num_history+num_frames, act_dim], raw delta actions
            instruction:    str
        Returns:
            future observations/latents for frames 1: of the generated chunk
        """
        self._check_inference_health()
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        self.requests.append((current_latent, latent_history, action_condition, instruction))
        self.promises.append(fut)
        return await fut

    def init_latent_state(self, obs_seq):
        """
        Worker 端调用：从观测序列初始化 latent_history。
        Args:
            obs_seq: [T,M,C,H,W] in [-1, 1] (CPU tensor)
        Returns:
            (latent_history, current_latent)
        """
        history, current = self.ctrl_world_env.init_latent_state(
            obs_seq.unsqueeze(0).to(self.device)
        )
        return history[0].cpu(), current[0].cpu()

    def get_num_history(self):
        return self.ctrl_world_env.num_history

    def get_stride(self):
        return self.ctrl_world_env.num_frames_pred - 1

    def _ensure_lpips(self, net_name: str = "alex"):
        if getattr(self, "_lpips_net_name", None) == net_name and hasattr(self, "_lpips"):
            return self._lpips
        import lpips
        self._lpips = lpips.LPIPS(net=net_name).to(self.device).eval()
        self._lpips_net_name = net_name
        return self._lpips

    @torch.no_grad()
    def evaluate_prediction_metrics(
        self,
        obs_np: np.ndarray,
        act_np: np.ndarray,
        instructions: List[str],
        max_horizon: int = 4,
        lpips_net: str = "alex",
        micro_batch_size: int = 4,
        seed: int = 12345,
    ) -> Dict[str, float]:
        """Deterministic, micro-batched metrics on a fixed held-out WM set."""
        env = self.ctrl_world_env
        num_history = env.num_history
        stride = env.num_frames_pred - 1
        if max_horizon < 1:
            raise ValueError("max_horizon must be >= 1")
        if micro_batch_size < 1:
            raise ValueError("micro_batch_size must be >= 1")

        obs = torch.as_tensor(obs_np, dtype=torch.float32)
        acts = torch.as_tensor(act_np, dtype=torch.float32)
        if obs.ndim != 6:
            raise ValueError(f"Expected obs [B,F,M,C,H,W], got {tuple(obs.shape)}")
        if acts.ndim != 3:
            raise ValueError(f"Expected act [B,F,A], got {tuple(acts.shape)}")

        B, nF, M, C, H, W = obs.shape
        context_len = num_history + 1
        available_future = nF - context_len
        if available_future < 1:
            raise ValueError(
                f"WM windows need at least {context_len + 1} frames, got F={nF}"
            )
        eval_horizon = min(int(max_horizon), int(available_future), int(stride))
        if eval_horizon < 1:
            raise ValueError("No future frames available for Ctrl-World eval")

        needed_act = num_history + 1 + stride
        lpips_model = self._ensure_lpips(lpips_net)
        samples = {
            name: {str(h): [] for h in range(1, eval_horizon + 1)}
            for name in ("mse", "psnr", "ssim", "lpips")
        }
        view_samples = {
            view: {name: [] for name in ("mse", "psnr", "ssim", "lpips")}
            for view in range(M)
        }

        for start in range(0, B, micro_batch_size):
            end = min(B, start + micro_batch_size)
            obs_mb = obs[start:end]
            acts_mb = acts[start:end].to(self.device)
            mb = end - start
            flat = obs_mb.reshape(mb * nF * M, C, H, W)
            flat = F.interpolate(
                flat,
                size=(env.target_height, env.target_width),
                mode="bilinear",
                align_corners=False,
            )
            obs_mb = flat.reshape(
                mb, nF, M, C, env.target_height, env.target_width
            )
            context = obs_mb[:, :context_len].to(self.device)
            gt_future = obs_mb[
                :, context_len : context_len + eval_horizon
            ].to(self.device)
            history, current = env.init_latent_state(
                context, deterministic=True
            )
            if acts_mb.shape[1] < needed_act:
                pad = acts_mb[:, -1:].repeat(
                    1, needed_act - acts_mb.shape[1], 1
                )
                act_cond = torch.cat([acts_mb, pad], dim=1)
            else:
                act_cond = acts_mb[:, :needed_act]
            generators = []
            for sample_index in range(start, end):
                generator = torch.Generator(device=self.device)
                generator.manual_seed(int(seed) + sample_index)
                generators.append(generator)
            future_obs, _ = env.predict_chunk_stateless(
                current_latent=current,
                latent_history=history,
                action_condition=act_cond,
                instructions=list(instructions[start:end]),
                output_size=(env.target_height, env.target_width),
                generator=generators,
            )
            pred_future = future_obs[:, :eval_horizon].float()
            gt_future = gt_future.float()

            for horizon in range(1, eval_horizon + 1):
                pred_h = pred_future[:, horizon - 1]
                gt_h = gt_future[:, horizon - 1]
                pred_flat = pred_h.reshape(
                    mb * M, C, env.target_height, env.target_width
                )
                gt_flat = gt_h.reshape(
                    mb * M, C, env.target_height, env.target_width
                )
                pred_01 = ((pred_flat + 1.0) * 0.5).clamp(0.0, 1.0)
                gt_01 = ((gt_flat + 1.0) * 0.5).clamp(0.0, 1.0)
                mse_view = (pred_01 - gt_01).square().mean(dim=(1, 2, 3)).reshape(mb, M)
                psnr_view = -10.0 * torch.log10(mse_view.clamp_min(1e-12))
                ssim_view = batch_ssim(pred_01, gt_01).reshape(mb, M)
                lpips_view = lpips_model(pred_flat, gt_flat).reshape(mb, M)
                key = str(horizon)
                for name, tensor in (
                    ("mse", mse_view.mean(dim=1)),
                    ("psnr", psnr_view.mean(dim=1)),
                    ("ssim", ssim_view.mean(dim=1)),
                    ("lpips", lpips_view.mean(dim=1)),
                ):
                    samples[name][key].extend(tensor.float().cpu().tolist())
                if horizon == eval_horizon:
                    for view in range(M):
                        for name, tensor in (
                            ("mse", mse_view[:, view]),
                            ("psnr", psnr_view[:, view]),
                            ("ssim", ssim_view[:, view]),
                            ("lpips", lpips_view[:, view]),
                        ):
                            view_samples[view][name].extend(
                                tensor.float().cpu().tolist()
                            )

        def _mean(values):
            return float(np.mean(np.asarray(values, dtype=np.float64)))

        def _std(values):
            return float(np.std(np.asarray(values, dtype=np.float64)))

        result = {
            "eval_horizon": float(eval_horizon),
            "num_windows": float(B),
            "eval_seed": float(seed),
        }
        for name in ("mse", "psnr", "ssim", "lpips"):
            all_values = []
            by_horizon, std_by_horizon = {}, {}
            for horizon in range(1, eval_horizon + 1):
                key = str(horizon)
                values = samples[name][key]
                all_values.extend(values)
                by_horizon[key] = _mean(values)
                std_by_horizon[key] = _std(values)
            result[name] = _mean(all_values)
            result[f"{name}_std"] = _std(all_values)
            result[f"{name}_by_horizon"] = by_horizon
            result[f"{name}_std_by_horizon"] = std_by_horizon
        view_names = ["agent_view", "wrist_view"]
        for view, metrics in view_samples.items():
            prefix = view_names[view] if view < len(view_names) else f"view{view}"
            for name, values in metrics.items():
                result[f"{prefix}_{name}"] = _mean(values)
                result[f"{prefix}_{name}_std"] = _std(values)

        return result

    async def _loop(self):
        while True:
            should_process = self.requests and (
                len(self.requests) >= self.batch_size or
                time.time() - self.last_process_time > self.timeout_sec
            )
            if not should_process:
                await asyncio.sleep(0.0005)
                continue

            count = min(len(self.requests), getattr(self, "training_batch_limit", len(self.requests)))
            requests_to_process, self.requests = self.requests[:count], self.requests[count:]
            promises_to_process, self.promises = self.promises[:count], self.promises[count:]
            capture_batch = batch_begin(self, requests_to_process, promises_to_process)
            self.last_process_time = time.time()

            # 拆分请求
            current_latent_list = [req[0] for req in requests_to_process]
            latent_history_list = [req[1] for req in requests_to_process]
            action_condition_list = [req[2] for req in requests_to_process]
            instructions = [req[3] for req in requests_to_process]

            capture_stage = stage_begin(self, "input_prepare", actual_batch=len(requests_to_process))
            # Stack into batch
            current_latent_batch = torch.stack(current_latent_list, dim=0).to(self.device)
            latent_history_batch = torch.stack(latent_history_list, dim=0).to(self.device)  # [B, num_history, 4, ...]
            action_condition_batch = torch.stack(action_condition_list, dim=0).to(self.device)
            stage_end(self, capture_stage)
            capture_core = stage_begin(self, "gpu_forward_and_copy", actual_batch=len(requests_to_process), cuda=True)

            t_loop_start = time.time()
            try:
                with torch.inference_mode():
                    future_obs, future_latents = self.ctrl_world_env.predict_chunk_stateless(
                        current_latent=current_latent_batch,
                        latent_history=latent_history_batch,
                        action_condition=action_condition_batch,
                        instructions=instructions,
                        output_size=(224, 224),
                    )
                    future_obs = future_obs.float().cpu()
                    future_latents = future_latents.cpu()
                stage_end(self, capture_core)

                for i, promise in enumerate(promises_to_process):
                    promise.set_result((future_obs[i], future_latents[i]))

                batch_end(self, capture_batch)
                _trim_cuda_cache("idle")
                loop_duration = time.time() - t_loop_start
                self.stats_actor.add_timing_metric.remote("Inference/ctrl_world_loop_time_s", loop_duration)
            except Exception as e:
                import traceback
                print(f"[ERROR] CtrlWorldInferenceActor {self.actor_id} 批处理失败: {e}", flush=True)
                traceback.print_exc()
                batch_end(self, capture_batch, error=e)
                for p in promises_to_process:
                    if not p.done():
                        p.set_exception(e)
                raise


# ================================================================
# 4. 训练器 (TrainerActor) — Actor + 可训练 Ctrl-World + Reward（含广播）
# ================================================================
WeaverWorldInferenceActor = ray.remote(num_gpus=1)(
    make_weaver_inference_actor(ColocatedInferenceMixin, InferenceActorCom, globals()))
WeaverOnlineTrainerActor = ray.remote(num_gpus=1)(make_weaver_online_trainer(globals()))
WeaverOnlineWorldInferenceActor = ray.remote(num_gpus=1)(
    make_weaver_online_inference(ColocatedInferenceMixin, InferenceActorCom, globals()))


@ray.remote(num_gpus=1)
class TrainerActor(TrainerActorCom):
    def export_snapshot_component(self, kind):
        return export_split_component(self, kind)

    async def save_parallel_snapshot(self, checkpoint_dir, reason="periodic", metadata=None):
        # Driver calls only rank 0 after both ranks finish the synchronized epoch.
        result = save_parallel_weights(self, checkpoint_dir, reason=reason, metadata=metadata)
        self.telemetry.event("checkpoint", reason=reason, steps=result["steps"],
                             path=str(checkpoint_dir), weights_only=True)
        self.telemetry.flush()
        return result

    def __init__(self, rank, world_size, replay_buffer, cfg, train_batch_size, accumulation_steps,
                 use_bf16, torch_dtype, policy_lr, value_lr, gamma, lambda_, clip_eps, vf_coef,
                 ent_coef, kl_coef, reward_scale, value_warmup_steps, policy_warmup_steps,
                 policy_train_start_step, train_iters, clip_mode,
                 # World Model 相关参数
                 wm_replay_buffer, agent_config_path, trainer_config_path,
                 reward_batch_size, reward_accumulation_steps, reward_lr, reward_warmup_steps,
                 reward_train_interval,
                 reward_checkpoint=None,
                 # Ctrl-World train/broadcast
                 ctrl_world_args=None,
                 ctrl_batch_size=1, ctrl_accumulation_steps=8, ctrl_lr=1e-5,
                 ctrl_warmup_steps=500, ctrl_train_interval=10, batch_coordinator=None,
                 training_role='combined', split_train_serial=False):
        super().__init__()
        if training_role not in ('combined', 'policy', 'ctrl', 'reward'):
            raise ValueError('Unknown training process role: ' + str(training_role))
        self.training_role = training_role
        self.world_config = WorldBackendConfig.from_mapping((ctrl_world_args or {}).get("world_backend_config"))
        self.fresh_config = FreshConfig.from_world_config(self.world_config)
        self.fresh_policy_replay = any((self.fresh_config.refresh_gae,
            self.fresh_config.replay_age_limits, self.fresh_config.real_policy_replay))
        self.world_backend = None
        self.active_kinds = build_service_plan(self.world_config).active_kinds
        if not self.world_config.world_trainable and training_role in ("combined", "ctrl"):
            raise ValueError("Frozen World must not construct a World optimizer owner")
        if split_train_serial and training_role == 'combined':
            raise ValueError('Serial split training requires an independent model owner')
        self.split_train_serial = bool(split_train_serial)
        self.cpu_replay_prefetch = bool(split_train_serial or os.environ.get("ACCERL_CPU_REPLAY_PREFETCH") == "1")
        self._serial_phase = SerialPhase(training_role, rank, world_size) if self.cpu_replay_prefetch else None
        self._serial_prefetch = None
        setup_debugger("trainer", rank)
        self.gpu_binding = verify_bound_gpu()
        self.rank = rank
        self.world_size = world_size
        self.replay_buffer = replay_buffer
        self.batch_coordinator = batch_coordinator
        self.cfg = cfg
        self.model = None
        self.optimizer = None
        self.base_model = None
        self.data_dtype = None
        self.next_ready_batch = None
        self.data_fetching_task = None
        self.data_fetch_error = None

        self.reward_checkpoint = reward_checkpoint

        self.train_batch_size = train_batch_size
        self.accumulation_steps = accumulation_steps
        self.super_batch_size = train_batch_size * accumulation_steps
        self.use_bf16 = use_bf16
        self.torch_dtype = torch_dtype
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.gamma = gamma
        self.lambda_ = lambda_
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.kl_coef = kl_coef
        self.reward_scale = reward_scale
        self.value_warmup_steps = value_warmup_steps
        self.policy_warmup_steps = policy_warmup_steps
        self.policy_train_start_step = policy_train_start_step
        self.train_iters = train_iters
        self.clip_mode = clip_mode

        self.global_step = 0

        # ============ World Model 相关 ============
        self.wm_replay_buffer = wm_replay_buffer
        self.agent_config_path = agent_config_path
        self.trainer_config_path = trainer_config_path

        # Reward Model 参数
        self.reward_batch_size = reward_batch_size
        self.reward_accumulation_steps = reward_accumulation_steps
        self.reward_super_batch_size = reward_batch_size * reward_accumulation_steps
        self.reward_lr = reward_lr
        self.reward_warmup_steps = reward_warmup_steps

        # Ctrl-World 训练参数
        self.ctrl_world_args = ctrl_world_args
        self.ctrl_batch_size = ctrl_batch_size
        self.ctrl_accumulation_steps = ctrl_accumulation_steps
        self.ctrl_super_batch_size = ctrl_batch_size * ctrl_accumulation_steps
        self.wm_local_batch_size = max(self.ctrl_super_batch_size if "ctrl" in self.active_kinds else 0,
            self.reward_super_batch_size if "reward" in self.active_kinds else 0)
        self.ctrl_lr = ctrl_lr
        self.ctrl_warmup_steps = ctrl_warmup_steps
        self.ctrl_train_interval = ctrl_train_interval

        # World Model 训练频率
        self.reward_train_interval = reward_train_interval
        self.total_ctrl_steps = max(1, math.ceil(self.train_iters / self.ctrl_train_interval))

        # World Model 独立训练步数计数器
        self.reward_step = 0
        self.ctrl_step = 0

        # World Model engines (将在 setup_deepspeed_group 中初始化)
        self.reward_engine = None
        self.reward_model = None
        self.reward_cfg = None
        self.processor = None
        self.ctrl_world_engine = None
        self.ctrl_world_model = None
        self.condition_low = None
        self.condition_high = None

        # WM 数据获取
        self.next_ready_wm_batch = None
        self.wm_data_fetching_task = None

        # Batch preparation runs in worker threads (created in setup_deepspeed_group);
        # the events wake each fetch loop as soon as its batch slot is consumed.
        self._prep_executor = None
        self._policy_slot_freed = None
        self._wm_slot_freed = None
        self._policy_replay_ready = False
        self._wm_replay_ready = False
        self.last_wm_batch_info = {"sample_time": 0.0, "prep_time": 0.0,
                                   "unique_episodes": 0, "fresh_episode_ratio": 0.0}

        print(f"TrainerActor Rank {self.rank} 初始化于 GPU: {ray.get_gpu_ids()}")

    def get_model_keys(self):
        if self.model is None:
            print("模型尚未初始化。请先调用 setup_deepspeed_group()。")
            return {}
        module = self.model.module if hasattr(self.model, "module") else self.model
        sd = module.state_dict()
        if isinstance(self.model, PolicyFSDPEngine):
            return {k: None for k in sd}
        res = {k: float(v.abs().sum().item()) for k, v in sd.items()}
        return res

    def get_training_state(self):
        return {"rank": self.rank, "gpu": self.gpu_binding, "pid": os.getpid(),
                "training_role": self.training_role, "world_size": self.world_size,
                "split_train_serial": self.split_train_serial,
                "cpu_replay_prefetch": self.cpu_replay_prefetch,
                "policy_sharding": self.model.shard_report() if isinstance(self.model, PolicyFSDPEngine) else None,
                "reward_sharding": self.reward_engine.shard_report() if isinstance(self.reward_engine, PolicyFSDPEngine) else None,
                "distributed_world_size": distributed.get_world_size(),
                "group_port": self.deepspeed_group_port,
                "policy_steps": self.global_step, "ctrl_steps": self.ctrl_step,
                "reward_steps": self.reward_step,
                "engines": {name: {"global_steps": int(engine.global_steps),
                                     "skipped_steps": int(engine.skipped_steps),
                                     "global_batch": int(engine.train_batch_size())}
                            for name, engine in (("policy", self.model), ("ctrl", self.ctrl_world_engine),
                                                  ("reward", self.reward_engine)) if engine is not None}}

    def export_rank_weights(self, kind):
        if self.training_role not in ('combined', kind):
            raise ValueError('A training process may export only its owned model')
        engine, version = {"policy": (self.model, self.global_step),
                           "ctrl": (self.ctrl_world_engine, self.ctrl_step),
                           "reward": (self.reward_engine, self.reward_step)}[kind]
        ipc = kind in ("policy", "ctrl")
        with self.telemetry.span("broadcast_send", cuda=True, model_version=version,
                                 transport="cuda_ipc" if ipc else "cpu_ray", model=kind, rank=self.rank):
            if ipc:
                if isinstance(engine, PolicyFSDPEngine):
                    raise RuntimeError("Policy/World IPC source requires full ZeRO2 parameter replicas")
                exports = getattr(self, "_ipc_weight_exports", None)
                if exports is None:
                    exports = self._ipc_weight_exports = {}
                if kind not in exports:
                    exports[kind] = CudaIPCWeightExporter(
                        engine.module, kind, self.gpu_binding["uuid"],
                        os.environ["ACCERL_RUN_ID"], self.rank,
                        world_backend=self.world_backend if kind == "ctrl" else None)
                payload = exports[kind].begin(int(version))
            else:
                payload = (engine.export_cpu_weights(kind, version) if isinstance(engine, PolicyFSDPEngine)
                           else export_cpu_weights(engine.module, kind, version))
        _trim_cuda_cache("idle")
        return payload

    def describe_rank_weights(self, payload):
        return {key: payload[key] for key in ("kind", "version", "hash", "bytes", "tensor_count")}

    def finish_ipc_weight_publish(self, kind, token, acknowledgement):
        if kind not in ("policy", "ctrl") or self.training_role not in ("combined", kind):
            raise ValueError("An IPC acknowledgement must target its Policy/World owner")
        result = self._ipc_weight_exports[kind].finish(token, acknowledgement)
        self.telemetry.event("ipc_weight_publish_released", model_version=result["version"],
                             details={"model":kind, "rank":self.rank, "token":token, **result})
        return result

    def _verify_optimizer_update(self, engine, name, before_steps, before_skips):
        state = torch.tensor([int(engine.global_steps) - before_steps,
                              int(engine.skipped_steps) - before_skips], device="cuda", dtype=torch.int64)
        states = [torch.empty_like(state) for _ in range(self.world_size)]
        distributed.all_gather(states, state)
        counts = [value.cpu().tolist() for value in states]
        if any(value != [1, 0] for value in counts):
            raise RuntimeError(f"{name}: expected one successful synchronized update per rank, got {counts}")

    def get_node_ip(self):
        return ray.util.get_node_ip_address()

    def setup_deepspeed_group(self, master_addr, master_port):
        self.deepspeed_group_port = int(master_port)
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["LOCAL_RANK"] = "0"
        torch.cuda.set_device(0)
        deepspeed.init_distributed(dist_backend="nccl")

        if self.training_role in ("combined", "policy"):
            # ============ 1. 初始化 Actor Engine (bf16) ============
            print(f"Trainer {self.rank}: 正在加载 OpenVLA ActorCritic...")
            model = ActorCritic(self.cfg, torch_dtype=self.torch_dtype)
            self.base_model = model

            param_groups = self.base_model.get_parameter_groups()
            optimizer_params = [
                {"params": pg["params"], "name": pg["name"],
                 "lr": self.policy_lr if pg["name"] == "policy" else self.value_lr}
                for pg in param_groups
            ]

            actor_ds_config = {
                "train_micro_batch_size_per_gpu": self.train_batch_size,
                "gradient_accumulation_steps": self.accumulation_steps,
                "optimizer": {"type": "AdamW", "params": {}},
                "bf16": {"enabled": self.use_bf16},
                "zero_optimization": {
                    "stage": 2, "allgather_partitions": True, "allgather_bucket_size": 5e8,
                    "reduce_scatter": True, "reduce_bucket_size": 5e8, "overlap_comm": True,
                    "contiguous_gradients": True
                },
                "gradient_clipping": 1.0,
            }

            if actor_ds_config.get("bf16", {}).get("enabled", False):
                self.data_dtype = torch.bfloat16
            else:
                self.data_dtype = torch.float32

            if os.environ.get("ACCERL_POLICY_BACKEND") == "fsdp2":
                if self.training_role != "policy":
                    raise ValueError("FSDP Policy requires a separate Policy owner")
                self.model = PolicyFSDPEngine(model, micro_batch=self.train_batch_size,
                    accumulation_steps=self.accumulation_steps,
                    policy_lr=self.policy_lr, value_lr=self.value_lr)
                self.optimizer = self.model.optimizer
                self.telemetry.event("fsdp_sharding_verified", details=self.model.shard_report())
                self.telemetry.flush()
            else:
                self.model, self.optimizer, _, _ = deepspeed.initialize(
                    model=model, config=actor_ds_config, model_parameters=optimizer_params
                )
            print(f"TrainerActor Rank {self.rank}: Actor DeepSpeed engine (bf16) 初始化完成。")

            n_total = sum(p.numel() for p in model.parameters())
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Actor 总参数量: {n_total:,}, 可训练参数量: {n_trainable:,}")


        if self.training_role in ("combined", "ctrl"):
            # ============ 2. 初始化 Ctrl-World Engine (bf16, trainable unet/action_encoder) ============
            print(f"Trainer {self.rank}: 正在加载可训练 Ctrl-World...")
            if self.ctrl_world_args is None:
                raise ValueError("ctrl_world_args is required for trainable Ctrl-World trainer")
            cw_args = self.ctrl_world_args
            if self.world_config.backend == "weaver":
                from world_backends.weaver import WeaverBackend
                options = self.world_config.backend_options
                self.world_backend = WeaverBackend.load(options["source_path"], options["checkpoint_dir"],
                    device="cuda", dtype=self.torch_dtype, config=self.world_config, for_training=True)
                self.ctrl_world_model = self.world_backend.model
            else:
                if wm_args is None:
                    raise RuntimeError("CTRL dependencies are unavailable") from _ctrl_import_error
                ctrl_cfg = wm_args()
                ctrl_cfg.svd_model_path = cw_args["svd_model_path"]
                ctrl_cfg.clip_model_path = cw_args["clip_model_path"]
                ctrl_cfg.num_cams = cw_args["num_cams"]
                ctrl_cfg.num_history = cw_args["num_history"]
                ctrl_cfg.num_frames = cw_args["num_frames_pred"]
                ctrl_cfg.action_dim = ACTION_DIM
                ctrl_cfg.text_cond = True
                ctrl_cfg.frame_level_cond = True
                ctrl_cfg.his_cond_zero = False
                self.ctrl_target_height = cw_args["target_height"]
                self.ctrl_target_width = cw_args["target_width"]
                self.ctrl_num_history = cw_args["num_history"]
                self.ctrl_num_frames = cw_args["num_frames_pred"]
    
                self.ctrl_world_model = AcceRLCtrlWorld(ctrl_cfg).cuda().to(torch.bfloat16)
                ckpt_path = cw_args["ctrl_world_ckpt"]
                print(f"Trainer {self.rank}: 加载 Ctrl-World checkpoint from {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
                state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
                self.ctrl_world_model.load_state_dict(state_dict, strict=True)
                _load_vae_decoder_checkpoint(
                    self.ctrl_world_model, cw_args.get("vae_decoder_checkpoint")
                )
    
                stat_path = cw_args.get("condition_stat_path")
                if not stat_path:
                    stat_path = str(Path(ckpt_path).resolve().parent / "condition_stat.json")
                with open(stat_path, "r", encoding="utf-8") as f:
                    condition_stats = json.load(f)
                self.condition_low = torch.tensor(condition_stats["condition_p01"], dtype=torch.float32, device="cuda")
                self.condition_high = torch.tensor(condition_stats["condition_p99"], dtype=torch.float32, device="cuda")
                self.world_backend = CtrlBackendAdapter(model=self.ctrl_world_model,
                    prepare_fn=lambda batch: self._prepare_ctrl_batch(batch["obs"], batch["actions"],
                                                                    batch["rewards"], batch["instructions"]))
    
            ctrl_params = [p for p in self.ctrl_world_model.parameters() if p.requires_grad]
            n_ctrl_train = sum(p.numel() for p in ctrl_params)
            n_ctrl_total = sum(p.numel() for p in self.ctrl_world_model.parameters())
            print(f"Trainer {self.rank}: Ctrl-World params total={n_ctrl_total:,}, trainable={n_ctrl_train:,}")

            ctrl_ds_config = {
                "train_micro_batch_size_per_gpu": self.ctrl_batch_size,
                "gradient_accumulation_steps": self.ctrl_accumulation_steps,
                "optimizer": {"type": "AdamW", "params": {"lr": self.ctrl_lr}},
                "bf16": {"enabled": True},
                "zero_optimization": {"stage": 2},
                "gradient_clipping": 1.0,
            }
            self.ctrl_world_engine, _, _, _ = deepspeed.initialize(
                model=self.ctrl_world_model,
                config=ctrl_ds_config,
                model_parameters=ctrl_params,
            )
            self.telemetry.event("world_memory_policy_verified", details={
                "zero_stage": 2,
                "reduce_bucket_size": int(self.ctrl_world_engine._config.zero_config.reduce_bucket_size),
                "allgather_bucket_size": int(self.ctrl_world_engine._config.zero_config.allgather_bucket_size),
                "original_zero_config_preserved": True, "batch_and_optimizer_unchanged": True})
            self.telemetry.flush()
            print(f"TrainerActor Rank {self.rank}: Ctrl-World DeepSpeed engine (bf16) 初始化完成。")


        if self.training_role in ("combined", "reward"):
            # ============ 3. 初始化 Reward Model Engine (bf16) ============
            print(f"Trainer {self.rank}: 正在加载 Reward Model...")
            from envs.utils import load_reward_model

            agent_cfg = OmegaConf.load(self.agent_config_path)
            reward_ckpt_path = self.reward_checkpoint if self.reward_checkpoint else agent_cfg.reward_model_path
            print(f"Trainer {self.rank}: 加载 Reward Model checkpoint from {reward_ckpt_path}")
            self.reward_model, self.reward_cfg = load_reward_model(
                model_path=reward_ckpt_path,
                device="cuda",
                pretrained_checkpoint=agent_cfg.openvla_path,
                focal_alpha=agent_cfg.reward_model.focal_alpha,
            )
            if os.environ.get("ACCERL_REWARD_FEATURES_ONLY") == "1":
                metadata = install_reward_features_only(self.reward_model)
                self.telemetry.event("reward_features_only_enabled", details=metadata)
                checkpointing = install_reward_checkpointing(self.reward_model)
                self.telemetry.event("reward_checkpointing_verified", details=checkpointing)
                self.telemetry.flush()
            self.processor = get_processor(self.reward_cfg)

            reward_ds_config = {
                "train_micro_batch_size_per_gpu": self.reward_batch_size,
                "gradient_accumulation_steps": self.reward_accumulation_steps,
                "optimizer": {"type": "AdamW", "params": {"lr": self.reward_lr}},
                "bf16": {"enabled": True},
                "zero_optimization": {"stage": 2},
                "gradient_clipping": 1.0,
            }

            if os.environ.get("ACCERL_REWARD_BACKEND") == "fsdp2":
                if self.training_role != "reward":
                    raise ValueError("FSDP Reward requires a separate Reward owner")
                self.reward_engine = PolicyFSDPEngine(self.reward_model,
                    micro_batch=self.reward_batch_size,
                    accumulation_steps=self.reward_accumulation_steps,
                    kind="reward", learning_rate=self.reward_lr)
                self.telemetry.event("fsdp_sharding_verified", details=self.reward_engine.shard_report())
                self.telemetry.flush()
            else:
                self.reward_engine, _, _, _ = deepspeed.initialize(
                    model=self.reward_model,
                    config=reward_ds_config,
                    model_parameters=self.reward_model.parameters(),
                )
            print(f"TrainerActor Rank {self.rank}: Reward Model engine (bf16) 初始化完成。")

            n_reward = sum(p.numel() for p in self.reward_model.parameters())
            print(f"Reward Model 总参数量: {n_reward:,}")


            # 启动数据获取循环。The training body below is synchronous and occupies

        # Each split actor owns one model and one bounded preparation slot.
        self._prep_executor = ThreadPoolExecutor(
            max_workers=2 if self.training_role == "combined" else 1,
            thread_name_prefix="trainer-" + self.training_role + "-prep")
        self._policy_slot_freed = asyncio.Event()
        self._wm_slot_freed = asyncio.Event()
        if self.cpu_replay_prefetch and self.training_role == "policy":
            self._policy_joint_batch_mode = PolicyJointBatchMode(self, writer=self.telemetry)
        if self.cpu_replay_prefetch and not (self.training_role == 'policy' and self.fresh_policy_replay):
            self._serial_prefetch = ObservedCpuReplayPrefetch(
                self._serial_fetch_cpu_batch, writer=self.telemetry, model=self.training_role,
                rank=self.rank, world_size=self.world_size,
                local_batch_size=(self.super_batch_size if self.training_role == 'policy'
                                  else self.wm_local_batch_size))
            task = self._serial_prefetch.start()
            if self.training_role == "policy":
                self.data_fetching_task = task
            else:
                self.wm_data_fetching_task = task
        elif not self.fresh_policy_replay and self.training_role in ("combined", "policy"):
            self.data_fetching_task = asyncio.get_event_loop().create_task(self._data_fetching_loop())
        if not self.cpu_replay_prefetch and self.training_role in ("combined", "ctrl", "reward"):
            self.wm_data_fetching_task = asyncio.get_event_loop().create_task(self._wm_data_fetching_loop())
        print(f"[Trainer owner] role={self.training_role} rank={self.rank} pid={os.getpid()} "
              f"group_port={self.deepspeed_group_port} engines={list(self.get_training_state()['engines'])}",
              flush=True)
    async def save_agent(self, ckpt_dir, step):
        os.makedirs(ckpt_dir, exist_ok=True)
        self.base_model.save_model(ckpt_dir, epoch=step)
        print(f"[Trainer {self.rank}] 已保存 checkpoint -> {ckpt_dir}/agent_lora_epoch_{step}, agent_extra_layers_epoch_{step}.pt")

    def _get_current_lr(self, current_step, peak_lr, warmup_steps, total_steps, start_step=0):
        return cosine_warmup_lr(
            current_step=current_step,
            peak_lr=peak_lr,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            start_step=start_step,
        )

    def configure_cost_scheduler(self, broker, settings, source_sha256):
        from cost_scheduler_runtime import TrainingAdmissionClient
        if self.training_role != 'policy':
            raise ValueError('Only reviewed Policy ZeRO2 owners use this adapter')
        self.cost_admission = TrainingAdmissionClient(broker,settings,source_sha256,ray_module=ray)
        return {'backend':'policy_zero2','boundary':'optimizer_update','rank':self.rank}

    def _train_policy(self, inputs_batch, act_token_t, adv_t, logits_old_t, v_targ_t, global_mean, global_std, current_lrs):
        from cost_scheduler_runtime import scheduled_policy_update
        with scheduled_policy_update(self):
            """Policy 训练（包含 Value 网络）— 与原版完全一致"""
            epoch_losses, epoch_p_losses, epoch_v_losses, epoch_e_losses, epoch_kl_losses = [], [], [], [], []
            epoch_ent, epoch_kl_divs = [], []
            before_steps, before_skips = int(self.model.global_steps), int(self.model.skipped_steps)

            num_updates_in_epoch = self.super_batch_size // self.train_batch_size
            t_policy_train_start = time.time()

            for i in range(num_updates_in_epoch):
                start = i * self.train_batch_size
                end = start + self.train_batch_size
                mini_inputs = {k: v[start:end] for k, v in inputs_batch.items()}
                action_mask = mini_inputs.pop('_fresh_action_mask', None)
                def token_mean(value):
                    if action_mask is None:
                        return torch.mean(value)
                    if value.shape != action_mask.shape:
                        raise ValueError('FRESH loss mask must match actual action-token geometry')
                    return (value * action_mask.to(value.dtype)).sum() / action_mask.sum().clamp_min(1)

                mini_act_token = act_token_t[start:end]
                mini_adv = adv_t[start:end]
                mini_logits_old = logits_old_t[start:end]
                mini_v_targ = v_targ_t[start:end]

                normalized_adv = (mini_adv - global_mean) / (global_std + 1e-8)
                capture_micro = micro_begin(self, "policy_train", self.model, len(mini_adv), start, end)
                action_logits, value = self.model.forward(mini_inputs)
                value = value.to(torch.float32)

                action_logits_reshape = action_logits.view(
                    -1, NUM_ACTIONS_CHUNK, ACTION_DIM, action_logits.shape[-1]
                )

                value_loss = self.vf_coef * torch.mean((value - mini_v_targ) ** 2)

                if self.global_step < self.policy_train_start_step:
                    loss = value_loss
                    policy_loss = torch.tensor(0.0, device=loss.device)
                    ent_loss = torch.tensor(0.0, device=loss.device)
                    kl_loss = torch.tensor(0.0, device=loss.device)
                    kl_div = 0.0
                    ent = torch.tensor(0.0, device=loss.device)
                else:
                    dist = torch.distributions.Categorical(logits=action_logits_reshape)
                    logp = dist.log_prob(mini_act_token)

                    with torch.no_grad():
                        dist_old = torch.distributions.Categorical(logits=mini_logits_old)
                        logp_old = dist_old.log_prob(mini_act_token)

                    kl_div_tensor = kl.kl_divergence(dist_old, dist)
                    kl_div = token_mean(kl_div_tensor).item()
                    kl_loss = self.kl_coef * token_mean(kl_div_tensor)
                    ratio = torch.exp(logp - logp_old)
                    adv_unsqueezed = normalized_adv.unsqueeze(dim=-1).unsqueeze(dim=-1)
                    surr1 = ratio * adv_unsqueezed
                    if self.clip_mode == "gipo":
                        eps = 1e-9
                        sigma = 1.0
                        r_detach = ratio.clamp_min(eps).detach()
                        coeff = torch.exp(-0.5 * (torch.log(r_detach) / sigma) ** 2)
                        surr_soft = surr1 * coeff
                        policy_loss = -token_mean(surr_soft)
                    elif self.clip_mode == "ppo":
                        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_unsqueezed
                        policy_loss = -token_mean(torch.min(surr1, surr2))
                    elif self.clip_mode == "sapo":
                        tau_pos = 1.0
                        tau_neg = 2.0
                        if tau_pos <= 0 or tau_neg <= 0:
                            raise ValueError(f"tau_pos/tau_neg must be > 0, got {tau_pos}, {tau_neg}")
                        ratio_min = 1e-6
                        ratio_max = 1e6
                        r = ratio.clamp(ratio_min, ratio_max)
                        tau_pos_t = torch.full_like(adv_unsqueezed, tau_pos)
                        tau_neg_t = torch.full_like(adv_unsqueezed, tau_neg)
                        tau = torch.where(adv_unsqueezed > 0, tau_pos_t, tau_neg_t)
                        x = tau * (r - 1.0)
                        gate = torch.sigmoid(x) * (4.0 / tau)
                        surr_sapo = gate * adv_unsqueezed
                        policy_loss = -token_mean(surr_sapo)
                    else:
                        raise ValueError(f"Invalid CLIP_MODE: {self.clip_mode}")
                    ent = token_mean(dist.entropy())
                    ent_loss = -self.ent_coef * ent
                    loss = policy_loss + value_loss + ent_loss + kl_loss

                self.model.backward(loss)
                micro_steps_before, micro_skips_before = int(self.model.global_steps), int(self.model.skipped_steps)
                self.model.step()
                epoch_losses.append(loss.item())
                epoch_p_losses.append(policy_loss.item())
                epoch_v_losses.append(value_loss.item())
                epoch_e_losses.append(ent_loss.item())
                epoch_kl_losses.append(kl_loss.item())
                epoch_ent.append(ent.item())
                epoch_kl_divs.append(kl_div)
                micro_end(self, capture_micro, self.model)
                _trim_cuda_cache("micro")
                self.global_step += successful_updates(micro_steps_before, micro_skips_before, self.model)

            self._verify_optimizer_update(self.model, "policy", before_steps, before_skips)
            _trim_cuda_cache("idle")
            policy_train_time = time.time() - t_policy_train_start

            return {
                'epoch_losses': epoch_losses,
                'epoch_p_losses': epoch_p_losses,
                'epoch_v_losses': epoch_v_losses,
                'epoch_e_losses': epoch_e_losses,
                'epoch_kl_losses': epoch_kl_losses,
                'epoch_ent': epoch_ent,
                'epoch_kl_divs': epoch_kl_divs,
                'policy_train_time': policy_train_time,
            }


    def _train_ctrl_world(self, pixel, action, text, current_lrs):
        """Train Ctrl-World unet/action_encoder on real multi-view windows."""
        should_train = (self.global_step % self.ctrl_train_interval == 0)
        epoch_losses = []
        train_time = 0.0
        if not should_train:
            return {
                'should_train': False,
                'epoch_ctrl_losses': epoch_losses,
                'ctrl_train_time': train_time,
            }

        ctrl_lr = self._get_current_lr(
            self.ctrl_step, self.ctrl_lr, self.ctrl_warmup_steps, self.total_ctrl_steps
        )
        for pg in self.ctrl_world_engine.optimizer.param_groups:
            pg['lr'] = ctrl_lr
        current_lrs['ctrl'] = ctrl_lr

        before_steps, before_skips = int(self.ctrl_world_engine.global_steps), int(self.ctrl_world_engine.skipped_steps)
        self.ctrl_world_engine.train()
        num_updates = self.ctrl_super_batch_size // self.ctrl_batch_size
        grad_norms = []
        t0 = time.time()
        # Use only the first ctrl_super_batch_size samples from the shared WM batch.
        pixel = pixel[: self.ctrl_super_batch_size]
        if self.world_config.backend == "ctrl":
            action = action[: self.ctrl_super_batch_size]
            text = list(text)[: self.ctrl_super_batch_size]

        for i in range(num_updates):
            start = i * self.ctrl_batch_size
            end = start + self.ctrl_batch_size
            batch = (self.world_backend.prepare_train_batch(pixel[start:end])
                     if self.world_config.backend == "weaver" else {
                         'pixel': pixel[start:end], 'action': action[start:end], 'text': text[start:end]})
            capture_micro = micro_begin(self, "ctrl_train", self.ctrl_world_engine, end-start, start, end)
            ctrl_loss, loss_components = self.ctrl_world_engine(batch)
            if self.world_config.backend == "weaver":
                self.telemetry.event("world_backend_losses", details={
                    "backend": self.world_config.backend, "rank": self.rank,
                    "update_attempt": self.ctrl_step + 1, "micro": i,
                    "losses": {key: float(value.detach().item()) if hasattr(value, "detach") else float(value)
                               for key, value in loss_components.items()}})
            self.ctrl_world_engine.backward(ctrl_loss)
            if self.ctrl_world_engine.is_gradient_accumulation_boundary():
                grad_norm = self.ctrl_world_engine.get_global_grad_norm()
                if grad_norm is not None and math.isfinite(float(grad_norm)):
                    grad_norms.append(float(grad_norm))
            self.ctrl_world_engine.step()
            epoch_losses.append(float(ctrl_loss.detach().item()))
            micro_end(self, capture_micro, self.ctrl_world_engine)
            _trim_cuda_cache("micro")

        train_time = time.time() - t0
        self._verify_optimizer_update(self.ctrl_world_engine, "ctrl", before_steps, before_skips)
        self.ctrl_step += successful_updates(before_steps, before_skips, self.ctrl_world_engine)
        if self.world_config.backend == "weaver":
            self.world_backend.update_ema()
        _trim_cuda_cache("idle")
        return {
            'should_train': True,
            'epoch_ctrl_losses': epoch_losses,
            'ctrl_train_time': train_time,
            'ctrl_grad_norm': float(np.mean(grad_norms)) if grad_norms else float('nan'),
            'ctrl_clip_ratio': float(np.mean(np.asarray(grad_norms) > 1.0)) if grad_norms else float('nan'),
        }

    def _train_reward_model(self, wm_reward_inputs, wm_labels, current_lrs):
        """Reward Model 训练 (bf16) — Diamond-aligned: window last-frame + sparse rew."""
        should_train = (self.global_step % self.reward_train_interval == 0)
        epoch_reward_losses = []
        epoch_reward_tp, epoch_reward_tn, epoch_reward_fp, epoch_reward_fn = 0, 0, 0, 0
        reward_train_time = 0.0

        if not should_train:
            return {
                'should_train': False,
                'epoch_reward_losses': epoch_reward_losses,
                'reward_train_time': reward_train_time,
                'reward_tp': epoch_reward_tp,
                'reward_tn': epoch_reward_tn,
                'reward_fp': epoch_reward_fp,
                'reward_fn': epoch_reward_fn,
            }

        # Match Diamond: schedule reward LR against train_iters (not planned reward updates).
        reward_lr = self._get_current_lr(
            self.reward_step,
            self.reward_lr,
            self.reward_warmup_steps,
            self.train_iters,
        )
        for pg in self.reward_engine.optimizer.param_groups:
            pg['lr'] = reward_lr
        current_lrs['reward'] = reward_lr

        before_steps, before_skips = int(self.reward_engine.global_steps), int(self.reward_engine.skipped_steps)
        self.reward_engine.train()
        num_reward_updates = self.reward_super_batch_size // self.reward_batch_size
        grad_norms = []
        t_reward_train_start = time.time()

        for i in range(num_reward_updates):
            start = i * self.reward_batch_size
            end = start + self.reward_batch_size
            mini_inputs = {k: v[start:end] for k, v in wm_reward_inputs.items()}
            mini_labels = wm_labels[start:end]

            capture_micro = micro_begin(self, "reward_train", self.reward_engine, len(mini_labels), start, end)
            # Engine.forward installs ZeRO accumulation cleanup and GAS scaling hooks.
            logits = self.reward_engine(mini_inputs)
            reward_loss = self.reward_model.focal_loss(logits, mini_labels)
            with torch.no_grad():
                preds_bool, labels_bool = logits.argmax(dim=-1).bool(), mini_labels.bool()
                metrics = {"tp": (preds_bool & labels_bool).sum(),
                           "tn": (~preds_bool & ~labels_bool).sum(),
                           "fp": (preds_bool & ~labels_bool).sum(),
                           "fn": (~preds_bool & labels_bool).sum()}
            self.reward_engine.backward(reward_loss)
            if self.reward_engine.is_gradient_accumulation_boundary():
                grad_norm = self.reward_engine.get_global_grad_norm()
                if grad_norm is not None and math.isfinite(float(grad_norm)):
                    grad_norms.append(float(grad_norm))
            self.reward_engine.step()

            epoch_reward_losses.append(reward_loss.item())
            epoch_reward_tp += metrics["tp"].item()
            epoch_reward_tn += metrics["tn"].item()
            epoch_reward_fp += metrics["fp"].item()
            epoch_reward_fn += metrics["fn"].item()
            micro_end(self, capture_micro, self.reward_engine)
            _trim_cuda_cache("micro")

        reward_train_time = time.time() - t_reward_train_start
        self._verify_optimizer_update(self.reward_engine, "reward", before_steps, before_skips)
        self.reward_step += successful_updates(before_steps, before_skips, self.reward_engine)
        _trim_cuda_cache("idle")

        return {
            'should_train': True,
            'epoch_reward_losses': epoch_reward_losses,
            'reward_train_time': reward_train_time,
            'reward_tp': epoch_reward_tp,
            'reward_tn': epoch_reward_tn,
            'reward_fp': epoch_reward_fp,
            'reward_fn': epoch_reward_fn,
            'reward_grad_norm': float(np.mean(grad_norms)) if grad_norms else float('nan'),
            'reward_clip_ratio': float(np.mean(np.asarray(grad_norms) > 1.0)) if grad_norms else float('nan'),
        }

    def _prepare_policy_batch(self, obs_list, action_token_np, adv_np, logits_old_np, v_targ_np):
        """Policy batch preparation; runs in a worker thread, off the event loop."""
        masks = [obs.get('_fresh_action_mask') for obs in obs_list]
        if any(mask is not None for mask in masks):
            if any(mask is None for mask in masks):
                raise ValueError('A FRESH Policy cohort cannot mix masked and unmasked transitions')
            inputs = [{key:value for key,value in obs.items() if key != '_fresh_action_mask'} for obs in obs_list]
        else:
            inputs = obs_list
        inputs_batch = self.base_model.prepare_inputs_batch(inputs)
        device = next(self.model.parameters()).device
        if masks and masks[0] is not None:
            inputs_batch['_fresh_action_mask'] = torch.as_tensor(np.stack(masks), dtype=torch.bool, device=device)
        act_token_t = torch.tensor(action_token_np, dtype=torch.long, device=device)
        adv_t = torch.tensor(adv_np, dtype=torch.float32, device=device)
        logits_old_t = torch.tensor(logits_old_np, dtype=torch.float32, device=device)
        v_targ_t = torch.tensor(v_targ_np, dtype=torch.float32, device=device)
        return inputs_batch, act_token_t, adv_t, logits_old_t, v_targ_t

    async def _data_fetching_loop(self):
        """Actor 数据获取循环 — 与原版一致"""
        print(f"Trainer {self.rank}: 后台数据准备循环已启动 (超级批次大小: {self.super_batch_size})。")
        loop = asyncio.get_running_loop()
        while True:
            try:
                if self.next_ready_batch is not None:
                    # run_training_epoch sets this right after taking the batch,
                    # allowing the next Replay request to be submitted. Its reply
                    # and CPU preparation dispatch may still wait for training.
                    self._policy_slot_freed.clear()
                    await self._policy_slot_freed.wait()
                    continue

                if not self._policy_replay_ready:
                    required = self.super_batch_size * (self.world_size if self.batch_coordinator else 1)
                    while await self.replay_buffer.size.remote() < required:
                        print(f"Trainer {self.rank} (BG): 等待 ReplayBuffer 填充至 {self.super_batch_size}...")
                        await asyncio.sleep(3)
                    # The replay is a bounded deque, so its size never drops again.
                    self._policy_replay_ready = True

                # sample_time spans submission to resumption; resumption waits for
                # the synchronous training body, so it includes that overlap.
                t_sample_start = time.time()
                obs_list, action_token_np, adv_np, logits_old_np, v_targ_np, capture_meta = \
                    await (self.batch_coordinator.sample_policy.remote(self.rank, self.super_batch_size)
                           if self.batch_coordinator else self.replay_buffer.sample.remote(self.super_batch_size))
                sample_time = time.time() - t_sample_start
                self.telemetry.event("data_fetch", start_ns=int(t_sample_start*1e9), end_ns=now_ns(), duration_ns=int(sample_time*1e9), replay="policy")

                t_prep_start = time.time()
                inputs_batch, act_token_t, adv_t, logits_old_t, v_targ_t = await loop.run_in_executor(
                    self._prep_executor, self._prepare_policy_batch,
                    obs_list, action_token_np, adv_np, logits_old_np, v_targ_np)
                prep_time = time.time() - t_prep_start
                self.telemetry.event("data_prepare", start_ns=int(t_prep_start*1e9), end_ns=now_ns(), duration_ns=int(prep_time*1e9), replay="policy")

                self.next_ready_batch = {
                    'inputs_batch': inputs_batch,
                    'act_token': act_token_t,
                    'advantage': adv_t,
                    'logits_old': logits_old_t,
                    'value_target': v_targ_t,
                    'sample_time': sample_time,
                    'prep_time': prep_time,
                    'capture_meta': capture_meta,
                }

            except Exception as e:
                if self.batch_coordinator:
                    # The other rank may already hold this global batch's half.
                    self.data_fetch_error = RuntimeError(f"Policy batch preparation failed on rank {self.rank}: {e}")
                    self.telemetry.event("data_prepare_failed", replay="policy", status="error", error=repr(e))
                    return
                print(f"Trainer {self.rank}: 数据采样失败: {e}。将在3秒后重试。")
                await asyncio.sleep(3)

    def _prepare_wm_batch(self, obs_np, act_np, rew_np, instructions, backend_samples=None):
        """Prepare only the owning model's inputs from the shared CPU shard."""
        prepared = {}
        if self.training_role in ("combined", "ctrl"):
            if self.world_config.backend == "weaver":
                from replay_projection import normalized_observations
                if backend_samples is None or len(backend_samples) != len(obs_np):
                    raise ValueError("WEAVER replay payload missing or not aligned with rank shard")
                world_obs = normalized_observations(obs_np)
                samples = [dict(payload, obs=world_obs[i], actions=act_np[i], instruction=instructions[i])
                           for i, payload in enumerate(backend_samples)]
                prepared.update(pixel=samples, action=None, text=None)
            else:
                prepared.update(self.world_backend.prepare_train_batch({"obs": obs_np, "actions": act_np,
                                "rewards": rew_np, "instructions": instructions}))
        if self.training_role in ("combined", "reward"):
            prepared.update(self._prepare_reward_batch(obs_np, rew_np, instructions))
        return prepared

    def _prepare_ctrl_batch(self, obs_np, act_np, rew_np, instructions):
        device = next(self.ctrl_world_engine.parameters()).device
        # ---- Ctrl-World batch: resize multi-view pixels + bounds-normalize actions ----
        # Canonical uint8 pixels normalize only in this owning trainer.
        from replay_projection import normalized_observations
        obs_t = torch.as_tensor(normalized_observations(obs_np), dtype=torch.float32)
        B, nF, M, C, H, W = obs_t.shape
        flat = obs_t.reshape(B * nF * M, C, H, W)
        flat = torch.nn.functional.interpolate(
            flat,
            size=(self.ctrl_target_height, self.ctrl_target_width),
            mode="bilinear",
            align_corners=False,
        )
        pixel = flat.reshape(B, nF, M, C, self.ctrl_target_height, self.ctrl_target_width)
        # Convert pixels to bf16 on the CPU before transfer; the destination
        # tensor is bf16. This alone does not establish overlap with training.
        pixel = pixel.to(dtype=torch.bfloat16).to(device=device)

        act_t = torch.as_tensor(act_np, dtype=torch.float32, device=device)
        if act_t.shape[1] != nF:
            raise ValueError(
                f"Ctrl-World expects frame-aligned actions [B,{nF},A], got {tuple(act_t.shape)}"
            )
        action = _normalize_actions_bounds(act_t, self.condition_low, self.condition_high)
        action = action.to(dtype=torch.bfloat16)


        ctrl_n = min(self.ctrl_super_batch_size, B)
        return {'pixel': pixel[:ctrl_n], 'action': action[:ctrl_n],
                'text': list(instructions[:ctrl_n]),
                'rew': torch.tensor(rew_np[:ctrl_n], dtype=torch.float32, device=device)}

    def _prepare_reward_batch(self, obs_np, rew_np, instructions):
        device = next(self.reward_engine.parameters()).device
        # Diamond-aligned reward: last primary-cam frame + sparse window reward.
        # Accept the compact RPC projection and the legacy combined-role path.
        from replay_projection import reward_last_primary
        last_primary = reward_last_primary(obs_np)
        reward_n = min(self.reward_super_batch_size, len(last_primary))
        inputs_list = []
        for i in range(reward_n):
            obs_tensor = last_primary[i]
            obs_np_hwc = np.transpose(obs_tensor, (1, 2, 0))
            obs_img = ((obs_np_hwc + 1) / 2 * 255).astype(np.uint8)
            inputs = prepare_one_obs(
                self.reward_cfg,
                self.processor,
                {"full_image": obs_img},
                instructions[i],
                torch.bfloat16,
            )
            inputs_list.append(inputs)

        reward_inputs_batch = self.reward_model.prepare_inputs_batch(inputs_list)
        labels = torch.tensor(
            (rew_np[:reward_n] > 0).astype(np.int64),
            dtype=torch.long,
            device=device,
        )
        reward_pos_ratio_batch = (
            float(labels.float().mean().item()) if labels.numel() else 0.0
        )


        return {'reward_inputs_batch': reward_inputs_batch, 'labels': labels,
                'reward_pos_ratio_batch': reward_pos_ratio_batch}
    async def _wm_data_fetching_loop(self):
        """Prepare Ctrl-World windows + Diamond-style reward batches from WM windows."""
        wm_super_batch_size = self.wm_local_batch_size
        print(
            f"Trainer {self.rank}: WM 后台数据准备循环已启动 "
            f"(wm_super={wm_super_batch_size}, "
            f"ctrl_super={self.ctrl_super_batch_size}, "
            f"reward_super={self.reward_super_batch_size})."
        )

        loop = asyncio.get_running_loop()
        while True:
            try:
                if self.next_ready_wm_batch is not None:
                    # Set by run_training_epoch once a WM-training epoch takes the batch.
                    self._wm_slot_freed.clear()
                    await self._wm_slot_freed.wait()
                    continue

                if not self._wm_replay_ready:
                    required = wm_super_batch_size * (self.world_size if self.batch_coordinator else 1)
                    while await self.wm_replay_buffer.size.remote() < required:
                        ctrl_size = await self.wm_replay_buffer.size.remote()
                        print(
                            f"Trainer {self.rank} (BG-WM): 等待 WM ReplayBuffer "
                            f"填充至 {wm_super_batch_size} (当前: {ctrl_size})..."
                        )
                        await asyncio.sleep(3)
                    # The WM replay is a bounded deque, so its size never drops again.
                    self._wm_replay_ready = True

                t_sample_start = time.time()
                obs_np, act_np, rew_np, instructions, ctrl_sample_stats = \
                    await (self.batch_coordinator.sample_wm.remote(
                               self.rank, wm_super_batch_size,
                               self.training_role if self.training_role in ("ctrl", "reward") else "combined")
                           if self.batch_coordinator else self.wm_replay_buffer.sample.remote(wm_super_batch_size))
                sample_time = time.time() - t_sample_start
                self.telemetry.event("data_fetch", start_ns=int(t_sample_start*1e9), end_ns=now_ns(), duration_ns=int(sample_time*1e9), replay="wm")

                t_prep_start = time.time()
                prepared = await loop.run_in_executor(
                    self._prep_executor, self._prepare_wm_batch, obs_np, act_np, rew_np, instructions,
                    ctrl_sample_stats.get("backend_samples"))
                prep_time = time.time() - t_prep_start
                self.telemetry.event("data_prepare", start_ns=int(t_prep_start*1e9), end_ns=now_ns(), duration_ns=int(prep_time*1e9), replay="wm")

                self.last_wm_batch_info = {
                    "sample_time": sample_time,
                    "prep_time": prep_time,
                    "unique_episodes": int(ctrl_sample_stats['unique_episodes']),
                    "fresh_episode_ratio": float(ctrl_sample_stats['fresh_episode_ratio']),
                }
                self.next_ready_wm_batch = {
                    **prepared,
                    'instructions': instructions,
                    'capture_meta': ctrl_sample_stats['capture_meta'],
                    'ctrl_sample_unique_episodes': int(ctrl_sample_stats['unique_episodes']),
                    'ctrl_sample_fresh_episode_ratio': float(ctrl_sample_stats['fresh_episode_ratio']),
                    'sample_time': sample_time,
                    'prep_time': prep_time,
                }

            except Exception as e:
                import traceback
                traceback.print_exc()
                if self.batch_coordinator:
                    self.data_fetch_error = RuntimeError(f"WM batch preparation failed on rank {self.rank}: {e}")
                    self.telemetry.event("data_prepare_failed", replay="wm", status="error", error=repr(e))
                    return
                print(f"Trainer {self.rank}: WM 数据采样失败: {e}。将在3秒后重试。")
                await asyncio.sleep(3)

    async def _wait_for_batch(self, attr, label):
        """Yield to the event loop until the fetch loop publishes a batch; returns seconds waited."""
        t_wait_start = time.time()
        if self.data_fetch_error is not None:
            raise self.data_fetch_error
        if getattr(self, attr) is None and self.global_step == 0:
            print(f"Trainer {self.rank}: 等待初始 {label} 超级批次...")
        while getattr(self, attr) is None:
            if self.data_fetch_error is not None:
                raise self.data_fetch_error
            await asyncio.sleep(0.05)
        if self.data_fetch_error is not None:
            raise self.data_fetch_error
        return time.time() - t_wait_start

    async def _run_policy_epoch(self):
        # ============ 等待数据准备 ============
        # The WM batch is awaited only in epochs that train Ctrl-World/Reward (below).
        policy_wait_time = await self._wait_for_batch("next_ready_batch", "Actor")

        current_lrs = {}
        if self.data_fetch_error:
            raise self.data_fetch_error
        value_lr = self._get_current_lr(self.global_step, self.value_lr, self.value_warmup_steps, self.train_iters)
        policy_lr = self._get_current_lr(self.global_step, self.policy_lr, self.policy_warmup_steps, self.train_iters, start_step=self.policy_train_start_step)

        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'value':
                param_group['lr'] = value_lr; current_lrs['value'] = value_lr
            elif param_group['name'] == 'policy':
                param_group['lr'] = policy_lr; current_lrs['policy'] = policy_lr

        current_batch = self.next_ready_batch
        self.capture_policy_metadata = current_batch['capture_meta']
        self.next_ready_batch = None
        # Yield once to let the fetch loop submit the next Replay request. If its
        # reply arrives later, CPU preparation dispatch still waits until the
        # synchronous training body releases the event loop.
        self._policy_slot_freed.set()
        await asyncio.sleep(0)
        if self.data_fetch_error is not None:
            raise self.data_fetch_error

        inputs_batch = current_batch['inputs_batch']
        act_token_t = current_batch['act_token']
        adv_t = current_batch['advantage']
        logits_old_t = current_batch['logits_old']
        v_targ_t = current_batch['value_target']
        policy_sample_time = current_batch['sample_time']
        policy_prep_time = current_batch['prep_time']

        # 修正std 归一化
        local_sum = adv_t.sum()
        local_sq_sum = (adv_t * adv_t).sum()
        local_count = torch.tensor([adv_t.numel()], device=adv_t.device, dtype=torch.float32)

        stats_tensor = torch.stack([local_sum, local_sq_sum, local_count.squeeze(0)])
        distributed.all_reduce(stats_tensor, op=distributed.ReduceOp.SUM)

        global_sum, global_sq_sum, global_count = stats_tensor[0], stats_tensor[1], stats_tensor[2]
        global_mean = global_sum / torch.clamp(global_count, min=1.0)
        global_var = torch.clamp(global_sq_sum / torch.clamp(global_count, min=1.0) - global_mean * global_mean, min=1e-12)
        global_std = torch.sqrt(global_var)

        policy_result = self._train_policy(inputs_batch, act_token_t, adv_t, logits_old_t, v_targ_t, global_mean, global_std, current_lrs)


        perf_metrics = {
            "policy_sample_time": policy_sample_time, "policy_prep_time": policy_prep_time,
            "policy_train_time": policy_result["policy_train_time"],
            "policy_batch_wait_time": policy_wait_time,
            "wm_sample_time": 0.0, "wm_prep_time": 0.0, "wm_batch_wait_time": 0.0,
        }
        return (np.mean(policy_result["epoch_losses"]), np.mean(policy_result["epoch_p_losses"]),
                np.mean(policy_result["epoch_v_losses"]), np.mean(policy_result["epoch_e_losses"]),
                np.mean(policy_result["epoch_kl_losses"]), current_lrs, self.global_step,
                np.mean(policy_result["epoch_ent"]), np.mean(policy_result["epoch_kl_divs"]),
                perf_metrics, {"trained_ctrl": False, "trained_reward": False})

    async def _run_auxiliary_epoch(self, policy_step):
        if (isinstance(policy_step, bool) or not isinstance(policy_step, int)
                or policy_step != self.global_step + 1):
            raise ValueError("Auxiliary trainer requires the next shared Policy boundary")
        self.global_step = policy_step
        ctrl_due = "ctrl" in self.active_kinds and policy_step % self.ctrl_train_interval == 0
        reward_due = "reward" in self.active_kinds and policy_step % self.reward_train_interval == 0
        wm_batch, wm_wait_time, cohort = None, 0.0, None
        # Both consumers advance the same Replay cohort at the union of schedules.
        # Off-schedule owners claim/discard a prepared shard without recording use.
        if ctrl_due or reward_due:
            wm_wait_time = await self._wait_for_batch("next_ready_wm_batch", "WM")
            wm_batch = self.next_ready_wm_batch
            self.capture_wm_metadata = wm_batch["capture_meta"]
            self.next_ready_wm_batch = None
            self._wm_slot_freed.set()
            await asyncio.sleep(0)
            if self.data_fetch_error is not None:
                raise self.data_fetch_error
            import hashlib
            meta = self.capture_wm_metadata
            cohort = {
                "reservation_id": meta["reservation_id"],
                "shard_start": meta["shard_start"], "shard_end": meta["shard_end"],
                "sample_ids_sha256": hashlib.sha256(
                    json.dumps(meta["sample_ids"], separators=(",", ":")).encode()).hexdigest(),
            }
        current_lrs = {}
        wm_info = self.last_wm_batch_info
        metrics = {"trained_ctrl": False, "trained_reward": False,
                   "ctrl_sample_unique_episodes": wm_info["unique_episodes"],
                   "ctrl_sample_fresh_episode_ratio": wm_info["fresh_episode_ratio"]}
        performance = {"wm_sample_time": wm_info["sample_time"],
                       "wm_prep_time": wm_info["prep_time"],
                       "wm_batch_wait_time": wm_wait_time}
        if self.training_role == "ctrl":
            result = self._train_ctrl_world(
                wm_batch["pixel"] if wm_batch else None,
                wm_batch["action"] if wm_batch else None,
                wm_batch["text"] if wm_batch else None, current_lrs)
            metrics["trained_ctrl"] = result["should_train"]
            if result["should_train"]:
                metrics.update(ctrl_loss=float(np.mean(result["epoch_ctrl_losses"])),
                               ctrl_grad_norm=result["ctrl_grad_norm"], ctrl_clip_ratio=result["ctrl_clip_ratio"])
                performance["ctrl_train_time"] = result["ctrl_train_time"]
            model_step = self.ctrl_step
        elif self.training_role == "reward":
            result = self._train_reward_model(
                wm_batch["reward_inputs_batch"] if wm_batch else None,
                wm_batch["labels"] if wm_batch else None, current_lrs)
            metrics["trained_reward"] = result["should_train"]
            if result["should_train"]:
                metrics.update(reward_loss=float(np.mean(result["epoch_reward_losses"])),
                               reward_grad_norm=result["reward_grad_norm"],
                               reward_clip_ratio=result["reward_clip_ratio"],
                               reward_pos_ratio_batch=float(wm_batch["reward_pos_ratio_batch"]))
                for key in ("reward_tp", "reward_tn", "reward_fp", "reward_fn"):
                    metrics[key] = result[key]
                performance["reward_train_time"] = result["reward_train_time"]
            model_step = self.reward_step
        else:
            raise RuntimeError("Auxiliary epoch dispatched to a non-auxiliary owner")
        return {"kind": self.training_role, "rank": self.rank, "policy_step": policy_step,
                "model_step": model_step, "learning_rates": current_lrs,
                "metrics": metrics, "performance": performance, "wm_cohort": cohort}

    async def _serial_fetch_cpu_batch(self):
        """Reserve the next raw shard before permission; never prepare GPU inputs."""
        policy = self.training_role == "policy"
        local_size = self.super_batch_size if policy else self.wm_local_batch_size
        replay = self.replay_buffer if policy else self.wm_replay_buffer
        ready_attr = "_policy_replay_ready" if policy else "_wm_replay_ready"
        if not getattr(self, ready_attr):
            required = local_size * (self.world_size if self.batch_coordinator else 1)
            while await replay.size.remote() < required:
                await asyncio.sleep(3)
            setattr(self, ready_attr, True)
        started = time.time()
        if self.batch_coordinator:
            raw = await (self.batch_coordinator.sample_policy.remote(self.rank, local_size) if policy
                         else self.batch_coordinator.sample_wm.remote(self.rank, local_size, self.training_role))
        else:
            raw = await replay.sample.remote(local_size)
        elapsed = time.time() - started
        self.telemetry.event("data_fetch", start_ns=int(started * 1e9), end_ns=now_ns(),
                             duration_ns=int(elapsed * 1e9), replay="policy" if policy else "wm",
                             serial_cpu_prefetch=True, gpu_prepared=False)
        return {"raw": raw, "sample_time": elapsed}

    async def _serial_publish_batch(self, policy_step):
        """Only the active owner transfers/prepares its raw CPU shard on GPU.

        This helper returns no GPU values. Its preparation Future and locals
        disappear before the learning RPC returns and before the final trim.
        The background producer only ever sees raw CPU Replay data.
        """
        key = (policy_step, self.training_role)
        if self._serial_phase.active != key:
            raise RuntimeError("GPU preparation attempted without its serial stage")
        policy = self.training_role == "policy"
        union_due = (("ctrl" in self.active_kinds and policy_step % self.ctrl_train_interval == 0) or
                     ("reward" in self.active_kinds and policy_step % self.reward_train_interval == 0))
        if not policy and not union_due:
            return
        if policy and self.fresh_policy_replay:
            # Draw only at the actual update boundary, after the preceding
            # publication. Earlier prefetch would invalidate current-critic and
            # consumption-age contracts before this cohort reaches the model.
            record = await self._serial_fetch_cpu_batch()
        else:
            record = await self._serial_prefetch.take()
        raw = record["raw"]
        loop = asyncio.get_running_loop()
        started = time.time()
        if policy:
            future = loop.run_in_executor(self._prep_executor, self._prepare_policy_batch, *raw[:-1])
            prepared = await finish_preparation(future)
            future = None
            inputs, actions, advantage, old_logits, target = prepared
            self.next_ready_batch = {"inputs_batch": inputs, "act_token": actions,
                                     "advantage": advantage, "logits_old": old_logits,
                                     "value_target": target, "capture_meta": raw[-1],
                                     "sample_time": record["sample_time"], "prep_time": time.time() - started}
        else:
            stats = raw[-1]
            interval = self.ctrl_train_interval if self.training_role == "ctrl" else self.reward_train_interval
            if policy_step % interval == 0:
                future = loop.run_in_executor(self._prep_executor, self._prepare_wm_batch, *raw[:-1], stats.get("backend_samples"))
                prepared = await finish_preparation(future)
                future = None
            else:
                # An off-schedule owner still claims the shared cohort, but
                # never allocates GPU inputs for a model that will not update.
                prepared = {"pixel": None, "action": None, "text": None,
                            "reward_inputs_batch": None, "labels": None}
            self.last_wm_batch_info = {"sample_time": record["sample_time"], "prep_time": time.time() - started,
                                       "unique_episodes": int(stats["unique_episodes"]),
                                       "fresh_episode_ratio": float(stats["fresh_episode_ratio"])}
            self.next_ready_wm_batch = {**prepared, "capture_meta": stats["capture_meta"],
                                        "sample_time": record["sample_time"], "prep_time": time.time() - started}
        self.telemetry.event("data_prepare", start_ns=int(started * 1e9), end_ns=now_ns(),
                             duration_ns=int((time.time() - started) * 1e9),
                             replay="policy" if policy else "wm", serial_phase_key=list(key))

    def _serial_key_barrier(self, key, boundary):
        """All four ranks agree on the same step/model/boundary before ACK."""
        model_id = ("policy", "ctrl", "reward").index(key[1])
        token = torch.tensor([key[0], model_id, boundary], dtype=torch.int64, device="cuda")
        tokens = [torch.empty_like(token) for _ in range(self.world_size)]
        distributed.all_gather(tokens, token)
        values = [item.cpu().tolist() for item in tokens]
        if any(value != [key[0], model_id, boundary] for value in values):
            raise RuntimeError(f"Serial four-rank phase keys disagree: {values}")

    def _serial_memory(self):
        free, total = torch.cuda.mem_get_info()
        return {"allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "free_bytes": free, "total_bytes": total}

    async def _run_serial_epoch(self, policy_step):
        key = self._serial_phase.begin(policy_step)
        try:
            self._serial_key_barrier(key, 0)
            if self.training_role == "policy":
                self._policy_joint_batch_mode.begin(policy_step)
            torch.cuda.reset_peak_memory_stats()
            self.telemetry.event("split_serial_phase_begin", details={
                "policy_step": policy_step, "model": self.training_role, "rank": self.rank,
                "phase_key": list(key), "cpu_prefetch_only": True, **self._serial_memory()})
            await self._serial_publish_batch(policy_step)
            # The unchanged epoch bodies own all learning/LR/update accounting.
            # Their local GPU inputs have gone out of scope before handoff.
            result = await (self._run_policy_epoch() if self.training_role == "policy"
                            else self._run_auxiliary_epoch(policy_step))
            if self.training_role == "policy":
                self._policy_joint_batch_mode.complete(policy_step)
            self.next_ready_batch = self.next_ready_wm_batch = None
            import gc
            gc.collect()
            torch.cuda.synchronize()
            self.telemetry.flush(force_gpu=True)
            torch.cuda.empty_cache()
            self._serial_key_barrier(key, 1)
            # The key collective's temporary tensors are now also unreferenced.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            self._serial_phase.complete(key)
            if self._serial_prefetch is not None:
                self._serial_prefetch.release_in_use()
            if self.training_role == 'policy' and self.fresh_policy_replay:
                metadata = self.capture_policy_metadata
                await self.replay_buffer.commit_fresh_use.remote(metadata['fresh_reservation_token'], metadata['sample_ids'])
            trained = self.training_role == "policy" or result["metrics"]["trained_" + self.training_role]
            self.telemetry.event("split_serial_phase_end", details={
                "policy_step": policy_step, "model": self.training_role, "rank": self.rank,
                "phase_key": list(key), "cpu_prefetch_only": True, "trained": bool(trained),
                "cohort": None if self.training_role == "policy" else result["wm_cohort"],
                "gpu_batch_refs_released": True, "completion": "cuda_sync_and_empty_cache",
                **self._serial_memory()})
            self.telemetry.flush()
            return result
        except BaseException as error:
            self._serial_phase.fail(error)
            self.next_ready_batch = self.next_ready_wm_batch = None
            if self._serial_prefetch is not None:
                await self._serial_prefetch.close()
            self.telemetry.event("split_serial_phase_failed", status="error", details={
                "phase_key": list(key), "rank": self.rank, "error": repr(error), "acknowledged": False})
            raise

    def serial_sample_identity(self, policy_step):
        if (not self.cpu_replay_prefetch or self._serial_phase.active is not None
                or self._serial_phase.completed_step != policy_step):
            raise RuntimeError("Sample identity requested before its complete serial phase ACK")
        needed = (self.training_role == "policy" or ("ctrl" in self.active_kinds and policy_step % self.ctrl_train_interval == 0)
                  or ("reward" in self.active_kinds and policy_step % self.reward_train_interval == 0))
        metadata = (getattr(self, "capture_policy_metadata", None) if self.training_role == "policy"
                    else getattr(self, "capture_wm_metadata", None))
        cohort = ({key: metadata[key] for key in ("reservation_id", "shard_start", "shard_end", "sample_ids")}
                  if needed and metadata else None)
        return {"model": self.training_role, "rank": self.rank, "policy_step": policy_step,
                "cohort": cohort}

    async def run_training_epoch(self, policy_step=None):
        for exporter in getattr(self, "_ipc_weight_exports", {}).values():
            exporter.assert_idle()
        if self.cpu_replay_prefetch:
            return await self._run_serial_epoch(policy_step)
        if self.training_role == "combined":
            if policy_step is not None:
                raise ValueError("Combined trainers own their Policy clock")
            return await self._run_combined_epoch()
        if self.training_role == "policy":
            if (isinstance(policy_step, bool) or not isinstance(policy_step, int)
                    or policy_step != self.global_step + 1):
                raise ValueError("Policy trainer requires its next shared boundary")
            return await self._run_policy_epoch()
        return await self._run_auxiliary_epoch(policy_step)

    async def _run_combined_epoch(self):
        # ============ 等待数据准备 ============
        # The WM batch is awaited only in epochs that train Ctrl-World/Reward (below).
        policy_wait_time = await self._wait_for_batch("next_ready_batch", "Actor")

        current_lrs = {}
        if self.data_fetch_error:
            raise self.data_fetch_error
        value_lr = self._get_current_lr(self.global_step, self.value_lr, self.value_warmup_steps, self.train_iters)
        policy_lr = self._get_current_lr(self.global_step, self.policy_lr, self.policy_warmup_steps, self.train_iters, start_step=self.policy_train_start_step)

        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'value':
                param_group['lr'] = value_lr; current_lrs['value'] = value_lr
            elif param_group['name'] == 'policy':
                param_group['lr'] = policy_lr; current_lrs['policy'] = policy_lr

        current_batch = self.next_ready_batch
        self.capture_policy_metadata = current_batch['capture_meta']
        self.next_ready_batch = None
        # Yield once to let the fetch loop submit the next Replay request. If its
        # reply arrives later, CPU preparation dispatch still waits until the
        # synchronous training body releases the event loop.
        self._policy_slot_freed.set()
        await asyncio.sleep(0)
        if self.data_fetch_error is not None:
            raise self.data_fetch_error

        inputs_batch = current_batch['inputs_batch']
        act_token_t = current_batch['act_token']
        adv_t = current_batch['advantage']
        logits_old_t = current_batch['logits_old']
        v_targ_t = current_batch['value_target']
        policy_sample_time = current_batch['sample_time']
        policy_prep_time = current_batch['prep_time']

        # 修正std 归一化
        local_sum = adv_t.sum()
        local_sq_sum = (adv_t * adv_t).sum()
        local_count = torch.tensor([adv_t.numel()], device=adv_t.device, dtype=torch.float32)

        stats_tensor = torch.stack([local_sum, local_sq_sum, local_count.squeeze(0)])
        distributed.all_reduce(stats_tensor, op=distributed.ReduceOp.SUM)

        global_sum, global_sq_sum, global_count = stats_tensor[0], stats_tensor[1], stats_tensor[2]
        global_mean = global_sum / torch.clamp(global_count, min=1.0)
        global_var = torch.clamp(global_sq_sum / torch.clamp(global_count, min=1.0) - global_mean * global_mean, min=1e-12)
        global_std = torch.sqrt(global_var)

        policy_result = self._train_policy(inputs_batch, act_token_t, adv_t, logits_old_t, v_targ_t, global_mean, global_std, current_lrs)

        # ============ 2. World Model 训练（Ctrl-World + Reward）============
        ctrl_should_train = (self.global_step % self.ctrl_train_interval == 0)
        reward_should_train = ("reward" in self.active_kinds and self.global_step % self.reward_train_interval == 0)
        wm_batch = None
        wm_wait_time = 0.0
        if ctrl_should_train or reward_should_train:
            wm_wait_time = await self._wait_for_batch("next_ready_wm_batch", "WM")
            wm_batch = self.next_ready_wm_batch
            self.capture_wm_metadata = wm_batch['capture_meta']
            self.next_ready_wm_batch = None
            self._wm_slot_freed.set()
            await asyncio.sleep(0)
            if self.data_fetch_error is not None:
                raise self.data_fetch_error

        # Off-schedule epochs pass None; both trainers return before reading inputs.
        ctrl_result = self._train_ctrl_world(
            wm_batch['pixel'] if wm_batch else None,
            wm_batch['action'] if wm_batch else None,
            wm_batch['text'] if wm_batch else None,
            current_lrs,
        )
        reward_result = self._train_reward_model(
            wm_batch['reward_inputs_batch'] if wm_batch else None,
            wm_batch['labels'] if wm_batch else None,
            current_lrs,
        )

        # ============ 汇总结果 ============
        avg_loss = np.mean(policy_result['epoch_losses'])
        avg_p_loss = np.mean(policy_result['epoch_p_losses'])
        avg_v_loss = np.mean(policy_result['epoch_v_losses'])
        avg_e_loss = np.mean(policy_result['epoch_e_losses'])
        avg_kl_loss = np.mean(policy_result['epoch_kl_losses'])
        avg_ent = np.mean(policy_result['epoch_ent'])
        avg_kl_div = np.mean(policy_result['epoch_kl_divs'])

        wm_info = self.last_wm_batch_info
        perf_metrics = {
            "policy_sample_time": policy_sample_time,
            "policy_prep_time": policy_prep_time,
            "policy_train_time": policy_result['policy_train_time'],
            "wm_sample_time": wm_info["sample_time"],
            "wm_prep_time": wm_info["prep_time"],
            # Time this epoch actually stalled waiting for a prefetched batch.
            "policy_batch_wait_time": policy_wait_time,
            "wm_batch_wait_time": wm_wait_time,
        }

        if ctrl_result['should_train']:
            perf_metrics["ctrl_train_time"] = ctrl_result['ctrl_train_time']
        if reward_result['should_train']:
            perf_metrics["reward_train_time"] = reward_result['reward_train_time']

        wm_metrics = {
            "trained_reward": reward_result['should_train'],
            "trained_ctrl": ctrl_result['should_train'],
            "ctrl_sample_unique_episodes": wm_info["unique_episodes"],
            "ctrl_sample_fresh_episode_ratio": wm_info["fresh_episode_ratio"],
        }

        if ctrl_result['should_train']:
            wm_metrics["ctrl_loss"] = float(np.mean(ctrl_result['epoch_ctrl_losses']))
            wm_metrics["ctrl_grad_norm"] = ctrl_result["ctrl_grad_norm"]
            wm_metrics["ctrl_clip_ratio"] = ctrl_result["ctrl_clip_ratio"]
            print(f"Trainer {self.rank}: Ctrl-World loss={wm_metrics['ctrl_loss']:.6f}")

        if reward_result['should_train']:
            print(f"Trainer {self.rank}: Reward Model 训练结果: {reward_result}")
            wm_metrics["reward_loss"] = np.mean(reward_result['epoch_reward_losses'])
            wm_metrics["reward_tp"] = reward_result['reward_tp']
            wm_metrics["reward_tn"] = reward_result['reward_tn']
            wm_metrics["reward_fp"] = reward_result['reward_fp']
            wm_metrics["reward_fn"] = reward_result['reward_fn']
            wm_metrics["reward_grad_norm"] = reward_result["reward_grad_norm"]
            wm_metrics["reward_clip_ratio"] = reward_result["reward_clip_ratio"]
            wm_metrics["reward_pos_ratio_batch"] = float(
                wm_batch.get("reward_pos_ratio_batch", 0.0)
            )

        return avg_loss, avg_p_loss, avg_v_loss, avg_e_loss, avg_kl_loss, current_lrs, self.global_step, avg_ent, avg_kl_div, perf_metrics, wm_metrics

    def broadcast_weights(self, group_name, prefix=None):
        """Broadcast Actor / Reward / Ctrl-World (trainable params only for Ctrl)."""
        if group_name.startswith("broadcast_ctrl"):
            return self._broadcast_ctrl_trainable(group_name)

        if group_name.startswith("broadcast_actor"):
            engine = self.model
        elif group_name.startswith("broadcast_reward"):
            engine = self.reward_engine
        else:
            engine = self.model

        try:
            original_model = self.model
            self.model = engine
            return super().broadcast_weights(group_name, prefix=None)
        finally:
            self.model = original_model

    def _broadcast_ctrl_trainable(self, group_name):
        # Keep this paired with TrainerActorCom.setup_broadcast_group(), which
        # stores groups on rl.ds_com._group_mgr inside the actor process.
        from rl.ds_com import _group_mgr as group_mgr

        group_handle = group_mgr.get_group_by_name(group_name)
        assert group_handle is not None, f"广播组 '{group_name}' 未初始化"
        module = _unwrap_module(self.ctrl_world_engine)
        params = sorted(
            [(n, p) for n, p in module.named_parameters(recurse=True) if p.requires_grad],
            key=lambda x: x[0],
        )
        zero_ctx = getattr(deepspeed.zero, "GatheredParameters", None)
        if zero_ctx is None:
            import contextlib
            zero_ctx = contextlib.nullcontext
        params_to_gather = [p for _, p in params]
        device = next(module.parameters()).device
        with zero_ctx(params_to_gather, modifier_rank=0):
            for _, p in params:
                t = p.detach().to(device=device, dtype=p.dtype).contiguous()
                dist.broadcast(t, src=0, group=group_handle)
        print(f"Trainer {self.rank}: broadcast {len(params)} Ctrl-World trainable tensors")


# ================================================================
# 5. 主逻辑
# ================================================================
def build_openvla_cfg(args):
    """构建 OpenVLA 配置"""
    cfg = GenerateConfig(
        pretrained_checkpoint=args.pretrained_checkpoint,
        use_l1_regression=False,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=args.num_images_in_input,
        use_proprio=args.use_proprio,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        unnorm_key=args.benchmark + "_no_noops",
        checkpoint2=args.checkpoint2,
    )
    return cfg


def validate_split_owners(states, bindings, active_kinds=("policy", "ctrl", "reward")):
    """Prove one physical-card owner per model/rank and three independent worlds."""
    kinds = tuple(active_kinds)
    expected = {(kind, rank) for kind in kinds for rank in range(4)}
    if (len(states) != 4 * len(kinds) or len({state["pid"] for state in states}) != 4 * len(kinds)
            or {(state["training_role"], state["rank"]) for state in states} != expected):
        raise RuntimeError("Split training requires twelve distinct model/rank processes")
    ports = {}
    for state in states:
        kind, rank = state["training_role"], state["rank"]
        if (set(state["engines"]) != {kind} or state["world_size"] != 4
                or state["distributed_world_size"] != 4
                or state["gpu"]["physical_gpu"] != bindings[rank]["index"]):
            raise RuntimeError("Split trainer engine, process group or physical GPU identity differs")
        ports.setdefault(kind, set()).add(state["group_port"])
    if any(len(values) != 1 for values in ports.values()) or len(set.union(*ports.values())) != len(kinds):
        raise RuntimeError("Each model needs its own independent four-rank training rendezvous")


def merge_split_epoch_results(epoch_results, world_size, policy_step, ctrl_interval, reward_interval,
                              active_kinds=("policy", "ctrl", "reward")):
    """Join completed role groups at the original Policy publication boundary."""
    if len(epoch_results) != len(active_kinds) * world_size:
        raise RuntimeError("Incomplete split training epoch")
    policy = epoch_results[:world_size]
    auxiliaries = {kind: epoch_results[index * world_size:(index + 1) * world_size]
                   for index, kind in enumerate(active_kinds) if kind != "policy"}
    union_due = (("ctrl" in active_kinds and policy_step % ctrl_interval == 0) or
                 ("reward" in active_kinds and policy_step % reward_interval == 0))
    for rank, result in enumerate(policy):
        if len(result) != 11 or result[6] != policy_step:
            raise RuntimeError("Policy group did not complete its common next update")
        for kind, interval in (("ctrl", ctrl_interval), ("reward", reward_interval)):
            if kind not in auxiliaries:
                continue
            item = auxiliaries[kind][rank]
            due = policy_step % interval == 0
            if (item["kind"] != kind or item["rank"] != rank
                    or item["policy_step"] != policy_step
                    or item["model_step"] != policy_step // interval
                    or bool(item["metrics"]["trained_" + kind]) != due
                    or (item["wm_cohort"] is not None) != union_due):
                raise RuntimeError(f"{kind}: split trainer counters or schedule diverged")
        if {"ctrl", "reward"} <= auxiliaries.keys() and auxiliaries["ctrl"][rank]["wm_cohort"] != auxiliaries["reward"][rank]["wm_cohort"]:
            raise RuntimeError("World and Reward did not consume the same global Replay draw")
    merged = []
    for rank, result in enumerate(policy):
        item = list(result)
        item[5], item[9], item[10] = dict(result[5]), dict(result[9]), {}
        ctrl = auxiliaries["ctrl"][rank] if "ctrl" in auxiliaries else None
        reward = auxiliaries["reward"][rank] if "reward" in auxiliaries else None
        sample_owner = ctrl if ctrl is not None else reward
        for auxiliary in [items[rank] for items in auxiliaries.values()]:
            item[5].update(auxiliary["learning_rates"])
            item[10].update({key: value for key, value in auxiliary["metrics"].items()
                             if not key.startswith(("ctrl_sample_", "trained_"))})
            item[9].update({key: value for key, value in auxiliary["performance"].items()
                            if key not in ("wm_sample_time", "wm_prep_time", "wm_batch_wait_time")})
        item[10].update(trained_ctrl=ctrl["metrics"]["trained_ctrl"] if ctrl else False,
                        trained_reward=reward["metrics"]["trained_reward"] if reward else False)
        for key in ("ctrl_sample_unique_episodes", "ctrl_sample_fresh_episode_ratio"):
            item[10][key] = sample_owner["metrics"][key]
        # The preparation owners now run concurrently; retain role timings too.
        for key in ("wm_sample_time", "wm_prep_time", "wm_batch_wait_time"):
            item[9][key] = max(role[rank]["performance"][key] for role in auxiliaries.values())
        item[9]["ctrl_batch_prep_time"] = ctrl["performance"]["wm_prep_time"] if ctrl else 0.0
        if reward is not None:
            item[9]["reward_batch_prep_time"] = reward["performance"]["wm_prep_time"]
        merged.append(tuple(item))
    return merged


def main(args):
    """Run the original training recipe with optional evaluation-based stop."""
    world_config = load_backend_config(args)
    fresh_config = FreshConfig.from_world_config(world_config)
    from cost_scheduler_runtime import load_settings, CostAdmissionBroker, source_fingerprint
    cost_settings=load_settings()
    cost_broker=None
    if cost_settings.get('enabled'):
        expected={'dense_reward':True,'signal_telemetry':True,'refresh_gae':True,
            'replay_age_limits':True,'max_policy_age':16,'max_world_age':3,'max_sample_uses':4,
            'seed_balancing':True,'early_seed_context':True,'real_policy_replay':True,'real_fraction':.25}
        if any(getattr(fresh_config,k)!=v for k,v in expected.items()):
            raise ValueError('Cost scheduling preserves the exact full FRESH learning rules')
    fresh_replay = any((fresh_config.refresh_gae, fresh_config.replay_age_limits, fresh_config.real_policy_replay))
    if fresh_replay and (args.gpu_layout != SPLIT_LAYOUT or world_config.backend != 'weaver'
            or os.environ.get('ACCERL_CPU_REPLAY_PREFETCH') != '1'):
        raise ValueError('FRESH segment Replay currently requires the audited split WEAVER update boundary')
    world_config.validate_admission()
    validate_online_schema(world_config, ACTION_DIM)
    service_plan = build_service_plan(world_config, args.num_trainer_gpus)
    online_weaver = isolated_online_weaver(world_config)
    validate_online_runtime_arguments(world_config, args)
    pipeline_config = PipelineBudgetConfig.from_json(getattr(args, 'pipeline_config', None))
    if pipeline_config.bytes_per_frame != 2 * 3 * 224 * 224:
        raise ValueError("Frame budget must use the actual two-camera uint8 geometry")
    geometry = context_geometry(world_config)
    if world_config.backend == 'ctrl':
        geometry = dict(required_context_frames=args.num_history + 1,
                        training_window_frames=args.num_history + args.num_frames_pred)
    if args.num_step_cond != geometry['required_context_frames']:
        raise ValueError("Producer context must match the loaded World backend")
    replay_window_frames = (geometry['training_window_frames'] if world_config.world_trainable
                            else int(world_config.backend_options.get('external_reward_window_frames', 11)))
    if not world_config.world_trainable and replay_window_frames != 11:
        raise ValueError("Frozen WEAVER preserves the existing external Reward 11-frame eligibility")
    if args.gpu_layout != SPLIT_LAYOUT or os.environ.get('ACCERL_CPU_REPLAY_PREFETCH') != '1':
        raise ValueError("Episode pipeline requires split owners and CPU raw Replay prefetch")
    if args.imagine_horizon % NUM_ACTIONS_CHUNK:
        raise ValueError("Imagination horizon must contain whole Policy chunks")
    if pipeline_config.initial_policy_credit < args.train_batch_size * args.accumulation_steps * args.num_trainer_gpus:
        raise ValueError("Initial imagination budget cannot fill one global Policy batch")

    active_kinds = service_plan.active_kinds
    inference_kinds = service_plan.inference_kinds
    if world_config.backend == "weaver":
        service_plan.validate_runtime(args.num_trainer_gpus)
        if args.gpu_layout != SPLIT_LAYOUT or serial_enabled(args.gpu_layout, environment=os.environ):
            raise ValueError("WEAVER requires the independent concurrent owner layout")
        if args.ctrl_train_interval != 5 or ("reward" in active_kinds and args.reward_train_interval != 5):
            raise ValueError("WEAVER service plan requires the 5:1 schedule")
        if os.environ.get("ACCERL_CPU_REPLAY_PREFETCH") != "1":
            raise ValueError("WEAVER v1 requires audited CPU raw prefetch for rank/cohort consumption receipts")
    run_policy = TrainingStopPolicy.from_environment(os.environ)
    evaluation_milestones = EvaluationMilestones()
    run_started_ns = int(os.environ.get("ACCERL_RUN_STARTED_NS", now_ns()))
    eval_target = os.environ.get("ACCERL_EVAL_TARGET")
    threshold_tracker = (ThresholdTracker(
        target=float(eval_target), run_started_ns=run_started_ns,
        required_consecutive=int(os.environ.get("ACCERL_EVAL_CONSECUTIVE", "2")),
        min_samples=int(os.environ.get("ACCERL_EVAL_MIN_EPISODES", str(args.moving_avg_window))))
        if eval_target is not None else None)
    target_reached = False
    benchmark = args.benchmark
    torch_dtype = torch.bfloat16 if args.use_bf16 else torch.float32
    agent_config_path = Path.cwd() / args.agent_config_path
    trainer_config_path = Path.cwd() / args.trainer_config_path

    if not os.path.exists(args.pretrained_checkpoint):
        print(f"错误: OpenVLA checkpoint 路径 '{args.pretrained_checkpoint}' 不存在。")
        return
    if world_config.backend == "ctrl":
        if not os.path.isfile(args.ctrl_world_ckpt):
            raise FileNotFoundError(f"Ctrl-World checkpoint not found: {args.ctrl_world_ckpt}")
        if args.vae_decoder_checkpoint and not os.path.isfile(args.vae_decoder_checkpoint):
            raise FileNotFoundError(
                f"VAE decoder checkpoint not found: {args.vae_decoder_checkpoint}"
            )
        if args.condition_stat_path is None:
            args.condition_stat_path = str(
                Path(args.ctrl_world_ckpt).resolve().parent / "condition_stat.json"
            )
        if not os.path.isfile(args.condition_stat_path):
            raise FileNotFoundError(
                f"Ctrl-World condition statistics not found: {args.condition_stat_path}"
            )
    if not 0.0 < args.ctrl_eval_episode_ratio < 1.0:
        raise ValueError("ctrl_eval_episode_ratio must be in (0, 1)")
    if args.ctrl_eval_batch_size < 1 or args.ctrl_eval_micro_batch_size < 1:
        raise ValueError("Ctrl eval batch sizes must be positive")
    if args.ctrl_eval_windows_per_episode < 1:
        raise ValueError("ctrl_eval_windows_per_episode must be positive")
    if args.ctrl_train_interval < 1 or ("reward" in active_kinds and args.reward_train_interval < 1):
        raise ValueError("World-model train intervals must be positive")
    total_ctrl_steps = math.ceil(args.train_iters / args.ctrl_train_interval)
    if "ctrl" in active_kinds and args.ctrl_warmup_steps >= total_ctrl_steps:
        raise ValueError(
            f"ctrl_warmup_steps={args.ctrl_warmup_steps} must be smaller than "
            f"the {total_ctrl_steps} planned Ctrl optimizer updates"
        )
    # Reward LR schedule uses train_iters (Diamond-aligned).
    if "reward" in active_kinds and args.reward_warmup_steps >= args.train_iters:
        raise ValueError(
            f"reward_warmup_steps={args.reward_warmup_steps} must be smaller than "
            f"train_iters={args.train_iters}"
        )

    os.environ["RAY_DEDUP_LOGS"] = "0"
    split_training = args.gpu_layout == SPLIT_LAYOUT
    split_train_serial = serial_enabled(args.gpu_layout, environment=os.environ)
    four_layout = args.gpu_layout in (FOUR_LAYOUT, SPLIT_LAYOUT)
    fixed_layout = args.gpu_layout in (DUAL_LAYOUT, FOUR_LAYOUT, SPLIT_LAYOUT)
    bindings = build_bindings(parse_gpu_ids(args.cuda_visible_devices)) if fixed_layout else None
    topology = ray_init_options(bindings, args.gpu_layout, active_kinds=active_kinds,
                               inference_kinds=inference_kinds) if fixed_layout else {"num_gpus": len(args.cuda_visible_devices.split(","))}
    object_store_size_gb = 328
    object_store_memory_bytes = int(object_store_size_gb * 1024 * 1024 * 1024)
    print("[Ray memory budget] " + json.dumps(validate_object_store_budget(object_store_memory_bytes)), flush=True)
    print(f"正在初始化 Ray，并为对象存储分配 {object_store_size_gb} GB 内存...")
    ray.init(
        address="local",
        num_cpus=32,
        **topology,
        include_dashboard=False,
        ignore_reinit_error=True,
        _temp_dir=os.environ.get("ACCERL_RAY_TMP", "/dev/shm/arlcap"),
        object_store_memory=object_store_memory_bytes,
        object_spilling_directory=os.environ.get("ACCERL_RAY_SPILL", "/mnt/data/lcx1/dong/ray_profile_spill"),
    )


    print(f"Ray 初始化完成，对象存储分配 {object_store_size_gb} GB 内存。")
    log_dir = f"runs/Libero/{args.benchmark}/{int(time.time())}_{args.exp_name}"
    writer = CaptureSummaryWriter(log_dir, tensorboard=(
        os.environ.get("ACCERL_CAPTURE_ONLY") != "1" or os.environ.get("ACCERL_TENSORBOARD") == "1"))
    if writer.tensorboard is not None:
        writer.telemetry.event("tensorboard_started", details={
            "log_dir": str(Path(log_dir).resolve()), "run_started_ns": run_started_ns,
            "metric_tag": "Eval/_Global/Average_Return", "window_size": args.moving_avg_window,
            "target": float(eval_target) if eval_target is not None else None,
            "mps_enabled": os.environ.get("ACCERL_USE_MPS") == "1"})
        writer.add_scalar("System/MPS_Enabled", int(os.environ.get("ACCERL_USE_MPS") == "1"), 0)
        writer.flush()
    stats_actor = StatsActor.remote(window_size=args.moving_avg_window)
    print(f"TensorBoard 日志将保存在: {log_dir}")

    cfg = build_openvla_cfg(args)

    print("--- 步骤 1: 创建 Actors ---")
    replay_count = 1 if fixed_layout else args.num_trainer_gpus
    # One shared Replay per model; draw each global batch once, then shard by rank.
    # Actor Replay Buffer (imagination rollout 数据)
    replay_buffers = [ReplayBufferActor.remote(capacity=args.replay_capacity,
        fresh_config=fresh_config.to_dict()) for _ in range(replay_count)]
    # World Model Replay Buffer (真实轨迹数据)
    wm_replay_buffers = [
        WMReplayBufferActor.remote(
            capacity=args.wm_replay_capacity,
            ctrl_eval_capacity=args.ctrl_eval_batch_size,
            ctrl_eval_episode_ratio=args.ctrl_eval_episode_ratio,
            ctrl_eval_windows_per_episode=args.ctrl_eval_windows_per_episode,
            pipeline_config=pipeline_config.to_dict(), world_config=world_config.to_dict(),
            context_frames=geometry['required_context_frames'], window_frames=replay_window_frames,
            policy_chunks_per_seed=args.imagine_horizon // NUM_ACTIONS_CHUNK,
            minimum_train_windows=max(args.ctrl_batch_size * args.ctrl_accumulation_steps if "ctrl" in active_kinds else 0,
                args.reward_batch_size * args.reward_accumulation_steps if "reward" in active_kinds else 0) * args.num_trainer_gpus,
        )
        for _ in range(replay_count)
    ]
    batch_coordinator = (ray.remote(GlobalBatchCoordinator).remote(
        replay_buffers[0], wm_replay_buffers[0], args.num_trainer_gpus,
        wm_consumers=service_plan.wm_consumers if split_training else ("combined",),
        pipeline_enabled=True, world_streaming=online_weaver,
        world_micro_batch=args.ctrl_batch_size) if fixed_layout else None)

    ctrl_world_args = {
        "world_backend_config": world_config.to_dict(),
        "svd_model_path": args.svd_model_path,
        "clip_model_path": args.clip_model_path,
        "ctrl_world_ckpt": args.ctrl_world_ckpt,
        "vae_decoder_checkpoint": args.vae_decoder_checkpoint,
        "condition_stat_path": args.condition_stat_path,
        "num_cams": args.num_cams,
        "num_history": args.num_history,
        "num_frames_pred": args.num_frames_pred,
        "target_height": args.target_height,
        "target_width": args.target_width,
    }
    def create_trainer(i, training_role):
        placement_role = training_role + "_trainer" if split_training else "trainer"
        trainer_class = WeaverOnlineTrainerActor if online_weaver and training_role == "ctrl" else TrainerActor
        return trainer_class.options(**(actor_options(bindings, placement_role, i, layout=args.gpu_layout, active_kinds=active_kinds,
                                                   inference_kinds=inference_kinds) if fixed_layout else {})).remote(
            rank=i, world_size=args.num_trainer_gpus, replay_buffer=replay_buffers[i % replay_count], cfg=cfg,
            train_batch_size=args.train_batch_size, accumulation_steps=args.accumulation_steps,
            use_bf16=args.use_bf16, torch_dtype=torch_dtype, policy_lr=args.policy_lr, value_lr=args.value_lr,
            gamma=args.gamma, lambda_=args.lambda_, clip_eps=args.clip_eps, vf_coef=args.vf_coef,
            ent_coef=args.ent_coef, kl_coef=args.kl_coef, reward_scale=args.reward_scale,
            value_warmup_steps=args.value_warmup_steps, policy_warmup_steps=args.policy_warmup_steps,
            policy_train_start_step=args.policy_train_start_step, train_iters=args.train_iters,
            clip_mode=args.clip_mode,
            # World Model 相关参数
            wm_replay_buffer=wm_replay_buffers[i % replay_count],
            agent_config_path=agent_config_path,
            trainer_config_path=trainer_config_path,
            reward_batch_size=args.reward_batch_size,
            reward_accumulation_steps=args.reward_accumulation_steps,
            reward_lr=args.reward_lr,
            reward_warmup_steps=args.reward_warmup_steps,
            reward_train_interval=args.reward_train_interval,
            reward_checkpoint=args.reward_checkpoint,
            ctrl_world_args=ctrl_world_args,
            ctrl_batch_size=args.ctrl_batch_size,
            ctrl_accumulation_steps=args.ctrl_accumulation_steps,
            ctrl_lr=args.ctrl_lr,
            ctrl_warmup_steps=args.ctrl_warmup_steps,
            ctrl_train_interval=args.ctrl_train_interval,
            batch_coordinator=batch_coordinator, training_role=training_role, split_train_serial=split_train_serial,
        )

    if split_training:
        training_groups = {kind: [create_trainer(i, kind) for i in range(args.num_trainer_gpus)]
                           for kind in active_kinds}
        trainer_group = training_groups["policy"]
        all_trainers = [actor for group in training_groups.values() for actor in group]
    else:
        trainer_group = [create_trainer(i, "combined") for i in range(args.num_trainer_gpus)]
        training_groups = {kind: trainer_group for kind in ("policy", "ctrl", "reward")}
        all_trainers = trainer_group

    if args.gpu_layout == DUAL_LAYOUT:
        shared_inference = PolicyRewardInferenceActor.options(**actor_options(bindings, "policy_reward")).remote(
            policy_cls=InferenceActor, reward_cls=RewardInferenceActor,
            policy_kwargs=dict(actor_id=0, cfg=cfg, stats_actor=stats_actor, torch_dtype=torch_dtype,
                               inference_batch=args.inference_batch, inference_timeout_ms=args.inference_timeout_ms),
            reward_kwargs=dict(actor_id=0, stats_actor=stats_actor, agent_config_path=agent_config_path,
                               inference_batch=args.inference_batch, inference_timeout_ms=args.inference_timeout_ms,
                               reward_checkpoint=args.reward_checkpoint), placement_guard=verify_bound_gpu,
        )
        inference_pool = [shared_inference]
        reward_inference_pool = [RewardInferenceHandle(shared_inference)]
    else:
        inference_pool = [
            InferenceActor.options(**(actor_options(bindings, "policy", i, layout=args.gpu_layout, active_kinds=active_kinds, inference_kinds=inference_kinds) if four_layout else {"num_gpus": float(os.environ.get("ACCERL_POLICY_GPU","1"))})).remote(
                actor_id=i, cfg=cfg, stats_actor=stats_actor, torch_dtype=torch_dtype,
                inference_batch=args.inference_batch, inference_timeout_ms=args.inference_timeout_ms
            ) for i in range(args.num_inference_actors)
        ]
        reward_inference_pool = [
            RewardInferenceActor.options(**(actor_options(bindings, "reward", i, layout=args.gpu_layout, active_kinds=active_kinds, inference_kinds=inference_kinds) if four_layout else {"num_gpus": float(os.environ.get("ACCERL_REWARD_GPU","1"))})).remote(
                actor_id=i, stats_actor=stats_actor,
                agent_config_path=agent_config_path,
                inference_batch=args.inference_batch,
                inference_timeout_ms=args.inference_timeout_ms,
                reward_checkpoint=args.reward_checkpoint,
            ) for i in range(args.num_reward_inference_actors)
        ]
    # ★ 替换 denoiser_inference_pool 为 ctrl_world_inference_pool
    ctrl_world_inference_pool = [
        (WeaverOnlineWorldInferenceActor if online_weaver else
         WeaverWorldInferenceActor if world_config.backend == "weaver" else CtrlWorldInferenceActor).options(**(actor_options(bindings, "ctrl", i, layout=args.gpu_layout, active_kinds=active_kinds, inference_kinds=inference_kinds) if fixed_layout else
                                          {"num_gpus": float(os.environ.get("ACCERL_CTRL_GPU", "1"))})).remote(
            actor_id=i,
            stats_actor=stats_actor,
            args=args,
            inference_batch=args.inference_batch,
            inference_timeout_ms=args.inference_timeout_ms,
        ) for i in range(args.num_ctrl_inference_actors)
    ]
    if cost_settings.get('enabled'):
        cost_settings=dict(cost_settings,run_id=os.environ['ACCERL_RUN_ID'],telemetry=True)
        cost_source_sha=source_fingerprint(Path(__file__).parent)
        cost_broker=ray.remote(num_cpus=0,max_concurrency=1000,max_restarts=0,max_task_retries=0)(CostAdmissionBroker).remote(
            cost_settings,os.environ['ACCERL_RUN_ID'],cost_source_sha)
        ray.get([actor.configure_cost_scheduler.remote(cost_broker,cost_settings,cost_source_sha) for actor in all_trainers])
        policy_dispatch_settings={k:cost_settings[k] for k in
            ('enabled','refresh_micro','action_micro','class_weights','max_pending')}
        ray.get([actor.configure_cost_scheduler.remote(cost_broker,policy_dispatch_settings)
            for actor in inference_pool+replay_buffers])
        ray.get([actor.configure_cost_scheduler.remote(cost_broker,cost_settings)
            for actor in ctrl_world_inference_pool])
    rollout_workers, eval_workers = [], []
    if not args.validate_only:
        rollout_workers = [
            RolloutWorkerActor.remote(
                inference_pool[i % args.num_inference_actors],
                replay_buffers[i % replay_count],
                i,
                stats_actor,
                cfg,
                benchmark,
                args.num_step_cond,
                args.imagine_horizon,
                torch_dtype,
                reward_inference_pool[i % len(reward_inference_pool)] if reward_inference_pool else None,
                ctrl_world_inference_pool[i % args.num_ctrl_inference_actors],  # ★ 替换
                args.reward_scale,
                args.gamma,
                args.lambda_,
                wm_replay_buffers[i % replay_count],
                args.real_traj_collect_interval, world_config=world_config.to_dict(),
                worker_mode="real" if i < pipeline_config.real_workers else "imagine",
                pipeline_config=pipeline_config.to_dict(),
            ) for i in range(pipeline_config.real_workers + pipeline_config.imagination_workers)
        ]
        eval_workers = [
            EvaluationWorkerActor.remote(
                inference_pool[i % args.num_inference_actors], i, stats_actor, cfg, benchmark, torch_dtype
            ) for i in range(args.num_eval_workers)
        ]
    print(f"已创建 {len(rollout_workers)} 个 Rollout workers 和 {len(eval_workers)} 个 Evaluation workers。")

    print("\n--- 步骤 2: 建立独立的四 rank 训练组 ---")
    used_train_ports = set()
    groups_to_initialize = training_groups.items() if split_training else (("combined", trainer_group),)
    for kind, group in groups_to_initialize:
        train_group_port = find_free_port()
        while train_group_port in used_train_ports:
            train_group_port = find_free_port()
        used_train_ports.add(train_group_port)
        trainer_master_addr = ray.get(group[0].get_node_ip.remote())
        ray.get([actor.setup_deepspeed_group.remote(trainer_master_addr, train_group_port) for actor in group])
        engine_label = "native Torch DDP flow matching" if online_weaver and kind == "ctrl" else "DeepSpeed/FSDP"
        print(f"{engine_label} group ready: role={kind}, world={len(group)}, port={train_group_port}", flush=True)

    if not four_layout:
        print(f"\n--- 步骤 3: 建立共享广播组 ---")
        broadcast_master_addr = ray.get(trainer_group[0].get_node_ip.remote())

        # 3.1 Actor 广播组
        broadcast_actor_port = find_free_port()
        broadcast_actor_participants = [trainer_group[0]] + inference_pool
        broadcast_actor_world_size = len(broadcast_actor_participants)
        broadcast_setup_tasks = [
            actor.setup_broadcast_group.remote(
                master_addr=broadcast_master_addr, master_port=broadcast_actor_port,
                group_name="broadcast_actor", group_world_size=broadcast_actor_world_size,
                my_rank_in_group=rank) for rank, actor in enumerate(broadcast_actor_participants)
        ]
        ray.get(broadcast_setup_tasks)
        print("Actor 广播组建立完成。")

        # 3.2 Reward Model 广播组 (Trainer -> RewardInferenceActor)
        broadcast_reward_port = find_free_port()
        broadcast_reward_participants = [trainer_group[0]] + reward_inference_pool
        broadcast_reward_world_size = len(broadcast_reward_participants)
        broadcast_setup_tasks = [
            actor.setup_broadcast_group.remote(
                master_addr=broadcast_master_addr, master_port=broadcast_reward_port,
                group_name="broadcast_reward", group_world_size=broadcast_reward_world_size,
                my_rank_in_group=rank) for rank, actor in enumerate(broadcast_reward_participants)
        ]
        ray.get(broadcast_setup_tasks)
        print("Reward Model 广播组建立完成。")

        # 3.3 Ctrl-World 广播组 (Trainer -> CtrlWorldInferenceActor)
        broadcast_ctrl_port = find_free_port()
        broadcast_ctrl_participants = [trainer_group[0]] + ctrl_world_inference_pool
        broadcast_ctrl_world_size = len(broadcast_ctrl_participants)
        broadcast_setup_tasks = [
            actor.setup_broadcast_group.remote(
                master_addr=broadcast_master_addr, master_port=broadcast_ctrl_port,
                group_name="broadcast_ctrl", group_world_size=broadcast_ctrl_world_size,
                my_rank_in_group=rank) for rank, actor in enumerate(broadcast_ctrl_participants)
        ]
        ray.get(broadcast_setup_tasks)
        print("Ctrl-World 广播组建立完成。")

    # 验证 Actor 模型键
    inf_keys = ray.get(inference_pool[0].get_model_keys.remote())
    trainer_keys = ray.get(trainer_group[0].get_model_keys.remote())
    for key in inf_keys:
        if key not in trainer_keys:
            print(f"警告: Actor 推理器中缺少训练器的键: {key}")
    for key in trainer_keys:
        if key not in inf_keys:
            print(f"警告: Actor 训练器中缺少推理器的键: {key}")

    forward_test_tasks = [inf.forward_test.remote() for inf in inference_pool]
    ray.get(forward_test_tasks)
    print("Actor 推理器前向测试完成 (广播前)。")

    experiment = Telemetry("driver", "fill_experiment")
    experiment.event("world_backend_service_plan", details={"service_plan": service_plan.to_dict(),
                     "world_backend_config": world_config.to_dict()})
    experiment.flush()

    frozen_world = None
    if online_weaver:
        experiment.event("world_backend_diagnostic_evaluation_deferred", details={
            "backend": "weaver", "world_trainable": True, "reason": "separate_bounded_native_evaluator_required",
            "policy_real_environment_evaluation_unchanged": True, "quality_measured": False})
        experiment.flush()
    if not world_config.world_trainable:
        receipts = ray.get([actor.get_frozen_world_state.remote() for actor in ctrl_world_inference_pool])
        required = ("checkpoint_id", "fixed_step", "bundle_sha256", "weight_dtype", "compute_dtype")
        if (len(receipts) != 4 or len({receipt["actor_id"] for receipt in receipts}) != 4
                or any(receipt["world_trainable"] or receipt["model_version"] != 0
                       or receipt["trainable_parameter_count"] != 0 for receipt in receipts)
                or any(len({receipt[key] for receipt in receipts}) != 1 for key in required)):
            raise RuntimeError("Frozen World replicas disagree on their admitted fixed checkpoint")
        frozen_world = {"backend": "weaver", "world_trainable": False,
                        "fixed_step": receipts[0]["fixed_step"], "bundle_sha256": receipts[0]["bundle_sha256"]}
        experiment.event("frozen_world_snapshot_verified", details={"frozen_world_identity": frozen_world,
                                                                 "receivers": receipts})
        experiment.flush()

    published_fresh_versions = {'policy':0, 'ctrl':0}
    if fresh_replay:
        ray.get([replay.configure_fresh.remote(inference_pool, wm_replay_buffers[index])
                 for index, replay in enumerate(replay_buffers)])

    def sync_model(kind, pool):
        publication_started_ns = time.time_ns()
        publication_timing = {"export_submitted_ns": publication_started_ns}
        if four_layout:
            owners = training_groups[kind]
            payloads = [trainer.export_rank_weights.remote(kind) for trainer in owners]
            if kind in ("policy", "ctrl"):
                # Only handles/metadata cross the Ray object store; the CPU driver
                # neither rebuilds CUDA tensors nor receives full weight snapshots.
                packets = ray.get(payloads, timeout=120)
                publication_timing["export_ready_ns"] = time.time_ns()
                inventories = [packet["inventory"] for packet in packets]
                if len({(item["version"], item["hash"]) for item in inventories}) != 1:
                    raise RuntimeError(f"Training weights diverged before {kind} IPC sync: {inventories}")
                receives = [actor.apply_rank_weights.remote(packet, kind)
                            for actor, packet in zip(pool, packets)]
                publication_timing["apply_submitted_ns"] = time.time_ns()
                publication_timing["rank_ACK_observed_ns"] = []
                received, failures = [], []
                for receive in receives:
                    try:
                        received.append(ray.get(receive, timeout=120))
                        publication_timing["rank_ACK_observed_ns"].append(time.time_ns())
                    except BaseException as error:
                        received.append(None)
                        failures.append(repr(error))
                if failures or received != inventories:
                    # Do not release producer references or start another update.
                    raise RuntimeError(f"Incomplete {kind} IPC publication: {failures}; ACKs={received}")
                released = ray.get([trainer.finish_ipc_weight_publish.remote(kind, packet["token"], ack)
                                    for trainer, packet, ack in zip(owners, packets, received)], timeout=120)
                if released != inventories:
                    raise RuntimeError(f"IPC release acknowledgement differs for {kind}")
                publication_timing["release_ready_ns"] = time.time_ns()
            else:
                inventories = ray.get([trainer.describe_rank_weights.remote(payload)
                                      for trainer, payload in zip(owners, payloads)])
                publication_timing["export_ready_ns"] = time.time_ns()
                if len({(item["version"], item["hash"]) for item in inventories}) != 1:
                    raise RuntimeError(f"Training weights diverged before {kind} sync: {inventories}")
                received = ray.get([actor.apply_rank_weights.remote(payload, kind)
                                    for actor, payload in zip(pool, payloads)])
                if received != inventories:
                    raise RuntimeError(f"Inference weights differ after {kind} sync")
                publication_timing["all_ACK_observed_ns"] = time.time_ns()
            experiment.event("weight_publication_committed", model_version=inventories[0]["version"],
                details={"kind": kind, "transport": "cuda_ipc" if kind in ("policy", "ctrl") else "cpu_ray",
                         "started_ns": publication_started_ns, "ack_completed_ns": time.time_ns(),
                         "inventories": inventories, "receiver_ranks": list(range(len(pool))),
                         "driver_timing": publication_timing,
                         "meaning": "all_rank_receivers_ACKed; per_receiver_model_version_event_is_its_visibility_point"})
            experiment.flush()
            if fresh_replay:
                published_fresh_versions[kind] = int(inventories[0]['version'])
                ray.get([producer.set_fresh_replay_versions.remote(published_fresh_versions['policy'],
                    published_fresh_versions['ctrl']) for producer in wm_replay_buffers])
                ray.get([replay.set_fresh_versions.remote(published_fresh_versions['policy'],
                    published_fresh_versions['ctrl']) for replay in replay_buffers])
        else:
            group = {"policy": "broadcast_actor", "reward": "broadcast_reward", "ctrl": "broadcast_ctrl"}[kind]
            send = trainer_group[0].broadcast_weights.remote(group)
            ray.get([send] + [actor.receive_and_update_weights.remote(group) for actor in pool])

    for kind, pool in (("policy", inference_pool), ("reward", reward_inference_pool),
                       ("ctrl", ctrl_world_inference_pool)):
        if kind in active_kinds:
            sync_model(kind, pool)
    states = ray.get([trainer.get_training_state.remote() for trainer in all_trainers])
    if split_training:
        validate_split_owners(states, bindings, active_kinds)
    print("[Training topology] " + json.dumps(states, ensure_ascii=False), flush=True)
    if args.gpu_layout == DUAL_LAYOUT:
        print("[Independent inference queues] " + json.dumps(ray.get(shared_inference.get_inference_owners.remote())), flush=True)
    if args.validate_only:
        print("VALIDATION_OK: real models loaded; GPU bindings and active initial publications completed", flush=True)
        experiment.close()
        writer.close()
        return

    forward_test_tasks = [inf.forward_test.remote() for inf in inference_pool]
    ray.get(forward_test_tasks)
    print("Actor 推理器前向测试完成 (广播后)。")

    snapshot_dir = os.environ.get("ACCERL_FINAL_CHECKPOINT") if fixed_layout else None
    def publish_snapshot(reason):
        if snapshot_dir:
            metadata = {"args": vars(args), "run_id": os.environ.get("ACCERL_RUN_ID"),
                        "stop_policy": run_policy.metadata(), "service_plan": service_plan.to_dict(),
                        "world_backend_config": world_config.to_dict(),
                        "frozen_weights": "Restore original checkpoints in args before applying snapshot"}
            if frozen_world is not None:
                metadata["frozen_world"] = frozen_world
            if pipeline_config.initial_state_enabled:
                metadata['initial_state_queue'] = ray.get(wm_replay_buffers[0].initial_state_checkpoint.remote())
            if split_training and online_weaver:
                result = save_online_split_snapshot(training_groups, world_config, snapshot_dir, reason, metadata,
                    ray_get=ray.get, save_split_weights=save_split_weights)
            elif split_training:
                kinds = service_plan.checkpoint_kinds
                components = ray.get([training_groups[kind][0].export_snapshot_component.remote(kind)
                                      for kind in kinds])
                result = save_split_weights(dict(zip(kinds, components)), snapshot_dir,
                                            reason=reason, metadata=metadata, active_kinds=active_kinds)
            else:
                result = ray.get(trainer_group[0].save_parallel_snapshot.remote(
                    snapshot_dir, reason, metadata))
            print("[Active-model checkpoint] " + json.dumps(result, ensure_ascii=False), flush=True)
    publish_snapshot("initial")
    if snapshot_dir and pipeline_config.initial_state_enabled:
        from checkpoint_evaluation import preserve_initial_checkpoint
        preserve_initial_checkpoint(snapshot_dir)

    print("\n--- 步骤 4: 启动 Rollout Workers 进行数据收集 ---")
    experiment.event("training_stop_policy", details={**run_policy.metadata(),
                     "train_iters_lr_horizon": args.train_iters})
    if split_training:
        experiment.event("split_training_topology_validated", details={
            "layout": SPLIT_LAYOUT, "states": states,
            "trainer_processes": len(all_trainers), "training_groups": len(training_groups),
            "world_size_per_group": args.num_trainer_gpus, "service_plan": service_plan.to_dict(),
            "world_backend_config": world_config.to_dict(),
            "split_train_serial": split_train_serial,
            "group_order": ["policy", "ctrl", "reward"] if split_train_serial else "concurrent"})
        experiment.flush()
    experiment.event("data_pipeline_config", details={"budget": pipeline_config.to_dict(),
        "geometry": geometry, "replay_window_frames": replay_window_frames,
        "real_workers": pipeline_config.real_workers,
        "imagination_workers": pipeline_config.imagination_workers,
        "canonical_frames": "uint8_episode_once", "wm_projection": "same_draw_per_owner"})
    fill_started = time.time()
    experiment.event("rollout_started", ts_ns=int(fill_started * 1e9),
                     details={"layout": args.gpu_layout, "physical_gpus": args.cuda_visible_devices,
                              "world_replicas": args.num_ctrl_inference_actors})
    experiment.flush()
    worker_run_refs = []
    for w in rollout_workers:
        worker_run_refs.append(w.run.remote())
    for w in eval_workers:
        worker_run_refs.append(w.run.remote())

    print("\n--- 步骤 5: 等待远程经验池填充初始数据 ---")
    shard_factor = args.num_trainer_gpus if fixed_layout else 1
    min_buffer_size_for_start = args.train_batch_size * args.accumulation_steps * shard_factor
    assert min_buffer_size_for_start < args.replay_capacity, "初始填充量必须小于回放池总容量"

    wm_min_buffer_size = max(
        args.reward_batch_size * args.reward_accumulation_steps * shard_factor if "reward" in active_kinds else 0,
        args.ctrl_batch_size * args.ctrl_accumulation_steps * shard_factor if "ctrl" in active_kinds else 0,
    )
    assert wm_min_buffer_size < args.wm_replay_capacity, "WM初始填充量必须小于WM回放池总容量"

    last_fill_snapshot = -float("inf")
    measured_twenty_minutes = False
    while True:
        if stop_requested():
            fill_completion_reason = run_policy.stop_reason(global_step=0, train_iters=args.train_iters,
                target_reached=False, requested_stop=True)
            ray.get([buffer.stop_pipeline.remote() for buffer in wm_replay_buffers])
            publish_snapshot("final_before_first_update")
            experiment.event("experiment_completed", details={"global_step": 0, "completed_epochs": 0,
                "run_started_ns": run_started_ns, "threshold_state": threshold_tracker.as_dict() if threshold_tracker else None,
                "stop_reason": fill_completion_reason, "target_reached": False,
                "service_plan": service_plan.to_dict(), "world_backend_config": world_config.to_dict(),
                "stop_on_target": run_policy.stop_on_target, "until_deadline": run_policy.until_deadline,
                "deadline_unix_seconds": run_policy.deadline_unix_seconds, "stop_policy": run_policy.metadata(),
                "states": ray.get([trainer.get_training_state.remote() for trainer in all_trainers])})
            experiment.close()
            print("Scheduled stop during initial replay fill; no optimizer update completed", flush=True)
            writer.close()
            ray.shutdown()
            return
        if time.time()-fill_started > float(os.environ.get("ACCERL_FILL_TIMEOUT_SEC","7200")):
            raise TimeoutError("Initial replay fill exceeded configured grace period")
        ready, _ = ray.wait(worker_run_refs, timeout=0, num_returns=1)
        if ready:
            ray.get(ready[0])
            raise RuntimeError("A rollout/eval worker exited during initial replay fill")
        actor_sizes = ray.get([rb.size.remote() for rb in replay_buffers])
        wm_sizes = ray.get([rb.size.remote() for rb in wm_replay_buffers])
        actor_ready = all(size >= min_buffer_size_for_start for size in actor_sizes)
        wm_ready = all(size >= wm_min_buffer_size for size in wm_sizes)
        fill_elapsed = time.time() - fill_started
        at_twenty = fill_elapsed >= 1200 and not measured_twenty_minutes
        if fill_elapsed - last_fill_snapshot >= 30 or at_twenty or (actor_ready and wm_ready):
            experiment.event("fill_snapshot", details={
                "elapsed_seconds": fill_elapsed, "policy_sizes": actor_sizes, "wm_sizes": wm_sizes,
                "policy_target": min_buffer_size_for_start, "wm_target": wm_min_buffer_size,
                "at_twenty_minutes": at_twenty, "ready": actor_ready and wm_ready})
            experiment.flush()
            last_fill_snapshot = fill_elapsed
            measured_twenty_minutes = measured_twenty_minutes or at_twenty
            writer.add_scalar("Fill/Policy_Resident", min(actor_sizes), int(fill_elapsed))
            writer.add_scalar("Fill/WM_Resident", min(wm_sizes), int(fill_elapsed))
            writer.flush()

        if actor_ready and wm_ready:
            experiment.event("replay_ready", details={"elapsed_seconds": fill_elapsed,
                             "policy_sizes": actor_sizes, "wm_sizes": wm_sizes})
            experiment.flush()
            break

        print(
            f"等待经验池填充... Actor (目标: {min_buffer_size_for_start}): {actor_sizes}, "
            f"WM (目标: {wm_min_buffer_size}): {wm_sizes}"
        )
        time.sleep(5)
    print("所有远程经验池已准备好，训练器将按需获取数据。")
    if four_layout:
        limits = ((inference_pool, 2), (ctrl_world_inference_pool, 3), (reward_inference_pool, 2))
        ray.get([actor.set_training_batch_limit.remote(cap) for pool, cap in limits for actor in pool])
        experiment.event("training_phase_limits", details={kind: {"policy": 2, "ctrl": 3, "reward": 2}[kind]
                                                          for kind in active_kinds})

    print("\n--- 步骤 6: 开始主训练与同步循环 ---")
    start_time = time.time()
    last_log_time = time.time()
    last_log_global_step = 0
    global_step = 0
    completed_epochs = 0
    measure_stop_epochs = int(os.environ.get("ACCERL_MEASURE_STOP_AFTER_EPOCHS", "0"))
    nsight_windows = NsightCaptureWindows(experiment)
    while run_policy.should_continue(global_step=global_step, train_iters=args.train_iters,
            target_reached=target_reached, requested_stop=stop_requested(),
            completed_epochs=completed_epochs, measure_stop_epochs=measure_stop_epochs):
        ready, _ = ray.wait(worker_run_refs, timeout=0, num_returns=1)
        if ready:
            ray.get(ready[0])
            raise RuntimeError("A rollout/eval worker exited unexpectedly")
        t_train_start = time.time()
        next_policy_step = global_step + 1
        if cost_broker is not None:
            inventory=ray.get(replay_buffers[0].scheduler_inventory.remote())
            wm_ready=ray.get(wm_replay_buffers[0].size.remote())>=1024
            ray.get(cost_broker.update_cycle.remote(dict(inventory,policy_version=global_step,
                world_version=global_step//5,phase='needs_batch',
                context='joint' if next_policy_step%5==0 else 'ordinary',
                completed_context='joint' if global_step and global_step%5==0 else 'ordinary',
                world_training_units_remaining=float(next_policy_step%5==0),world_batch_ready=wm_ready,
                policy_action_observations_per_real_chunk=1.,
                policy_action_observations_per_imagined_chunk=1.25)))
        nsight_windows.before_epoch(next_policy_step)
        epoch_owners = all_trainers if split_training else trainer_group
        # Observe each rank separately so one failed rank cannot hide behind
        # another rank blocked inside a collective or a missing data shard.
        train_tasks = []
        rank_results = {}
        serial_sample_records = {}
        deadline = time.monotonic() + float(os.environ.get("ACCERL_TRAIN_EPOCH_TIMEOUT_SEC", "3600"))
        for group in epoch_groups(training_groups, epoch_owners, split_train_serial):
            tasks = [trainer.run_training_epoch.remote(next_policy_step) if split_training
                     else trainer.run_training_epoch.remote() for trainer in group]
            train_tasks.extend(tasks)
            pending = list(tasks)
            while pending:
                completed, pending = ray.wait(pending, num_returns=1, timeout=5)
                for result_ref in completed:
                    rank_results[result_ref] = ray.get(result_ref)
                exited, _ = ray.wait(worker_run_refs, timeout=0, num_returns=1)
                if exited:
                    ray.get(exited[0])
                    raise RuntimeError("A rollout/eval worker exited during training")
                if time.monotonic() > deadline:
                    raise TimeoutError("A synchronized training epoch exceeded its progress deadline")
        if split_training and (split_train_serial or os.environ.get("ACCERL_CPU_REPLAY_PREFETCH") == "1"):
            for model, group in training_groups.items():
                records = ray.get([trainer.serial_sample_identity.remote(next_policy_step) for trainer in group],
                                  timeout=max(1.0, deadline - time.monotonic()))
                union_due = (("ctrl" in active_kinds and next_policy_step % args.ctrl_train_interval == 0)
                             or ("reward" in active_kinds and next_policy_step % args.reward_train_interval == 0))
                local_size = (args.train_batch_size * args.accumulation_steps if model == "policy" else
                              max(args.ctrl_batch_size * args.ctrl_accumulation_steps if "ctrl" in active_kinds else 0,
                                  args.reward_batch_size * args.reward_accumulation_steps if "reward" in active_kinds else 0))
                identity_audit = validate_sample_group(records, model, next_policy_step,
                    args.num_trainer_gpus, local_size, required=model == "policy" or union_due)
                serial_sample_records[model] = records
                experiment.event("split_sample_identity_validated", details=identity_audit)
        epoch_results = [rank_results[ref] for ref in train_tasks]
        results = (merge_split_epoch_results(
            epoch_results, args.num_trainer_gpus, next_policy_step,
            args.ctrl_train_interval, args.reward_train_interval, active_kinds) if split_training else epoch_results)
        rank_steps = [int(result[6]) for result in results]
        rank_schedules = [(result[10]['trained_ctrl'], result[10]['trained_reward']) for result in results]
        if len(set(rank_steps)) != 1 or len(set(rank_schedules)) != 1:
            raise RuntimeError(f"Training ranks diverged: steps={rank_steps}, schedules={rank_schedules}")
        if split_training and (split_train_serial or os.environ.get("ACCERL_CPU_REPLAY_PREFETCH") == "1") and (("ctrl" in active_kinds and next_policy_step % args.ctrl_train_interval == 0)
                                   or ("reward" in active_kinds and next_policy_step % args.reward_train_interval == 0)):
            if {"ctrl", "reward"} <= set(active_kinds):
                validate_shared_wm_samples(serial_sample_records["ctrl"], serial_sample_records["reward"])
            experiment.event("split_wm_cohort_validated", details={
                "policy_step": next_policy_step, "expected_consumers": service_plan.expected_wm_consumers,
                "expected_wm_consumers": service_plan.expected_wm_consumers,
                "wm_consumers": list(service_plan.wm_consumers),
                "actual_sample_ids_verified": True, "rank_shards_disjoint": True,
                "world_reward_ordered_ids_identical": True if {"ctrl", "reward"} <= set(active_kinds) else None,
                "consumers": [{"model": kind, "rank": rank, **epoch_results[offset + rank]["wm_cohort"]}
                              for kind, offset in ((kind, active_kinds.index(kind) * args.num_trainer_gpus)
                                                   for kind in service_plan.wm_consumers)
                              for rank in range(args.num_trainer_gpus)]})
            experiment.flush()
        nsight_windows.after_epoch(next_policy_step)
        _, _, _, _, _, _, global_step, _, _, _, wm_metrics = results[0]
        wm_metrics = combine_wm_metrics([result[10] for result in results])
        train_time = time.time() - t_train_start

        # ============ 权重同步 ============
        t_sync_start = time.time()
        if cost_broker is not None:ray.get(cost_broker.publication.remote(True))

        sync_model("policy", inference_pool)
        if wm_metrics.get("trained_reward", False):
            sync_model("reward", reward_inference_pool)
        if wm_metrics.get("trained_ctrl", False):
            sync_model("ctrl", ctrl_world_inference_pool)

        sync_time = time.time() - t_sync_start
        if cost_broker is not None:ray.get(cost_broker.publication.remote(False,
            policy_version=global_step,world_version=global_step//5))
        completed_epochs += 1
        experiment.event("synchronized_epoch_completed", details={
            "completed_epochs": completed_epochs, "global_step": global_step, "rank_steps": rank_steps,
            "trained_ctrl": wm_metrics.get("trained_ctrl", False),
            "trained_reward": wm_metrics.get("trained_reward", False),
            "train_seconds": train_time, "sync_seconds": sync_time,
            "elapsed_from_rollout_seconds": time.time() - fill_started})
        experiment.flush()

        epoch_stats = None
        if threshold_tracker is not None:
            epoch_stats = ray.get(stats_actor.get_stats.remote())
            evaluation = epoch_stats["_global_eval_"]
            watermark = int(evaluation["total_episodes_processed"])
            fresh = evaluation_milestones.observe_watermark(watermark)
            evaluation_ns = int(evaluation["snapshot_completed_ns"])
            decision = threshold_tracker.update(
                completed_epochs, float(evaluation["avg_return"]), evaluation_ns,
                int(evaluation["window_episodes"]), int(global_step), fresh_count=fresh,
                evaluation_watermark=watermark,
                metadata={"metric_tag": "Eval/_Global/Average_Return",
                          "scope": "original_task0_rolling_window",
                          "snapshot_completed_ns": evaluation_ns,
                          "latest_episode_completed_ns": evaluation["latest_episode_completed_ns"],
                          "latest_episode_policy_versions": evaluation["latest_episode_policy_versions"],
                          "rolling_window_can_span_policy_versions": True})
            threshold_state = threshold_tracker.as_dict()
            experiment.event("evaluation_threshold", ts_ns=evaluation_ns, details={
                "state": threshold_state, "decision": decision, "snapshot_completed_ns": evaluation_ns,
                "latest_episode_completed_ns": evaluation["latest_episode_completed_ns"],
                "total_episodes": watermark, "fresh_episodes": fresh,
                "live_evaluation_watermark": evaluation_milestones.last_watermark,
                "rate": float(evaluation["avg_return"]), "policy_step": int(global_step)})
            walltime = evaluation_ns / 1e9
            writer.add_scalar("Eval/_Global/Average_Return", evaluation["avg_return"], global_step, walltime=walltime)
            writer.add_scalar("Eval/Target85/Window_Episodes", evaluation["window_episodes"], global_step, walltime=walltime)
            writer.add_scalar("Eval/Target85/New_Completed_Episodes", fresh, global_step, walltime=walltime)
            writer.add_scalar("Eval/Target85/Consecutive_Qualified", threshold_tracker.streak, global_step, walltime=walltime)
            writer.add_scalar("Eval/Target85/Threshold", threshold_tracker.target, global_step, walltime=walltime)
            milestones = evaluation_milestones.new_marks(threshold_tracker)
            for label, milestone in milestones:
                writer.add_scalar(f"TimeTo85/{label}_Elapsed_Seconds", milestone["elapsed_seconds"],
                                  global_step, walltime=milestone["completed_ns"] / 1e9, double_precision=True)
                writer.add_scalar(f"TimeTo85/{label}_Completion_Unix_Seconds", milestone["completed_ns"] / 1e9,
                                  global_step, walltime=milestone["completed_ns"] / 1e9, double_precision=True)
            target_reached = threshold_tracker.should_stop
            if evaluation_milestones.announce_target(target_reached):
                experiment.event("accuracy_target_reached", ts_ns=evaluation_ns, details={
                    "state": threshold_state, "global_step": int(global_step),
                    "stop_policy": run_policy.metadata(),
                    "reason": "consecutive_eval_85", "tensorboard_dir": str(Path(log_dir).resolve())})
                print("[85% target completed] " + json.dumps(threshold_state, ensure_ascii=False), flush=True)
            experiment.flush()
            writer.flush()

        if (world_config.backend == "weaver" and world_config.world_trainable and not online_weaver
                and not run_policy.target_stop_active(target_reached)
                and args.ctrl_eval_interval > 0 and global_step % args.ctrl_eval_interval == 0):
            eval_size = ray.get(wm_replay_buffers[0].eval_size.remote())
            if eval_size >= args.ctrl_eval_batch_size:
                windows = ray.get(wm_replay_buffers[0].sample_eval_backend.remote(args.ctrl_eval_batch_size))
                backend_metrics = ray.get(ctrl_world_inference_pool[0].evaluate_backend_prediction_metrics.remote(
                    windows, seed=args.ctrl_eval_seed))
                for key, value in backend_metrics.items():
                    if isinstance(value, (float, int)):
                        writer.add_scalar("WorldModel/WeaverEval/" + key, value, global_step)
                experiment.event("world_backend_heldout_evaluation", details=backend_metrics)
                experiment.flush()

        # Ctrl-World prediction quality eval (after latest weight sync)
        if (
            world_config.backend == "ctrl" and not run_policy.target_stop_active(target_reached) and args.ctrl_eval_interval > 0
            and global_step > 0
            and global_step % args.ctrl_eval_interval == 0
        ):
            try:
                eval_size = ray.get(wm_replay_buffers[0].eval_size.remote())
                if eval_size >= args.ctrl_eval_batch_size:
                    obs_np, act_np, instructions = ray.get(
                        wm_replay_buffers[0].sample_eval.remote(
                            args.ctrl_eval_batch_size
                        )
                    )
                    ctrl_eval_metrics = ray.get(
                        ctrl_world_inference_pool[0].evaluate_prediction_metrics.remote(
                            obs_np,
                            act_np,
                            instructions,
                            max_horizon=args.ctrl_eval_max_horizon,
                            lpips_net=args.ctrl_eval_lpips_net,
                            micro_batch_size=args.ctrl_eval_micro_batch_size,
                            seed=args.ctrl_eval_seed,
                        )
                    )
                    for metric in ('mse', 'psnr', 'ssim', 'lpips'):
                        label = metric.upper()
                        writer.add_scalar(
                            f'WorldModel/Eval/{label}',
                            ctrl_eval_metrics[metric],
                            global_step,
                        )
                        writer.add_scalar(
                            f'WorldModel/Eval/{label}_Std',
                            ctrl_eval_metrics[f'{metric}_std'],
                            global_step,
                        )
                        for h, value in ctrl_eval_metrics[
                            f'{metric}_by_horizon'
                        ].items():
                            writer.add_scalar(
                                f'WorldModel/Eval/{label}_h{h}', value, global_step
                            )
                            writer.add_scalar(
                                f'WorldModel/Eval/{label}_h{h}_Std',
                                ctrl_eval_metrics[f'{metric}_std_by_horizon'][h],
                                global_step,
                            )
                    for view_prefix, tb_prefix in (
                        ('agent_view', 'AgentView'),
                        ('wrist_view', 'WristView'),
                    ):
                        for metric in ('psnr', 'ssim', 'lpips'):
                            if f'{view_prefix}_{metric}' not in ctrl_eval_metrics:
                                continue
                            label = metric.upper()
                            writer.add_scalar(
                                f'WorldModel/Eval/{tb_prefix}_{label}',
                                ctrl_eval_metrics[f'{view_prefix}_{metric}'],
                                global_step,
                            )
                            writer.add_scalar(
                                f'WorldModel/Eval/{tb_prefix}_{label}_Std',
                                ctrl_eval_metrics[f'{view_prefix}_{metric}_std'],
                                global_step,
                            )
                    writer.add_scalar(
                        'WorldModel/Eval/Fixed_Set_Size', eval_size, global_step
                    )
                    writer.flush()
                    print(
                        f"[CtrlEval] step={global_step} "
                        f"LPIPS={ctrl_eval_metrics['lpips']:.4f} "
                        f"PSNR={ctrl_eval_metrics['psnr']:.3f} "
                        f"SSIM={ctrl_eval_metrics['ssim']:.4f} "
                        f"horizon={int(ctrl_eval_metrics['eval_horizon'])} "
                        f"windows={int(ctrl_eval_metrics['num_windows'])}"
                    )
                else:
                    print(
                        f"[CtrlEval] skip step={global_step}: "
                        f"fixed holdout {eval_size} < {args.ctrl_eval_batch_size}"
                    )
            except Exception as e:
                import traceback
                print(f"[CtrlEval] failed at step={global_step}: {e}", flush=True)
                traceback.print_exc()

        if os.environ.get("ACCERL_CAPTURE_ONLY") != "1" and global_step > 0 and global_step % args.ckpt_every_steps == 0:
            ray.get(trainer_group[0].save_agent.remote(args.ckpt_dir, global_step))

        snapshot_interval = int(os.environ.get("ACCERL_SNAPSHOT_INTERVAL", "25"))
        if not run_policy.target_stop_active(target_reached) and snapshot_dir and snapshot_interval > 0 and global_step % snapshot_interval == 0:
            publish_snapshot("periodic")

        current_time = time.time()
        if current_time - last_log_time > args.log_interval_seconds:
            all_stats = epoch_stats if epoch_stats is not None else ray.get(stats_actor.get_stats.remote())

            elapsed_log_time = current_time - last_log_time
            steps_since_last_log = global_step - last_log_global_step
            training_speed_steps_per_sec = steps_since_last_log / elapsed_log_time if elapsed_log_time > 0 else 0.0

            timing_stats = all_stats.pop("_timings_", {})
            global_stats = all_stats.pop("_global_rollout_")
            eval_stats = all_stats.pop("_global_eval_")
            avg_return = global_stats["avg_return"]
            avg_ep_len = global_stats["avg_ep_len"]
            total_episodes = global_stats["total_episodes_processed"]
            total_env_steps = global_stats["total_env_steps"]
            avg_step_time = global_stats["avg_step_time"]

            eval_avg_return = eval_stats["avg_return"]
            eval_avg_ep_len = eval_stats["avg_ep_len"]
            eval_total_episodes = eval_stats["total_episodes_processed"]
            eval_env_steps = eval_stats["total_env_steps"]
            eval_avg_step_time = eval_stats["avg_step_time"]

            total_losses, p_losses, v_losses, e_losses, kl_losses, lrs_list, _, ents, avg_kl_divs, perf_metrics_list, wm_metrics_list = zip(*results)
            current_lrs = lrs_list[0]
            wm_metrics = combine_wm_metrics(wm_metrics_list)

            elapsed_time = current_time - start_time
            total_buffer_size = sum(ray.get([rb.size.remote() for rb in replay_buffers]))
            wm_buffer_size = sum(ray.get([rb.size.remote() for rb in wm_replay_buffers]))
            wm_eval_size = sum(ray.get([rb.eval_size.remote() for rb in wm_replay_buffers]))
            episode_stats_list = ray.get(
                [rb.episode_stats.remote() for rb in wm_replay_buffers]
            )
            wm_train_episodes = sum(s["train"] for s in episode_stats_list)
            wm_eval_episodes = sum(s["eval"] for s in episode_stats_list)

            wm_loss_str = ""
            if 'reward_loss' in wm_metrics:
                wm_loss_str += f" | reward loss: {wm_metrics['reward_loss']:.4f}"

            print(f"更新步 {global_step}/{args.train_iters} | 时间: {elapsed_time:.1f}s | "
                  f"全局平均奖励: {avg_return:.2f} | 全局平均幕长: {avg_ep_len:.1f} | Eval奖励: {eval_avg_return:.2f} | "
                  f"value loss: {np.mean(v_losses):.4f}{wm_loss_str} | "
                  f"LR(V/P): {current_lrs['value']:.7f}/{current_lrs['policy']:.7f} | "
                  f"Episodes数量: {total_episodes:,} | Step平均时间: {avg_step_time:.3f}s")

            writer.add_scalar('Train/Learning_Rate/Value', current_lrs['value'], global_step)
            writer.add_scalar('Train/Learning_Rate/Policy', current_lrs['policy'], global_step)
            if 'reward' in current_lrs:
                writer.add_scalar('Train/Learning_Rate/Reward', current_lrs['reward'], global_step)
            if 'ctrl' in current_lrs:
                writer.add_scalar('Train/Learning_Rate/CtrlWorld', current_lrs['ctrl'], global_step)

            writer.add_scalar('Loss/Total', np.mean(total_losses), global_step)
            writer.add_scalar('Loss/Policy', np.mean(p_losses), global_step)
            writer.add_scalar('Loss/Value', np.mean(v_losses), global_step)
            writer.add_scalar('Loss/Entropy', np.mean(e_losses), global_step)
            writer.add_scalar('Loss/KL', np.mean(kl_losses), global_step)

            if 'ctrl_loss' in wm_metrics:
                writer.add_scalar('Loss/CtrlWorld', wm_metrics['ctrl_loss'], global_step)
                if math.isfinite(wm_metrics.get('ctrl_grad_norm', float('nan'))):
                    writer.add_scalar('WorldModel/Diagnostics/Ctrl_Grad_Norm', wm_metrics['ctrl_grad_norm'], global_step)
                    writer.add_scalar('WorldModel/Diagnostics/Ctrl_Grad_Clip_Ratio', wm_metrics['ctrl_clip_ratio'], global_step)
            writer.add_scalar(
                'WorldModel/Diagnostics/Sampled_Unique_Episodes',
                wm_metrics.get('ctrl_sample_unique_episodes', 0),
                global_step,
            )
            writer.add_scalar(
                'WorldModel/Diagnostics/Fresh_Episode_Ratio',
                wm_metrics.get('ctrl_sample_fresh_episode_ratio', 0.0),
                global_step,
            )
            if 'reward_loss' in wm_metrics:
                writer.add_scalar('Loss/Reward_Model', wm_metrics['reward_loss'], global_step)
                if math.isfinite(wm_metrics.get('reward_grad_norm', float('nan'))):
                    writer.add_scalar('WorldModel/Diagnostics/Reward_Grad_Norm', wm_metrics['reward_grad_norm'], global_step)
                    writer.add_scalar('WorldModel/Diagnostics/Reward_Grad_Clip_Ratio', wm_metrics['reward_clip_ratio'], global_step)
            if 'reward_tp' in wm_metrics:
                writer.add_scalar('WorldModel/Reward_TP', wm_metrics['reward_tp'], global_step)
                writer.add_scalar('WorldModel/Reward_TN', wm_metrics['reward_tn'], global_step)
                writer.add_scalar('WorldModel/Reward_FP', wm_metrics['reward_fp'], global_step)
                writer.add_scalar('WorldModel/Reward_FN', wm_metrics['reward_fn'], global_step)
                pos_total = wm_metrics['reward_tp'] + wm_metrics['reward_fn']
                neg_total = wm_metrics['reward_tn'] + wm_metrics['reward_fp']
                reward_pos_acc = wm_metrics['reward_tp'] / pos_total if pos_total > 0 else 0.0
                reward_neg_acc = wm_metrics['reward_tn'] / neg_total if neg_total > 0 else 0.0
                writer.add_scalar('WorldModel/Reward_Pos_Acc', reward_pos_acc, global_step)
                writer.add_scalar('WorldModel/Reward_Neg_Acc', reward_neg_acc, global_step)
            if 'reward_pos_ratio_batch' in wm_metrics:
                writer.add_scalar(
                    'WorldModel/Reward_Batch_Pos_Ratio',
                    wm_metrics['reward_pos_ratio_batch'],
                    global_step,
                )

            writer.add_scalar('Metrics/Entropy', np.mean(ents), global_step)
            writer.add_scalar('Metrics/KL_Divergence', np.mean(avg_kl_divs), global_step)
            writer.add_scalar('Metrics/Training_Speed_Steps_per_Sec', training_speed_steps_per_sec, global_step)
            for metric_name, metric_value in timing_stats.items():
                writer.add_scalar(f'Performance/{metric_name}', metric_value, global_step)
            avg_policy_sample_time = np.mean([pm["policy_sample_time"] for pm in perf_metrics_list])
            avg_policy_prep_time = np.mean([pm["policy_prep_time"] for pm in perf_metrics_list])
            avg_policy_train_time = np.mean([pm["policy_train_time"] for pm in perf_metrics_list])
            avg_wm_sample_time = np.mean([pm.get("wm_sample_time", 0.0) for pm in perf_metrics_list])
            avg_wm_prep_time = np.mean([pm.get("wm_prep_time", 0.0) for pm in perf_metrics_list])
            writer.add_scalar('Performance/policy_sample_time', avg_policy_sample_time, global_step)
            writer.add_scalar('Performance/policy_prep_time', avg_policy_prep_time, global_step)
            writer.add_scalar('Performance/policy_train_time', avg_policy_train_time, global_step)
            writer.add_scalar('Performance/wm_sample_time', avg_wm_sample_time, global_step)
            writer.add_scalar('Performance/wm_prep_time', avg_wm_prep_time, global_step)
            writer.add_scalar('Performance/policy_batch_wait_time',
                              np.mean([pm.get("policy_batch_wait_time", 0.0) for pm in perf_metrics_list]), global_step)
            writer.add_scalar('Performance/wm_batch_wait_time',
                              np.mean([pm.get("wm_batch_wait_time", 0.0) for pm in perf_metrics_list]), global_step)

            reward_times = [pm.get("reward_train_time") for pm in perf_metrics_list if "reward_train_time" in pm]
            if reward_times:
                avg_reward_train_time = np.mean(reward_times)
                writer.add_scalar('Performance/reward_train_time', avg_reward_train_time, global_step)
            ctrl_times = [pm.get("ctrl_train_time") for pm in perf_metrics_list if "ctrl_train_time" in pm]
            if ctrl_times:
                writer.add_scalar('Performance/ctrl_train_time', np.mean(ctrl_times), global_step)
            writer.add_scalar('Performance/train_time', train_time, global_step)
            writer.add_scalar('Performance/sync_time', sync_time, global_step)
            writer.add_scalar('Performance/train_time_total', time.time() - t_train_start, global_step)

            writer.add_scalar('Rollout/_Global/Average_Return', avg_return, global_step)
            writer.add_scalar('Rollout/_Global/Average_Episode_Length', avg_ep_len, global_step)
            writer.add_scalar('Rollout/_Global/Average_Imagine_Reward', global_stats.get("avg_imagine_reward", 0.0), global_step)
            if threshold_tracker is None:
                writer.add_scalar('Eval/_Global/Average_Return', eval_avg_return, global_step)
            writer.add_scalar('Eval/_Global/Average_Episode_Length', eval_avg_ep_len, global_step)

            pipeline_status = ray.get(wm_replay_buffers[0].pipeline_status.remote())
            experiment.event("data_pipeline_status", details={"policy_step": global_step, **pipeline_status})
            for section, fields in pipeline_status.items():
                if isinstance(fields, dict):
                    for metric, value in fields.items():
                        if isinstance(value, (int, float)):
                            writer.add_scalar(f'Pipeline/{section}/{metric}', value, global_step)
            writer.add_scalar('System/Replay_Buffer_Size_Total', total_buffer_size, global_step)
            writer.add_scalar('System/WM_Replay_Buffer_Size_Total', wm_buffer_size, global_step)
            writer.add_scalar('System/WM_Fixed_Eval_Size', wm_eval_size, global_step)
            writer.add_scalar('System/WM_Train_Episodes', wm_train_episodes, global_step)
            writer.add_scalar('System/WM_Eval_Episodes', wm_eval_episodes, global_step)
            writer.add_scalar('System/Total_Episodes_Processed', total_episodes, global_step)
            writer.add_scalar('System/Total_Env_Steps', total_env_steps, global_step)
            writer.add_scalar('System/Avg_Step_Time', avg_step_time, global_step)
            writer.add_scalar('System/Eval_Total_Episodes_Processed', eval_total_episodes, global_step)
            writer.add_scalar('System/Eval_Total_Env_Steps', eval_env_steps, global_step)
            writer.add_scalar('System/Eval_Avg_Step_Time', eval_avg_step_time, global_step)
            writer.add_scalar('System/Active_Rollout_Actors', global_stats.get("active_actor_count", 0), global_step)
            writer.add_scalar('System/Total_Samples_Produced', global_stats.get("total_samples_produced", 0), global_step)
            writer.add_scalar('System/Total_Imagine_Samples', global_stats.get("total_imagine_samples", 0), global_step)

            for env_name, env_stats in all_stats.items():
                if env_name.startswith("eval_"):
                    tag_prefix = f"Eval/{env_name.replace('eval_', '')}"
                    writer.add_scalar(f'{tag_prefix}/Average_Return', env_stats['avg_return'], global_step)
                    writer.add_scalar(f'{tag_prefix}/Average_Episode_Length', env_stats['avg_ep_len'], global_step)
                    writer.add_scalar(f'{tag_prefix}/Success_Rate', env_stats['avg_success_rate'], global_step)
                    writer.add_scalar(f'{tag_prefix}/Total_Episodes', env_stats['total_episodes'], global_step)
                else:
                    tag_prefix = f"Rollout/{env_name}"
                    writer.add_scalar(f'{tag_prefix}/Average_Return', env_stats['avg_return'], global_step)
                    writer.add_scalar(f'{tag_prefix}/Average_Episode_Length', env_stats['avg_ep_len'], global_step)
                    writer.add_scalar(f'{tag_prefix}/Success_Rate', env_stats['avg_success_rate'], global_step)
                    writer.add_scalar(f'{tag_prefix}/Total_Episodes', env_stats['total_episodes'], global_step)

            last_log_time = current_time
            last_log_global_step = global_step

    completion_reason = run_policy.stop_reason(global_step=global_step, train_iters=args.train_iters,
        target_reached=target_reached, requested_stop=stop_requested(),
        completed_epochs=completed_epochs, measure_stop_epochs=measure_stop_epochs)
    nsight_windows.close()
    ray.get([buffer.stop_pipeline.remote() for buffer in wm_replay_buffers])
    publish_snapshot("final")
    experiment.event("experiment_completed", details={"global_step": global_step,
                     "completed_epochs": completed_epochs,
                     "service_plan": service_plan.to_dict(), "world_backend_config": world_config.to_dict(),
                     "run_started_ns": run_started_ns,
                     "threshold_state": threshold_tracker.as_dict() if threshold_tracker else None,
                     "stop_reason": completion_reason, "target_reached": bool(target_reached),
                     "stop_on_target": run_policy.stop_on_target, "until_deadline": run_policy.until_deadline,
                     "deadline_unix_seconds": run_policy.deadline_unix_seconds,
                     "stop_policy": run_policy.metadata(),
                     "states": ray.get([trainer.get_training_state.remote() for trainer in all_trainers])})
    experiment.close()
    print(f"\n完成 {global_step} 次训练与同步循环；scheduled_stop={stop_requested()}", flush=True)
    writer.close()
    ray.shutdown()



class CaptureSummaryWriter:
    """One metric stream, optionally mirrored to a standard TensorBoard run."""
    def __init__(self, path, tensorboard=False):
        import hashlib
        self.tensorboard = SummaryWriter(path, flush_secs=5) if tensorboard else None
        self.telemetry = Telemetry("driver", "metrics")
        self.telemetry.config({"args": vars(args), "source": __file__, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "cwd": os.getcwd(), "python": sys.executable, "torch": torch.__version__, "ray": ray.__version__, "module_paths": {key: getattr(sys.modules.get(key),"__file__",None) for key in ("rl","prismatic","ctrl_world","libero")}})
    def add_scalar(self, tag, value, step, walltime=None, double_precision=False):
        self.telemetry.event("metric", tag=tag, value=float(value), step=int(step),
                             **({"ts_ns": int(walltime * 1e9)} if walltime is not None else {}))
        if self.tensorboard is not None:
            self.tensorboard.add_scalar(tag, value, step, walltime=walltime,
                                        new_style=double_precision, double_precision=double_precision)
    def flush(self):
        self.telemetry.flush()
        if self.tensorboard is not None:
            self.tensorboard.flush()
    def close(self):
        self.telemetry.close()
        if self.tensorboard is not None:
            self.tensorboard.close()

if os.environ.get("ACCERL_TELEMETRY_DB"):
    install_hooks(globals())
    install_training_hooks(globals())

if __name__ == "__main__":
    args = parse_args()
    print("=" * 80)
    print("命令行参数:")
    print("-" * 80)
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("=" * 80)
    try:
        main(args)
    finally:
        ray.shutdown()
