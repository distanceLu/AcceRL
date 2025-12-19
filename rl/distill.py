"""
Knowledge Distillation Training Script for OpenVLA ActorCritic Models

This script implements knowledge distillation where:
- Teacher model: Larger, pre-trained model (frozen)
- Student model: Smaller model that learns from teacher's logits
- Data: Generated on-the-fly from simulation environments
- Logging: TensorBoard for metrics tracking

Example usage:
    python rl/distill.py \
        --teacher_checkpoint /path/to/teacher/checkpoint \
        --student_checkpoint /path/to/student/checkpoint \
        --batch_size 4 \
        --learning_rate 1e-4 \
        --temperature 4.0 \
        --num_steps 10000 \
        --benchmark libero_spatial \
        --task_ids 0,1,2,3,4 \
        --teacher_num_images 2 \
        --teacher_use_proprio \
        --student_num_images 1 \
        --output_dir ./runs/distill \
        --exp_name my_distill_experiment \
        --device cuda:0
"""

import os
import time
import random
import argparse
from pathlib import Path
from collections import deque
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from rl.actor_critic_model_discrete import ActorCritic
from rl.libero_env import LiberoEnvWrapper
from rl.utils import prepare_one_obs, check_unnorm_key
from experiments.robot.libero.libero_utils import GenerateConfig, TaskSuite
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM


class DistillationLoss(nn.Module):
    """Knowledge distillation loss combining KL divergence and optional temperature scaling."""
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        """
        Args:
            temperature: Temperature for softmax scaling (higher = softer distribution)
            alpha: Weight for distillation loss vs hard target loss (if applicable)
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute distillation loss (KL divergence between soft targets).
        
        Args:
            student_logits: (B, num_dims, n_action_bins) student model logits
            teacher_logits: (B, num_dims, n_action_bins) teacher model logits
            temperature: Optional override for temperature
            
        Returns:
            Scalar loss value
        """
        student_logits = student_logits.to(torch.float32)
        teacher_logits = teacher_logits.to(torch.float32)
        temp = temperature if temperature is not None else self.temperature
        
        # Apply temperature scaling
        student_soft = F.log_softmax(student_logits / temp, dim=-1)
        teacher_soft = F.softmax(teacher_logits / temp, dim=-1)
        
        # KL divergence: KL(teacher || student) = sum(teacher * log(teacher/student))
        # = sum(teacher * log(teacher)) - sum(teacher * log(student))
        # We compute: -sum(teacher * log(student)) since first term is constant
        kl_div = F.kl_div(student_soft, teacher_soft, reduction='batchmean', log_target=False)
        
        # Scale by temperature^2 to match original distillation paper
        loss = (temp ** 2) * kl_div
        
        return loss


class DistillationTrainer:
    """Trainer for knowledge distillation between teacher and student models."""
    
    def __init__(
        self,
        teacher_cfg: GenerateConfig,
        student_cfg: GenerateConfig,
        torch_dtype: torch.dtype,
        temperature: float = 4.0,
        learning_rate: float = 1e-4,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize distillation trainer.
        
        Args:
            teacher_cfg: Configuration for teacher model
            student_cfg: Configuration for student model
            torch_dtype: Data type for model weights
            temperature: Temperature for distillation loss
            learning_rate: Learning rate for student optimizer
            device: Device to run training on
        """
        self.teacher_cfg = teacher_cfg
        self.student_cfg = student_cfg
        self.torch_dtype = torch_dtype
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize teacher model (frozen)
        print("Initializing teacher model...")
        self.teacher = ActorCritic(teacher_cfg, torch_dtype)
        check_unnorm_key(teacher_cfg, self.teacher.vla)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        print("Teacher model initialized and frozen.")
        
        # Initialize student model (trainable)
        print("Initializing student model...")
        self.student = ActorCritic(student_cfg, torch_dtype)
        check_unnorm_key(student_cfg, self.student.vla)
        self.student.train()
        print("Student model initialized.")
        
        # Distillation loss
        self.distill_loss_fn = DistillationLoss(temperature=temperature)
        
        # Optimizer (only for student parameters)
        parameter_groups = self.student.get_parameter_groups()
        self.optimizer = torch.optim.AdamW(
            [p for group in parameter_groups for p in group["params"] if p.requires_grad],
            lr=learning_rate,
            weight_decay=1e-5,
        )
        
        # Mixed precision training
        # Note: bfloat16 doesn't need gradient scaling, so we disable scaler for bfloat16
        # float16 needs scaling to prevent underflow
        use_scaler = (torch_dtype == torch.float16)
        self.scaler = torch.amp.GradScaler('cuda', enabled=use_scaler) if use_scaler else None
        
        print(f"Distillation trainer initialized on device: {self.device}")
    
    def collect_trajectory_with_teacher(
        self,
        env: LiberoEnvWrapper,
        max_steps: int = 200,
    ) -> List[Dict]:
        """
        Collect a full trajectory using teacher's actions.
        
        Args:
            env: Environment wrapper
            max_steps: Maximum steps in trajectory
            
        Returns:
            List of observation dictionaries from the trajectory
        """
        trajectory = []
        
        # Reset environment
        obs, info = env.reset(seed=random.randint(0, 10000))
        task_description = info.get("task_description", env.task_description)
        
        action_queue = deque()
        step_count = 0
        
        while step_count < max_steps:
            # Store current observation in trajectory BEFORE executing action
            # This ensures we capture the state where teacher would generate actions
            trajectory.append({
                "observation": obs.copy() if isinstance(obs, dict) else obs,
                "task_description": task_description,
            })
            
            # If action queue is empty, generate new actions using teacher
            if len(action_queue) == 0:
                # Prepare input for teacher using current observation
                inputs_t = prepare_one_obs(
                    self.teacher_cfg,
                    self.teacher.processor,
                    obs,
                    task_description,
                    self.torch_dtype,
                )
                inputs_batch = self.teacher.prepare_inputs_batch([inputs_t])
                
                # Remove proprio if not used
                if not self.teacher_cfg.use_proprio and "proprio" in inputs_batch:
                    inputs_batch.pop("proprio", None)
                
                # Teacher generates actions (use inference_mode for better performance)
                with torch.inference_mode():
                    action_logits, _ = self.teacher.forward(inputs_batch)
                
                # Post-process to get actions
                deterministic_flags = [False]  # Use stochastic for exploration
                _, _, normalized_actions = self.teacher.post_process(action_logits, deterministic_flags)
                
                # Add actions to queue (normalized_actions shape: (1, 8, 7))
                action_sequence = normalized_actions[0]  # (8, 7)
                action_queue.extend(action_sequence)
            
            # Execute teacher's action to get next observation
            if len(action_queue) > 0:
                action_norm = action_queue.popleft()
                action_env = self.teacher.vla._unnormalize_actions(action_norm, self.teacher_cfg.unnorm_key)
                obs, reward, terminated, truncated, info = env.step(action_env)
                step_count += 1
                
                # Check if episode ended
                if terminated or truncated:
                    break
        
        return trajectory
    
    def compute_distillation_loss(
        self,
        teacher_inputs_batch: Dict[str, torch.Tensor],
        student_inputs_batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute distillation loss between teacher and student.
        
        Args:
            teacher_inputs_batch: Batch of inputs for teacher (from teacher's prepare_inputs_batch)
            student_inputs_batch: Batch of inputs for student (from student's prepare_inputs_batch)
            
        Returns:
            Tuple of (loss tensor, metrics dict)
        """
        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_logits, _ = self.teacher.forward(teacher_inputs_batch)
        
        # Student forward (with grad)
        student_logits, _ = self.student.forward(student_inputs_batch)
        
        # Ensure logits have the same shape (should be (B, NUM_ACTIONS_CHUNK * ACTION_DIM, n_action_bins))
        assert teacher_logits.shape == student_logits.shape, \
            f"Logits shape mismatch: teacher {teacher_logits.shape} vs student {student_logits.shape}"
        
        # Compute distillation loss
        loss = self.distill_loss_fn(student_logits, teacher_logits)
        
        # Compute additional metrics
        with torch.no_grad():
            # KL divergence (for logging)
            student_probs = F.softmax(student_logits, dim=-1)
            teacher_probs = F.softmax(teacher_logits, dim=-1)
            kl_div = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                teacher_probs,
                reduction='batchmean',
                log_target=False,
            )
            
            # Agreement: how often student and teacher predict the same action
            teacher_preds = torch.argmax(teacher_logits, dim=-1)
            student_preds = torch.argmax(student_logits, dim=-1)
            agreement = (teacher_preds == student_preds).float().mean()
            
            # Entropy (diversity of predictions)
            student_entropy = -(student_probs * F.log_softmax(student_logits, dim=-1)).sum(dim=-1).mean()
            teacher_entropy = -(teacher_probs * F.log_softmax(teacher_logits, dim=-1)).sum(dim=-1).mean()
        
        metrics = {
            "distill_loss": loss.item(),
            "kl_divergence": kl_div.item(),
            "agreement": agreement.item(),
            "student_entropy": student_entropy.item(),
            "teacher_entropy": teacher_entropy.item(),
        }
        
        return loss, metrics
    
    def train_step(
        self,
        teacher_inputs_batch: Dict[str, torch.Tensor],
        student_inputs_batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            teacher_inputs_batch: Batch of inputs for teacher
            student_inputs_batch: Batch of inputs for student
            
        Returns:
            Dictionary of metrics
        """
        self.student.train()
        self.optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with torch.amp.autocast('cuda', enabled=(self.torch_dtype in [torch.bfloat16, torch.float16]), dtype=self.torch_dtype):
            loss, metrics = self.compute_distillation_loss(teacher_inputs_batch, student_inputs_batch)
        
        # Backward pass
        if self.scaler is not None:
            # Use scaler for float16
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Direct backward for bfloat16 (no scaling needed)
            loss.backward()
            self.optimizer.step()
        
        return metrics
    
    def save_checkpoint(self, save_dir: Path, step: int, metrics: Dict[str, float], keep_latest_only: bool = True):
        """
        Save training checkpoint.
        
        Args:
            save_dir: Directory to save checkpoint
            step: Training step number
            metrics: Training metrics
            keep_latest_only: If True, only keep the latest checkpoint (saves space)
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_data = {
            "step": step,
            "student_state_dict": self.student.state_dict(),
            "metrics": metrics,
            "student_cfg": self.student_cfg.__dict__,
            "teacher_cfg": self.teacher_cfg.__dict__,
        }
        # Only save scaler state if it exists (for float16)
        if self.scaler is not None:
            checkpoint_data["scaler_state_dict"] = self.scaler.state_dict()
        # Note: optimizer state dict can be large, comment out if not needed for resuming
        # checkpoint_data["optimizer_state_dict"] = self.optimizer.state_dict()
        
        if keep_latest_only:
            # Only keep the latest checkpoint to save space
            latest_path = save_dir / "checkpoint_latest.pt"
            
            # Delete old latest checkpoint if exists
            if latest_path.exists():
                latest_path.unlink()
            
            # Save new latest checkpoint
            torch.save(checkpoint_data, latest_path)
            print(f"Checkpoint saved to {latest_path} (step {step})")
        else:
            # Save checkpoint with step number (for final checkpoint)
            checkpoint_path = save_dir / f"checkpoint_step_{step}.pt"
            torch.save(checkpoint_data, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """
        Load checkpoint for evaluation or resuming training.
        
        Args:
            checkpoint_path: Path to checkpoint file (can be checkpoint_latest.pt or checkpoint_step_X.pt, or directory)
        """
        # If path is a directory, try to load latest checkpoint
        if checkpoint_path.is_dir():
            latest_path = checkpoint_path / "checkpoint_latest.pt"
            if latest_path.exists():
                checkpoint_path = latest_path
                print(f"Found latest checkpoint in directory, loading: {latest_path}")
            else:
                raise FileNotFoundError(f"No checkpoint_latest.pt found in {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.student.load_state_dict(checkpoint["student_state_dict"], strict=True)
        print(f"Loaded checkpoint from {checkpoint_path} (step {checkpoint.get('step', 'unknown')})")
    
    def evaluate(
        self,
        envs: List[LiberoEnvWrapper],
        num_episodes: int = 10,
        deterministic: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluate student model in environments.
        
        Args:
            envs: List of environment wrappers
            num_episodes: Number of episodes to evaluate
            deterministic: If True, use deterministic (greedy) actions
            
        Returns:
            Dictionary with evaluation metrics (success_rate, avg_reward, etc.)
        """
        self.student.eval()
        
        total_episodes = 0
        total_successes = 0
        total_rewards = []
        episode_lengths = []
        
        # Initialize environment queues
        env_queues = [deque() for _ in range(len(envs))]
        observations = []
        task_descriptions = []
        
        # Initial reset
        for i, env in enumerate(envs):
            obs, info = env.reset(seed=int(time.time()) + i)
            observations.append(obs)
            task_descriptions.append(info.get("task_description", env.task_description))
            env_queues[i].clear()
        
        active_envs = [True] * len(envs)
        episode_rewards = [0.0] * len(envs)
        episode_steps = [0] * len(envs)
        
        print(f"\n开始评估，目标 {num_episodes} 个回合...")
        
        while total_episodes < num_episodes:
            # 1. 收集需要生成新动作的环境
            need_generation_indices = []
            inputs_t_list = []
            
            for i in range(len(envs)):
                if active_envs[i] and len(env_queues[i]) == 0:
                    inputs_t = prepare_one_obs(
                        self.student_cfg,
                        self.student.processor,
                        observations[i],
                        task_descriptions[i],
                        self.torch_dtype,
                    )
                    inputs_t_list.append(inputs_t)
                    need_generation_indices.append(i)
            
            # 2. 批量生成动作
            if inputs_t_list:
                inputs_batch = self.student.prepare_inputs_batch(inputs_t_list)
                
                with torch.inference_mode():
                    action_logits, _ = self.student.forward(inputs_batch)
                
                B = action_logits.size(0)
                deterministic_flags = [deterministic] * B
                _, _, normalized_actions = self.student.post_process(action_logits, deterministic_flags)
                
                # 将动作序列添加到队列
                for idx, env_idx in enumerate(need_generation_indices):
                    action_sequence = normalized_actions[idx]  # (8, 7)
                    env_queues[env_idx].extend(action_sequence)
            
            # 3. 执行动作
            for i in range(len(envs)):
                if not active_envs[i]:
                    continue
                
                if len(env_queues[i]) == 0:
                    continue
                
                # 从队列取出动作
                action_norm = env_queues[i].popleft()
                
                # 转换为环境动作
                action_env = self.student.vla._unnormalize_actions(action_norm, self.student_cfg.unnorm_key)
                
                # 执行动作
                obs, reward, terminated, truncated, info = envs[i].step(action_env)
                
                # 更新状态
                observations[i] = obs
                episode_rewards[i] += float(reward)
                episode_steps[i] += 1
                
                # 检查是否完成
                if terminated or truncated:
                    is_success = info.get('is_success', False)
                    total_successes += is_success
                    total_episodes += 1
                    total_rewards.append(episode_rewards[i])
                    episode_lengths.append(episode_steps[i])
                    
                    if total_episodes % 10 == 0:
                        current_success_rate = total_successes / total_episodes
                        print(f"已完成 {total_episodes}/{num_episodes} 回合, "
                              f"当前成功率: {current_success_rate:.3f} ({total_successes}/{total_episodes})")
                    
                    # 重置环境并保持活跃状态（修复：不要设置为 False）
                    episode_rewards[i] = 0.0
                    episode_steps[i] = 0
                    obs, info = envs[i].reset(seed=random.randint(0, 10000))
                    observations[i] = obs
                    task_descriptions[i] = info.get("task_description", envs[i].task_description)
                    env_queues[i].clear()
                    # 保持环境活跃，继续用于下一个回合
                    
                    # 如果达到目标回合数，停止
                    if total_episodes >= num_episodes:
                        break
        
        # 计算最终指标
        success_rate = total_successes / total_episodes if total_episodes > 0 else 0.0
        avg_reward = np.mean(total_rewards) if total_rewards else 0.0
        avg_length = np.mean(episode_lengths) if episode_lengths else 0.0
        
        metrics = {
            "success_rate": success_rate,
            "total_episodes": total_episodes,
            "total_successes": total_successes,
            "avg_reward": avg_reward,
            "avg_episode_length": avg_length,
        }
        
        print("\n" + "=" * 60)
        print("评估完成!")
        print(f"总回合数: {total_episodes}")
        print(f"成功次数: {total_successes}")
        print(f"成功率: {success_rate:.3f}")
        print(f"平均奖励: {avg_reward:.4f}")
        print(f"平均回合长度: {avg_length:.2f}")
        print("=" * 60)
        
        return metrics


def main():
    parser = argparse.ArgumentParser(description="Knowledge Distillation Training")
    
    # Model configs
    parser.add_argument("--teacher_checkpoint", type=str, required=True,
                       help="Path to teacher model checkpoint")
    parser.add_argument("--student_checkpoint", type=str, required=True,
                       help="Path to student model checkpoint (or base model)")
    
    # Training configs
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--temperature", type=float, default=4.0,
                       help="Temperature for distillation loss")
    parser.add_argument("--num_steps", type=int, default=10000,
                       help="Number of training steps")
    parser.add_argument("--save_interval", type=int, default=1000,
                       help="Save checkpoint every N steps")
    parser.add_argument("--log_interval", type=int, default=100,
                       help="Log metrics every N steps")
    parser.add_argument("--trajectory_length", type=int, default=200,
                       help="Maximum length of trajectory collected by teacher")
    parser.add_argument("--samples_per_trajectory", type=int, default=10,
                       help="Number of samples to extract from each trajectory")
    
    # Environment configs
    parser.add_argument("--benchmark", type=str, default="libero_spatial",
                       choices=["libero_spatial", "libero_goal", "libero_object", "libero_10", "libero_90"],
                       help="LIBERO benchmark name")
    parser.add_argument("--task_ids", type=str, default=None,
                       help="Comma-separated list of task IDs (e.g., '0,1,2,3,4'). If None, uses all tasks.")
    parser.add_argument("--num_envs", type=int, default=10,
                       help="Number of parallel environments")
    
    # Teacher/Student config differences
    parser.add_argument("--teacher_num_images", type=int, default=2,
                       help="Number of images for teacher model")
    parser.add_argument("--teacher_use_proprio", action="store_true",
                       help="Use proprioception for teacher")
    parser.add_argument("--student_num_images", type=int, default=1,
                       help="Number of images for student model")
    parser.add_argument("--student_use_proprio", action="store_true",
                       help="Use proprioception for student")
    
    # Output configs
    parser.add_argument("--output_dir", type=str, default="./runs/distill",
                       help="Output directory for checkpoints and logs")
    parser.add_argument("--exp_name", type=str, default="distill",
                       help="Experiment name for logging")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to run training on")
    
    # Evaluation configs
    parser.add_argument("--eval_only", action="store_true",
                       help="Only evaluate, do not train")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Path to checkpoint for evaluation (e.g., checkpoint_step_2000.pt)")
    parser.add_argument("--eval_episodes", type=int, default=50,
                       help="Number of episodes for evaluation")
    parser.add_argument("--eval_deterministic", action="store_true",
                       help="Use deterministic (greedy) actions for evaluation")
    parser.add_argument("--eval_interval", type=int, default=100,
                       help="Evaluate every N training steps (0 to disable periodic evaluation)")
    
    args = parser.parse_args()
    
    # Setup
    USE_BF16 = True
    TORCH_DTYPE = torch.bfloat16 if USE_BF16 else torch.float32
    device = torch.device(args.device)
    
    # Parse task IDs
    if args.task_ids:
        task_ids = [int(x.strip()) for x in args.task_ids.split(",")]
    else:
        # Use all available tasks (adjust based on benchmark)
        max_tasks = {
            "libero_spatial": 10,
            "libero_goal": 10,
            "libero_object": 10,
            "libero_10": 10,
            "libero_90": 90,
        }
        task_ids = list(range(min(args.num_envs, max_tasks.get(args.benchmark, 10))))
    
    benchmark_enum = getattr(TaskSuite, f"LIBERO_{args.benchmark.upper().replace('LIBERO_', '')}", TaskSuite.LIBERO_SPATIAL)
    unnorm_key = f"{args.benchmark}_no_noops"
    
    # Create configs
    teacher_cfg = GenerateConfig(
        pretrained_checkpoint=args.teacher_checkpoint,
        use_l1_regression=False,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=args.teacher_num_images,
        use_proprio=args.teacher_use_proprio,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        unnorm_key=unnorm_key,
        device=device,
    )
    print(f"Teacher config: {teacher_cfg}")
    
    student_cfg = GenerateConfig(
        pretrained_checkpoint=args.student_checkpoint,
        use_l1_regression=False,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=args.student_num_images,
        use_proprio=args.student_use_proprio,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        unnorm_key=unnorm_key,
        device=device,
    )
    print(f"Student config: {student_cfg}")
    
    # Initialize trainer
    trainer = DistillationTrainer(
        teacher_cfg=teacher_cfg,
        student_cfg=student_cfg,
        torch_dtype=TORCH_DTYPE,
        temperature=args.temperature,
        learning_rate=args.learning_rate,
        device=device,
    )
    
    # Initialize environments
    print(f"Initializing {len(task_ids)} environments for benchmark: {args.benchmark}")
    envs = [
        LiberoEnvWrapper(
            benchmark_name=args.benchmark,
            task_id=task_id,
            image_size=224,
            render_mode="rgb_array",
        )
        for task_id in task_ids
    ]
    print("Environments initialized.")
    
    # Load checkpoint if provided (for evaluation or resuming)
    if args.checkpoint_path:
        checkpoint_path = Path(args.checkpoint_path)
        if not checkpoint_path.is_absolute():
            # Try relative to output_dir first, then as-is
            potential_path = Path(args.output_dir) / args.checkpoint_path
            if potential_path.exists():
                checkpoint_path = potential_path
        trainer.load_checkpoint(checkpoint_path)
    
    # Evaluation mode
    if args.eval_only:
        print("\n" + "=" * 60)
        print("评估模式")
        print("=" * 60)
        metrics = trainer.evaluate(
            envs=envs,
            num_episodes=args.eval_episodes,
            deterministic=args.eval_deterministic,
        )
        print(f"\n最终成功率: {metrics['success_rate']:.3f}")
        return
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{timestamp}_{args.exp_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tb_log_dir = output_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(tb_log_dir))
    print(f"TensorBoard logs will be saved to: {tb_log_dir}")
    
    # Training loop
    print(f"\nStarting distillation training for {args.num_steps} steps...")
    print(f"Teacher: {args.teacher_checkpoint}")
    print(f"Student: {args.student_checkpoint}")
    print(f"Batch size: {args.batch_size}, LR: {args.learning_rate}, Temperature: {args.temperature}\n")
    
    global_step = 0
    start_time = time.time()
    
    # Training loop: collect trajectories with teacher and train student
    print(f"Using teacher actions to collect trajectories (max length: {args.trajectory_length})")
    print(f"Sampling {args.samples_per_trajectory} samples per trajectory\n")
    
    while global_step < args.num_steps:
        # Collect trajectories using teacher actions
        all_trajectory_samples = []
        
        for env_idx in range(min(args.batch_size, len(envs))):
            # Collect a full trajectory using teacher
            trajectory = trainer.collect_trajectory_with_teacher(
                envs[env_idx],
                max_steps=args.trajectory_length,
            )
            
            # Sample multiple timesteps from the trajectory
            if len(trajectory) > 0:
                # Sample evenly or randomly from trajectory
                num_samples = min(args.samples_per_trajectory, len(trajectory))
                if num_samples == len(trajectory):
                    sampled_indices = list(range(len(trajectory)))
                else:
                    # Evenly sample
                    step = len(trajectory) / num_samples
                    sampled_indices = [int(i * step) for i in range(num_samples)]
                
                for idx in sampled_indices:
                    all_trajectory_samples.append(trajectory[idx])
        
        # If we don't have enough samples, collect more trajectories
        while len(all_trajectory_samples) < args.batch_size:
            env_idx = random.randint(0, len(envs) - 1)
            trajectory = trainer.collect_trajectory_with_teacher(
                envs[env_idx],
                max_steps=args.trajectory_length,
            )
            if len(trajectory) > 0:
                # Sample one random timestep
                idx = random.randint(0, len(trajectory) - 1)
                all_trajectory_samples.append(trajectory[idx])
        
        # Mini-batch training: use all collected samples for multiple gradient updates
        # Shuffle samples for better training
        random.shuffle(all_trajectory_samples)
        
        # Process all samples in mini-batches
        num_mini_batches = max(1, len(all_trajectory_samples) // args.batch_size)
        accumulated_metrics = {}
        
        for mini_batch_idx in range(num_mini_batches):
            start_idx = mini_batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, len(all_trajectory_samples))
            batch_samples = all_trajectory_samples[start_idx:end_idx]
            
            # Prepare inputs for teacher and student
            teacher_inputs_list = []
            student_inputs_list = []
            
            for sample in batch_samples:
                obs = sample["observation"]
                task_desc = sample["task_description"]
                
                # Teacher inputs
                teacher_inputs_t = prepare_one_obs(
                    teacher_cfg,
                    trainer.teacher.processor,
                    obs,
                    task_desc,
                    TORCH_DTYPE,
                )
                teacher_inputs_list.append(teacher_inputs_t)
                
                # Student inputs
                student_inputs_t = prepare_one_obs(
                    student_cfg,
                    trainer.student.processor,
                    obs,
                    task_desc,
                    TORCH_DTYPE,
                )
                student_inputs_list.append(student_inputs_t)
            
            # Batch inputs separately for teacher and student
            teacher_inputs_batch = trainer.teacher.prepare_inputs_batch(teacher_inputs_list)
            student_inputs_batch = trainer.student.prepare_inputs_batch(student_inputs_list)
            
            # Remove proprio from batch if not used (to avoid warnings)
            if not teacher_cfg.use_proprio and "proprio" in teacher_inputs_batch:
                teacher_inputs_batch.pop("proprio", None)
            if not student_cfg.use_proprio and "proprio" in student_inputs_batch:
                student_inputs_batch.pop("proprio", None)
            
            # Training step
            metrics = trainer.train_step(teacher_inputs_batch, student_inputs_batch)
            
            # Accumulate metrics (average later)
            for key, value in metrics.items():
                if key not in accumulated_metrics:
                    accumulated_metrics[key] = []
                accumulated_metrics[key].append(value)
            
            global_step += 1
        
        # Average metrics across mini-batches
        metrics = {key: np.mean(values) for key, values in accumulated_metrics.items()}
        
        # Logging
        if global_step % args.log_interval == 0:
            elapsed_time = time.time() - start_time
            steps_per_sec = global_step / elapsed_time
            
            print(f"Step {global_step}/{args.num_steps} | "
                  f"Loss: {metrics['distill_loss']:.4f} | "
                  f"KL: {metrics['kl_divergence']:.4f} | "
                  f"Agreement: {metrics['agreement']:.4f} | "
                  f"Speed: {steps_per_sec:.2f} steps/s")
            
            # Log to TensorBoard
            for key, value in metrics.items():
                writer.add_scalar(f"Train/{key}", value, global_step)
            writer.add_scalar("Train/steps_per_sec", steps_per_sec, global_step)
        
        # Periodic evaluation
        if args.eval_interval > 0 and global_step % args.eval_interval == 0:
            print(f"\n{'='*60}")
            print(f"开始评估 (Step {global_step})...")
            print(f"{'='*60}")
            
            eval_metrics = trainer.evaluate(
                envs=envs,
                num_episodes=args.eval_episodes,
                deterministic=args.eval_deterministic,
            )
            
            # Log evaluation metrics to TensorBoard
            for key, value in eval_metrics.items():
                writer.add_scalar(f"Eval/{key}", value, global_step)
            
            print(f"{'='*60}\n")
        
        # Save checkpoint (only keep latest to save space)
        if global_step % args.save_interval == 0:
            trainer.save_checkpoint(output_dir / "checkpoints", global_step, metrics, keep_latest_only=True)
    
    # Final save (save with step number for final checkpoint)
    trainer.save_checkpoint(output_dir / "checkpoints", global_step, metrics, keep_latest_only=False)
    writer.close()
    
    print(f"\nTraining completed! Final checkpoint saved.")
    print(f"TensorBoard logs: {tb_log_dir}")
    print(f"Checkpoints: {output_dir / 'checkpoints'}")


if __name__ == "__main__":
    main()

