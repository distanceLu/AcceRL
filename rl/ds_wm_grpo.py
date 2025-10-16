import os
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["TMPDIR"] = "/dev/shm"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import time
import asyncio
from collections import defaultdict
from typing import Dict, Optional, Tuple, List
import math
import shutil
import socket
import contextlib

import numpy as np

import ray
import torch
import deepspeed
from torch.utils.tensorboard import SummaryWriter

# OpenVLA 和 Libero 工具
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM
from experiments.robot.libero.libero_utils import GenerateConfig

from rl.world_model import WorldModel, compute_grpo_advantages, create_validity_mask, compute_grpo_policy_loss
from ds_com import TrainerActorCom, InferenceActorCom
from rl.ds_wm import StatsActor, ReplayBufferActor, ImaginationBufferActor, InferenceActor, RolloutWorkerActor, ImaginedExperience

# ================================================================
# 0. 超参数与配置
# ================================================================
EXP_NAME = "grpo"
BENCHMARK = "libero_spatial"

# 分布式系统参数
NUM_TRAINER_GPUS = 2
NUM_INFERENCE_ACTORS = 1
NUM_IMAGINATION_ACTORS = 1 # 使用1个专用的GPU Actor来生成想象数据
NUM_ROLLOUT_WORKERS = 9
ROLLOUT_LOCAL_BUF = 64
INFERENCE_BATCH = 4
INFERENCE_TIMEOUT_MS = 300
REPLAY_CAPACITY = 10000
IMAGINATION_REPLAY_CAPACITY = 1000
TRAIN_BATCH_SIZE = 24
WORLD_ACCUM = 6
AGENT_ACCUM = 21
TRAIN_ITERS = 100000

# GRPO / PPO
GRPO_N = 16      # 每个初始状态生成的轨迹数
GAMMA = 0.99
LAMBDA = 0.95   # GAE lambda, not used for GRPO advantage calculation
CLIP_EPS = 0.2
VF_COEF = 0.0   # 价值函数损失系数, 在GRPO中设为0
ENT_COEF = 0.0
KL_COEF = 0.02

# 世界模型想象步数
IMAGINE_MAX_HORIZON = 15

# AE 和 IL 损失的系数
RT_LOSS_COEF = 1.0  # reward-termination分类损失系数

AE_LOSS_COEF = 1.0

# 奖励缩放
REWARD_SCALE = 1.0

# 学习率调度参数
WORLD_LR = 3e-5
POLICY_LR = 1e-6
WORLD_WARMUP_STEPS = 500
POLICY_WARMUP_STEPS = 500
POLICY_TRAIN_START_STEP = 10

# 日志
MOVING_AVG_WINDOW = 1000
LOG_INTERVAL_SECONDS = 10
SAVE_INTERVAL_STEPS = 1000

# 通信组
BROADCAST_GROUP_NAME = "trainer_to_inference_broadcast"

# OpenVLA 加载配置
USE_BF16: bool = True
TORCH_DTYPE = torch.bfloat16 if USE_BF16 else torch.float32
PRETRAINED_CHECKPOINT = "/cpfs01/lcx_workspace/models/openvla-7b-oft-finetuned-libero-spatial-object-goal-10/"
CHECKPOINT2 = "/cpfs01/lcx_workspace/models/WorldModel_ds_rew_termin_3class_1760519458/checkpoint_1000"

INP_MAX_LEN = 100  # 输入input_id的最大长度


@ray.remote(num_gpus=1)
class ImaginationRolloutActor(InferenceActorCom):
    """这个Actor现在只负责生成想象数据，并将其分发到多个Buffer中"""
    def __init__(self, actor_id: int, cfg: GenerateConfig, real_replay_buffer: ray.actor.ActorHandle, imagination_buffers: List[ray.actor.ActorHandle]):
        super().__init__()
        self.actor_id = actor_id
        self.cfg = cfg
        self.real_replay_buffer = real_replay_buffer
        self.imagination_buffers = imagination_buffers # <--- 接收Buffer列表
        self.num_buffers = len(imagination_buffers)
        
        print(f"ImaginationRolloutActor {actor_id}: 正在加载 WorldModel...")
        self.model = WorldModel(cfg, torch_dtype=TORCH_DTYPE, checkpoint_dir=cfg.checkpoint2, freeze_value=True)
        self.model.cuda()
        self.model.eval()
        
        self.generation_batch_size = 32 * self.num_buffers # 一次生成足够分发给所有buffer的轨迹
        print(f"ImaginationRolloutActor {self.actor_id} 初始化于 GPU: {ray.get_gpu_ids()}, 将分发数据到 {self.num_buffers} 个Buffer。")

    def get_model_keys(self):
        if self.model is None: return {}
        return {k: v.abs().sum().item() for k, v in self.model.state_dict().items()}

    async def run_generation_loop(self):
        print(f"ImaginationRolloutActor {self.actor_id}: 想象数据生成循环已启动。")
        while True:
            try:
                # 1. 为 GRPO 准备初始状态
                num_start_states = self.generation_batch_size // GRPO_N
                if num_start_states == 0:
                    raise RuntimeError(f"ImaginationRolloutActor {self.actor_id}: generation_batch_size ({self.generation_batch_size}) 太小，无法为 GRPO_N={GRPO_N} 生成至少一个组。")

                start_obs_list_raw = await self.real_replay_buffer.sample.remote(num_start_states)
                if start_obs_list_raw is None:
                    print(f"ImaginationRolloutActor {self.actor_id}: 无法从真实回放池采样，等待...")
                    await asyncio.sleep(5)
                    continue
                
                start_obs_list = start_obs_list_raw[0] 
                start_states_batch_single = self.model.prepare_inputs_batch(start_obs_list, INP_MAX_LEN)

                # 为 GRPO 复制初始状态
                start_states_batch = {}
                for key, tensor in start_states_batch_single.items():
                    start_states_batch[key] = tensor.repeat_interleave(GRPO_N, dim=0)

                # 2. 进行想象
                with torch.inference_mode():
                    imagine_start_time = time.time()
                    (imagined_mus, imagined_log_stds, _, imagined_rewards, imagined_dones, 
                     _, imagined_actions, imagined_multimodal_embs, 
                     imagined_att_masks, imagined_step_counts) = self.model.imagine(start_states_batch, IMAGINE_MAX_HORIZON)
                    imagine_duration = time.time() - imagine_start_time

                # 3. 计算 GRPO 优势和有效性掩码
                imagined_advs, trajectory_validity_mask = compute_grpo_advantages(imagined_rewards, imagined_dones, GAMMA, GRPO_N)
                step_validity_mask, _ = create_validity_mask(imagined_dones)

                # 4. 展平并处理数据
                T, B = imagined_dones.shape
                imagined_multimodal_embs_flat = imagined_multimodal_embs.view(T * B, *imagined_multimodal_embs.shape[2:])
                imagined_att_masks_flat = imagined_att_masks.view(T * B, *imagined_att_masks.shape[2:])
                imagined_actions_flat = imagined_actions.view(T * B, *imagined_actions.shape[2:])
                imagined_mus_flat = imagined_mus.view(T * B, *imagined_mus.shape[2:])
                imagined_log_stds_flat = imagined_log_stds.view(T * B, *imagined_log_stds.shape[2:])
                step_validity_mask_flat = step_validity_mask.view(T * B)
                labels_np = start_states_batch['labels'].cpu().numpy()
                imagined_step_counts_flat = imagined_step_counts.view(T * B).cpu().numpy()

                # 5. 将有效数据打包成 ImaginedExperience
                all_new_experiences = []
                for i in range(T * B):
                    traj_idx = i % B
                    # 一个步骤是有效的，当且仅当它在第一次done之前，并且它所属的GRPO组是有效的（回报不全相同）
                    if step_validity_mask_flat[i] and trajectory_validity_mask[traj_idx]:
                        advantage_for_this_traj = imagined_advs[traj_idx].item()
                        
                        exp = ImaginedExperience(
                            multimodal_emb=imagined_multimodal_embs_flat[i].float().cpu().numpy(),
                            attention_mask=imagined_att_masks_flat[i].cpu().numpy(),
                            labels=labels_np[traj_idx], # 使用复制后的索引
                            action=imagined_actions_flat[i].cpu().numpy(),
                            old_mu=imagined_mus_flat[i].cpu().numpy(),
                            old_log_std=imagined_log_stds_flat[i].cpu().numpy(),
                            advantage=advantage_for_this_traj,
                            value_target=0.0,  # 在GRPO中不使用
                            step_count=int(imagined_step_counts_flat[i])
                        )
                        all_new_experiences.append(exp)
                print(f"ImaginationRolloutActor {self.actor_id}: self.model.imagine() took {imagine_duration:.4f} seconds. imagined_actions: {imagined_actions.shape}. Valid sample: {len(all_new_experiences)}", flush=True)
                
                if all_new_experiences:
                    chunks = np.array_split(all_new_experiences, self.num_buffers)
                    for i, chunk in enumerate(chunks):
                        if len(chunk) > 0:
                            self.imagination_buffers[i].add_batch.remote(list(chunk))

            except Exception as e:
                import traceback
                print(f"[ERROR] ImaginationRolloutActor {self.actor_id} 生成循环失败: {e}", flush=True)
                traceback.print_exc()
                await asyncio.sleep(5)

# ================================================================
# 4. 训练器 (TrainerActor)
# ================================================================
@ray.remote(num_gpus=1)
class TrainerActor(TrainerActorCom):
    def __init__(self, rank, world_size, replay_buffer, imagination_buffer, cfg):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.replay_buffer = replay_buffer
        self.imagination_buffer = imagination_buffer # 这是指向一个ImaginationBufferActor的句柄
        self.cfg = cfg
        self.model = None
        self.optimizer = None
        self.base_model = None
        self.data_dtype = None
        
        self.next_wm_batch: Optional[Tuple] = None
        self.next_policy_batch: Optional[Dict] = None
        
        self.wm_data_fetching_task = None
        self.policy_data_fetching_task = None
        
        self.global_step = 0
        print(f"TrainerActor Rank {self.rank} 初始化于 GPU: {ray.get_gpu_ids()}")

    def get_rank(self):
        """返回当前 actor 的 rank。"""
        return self.rank

    def save_model(self, save_dir: str):
        """由 rank 0 调用，用于保存模型。"""
        if self.rank != 0:
            print(f"警告: save_model 应该只在 rank 0 上调用，但被 rank {self.rank} 调用。跳过。")
            return
        
        # self.model.module 是底层的 WorldModel
        # 我们只从 rank 0 发起并处理文件IO。
        # 获取底层的模型
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        print(f"\nTrainer Rank 0: 正在保存模型到 '{save_dir}'...")
        model_to_save.save_checkpoint(save_dir)
        print(f"Trainer Rank 0: 模型保存完成。")

    def get_model_keys(self):
        if self.model is None:
            print("模型尚未初始化。请先调用 setup_deepspeed_group()。")
            return {}
        module = self.model.module if hasattr(self.model, "module") else self.model
        sd = module.state_dict()
        return {k: float(v.abs().sum().item()) for k, v in sd.items()}

    def get_node_ip(self):
        return ray.util.get_node_ip_address()

    def setup_deepspeed_group(self, master_addr, master_port):
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["LOCAL_RANK"] = "0"
        deepspeed.init_distributed(dist_backend="nccl")

        # 使用 WorldModel 进行训练
        print(f"Trainer {self.rank}: 正在加载 OpenVLA WorldModel...")
        model = WorldModel(self.cfg, torch_dtype=TORCH_DTYPE, checkpoint_dir=self.cfg.checkpoint2, freeze_value=True)
        self.base_model = model

        param_groups = self.base_model.get_parameter_groups()
        optimizer_params = [
            {"params": pg["params"], "name": pg["name"], "lr": POLICY_LR if pg["name"] == "policy" else WORLD_LR}
            for pg in param_groups
        ]
        
        ds_config = {
            "train_micro_batch_size_per_gpu": TRAIN_BATCH_SIZE,
            "gradient_accumulation_steps": 1,
            "optimizer": {"type": "AdamW", "params": {}},
            "bf16": {"enabled": USE_BF16},
            "zero_optimization": {"stage": 2, "overlap_comm": True, "contiguous_gradients": True},
            "gradient_clipping": 1.0,
        }

        if ds_config.get("bf16", {}).get("enabled", False): self.data_dtype = torch.bfloat16
        else: self.data_dtype = torch.float32

        self.model, self.optimizer, _, _ = deepspeed.initialize(
            model=model, config=ds_config, model_parameters=optimizer_params
        )
        print(f"TrainerActor Rank {self.rank}: DeepSpeed 训练组 (ZeRO-2) 初始化完成。")
        
        loop = asyncio.get_event_loop()
        self.wm_data_fetching_task = loop.create_task(self._wm_data_fetching_loop())
        self.policy_data_fetching_task = loop.create_task(self._policy_data_fetching_loop())

    def _get_current_lr(self, current_step, peak_lr, warmup_steps, total_steps, start_step=0):
        if current_step < start_step: return 0.0
        effective_step = current_step - start_step
        if effective_step < warmup_steps:
            return peak_lr * (effective_step / warmup_steps)
        progress = (effective_step - warmup_steps) / (total_steps - start_step - warmup_steps)
        progress = min(progress, 1.0)
        return peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    async def _wm_data_fetching_loop(self):
        print(f"Trainer {self.rank}: (WM)后台数据准备循环已启动。")
        while True:
            try:
                if self.next_wm_batch is not None:
                    await asyncio.sleep(0.1)
                    continue

                while await self.replay_buffer.size.remote() < TRAIN_BATCH_SIZE:
                    await asyncio.sleep(1)

                sampled_data = await self.replay_buffer.sample.remote(TRAIN_BATCH_SIZE)
                if sampled_data is None: continue
                
                obs_list, act_np, _, _, _, _, done_np, next_teacher_proj_feat_np, reward_np = sampled_data
                inputs_batch = self.base_model.prepare_inputs_batch(obs_list, INP_MAX_LEN)
                device = self.model.device
                
                self.next_wm_batch = (
                    inputs_batch,
                    torch.tensor(act_np, dtype=torch.float32, device=device),
                    torch.tensor(done_np, dtype=torch.bool, device=device),
                    torch.tensor(next_teacher_proj_feat_np, dtype=self.data_dtype, device=device),
                    torch.tensor(reward_np, dtype=torch.float32, device=device)
                )
            except Exception as e:
                print(f"Trainer {self.rank}: (WM)数据采样失败: {e}。将在3秒后重试。")
                await asyncio.sleep(3)

    async def _policy_data_fetching_loop(self):
        print(f"Trainer {self.rank}: (Policy)后台数据准备循环已启动。")
        while True:
            try:
                if self.next_policy_batch is not None:
                    await asyncio.sleep(0.01)
                    continue

                while await self.imagination_buffer.size.remote() < TRAIN_BATCH_SIZE:
                    await asyncio.sleep(1)

                sampled_data = await self.imagination_buffer.sample.remote(TRAIN_BATCH_SIZE)
                if sampled_data is None: continue
                
                device = self.model.device
                self.next_policy_batch = {k: torch.tensor(v, device=device) for k, v in sampled_data.items()}

            except Exception as e:
                print(f"Trainer {self.rank}: (Policy)数据采样失败: {e}。将在3秒后重试。")
                await asyncio.sleep(3)

    async def run_training_epoch(self) -> Tuple[Dict[str, float], Dict[str, int], int]:
        if self.next_wm_batch is None or self.next_policy_batch is None:
            print(f"Trainer {self.rank}: 等待初始批次...", flush=True)
            while self.next_wm_batch is None or self.next_policy_batch is None: await asyncio.sleep(0.2)
            print(f"Trainer {self.rank}: 初始数据已收到，开始训练。", flush=True)

        current_lrs = {}
        world_lr = self._get_current_lr(self.global_step, WORLD_LR, WORLD_WARMUP_STEPS, TRAIN_ITERS)
        policy_lr = self._get_current_lr(self.global_step, POLICY_LR, POLICY_WARMUP_STEPS, TRAIN_ITERS, start_step=POLICY_TRAIN_START_STEP)
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'world': param_group['lr'] = world_lr; current_lrs['world'] = world_lr
            elif param_group['name'] == 'policy': param_group['lr'] = policy_lr; current_lrs['policy'] = policy_lr
        
        epoch_losses = defaultdict(list)
        self.model.train()
        total_accum = WORLD_ACCUM + AGENT_ACCUM

        # === 阶段 1: 世界模型训练 ===
        for _ in range(WORLD_ACCUM):
            while self.next_wm_batch is None: await asyncio.sleep(0.01)
            (wm_inputs, mini_act, mini_done, mini_next_teacher_proj_feat, mini_reward) = self.next_wm_batch
            self.next_wm_batch = None
            
            wm_inp = {**wm_inputs, 'this_action': mini_act.to(self.data_dtype)}

            ae_loss, rt_loss, reward_acc, reward_mean, termin_acc, termi_mean, rt_acc = \
                self.model.module.compute_world_model_loss(wm_inp, mini_done, mini_next_teacher_proj_feat, mini_reward)
            
            world_model_loss = AE_LOSS_COEF * ae_loss + RT_LOSS_COEF * rt_loss
            self.model.backward(world_model_loss / total_accum)
            
            epoch_losses["ae_loss"].append(ae_loss.item())
            epoch_losses["rt_loss"].append(rt_loss.item())
            epoch_losses["reward_acc"].append(reward_acc.item())
            epoch_losses["reward_mean"].append(reward_mean.item())
            epoch_losses["termin_acc"].append(termin_acc.item())
            epoch_losses["termi_mean"].append(termi_mean.item())
            epoch_losses["rt_classification_acc"].append(rt_acc.item())

        # === 阶段 2: 策略训练 (GRPO) ===
        for _ in range(AGENT_ACCUM):
            while self.next_policy_batch is None: await asyncio.sleep(0.01)
            mini_policy_batch = self.next_policy_batch
            self.next_policy_batch = None
            
            # GRPO: 优势已经是归一化过的，直接使用
            advantage = mini_policy_batch['advantage']
            
            mu, log_std, _ = self.model.module.agent.forward(
                mini_policy_batch["attention_mask"].to(self.data_dtype), 
                mini_policy_batch["inputs_embeds"].to(self.data_dtype),
                mini_policy_batch["labels"],
                mini_policy_batch["step_count"]
            )
            
            # 使用 GRPO 的 PPO 裁剪损失
            policy_loss, value_loss, entropy_loss, kl_loss, entropy, kl_div_metric = compute_grpo_policy_loss(
                mu, log_std,
                mini_policy_batch['old_mu'],
                mini_policy_batch['old_log_std'],
                mini_policy_batch['action'],
                advantage,
                CLIP_EPS,
                ENT_COEF,
                KL_COEF
            )
            
            total_policy_loss = policy_loss + value_loss + entropy_loss + kl_loss
            self.model.backward(total_policy_loss / total_accum)
            
            epoch_losses["imagination_policy_loss"].append(policy_loss.item())
            epoch_losses["imagination_value_loss"].append(value_loss.item())
            epoch_losses["imagination_entropy_loss"].append(entropy_loss.item())
            epoch_losses["imagination_kl_loss"].append(kl_loss.item())
            epoch_losses["imagination_entropy"].append(entropy.item())
            epoch_losses["imagination_kl_div"].append(kl_div_metric.item())

        # === 阶段 3: 优化器步骤 ===
        self.model.step()
        self.global_step += 1

        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        return avg_losses, current_lrs, self.global_step


def build_openvla_cfg() -> GenerateConfig:
    cfg = GenerateConfig(
        pretrained_checkpoint=PRETRAINED_CHECKPOINT,
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        num_open_loop_steps=NUM_ACTIONS_CHUNK,  # 与常量保持一致
        unnorm_key="libero_spatial_no_noops",
    )
    cfg.checkpoint2 = CHECKPOINT2
    return cfg


def find_free_port() -> int:
    """
    利用 socket 绑定到端口 0 的技巧，由操作系统找到一个当前未被使用的临时端口。
    """
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def main():
    if not os.path.exists(PRETRAINED_CHECKPOINT):
        print(f"错误: OpenVLA checkpoint 路径 '{PRETRAINED_CHECKPOINT}' 不存在。")
        return

    ray.init(ignore_reinit_error=True, _temp_dir='/dev/shm')

    exp_name = f"{EXP_NAME}_{int(time.time())}"
    save_dir = f"/cpfs01/lcx_workspace/models/{exp_name}"
    log_dir = f"runs/wm2/{exp_name}"
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    stats_actor = StatsActor.remote(window_size=MOVING_AVG_WINDOW)
    print(f"TensorBoard 日志将保存在: {log_dir}")
    print(f"模型检查点将保存在: {save_dir}")

    cfg = build_openvla_cfg()

    print("--- 步骤 1: 创建 Actors ---")
    replay_buffers = [ReplayBufferActor.remote(capacity=REPLAY_CAPACITY) for _ in range(NUM_TRAINER_GPUS)]
    
    imagination_buffers = [ImaginationBufferActor.remote(capacity=IMAGINATION_REPLAY_CAPACITY) for _ in range(NUM_TRAINER_GPUS)]
    imagination_rollout_actors = [
        ImaginationRolloutActor.remote(
            actor_id=i, 
            cfg=cfg, 
            real_replay_buffer=replay_buffers[0], # 所有生成器从同一个真实池采样
            imagination_buffers=imagination_buffers # 传递所有Buffer的句柄
        ) for i in range(NUM_IMAGINATION_ACTORS)
    ]
    
    trainer_group = [
        TrainerActor.remote(
            rank=i, world_size=NUM_TRAINER_GPUS, 
            replay_buffer=replay_buffers[i], 
            imagination_buffer=imagination_buffers[i], # 每个Trainer使用其专属的Buffer
            cfg=cfg
        ) for i in range(NUM_TRAINER_GPUS)
    ]
    inference_pool = [InferenceActor.remote(actor_id=i, cfg=cfg, dtype=TORCH_DTYPE, infer_bs=INFERENCE_BATCH, infer_timeout=INFERENCE_TIMEOUT_MS, freeze_value=True, max_len=INP_MAX_LEN) for i in range(NUM_INFERENCE_ACTORS)]
    rollout_workers = [
        RolloutWorkerActor.remote(
            inference_pool[i % NUM_INFERENCE_ACTORS],
            replay_buffers[i % NUM_TRAINER_GPUS], i, stats_actor, cfg,
            benchmark_name=BENCHMARK, dtype=TORCH_DTYPE, local_buff_len=ROLLOUT_LOCAL_BUF, gamma=GAMMA, lamb=LAMBDA
        ) for i in range(NUM_ROLLOUT_WORKERS)
    ]

    print("\n--- 正在为通信组查找空闲端口... ---")
    train_group_port = find_free_port()
    broadcast_group_port = find_free_port()
    # 确保两个端口不同，尽管可能性极小
    while broadcast_group_port == train_group_port:
        broadcast_group_port = find_free_port()
    print(f"找到端口: 训练组 = {train_group_port}, 广播组 = {broadcast_group_port}")

    print("\n--- 步骤 2: 建立 DeepSpeed 训练组 ---")
    trainer_master_addr = ray.get(trainer_group[0].get_node_ip.remote())
    ray.get([actor.setup_deepspeed_group.remote(trainer_master_addr, train_group_port) for actor in trainer_group])
    print("DeepSpeed 训练组建立完成。")

    print(f"\n--- 步骤 3: 建立共享广播组 ---")
    broadcast_participants = [trainer_group[0]] + inference_pool + imagination_rollout_actors
    broadcast_group_world_size = len(broadcast_participants)
    broadcast_master_addr = ray.get(trainer_group[0].get_node_ip.remote())
    ray.get([
        actor.setup_broadcast_group.remote(
            master_addr=broadcast_master_addr, master_port=broadcast_group_port,
            group_name=BROADCAST_GROUP_NAME, group_world_size=broadcast_group_world_size,
            my_rank_in_group=rank) for rank, actor in enumerate(broadcast_participants)
    ])
    print("共享广播组建立完成。")
    all_actors_for_broadcast = inference_pool + imagination_rollout_actors
    
    inf_keys = ray.get(inference_pool[0].get_model_keys.remote())
    trainer_keys = ray.get(trainer_group[0].get_model_keys.remote())
    for key in inf_keys:
        if key not in trainer_keys:
            print(f"警告: 推理器中缺少训练器的键: {key}")
    for key in trainer_keys:
        if key not in inf_keys:
            print(f"警告: 训练器中缺少推理器的键: {key}")
    train_sig = ray.get(trainer_group[0].get_broadcast_signature.remote())
    infer_sig = ray.get(inference_pool[0].get_broadcast_signature.remote())
    # 打印前几十个，或计算哈希对比
    if len(train_sig) != len(infer_sig):
        raise RuntimeError(f"训练器与推理器的广播签名长度不匹配: {len(train_sig)} vs {len(infer_sig)}")
    for i, (a, b) in enumerate(zip(train_sig, infer_sig)):
        if a != b:
            raise RuntimeError(f"First mismatch at idx: {i}, trainer: {a}, inference: {b}")
    forward_test_tasks = [inf.forward_test.remote(NUM_ACTIONS_CHUNK, ACTION_DIM) for inf in inference_pool]
    ray.get(forward_test_tasks)
    print("推理器前向测试完成。before broadcast")
    broadcast_task = trainer_group[0].broadcast_weights.remote(BROADCAST_GROUP_NAME)
    receive_tasks = [actor.receive_and_update_weights.remote(BROADCAST_GROUP_NAME) for actor in all_actors_for_broadcast]
    ray.get([broadcast_task] + receive_tasks)
    print("初始权重已广播到所有推理和想象生成器。")
    
    print("\n--- 正在运行前向测试... ---")
    ray.get([inf.forward_test.remote(NUM_ACTIONS_CHUNK, ACTION_DIM) for inf in inference_pool])
    print("所有 Actor 前向测试完成。")

    print("\n--- 步骤 4: 启动 Rollout 和想象数据生成 ---")
    for w in rollout_workers: w.run.remote()
    for actor in imagination_rollout_actors: actor.run_generation_loop.remote()

    print("\n--- 步骤 5: 等待经验池填充 ---")
    min_real_buffer_size = TRAIN_BATCH_SIZE * WORLD_ACCUM
    min_imagined_buffer_size = TRAIN_BATCH_SIZE * AGENT_ACCUM
    
    while True:
        real_sizes = ray.get([rb.size.remote() for rb in replay_buffers])
        imagined_sizes = ray.get([ib.size.remote() for ib in imagination_buffers])
        
        real_ready = all(size >= min_real_buffer_size for size in real_sizes)
        imagined_ready = all(size >= min_imagined_buffer_size for size in imagined_sizes)
        
        print(f"等待经验池填充... "
              f"真实数据: {real_sizes} (需 >={min_real_buffer_size}) | "
              f"想象数据: {imagined_sizes} (需 >={min_imagined_buffer_size})")
              
        if real_ready and imagined_ready: break
        time.sleep(5)
    
    print("\n--- 步骤 6: 开始主训练循环 ---")
    start_time = time.time()
    last_log_time = time.time()
    global_step = 0
    last_saved_step = -1
    while global_step < TRAIN_ITERS:
        train_tasks = [trainer.run_training_epoch.remote() for trainer in trainer_group]
        results = ray.get(train_tasks)
        
        avg_losses_list, lrs_list, steps_list = zip(*results)
        global_step = steps_list[0]

        broadcast_task = trainer_group[0].broadcast_weights.remote(BROADCAST_GROUP_NAME)
        receive_tasks = [actor.receive_and_update_weights.remote(BROADCAST_GROUP_NAME) for actor in all_actors_for_broadcast]
        ray.get([broadcast_task] + receive_tasks)

        # 每 SAVE_INTERVAL_STEPS 步保存一次模型，且只保留最新的一个
        if global_step > 0 and global_step % SAVE_INTERVAL_STEPS == 0 and global_step != last_saved_step:
            if ray.get(trainer_group[0].get_rank.remote()) == 0:
                current_checkpoint_dir = os.path.join(save_dir, f"checkpoint_{global_step}")
                ray.get(trainer_group[0].save_model.remote(current_checkpoint_dir))
                previous_checkpoint_dir = os.path.join(save_dir, f"checkpoint_{last_saved_step}") if last_saved_step > 0 else None
                if previous_checkpoint_dir and os.path.exists(previous_checkpoint_dir):
                    shutil.rmtree(previous_checkpoint_dir, ignore_errors=True)
                last_saved_step = global_step

        current_time = time.time()
        if current_time - last_log_time > LOG_INTERVAL_SECONDS:
            all_stats = ray.get(stats_actor.get_stats.remote())
            global_stats = all_stats.pop("_global_")
            avg_losses = {k: np.mean([d[k] for d in avg_losses_list]) for k in avg_losses_list[0]}
            for k, v in avg_losses.items(): writer.add_scalar(f'Loss/{k}', v, global_step)
            
            current_lrs = lrs_list[0]
            total_real_buffer = sum(ray.get([rb.size.remote() for rb in replay_buffers]))
            total_imagined_buffer = sum(ray.get([ib.size.remote() for ib in imagination_buffers]))

            print(f"步 {global_step}/{TRAIN_ITERS} | 时间: {time.time() - start_time:.1f}s | "
                  f"奖励: {global_stats['avg_return']:.2f} | "
                  f"AE Loss: {avg_losses.get('ae_loss', 0):.4f} | "
                  f"P Loss(i): {avg_losses.get('imagination_policy_loss', 0):.4f} | "
                  f"V Loss(i): {avg_losses.get('imagination_value_loss', 0):.4f} | "
                  f"LR(W/P): {current_lrs['world']:.7f}/{current_lrs['policy']:.7f}")

            writer.add_scalar('Train/Learning_Rate/World', current_lrs['world'], global_step)
            writer.add_scalar('Train/Learning_Rate/Policy', current_lrs['policy'], global_step)
            writer.add_scalar('Metrics/Imagination_Entropy', avg_losses.get('imagination_entropy', 0), global_step)
            writer.add_scalar('Metrics/Imagination_KL_Divergence', avg_losses.get('imagination_kl_div', 0), global_step)
            writer.add_scalar('Rollout/_Global/Average_Return', global_stats['avg_return'], global_step)
            writer.add_scalar('Rollout/_Global/Average_Success_Rate', global_stats['avg_success_rate'], global_step)
            writer.add_scalar('System/Buffer_Size_Real', total_real_buffer, global_step)
            writer.add_scalar('System/Buffer_Size_Imagined', total_imagined_buffer, global_step)
            writer.add_scalar('Rollout/_Global/Average_Episode_Length', global_stats['avg_ep_len'], global_step)
            writer.add_scalar('System/Total_Episodes_Processed', global_stats["total_episodes_processed"], global_step)
            writer.add_scalar('System/Avg_Step_Time', global_stats["avg_step_time"] , global_step)
            for env_name, env_stats in all_stats.items():
                tag_prefix = f"Rollout/{env_name}"
                writer.add_scalar(f'{tag_prefix}/Average_Return', env_stats['avg_return'], global_step)
                writer.add_scalar(f'{tag_prefix}/Average_Episode_Length', env_stats['avg_ep_len'], global_step)
                writer.add_scalar(f'{tag_prefix}/Success_Rate', env_stats['avg_success_rate'], global_step)
                writer.add_scalar(f'{tag_prefix}/Total_Episodes', env_stats['total_episodes'], global_step)
            last_log_time = current_time

    print(f"\n成功完成 {TRAIN_ITERS} 次训练！")
    writer.close()
    ray.shutdown()


if __name__ == "__main__":
    main()