import os
# 设置临时文件目录，避免磁盘I/O瓶颈
os.environ["TMPDIR"] = "/dev/shm"
# 为了让 Ray 能看到所有可用的 GPU，我们在脚本开头设置。
# Ray 会根据 @ray.remote(num_gpus=1) 的请求来为每个 Actor 分配和隔离 GPU。
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,4,5,6" 
# 防止 transformers 库的 tokenizer 并行化警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import random
import asyncio
from collections import deque, defaultdict
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

# --- 新增: 引入新网络和环境所需的库 ---
import numpy as np
from PIL import Image # 用于处理图像

import ray
import torch

from torch.distributions import Normal, TransformedDistribution, TanhTransform
import deepspeed
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoProcessor

from ds_com import TrainerActorCom, InferenceActorCom
from qwen_actor_critic import QwenVLWithPPOHeads
from rl.libero_env import LiberoEnvWrapper
from libero.libero import benchmark
from rl.utils import prepare_one_obs

# ================================================================
# 0. 超参数 (已为 MetaWorld reach-v3 和 Qwen-VL 模型调整)
# ================================================================
# --- MetaWorld 环境参数 ---
# Qwen-VL 的视觉编码器通常使用 448x448 的图像
OBS_SHAPE = (64, 64, 3)   # 图像观测空间 (H, W, C)
ACT_DIM = 4                # 动作空间维度
BENCHMARK = "libero_spatial"

# --- 分布式系统参数 ---
NUM_TRAINER_GPUS = 4       # Trainer使用GPU数量
NUM_INFERENCE_ACTORS = 1   # 推理Actor数量
NUM_ROLLOUT_WORKERS = 8  # 数据收集Worker数量
ROLLOUT_LOCAL_BUF = 8     # 每个Rollout Worker的本地缓冲区大小
INFERENCE_BATCH = 2        # 推理服务的批处理大小 (VLM模型需要更小的批次)
INFERENCE_TIMEOUT_MS = 300 # 推理请求的批次级超时时间 (毫秒)
REPLAY_CAPACITY = 10_000   # 每个经验池的容量 (VLM占用内存大，适当减小)
TRAIN_BATCH_SIZE = 64      # PPO的训练批次大小 (VLM模型必须使用小批次)
ACCUMULATION_STEPS = 16
TRAIN_ITERS = 100000       # 总训练迭代次数

# --- PPO 算法参数 ---
GAMMA = 0.99
LAMBDA = 0.95
LR = 1e-5                  # VLM微调通常需要更小的学习率
CLIP_EPS = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01

# --- 奖励归一化参数 ---
REWARD_SCALE = 0.01

# --- 学习率调度器参数 ---
WARMUP_STEPS = 500 # 线性预热的步数

# --- 日志和统计参数 ---
MOVING_AVG_WINDOW = 100
LOG_INTERVAL_SECONDS = 10

# --- 通信组端口和名称 ---
TRAIN_GROUP_PORT = 29531
BROADCAST_GROUP_NAME = "trainer_to_inference_broadcast"
BROADCAST_GROUP_PORT = 29532
MODEL_NAME = "/cpfs01/lcx_workspace/models/Qwen2.5-VL-3B-Instruct"

USE_BF16: bool = True  # True 使用 bfloat16；False 使用 float32
TORCH_DTYPE = torch.bfloat16 if USE_BF16 else torch.float32

@dataclass
class Experience:
    obs: dict
    action: np.ndarray
    advantage: float
    behaviour_mu: np.ndarray
    behaviour_log_std: np.ndarray
    value_target: float


# ================================================================
# 1.5. 统计模块 (StatsActor)
# ================================================================
@ray.remote
class StatsActor:
    """
    一个用于跟踪和计算统计数据的Actor，现在支持按环境名称进行多任务统计。
    """
    def __init__(self, window_size=MOVING_AVG_WINDOW):
        # 使用 defaultdict，当遇到新的环境名称时，自动创建统计数据结构
        self.stats = defaultdict(lambda: {
            "episode_returns": deque(maxlen=window_size),
            "step_times": deque(maxlen=window_size),
            "episode_lengths": deque(maxlen=window_size),
            "successes": deque(maxlen=window_size),
            "total_episodes_processed": 0
        })

    def add_episode_return(self, env_name: str, ep_return: float, step_time: float, ep_length: int, success: float):
        """
        为指定环境添加一幕的统计数据。

        参数:
            env_name (str): 完成该幕的环境名称 (例如, 'reach-v3')。
            ep_return (float): 该幕的总回报。
            step_time (float): 该幕的平均步长时间。
            ep_length (int): 该幕的总步数。
            success (float): 该幕的成功状态 (1.0 表示成功, 0.0 表示失败)。
        """
        env_stats = self.stats[env_name]
        env_stats["episode_returns"].append(ep_return)
        env_stats["step_times"].append(step_time)
        env_stats["episode_lengths"].append(ep_length)
        env_stats["successes"].append(success)
        env_stats["total_episodes_processed"] += 1

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """
        计算并返回所有环境的统计数据。

        返回:
            一个字典，键是环境名称，值是包含该环境统计数据的字典。
            此外，还包含一个特殊的 '_global_' 键，用于提供所有环境的聚合统计。
        """
        per_env_stats = {}
        
        # 用于计算全局统计的聚合列表
        all_returns = []
        all_lengths = []
        all_step_times = []
        total_episodes_processed = 0

        for env_name, env_data in self.stats.items():
            total_episodes_processed += env_data["total_episodes_processed"]
            all_returns.extend(env_data["episode_returns"])
            all_lengths.extend(env_data["episode_lengths"])
            all_step_times.extend(env_data["step_times"])
            
            if not env_data["episode_returns"]:
                per_env_stats[env_name] = {
                    "avg_return": 0.0, "avg_ep_len": 0.0, "avg_success_rate": 0.0,
                    "num_episodes_in_avg": 0, "total_episodes": env_data["total_episodes_processed"]
                }
            else:
                per_env_stats[env_name] = {
                    "avg_return": np.mean(env_data["episode_returns"]),
                    "avg_ep_len": np.mean(env_data["episode_lengths"]),
                    "avg_success_rate": np.mean(env_data["successes"]),
                    "num_episodes_in_avg": len(env_data["episode_returns"]),
                    "total_episodes": env_data["total_episodes_processed"]
                }

        # 添加一个特殊的 "_global_" 键，用于方便地获取总体统计数据
        per_env_stats["_global_"] = {
            "avg_return": np.mean(all_returns) if all_returns else 0.0,
            "avg_ep_len": np.mean(all_lengths) if all_lengths else 0.0,
            "avg_step_time": np.mean(all_step_times) if all_step_times else 0.0,
            "total_episodes_processed": total_episodes_processed,
        }

        return per_env_stats


# ================================================================
# 2. 模型、经验回放、数据收集器
# ================================================================
@ray.remote
class ReplayBufferActor:
    def __init__(self, capacity=REPLAY_CAPACITY):
        self.buffer = deque(maxlen=capacity)
    def add_batch(self, batch):
        self.buffer.extend(batch)
    def size(self):
        return len(self.buffer)
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs = np.stack([b.obs for b in batch]) 
        instructions = [b.instruction for b in batch]
        act = np.stack([b.action for b in batch])
        adv = np.asarray([b.advantage for b in batch], np.float32)
        mu_old = np.stack([b.behaviour_mu for b in batch])
        log_std_old = np.stack([b.behaviour_log_std for b in batch])
        v_targ = np.asarray([b.value_target for b in batch], np.float32)
        return obs, instructions, act, adv, mu_old, log_std_old, v_targ


@ray.remote
class RolloutWorkerActor:
    def __init__(self, infer, replay, wid, stats_actor, cfg, processor):
        self.infer, self.replay = infer, replay
        self.stats_actor = stats_actor
        self.cfg = cfg
        self.processor = processor
        self.env = LiberoEnvWrapper(
            benchmark_name=BENCHMARK,
            task_id=3,
            image_size=224,
            render_mode="rgb_array"
        )
        
        self.wid = wid
        self.local_buffer = []
        self.task_description = None
        self.current_env_name = None 

    def run(self):
        obs, info = self.env.reset(seed=self.wid)
        self.task_description = self.env.task_description
        self.current_env_name = self.env.task.name
        reward_sum = 0.0
        step_count = 0
        time_start = time.time()
        step_count_total = 0
        while True:
            inputs_t = prepare_one_obs(self.cfg, self.processor, obs, self.task_description, TORCH_DTYPE)
            action, mu, log_std, value = ray.get(self.infer.request.remote(inputs_t))
            nxt, r, term, trunc, info = self.env.step(action)
            reward_sum += r
            r *= REWARD_SCALE
            step_count += 1
            self.local_buffer.append((inputs_t, action, r, mu, log_std, value))
            obs = nxt
            step_count_total += 1
            if term or trunc:
                step_time = (time.time() - time_start) / step_count_total
                # 从最后一步的info中获取成功状态
                success = float(info.get('success', 0.0))
                # 将环境名称和成功状态传递给统计Actor
                self.stats_actor.add_episode_return.remote(
                    self.current_env_name, reward_sum, step_time, step_count, success
                )
                reward_sum = 0.0
                step_count = 0
                if self.local_buffer:
                    self._process_traj(self.local_buffer, 0.0)
                self.local_buffer.clear()
                obs, info = self.env.reset()
                self.task_description = info['instruction']
                self.current_env_name = info['env_name'] # 重置后记录新的环境名称
            elif len(self.local_buffer) == ROLLOUT_LOCAL_BUF + 1:
                _, _, _, _, _, _, bootstrap_val = self.local_buffer[-1]
                self._process_traj(self.local_buffer[:-1], bootstrap_val)
                self.local_buffer = [self.local_buffer[-1]]

    def _process_traj(self, traj_segment, bootstrap_val):
        rets, advs = [], []
        gae = 0.0
        for i in reversed(range(len(traj_segment))):
            _, _, _, r, _, _, v = traj_segment[i]
            nv = bootstrap_val if i == len(traj_segment) - 1 else traj_segment[i+1][6]
            delta = r + GAMMA * nv - v
            gae = delta + GAMMA * LAMBDA * gae
            advs.append(gae)
            rets.append(gae + v)
        advs.reverse(); rets.reverse()
        advs_np = np.array(advs, dtype=np.float32)
        advs_np = (advs_np - np.mean(advs_np)) / (np.std(advs_np) + 1e-8)
        batch = [Experience(s, instruction, a, advs_np[i], mu, log_std, rets[i]) for i, (s, instruction, a, _, mu, log_std, _) in enumerate(traj_segment)]
        self.replay.add_batch.remote(batch)


# ================================================================
# 3. 融合后的推理器 (InferenceActor)
# ================================================================
@ray.remote(num_gpus=1)
class InferenceActor(InferenceActorCom):
    def __init__(self, actor_id):
        super().__init__()
        self.actor_id = actor_id
        
        # --- 修改: 加载 Qwen-VL 模型和处理器 ---
        print(f"InferenceActor {actor_id}: 正在从 '{MODEL_NAME}' 加载模型和处理器...")
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
        self.model = QwenVLWithPPOHeads.from_pretrained(
            MODEL_NAME,
            num_labels=ACT_DIM,
            torch_dtype=torch.bfloat16, # 明确使用 bfloat16
            device_map="cuda", # 直接将模型加载到GPU
            trust_remote_code=True
        ).eval()
        
        self.batch_size = INFERENCE_BATCH
        self.timeout_sec = INFERENCE_TIMEOUT_MS / 1000.0
        self.requests, self.promises = [], []
        self.last_process_time = time.time()
        asyncio.get_event_loop().create_task(self._loop())
        print(f"InferenceActor {self.actor_id} (bf16) 初始化于 GPU: {ray.get_gpu_ids()} (批次超时: {INFERENCE_TIMEOUT_MS}ms)")

    # --- 修改: request 接受 obs 和 instruction ---
    async def request(self, obs, instruction):
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        self.requests.append((obs, instruction))
        self.promises.append(fut)
        return await fut
    
    async def _loop(self):
        while True:
            should_process = self.requests and (
                len(self.requests) >= self.batch_size or 
                time.time() - self.last_process_time > self.timeout_sec
            )
            if should_process:
                requests_to_process = self.requests
                promises_to_process = self.promises
                self.requests, self.promises = [], []
                self.last_process_time = time.time()
                
                # --- 修改: 为 VLM 准备动态指令的输入 ---
                obs_batch, instruction_batch = zip(*requests_to_process)
                
                image_batch = [Image.fromarray(obs) for obs in obs_batch]
                
                # --- 修改: 为批次中的每个指令动态创建提示 ---
                text_batch = []
                for instruction in instruction_batch:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": instruction},
                            ],
                        }
                    ]
                    prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    text_batch.append(prompt)
                
                inputs = self.processor(
                    text=text_batch,
                    images=image_batch,
                    padding=True,
                    return_tensors="pt"
                ).to(self.model.device)
                with torch.no_grad():
                    mu, log_std, values = self.model(**inputs)
                    mu = mu.float()
                    log_std = log_std.float()
                    values = values.float()
                    std = torch.exp(log_std)
                    base_dist = Normal(mu, std)
                    dist = TransformedDistribution(base_dist, TanhTransform())
                    actions = dist.sample()
                actions_np = actions.cpu().numpy()
                mu_np = mu.cpu().numpy()
                log_std_np = log_std.cpu().numpy()
                values_np = values.cpu().numpy()
                
                for i in range(len(promises_to_process)):
                    promises_to_process[i].set_result((actions_np[i], mu_np[i], log_std_np[i], values_np[i]))
            else:
                await asyncio.sleep(0.0005)


# ================================================================
# 4. 融合后的训练器 (TrainerActor)
# ================================================================
@ray.remote(num_gpus=1)
class TrainerActor(TrainerActorCom):
    def __init__(self, rank, world_size, replay_buffer):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.replay_buffer = replay_buffer
        self.model = None
        self.processor = None
        self.data_dtype = None # 将在 setup 中设置
        self.training_batch: Optional[Tuple[torch.Tensor, ...]] = None
        self.data_fetching_task = None
        print(f"TrainerActor Rank {self.rank} 初始化于 GPU: {ray.get_gpu_ids()}")

    def get_node_ip(self):
        return ray.util.get_node_ip_address()
    
    async def _data_fetching_loop(self):
        print(f"Trainer {self.rank}: 后台数据准备循环已启动。")
        while True:
            try:
                if await self.replay_buffer.size.remote() < TRAIN_BATCH_SIZE:
                    await asyncio.sleep(3)
                    continue
                # 1. 异步获取 NumPy 数据和指令
                obs_np, instructions, act_np, adv_np, mu_old_np, log_std_old_np, v_targ_np = await self.replay_buffer.sample.remote(TRAIN_BATCH_SIZE)
                
                # 2. 将数据转换为Tensor并移动到设备，但不进行VLM预处理
                act_t = torch.tensor(act_np, dtype=torch.float32).to(self.model.device)
                adv_t = torch.tensor(adv_np, dtype=torch.float32).to(self.model.device)
                mu_old_t = torch.tensor(mu_old_np, dtype=torch.float32).to(self.model.device)
                log_std_old_t = torch.tensor(log_std_old_np, dtype=torch.float32).to(self.model.device)
                v_targ_t = torch.tensor(v_targ_np, dtype=torch.float32).to(self.model.device)
                
                # 3. 原子地更新训练批次，包含指令
                self.training_batch = (obs_np, instructions, act_t, adv_t, mu_old_t, log_std_old_t, v_targ_t)
            except Exception as e:
                print(f"Trainer {self.rank}: 数据采样失败: {e}。将在3秒后重试。")
                await asyncio.sleep(3)

    def setup_deepspeed_group(self, master_addr, master_port):
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["LOCAL_RANK"] = "0"
        deepspeed.init_distributed(dist_backend="nccl")

        # --- 修改: 加载 Qwen-VL 模型 ---
        print(f"Trainer {self.rank}: 正在从 '{MODEL_NAME}' 加载模型...")
        model = QwenVLWithPPOHeads.from_pretrained(
            MODEL_NAME,
            num_labels=ACT_DIM,
            trust_remote_code=True,
            # deepspeed 会处理 torch_dtype 和 device
        )
        
        # --- 修改: 初始化处理器和文本提示 ---
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
        
        ds_config = {
            "train_micro_batch_size_per_gpu": TRAIN_BATCH_SIZE,
            "gradient_accumulation_steps": ACCUMULATION_STEPS,
            "optimizer": {"type": "AdamW", "params": {"lr": LR}}, # AdamW is often better for transformers
            "scheduler": {
                "type": "WarmupCosineLR", "params": {
                    "total_num_steps": TRAIN_ITERS, "warmup_num_steps": WARMUP_STEPS,
                    "warmup_type": "linear", "warmup_min_ratio": 0.0, "cos_min_ratio": 0.0,
                }
            },
            "bf16": {"enabled": True}, # 启用 bfloat16
            "zero_optimization": {
                "stage": 2, "allgather_partitions": True, "allgather_bucket_size": 5e8,
                "reduce_scatter": True, "reduce_bucket_size": 5e8, "overlap_comm": True,
                "contiguous_gradients": True
            },
            "gradient_clipping": 1.0, # Add gradient clipping
        }

        if ds_config.get("fp16", {}).get("enabled", False): self.data_dtype = torch.float16
        elif ds_config.get("bf16", {}).get("enabled", False): self.data_dtype = torch.bfloat16
        else: self.data_dtype = torch.float32

        self.model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)
        print(f"TrainerActor Rank {self.rank}: DeepSpeed 训练组 (ZeRO-2) 初始化完成。")
        self.data_fetching_task = asyncio.get_event_loop().create_task(self._data_fetching_loop())

    async def train_step(self) -> Tuple[float, float, float, float, float]:
        if self.training_batch is None:
            print(f"Trainer {self.rank}: 首次训练，等待初始数据批次...")
            while self.training_batch is None:
                await asyncio.sleep(0.2)
            print(f"Trainer {self.rank}: 初始数据已收到，开始训练。")
        
        obs_np, instructions, act_t, adv_t, mu_old_t, log_std_old_t, v_targ_t = self.training_batch
        
        # --- 修改: 在训练步骤中为 VLM 准备动态指令的输入 ---
        batch_size = obs_np.shape[0]
        image_batch = [Image.fromarray(obs) for obs in obs_np]
        
        # --- 修改: 为批次中的每个指令动态创建提示 ---
        text_batch = []
        for instruction in instructions:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": instruction},
                    ],
                }
            ]
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            text_batch.append(prompt)
        inputs = self.processor(
            text=text_batch,
            images=image_batch,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)
        # --- MODIFIED: 显式转换 pixel_values 为 bf16，以匹配模型期望的输入类型 ---
        inputs['pixel_values'] = inputs['pixel_values'].to(self.data_dtype)

        mu, log_std, value = self.model(**inputs)
        mu = mu.float()
        log_std = log_std.float()
        value = value.float()
        std = torch.exp(log_std)
        base_dist = Normal(mu, std)
        dist = TransformedDistribution(base_dist, TanhTransform())
        
        # [FIX] 关键修复点：对从经验池中采样的动作进行裁剪，以避免 log_prob 计算中的数值问题。
        # 这个裁剪后的动作必须同时用于计算 logp 和 logp_old。
        epsilon = 1e-6
        clipped_act_t = torch.clamp(act_t, -1.0 + epsilon, 1.0 - epsilon)
            
        # 使用裁剪后的动作计算新策略的 log_prob
        logp = dist.log_prob(clipped_act_t).sum(axis=-1)
        
        # 使用同样的裁剪后的动作计算旧策略的 log_prob
        with torch.no_grad():
            std_old = torch.exp(log_std_old_t)
            base_dist_old = Normal(mu_old_t, std_old)
            dist_old = TransformedDistribution(base_dist_old, TanhTransform())
            logp_old = dist_old.log_prob(clipped_act_t).sum(axis=-1)
            
        ratio = torch.exp(logp - logp_old)
        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_t
        policy_loss = -torch.mean(torch.min(surr1, surr2))
        value_loss = VF_COEF * torch.mean((value - v_targ_t)**2)
        ent_loss = -ENT_COEF * torch.mean(base_dist.entropy().sum(axis=-1))
        loss = policy_loss + value_loss + ent_loss
        
        self.model.backward(loss)
        self.model.step()
        updated = self.model.is_gradient_accumulation_boundary()
        
        current_lr = self.model.get_lr()[0] if self.model.get_lr() else 0.0
        
        return loss.item(), policy_loss.item(), value_loss.item(), ent_loss.item(), current_lr, updated
    

# ================================================================
# 5. 主逻辑
# ================================================================
def main():
    # 检查模型路径是否存在
    if not os.path.exists(MODEL_NAME):
        print(f"错误: 模型路径 '{MODEL_NAME}' 不存在。请更新 'MODEL_NAME' 变量。")
        return
        
    os.environ["RAY_DEDUP_LOGS"] = "0"
    ray.init(ignore_reinit_error=True, _temp_dir='/dev/shm')

    log_dir = "runs/MetaWorld/DS_PPO_QwenVL_MultiTask_bs4096_" + str(int(time.time()))
    writer = SummaryWriter(log_dir)
    stats_actor = StatsActor.remote(window_size=MOVING_AVG_WINDOW)
    print(f"TensorBoard 日志将保存在: {log_dir}")

    print("--- 步骤 1: 创建 Actors ---")
    replay_buffers = [ReplayBufferActor.remote(capacity=REPLAY_CAPACITY) for _ in range(NUM_TRAINER_GPUS)]
    trainer_group = [TrainerActor.remote(rank=i, world_size=NUM_TRAINER_GPUS, replay_buffer=replay_buffers[i]) for i in range(NUM_TRAINER_GPUS)]
    inference_pool = [InferenceActor.remote(actor_id=i) for i in range(NUM_INFERENCE_ACTORS)]
    rollout_workers = [
        RolloutWorkerActor.remote(
            inference_pool[i % NUM_INFERENCE_ACTORS], 
            replay_buffers[i % NUM_TRAINER_GPUS],
            i, 
            stats_actor
        ) for i in range(NUM_ROLLOUT_WORKERS)
    ]
    
    print("\n--- 步骤 2: 建立独立的 DeepSpeed 训练组 ---")
    trainer_master_addr = ray.get(trainer_group[0].get_node_ip.remote())
    train_setup_tasks = [actor.setup_deepspeed_group.remote(trainer_master_addr, TRAIN_GROUP_PORT) for actor in trainer_group]
    ray.get(train_setup_tasks)
    print("DeepSpeed 训练组建立完成。")

    print(f"\n--- 步骤 3: 建立共享广播组 ({BROADCAST_GROUP_NAME}) ---")
    broadcast_participants = [trainer_group[0]] + inference_pool
    broadcast_group_world_size = len(broadcast_participants)
    broadcast_master_addr = ray.get(trainer_group[0].get_node_ip.remote())
    broadcast_setup_tasks = [
        actor.setup_broadcast_group.remote(
            master_addr=broadcast_master_addr, master_port=BROADCAST_GROUP_PORT,
            group_name=BROADCAST_GROUP_NAME, group_world_size=broadcast_group_world_size,
            my_rank_in_group=rank) for rank, actor in enumerate(broadcast_participants)
    ]
    ray.get(broadcast_setup_tasks)
    print("共享广播组建立完成。")

    print("\n--- 步骤 4: 启动 Rollout Workers 进行数据收集 ---")
    for w in rollout_workers:
        w.run.remote()
    
    print("\n--- 步骤 5: 等待远程经验池填充初始数据 ---")
    # VLM模型需要更小的批次，所以等待条件也相应调整
    min_buffer_size_for_start = TRAIN_BATCH_SIZE
    while not all(size >= min_buffer_size_for_start for size in ray.get([rb.size.remote() for rb in replay_buffers])):
        sizes = ray.get([rb.size.remote() for rb in replay_buffers])
        print(f"等待所有经验池填充初始数据 (目标: {min_buffer_size_for_start})... (当前大小: {sizes})")
        time.sleep(5)
    print("远程经验池已准备好，训练器将按需获取数据。")
    
    print("\n--- 步骤 6: 开始主训练与同步循环 ---")
    start_time = time.time()
    last_log_time = time.time()
    
    for i in range(TRAIN_ITERS):
        results = []
        while True:
            train_tasks = [trainer.train_step.remote() for trainer in trainer_group]
            result = ray.get(train_tasks)
            _, _, _, _, _, updated = result[0]
            results.extend(result)
            if updated:
                break
        broadcast_task = trainer_group[0].broadcast_weights.remote(BROADCAST_GROUP_NAME)
        receive_tasks = [inf.receive_and_update_weights.remote(BROADCAST_GROUP_NAME) for inf in inference_pool]
        ray.get([broadcast_task] + receive_tasks)
        
        current_time = time.time()
        if current_time - last_log_time > LOG_INTERVAL_SECONDS:
            # 获取包含所有环境统计数据的字典
            all_stats = ray.get(stats_actor.get_stats.remote())
            
            # 从 '_global_' 键中提取全局统计数据用于打印
            global_stats = all_stats.pop("_global_")
            avg_return = global_stats["avg_return"]
            avg_ep_len = global_stats["avg_ep_len"]
            total_episodes = global_stats["total_episodes_processed"]
            avg_step_time = global_stats["avg_step_time"]
            
            total_losses, p_losses, v_losses, e_losses, lrs, _ = zip(*results)
            current_lr = lrs[0]
            
            elapsed_time = current_time - start_time
            total_buffer_size = sum(ray.get([rb.size.remote() for rb in replay_buffers]))

            print(f"迭代 {i+1}/{TRAIN_ITERS} | 时间: {elapsed_time:.1f}s | "
                  f"全局平均奖励: {avg_return:.2f} | "
                  f"全局平均幕长: {avg_ep_len:.1f} | "
                  f"value loss: {np.mean(v_losses):.4f} | "
                  f"学习率: {current_lr:.7f} | "
                  f"经验池总大小: {total_buffer_size:,} | "
                  f"Step平均时间: {avg_step_time:.3f}s")
            
            # 记录全局训练指标
            writer.add_scalar('Train/Learning_Rate', current_lr, i)
            writer.add_scalar('Loss/Total', np.mean(total_losses), i)
            writer.add_scalar('Loss/Policy', np.mean(p_losses), i)
            writer.add_scalar('Loss/Value', np.mean(v_losses), i)
            writer.add_scalar('Loss/Entropy', np.mean(e_losses), i)
            
            # 记录全局系统和 rollout 指标
            writer.add_scalar('Rollout/_Global/Average_Return', avg_return, i)
            writer.add_scalar('Rollout/_Global/Average_Episode_Length', avg_ep_len, i)
            writer.add_scalar('System/Replay_Buffer_Size_Total', total_buffer_size, i)
            writer.add_scalar('System/Total_Episodes_Processed', total_episodes, i)
            writer.add_scalar('System/Avg_Step_Time', avg_step_time, i)
            
            # 循环记录每个环境的独立指标
            for env_name, env_stats in all_stats.items():
                # 在TensorBoard中使用类似文件夹的结构来组织
                tag_prefix = f"Rollout/{env_name}"
                writer.add_scalar(f'{tag_prefix}/Average_Return', env_stats['avg_return'], i)
                writer.add_scalar(f'{tag_prefix}/Average_Episode_Length', env_stats['avg_ep_len'], i)
                writer.add_scalar(f'{tag_prefix}/Success_Rate', env_stats['avg_success_rate'], i)
                writer.add_scalar(f'{tag_prefix}/Total_Episodes', env_stats['total_episodes'], i)

            last_log_time = current_time

    print(f"\n成功完成 {TRAIN_ITERS} 次训练与同步循环！")
    writer.close()
    ray.shutdown()


if __name__ == "__main__":
    main()