import os
# 设置临时文件目录，避免磁盘I/O瓶颈
os.environ["TMPDIR"] = "/dev/shm"
# 设置MuJoCo的渲染后端为osmesa，用于无头服务器渲染 (在Mock中非必需，但保留)
os.environ["MUJOCO_GL"] = "osmesa"
# 为了让 Ray 能看到所有可用的 GPU，我们在脚本开头设置。
# Ray 会根据 @ray.remote(num_gpus=1) 的请求来为每个 Actor 分配和隔离 GPU。
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3,5" # 可根据实际情况调整
# 防止 transformers 库的 tokenizer 并行化警告 (在Mock中非必需，但保留)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import random
import asyncio
from collections import deque, defaultdict
from typing import Union, Dict, List, Optional, Tuple
from dataclasses import dataclass

# --- 库引入 ---
import gymnasium
# import metaworld # 在Mock版本中不再直接使用
import numpy as np
from gymnasium.spaces import Box
# from PIL import Image # 在Mock版本中不再需要

import ray
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import Backend
from torch.distributions import Normal, TransformedDistribution, TanhTransform
import deepspeed
from torch.utils.tensorboard import SummaryWriter


# ================================================================
# 0. 超参数 (已为 Mock 仿真和 SimpleNet 调整)
# ================================================================
# --- Mock 环境参数 ---
OBS_SHAPE = (64, 64, 3)   # 图像观测空间 (H, W, C)
ACT_DIM = 4                # 动作空间维度

# --- 分布式系统参数 ---
NUM_TRAINER_GPUS = 2       # Trainer使用GPU数量 (可根据硬件调整)
NUM_INFERENCE_ACTORS = 1   # 推理Actor数量
NUM_ROLLOUT_WORKERS = 8    # 数据收集Worker数量
ROLLOUT_LOCAL_BUF = 128    # 每个Rollout Worker的本地缓冲区大小 (简单模型可增大)
INFERENCE_BATCH = 64       # 推理服务的批处理大小 (简单模型可增大)
INFERENCE_TIMEOUT_MS = 50  # 推理请求的批次级超时时间 (毫秒)
REPLAY_CAPACITY = 50_000   # 每个经验池的容量
TRAIN_BATCH_SIZE = 256     # PPO的训练批次大小 (简单模型可增大)
ACCUMULATION_STEPS = 1     # 梯度累积步数
TRAIN_ITERS = 100000       # 总训练迭代次数

# --- PPO 算法参数 ---
GAMMA = 0.99
LAMBDA = 0.95
LR = 0.0                  # 简单网络可以使用较大的学习率
CLIP_EPS = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01

# --- 奖励归一化参数 ---
REWARD_SCALE = 1.0 # Mock环境奖励范围小，无需缩放

# --- 学习率调度器参数 ---
WARMUP_STEPS = 500 # 线性预热的步数

# --- 日志和统计参数 ---
MOVING_AVG_WINDOW = 100
LOG_INTERVAL_SECONDS = 10

# --- 通信组端口和名称 ---
TRAIN_GROUP_PORT = 29528
BROADCAST_GROUP_NAME = "trainer_to_inference_broadcast"
BROADCAST_GROUP_PORT = 29529
# MODEL_NAME = "/path/to/your/model" # 不再需要模型路径

# ================================================================
# 0.5. 环境封装器与数据类 (MOCK 版本)
# ================================================================

# --- 任务定义保持不变 ---
TASKS = [
    ("reach-v3", "reach the target position"),
    ("push-v3", "push the puck to the goal"),
    ("pick-place-v3", "pick the block and place it at the target"),
    ("door-open-v3", "pull the door handle to open it"),
    ("drawer-open-v3", "pull the drawer handle to open it"),
    ("button-press-v3", "press the green button"),
]

# --- 新增: MockEnv，一个轻量级的模拟环境 ---
class MockEnv(gymnasium.Wrapper):
    """
    一个模拟的 Gymnasium 环境，用于替代 Meta-World。
    它遵循相同的接口，但内部逻辑被简化，无需物理仿真。
    """
    def __init__(self, tasks, shape, **kwargs):
        # 为了初始化父类，我们创建一个虚拟的、不使用的环境
        super().__init__(gymnasium.make("CartPole-v1"))
        
        self.tasks = tasks
        self.observation_space = Box(low=0, high=255, shape=shape, dtype=np.uint8)
        self.action_space = Box(low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32)
        
        self.current_task_idx = 0
        self.current_instruction = ""
        self.current_env_name = ""
        
        self._episode_step = 0
        self._max_episode_steps = 50 # 设定每幕的最大步数

    def reset(self, **kwargs):
        """
        重置环境，随机选择一个新任务，并返回初始观测。
        """
        self._episode_step = 0
        
        # 随机选择一个新任务
        self.current_task_idx = random.randint(0, len(self.tasks) - 1)
        self.current_env_name, self.current_instruction = self.tasks[self.current_task_idx]

        # 生成一个假的图像观测
        obs = np.random.randint(0, 1, size=self.observation_space.shape, dtype=np.uint8)
        
        info = {
            "life_loss": False,
            "env_name": self.current_env_name,
            "instruction": self.current_instruction,
            "success": 0.0 # 初始时任务未成功
        }
        
        return obs, info

    def step(self, action):
        """
        执行一个动作，返回下一步的状态。
        """
        self._episode_step += 1
        
        # 模拟环境动态
        terminated = False
        truncated = self._episode_step >= self._max_episode_steps
        
        # 模拟奖励和成功条件
        # 假设在接近终点时有更高概率成功
        success_prob = self._episode_step / self._max_episode_steps
        is_success = random.random() < success_prob
        
        reward = 0.0
        info = {"life_loss": False, "success": 0.0}

        if is_success and not terminated:
            reward = 1.0
            info["success"] = 1.0
            terminated = True # 成功后结束本幕

        # 生成下一个假的图像观测
        next_obs = np.random.randint(0, 1, size=self.observation_space.shape, dtype=np.uint8)
            
        return next_obs, reward, terminated, truncated, info

    def render(self):
        # 模拟渲染，直接返回一个随机图像
        return np.random.randint(0, 1, size=self.observation_space.shape, dtype=np.uint8)

    def close(self):
        # 无需关闭任何资源
        pass

# --- Experience Dataclass 保持不变 ---
@dataclass
class Experience:
    obs: np.ndarray
    instruction: str
    action: np.ndarray
    advantage: float
    behaviour_mu: np.ndarray
    behaviour_log_std: np.ndarray
    value_target: float

# ================================================================
# 1. 通信模块 (与之前代码相同)
# ================================================================
def init_custom_process_group(
    backend=None, init_method=None, timeout=None, world_size=-1, rank=-1,
    store=None, group_name=None, pg_options=None,):
    from torch.distributed.distributed_c10d import (
        Backend, PrefixStore, _new_process_group_helper, _world,
        default_pg_timeout, rendezvous)
    assert (store is None) or (init_method is None), "Cannot specify both init_method and store."
    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"
    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("nccl")
    if timeout is None:
        from datetime import timedelta
        timeout = timedelta(minutes=30)
    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)
        store = PrefixStore(group_name, store)
    pg_options_param_name = "backend_options" if str(torch.__version__) >= "2.6" else "pg_options"
    pg, _ = _new_process_group_helper(
        world_size, rank, [], backend, store, group_name=group_name,
        **{pg_options_param_name: pg_options}, timeout=timeout)
    if _world:
        _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
    return pg

class GroupManager:
    def __init__(self):
        self._name_group_map = {}
    def create_collective_group(self, backend, world_size, rank, master_addr: str, master_port: int, group_name):
        init_method = f"tcp://{master_addr}:{master_port}"
        pg_handle = init_custom_process_group(
            backend=backend, init_method=init_method, world_size=world_size, rank=rank, group_name=group_name)
        self._name_group_map[group_name] = pg_handle
        return pg_handle
    def is_group_exist(self, group_name):
        return group_name in self._name_group_map
    def get_group_by_name(self, group_name):
        if not self.is_group_exist(group_name):
            print(f"警告: 通信组 '{group_name}' 未初始化。")
            return None
        return self._name_group_map[group_name]

_group_mgr = GroupManager()

def init_collective_group(
    world_size: int, rank: int, master_addr: str, master_port: int,
    backend: Union[str, Backend] = "nccl", group_name: str = "default"):
    global _group_mgr
    if not group_name: raise ValueError(f"group_name '{group_name}' 必须是一个非空字符串。")
    if _group_mgr.is_group_exist(group_name): return
    _group_mgr.create_collective_group(backend, world_size, rank, master_addr, master_port, group_name)

def broadcast(tensor, src_rank: int = 0, group_name: str = "default"):
    group_handle = _group_mgr.get_group_by_name(group_name)
    dist.broadcast(tensor, src=src_rank, group=group_handle)

# ================================================================
# 1.5. 统计模块 (StatsActor) (与之前代码相同)
# ================================================================
@ray.remote
class StatsActor:
    def __init__(self, window_size=MOVING_AVG_WINDOW):
        self.stats = defaultdict(lambda: {
            "episode_returns": deque(maxlen=window_size),
            "step_times": deque(maxlen=window_size),
            "episode_lengths": deque(maxlen=window_size),
            "successes": deque(maxlen=window_size),
            "total_episodes_processed": 0
        })

    def add_episode_return(self, env_name: str, ep_return: float, step_time: float, ep_length: int, success: float):
        env_stats = self.stats[env_name]
        env_stats["episode_returns"].append(ep_return)
        env_stats["step_times"].append(step_time)
        env_stats["episode_lengths"].append(ep_length)
        env_stats["successes"].append(success)
        env_stats["total_episodes_processed"] += 1

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        per_env_stats = {}
        all_returns = []
        all_lengths = []
        all_step_times = []
        all_successes = []
        total_episodes_processed = 0

        for env_name, env_data in self.stats.items():
            total_episodes_processed += env_data["total_episodes_processed"]
            all_returns.extend(env_data["episode_returns"])
            all_lengths.extend(env_data["episode_lengths"])
            all_step_times.extend(env_data["step_times"])
            all_successes.extend(env_data["successes"])
            
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

        per_env_stats["_global_"] = {
            "avg_return": np.mean(all_returns) if all_returns else 0.0,
            "avg_ep_len": np.mean(all_lengths) if all_lengths else 0.0,
            "avg_step_time": np.mean(all_step_times) if all_step_times else 0.0,
            "avg_success_rate": np.mean(all_successes) if all_successes else 0.0,
            "total_episodes_processed": total_episodes_processed,
        }

        return per_env_stats

# ================================================================
# 2. 模型、经验回放、数据收集器 (MOCK 版本)
# ================================================================

# --- 新模型: SimpleNet，一个简单的MLP网络 ---
class SimpleNet(nn.Module):
    """
    一个简单的神经网络，用于替代复杂的视觉语言模型。
    它接收一个扁平化的图像观测，并输出策略和价值。
    """
    def __init__(self, obs_shape: Tuple[int, int, int], act_dim: int):
        super().__init__()
        
        # 计算扁平化后的输入维度
        obs_dim = np.prod(obs_shape)
        
        # 定义共享的网络主体
        self.body = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # 定义策略头 (输出动作均值)
        self.policy_head = nn.Linear(256, act_dim)
        
        # 定义价值头 (输出状态价值)
        self.value_head = nn.Linear(256, 1)
        
        # 定义动作分布的对数标准差，作为可学习的参数
        self.log_std = nn.Parameter(torch.zeros(1, act_dim))

    def forward(self, pixel_values: torch.Tensor, **kwargs):
        """
        前向传播函数。
        
        参数:
            pixel_values (torch.Tensor): 批量的图像观测，形状为 (B, H, W, C)。
            **kwargs: 忽略其他可能的输入 (如指令文本)。
        """
        # 将输入张量转换为正确的浮点类型并扁平化
        # 输入的 pixel_values 可能是 (B, H, W, C)
        x = pixel_values
        batch_size = x.shape[0]
        x = x.view(batch_size, -1) # 扁平化为 (B, H*W*C)
        
        # 通过网络主体
        features = self.body(x)
        
        # 计算动作均值和状态价值
        mu = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        
        # 扩展 log_std 以匹配批次大小
        expanded_log_std = self.log_std.expand(batch_size, -1)
        
        return mu, expanded_log_std, value

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        模拟的 from_pretrained 方法，用于无缝替换原有的加载逻辑。
        它会忽略模型路径，直接初始化一个新的 SimpleNet。
        """
        print(f"--- MOCK MODE ---")
        print(f"忽略模型路径 '{pretrained_model_name_or_path}'。")
        print(f"正在初始化一个全新的 SimpleNet。")
        
        num_labels = kwargs.get('num_labels', ACT_DIM)
        if num_labels is None:
            raise ValueError("必须提供 `num_labels` (即 act_dim) 来初始化 SimpleNet。")
        
        # 创建 SimpleNet 实例
        model = cls(obs_shape=OBS_SHAPE, act_dim=num_labels)
        
        # 检查并应用数据类型，以兼容 DeepSpeed 和 InferenceActor 的设置
        dtype = kwargs.get('torch_dtype', torch.float32)
        if dtype == torch.bfloat16:
            model.to(torch.bfloat16)
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"SimpleNet 模型参数量: {num_params:,}")
        
        return model


@ray.remote
class ReplayBufferActor:
    def __init__(self, capacity=REPLAY_CAPACITY):
        self.buffer = deque(maxlen=capacity)
    def add_batch(self, batch):
        self.buffer.extend(batch)
    def size(self):
        return len(self.buffer)
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            batch = list(self.buffer)
        else:
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
    def __init__(self, infer, replay, wid, stats_actor):
        self.infer, self.replay = infer, replay
        self.stats_actor = stats_actor
        
        # --- 修改: 初始化 Mock 环境 ---
        self.env = MockEnv(tasks=TASKS, shape=OBS_SHAPE)
        
        self.wid = wid
        self.local_buffer = []
        self.current_instruction = None
        self.current_env_name = None 

    def run(self):
        obs, info = self.env.reset(seed=self.wid)
        self.current_instruction = info['instruction']
        self.current_env_name = info['env_name']
        reward_sum = 0.0
        step_count = 0
        time_start = time.time()
        step_count_total = 0
        while True:
            # 传递图像观测和当前任务指令 (指令在SimpleNet中被忽略)
            action, mu, log_std, value = ray.get(self.infer.request.remote(obs, self.current_instruction))
            nxt, r, term, trunc, info = self.env.step(action)
            reward_sum += r
            r *= REWARD_SCALE
            step_count += 1
            self.local_buffer.append((obs, self.current_instruction, action, r, mu, log_std, value))
            obs = nxt
            step_count_total += 1
            if term or trunc:
                step_time = (time.time() - time_start) / step_count_total if step_count_total > 0 else 0
                success = float(info.get('success', 0.0))
                self.stats_actor.add_episode_return.remote(
                    self.current_env_name, reward_sum, step_time, step_count, success
                )
                reward_sum = 0.0
                step_count = 0
                if self.local_buffer:
                    self._process_traj(self.local_buffer, 0.0)
                self.local_buffer.clear()
                obs, info = self.env.reset()
                self.current_instruction = info['instruction']
                self.current_env_name = info['env_name']
                time_start = time.time()
                step_count_total = 0
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
# 3. 融合后的推理器 (InferenceActor) (MOCK 版本)
# ================================================================
@ray.remote(num_gpus=1)
class InferenceActor:
    def __init__(self, actor_id):
        self.actor_id = actor_id
        
        # --- 修改: 加载 SimpleNet 模型 ---
        print(f"InferenceActor {actor_id}: 正在加载 SimpleNet 模型...")
        
        # --- 关键修正 1: 显式存储设备信息 ---
        self.device = "cuda"
        self.model = SimpleNet.from_pretrained(
            "mock_model_path", num_labels=ACT_DIM, torch_dtype=torch.bfloat16,
        ).to(self.device).eval()
        
        self.batch_size = INFERENCE_BATCH
        self.timeout_sec = INFERENCE_TIMEOUT_MS / 1000.0
        self.requests, self.promises = [], []
        self.last_process_time = time.time()
        asyncio.get_event_loop().create_task(self._loop())
        print(f"InferenceActor {self.actor_id} (bf16) 初始化于 GPU: {ray.get_gpu_ids()} (批次超时: {INFERENCE_TIMEOUT_MS}ms)")

    # --- request 接受 obs 和 instruction，但 instruction 会被忽略 ---
    async def request(self, obs, instruction):
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        # 只需传递 obs，instruction 被忽略
        self.requests.append(obs)
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
                
                # --- 修改: 为 SimpleNet 准备输入 ---
                obs_batch_np = np.stack(requests_to_process)
                
                # --- 关键修正 2: 使用 self.device 而不是 self.model.device ---
                obs_batch_t = torch.tensor(obs_batch_np, dtype=torch.bfloat16).to(self.device)
                
                with torch.no_grad():
                    # 模型调用已简化，不再需要 processor 或 text
                    mu, log_std, values = self.model(pixel_values=obs_batch_t)
                    
                    # 转换为 float32 进行后续计算
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

    def setup_broadcast_group(self, master_addr, master_port, group_name, group_world_size, my_rank_in_group):
        init_collective_group(
            world_size=group_world_size, rank=my_rank_in_group, master_addr=master_addr,
            master_port=master_port, group_name=group_name)
        print(f"InferenceActor {self.actor_id}: 已作为 rank {my_rank_in_group} 加入广播组 '{group_name}'。")

    def receive_and_update_weights(self, group_name):
        state_dict = self.model.state_dict()
        for key, value in state_dict.items():
            received_tensor = torch.empty_like(value, device=self.device)
            broadcast(received_tensor, src_rank=0, group_name=group_name)
            state_dict[key] = received_tensor
        self.model.load_state_dict(state_dict)

# ================================================================
# 4. 融合后的训练器 (TrainerActor) (MOCK 版本)
# ================================================================
@ray.remote(num_gpus=1)
class TrainerActor:
    def __init__(self, rank, world_size, replay_buffer):
        self.rank = rank
        self.world_size = world_size
        self.replay_buffer = replay_buffer
        self.model = None
        self.data_dtype = None
        self.training_batch: Optional[Tuple[torch.Tensor, ...]] = None
        self.data_fetching_task = None
        print(f"TrainerActor Rank {self.rank} 初始化于 GPU: {ray.get_gpu_ids()}")

    def get_node_ip(self):
        return ray.util.get_node_ip_address()
    
    async def _data_fetching_loop(self):
        print(f"Trainer {self.rank}: 后台数据准备循环已启动。")
        while True:
            try:
                # 等待缓冲区有足够数据
                while await self.replay_buffer.size.remote() < TRAIN_BATCH_SIZE:
                    await asyncio.sleep(1)
                
                # 异步获取 NumPy 数据 (指令被获取但后续被忽略)
                obs_np, instructions, act_np, adv_np, mu_old_np, log_std_old_np, v_targ_np = await self.replay_buffer.sample.remote(TRAIN_BATCH_SIZE)
                
                # 将数据转换为Tensor并移动到设备
                act_t = torch.tensor(act_np, dtype=torch.float32).to(self.model.device)
                adv_t = torch.tensor(adv_np, dtype=torch.float32).to(self.model.device)
                mu_old_t = torch.tensor(mu_old_np, dtype=torch.float32).to(self.model.device)
                log_std_old_t = torch.tensor(log_std_old_np, dtype=torch.float32).to(self.model.device)
                v_targ_t = torch.tensor(v_targ_np, dtype=torch.float32).to(self.model.device)
                
                # obs_np 保持为 numpy 数组，在训练步骤中再转换为 tensor
                self.training_batch = (obs_np, instructions, act_t, adv_t, mu_old_t, log_std_old_t, v_targ_t)
            except Exception as e:
                print(f"Trainer {self.rank}: 数据采样失败: {e}。将在3秒后重试。")
                await asyncio.sleep(3)

    def setup_deepspeed_group(self, master_addr, master_port):
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        # --- 关键修正 ---
        # 每个Actor进程只看到一个GPU，所以它的本地rank永远是0
        os.environ["LOCAL_RANK"] = "0"
        deepspeed.init_distributed(dist_backend="nccl")

        # --- 修改: 加载 SimpleNet 模型 ---
        print(f"Trainer {self.rank}: 正在加载 SimpleNet 模型...")
        model = SimpleNet.from_pretrained(
            "mock_model_path", # 路径被忽略
            num_labels=ACT_DIM,
            torch_dtype=torch.bfloat16,
        )
        
        ds_config = {
            "train_micro_batch_size_per_gpu": TRAIN_BATCH_SIZE,
            "gradient_accumulation_steps": ACCUMULATION_STEPS,
            "optimizer": {"type": "AdamW", "params": {"lr": LR}},
            "scheduler": {
                "type": "WarmupCosineLR", "params": {
                    "total_num_steps": TRAIN_ITERS, "warmup_num_steps": WARMUP_STEPS,
                    "warmup_type": "linear", "warmup_min_ratio": 0.0, "cos_min_ratio": 0.0,
                }
            },
            "bf16": {"enabled": True},
            "zero_optimization": {
                "stage": 2, "allgather_partitions": True, "allgather_bucket_size": 5e8,
                "reduce_scatter": True, "reduce_bucket_size": 5e8, "overlap_comm": True,
                "contiguous_gradients": True
            },
            "gradient_clipping": 1.0,
        }

        if ds_config.get("bf16", {}).get("enabled", False): self.data_dtype = torch.bfloat16
        else: self.data_dtype = torch.float32

        self.model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)
        print(f"TrainerActor Rank {self.rank}: DeepSpeed 训练组 (ZeRO-2) 初始化完成。")
        self.data_fetching_task = asyncio.get_event_loop().create_task(self._data_fetching_loop())

    def setup_broadcast_group(self, master_addr, master_port, group_name, group_world_size, my_rank_in_group):
        init_collective_group(
            world_size=group_world_size, rank=my_rank_in_group, master_addr=master_addr,
            master_port=master_port, group_name=group_name)
        print(f"TrainerActor Rank {self.rank}: 已作为 rank {my_rank_in_group} 加入广播组 '{group_name}'。")

    async def train_step(self) -> Tuple[float, float, float, float, float, bool]:
        if self.training_batch is None:
            print(f"Trainer {self.rank}: 等待初始数据批次...")
            while self.training_batch is None: await asyncio.sleep(0.2)
            print(f"Trainer {self.rank}: 初始数据已收到，开始训练。")
        
        obs_np, _, act_t, adv_t, mu_old_t, log_std_old_t, v_targ_t = self.training_batch
        
        # --- 修改: 准备 SimpleNet 的输入 ---
        # 移除所有 processor 和 text 相关逻辑
        obs_t = torch.tensor(obs_np).to(self.model.device, self.data_dtype)
        
        # 模型调用简化
        mu, log_std, value = self.model(pixel_values=obs_t)
        
        # 将输出转为 float32 进行损失计算
        mu, log_std, value = mu.float(), log_std.float(), value.float()
        
        std = torch.exp(log_std)
        if mu.isnan().any():
            print(mu)
            raise ValueError("mu包含 NaN 值，请检查输入数据和模型参数。")
        if std.isnan().any():
            print(std)
            raise ValueError("模型输出的标准差包含 NaN 值，请检查输入数据和模型参数。")
        base_dist = Normal(mu, std)
        dist = TransformedDistribution(base_dist, TanhTransform())
        
        epsilon = 1e-6
        clipped_act_t = torch.clamp(act_t, -1.0 + epsilon, 1.0 - epsilon)
        logp = dist.log_prob(clipped_act_t).sum(axis=-1)
        
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
    
    def broadcast_weights(self, group_name):
        with deepspeed.zero.GatheredParameters(self.model.parameters(), modifier_rank=0):
            if self.rank == 0:
                state_dict = self.model.state_dict()
                for key, tensor in state_dict.items():
                    # 确保广播的张量是 bfloat16
                    tensor_gpu = tensor.to(device=self.model.device, dtype=torch.bfloat16)
                    broadcast(tensor_gpu, src_rank=0, group_name=group_name)

# ================================================================
# 5. 主逻辑
# ================================================================
def main():
    os.environ["RAY_DEDUP_LOGS"] = "0"
    ray.init(ignore_reinit_error=True, _temp_dir='/dev/shm')

    log_dir = "runs/MockEnv/DS_PPO_SimpleNet_MultiTask_" + str(int(time.time()))
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
    min_buffer_size_for_start = TRAIN_BATCH_SIZE * NUM_TRAINER_GPUS
    while not all(size >= TRAIN_BATCH_SIZE for size in ray.get([rb.size.remote() for rb in replay_buffers])):
        sizes = ray.get([rb.size.remote() for rb in replay_buffers])
        print(f"等待所有经验池填充数据 (每个池目标: {TRAIN_BATCH_SIZE})... (当前大小: {sizes})")
        time.sleep(5)
    print("远程经验池已准备好，训练器将按需获取数据。")
    
    print("\n--- 步骤 6: 开始主训练与同步循环 ---")
    start_time = time.time()
    last_log_time = time.time()
    
    for i in range(TRAIN_ITERS):
        results = []
        # 循环直到梯度累积完成并更新模型
        while True:
            train_tasks = [trainer.train_step.remote() for trainer in trainer_group]
            result_list = ray.get(train_tasks)
            # 检查任一训练器是否完成了更新 (在ZeRO-2中所有训练器会同步更新)
            updated = any(res[5] for res in result_list)
            results.extend(result_list)
            if updated:
                break

        # 广播更新后的权重
        broadcast_task = trainer_group[0].broadcast_weights.remote(BROADCAST_GROUP_NAME)
        receive_tasks = [inf.receive_and_update_weights.remote(BROADCAST_GROUP_NAME) for inf in inference_pool]
        ray.get([broadcast_task] + receive_tasks)
        
        current_time = time.time()
        if current_time - last_log_time > LOG_INTERVAL_SECONDS:
            all_stats = ray.get(stats_actor.get_stats.remote())
            global_stats = all_stats.pop("_global_")
            avg_return = global_stats["avg_return"]
            avg_ep_len = global_stats["avg_ep_len"]
            total_episodes = global_stats["total_episodes_processed"]
            avg_step_time = global_stats["avg_step_time"]
            avg_success_rate = global_stats["avg_success_rate"]
            
            total_losses, p_losses, v_losses, e_losses, lrs, _ = zip(*results)
            current_lr = lrs[-1]
            
            elapsed_time = current_time - start_time
            total_buffer_size = sum(ray.get([rb.size.remote() for rb in replay_buffers]))

            print(f"迭代 {i+1}/{TRAIN_ITERS} | 时间: {elapsed_time:.1f}s | "
                  f"全局Avg奖励: {avg_return:.2f} | "
                  f"全局Avg幕长: {avg_ep_len:.1f} | "
                  f"全局成功率: {avg_success_rate:.2f} | "
                  f"Value Loss: {np.mean(v_losses):.4f} | "
                  f"LR: {current_lr:.7f} | "
                  f"经验池: {total_buffer_size:,}")
            
            writer.add_scalar('Train/Learning_Rate', current_lr, i)
            writer.add_scalar('Loss/Total', np.mean(total_losses), i)
            writer.add_scalar('Loss/Policy', np.mean(p_losses), i)
            writer.add_scalar('Loss/Value', np.mean(v_losses), i)
            writer.add_scalar('Loss/Entropy', np.mean(e_losses), i)
            
            writer.add_scalar('Rollout/_Global/Average_Return', avg_return, i)
            writer.add_scalar('Rollout/_Global/Average_Episode_Length', avg_ep_len, i)
            writer.add_scalar('Rollout/_Global/Success_Rate', avg_success_rate, i)
            writer.add_scalar('System/Replay_Buffer_Size_Total', total_buffer_size, i)
            writer.add_scalar('System/Total_Episodes_Processed', total_episodes, i)
            writer.add_scalar('System/Avg_Step_Time', avg_step_time, i)
            
            for env_name, env_stats in all_stats.items():
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