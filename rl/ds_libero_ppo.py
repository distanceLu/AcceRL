import os
os.environ["MUJOCO_GL"] = "osmesa"           # 强制软件渲染
os.environ["PYOPENGL_PLATFORM"] = "osmesa"   # 保险起见，给 PyOpenGL 也指明
# 设置临时文件目录，避免磁盘I/O瓶颈
os.environ["TMPDIR"] = "/dev/shm"
# 为了让 Ray 能看到所有可用的 GPU，我们在脚本开头设置。
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6,7"
# 防止 transformers 库的 tokenizer 并行化警告
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import random
import asyncio
from collections import deque, defaultdict
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import math

import numpy as np

import ray
import torch
from torch.distributions import Normal, kl
from torch.distributions.transforms import TanhTransform
import deepspeed
import torch.distributed as distributed # 新增：为了分布式通信
from torch.utils.tensorboard import SummaryWriter

# Libero env 与工具

# OpenVLA 组件与常量
from experiments.robot.openvla_utils import (
    get_processor,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK

from experiments.robot.libero.libero_utils import GenerateConfig

# 替换为你的 ActorCritic 类实现（来自你给的示例）
from rl.actor_critic_model import ActorCritic
from rl.utils import prepare_one_obs
# 训练/推理通信（保持接口不变）
from ds_com import TrainerActorCom, InferenceActorCom

# ================================================================
# 0. 超参数与配置
# ================================================================
# Libero benchmark
BENCHMARK = "libero_spatial"

# 分布式系统参数
NUM_TRAINER_GPUS = 4
NUM_INFERENCE_ACTORS = 1
NUM_ROLLOUT_WORKERS = 40
ROLLOUT_LOCAL_BUF = 64
INFERENCE_BATCH = 8
INFERENCE_TIMEOUT_MS = 300
REPLAY_CAPACITY = 1000
TRAIN_BATCH_SIZE = 20
ACCUMULATION_STEPS = 13
SUPER_BATCH_SIZE = 260
TRAIN_ITERS = 10000

# PPO
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
VF_COEF = 0.5
ENT_COEF = 0.0
KL_COEF = 0.1

# log_std
START_LOG_STD = -2.0
END_LOG_STD = -3.0

# 奖励缩放
REWARD_SCALE = 1.0

# ================================================================
# 学习率调度参数
# ================================================================
VALUE_LR = 1e-4
POLICY_LR = 3e-6
VALUE_WARMUP_STEPS = 500
POLICY_WARMUP_STEPS = 500
POLICY_TRAIN_START_STEP = 0 # 策略网络从第500个 *更新步* 开始训练

# 日志
MOVING_AVG_WINDOW = 1000
LOG_INTERVAL_SECONDS = 10

# 通信组
TRAIN_GROUP_PORT = 64794
BROADCAST_GROUP_NAME = "trainer_to_inference_broadcast"
BROADCAST_GROUP_PORT = 64795

# OpenVLA 加载配置
USE_BF16: bool = True
TORCH_DTYPE = torch.bfloat16 if USE_BF16 else torch.float32
PRETRAINED_CHECKPOINT = "/cpfs01/lcx_workspace/models/openvla-7b-oft-finetuned-libero-spatial-object-goal-10/"
# PRETRAINED_CHECKPOINT = "/cpfs01/liuwei_workspace/openvla_oft_rl/ckpt/finetune_nll_16/openvla-7b-oft-finetuned-libero-spatial-object-goal-10+libero_spatial_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state"

# ================================================================
# 数据结构
# ================================================================
@dataclass
class Experience:
    obs: Dict[str, torch.Tensor]            # prepare_one_obs 的结果（CPU tensors）
    action: np.ndarray                      # 标准化后的动作（tanh 后，范围在 (-1,1)）
    advantage: float
    behaviour_mu: np.ndarray                # 策略均值（对应 action 的 chunk）
    behaviour_log_std: np.ndarray           # 策略对数标准差（对应 action 的 chunk）
    behaviour_value: float

# ================================================================
# 1.5. 统计模块 (StatsActor)
# ================================================================
@ray.remote
class StatsActor:
    def __init__(self, window_size=MOVING_AVG_WINDOW):
        self.stats = defaultdict(lambda: {
            "episode_returns": deque(maxlen=window_size),
            "step_times": deque(maxlen=window_size),
            "episode_lengths": deque(maxlen=window_size),
            "successes": deque(maxlen=window_size),
            "total_episodes_processed": 0,
            "total_env_steps": 0
        })

    def add_episode_return(self, env_name: str, ep_return: float, step_time: float, ep_length: int, success: float):
        env_stats = self.stats[env_name]
        env_stats["episode_returns"].append(ep_return)
        env_stats["step_times"].append(step_time)
        env_stats["episode_lengths"].append(ep_length)
        env_stats["successes"].append(success)
        env_stats["total_episodes_processed"] += 1
        env_stats["total_env_steps"] += ep_length

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        per_env_stats = {}
        all_returns, all_lengths, all_step_times = [], [], []
        total_episodes_processed = 0
        total_env_steps = 0

        for env_name, env_data in self.stats.items():
            total_episodes_processed += env_data["total_episodes_processed"]
            total_env_steps += env_data["total_env_steps"]
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

        per_env_stats["_global_"] = {
            "avg_return": np.mean(all_returns) if all_returns else 0.0,
            "avg_ep_len": np.mean(all_lengths) if all_lengths else 0.0,
            "avg_step_time": np.mean(all_step_times) if all_step_times else 0.0,
            "total_episodes_processed": total_episodes_processed,
            "total_env_steps": total_env_steps
        }
        return per_env_stats

# ================================================================
# 2. 经验回放与 Rollout
# ================================================================
@ray.remote
class ReplayBufferActor:
    def __init__(self, capacity=REPLAY_CAPACITY):
        self.buffer = deque(maxlen=capacity)

    def add_batch(self, batch: List[Experience]):
        self.buffer.extend(batch)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # obs 是 prepare_one_obs 的字典，不能 stack，保持 list 返回
        obs_list = [b.obs for b in batch]
        act = np.stack([b.action for b in batch])  # 标准化动作（tanh 后）
        adv = np.asarray([b.advantage for b in batch], np.float32)
        mu_old = np.stack([b.behaviour_mu for b in batch])
        log_std_old = np.stack([b.behaviour_log_std for b in batch])
        v_old = np.asarray([b.behaviour_value for b in batch], np.float32)
        return obs_list, act, adv, mu_old, log_std_old, v_old


@ray.remote
class RolloutWorkerActor:
    def __init__(self, infer, replay, wid, stats_actor, cfg, benchmark_name=BENCHMARK):
        self.infer, self.replay = infer, replay
        self.stats_actor = stats_actor
        self.cfg = cfg
        # 仅需 processor，Worker 不加载大模型
        self.processor = get_processor(cfg)
        self.benchmark_name = benchmark_name
        from rl.libero_env import LiberoEnvWrapper
        from libero.libero import benchmark

        benchmark_dict = benchmark.get_benchmark_dict()
        if self.benchmark_name not in benchmark_dict:
            err_info = f"基准 '{self.benchmark_name}' 不存在。可用选项: {list(benchmark_dict.keys())}"
            print(err_info, flush=True)  # ray可能不会打印报错信息，所以这里用print及时打印
            raise ValueError(err_info)
        task_suite = benchmark_dict[self.benchmark_name]()
        task_id = int(wid % 10)
        # print(f"RolloutWorker {wid} 正在加载任务: {task_id} ({task_suite.get_task(task_id).name})")
        # task_id = 5
        self.env = LiberoEnvWrapper(
            benchmark_name=self.benchmark_name,
            task_id=task_id,
            image_size=224,
            render_mode="rgb_array",
        )
        self.wid = wid
        self.local_buffer = []
        self.task_description = None
        self.current_env_name = None

    def run(self):
        try:
            obs, info = self.env.reset(seed=self.wid)
            self.task_description = self.env.task_description
            self.current_env_name = self.env.get_name()

            reward_sum = 0.0
            step_count = 0
            time_start = time.time()

            while True:
                # 2) 用 prepare_one_obs 生成单条样本
                inputs_t = prepare_one_obs(self.cfg, self.processor, obs, self.task_description, TORCH_DTYPE)
                inputs_t["step_count"] = torch.tensor([step_count], dtype=torch.long)
                # 3) 发给 InferenceActor：它返回 env 动作（已 unnormalize），以及标准化动作与策略信息
                action_env, action_norm, mu, log_std, value = ray.get(self.infer.request.remote(inputs_t))
                chunk_reward = 0.0
                done = False
                for i in range(len(action_env)):
                    single_action = action_env[i]
                    nxt, r, term, trunc, info = self.env.step(single_action)
                
                    reward_sum += r
                    r_scaled = r * REWARD_SCALE
                    chunk_reward += r_scaled
                
                    step_count += 1
                    if term or trunc:
                        done = True
                        break
                self.local_buffer.append((inputs_t, action_norm, chunk_reward, mu, log_std, value))
                obs = nxt 
                if done:
                    step_time = (time.time() - time_start) / max(step_count, 1)
                    success = float(info.get('is_success', 0.0))  # Libero 用 is_success
                    self.stats_actor.add_episode_return.remote(
                        self.current_env_name, reward_sum, step_time, step_count, success
                    )
                    reward_sum = 0.0
                    step_count = 0
                    if self.local_buffer:
                        self._process_traj(self.local_buffer, 0.0)
                    self.local_buffer.clear()
                    obs, info = self.env.reset()
                    self.task_description = self.env.task_description
                    self.current_env_name = self.env.get_name()
                    time_start = time.time()
                elif len(self.local_buffer) == ROLLOUT_LOCAL_BUF + 1:
                    _, _, _, _, _, bootstrap_val = self.local_buffer[-1]
                    self._process_traj(self.local_buffer[:-1], bootstrap_val)
                    self.local_buffer = [self.local_buffer[-1]]
        except Exception as e:
            import traceback
            print(f"[ERROR] RolloutWorker {self.wid} run() 崩溃: {e}", flush=True)
            traceback.print_exc()
            # 调试期可以选择 re-raise，让 Ray 标记该任务失败
            raise

    def _process_traj(self, traj_segment, bootstrap_val):
        advs = []
        gae = 0.0
        for i in reversed(range(len(traj_segment))):
            _, _, r, _, _, v = traj_segment[i]
            nv = bootstrap_val if i == len(traj_segment) - 1 else traj_segment[i+1][5]
            delta = r + GAMMA * nv - v
            gae = delta + GAMMA * LAMBDA * gae
            advs.append(gae)
        advs.reverse()
        advs_np = np.array(advs, dtype=np.float32)

        batch: List[Experience] = []
        for i, (s, a_norm, _, mu, log_std, v) in enumerate(traj_segment):
            batch.append(
                Experience(
                    obs=s,
                    action=a_norm.astype(np.float32),
                    advantage=float(advs_np[i]),
                    behaviour_mu=mu.astype(np.float32),
                    behaviour_log_std=log_std.astype(np.float32),
                    behaviour_value=float(v),
                )
            )
        self.replay.add_batch.remote(batch)

# ================================================================
# 3. 推理器 (InferenceActor) — 使用 ActorCritic，并仅在此处反归一化动作
# ================================================================
@ray.remote(num_gpus=1)
class InferenceActor(InferenceActorCom):
    def __init__(self, actor_id, cfg):
        super().__init__()
        self.actor_id = actor_id
        # 加载 ActorCritic（包含 VLA）与 processor
        print(f"InferenceActor {actor_id}: 正在加载 OpenVLA ActorCritic...")
        self.model = ActorCritic(cfg, torch_dtype=TORCH_DTYPE)
        # self.model.load_log_std(cfg.pretrained_checkpoint, "latest")
        self.model.cuda()
        self.model.eval()
        self.processor = self.model.processor
        self.cfg = cfg

        self.batch_size = INFERENCE_BATCH
        self.timeout_sec = INFERENCE_TIMEOUT_MS / 1000.0
        self.requests, self.promises = [], []
        self.last_process_time = time.time()

        loop = asyncio.get_event_loop()
        # 保存 task 句柄，防止被 GC；并加回调打印异常
        self._bg_task = loop.create_task(self._loop())
        self._bg_task.add_done_callback(self._on_bg_task_done)
        print(f"InferenceActor {self.actor_id} 初始化于 GPU: {ray.get_gpu_ids()} (批次超时: {INFERENCE_TIMEOUT_MS}ms)")

    def get_model_keys(self):
        if self.model is None:
            print("模型尚未初始化。")
            return {}
        sd = self.model.state_dict()
        # 返回一个小型摘要，便于比较键是否一致，且便于序列化
        res = {k: float(v.abs().sum().item()) for k, v in sd.items()}
        return res

    def _on_bg_task_done(self, task: asyncio.Task):
        try:
            task.result()
        except Exception as e:
            import traceback
            print(f"[ERROR] InferenceActor {self.actor_id} 后台任务异常: {e}", flush=True)
            traceback.print_exc()

    async def request(self, inputs_t: Dict[str, torch.Tensor]):
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        self.requests.append(inputs_t)
        self.promises.append(fut)
        return await fut

    async def _loop(self):
        while True:
            # 周期性检查
            should_process = self.requests and (
                len(self.requests) >= self.batch_size or
                time.time() - self.last_process_time > self.timeout_sec
            )
            if not should_process:
                await asyncio.sleep(0.0005)
                continue

            requests_to_process = self.requests
            promises_to_process = self.promises
            self.requests, self.promises = [], []
            self.last_process_time = time.time()

            try:
                # 准备 batch
                inputs_batch = self.model.prepare_inputs_batch(requests_to_process)
                with torch.inference_mode():
                    # 前向：得到所有 chunk 的动作、mu、log_std、value
                    actions_all, mu_all, log_std_all, value = self.model(inputs_batch)
                # 只用第一个 chunk
                actions_norm = actions_all.to(torch.float32).detach().cpu().numpy()          # (-1,1)
                mu = mu_all.to(torch.float32).detach().cpu().numpy()
                log_std = log_std_all.to(torch.float32).detach().cpu().numpy()
                values = value.to(torch.float32).detach().cpu().numpy()
                # 仅在推理器中将标准化动作转换为环境动作（反归一化）

                actions_env_clip = np.clip(actions_norm, -1, 1)

                actions_env = []
                for i in range(actions_norm.shape[0]):
                    a_env = self.model.vla._unnormalize_actions(actions_env_clip[i], self.cfg.unnorm_key)
                    actions_env.append(a_env.astype(np.float32))
                for i in range(len(promises_to_process)):
                    # 返回：
                    #  - env 动作（反归一化）
                    #  - 标准化动作（用于训练 log_prob）
                    #  - mu/log_std（标准化空间）
                    #  - value 估计
                    promises_to_process[i].set_result((
                        actions_env[i], actions_norm[i], mu[i], log_std[i], values[i]
                    ))
            except Exception as e:
                # 1) 打印详细堆栈
                import traceback
                print(f"[ERROR] InferenceActor {self.actor_id} 批处理失败: {e}", flush=True)
                traceback.print_exc()

                # 2) 把异常传给所有请求者，避免上游永远等待
                for p in promises_to_process:
                    if not p.done():
                        p.set_exception(e)
                # 3) 也可选择 re-raise 让后台任务整体崩溃（若希望 actor 直接失败）：
                raise
    
    def forward_test(self):
        import pickle
        with open("experiments/robot/libero/sample_libero_spatial_observation.pkl", "rb") as file:
            observation = pickle.load(file)
        inputs_t = prepare_one_obs(self.cfg, self.processor, observation, observation['task_description'], TORCH_DTYPE)
        inputs_t["step_count"] = torch.tensor([0], dtype=torch.long)
        inputs_batch = self.model.prepare_inputs_batch([inputs_t])
        with torch.no_grad():
            actions_all, mu_all, log_std_all, value = self.model(inputs_batch)
        return actions_all
    

# ================================================================
# 4. 训练器 (TrainerActor) — 使用 ActorCritic + DeepSpeed
# ================================================================
@ray.remote(num_gpus=1)
class TrainerActor(TrainerActorCom):
    def __init__(self, rank, world_size, replay_buffer, cfg):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.replay_buffer = replay_buffer
        self.cfg = cfg
        self.model = None             # DeepSpeed engine
        self.optimizer = None         # DeepSpeed optimizer
        self.base_model = None        # 原始 PyTorch 模型
        self.data_dtype = None
        # self.training_batch: Optional[Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None
        self.next_ready_batch: Optional[Tuple] = None
        self.data_fetching_task = None
        self.super_batch_size = SUPER_BATCH_SIZE

        # 新增: 用于手动学习率调度的状态
        self.global_step = 0

        print(f"TrainerActor Rank {self.rank} 初始化于 GPU: {ray.get_gpu_ids()}")

    def get_model_keys(self):
        if self.model is None:
            print("模型尚未初始化。请先调用 setup_deepspeed_group()。")
            return {}
        # self.model 是 DeepSpeedEngine，取其 module 的 state_dict 更稳妥
        module = self.model.module if hasattr(self.model, "module") else self.model
        sd = module.state_dict()
        res = {k: float(v.abs().sum().item()) for k, v in sd.items()}
        return res

    def get_node_ip(self):
        return ray.util.get_node_ip_address()

    def setup_deepspeed_group(self, master_addr, master_port):
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["LOCAL_RANK"] = "0"
        deepspeed.init_distributed(dist_backend="nccl")

        print(f"Trainer {self.rank}: 正在加载 OpenVLA ActorCritic...")
        model = ActorCritic(self.cfg, torch_dtype=TORCH_DTYPE)
        # model.load_log_std(self.cfg.pretrained_checkpoint, "latest")
        self.base_model = model


        # 修改: 使用参数分组来配置优化器
        param_groups = self.base_model.get_parameter_groups()
        optimizer_params = [
            {
                "params": pg["params"], 
                "name": pg["name"], 
                # 为每个组设置其峰值学习率
                "lr": POLICY_LR if pg["name"] == "policy" else VALUE_LR
            }
            for pg in param_groups
        ]
        
        ds_config = {
            "train_micro_batch_size_per_gpu": TRAIN_BATCH_SIZE,
            "gradient_accumulation_steps": ACCUMULATION_STEPS,
            "optimizer": {
                "type": "AdamW", 
                "params": {
                    # 此处为空，因为参数和学习率由 `model_parameters` 提供
                    # 且学习率将被手动调度。可以添加如 'betas': [0.9, 0.999] 等
                }
            },
            # 移除scheduler，我们将手动实现调度器
            "bf16": {"enabled": USE_BF16},
            "zero_optimization": {
                "stage": 2,
                "allgather_partitions": True,
                "allgather_bucket_size": 5e8,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8,
                "overlap_comm": True,
                "contiguous_gradients": True
            },
            "gradient_clipping": 1.0,
        }

        if ds_config.get("fp16", {}).get("enabled", False): self.data_dtype = torch.float16
        elif ds_config.get("bf16", {}).get("enabled", False): self.data_dtype = torch.bfloat16
        else: self.data_dtype = torch.float32

        # 修改: DeepSpeed 初始化现在返回优化器实例
        self.model, self.optimizer, _, _ = deepspeed.initialize(
            model=model, 
            config=ds_config,
            # 将构造好的参数组列表传递给这里
            model_parameters=optimizer_params
        )
        print(f"TrainerActor Rank {self.rank}: DeepSpeed 训练组 (ZeRO-2) 初始化完成。")

        # 后台取数
        self.data_fetching_task = asyncio.get_event_loop().create_task(self._data_fetching_loop())

        n_total = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数量: {n_total:,}, 可训练参数量: {n_trainable:,}")

    # 新增: 手动学习率调度器逻辑
    def _get_current_lr(self, current_step: int, peak_lr: float, warmup_steps: int, total_steps: int, start_step: int = 0) -> float:
        """计算给定步骤的学习率，支持延迟启动、线性预热和余弦退火。"""
        if current_step < start_step:
            return 0.0
        
        effective_step = current_step - start_step
        
        # 1. 线性预热
        if effective_step < warmup_steps:
            return peak_lr * (effective_step / warmup_steps)
        
        # 2. 余弦退火
        progress = (effective_step - warmup_steps) / (total_steps - start_step - warmup_steps)
        progress = min(progress, 1.0) # 确保不超调
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        return peak_lr * cosine_decay

    async def _data_fetching_loop(self):
        print(f"Trainer {self.rank}: 后台数据准备循环已启动 (超级批次大小: {self.super_batch_size})。")
        while True:
            try:
                # 如果下一个批次的缓冲区已经满了，就等待，避免内存过度占用
                if self.next_ready_batch is not None:
                    await asyncio.sleep(0.1)
                    continue

                # 等待 ReplayBuffer 中有足够的数据
                while await self.replay_buffer.size.remote() < self.super_batch_size:
                    print(f"Trainer {self.rank} (BG): 等待 ReplayBuffer 填充至 {self.super_batch_size}...")
                    await asyncio.sleep(3)

                # 1. 获取经验 (仍然是 numpy 数组)
                obs_list, act_np, adv_np, mu_old_np, log_std_old_np, v_old_np = \
                    await self.replay_buffer.sample.remote(self.super_batch_size)

                # 准备 batch（右侧 padding + proprio 归一化；放到 ActorCritic 的 device）
                inputs_batch = self.base_model.prepare_inputs_batch(obs_list)

                # 其它张量
                device = next(self.model.parameters()).device
                act_t = torch.tensor(act_np, dtype=torch.float32, device=device)
                adv_t = torch.tensor(adv_np, dtype=torch.float32, device=device)
                mu_old_t = torch.tensor(mu_old_np, dtype=torch.float32, device=device)
                log_std_old_t = torch.tensor(log_std_old_np, dtype=torch.float32, device=device)
                v_old_t = torch.tensor(v_old_np, dtype=torch.float32, device=device)

                # 缓存本轮训练 batch
                self.next_ready_batch = (inputs_batch, act_t, adv_t, mu_old_t, log_std_old_t, v_old_t)

            except Exception as e:
                print(f"Trainer {self.rank}: 数据采样失败: {e}。将在3秒后重试。")
                await asyncio.sleep(3)

    async def run_training_epoch(self) -> Tuple[float, float, float, float, float, Dict[str, float], int, float, float]:
        # 等待后台任务准备好第一个批次 (仅在启动时发生一次)
        if self.next_ready_batch is None:
            print(f"Trainer {self.rank}: 等待初始超级批次...", flush=True)
            while self.next_ready_batch is None:
                await asyncio.sleep(0.2)
            print(f"Trainer {self.rank}: 初始数据已收到，开始第一个训练周期。", flush=True)

        # 根据 self.global_step 线性插值
        progress = min(self.global_step / TRAIN_ITERS, 1.0)
        current_log_std_val = START_LOG_STD + (END_LOG_STD - START_LOG_STD) * progress
        
        # 更新模型中的 log_std 缓冲区
        self.model.module.log_std_param.fill_(current_log_std_val)

        # 手动更新学习率
        current_lrs = {}
        value_lr = self._get_current_lr(self.global_step, VALUE_LR, VALUE_WARMUP_STEPS, TRAIN_ITERS)
        policy_lr = self._get_current_lr(self.global_step, POLICY_LR, POLICY_WARMUP_STEPS, TRAIN_ITERS, start_step=POLICY_TRAIN_START_STEP)
        
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'value':
                param_group['lr'] = value_lr
                current_lrs['value'] = value_lr
            elif param_group['name'] == 'policy':
                param_group['lr'] = policy_lr
                current_lrs['policy'] = policy_lr
        # 2. "原子"地获取当前批次，并触发后台准备下一个批次
        current_batch = self.next_ready_batch
        self.next_ready_batch = None  # 清空，信号后台任务开始工作

        inputs_batch, act_t, adv_t, mu_old_t, log_std_old_t, v_old_t = current_batch
        with torch.no_grad():
            v_targ_t = adv_t + v_old_t
        
        # 3. **关键：在整个超级批次上计算优势的全局统计量**
        # 本地统计
        local_sum = adv_t.sum()
        local_sq_sum = (adv_t * adv_t).sum()
        local_count = torch.tensor([adv_t.numel()], device=adv_t.device, dtype=torch.float32)

        stats_tensor = torch.stack([local_sum, local_sq_sum, local_count.squeeze(0)])
        distributed.all_reduce(stats_tensor, op=distributed.ReduceOp.SUM)

        global_sum, global_sq_sum, global_count = stats_tensor[0], stats_tensor[1], stats_tensor[2]
        global_mean = global_sum / torch.clamp(global_count, min=1.0)
        global_var = torch.clamp(global_sq_sum / torch.clamp(global_count, min=1.0) - global_mean * global_mean, min=1e-12)
        global_std = torch.sqrt(global_var)

        # 记录本周期的所有损失和指标
        epoch_losses, epoch_p_losses, epoch_v_losses, epoch_e_losses, epoch_kl_losses = [], [], [], [], [] 
        epoch_ent, epoch_kl_divs = [], []
        
        num_updates_in_epoch = self.super_batch_size // TRAIN_BATCH_SIZE
        
        # 4. 内循环：对小批次进行梯度更新
        for i in range(num_updates_in_epoch):
            start = i * TRAIN_BATCH_SIZE
            end = start + TRAIN_BATCH_SIZE

            # 切分小批次
            mini_inputs = {k: v[start:end] for k, v in inputs_batch.items()}
            mini_act = act_t[start:end]
            mini_adv = adv_t[start:end]
            mini_mu_old = mu_old_t[start:end]
            mini_log_std = log_std_old_t[start:end]
            mini_v_old = v_old_t[start:end]
            mini_v_targ = v_targ_t[start:end]
            
            # 使用全局统计量进行归一化
            normalized_adv = (mini_adv - global_mean) / (global_std + 1e-8)
            
            # 前向
            actions_all, mu_all, log_std_all, value = self.model(mini_inputs)
            # 仅用第一个 chunk
            mu = mu_all.to(torch.float32)
            log_std = log_std_all.to(torch.float32)
            value = value.to(torch.float32)

            # 3. 根据 global_step 计算损失
            # value_loss = VF_COEF * torch.mean((value - mini_v_targ) ** 2)
            value_loss_unclipped = (value - mini_v_targ) ** 2

            value_clipped = mini_v_old + torch.clamp(value - mini_v_old, -CLIP_EPS, CLIP_EPS)
            value_loss_clipped = (value_clipped - mini_v_targ) ** 2
            
            value_loss = VF_COEF * 0.5 * torch.mean(torch.maximum(value_loss_unclipped, value_loss_clipped))

            if self.global_step < POLICY_TRAIN_START_STEP:
                # 阶段一: 只训练 value head
                loss = value_loss
                policy_loss = torch.tensor(0.0, device=loss.device)
                ent_loss = torch.tensor(0.0, device=loss.device)
                kl_loss = torch.tensor(0.0, device=loss.device) 
                kl_div = 0.0
            else:
                # 阶段二: 训练所有组件
                std = torch.exp(log_std)
                base_dist = Normal(mu, std)
                logp = base_dist.log_prob(mini_act)

                with torch.no_grad():
                    std_old = torch.exp(mini_log_std)
                    base_dist_old = Normal(mini_mu_old, std_old)
                    logp_old = base_dist_old.log_prob(mini_act)
                
                # KL 散度计算 (作为指标和损失项)
                kl_div_tensor = kl.kl_divergence(base_dist_old, base_dist)
                kl_div = torch.mean(kl_div_tensor).item() # 作为指标
                kl_loss = KL_COEF * torch.mean(kl_div_tensor) # 作为损失

                ratio = torch.exp(logp - logp_old)
                adv_unsqueezed = normalized_adv.unsqueeze(-1).unsqueeze(-1)
                surr1 = ratio * adv_unsqueezed
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_unsqueezed
                policy_loss = -torch.mean(torch.min(surr1, surr2))
                ent = torch.mean(base_dist.entropy())
                ent_loss = -ENT_COEF * ent
                
                loss = policy_loss + value_loss + ent_loss + kl_loss

            self.model.backward(loss)
            self.model.step()
            epoch_losses.append(loss.item())
            epoch_p_losses.append(policy_loss.item())
            epoch_v_losses.append(value_loss.item())
            epoch_e_losses.append(ent_loss.item())
            epoch_kl_losses.append(kl_loss.item())
            epoch_ent.append(ent.item())
            epoch_kl_divs.append(kl_div)
            if self.model.is_gradient_accumulation_boundary():
                self.global_step += 1
        avg_loss = np.mean(epoch_losses)
        avg_p_loss = np.mean(epoch_p_losses)
        avg_v_loss = np.mean(epoch_v_losses)
        avg_e_loss = np.mean(epoch_e_losses)
        avg_kl_loss = np.mean(epoch_kl_losses)
        avg_ent = np.mean(epoch_ent)
        avg_kl_div = np.mean(epoch_kl_divs)

        return avg_loss, avg_p_loss, avg_v_loss, avg_e_loss, avg_kl_loss, current_lrs, self.global_step, avg_ent, avg_kl_div


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
        device="cuda",
        use_lora=True,
        lora_rank=32,
        lora_dropout=0.0,
    )
    return cfg


def main():
    if not os.path.exists(PRETRAINED_CHECKPOINT):
        print(f"错误: OpenVLA checkpoint 路径 '{PRETRAINED_CHECKPOINT}' 不存在。请更新 PRETRAINED_CHECKPOINT。")
        return

    os.environ["RAY_DEDUP_LOGS"] = "0"
    ray.init(ignore_reinit_error=True, _temp_dir='/dev/shm')

    log_dir = f"runs/Libero/{BENCHMARK}/OpenVLA_DS_PPO_spatial_step_emb_{int(time.time())}"
    writer = SummaryWriter(log_dir)
    stats_actor = StatsActor.remote(window_size=MOVING_AVG_WINDOW)
    print(f"TensorBoard 日志将保存在: {log_dir}")

    # 构建 OpenVLA 配置
    cfg = build_openvla_cfg()

    print("--- 步骤 1: 创建 Actors ---")
    replay_buffers = [ReplayBufferActor.remote(capacity=REPLAY_CAPACITY) for _ in range(NUM_TRAINER_GPUS)]
    trainer_group = [
        TrainerActor.remote(rank=i, world_size=NUM_TRAINER_GPUS, replay_buffer=replay_buffers[i], cfg=cfg)
        for i in range(NUM_TRAINER_GPUS)
    ]
    inference_pool = [InferenceActor.remote(actor_id=i, cfg=cfg) for i in range(NUM_INFERENCE_ACTORS)]
    rollout_workers = [
        RolloutWorkerActor.remote(
            inference_pool[i % NUM_INFERENCE_ACTORS],
            replay_buffers[i % NUM_TRAINER_GPUS],
            i,
            stats_actor,
            cfg,
        )
        for i in range(NUM_ROLLOUT_WORKERS)
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
    forward_test_tasks = [inf.forward_test.remote() for inf in inference_pool]
    ray.get(forward_test_tasks)
    print("推理器前向测试完成。before broadcast")
    broadcast_task = trainer_group[0].broadcast_weights.remote(BROADCAST_GROUP_NAME)
    receive_tasks = [inf.receive_and_update_weights.remote(BROADCAST_GROUP_NAME) for inf in inference_pool]
    ray.get([broadcast_task] + receive_tasks)
    print("初始权重已广播到所有推理器。")
    forward_test_tasks = [inf.forward_test.remote() for inf in inference_pool]
    ray.get(forward_test_tasks)
    print("推理器前向测试完成。after broadcast")

    print("\n--- 步骤 4: 启动 Rollout Workers 进行数据收集 ---")
    for w in rollout_workers:
        w.run.remote()

    print("\n--- 步骤 5: 等待远程经验池填充初始数据 ---")
    min_buffer_size_for_start = SUPER_BATCH_SIZE
    assert min_buffer_size_for_start < REPLAY_CAPACITY
    while not all(size >= min_buffer_size_for_start for size in ray.get([rb.size.remote() for rb in replay_buffers])):
        sizes = ray.get([rb.size.remote() for rb in replay_buffers])
        print(f"等待所有经验池填充初始数据 (目标: {min_buffer_size_for_start})... (当前大小: {sizes})")
        time.sleep(5)
    print("远程经验池已准备好，训练器将按需获取数据。")

    print("\n--- 步骤 6: 开始主训练与同步循环 ---")
    start_time = time.time()
    last_log_time = time.time()
    last_log_global_step = 0
    global_step = 0
    while global_step < TRAIN_ITERS:
        # 每个 trainer 独立运行一个训练周期
        train_tasks = [trainer.run_training_epoch.remote() for trainer in trainer_group]
        results = ray.get(train_tasks)
        
        # 从结果中解包 (现在返回的是整个周期的平均值)
        # 注意：global_step 现在由 trainer 内部管理和返回
        _, _, _, _, _, _, global_step, _, _ = results[0]

        # 广播权重到推理器
        broadcast_task = trainer_group[0].broadcast_weights.remote(BROADCAST_GROUP_NAME)
        receive_tasks = [inf.receive_and_update_weights.remote(BROADCAST_GROUP_NAME) for inf in inference_pool]
        ray.get([broadcast_task] + receive_tasks)

        current_time = time.time()
        if current_time - last_log_time > LOG_INTERVAL_SECONDS:
            all_stats = ray.get(stats_actor.get_stats.remote())

            elapsed_log_time = current_time - last_log_time
            steps_since_last_log = global_step - last_log_global_step
            training_speed_steps_per_sec = steps_since_last_log / elapsed_log_time if elapsed_log_time > 0 else 0.0

            global_stats = all_stats.pop("_global_")
            avg_return = global_stats["avg_return"]
            avg_ep_len = global_stats["avg_ep_len"]
            total_episodes = global_stats["total_episodes_processed"]
            total_env_steps = global_stats["total_env_steps"]
            avg_step_time = global_stats["avg_step_time"]

            total_losses, p_losses, v_losses, e_losses, kl_losses, lrs_list, _, ents, avg_kl_divs = zip(*results)
            # lrs_list 是一个元组，每个元素是一个字典: ({'value': lr_v, 'policy': lr_p}, ...)
            # 我们从第一个 worker 的结果中获取学习率
            current_lrs = lrs_list[0]

            # === 为日志记录重新计算当前的 log_std ===
            progress = min(global_step / TRAIN_ITERS, 1.0)
            current_log_std = START_LOG_STD + (END_LOG_STD - (START_LOG_STD)) * progress

            elapsed_time = current_time - start_time
            total_buffer_size = sum(ray.get([rb.size.remote() for rb in replay_buffers]))

            print(f"更新步 {global_step}/{TRAIN_ITERS} | 时间: {elapsed_time:.1f}s | "
                  f"全局平均奖励: {avg_return:.2f} | "
                  f"全局平均幕长: {avg_ep_len:.1f} | "
                  f"value loss: {np.mean(v_losses):.4f} | "
                  f"LR(V/P): {current_lrs['value']:.7f}/{current_lrs['policy']:.7f} | "
                  f"Log_Std: {current_log_std:.4f} | "
                  f"Episodes数量: {total_episodes:,} | "
                  f"Step平均时间: {avg_step_time:.3f}s")

            writer.add_scalar('Train/Log_Std', current_log_std, global_step)
            writer.add_scalar('Train/Learning_Rate/Value', current_lrs['value'], global_step)
            writer.add_scalar('Train/Learning_Rate/Policy', current_lrs['policy'], global_step)
            
            writer.add_scalar('Loss/Total', np.mean(total_losses), global_step)
            writer.add_scalar('Loss/Policy', np.mean(p_losses), global_step)
            writer.add_scalar('Loss/Value', np.mean(v_losses), global_step)
            writer.add_scalar('Loss/Entropy', np.mean(e_losses), global_step)
            writer.add_scalar('Loss/KL', np.mean(kl_losses), global_step)

            writer.add_scalar('Metrics/Entropy', np.mean(ents), global_step)
            writer.add_scalar('Metrics/KL_Divergence', np.mean(avg_kl_divs), global_step)
            writer.add_scalar('Metrics/Training_Speed_Steps_per_Sec', training_speed_steps_per_sec, global_step)

            writer.add_scalar('Rollout/_Global/Average_Return', avg_return, global_step)
            writer.add_scalar('Rollout/_Global/Average_Episode_Length', avg_ep_len, global_step)
            writer.add_scalar('System/Replay_Buffer_Size_Total', total_buffer_size, global_step)
            writer.add_scalar('System/Total_Episodes_Processed', total_episodes, global_step)
            writer.add_scalar('System/Total_Env_Steps', total_env_steps, global_step)
            writer.add_scalar('System/Avg_Step_Time', avg_step_time, global_step)

            for env_name, env_stats in all_stats.items():
                tag_prefix = f"Rollout/{env_name}"
                writer.add_scalar(f'{tag_prefix}/Average_Return', env_stats['avg_return'], global_step)
                writer.add_scalar(f'{tag_prefix}/Average_Episode_Length', env_stats['avg_ep_len'], global_step)
                writer.add_scalar(f'{tag_prefix}/Success_Rate', env_stats['avg_success_rate'], global_step)
                writer.add_scalar(f'{tag_prefix}/Total_Episodes', env_stats['total_episodes'], global_step)

            last_log_time = current_time
            last_log_global_step = global_step

    print(f"\n成功完成 {TRAIN_ITERS} 次训练与同步循环！")
    writer.close()
    ray.shutdown()


if __name__ == "__main__":
    main()