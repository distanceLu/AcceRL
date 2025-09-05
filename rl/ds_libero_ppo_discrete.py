import os
os.environ["MUJOCO_GL"] = "osmesa"           # 强制软件渲染
os.environ["PYOPENGL_PLATFORM"] = "osmesa"   # 保险起见，给 PyOpenGL 也指明
# 设置临时文件目录，避免磁盘I/O瓶颈
os.environ["TMPDIR"] = "/dev/shm"
# 为了让 Ray 能看到所有可用的 GPU，我们在脚本开头设置。
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"
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
import torch.distributions
import deepspeed
import torch.distributed as distributed 
from torch.utils.tensorboard import SummaryWriter

# OpenVLA 组件与常量
from experiments.robot.openvla_utils import (
    get_processor,
)

from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM
from experiments.robot.libero.libero_utils import GenerateConfig
from rl.actor_critic_model_discrete import ActorCritic
from rl.utils import prepare_one_obs
# 训练/推理通信（保持接口不变）
from ds_com import TrainerActorCom, InferenceActorCom

# ================================================================
# 0. 超参数与配置
# ================================================================
# Libero benchmark
BENCHMARK = "libero_spatial"   

# 分布式系统参数
NUM_TRAINER_GPUS = 2
NUM_INFERENCE_ACTORS = 1
NUM_ROLLOUT_WORKERS = 20
ROLLOUT_LOCAL_BUF = 64
INFERENCE_BATCH = 8
INFERENCE_TIMEOUT_MS = 300
REPLAY_CAPACITY = 1000
TRAIN_BATCH_SIZE = 16
ACCUMULATION_STEPS = 32
SUPER_BATCH_SIZE = 512
TRAIN_ITERS = 100000

# PPO
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01

# 奖励缩放
REWARD_SCALE = 1.0

# ================================================================
# 学习率调度参数
# ================================================================
VALUE_LR = 1e-4
POLICY_LR = 1e-5
VALUE_WARMUP_STEPS = 500
POLICY_WARMUP_STEPS = 500
POLICY_TRAIN_START_STEP = 0 # 策略网络从第500个 *更新步* 开始训练

# 日志
MOVING_AVG_WINDOW = 1000
LOG_INTERVAL_SECONDS = 10

# 通信组
TRAIN_GROUP_PORT = 29531
BROADCAST_GROUP_NAME = "trainer_to_inference_broadcast"
BROADCAST_GROUP_PORT = 29532

# OpenVLA 加载配置
USE_BF16: bool = True
TORCH_DTYPE = torch.bfloat16 if USE_BF16 else torch.float32
PRETRAINED_CHECKPOINT = "/cpfs01/jinshiji_workspace/openvla_oft_rl/runs/openvla-7b-oft-finetuned-2_gpus_batch_size_16_100_000"

# ================================================================
# 数据结构 更新经验数据结
# ================================================================
@dataclass
class Experience:
    obs: Dict[str, torch.Tensor]            # prepare_one_obs 的结果（CPU tensors）
    action_token: np.ndarray                # 采样的离散动作 token (shape: [ACTION_DIM,])
    advantage: float
    behaviour_logits: np.ndarray            # 行为策略的 logits (shape: [ACTION_DIM, VOCAB_SIZE])
    value_target: float

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
        all_returns, all_lengths, all_step_times = [], [], []
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

        per_env_stats["_global_"] = {
            "avg_return": np.mean(all_returns) if all_returns else 0.0,
            "avg_ep_len": np.mean(all_lengths) if all_lengths else 0.0,
            "avg_step_time": np.mean(all_step_times) if all_step_times else 0.0,
            "total_episodes_processed": total_episodes_processed,
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
    
    # ############################# 核心修改：更新采样逻辑 ##############################
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # obs 是 prepare_one_obs 的字典，不能 stack，保持 list 返回
        obs_list = [b.obs for b in batch]
        action_token = np.stack([b.action_token for b in batch])
        adv = np.asarray([b.advantage for b in batch], np.float32)
        logits_old = np.stack([b.behaviour_logits for b in batch])
        v_targ = np.asarray([b.value_target for b in batch], np.float32)
        return obs_list, action_token, adv, logits_old, v_targ
    # ################################################################################


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

        task_id = 5
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
            current_seed = int(time.time() * 1000) + self.wid + os.getpid()
            obs, info = self.env.reset(seed=current_seed)

            self.task_description = self.env.task_description
            self.current_env_name = self.env.get_name()

            reward_sum = 0.0
            time_start = time.time()
            step_count_total = 0

            while True:
                inputs_t = prepare_one_obs(self.cfg, self.processor, obs, self.task_description, TORCH_DTYPE)
                
                action_env, action_token, logits, value = ray.get(self.infer.request.remote(inputs_t))
                chunk_reward = 0.0
                # 假设action_env的形状是(8, action_dim)，循环执行每个动作
                done = False
                for i in range(len(action_env)):
                    single_action = action_env[i]  # 取出第i个动作
                    nxt, r, term, trunc, info = self.env.step(single_action)
                
                    reward_sum += r
                    r_scaled = r * REWARD_SCALE
                    chunk_reward += r_scaled
                
                    step_count_total += 1
                    if term or trunc:
                        done = True
                        break
                self.local_buffer.append((inputs_t, action_token, chunk_reward, logits, value))
                obs = nxt

                if done:
                    step_time = (time.time() - time_start) / max(step_count_total, 1)
                    success = float(info.get('is_success', 0.0))
                    self.stats_actor.add_episode_return.remote(
                        self.current_env_name, reward_sum, step_time, step_count_total, success
                    )
                    reward_sum = 0.0
                    if self.local_buffer:
                        self._process_traj(self.local_buffer, 0.0)
                    self.local_buffer.clear()
                    current_seed = int(time.time() * 1000) + self.wid + os.getpid()
                    obs, info = self.env.reset(seed=current_seed)
                    self.task_description = self.env.task_description
                    self.current_env_name = self.env.get_name()
                    time_start = time.time()
                    step_count_total = 0
                elif len(self.local_buffer) == ROLLOUT_LOCAL_BUF + 1:
                    _, _, _, _, bootstrap_val = self.local_buffer[-1]
                    self._process_traj(self.local_buffer[:-1], bootstrap_val)
                    self.local_buffer = [self.local_buffer[-1]]
        except Exception as e:
            import traceback
            print(f"[ERROR] RolloutWorker {self.wid} run() 崩溃: {e}", flush=True)
            traceback.print_exc()
            raise

    def _process_traj(self, traj_segment, bootstrap_val):
        rets, advs = [], []
        gae = 0.0
        for i in reversed(range(len(traj_segment))):
            _, _, r, _, v = traj_segment[i]
            nv = bootstrap_val if i == len(traj_segment) - 1 else traj_segment[i+1][4]
            delta = r + GAMMA * nv - v
            gae = delta + GAMMA * LAMBDA * gae
            advs.append(gae)
            rets.append(gae + v)
        advs.reverse(); rets.reverse()
        advs_np = np.array(advs, dtype=np.float32)

        batch: List[Experience] = []
        # ############################# 核心修改：创建新的 Experience 对象 ##############################
        for i, (s, a_token, _, logits, _) in enumerate(traj_segment):
            batch.append(
                Experience(
                    obs=s,
                    action_token=a_token.astype(np.int64), # token 是整数
                    advantage=float(advs_np[i]),
                    behaviour_logits=logits.astype(np.float32),
                    value_target=float(rets[i]),
                )
            )
        # ##########################################################################################
        self.replay.add_batch.remote(batch)

# ================================================================
# 3. 推理器 (InferenceActor)
# ================================================================
@ray.remote(num_gpus=1)
class InferenceActor(InferenceActorCom):
    def __init__(self, actor_id, cfg):
        super().__init__()
        self.actor_id = actor_id
        print(f"InferenceActor {actor_id}: 正在加载 OpenVLA ActorCritic...")
        self.model = ActorCritic(cfg, torch_dtype=TORCH_DTYPE)
        self.model.cuda()
        self.model.eval()
        self.processor = self.model.processor
        self.cfg = cfg

        self.batch_size = INFERENCE_BATCH
        self.timeout_sec = INFERENCE_TIMEOUT_MS / 1000.0
        self.requests, self.promises = [], []
        self.last_process_time = time.time()

        loop = asyncio.get_event_loop()
        self._bg_task = loop.create_task(self._loop())
        self._bg_task.add_done_callback(self._on_bg_task_done)
        print(f"InferenceActor {self.actor_id} 初始化于 GPU: {ray.get_gpu_ids()} (批次超时: {INFERENCE_TIMEOUT_MS}ms)")

    def get_model_keys(self):
        if self.model is None:
            print("模型尚未初始化。")
            return {}
        sd = self.model.state_dict()
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
                inputs_batch = self.model.prepare_inputs_batch(requests_to_process)
                with torch.inference_mode():
                    # ############################# 核心修改：使用新的模型接口 ##############################
                    # 1. 前向传播获取 logits 和 value
                    action_logits, value = self.model(inputs_batch)

                    # 2. 后处理以采样动作 tokens 和对应的归一化连续动作
                    _, action_tokens_all, normalized_actions_all = self.model.post_process(action_logits)
                    
                    # action_tokens_all 的形状是 (B, NUM_ACTIONS_CHUNK * ACTION_DIM)
                    action_tokens = action_tokens_all.view(
                        -1, NUM_ACTIONS_CHUNK, ACTION_DIM
                    ).cpu().numpy()

                    # action_logits 的形状是 (B, NUM_ACTIONS_CHUNK * ACTION_DIM, VocabSize)
                    logits_for_first_action = action_logits.view(
                        -1, NUM_ACTIONS_CHUNK, ACTION_DIM, action_logits.shape[-1]
                    ).float().cpu().numpy()
                    
                    values = value.to(torch.float32).cpu().numpy()
                    # ####################################################################################

                # 将标准化动作转换为环境动作
                actions_env = []
                for i in range(normalized_actions_all.shape[0]):
                    a_env = self.model.vla._unnormalize_actions(normalized_actions_all[i], self.cfg.unnorm_key)
                    actions_env.append(a_env.astype(np.float32))

                for i in range(len(promises_to_process)):
                    # ############################# 核心修改：返回新的数据 ##############################
                    promises_to_process[i].set_result((
                        actions_env[i],           # 反归一化的环境动作
                        action_tokens[i],         # 离散动作 token
                        logits_for_first_action[i], # 对应的 logits
                        values[i]                 # 价值估计
                    ))
                    # ###############################################################################
            except Exception as e:
                import traceback
                print(f"[ERROR] InferenceActor {self.actor_id} 批处理失败: {e}", flush=True)
                traceback.print_exc()
                for p in promises_to_process:
                    if not p.done():
                        p.set_exception(e)
                raise
    
    def forward_test(self):
        import pickle
        with open("experiments/robot/libero/sample_libero_spatial_observation.pkl", "rb") as file:
            observation = pickle.load(file)
        inputs_t = prepare_one_obs(self.cfg, self.processor, observation, observation['task_description'], TORCH_DTYPE)
        #print("cfg.use_proprio:",self.cfg.use_proprio)
        inputs_batch = self.model.prepare_inputs_batch([inputs_t])
        with torch.no_grad():
            action_logits, value = self.model(inputs_batch)
        return action_logits, value
    

# ================================================================
# 4. 训练器 (TrainerActor)
# ================================================================
@ray.remote(num_gpus=1)
class TrainerActor(TrainerActorCom):
    def __init__(self, rank, world_size, replay_buffer, cfg):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.replay_buffer = replay_buffer
        self.cfg = cfg
        self.model = None
        self.optimizer = None
        self.base_model = None
        self.data_dtype = None
        self.next_ready_batch: Optional[Tuple] = None
        self.data_fetching_task = None
        self.super_batch_size = SUPER_BATCH_SIZE
        self.global_step = 0

        print(f"TrainerActor Rank {self.rank} 初始化于 GPU: {ray.get_gpu_ids()}")

    def get_model_keys(self):
        if self.model is None:
            print("模型尚未初始化。请先调用 setup_deepspeed_group()。")
            return {}
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
        self.base_model = model

        # 参数分组（与之前代码一致）
        param_groups = self.base_model.get_parameter_groups()
        optimizer_params = [
            {"params": pg["params"], "name": pg["name"], "lr": POLICY_LR if pg["name"] == "policy" else VALUE_LR}
            for pg in param_groups
        ]
        
        ds_config = {
            "train_micro_batch_size_per_gpu": TRAIN_BATCH_SIZE,
            "gradient_accumulation_steps": ACCUMULATION_STEPS,
            "optimizer": {"type": "AdamW", "params": {}},
            "bf16": {"enabled": USE_BF16},
            "zero_optimization": {
                "stage": 2, "allgather_partitions": True, "allgather_bucket_size": 5e8,
                "reduce_scatter": True, "reduce_bucket_size": 5e8, "overlap_comm": True,
                "contiguous_gradients": True
            },
            "gradient_clipping": 1.0,
        }

        if ds_config.get("bf16", {}).get("enabled", False): self.data_dtype = torch.bfloat16
        else: self.data_dtype = torch.float32

        self.model, self.optimizer, _, _ = deepspeed.initialize(model=model, config=ds_config, model_parameters=optimizer_params)
        print(f"TrainerActor Rank {self.rank}: DeepSpeed 训练组 (ZeRO-2) 初始化完成。")

        self.data_fetching_task = asyncio.get_event_loop().create_task(self._data_fetching_loop())

        n_total = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数量: {n_total:,}, 可训练参数量: {n_trainable:,}")

    def _get_current_lr(self, current_step: int, peak_lr: float, warmup_steps: int, total_steps: int, start_step: int = 0) -> float:
        if current_step < start_step: return 0.0
        effective_step = current_step - start_step
        if effective_step < warmup_steps: return peak_lr * (effective_step / warmup_steps)
        progress = (effective_step - warmup_steps) / (total_steps - start_step - warmup_steps)
        progress = min(progress, 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return peak_lr * cosine_decay

    async def _data_fetching_loop(self):
        print(f"Trainer {self.rank}: 后台数据准备循环已启动 (超级批次大小: {self.super_batch_size})。")
        while True:
            try:
                if self.next_ready_batch is not None:
                    await asyncio.sleep(0.1)
                    continue

                while await self.replay_buffer.size.remote() < self.super_batch_size:
                    print(f"Trainer {self.rank} (BG): 等待 ReplayBuffer 填充至 {self.super_batch_size}...")
                    await asyncio.sleep(3)

                # ############################# 核心修改：获取新的经验数据 ##############################
                obs_list, action_token_np, adv_np, logits_old_np, v_targ_np = \
                    await self.replay_buffer.sample.remote(self.super_batch_size)
                # ################################################################################

                inputs_batch = self.base_model.prepare_inputs_batch(obs_list)

                device = next(self.model.parameters()).device
                # ############################# 核心修改：准备新的 Tensors ##############################
                act_token_t = torch.tensor(action_token_np, dtype=torch.long, device=device) # Tokens 是 long 类型
                adv_t = torch.tensor(adv_np, dtype=torch.float32, device=device)
                logits_old_t = torch.tensor(logits_old_np, dtype=torch.float32, device=device)
                v_targ_t = torch.tensor(v_targ_np, dtype=torch.float32, device=device)

                self.next_ready_batch = (inputs_batch, act_token_t, adv_t, logits_old_t, v_targ_t)
                # ####################################################################################

            except Exception as e:
                print(f"Trainer {self.rank}: 数据采样失败: {e}。将在3秒后重试。")
                await asyncio.sleep(3)

    async def run_training_epoch(self) -> Tuple[float, float, float, float, Dict[str, float], int]:
        if self.next_ready_batch is None:
            print(f"Trainer {self.rank}: 等待初始超级批次...")
            while self.next_ready_batch is None:
                await asyncio.sleep(0.2)
            print(f"Trainer {self.rank}: 初始数据已收到，开始第一个训练周期。")

        current_lrs = {}
        value_lr = self._get_current_lr(self.global_step, VALUE_LR, VALUE_WARMUP_STEPS, TRAIN_ITERS)
        policy_lr = self._get_current_lr(self.global_step, POLICY_LR, POLICY_WARMUP_STEPS, TRAIN_ITERS, start_step=POLICY_TRAIN_START_STEP)
        
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'value': param_group['lr'] = value_lr; current_lrs['value'] = value_lr
            elif param_group['name'] == 'policy': param_group['lr'] = policy_lr; current_lrs['policy'] = policy_lr

        current_batch = self.next_ready_batch
        self.next_ready_batch = None
        
        # ############################# 核心修改：解包新的批处理数据 ##############################
        inputs_batch, act_token_t, adv_t, logits_old_t, v_targ_t = current_batch
        # ####################################################################################

        # 修正std 归一化（消融1）
        # 计算本地统计量
        local_sum = adv_t.sum()
        local_sq_sum = (adv_t * adv_t).sum()
        local_count = torch.tensor([adv_t.numel()], device=adv_t.device, dtype=torch.float32)

        # 使用分布式all_reduce获取全局统计量
        stats_tensor = torch.stack([local_sum, local_sq_sum, local_count.squeeze(0)])
        distributed.all_reduce(stats_tensor, op=distributed.ReduceOp.SUM)

        global_sum, global_sq_sum, global_count = stats_tensor[0], stats_tensor[1], stats_tensor[2]
        global_mean = global_sum / torch.clamp(global_count, min=1.0)
        global_var = torch.clamp(global_sq_sum / torch.clamp(global_count, min=1.0) - global_mean * global_mean, min=1e-12)
        global_std = torch.sqrt(global_var)
        # ========================================================

        # local_adv_mean = adv_t.mean()
        # local_adv_std = adv_t.std()
        # global_adv_stats = torch.tensor([local_adv_mean.item(), local_adv_std.item()], device=adv_t.device, dtype=self.data_dtype)
        # distributed.all_reduce(global_adv_stats, op=distributed.ReduceOp.AVG)
        # global_adv_mean, global_adv_std = global_adv_stats[0], global_adv_stats[1]

        epoch_losses, epoch_p_losses, epoch_v_losses, epoch_e_losses = [], [], [], []
        num_updates_in_epoch = self.super_batch_size // TRAIN_BATCH_SIZE
        
        for i in range(num_updates_in_epoch):
            start = i * TRAIN_BATCH_SIZE; end = start + TRAIN_BATCH_SIZE
            mini_inputs = {k: v[start:end] for k, v in inputs_batch.items()}
            
            # ############################# 核心修改：切分新的小批次数据 ##############################
            mini_act_token = act_token_t[start:end]
            mini_adv = adv_t[start:end]
            mini_logits_old = logits_old_t[start:end]
            mini_v_targ = v_targ_t[start:end]
            
            # 使用全局统计量进行归一化
            normalized_adv = (mini_adv - global_mean) / (global_std + 1e-8)
            # 前向
            action_logits, value = self.model.forward(mini_inputs)
            value = value.to(torch.float32)

            # 只使用第一个动作块进行训练
            action_logits_first_chunk = action_logits.view(
                -1, NUM_ACTIONS_CHUNK, ACTION_DIM, action_logits.shape[-1]
            )

            # 价值损失 (不变)
            value_loss = VF_COEF * torch.mean((value - mini_v_targ) ** 2)
            
            if self.global_step < POLICY_TRAIN_START_STEP:
                loss = value_loss
                policy_loss = torch.tensor(0.0, device=loss.device)
                ent_loss = torch.tensor(0.0, device=loss.device)
            else:
                # 策略与熵损失 (离散版本)
                dist = torch.distributions.Categorical(logits=action_logits_first_chunk)
                logp = dist.log_prob(mini_act_token).sum(dim=-1) # 对动作维度求和

                with torch.no_grad():
                    dist_old = torch.distributions.Categorical(logits=mini_logits_old)
                    logp_old = dist_old.log_prob(mini_act_token).sum(dim=-1)

                ratio = torch.exp(logp - logp_old)
                surr1 = ratio * normalized_adv.unsqueeze(dim=-1)
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * normalized_adv.unsqueeze(dim=-1)
                policy_loss = -torch.mean(torch.min(surr1, surr2))
                
                # 熵是在动作维度上求和，然后在批次上求平均
                ent_loss = -ENT_COEF * torch.mean(dist.entropy().sum(dim=-1))
                loss = policy_loss + value_loss + ent_loss
            # #######################################################################################

            self.model.backward(loss)
            self.model.step()
            epoch_losses.append(loss.item())
            epoch_p_losses.append(policy_loss.item())
            epoch_v_losses.append(value_loss.item())
            epoch_e_losses.append(ent_loss.item())
            if self.model.is_gradient_accumulation_boundary():
                self.global_step += 1

        avg_loss = np.mean(epoch_losses)
        avg_p_loss = np.mean(epoch_p_losses)
        avg_v_loss = np.mean(epoch_v_losses)
        avg_e_loss = np.mean(epoch_e_losses)

        return avg_loss, avg_p_loss, avg_v_loss, avg_e_loss, current_lrs, self.global_step

# ================================================================
# 5. 主逻辑
# ================================================================
def build_openvla_cfg() -> GenerateConfig:
    cfg = GenerateConfig(
        pretrained_checkpoint=PRETRAINED_CHECKPOINT,
        use_l1_regression=False, # Note: ActorCritic in discrete model doesn't use this
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=False, # Note: ActorCritic in discrete model can handle this
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        unnorm_key="libero_spatial_no_noops",
    )
    return cfg

def main():
    if not os.path.exists(PRETRAINED_CHECKPOINT):
        print(f"错误: OpenVLA checkpoint 路径 '{PRETRAINED_CHECKPOINT}' 不存在。请更新 PRETRAINED_CHECKPOINT。")
        return

    os.environ["RAY_DEDUP_LOGS"] = "0"
    ray.init(ignore_reinit_error=True, _temp_dir='/dev/shm')

    log_dir = f"runs/Libero/{BENCHMARK}/OpenVLA_DS_PPO_DISCRETE_ATTE_POOL_{int(time.time())}"
    writer = SummaryWriter(log_dir)
    stats_actor = StatsActor.remote(window_size=MOVING_AVG_WINDOW)
    print(f"TensorBoard 日志将保存在: {log_dir}")

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
            replay_buffers[i % NUM_TRAINER_GPUS], i, stats_actor, cfg,
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
    print("推理器前向测试完成 (广播前)。")
    
    broadcast_task = trainer_group[0].broadcast_weights.remote(BROADCAST_GROUP_NAME)
    receive_tasks = [inf.receive_and_update_weights.remote(BROADCAST_GROUP_NAME) for inf in inference_pool]
    ray.get([broadcast_task] + receive_tasks)
    print("初始权重已广播到所有推理器。")

    forward_test_tasks = [inf.forward_test.remote() for inf in inference_pool]
    ray.get(forward_test_tasks)
    print("推理器前向测试完成 (广播后)。")

    print("\n--- 步骤 4: 启动 Rollout Workers 进行数据收集 ---")
    for w in rollout_workers: w.run.remote()

    print("\n--- 步骤 5: 等待远程经验池填充初始数据 ---")
    min_buffer_size_for_start = SUPER_BATCH_SIZE
    assert min_buffer_size_for_start < REPLAY_CAPACITY, "初始填充量必须小于回放池总容量"
    while not all(size >= min_buffer_size_for_start for size in ray.get([rb.size.remote() for rb in replay_buffers])):
        sizes = ray.get([rb.size.remote() for rb in replay_buffers])
        print(f"等待所有经验池填充初始数据 (目标: {min_buffer_size_for_start})... (当前大小: {sizes})")
        time.sleep(5)
    print("远程经验池已准备好，训练器将按需获取数据。")

    print("\n--- 步骤 6: 开始主训练与同步循环 ---")
    start_time = time.time()
    last_log_time = time.time()
    global_step = 0
    while global_step < TRAIN_ITERS:
        train_tasks = [trainer.run_training_epoch.remote() for trainer in trainer_group]
        results = ray.get(train_tasks)
        _, _, _, _, _, global_step = results[0]

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

            total_losses, p_losses, v_losses, e_losses, lrs_list, _ = zip(*results)
            current_lrs = lrs_list[0]

            elapsed_time = current_time - start_time
            total_buffer_size = sum(ray.get([rb.size.remote() for rb in replay_buffers]))

            print(f"更新步 {global_step}/{TRAIN_ITERS} | 时间: {elapsed_time:.1f}s | "
                  f"全局平均奖励: {avg_return:.2f} | 全局平均幕长: {avg_ep_len:.1f} | "
                  f"value loss: {np.mean(v_losses):.4f} | LR(V/P): {current_lrs['value']:.7f}/{current_lrs['policy']:.7f} | "
                  f"Episodes数量: {total_episodes:,} | Step平均时间: {avg_step_time:.3f}s")

            writer.add_scalar('Train/Learning_Rate/Value', current_lrs['value'], global_step)
            writer.add_scalar('Train/Learning_Rate/Policy', current_lrs['policy'], global_step)
            writer.add_scalar('Loss/Total', np.mean(total_losses), global_step)
            writer.add_scalar('Loss/Policy', np.mean(p_losses), global_step)
            writer.add_scalar('Loss/Value', np.mean(v_losses), global_step)
            writer.add_scalar('Loss/Entropy', np.mean(e_losses), global_step)
            writer.add_scalar('Rollout/_Global/Average_Return', avg_return, global_step)
            writer.add_scalar('Rollout/_Global/Average_Episode_Length', avg_ep_len, global_step)
            writer.add_scalar('System/Replay_Buffer_Size_Total', total_buffer_size, global_step)
            writer.add_scalar('System/Total_Episodes_Processed', total_episodes, global_step)
            writer.add_scalar('System/Avg_Step_Time', avg_step_time, global_step)

            for env_name, env_stats in all_stats.items():
                tag_prefix = f"Rollout/{env_name}"
                writer.add_scalar(f'{tag_prefix}/Average_Return', env_stats['avg_return'], global_step)
                writer.add_scalar(f'{tag_prefix}/Average_Episode_Length', env_stats['avg_ep_len'], global_step)
                writer.add_scalar(f'{tag_prefix}/Success_Rate', env_stats['avg_success_rate'], global_step)
                writer.add_scalar(f'{tag_prefix}/Total_Episodes', env_stats['total_episodes'], global_step)

            last_log_time = current_time

    print(f"\n成功完成 {TRAIN_ITERS} 次训练与同步循环！")
    writer.close()
    ray.shutdown()


if __name__ == "__main__":
    main()