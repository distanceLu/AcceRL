import os
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["TMPDIR"] = "/dev/shm"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"

import time
import random
import asyncio
from collections import deque, defaultdict
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import math
import shutil
import socket
import contextlib

import numpy as np

import ray
import torch
import torch.distributed as distributed
import deepspeed
from torch.utils.tensorboard import SummaryWriter

# OpenVLA 和 Libero 工具
from experiments.robot.openvla_utils import get_processor
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM
from experiments.robot.libero.libero_utils import GenerateConfig

from rl.world_model_discrete import WorldModel
from rl.utils import prepare_one_obs

# ================================================================
# 0. 超参数与配置
# ================================================================
EXP_NAME = "only_wm_env_idx1_8layer"
BENCHMARK = "libero_spatial"

# 分布式系统参数
NUM_TRAINER_GPUS = 2
NUM_INFERENCE_ACTORS = 1
NUM_ROLLOUT_WORKERS = 20
ROLLOUT_LOCAL_BUF = 64
INFERENCE_BATCH = 8
INFERENCE_TIMEOUT_MS = 300
REPLAY_CAPACITY = 10000
TRAIN_BATCH_SIZE = 8
WORLD_ACCUM = 32
TRAIN_ITERS = 30000

# RT 和 AE 损失的系数
RT_LOSS_COEF = 1.0
AE_LOSS_COEF = 1.0

# 学习率调度参数
WORLD_LR = 3e-5
WORLD_WARMUP_STEPS = 500

# 日志
MOVING_AVG_WINDOW = 1000
LOG_INTERVAL_SECONDS = 10
SAVE_INTERVAL_STEPS = 200

# OpenVLA 加载配置
USE_BF16: bool = True
TORCH_DTYPE = torch.bfloat16 if USE_BF16 else torch.float32
PRETRAINED_CHECKPOINT = "/cpfs01/liuwei_workspace/models/finetune_im/openvla-7b+libero_spatial_no_noops+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--discrete_acts--proprio_state--100000_chkpt"
CHECKPOINT2 = "/cpfs01/lcx_workspace/models/ppo_wm_discrete_env_idx1_rt_token_1762259733/checkpoint_3000"

INP_MAX_LEN = 100  # 输入input_id的最大长度

# ================================================================
# 数据结构
# ================================================================
@dataclass
class Experience:
    obs: Dict[str, torch.Tensor]
    action: np.ndarray                      # 离散动作token (NUM_ACTIONS_CHUNK,)
    next_obs: Optional[Dict[str, torch.Tensor]] # 下一状态的原始观测
    done: bool                              # 结束标志
    reward: float

# ================================================================
# 1.5. 统计模块 (StatsActor)
# ================================================================
@ray.remote
class StatsActor:
    def __init__(self, window_size):
        self.stats = defaultdict(lambda: {
            "episode_returns": deque(maxlen=window_size),
            "step_times": deque(maxlen=window_size),
            "episode_lengths": deque(maxlen=window_size),
            "successes": deque(maxlen=window_size),
            "total_episodes_processed": 0
        })
        self.timings = defaultdict(lambda: deque(maxlen=window_size))
        self.actor_last_active = {}  # {actor_id: timestamp}
        self.active_window_seconds = 600  # 10分钟

    def add_episode_return(self, env_name: str, ep_return: float, step_time: float, ep_length: int, success: float, actor_id: int, step_num: int):
        env_stats = self.stats[env_name]
        env_stats["episode_returns"].append(ep_return)
        env_stats["step_times"].append(step_time)
        env_stats["episode_lengths"].append(ep_length)
        env_stats["successes"].append(success)
        env_stats["total_episodes_processed"] += 1
        if "total_samples_produced" not in self.stats["_global_"]:
            self.stats["_global_"]["total_samples_produced"] = 0
        self.stats["_global_"]["total_samples_produced"] += step_num
        self.actor_last_active[actor_id] = time.time()

    def add_timing_metric(self, metric_name: str, value: float):
        """记录一个通用的性能计时值 (例如，一个循环或函数的执行时间)"""
        self.timings[metric_name].append(value)

    def get_active_actor_count(self) -> int:
        """返回最近10分钟内活跃的actor数量"""
        current_time = time.time()
        cutoff_time = current_time - self.active_window_seconds
        active_count = sum(1 for last_active in self.actor_last_active.values() 
                          if last_active >= cutoff_time)
        return active_count

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        per_env_stats = {}
        all_returns, all_lengths, all_step_times, all_successes = [], [], [], []
        total_episodes_processed = 0

        for env_name, env_data in self.stats.items():
            if env_name == "_global_":
                continue
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
            "total_samples_produced": self.stats["_global_"].get("total_samples_produced", 0),
            "active_actor_count": self.get_active_actor_count()
        }
        
        timing_stats = {}
        for name, deq in self.timings.items():
            if deq:
                timing_stats[name] = np.mean(deq)
            else:
                timing_stats[name] = 0.0
        per_env_stats["_timings_"] = timing_stats
        
        return per_env_stats

# ================================================================
# 2. 经验回放与 Rollout
# ================================================================
@ray.remote
class ReplayBufferActor:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add_batch(self, batch: List[Experience]):
        self.buffer.extend(batch)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        batch = random.sample(self.buffer, batch_size)
        obs_list = [b.obs for b in batch]
        
        # 为None的next_obs（在done=True时）使用其对应的obs作为占位符
        next_obs_list = [b.next_obs if b.next_obs is not None else b.obs for b in batch]
        
        act = np.stack([b.action for b in batch])  # (N, NUM_ACTIONS_CHUNK)
        done = np.asarray([b.done for b in batch], np.bool_)
        reward = np.asarray([b.reward for b in batch], np.float32)

        return obs_list, next_obs_list, act, done, reward

@ray.remote
class RolloutWorkerActor:
    def __init__(self, infer, replay, wid, stats_actor, cfg, benchmark_name, dtype, local_buff_len):
        self.infer, self.replay = infer, replay
        self.stats_actor = stats_actor
        self.cfg = cfg
        self.processor = get_processor(cfg)
        self.benchmark_name = benchmark_name
        self.dtype = dtype
        self.local_buff_len = local_buff_len
        from rl.libero_env import LiberoEnvWrapper

        # 1. 初始化所有任务的环境
        self.num_tasks = 3
        print(f"Worker {wid}: 正在初始化 {self.num_tasks} 个 Libero 环境...")
        self.envs = [
            LiberoEnvWrapper(
                benchmark_name=self.benchmark_name,
                task_id=i,
                image_size=224,
                render_mode="rgb_array",
            ) for i in range(self.num_tasks)
        ]
        print(f"Worker {wid}: 环境初始化完成。")
        # 2. 初始化所有环境的采样权重，初始值均为1
        self.env_weights = np.ones(self.num_tasks, dtype=np.float32)
        # 3. 用于存储当前活动环境的占位符
        self.env = None               # 指向当前选择的 env 对象
        self.current_env_idx = -1     # 当前选择的 env 在列表中的索引

        self.wid = wid
        self.local_buffer = []
        self.task_description = None
        self.current_env_name = None

    def _reset_and_select_env(self, seed: Optional[int] = None) -> Tuple[Dict, Dict]:
        """
        根据权重随机选择一个环境并重置它。
        """
        probabilities = self.env_weights / np.sum(self.env_weights)
        self.current_env_idx = np.random.choice(self.num_tasks, p=probabilities)
        self.current_env_idx = 1  # 强制选择任务1进行调试 TODO
        seed = 0  # TODO
        self.env = self.envs[self.current_env_idx]
        obs, info = self.env.reset(seed=seed)
        self.task_description = self.env.task_description
        self.current_env_name = self.env.get_name()
        return obs, info

    def run(self):
        try:
            current_seed = int(time.time() * 1000) + self.wid + os.getpid()
            obs, info = self._reset_and_select_env(seed=current_seed)

            reward_sum = 0.0
            step_count = 0
            infer_step = 0
            time_start = time.time()

            while True:
                inputs_t = prepare_one_obs(self.cfg, self.processor, obs, self.task_description, self.dtype)
                inputs_t['step_count'] = torch.tensor([step_count], dtype=torch.long)
                (action_token, continuous_action) = ray.get(self.infer.request.remote(inputs_t))
                infer_step += 1
                chunk_reward = 0.0
                done = False
                for i in range(len(continuous_action)):
                    single_action = continuous_action[i]
                    nxt, r, term, trunc, info = self.env.step(single_action)
                
                    reward_sum += r
                    chunk_reward += r
                
                    step_count += 1
                    if term or trunc:
                        done = True
                        break
                
                # 存储所有信息，包括下一个原始观测和结束标志
                self.local_buffer.append((
                    inputs_t, action_token, chunk_reward, done
                ))
                obs = nxt 
                if done:
                    step_time = (time.time() - time_start) / max(step_count, 1)
                    success = float(info.get('is_success', 0.0))

                    # 如果任务失败，增加对应环境的权重
                    if success < 1.0:
                        self.env_weights[self.current_env_idx] += 1

                    self.stats_actor.add_episode_return.remote(
                        self.current_env_name, reward_sum, step_time, step_count, success, self.wid, infer_step
                    )
                    if self.local_buffer:
                        self._process_traj(self.local_buffer)
                    self.local_buffer.clear()
                    
                    current_seed = int(time.time() * 1000) + self.wid + os.getpid()
                    obs, info = self._reset_and_select_env(seed=current_seed)
                    reward_sum = 0.0
                    step_count = 0
                    infer_step = 0
                    time_start = time.time()
                elif len(self.local_buffer) == self.local_buff_len + 1:
                    self._process_traj(self.local_buffer[:-1])
                    self.local_buffer = [self.local_buffer[-1]]
        except Exception as e:
            import traceback
            print(f"[ERROR] RolloutWorker {self.wid} run() 崩溃: {e}", flush=True)
            traceback.print_exc()
            raise

    def _process_traj(self, traj_segment):
        batch: List[Experience] = []
        for i, (s, action_token, rew, done) in enumerate(traj_segment):
            if done:
                next_obs_processed = s
            elif i < len(traj_segment) - 1:
                next_obs_processed = traj_segment[i+1][0]  # 使用traj中的下一个obs
            else:
                # traj的最后一步，使用当前obs作为next_obs
                next_obs_processed = s
                
            batch.append(
                Experience(
                    obs=s,
                    action=action_token.astype(np.int64),
                    done=done,
                    next_obs=next_obs_processed,
                    reward=rew,
                )
            )
        self.replay.add_batch.remote(batch)

# ================================================================
# 3. 推理器 (InferenceActor)
# ================================================================
@ray.remote(num_gpus=1)
class InferenceActor:
    def __init__(self, actor_id, cfg, dtype, infer_bs, infer_timeout, freeze_value, max_len, stats_actor):
        self.actor_id = actor_id
        self.stats_actor = stats_actor
        # 加载世界模型
        print(f"InferenceActor {actor_id}: 正在加载 WorldModel...")
        self.model = WorldModel(cfg, torch_dtype=dtype, checkpoint_dir=cfg.checkpoint2, freeze_value=freeze_value)  
        self.model.cuda()
        self.model.eval()

        self.processor = self.model.processor
        self.cfg = cfg
        self.max_len = max_len
        self.dtype = dtype

        self.batch_size = infer_bs
        self.timeout_sec = infer_timeout / 1000.0
        self.requests, self.promises = [], []
        self.last_process_time = time.time()

        loop = asyncio.get_event_loop()
        self._bg_task = loop.create_task(self._loop())
        self._bg_task.add_done_callback(self._on_bg_task_done)
        print(f"InferenceActor {self.actor_id} 初始化于 GPU: {ray.get_gpu_ids()}")

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
            
            t_loop_start = time.time()

            try:
                inputs_batch = self.model.prepare_inputs_batch(requests_to_process, self.max_len)
                with torch.inference_mode():
                    logits, _ = self.model.agent_super_forward(inputs_batch, return_vit_out=False)
                
                # 使用agent的post_process方法获取离散和连续动作
                action_tokens = []
                normalized_actions = []
                _, action_tokens, normalized_actions = self.model.post_process(logits, deterministic=[False]*logits.shape[0])
                action_tokens = action_tokens.detach().cpu().numpy()
                
                # 将标准化动作转换为环境动作
                actions_env = []
                for i in range(normalized_actions.shape[0]):
                    a_env = self.model.vla._unnormalize_actions(normalized_actions[i], self.cfg.unnorm_key)
                    actions_env.append(a_env.astype(np.float32))
                for i in range(len(promises_to_process)):
                    promises_to_process[i].set_result((
                        action_tokens[i],           # 离散动作token
                        actions_env[i],             # 连续动作（用于环境）
                    ))
                loop_duration = time.time() - t_loop_start
                self.stats_actor.add_timing_metric.remote("Inference/loop_time_ms", loop_duration * 1000)

            except Exception as e:
                import traceback
                print(f"[ERROR] InferenceActor {self.actor_id} 批处理失败: {e}", flush=True)
                traceback.print_exc()
                for p in promises_to_process:
                    if not p.done():
                        p.set_exception(e)
                raise
    
    def forward_test(self, chunk_num, action_dim):
        import pickle
        with open("experiments/robot/libero/sample_libero_spatial_observation.pkl", "rb") as file:
            observation = pickle.load(file)
        inputs_t = prepare_one_obs(self.cfg, self.processor, observation, observation['task_description'], self.dtype)
        inputs_t['step_count'] = torch.tensor([0], dtype=torch.long)  # 添加 step_count 信息
        inputs_batch = self.model.prepare_inputs_batch([inputs_t])
        inputs_batch['this_action'] = torch.zeros((1, chunk_num, action_dim), dtype=self.dtype).cuda()
        with torch.no_grad():
            self.model(inputs_batch)

# ================================================================
# 4. 训练器 (TrainerActor)
# ================================================================
@ray.remote(num_gpus=1)
class TrainerActor:
    def __init__(self, rank, world_size, replay_buffer, cfg):
        self.rank = rank
        self.world_size = world_size
        self.replay_buffer = replay_buffer
        self.cfg = cfg
        self.model = None
        self.optimizer = None
        self.base_model = None
        self.data_dtype = None
        
        self.next_wm_batch: Optional[Tuple] = None
        self.wm_data_fetching_task = None
        
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
        
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        print(f"\nTrainer Rank 0: 正在保存模型到 '{save_dir}'...")
        model_to_save.save_checkpoint(save_dir)
        print(f"Trainer Rank 0: 模型保存完成。")

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
        model = WorldModel(self.cfg, torch_dtype=TORCH_DTYPE, checkpoint_dir=self.cfg.checkpoint2, freeze_value=False)
        self.base_model = model

        param_groups = self.base_model.get_parameter_groups()
        optimizer_params = [
            {"params": pg["params"], "name": pg["name"], "lr": WORLD_LR}
            for pg in param_groups if pg["name"] == "world"  # 只训练世界模型参数
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

    def _get_current_lr(self, current_step, peak_lr, warmup_steps, total_steps):
        if current_step < warmup_steps:
            return peak_lr * (current_step / warmup_steps)
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        progress = min(progress, 1.0)
        return peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _slice_and_to_gpu(self, data_cpu, start_idx, end_idx, device):
        """将CPU上的数据切片并转移到GPU"""
        result = {}
        for k, v in data_cpu.items():
            # 特殊处理字典类型的批次数据 (inputs_batch, next_inputs_batch)
            if isinstance(v, dict):
                result[k] = {}
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, torch.Tensor):
                        result[k][sub_k] = sub_v[start_idx:end_idx].to(device)
                    elif isinstance(sub_v, np.ndarray):
                        result[k][sub_k] = torch.tensor(sub_v[start_idx:end_idx], device=device)
                    else:
                        result[k][sub_k] = sub_v
            elif isinstance(v, torch.Tensor):
                result[k] = v[start_idx:end_idx].to(device)
            elif isinstance(v, np.ndarray):
                result[k] = torch.tensor(v[start_idx:end_idx], device=device)
            else:
                result[k] = v
        return result

    async def _wm_data_fetching_loop(self):
        print(f"Trainer {self.rank}: (WM)后台数据准备循环已启动。")
        while True:
            try:
                if self.next_wm_batch is not None:
                    await asyncio.sleep(0.1)
                    continue

                required_samples = TRAIN_BATCH_SIZE * WORLD_ACCUM
                while await self.replay_buffer.size.remote() < required_samples:
                    await asyncio.sleep(1)

                t_sample_start = time.time()
                sampled_data = await self.replay_buffer.sample.remote(required_samples)
                if sampled_data is None:
                    continue
                
                obs_list, next_obs_list, act_np, done_np, reward_np = sampled_data
                sample_time = time.time() - t_sample_start
                
                t_prep_start = time.time()
                inputs_batch = self.base_model.prepare_inputs_batch(obs_list, INP_MAX_LEN)
                next_inputs_batch = self.base_model.prepare_inputs_batch(next_obs_list, INP_MAX_LEN)
                prep_time = time.time() - t_prep_start
                
                # 存储完整宏批次（保持在CPU）
                self.next_wm_batch = {
                    'inputs_batch': inputs_batch,
                    'next_inputs_batch': next_inputs_batch,
                    'actions': torch.tensor(act_np, dtype=torch.long),
                    'dones': torch.tensor(done_np, dtype=torch.bool),
                    'rewards': torch.tensor(reward_np, dtype=torch.float32),
                    'sample_time': sample_time,
                    'prep_time': prep_time
                }
                
                if self.rank == 0 and random.random() < 0.1:
                    print(f"Trainer {self.rank}: WM宏批次拉取完成 - 采样: {sample_time:.3f}s, 准备: {prep_time:.3f}s, 样本数: {required_samples}")
                    
            except Exception as e:
                print(f"Trainer {self.rank}: (WM)数据采样失败: {e}。将在3秒后重试。")
                await asyncio.sleep(3)

    async def run_training_epoch(self) -> Tuple[Dict[str, float], Dict[str, int], int, Dict[str, float]]:
        perf_timings = {}
        
        # 等待数据准备完成
        t_wait_data_start = time.time()
        while self.next_wm_batch is None:
            await asyncio.sleep(0.1)
        perf_timings["data_wait_time"] = time.time() - t_wait_data_start

        # 获取批次数据并立即设为None以触发后台拉取
        wm_batch = self.next_wm_batch
        
        perf_timings['wm_sample_time'] = wm_batch['sample_time']
        perf_timings['wm_prep_time'] = wm_batch['prep_time']
        
        self.next_wm_batch = None

        # 更新学习率
        current_lrs = {}
        world_lr = self._get_current_lr(self.global_step, WORLD_LR, WORLD_WARMUP_STEPS, TRAIN_ITERS)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = world_lr
            current_lrs['world'] = world_lr
        
        epoch_losses = defaultdict(list)
        self.model.train()
        device = self.model.device

        # 世界模型训练
        t_wm_start = time.time()
        wm_to_gpu_times = []
        
        wm_data_cpu = {
            'inputs_batch': wm_batch['inputs_batch'],
            'next_inputs_batch': wm_batch['next_inputs_batch'],
            'actions': wm_batch['actions'],
            'dones': wm_batch['dones'],
            'rewards': wm_batch['rewards']
        }
        
        for i in range(WORLD_ACCUM):
            start_idx = i * TRAIN_BATCH_SIZE
            end_idx = (i + 1) * TRAIN_BATCH_SIZE
            
            t_to_gpu_start = time.time()
            wm_mini_batch = self._slice_and_to_gpu(wm_data_cpu, start_idx, end_idx, device)
            wm_to_gpu_times.append(time.time() - t_to_gpu_start)
            
            wm_inp = {**wm_mini_batch['inputs_batch'], 'this_action': wm_mini_batch['actions']}

            # 动态计算教师特征作为AE损失的目标
            with torch.no_grad():
                self.model.module.eval() 
                _, _, mini_next_teacher_proj_feat = self.model.forward_vision(wm_mini_batch['next_inputs_batch'])
                self.model.module.train()

            ae_loss, rt_loss, reward_acc, reward_mean, termin_acc, termi_mean, rt_acc, mae_loss, relative_error = \
                self.model.module.compute_world_model_loss(
                    wm_inp, 
                    wm_mini_batch['dones'], 
                    mini_next_teacher_proj_feat, 
                    wm_mini_batch['rewards']
                )
            
            world_model_loss = AE_LOSS_COEF * mae_loss + RT_LOSS_COEF * rt_loss
            self.model.backward(world_model_loss / WORLD_ACCUM)
            
            epoch_losses["ae_loss"].append(ae_loss.item())
            epoch_losses["rt_loss"].append(rt_loss.item())
            epoch_losses["reward_acc"].append(reward_acc.item())
            epoch_losses["reward_mean"].append(reward_mean.item())
            epoch_losses["termin_acc"].append(termin_acc.item())
            epoch_losses["termi_mean"].append(termi_mean.item())
            epoch_losses["rt_classification_acc"].append(rt_acc.item())
            epoch_losses["mae_loss"].append(mae_loss.item())
            epoch_losses["relative_error"].append(relative_error.item())

        perf_timings["wm_train_time"] = time.time() - t_wm_start
        perf_timings["wm_to_gpu_time"] = np.mean(wm_to_gpu_times)

        # 优化器步骤
        self.model.step()
        self.global_step += 1

        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        return avg_losses, current_lrs, self.global_step, perf_timings


def build_openvla_cfg() -> GenerateConfig:
    cfg = GenerateConfig(
        pretrained_checkpoint=PRETRAINED_CHECKPOINT,
        use_l1_regression=False,
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

    object_store_size_gb = 256 # 分配的GB数
    object_store_memory_bytes = int(object_store_size_gb * 1024 * 1024 * 1024)

    print(f"正在初始化 Ray，并为对象存储分配 {object_store_size_gb} GB 内存...")
    
    ray.init(
        ignore_reinit_error=True, 
        _temp_dir='/dev/shm',
        object_store_memory=object_store_memory_bytes
    )
    exp_name = f"{EXP_NAME}_{int(time.time())}"
    save_dir = f"/cpfs01/lcx_workspace/models/{exp_name}"
    log_dir = f"runs/wm_only/{exp_name}"
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    stats_actor = StatsActor.remote(window_size=MOVING_AVG_WINDOW)
    print(f"TensorBoard 日志将保存在: {log_dir}")
    print(f"模型检查点将保存在: {save_dir}")

    cfg = build_openvla_cfg()

    print("--- 步骤 1: 创建 Actors ---")
    replay_buffers = [ReplayBufferActor.remote(capacity=REPLAY_CAPACITY) for _ in range(NUM_TRAINER_GPUS)]
    
    trainer_group = [
        TrainerActor.remote(
            rank=i, world_size=NUM_TRAINER_GPUS, 
            replay_buffer=replay_buffers[i], 
            cfg=cfg
        ) for i in range(NUM_TRAINER_GPUS)
    ]
    
    inference_pool = [InferenceActor.remote(
        actor_id=i, 
        cfg=cfg, 
        dtype=TORCH_DTYPE, 
        infer_bs=INFERENCE_BATCH, 
        infer_timeout=INFERENCE_TIMEOUT_MS, 
        freeze_value=False, 
        max_len=INP_MAX_LEN,
        stats_actor=stats_actor
    ) for i in range(NUM_INFERENCE_ACTORS)]

    rollout_workers = [
        RolloutWorkerActor.remote(
            inference_pool[i % NUM_INFERENCE_ACTORS],
            replay_buffers[i % NUM_TRAINER_GPUS], i, stats_actor, cfg,
            benchmark_name=BENCHMARK, dtype=TORCH_DTYPE, local_buff_len=ROLLOUT_LOCAL_BUF
        ) for i in range(NUM_ROLLOUT_WORKERS)
    ]

    print("\n--- 正在为通信组查找空闲端口... ---")
    train_group_port = find_free_port()
    print(f"找到端口: 训练组 = {train_group_port}")

    print("\n--- 步骤 2: 建立 DeepSpeed 训练组 ---")
    trainer_master_addr = ray.get(trainer_group[0].get_node_ip.remote())
    ray.get([actor.setup_deepspeed_group.remote(trainer_master_addr, train_group_port) for actor in trainer_group])
    print("DeepSpeed 训练组建立完成。")

    print("\n--- 正在运行前向测试... ---")
    ray.get([inf.forward_test.remote(NUM_ACTIONS_CHUNK, ACTION_DIM) for inf in inference_pool])
    print("所有 Actor 前向测试完成。")

    print("\n--- 步骤 3: 启动 Rollout ---")
    for w in rollout_workers: w.run.remote()

    print("\n--- 步骤 4: 等待经验池填充 ---")
    min_real_buffer_size = TRAIN_BATCH_SIZE * WORLD_ACCUM
    
    while True:
        real_sizes = ray.get([rb.size.remote() for rb in replay_buffers])
        
        real_ready = all(size >= min_real_buffer_size for size in real_sizes)
        
        print(f"等待经验池填充... "
              f"真实数据: {real_sizes} (需 >={min_real_buffer_size})")
              
        if real_ready: break
        time.sleep(5)
    
    print("\n--- 步骤 5: 开始主训练循环 ---")
    start_time = time.time()
    last_log_time = time.time()
    global_step = 0
    last_saved_step = -1
    while global_step < TRAIN_ITERS:
        t_train_start = time.time()
        train_tasks = [trainer.run_training_epoch.remote() for trainer in trainer_group]
        results = ray.get(train_tasks)
        
        avg_losses_list, lrs_list, steps_list, perf_timings_list = zip(*results)
        global_step = steps_list[0]
        train_time = time.time() - t_train_start

        # 每 SAVE_INTERVAL_STEPS 步保存一次模型，且只保留最新的一个
        if global_step == 10 or global_step > 0 and global_step % SAVE_INTERVAL_STEPS == 0 and global_step != last_saved_step:
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
            timing_stats = all_stats.pop("_timings_", {})
            avg_losses = {k: np.mean([d[k] for d in avg_losses_list]) for k in avg_losses_list[0]}
            
            # 汇总性能指标
            avg_perf_timings = {}
            for key in perf_timings_list[0].keys():
                avg_perf_timings[key] = np.mean([pt[key] for pt in perf_timings_list])
            avg_perf_timings['train_time'] = train_time
            
            # 损失指标
            for k, v in avg_losses.items():
                writer.add_scalar(f'Loss/{k}', v, global_step)
            # 记录性能监控指标
            for k, v in avg_perf_timings.items():
                writer.add_scalar(f'Performance/{k}', v, global_step)
            for k, v in timing_stats.items():
                writer.add_scalar(f'Performance/{k}', v, global_step)

            current_lrs = lrs_list[0]
            total_real_buffer = sum(ray.get([rb.size.remote() for rb in replay_buffers]))

            print(f"步 {global_step}/{TRAIN_ITERS} | 时间: {time.time() - start_time:.1f}s | "
                  f"奖励: {global_stats['avg_return']:.2f} | "
                  f"AE Loss: {avg_losses.get('ae_loss', 0):.4f} | "
                  f"RT Loss: {avg_losses.get('rt_loss', 0):.4f} | "
                  f"LR(W): {current_lrs['world']:.7f}")

            writer.add_scalar('Train/Learning_Rate/World', current_lrs['world'], global_step)
            writer.add_scalar('Rollout/_Global/Average_Return', global_stats['avg_return'], global_step)
            writer.add_scalar('Rollout/_Global/Average_Success_Rate', global_stats['avg_success_rate'], global_step)
            writer.add_scalar('System/Buffer_Size_Real', total_real_buffer, global_step)
            writer.add_scalar('Rollout/_Global/Average_Episode_Length', global_stats['avg_ep_len'], global_step)
            writer.add_scalar('System/Total_Episodes_Processed', global_stats["total_episodes_processed"], global_step)
            writer.add_scalar('System/Avg_Step_Time', global_stats["avg_step_time"] , global_step)
            writer.add_scalar('System/Total_Samples_Produced', global_stats["total_samples_produced"], global_step)
            writer.add_scalar('System/Active_Rollout_Actors', global_stats["active_actor_count"], global_step)
            
            for env_name, env_stats in all_stats.items():
                tag_prefix = f"Rollout/{env_name}"
                writer.add_scalar(f'{tag_prefix}/Average_Return', env_stats['avg_return'], global_step)
                writer.add_scalar(f'{tag_prefix}/Average_Episode_Length', env_stats['avg_ep_len'], global_step)
                writer.add_scalar(f'{tag_prefix}/Success_Rate', env_stats['avg_success_rate'], global_step)
                writer.add_scalar(f'{tag_prefix}/Total_Episodes', env_stats['total_episodes'], global_step)
            writer.add_scalar('Performance/train_time_total', time.time() - t_train_start, global_step)
            last_log_time = current_time

    print(f"\n成功完成 {TRAIN_ITERS} 次训练！")
    writer.close()
    ray.shutdown()


if __name__ == "__main__":
    main()