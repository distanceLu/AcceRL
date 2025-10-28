import os
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["TMPDIR"] = "/dev/shm"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,4,5,6"

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
import deepspeed
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as distributed 

# OpenVLA 和 Libero 工具
from experiments.robot.openvla_utils import get_processor
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM
from experiments.robot.libero.libero_utils import GenerateConfig

from rl.world_model_discrete import WorldModel, compute_imagined_gae, create_validity_mask, compute_ppo_loss
from rl.utils import prepare_one_obs

# ================================================================
# 0. 超参数与配置
# ================================================================
EXP_NAME = "ppo_wm_param_server_ent0d003_roll40"
BENCHMARK = "libero_spatial"

# 分布式系统参数
NUM_TRAINER_GPUS = 4
NUM_INFERENCE_ACTORS = 1
NUM_IMAGINATION_ACTORS = 1 # 使用1个专用的GPU Actor来生成想象数据
NUM_ROLLOUT_WORKERS = 12
ROLLOUT_LOCAL_BUF = 64
INFERENCE_BATCH = 8
INFERENCE_TIMEOUT_MS = 300
REPLAY_CAPACITY = 10000
IMAGINATION_REPLAY_CAPACITY = 1000
TRAIN_BATCH_SIZE = 22
WORLD_ACCUM = 12
AGENT_ACCUM = 12
TRAIN_ITERS = 100000

# PPO
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
VF_COEF = 0.5
ENT_COEF = 0.003
KL_COEF = 0.02

# 世界模型想象步数
IMAGINE_MAX_HORIZON = 10 

# AE 和 RT 损失的系数
RT_LOSS_COEF = 1.0
AE_LOSS_COEF = 1.0

# 学习率调度参数
WORLD_LR = 3e-5
POLICY_LR = 3e-6
WORLD_WARMUP_STEPS = 500
POLICY_WARMUP_STEPS = 500
POLICY_TRAIN_START_STEP = 100

# 日志
MOVING_AVG_WINDOW = 1000
LOG_INTERVAL_SECONDS = 10
SAVE_INTERVAL_STEPS = 100

# OpenVLA 加载配置
USE_BF16: bool = True
TORCH_DTYPE = torch.bfloat16 if USE_BF16 else torch.float32
PRETRAINED_CHECKPOINT = "/cpfs01/jinshiji_workspace/openvla_oft_rl/runs/openvla-7b-oft-finetuned-2_gpus_batch_size_16_100_000"
CHECKPOINT2 = "/cpfs01/lcx_workspace/models/ppo_wm_param_server2_1761469739/checkpoint_1700"


INP_MAX_LEN = 100  # 输入input_id的最大长度
# ================================================================
# 数据结构
# ================================================================
@dataclass
class Experience:
    obs: Dict[str, torch.Tensor]
    action: np.ndarray                      # 离散动作token (NUM_ACTIONS_CHUNK,)
    advantage: float
    old_logits: np.ndarray                  # (NUM_ACTIONS_CHUNK, VOCAB_SIZE)
    value_target: float
    done: bool                              # 结束标志
    next_teacher_projector_features: Optional[torch.Tensor] # 下一状态的教师视觉特征 (bf16 Tensor)
    reward: float

@dataclass
class ImaginedExperience:
    multimodal_emb: torch.Tensor            # bf16 Tensor
    attention_mask: np.ndarray
    labels: np.ndarray
    action: np.ndarray                      # 离散动作token
    old_logits: np.ndarray                  # (NUM_ACTIONS_CHUNK, VOCAB_SIZE)
    advantage: float
    value_target: float
    step_count: int 
    
# ================================================================
# 0.5. 参数服务器
# ================================================================
@ray.remote
class ParameterServer:
    """轻量级参数服务器，不占用GPU，用于存储和分发最新的模型权重"""
    def __init__(self):
        self.latest_weights_ref = None
        self.version = 0
        print("ParameterServer 初始化完成")

    def set_latest_weights(self, weights_ref, version: int):
        """由 Trainer 调用，更新最新的权重引用"""
        self.latest_weights_ref = weights_ref
        self.version = version
    
    def get_version(self) -> int:
        """返回当前权重版本号"""
        return self.version
        
    def get_weights(self) -> ray.ObjectRef:
        """返回权重引用（仅在版本检查通过后调用）"""
        return self.latest_weights_ref

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
        self.perf_stats = defaultdict(lambda: deque(maxlen=window_size))

    def add_episode_return(self, env_name: str, ep_return: float, step_time: float, ep_length: int, success: float):
        env_stats = self.stats[env_name]
        env_stats["episode_returns"].append(ep_return)
        env_stats["step_times"].append(step_time)
        env_stats["episode_lengths"].append(ep_length)
        env_stats["successes"].append(success)
        env_stats["total_episodes_processed"] += 1

    def add_perf_metric(self, metric_name: str, value: float):
        """添加性能监控指标"""
        self.perf_stats[metric_name].append(value)

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        per_env_stats = {}
        all_returns, all_lengths, all_step_times, all_successes = [], [], [], []
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
    
    def get_perf_stats(self) -> Dict[str, float]:
        """获取性能监控指标的平均值"""
        return {k: np.mean(v) if v else 0.0 for k, v in self.perf_stats.items()}

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
        act = np.stack([b.action for b in batch])  # (N, NUM_ACTIONS_CHUNK)
        adv = np.asarray([b.advantage for b in batch], np.float32)
        old_logits = np.stack([b.old_logits for b in batch])  # (N, NUM_ACTIONS_CHUNK, VOCAB_SIZE)
        v_targ = np.asarray([b.value_target for b in batch], np.float32)
        done = np.asarray([b.done for b in batch], np.bool_)
        # 如果 next_teacher_projector_features 为 None (在 done=True 时)，用零填充
        feasible_b = None
        for b in batch:
            if b.next_teacher_projector_features is not None:
                feasible_b = b
                break
        if feasible_b is None:
            return None
        for b in batch:
            if b.next_teacher_projector_features is None:
                b.next_teacher_projector_features = torch.zeros_like(feasible_b.next_teacher_projector_features)
            elif b.next_teacher_projector_features.shape != feasible_b.next_teacher_projector_features.shape:
                print_str = f"[ERROR] ReplayBufferActor.sample(): next_teacher_projector_features 形状不匹配: {b.next_teacher_projector_features.shape} vs {feasible_b.next_teacher_projector_features.shape}"
                print(print_str, flush=True)
                raise RuntimeError(print_str)
        
        next_teacher_proj_feat = torch.stack([b.next_teacher_projector_features for b in batch])
        reward = np.asarray([b.reward for b in batch], np.float32)
        return obs_list, act, adv, old_logits, v_targ, done, next_teacher_proj_feat, reward

@ray.remote
class ImaginationBufferActor:
    """存储由世界模型想象出的经验，供策略学习使用"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add_batch(self, batch: List[ImaginedExperience]):
        self.buffer.extend(batch)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        
        batch = random.sample(self.buffer, batch_size)
        first_sample = batch[0].multimodal_emb.shape
        for b in batch:
            if b.multimodal_emb.shape != first_sample:
                print(f"b.shape: {b.multimodal_emb.shape}, first_sample shape: {first_sample}")
        
        return {
            "inputs_embeds": torch.stack([b.multimodal_emb for b in batch]),  # stack Tensor
            "attention_mask": np.stack([b.attention_mask for b in batch]),
            "labels": np.stack([b.labels for b in batch]),
            "action": np.stack([b.action for b in batch]),
            "old_logits": np.stack([b.old_logits for b in batch]),
            "advantage": np.array([b.advantage for b in batch], dtype=np.float32),
            "value_target": np.array([b.value_target for b in batch], dtype=np.float32),
            "step_count": np.array([b.step_count for b in batch], dtype=np.int64)
        }

@ray.remote
class RolloutWorkerActor:
    def __init__(self, infer, replay, wid, stats_actor, cfg, benchmark_name, dtype, local_buff_len, gamma, lamb):
        self.infer, self.replay = infer, replay
        self.stats_actor = stats_actor
        self.cfg = cfg
        self.processor = get_processor(cfg)
        self.benchmark_name = benchmark_name
        self.dtype = dtype
        self.local_buff_len = local_buff_len
        self.gamma = gamma
        self.lamb = lamb
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
            time_start = time.time()

            while True:
                inputs_t = prepare_one_obs(self.cfg, self.processor, obs, self.task_description, self.dtype)
                inputs_t['step_count'] = torch.tensor([step_count], dtype=torch.long)
                (action_token, continuous_action, logits, value, teacher_proj_features) = ray.get(self.infer.request.remote(inputs_t))

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
                
                # 存储所有信息，包括教师信号和结束标志
                self.local_buffer.append((
                    inputs_t, action_token, chunk_reward, logits, value, teacher_proj_features, done
                ))
                obs = nxt 
                if done:
                    step_time = (time.time() - time_start) / max(step_count, 1)
                    success = float(info.get('is_success', 0.0))

                    # 如果任务失败，增加对应环境的权重
                    if success < 1.0:
                        self.env_weights[self.current_env_idx] += 1

                    self.stats_actor.add_episode_return.remote(
                        self.current_env_name, reward_sum, step_time, step_count, success
                    )
                    if self.local_buffer:
                        # 最后一个状态的价值为0，没有下一个状态的特征
                        self._process_traj(self.local_buffer, 0.0, None)
                    self.local_buffer.clear()
                    
                    current_seed = int(time.time() * 1000) + self.wid + os.getpid()
                    obs, info = self._reset_and_select_env(seed=current_seed)
                    reward_sum = 0.0
                    step_count = 0
                    time_start = time.time()
                elif len(self.local_buffer) == self.local_buff_len + 1:
                    _, _, _, _, bootstrap_val, bootstrap_proj_feat, _ = self.local_buffer[-1]
                    self._process_traj(self.local_buffer[:-1], bootstrap_val, bootstrap_proj_feat)
                    self.local_buffer = [self.local_buffer[-1]]
        except Exception as e:
            import traceback
            print(f"[ERROR] RolloutWorker {self.wid} run() 崩溃: {e}", flush=True)
            traceback.print_exc()
            raise

    def _process_traj(self, traj_segment, bootstrap_val, bootstrap_proj_features):
        rets, advs = [], []
        gae = 0.0
        # 从后向前计算 GAE
        for i in reversed(range(len(traj_segment))):
            _, _, r, _, v, _, done_flag = traj_segment[i]
            nv = bootstrap_val if i == len(traj_segment) - 1 else traj_segment[i+1][4]
            next_val = nv * (1.0 - float(done_flag))
            delta = r + self.gamma * next_val - v
            gae = delta + self.gamma * self.lamb * gae * (1.0 - float(done_flag))
            advs.append(gae)
            rets.append(gae + v)
        advs.reverse(); rets.reverse()
        advs_np = np.array(advs, dtype=np.float32)

        batch: List[Experience] = []
        for i, (s, action_token, rew, logits_val, _, _, done) in enumerate(traj_segment):
            if i < len(traj_segment) - 1:
                next_teacher_features = traj_segment[i+1][5] 
            else:
                next_teacher_features = bootstrap_proj_features if not done else None
            
            batch.append(
                Experience(
                    obs=s,
                    action=action_token.astype(np.int64),  # 离散动作token
                    advantage=float(advs_np[i]),
                    old_logits=logits_val.astype(np.float32),
                    value_target=float(rets[i]),
                    done=done,
                    next_teacher_projector_features=next_teacher_features if next_teacher_features is None else next_teacher_features.to(torch.bfloat16),
                    reward=rew,
                )
            )
        self.replay.add_batch.remote(batch)

# ================================================================
# 3. 推理器 (InferenceActor)
# ================================================================
@ray.remote(num_gpus=1)
class InferenceActor:
    def __init__(self, actor_id, cfg, dtype, infer_bs, infer_timeout, freeze_value, max_len, param_server, stats_actor, update_interval=3):
        self.actor_id = actor_id
        self.param_server = param_server
        self.stats_actor = stats_actor
        self.current_version = -1
        self.update_interval = update_interval  # 每N次循环更新一次权重
        self.loop_counter = 0  # 循环计数器
        
        # 加载学生模型(WorldModel)
        print(f"InferenceActor {actor_id}: 正在加载 WorldModel (学生)...")
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

            # 每N次循环更新一次权重
            self.loop_counter += 1
            if self.loop_counter % self.update_interval == 0:
                await self.maybe_update_weights()

            requests_to_process = self.requests
            promises_to_process = self.promises
            self.requests, self.promises = [], []
            self.last_process_time = time.time()

            try:
                t_start = time.time()
                inputs_batch = self.model.prepare_inputs_batch(requests_to_process, self.max_len)
                with torch.inference_mode():
                    logits, value, teacher_proj_features = self.model.agent_super_forward(inputs_batch, return_vit_out=True)
                
                # 使用agent的post_process方法获取离散和连续动作
                action_tokens = []
                normalized_actions = []
                _, action_tokens, normalized_actions = self.model.post_process(logits, deterministic=[False]*logits.shape[0])
                action_tokens = action_tokens.detach().cpu().numpy()
                logits_np = logits.to(torch.float32).detach().cpu().numpy()
                values_np = value.to(torch.float32).detach().cpu().numpy()
                
                # teacher_proj_features保持为bf16 Tensor，转到CPU
                teacher_proj_features_cpu = teacher_proj_features.to(torch.bfloat16).cpu()
                
                # 将标准化动作转换为环境动作
                actions_env = []
                for i in range(normalized_actions.shape[0]):
                    a_env = self.model.vla._unnormalize_actions(normalized_actions[i], self.cfg.unnorm_key)
                    actions_env.append(a_env.astype(np.float32))
                for i in range(len(promises_to_process)):
                    promises_to_process[i].set_result((
                        action_tokens[i],           # 离散动作token
                        actions_env[i],      # 连续动作（用于环境）
                        logits_np[i],              # logits (NUM_ACTIONS_CHUNK, VOCAB_SIZE)
                        values_np[i],              # 价值估计
                        teacher_proj_features_cpu[i] # 教师视觉特征 (bf16 Tensor)
                    ))
                
                # 记录处理时间
                process_time = time.time() - t_start
                self.stats_actor.add_perf_metric.remote(f"inference_{self.actor_id}_process_time", process_time)
                
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
        inputs_t['step_count'] = torch.tensor([0], dtype=torch.long)
        inputs_batch = self.model.prepare_inputs_batch([inputs_t])
        inputs_batch['this_action'] = torch.zeros((1, chunk_num, action_dim), dtype=self.dtype).cuda()
        with torch.no_grad():
            self.model(inputs_batch)
    
    async def maybe_update_weights(self):
        """异步拉取最新权重（如果有更新）- 优化版本，先检查版本号"""
        t_start = time.time()
        # 1. 先获取版本号
        latest_version = await self.param_server.get_version.remote()
        
        # 2. 检查是否需要更新
        if latest_version == self.current_version:
            return  # 已经是最新版本，无需更新
        
        if latest_version < 0:
            return  # 还没有发布任何权重
        
        # 3. 只有在需要更新时才拉取权重
        weights_ref = await self.param_server.get_weights.remote()
        if weights_ref is None:
            return  # 安全检查
        
        # 4. 获取并加载新权重
        new_weights = weights_ref
        self._load_trainable_weights(new_weights)
        self.current_version = latest_version
        
        # 记录更新时间
        update_time = time.time() - t_start
        self.stats_actor.add_perf_metric.remote(f"inference_{self.actor_id}_update_time", update_time)
        
        if random.random() < 0.1:  # 10%概率打印日志
            print(f"InferenceActor {self.actor_id}: 已更新到版本 {latest_version}, 耗时 {update_time:.3f}s")
    
    def _load_trainable_weights(self, state_dict: Dict[str, torch.Tensor]):
        """只加载requires_grad=True的参数"""
        model_state = self.model.state_dict()
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in state_dict:
                model_state[name].copy_(state_dict[name])
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()


@ray.remote(num_gpus=1)
class ImaginationRolloutActor:
    """这个Actor现在只负责生成想象数据，并将其分发到多个Buffer中"""
    def __init__(self, actor_id: int, cfg: GenerateConfig, real_replay_buffer: ray.actor.ActorHandle, 
                 imagination_buffers: List[ray.actor.ActorHandle], param_server: ray.actor.ActorHandle, stats_actor: ray.actor.ActorHandle):
        self.actor_id = actor_id
        self.cfg = cfg
        self.real_replay_buffer = real_replay_buffer
        self.imagination_buffers = imagination_buffers
        self.num_buffers = len(imagination_buffers)
        self.param_server = param_server
        self.stats_actor = stats_actor
        
        # 权重版本追踪
        self.current_version = -1
        
        print(f"ImaginationRolloutActor {actor_id}: 正在加载 WorldModel...")
        self.model = WorldModel(cfg, torch_dtype=TORCH_DTYPE, checkpoint_dir=cfg.checkpoint2, freeze_value=False)
        self.model.cuda()
        self.model.eval()
        
        self.generation_batch_size = 32 * self.num_buffers # 一次生成足够分发给所有buffer的轨迹
        print(f"ImaginationRolloutActor {self.actor_id} 初始化于 GPU: {ray.get_gpu_ids()}, 将分发数据到 {self.num_buffers} 个Buffer。")

    async def maybe_update_weights(self):
        """在每次生成想象数据之前，检查并更新权重 - 优化版本，先检查版本号"""
        t_start = time.time()
        # 1. 先获取版本号
        latest_version = await self.param_server.get_version.remote()
        
        # 2. 检查是否需要更新
        if latest_version == self.current_version:
            return  # 已经是最新版本，无需更新
        
        if latest_version < 0:
            return  # 还没有发布任何权重
        
        # 3. 只有在需要更新时才拉取权重
        weights_ref = await self.param_server.get_weights.remote()
        if weights_ref is None:
            return  # 安全检查
        
        # 4. 获取并加载新权重
        new_weights = weights_ref
        self._load_trainable_weights(new_weights)
        self.current_version = latest_version
        
        # 记录更新时间
        update_time = time.time() - t_start
        self.stats_actor.add_perf_metric.remote(f"imagination_{self.actor_id}_update_time", update_time)
        if random.random() < 0.1:  # 10%概率打印日志
            print(f"ImaginationRolloutActor {self.actor_id}: 已更新到版本 {latest_version}, 耗时 {update_time:.3f}s")
    
    def _load_trainable_weights(self, state_dict: Dict[str, torch.Tensor]):
        """只加载requires_grad=True的参数"""
        model_state = self.model.state_dict()
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in state_dict:
                model_state[name].copy_(state_dict[name])
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    async def run_generation_loop(self):
        print(f"ImaginationRolloutActor {self.actor_id}: 想象数据生成循环已启动。")
        while True:
            try:
                # 0. 在开始生成之前，检查并更新权重
                await self.maybe_update_weights()
                
                # 1. 从真实回放池中采样初始状态
                sample_list = await self.real_replay_buffer.sample.remote(self.generation_batch_size)
                if sample_list is None:
                    print(f"ImaginationRolloutActor {self.actor_id}: 无法从真实回放池采样，等待...")
                    await asyncio.sleep(5)
                    continue
                
                start_obs_list = sample_list[0] 
                old_logits = sample_list[3]
                start_states_batch = self.model.prepare_inputs_batch(start_obs_list, INP_MAX_LEN)

                # 2. 进行想象
                t_imagine_start = time.time()
                with torch.inference_mode():
                    (imagined_logits, imagined_values, imagined_rewards, imagined_dones, 
                     last_value, imagined_actions, imagined_multimodal_embs, 
                     imagined_att_masks, imagined_step_counts) = self.model.imagine(start_states_batch, IMAGINE_MAX_HORIZON, old_logits)
                if random.random() < 0.1:
                    print(f"imagined_multimodal_embs.dtype: {imagined_multimodal_embs.dtype}, shape: {imagined_multimodal_embs.shape}")
                imagine_time = time.time() - t_imagine_start
                self.stats_actor.add_perf_metric.remote(f"imagination_{self.actor_id}_imagine_time", imagine_time)

                # 3. 计算 GAE 和有效性掩码
                imagined_advs, imagined_rets = compute_imagined_gae(imagined_rewards, imagined_values, imagined_dones, last_value, GAMMA, LAMBDA)
                valid_mask, _ = create_validity_mask(imagined_dones)

                # 4. 展平并处理数据
                T, B = imagined_dones.shape
                imagined_multimodal_embs_flat = imagined_multimodal_embs.view(T * B, *imagined_multimodal_embs.shape[2:])
                imagined_att_masks_flat = imagined_att_masks.view(T * B, *imagined_att_masks.shape[2:])
                imagined_actions_flat = imagined_actions.view(T * B, *imagined_actions.shape[2:])
                imagined_logits_flat = imagined_logits.view(T * B, *imagined_logits.shape[2:])
                imagined_advs_flat = imagined_advs.view(T * B)
                imagined_rets_flat = imagined_rets.view(T * B)
                valid_mask_flat = valid_mask.view(T * B)
                labels_np = start_states_batch['labels'].cpu().numpy()
                imagined_step_counts_flat = imagined_step_counts.view(T * B).cpu().numpy()

                # 5. 将有效数据打包成 ImaginedExperience
                all_new_experiences = []
                for i in range(T * B):
                    if valid_mask_flat[i]:
                        exp = ImaginedExperience(
                            multimodal_emb=imagined_multimodal_embs_flat[i].to(torch.bfloat16).cpu(),  # bf16 Tensor
                            attention_mask=imagined_att_masks_flat[i].cpu().numpy(),
                            labels=labels_np[i % B],
                            action=imagined_actions_flat[i].cpu().numpy(),
                            old_logits=imagined_logits_flat[i].cpu().numpy(),
                            advantage=imagined_advs_flat[i].item(),
                            value_target=imagined_rets_flat[i].item(),
                            step_count=int(imagined_step_counts_flat[i])
                        )
                        all_new_experiences.append(exp)
                
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
class TrainerActor:
    def __init__(self, rank, world_size, replay_buffer, imagination_buffer, cfg, param_server, stats_actor):
        self.rank = rank
        self.world_size = world_size
        self.replay_buffer = replay_buffer
        self.imagination_buffer = imagination_buffer
        self.cfg = cfg
        self.param_server = param_server
        self.stats_actor = stats_actor
        self.model = None
        self.optimizer = None
        self.base_model = None
        self.data_dtype = None
        
        self.next_wm_batch = None
        self.next_policy_batch = None
        
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

        print(f"Trainer {self.rank}: 正在加载 OpenVLA WorldModel...")
        model = WorldModel(self.cfg, torch_dtype=TORCH_DTYPE, checkpoint_dir=self.cfg.checkpoint2, freeze_value=False)
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
        """后台任务：拉取完整宏批次的世界模型训练数据到CPU"""
        print(f"Trainer {self.rank}: (WM)后台数据准备循环已启动。")
        while True:
            try:
                # 等待上一个批次被消费
                if self.next_wm_batch is not None:
                    await asyncio.sleep(0.1)
                    continue

                # 等待buffer有足够数据
                required_samples = TRAIN_BATCH_SIZE * WORLD_ACCUM
                while await self.replay_buffer.size.remote() < required_samples:
                    await asyncio.sleep(1)

                # 一次性采样整个宏批次
                t_sample_start = time.time()
                sampled_data = await self.replay_buffer.sample.remote(required_samples)
                if sampled_data is None:
                    continue
                
                obs_list, act_np, _, _, _, done_np, next_teacher_proj_feat_tensor, reward_np = sampled_data
                sample_time = time.time() - t_sample_start
                
                # 准备输入批次（在CPU上）
                t_prep_start = time.time()
                inputs_batch = self.base_model.prepare_inputs_batch(obs_list, INP_MAX_LEN)
                prep_time = time.time() - t_prep_start
                
                # 存储完整宏批次（保持在CPU）
                self.next_wm_batch = {
                    'inputs_batch': inputs_batch,
                    'actions': act_np,
                    'dones': done_np,
                    'next_teacher_proj_feat': next_teacher_proj_feat_tensor,
                    'rewards': reward_np,
                    'sample_time': sample_time,
                    'prep_time': prep_time
                }
                
                # 记录性能指标
                if self.rank == 0 and random.random() < 0.1:
                    print(f"Trainer {self.rank}: WM宏批次拉取完成 - 采样: {sample_time:.3f}s, 准备: {prep_time:.3f}s, 样本数: {required_samples}")
                    
            except Exception as e:
                print(f"Trainer {self.rank}: (WM)数据采样失败: {e}。将在3秒后重试。")
                await asyncio.sleep(3)

    async def _policy_data_fetching_loop(self):
        """后台任务：拉取完整宏批次的策略训练数据到CPU"""
        print(f"Trainer {self.rank}: (Policy)后台数据准备循环已启动。")
        while True:
            try:
                # 等待上一个批次被消费
                if self.next_policy_batch is not None:
                    await asyncio.sleep(0.1)
                    continue

                # 等待buffer有足够数据
                required_samples = TRAIN_BATCH_SIZE * AGENT_ACCUM
                while await self.imagination_buffer.size.remote() < required_samples:
                    await asyncio.sleep(1)

                # 一次性采样整个宏批次
                t_sample_start = time.time()
                sampled_data = await self.imagination_buffer.sample.remote(required_samples)
                if sampled_data is None:
                    continue
                
                sample_time = time.time() - t_sample_start
                
                # 转换数据到CPU (保持在CPU内存中)
                t_prep_start = time.time()
                macro_batch = {}
                for k, v in sampled_data.items():
                    if isinstance(v, np.ndarray):
                        macro_batch[k] = torch.tensor(v)
                    elif isinstance(v, torch.Tensor):
                        macro_batch[k] = v.cpu()
                    else:
                        raise ValueError(f"Unsupported data type for key '{k}': {type(v)}")
                prep_time = time.time() - t_prep_start
                
                # 存储完整宏批次（保持在CPU）
                self.next_policy_batch = {
                    'data': macro_batch,
                    'sample_time': sample_time,
                    'prep_time': prep_time
                }
                
                # 记录性能指标
                if self.rank == 0 and random.random() < 0.1:
                    print(f"Trainer {self.rank}: Policy宏批次拉取完成 - 采样: {sample_time:.3f}s, 准备: {prep_time:.3f}s, 样本数: {required_samples}")
                    
            except Exception as e:
                print(f"Trainer {self.rank}: (Policy)数据采样失败: {e}。将在3秒后重试。")
                await asyncio.sleep(3)
    
    def _extract_trainable_weights(self) -> Dict[str, torch.Tensor]:
        """提取所有requires_grad=True的参数"""
        module = self.model.module if hasattr(self.model, 'module') else self.model
        
        # 使用 DeepSpeed 的 GatheredParameters 上下文管理器
        zero_ctx = getattr(deepspeed.zero, "GatheredParameters", None)
        if zero_ctx is None:
            zero_ctx = contextlib.nullcontext
        
        state_dict = {}
        with zero_ctx(module.parameters(), modifier_rank=0):
            if self.rank == 0:  # 只在 rank 0 上收集
                for name, param in module.named_parameters():
                    if param.requires_grad:
                        state_dict[name] = param.detach().cpu().clone()
        
        return state_dict

    async def run_training_epoch(self) -> Tuple[Dict[str, float], Dict[str, int], int]:
        perf_timings = {}
        
        # 等待数据准备完成
        t_wait_data_start = time.time()
        while self.next_wm_batch is None or self.next_policy_batch is None:
            await asyncio.sleep(0.1)
        perf_timings["data_wait_time"] = time.time() - t_wait_data_start

        # 获取批次数据并立即设为None以触发后台拉取
        wm_batch = self.next_wm_batch
        policy_batch = self.next_policy_batch
        
        # 记录数据准备时间到 perf_timings
        perf_timings['wm_sample_time'] = wm_batch['sample_time']
        perf_timings['wm_prep_time'] = wm_batch['prep_time']
        perf_timings['policy_sample_time'] = policy_batch['sample_time']
        perf_timings['policy_prep_time'] = policy_batch['prep_time']
        
        self.next_wm_batch = None
        self.next_policy_batch = None

        # 更新学习率
        current_lrs = {}
        world_lr = self._get_current_lr(self.global_step, WORLD_LR, WORLD_WARMUP_STEPS, TRAIN_ITERS)
        policy_lr = self._get_current_lr(self.global_step, POLICY_LR, POLICY_WARMUP_STEPS, TRAIN_ITERS, start_step=POLICY_TRAIN_START_STEP)
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'world': 
                param_group['lr'] = world_lr
                current_lrs['world'] = world_lr
            elif param_group['name'] == 'policy': 
                param_group['lr'] = policy_lr
                current_lrs['policy'] = policy_lr
        
        epoch_losses = defaultdict(list)
        self.model.train()
        device = self.model.device

        # === 阶段 1: 世界模型训练 ===
        t_wm_start = time.time()
        wm_to_gpu_times = []
        
        # 准备CPU上的完整数据
        wm_data_cpu = {
            'inputs_batch': wm_batch['inputs_batch'],
            'actions': wm_batch['actions'],
            'dones': wm_batch['dones'],
            'next_teacher_proj_feat': wm_batch['next_teacher_proj_feat'],
            'rewards': wm_batch['rewards']
        }
        
        for i in range(WORLD_ACCUM):
            start_idx = i * TRAIN_BATCH_SIZE
            end_idx = (i + 1) * TRAIN_BATCH_SIZE
            
            # 切片并转移到GPU
            t_to_gpu_start = time.time()
            wm_mini_batch = self._slice_and_to_gpu(wm_data_cpu, start_idx, end_idx, device)
            wm_to_gpu_times.append(time.time() - t_to_gpu_start)
            
            # 构造输入
            wm_inp = {**wm_mini_batch['inputs_batch'], 'this_action': wm_mini_batch['actions']}

            # 前向+反向
            ae_loss, rt_loss, reward_acc, reward_mean, termin_acc, termi_mean, rt_acc, mae_loss = \
                self.model.module.compute_world_model_loss(
                    wm_inp, 
                    wm_mini_batch['dones'], 
                    wm_mini_batch['next_teacher_proj_feat'], 
                    wm_mini_batch['rewards']
                )
            
            world_model_loss = AE_LOSS_COEF * ae_loss + RT_LOSS_COEF * rt_loss
            self.model.backward(world_model_loss / WORLD_ACCUM)
            
            # 记录损失
            epoch_losses["ae_loss"].append(ae_loss.item())
            epoch_losses["rt_loss"].append(rt_loss.item())
            epoch_losses["reward_acc"].append(reward_acc.item())
            epoch_losses["reward_mean"].append(reward_mean.item())
            epoch_losses["termin_acc"].append(termin_acc.item())
            epoch_losses["termi_mean"].append(termi_mean.item())
            epoch_losses["rt_classification_acc"].append(rt_acc.item())
            epoch_losses["mae_loss"].append(mae_loss.item())

        perf_timings["wm_train_time"] = time.time() - t_wm_start
        perf_timings["wm_to_gpu_time"] = np.mean(wm_to_gpu_times)

        # === 阶段 2: 策略训练 ===
        t_policy_start = time.time()
        
        # 计算advantage的全局mean和std
        t_adv_stats_start = time.time()
        macro_batch = policy_batch['data']
        adv_t = macro_batch['advantage'].to(device)
        local_sum = adv_t.sum()
        local_sq_sum = (adv_t * adv_t).sum()
        local_count = torch.tensor([adv_t.numel()], dtype=torch.float32).to(device)

        stats_tensor = torch.stack([local_sum, local_sq_sum, local_count.squeeze(0)])
        distributed.all_reduce(stats_tensor, op=distributed.ReduceOp.SUM)

        global_sum, global_sq_sum, global_count = stats_tensor[0], stats_tensor[1], stats_tensor[2]
        global_mean = global_sum / torch.clamp(global_count, min=1.0)
        global_var = torch.clamp(global_sq_sum / torch.clamp(global_count, min=1.0) - global_mean * global_mean, min=1e-12)
        global_std = torch.sqrt(global_var)
        perf_timings["adv_stats_time"] = time.time() - t_adv_stats_start

        # 逐个微批次训练
        policy_to_gpu_times = []
        for i in range(AGENT_ACCUM):
            start_idx = i * TRAIN_BATCH_SIZE
            end_idx = (i + 1) * TRAIN_BATCH_SIZE
            
            # 切片并转移到GPU
            t_to_gpu_start = time.time()
            mini_policy_batch_gpu = self._slice_and_to_gpu(macro_batch, start_idx, end_idx, device)
            policy_to_gpu_times.append(time.time() - t_to_gpu_start)
            
            # 前向传播
            step_count_tensor = mini_policy_batch_gpu['step_count'].to(torch.long)
            logits, value = self.model.module.agent.forward(
                mini_policy_batch_gpu["attention_mask"].to(self.data_dtype), 
                mini_policy_batch_gpu["inputs_embeds"].to(self.data_dtype),
                mini_policy_batch_gpu["labels"],
                step_count_tensor
            )
            
            # 使用全局mean/std归一化advantage
            normalized_adv = (mini_policy_batch_gpu['advantage'] - global_mean) / (global_std + 1e-8)
            
            # 计算损失
            policy_loss, value_loss, entropy_loss, kl_loss, entropy, kl_div_metric = compute_ppo_loss(
                logits, value, 
                mini_policy_batch_gpu['old_logits'],
                mini_policy_batch_gpu['action'],
                normalized_adv, 
                mini_policy_batch_gpu['value_target'], 
                CLIP_EPS, 
                VF_COEF, 
                ENT_COEF,
                KL_COEF
            )
            
            total_policy_loss = policy_loss + value_loss + entropy_loss + kl_loss
            self.model.backward(total_policy_loss / AGENT_ACCUM)
            
            # 记录损失
            epoch_losses["imagination_policy_loss"].append(policy_loss.item())
            epoch_losses["imagination_value_loss"].append(value_loss.item())
            epoch_losses["imagination_entropy_loss"].append(entropy_loss.item())
            epoch_losses["imagination_kl_loss"].append(kl_loss.item())
            epoch_losses["imagination_entropy"].append(entropy.item())
            epoch_losses["imagination_kl_div"].append(kl_div_metric.item())

        perf_timings["policy_train_time"] = time.time() - t_policy_start
        perf_timings["policy_to_gpu_time"] = np.mean(policy_to_gpu_times)

        # === 阶段 3: 优化器步骤 ===
        self.model.step()
        self.global_step += 1
        
        # === 阶段 4: 发布权重到参数服务器（仅 rank 0）===
        if self.rank == 0:
            t_publish_start = time.time()
            trainable_weights = self._extract_trainable_weights()
            weights_ref = ray.put(trainable_weights)
            self.param_server.set_latest_weights.remote(weights_ref, self.global_step)
            perf_timings["publish_weights_time"] = time.time() - t_publish_start
            
            for key, val in perf_timings.items():
                self.stats_actor.add_perf_metric.remote(f"trainer_{self.rank}_{key}", val)

        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        return avg_losses, current_lrs, self.global_step

    def _slice_and_to_gpu(self, data, start_idx, end_idx, device):
        """递归地切片数据并转移到GPU
        
        Args:
            data: 可以是dict, tensor, numpy array, list等
            start_idx: 起始索引
            end_idx: 结束索引
            device: 目标GPU设备
        
        Returns:
            切片后并转移到GPU的数据
        """
        if isinstance(data, dict):
            return {k: self._slice_and_to_gpu(v, start_idx, end_idx, device) for k, v in data.items()}
        elif isinstance(data, torch.Tensor):
            # 如果是批次数据（第一维度等于总样本数），需要切片
            if data.shape[0] >= end_idx:
                return data[start_idx:end_idx].to(device)
            else:
                # 否则只转移到GPU（如全局配置等）
                return data.to(device)
        elif isinstance(data, np.ndarray):
            # numpy数组先切片再转tensor再转GPU
            sliced = data[start_idx:end_idx]
            return torch.from_numpy(sliced).to(device)
        elif isinstance(data, list):
            return [self._slice_and_to_gpu(item, start_idx, end_idx, device) for item in data]
        else:
            # 其他类型（如标量、字符串等）直接返回
            raise ValueError(f"Unsupported data type for slicing and GPU transfer: {type(data)}")
    

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
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
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

    object_store_size_gb = 896 # 分配的GB数
    object_store_memory_bytes = int(object_store_size_gb * 1024 * 1024 * 1024)

    print(f"正在初始化 Ray，并为对象存储分配 {object_store_size_gb} GB 内存...")
    
    ray.init(
        ignore_reinit_error=True, 
        _temp_dir='/dev/shm',
        object_store_memory=object_store_memory_bytes
    )
    exp_name = f"{EXP_NAME}_{int(time.time())}"
    save_dir = f"/cpfs01/lcx_workspace/models/{exp_name}"
    log_dir = f"runs/wm3/{exp_name}"
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    stats_actor = StatsActor.remote(window_size=MOVING_AVG_WINDOW)
    print(f"TensorBoard 日志将保存在: {log_dir}")
    print(f"模型检查点将保存在: {save_dir}")

    cfg = build_openvla_cfg()

    print("--- 步骤 1: 创建参数服务器 ---")
    param_server = ParameterServer.remote()

    print("--- 步骤 2: 创建 Actors ---")
    replay_buffers = [ReplayBufferActor.remote(capacity=REPLAY_CAPACITY) for _ in range(NUM_TRAINER_GPUS)]
    
    imagination_buffers = [ImaginationBufferActor.remote(capacity=IMAGINATION_REPLAY_CAPACITY) for _ in range(NUM_TRAINER_GPUS)]
    imagination_rollout_actors = [
        ImaginationRolloutActor.remote(
            actor_id=i, 
            cfg=cfg, 
            real_replay_buffer=replay_buffers[0],
            imagination_buffers=imagination_buffers,
            param_server=param_server,
            stats_actor=stats_actor
        ) for i in range(NUM_IMAGINATION_ACTORS)
    ]
    
    trainer_group = [
        TrainerActor.remote(
            rank=i, world_size=NUM_TRAINER_GPUS, 
            replay_buffer=replay_buffers[i], 
            imagination_buffer=imagination_buffers[i],
            cfg=cfg,
            param_server=param_server,
            stats_actor=stats_actor
        ) for i in range(NUM_TRAINER_GPUS)
    ]
    inference_pool = [
        InferenceActor.remote(
            actor_id=i, cfg=cfg, dtype=TORCH_DTYPE, infer_bs=INFERENCE_BATCH, 
            infer_timeout=INFERENCE_TIMEOUT_MS, freeze_value=False, max_len=INP_MAX_LEN,
            param_server=param_server,
            stats_actor=stats_actor,
            update_interval=3
        ) for i in range(NUM_INFERENCE_ACTORS)
    ]
    rollout_workers = [
        RolloutWorkerActor.remote(
            inference_pool[i % NUM_INFERENCE_ACTORS],
            replay_buffers[i % NUM_TRAINER_GPUS], i, stats_actor, cfg,
            benchmark_name=BENCHMARK, dtype=TORCH_DTYPE, local_buff_len=ROLLOUT_LOCAL_BUF, gamma=GAMMA, lamb=LAMBDA
        ) for i in range(NUM_ROLLOUT_WORKERS)
    ]

    print("\n--- 正在为训练组查找空闲端口... ---")
    train_group_port = find_free_port()
    print(f"找到端口: 训练组 = {train_group_port}")

    print("\n--- 步骤 3: 建立 DeepSpeed 训练组 ---")
    trainer_master_addr = ray.get(trainer_group[0].get_node_ip.remote())
    ray.get([actor.setup_deepspeed_group.remote(trainer_master_addr, train_group_port) for actor in trainer_group])
    print("DeepSpeed 训练组建立完成。")

    print("\n--- 步骤 4: 初始化参数服务器权重 ---")
    # 从 rank 0 获取初始权重并发布到参数服务器
    initial_weights = ray.get(trainer_group[0]._extract_trainable_weights.remote())
    initial_weights_ref = ray.put(initial_weights)
    ray.get(param_server.set_latest_weights.remote(initial_weights_ref, 0))
    print("初始权重已发布到参数服务器。")

    print("\n--- 步骤 5: 所有推理和想象Actor拉取初始权重 ---")
    all_inference_actors = inference_pool + imagination_rollout_actors
    ray.get([actor.maybe_update_weights.remote() for actor in all_inference_actors])
    print("所有推理和想象Actor已同步初始权重。")
    
    print("\n--- 正在运行前向测试... ---")
    ray.get([inf.forward_test.remote(NUM_ACTIONS_CHUNK, ACTION_DIM) for inf in inference_pool])
    print("所有 Actor 前向测试完成。")

    print("\n--- 步骤 6: 启动 Rollout 和想象数据生成 ---")
    for w in rollout_workers: w.run.remote()
    for actor in imagination_rollout_actors: actor.run_generation_loop.remote()

    print("\n--- 步骤 7: 等待经验池填充 ---")
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
    
    print("\n--- 步骤 8: 开始主训练循环 ---")
    start_time = time.time()
    last_log_time = time.time()
    global_step = 0
    last_saved_step = -1
    while global_step < TRAIN_ITERS:
        t_train_start = time.time()
        train_tasks = [trainer.run_training_epoch.remote() for trainer in trainer_group]
        results = ray.get(train_tasks)
        
        avg_losses_list, lrs_list, steps_list = zip(*results)
        global_step = steps_list[0]
        train_time = time.time() - t_train_start

        # 每 SAVE_INTERVAL_STEPS 步保存一次模型
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
            perf_stats = ray.get(stats_actor.get_perf_stats.remote())
            perf_stats['train_time'] = train_time
            global_stats = all_stats.pop("_global_")
            avg_losses = {k: np.mean([d[k] for d in avg_losses_list]) for k in avg_losses_list[0]}
            for k, v in avg_losses.items(): writer.add_scalar(f'Loss/{k}', v, global_step)
            
            # 记录性能监控指标
            for k, v in perf_stats.items():
                writer.add_scalar(f'Performance/{k}', v, global_step)
            
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
            writer.add_scalar('Performance/train_time_total', time.time() - t_train_start, global_step)
            last_log_time = current_time

    print(f"\n成功完成 {TRAIN_ITERS} 次训练！")
    writer.close()
    ray.shutdown()


if __name__ == "__main__":
    main()