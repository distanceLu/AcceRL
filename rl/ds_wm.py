import os
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["TMPDIR"] = "/dev/shm"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

import time
import random
import asyncio
from collections import deque, defaultdict
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import math

import numpy as np
import torch.nn.functional as F

import ray
import torch
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform
import deepspeed
import torch.distributed as distributed
from torch.utils.tensorboard import SummaryWriter

# OpenVLA 和 Libero 工具
from experiments.robot.openvla_utils import get_processor
from prismatic.vla.constants import NUM_ACTIONS_CHUNK
from experiments.robot.libero.libero_utils import GenerateConfig

# --- 修改: 导入 WorldModel 和 原始的 ActorCritic ---
from rl.world_model import WorldModel
from rl.actor_critic_model import ActorCritic
from rl.utils import prepare_one_obs
from ds_com import TrainerActorCom, InferenceActorCom
from storm.functions_losses import SymLogTwoHotLoss

# ================================================================
# 0. 超参数与配置
# ================================================================
# Libero benchmark
BENCHMARK = "libero_spatial"

# 分布式系统参数
NUM_TRAINER_GPUS = 2
NUM_INFERENCE_ACTORS = 1
NUM_ROLLOUT_WORKERS = 10
ROLLOUT_LOCAL_BUF = 64
INFERENCE_BATCH = 2
INFERENCE_TIMEOUT_MS = 300
REPLAY_CAPACITY = 1000
TRAIN_BATCH_SIZE = 24
ACCUMULATION_STEPS = 1
SUPER_BATCH_SIZE = 24
TRAIN_ITERS = 100000

# PPO
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01

# AE 和 IL 损失的系数
TERMINATION_LOSS_COEF = 0.3
REWARD_LOSS_COEF = 0.3

# 奖励缩放
REWARD_SCALE = 1.0

# 学习率调度参数
VALUE_LR = 1e-4
POLICY_LR = 1e-4
VALUE_WARMUP_STEPS = 0
POLICY_WARMUP_STEPS = 0
POLICY_TRAIN_START_STEP = 0

# 日志
MOVING_AVG_WINDOW = 1000
LOG_INTERVAL_SECONDS = 10

# 通信组
TRAIN_GROUP_PORT = 42364
BROADCAST_GROUP_NAME = "trainer_to_inference_broadcast"
BROADCAST_GROUP_PORT = 43265

# OpenVLA 加载配置
USE_BF16: bool = True
TORCH_DTYPE = torch.bfloat16 if USE_BF16 else torch.float32
PRETRAINED_CHECKPOINT = "/cpfs01/lcx_workspace/models/openvla-7b-oft-finetuned-libero-spatial-object-goal-10/"

# ================================================================
# 数据结构
# ================================================================
@dataclass
class Experience:
    obs: Dict[str, torch.Tensor]
    action: np.ndarray                      # 学生动作 (normalized)
    advantage: float
    behaviour_mu: np.ndarray
    behaviour_log_std: np.ndarray
    value_target: float
    done: bool                              # 结束标志
    teacher_action: np.ndarray              # 教师动作 (normalized)
    next_teacher_projector_features: Optional[np.ndarray] # 下一状态的教师视觉特征
    reward: float

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

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs_list = [b.obs for b in batch]
        act = np.stack([b.action for b in batch])
        adv = np.asarray([b.advantage for b in batch], np.float32)
        mu_old = np.stack([b.behaviour_mu for b in batch])
        log_std_old = np.stack([b.behaviour_log_std for b in batch])
        v_targ = np.asarray([b.value_target for b in batch], np.float32)
        done = np.asarray([b.done for b in batch], np.bool_)
        teacher_act = np.stack([b.teacher_action for b in batch])
        # 如果 next_teacher_projector_features 为 None (在 done=True 时)，用零填充
        for b in batch:
            if b.next_teacher_projector_features is not None:
                feasible_b = b
                break
        for b in batch:
            if b.next_teacher_projector_features is None:
                b.next_teacher_projector_features = np.zeros_like(feasible_b.next_teacher_projector_features)
            elif b.next_teacher_projector_features.shape != feasible_b.next_teacher_projector_features.shape:
                print_str = f"[ERROR] ReplayBufferActor.sample(): next_teacher_projector_features 形状不匹配: {b.next_teacher_projector_features.shape} vs {batch[0].next_teacher_projector_features.shape}"
                print(print_str, flush=True)
                raise RuntimeError(print_str)
        next_teacher_proj_feat = np.stack([b.next_teacher_projector_features for b in batch])
        reward = np.asarray([b.reward for b in batch], np.float32)
        return obs_list, act, adv, mu_old, log_std_old, v_targ, done, teacher_act, next_teacher_proj_feat, reward


@ray.remote
class RolloutWorkerActor:
    def __init__(self, infer, replay, wid, stats_actor, cfg, benchmark_name=BENCHMARK):
        self.infer, self.replay = infer, replay
        self.stats_actor = stats_actor
        self.cfg = cfg
        self.processor = get_processor(cfg)
        self.benchmark_name = benchmark_name
        from rl.libero_env import LiberoEnvWrapper
        from libero.libero import benchmark

        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[self.benchmark_name]()
        task_id = wid % 10
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
                inputs_t = prepare_one_obs(self.cfg, self.processor, obs, self.task_description, TORCH_DTYPE)
                inputs_t['step_count'] = torch.tensor([step_count], dtype=torch.long)  # 添加 step_count 信息
                # --- 修改: 从InferenceActor获取更多信息 ---
                (student_action_env, student_action_norm, mu, log_std, value, 
                 teacher_action_norm, teacher_proj_features) = ray.get(self.infer.request.remote(inputs_t))

                chunk_reward = 0.0
                done = False
                for i in range(len(student_action_env)):
                    single_action = student_action_env[i]
                    nxt, r, term, trunc, info = self.env.step(single_action)
                
                    reward_sum += r
                    r_scaled = r * REWARD_SCALE
                    chunk_reward += r_scaled
                
                    step_count += 1
                    if term or trunc:
                        done = True
                        break
                
                # 存储所有信息，包括教师信号和结束标志
                self.local_buffer.append((
                    inputs_t, student_action_norm, chunk_reward, mu, log_std, value,
                    teacher_action_norm, teacher_proj_features, done
                ))
                obs = nxt 
                if done:
                    step_time = (time.time() - time_start) / max(step_count, 1)
                    success = float(info.get('is_success', 0.0))
                    self.stats_actor.add_episode_return.remote(
                        self.current_env_name, reward_sum, step_time, step_count, success
                    )
                    if self.local_buffer:
                        # 最后一个状态的价值为0，没有下一个状态的特征
                        self._process_traj(self.local_buffer, 0.0, None)
                    self.local_buffer.clear()
                    
                    obs, info = self.env.reset()
                    self.task_description = self.env.task_description
                    self.current_env_name = self.env.get_name()
                    reward_sum = 0.0
                    step_count = 0
                    time_start = time.time()
                elif len(self.local_buffer) == ROLLOUT_LOCAL_BUF + 1:
                    # 使用最后一个状态的信息进行引导
                    _, _, _, _, _, bootstrap_val, _, bootstrap_proj_feat, _ = self.local_buffer[-1]
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
            _, _, r, _, _, v, _, _, _ = traj_segment[i]
            # 下一个状态的价值
            nv = bootstrap_val if i == len(traj_segment) - 1 else traj_segment[i+1][5]
            delta = r + GAMMA * nv - v
            gae = delta + GAMMA * LAMBDA * gae
            advs.append(gae)
            rets.append(gae + v)
        advs.reverse(); rets.reverse()
        advs_np = np.array(advs, dtype=np.float32)

        batch: List[Experience] = []
        for i, (s, a_norm, rew, mu, log_std, _, teacher_a, _, done) in enumerate(traj_segment):
            # 获取下一个状态的教师视觉特征
            if i < len(traj_segment) - 1:
                # 从轨迹的下一个时间步获取
                next_teacher_features = traj_segment[i+1][7] 
            else:
                # 这是段的末尾，使用引导特征 (如果 episode 没结束)
                next_teacher_features = bootstrap_proj_features if not done else None
            
            batch.append(
                Experience(
                    obs=s,
                    action=a_norm.astype(np.float32),
                    advantage=float(advs_np[i]),
                    behaviour_mu=mu.astype(np.float32),
                    behaviour_log_std=log_std.astype(np.float32),
                    value_target=float(rets[i]),
                    done=done,
                    teacher_action=teacher_a.astype(np.float32),
                    next_teacher_projector_features=next_teacher_features.astype(np.float32) if next_teacher_features is not None else None,
                    reward=rew,
                )
            )
        self.replay.add_batch.remote(batch)

# ================================================================
# 3. 推理器 (InferenceActor)
# ================================================================
@ray.remote(num_gpus=1)
class InferenceActor(InferenceActorCom):
    def __init__(self, actor_id, cfg):
        super().__init__()
        self.actor_id = actor_id
        # 加载学生模型(WorldModel)和教师模型(ActorCritic)
        print(f"InferenceActor {actor_id}: 正在加载 WorldModel (学生)...")
        # InferenceActorCom会调用self.model，要更新的模型名字必须是self.model。不要删除本注释！
        self.model = WorldModel(cfg, torch_dtype=TORCH_DTYPE)  
        self.model.cuda()
        self.model.eval()

        print(f"InferenceActor {actor_id}: 正在加载 ActorCritic (教师)...")
        self.teacher_model = ActorCritic(cfg, torch_dtype=TORCH_DTYPE)
        self.teacher_model.cuda()
        self.teacher_model.eval()

        self.processor = self.model.processor
        self.cfg = cfg

        self.batch_size = INFERENCE_BATCH
        self.timeout_sec = INFERENCE_TIMEOUT_MS / 1000.0
        self.requests, self.promises = [], []
        self.last_process_time = time.time()

        loop = asyncio.get_event_loop()
        self._bg_task = loop.create_task(self._loop())
        self._bg_task.add_done_callback(self._on_bg_task_done)
        print(f"InferenceActor {self.actor_id} 初始化于 GPU: {ray.get_gpu_ids()}")

    def get_model_keys(self):
        # 我们只关心学生模型的键，因为它是被训练和同步的
        if self.model is None:
            return {}
        sd = self.model.state_dict()
        return {k: float(v.abs().sum().item()) for k, v in sd.items()}

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
                    student_mu, student_log_std, student_value, _, _, _, _ = self.model.forward(inputs_batch)
                    _, teacher_action_norm_chunks, _, _, teacher_proj_features = self.teacher_model.forward(inputs_batch, return_vit_out=True)
                
                # 只使用第一个动作块
                student_mu_chunk = student_mu.to(torch.float32).detach().cpu().numpy()
                student_log_std_chunk = student_log_std.to(torch.float32).detach().cpu().numpy()
                student_values = student_value.to(torch.float32).detach().cpu().numpy()

                # 将学生动作反归一化以用于环境
                student_actions_env = []
                for i in range(student_mu.shape[0]):
                    # 创建一个分布来采样或直接使用均值
                    dist = TransformedDistribution(Normal(student_mu[i], torch.exp(student_log_std[i])), [TanhTransform(cache_size=1)])
                    action_norm_i = dist.sample() # or student_mu[i] for deterministic action
                    action_norm_i = torch.tanh(student_mu[i]) # 确定性动作
                    a_env = self.model.vla._unnormalize_actions(action_norm_i.cpu().numpy(), self.cfg.unnorm_key)
                    student_actions_env.append(a_env.astype(np.float32))

                # 教师信号
                teacher_action_norm = teacher_action_norm_chunks.to(torch.float32).detach().cpu().numpy()
                teacher_proj_features_np = teacher_proj_features.to(torch.float32).detach().cpu().numpy()

                for i in range(len(promises_to_process)):
                    promises_to_process[i].set_result((
                        student_actions_env[i],      # 用于环境的动作
                        torch.tanh(student_mu)[i].cpu().numpy(), # 学生动作 (normalized, for experience)
                        student_mu_chunk[i],         # 学生策略 mu
                        student_log_std_chunk[i],    # 学生策略 log_std
                        student_values[i],           # 学生价值估计
                        teacher_action_norm[i],      # 教师动作 (normalized)
                        teacher_proj_features_np[i]  # 教师视觉特征
                    ))
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
        inputs_batch = self.model.prepare_inputs_batch([inputs_t])
        with torch.no_grad():
            self.model(inputs_batch)
    

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
        self.old_ready_batch: Optional[Tuple] = None
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
        model = WorldModel(self.cfg, torch_dtype=TORCH_DTYPE)
        self.base_model = model

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
            "zero_optimization": {"stage": 2, "overlap_comm": True, "contiguous_gradients": True},
            "gradient_clipping": 1.0,
        }

        if ds_config.get("bf16", {}).get("enabled", False): self.data_dtype = torch.bfloat16
        else: self.data_dtype = torch.float32

        self.model, self.optimizer, _, _ = deepspeed.initialize(
            model=model, config=ds_config, model_parameters=optimizer_params
        )
        print(f"TrainerActor Rank {self.rank}: DeepSpeed 训练组 (ZeRO-2) 初始化完成。")
        self.data_fetching_task = asyncio.get_event_loop().create_task(self._data_fetching_loop())

    def _get_current_lr(self, current_step, peak_lr, warmup_steps, total_steps, start_step=0):
        if current_step < start_step: return 0.0
        effective_step = current_step - start_step
        if effective_step < warmup_steps:
            return peak_lr * (effective_step / warmup_steps)
        progress = (effective_step - warmup_steps) / (total_steps - start_step - warmup_steps)
        progress = min(progress, 1.0)
        return peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    async def _data_fetching_loop(self):
        print(f"Trainer {self.rank}: 后台数据准备循环已启动。")
        while True:
            try:
                if self.next_ready_batch is not None:
                    await asyncio.sleep(0.1)
                    continue

                while await self.replay_buffer.size.remote() < self.super_batch_size:
                    print(f"Trainer {self.rank} (BG): 等待 ReplayBuffer 填充至 {self.super_batch_size}...")
                    await asyncio.sleep(3)

                obs_list, act_np, adv_np, mu_old_np, log_std_old_np, v_targ_np, done_np, teacher_act_np, next_teacher_proj_feat_np, reward_np = \
                    await self.replay_buffer.sample.remote(self.super_batch_size)

                inputs_batch = self.base_model.prepare_inputs_batch(obs_list)
                device = next(self.model.parameters()).device
                
                # --- 转换所有数据为张量 ---
                act_t = torch.tensor(act_np, dtype=torch.float32, device=device)
                adv_t = torch.tensor(adv_np, dtype=torch.float32, device=device)
                mu_old_t = torch.tensor(mu_old_np, dtype=torch.float32, device=device)
                log_std_old_t = torch.tensor(log_std_old_np, dtype=torch.float32, device=device)
                v_targ_t = torch.tensor(v_targ_np, dtype=torch.float32, device=device)
                done_t = torch.tensor(done_np, dtype=torch.bool, device=device)
                teacher_act_t = torch.tensor(teacher_act_np, dtype=torch.float32, device=device)
                next_teacher_proj_feat_t = torch.tensor(next_teacher_proj_feat_np, dtype=torch.float32, device=device)
                reward_t = torch.tensor(reward_np, dtype=torch.float32, device=device)
                self.next_ready_batch = (inputs_batch, act_t, adv_t, mu_old_t, log_std_old_t, v_targ_t, done_t, teacher_act_t, next_teacher_proj_feat_t, reward_t)

            except Exception as e:
                print(f"Trainer {self.rank}: 数据采样失败: {e}。将在3秒后重试。")
                await asyncio.sleep(3)
    
    async def run_training_epoch(self) -> Tuple[Dict[str, float], Dict[str, float], int]:
        if self.next_ready_batch is None and self.old_ready_batch is None:
            print(f"Trainer {self.rank}: 等待初始超级批次...", flush=True)  # 打印代码不要删
            while self.next_ready_batch is None:
                await asyncio.sleep(0.2)
            print(f"Trainer {self.rank}: 初始数据已收到，开始第一次训练。", flush=True)  # 打印代码不要删

        # 更新学习率
        current_lrs = {}
        value_lr = self._get_current_lr(self.global_step, VALUE_LR, VALUE_WARMUP_STEPS, TRAIN_ITERS)
        policy_lr = self._get_current_lr(self.global_step, POLICY_LR, POLICY_WARMUP_STEPS, TRAIN_ITERS, start_step=POLICY_TRAIN_START_STEP)
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'value': param_group['lr'] = value_lr; current_lrs['value'] = value_lr
            elif param_group['name'] == 'policy': param_group['lr'] = policy_lr; current_lrs['policy'] = policy_lr
        
        if self.next_ready_batch is None:
            current_batch = self.old_ready_batch
        else:
            current_batch = self.next_ready_batch
            self.old_ready_batch = self.next_ready_batch
        self.next_ready_batch = None
        (inputs_batch, act_t, adv_t, mu_old_t, log_std_old_t, v_targ_t, 
         done_t, teacher_act_t, next_teacher_proj_feat_t, reward_t) = current_batch
        
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
        
        epoch_losses = defaultdict(list)
        num_updates_in_epoch = self.super_batch_size // TRAIN_BATCH_SIZE
        
        for i in range(num_updates_in_epoch):
            start = i * TRAIN_BATCH_SIZE; end = start + TRAIN_BATCH_SIZE

            # 切分小批次
            mini_inputs = {k: v[start:end] for k, v in inputs_batch.items()}
            mini_act = act_t[start:end]
            mini_adv = adv_t[start:end]
            mini_mu_old = mu_old_t[start:end]
            mini_log_std_old = log_std_old_t[start:end]
            mini_v_targ = v_targ_t[start:end]
            mini_done = done_t[start:end]
            mini_reward = reward_t[start:end]
            mini_teacher_act = teacher_act_t[start:end]
            mini_next_teacher_proj_feat = next_teacher_proj_feat_t[start:end]
            # 使用全局统计量进行归一化
            normalized_adv = (mini_adv - global_mean) / (global_std + 1e-8)

            # --- 修改: 前向传播和损失计算 ---
            mu, log_std, value, _, _, _, _ = self.model(mini_inputs)
            # 1. PPO 价值损失
            value_loss = VF_COEF * F.mse_loss(value.squeeze(), mini_v_targ)
            
            if self.global_step < POLICY_TRAIN_START_STEP:
                loss = value_loss
                policy_loss = torch.tensor(0.0, device=loss.device)
                ent_loss = torch.tensor(0.0, device=loss.device)
                ent = torch.tensor(0.0, device=loss.device)
                imitation_loss = torch.tensor(0.0, device=loss.device)
                ae_loss = torch.tensor(0.0, device=loss.device)
            else:
                # 2. PPO 策略损失
                dist = TransformedDistribution(Normal(mu, torch.exp(log_std)), [TanhTransform(cache_size=1)])
                logp = dist.log_prob(torch.clamp(mini_act, -1.0 + 1e-6, 1.0 - 1e-6))
                with torch.no_grad():
                    dist_old = TransformedDistribution(Normal(mini_mu_old, torch.exp(mini_log_std_old)), [TanhTransform(cache_size=1)])
                    logp_old = dist_old.log_prob(torch.clamp(mini_act, -1.0 + 1e-6, 1.0 - 1e-6))
                
                ratio = torch.exp(logp - logp_old)
                adv_unsqueezed = normalized_adv.unsqueeze(-1).unsqueeze(-1)
                surr1 = ratio * adv_unsqueezed
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_unsqueezed
                policy_loss = -torch.mean(torch.min(surr1, surr2))
                ent = torch.mean(dist.base_dist.entropy())
                ent_loss = -ENT_COEF * ent

                # 3. 模仿学习损失
                imitation_loss = F.mse_loss(torch.tanh(mu), mini_teacher_act)
                loss1 = imitation_loss
                self.model.backward(loss1)  # backward掉，释放显存
                self.model.step()

                mini_inputs['this_action'] = mini_act  # 用于自编码器损失
                _, _, _, post_patch_proj, _, reward_hat, termi_hat = self.model(mini_inputs)
                non_terminal_mask = ~mini_done.squeeze()
                if torch.any(non_terminal_mask):
                    ae_loss = F.mse_loss(
                        post_patch_proj[non_terminal_mask],
                        mini_next_teacher_proj_feat[non_terminal_mask]
                    )  # 自编码器损失 (仅对非终止状态)
                else:
                    ae_loss = torch.tensor(0.0, device=value_loss.device)
                self.model.symlog_twohot_loss_func: SymLogTwoHotLoss
                reward_loss = self.model.symlog_twohot_loss_func(reward_hat, mini_reward)
                reward_predict = self.model.symlog_twohot_loss_func.decode(reward_hat)
                reward_mae = F.l1_loss(reward_predict, mini_reward)
                reward_mean = mini_reward.mean()
                termi_loss = self.model.bce_with_logits_loss_func(termi_hat.squeeze(), mini_done.float())
                termi_predict = termi_hat > 0
                termi_acc = (termi_predict.squeeze() == mini_done).float().mean()
                termi_mean = mini_done.float().mean()
                loss = ae_loss + REWARD_LOSS_COEF * reward_loss + TERMINATION_LOSS_COEF * termi_loss

            self.model.backward(loss)
            self.model.step()
            epoch_losses["total_loss"].append(loss.item())
            epoch_losses["policy_loss"].append(policy_loss.item())
            epoch_losses["value_loss"].append(value_loss.item())
            epoch_losses["entropy_loss"].append(ent_loss.item())
            epoch_losses["entropy"].append(ent.item())
            epoch_losses["imitation_loss"].append(imitation_loss.item())
            epoch_losses["reward_loss"].append(reward_loss.item())
            epoch_losses["ae_loss"].append(ae_loss.item())
            epoch_losses["termi_loss"].append(termi_loss.item())
            epoch_losses["reward_mae"].append(reward_mae.item())
            epoch_losses["reward_mean"].append(reward_mean.item())
            epoch_losses["termi_acc"].append(termi_acc.item())
            epoch_losses["termi_mean"].append(termi_mean.item())
            
            if self.model.is_gradient_accumulation_boundary():
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
    return cfg


def main():
    if not os.path.exists(PRETRAINED_CHECKPOINT):
        print(f"错误: OpenVLA checkpoint 路径 '{PRETRAINED_CHECKPOINT}' 不存在。")
        return

    ray.init(ignore_reinit_error=True, _temp_dir='/dev/shm')

    log_dir = f"runs/wm/WorldModel_ds_reward_loss_0d3_{int(time.time())}"
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
            replay_buffers[i % NUM_TRAINER_GPUS], i, stats_actor, cfg
        ) for i in range(NUM_ROLLOUT_WORKERS)
    ]

    print("\n--- 步骤 2: 建立 DeepSpeed 训练组 ---")
    trainer_master_addr = ray.get(trainer_group[0].get_node_ip.remote())
    ray.get([actor.setup_deepspeed_group.remote(trainer_master_addr, TRAIN_GROUP_PORT) for actor in trainer_group])
    print("DeepSpeed 训练组建立完成。")

    print(f"\n--- 步骤 3: 建立共享广播组 ---")
    broadcast_participants = [trainer_group[0]] + inference_pool
    broadcast_group_world_size = len(broadcast_participants)
    broadcast_master_addr = ray.get(trainer_group[0].get_node_ip.remote())
    ray.get([
        actor.setup_broadcast_group.remote(
            master_addr=broadcast_master_addr, master_port=BROADCAST_GROUP_PORT,
            group_name=BROADCAST_GROUP_NAME, group_world_size=broadcast_group_world_size,
            my_rank_in_group=rank) for rank, actor in enumerate(broadcast_participants)
    ])
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

    print("\n--- 步骤 4: 启动 Rollout Workers ---")
    for w in rollout_workers: w.run.remote()

    print("\n--- 步骤 5: 等待经验池填充 ---")
    min_buffer_size_for_start = SUPER_BATCH_SIZE
    assert min_buffer_size_for_start < REPLAY_CAPACITY
    while not all(size >= min_buffer_size_for_start for size in ray.get([rb.size.remote() for rb in replay_buffers])):
        sizes = ray.get([rb.size.remote() for rb in replay_buffers])
        print(f"等待经验池填充... (目标: {min_buffer_size_for_start}, 当前: {sizes})")
        time.sleep(5)
    
    print("\n--- 步骤 6: 开始主训练循环 ---")
    start_time = time.time()
    last_log_time = time.time()
    global_step = 0
    while global_step < TRAIN_ITERS:
        train_tasks = [trainer.run_training_epoch.remote() for trainer in trainer_group]
        results = ray.get(train_tasks)
        
        avg_losses_list, lrs_list, steps_list = zip(*results)
        global_step = steps_list[0]

        broadcast_task = trainer_group[0].broadcast_weights.remote(BROADCAST_GROUP_NAME)
        receive_tasks = [inf.receive_and_update_weights.remote(BROADCAST_GROUP_NAME) for inf in inference_pool]
        ray.get([broadcast_task] + receive_tasks)

        current_time = time.time()
        if current_time - last_log_time > LOG_INTERVAL_SECONDS:
            all_stats = ray.get(stats_actor.get_stats.remote())
            global_stats = all_stats.pop("_global_")
            total_episodes = global_stats["total_episodes_processed"]
            avg_step_time = global_stats["avg_step_time"]            
            # 平均所有训练器的损失
            avg_losses = {}
            for k in avg_losses_list[0].keys():
                v_mean = np.mean([d[k] for d in avg_losses_list])
                avg_losses[k] = v_mean
                writer.add_scalar(f'Loss/{k.capitalize()}', v_mean, global_step)
            current_lrs = lrs_list[0]

            elapsed_time = current_time - start_time
            total_buffer_size = sum(ray.get([rb.size.remote() for rb in replay_buffers]))

            print(f"更新步 {global_step}/{TRAIN_ITERS} | 时间: {elapsed_time:.1f}s | "
                  f"奖励: {global_stats['avg_return']:.2f} | 幕长: {global_stats['avg_ep_len']:.1f} | "
                  f"总损失: {avg_losses['total_loss']:.4f} | V Loss: {avg_losses['value_loss']:.4f} | P Loss: {avg_losses['policy_loss']:.4f} | "
                  f"IL Loss: {avg_losses['imitation_loss']:.4f} | AE Loss: {avg_losses['ae_loss']:.4f} | "
                  f"LR(V/P): {current_lrs['value']:.7f}/{current_lrs['policy']:.7f}")

            writer.add_scalar('Train/Learning_Rate/Value', current_lrs['value'], global_step)
            writer.add_scalar('Train/Learning_Rate/Policy', current_lrs['policy'], global_step)
            writer.add_scalar('Loss/Total', avg_losses['total_loss'], global_step)
            writer.add_scalar('Loss/Policy', avg_losses['policy_loss'], global_step)
            writer.add_scalar('Loss/Value', avg_losses['value_loss'], global_step)
            writer.add_scalar('Loss/Imitation', avg_losses['imitation_loss'], global_step)
            writer.add_scalar('Loss/Entropy', avg_losses['entropy_loss'], global_step)
            writer.add_scalar('Loss/AutoEncoder', avg_losses['ae_loss'], global_step)
            writer.add_scalar('Loss/Reward', avg_losses['reward_loss'], global_step)
            writer.add_scalar('Loss/Termination', avg_losses['termi_loss'], global_step)
            writer.add_scalar('Metrics/Entropy', avg_losses['entropy'], global_step)
            writer.add_scalar('Rollout/_Global/Average_Return', global_stats['avg_return'], global_step)
            writer.add_scalar('Rollout/_Global/Average_Episode_Length', global_stats['avg_ep_len'], global_step)
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

    print(f"\n成功完成 {TRAIN_ITERS} 次训练！")
    writer.close()
    ray.shutdown()


if __name__ == "__main__":
    main()