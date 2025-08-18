import os
os.environ["MUJOCO_GL"] = "osmesa"           # 强制软件渲染
os.environ["PYOPENGL_PLATFORM"] = "osmesa"   # 保险起见，给 PyOpenGL 也指明
# 设置临时文件目录，避免磁盘I/O瓶颈
os.environ["TMPDIR"] = "/dev/shm/ray"
# 为了让 Ray 能看到所有可用的 GPU，我们在脚本开头设置。
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6"
# 防止 transformers 库的 tokenizer 并行化警告
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import random
import asyncio
from collections import deque, defaultdict
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

import numpy as np

import ray
import torch
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform
import deepspeed
from torch.utils.tensorboard import SummaryWriter

# Libero env 与工具

# OpenVLA 组件与常量
from experiments.robot.openvla_utils import (
    get_processor,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK

from experiments.robot.libero.run_libero_eval import GenerateConfig

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
NUM_ROLLOUT_WORKERS = 20
ROLLOUT_LOCAL_BUF = 64
INFERENCE_BATCH = 8
INFERENCE_TIMEOUT_MS = 300
REPLAY_CAPACITY = 1000
TRAIN_BATCH_SIZE = 8
ACCUMULATION_STEPS = 32
TRAIN_ITERS = 100000

# PPO
GAMMA = 0.99
LAMBDA = 0.95
LR = 1e-5
CLIP_EPS = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01

# 奖励缩放
REWARD_SCALE = 1.0

# LR 调度
WARMUP_STEPS = 500

# 日志
MOVING_AVG_WINDOW = 100
LOG_INTERVAL_SECONDS = 10

# 通信组
TRAIN_GROUP_PORT = 29531
BROADCAST_GROUP_NAME = "trainer_to_inference_broadcast"
BROADCAST_GROUP_PORT = 29532

# OpenVLA 加载配置
USE_BF16: bool = True
TORCH_DTYPE = torch.bfloat16 if USE_BF16 else torch.float32
PRETRAINED_CHECKPOINT = "/cpfs01/lcx_workspace/models/openvla-7b-oft-finetuned-libero-spatial-object-goal-10/"

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

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # obs 是 prepare_one_obs 的字典，不能 stack，保持 list 返回
        obs_list = [b.obs for b in batch]
        act = np.stack([b.action for b in batch])  # 标准化动作（tanh 后）
        adv = np.asarray([b.advantage for b in batch], np.float32)
        mu_old = np.stack([b.behaviour_mu for b in batch])
        log_std_old = np.stack([b.behaviour_log_std for b in batch])
        v_targ = np.asarray([b.value_target for b in batch], np.float32)
        return obs_list, act, adv, mu_old, log_std_old, v_targ


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
        # from libero.libero import benchmark

        # benchmark_dict = benchmark.get_benchmark_dict()
        # if self.benchmark_name not in benchmark_dict:
        #     raise ValueError(f"基准 '{self.benchmark_name}' 不存在。可用选项: {list(benchmark_dict.keys())}")
        # task_suite = benchmark_dict[self.benchmark_name]()
        # task_id = int(wid % task_suite.n_tasks)
        # print(f"RolloutWorker {wid} 正在加载任务: {task_id} ({task_suite.get_task(task_id).name})")
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
            obs, info = self.env.reset(seed=self.wid)
            self.task_description = self.env.task_description
            self.current_env_name = self.env.get_name()

            reward_sum = 0.0
            step_count = 0
            time_start = time.time()
            step_count_total = 0

            while True:
                # 2) 用 prepare_one_obs 生成单条样本
                inputs_t = prepare_one_obs(self.cfg, self.processor, obs, self.task_description, TORCH_DTYPE)
                # 3) 发给 InferenceActor：它返回 env 动作（已 unnormalize），以及标准化动作与策略信息
                action_env, action_norm, mu, log_std, value = ray.get(self.infer.request.remote(inputs_t))
                # 环境交互使用 env 动作（已反归一化）
                nxt, r, term, trunc, info = self.env.step(action_env)
                reward_sum += r

                # 训练使用缩放后的奖励
                r_scaled = r * REWARD_SCALE

                step_count += 1
                step_count_total += 1
                # 只在 buffer 存标准化后的动作与策略统计量
                self.local_buffer.append((inputs_t, action_norm, r_scaled, mu, log_std, value))
                obs = nxt

                if term or trunc:
                    step_time = (time.time() - time_start) / max(step_count_total, 1)
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
                    step_count_total = 0
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
        rets, advs = [], []
        gae = 0.0
        for i in reversed(range(len(traj_segment))):
            _, _, r, _, _, v = traj_segment[i]
            nv = bootstrap_val if i == len(traj_segment) - 1 else traj_segment[i+1][5]
            delta = r + GAMMA * nv - v
            gae = delta + GAMMA * LAMBDA * gae
            advs.append(gae)
            rets.append(gae + v)
        advs.reverse(); rets.reverse()
        advs_np = np.array(advs, dtype=np.float32)
        advs_np = (advs_np - np.mean(advs_np)) / (np.std(advs_np) + 1e-8)

        batch: List[Experience] = []
        for i, (s, a_norm, _, mu, log_std, _) in enumerate(traj_segment):
            batch.append(
                Experience(
                    obs=s,
                    action=a_norm.astype(np.float32),
                    advantage=float(advs_np[i]),
                    behaviour_mu=mu.astype(np.float32),
                    behaviour_log_std=log_std.astype(np.float32),
                    value_target=float(rets[i]),
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
                actions_norm = actions_all[:, 0, :].to(torch.float32).detach().cpu().numpy()          # (-1,1)
                mu = mu_all[:, 0, :].to(torch.float32).detach().cpu().numpy()
                log_std = log_std_all[:, 0, :].to(torch.float32).detach().cpu().numpy()
                values = value.to(torch.float32).detach().cpu().numpy()
                # 仅在推理器中将标准化动作转换为环境动作（反归一化）
                actions_env = []
                for i in range(actions_norm.shape[0]):
                    a_env = self.model.vla._unnormalize_actions(actions_norm[i], self.cfg.unnorm_key)
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
        self.data_dtype = None
        self.training_batch: Optional[Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None
        self.data_fetching_task = None
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
        model = ActorCritic(self.cfg, torch_dtype=TORCH_DTYPE)  # 原始 PyTorch 模型
        self.base_model = model  # <-- 改为使用不同的属性名

        ds_config = {
            "train_micro_batch_size_per_gpu": TRAIN_BATCH_SIZE,
            "gradient_accumulation_steps": ACCUMULATION_STEPS,
            "optimizer": {"type": "AdamW", "params": {"lr": LR}},
            "scheduler": {
                "type": "WarmupCosineLR",
                "params": {
                    "total_num_steps": TRAIN_ITERS,
                    "warmup_num_steps": WARMUP_STEPS,
                    "warmup_type": "linear",
                    "warmup_min_ratio": 0.0,
                    "cos_min_ratio": 0.0,
                },
            },
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

        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        self.model, _, _, _ = deepspeed.initialize(model=model, model_parameters=trainable_params, config=ds_config)
        print(f"TrainerActor Rank {self.rank}: DeepSpeed 训练组 (ZeRO-2) 初始化完成。")

        # 后台取数
        self.data_fetching_task = asyncio.get_event_loop().create_task(self._data_fetching_loop())

        n_total = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数量: {n_total:,}, 可训练参数量: {n_trainable:,}")

    async def _data_fetching_loop(self):
        print(f"Trainer {self.rank}: 后台数据准备循环已启动。")
        while True:
            try:
                if await self.replay_buffer.size.remote() < TRAIN_BATCH_SIZE:
                    await asyncio.sleep(3)
                    continue

                # 获取经验
                obs_list, act_np, adv_np, mu_old_np, log_std_old_np, v_targ_np = await self.replay_buffer.sample.remote(TRAIN_BATCH_SIZE)

                # 准备 batch（右侧 padding + proprio 归一化；放到 ActorCritic 的 device）
                inputs_batch = self.base_model.prepare_inputs_batch(obs_list)

                # 其它张量
                device = next(self.model.parameters()).device
                act_t = torch.tensor(act_np, dtype=torch.float32, device=device)
                adv_t = torch.tensor(adv_np, dtype=torch.float32, device=device)
                mu_old_t = torch.tensor(mu_old_np, dtype=torch.float32, device=device)
                log_std_old_t = torch.tensor(log_std_old_np, dtype=torch.float32, device=device)
                v_targ_t = torch.tensor(v_targ_np, dtype=torch.float32, device=device)

                # 缓存本轮训练 batch
                self.training_batch = (inputs_batch, act_t, adv_t, mu_old_t, log_std_old_t, v_targ_t)

            except Exception as e:
                print(f"Trainer {self.rank}: 数据采样失败: {e}。将在3秒后重试。")
                await asyncio.sleep(3)

    async def train_step(self) -> Tuple[float, float, float, float, float, bool]:
        if self.training_batch is None:
            print(f"Trainer {self.rank}: 首次训练，等待初始数据批次...")
            while self.training_batch is None:
                await asyncio.sleep(0.2)
            print(f"Trainer {self.rank}: 初始数据已收到，开始训练。")

        inputs_batch, act_t, adv_t, mu_old_t, log_std_old_t, v_targ_t = self.training_batch

        # 前向
        actions_all, mu_all, log_std_all, value = self.model(inputs_batch)
        # 仅用第一个 chunk
        mu = mu_all[:, 0, :].to(torch.float32)
        log_std = log_std_all[:, 0, :].to(torch.float32)
        value = value.to(torch.float32)

        # 分布（标准化动作空间，Squashed Gaussian）
        std = torch.exp(log_std)
        base_dist = Normal(mu, std)
        dist = TransformedDistribution(base_dist, [TanhTransform(cache_size=1)])

        # 防数值问题：对经验的标准化动作进行微小裁剪
        epsilon = 1e-6
        clipped_act_t = torch.clamp(act_t, -1.0 + epsilon, 1.0 - epsilon)
        logp = dist.log_prob(clipped_act_t).sum(dim=-1)

        with torch.no_grad():
            std_old = torch.exp(log_std_old_t)
            base_dist_old = Normal(mu_old_t, std_old)
            dist_old = TransformedDistribution(base_dist_old, [TanhTransform(cache_size=1)])
            logp_old = dist_old.log_prob(clipped_act_t).sum(dim=-1)

        ratio = torch.exp(logp - logp_old)
        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_t
        policy_loss = -torch.mean(torch.min(surr1, surr2))
        value_loss = VF_COEF * torch.mean((value - v_targ_t) ** 2)
        ent_loss = -ENT_COEF * torch.mean(base_dist.entropy().sum(dim=-1))
        loss = policy_loss + value_loss + ent_loss

        self.model.backward(loss)
        self.model.step()
        updated = self.model.is_gradient_accumulation_boundary()
        current_lr = self.model.get_lr()[0] if self.model.get_lr() else 0.0

        return loss.item(), policy_loss.item(), value_loss.item(), ent_loss.item(), current_lr, updated

# ================================================================
# 5. 主逻辑
# ================================================================
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
        print(f"错误: OpenVLA checkpoint 路径 '{PRETRAINED_CHECKPOINT}' 不存在。请更新 PRETRAINED_CHECKPOINT。")
        return

    os.environ["RAY_DEDUP_LOGS"] = "0"
    ray.init(ignore_reinit_error=True, _temp_dir='/dev/shm')

    log_dir = f"runs/Libero/{BENCHMARK}/OpenVLA_DS_PPO_bs1024_id5_{int(time.time())}"
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
    min_buffer_size_for_start = TRAIN_BATCH_SIZE * ACCUMULATION_STEPS
    assert min_buffer_size_for_start < REPLAY_CAPACITY
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

        # 广播权重到推理器
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

            total_losses, p_losses, v_losses, e_losses, lrs, _ = zip(*results)
            current_lr = lrs[0]

            elapsed_time = current_time - start_time
            total_buffer_size = sum(ray.get([rb.size.remote() for rb in replay_buffers]))

            print(f"迭代 {i+1}/{TRAIN_ITERS} | 时间: {elapsed_time:.1f}s | "
                  f"全局平均奖励: {avg_return:.2f} | "
                  f"全局平均幕长: {avg_ep_len:.1f} | "
                  f"value loss: {np.mean(v_losses):.4f} | "
                  f"学习率: {current_lr:.7f} | "
                  f"Episodes数量: {total_episodes:,} | "
                  f"Step平均时间: {avg_step_time:.3f}s")

            writer.add_scalar('Train/Learning_Rate', current_lr, i)
            writer.add_scalar('Loss/Total', np.mean(total_losses), i)
            writer.add_scalar('Loss/Policy', np.mean(p_losses), i)
            writer.add_scalar('Loss/Value', np.mean(v_losses), i)
            writer.add_scalar('Loss/Entropy', np.mean(e_losses), i)

            writer.add_scalar('Rollout/_Global/Average_Return', avg_return, i)
            writer.add_scalar('Rollout/_Global/Average_Episode_Length', avg_ep_len, i)
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