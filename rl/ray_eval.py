import os
os.environ["MUJOCO_GL"] = "osmesa"           # 强制软件渲染
os.environ["PYOPENGL_PLATFORM"] = "osmesa"   # 保险起见，给 PyOpenGL 也指明
# 设置临时文件目录，避免磁盘I/O瓶颈
os.environ["TMPDIR"] = "/dev/shm"
# 为了让 Ray 能看到所有可用的 GPU，我们在脚本开头设置。
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import time
import random
import asyncio
from collections import deque, defaultdict
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np

import ray
import torch
from torch.utils.tensorboard import SummaryWriter

# OpenVLA 组件与常量
from experiments.robot.openvla_utils import (
    get_processor,
)

from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM
from experiments.robot.libero.libero_utils import GenerateConfig, TaskSuite
from rl.actor_critic_model_discrete import ActorCritic
from rl.utils import prepare_one_obs

# ================================================================
# 0. 超参数与配置
# ================================================================
# Libero benchmark
BENCHMARK = TaskSuite.LIBERO_OBJECT

# 分布式系统参数
NUM_INFERENCE_ACTORS = 1
NUM_EVAL_WORKERS = 10
INFERENCE_BATCH = 8
INFERENCE_TIMEOUT_MS = 300

# 日志
MOVING_AVG_WINDOW = 1000
LOG_INTERVAL_SECONDS = 30

# OpenVLA 加载配置
USE_BF16: bool = True
TORCH_DTYPE = torch.bfloat16 if USE_BF16 else torch.float32
PRETRAINED_CHECKPOINT = "/cpfs01/liuwei_workspace/models/finetune_im/openvla-7b+libero_object_no_noops+b40+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--discrete_acts--proprio_state--80000_chkpt"  # 需要修改为实际的checkpoint路径

# ================================================================
# 1. 统计模块 (StatsActor)
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
        eval_returns, eval_lengths, eval_step_times, eval_successes = [], [], [], []
        eval_total_episodes_processed = 0
        eval_total_env_steps = 0

        for env_name, env_data in self.stats.items():
            if not env_data["episode_returns"]:
                per_env_stats[env_name] = { 
                    "avg_return": 0.0, 
                    "avg_ep_len": 0.0, 
                    "avg_success_rate": 0.0, 
                    "num_episodes_in_avg": 0, 
                    "total_episodes": env_data["total_episodes_processed"]}
                continue
            
            per_env_stats[env_name] = {
                "avg_return": np.mean(env_data["episode_returns"]),
                "avg_ep_len": np.mean(env_data["episode_lengths"]),
                "avg_success_rate": np.mean(env_data["successes"]),
                "num_episodes_in_avg": len(env_data["episode_returns"]),
                "total_episodes": env_data["total_episodes_processed"]
            }
            print(env_name, np.mean(env_data["episode_returns"]), env_data["episode_returns"])
            
            eval_total_episodes_processed += env_data["total_episodes_processed"]
            eval_total_env_steps += env_data["total_env_steps"]
            eval_returns.extend(env_data["episode_returns"])
            eval_lengths.extend(env_data["episode_lengths"])
            eval_step_times.extend(env_data["step_times"])
            eval_successes.extend(env_data["successes"])

        per_env_stats["_global_eval_"] = {
            "avg_return": np.mean(eval_returns) if eval_returns else 0.0,
            "avg_ep_len": np.mean(eval_lengths) if eval_lengths else 0.0,
            "avg_step_time": np.mean(eval_step_times) if eval_step_times else 0.0,
            "avg_success_rate": np.mean(eval_successes) if eval_successes else 0.0,
            "total_episodes_processed": eval_total_episodes_processed,
            "total_env_steps": eval_total_env_steps
        }
        return per_env_stats

# ================================================================
# 2. Evaluation Worker
# ================================================================
class BaseWorkerActor:
    """eval worker 的共享逻辑。"""
    def __init__(self, infer, wid, stats_actor, cfg, benchmark_name=BENCHMARK):
        self.infer = infer
        self.stats_actor = stats_actor
        self.cfg = cfg
        # 仅需 processor，Worker 不加载大模型
        self.processor = get_processor(cfg)
        self.benchmark_name = benchmark_name
        from rl.libero_env import LiberoEnvWrapper

        self.num_tasks = 10
        print(f"BaseWorker {wid}: 正在初始化 {self.num_tasks} 个 Libero 环境...")
        self.envs = [
            LiberoEnvWrapper(
                benchmark_name=self.benchmark_name,
                task_id=i,
                image_size=224,
                render_mode="rgb_array"
            ) for i in range(self.num_tasks)]
        print(f"BaseWorker {wid}: 环境初始化完成。")
        
        self.env = None
        self.current_env_idx = -1
        self.wid = wid
        self.task_description = None
        self.current_env_name = None

@ray.remote
class EvaluationWorkerActor(BaseWorkerActor):
    def __init__(self, infer, wid, stats_actor, cfg, benchmark_name=BENCHMARK):
        super().__init__(infer, wid, stats_actor, cfg, benchmark_name)
        print(f"EvaluationWorker {self.wid}: 环境初始化完成。")

    def _reset_and_select_env(self, seed: Optional[int] = None) -> Tuple[Dict, Dict]:
        self.current_env_idx = (self.current_env_idx + 1) % self.num_tasks
        self.env = self.envs[self.current_env_idx]
        obs, info = self.env.reset(seed=seed)
        self.task_description = self.env.task_description
        self.current_env_name = self.env.get_name()
        return obs, info

    def run(self):
        try:
            current_seed = int(time.time() * 1000) + os.getpid() + random.randint(0, 10000)
            obs, info = self._reset_and_select_env(seed=current_seed)
            while True:
                reward_sum, time_start, step_count_total, done = 0.0, time.time(), 0, False
                while not done:
                    inputs_t = prepare_one_obs(self.cfg, self.processor, obs, self.task_description, TORCH_DTYPE)
                    action_env, _, _, _ = ray.get(self.infer.request.remote(inputs_t, deterministic=True))
                    for i in range(len(action_env)):
                        single_action = action_env[i]
                        obs, r, term, trunc, info = self.env.step(single_action)
                        reward_sum += r; step_count_total += 1
                        if term or trunc: 
                            done = True
                            break
                step_time = (time.time() - time_start) / max(step_count_total, 1)
                success = float(info.get('is_success', 0.0))
                self.stats_actor.add_episode_return.remote(f"eval_{self.current_env_name}", reward_sum, step_time, step_count_total, success)
                current_seed = int(time.time() * 1000) + os.getpid() + random.randint(0, 10000)
                obs, info = self._reset_and_select_env(seed=current_seed)
        except Exception as e: import traceback; print(f"[ERROR] EvaluationWorker {self.wid} run() 崩溃: {e}", flush=True); traceback.print_exc(); raise


# ================================================================
# 3. 推理器 (InferenceActor)
# ================================================================
@ray.remote(num_gpus=1)
class InferenceActor:
    def __init__(self, actor_id, cfg):
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

    def _on_bg_task_done(self, task: asyncio.Task):
        try:
            task.result()
        except Exception as e:
            import traceback
            print(f"[ERROR] InferenceActor {self.actor_id} 后台任务异常: {e}", flush=True)
            traceback.print_exc()

    async def request(self, inputs_t: Dict[str, torch.Tensor], deterministic: bool = False):
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        self.requests.append((inputs_t, deterministic))
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
            
            inputs_list = [r[0] for r in requests_to_process]
            deterministic_flags = [r[1] for r in requests_to_process]
            
            try:
                inputs_batch = self.model.prepare_inputs_batch(inputs_list)
                with torch.inference_mode():
                    # 1. 前向传播获取 logits 和 value
                    action_logits, value = self.model(inputs_batch)

                    # 2. 后处理以采样动作 tokens 和对应的归一化连续动作
                    _, action_tokens_all, normalized_actions_all = self.model.post_process(action_logits, deterministic=deterministic_flags)
                    
                    # action_tokens_all 的形状是 (B, NUM_ACTIONS_CHUNK * ACTION_DIM)
                    action_tokens = action_tokens_all.view(
                        -1, NUM_ACTIONS_CHUNK, ACTION_DIM
                    ).cpu().numpy()

                    # action_logits 的形状是 (B, NUM_ACTIONS_CHUNK * ACTION_DIM, VocabSize)
                    logits = action_logits.view(
                        -1, NUM_ACTIONS_CHUNK, ACTION_DIM, action_logits.shape[-1]
                    ).float().cpu().numpy()
                    
                    values = value.to(torch.float32).cpu().numpy()

                # 将标准化动作转换为环境动作
                actions_env = []
                for i in range(normalized_actions_all.shape[0]):
                    a_env = self.model.vla._unnormalize_actions(normalized_actions_all[i], self.cfg.unnorm_key)
                    actions_env.append(a_env.astype(np.float32))

                for i in range(len(promises_to_process)):
                    promises_to_process[i].set_result((
                        actions_env[i],           # 反归一化的环境动作
                        action_tokens[i],         # 离散动作 token
                        logits[i],                # 对应的 logits
                        values[i]                 # 价值估计
                    ))
            except Exception as e:
                import traceback
                print(f"[ERROR] InferenceActor {self.actor_id} 批处理失败: {e}", flush=True)
                traceback.print_exc()
                for p in promises_to_process:
                    if not p.done():
                        p.set_exception(e)
                raise


# ================================================================
# 4. 主逻辑
# ================================================================
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
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        unnorm_key=BENCHMARK+"_no_noops",
    )
    return cfg

def main():
    if not os.path.exists(PRETRAINED_CHECKPOINT):
        print(f"错误: OpenVLA checkpoint 路径 '{PRETRAINED_CHECKPOINT}' 不存在。请更新 PRETRAINED_CHECKPOINT。")
        return

    os.environ["RAY_DEDUP_LOGS"] = "0"
    ray.init(ignore_reinit_error=True, _temp_dir='/dev/shm')

    log_dir = f"runs/Libero/{BENCHMARK}/OpenVLA_Eval_{int(time.time())}"
    writer = SummaryWriter(log_dir)
    stats_actor = StatsActor.remote(window_size=MOVING_AVG_WINDOW)
    print(f"TensorBoard 日志将保存在: {log_dir}")

    cfg = build_openvla_cfg()

    print("--- 步骤 1: 创建 Actors ---")
    inference_pool = [InferenceActor.remote(actor_id=i, cfg=cfg) for i in range(NUM_INFERENCE_ACTORS)]
    eval_workers = [
        EvaluationWorkerActor.remote(
            inference_pool[i % NUM_INFERENCE_ACTORS], f"eval_{i}", stats_actor, cfg
        ) for i in range(NUM_EVAL_WORKERS)
    ]
    print(f"已创建 {NUM_EVAL_WORKERS} 个 Evaluation workers。")

    print("\n--- 步骤 2: 启动 Evaluation Workers 进行评估 ---")
    for w in eval_workers: w.run.remote()

    print("\n--- 步骤 3: 开始评估循环 ---")
    start_time = time.time()
    last_log_time = time.time()

    eval_step = 0
    try:
        while True:
            time.sleep(1)  # 主循环等待
            
            current_time = time.time()
            if current_time - last_log_time > LOG_INTERVAL_SECONDS:
                all_stats = ray.get(stats_actor.get_stats.remote())

                eval_stats = all_stats.pop("_global_eval_")
                eval_avg_return = eval_stats["avg_return"]
                eval_avg_ep_len = eval_stats["avg_ep_len"]
                eval_avg_success_rate = eval_stats["avg_success_rate"]
                eval_total_episodes = eval_stats["total_episodes_processed"]
                eval_env_steps = eval_stats["total_env_steps"]
                eval_avg_step_time = eval_stats["avg_step_time"]

                elapsed_time = current_time - start_time

                print(f"评估步 {eval_step} | 时间: {elapsed_time:.1f}s | "
                      f"平均奖励: {eval_avg_return:.2f} | 平均幕长: {eval_avg_ep_len:.1f} | "
                      f"成功率: {eval_avg_success_rate:.2%} | Episodes数量: {eval_total_episodes:,} | "
                      f"Step平均时间: {eval_avg_step_time:.3f}s")

                writer.add_scalar('Eval/_Global/Average_Return', eval_avg_return, eval_step)
                writer.add_scalar('Eval/_Global/Average_Episode_Length', eval_avg_ep_len, eval_step)
                writer.add_scalar('Eval/_Global/Success_Rate', eval_avg_success_rate, eval_step)
                writer.add_scalar('System/Eval_Total_Episodes_Processed', eval_total_episodes, eval_step)
                writer.add_scalar('System/Eval_Total_Env_Steps', eval_env_steps, eval_step)
                writer.add_scalar('System/Eval_Avg_Step_Time', eval_avg_step_time, eval_step)

                for env_name, env_stats in all_stats.items():
                    tag_prefix = f"Eval/{env_name.replace('eval_', '')}"
                    writer.add_scalar(f'{tag_prefix}/Average_Return', env_stats['avg_return'], eval_step)
                    writer.add_scalar(f'{tag_prefix}/Average_Episode_Length', env_stats['avg_ep_len'], eval_step)
                    writer.add_scalar(f'{tag_prefix}/Success_Rate', env_stats['avg_success_rate'], eval_step)
                    writer.add_scalar(f'{tag_prefix}/Total_Episodes', env_stats['total_episodes'], eval_step)

                last_log_time = current_time
                eval_step += 1

    except KeyboardInterrupt:
        print("\n收到中断信号，正在关闭...")
    finally:
        writer.close()
        ray.shutdown()
        print("评估结束。")


if __name__ == "__main__":
    main()