import time
import random
import os
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Dict, Any, List

import csv
import torch
import numpy as np

# Optional logging dependencies
try:
    import swanlab  # type: ignore
    SWANLAB_AVAILABLE = True
    SWANLAB_RUN = None  # type: ignore
except Exception:
    swanlab = None  # type: ignore
    SWANLAB_AVAILABLE = False
    SWANLAB_RUN = None  # type: ignore

try:
    import pandas as pd  # type: ignore
    PANDAS_AVAILABLE = True
except Exception:
    pd = None  # type: ignore
    PANDAS_AVAILABLE = False

from prismatic.vla.constants import NUM_ACTIONS_CHUNK
from rl.libero_env import LiberoEnvWrapper
from rl.utils import prepare_one_obs, check_unnorm_key
from experiments.robot.libero.libero_utils import GenerateConfig, TaskSuite

# ✅ 按你的要求：从 rl/actor_critic_model_discrete.py 导入 ActorCritic
from rl.actor_critic_model_discrete import ActorCritic


def _parse_benchmark(benchmark_name: str) -> Any:
    """
    将命令行传入的基准名称（字符串）映射到 TaskSuite 中的枚举 / 常量。
    例如: --benchmark LIBERO_SPATIAL -> TaskSuite.LIBERO_SPATIAL
    """
    if hasattr(TaskSuite, benchmark_name):
        return getattr(TaskSuite, benchmark_name)

    # 尝试大小写宽松匹配
    candidates = {name.upper(): name for name in dir(TaskSuite) if not name.startswith("_")}
    key = benchmark_name.upper()
    if key in candidates:
        return getattr(TaskSuite, candidates[key])

    available = [name for name in dir(TaskSuite) if not name.startswith("_")]
    raise ValueError(
        f"Unknown benchmark '{benchmark_name}'. "
        f"Available TaskSuite members: {', '.join(available)}"
    )


def run_eval(
    pretrained_checkpoint: str,
    benchmark_name: str,
    device: str,
    envs_num: int = 10,
    use_bf16: bool = True,
    lora_merge_dir: str | None = None,
) -> None:
    """
    等价于你原来的 main 测试逻辑，但改成可复用函数形式。
    """


    print(f"[Debug]device: {device}")
    # ====== 精度设置 ======
    TORCH_DTYPE = torch.bfloat16 if use_bf16 else torch.float32

    # ====== 基准 / 配置 ======
    ENVS_ID = list(range(envs_num))
    BENCHMARK = _parse_benchmark(benchmark_name)
    unnorm_key = f"{BENCHMARK}_no_noops"

    # 生成用于 VLA 的配置（这里使用命令行传入的 pretrained_checkpoint 和 device）
    cfg = GenerateConfig(
        pretrained_checkpoint=pretrained_checkpoint,
        use_l1_regression=False,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        unnorm_key=unnorm_key,
        device=device,
    )

    # ====== 创建策略 ======
    actor = ActorCritic(cfg, TORCH_DTYPE)
    lora_flag = bool(lora_merge_dir)
    print(f"[Debug]use_lora: {lora_flag}, lora_merge_dir: {lora_merge_dir}")
    if lora_merge_dir:
        actor.safe_load_model(lora_merge_dir, strict=True)
    # 这里仅用于验证分组是否完整，不会实际用到 parameter_groups
    parameter_groups = actor.get_parameter_groups()  # noqa: F841

    # 检查 unnorm_key 是否在模型的 norm_stats 中
    check_unnorm_key(cfg, actor.vla)
    actor.eval()

    # 检查参数 dtype 是否一致
    for key, value in actor.named_parameters():
        if value.dtype != TORCH_DTYPE:
            print(f"Warning: Parameter {key} has dtype {value.dtype}, expected {TORCH_DTYPE}.")
    print("策略初始化完成。")

    # ===== SwanLab & Excel setup =====
    pretrained_str = str(cfg.pretrained_checkpoint)
    pretrained_name = os.path.basename(pretrained_str.rstrip("/")) or pretrained_str
    lora_tag = "lora" if lora_flag else "nolora"

    if SWANLAB_AVAILABLE:
        try:
            project_name = f"inference_{pretrained_name}_{lora_tag}"
            exp_name = f"libero_discrete_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            global SWANLAB_RUN  # type: ignore
            SWANLAB_RUN = swanlab.init(
                project=project_name,
                experiment_name=exp_name,
                description=f"original_pretrained_checkpoint={pretrained_str}",
                config={
                    "pretrained_checkpoint": pretrained_str,
                    "use_lora": lora_flag,
                    "lora_rank": getattr(cfg, "lora_rank", None),
                    "benchmark": str(BENCHMARK),
                    "unnorm_key": str(cfg.unnorm_key),
                    "num_images_in_input": cfg.num_images_in_input,
                    "use_proprio": cfg.use_proprio,
                    "device": cfg.device,
                    "envs_num": envs_num,
                    "lora_loaded": bool(lora_merge_dir),
                    "lora_merge_dir": lora_merge_dir,
                    "lora_loaded": bool(lora_merge_dir),
                    "lora_merge_dir": lora_merge_dir,
                },
            )
        except Exception as e:
            print(f"SwanLab 初始化失败: {e}")

    records: List[Dict[str, Any]] = []
    out_dir = Path("./swanlog")
    out_dir.mkdir(parents=True, exist_ok=True)
    excel_path = out_dir / f"inference_{pretrained_name}_{lora_tag}.xlsx"
    csv_path = out_dir / f"inference_{pretrained_name}_{lora_tag}.csv"

    # ===== 初始化环境 =====
    print(f"正在初始化 {len(ENVS_ID)} 个并行的 Libero 环境...")
    envs = [
        LiberoEnvWrapper(
            benchmark_name=BENCHMARK,
            task_id=env_id,
            image_size=224,
            render_mode="rgb_array",
        )
        for env_id in ENVS_ID
    ]
    print("所有环境初始化完成。")

    total_episodes_finished = 0
    total_successes = 0

    # 为每个环境维护一个动作队列
    env_queues = [deque() for _ in range(len(ENVS_ID))]

    # ===================== 主循环 =====================
    while True:
        # 初始化环境状态
        observations = []
        task_descriptions = []
        for i, env in enumerate(envs):
            obs, info = env.reset(seed=int(time.time()) + i)
            observations.append(obs)
            task_descriptions.append(env.task_description)
            print(f"环境 {i}: 任务 ID = {env.task_id}, 任务描述 = {env.task_description}")
            env_queues[i].clear()

        active_envs = [True] * envs_num
        total_rewards = [0.0] * envs_num
        episode_steps = [0] * envs_num
        success_info = [False] * envs_num

        print(f"\n开始第 {total_episodes_finished // envs_num + 1} 轮并行执行...")

        # -------- 环境执行循环 --------
        while any(active_envs):
            # 1. 收集需要生成新动作的环境
            need_generation_indices: List[int] = []
            inputs_t_list: List[Dict[str, Any]] = []

            for i in range(envs_num):
                if active_envs[i] and len(env_queues[i]) == 0:
                    inputs_t = prepare_one_obs(
                        cfg,
                        actor.processor,
                        observations[i],
                        task_descriptions[i],
                        TORCH_DTYPE,
                    )
                    inputs_t_list.append(inputs_t)
                    need_generation_indices.append(i)

            # 2. 为需要生成动作的环境批量推理
            if inputs_t_list:
                inputs_batch = actor.prepare_inputs_batch(inputs_t_list)

                with torch.inference_mode():
                    action_logits, _ = actor.forward(inputs_batch)
                B = action_logits.size(0)
                deterministic_flags = [True] * B  # 如需随机采样改成 False
                _, _, normalized_actions = actor.post_process(
                    action_logits,
                    deterministic_flags,
                )  # 形状 (B, NUM_ACTIONS_CHUNK, ACTION_DIM)

                # 将动作序列分发到各环境队列中
                for idx, env_idx in enumerate(need_generation_indices):
                    action_sequence = normalized_actions[idx]  # (NUM_ACTIONS_CHUNK, ACTION_DIM)
                    env_queues[env_idx].extend(action_sequence)

            # 3. 执行动作
            for i in range(envs_num):
                if not active_envs[i]:
                    continue

                if len(env_queues[i]) == 0:
                    print(f"错误：环境 {i} 动作队列为空但未生成新动作")
                    continue

                action_norm = env_queues[i].popleft()
                action_env = actor.vla._unnormalize_actions(action_norm, cfg.unnorm_key)

                obs, reward, terminated, truncated, info = envs[i].step(action_env)

                observations[i] = obs
                total_rewards[i] += float(reward)
                episode_steps[i] += 1

                if episode_steps[i] % 50 == 0:
                    print(
                        f"环境 {i}, Step: {episode_steps[i]}, 奖励: {reward:.4f}, "
                        f"终止: {terminated}, 截断: {truncated}"
                    )

                if terminated or truncated:
                    is_success = info.get("is_success", False)
                    total_successes += is_success
                    total_episodes_finished += 1
                    success_info[i] = is_success

                    print("-" * 40)
                    print(f"环境 {i} 已完成 (任务: {envs[i].task_description[:50]}...)")
                    print(
                        f"总步数: {episode_steps[i]}, "
                        f"总奖励: {total_rewards[i]:.4f}, "
                        f"是否成功: {is_success}"
                    )
                    print(
                        f"成功率: {total_successes/total_episodes_finished:.3f}, "
                        f"总回合数: {total_episodes_finished}"
                    )
                    print("-" * 40)

                    active_envs[i] = False
                    episode_steps[i] = 0
                    total_rewards[i] = 0.0
                    obs, info = envs[i].reset(seed=random.randint(0, 1000))
                    observations[i] = obs
                    env_queues[i].clear()

        # -------- 每一轮结束统计 --------
        print("=" * 60)
        print(f"第 {total_episodes_finished // envs_num} 轮完成!")
        print(f"累计总回合数: {total_episodes_finished}, 成功次数: {total_successes}")
        print(f"总体成功率: {total_successes/total_episodes_finished:.3f}")
        print("=" * 60)

        round_idx = total_episodes_finished // envs_num
        current_round_successes = sum(1 for s in success_info if s)
        current_round_success_rate = current_round_successes / float(envs_num)
        global_success_rate = (
            total_successes / float(total_episodes_finished)
            if total_episodes_finished > 0
            else 0.0
        )

        # 每 5 轮记录一次
        if round_idx > 0 and round_idx % 5 == 0:
            if SWANLAB_AVAILABLE and SWANLAB_RUN is not None:
                try:
                    SWANLAB_RUN.log(
                        {
                            "round_idx": round_idx,
                            "round_success_rate": current_round_success_rate,
                            "round_successes": current_round_successes,
                            "global_success_rate": global_success_rate,
                            "total_successes": total_successes,
                            "total_episodes": total_episodes_finished,
                        },
                        step=round_idx,
                    )
                except Exception as e:
                    print(f"SwanLab 记录失败: {e}")

            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "round_idx": round_idx,
                "round_success_rate": current_round_success_rate,
                "round_successes": current_round_successes,
                "global_success_rate": global_success_rate,
                "total_successes": total_successes,
                "total_episodes": total_episodes_finished,
                "pretrained_checkpoint": pretrained_str,
                "use_lora": lora_flag,
                "lora_rank": getattr(cfg, "lora_rank", None),
                "benchmark": str(BENCHMARK),
                "unnorm_key": str(cfg.unnorm_key),
                "envs_num": envs_num,
            }
            records.append(record)

            saved_path = None
            if PANDAS_AVAILABLE:
                try:
                    df = pd.DataFrame(records)
                    df.to_excel(excel_path, index=False)
                    saved_path = excel_path
                except Exception as e:
                    print(f"写入 Excel 失败，降级为 CSV: {e}")

            if saved_path is None:
                try:
                    write_header = not csv_path.exists()
                    with open(csv_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=list(record.keys()))
                        if write_header:
                            writer.writeheader()
                        writer.writerow(record)
                    saved_path = csv_path
                except Exception as e:
                    print(f"保存记录失败: {e}")

            if saved_path is not None:
                print(f"✅ 记录已保存: {saved_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Libero 并行评测脚本（离散动作 ActorCritic）。\n"
            "示例：\n"
            "  python run_discrete_policy_eval.py "
            "--pretrained_checkpoint /path/to/checkpoint "
            "--benchmark LIBERO_SPATIAL "
            "--device cuda:4"
        )
    )

    parser.add_argument(
        "--pretrained_checkpoint",
        type=str,
        required=True,
        help="用于初始化 GenerateConfig 的预训练 checkpoint 路径",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help="TaskSuite 中的基准名称，例如: LIBERO_SPATIAL / LIBERO_OBJECT / LIBERO_GOAL 等",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="指定使用的 GPU 序号（整数），例如: 0 / 1 / 2；将使用 cuda:{index}",
    )
    parser.add_argument(
        "--envs_num",
        type=int,
        default=10,
        help="并行 Libero 环境数量（默认 10）",
    )
    parser.add_argument(
        "--lora_merge_dir",
        type=str,
        default=None,
        help="LoRA 目录（例如 agent_checkpoint_*）；提供时将加载 LoRA 权重",
    )
    parser.add_argument(
        "--no_bf16",
        action="store_true",
        help="默认使用 bfloat16；如果指定此参数，则改用 float32。",
    )

    args = parser.parse_args()
    print(f"[Debug]args.device: {args.device}")
    run_eval(
        pretrained_checkpoint=args.pretrained_checkpoint,
        benchmark_name=args.benchmark,
        device=f"cuda:{args.device}",
        envs_num=args.envs_num,
        use_bf16=not args.no_bf16,
        lora_merge_dir=args.lora_merge_dir,
    )
