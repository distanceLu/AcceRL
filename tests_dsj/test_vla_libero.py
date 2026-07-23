"""
test_vla_libero.py — 独立测试离散 ActorCritic 模型在 LIBERO 上的表现

用法示例:
  CUDA_VISIBLE_DEVICES=7 python test_vla_libero.py \
    --pretrained-checkpoint /mnt/data/lcx2/yanjieworkspace/models/finetune_im/openvla-7b+libero_spatial_no_noops+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--discrete_acts--proprio_state--100000_chkpt \
    --checkpoint2 /mnt/data/lcx2/yanjieworkspace/openvla_oft_rl/runs/wm_reward_denoiser_distill_named/distill_object_teacher_2img_proprio_student_1img_no_proprio/checkpoint_step_100.pt \
    --benchmark libero_spatial \
    --task-id 0 \
    --num-episodes 10 \
    --deterministic

  # 不加载 checkpoint2，只测试 SFT 基础模型
  CUDA_VISIBLE_DEVICES=7 python test_vla_libero.py \
    --pretrained-checkpoint /mnt/data/lcx2/yanjieworkspace/models/finetune_im/openvla-7b+libero_spatial_no_noops+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--discrete_acts--proprio_state--100000_chkpt \
    --no-checkpoint2 \
    --benchmark libero_spatial \
    --task-id 0 \
    --num-episodes 10
"""

import os
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import argparse
import sys
import time
import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.robot.libero.libero_utils import GenerateConfig, TaskSuite
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM
from rl.actor_critic_model_discrete import ActorCritic
from rl.utils import prepare_one_obs
from rl.libero_env import LiberoEnvWrapper, TASK_MAX_STEPS
from experiments.robot.robot_utils import invert_gripper_action, normalize_gripper_action


def build_cfg(args):
    cfg = GenerateConfig(
        pretrained_checkpoint=args.pretrained_checkpoint,
        use_l1_regression=False,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=args.num_images_in_input,
        use_proprio=args.use_proprio,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        unnorm_key=args.benchmark + "_no_noops",
        checkpoint2=args.checkpoint2 if args.checkpoint2 else "",
        use_lora=True,
        lora_rank=32,
        lora_dropout=0.0,
    )
    return cfg


def run_one_episode(model, cfg, env, task_description, torch_dtype, deterministic=True, max_steps=None, episode_idx=0):
    obs, info = env.reset()

    # 保存 reset 后的观测图像用于调试
    save_dir = "./debug_obs"
    os.makedirs(save_dir, exist_ok=True)
    if "full_image" in obs:
        img_path = os.path.join(save_dir, f"reset_obs_ep{episode_idx}.png")
        Image.fromarray(obs["full_image"]).save(img_path)
        print(f"  [debug] reset obs 图像已保存: {img_path}")
    else:
        print(f"  [debug] obs 中没有 full_image 键，可用 keys: {list(obs.keys())}")

    if max_steps is None:
        max_steps = TASK_MAX_STEPS.get(cfg.task_suite_name if hasattr(cfg, 'task_suite_name') else args.benchmark, 220)

    step_count = 0
    reward_sum = 0.0
    replay_images = []

    while True:
        inputs_t = prepare_one_obs(cfg, model.processor, obs, task_description, torch_dtype)
        inputs_batch = model.prepare_inputs_batch([inputs_t])

        with torch.inference_mode():
            action_logits, value = model(inputs_batch)
            _, action_tokens_all, normalized_actions_all = model.post_process(
                action_logits, deterministic=[deterministic]
            )

        action_tokens = action_tokens_all.view(-1, NUM_ACTIONS_CHUNK, ACTION_DIM).cpu().numpy()
        normalized_actions = normalized_actions_all  # (1, NUM_ACTIONS_CHUNK, ACTION_DIM)

        a_env = model.vla._unnormalize_actions(normalized_actions[0], cfg.unnorm_key)

        done = False
        for i in range(len(a_env)):
            single_action = a_env[i].astype(np.float32)
            single_action = normalize_gripper_action(single_action.copy(), binarize=True)
            single_action = invert_gripper_action(single_action)

            nxt, r, term, trunc, info = env.step(single_action)
            replay_images.append(getattr(env, 'last_full_image', None))
            step_count += 1
            reward_sum += r
            if term or trunc:
                done = True
                break

        obs = nxt
        if done:
            break
        if step_count >= max_steps:
            break

    success = float(info.get('is_success', 0.0))
    return success, step_count, reward_sum, replay_images


def main():
    parser = argparse.ArgumentParser(description="测试离散 ActorCritic VLA 模型在 LIBERO 上的表现")
    parser.add_argument('--pretrained-checkpoint', type=str, required=True,
                        help='OpenVLA SFT checkpoint 路径')
    parser.add_argument('--checkpoint2', type=str, default=None,
                        help='蒸馏 checkpoint 路径（不传则只测试 SFT 基础模型）')
    parser.add_argument('--no-checkpoint2', action='store_true',
                        help='显式跳过 checkpoint2 加载')
    parser.add_argument('--benchmark', type=str, default='libero_spatial',
                        choices=['libero_spatial', 'libero_object', 'libero_goal', 'libero_10'],
                        help='LIBERO benchmark 名称')
    parser.add_argument('--task-id', type=int, default=0,
                        help='任务 ID')
    parser.add_argument('--num-episodes', type=int, default=10,
                        help='测试 episode 数量')
    parser.add_argument('--use-bf16', action='store_true', default=True)
    parser.add_argument('--no-bf16', action='store_false', dest='use_bf16')
    parser.add_argument('--use-proprio', action='store_true', default=False)
    parser.add_argument('--num-images-in-input', type=int, default=1)
    parser.add_argument('--deterministic', action='store_true', default=True,
                        help='使用 argmax（确定性）动作')
    parser.add_argument('--stochastic', action='store_false', dest='deterministic',
                        help='使用采样的（随机）动作')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--save-video', action='store_true', default=False,
                        help='保存失败 episode 的视频')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    torch_dtype = torch.bfloat16 if args.use_bf16 else torch.float32

    if args.no_checkpoint2:
        args.checkpoint2 = None

    print("=" * 60)
    print("VLA LIBERO 测试脚本")
    print("=" * 60)
    print(f"Pretrained checkpoint: {args.pretrained_checkpoint}")
    print(f"Checkpoint2: {args.checkpoint2 if args.checkpoint2 else '(不加载)'}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Task ID: {args.task_id}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Deterministic: {args.deterministic}")
    print(f"Use BF16: {args.use_bf16}")
    print(f"Use Proprio: {args.use_proprio}")
    print(f"Num Images: {args.num_images_in_input}")
    print("=" * 60)

    # 1. 构建 cfg
    cfg = build_cfg(args)
    cfg.task_suite_name = args.benchmark

    # 2. 加载模型
    print("\n[1/3] 加载 ActorCritic 模型...")
    t0 = time.time()
    model = ActorCritic(cfg, torch_dtype=torch_dtype)
    model.cuda()
    model.eval()
    print(f"  模型加载完成，耗时 {time.time() - t0:.1f}s")

    # 验证 unnorm_key
    unnorm_key = cfg.unnorm_key
    if unnorm_key not in model.vla.norm_stats:
        alt_key = f"{args.benchmark}_no_noops"
        if alt_key in model.vla.norm_stats:
            unnorm_key = alt_key
            cfg.unnorm_key = unnorm_key
        else:
            print(f"  [错误] unnorm_key '{unnorm_key}' 不在 norm_stats 中！")
            print(f"  可用 keys: {list(model.vla.norm_stats.keys())}")
            return
    print(f"  unnorm_key = {unnorm_key}")

    # 打印 lm_head 信息
    lm_head = model.vla.language_model.lm_head
    print(f"  lm_head: weight={lm_head.weight.shape}, n_action_bins={model.n_action_bins}")

    # 3. 创建环境
    print("\n[2/3] 创建 LIBERO 环境...")
    env = LiberoEnvWrapper(
        benchmark_name=args.benchmark,
        task_id=args.task_id,
        image_size=224,
        render_mode="rgb_array",
    )
    task_description = env.task_description
    max_steps = TASK_MAX_STEPS.get(args.benchmark, 220)
    print(f"  任务: {env.get_name()}")
    print(f"  描述: {task_description}")
    print(f"  最大步数: {max_steps}")

    # 4. 运行测试
    print(f"\n[3/3] 开始测试 ({args.num_episodes} episodes)...")
    print("-" * 60)

    successes = 0
    total_steps = 0
    episode_lens = []
    episode_rewards = []

    for ep in range(args.num_episodes):
        t_start = time.time()
        success, steps, reward_sum, images = run_one_episode(
            model, cfg, env, task_description, torch_dtype,
            deterministic=args.deterministic, max_steps=max_steps,
            episode_idx=ep,
        )
        elapsed = time.time() - t_start

        successes += int(success)
        total_steps += steps
        episode_lens.append(steps)
        episode_rewards.append(reward_sum)

        status = "✓ 成功" if success else "✗ 失败"
        print(f"  Episode {ep+1}/{args.num_episodes}: {status} | "
              f"步数={steps} | 奖励={reward_sum:.2f} | 耗时={elapsed:.1f}s")

        if args.save_video and not success and images:
            try:
                import imageio
                video_dir = "./test_rollouts"
                os.makedirs(video_dir, exist_ok=True)
                video_path = f"{video_dir}/ep{ep+1}_fail_{args.benchmark}_task{args.task_id}.mp4"
                writer = imageio.get_writer(video_path, fps=30)
                for img in images:
                    if img is not None:
                        writer.append_data(img)
                writer.close()
                print(f"    视频已保存: {video_path}")
            except Exception as e:
                print(f"    视频保存失败: {e}")

    # 5. 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    print(f"  成功率: {successes}/{args.num_episodes} = {successes/args.num_episodes*100:.1f}%")
    print(f"  平均步数: {np.mean(episode_lens):.1f} (min={min(episode_lens)}, max={max(episode_lens)})")
    print(f"  平均奖励: {np.mean(episode_rewards):.4f}")
    print(f"  总步数: {total_steps}")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()