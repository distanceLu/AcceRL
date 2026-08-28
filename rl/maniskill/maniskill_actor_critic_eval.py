"""Evaluate an OpenVLA ActorCritic checkpoint in a ManiSkill environment.

All task- and checkpoint-specific settings are command-line arguments. Use one
of the companion shell launchers for the historical PickCube setup or the
single-camera DrawTriangle setup.
"""

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--gpu-id", type=str, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-output-dir", type=Path, required=True)

    parser.add_argument("--task-id", type=str, required=True)
    parser.add_argument("--unnorm-key", type=str, required=True)
    parser.add_argument("--language-instruction", type=str, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--num-eval-episodes", type=int, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--base-seed", type=int, required=True)

    parser.add_argument("--camera-name", type=str, required=True)
    parser.add_argument("--wrist-camera-name", type=str, default=None)
    parser.add_argument("--camera-res", type=int, required=True)
    parser.add_argument("--num-images-in-input", type=int, choices=(1, 2), required=True)
    parser.add_argument("--robot-uids", type=str, required=True)
    parser.add_argument("--env-action-dim", type=int, required=True)
    parser.add_argument("--control-mode", type=str, required=True)
    parser.add_argument("--sim-backend", type=str, required=True)
    parser.add_argument("--reward-mode", type=str, required=True)
    parser.add_argument("--render-mode", type=str, required=True)
    parser.add_argument("--exec-actions-per-inference", type=int, required=True)

    parser.add_argument("--use-bf16", action=argparse.BooleanOptionalAction, required=True)
    parser.add_argument("--use-proprio", action=argparse.BooleanOptionalAction, required=True)
    parser.add_argument("--center-crop", action=argparse.BooleanOptionalAction, required=True)
    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction, required=True)
    parser.add_argument("--lora-rank", type=int, required=True)
    parser.add_argument("--lora-dropout", type=float, required=True)
    parser.add_argument("--load-in-8bit", action=argparse.BooleanOptionalAction, required=True)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, required=True)
    parser.add_argument("--use-film", action=argparse.BooleanOptionalAction, required=True)
    parser.add_argument("--enable-pmvt", action=argparse.BooleanOptionalAction, required=True)
    parser.add_argument("--checkpoint2", type=str, required=True)

    parser.add_argument("--record-eval-video", action=argparse.BooleanOptionalAction, required=True)
    parser.add_argument("--record-video-num-episodes", type=int, required=True)
    parser.add_argument(
        "--show-pickcube-goal-in-policy-obs",
        action=argparse.BooleanOptionalAction,
        required=True,
    )
    args = parser.parse_args()

    if args.num_images_in_input == 2 and not args.wrist_camera_name:
        parser.error("--wrist-camera-name is required when --num-images-in-input=2")
    if args.env_action_dim <= 0:
        parser.error("--env-action-dim must be positive")
    if args.exec_actions_per_inference <= 0:
        parser.error("--exec-actions-per-inference must be positive")
    return args


# Device visibility must be configured before importing torch, TensorFlow, or
# ManiSkill/SAPIEN. The selected physical GPU is logical cuda:0 in this process.
ARGS = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = ARGS.gpu_id
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TF_CUDA_VISIBLE_DEVICES"] = ARGS.gpu_id
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "6"
os.environ["VULKAN_VISIBLE_DEVICES"] = ARGS.gpu_id
os.environ["SAPIEN_VULKAN_DEVICE"] = ARGS.gpu_id
os.environ["EGL_DEVICE_ID"] = ARGS.gpu_id

import numpy as np
import torch

_MANISKILL_DIR = Path(__file__).resolve().parent
_RL_DIR = _MANISKILL_DIR.parent
_REPO_ROOT = _RL_DIR.parent
for _p in (_REPO_ROOT, _RL_DIR, _MANISKILL_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
from rl.actor_critic_model_discrete import ActorCritic
from rl.maniskill.maniskill_utils import (
    adapt_maniskill_action,
    extract_done_mask,
    extract_maniskill_observation,
    extract_success_mask,
)
from rl.utils import check_unnorm_key, prepare_one_obs


def extract_policy_observation(obs, env_idx, args):
    return extract_maniskill_observation(
        obs,
        env_idx=env_idx,
        camera_name=args.camera_name,
        use_proprio=args.use_proprio,
        wrist_camera_name=args.wrist_camera_name,
        include_wrist_image=args.num_images_in_input > 1,
    )


def main(args):
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {args.checkpoint}")
    if args.exec_actions_per_inference > NUM_ACTIONS_CHUNK:
        raise ValueError(
            f"--exec-actions-per-inference={args.exec_actions_per_inference} exceeds "
            f"NUM_ACTIONS_CHUNK={NUM_ACTIONS_CHUNK}"
        )

    torch_dtype = torch.bfloat16 if args.use_bf16 else torch.float32
    cfg = SimpleNamespace(
        pretrained_checkpoint=str(args.checkpoint),
        use_l1_regression=False,
        use_diffusion=False,
        use_film=args.use_film,
        num_images_in_input=args.num_images_in_input,
        use_proprio=args.use_proprio,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        center_crop=args.center_crop,
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        unnorm_key=args.unnorm_key,
        device=torch.device("cuda"),
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_dropout=args.lora_dropout,
        checkpoint2=args.checkpoint2,
        enable_pmvt=args.enable_pmvt,
    )

    actor = ActorCritic(cfg, torch_dtype)
    check_unnorm_key(cfg, actor.vla)
    actor.eval()
    print("策略初始化完成。")

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from mani_skill.utils.wrappers import RecordEpisode

    sensor_configs = {
        args.camera_name: {"width": args.camera_res, "height": args.camera_res}
    }
    if args.num_images_in_input > 1:
        sensor_configs[args.wrist_camera_name] = {
            "width": args.camera_res,
            "height": args.camera_res,
        }

    env = gym.make(
        args.task_id,
        obs_mode="rgbd",
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        robot_uids=args.robot_uids,
        num_envs=args.num_envs,
        sim_backend=args.sim_backend,
        render_mode=args.render_mode,
        sensor_configs=sensor_configs,
        max_episode_steps=args.max_steps,
    )
    actual_action_dim = int(env.action_space.shape[-1])
    if actual_action_dim != args.env_action_dim:
        env.close()
        raise ValueError(
            f"Configured --env-action-dim={args.env_action_dim}, but "
            f"{args.task_id} exposes action dimension {actual_action_dim}"
        )

    if args.show_pickcube_goal_in_policy_obs:
        if args.task_id != "PickCube-v1":
            raise ValueError("The PickCube goal-visibility option is only valid for PickCube-v1")
        base_env = env.unwrapped
        goal_site = getattr(base_env, "goal_site", None)
        hidden_objects = getattr(base_env, "_hidden_objects", None)
        if goal_site is None or hidden_objects is None:
            raise RuntimeError(
                "Cannot expose PickCube goal marker: ManiSkill environment "
                "does not provide goal_site/_hidden_objects."
            )
        base_env._hidden_objects = [obj for obj in hidden_objects if obj is not goal_site]
        goal_site.show_visual()
        print("PickCube green goal marker enabled in policy camera observations.")

    video_task_dir = args.task_id
    if args.exec_actions_per_inference != NUM_ACTIONS_CHUNK:
        video_task_dir += f"_exec{args.exec_actions_per_inference}"
    video_dir = args.eval_output_dir / "eval_videos" / video_task_dir
    if args.record_eval_video:
        video_dir.mkdir(parents=True, exist_ok=True)
        record_step_limit = args.record_video_num_episodes * args.max_steps * args.num_envs
        env = RecordEpisode(
            env,
            output_dir=str(video_dir),
            save_trajectory=False,
            save_video=True,
            info_on_video=True,
            save_video_trigger=lambda elapsed_steps: elapsed_steps < record_step_limit,
            max_steps_per_video=args.max_steps,
            avoid_overwriting_video=True,
        )
        print(
            f"Eval video recording enabled: {video_dir} "
            f"(up to first {args.record_video_num_episodes} episodes)"
        )

    try:
        obs, _ = env.reset()
        available_cameras = list(obs["sensor_data"].keys())
        expected_cameras = [args.camera_name]
        if args.num_images_in_input > 1:
            expected_cameras.append(args.wrist_camera_name)
        missing_cameras = [name for name in expected_cameras if name not in available_cameras]
        if missing_cameras:
            raise RuntimeError(
                f"Missing ManiSkill sensor camera(s): {missing_cameras}; "
                f"available cameras: {available_cameras}"
            )
        print(
            f"\nManiSkill 环境已创建: {args.task_id}, robot={args.robot_uids}, "
            f"num_envs={args.num_envs}, cameras={available_cameras}, "
            f"unnorm_key={cfg.unnorm_key}, max_steps={args.max_steps}, "
            f"env_action_dim={args.env_action_dim}, "
            f"exec_actions_per_inference={args.exec_actions_per_inference}"
        )

        total_successes = 0
        total_episodes = 0
        ep_idx = 0
        times = deque(maxlen=200)

        while ep_idx < args.num_eval_episodes:
            this_batch = min(args.num_envs, args.num_eval_episodes - ep_idx)
            seeds = [args.base_seed + ep_idx + i for i in range(args.num_envs)]
            obs, _ = env.reset(seed=seeds)

            succeeded = np.zeros(args.num_envs, dtype=bool)
            finished = np.zeros(args.num_envs, dtype=bool)
            action_queues = [deque() for _ in range(args.num_envs)]

            for _ in range(args.max_steps):
                need_inference = [
                    i
                    for i in range(args.num_envs)
                    if not finished[i] and len(action_queues[i]) == 0
                ]

                if need_inference:
                    obs_list = [
                        extract_policy_observation(obs, env_idx=i, args=args)
                        for i in need_inference
                    ]
                    inputs_list = [
                        prepare_one_obs(
                            cfg,
                            actor.processor,
                            policy_obs,
                            args.language_instruction,
                            torch_dtype,
                        )
                        for policy_obs in obs_list
                    ]
                    inputs_batch = actor.prepare_inputs_batch(inputs_list)

                    with torch.inference_mode():
                        action_logits, _ = actor.forward(inputs_batch)
                    _, _, normalized_actions = actor.post_process(
                        action_logits, [True] * len(need_inference)
                    )
                    for idx, env_i in enumerate(need_inference):
                        action_chunk = np.asarray(normalized_actions[idx])[
                            : args.exec_actions_per_inference
                        ]
                        action_queues[env_i].extend(action_chunk)

                model_actions = []
                for i in range(args.num_envs):
                    if action_queues[i]:
                        normalized_action = action_queues[i].popleft()
                        model_action = actor.vla._unnormalize_actions(
                            normalized_action, cfg.unnorm_key
                        )
                    else:
                        model_action = np.zeros(ACTION_DIM, dtype=np.float32)
                    model_actions.append(model_action)

                action_array = adapt_maniskill_action(
                    np.stack(model_actions, axis=0), args.env_action_dim
                )
                t0 = time.time()
                obs, _, terminated, truncated, info = env.step(action_array)
                times.append(time.time() - t0)

                succ = extract_success_mask(info, args.num_envs)
                done = extract_done_mask(terminated, truncated, args.num_envs)
                succeeded |= succ & ~finished
                finished |= succ | done
                if finished.all():
                    break

            batch_succ = int(succeeded[:this_batch].sum())
            total_successes += batch_succ
            total_episodes += this_batch
            ep_idx += this_batch

            avg_step_ms = np.mean(times) * 1000 if times else 0
            print(
                f"{args.task_id}: Episodes {ep_idx}/{args.num_eval_episodes}, "
                f"success_rate={total_successes / total_episodes:.3f}, "
                f"avg_step={avg_step_ms:.1f}ms"
            )

        success_rate = total_successes / total_episodes if total_episodes else 0.0
        print("\n========== ManiSkill evaluation summary ==========")
        print(
            f"{args.task_id}: {total_successes}/{total_episodes} = "
            f"{success_rate:.3f} (max_steps={args.max_steps})"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main(ARGS)
