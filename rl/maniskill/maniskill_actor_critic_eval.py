"""
ManiSkill evaluation script using ActorCritic (imported).
Replaces the Libero simulation in new_actor_critic.py with ManiSkill envs.
"""

import time
import random
import os
import sys
from pathlib import Path

# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
# os.environ["VULKAN_VISIBLE_DEVICES"] = "6" 
# os.environ["SAPIEN_VULKAN_DEVICE"] = "6"

GPU_ID = "6"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# Prevent TensorFlow from grabbing other GPUs on import
os.environ["TF_CUDA_VISIBLE_DEVICES"] = GPU_ID
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Vulkan / SAPIEN device pinning
os.environ["VULKAN_VISIBLE_DEVICES"] = GPU_ID
os.environ["SAPIEN_VULKAN_DEVICE"] = GPU_ID
os.environ["EGL_DEVICE_ID"] = GPU_ID

import warnings
from collections import deque
from types import SimpleNamespace

import numpy as np
import torch

_MANISKILL_DIR = Path(__file__).resolve().parent
_RL_DIR = _MANISKILL_DIR.parent
_REPO_ROOT = _RL_DIR.parent
for _p in (_REPO_ROOT, _RL_DIR, _MANISKILL_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM

# Import ActorCritic — no need to redefine
from new_actor_critic import ActorCritic
#from rl.actor_critic_model_discrete import ActorCritic
from rl.utils import prepare_one_obs, check_unnorm_key

# ManiSkill helpers
from rl.maniskill.maniskill_utils import (
    build_maniskill_env,
    extract_maniskill_observation,
    clip_maniskill_action,
    extract_success_mask,
    extract_done_mask,
)


def extract_camera_rgb(obs, env_idx, camera_name):
    if camera_name not in obs["sensor_data"]:
        available = ", ".join(obs["sensor_data"].keys())
        raise KeyError(
            f"Camera '{camera_name}' not found in obs['sensor_data']; "
            f"available cameras: {available}"
        )

    rgb = obs["sensor_data"][camera_name]["rgb"]
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().numpy()
    else:
        rgb = np.asarray(rgb)
    if rgb.ndim == 4:
        rgb = rgb[env_idx]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb)


def extract_two_image_maniskill_observation(
    obs,
    env_idx,
    camera_name,
    wrist_camera_name,
    use_proprio=False,
):
    obs_dict = extract_maniskill_observation(
        obs,
        env_idx=env_idx,
        camera_name=camera_name,
        use_proprio=use_proprio,
    )
    obs_dict["wrist_image"] = extract_camera_rgb(obs, env_idx, wrist_camera_name)
    return obs_dict


def main():
    USE_BF16 = True
    TORCH_DTYPE = torch.bfloat16 if USE_BF16 else torch.float32

    # ── ManiSkill env config ──
    NUM_ENVS = 1
    TASK_ID = "PickCube-v1"
    CAMERA_NAME = "base_camera"
    WRIST_CAMERA_NAME = "hand_camera"
    ROBOT_UIDS = "panda_wristcam"
    CAMERA_RES = 224
    MAX_STEPS = 200
    NUM_EVAL_EPISODES = 50
    LANGUAGE_INSTRUCTION = "pick up the red cube and place it at the green target"

    maniskill_checkpoint = (
        #"/cpfs01/lcx_stu4_workspace/openvla_oft_rl/runs/imitation/20260428_181954_openvla-7b+maniskill_pickcube+b64+lr-0.0005+lora-r32+dropout-0.0--image_aug"
        "/cpfs01/lcx_stu4_workspace/openvla_oft_rl/runs/imitation/20260511_203047_openvla-7b+maniskill_pickcube+b64+lr-0.0005+lora-r32+dropout-0.0--image_aug_2images"
        #"/cpfs01/lcx_stu4_workspace/openvla_oft_rl/runs/imitation/20260429_182819_openvla-7b+maniskill_pickcube+b64+lr-0.0005+lora-r32+dropout-0.0--image_aug"
    )

    cfg = SimpleNamespace(
        pretrained_checkpoint=maniskill_checkpoint,
        use_l1_regression=False,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=False,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        unnorm_key="maniskill_pickcube",
        device=torch.device("cuda"),
        use_lora=True,
        lora_rank=32,
        lora_dropout=0.0,
        checkpoint2="",
        enable_pmvt=True,
    )

    # ── Build actor (imported) ──
    actor = ActorCritic(cfg, TORCH_DTYPE)
    check_unnorm_key(cfg, actor.vla)
    actor.eval()
    print("策略初始化完成。")

    # ── Build ManiSkill env (GPU sim+render, pinned to cuda:0 = physical GPU_ID) ──
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    env = gym.make(
        TASK_ID,
        obs_mode="rgbd",
        reward_mode="sparse",
        control_mode="pd_ee_delta_pose",
        robot_uids=ROBOT_UIDS,
        num_envs=NUM_ENVS,
        sim_backend="gpu",
        render_mode="rgb_array",
        sensor_configs={
            CAMERA_NAME: {"width": CAMERA_RES, "height": CAMERA_RES},
            WRIST_CAMERA_NAME: {"width": CAMERA_RES, "height": CAMERA_RES},
        },
        max_episode_steps=MAX_STEPS,
    )

    obs, _ = env.reset()
    available_cameras = list(obs["sensor_data"].keys())
    expected_cameras = [CAMERA_NAME, WRIST_CAMERA_NAME]
    missing_cameras = [name for name in expected_cameras if name not in available_cameras]
    if missing_cameras:
        raise RuntimeError(
            f"Missing ManiSkill sensor camera(s): {missing_cameras}; "
            f"available cameras: {available_cameras}"
        )

    VIDEO_DIR = os.path.join(maniskill_checkpoint, "eval_videos")
    os.makedirs(VIDEO_DIR, exist_ok=True)
    from mani_skill.utils.wrappers import RecordEpisode
    # env = RecordEpisode(
    #     env,
    #     output_dir=VIDEO_DIR,
    #     save_video=True,
    #     info_on_video=True,
    #     max_steps_per_video=MAX_STEPS,
    # )
    print(
        f"ManiSkill 环境已创建: {TASK_ID}, robot={ROBOT_UIDS}, "
        f"num_envs={NUM_ENVS}, cameras={available_cameras}"
    )
    # ── Eval loop ──
    total_successes = 0
    total_episodes = 0
    ep_idx = 0
    times = deque(maxlen=200)

    while ep_idx < NUM_EVAL_EPISODES:
        this_batch = min(NUM_ENVS, NUM_EVAL_EPISODES - ep_idx)
        seeds = [ep_idx + i for i in range(NUM_ENVS)]
        obs, _ = env.reset(seed=seeds)

        succeeded = np.zeros(NUM_ENVS, dtype=bool)
        finished = np.zeros(NUM_ENVS, dtype=bool)
        action_queues = [deque() for _ in range(NUM_ENVS)]

        for step in range(MAX_STEPS):
            need_inference = [
                i for i in range(NUM_ENVS)
                if not finished[i] and len(action_queues[i]) == 0
            ]

            if need_inference:
                obs_list = [
                    extract_two_image_maniskill_observation(
                        obs, env_idx=i,
                        camera_name=CAMERA_NAME,
                        wrist_camera_name=WRIST_CAMERA_NAME,
                        use_proprio=cfg.use_proprio,
                    )
                    for i in need_inference
                ]
                task_labels = [LANGUAGE_INSTRUCTION] * len(need_inference)
                inputs_list = [
                    prepare_one_obs(cfg, actor.processor, o, t, TORCH_DTYPE)
                    for o, t in zip(obs_list, task_labels)
                ]

                inputs_batch = actor.prepare_inputs_batch(inputs_list)

                # 验证输入图片数量
                # pv = inputs_batch["pixel_values"]
                # num_images = pv.shape[1] // 3  # 每张图片3个channel
                # print(f"[DEBUG] pixel_values shape: {pv.shape}, 输入图片数量: {num_images}")

                with torch.inference_mode():
                    action_logits, _ = actor.forward(inputs_batch)
                _, _, normalized_actions = actor.post_process(
                    action_logits, [True] * len(need_inference)
                )
                for idx, env_i in enumerate(need_inference):
                    action_queues[env_i].extend(normalized_actions[idx])

            # Execute one action per env
            step_actions = []
            for i in range(NUM_ENVS):
                if len(action_queues[i]) > 0:
                    a = action_queues[i].popleft()
                    a_env = actor.vla._unnormalize_actions(a, cfg.unnorm_key)
                    step_actions.append(a_env)
                else:
                    step_actions.append(np.zeros(ACTION_DIM, dtype=np.float32))

            action_array = clip_maniskill_action(np.stack(step_actions, axis=0))
            t0 = time.time()
            obs, _, terminated, truncated, info = env.step(action_array)
            times.append(time.time() - t0)

            succ = extract_success_mask(info, NUM_ENVS)
            done = extract_done_mask(terminated, truncated, NUM_ENVS)
            succeeded |= (succ & ~finished)
            finished |= (succ | done)
            if finished.all():
                break

        batch_succ = int(succeeded[:this_batch].sum())
        total_successes += batch_succ
        total_episodes += this_batch
        ep_idx += this_batch

        avg_step_ms = np.mean(times) * 1000 if times else 0
        print(
            f"Episodes {ep_idx}/{NUM_EVAL_EPISODES}, "
            f"success_rate={total_successes / total_episodes:.3f}, "
            f"avg_step={avg_step_ms:.1f}ms"
        )

    env.close()
    print(
        f"\nFinal: {total_successes}/{total_episodes} = "
        f"{total_successes / total_episodes:.3f}"
    )


if __name__ == "__main__":
    main()
