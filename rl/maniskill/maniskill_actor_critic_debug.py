"""
Debug PegInsertionSide eval failures with two small experiments:

1. default_chunk: execute the full OpenVLA action chunk before re-inference.
2. chunk1: execute only the first action from each chunk before re-inference.

Each experiment records videos and writes an actions.csv file so the gripper
command can be aligned with the rollout video.
"""

import argparse
import csv
import os
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

GPU_ID = os.environ.get("MANISKILL_GPU_ID", "4")
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TF_CUDA_VISIBLE_DEVICES"] = GPU_ID
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["VULKAN_VISIBLE_DEVICES"] = GPU_ID
os.environ["SAPIEN_VULKAN_DEVICE"] = GPU_ID
os.environ["EGL_DEVICE_ID"] = GPU_ID

import numpy as np
import torch

_MANISKILL_DIR = Path(__file__).resolve().parent
_RL_DIR = _MANISKILL_DIR.parent
_REPO_ROOT = _RL_DIR.parent
for _p in (_REPO_ROOT, _RL_DIR, _MANISKILL_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
from new_actor_critic import ActorCritic
from rl.utils import check_unnorm_key, prepare_one_obs
from rl.maniskill.maniskill_utils import (
    clip_maniskill_action,
    extract_done_mask,
    extract_maniskill_observation,
    extract_success_mask,
)


TASK_CFG = {
    "task_id": "PegInsertionSide-v1",
    "unnorm_key": "maniskill_peginsertionside",
    "language_instruction": "pick up the orange-white peg and insert the orange end into the box with a hole in it",
    "max_steps": 500,
}
CAMERA_NAME = "base_camera"
WRIST_CAMERA_NAME = "hand_camera"
ROBOT_UIDS = "panda_wristcam"
CAMERA_RES = 224
NUM_ENVS = 1
DEFAULT_CHECKPOINT = (
    "/cpfs01/lcx_stu4_workspace/openvla_oft_rl/runs/imitation/"
    "20260523_152554_openvla-7b+maniskill_three_tasks+b128+lr-0.0005+lora-r32+dropout-0.0--image_aug--three_tasks_2cam_preprocessed"
)


def extract_camera_rgb(obs, env_idx, camera_name):
    if camera_name not in obs["sensor_data"]:
        available = ", ".join(obs["sensor_data"].keys())
        raise KeyError(f"Camera {camera_name!r} not found; available cameras: {available}")

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


def extract_two_image_observation(obs, env_idx, use_proprio=False):
    obs_dict = extract_maniskill_observation(
        obs,
        env_idx=env_idx,
        camera_name=CAMERA_NAME,
        use_proprio=use_proprio,
    )
    obs_dict["wrist_image"] = extract_camera_rgb(obs, env_idx, WRIST_CAMERA_NAME)
    return obs_dict


def make_actor_cfg(checkpoint):
    return SimpleNamespace(
        pretrained_checkpoint=checkpoint,
        use_l1_regression=False,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=False,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        unnorm_key=TASK_CFG["unnorm_key"],
        device=torch.device("cuda"),
        use_lora=True,
        lora_rank=32,
        lora_dropout=0.0,
        checkpoint2="",
        enable_pmvt=True,
    )


def make_env(max_steps):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    return gym.make(
        TASK_CFG["task_id"],
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
        max_episode_steps=max_steps,
    )


def wrap_record_episode(env, output_dir, episodes, max_steps):
    from mani_skill.utils.wrappers import RecordEpisode

    os.makedirs(output_dir, exist_ok=True)
    record_step_limit = episodes * max_steps * NUM_ENVS
    return RecordEpisode(
        env,
        output_dir=output_dir,
        save_trajectory=False,
        save_video=True,
        info_on_video=True,
        save_video_trigger=lambda elapsed_steps: elapsed_steps < record_step_limit,
        max_steps_per_video=max_steps,
        avoid_overwriting_video=True,
    )


def write_action_row(writer, experiment, ep_idx, step, env_idx, action, succ, done):
    writer.writerow(
        {
            "experiment": experiment,
            "episode": ep_idx + env_idx,
            "step": step,
            "env_idx": env_idx,
            "dx": float(action[0]),
            "dy": float(action[1]),
            "dz": float(action[2]),
            "drx": float(action[3]),
            "dry": float(action[4]),
            "drz": float(action[5]),
            "gripper": float(action[6]),
            "success": bool(succ[env_idx]),
            "done": bool(done[env_idx]),
        }
    )


def run_experiment(actor, cfg, experiment, exec_actions_per_inference, episodes, checkpoint):
    import mani_skill.envs  # noqa: F401

    task_id = TASK_CFG["task_id"]
    max_steps = TASK_CFG["max_steps"]
    cfg.unnorm_key = TASK_CFG["unnorm_key"]
    check_unnorm_key(cfg, actor.vla)

    output_dir = os.path.join(checkpoint, "eval_videos", f"{task_id}_{experiment}")
    env = wrap_record_episode(make_env(max_steps), output_dir, episodes, max_steps)
    action_csv = os.path.join(output_dir, "actions.csv")

    try:
        obs, _ = env.reset()
        cameras = list(obs["sensor_data"].keys())
        missing = [name for name in (CAMERA_NAME, WRIST_CAMERA_NAME) if name not in cameras]
        if missing:
            raise RuntimeError(f"Missing cameras {missing}; available cameras: {cameras}")

        print(
            f"\n=== Experiment: {experiment} ===\n"
            f"videos: {output_dir}\n"
            f"actions: {action_csv}\n"
            f"exec_actions_per_inference={exec_actions_per_inference}, "
            f"episodes={episodes}, max_steps={max_steps}, cameras={cameras}",
            flush=True,
        )

        total_successes = 0
        total_episodes = 0
        ep_idx = 0
        times = deque(maxlen=200)

        with open(action_csv, "w", newline="") as f:
            fieldnames = [
                "experiment",
                "episode",
                "step",
                "env_idx",
                "dx",
                "dy",
                "dz",
                "drx",
                "dry",
                "drz",
                "gripper",
                "success",
                "done",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            while ep_idx < episodes:
                this_batch = min(NUM_ENVS, episodes - ep_idx)
                seeds = [ep_idx + i for i in range(NUM_ENVS)]
                obs, _ = env.reset(seed=seeds)

                succeeded = np.zeros(NUM_ENVS, dtype=bool)
                finished = np.zeros(NUM_ENVS, dtype=bool)
                action_queues = [deque() for _ in range(NUM_ENVS)]

                for step in range(max_steps):
                    need_inference = [
                        i for i in range(NUM_ENVS)
                        if not finished[i] and len(action_queues[i]) == 0
                    ]

                    if need_inference:
                        obs_list = [
                            extract_two_image_observation(
                                obs,
                                env_idx=i,
                                use_proprio=cfg.use_proprio,
                            )
                            for i in need_inference
                        ]
                        task_labels = [TASK_CFG["language_instruction"]] * len(need_inference)
                        inputs_list = [
                            prepare_one_obs(cfg, actor.processor, o, t, torch.bfloat16)
                            for o, t in zip(obs_list, task_labels)
                        ]
                        inputs_batch = actor.prepare_inputs_batch(inputs_list)

                        with torch.inference_mode():
                            action_logits, _ = actor.forward(inputs_batch)
                        _, _, normalized_actions = actor.post_process(
                            action_logits, [True] * len(need_inference)
                        )
                        for idx, env_i in enumerate(need_inference):
                            chunk = np.asarray(normalized_actions[idx])
                            if exec_actions_per_inference > 0:
                                chunk = chunk[:exec_actions_per_inference]
                            action_queues[env_i].extend(chunk)

                    step_actions = []
                    for i in range(NUM_ENVS):
                        if action_queues[i]:
                            action = action_queues[i].popleft()
                            action = actor.vla._unnormalize_actions(action, cfg.unnorm_key)
                            step_actions.append(action)
                        else:
                            step_actions.append(np.zeros(ACTION_DIM, dtype=np.float32))

                    action_array = clip_maniskill_action(np.stack(step_actions, axis=0))
                    t0 = time.time()
                    obs, _, terminated, truncated, info = env.step(action_array)
                    times.append(time.time() - t0)

                    succ = extract_success_mask(info, NUM_ENVS)
                    done = extract_done_mask(terminated, truncated, NUM_ENVS)
                    for i, action in enumerate(action_array):
                        write_action_row(writer, experiment, ep_idx, step, i, action, succ, done)
                        if step < 10 or step % 25 == 0:
                            print(
                                f"[{experiment}] ep={ep_idx + i} step={step:03d} "
                                f"xyz={np.array2string(action[:3], precision=4, suppress_small=True)} "
                                f"rot={np.array2string(action[3:6], precision=4, suppress_small=True)} "
                                f"gripper={action[6]: .4f} success={bool(succ[i])}",
                                flush=True,
                            )

                    succeeded |= (succ & ~finished)
                    finished |= (succ | done)
                    if finished.all():
                        break

                batch_succ = int(succeeded[:this_batch].sum())
                total_successes += batch_succ
                total_episodes += this_batch
                ep_idx += this_batch

                avg_step_ms = np.mean(times) * 1000 if times else 0.0
                print(
                    f"[{experiment}] episodes {ep_idx}/{episodes}, "
                    f"success_rate={total_successes / total_episodes:.3f}, "
                    f"avg_step={avg_step_ms:.1f}ms",
                    flush=True,
                )

        success_rate = total_successes / total_episodes if total_episodes else 0.0
        print(
            f"[{experiment}] Final: {total_successes}/{total_episodes} = {success_rate:.3f}",
            flush=True,
        )
        return {
            "experiment": experiment,
            "successes": total_successes,
            "episodes": total_episodes,
            "success_rate": success_rate,
            "video_dir": output_dir,
            "action_csv": action_csv,
        }
    finally:
        env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--experiment",
        choices=("both", "default_chunk", "chunk1"),
        default="both",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = make_actor_cfg(args.checkpoint)
    actor = ActorCritic(cfg, torch.bfloat16)
    check_unnorm_key(cfg, actor.vla)
    actor.eval()
    print(f"Actor initialized on GPU_ID={GPU_ID}. NUM_ACTIONS_CHUNK={NUM_ACTIONS_CHUNK}")

    experiments = []
    if args.experiment in ("both", "default_chunk"):
        experiments.append(("default_chunk", NUM_ACTIONS_CHUNK))
    if args.experiment in ("both", "chunk1"):
        experiments.append(("chunk1", 1))

    results = [
        run_experiment(actor, cfg, name, exec_count, args.episodes, args.checkpoint)
        for name, exec_count in experiments
    ]

    print("\n========== Debug experiment summary ==========")
    for result in results:
        print(
            f"{result['experiment']}: {result['successes']}/{result['episodes']} = "
            f"{result['success_rate']:.3f}"
        )
        print(f"  videos: {result['video_dir']}")
        print(f"  actions: {result['action_csv']}")


if __name__ == "__main__":
    main()
