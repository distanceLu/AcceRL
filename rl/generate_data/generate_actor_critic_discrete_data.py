'''
example
python rl/generate_data/generate_actor_critic_discrete_data.py --output_dir /mnt/data/lcx3/AcceRL/tests_dsj/dataset_episode --benchmark_name libero_spatial --num_tasks 5 --episodes_per_task 10 --pretrained_checkpoint /mnt/data/lcx3/checkpoint/dsj/openvla-7b+libero_spatial_no_noops+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--discrete_acts--proprio_state--100000_chkpt --device cuda:0 --use_bf16 --use_proprio
'''
import os
import json
import time
import random
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from tqdm import tqdm

from rl.libero_env import LiberoEnvWrapper
from rl.utils import prepare_one_obs, check_unnorm_key
from rl.actor_critic_model_discrete import ActorCritic
from experiments.robot.libero.libero_utils import GenerateConfig
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


def discretize_action(action: np.ndarray, bins: int = 256, min_val: float = -1.0, max_val: float = 1.0) -> np.ndarray:
    """Discretize continuous action to integer bins (shape (..., 7))."""
    action = np.asarray(action)
    action = np.clip(action, min_val, max_val)
    return ((action - min_val) / (max_val - min_val) * (bins - 1)).astype(int)


def generate_data(
    output_dir: str,
    benchmark_name: str = "libero_spatial",
    num_tasks: int = 5,
    episodes_per_task: int = 5,
    max_frames: int = 16,
    image_size: int = 256,
    require_full_window: bool = True,
    pretrained_checkpoint: Optional[str] = None,
    device: str = "cuda:0",
    use_bf16: bool = True,
    greedy: bool = False,
    seed: int = None,
    print_success_interval: int = 100,
    num_images_in_input: int = 2,
    use_proprio: bool = True,
):
    """Generate full-episode samples using ActorCritic discrete policy.

    Output format is episode-oriented for Ctrl-World training:
      - one `.pt` file per completed episode
      - `video`: full agentview RGB sequence, shape [T, H, W, 3]
      - `wrist_video`: full wrist RGB sequence, shape [T, H, W, 3]
      - `proprio`: full proprio sequence, shape [T, 8]
      - `actions`: discretized actions, shape [T, 7], action[0] is padding
      - `actions_continuous`: continuous env delta actions, shape [T, 7],
        action[0] is padding and action[t] produced observation frame t
      - `mask`: all True, shape [T]
      - `metadata.json`: per-frame sample entries under `metadata`, plus full
        episode entries under `episodes`, matching Ctrl-World's frame-anchor
        sampling convention

    Notes:
      - Policy generates `NUM_ACTIONS_CHUNK` actions at once; we queue them per-env.
      - We discretize the executed env action into 256 bins for tokenization compatibility.
      - `max_frames` and `require_full_window` are kept as CLI-compatible no-ops;
        episode files always save the full rollout.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Reproducibility (best-effort)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"Set seed to {seed}")
    else:
        print("No seed provided, using random seed")

    unnorm_key = f"{benchmark_name}_no_noops"

    torch_dtype = torch.bfloat16 if use_bf16 else torch.float32
    cfg = GenerateConfig(
        pretrained_checkpoint=pretrained_checkpoint or GenerateConfig.pretrained_checkpoint,
        use_l1_regression=False,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=num_images_in_input,
        use_proprio=use_proprio,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        unnorm_key=unnorm_key,
        device=torch.device(device),
        task_suite_name=benchmark_name,
    )

    # Policy
    actor = ActorCritic(cfg, torch_dtype)
    check_unnorm_key(cfg, actor.vla)
    actor.eval()

    # Envs (one env per task_id)
    task_ids: List[int] = list(range(num_tasks))
    envs = [
        LiberoEnvWrapper(
            benchmark_name=str(benchmark_name),
            task_id=task_id,
            image_size=image_size,
            render_mode="rgb_array",
        )
        for task_id in task_ids
    ]

    from collections import deque

    env_queues = [deque() for _ in envs]  # queued normalized actions (each element shape (7,))

    # Per-env episode state
    episodes_done = [0 for _ in envs]
    observations: List[Optional[Dict[str, Any]]] = [None for _ in envs]
    task_descriptions: List[str] = ["" for _ in envs]

    buffer_obs: List[List[np.ndarray]] = [[] for _ in envs]
    buffer_wrist_obs: List[List[np.ndarray]] = [[] for _ in envs]
    buffer_proprio: List[List[np.ndarray]] = [[] for _ in envs]
    buffer_actions: List[List[np.ndarray]] = [[] for _ in envs]
    buffer_actions_continuous: List[List[np.ndarray]] = [[] for _ in envs]
    step_counts = [0 for _ in envs]

    episode_metadata: List[Dict[str, Any]] = []
    sample_metadata: List[Dict[str, Any]] = []
    sample_idx = 0
    
    # Success rate tracking
    total_success = 0
    total_episodes = 0

    if max_frames != 16 or require_full_window:
        print(
            "Note: full-episode save mode ignores --max_frames and "
            "--require_full_window; complete episodes are saved at termination."
        )

    def _save_episode(i: int, reward: float, terminated: bool, truncated: bool, info: Dict[str, Any]):
        nonlocal sample_idx, total_success

        valid_frames = len(buffer_obs[i])
        if valid_frames <= 1:
            return

        video_tensor_seq = np.asarray(buffer_obs[i], dtype=np.uint8)
        wrist_video_tensor_seq = np.asarray(buffer_wrist_obs[i], dtype=np.uint8)
        proprio_seq_final = np.asarray(buffer_proprio[i], dtype=np.float32)

        # Align actions to frames. actions[0] is padding for the initial frame;
        # actions[t] is the control that led from frame t-1 to frame t.
        action_seq_final = np.zeros((valid_frames, 7), dtype=int)
        action_continuous_seq_final = np.zeros((valid_frames, 7), dtype=np.float32)
        if buffer_actions[i]:
            action_seq_final[1:] = np.asarray(buffer_actions[i], dtype=int)
            action_continuous_seq_final[1:] = np.asarray(buffer_actions_continuous[i], dtype=np.float32)
        mask_seq = np.ones((valid_frames,), dtype=bool)

        task_id = envs[i].task_id
        ep = episodes_done[i]
        sample_name = f"task{task_id}_ep{ep}_{sample_idx:06d}.pt"
        save_path = os.path.join(output_dir, sample_name)
        is_success = bool(info.get("is_success", False))

        torch.save(
            {
                "video": video_tensor_seq,
                "wrist_video": wrist_video_tensor_seq,
                "view_names": ["agentview_image", "robot0_eye_in_hand_image"],
                "proprio": proprio_seq_final,
                "actions": action_seq_final,
                "actions_continuous": action_continuous_seq_final,
                "mask": mask_seq,
                "instruction": task_descriptions[i],
                "reward": float(reward),
                "success": is_success,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "task_id": int(task_id),
                "episode": int(ep),
                "num_frames": int(valid_frames),
            },
            save_path,
        )

        episode_meta = {
            "path": save_path,
            "task_id": int(task_id),
            "instruction": task_descriptions[i],
            "valid_frames": int(valid_frames),
            "episode": int(ep),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "is_success": is_success,
            "views": ["agentview_image", "robot0_eye_in_hand_image"],
            "proprio_dim": 8,
            "action_dim": 7,
            "action_format": {
                "actions": "discrete_0_255_frame_aligned_padding_at_0",
                "actions_continuous": "unnormalized_env_delta_frame_aligned_padding_at_0",
            },
        }
        episode_metadata.append(episode_meta)
        for frame_id in range(valid_frames):
            sample_metadata.append(
                {
                    **episode_meta,
                    "frame_ids": [int(frame_id)],
                    "sample_id": int(len(sample_metadata)),
                }
            )
        sample_idx += 1

        if is_success:
            total_success += 1

    def _reset_env(i: int):
        if seed is not None:
            obs, info = envs[i].reset(seed=seed + i)
        else:
            obs, info = envs[i].reset(seed=int(time.time()) + i)
        observations[i] = obs
        task_descriptions[i] = info.get("task_description", envs[i].task_description)
        # start buffers with initial observation (same alignment as random script)
        buffer_obs[i] = [obs["full_image"]]
        buffer_wrist_obs[i] = [obs["wrist_image"]]
        buffer_proprio[i] = [obs["state"]]
        buffer_actions[i] = []
        buffer_actions_continuous[i] = []
        step_counts[i] = 0
        env_queues[i].clear()

    # Initialize all envs
    for i in range(len(envs)):
        _reset_env(i)

    active_envs = [True for _ in envs]

    progress = tqdm(total=num_tasks * episodes_per_task, desc="Episodes", dynamic_ncols=True)

    try:
        while any(active_envs):
            # 1) Batch-generate new action chunks for envs that need them
            need_gen_indices: List[int] = []
            inputs_t_list: List[Dict[str, Any]] = []
            for i in range(len(envs)):
                if not active_envs[i]:
                    continue
                if len(env_queues[i]) == 0:
                    assert observations[i] is not None
                    inputs_t = prepare_one_obs(cfg, actor.processor, observations[i], task_descriptions[i], torch_dtype)
                    inputs_t_list.append(inputs_t)
                    need_gen_indices.append(i)

            if inputs_t_list:
                inputs_batch = actor.prepare_inputs_batch(inputs_t_list)
                with torch.inference_mode():
                    action_logits, _value = actor.forward(inputs_batch)
                B = int(action_logits.size(0))
                deterministic_flags = [bool(greedy) for _ in range(B)]
                _dist, _token_ids, normalized_actions = actor.post_process(action_logits, deterministic_flags)
                # normalized_actions: (B, NUM_ACTIONS_CHUNK, 7)
                for b, env_i in enumerate(need_gen_indices):
                    env_queues[env_i].extend(normalized_actions[b])

            # 2) Step each active env by one action
            for i in range(len(envs)):
                if not active_envs[i]:
                    continue

                if len(env_queues[i]) == 0:
                    # should not happen; skip to next env
                    continue

                action_norm = env_queues[i].popleft()  # shape (7,)
                action_env = actor.vla._unnormalize_actions(action_norm, cfg.unnorm_key)

                obs, reward, terminated, truncated, info = envs[i].step(action_env)
                observations[i] = obs

                # Update buffers
                buffer_obs[i].append(obs["full_image"])
                buffer_wrist_obs[i].append(obs["wrist_image"])
                buffer_proprio[i].append(obs["state"])
                disc_action = discretize_action(action_env)  # (7,)
                buffer_actions[i].append(disc_action)
                buffer_actions_continuous[i].append(np.asarray(action_env, dtype=np.float32))

                step_counts[i] += 1

                if terminated or truncated:
                    _save_episode(i, reward, terminated, truncated, info)
                    total_episodes += 1
                     # Print success rate periodically
                    if total_episodes % print_success_interval == 0:
                        success_rate = total_success / total_episodes if total_episodes > 0 else 0.0
                        print(f"\n[Success Rate] Episodes: {total_episodes}, Success: {total_success}, Rate: {success_rate:.4f} ({success_rate*100:.2f}%)")
                    episodes_done[i] += 1
                    progress.update(1)

                    if episodes_done[i] >= episodes_per_task:
                        active_envs[i] = False
                        continue

                    _reset_env(i)

    finally:
        progress.close()
        for env in envs:
            try:
                env.close()
            except Exception:
                pass
        
        # Print final success rate
        if total_episodes > 0:
            final_success_rate = total_success / total_episodes
            print(f"\n[Final Success Rate] Total Episodes: {total_episodes}, Total Success: {total_success}, Rate: {final_success_rate:.4f} ({final_success_rate*100:.2f}%)")

        # Save metadata
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(
                {
                    "metadata": sample_metadata,
                    "episodes": episode_metadata,
                    "config": {
                        "benchmark_name": benchmark_name,
                        "num_tasks": num_tasks,
                        "episodes_per_task": episodes_per_task,
                        "max_frames": max_frames,
                        "image_size": image_size,
                        "require_full_window": require_full_window,
                        "save_format": "full_episode",
                        "pretrained_checkpoint": str(cfg.pretrained_checkpoint),
                        "device": device,
                        "use_bf16": use_bf16,
                        "greedy": greedy,
                        "unnorm_key": cfg.unnorm_key,
                        "num_open_loop_steps": int(cfg.num_open_loop_steps),
                        "torch_dtype": str(torch_dtype),
                        "saved_observation_keys": [
                            "video",
                            "wrist_video",
                            "proprio",
                        ],
                        "view_names": ["agentview_image", "robot0_eye_in_hand_image"],
                        "proprio_dim": 8,
                        "action_keys": {
                            "actions": "discrete_0_255_frame_aligned_padding_at_0",
                            "actions_continuous": "unnormalized_env_delta_frame_aligned_padding_at_0",
                        },
                    },
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--benchmark_name", type=str, default="libero_spatial")
    parser.add_argument("--num_tasks", type=int, default=5)
    parser.add_argument("--episodes_per_task", type=int, default=10)
    parser.add_argument("--max_frames", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--require_full_window", action="store_true")
    parser.add_argument("--pretrained_checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint2", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--print_success_interval", type=int, default=10, help="Print success rate every N samples")
    parser.add_argument("--num_images_in_input", type=int, default=2)
    parser.add_argument("--use_proprio", action="store_true")

    args = parser.parse_args()
    print(f"args: {args}")

    generate_data(
        output_dir=args.output_dir,
        benchmark_name=args.benchmark_name,
        num_tasks=args.num_tasks,
        episodes_per_task=args.episodes_per_task,
        max_frames=args.max_frames,
        image_size=args.image_size,
        require_full_window=args.require_full_window,
        pretrained_checkpoint=args.pretrained_checkpoint,
        device=args.device,
        use_bf16=args.use_bf16,
        greedy=args.greedy,
        seed=args.seed,
        print_success_interval=args.print_success_interval,
        num_images_in_input=args.num_images_in_input,
        use_proprio=args.use_proprio,
    )
