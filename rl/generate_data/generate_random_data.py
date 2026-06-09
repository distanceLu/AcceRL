import os
import numpy as np
import torch
import json
from tqdm import tqdm
from rl.libero_env import LiberoEnvWrapper


def discretize_action(action, bins=256, min_val=-1.0, max_val=1.0):
    """Discretize continuous action to integer bins."""
    action = np.clip(action, min_val, max_val)
    return ((action - min_val) / (max_val - min_val) * (bins - 1)).astype(int)


def generate_data(
    output_dir="/cpfs01/lcx_workspace/Open-Sora/debug/libero_data_full_775ep",
    benchmark_name="libero_spatial",
    num_tasks=5, # Generate data for first N tasks
    episodes_per_task=155,
    max_frames=16,
    image_size=256, # OpenSora default resolution
    require_full_window: bool = True,  # if True, only save when valid_frames >= max_frames
):
    os.makedirs(output_dir, exist_ok=True)
    
    metadata = []
    sample_idx = 0
    
    print(f"Generating data to {output_dir} from benchmark {benchmark_name}...")
    
    for task_id in range(num_tasks):
        try:
            print(f"Initializing task {task_id}...")
            # Initialize environment for the current task
            # Use image_size=256 directly to avoid resizing later if possible, 
            # but LiberoEnvWrapper defaults to 224. Let's use 256 in wrapper init if supported, 
            # or resize. Wrapper init supports image_size.
            env = LiberoEnvWrapper(
                benchmark_name=benchmark_name,
                task_id=task_id,
                image_size=image_size,
                render_mode="rgb_array"
            )
        except Exception as e:
            print(f"Failed to initialize task {task_id}: {e}")
            continue
            
        print(f"Generating {episodes_per_task} episodes for task: {env.task.name}")
        
        for ep in tqdm(range(episodes_per_task), desc=f"Task {task_id}"):
            obs, info = env.reset()
            # obs['full_image'] is (image_size, image_size, 3) uint8
            
            current_img = obs["full_image"]
            instruction = info["task_description"]
            
            buffer_obs = [current_img]
            buffer_actions = [] 
            
            terminated = False
            truncated = False
            step_count = 0
            
            while not (terminated or truncated):
                # For data generation, we use random actions since we don't have a policy
                # In a real scenario, you'd use an expert policy or human demonstrations.
                # Libero doesn't provide built-in expert policies easily accessible here without extra setup.
                # So we generate random walks.
                action = env.action_space.sample()
                
                obs, reward, terminated, truncated, info = env.step(action)
                current_img = obs["full_image"]
                
                buffer_obs.append(current_img)
                
                # Discretize action for tokenization compatibility
                # Action shape is (7,). We keep it as array of 7 ints.
                disc_action = discretize_action(action)
                buffer_actions.append(disc_action)
                
                step_count += 1
                
                # Save sample window
                current_obs_seq = buffer_obs[-max_frames:]
                # Action sequence corresponding to transitions. 
                # buffer_obs[0] is initial. action[0] leads to buffer_obs[1].
                # So if we have obs[0...T], we have actions[0...T-1].
                # We want to align them. Let's say we want to predict obs[t] given obs[t-1] and action[t-1].
                # Or standard video generation: conditioned on text + actions.
                # Let's take last max_frames actions. 
                current_action_seq = buffer_actions[-max_frames:]
                
                valid_frames = len(current_obs_seq)

                # Optionally require a full 16-frame window before saving
                if require_full_window and valid_frames < max_frames:
                    continue
                
                # Prepare tensors
                video_tensor_seq = np.zeros((max_frames, image_size, image_size, 3), dtype=np.uint8)
                # Actions: [max_frames, 7] (7-dim action)
                # Padding with 0 (or specific token)
                action_seq_final = np.zeros((max_frames, 7), dtype=int) 
                mask_seq = np.zeros((max_frames,), dtype=bool)
                
                # Fill data (Right aligned / Latest at end)
                video_tensor_seq[-valid_frames:] = np.array(current_obs_seq)
                mask_seq[-valid_frames:] = True
                
                valid_actions = len(current_action_seq)
                if valid_actions > 0:
                    action_seq_final[-valid_actions:] = np.array(current_action_seq)
                
                # Save
                sample_name = f"task{task_id}_ep{ep}_step{step_count}_{sample_idx:06d}.pt"
                save_path = os.path.join(output_dir, sample_name)
                
                torch.save({
                    "video": video_tensor_seq, # [T, H, W, C]
                    "actions": action_seq_final, # [T, 7] -> need to handle this in train script
                    "mask": mask_seq, # [T]
                    "instruction": instruction
                }, save_path)
                
                metadata.append({
                    "path": save_path,
                    "task_id": task_id,
                    "instruction": instruction,
                    "valid_frames": valid_frames
                })
                
                sample_idx += 1
                
                if step_count >= 50: # Limit length
                    break
        
        env.close()
                
    # Save metadata
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Generated {sample_idx} samples.")


if __name__ == "__main__":
    # Ensure LIBERO dataset is downloaded or path is set correctly
    # You might need to set LIBERO_DATASET_PATH env var or similar if Libero wrapper requires it
    generate_data()

