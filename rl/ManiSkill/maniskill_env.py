"""ManiSkill single-environment wrapper for RL training.

Provides the same interface as LiberoEnvWrapper so that the PPO workers
in ds_maniskill_ppo_discrete.py can use it as a drop-in replacement.

Interface contract (mirrors LiberoEnvWrapper):
    - reset(seed=None)  -> (obs_dict, info)
    - step(action)      -> (obs_dict, reward, terminated, truncated, info)
    - task_description   : str property
    - get_name()        : str
    - close()
    obs_dict keys: "full_image" (H,W,3 uint8), "state" (numpy 1-D float)
    info must contain "is_success" after step.
"""
import os

import numpy as np
from typing import Any, Dict, Optional

from rl.ManiSkill.maniskill.maniskill_utils import (
    build_maniskill_env,
    extract_maniskill_observation,
    clip_maniskill_action,
    extract_success_mask,
    convert_torch_to_numpy,
    LANGUAGE_INSTRUCTION, 
)

# from experiments.robot.maniskill.maniskill_utils import (
#     build_maniskill_env,
#     extract_maniskill_observation,
#     clip_maniskill_action,
#     extract_success_mask,
#     convert_torch_to_numpy,
#     LANGUAGE_INSTRUCTION,
# )


class ManiSkillSingleEnv:
    """Single ManiSkill env with the same API as LiberoEnvWrapper."""

    def __init__(
        self,
        task_id: str = "PickCube-v1",
        camera_name: str = "base_camera",
        camera_res: int = 224,
        max_episode_steps: int = 100,
        use_proprio: bool = False,
        sim_backend: str = "cpu",
        language_instruction: Optional[str] = None,
        render_backend: Optional[str] = None,
        wrist_camera_name: Optional[str] = None,
        robot_uids: Optional[str] = None,
    ):
        self.task_id = task_id
        self.camera_name = camera_name
        self.wrist_camera_name = wrist_camera_name
        self.include_wrist_image = wrist_camera_name is not None
        self.use_proprio = use_proprio
        self.task_description = language_instruction or LANGUAGE_INSTRUCTION

        self.env = build_maniskill_env(
            task_id=task_id,
            num_envs=1,
            obs_mode="rgbd",
            control_mode="pd_ee_delta_pose",
            camera_name=camera_name,
            wrist_camera_name=wrist_camera_name,
            camera_res=camera_res,
            max_episode_steps=max_episode_steps,
            sim_backend=sim_backend,
            render_backend=render_backend,
            robot_uids=robot_uids,
        )

        # 初始势能
        self._prev_potential = 0.0
        wrist_info = f" + wrist={wrist_camera_name}" if wrist_camera_name else ""
        print(
            f"ManiSkillSingleEnv created: task={task_id}, "
            f"camera={camera_name}{wrist_info}@{camera_res}, backend={sim_backend}"
        )

    def reset(self, seed=None):
        obs_raw, info = self.env.reset(seed=seed)
        obs_dict = extract_maniskill_observation(
            obs_raw, env_idx=0,
            camera_name=self.camera_name,
            use_proprio=self.use_proprio,
            wrist_camera_name=self.wrist_camera_name,
            include_wrist_image=self.include_wrist_image,
        )
        # 环境重置后势能变回0
        self._prev_potential = 0.0
        return obs_dict, info

    def step(self, action: np.ndarray):
        action = clip_maniskill_action(
            np.asarray(action, dtype=np.float32).reshape(1, -1)
        )
        obs_raw, reward, terminated, truncated, info = self.env.step(action)

        obs_dict = extract_maniskill_observation(
            obs_raw, env_idx=0,
            camera_name=self.camera_name,
            use_proprio=self.use_proprio,
            wrist_camera_name=self.wrist_camera_name,
            include_wrist_image=self.include_wrist_image,
        )
        r = float(convert_torch_to_numpy(reward).item())
        term = bool(convert_torch_to_numpy(terminated).item())
        trunc = bool(convert_torch_to_numpy(truncated).item())

        succ = extract_success_mask(info, 1)
        info["is_success"] = float(succ[0]) if succ is not None else 0.0

        if info["is_success"] > 0:
            r = 1.0
            term = True
        else:
            # Potential-based reward shaping: r = gamma * Phi(s') - Phi(s)
            gamma = 0.99
            current_potential = r  # normalized_dense ∈ [0,1]
            r = (gamma * current_potential - self._prev_potential) * 0.1
            self._prev_potential = current_potential

        return obs_dict, r, term, trunc, info

    def get_name(self) -> str:
        return self.task_id

    def close(self):
        self.env.close()
