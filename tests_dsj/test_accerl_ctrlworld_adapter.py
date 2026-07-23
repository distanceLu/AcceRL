#!/usr/bin/env python3
"""
用假数据验证 AcceRL-CtrlWorld 双视角 chunk 接口与数据格式。
    obs: [B, num_steps_conditioning, num_cams, 3, H, W] in [-1, 1]
    act condition: [B, num_history + num_frames, action_dim]
    instruction: list of str
"""
import sys, os
ACCE_RL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ACCE_RL_ROOT)

import argparse
from dataclasses import dataclass
import torch
import numpy as np

from ctrl_world_env_batch import CtrlWorldEnvBatch
from ctrl_world.config import wm_args
from ctrl_world.models.ctrl_world import CrtlWorld


@dataclass
class FakeCfg:
    horizon: int = 16


def make_fake_accerl_input(
    batch_size=2, num_step_cond=7, num_cams=2, act_dim=7, image_size=224
):
    """生成与 AcceRL WMExperience 堆叠后一致的假数据。"""
    obs = np.random.uniform(
        -1, 1, (batch_size, num_step_cond, num_cams, 3, image_size, image_size)
    ).astype(np.float32)
    act = np.random.uniform(
        -0.5, 0.5, (batch_size, num_step_cond + 4, act_dim)
    ).astype(np.float32)
    instructions = ["pick up the black bowl and place it on the plate"] * batch_size
    return obs, act, instructions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--svd-model-path", type=str, default=None)
    parser.add_argument("--clip-model-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-cams", type=int, default=2)
    parser.add_argument("--num-step-cond", type=int, default=7)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-steps", type=int, default=3)
    args = parser.parse_args()

    cfg = wm_args()
    if args.svd_model_path:
        cfg.svd_model_path = args.svd_model_path
    if args.clip_model_path:
        cfg.clip_model_path = args.clip_model_path
    cfg.num_cams = args.num_cams

    print("Loading CrtlWorld...")
    ctrl_world = CrtlWorld(cfg).to(args.device).to(torch.bfloat16)
    ctrl_world.eval()

    env_cfg = FakeCfg(horizon=16)
    env = CtrlWorldEnvBatch(
        ctrl_world_model=ctrl_world,
        cfg=env_cfg,
        reward_model=None,          # smoke test：不加载 reward model
        reward_cfg=None,
        processor=None,
        torch_dtype=torch.bfloat16,
        instructions=["pick up the black bowl"] * args.batch_size,
        num_cams=args.num_cams,
        num_frames_pred=5,
        num_inference_steps=4,
        condition_low=torch.full((7,), -1.0),
        condition_high=torch.full((7,), 1.0),
    )

    obs_np, act_np, instructions = make_fake_accerl_input(
        args.batch_size, args.num_step_cond, args.num_cams, 7, args.image_size
    )
    obs_t = torch.from_numpy(obs_np).to(args.device)
    act_t = torch.from_numpy(act_np).to(args.device)

    print("\n[1/2] Testing true multi-view latent initialization...")
    history, current = env.init_latent_state(obs_t)
    assert history.shape[:3] == (args.batch_size, cfg.num_history, 4)
    assert current.shape[:2] == (args.batch_size, 4)

    print("\n[2/2] Testing 5-frame chunk inference...")
    future_obs, future_latents = env.predict_chunk_stateless(
        current_latent=current,
        latent_history=history,
        action_condition=act_t[:, : cfg.num_history + 5],
        instructions=instructions,
        output_size=(args.image_size, args.image_size),
    )
    assert future_obs.shape == (
        args.batch_size, 4, args.num_cams, 3, args.image_size, args.image_size
    )
    assert future_latents.shape[:3] == (args.batch_size, 4, 4)

    print("\nAll AcceRL-CtrlWorld Adapter smoke tests passed.")


if __name__ == "__main__":
    main()
