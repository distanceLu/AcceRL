#!/usr/bin/env python3
"""
用假数据验证 AcceRL-CtrlWorld Adapter 的接口与数据格式。
输入严格模仿 AcceRL 的 WMExperience / WorldModelEnvBatch.reset 格式：
    obs: [B, num_steps_conditioning+1, 3, H, W] in [-1, 1]
    act: [B, num_steps_conditioning, action_dim]
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


def make_fake_accerl_input(batch_size=2, num_step_cond=4, act_dim=7, image_size=224):
    """生成与 AcceRL WMExperience 堆叠后一致的假数据。"""
    obs = np.random.uniform(
        -1, 1, (batch_size, num_step_cond + 1, 3, image_size, image_size)
    ).astype(np.float32)
    act = np.random.randn(batch_size, num_step_cond, act_dim).astype(np.float32)
    instructions = ["pick up the black bowl and place it on the plate"] * batch_size
    return obs, act, instructions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--svd-model-path", type=str, default=None)
    parser.add_argument("--clip-model-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-cams", type=int, default=2)
    parser.add_argument("--num-step-cond", type=int, default=4)
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
        num_frames_pred=1,
        num_inference_steps=4,
    )

    obs_np, act_np, instructions = make_fake_accerl_input(
        args.batch_size, args.num_step_cond, 7, args.image_size
    )
    obs_t = torch.from_numpy(obs_np).to(args.device)
    act_t = torch.from_numpy(act_np).to(args.device)

    print("\n[1/3] Testing reset...")
    current_obs, info = env.reset(obs_t, act_t, instructions)
    print(f"  current_obs shape: {tuple(current_obs.shape)}")  # [B,3,H,W]
    assert current_obs.shape == (args.batch_size, 3, args.image_size, args.image_size)

    print("\n[2/3] Testing step...")
    for step in range(args.num_steps):
        action = torch.randn(args.batch_size, 7, device=args.device, dtype=act_t.dtype)
        next_obs, rew, end, trunc, info = env.step(action)
        print(
            f"  step {step}: next_obs={tuple(next_obs.shape)}, "
            f"rew={tuple(rew.shape)}, end={tuple(end.shape)}, "
            f"alive={info['alive'].sum()}/{args.batch_size}"
        )
        assert next_obs.shape == (args.batch_size, 3, args.image_size, args.image_size)
        assert rew.shape == (args.batch_size,)
        assert end.shape == (args.batch_size,)

    print("\n[3/3] Testing imagine...")
    imagined = env.imagine(obs_t, act_t, instructions)
    for k, v in imagined.items():
        print(f"  {k}: {tuple(v.shape)}")
    assert imagined["obs"].shape[0] == args.batch_size
    assert imagined["act"].shape[0] == args.batch_size

    print("\nAll AcceRL-CtrlWorld Adapter smoke tests passed.")


if __name__ == "__main__":
    main()
