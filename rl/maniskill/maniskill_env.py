"""Single-environment ManiSkill wrapper used by the RL workers.

DrawTriangle-v1 only exposes a sparse reward in ManiSkill. For that task this
wrapper can compute a dense geometric reward from the reference-point coverage
and paint-dot validity already maintained by the simulator. No extra render
pass or image thresholding is required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) in sys.path:
    sys.path.remove(str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT))

from rl.maniskill.maniskill_utils import (
    LANGUAGE_INSTRUCTION,
    adapt_maniskill_action,
    build_maniskill_env,
    convert_torch_to_numpy,
    extract_maniskill_observation,
    extract_success_mask,
)


DRAW_TRIANGLE_TASK = "DrawTriangle-v1"
DRAW_TRIANGLE_REWARD_MODES = ("geometric_dense", "sparse")


class ManiSkillSingleEnv:
    """Single ManiSkill env with the same API as ``LiberoEnvWrapper``."""

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
        drawtriangle_reward_mode: str = "geometric_dense",
        drawtriangle_coverage_coef: float = 1.0,
        drawtriangle_overflow_coef: float = 0.3,
        drawtriangle_approach_coef: float = 0.1,
        drawtriangle_success_bonus: float = 1.0,
    ):
        if drawtriangle_reward_mode not in DRAW_TRIANGLE_REWARD_MODES:
            raise ValueError(
                f"drawtriangle_reward_mode must be one of {DRAW_TRIANGLE_REWARD_MODES}, "
                f"got {drawtriangle_reward_mode!r}"
            )
        for name, value in (
            ("drawtriangle_coverage_coef", drawtriangle_coverage_coef),
            ("drawtriangle_overflow_coef", drawtriangle_overflow_coef),
            ("drawtriangle_approach_coef", drawtriangle_approach_coef),
            ("drawtriangle_success_bonus", drawtriangle_success_bonus),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")

        self.task_id = task_id
        self.camera_name = camera_name
        self.wrist_camera_name = wrist_camera_name
        self.include_wrist_image = wrist_camera_name is not None
        self.use_proprio = use_proprio
        self.task_description = language_instruction or LANGUAGE_INSTRUCTION
        self.drawtriangle_reward_mode = drawtriangle_reward_mode
        self.drawtriangle_coverage_coef = float(drawtriangle_coverage_coef)
        self.drawtriangle_overflow_coef = float(drawtriangle_overflow_coef)
        self.drawtriangle_approach_coef = float(drawtriangle_approach_coef)
        self.drawtriangle_success_bonus = float(drawtriangle_success_bonus)

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
        self.action_dim = int(self.env.action_space.shape[-1])
        if self.action_dim not in (6, 7):
            self.env.close()
            raise ValueError(
                f"{task_id} exposes an unsupported {self.action_dim}-D action space; "
                "only 6-D and 7-D actions are supported"
            )

        self._prev_potential = 0.0
        self._drawtriangle_prev_score = 0.0
        self._drawtriangle_best_approach = 0.0
        self.last_reward_metrics: Dict[str, float] = {}
        wrist_info = f" + wrist={wrist_camera_name}" if wrist_camera_name else ""
        reward_info = (
            f", reward={drawtriangle_reward_mode}"
            if task_id == DRAW_TRIANGLE_TASK
            else ""
        )
        print(
            f"ManiSkillSingleEnv created: task={task_id}, "
            f"camera={camera_name}{wrist_info}@{camera_res}, backend={sim_backend}, "
            f"action_dim={self.action_dim}{reward_info}"
        )

    def reset(self, seed=None):
        obs_raw, info = self.env.reset(seed=seed)
        obs_dict = self._extract_observation(obs_raw)
        self._prev_potential = 0.0
        self._drawtriangle_best_approach = 0.0
        if self._uses_drawtriangle_dense_reward:
            metrics = self._drawtriangle_metrics()
            self._drawtriangle_prev_score = metrics["score"]
            metrics.update(reward=0.0, score_delta=0.0, success_bonus=0.0, original_reward=0.0)
            self.last_reward_metrics = metrics
        else:
            self._drawtriangle_prev_score = 0.0
            self.last_reward_metrics = {}
        return obs_dict, info

    @property
    def _uses_drawtriangle_dense_reward(self) -> bool:
        return (
            self.task_id == DRAW_TRIANGLE_TASK
            and self.drawtriangle_reward_mode == "geometric_dense"
        )

    def _extract_observation(self, obs_raw: Dict[str, Any]) -> Dict[str, Any]:
        return extract_maniskill_observation(
            obs_raw,
            env_idx=0,
            camera_name=self.camera_name,
            use_proprio=self.use_proprio,
            wrist_camera_name=self.wrist_camera_name,
            include_wrist_image=self.include_wrist_image,
        )

    def step(self, action: np.ndarray):
        action = adapt_maniskill_action(
            np.asarray(action, dtype=np.float32).reshape(1, -1),
            self.action_dim,
        )
        obs_raw, original_reward, terminated, truncated, info = self.env.step(action)
        obs_dict = self._extract_observation(obs_raw)
        original_r = float(convert_torch_to_numpy(original_reward).item())
        term = bool(convert_torch_to_numpy(terminated).item())
        trunc = bool(convert_torch_to_numpy(truncated).item())

        succ = extract_success_mask(info, 1)
        success = float(succ[0]) if succ is not None else 0.0
        info["is_success"] = success

        if self._uses_drawtriangle_dense_reward:
            r = self._drawtriangle_dense_reward(original_r, success)
            info["dense_reward"] = dict(self.last_reward_metrics)
            if success > 0:
                term = True
        elif success > 0:
            r = 1.0
            term = True
        else:
            # Preserve the existing shaping behavior for ManiSkill tasks that
            # actually expose normalized dense rewards.
            gamma = 0.99
            current_potential = original_r
            r = (gamma * current_potential - self._prev_potential) * 0.1
            self._prev_potential = current_potential

        return obs_dict, r, term, trunc, info

    def _drawtriangle_dense_reward(self, original_reward: float, success: float) -> float:
        metrics = self._drawtriangle_metrics()
        score_delta = metrics["score"] - self._drawtriangle_prev_score
        success_bonus = self.drawtriangle_success_bonus if success > 0 else 0.0
        reward = score_delta + success_bonus
        self._drawtriangle_prev_score = metrics["score"]
        metrics.update(
            reward=float(reward),
            score_delta=float(score_delta),
            success_bonus=float(success_bonus),
            original_reward=float(original_reward),
        )
        self.last_reward_metrics = metrics
        return float(reward)

    def _drawtriangle_metrics(self) -> Dict[str, float]:
        base_env = self.env.unwrapped
        ref_dist = np.asarray(convert_torch_to_numpy(base_env.ref_dist))[0].astype(bool)
        dots_dist = np.asarray(convert_torch_to_numpy(base_env.dots_dist))[0]
        valid_mask = dots_dist >= 0
        valid_dot_count = int(valid_mask.sum())
        on_target_count = int(np.logical_and(valid_mask, dots_dist > 0).sum())
        off_target_count = int(np.logical_and(valid_mask, dots_dist == 0).sum())
        max_dots = max(int(getattr(base_env, "MAX_DOTS", dots_dist.size)), 1)

        coverage = float(ref_dist.mean()) if ref_dist.size else 0.0
        overflow = off_target_count / max_dots
        on_target_ratio = on_target_count / max(valid_dot_count, 1)
        approach = self._drawtriangle_approach()
        self._drawtriangle_best_approach = max(self._drawtriangle_best_approach, approach)
        score = (
            self.drawtriangle_coverage_coef * coverage
            - self.drawtriangle_overflow_coef * overflow
            + self.drawtriangle_approach_coef * self._drawtriangle_best_approach
        )
        return {
            "score": float(score),
            "coverage": coverage,
            "overflow": float(overflow),
            "approach": float(approach),
            "best_approach": float(self._drawtriangle_best_approach),
            "valid_dot_count": float(valid_dot_count),
            "on_target_dot_count": float(on_target_count),
            "off_target_dot_count": float(off_target_count),
            "on_target_ratio": float(on_target_ratio),
            "covered_reference_count": float(ref_dist.sum()),
            "reference_count": float(ref_dist.size),
        }

    def _drawtriangle_approach(self) -> float:
        """Return closeness of the TCP to any target point in [0, 1]."""
        base_env = self.env.unwrapped
        tcp = np.asarray(convert_torch_to_numpy(base_env.agent.tcp.pose.p))[0]
        triangle = np.asarray(convert_torch_to_numpy(base_env.triangles))[0]
        canvas_z = float(base_env.CANVAS_THICKNESS + base_env.DOT_THICKNESS / 2)
        target_xyz = np.concatenate(
            [triangle, np.full((triangle.shape[0], 1), canvas_z, dtype=triangle.dtype)],
            axis=-1,
        )
        min_distance = float(np.linalg.norm(target_xyz - tcp[None, :], axis=-1).min())
        # The initial TCP is roughly 0.4--0.5 m from the canvas. Only progress
        # toward the nearest target point is rewarded, and best_approach makes
        # this one-way so oscillation cannot farm reward.
        return float(np.clip(1.0 - min_distance / 0.5, 0.0, 1.0))

    def get_name(self) -> str:
        return self.task_id

    def close(self):
        self.env.close()


def _episode_seed(trajectory_path: Path, trajectory_key: str, fallback: int) -> int:
    metadata_path = trajectory_path.with_suffix(".json")
    if not metadata_path.exists() or not trajectory_key.startswith("traj_"):
        return fallback
    episode_id = int(trajectory_key.removeprefix("traj_"))
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    for episode in metadata.get("episodes", []):
        if int(episode.get("episode_id", -1)) == episode_id:
            return int(episode.get("episode_seed", fallback))
    return fallback


def _annotate_frame(frame: np.ndarray, lines: list[str], scale: int = 3) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.fromarray(frame).resize(
        (frame.shape[1] * scale, frame.shape[0] * scale),
        resample=Image.Resampling.NEAREST,
    )
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24
        )
    except OSError:
        font = ImageFont.load_default()
    line_height = 30
    box_width = max(int(draw.textlength(line, font=font)) for line in lines) + 20
    draw.rectangle((5, 5, box_width, 15 + line_height * len(lines)), fill=(0, 0, 0))
    for idx, line in enumerate(lines):
        draw.text((12, 10 + idx * line_height), line, fill=(255, 255, 255), font=font)
    return np.asarray(image)


def main() -> None:
    """Replay a successful demonstration and visualize the dense reward."""
    parser = argparse.ArgumentParser(description="Test DrawTriangle geometric dense reward")
    parser.add_argument(
        "--trajectory-path",
        type=Path,
        default=Path(
            "/mnt/data/lcx/data/maniskill/DrawTriangle-v1/motionplanning/"
            "trajectory.rgbd.pd_ee_delta_pose.physx_cpu.h5"
        ),
    )
    parser.add_argument("--trajectory-key", default="traj_0")
    parser.add_argument(
        "--video-path", type=Path, default=Path("runs/drawtriangle_dense_reward.mp4")
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--camera-res", type=int, default=224)
    parser.add_argument("--sim-backend", choices=("cpu", "gpu", "auto"), default="cpu")
    parser.add_argument("--render-backend", default="gpu")
    parser.add_argument("--coverage-coef", type=float, default=1.0)
    parser.add_argument("--overflow-coef", type=float, default=0.3)
    parser.add_argument("--approach-coef", type=float, default=0.1)
    parser.add_argument("--success-bonus", type=float, default=1.0)
    args = parser.parse_args()

    import h5py
    import imageio.v2 as imageio

    with h5py.File(args.trajectory_path, "r") as trajectory_file:
        if args.trajectory_key not in trajectory_file:
            raise KeyError(
                f"{args.trajectory_key!r} not found in {args.trajectory_path}; "
                f"examples: {list(trajectory_file.keys())[:5]}"
            )
        actions = np.asarray(trajectory_file[args.trajectory_key]["actions"])
    actions = actions[: args.max_steps]
    seed = _episode_seed(args.trajectory_path, args.trajectory_key, args.seed)

    env = ManiSkillSingleEnv(
        task_id=DRAW_TRIANGLE_TASK,
        camera_name="base_camera",
        camera_res=args.camera_res,
        max_episode_steps=max(len(actions), 1),
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        robot_uids="panda_stick",
        drawtriangle_reward_mode="geometric_dense",
        drawtriangle_coverage_coef=args.coverage_coef,
        drawtriangle_overflow_coef=args.overflow_coef,
        drawtriangle_approach_coef=args.approach_coef,
        drawtriangle_success_bonus=args.success_bonus,
    )
    args.video_path.parent.mkdir(parents=True, exist_ok=True)
    episode_return = 0.0
    try:
        obs, _ = env.reset(seed=seed)
        with imageio.get_writer(
            args.video_path,
            fps=20,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=16,
        ) as writer:
            initial = env.last_reward_metrics
            writer.append_data(
                _annotate_frame(
                    obs["full_image"],
                    [
                        "DrawTriangle geometric dense reward",
                        "step=0 reward=0.000000 return=0.000000",
                        f"coverage={initial['coverage']:.4f} score={initial['score']:.4f}",
                        f"overflow={initial['overflow']:.4f} approach={initial['approach']:.4f}",
                        "dots valid/on/off=0/0/0",
                    ],
                )
            )
            for step, action in enumerate(actions, start=1):
                obs, reward, terminated, truncated, info = env.step(action)
                episode_return += reward
                metrics = info["dense_reward"]
                writer.append_data(
                    _annotate_frame(
                        obs["full_image"],
                        [
                            "DrawTriangle geometric dense reward",
                            f"step={step} reward={reward:+.6f} return={episode_return:+.6f}",
                            f"coverage={metrics['coverage']:.4f} overflow={metrics['overflow']:.4f}",
                            f"score={metrics['score']:.4f} delta={metrics['score_delta']:+.6f}",
                            "dots valid/on/off="
                            f"{int(metrics['valid_dot_count'])}/"
                            f"{int(metrics['on_target_dot_count'])}/"
                            f"{int(metrics['off_target_dot_count'])} "
                            f"approach={metrics['approach']:.4f}",
                            f"official_success={bool(info['is_success'])}",
                        ],
                    )
                )
                print(
                    f"step={step:03d} reward={reward:+.6f} "
                    f"coverage={metrics['coverage']:.4f} "
                    f"on/off={int(metrics['on_target_dot_count'])}/"
                    f"{int(metrics['off_target_dot_count'])} "
                    f"success={bool(info['is_success'])}",
                    flush=True,
                )
                if terminated or truncated:
                    break
    finally:
        env.close()

    print(f"episode_return={episode_return:+.6f}")
    print(f"Saved dense reward debug video to: {args.video_path.resolve()}")


if __name__ == "__main__":
    main()
