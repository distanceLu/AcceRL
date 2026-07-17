"""Replay seven ManiSkill task trajectories with panda_wristcam dual cameras.

This script prepares RGBD 224x224 replay files for the seven additional tasks
used in the 10-task ManiSkill mixture. It prefers motion-planning trajectories
when present and falls back to existing RL pd_ee_delta_pose trajectories.

Usage:
    cd /cpfs01/lcx_stu4_workspace/openvla_oft_rl
    python rl/maniskill/replay_224_two_cam_7tasks.py

Optional overrides:
    DATA_ROOT=/path/to/maniskill/demos \
    TASKS=PushCube-v1,PullCube-v1 \
    SKIP_MISSING=1 \
    FORCE=1 \
    python rl/maniskill/replay_224_two_cam_7tasks.py
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable, Optional

# Keep Vulkan/SAPIEN rendering on a single visible GPU. These must be set before
# importing gymnasium/mani_skill because SAPIEN initializes rendering at import
# and environment construction time.
RENDER_GPU = os.environ.get("RENDER_GPU", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", RENDER_GPU)
os.environ.setdefault("VULKAN_VISIBLE_DEVICES", RENDER_GPU)
os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")

# Monkey-patch gymnasium.make BEFORE importing replay_trajectory.
import gymnasium as gym

DEFAULT_TASKS = (
    "PushCube-v1",
    "PullCube-v1",
    "PokeCube-v1",
    "LiftPegUpright-v1",
    "RollBall-v1",
    "PlaceSphere-v1",
    "PullCubeTool-v1",
)

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/data/disk1/lcx_stu4/maniskill/demos"))
SKIP_MISSING = os.environ.get("SKIP_MISSING", "0") == "1"
FORCE = os.environ.get("FORCE", "0") == "1"
RENDER_BACKEND = os.environ.get("RENDER_BACKEND", "sapien_cuda:0")

_current_task_for_log = "unknown"
_original_gym_make = gym.make


def _split_tasks(value: Optional[str]) -> tuple[str, ...]:
    if not value:
        return DEFAULT_TASKS
    tasks = tuple(item.strip() for item in value.split(",") if item.strip())
    if not tasks:
        raise ValueError("TASKS was provided but no valid task names were found.")
    return tasks


def _candidate_paths(task: str) -> list[Path]:
    return [
        DATA_ROOT / task / "motionplanning" / "trajectory.h5",
        DATA_ROOT / task / "rl" / "trajectory.none.pd_ee_delta_pose.physx_cuda.h5",
    ]


def _resolve_source_traj(task: str) -> Path:
    source_override = os.environ.get(f"{task.upper().replace('-', '_')}_SOURCE_TRAJ")
    if source_override:
        return Path(source_override)

    source_traj = next((path for path in _candidate_paths(task) if path.exists()), None)
    if source_traj is None:
        candidates = "\n  ".join(str(path) for path in _candidate_paths(task))
        raise FileNotFoundError(f"No source trajectory found for {task}. Tried:\n  {candidates}")
    return source_traj


def _patched_gym_make(env_id, **kwargs):
    """Intercept gym.make to inject dual-camera 224x224 rendering settings."""
    kwargs["robot_uids"] = "panda_wristcam"
    if RENDER_BACKEND:
        kwargs["render_backend"] = RENDER_BACKEND
    kwargs["sensor_configs"] = {
        "base_camera": {"width": 224, "height": 224},
        "hand_camera": {"width": 224, "height": 224},
    }
    print(
        f"[replay_224_two_cam_7tasks:{_current_task_for_log}] "
        "Injecting robot_uids='panda_wristcam', "
        f"render_backend={RENDER_BACKEND!r}, sensor_configs 224x224 "
        f"into gym.make({env_id!r})"
    )
    return _original_gym_make(env_id, **kwargs)


gym.make = _patched_gym_make

# Now import replay_trajectory; it will use the patched gym.make.
from mani_skill.trajectory.replay_trajectory import Args, main as replay_main


def _copy_source_trajectory(task: str, source_traj: Path, output_dir: Path) -> Path:
    if not source_traj.exists():
        raise FileNotFoundError(f"{task} source trajectory not found: {source_traj}")

    output_dir.mkdir(parents=True, exist_ok=True)
    replay_traj = output_dir / source_traj.name
    if FORCE or not replay_traj.exists() or replay_traj.resolve() != source_traj.resolve():
        shutil.copy2(source_traj, replay_traj)

    source_json = source_traj.with_suffix(".json")
    if source_json.exists():
        replay_json = output_dir / source_json.name
        if FORCE or not replay_json.exists() or replay_json.resolve() != source_json.resolve():
            shutil.copy2(source_json, replay_json)
    else:
        print(f"Warning: source json not found next to trajectory: {source_json}")

    return replay_traj


def _replay_task(task: str) -> None:
    global _current_task_for_log
    _current_task_for_log = task

    source_traj = _resolve_source_traj(task)
    output_dir = Path(os.environ.get(
        f"{task.upper().replace('-', '_')}_OUTPUT_DIR",
        str(DATA_ROOT / task / "motionplanning_rgbd_224_two_cam"),
    ))
    replay_traj = _copy_source_trajectory(task, source_traj, output_dir)

    print("=" * 100)
    print(f"Task: {task}")
    print(f"Source trajectory: {source_traj}")
    print(f"Replaying trajectory from: {replay_traj}")
    print(f"Replay output directory: {output_dir}")

    args = Args(
        traj_path=str(replay_traj),
        sim_backend="physx_cpu",
        obs_mode="rgbd",
        target_control_mode="pd_ee_delta_pose",
        save_traj=True,
        use_env_states=False,
        allow_failure=False,
        verbose=True,
    )
    replay_main(args)


def main(tasks: Iterable[str]) -> None:
    completed: list[str] = []
    skipped: list[str] = []
    for task in tasks:
        try:
            _replay_task(task)
            completed.append(task)
        except FileNotFoundError as exc:
            if not SKIP_MISSING:
                raise
            print(f"[SKIP] {exc}")
            skipped.append(task)

    print("=" * 100)
    print(f"Completed tasks ({len(completed)}): {completed}")
    if skipped:
        print(f"Skipped tasks ({len(skipped)}): {skipped}")


if __name__ == "__main__":
    main(_split_tasks(os.environ.get("TASKS")))
