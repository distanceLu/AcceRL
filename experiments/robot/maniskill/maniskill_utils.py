"""Utils for evaluating policies in ManiSkill simulation environments.

Mirrors the role of ``experiments.robot.libero.libero_utils`` for the LIBERO
suite. The functions here intentionally do *not* depend on the rest of
``openvla_oft_rl`` so they can be unit-tested in isolation; the eval-loop glue
lives next to ``run_libero_real_eval`` in ``vla-scripts/finetune_debug.py``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np

LANGUAGE_INSTRUCTION = "pick up the red cube and place it at the green target"
MANISKILL_GRIPPER_RANGE: Tuple[float, float] = (-1.0, 1.0)

import os

def _to_numpy(value: Any) -> np.ndarray:
    """Convert a torch tensor / numpy array / list / scalar into a numpy array.

    ManiSkill 3.x's GPU-backed vectorized envs return torch tensors, while
    the older / CPU code paths return numpy. We normalize everything to numpy
    so downstream code (image preprocessing, action dispatch, success bookkeeping)
    can stay framework-agnostic.
    """
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except ImportError:
        pass
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def convert_torch_to_numpy(value: Any) -> np.ndarray:
    """Public alias kept for parity with the plan / other call sites."""
    return _to_numpy(value)


def quat_wxyz_to_axisangle(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert a (..., 4) wxyz quaternion to (..., 3) axis-angle (rotvec).

    Matches the conversion used by
    ``rlds_dataset_builder/maniskill_pickcube/maniskill_pickcube_dataset_builder.py``
    so policy-side proprio is in the exact same coordinate system as training.
    """
    from scipy.spatial.transform import Rotation as R

    q = np.asarray(quat_wxyz, dtype=np.float64)
    q_xyzw = np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)
    return R.from_quat(q_xyzw).as_rotvec().astype(np.float32)






##########################
def print_gpu_hits(tag, pid):
    print(f"\n=== GPU hits: {tag} ===", flush=True)
    print(query_pid_gpus(pid), flush=True)


def query_pid_gpus(pid: int):
    try:
        import pynvml

        pynvml.nvmlInit()
        hits = []

        getters = [
            ("compute", pynvml.nvmlDeviceGetComputeRunningProcesses),
            ("graphics", getattr(pynvml, "nvmlDeviceGetGraphicsRunningProcesses", None)),
        ]

        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()

            for proc_type, getter in getters:
                if getter is None:
                    continue

                try:
                    procs = getter(h)
                except Exception:
                    continue

                for p in procs:
                    if int(p.pid) == int(pid):
                        used = getattr(p, "usedGpuMemory", 0)
                        hits.append({
                            "gpu_index": i,
                            "name": name,
                            "type": proc_type,
                            "used_mb": used / 1024 / 1024,
                        })

        return hits
    except Exception as e:
        return [{"error": repr(e)}]
##########################

def build_maniskill_env(
    task_id: str,
    num_envs: int,
    obs_mode: str = "rgbd",
    control_mode: str = "pd_ee_delta_pose",
    camera_name: str = "base_camera",
    wrist_camera_name: Optional[str] = None,
    camera_res: int = 128,
    max_episode_steps: Optional[int] = None,
    sim_backend: str = "gpu",
    render_backend: Optional[str] = None,
    robot_uids: Optional[str] = None,
):
    """Build a (vectorized) ManiSkill env tuned for VLA-style evaluation.

    Notes:
      - We force the configured camera(s) to ``camera_res x camera_res`` so the
        observation matches the resolution the OpenVLA backbone was trained on
        (training data is typically 224x224 base/wrist RGB).
      - ``render_mode`` is intentionally left None: success rate evaluation does
        not need an off-screen render pass, and disabling it removes the
        viewer overhead.
    """
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401  (registers ManiSkill tasks with gym)

    sensor_cfg: Dict[str, Dict[str, int]] = {
        camera_name: {"width": camera_res, "height": camera_res}
    }
    if wrist_camera_name is not None:
        sensor_cfg[wrist_camera_name] = {"width": camera_res, "height": camera_res}

    # debug
    pid = os.getpid()


    env_kwargs: Dict[str, Any] = dict(
        obs_mode=obs_mode,
        control_mode=control_mode,
        sensor_configs=sensor_cfg,
        sim_backend=sim_backend,
        render_mode=None
    )
    
    if render_backend is not None:
        env_kwargs["render_backend"] = render_backend
    if robot_uids is not None:
        env_kwargs["robot_uids"] = robot_uids
    if num_envs is not None and num_envs > 0:
        env_kwargs["num_envs"] = num_envs
    if max_episode_steps is not None and max_episode_steps > 0:
        env_kwargs["max_episode_steps"] = max_episode_steps
    
    # debug
    # def patch_sapien_device_debug():
    #     import os
    #     import sapien

    #     if getattr(sapien, "_debug_device_patched", False):
    #         return

    #     original_device = sapien.Device

    #     def debug_device(device_str=None, *args, **kwargs):
    #         print(f"\n=== sapien.Device called with: {device_str} ===", flush=True)
    #         print_gpu_hits(f"before sapien.Device({device_str})", os.getpid())

    #         dev = original_device(device_str, *args, **kwargs)

    #         print_gpu_hits(f"after sapien.Device({device_str})", os.getpid())
    #         return dev

    #     sapien.Device = debug_device
    #     sapien._debug_device_patched = True

    # patch_sapien_device_debug()

    # print_gpu_hits("before gym.make", pid)

    env = gym.make(task_id, **env_kwargs)

    # print_gpu_hits("after gym.make", pid)


    return env


def _index_nested(value: Any, env_idx: int) -> Any:
    """Recursively index the leading (num_envs) dimension of a possibly-nested dict.

    ManiSkill's vectorized obs is a dict-of-dicts whose leaf tensors all have
    leading dim ``num_envs``. We slice out one env at a time so the per-env dict
    can be fed straight into ``get_vla_action_batch``.
    """
    if isinstance(value, dict):
        return {k: _index_nested(v, env_idx) for k, v in value.items()}
    arr = _to_numpy(value)
    if arr.ndim == 0:
        return arr
    return arr[env_idx]


def extract_maniskill_observation(
    obs: Dict[str, Any],
    env_idx: int,
    camera_name: str = "base_camera",
    use_proprio: bool = False,
    wrist_camera_name: Optional[str] = None,
    include_wrist_image: bool = False,
) -> Dict[str, Any]:
    """Extract the observation dict expected by ``get_vla_action_batch`` from a
    single sub-env of a ManiSkill batched obs.

    Returned keys:
      - ``full_image`` (uint8 ``(H, W, 3)``): primary RGB observation.
      - ``wrist_image`` (uint8 ``(H, W, 3)``, only if ``include_wrist_image=True``):
        wrist-camera RGB observation.
      - ``state`` (float32 ``(8,)``, only if ``use_proprio=True``):
        ``[tcp_xyz(3), tcp_axis_angle(3), finger_left, finger_right]`` (matches
        the layout of the training RLDS state).
    """
    rgb = _index_nested(obs["sensor_data"][camera_name]["rgb"], env_idx)
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    rgb = np.ascontiguousarray(rgb)

    out: Dict[str, Any] = {"full_image": rgb}

    if include_wrist_image:
        if wrist_camera_name is None:
            raise ValueError("wrist_camera_name must be provided when include_wrist_image=True")
        if wrist_camera_name not in obs["sensor_data"]:
            available = ", ".join(obs["sensor_data"].keys())
            raise KeyError(
                f"Camera '{wrist_camera_name}' not found in obs['sensor_data']; "
                f"available cameras: {available}"
            )
        wrist_rgb = _index_nested(obs["sensor_data"][wrist_camera_name]["rgb"], env_idx)
        wrist_rgb = np.asarray(wrist_rgb)
        if wrist_rgb.dtype != np.uint8:
            wrist_rgb = np.clip(wrist_rgb, 0, 255).astype(np.uint8)
        out["wrist_image"] = np.ascontiguousarray(wrist_rgb)

    if use_proprio:
        tcp_pose = _index_nested(obs["extra"]["tcp_pose"], env_idx)
        tcp_pose = np.asarray(tcp_pose, dtype=np.float64)
        tcp_xyz = tcp_pose[:3].astype(np.float32)
        tcp_axisangle = quat_wxyz_to_axisangle(tcp_pose[3:7])

        qpos = _index_nested(obs["agent"]["qpos"], env_idx)
        qpos = np.asarray(qpos, dtype=np.float32)
        fingers = qpos[7:9].astype(np.float32)

        out["state"] = np.concatenate([tcp_xyz, tcp_axisangle, fingers], axis=-1).astype(np.float32)

    return out


def extract_success_mask(info: Dict[str, Any], num_envs: int) -> np.ndarray:
    """Pull a ``(num_envs,)`` bool numpy array out of ManiSkill's ``info`` dict.

    Across ManiSkill / SAPIEN minor versions, ``info["success"]`` may come back
    as a torch BoolTensor, a numpy array, a list of bools, or even a Python
    bool when ``num_envs == 1``. We coerce all of them.
    """
    if "success" not in info:
        return np.zeros(num_envs, dtype=bool)
    raw = info["success"]
    arr = _to_numpy(raw)
    if arr.ndim == 0:
        arr = np.asarray([bool(arr)] * num_envs)
    return arr.astype(bool)


def extract_done_mask(
    terminated: Any,
    truncated: Any,
    num_envs: int,
) -> np.ndarray:
    """Combine ``terminated`` and ``truncated`` into a single ``(num_envs,)`` bool array.

    Treats either flag as "this episode is finished" so the eval loop can stop
    feeding actions to that sub-env. Truncation is included so that envs which
    hit ManiSkill's internal ``max_episode_steps`` (and would otherwise
    auto-reset) are also marked done from the policy's point of view.
    """
    term = _to_numpy(terminated)
    trunc = _to_numpy(truncated)
    if term.ndim == 0:
        term = np.asarray([bool(term)] * num_envs)
    if trunc.ndim == 0:
        trunc = np.asarray([bool(trunc)] * num_envs)
    return (term.astype(bool) | trunc.astype(bool))


def clip_maniskill_action(action: np.ndarray) -> np.ndarray:
    """Clip the gripper dim of an action (or batch of actions) into ManiSkill's
    valid range. Other dims are left untouched - upstream ``_unnormalize_actions``
    in the OpenVLA model already maps them to a reasonable continuous range.
    """
    a = np.asarray(action, dtype=np.float32).copy()
    lo, hi = MANISKILL_GRIPPER_RANGE
    if a.ndim == 1:
        a[6] = float(np.clip(a[6], lo, hi))
    else:
        a[..., 6] = np.clip(a[..., 6], lo, hi)
    return a


def seeds_for_batch(base_seed: int, start_episode: int, num_envs: int) -> Iterable[int]:
    """Yield deterministic per-env seeds for one batch of episodes."""
    return [int(base_seed + start_episode + i) for i in range(num_envs)]
