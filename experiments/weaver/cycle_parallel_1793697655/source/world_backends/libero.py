"""Explicit LIBERO observation adapter; never labels EEF state as robot joints."""
from __future__ import annotations

ACTION_SCHEMA = "libero_eef_delta_7_v1"
STATE_SCHEMA = "libero_eef_pos_axisangle_gripper_v1"


def validate_online_schema(config, action_dim=7):
    if config.backend != "weaver":
        return
    schema = config.schema
    expected = {"domain": "libero", "action_schema_id": ACTION_SCHEMA,
                "state_schema_id": STATE_SCHEMA, "action_dim": action_dim,
                "alignment": "action_into_frame", "image_value_range": "minus_one_one"}
    for key, value in expected.items():
        if schema.get(key) != value:
            raise ValueError(f"LIBERO collector requires {key}={value!r}; incompatible robot schema")
    if list(schema["image_keys"]) != ["full_image", "wrist_image"]:
        raise ValueError("LIBERO collector requires ordered full_image, wrist_image views")


def environment_dt(env):
    """Read the simulation time scale, walking explicit wrapper links only."""
    visited = set()
    pending = [env]
    while pending:
        obj = pending.pop()
        if id(obj) in visited:
            continue
        visited.add(id(obj))
        for key in ("control_freq",):
            frequency = getattr(obj, key, None)
            if frequency is not None and float(frequency) > 0:
                return 1.0 / float(frequency)
        for key in ("env", "_env", "unwrapped"):
            child = getattr(obj, key, None)
            if child is not None and child is not obj:
                pending.append(child)
    raise ValueError("Cannot verify LIBERO control timestep from environment; explicit adapter required")


def backend_window_payload(config, observations, states, actions, rewards, success,
                           terminated, truncated, start, end, instruction, episode_id, sample_id):
    """Serialize measured frame labels and state, with the reset action masked."""
    import numpy as np
    from .data import EpisodeWindow
    count = end - start
    action_mask = np.ones(count, dtype=bool)
    if start == 0:
        action_mask[0] = False
    dt = np.full(count, float(config.schema["dt_seconds"]), dtype=np.float32)
    dt[~action_mask] = 0
    window = EpisodeWindow(
        observations=observations[start:end], actions=actions[start:end], states=states[start:end],
        rewards=np.asarray(rewards[start:end], dtype=np.float32),
        success=np.asarray(success[start:end], dtype=bool),
        terminated=np.asarray(terminated[start:end], dtype=bool),
        truncated=np.asarray(truncated[start:end], dtype=bool),
        valid_mask=np.ones(count, dtype=bool), action_mask=action_mask, dt=dt,
        episode_id=str(episode_id), sample_id=sample_id, task_id=instruction,
        instruction=instruction, schema=dict(config.schema), label_source="libero_env_step",
    ).validate(config)
    # Images/actions already live in WMExperience, and are attached by the
    # training owner after rank sharding. The rest remains a CPU-only payload.
    return {"states": window.states, "rewards": window.rewards, "success": window.success,
            "done": window.terminated, "truncated": window.truncated, "mask": window.valid_mask,
            "action_mask": window.action_mask, "dt": window.dt,
            "dt_seconds": config.schema["dt_seconds"], "schema": dict(config.schema),
            "instruction": instruction, "episode_id": str(episode_id), "sample_id": sample_id,
            "label_source": "libero_env_step"}
