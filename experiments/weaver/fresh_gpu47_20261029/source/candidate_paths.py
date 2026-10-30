"""CPU-only path and identity boundary for this isolated source candidate."""
from __future__ import annotations

import os
from pathlib import Path
import re

SOURCE = Path(__file__).resolve().parent
RUN_ID_PREFIX = "weaverintegrated_"
PROTECTED_ROOTS = (
    Path("/mnt/data/lcx1/dong/need_data/weaver_optimized_gpu47_20261027_v2"),
    Path("/mnt/data/lcx1/dong/need_data/weaver_optimized_20261027"),
    Path("/mnt/data/lcx1/dong/need_data/cuda_ipc_weight_sync_gpu47_20261022"),
    Path("/mnt/data/lcx1/dong/need_data/archive"),
    SOURCE.parent / "analysis",
    SOURCE,
    SOURCE.parent / "world_backends_v1",
    Path("/mnt/data/lcx1/dong/need_data/weaver_libero_offline_20261022"),
    Path("/mnt/data/lcx1/dong/need_data/initial_state_budget_gpu47_20261026"),
    Path("/mnt/data/lcx1/dong/need_data/initial_state_budget_20261026"),
    Path("/mnt/data/lcx1/dong/need_data/weaver_policy_gpu47_20261026"),
    Path("/mnt/data/lcx1/dong/need_data/weaver_policy_20261026"),
    Path("/mnt/data/lcx1/dong/need_data/weaver_online_gpu47_20261027"),
    Path("/mnt/data/lcx1/dong/need_data/weaver_online_20261027"),
)


def experiment_root(value=None, *, environ=None):
    env = os.environ if environ is None else environ
    value = value if value is not None else env.get("ACCERL_EXPERIMENT_ROOT")
    if not value or not Path(value).expanduser().is_absolute():
        raise ValueError("Candidate requires an explicit absolute ACCERL_EXPERIMENT_ROOT")
    root = Path(value).expanduser().resolve()
    for protected in PROTECTED_ROOTS:
        protected = protected.resolve()
        if root == protected or protected in root.parents or root in protected.parents:
            raise ValueError("Candidate experiment root overlaps protected live/archive/source paths")
    return root


def candidate_run_id(value):
    if not isinstance(value, str) or re.fullmatch(RUN_ID_PREFIX + r"[A-Za-z0-9_-]+", value) is None:
        raise ValueError("Candidate run ID must use the fresh weaverintegrated_ namespace")
    return value


def candidate_data(data, *, root=None):
    root = experiment_root(root)
    data = Path(data).resolve()
    if data != root / "formal_v3":
        raise ValueError("Candidate data must be ACCERL_EXPERIMENT_ROOT/formal_v3")
    for relative in ("training", "runnerstatus.json", "reportstatus.json", "README.md",
                     "runner.log", "world_backend_config.json", "pipeline_config.json", "training/collection.sqlite",
                     "training/checkpoints", "training/checkpoints_initial", "training/initial_state_queue.json",
                     "training/stop.flag", "training/ray_spill"):
        target = data / relative
        if target.resolve() != target:
            raise ValueError("Candidate output paths must not redirect through symbolic links")
    return data


def validate_candidate_state(data, state, *, root=None, run_id=None):
    root = experiment_root(root)
    candidate_data(data, root=root)
    candidate_run_id(state.get("run_id"))
    if run_id is not None and state["run_id"] != candidate_run_id(run_id):
        raise ValueError("Candidate run ID differs from the requested run")
    if (state.get("candidate") != "weaver_integrated_v1"
            or state.get("experiment_root") != str(root)
            or state.get("source_path") != str(SOURCE)):
        raise ValueError("Candidate source/root identity differs; refusing existing run state")
    return state


def launch_paths(environ=None):
    """Validate launch outputs before a shell creates files or probes GPUs."""
    env = os.environ if environ is None else environ
    root = experiment_root(environ=env)
    candidate_data(root / "formal_v3", root=root)
    if env.get("ACCERL_RUN_ID"):
        candidate_run_id(env["ACCERL_RUN_ID"])
    data = Path(env.get("ACCERL_DATA_DIR") or root / "formal_v3" / "training").resolve()
    if data != root / "formal_v3" / "training":
        raise ValueError("Candidate training outputs must use its isolated formal_v3/training")
    if env.get("ACCERL_RAY_TMP"):
        temporary = Path(env["ACCERL_RAY_TMP"])
        if (not temporary.is_absolute() or temporary.parent != Path("/dev/shm")
                or not temporary.name.startswith("weaverintegrated_") or temporary.exists()):
            raise ValueError("Candidate Ray temporary directory must be a fresh /dev/shm/weaverintegrated_ path")
    return root, data
