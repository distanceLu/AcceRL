"""Bounded, atomic snapshots of all three replicated ZeRO-2 model weights.

Export only from each model's trainer rank 0, between completed synchronized
training epochs. Combined trainers retain the original single-owner API.
This deliberately does not save optimizer partitions, Replay, or random state.
Restore the original model recipe/checkpoints into bare models, apply these
tensors, and only then initialize DeepSpeed engines and their optimizers.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time


FORMAT = "accerl_parallel_weights_v1"


def stop_requested() -> bool:
    flag = os.environ.get("ACCERL_STOP_FILE")
    deadline = os.environ.get("ACCERL_SOFT_STOP_AT")
    return bool((flag and Path(flag).is_file()) or (deadline and time.time() >= float(deadline)))


def _module(engine):
    stage = getattr(engine, "zero_optimization_stage", None)
    if stage is not None and int(stage() if callable(stage) else stage) > 2:
        raise RuntimeError("Rank-0-only checkpoint requires replicated ZeRO stage 0-2 parameters")
    return getattr(engine, "module", engine)


def _weights(module, *, full=False):
    parameters = dict(module.named_parameters())
    selected = {name for name, value in parameters.items() if full or value.requires_grad}
    state = module.state_dict()
    missing = selected - state.keys()
    if missing:
        raise RuntimeError(f"Checkpoint tensors are absent from state_dict: {sorted(missing)}")
    selected.update(name for name, _ in module.named_buffers() if name in state)
    # Non-persistent buffers are not required for restoration. Model state_dict
    # contains all persistent buffers and includes parameter aliases as needed.
    if full:
        selected = set(state)
    else:
        trainable_ids = {id(value) for value in parameters.values() if value.requires_grad}
        try:
            aliases = module.named_parameters(remove_duplicate=False)
        except TypeError:
            aliases = module.named_parameters()
        selected.update(name for name, value in aliases if id(value) in trainable_ids and name in state)
    tensors = {name: state[name].detach().cpu().clone() for name in sorted(selected)}
    return {"state_dict": tensors,
            "trainable_names": sorted(name for name, value in parameters.items() if value.requires_grad),
            "full_state": bool(full)}


def _atomic_json(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_parallel_weights(trainer, checkpoint_dir, *, reason="periodic", metadata=None, torch_module=None):
    """Save one overwriteable snapshot; failures preserve the previous weights.

    The embedded manifest is authoritative if interruption occurs between
    publishing latest_weights.pt and its human-readable manifest.json sidecar.
    """
    if int(trainer.rank) != 0:
        raise RuntimeError("Only trainer rank 0 publishes replicated model weights")
    models = {}
    for name, engine, full in (("policy", trainer.model, False),
                               ("ctrl", trainer.ctrl_world_engine, False),
                               ("reward", trainer.reward_engine, True)):
        models[name] = _weights(_module(engine), full=full)
    return _save_weights(models, int(trainer.world_size),
                         {"policy": int(trainer.global_step), "ctrl": int(trainer.ctrl_step),
                          "reward": int(trainer.reward_step)},
                         checkpoint_dir, reason=reason, metadata=metadata, torch_module=torch_module)


def _counter(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < int(positive):
        raise ValueError(f"{label} must be a {'positive' if positive else 'nonnegative'} integer")
    return value


def export_split_component(trainer, kind):
    """Copy the owning rank-0 engine at a completed shared Policy boundary."""
    engine_names = {"policy": "model", "ctrl": "ctrl_world_engine", "reward": "reward_engine"}
    if kind not in engine_names or getattr(trainer, "training_role", None) != kind:
        raise ValueError("Split checkpoint component must be exported by its owning training role")
    if _counter(trainer.rank, "Training rank") != 0:
        raise RuntimeError("Only trainer rank 0 exports replicated model weights")
    world_size = _counter(trainer.world_size, "Training world size", positive=True)
    policy_step = _counter(trainer.global_step, "Policy step")
    step_name = {"policy": "global_step", "ctrl": "ctrl_step", "reward": "reward_step"}[kind]
    model_step = _counter(getattr(trainer, step_name), f"{kind} model step")
    return {"kind": kind, "rank": 0, "world_size": world_size,
            "policy_step": policy_step, "model_step": model_step,
            "model": (getattr(trainer, engine_names[kind]).snapshot_weights(model_step)
                      if hasattr(getattr(trainer, engine_names[kind]), "snapshot_weights") else
                      _weights(_module(getattr(trainer, engine_names[kind])), full=kind == "reward"))}


def save_split_weights(components, checkpoint_dir, *, reason="periodic", metadata=None, torch_module=None):
    """Publish the original three-model format from three distinct owners.

    The caller waits for all three four-rank groups to finish the epoch before
    exporting components. Reject mixed Policy boundaries before touching the
    existing checkpoint, even when a component has otherwise valid tensors.
    """
    kinds = ("policy", "ctrl", "reward")
    if not isinstance(components, dict) or set(components) != set(kinds):
        raise ValueError("Split snapshot requires exactly policy, ctrl and reward components")
    for kind in kinds:
        component = components[kind]
        if not isinstance(component, dict) or component.get("kind") != kind:
            raise ValueError(f"{kind}: split checkpoint owner identity differs")
        if _counter(component.get("rank"), "Component rank") != 0:
            raise ValueError("Only rank-0 components can publish replicated model weights")
        _counter(component.get("world_size"), "Component world size", positive=True)
        _counter(component.get("policy_step"), "Component Policy step")
        _counter(component.get("model_step"), "Component model step")
        model = component.get("model")
        if (not isinstance(model, dict) or not isinstance(model.get("state_dict"), dict)
                or not model["state_dict"] or not isinstance(model.get("trainable_names"), list)
                or model.get("full_state") is not (kind == "reward")):
            raise ValueError(f"{kind}: split checkpoint model state differs from the original format")
    world_sizes = {component["world_size"] for component in components.values()}
    policy_steps = {component["policy_step"] for component in components.values()}
    if len(world_sizes) != 1:
        raise ValueError("Split checkpoint components have different world sizes")
    if len(policy_steps) != 1 or components["policy"]["model_step"] != components["policy"]["policy_step"]:
        raise ValueError("Split checkpoint components have different Policy boundaries")
    return _save_weights({kind: components[kind]["model"] for kind in kinds}, world_sizes.pop(),
                         {kind: components[kind]["model_step"] for kind in kinds}, checkpoint_dir,
                         reason=reason, metadata=metadata, torch_module=torch_module)


def _save_weights(models, world_size, steps, checkpoint_dir, *, reason, metadata, torch_module):
    if torch_module is None:
        import torch as torch_module
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"format": FORMAT,
               "manifest": {"format": FORMAT, "saved_at_unix": time.time(), "reason": str(reason),
                            "rank": 0, "world_size": world_size, "steps": steps,
                            "weights_only": True, "exact_resume_supported": False,
                            "missing_resume_state": ["optimizer", "Replay", "RNG", "prefetched_batches"],
                            "restore_method": "Initialize original models, then apply saved state_dict tensors",
                            "metadata": metadata or {}},
               "models": models}
    payload["manifest"]["tensor_counts"] = {
        name: len(model["state_dict"]) for name, model in payload["models"].items()}
    target = directory / "latest_weights.pt"
    temporary = directory / "latest_weights.pt.tmp"
    try:
        with temporary.open("wb") as handle:
            torch_module.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        result = dict(payload["manifest"], weights_file=target.name, weights_bytes=target.stat().st_size)
        _atomic_json(directory / "manifest.json", result)
        if os.name != "nt":
            descriptor = os.open(str(directory), os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return result
    finally:
        temporary.unlink(missing_ok=True)


def apply_parallel_weights(trainer, snapshot, *, torch_module=None):
    """Apply a weights-only snapshot before initializing DeepSpeed engines.

    The three model fields must contain bare models initialized from the
    original recipe and frozen checkpoints. Apply this snapshot to each rank's
    bare models, then initialize DeepSpeed so FP32 optimizer master parameters
    are built from the imported weights. Initialized DeepSpeed engines are rejected
    before any tensor is changed: module.load_state_dict alone would leave
    their optimizer master copies stale.

    Do not restore training counters here: without optimizer/Replay/RNG this is
    a warm start, not continuation of the original optimizer trajectory.
    """
    models = (("policy", trainer.model), ("ctrl", trainer.ctrl_world_engine),
              ("reward", trainer.reward_engine))
    # Preflight every model before loading or applying any state_dict. Otherwise
    # a later initialized engine could be rejected after an earlier model was
    # already overwritten.
    for name, model in models:
        stage = getattr(model, "zero_optimization_stage", None)
        if hasattr(model, "zero_optimization_stage"):
            stage = stage() if callable(stage) else stage
            raise RuntimeError(
                f"{name}: apply weights to bare models before DeepSpeed initialization; "
                f"initialized DeepSpeed engine (ZeRO stage {stage}) may have separate optimizer master weights")
    if torch_module is None:
        import torch as torch_module
    payload = torch_module.load(snapshot, map_location="cpu", weights_only=False)
    if payload.get("format") != FORMAT:
        raise ValueError("Unsupported parallel weights snapshot")
    for name, model in models:
        module = _module(model)
        saved = payload["models"][name]
        trainable_names = {key for key, value in module.named_parameters() if value.requires_grad}
        if trainable_names != set(saved["trainable_names"]):
            raise ValueError(f"{name}: trainable parameter names differ from saved recipe")
        result = module.load_state_dict(saved["state_dict"], strict=bool(saved["full_state"]))
        if result.unexpected_keys or trainable_names.intersection(result.missing_keys):
            raise ValueError(f"{name}: snapshot does not cover trainable model state")
    return payload["manifest"]
