"""Bounded, atomic snapshots of enabled replicated ZeRO-2 model weights.

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

from training_topology import active_model_kinds, _publication_state


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


def _weights(module, *, full=False, world_backend=None, version=0):
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
    publication = None
    sources = state
    if world_backend is not None:
        declared, publication = _publication_state(module, "ctrl", version, world_backend, for_export=True)
        selected.update(name for name, _ in declared if name in state)
        sources = dict(state)
        sources.update((name, tensor) for name, tensor in declared if name in state)
    tensors = {name: sources[name].detach().cpu().clone() for name in sorted(selected)}
    return {"state_dict": tensors,
            **({"publication": publication} if publication is not None else {}),
            **({"publication_restore_names": sorted(name for name, _ in declared if name in state)}
               if publication is not None else {}),
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


def save_parallel_weights(trainer, checkpoint_dir, *, reason="periodic", metadata=None, torch_module=None,
                          active_kinds=None):
    """Save one overwriteable snapshot; failures preserve the previous weights.

    The embedded manifest is authoritative if interruption occurs between
    publishing latest_weights.pt and its human-readable manifest.json sidecar.
    """
    if int(trainer.rank) != 0:
        raise RuntimeError("Only trainer rank 0 publishes replicated model weights")
    kinds = active_model_kinds(active_kinds)
    models, steps = {}, {}
    engine_names = {"policy": "model", "ctrl": "ctrl_world_engine", "reward": "reward_engine"}
    step_names = {"policy": "global_step", "ctrl": "ctrl_step", "reward": "reward_step"}
    for kind in kinds:
        steps[kind] = int(getattr(trainer, step_names[kind]))
        backend = getattr(trainer, "world_backend", None) if kind == "ctrl" else None
        models[kind] = _weights(_module(getattr(trainer, engine_names[kind])), full=kind == "reward",
                                world_backend=backend, version=steps[kind])
    return _save_weights(models, int(trainer.world_size),
                         steps,
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
    backend = getattr(trainer, "world_backend", None) if kind == "ctrl" else None
    engine = getattr(trainer, engine_names[kind])
    return {"kind": kind, "rank": 0, "world_size": world_size,
            "policy_step": policy_step, "model_step": model_step,
            "model": (engine.snapshot_weights(model_step)
                      if backend is None and hasattr(engine, "snapshot_weights") else
                      _weights(_module(engine), full=kind == "reward", world_backend=backend,
                               version=model_step))}


def save_split_weights(components, checkpoint_dir, *, reason="periodic", metadata=None, torch_module=None,
                       active_kinds=None):
    """Publish one snapshot containing exactly the explicitly enabled owners.

    The caller waits for all enabled four-rank groups to finish the epoch before
    exporting components. Reject mixed Policy boundaries before touching the
    existing checkpoint, even when a component has otherwise valid tensors.
    """
    config = (metadata or {}).get('world_backend_config', {})
    model_free = isinstance(config, dict) and config.get('backend') == 'model_free'
    if tuple(active_kinds or ()) == ('policy',) and model_free:
        from world_backends import build_service_plan
        if (config.get('world_trainable') is not False or config.get('reward_provider') != 'environment'
                or 'online_world' in metadata or 'frozen_world' in metadata
                or metadata.get('service_plan') != build_service_plan(config, world_size=4).to_dict()):
            raise ValueError('Model-free Policy snapshot requires its exact environment-only config and service plan')
        kinds = ('policy',)
    elif tuple(active_kinds or ()) == ('policy',):
        # This is only the controller's monolithic file selection. Native
        # World/RM owners remain active and save their separate full artifact.
        native = (metadata or {}).get('online_world', {})
        plan = (metadata or {}).get('service_plan', {})
        if (native.get('protocol') != 'weaver_online_joint_weights_v1'
                or native.get('run_id') != (metadata or {}).get('run_id')
                or native.get('world_size') != 4 or native.get('weights_only') is not True
                or native.get('tensor_count') != 866 or native.get('reason') != reason
                or plan.get('trainer_kinds') != ['policy', 'ctrl']
                or plan.get('inference_kinds') != ['policy', 'ctrl']
                or plan.get('reward_provider') != 'world_head'
                or plan.get('expected_gpu_model_processes') != 16):
            raise ValueError('Policy-only monolith requires the separate integrated native World/RM artifact and plan')
        kinds = ('policy',)
    else:
        kinds = active_model_kinds(active_kinds)
    if not isinstance(components, dict) or set(components) != set(kinds):
        raise ValueError("Split snapshot requires exactly the enabled model components: %s" % (kinds,))
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
    if model_free and (kinds != ('policy',) or world_sizes != {4}):
        raise ValueError('Model-free snapshots require exactly four Policy ranks and no World owners')
    validation = (metadata or {}).get('fixed_validation') or {}
    if model_free and 'model_free_success' in metadata:
        from model_free_success import validate_model_free_success
        snapshot = validate_model_free_success(metadata['model_free_success'])
        if reason in ('final', 'final_before_first_update'):
            from model_free_completion import LEGACY_VALIDATION_REASONS, ROLLOUT_TARGET_REASON
            if _counter(metadata.get('global_step'), 'Final Policy step') != components['policy']['policy_step']:
                raise ValueError('Final rollout statistics must accompany the synchronized Policy')
            stop_reason = metadata.get('stop_reason')
            if (not isinstance(stop_reason, str) or not stop_reason or stop_reason in LEGACY_VALIDATION_REASONS
                    or (stop_reason == ROLLOUT_TARGET_REASON and '99' not in snapshot['crossings'])):
                raise ValueError('Final stop reason disagrees with rollout statistics')
    elif model_free and validation.get('target_reached') is True:
        target = validation.get('target_validation') or {}
        if (target != validation.get('last') or target.get('cases') != 50
                or target.get('successes') != 50
                or target.get('policy_step') != components['policy']['policy_step']):
            raise ValueError('Full-success checkpoint must preserve the exact validated Policy')
    if kinds == ('policy',) and not model_free and (native.get('policy_step') != components['policy']['policy_step']
            or native.get('successful_updates') != components['policy']['policy_step'] // 5
            or native.get('model_version') != native.get('successful_updates')):
        raise ValueError('Native World/RM artifact and Policy-only monolith have different synchronized boundaries')
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


def apply_parallel_weights(trainer, snapshot, *, torch_module=None, active_kinds=None):
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
    kinds = active_model_kinds(active_kinds)
    engine_names = {"policy": "model", "ctrl": "ctrl_world_engine", "reward": "reward_engine"}
    models = tuple((kind, getattr(trainer, engine_names[kind])) for kind in kinds)
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
    if set(payload.get("models", {})) != set(kinds):
        raise ValueError("Snapshot model roles differ from enabled owners")
    # Validate backend identity/selected weight family before mutating any model.
    for name, model in models:
        saved = payload["models"][name]
        backend = getattr(trainer, "world_backend", None) if name == "ctrl" else None
        if backend is not None or "publication" in saved:
            selected, publication = _publication_state(_module(model), name,
                payload["manifest"]["steps"][name], backend)
            if saved.get("publication") != publication:
                raise ValueError("Snapshot publication differs from destination backend bundle")
            restore_names = sorted(key for key, _ in selected if key in _module(model).state_dict())
            if (saved.get("publication_restore_names") != restore_names
                    or not set(restore_names) <= saved["state_dict"].keys()):
                raise ValueError("Snapshot is missing declared backend restore tensors")
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
