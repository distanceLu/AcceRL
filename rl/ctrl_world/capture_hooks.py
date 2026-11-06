"""Instrumentation around the existing actors; learning methods stay unchanged.

The driver must call install_hooks(globals()) before creating Ray actors. Infer
loops call batch_begin/batch_end around the actual work. Client profile_rpc is
synchronous because the existing rollout workers use ray.get.
"""
from __future__ import annotations

import collections
import functools
import hashlib
import inspect
import json
import os
import random
import subprocess
import time

from capture_telemetry import Telemetry, new_id, now_ns


MODEL_KEY = {"policy_infer": "policy", "reward_infer": "reward", "ctrl_infer": "ctrl"}
ROLES = {
    "InferenceActor": "policy_infer",
    "RewardInferenceActor": "reward_infer",
    "CtrlWorldInferenceActor": "ctrl_infer",
    "TrainerActor": "trainer",
    "RolloutWorkerActor": "rollout",
    "EvaluationWorkerActor": "evaluation",
    "ReplayBufferActor": "policy_replay",
    "WMReplayBufferActor": "wm_replay",
    "StatsActor": "stats",
}


def _writer(owner):
    writer = getattr(owner, "telemetry", None)
    if writer is None:
        writer = Telemetry(getattr(owner, "capture_role", type(owner).__name__),
                           str(getattr(owner, "actor_id", getattr(owner, "rank", getattr(owner, "wid", 0)))))
        owner.telemetry = writer
    return writer


def _role(owner):
    return getattr(owner, "capture_role", "unknown")


def _version(owner):
    return int(getattr(owner, "capture_model_version", -1))


def _seed(role, actor_id):
    base = int(os.environ.get("ACCERL_SEED", "20261010"))
    offset = int.from_bytes(hashlib.sha256(role.encode()).digest()[:4], "big") % 1000000
    try:
        actor_offset = int(actor_id)
    except (ValueError, TypeError):
        actor_offset = int.from_bytes(hashlib.sha256(str(actor_id).encode()).digest()[:4], "big")
    seed = (base + offset + actor_offset) % (2 ** 32)
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    import torch
    if os.environ.get("ACCERL_CPU_THREADS"):
        torch.set_num_threads(int(os.environ["ACCERL_CPU_THREADS"]))
    torch.manual_seed(seed)
    return seed


def _actor_identity(original, owner, args, kwargs):
    try:
        bound = inspect.signature(original).bind(owner, *args, **kwargs)
        for key in ("actor_id", "rank", "wid"):
            if key in bound.arguments:
                return str(bound.arguments[key])
    except (TypeError, ValueError):
        pass
    return str(getattr(owner, "actor_id", getattr(owner, "rank", getattr(owner, "wid", 0))))


def _placement(owner):
    details = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
               "role": _role(owner), "seed": owner.capture_seed}
    import torch
    details["cpu_threads"] = torch.get_num_threads()
    try:
        import ray
        details["ray_gpu_ids"] = ray.get_gpu_ids()
    except Exception as error:
        details["ray_gpu_ids_error"] = repr(error)
    if _role(owner) in MODEL_KEY or _role(owner) in (
        "trainer", "policy_trainer", "ctrl_trainer", "reward_trainer"
    ):
        if hasattr(owner, "training_role"):
            details["training_role"] = owner.training_role
            details["model_role"] = owner.training_role
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            details.update(cuda_local_device=torch.cuda.current_device(), name=props.name,
                           total_bytes=props.total_memory)
            if getattr(props, "uuid", None) is not None:
                details["physical_gpu_uuid"] = str(props.uuid)
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=3, check=True,
                )
                devices = [token.strip() for token in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if token.strip()]
                mapping = []
                for line in result.stdout.splitlines():
                    index, uuid = [part.strip() for part in line.split(",", 1)]
                    if index in devices or any(uuid.startswith(token) for token in devices):
                        mapping.append({"physical_index": int(index), "uuid": uuid})
                details["physical_gpus"] = mapping
            except Exception as error:
                details["physical_lookup_error"] = repr(error)
    _writer(owner).event("placement", model_version=_version(owner), details=details)


def request_begin(owner, metadata=None):
    """Called once by the request wrapper, immediately before original enqueue."""
    metadata = dict(metadata or {})
    metadata.setdefault("request_id", new_id("request"))
    metadata.setdefault("submitted_ns", now_ns())
    metadata["arrival_ns"] = now_ns()
    if not hasattr(owner, "capture_pending_meta"):
        owner.capture_pending_meta = collections.deque()
    if not hasattr(owner, "capture_served_versions"):
        owner.capture_served_versions = {}
    owner.capture_pending_meta.append(metadata)
    _writer(owner).event("request_enqueue", workload=_role(owner), request_id=metadata["request_id"],
                         submitted_ns=metadata["submitted_ns"], start_ns=metadata["arrival_ns"],
                         details={key: value for key, value in metadata.items() if key not in ("request_id", "submitted_ns", "arrival_ns")})
    return metadata


def batch_begin(owner, requests, promises=None):
    """Freeze served versions and record actual, rather than configured, batch."""
    count = len(requests)
    pending = getattr(owner, "capture_pending_meta", None)
    metadata = []
    for _ in range(count):
        if pending:
            metadata.append(pending.popleft())
        else:
            timestamp = now_ns()
            metadata.append({"request_id": new_id("untracked_request"),
                             "submitted_ns": timestamp, "arrival_ns": timestamp,
                             "missing_client_metadata": True})
    version = _version(owner)
    key = MODEL_KEY.get(_role(owner), _role(owner))
    served = getattr(owner, "capture_served_versions", None)
    if served is None:
        served = owner.capture_served_versions = {}
    for item in metadata:
        served[item["request_id"]] = {key: version}
    starts = [item["arrival_ns"] for item in metadata]
    submitted = [item["submitted_ns"] for item in metadata]
    wall = now_ns()
    delays = sorted(max(0, wall - timestamp) for timestamp in starts)
    _writer(owner).event("batch_dispatch", workload=_role(owner), actual_batch=count,
                         model_version=version, start_ns=wall,
                         queue_mean_ns=sum(delays) / max(len(delays), 1),
                         queue_p95_ns=delays[int((len(delays) - 1) * .95)] if delays else None,
                         queue_max_ns=max(delays) if delays else None,
                         details={"request_ids": [item["request_id"] for item in metadata],
                                  "queue_scope": "actor_arrival_to_batch_dispatch",
                                  "configured_batch": getattr(owner, "batch_size", None)})
    span = _writer(owner).span(_role(owner), actual_batch=count, model_version=version,
                               request_ids=[item["request_id"] for item in metadata],
                               submitted_ns=submitted, cuda=True,
                               timing_scope="prepare_and_gpu_work_and_output_copy",
                               gpu_interval_includes_cpu_enqueue_gaps=True,
                               queue_scope="client_submission_to_service_start")
    span.__enter__()
    return span


def batch_end(owner, span, error=None):
    if span is None or getattr(span, "capture_closed", False):
        return
    span.capture_closed = True
    span.__exit__(type(error) if error is not None else None, error,
                  error.__traceback__ if error is not None else None)


def stage_begin(owner, stage, actual_batch=1, cuda=False):
    span = _writer(owner).span(_role(owner) + "_" + stage, actual_batch=actual_batch,
                               model_version=_version(owner), cuda=cuda, stage=stage)
    span.__enter__()
    return span


def stage_end(owner, span, error=None):
    batch_end(owner, span, error)


def profile_rpc(owner, workload, actor, method, *args, **kwargs):
    """Synchronous request/reply timing, plus the actual served model version."""
    import ray
    context = kwargs.pop("_capture_context", None) or {}
    submitted = now_ns()
    perf = time.perf_counter_ns()
    request_id = new_id("request")
    metadata = {"request_id": request_id, "submitted_ns": submitted,
                "caller_role": _role(owner), "caller_id": str(getattr(owner, "wid", getattr(owner, "rank", 0))),
                "episode_id": getattr(owner, "capture_episode_id", getattr(owner, "current_episode_id", None)),
                "method": method, **context}
    writer = _writer(owner)
    writer.event("request_submit", workload=workload, request_id=request_id,
                 submitted_ns=submitted, details=metadata)
    if method == "request":
        kwargs["telemetry_meta"] = metadata
    try:
        result = ray.get(getattr(actor, method).remote(*args, **kwargs))
        versions = {}
        if isinstance(result, dict) and "_capture_payload" in result:
            versions = dict(result.get("versions", {}))
            payload = result["_capture_payload"]
        else:
            payload = result
        existing = getattr(owner, "capture_last_versions", None)
        if existing is None:
            existing = owner.capture_last_versions = {}
        existing.update(versions)
        owner.capture_last_request_id = request_id
        model_key = MODEL_KEY.get(workload)
        if model_key is not None and method == "request":
            existing[model_key + "_request_id"] = request_id
        end = now_ns()
        writer.event("client_wait", workload=workload, request_id=request_id,
                     submitted_ns=submitted, start_ns=submitted, end_ns=end,
                     duration_ns=time.perf_counter_ns() - perf, model_version=json.dumps(versions, sort_keys=True),
                     status="ok", details={"method": method, "served_versions": versions,
                                           "scope": "Ray_submission_to_deserialized_reply"})
        return payload
    except BaseException as error:
        end = now_ns()
        writer.event("client_wait", workload=workload, request_id=request_id,
                     submitted_ns=submitted, start_ns=submitted, end_ns=end,
                     duration_ns=time.perf_counter_ns() - perf, status="error",
                     details={"method": method, "error": repr(error)})
        raise


def _request_wrapper(original):
    # Deliberately do not copy the old signature: Ray must recognize the added
    # telemetry_meta keyword when it builds its actor method signatures.
    async def request(self, *args, telemetry_meta=None, **kwargs):
        metadata = request_begin(self, telemetry_meta)
        request_id = metadata["request_id"]
        try:
            payload = await original(self, *args, **kwargs)
            versions = self.capture_served_versions.pop(request_id, {MODEL_KEY[_role(self)]: _version(self)})
            return {"_capture_payload": payload, "versions": versions, "request_id": request_id}
        except BaseException:
            self.capture_served_versions.pop(request_id, None)
            try:
                self.capture_pending_meta.remove(metadata)
            except ValueError:
                pass
            raise
    request.__name__ = original.__name__
    request.__doc__ = original.__doc__
    return request


def _receive_wrapper(original):
    @functools.wraps(original)
    def receive(self, *args, **kwargs):
        before = _version(self)
        with _writer(self).span("broadcast_receive", model_version=before,
                                cuda=True, role=_role(self), group_name=args[0] if args else kwargs.get("group_name")):
            result = original(self, *args, **kwargs)
        self.capture_model_version = before + 1
        self.model_version = self.capture_model_version
        _writer(self).event("model_version", workload=_role(self), model_version=self.capture_model_version,
                            details={"previous_version": before, "meaning": "completed_broadcast_receive_count_initial_zero"})
        return result
    return receive


def _broadcast_inventory(owner, group):
    if str(group).startswith("broadcast_ctrl"):
        engine = owner.ctrl_world_engine
        include_buffers = False
    elif str(group).startswith("broadcast_reward"):
        engine = owner.reward_engine
        include_buffers = True
    else:
        engine = owner.model
        include_buffers = True
    module = getattr(engine, "module", engine)
    tensors = [parameter for parameter in module.parameters() if parameter.requires_grad]
    if include_buffers:
        tensors.extend(module.buffers())
    key = ("ctrl" if str(group).startswith("broadcast_ctrl") else
           "reward" if str(group).startswith("broadcast_reward") else "policy")
    version = (getattr(owner, "ctrl_step", 0) if str(group).startswith("broadcast_ctrl") else
               getattr(owner, "reward_step", 0) if str(group).startswith("broadcast_reward") else
               getattr(owner, "global_step", 0))
    optimizer_version = getattr(owner, "capture_versions", {}).get(key)
    return {"tensor_count": len(tensors), "payload_bytes": sum(t.numel() * t.element_size() for t in tensors),
            "source_step": int(version), "group_name": group,
            "optimizer_version": optimizer_version, "model_key": key,
            "meaning": "trainable_parameters_and_required_buffers"}


def _broadcast_wrapper(original):
    @functools.wraps(original)
    def broadcast(self, *args, **kwargs):
        group = args[0] if args else kwargs.get("group_name", "")
        inventory = _broadcast_inventory(self, group)
        with _writer(self).span("broadcast_send", model_version=inventory["source_step"], cuda=True, **inventory):
            return original(self, *args, **kwargs)
    return broadcast


def _cpu_method(original, workload):
    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        with _writer(self).span(workload, cuda=False):
            return original(self, *args, **kwargs)
    return wrapper


def _patch_envs(owner):
    for environment in getattr(owner, "envs", []):
        for method_name in ("step", "reset"):
            original = getattr(environment, method_name)
            def measured(*args, _original=original, _method=method_name, **kwargs):
                with _writer(owner).span("cpu_env_" + _method, cuda=False):
                    return _original(*args, **kwargs)
            setattr(environment, method_name, measured)


def _init_wrapper(original, role):
    @functools.wraps(original)
    def init(self, *args, **kwargs):
        actor_id = _actor_identity(original, self, args, kwargs)
        training_role = "combined"
        if role == "trainer":
            # TrainerActor remains a single Ray class, while split instances
            # need distinct telemetry identities on the same rank/card.
            bound = inspect.signature(original).bind(self, *args, **kwargs)
            training_role = bound.arguments.get("training_role", "combined")
        effective_role = (training_role + "_trainer" if role == "trainer"
                          and training_role in ("policy", "ctrl", "reward") else role)
        self.capture_role = effective_role
        # Preserve the baseline constructor's seed derivation. A telemetry role
        # rename must not independently change model initialization seeds.
        self.capture_seed = _seed(role, actor_id)
        self.capture_model_version = -1
        self.model_version = -1
        self.capture_pending_meta = collections.deque()
        self.capture_served_versions = {}
        self.capture_last_versions = {}
        start = now_ns()
        perf = time.perf_counter_ns()
        original(self, *args, **kwargs)
        if getattr(self, "telemetry", None) is None:
            self.telemetry = Telemetry(effective_role, actor_id)
        details = {"role": effective_role, "seed": self.capture_seed}
        if role == "trainer":
            details.update(training_role=training_role, seed_role=role)
        self.telemetry.event("actor_init", start_ns=start, end_ns=now_ns(),
                             duration_ns=time.perf_counter_ns() - perf,
                             details=details)
        _placement(self)
        if role in ("rollout", "evaluation"):
            _patch_envs(self)
    return init


def _eval_wrapper(original):
    @functools.wraps(original)
    def evaluate(self, *args, **kwargs):
        count = len(args[0]) if args and hasattr(args[0], "__len__") else 1
        with _writer(self).span("ctrl_eval", actual_batch=count, model_version=_version(self), cuda=True):
            return original(self, *args, **kwargs)
    return evaluate


def install_hooks(namespace):
    """Patch Ray's modified classes and refresh cached method signatures."""
    patched = []
    for name, role in ROLES.items():
        actor_class = namespace.get(name)
        if actor_class is None:
            continue
        metadata = getattr(actor_class, "__ray_metadata__", None)
        cls = getattr(metadata, "modified_class", None)
        if cls is None or cls.__dict__.get("_capture_hooks_installed", False):
            continue
        cls.__init__ = _init_wrapper(cls.__init__, role)
        if role in MODEL_KEY:
            cls.request = _request_wrapper(cls.request)
            cls.receive_and_update_weights = _receive_wrapper(cls.receive_and_update_weights)
        if role == "trainer":
            cls.broadcast_weights = _broadcast_wrapper(cls.broadcast_weights)
        if role == "ctrl_infer" and hasattr(cls, "evaluate_prediction_metrics"):
            cls.evaluate_prediction_metrics = _eval_wrapper(cls.evaluate_prediction_metrics)
        if role == "rollout":
            for method, workload in (("_obs_to_views", "cpu_obs_views"), ("obs2inp", "cpu_obs_prepare"),
                                     ("_process_episode_for_wm", "cpu_wm_windows")):
                if hasattr(cls, method):
                    setattr(cls, method, _cpu_method(getattr(cls, method), workload))
        cls._capture_hooks_installed = True
        # Ray has already inspected methods at @ray.remote decoration time.
        # Refreshing this map is necessary for the added telemetry_meta keyword.
        try:
            try:
                from ray._common.signature import extract_signature
            except ImportError:
                from ray._private.signature import extract_signature
            method_meta = metadata.method_meta
            for method in ("request",):
                if hasattr(cls, method) and hasattr(method_meta, "signatures"):
                    method_meta.signatures[method] = extract_signature(getattr(cls, method), ignore_first=True)
                    if hasattr(method_meta, "methods"):
                        method_meta.methods[method] = getattr(cls, method)
        except (AttributeError, ImportError, TypeError):
            pass
        patched.append(name)
    return patched
