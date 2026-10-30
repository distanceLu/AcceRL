"""CPU-safe configuration and actor operations; Ray and model imports stay lazy."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from .contracts import BackendCapabilities, WorldBackendConfig
from .service_plan import build_service_plan


def _argument(args, name, default=None):
    return args.get(name, default) if isinstance(args, Mapping) else getattr(args, name, default)


def load_backend_config(args=None, environ=None):
    """Read optional JSON, then explicit CLI selections, then environment defaults.

    CLI selection defaults should be ``None`` so a JSON backend choice survives.
    No model imports, subprocesses, downloads, or environment mutation occur.
    """
    env = os.environ if environ is None else environ
    source = _argument(args, "world_backend_config") or env.get("ACCERL_WORLD_BACKEND_CONFIG")
    if isinstance(source, WorldBackendConfig):
        values = source.to_dict()
    elif isinstance(source, Mapping):
        values = dict(source)
    elif source:
        with Path(source).expanduser().open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        if not isinstance(values, dict):
            raise ValueError("Backend configuration JSON must contain an object")
    else:
        values = {}
    for arg_name, field, env_name in (("world_backend", "backend", "ACCERL_WORLD_BACKEND"),
                                      ("reward_provider", "reward_provider", "ACCERL_REWARD_PROVIDER")):
        choice = _argument(args, arg_name)
        if choice is not None and choice != "":
            values[field] = choice
        elif env.get(env_name):
            values[field] = env[env_name]
    options = dict(values.get("backend_options", {}))
    for arg_name, field, env_name in (("weaver_source_path", "source_path", "ACCERL_WEAVER_SOURCE_PATH"),
                                      ("weaver_checkpoint_dir", "checkpoint_dir", "ACCERL_WEAVER_CHECKPOINT_DIR")):
        choice = _argument(args, arg_name) or env.get(env_name)
        if choice:
            options[field] = choice
    if options:
        values["backend_options"] = options
    # A separately saved audited manifest can be the single metadata source.
    # Missing manifests remain missing and fail admission; none are fabricated.
    if not values.get("checkpoint_metadata") and options.get("checkpoint_dir"):
        metadata_path = Path(options["checkpoint_dir"]) / options.get("metadata_file", "backend_metadata.json")
        if metadata_path.is_file():
            values["checkpoint_metadata"] = json.loads(metadata_path.read_text(encoding="utf-8"))
    return WorldBackendConfig.from_mapping(values)


def prepare_runtime(config=None, world_size=4):
    """Validate semantics before constructing the four-rank service owners."""
    if not isinstance(config, WorldBackendConfig):
        config = WorldBackendConfig.from_mapping(config)
    config.validate_admission()
    return build_service_plan(config, world_size).validate_runtime()


def service_plan_from_state(state, environ=None):
    """Recover recorded owner expectations without reading ambient environment.

    This is a telemetry/controller reader, not a replacement for startup
    admission. Saved derived counts must agree with the saved configuration.
    """
    if not isinstance(state, Mapping):
        raise ValueError("Controller state must be a mapping")
    recorded = state.get("service_plan")
    if recorded is not None and not isinstance(recorded, Mapping):
        raise ValueError("Recorded service_plan must be a mapping")
    if state.get("world_backend_config") is not None:
        config = WorldBackendConfig.from_mapping(state["world_backend_config"])
    elif recorded is not None:
        config = WorldBackendConfig.from_mapping({key: recorded[key] for key in ("backend", "reward_provider", "world_trainable") if key in recorded})
    else:
        config = load_backend_config(environ=environ) if environ is not None else WorldBackendConfig()
    plan = build_service_plan(config, world_size=(recorded or {}).get("world_size", 4)).validate_runtime()
    if recorded is not None:
        expected = plan.to_dict()
        for name, value in recorded.items():
            if name in expected:
                actual = list(value) if isinstance(value, tuple) else value
                if actual != expected[name]:
                    raise ValueError("Recorded service_plan.%s disagrees with the configured active owners" % name)
    return plan


def context_geometry(config):
    if not isinstance(config, WorldBackendConfig):
        config = WorldBackendConfig.from_mapping(config)
    if config.backend == "ctrl":
        history = int(config.backend_options.get("num_history", 6))
        horizon = int(config.backend_options.get("num_frames_pred", 5)) - 1
        memory, stride = 0, 1
        required = history + 1
    else:
        history = int(config.schema["n_history"])
        memory = int(config.schema["n_memory_frames"])
        stride = int(config.schema["t_memory"])
        horizon = int(config.backend_options.get("horizon", 8))
        required = history + memory * stride
    if min(history, horizon, stride) < 1 or memory < 0:
        raise ValueError("Backend context geometry must have positive history, horizon and stride")
    return {"required_context_frames": required, "training_window_frames": required + horizon,
            "horizon": horizon, "n_history": history, "n_memory_frames": memory, "t_memory": stride}


def _version(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Published bundle version must be a nonnegative integer")
    return value


def _require_cpu(value, name="payload"):
    device = getattr(value, "device", None)
    if device is not None and str(device) != "cpu":
        raise ValueError("%s must use CPU RPC tensors" % name)
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_cpu(item, "%s.%s" % (name, key))
    elif isinstance(value, (list, tuple)):
        for item in value:
            _require_cpu(item, name)


def _require_owned_state(value, device, name='resident_state'):
    """Only internal child-owned state may use its admitted GPU device."""
    actual=getattr(value,'device',None)
    expected=str(device)
    if expected=='cuda':
        expected='cuda:0'  # the worker admits exactly one visible device
    if actual is not None and str(actual) not in ('cpu',expected):
        raise ValueError('%s uses another device than its owned backend' % name)
    if actual is not None and str(actual)!='cpu' and not expected.startswith('cuda:'):
        raise ValueError('%s backend does not admit GPU state' % name)
    if isinstance(value,Mapping):
        for key,item in value.items():_require_owned_state(item,device,'%s.%s'%(name,key))
    elif isinstance(value,(list,tuple)):
        for item in value:_require_owned_state(item,device,name)


class CtrlBackendAdapter:
    """Delegate legacy CTRL behavior while making its state/version opaque.

    ``init_fn(context)`` and ``predict_fn(payload, actions, requested_outputs)``
    wrap the existing CTRL routines. The predictor returns a mapping containing
    ``observations`` and ``next_state``. History evolution stays in that existing
    implementation; this adapter does not reimplement diffusion or normalization.
    """
    name = "ctrl"

    def __init__(self, model=None, env=None, *, config=None, init_fn=None, predict_fn=None,
                 prepare_fn=None, loss_fn=None, published_version=0):
        self.config = config if isinstance(config, WorldBackendConfig) else WorldBackendConfig.from_mapping(config)
        if self.config.backend != "ctrl":
            raise ValueError("CtrlBackendAdapter requires backend='ctrl'")
        self.config.validate_admission()
        self.model, self.env = model, env
        self._init = init_fn or getattr(env, "init_rollout", None)
        self._predict = predict_fn or getattr(env, "predict_chunk", None)
        self._prepare, self._loss = prepare_fn, loss_fn
        self.published_version = _version(published_version)

    def capabilities(self):
        return BackendCapabilities("ctrl")

    def set_published_version(self, version):
        version = _version(version)
        if version < self.published_version:
            raise ValueError("Cannot roll CTRL publication version backward")
        self.published_version = version

    def prepare_train_batch(self, episode_window):
        if self._prepare is None:
            raise ValueError("CTRL preparation must delegate the existing data preparation routine")
        return self._prepare(episode_window)

    def compute_losses(self, prepared_batch):
        function = self._loss or self.model
        if function is None:
            raise ValueError("CTRL loss delegate/model is missing")
        return function(prepared_batch)

    def init_rollout(self, real_context, published_version=None):
        version = self.published_version if published_version is None else _version(published_version)
        if version != self.published_version:
            raise ValueError("Requested CTRL version is not the currently published bundle")
        if self._init is not None:
            payload = self._init(real_context)
        elif self.env is not None and hasattr(self.env, "init_latent_state"):
            payload = self.env.init_latent_state(real_context["observations"])
        else:
            raise ValueError("CTRL rollout initializer delegate is missing")
        return {"backend": "ctrl", "version": version, "payload": payload}

    def predict_chunk(self, opaque_state, action_chunk, requested_outputs=None):
        if not isinstance(opaque_state, Mapping) or opaque_state.get("backend") != "ctrl" or opaque_state.get("version") != self.published_version:
            raise ValueError("Stale CTRL rollout state; initialize real context after publication")
        if self._predict is None:
            raise ValueError("CTRL prediction must delegate the existing latent history/prediction routine")
        result = self._predict(opaque_state["payload"], action_chunk, requested_outputs)
        if not isinstance(result, Mapping) or "observations" not in result or "next_state" not in result:
            raise ValueError("CTRL predictor must return observations and next_state")
        return {**result, "served_version": self.published_version,
                "next_state": {"backend": "ctrl", "version": self.published_version, "payload": result["next_state"]}}

    def publication_inventory(self):
        # None deliberately retains the legacy parameter selector/buffer policy.
        return None


class WeaverActorOperations:
    """Shared synchronous operations used by the Ray actor's queued RPC shell."""
    def __init__(self, backend, config=None, published_version=0):
        self.backend = backend
        if config is None:
            raw = backend.config
            config = {name: raw[name] for name in WorldBackendConfig.__dataclass_fields__ if name in raw}
        self.config = config if isinstance(config, WorldBackendConfig) else WorldBackendConfig.from_mapping(config)
        if self.config.backend != "weaver":
            raise ValueError("WeaverActorOperations requires backend='weaver'")
        self.config.validate_admission()
        self.published_version = None
        self.set_published_version(published_version)

    def set_published_version(self, version):
        version = _version(version)
        if self.published_version is not None and version < self.published_version:
            raise ValueError("Cannot roll WEAVER publication version backward")
        # Backend invalidates model-dependent caches; frozen features remain
        # reusable only under their explicitly checked encoder identity.
        self.backend.set_published_version(version)
        self.published_version = version

    def get_context_geometry(self):
        geometry = context_geometry(self.config)
        for key in ("required_context_frames", "training_window_frames", "horizon"):
            if hasattr(self.backend, key):
                geometry[key] = int(getattr(self.backend, key))
        geometry.update({key: self.config.schema[key] for key in ("action_dim", "state_dim", "dt_seconds")})
        return geometry

    def init_rollout(self, real_context, published_version=None):
        return self._init_rollout(real_context,published_version,owned=False)

    def init_resident_rollout(self, real_context, published_version=None):
        return self._init_rollout(real_context,published_version,owned=True)

    def _init_rollout(self, real_context, published_version, *, owned):
        if not isinstance(real_context, Mapping):
            raise ValueError("WEAVER real_context must be an explicit CPU payload mapping")
        _require_cpu(real_context)
        version = self.published_version if published_version is None else _version(published_version)
        if version != self.published_version:
            raise ValueError("Requested WEAVER version is not the currently published bundle")
        state = self.backend.init_rollout(real_context, version)
        self._validate_state(state,owned=owned)
        if self.config.reward_provider == "world_head":
            native_score = self.config.reward_semantics == 'delta_native_score'
            initial = state.get("initial_reward_score" if native_score else "initial_success_probability")
            if initial is None or not math.isfinite(float(initial)) or not 0 <= float(initial) <= 1:
                raise ValueError("Integrated rollout needs its bounded real seed reward score" if native_score
                                 else "Integrated rollout needs the calibrated real seed success probability")
            if native_score and state.get('initial_terminal') is not False:
                raise ValueError('Uncalibrated native reward cannot terminate an imagined rollout')
        return state

    def _validate_state(self, state, *, owned=False):
        if (not isinstance(state, Mapping) or state.get("backend") != "weaver"
                or type(state.get("version")) is not int or state.get("version") != self.published_version):
            raise ValueError("Stale WEAVER rollout/cache version; initialize from real context after publication")
        for name in ("encoder_id", "normalizer_id"):
            if state.get(name) != self.config.schema[name]:
                raise ValueError("WEAVER rollout %s differs from the admitted bundle" % name)
        if owned:
            _require_owned_state(state,getattr(self.backend,'device',None))
        else:
            _require_cpu(state)

    def request(self, state, actions, instruction=None):
        return self._request(state,actions,instruction,owned=False)

    def request_wire(self, state, actions, instruction=None):
        return self._request(state,actions,instruction,owned=True)

    def _request(self, state, actions, instruction, *, owned):
        self._validate_state(state,owned=owned)
        _require_cpu(actions, "actions")
        if instruction is not None and instruction != state.get("instruction"):
            raise ValueError("Rollout task instruction changed; initialize a new real context")
        shape = getattr(actions, "shape", None)
        if shape is not None and (len(shape) != 2 or shape[-1] != self.config.schema["action_dim"]):
            raise ValueError("WEAVER RPC actions must be [time, action_dim]")
        if len(actions) < 1:
            raise ValueError("WEAVER RPC action chunk must not be empty")
        for action in actions:
            if len(action) != self.config.schema["action_dim"] or any(not math.isfinite(float(v)) for v in action):
                raise ValueError("WEAVER RPC actions must be finite and match the declared action schema")
        result = self.backend.predict_chunk(state, actions, published_version=self.published_version)
        if (not isinstance(result, Mapping) or type(result.get("served_version")) is not int
                or result.get("served_version") != self.published_version):
            raise ValueError("WEAVER result served_version differs from the requested bundle")
        self._validate_state(result.get("next_state"),owned=owned)
        _require_cpu({k:v for k,v in result.items() if k!='next_state'} if owned else result,"result")
        output = dict(result)
        from .sparse_output import sparse_last_geometry
        sparse=sparse_last_geometry(output,len(actions),allowed=(
            self.config.reward_provider=='world_head' and self.config.reward_semantics=='delta_native_score'
            and self.config.terminal_rule=='horizon_only'))
        # The backend predicts a batch of one. RPC consumers operate on time.
        for name, dimensions in (("observations", 6), ("predicted_states", 3), ("success_probabilities", 2),
                                 ("terminal_predictions", 2), ("reward_predictions", 2), ("valid_mask", 2)):
            value = output.get(name)
            if value is None:
                continue
            result_shape = getattr(value, "shape", None)
            if result_shape is not None and len(result_shape) == dimensions:
                if result_shape[0] != 1:
                    raise ValueError("WEAVER RPC accepts a single rollout per request")
                value = output[name] = value[0]
            if len(value) != (1 if name=='observations' and sparse else len(actions)):
                raise ValueError("WEAVER %s does not align with the requested actions" % name)
        if "observations" not in output or "predicted_states" not in output:
            raise ValueError("WEAVER result requires decoded observations and predicted states")
        if sparse and ('valid_mask' not in output or any(not bool(v) for v in output['valid_mask'])):
            raise ValueError('Sparse native output requires all full-horizon validity masks')
        for name in ("encoder_id", "normalizer_id"):
            if output.get(name) != self.config.schema[name]:
                raise ValueError("WEAVER result %s differs from the admitted bundle" % name)
        intervals = output.get("dt")
        if isinstance(intervals, (int, float)) and not isinstance(intervals, bool):
            intervals = output["dt"] = [float(intervals)] * len(actions)
        if intervals is None or len(intervals) != len(actions) or any(
                not math.isfinite(float(v)) or not math.isclose(float(v), self.config.schema["dt_seconds"], rel_tol=1e-5, abs_tol=1e-8)
                for v in intervals):
            raise ValueError("WEAVER result dt must match each requested transition's time scale")
        for name in ("terminal_predictions", "valid_mask"):
            for value in output.get(name, ()):
                scalar = value.item() if hasattr(value, "item") else value
                if type(scalar) is not bool:
                    raise ValueError("WEAVER %s must contain booleans" % name)
        if self.config.reward_provider == "world_head":
            native_score = self.config.reward_semantics == 'delta_native_score'
            key = 'reward_predictions' if native_score else 'success_probabilities'
            if key not in output or "terminal_predictions" not in output:
                raise ValueError("Integrated WEAVER result is missing calibrated success/terminal outputs")
            if any(not math.isfinite(float(v)) or not 0 <= float(v) <= 1 for v in output[key]):
                raise ValueError("Integrated WEAVER returned invalid success probabilities")
            if native_score and (any(bool(v) for v in output['terminal_predictions']) or
                                 'success_probabilities' in output):
                raise ValueError('Uncalibrated native score must not claim calibrated probabilities or termination')
        return output

    predict_chunk = request
