"""Small, dependency-free contracts at the world-model boundary.

These types describe data and admission requirements; they do not assert that
metadata declarations replace offline validation or a strict checkpoint load.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable


SCHEMA_KEYS = (
    "domain", "action_schema_id", "state_schema_id", "encoder_id", "normalizer_id",
    "action_dim", "state_dim", "alignment", "image_keys", "image_size",
    "image_value_range", "dt_seconds", "n_history", "n_memory_frames", "t_memory",
)


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("%s must be a nonempty string" % name)
    return value


def _positive_int(value: Any, name: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise ValueError("%s must be a %s integer" % (name, "nonnegative" if allow_zero else "positive"))
    return value


def validate_schema(schema: Mapping[str, Any], *, require_state: bool = False) -> None:
    """Validate the WEAVER input description without guessing robot semantics."""
    if not isinstance(schema, Mapping):
        raise ValueError("schema must be a mapping")
    missing = [name for name in SCHEMA_KEYS if name not in schema]
    if missing:
        raise ValueError("World schema is missing: %s" % ", ".join(missing))
    for name in ("domain", "action_schema_id", "encoder_id", "normalizer_id"):
        _nonempty(schema[name], "schema." + name)
    _positive_int(schema["action_dim"], "schema.action_dim")
    _positive_int(schema["state_dim"], "schema.state_dim", allow_zero=not require_state)
    if schema["state_dim"]:
        _nonempty(schema["state_schema_id"], "schema.state_schema_id")
    if schema["alignment"] not in ("action_into_frame", "action_between_frames"):
        raise ValueError("schema.alignment must explicitly identify action/frame alignment")
    image_keys = schema["image_keys"]
    if not isinstance(image_keys, (tuple, list)) or len(image_keys) != 2 or len(set(image_keys)) != 2:
        raise ValueError("schema.image_keys must name the two ordered observation views")
    for name in image_keys:
        _nonempty(name, "schema.image_keys entry")
    image_size = schema["image_size"]
    if not isinstance(image_size, (tuple, list)) or len(image_size) != 2:
        raise ValueError("schema.image_size must be [height, width]")
    for value in image_size:
        _positive_int(value, "schema.image_size entry")
    if schema["image_value_range"] not in ("zero_one", "minus_one_one", "uint8"):
        raise ValueError("schema.image_value_range must name the actual image range")
    dt = schema["dt_seconds"]
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("schema.dt_seconds must be finite and positive")
    _positive_int(schema["n_history"], "schema.n_history")
    _positive_int(schema["n_memory_frames"], "schema.n_memory_frames", allow_zero=True)
    _positive_int(schema["t_memory"], "schema.t_memory")


@dataclass(frozen=True)
class BackendCapabilities:
    backend: str
    requires_state: bool = False
    provides_decoded_observations: bool = True
    provides_success: bool = False
    provides_terminal: bool = False
    native_progress: bool = False
    opaque_rollout_state: bool = True


@dataclass(frozen=True)
class WorldBackendConfig:
    backend: str = "ctrl"
    # A frozen World is an inference service, never an optimizer/Replay owner.
    world_trainable: bool = True
    reward_provider: str = "external_vlm"
    reward_semantics: str = "delta_success"
    terminal_rule: str = "argmax"
    terminal_threshold: float = 0.5
    reward_scale: float = 1.0
    schema: Mapping[str, Any] = field(default_factory=dict)
    checkpoint_metadata: Mapping[str, Any] = field(default_factory=dict)
    # Backend-local paths/tuning are carried without making the core import ML packages.
    backend_options: Mapping[str, Any] = field(default_factory=dict)
    critic_mode: str = "disabled"

    def __post_init__(self):
        if type(self.world_trainable) is not bool:
            raise ValueError("world_trainable must be an explicit boolean")
        if not self.world_trainable and (self.backend != "weaver" or self.reward_provider != "external_vlm"):
            raise ValueError("Frozen World currently requires WEAVER with the external Reward")
        if self.backend not in ("ctrl", "weaver"):
            raise ValueError("Unknown world backend: %s" % self.backend)
        if self.reward_provider not in ("external_vlm", "world_head"):
            raise ValueError("Unknown reward provider: %s" % self.reward_provider)
        if self.reward_semantics not in ("delta_success", "native_progress", "delta_native_score"):
            raise ValueError("Unknown reward semantics: %s" % self.reward_semantics)
        if self.terminal_rule not in ("argmax", "success_probability", "horizon_only"):
            raise ValueError("Unknown terminal rule: %s" % self.terminal_rule)
        if self.reward_semantics == 'delta_native_score':
            if (self.backend != 'weaver' or self.reward_provider != 'world_head'
                    or not self.world_trainable or self.terminal_rule != 'horizon_only'):
                raise ValueError('Uncalibrated native score requires trainable WEAVER world_head and horizon_only')
        elif self.terminal_rule == 'horizon_only':
            raise ValueError('horizon_only belongs to the explicit delta_native_score algorithm')
        if not isinstance(self.terminal_threshold, (int, float)) or isinstance(self.terminal_threshold, bool) or not 0 <= self.terminal_threshold <= 1:
            raise ValueError("terminal_threshold must be a probability")
        if not isinstance(self.reward_scale, (int, float)) or isinstance(self.reward_scale, bool) or not math.isfinite(self.reward_scale):
            raise ValueError("reward_scale must be finite")
        for name in ("schema", "checkpoint_metadata", "backend_options"):
            if not isinstance(getattr(self, name), Mapping):
                raise ValueError("%s must be a mapping" % name)
        if self.critic_mode != "disabled":
            raise ValueError("WEAVER critic integration is not implemented; Policy Value remains the bootstrap source")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None = None) -> "WorldBackendConfig":
        if mapping is None:
            return cls()
        if not isinstance(mapping, Mapping):
            raise ValueError("World backend configuration must be a mapping")
        values = dict(mapping)
        # A descriptive alias is convenient for top-level experiment config files.
        if "world_backend" in values:
            if "backend" in values and values["backend"] != values["world_backend"]:
                raise ValueError("Conflicting backend and world_backend configuration")
            values["backend"] = values.pop("world_backend")
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError("Unknown world backend configuration fields: %s" % sorted(unknown))
        return cls(**values)

    @classmethod
    def from_json(cls, path: str | Path) -> "WorldBackendConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_mapping(json.load(handle))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def capabilities(self) -> BackendCapabilities:
        # Capability claims are gated by the same metadata requirements as startup.
        self.validate_admission()
        integrated = self.reward_provider == "world_head" and self.reward_semantics != 'delta_native_score'
        return BackendCapabilities(self.backend, requires_state=self.backend == "weaver",
                                   provides_success=integrated, provides_terminal=integrated)

    def validate_admission(self) -> None:
        """Fail closed before allocating services or loading GPU models."""
        if 'fresh' in self.backend_options:
            from fresh_config import FreshConfig
            fresh = FreshConfig.from_mapping(self.backend_options['fresh'])
            fresh.validate_reward_binding(self.backend_options.get('native_reward_head'))
            if fresh.enabled and (self.backend != 'weaver' or self.reward_semantics != 'delta_native_score'):
                raise ValueError('FRESH requires the explicit WEAVER native score candidate')
        if self.reward_semantics == "native_progress":
            raise NotImplementedError("native_progress is a distinct algorithm and is not implemented; specify reward scale, time discount, terminal and bootstrap rules in a separate experiment")
        if self.backend == "ctrl":
            if self.reward_provider != "external_vlm":
                raise ValueError("CTRL has no calibrated integrated reward/terminal head")
            if self.terminal_rule != "argmax":
                raise ValueError("The legacy CTRL rollout preserves argmax terminal semantics; an explicit probability rule requires the WEAVER rollout adapter")
            return
        validate_schema(self.schema, require_state=True)
        if self.schema["alignment"] != "action_into_frame":
            raise ValueError("The initial WEAVER adapter requires action_into_frame alignment")
        metadata = self.checkpoint_metadata
        for name in ("checkpoint_id", "code_revision"):
            _nonempty(metadata.get(name), "checkpoint_metadata." + name)
        recorded = metadata.get("schema")
        if not isinstance(recorded, Mapping):
            raise ValueError("WEAVER checkpoint metadata must record its input schema")
        for name in SCHEMA_KEYS:
            configured, checkpoint = self.schema[name], recorded.get(name)
            if isinstance(configured, (tuple, list)) and isinstance(checkpoint, (tuple, list)):
                configured, checkpoint = list(configured), list(checkpoint)
            if configured != checkpoint:
                raise ValueError("WEAVER checkpoint schema mismatch: %s" % name)
        load_report = metadata.get("load_report")
        if not isinstance(load_report, Mapping) or load_report.get("strict") is not True:
            raise ValueError("WEAVER checkpoint requires a strict load_report as offline admission evidence")
        if load_report.get("missing_keys") != [] or load_report.get("unexpected_keys") != []:
            raise ValueError("WEAVER checkpoint load_report must have no missing or unexpected keys")
        if load_report.get("shape_mismatches", []) != []:
            raise ValueError("WEAVER checkpoint load_report must have no tensor shape mismatches")
        _nonempty(load_report.get("evidence_id"), "checkpoint_metadata.load_report.evidence_id")
        if self.world_trainable and self.backend_options.get('transport') == 'isolated_online_ddp':
            native_score = self.reward_semantics == 'delta_native_score'
            if self.reward_provider != 'external_vlm' and not native_score:
                raise ValueError('Online WEAVER keeps the existing external Reward')
            sha = self.backend_options.get('initial_checkpoint_sha256')
            if (not isinstance(sha, str) or len(sha) != 64
                    or any(c not in '0123456789abcdef' for c in sha)
                    or load_report.get('checkpoint_sha256') != sha):
                raise ValueError('Online WEAVER must bind the strict admitted initial native snapshot SHA')
            provenance = metadata.get('training_provenance', {})
            if provenance.get('optimizer_steps') != 3200 or not provenance.get('run_id'):
                raise ValueError('Online WEAVER starts fresh optimizer/Replay from evaluated step3200 weights')
            for field in ('training_python', 'inference_python'):
                _nonempty(self.backend_options.get(field), 'backend_options.' + field)
            wanted = {'backend': 'torch_ddp', 'world_size': 4, 'trainable_scope': 'wm+rm' if native_score else 'wm',
                'micro_batch_size': 4, 'gradient_accumulation_steps': 64, 'global_batch_size': 1024,
                'optimizer': 'adamw', 'betas': [0.9, 0.99], 'lr': 1e-5, 'warmup_steps': 20,
                'schedule': 'constant_after_linear_warmup', 'weight_decay': 0.01,
                'gradient_clip': 1.0, 'parameter_dtype': 'float32', 'autocast_dtype': 'bfloat16', 'ema': False}
            if self.backend_options.get('online_training') != wanted:
                raise ValueError('Online WEAVER native DDP/AdamW recipe differs from the explicit candidate')
            if (self.schema.get('n_history'), self.schema.get('n_memory_frames'),
                    self.schema.get('t_memory'), self.schema.get('dt_seconds')) != (2, 6, 5, 0.05):
                raise ValueError('Online WEAVER requires the admitted 32 context plus 8 future LIBERO semantics')
        if not self.world_trainable:
            digest = self.backend_options.get("frozen_checkpoint_sha256")
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)):
                raise ValueError("Frozen WEAVER requires a lowercase SHA256 for its exact native checkpoint")
            if load_report.get("checkpoint_sha256") != digest:
                raise ValueError("Frozen WEAVER SHA256 differs from its strict native load evidence")
            provenance = metadata.get("training_provenance", {})
            if not isinstance(provenance, Mapping) or not provenance.get("run_id"):
                raise ValueError("Frozen WEAVER requires recorded training provenance")
            steps = provenance.get("optimizer_steps", provenance.get("global_step", provenance.get("step")))
            _positive_int(steps, "checkpoint_metadata.training_provenance.optimizer_steps")
            if self.backend_options.get("publication_weights", "online") != "online":
                raise ValueError("Frozen WEAVER must use the audited online native snapshot")
        if self.reward_semantics == 'delta_native_score':
            from weaver_native_reward import validate_specification
            validate_specification(self.backend_options.get('native_reward_head'))
            if self.backend_options.get('train_reward_head') is not True:
                raise ValueError('Native reward head must be trained from true per-frame simulator labels')
            if any(self.checkpoint_metadata.get(key,{}).get('calibrated') is not False
                   for key in ('success_head','terminal_head')):
                raise ValueError('Native score must retain false success/terminal calibration provenance')
        elif self.reward_provider == "world_head":
            self._validate_integrated_heads()

    def _validate_integrated_heads(self) -> None:
        for name, label in (("success_head", "environment_success"), ("terminal_head", "environment_terminal")):
            head = self.checkpoint_metadata.get(name)
            if not isinstance(head, Mapping) or head.get("calibrated") is not True or head.get("label_semantics") != label:
                raise ValueError("Integrated reward requires calibrated %s with %s labels; native progress is insufficient" % (name, label))
            _nonempty(head.get("calibration_id"), "checkpoint_metadata.%s.calibration_id" % name)
        success = self.checkpoint_metadata["success_head"]
        if success.get("source") != "rm" or success.get("output") not in ("probability", "logit"):
            raise ValueError("success_head must declare its rm source and probability/logit output")
        terminal = self.checkpoint_metadata["terminal_head"]
        if terminal.get("source") not in ("same_success", "terminal_head") or terminal.get("rule") != self.terminal_rule:
            raise ValueError("terminal_head must declare a calibrated source and the configured terminal rule")
        if self.terminal_rule == "success_probability" and terminal.get("threshold") != self.terminal_threshold:
            raise ValueError("terminal_head calibration threshold differs from terminal_threshold")


@dataclass
class RolloutResult:
    observations_for_policy: Any
    opaque_next_state: Any
    served_bundle_version: int
    encoder_id: str
    normalizer_id: str
    request_lineage: Mapping[str, Any]
    predicted_state: Any = None
    reward_predictions: Any = None
    terminal_predictions: Any = None
    success_probabilities: Any = None
    success_logits: Any = None
    truncated: Any = None
    valid_mask: Any = None
    dt: Any = None
    schema_id: str = ""
    reward_semantics: str = 'delta_success'
    reward_calibrated: bool | None = None

    def validate(self, expected_steps: int | None = None, *, require_reward: bool = False,
                 require_state: bool = False) -> "RolloutResult":
        _positive_int(self.served_bundle_version, "served_bundle_version", allow_zero=True)
        _nonempty(self.encoder_id, "encoder_id")
        _nonempty(self.normalizer_id, "normalizer_id")
        if not isinstance(self.request_lineage, Mapping) or not self.request_lineage:
            raise ValueError("Rollout result must retain request_lineage")
        count = len(self.observations_for_policy)
        if count < 1 or (expected_steps is not None and count != expected_steps):
            raise ValueError("Rollout observation count differs from the requested action count")
        if require_state and self.predicted_state is None:
            raise ValueError("Rollout result is missing predicted state")
        if require_reward:
            if self.reward_semantics=='delta_native_score':
                if (self.reward_predictions is None or self.terminal_predictions is None
                        or self.reward_calibrated is not False or self.success_probabilities is not None
                        or self.success_logits is not None or any(bool(v) for v in self.terminal_predictions)):
                    raise ValueError('Native reward rollout requires uncalibrated bounded scores and horizon-only continuation')
                if any(not math.isfinite(float(v)) or not 0<=float(v)<=1 for v in self.reward_predictions):
                    raise ValueError('Native reward scores must be bounded and finite')
            elif self.terminal_predictions is None or (self.success_probabilities is None and self.success_logits is None):
                raise ValueError("Integrated rollout requires calibrated success and terminal predictions")
        for name in ("predicted_state", "reward_predictions", "terminal_predictions", "success_probabilities",
                     "success_logits", "truncated", "valid_mask", "dt"):
            value = getattr(self, name)
            if value is not None and len(value) != count:
                raise ValueError("Rollout %s must align with every predicted frame" % name)
        if self.success_probabilities is not None:
            for value in self.success_probabilities:
                if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
                    raise ValueError("Success predictions must be probabilities, not native progress")
        if self.dt is not None and any(not math.isfinite(float(v)) or float(v) <= 0 for v in self.dt):
            raise ValueError("Rollout dt values must be finite and positive")
        return self

    def to_dict(self) -> dict[str, Any]:
        # Do not deepcopy opaque caches/tensors or interpret backend-internal state.
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@runtime_checkable
class WorldBackend(Protocol):
    def capabilities(self) -> BackendCapabilities | Mapping[str, Any]: ...
    def prepare_train_batch(self, episode_window: Any) -> Any: ...
    def compute_losses(self, prepared_batch: Any) -> Any: ...
    def init_rollout(self, real_context: Any, published_version: int) -> Any: ...
    def predict_chunk(self, opaque_state: Any, action_chunk: Any, requested_outputs: Any = None) -> Any: ...
    def publication_inventory(self) -> Mapping[str, Any]:
        """Name parameters/buffers and bundle identity for the existing IPC transport."""
        ...
