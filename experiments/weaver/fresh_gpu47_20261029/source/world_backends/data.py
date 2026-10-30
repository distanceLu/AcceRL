"""Frame-aligned real-world replay contract; no invented state or labels."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from .contracts import SCHEMA_KEYS, WorldBackendConfig, validate_schema


def _length(value: Any, name: str) -> int:
    if value is None or isinstance(value, (str, bytes, Mapping)):
        raise ValueError("%s must be a frame sequence" % name)
    try:
        return len(value)
    except TypeError as error:
        raise ValueError("%s must be a frame sequence" % name) from error


def _matrix_width(value: Any, width: int, name: str) -> None:
    shape = getattr(value, "shape", None)
    if shape is not None:
        if len(shape) != 2 or shape[1] != width:
            raise ValueError("%s must be [time, %s]" % (name, width))
    else:
        for row in value:
            if _length(row, name + " row") != width:
                raise ValueError("%s width differs from its declared schema" % name)
    for row in value:
        if any(not math.isfinite(float(item)) for item in row):
            raise ValueError("%s must contain finite values" % name)


def _boolean_sequence(value: Any, name: str) -> None:
    # Accept bool/NumPy bool scalars, but not arbitrary truthy strings or numbers.
    for item in value:
        if type(item).__name__ not in ("bool", "bool_"):
            raise ValueError("%s must contain boolean values" % name)


@dataclass
class EpisodeWindow:
    observations: Any
    actions: Any
    states: Any
    rewards: Any
    success: Any
    terminated: Any
    truncated: Any
    valid_mask: Any
    dt: Any
    episode_id: str
    sample_id: str
    task_id: str
    instruction: str
    schema: Mapping[str, Any]
    label_source: str
    action_mask: Any = None
    schema_version: str = "episode_window_v2"

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "EpisodeWindow":
        if not isinstance(values, Mapping):
            raise ValueError("EpisodeWindow requires a mapping")
        # Aliases only rename fields; missing state or sparse tail rewards are
        # never expanded into unobserved trajectories.
        aliases = {"obs": "observations", "act": "actions", "state": "states",
                   "rew": "rewards", "done": "terminated"}
        normalized = dict(values)
        for old, new in aliases.items():
            if old in normalized:
                if new in normalized:
                    raise ValueError("Duplicate episode field aliases: %s and %s" % (old, new))
                normalized[new] = normalized.pop(old)
        unknown = set(normalized) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError("Unknown EpisodeWindow fields: %s" % sorted(unknown))
        try:
            return cls(**normalized)
        except TypeError as error:
            raise ValueError("EpisodeWindow requires explicit per-frame states, labels, masks and identity") from error

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def validate(self, config: WorldBackendConfig | None = None) -> "EpisodeWindow":
        require_state = config is not None and config.backend == "weaver"
        validate_schema(self.schema, require_state=require_state)
        if self.schema_version != "episode_window_v2":
            raise ValueError("Unsupported episode schema version")
        for name in ("episode_id", "sample_id", "task_id", "label_source"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError("EpisodeWindow requires nonempty %s" % name)
        if not isinstance(self.instruction, str):
            raise ValueError("instruction must be a string")
        count = _length(self.observations, "observations")
        if count < 2:
            raise ValueError("EpisodeWindow requires at least two observation frames")
        shape = getattr(self.observations, "shape", None)
        if shape is not None:
            height, width = self.schema["image_size"]
            if len(shape) != 5 or tuple(shape[1:]) != (len(self.schema["image_keys"]), 3, height, width):
                raise ValueError("observations must be [time, ordered views, 3, height, width] from schema")
        for name in ("rewards", "success", "terminated", "truncated", "valid_mask"):
            if _length(getattr(self, name), name) != count:
                raise ValueError("%s must align with every observation frame" % name)
        for name in ("success", "terminated", "truncated", "valid_mask"):
            _boolean_sequence(getattr(self, name), name)
        if any(not math.isfinite(float(value)) for value in self.rewards):
            raise ValueError("rewards must be finite per-frame environment rewards")
        state_dim = self.schema["state_dim"]
        if self.states is None:
            if require_state or state_dim:
                raise ValueError("Missing measured state sequence; image replay cannot supply WEAVER state")
        else:
            if _length(self.states, "states") != count:
                raise ValueError("states must align with every observation frame")
            _matrix_width(self.states, state_dim, "states")
        action_count = count if self.schema["alignment"] == "action_into_frame" else count - 1
        if _length(self.actions, "actions") != action_count:
            raise ValueError("Action count differs from the declared action/frame alignment")
        _matrix_width(self.actions, self.schema["action_dim"], "actions")
        if _length(self.dt, "dt") != action_count:
            raise ValueError("dt must align with every action")
        if self.action_mask is None:
            raise ValueError("action_mask is required; reset or padded actions must be explicit")
        if _length(self.action_mask, "action_mask") != action_count:
            raise ValueError("action_mask must align with every action")
        _boolean_sequence(self.action_mask, "action_mask")
        for index, value in enumerate(self.dt):
            elapsed = float(value)
            if not math.isfinite(elapsed) or elapsed < 0 or (self.action_mask[index] and elapsed <= 0):
                raise ValueError("dt must be positive for valid transitions; masked reset dt may be zero")
            if self.action_mask[index] and not math.isclose(elapsed, self.schema["dt_seconds"], rel_tol=1e-5, abs_tol=1e-8):
                raise ValueError("Valid transition dt differs from the configured model time scale")
            destination = index if self.schema["alignment"] == "action_into_frame" else index + 1
            if self.action_mask[index] and not self.valid_mask[destination]:
                raise ValueError("A valid action cannot target a padded observation")
        if config is not None and config.schema:
            for name in SCHEMA_KEYS:
                left, right = self.schema[name], config.schema.get(name)
                if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
                    left, right = list(left), list(right)
                if left != right:
                    raise ValueError("Replay and backend schema mismatch: %s" % name)
        return self


def validate_episode_window(window: EpisodeWindow | Mapping[str, Any],
                            config: WorldBackendConfig | None = None) -> EpisodeWindow:
    if not isinstance(window, EpisodeWindow):
        window = EpisodeWindow.from_mapping(window)
    return window.validate(config)
