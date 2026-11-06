"""CPU episode ownership and lightweight, frame-aligned replay references.

Only the canonical PIL-resized uint8 frames are retained. Float conversion is
deliberately deferred and uses the same three float32 operations as
envs.utils.image_to_tensor. A replay reference or imagination lease owns its
episode strongly; the registry never keeps expired episodes alive.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import operator
import threading
from types import SimpleNamespace
import weakref

import numpy as np
from PIL import Image


CANONICAL_SIZE = (224, 224)
IMAGE_KEYS = ("full_image", "wrist_image")


@dataclass(frozen=True)
class FrozenMapping(Mapping):
    """Small pickleable immutable metadata mapping (unlike MappingProxyType)."""
    _items: tuple

    def __iter__(self):
        return (key for key, _ in self._items)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, key):
        for existing, value in self._items:
            if existing == key:
                return value
        raise KeyError(key)


def _immutable_array(value, dtype=None):
    # A read-only owning ndarray can have its writeable flag re-enabled. Bytes
    # backing prevents that, as well as aliasing the caller's writable input.
    array = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _freeze(value):
    if isinstance(value, Mapping):
        return FrozenMapping(tuple((key, _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.ndarray):
        return _immutable_array(value)
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool, bytes)):
        return value
    raise TypeError("Episode metadata must contain primitive values, mappings, sequences or arrays")


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def encode_observations(observations):
    """Pack raw LIBERO RGB uint8 observations without float quantization.

    Ordered views and PIL BILINEAR resize exactly match the original
    _obs_to_views/image_to_tensor preprocessing, including 224x224 inputs.
    Floating point images are rejected rather than guessed or requantized.
    """
    if not len(observations):
        raise ValueError("An episode requires at least one observation")
    frames = np.empty((len(observations), 2, 3, *CANONICAL_SIZE), dtype=np.uint8)
    for index, observation in enumerate(observations):
        for view, key in enumerate(IMAGE_KEYS):
            raw = np.asarray(observation[key])
            if raw.dtype != np.uint8 or raw.ndim != 3 or raw.shape[2] != 3:
                raise ValueError("%s must be an HWC RGB uint8 image" % key)
            if not raw.shape[0] or not raw.shape[1]:
                raise ValueError("Empty images cannot be encoded")
            resized = Image.fromarray(raw).resize(CANONICAL_SIZE, Image.Resampling.BILINEAR)
            frames[index, view] = np.asarray(resized).transpose(2, 0, 1)
    return frames


def normalize_frames(frames):
    """Return fresh float32 [-1,1] storage; do not fuse the arithmetic."""
    if np.asarray(frames).dtype != np.uint8:
        raise ValueError("normalize_frames requires canonical uint8 pixels")
    result = np.asarray(frames).astype(np.float32)
    result /= np.float32(255.0)
    result *= np.float32(2.0)
    result -= np.float32(1.0)
    return result


def _finite_matrix(value, count, name):
    value = np.asarray(value)
    if value.ndim != 2 or value.shape[0] != count or value.shape[1] < 1:
        raise ValueError("%s must be [T, dimension]" % name)
    if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
        raise ValueError("%s must be finite numeric values" % name)
    result = _immutable_array(value, np.float32)
    if not np.isfinite(result).all():
        raise ValueError("%s overflows float32" % name)
    return result


def _backend_values(config):
    if config is None:
        return {"backend": "ctrl", "schema": {}}
    if isinstance(config, Mapping):
        return {"backend": config.get("backend", config.get("world_backend", "ctrl")),
                "schema": config.get("schema", {}), "world_trainable": config.get("world_trainable", True),
                "backend_options": dict(config.get('backend_options', {}))}
    return {"backend": config.backend, "schema": dict(config.schema),
            "world_trainable": getattr(config, "world_trainable", True),
            "backend_options": dict(getattr(config, 'backend_options', {}))}


@dataclass(frozen=True, eq=False)
class EpisodeRecord:
    episode_id: str
    instruction: str
    frames: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    versions: Mapping = field(default_factory=dict)
    frame_groups: tuple = ()
    states: np.ndarray | None = None
    terminated: np.ndarray | None = None
    truncated: np.ndarray | None = None
    success: np.ndarray | None = None
    metadata: Mapping = field(default_factory=dict)
    backend_config: Mapping | None = None

    def __post_init__(self):
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise ValueError("episode_id must be a nonempty string")
        if not isinstance(self.instruction, str):
            raise ValueError("instruction must be a string")
        frames = np.asarray(self.frames)
        if frames.dtype != np.uint8 or frames.ndim != 5 or tuple(frames.shape[1:]) != (2, 3, *CANONICAL_SIZE):
            raise ValueError("frames must be canonical uint8 [T,2,3,224,224]")
        count = len(frames)
        if count < 1:
            raise ValueError("An episode requires at least one frame")
        actions = _finite_matrix(self.actions, count, "actions")
        if np.any(actions[0] != 0):
            raise ValueError("Reset frame must have a synthetic zero action")
        rewards = np.asarray(self.rewards)
        if rewards.shape != (count,) or not np.issubdtype(rewards.dtype, np.number) or not np.isfinite(rewards).all():
            raise ValueError("rewards must be a finite frame-aligned [T] array")
        if rewards[0] != 0:
            raise ValueError("Reset frame must have a synthetic zero reward")
        object.__setattr__(self, "frames", _immutable_array(frames))
        object.__setattr__(self, "actions", actions)
        # Preserve Python reward precision used by old WMExperience.rew; WEAVER
        # explicitly casts its per-frame reward payload to float32 downstream.
        object.__setattr__(self, "rewards", _immutable_array(rewards, np.float64))
        if self.states is not None:
            object.__setattr__(self, "states", _finite_matrix(self.states, count, "states"))
        labels = (self.terminated, self.truncated, self.success)
        if any(value is not None for value in labels) and not all(value is not None for value in labels):
            raise ValueError("terminated, truncated and success must be supplied together")
        for name in ("terminated", "truncated", "success"):
            value = getattr(self, name)
            if value is not None:
                value = np.asarray(value)
                if value.dtype != np.bool_ or value.shape != (count,):
                    raise ValueError("%s must be a boolean frame-aligned [T] array" % name)
                if name in ("terminated", "truncated") and value[:-1].any():
                    raise ValueError("An episode must end at its first terminal/truncated frame")
                object.__setattr__(self, name, _immutable_array(value))
        for name in ("versions", "metadata"):
            if not isinstance(getattr(self, name), Mapping):
                raise ValueError("%s must be a mapping" % name)
            object.__setattr__(self, name, _freeze(getattr(self, name)))
        object.__setattr__(self, "frame_groups", _freeze(self.frame_groups))
        config = _backend_values(self.backend_config)
        if config["backend"] not in ("ctrl", "weaver"):
            raise ValueError("Unknown world backend")
        if config["backend"] == "weaver":
            if self.states is None or self.terminated is None:
                raise ValueError("WEAVER requires measured states and all per-frame labels")
            from world_backends.libero import validate_online_schema
            from world_backends.contracts import validate_schema
            validate_schema(config["schema"], require_state=True)
            validate_online_schema(SimpleNamespace(**config), action_dim=self.actions.shape[1])
            if self.states.shape[1] != config["schema"]["state_dim"]:
                raise ValueError("Measured state width differs from backend schema")
        object.__setattr__(self, "backend_config", _freeze(config))

    @property
    def frame_count(self):
        return len(self.frames)

    @property
    def image_nbytes(self):
        return self.frames.nbytes

    @property
    def array_nbytes(self):
        return sum(value.nbytes for value in (self.frames, self.actions, self.rewards,
                   self.states, self.terminated, self.truncated, self.success) if value is not None)

    def window(self, start, length):
        return EpisodeWindowRef(self, start, length)

    def windows(self, length):
        length = _index(length, "length")
        if length < 1:
            raise ValueError("Window length must be positive")
        return [self.window(start, length) for start in range(max(0, self.frame_count - length + 1))]

    def __reduce__(self):
        # NumPy's default pickle recreates writable arrays. Reconstruct through
        # validation so cross-process use retains the ownership contract.
        names = self.__dataclass_fields__
        return (_restore_record, ({name: getattr(self, name) for name in names},))


def _restore_record(values):
    return EpisodeRecord(**values)


def _index(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("%s must be an integer" % name)
    try:
        return operator.index(value)
    except TypeError as error:
        raise ValueError("%s must be an integer" % name) from error


class _ObservationShape:
    """Shape-only input for the existing metadata validator, with no pixels.

    backend_window_payload only inspects the observation length/shape while
    validating labels and state. It never exports or accesses image values.
    """
    def __init__(self, count, image_size):
        self.shape = (count, 2, 3, *image_size)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, item):
        if not isinstance(item, slice):
            raise TypeError("Metadata validation supports frame slices only")
        return _ObservationShape(len(range(*item.indices(len(self)))), self.shape[-2:])


@dataclass(frozen=True, eq=False)
class EpisodeWindowRef:
    episode: EpisodeRecord
    start: int
    length: int

    def __post_init__(self):
        if not isinstance(self.episode, EpisodeRecord):
            raise TypeError("A window requires one EpisodeRecord")
        start, length = _index(self.start, "start"), _index(self.length, "length")
        if start < 0 or length < 1 or start + length > self.episode.frame_count:
            raise ValueError("Window bounds must remain inside one episode")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "length", length)

    @property
    def end(self):
        return self.start + self.length

    @property
    def sample_index(self):
        return self.start

    @property
    def episode_id(self):
        return self.episode.episode_id

    @property
    def sample_id(self):
        return "%s:window%d" % (self.episode_id, self.start)

    @property
    def instruction(self):
        return self.episode.instruction

    @property
    def model_versions(self):
        return _thaw(self.episode.versions)

    @property
    def obs_uint8(self):
        return self.episode.frames[self.start:self.end]

    @property
    def image_size(self):
        config = self.episode.backend_config
        return tuple(config["schema"]["image_size"]) if config["backend"] == "weaver" else CANONICAL_SIZE

    @property
    def world_observations(self):
        """Compact wire representation; only alternate WEAVER sizes need floats."""
        if self.episode.backend_config.get('backend_options', {}).get('transport') == 'isolated_online_ddp':
            # Resize/normalize only the current streamed micro in its native
            # owner, never a whole global/rank reservation inside Replay.
            return self.obs_uint8
        return self.obs_uint8 if self.image_size == CANONICAL_SIZE else self.obs

    @property
    def reward_primary(self):
        """Only the final primary view, preserving the legacy resize path.

        The consumer still performs the old normalization/inverse conversion.
        A uint8 return value is canonical, not an already round-tripped image.
        """
        image = self.episode.frames[self.end - 1, 0]
        if (not self.episode.backend_config.get("world_trainable", True) or self.image_size == CANONICAL_SIZE
                or self.episode.backend_config.get('backend_options', {}).get('transport') == 'isolated_online_ddp'):
            return image
        import torch
        import torch.nn.functional as functional
        tensor = torch.from_numpy(normalize_frames(image)).unsqueeze(0)
        return functional.interpolate(tensor, size=self.image_size,
            mode="bilinear", align_corners=False)[0].numpy()

    @property
    def obs(self):
        observations = normalize_frames(self.obs_uint8)
        if self.episode.backend_config["backend"] == "weaver":
            # Old path: PIL resize -> normalize float32 -> torch bilinear.
            # Never resize uint8 to the target or quantize resized floats.
            import torch
            import torch.nn.functional as functional
            target = tuple(self.episode.backend_config["schema"]["image_size"])
            tensor = torch.from_numpy(observations)
            observations = functional.interpolate(tensor.flatten(0, 1), size=target,
                mode="bilinear", align_corners=False).reshape(self.length, 2, 3, *target).numpy()
        return observations

    @property
    def act(self):
        return self.episode.actions[self.start:self.end]

    @property
    def rew(self):
        return float(self.episode.rewards[self.end - 1])

    @property
    def backend_payload(self):
        config = self.episode.backend_config
        if config["backend"] != "weaver":
            return {}
        from world_backends.libero import backend_window_payload
        config = SimpleNamespace(**_thaw(config))
        ep = self.episode
        result = backend_window_payload(config,
            _ObservationShape(ep.frame_count, config.schema["image_size"]),
            ep.states, ep.actions, ep.rewards, ep.success, ep.terminated, ep.truncated,
            self.start, self.end, ep.instruction, ep.episode_id, self.sample_id)
        specification = config.backend_options.get('native_reward_head', {})
        if specification.get('mode') == 'libero_discounted_success_value_v1':
            from fresh_reward_targets import reward_target_payload
            result.update(reward_target_payload(ep, self.start, self.end, specification))
        return result


def materialize_imagination_context(record, start, context_frames, dt_seconds):
    """Project one finite context, explicitly marking repeated-reset history.

    A negative context origin is permitted only by this opt-in helper. Padded
    slots repeat measured reset pixels/state and zero action, but carry false
    validity/action masks and zero elapsed time. ``frame_indices`` preserves the
    physical source mapping and does not invent negative physical frame IDs.
    """
    import math
    if not isinstance(record, EpisodeRecord):
        raise TypeError('An imagination context requires an EpisodeRecord')
    start = _index(start, 'start')
    context_frames = _index(context_frames, 'context_frames')
    if (isinstance(dt_seconds, bool) or not isinstance(dt_seconds, (int, float))
            or not math.isfinite(dt_seconds) or dt_seconds <= 0):
        raise ValueError('Context dt_seconds must be finite and positive')
    anchor = start + context_frames - 1
    if context_frames < 1 or anchor < 0 or anchor >= record.frame_count - 1:
        raise ValueError('A context must anchor a measured frame with a following transition')
    offsets = list(range(start, start + context_frames))
    indices = [max(index, 0) for index in offsets]
    valid = [index >= 0 for index in offsets]
    action_mask = [index > 0 for index in offsets]
    result = {'frames': _immutable_array(record.frames[indices]),
              'actions': _immutable_array(record.actions[indices]),
              'frame_indices': indices, 'valid_mask': valid,
              'action_mask': action_mask,
              'dt': [float(dt_seconds) if flag else 0.0 for flag in action_mask],
              'frame_start': max(0, start), 'context_start': start,
              'seed_anchor': anchor}
    if record.states is not None:
        result['states'] = _immutable_array(record.states[indices])
    padding = max(0, -start)
    if padding:
        result['context_padding'] = {
            'protocol': 'fresh_reset_left_repeat_v1', 'count': padding,
            'context_start': start, 'anchor_frame_index': anchor,
            'real_frame_start': 0, 'real_frame_end_exclusive': anchor + 1,
            'synthetic_history': True,
        }
    return result


def pack_episode(observations, transition_actions, transition_rewards, instruction,
                 episode_id, *, versions=None, frame_groups=(), frame_labels=None,
                 metadata=None, backend_config=None):
    """Convert raw observations and T-1 environment transitions to one record.

    frame_labels, when present, are (terminated, truncated, success), each of
    length T; no labels or measured state are invented from sparse rewards.
    """
    count = len(observations)
    actions = np.asarray(transition_actions, dtype=np.float32)
    rewards = np.asarray(transition_rewards)
    if actions.ndim != 2 or actions.shape[0] != count - 1:
        raise ValueError("Transition actions must be [T-1, action_dim]")
    if rewards.shape != (count - 1,):
        raise ValueError("Transition rewards must have length T-1")
    aligned_actions = np.concatenate((np.zeros((1, actions.shape[1]), np.float32), actions), axis=0)
    aligned_rewards = np.concatenate((np.zeros(1, np.float64), rewards))
    states = None
    if any("state" in observation for observation in observations):
        if not all("state" in observation for observation in observations):
            raise ValueError("Measured state must be present on every frame or none")
        states = np.stack([observation["state"] for observation in observations])
    labels = (None, None, None) if frame_labels is None else frame_labels
    if len(labels) != 3:
        raise ValueError("frame_labels must contain terminated, truncated and success")
    return EpisodeRecord(str(episode_id), instruction, encode_observations(observations),
        aligned_actions, aligned_rewards, versions={} if versions is None else versions,
        frame_groups=frame_groups, states=states, terminated=labels[0], truncated=labels[1],
        success=labels[2], metadata={} if metadata is None else metadata, backend_config=backend_config)


class EpisodeFrameStore:
    """Weak episode registry, with unique episode identities for this lifetime.

    Replay deques, evaluation reservations and imagination jobs own records.
    Dropping their final reference releases frame storage immediately; the
    registry retains only a small set of previously registered identity strings.
    """
    def __init__(self):
        self._records = weakref.WeakValueDictionary()
        self._seen = set()
        self._lock = threading.RLock()

    def register(self, record):
        if not isinstance(record, EpisodeRecord):
            raise TypeError("Only validated EpisodeRecord objects can be registered")
        with self._lock:
            if record.episode_id in self._seen:
                raise ValueError("Duplicate episode identity: %s" % record.episode_id)
            self._records[record.episode_id] = record
            self._seen.add(record.episode_id)
        return record

    add = register

    def __contains__(self, episode_id):
        """Identity admission check, including records whose pixels expired."""
        with self._lock:
            return str(episode_id) in self._seen

    def get(self, episode_id):
        with self._lock:
            return self._records[str(episode_id)]

    def stats(self):
        with self._lock:
            records = list(self._records.values())
            return {"resident_episodes": len(records),
                    "registered_episodes": len(self._seen),
                    "resident_frames": sum(record.frame_count for record in records),
                    "image_bytes": sum(record.image_nbytes for record in records),
                    "array_bytes": sum(record.array_nbytes for record in records)}
