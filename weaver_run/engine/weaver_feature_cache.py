"""Selected-frame, frozen-encoder features with a bounded CPU-only cache.

Only image features are cached. State/action/reward/dt/window checks and all
trainable WM/RM computation remain per sample. Encoder weights must be frozen,
posterior encoding deterministic (the pinned SD3Encoder uses posterior.mode),
and cache identity must describe the loaded encoder and exact preprocessing.
The caller must never equate CUDA shape changes with bitwise equivalence.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
import re


DEFAULT_CACHE_BYTES = 64 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SCHEMA_FIELDS = ("domain", "action_schema_id", "state_schema_id", "encoder_id",
                  "normalizer_id", "action_dim", "state_dim", "alignment", "image_keys",
                  "image_size", "image_value_range", "dt_seconds")


def _positive_int(value, name, *, zero=False):
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError(name + " must be a bounded nonnegative integer" if zero else name + " must be a positive integer")
    return value


def fingerprint(value):
    """Hash strict, finite primitive metadata; never repr an opaque object."""
    def checked(item):
        if item is None or type(item) in (str, bool, int):
            return item
        if type(item) is float and math.isfinite(item):
            return item
        if isinstance(item, (list, tuple)):
            return [checked(part) for part in item]
        if type(item) is dict and all(type(key) is str for key in item):
            return {key: checked(part) for key, part in item.items()}
        raise ValueError("Feature identity must contain only finite primitive metadata")
    return hashlib.sha256(json.dumps(checked(value), sort_keys=True,
        separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()


def training_frame_selection(window_frames, n_history, horizon, n_memory_frames, t_memory):
    """Return unique original positions and compact main/memory gather maps."""
    for name, value in (("window_frames", window_frames), ("n_history", n_history),
                        ("horizon", horizon), ("t_memory", t_memory)):
        _positive_int(value, name)
    _positive_int(n_memory_frames, "n_memory_frames", zero=True)
    start = window_frames - n_history - horizon
    current = window_frames - horizon - 1
    memory = [current - (n_memory_frames - index) * t_memory for index in range(n_memory_frames)]
    main = list(range(start, window_frames))
    if start < 0 or any(index < 0 or index >= window_frames for index in memory):
        raise ValueError("Training window cannot provide the required actual history/memory/future frames")
    selected = sorted(set(main + memory))
    inverse = {index: offset for offset, index in enumerate(selected)}
    return {"selected_indices": selected, "main_indices": main, "memory_indices": memory,
            "main_positions": [inverse[index] for index in main],
            "memory_positions": [inverse[index] for index in memory]}


def _cpu_array(value, name):
    import numpy as np
    if hasattr(value, "device") and hasattr(value, "detach"):
        if value.device.type != "cpu":
            raise ValueError(name + " must be CPU data before image selection or hashing")
        value = value.detach()
        # NumPy has no native bfloat16. Conversion is exact for BF16 values.
        if str(value.dtype) == "torch.bfloat16":
            value = value.float()
        value = value.numpy()
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(name + " cannot be converted to a CPU array") from error
    if array.dtype.kind not in "buif":
        raise ValueError(name + " must contain real numeric values")
    return array


def validate_cpu_window(sample, schema, min_frames, training, *, require_real_labels=False,
                        allow_context_padding=False):
    """Preserve full-window admission on CPU before selecting image frames.

    The image normalization/resize itself remains the original float32
    /255,*2,-1 -> inverse normalization -> bilinear path owned by the backend.
    uint8 frames are necessarily finite and in the canonical byte range.
    """
    import numpy as np
    _positive_int(min_frames, "min_frames")
    if not isinstance(sample, dict) or not isinstance(sample.get("schema"), dict):
        raise ValueError("Every WEAVER sample/context must carry its explicit schema")
    for key in _SCHEMA_FIELDS:
        if sample["schema"].get(key) != schema.get(key):
            raise ValueError("Sample schema." + key + " does not match the admitted checkpoint")
    if training and require_real_labels and sample.get("label_source") != "libero_env_step":
        raise ValueError("Native reward accepts only measured per-frame simulator labels")
    obs = _cpu_array(sample.get("obs", sample.get("observations")), "observations")
    states = _cpu_array(sample.get("states"), "states")
    actions = _cpu_array(sample.get("actions"), "actions")
    if obs.ndim != 5 or obs.shape[1] != len(schema["image_keys"]) or obs.shape[2] != 3:
        raise ValueError("Observations must be [T,views,3,H,W] in declared camera order")
    count = obs.shape[0]
    if (count < min_frames or states.shape != (count, schema["state_dim"]) or
            actions.shape != (count, schema["action_dim"]) or min(obs.shape[-2:]) < 1):
        raise ValueError("Insufficient or misaligned observation/state/action frames")
    # The actual candidate's master/input state dtype is FP32. A finite
    # float64 value overflowing that conversion must be rejected here just
    # as the original GPU-side _tensor + finite check rejected it.
    with np.errstate(over="ignore",invalid="ignore"):
        states=states.astype(np.float32,copy=False)
        actions=actions.astype(np.float32,copy=False)
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("State/action values must be finite")
    if obs.dtype != np.uint8:
        if obs.dtype.kind != "f" or not np.isfinite(obs).all():
            raise ValueError("Images must be canonical uint8 or finite normalized floats")
        low, high = (-1., 1.) if schema["image_value_range"] == "minus_one_one" else (0., 1.)
        if obs.min() < low - 1e-5 or obs.max() > high + 1e-5:
            raise ValueError("Image range differs from schema")
    dt_value = sample.get("dt", sample.get("dt_seconds"))
    if dt_value is None:
        raise ValueError("Every sample must declare the actual control interval")
    intervals = _cpu_array(dt_value, "dt").astype(np.float64, copy=False)
    if intervals.ndim == 0:
        intervals = np.full(count, intervals.item(), dtype=np.float64)
    if intervals.shape != (count,) or not np.isfinite(intervals).all():
        raise ValueError("dt must be a finite frame-aligned vector")
    mask_value = sample.get("action_mask")
    if mask_value is None:
        raise ValueError("Action mask must identify reset sentinel versus real transitions")
    action_mask = _cpu_array(mask_value, "action_mask").astype(bool, copy=False)
    if action_mask.shape != (count,):
        raise ValueError("Action mask must align with every context frame")
    from fresh_context import validate_reset_padding
    padded_indices = validate_reset_padding(sample, obs, states, actions, intervals, action_mask,
        allowed=allow_context_padding, training=training)
    if padded_indices is None and not action_mask[1:].all():
        raise ValueError("Only the first reset action may be masked")
    if (intervals[action_mask] <= 0).any() or (intervals[~action_mask] != 0).any():
        raise ValueError("Masked reset interval must be zero; actual transitions must be positive")
    expected_dt = schema.get("dt_seconds")
    if expected_dt is None or not np.allclose(intervals[action_mask], float(expected_dt), atol=1e-6, rtol=1e-4):
        raise ValueError("Actual control intervals differ from the trained timestep")
    if training:
        valid_value = sample.get("valid_mask", sample.get("mask"))
        if valid_value is None:
            raise ValueError("Native WEAVER has no training padding mask")
        valid = _cpu_array(valid_value, "valid_mask").astype(bool, copy=False)
        if valid.shape != (count,) or not valid.all():
            raise ValueError("All frames must be genuine valid training frames")
        terminals_value = sample.get("terminated", sample.get("done"))
        truncated_value = sample.get("truncated")
        if terminals_value is None or truncated_value is None:
            raise ValueError("Frame-aligned terminated/truncated labels are required")
        terminal = _cpu_array(terminals_value, "terminated").astype(bool, copy=False)
        truncated = _cpu_array(truncated_value, "truncated").astype(bool, copy=False)
        if terminal.shape != (count,) or truncated.shape != (count,) or terminal[:-1].any() or truncated[:-1].any():
            raise ValueError("Training windows must not cross episode boundaries")
    return {"raw": sample, "observations": obs, "states": states, "actions": actions,
            "frame_count": count, "dt": intervals, "action_mask": action_mask}


@dataclass(frozen=True, slots=True)
class FeatureNamespace:
    # These are digests of admitted identities, not claims that this helper
    # independently rescanned encoder weight files. The caller includes the
    # admitted bundle receipt, encoder/source ID and current loaded scope.
    run_id: str
    encoder_sha256: str
    preprocessing_sha256: str

    def __post_init__(self):
        if type(self.run_id) is not str or not self.run_id.strip():
            raise ValueError("Feature namespace requires the exact run identity")
        for value in (self.encoder_sha256, self.preprocessing_sha256):
            if type(value) is not str or _SHA256.fullmatch(value) is None:
                raise ValueError("Feature namespace requires explicit SHA256 identities")

    @property
    def sha256(self):
        return fingerprint({"run_id": self.run_id, "encoder": self.encoder_sha256,
                            "preprocessing": self.preprocessing_sha256})


@dataclass(frozen=True, slots=True)
class FrameFeatureKey:
    namespace_sha256: str
    run_id: str
    episode_id: str
    frame_index: int
    camera: str
    content_sha256: str

    @property
    def identity(self):
        return (self.namespace_sha256, self.run_id, self.episode_id, self.frame_index, self.camera)


def build_frame_keys(source, namespace, indices, *, require_window=False):
    """Hash only selected raw CPU frames using actual episode/global-frame IDs.

    Canonical Replay window IDs and real-context seed IDs encode their true
    starts. Explicit frame_start and seed lineage must agree when present.
    No image hash is derived from state, timestep number, or model version.
    """
    import numpy as np
    if not isinstance(namespace, FeatureNamespace):
        raise ValueError("Feature keys require a concrete namespace")
    episode = source.get("episode_id")
    if type(episode) is not str or not episode.strip():
        raise ValueError("Feature cache needs a real episode identity")
    sample_id = source.get("sample_id")
    match = re.fullmatch(re.escape(episode) + r":(window|seed)(0|[1-9][0-9]*)", sample_id or "")
    if require_window and (match is None or match.group(1) != "window"):
        raise ValueError("Training features require a canonical real Replay window identity")
    explicit = source.get("frame_start")
    if explicit is not None:
        _positive_int(explicit, "frame_start", zero=True)
    if match is None and explicit is None:
        raise ValueError("Feature cache needs an actual window frame start, not a guessed sample counter")
    start = int(match.group(2)) if match else explicit
    if match and explicit is not None and explicit != start:
        raise ValueError("Explicit frame start differs from canonical Replay sample identity")
    lineage = source.get("request_lineage", {})
    if not isinstance(lineage, dict):
        raise ValueError("Context request lineage must be explicit primitive metadata")
    if "seed_start" in lineage:
        _positive_int(lineage["seed_start"], "lineage.seed_start", zero=True)
        if lineage["seed_start"] != start:
            raise ValueError("Real-context seed lineage differs from its actual frame start")
    if "episode_id" in lineage and lineage["episode_id"] != episode:
        raise ValueError("Real-context lineage refers to a different episode")
    obs = _cpu_array(source.get("obs", source.get("observations")), "raw cache observations")
    cameras = source.get("schema", {}).get("image_keys")
    if (not isinstance(cameras, (list, tuple)) or not cameras or len(set(cameras)) != len(cameras)
            or any(type(name) is not str or not name for name in cameras) or obs.dtype != np.uint8
            or obs.ndim != 5 or obs.shape[1:3] != (len(cameras), 3)):
        raise ValueError("Cache identity accepts only canonical uint8 frames and explicit unique cameras")
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("Cache keys require distinct actual selected frame positions")
    frame_indices = source.get('frame_indices')
    if source.get('context_padding') is not None and frame_indices is None:
        raise ValueError('Padded cache history requires its actual source frame mapping')
    if frame_indices is not None:
        if (not isinstance(frame_indices, (list, tuple)) or len(frame_indices) != len(obs)
                or any(type(i) is not int or i < 0 for i in frame_indices)):
            raise ValueError('Cache frame indices must identify actual nonnegative source positions')
        if source.get('context_padding') is not None:
            from fresh_context import validate_reset_padding
            validate_reset_padding(source, obs, _cpu_array(source.get('states'), 'states'),
                _cpu_array(source.get('actions'), 'actions'), _cpu_array(source.get('dt'), 'dt'),
                _cpu_array(source.get('action_mask'), 'action_mask').astype(bool),
                allowed=not require_window, training=require_window)
        elif list(frame_indices) != list(range(start, start + len(obs))):
            raise ValueError('Unpadded cache positions must match genuine contiguous frame history')
    result = []
    for index in indices:
        if type(index) is not int or not 0 <= index < len(obs):
            raise ValueError("Selected frame position is outside the raw window")
        views = []
        for view, camera in enumerate(cameras):
            image = np.ascontiguousarray(obs[index, view])
            digest = hashlib.sha256()
            digest.update(json.dumps({"dtype": image.dtype.str, "shape": image.shape,
                                      "layout": "CHW"}, separators=(",", ":")).encode())
            digest.update(memoryview(image).cast("B"))
            views.append(FrameFeatureKey(namespace.sha256, namespace.run_id, episode,
                                         frame_indices[index] if frame_indices is not None else start + index,
                                         camera, digest.hexdigest()))
        result.append(views)
    return result


class FrozenFeatureCache:
    """CPU tensor clones only; LRU bounded in both bytes and entry count."""
    def __init__(self, namespace, max_bytes=DEFAULT_CACHE_BYTES, max_entries=8192):
        if not isinstance(namespace, FeatureNamespace):
            raise ValueError("Cache requires encoder/preprocessing/run identity")
        self.namespace = namespace
        self.max_bytes = _positive_int(max_bytes, "max_bytes", zero=True)
        self.max_entries = _positive_int(max_entries, "max_entries")
        self._entries = OrderedDict()
        self.resident_bytes = self.hits = self.misses = self.evictions = 0
        self.feature_payload_bytes = 0
        self.generation = 0

    @staticmethod
    def _entry_bytes(key, feature):
        # Conservative owned-object accounting as well as tensor payload.
        # Repeated immutable strings are counted again (an overestimate).
        # CPython/torch allocator reservations and transient caller tensors are
        # not process-RSS limits; max_entries bounds the index independently.
        import sys
        fields = (key.namespace_sha256, key.run_id, key.episode_id,
                  key.frame_index, key.camera, key.content_sha256)
        return (feature.numel() * feature.element_size() + sys.getsizeof(feature) +
                sys.getsizeof(key) + sys.getsizeof(key.identity) +
                sys.getsizeof((key, feature, 0)) + sum(sys.getsizeof(part) for part in fields) + 256)

    def _check(self, key):
        if not isinstance(key, FrameFeatureKey) or key.namespace_sha256 != self.namespace.sha256 or key.run_id != self.namespace.run_id:
            raise ValueError("Feature key belongs to a different encoder/preprocessing/run namespace")
        if _SHA256.fullmatch(key.content_sha256) is None:
            raise ValueError("Feature key needs a SHA256 content identity")
        old = self._entries.get(key.identity)
        if old is not None and old[0].content_sha256 != key.content_sha256:
            raise ValueError("The same retained episode/frame/camera identity changed pixel content")
        return old

    def get(self, key):
        entry = self._check(key)
        if entry is None:
            self.misses += 1
            return None
        self._entries.move_to_end(key.identity)
        self.hits += 1
        return entry[1].clone()

    def put(self, key, feature):
        import torch
        self._check(key)
        if (not isinstance(feature, torch.Tensor) or feature.device.type != "cpu" or
                feature.requires_grad or not feature.is_floating_point() or feature.numel() == 0):
            raise ValueError("Only nonempty detached floating CPU features may be cached")
        if not torch.isfinite(feature).all().item():
            raise ValueError("Nonfinite encoder features must never enter the cache")
        payload_size = feature.numel() * feature.element_size()
        size = self._entry_bytes(key, feature)
        if size > self.max_bytes or self.max_bytes == 0:
            return False
        feature = feature.detach().clone().contiguous()
        old = self._entries.pop(key.identity, None)
        if old is not None:
            self.resident_bytes -= old[2]
            self.feature_payload_bytes -= old[1].numel() * old[1].element_size()
        while self._entries and (self.resident_bytes + size > self.max_bytes or len(self._entries) >= self.max_entries):
            _, evicted = self._entries.popitem(last=False)
            self.resident_bytes -= evicted[2]
            self.feature_payload_bytes -= evicted[1].numel() * evicted[1].element_size()
            self.evictions += 1
        self._entries[key.identity] = (key, feature, size)
        self.resident_bytes += size
        self.feature_payload_bytes += payload_size
        return True

    def clear(self):
        self._entries.clear()
        self.resident_bytes = 0
        self.feature_payload_bytes = 0

    def rebind(self, namespace):
        if not isinstance(namespace, FeatureNamespace):
            raise ValueError("Cache namespace must be concrete")
        if namespace != self.namespace:
            self.clear()
            self.namespace = namespace
            self.generation += 1

    def report(self):
        return {"resident_bytes": self.resident_bytes, "max_bytes": self.max_bytes,
                "feature_payload_bytes": self.feature_payload_bytes,
                "byte_accounting": "tensor_payload_plus_conservative_entry_storage",
                "entries": len(self._entries), "max_entries": self.max_entries,
                "hits": self.hits, "misses": self.misses, "evictions": self.evictions,
                "generation": self.generation, "storage_device": "cpu",
                "namespace_sha256": self.namespace.sha256}


class FrozenEncoderGuard:
    """Cheap mutation/version checks, without rescanning frozen GPU weights.

    A loaded checkpoint/encoder fingerprint is still mandatory in namespace.
    Unsafe external .data writes bypassing PyTorch's version counter violate
    this contract; no API can prove their absence without rescanning weights.
    """
    def __init__(self, encoder, encoder_sha256, max_buffer_snapshot_bytes=1024*1024):
        self.encoder = encoder
        if type(encoder_sha256) is not str or _SHA256.fullmatch(encoder_sha256) is None:
            raise ValueError("Frozen encoder needs its admitted loaded identity digest")
        self.encoder_sha256 = encoder_sha256
        _positive_int(max_buffer_snapshot_bytes,"max_buffer_snapshot_bytes",zero=True)
        self._module_inventory = []
        self._tracked = []
        self._stochastic_layers = []
        import torch
        for module in encoder.modules():
            self._module_inventory.append((module, tuple(module._parameters.items()),
                tuple(module._buffers.items()), tuple(module._modules.items())))
            if isinstance(module,(torch.nn.Dropout,torch.nn.Dropout1d,torch.nn.Dropout2d,
                    torch.nn.Dropout3d,torch.nn.AlphaDropout,torch.nn.FeatureAlphaDropout,
                    torch.nn.modules.batchnorm._BatchNorm,torch.nn.RReLU)):
                self._stochastic_layers.append(module)
        for kind,items in (("parameter",encoder.named_parameters()),("buffer",encoder.named_buffers())):
            self._tracked.extend((kind,name,value) for name,value in items)
        self._inventory = self._read()
        self._buffer_values = {}
        used=0
        for kind,name,value in self._tracked:
            if kind != "buffer":continue
            size=value.numel()*value.element_size()
            if used+size <= max_buffer_snapshot_bytes:
                self._buffer_values[name]=value.detach().to(device="cpu").clone()
                used+=size
        self.buffer_snapshot_bytes=used

    def _read(self):
        import torch
        entries = []
        for module,params,buffers,children in self._module_inventory:
            for actual,recorded in ((module._parameters,params),(module._buffers,buffers),(module._modules,children)):
                if len(actual)!=len(recorded) or any(actual.get(name) is not value for name,value in recorded):
                    raise ValueError("Frozen encoder module/parameter/buffer registration changed")
        for kind,name,value in self._tracked:
            if value.requires_grad:
                raise ValueError("Trainable image encoders cannot reuse frozen features")
            entries.append((kind,name,id(value),tuple(value.shape),str(value.dtype),str(value.device),value._version))
        for layer in self._stochastic_layers:
            if isinstance(layer, (torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d, torch.nn.Dropout3d, torch.nn.AlphaDropout, torch.nn.FeatureAlphaDropout)) and layer.training and layer.p:
                raise ValueError("Active stochastic dropout cannot use frozen feature caching")
            if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm) and layer.training:
                raise ValueError("Training batch normalization cannot use frame-independent feature caching")
            if isinstance(layer, torch.nn.RReLU) and layer.training:
                raise ValueError("Active stochastic activation cannot use frozen feature caching")
        return tuple(entries)

    def check(self, namespace=None):
        if namespace is not None and namespace.encoder_sha256 != self.encoder_sha256:
            raise ValueError("Encoder guard and cache namespace have different weight identities")
        current=self._read()
        if current != self._inventory:
            for before,after,tracked in zip(self._inventory,current,self._tracked):
                if before == after:continue
                kind,name,value=tracked
                # Publication may copy the same immutable normalization
                # buffers. Only bounded CPU snapshots with exact values and
                # unchanged layout/registration may refresh their versions.
                if (before[:-1]!=after[:-1] or kind!="buffer" or name not in self._buffer_values or
                        not self.torch_equal_buffer(value,self._buffer_values[name])):
                    raise ValueError("Frozen encoder parameters/buffers changed; cached features are invalid")
            self._inventory=current

    @staticmethod
    def torch_equal_buffer(value,snapshot):
        import torch
        return torch.equal(value.detach().to(device="cpu"),snapshot)


def encode_feature_batch(encoder, images, frame_keys, image_keys, *, cache=None, chunk_size=4, guard=None):
    """Encode only selected [B,S,V,3,H,W] images, returning B,S features.

    The VAE must be deterministic, frozen and frame-independent. Native SD3
    posterior.mode consumes no sampling RNG. CUDA kernel rounding can still
    differ when image batch shapes/order change; tests use tolerances there.
    """
    import torch
    _positive_int(chunk_size, "chunk_size")
    if (not isinstance(images, torch.Tensor) or images.ndim != 6 or images.shape[2:4] != (len(image_keys), 3)
            or min(images.shape) < 1 or images.requires_grad):
        raise ValueError("Feature encoder needs detached selected images [B,S,V,3,H,W]")
    if len(set(image_keys)) != len(image_keys):
        raise ValueError("Camera names must be unique")
    if guard is None:
        if cache is not None:
            raise ValueError("Cached image encoding requires a frozen encoder mutation guard")
    else:
        guard.check(cache.namespace if cache is not None else None)
    batch, count = images.shape[:2]
    if cache is not None:
        if (not isinstance(frame_keys, (list, tuple)) or len(frame_keys) != batch or
                any(len(row) != count or any(len(views) != len(image_keys) for views in row) for row in frame_keys)):
            raise ValueError("Every selected image needs aligned actual frame/camera identity")
        for row in frame_keys:
            for views in row:
                if [key.camera for key in views] != list(image_keys):
                    raise ValueError("Feature cache camera identities differ from encoded view order")
    output = {}
    with torch.no_grad():
        for view, name in enumerate(image_keys):
            if cache is None:
                output[name + "_features"] = torch.cat([
                    encoder(images[:, start:start + chunk_size, view]).detach()
                    for start in range(0, count, chunk_size)], dim=1)
                continue
            features = [None] * (batch * count)
            for b in range(batch):
                for index in range(count):
                    offset = b * count + index
                    value = cache.get(frame_keys[b][index][view])
                    if value is not None:
                        features[offset] = value
            # Preserve the uncached [B, chunk_time, C, H, W] call geometry.
            # A partial hit re-encodes only that original selected time chunk,
            # including its hit members, instead of repacking arbitrary misses
            # as [B*chunk, 1]. It still never encodes an unselected raw frame.
            for first in range(0, count, chunk_size):
                end = min(first + chunk_size, count)
                if all(features[b * count + index] is not None
                       for b in range(batch) for index in range(first, end)):
                    continue
                encoded = encoder(images[:, first:end, view]).detach()
                if encoded.ndim < 3 or encoded.shape[:2] != (batch, end - first):
                    raise ValueError("Encoder features must preserve actual image batch/time axes")
                cpu = encoded.to(device="cpu")
                for b in range(batch):
                    for index in range(first, end):
                        value = cpu[b, index - first]
                        cache.put(frame_keys[b][index][view], value)
                        features[b * count + index] = value
            shape, dtype = tuple(features[0].shape), features[0].dtype
            if any(tuple(value.shape) != shape or value.dtype != dtype for value in features):
                raise ValueError("Cached and fresh encoder feature layout/dtype differs")
            output[name + "_features"] = torch.stack(features).reshape(batch, count, *shape).to(device=images.device)
    return output
