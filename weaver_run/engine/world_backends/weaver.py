"""WEAVER adapter using the audited upstream model and its native objectives.

Imports are lazy; importing this module neither allocates a model nor downloads
encoders. CPU test injection is explicit through ``model=``. Production ``load``
only constructs the checked upstream implementation and strictly loads weights.
"""
from __future__ import annotations

from contextlib import nullcontext
from collections import OrderedDict
from dataclasses import fields, is_dataclass
import importlib
import json
from pathlib import Path
import sys
import threading

from weaver_checkpoint import (PINNED_COMMIT, WeaverAdmissionError, admit_metadata,
    audit_state_dict, choose_checkpoint, environment_report, validate_normalization, verify_source)

_LOAD_LOCK = threading.RLock()


def _mapping(value):
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return {field.name: getattr(value, field.name) for field in fields(value)}
    return dict(value)


def _config_mapping(value):
    result = _mapping(value)
    options = result.pop("backend_options", {})
    overlap = set(options) & set(result)
    if overlap:
        raise WeaverAdmissionError(f"backend_options cannot override core configuration fields: {sorted(overlap)}")
    return {**result, **options}


def _cpu(value):
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if hasattr(value, "detach"):
        return value.detach().cpu()
    return value


class _NoEMA:
    """Inference and online-only training need no duplicate trainable snapshot."""
    def __init__(self, *args, **kwargs):
        pass
    def to(self, *args, **kwargs):
        return self
    def state_dict(self):
        return {}
    def update(self):
        raise RuntimeError("EMA is disabled: publication_weights is online")


def _make_training_module(torch, upstream, normalization):
    class WeaverTrainingModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.upstream = upstream
            for kind in ("state", "actions"):
                for field in ("mean", "std"):
                    self.register_buffer(f"{kind}_{field}", torch.tensor(normalization[kind][field], dtype=torch.float32))

        def forward(self, prepared):
            # Goes through the distributed engine's actual Module.__call__.
            # No detached scalar, replacement objective, or trainer bypass.
            controller=getattr(self.upstream,'native_reward_controller',None)
            if controller is not None and controller.dense:
                arguments=dict(prepared)
                mask=arguments.pop('gt_reward_mask',None)
                controller.set_micro_targets(arguments['gt_rewards'],mask)
                try:
                    return self.upstream(**arguments)
                finally:
                    controller.clear_micro_targets()
            return self.upstream(**prepared)
    return WeaverTrainingModule()


class WeaverBackend:
    name = "weaver"

    def __init__(self, config, model=None):
        self.config = _config_mapping(config)
        self.schema = dict(self.config.get("schema", {}))
        self.metadata = dict(self.config.get("checkpoint_metadata", {}))
        admit_metadata(self.config, self.metadata)
        if model is None:
            raise WeaverAdmissionError("Use WeaverBackend.load(source_path, checkpoint_dir, device, dtype, config=...) for real WEAVER weights; model= is explicit test injection")
        import torch
        self.torch = torch
        self.world_trainable = bool(self.config.get("world_trainable", True))
        self.native_score = self.config.get('reward_semantics') == 'delta_native_score'
        if 'fresh' in self.config:
            from fresh_config import FreshConfig
            FreshConfig.from_mapping(self.config['fresh']).validate_reward_binding(self.config.get('native_reward_head'))
        self.upstream_model = model
        stats = self.config.get("normalization", self.metadata.get("normalization", {}))
        self.normalization = validate_normalization(stats, self.schema)
        self.model = _make_training_module(torch, model, self.normalization)
        self.action_dim, self.state_dim = int(self.schema["action_dim"]), int(self.schema["state_dim"])
        self.image_keys = list(self.schema["image_keys"])
        self.n_history = int(getattr(model, "_n_history", self.config.get("n_history", 2)))
        self.horizon = int(getattr(model, "_n_horizon", self.config.get("horizon", 8)))
        self.n_memory_frames = int(getattr(model, "_n_memory_frames", self.config.get("n_memory_frames", 0)))
        self.t_memory = int(getattr(model, "_t_memory", self.config.get("t_memory", 1)))
        if min(self.n_history, self.horizon, self.t_memory) < 1 or self.n_memory_frames < 0:
            raise WeaverAdmissionError("history/horizon/memory stride must be positive and memory count nonnegative")
        self.required_context_frames = self.n_history + self.n_memory_frames * self.t_memory
        self.training_window_frames = self.required_context_frames + self.horizon
        self.published_version = None
        self.publication_weights = self.config.get("publication_weights", "online")
        if self.publication_weights not in ("online", "ema"):
            raise WeaverAdmissionError("publication_weights must be explicitly online or ema")
        self.critic_mode = self.config.get("critic_mode", "disabled")
        if self.critic_mode not in ("disabled", "train_unused"):
            raise WeaverAdmissionError("critic_mode must be disabled or train_unused; the policy value head remains authoritative")
        if self.critic_mode == "disabled":
            self.upstream_model._critic_loss_coeff = 0.0
        self.reward_training = bool(self.config.get("train_reward_head", self.config.get("reward_provider") == "world_head"))
        self.native_reward_controller=None
        if self.native_score:
            from weaver_native_reward import install_native_reward_head
            self.native_reward_controller=install_native_reward_head(model,self.config['native_reward_head'])
            model.native_reward_controller=self.native_reward_controller
            model._rm_loss_coeff=1.0
        if self.config.get("reward_provider") == "world_head" and not self.reward_training:
            raise WeaverAdmissionError("world_head requires train_reward_head=true")
        self._ema_parameter_names = ["upstream." + n for n, p in model.named_parameters() if p.requires_grad]
        if self.critic_mode == "disabled":
            for parameter in model.critic.parameters():
                parameter.requires_grad_(False)
        if not self.reward_training:
            model._rm_loss_coeff = 0.0
            for parameter in model.rm.parameters():
                parameter.requires_grad_(False)
        success = self.metadata.get("success_head", {})
        if (self.config.get("reward_provider") == "world_head" and not self.native_score
                and self.metadata["terminal_head"]["source"] != "same_success"):
            raise WeaverAdmissionError("Pinned WEAVER has no separate trainable terminal_head; this adapter requires calibrated same_success semantics and matching real labels")
        if self.reward_training and success.get("output") == "logit" and len(success.get("training_target_values", [])) != 2:
            raise WeaverAdmissionError("Native MSE training of a logit success head requires two calibrated training_target_values for false/true labels")
        self.checkpoint_audit = None
        self.environment = None
        self.loaded_snapshot = False
        self.compute_dtype = getattr(torch, self.config.get("autocast_dtype", "float32"))
        from weaver_optimizations import optimization_options
        self.optimizations = optimization_options(self.config)
        self._feature_cache = self._encoder_guard = self._feature_namespace = None
        self._task_cache = OrderedDict()
        self._task_guard = None
        if not self.world_trainable:
            self.model.requires_grad_(False)
            self.model.eval()

    @classmethod
    def load(cls, source_path, checkpoint_dir, device="cpu", dtype="float32", *, config, for_training=False):
        cfg = _config_mapping(config)
        if for_training and not cfg.get("world_trainable", True):
            raise WeaverAdmissionError("Frozen WEAVER cannot construct an online training model")
        root = verify_source(source_path)
        ckpt_root = Path(checkpoint_dir).resolve()
        metadata_path = ckpt_root / cfg.get("metadata_file", "backend_metadata.json")
        if metadata_path.is_file():
            disk_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if cfg.get("checkpoint_metadata") and cfg["checkpoint_metadata"] != disk_metadata:
                raise WeaverAdmissionError("Requested checkpoint_metadata differs from backend_metadata.json")
            cfg["checkpoint_metadata"] = disk_metadata
        admit_metadata(cfg, cfg.get("checkpoint_metadata", {}))
        # Resolve a concrete weight file before imports or model allocation.
        weight_path = choose_checkpoint(ckpt_root, cfg.get("checkpoint_file"))
        report = environment_report()
        if report["missing"] or (report["incompatible"] and not cfg.get("allow_untested_dependency_versions", False)):
            raise WeaverAdmissionError("Current environment cannot load audited WEAVER (no environment changes were made): " + json.dumps(report, sort_keys=True))
        encoder_paths = cfg.get("encoder_paths", {})
        for key in ("image", "task"):
            if not Path(encoder_paths.get(key, "__missing__")).is_dir():
                raise WeaverAdmissionError(f"encoder_paths.{key} must name an existing local pretrained encoder directory; automatic downloads are disabled")
        with _LOAD_LOCK:
            existing = sys.modules.get("weaver")
            if existing is not None and not Path(existing.__file__).resolve().is_relative_to(root):
                raise WeaverAdmissionError("A different weaver package is already imported; use a clean worker with the declared source_path")
            sys.path.insert(0, str(root))
            try:
                import torch
                import yaml
                model_module = importlib.import_module("weaver.wm.model")
                encoders = importlib.import_module("weaver.wm.encoders")
                conf = importlib.import_module("weaver.utils.config")
                defaults = conf.load_config(root / "weaver/config.yaml")
                released = yaml.safe_load((ckpt_root / "config.yaml").read_text(encoding="utf-8"))
                released = released.get("defaults", released)
                merged = conf.merge_dicts(defaults, released)
                admit_metadata(cfg, cfg["checkpoint_metadata"], merged)
                ns = conf.dict_to_namespace(merged)
                im_encoder = encoders.SD3Encoder(model_name=str(Path(encoder_paths["image"]).resolve()), image_size=tuple(ns.dataset.image_size), spatial_size=ns.im_encoder.spatial_size, device=device)
                task_encoder = encoders.ClipEncoder(model_name=str(Path(encoder_paths["task"]).resolve()), device=device)
                original_ema = model_module.EMA
                if not (for_training and cfg.get("publication_weights", "online") == "ema"):
                    model_module.EMA = _NoEMA
                try:
                    upstream = model_module.WEAVER(img_keys=cfg["schema"]["image_keys"], im_encoder=im_encoder,
                        train_decoder=False, task_encoder=task_encoder, n_history=ns.n_history,
                        n_horizon=ns.horizon, config=ns.model, use_precomputed_features=True,
                        n_states=cfg["schema"]["state_dim"], n_actions=cfg["schema"]["action_dim"],
                        image_size=tuple(ns.dataset.image_size), device=device,
                        n_memory_frames=ns.n_memory_frames, t_memory=ns.t_memory, inference_config=ns.inference)
                finally:
                    model_module.EMA = original_ema
                if not bool(merged.get("use_compile", False)):
                    original_forward = getattr(upstream.wm.forward, "_torchdynamo_orig_callable", None)
                    if original_forward is not None:
                        upstream.wm.forward = original_forward.__get__(upstream.wm, type(upstream.wm))
                checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
                if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
                    raise WeaverAdmissionError("Checkpoint must contain the native WEAVER model state_dict")
                cleaned, audit = audit_state_dict(upstream.state_dict(), checkpoint["model"])
                upstream.load_state_dict(cleaned, strict=True)
                if cfg.get("publication_weights", "online") == "ema":
                    shadows = checkpoint.get("ema", {}).get("shadow_params")
                    params = [(n, p) for n, p in upstream.named_parameters() if p.requires_grad]
                    if shadows is None or len(shadows) != len(params):
                        raise WeaverAdmissionError("EMA checkpoint shadow_params do not cover every trainable parameter")
                    for (name, param), shadow in zip(params, shadows):
                        if tuple(param.shape) != tuple(shadow.shape):
                            raise WeaverAdmissionError(f"EMA shape mismatch for {name}: {tuple(param.shape)} vs {tuple(shadow.shape)}")
                    if for_training:
                        upstream.ema.load_state_dict(checkpoint["ema"])
                    else:
                        with torch.no_grad():
                            for (_, param), shadow in zip(params, shadows):
                                param.copy_(shadow)
                norm_path = ckpt_root / cfg.get("normalization_file", "norm_stats_relabel.json")
                cfg["normalization"] = json.loads(norm_path.read_text(encoding="utf-8"))
                backend = cls(cfg, model=upstream)
                resolved_dtype = getattr(torch, str(dtype).replace("torch.", "")) if isinstance(dtype, str) else dtype
                backend.model.to(device=device, dtype=resolved_dtype)
                upstream.device = str(device)
                upstream.ema.to(device)
                backend.model.train(for_training)
                if "autocast_dtype" not in cfg:
                    backend.compute_dtype = resolved_dtype
                backend.loaded_snapshot = not for_training
                backend.checkpoint_audit = audit
                backend.environment = report
                return backend
            except (ImportError, OSError) as exc:
                raise WeaverAdmissionError(f"WEAVER dependency or local encoder load failed; environment unchanged: {exc}") from exc
            finally:
                sys.path.remove(str(root))

    def capabilities(self):
        return {"backend": "weaver", "requires_state": True, "native_reward": True,
                "integrated_reward": self.config.get("reward_provider") == "world_head",
                "provides_success": self.config.get('reward_provider')=='world_head' and not self.native_score,
                "provides_terminal": self.config.get('reward_provider')=='world_head' and not self.native_score,
                "reward_semantics": self.config.get('reward_semantics','delta_success'),
                "variable_horizon": True, "history_owned_by_backend": True,
                "memory_owned_by_backend": True, "image_output": True,
                "training_window_frames": self.training_window_frames,
                "required_context_frames": self.required_context_frames}

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    def _tensor(self, value):
        return self.torch.as_tensor(value, device=self.device, dtype=self.dtype)

    def _normal(self, value, name):
        return (value - getattr(self.model, name + "_mean")) / getattr(self.model, name + "_std")

    def _denormal(self, value, name):
        return value * getattr(self.model, name + "_std") + getattr(self.model, name + "_mean")

    def _check_schema(self, sample):
        given = sample.get("schema")
        if given is None:
            raise WeaverAdmissionError("Every WEAVER sample/context must carry its explicit schema")
        for key in ("domain", "action_schema_id", "state_schema_id", "encoder_id", "normalizer_id", "action_dim", "state_dim", "alignment", "image_keys", "image_size", "image_value_range", "dt_seconds"):
            if given.get(key) != self.schema.get(key):
                raise WeaverAdmissionError(f"Sample schema.{key} does not match admitted WEAVER checkpoint")

    def _raw_sample(self, sample, min_frames, training):
        sample = _mapping(sample)
        if sample.get('context_padding') is not None:
            raise WeaverAdmissionError('Reset-padded FRESH contexts require the explicit CPU admission path')
        if training and self.native_score and sample.get('label_source') != 'libero_env_step':
            raise WeaverAdmissionError('Native reward training accepts only measured per-frame simulator success labels')
        self._check_schema(sample)
        obs = self._tensor(sample.get("obs", sample.get("observations")))
        states = sample.get("states")
        if states is None:
            raise WeaverAdmissionError("WEAVER requires measured state sequence; legacy image/action replay cannot supply it")
        states, actions = self._tensor(states), self._tensor(sample["actions"])
        if obs.ndim != 5 or obs.shape[1] != len(self.image_keys) or obs.shape[2] != 3:
            raise WeaverAdmissionError("obs must be (T, views, 3, height, width) in declared view order")
        count = obs.shape[0]
        if count < min_frames or states.shape != (count, self.state_dim) or actions.shape != (count, self.action_dim):
            raise WeaverAdmissionError(f"Need >= {min_frames} aligned frames; obs={tuple(obs.shape)}, states={tuple(states.shape)}, actions={tuple(actions.shape)}")
        if not all(self.torch.isfinite(value).all().item() for value in (obs, states, actions)):
            raise WeaverAdmissionError("Observation/state/action values must be finite")
        low, high = (-1, 1) if self.schema["image_value_range"] == "minus_one_one" else (0, 1)
        if obs.min().item() < low - 1e-5 or obs.max().item() > high + 1e-5:
            raise WeaverAdmissionError("Observation range disagrees with schema.image_value_range")
        if low == -1:
            obs = (obs + 1) * 0.5
        height, width = self.schema["image_size"]
        if tuple(obs.shape[-2:]) != (height, width):
            obs = self.torch.nn.functional.interpolate(obs.flatten(0, 1), size=(height, width), mode="bilinear", align_corners=False).reshape(count, len(self.image_keys), 3, height, width)
        dt = sample.get("dt", sample.get("dt_seconds"))
        if dt is None:
            raise WeaverAdmissionError("WEAVER sample needs explicit dt or dt_seconds")
        # Time-scale admission uses full precision even when the network runs
        # in bf16; rounding a mismatched control period must not admit it.
        intervals = self.torch.as_tensor(dt, device=self.device, dtype=self.torch.float64)
        if intervals.ndim == 0:
            intervals = intervals.expand(count)
        if intervals.shape != (count,) or not self.torch.isfinite(intervals).all():
            raise WeaverAdmissionError("dt must be a finite per-frame vector or explicit uniform interval")
        action_mask = sample.get("action_mask")
        if action_mask is None or len(action_mask) != count:
            raise WeaverAdmissionError("action_mask must identify each actual incoming action and the reset sentinel")
        action_mask = self.torch.as_tensor(action_mask, device=self.device, dtype=self.torch.bool)
        if not action_mask[1:].all() or (intervals[action_mask] <= 0).any() or (intervals[~action_mask] != 0).any():
            raise WeaverAdmissionError("Only the first reset action may be masked with dt=0; actual transitions require positive dt")
        expected_dt = self.schema.get("dt_seconds", self.metadata.get("dt_seconds"))
        if expected_dt is None or not self.torch.allclose(intervals[action_mask], self.torch.full_like(intervals[action_mask], float(expected_dt)), atol=1e-6, rtol=1e-4):
            raise WeaverAdmissionError("Frame dt differs from checkpoint dt_seconds; resample controls explicitly")
        if training:
            valid = sample.get("valid_mask", sample.get("mask"))
            if valid is None or len(valid) != count or not self.torch.as_tensor(valid).bool().all():
                raise WeaverAdmissionError("Native WEAVER loss has no padding mask; training windows must contain only valid frames")
            terminals = sample.get("terminated", sample.get("done"))
            truncated = sample.get("truncated")
            if terminals is None or truncated is None or len(terminals) != count or len(truncated) != count:
                raise WeaverAdmissionError("Frame-aligned terminated and truncated labels are required")
            if self.torch.as_tensor(terminals[:-1]).bool().any() or self.torch.as_tensor(truncated[:-1]).bool().any():
                raise WeaverAdmissionError("Training window crosses an episode boundary")
        return sample, obs, self._normal(states, "state"), self._normal(actions, "actions")

    def _encoded(self, obs, states):
        chunk_size = int(self.config.get("encode_chunk_size", 4))
        if chunk_size < 1:
            raise WeaverAdmissionError("encode_chunk_size must be positive")
        with self.torch.no_grad():
            data = {key + "_features": self.torch.cat([
                        self.upstream_model.im_encoder(obs[:, start:start + chunk_size, index]).detach()
                        for start in range(0, obs.shape[1], chunk_size)], dim=1)
                    for index, key in enumerate(self.image_keys)}
        data["states"] = states
        return data

    def _feature_resources(self):
        """Bind only this loaded frozen encoder; never persist cache across loads."""
        from weaver_feature_cache import (FeatureNamespace, FrozenFeatureCache,
            FrozenEncoderGuard, fingerprint)
        if self._encoder_guard is None:
            import os
            encoder = self.upstream_model.im_encoder
            # This is the admitted artifact identity + in-process loaded scope,
            # not a claim that GPU weights were independently rehashed here.
            identity = fingerprint({'encoder_id': self.schema['encoder_id'],
                'paths': self.config.get('encoder_paths', {}),
                'initial_bundle': self.config.get('initial_checkpoint_sha256',
                    self.config.get('frozen_checkpoint_sha256')),
                'loaded_encoder_scope': id(encoder)})
            run = os.environ.get('ACCERL_RUN_ID') or self.config.get('cache_run_id')
            if not run:
                run = 'process-local-%s-%s' % (os.getpid(), id(self))
            self._feature_namespace = FeatureNamespace(run, identity, fingerprint({
                'image_size': self.schema['image_size'], 'image_range': self.schema['image_value_range'],
                'normalization': 'canonical-f32-minus-one-one-inverse-v1',
                'resize': 'bilinear-align-corners-false', 'dtype': str(self.dtype),
                'compute_dtype': str(self.compute_dtype)}))
            self._encoder_guard = FrozenEncoderGuard(encoder, identity)
            if self.optimizations['feature_cache_max_bytes']:
                self._feature_cache = FrozenFeatureCache(self._feature_namespace,
                    max_bytes=self.optimizations['feature_cache_max_bytes'])
        self._encoder_guard.check(self._feature_namespace)

    def _cpu_row(self, sample, min_frames, training):
        from weaver_feature_cache import validate_cpu_window
        from fresh_config import FreshConfig
        try:
            return validate_cpu_window(_mapping(sample), self.schema, min_frames, training,
                require_real_labels=self.native_score,
                allow_context_padding=FreshConfig.from_mapping(self.config.get('fresh')).early_seed_context)
        except ValueError as error:
            raise WeaverAdmissionError(str(error)) from error

    def _selected_images(self, rows, indices):
        """Move/resize selected images only, keeping the old FP32 pixel math."""
        torch = self.torch
        images = []
        for row in rows:
            value = torch.as_tensor(row['observations'][indices], device=self.device)
            if value.dtype == torch.uint8:
                value = value.float().div(255.)
                if self.schema['image_value_range'] == 'minus_one_one':
                    value = value.mul(2.).sub(1.)
            else:
                value = value.to(dtype=self.dtype)
            if self.schema['image_value_range'] == 'minus_one_one':
                value = (value + 1) * .5
            elif self.schema['image_value_range'] == 'uint8':
                raise WeaverAdmissionError('Normalized WEAVER encoder input needs an admitted floating image range')
            height, width = self.schema['image_size']
            if tuple(value.shape[-2:]) != (height, width):
                value = torch.nn.functional.interpolate(value.flatten(0, 1), size=(height, width),
                    mode='bilinear', align_corners=False).reshape(len(indices), len(self.image_keys), 3, height, width)
            images.append(value)
        return torch.stack(images)

    def _encoded_cpu_rows(self, rows, indices, states, *, training=False):
        from weaver_feature_cache import build_frame_keys, encode_feature_batch
        self._feature_resources()
        keys = None
        cache = self._feature_cache
        # Float contexts remain supported, but cannot impersonate canonical
        # byte frame identities or hit a cache created from different pixels.
        if cache is not None:
            import numpy as np
            if all(row['observations'].dtype == np.uint8 for row in rows):
                keys = [build_frame_keys(row['raw'], self._feature_namespace, indices,
                    require_window=training) for row in rows]
            else:
                cache = None
        encoded = encode_feature_batch(self.upstream_model.im_encoder,
            self._selected_images(rows, indices), keys, self.image_keys, cache=cache,
            chunk_size=int(self.config.get('encode_chunk_size', 4)), guard=self._encoder_guard)
        encoded['states'] = states
        return encoded

    def _task_features(self, texts):
        limit = self.optimizations['task_cache_entries']
        if not limit or self.model.training:
            return self.upstream_model.encode_task({'text': texts}).detach()
        from weaver_feature_cache import FrozenEncoderGuard, fingerprint
        encoder = self.upstream_model.task_encoder
        if self._task_guard is None:
            self._task_guard = FrozenEncoderGuard(encoder, fingerprint({
                'encoder_id': self.schema['encoder_id'], 'loaded_task_scope': id(encoder)}))
        self._task_guard.check()
        values = []
        for text in texts:
            if not isinstance(text, str):
                raise WeaverAdmissionError('Task instruction must be text')
            value = self._task_cache.get(text)
            if value is None:
                value = self.upstream_model.encode_task({'text': [text]}).detach()
                # Both entries and instruction size are bounded. A malicious
                # arbitrarily large instruction never enters resident storage.
                if len(text.encode('utf-8')) <= 8192 and value.numel() * value.element_size() <= 1024 * 1024:
                    while len(self._task_cache) >= limit:
                        self._task_cache.popitem(last=False)
                    self._task_cache[text] = value
            else:
                self._task_cache.move_to_end(text)
            values.append(value)
        return self.torch.cat(values, dim=0)

    def _memory_indices(self, current):
        return [current - (self.n_memory_frames - index) * self.t_memory for index in range(self.n_memory_frames)]

    def prepare_train_batch(self, episode_windows):
        if not self.world_trainable:
            raise WeaverAdmissionError("Frozen WEAVER cannot prepare training batches")
        if isinstance(episode_windows, dict) and "gt_rewards" in episode_windows:
            raise WeaverAdmissionError("prepare_train_batch requires raw validated trajectory windows; pass prepared data directly to model")
        if isinstance(episode_windows, dict) or is_dataclass(episode_windows):
            episode_windows = [episode_windows]
        if not episode_windows:
            raise WeaverAdmissionError("Cannot prepare an empty WEAVER batch")
        t = self.training_window_frames
        start = t - (self.n_history + self.horizon)
        if self.optimizations['validate_on_cpu']:
            from weaver_feature_cache import training_frame_selection
            cpu_rows = [self._cpu_row(row, t, True) for row in episode_windows]
            # Streaming protocol provides exactly 40 frames. Reject overlong
            # windows here rather than guessing their global frame identities.
            if any(row['frame_count'] != t for row in cpu_rows):
                raise WeaverAdmissionError('Optimized training requires the exact canonical window length')
            states = self.torch.stack([self._normal(self._tensor(r['states']), 'state') for r in cpu_rows])
            actions = self.torch.stack([self._normal(self._tensor(r['actions']), 'actions') for r in cpu_rows])
            if self.optimizations['selected_frame_encoding']:
                selection = training_frame_selection(t, self.n_history, self.horizon, self.n_memory_frames, self.t_memory)
            else:
                selection = {'selected_indices': list(range(t)), 'main_positions': list(range(start, t)),
                             'memory_positions': self._memory_indices(t - self.horizon - 1)}
            selected = selection['selected_indices']
            compact = self._encoded_cpu_rows(cpu_rows, selected, states[:, selected], training=True)
            # Restore the legacy latent storage and view geometry without
            # encoding the other raw images. Only actual selected positions
            # are ever read below; zero scaffold slots never enter native loss.
            all_obs = {'states': states}
            for key, value in compact.items():
                if key == 'states':
                    continue
                full = value.new_zeros(value.shape[0], t, *value.shape[2:])
                positions = self.torch.as_tensor(selected, device=value.device, dtype=self.torch.long)
                full.index_copy_(1, positions, value)
                all_obs[key] = full
            # Keep native image-key/state insertion order as well as strides:
            # sample_noise iterates this mapping and draws one tensor per key.
            all_obs = {key: all_obs[key] for key in compact}
            obs = {key: value[:, start:] for key, value in all_obs.items()}
            memory = ({key: value[:, self._memory_indices(t - self.horizon - 1)] for key, value in all_obs.items()}
                      if self.n_memory_frames else None)
            rows = [(r['raw'], None, None, None) for r in cpu_rows]
        else:
            rows = [self._raw_sample(row, t, True) for row in episode_windows]
            images = self.torch.stack([r[1][-t:] for r in rows])
            states = self.torch.stack([r[2][-t:] for r in rows])
            actions = self.torch.stack([r[3][-t:] for r in rows])
            all_obs = self._encoded(images, states)
            obs = {key: value[:, start:] for key, value in all_obs.items()}
            memory = ({key: value[:, self._memory_indices(t - self.horizon - 1)] for key, value in all_obs.items()}
                      if self.n_memory_frames else None)
        labels,masks = [],[]
        dense=bool(self.native_reward_controller and self.native_reward_controller.dense)
        for raw, _, _, _ in rows:
            if dense:
                from fresh_reward_targets import validate_reward_target_payload
                try:
                    rewards,mask,_=validate_reward_target_payload(raw,self.native_reward_controller.specification,
                                                               expected_count=t)
                except ValueError as error:
                    raise WeaverAdmissionError(str(error)) from error
                labels.append(self.torch.as_tensor(rewards[-(self.n_history+self.horizon):],
                    device=self.device,dtype=self.torch.float32))
                masks.append(self.torch.as_tensor(mask[-(self.n_history+self.horizon):],
                    device=self.device,dtype=self.torch.bool))
                continue
            rewards = raw.get("success") if self.config.get("reward_provider") == "world_head" else raw.get("rewards", raw.get("reward"))
            if self.config.get("reward_provider") == "world_head" and not self.native_score:
                terminals = raw.get("terminated", raw.get("done"))
                if rewards is None or not self.torch.equal(self.torch.as_tensor(rewards).bool(), self.torch.as_tensor(terminals).bool()):
                    raise WeaverAdmissionError("Calibrated same_success terminal head requires real success and terminated labels to agree on every frame")
            elif self.reward_training and not self.native_score:
                semantics = self.metadata.get("reward_head", {}).get("label_semantics")
                if not semantics or raw.get("reward_label_semantics") != semantics or "native_rewards" not in raw:
                    raise WeaverAdmissionError("Native WEAVER reward-head training requires native_rewards and matching reward_label_semantics; sparse environment rewards cannot substitute for progress labels")
                rewards = raw["native_rewards"]
            if rewards is None or len(rewards) != len(raw.get("obs", raw.get("observations"))):
                raise WeaverAdmissionError("Frame-aligned real reward/success labels required by native WEAVER loss")
            labels.append(self._tensor(rewards)[-(self.n_history + self.horizon):])
        gt_rewards = self.torch.stack(labels)
        if not self.torch.isfinite(gt_rewards).all():
            raise WeaverAdmissionError("Reward labels must be finite")
        if self.config.get("reward_provider") == "world_head" and not dense:
            if ((gt_rewards != 0) & (gt_rewards != 1)).any():
                raise WeaverAdmissionError("Integrated success labels must be real binary environment success")
            head = self.metadata["success_head"]
            if not self.native_score and head["output"] == "logit":
                lo, hi = head["training_target_values"]
                gt_rewards = lo + (hi - lo) * gt_rewards
        text = [r[0]["instruction"] for r in rows]
        with self.torch.no_grad():
            tasks = {"features": self.upstream_model.encode_task({"text": text}).detach()}
        prepared={"obs": obs, "actions": actions[:, start:], "tasks": tasks,
                "gt_rewards": gt_rewards, "memory": memory,
                "update_rm": self.reward_training or self.critic_mode == "train_unused"}
        if dense:
            prepared['gt_reward_mask']=self.torch.stack(masks)
        return prepared

    def compute_losses(self, prepared_batch):
        if not self.world_trainable:
            raise WeaverAdmissionError("Frozen WEAVER cannot compute online training losses")
        return self.model(prepared_batch)

    def set_published_version(self, version):
        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            raise WeaverAdmissionError("WEAVER published version must be a nonnegative integer")
        if self.published_version is not None and version < self.published_version:
            raise WeaverAdmissionError("Cannot roll WEAVER publication version backward")
        if version != self.published_version:
            self.published_version = version
            # Upstream creates KV caches within each generation; never reuse
            # weight-dependent memory tokens or task features across versions.
            cache = getattr(self.upstream_model, "_pyramid_schedule_cache", None)
            if cache is not None:
                cache.clear()

    def init_rollout(self, real_context, published_version):
        self.set_published_version(published_version)
        n = self.required_context_frames
        self.model.eval()
        cpu_row = None
        if self.optimizations['validate_on_cpu']:
            cpu_row = self._cpu_row(real_context, n, False)
            if cpu_row['frame_count'] != n:
                raise WeaverAdmissionError('Optimized seed requires its exact canonical context length')
            raw = cpu_row['raw']
            states = self._normal(self._tensor(cpu_row['states']), 'state')
            actions = self._normal(self._tensor(cpu_row['actions']), 'actions')
        else:
            raw, images, states, actions = self._raw_sample(real_context, n, False)
        with self.torch.no_grad(), self._autocast():
            encoded = (self._encoded_cpu_rows([cpu_row], list(range(n)), states.unsqueeze(0))
                       if cpu_row is not None else self._encoded(images[-n:].unsqueeze(0), states[-n:].unsqueeze(0)))
        latents = {key: encoded[key + "_features"] for key in self.image_keys}
        latents["states"] = encoded["states"]
        initial = {}
        if self.config.get("reward_provider") == "world_head":
            with self.torch.no_grad(), self._autocast():
                task = self._task_features([raw['instruction']])
                hist = {key: value[:, -self.n_history:] for key, value in latents.items()}
                score = self.upstream_model.rm(hist, actions[-self.n_history:].unsqueeze(0), task).float()[0, -1]
                probability = (score if self.native_score else score.sigmoid()
                    if self.metadata["success_head"]["output"] == "logit" else score)
                if not self.torch.isfinite(probability) or not 0 <= probability.item() <= 1:
                    raise RuntimeError("Calibrated success head returned invalid probabilities for real context")
                if self.native_score:
                    initial={'initial_reward_score':probability.item(),'initial_terminal':False,
                             'reward_semantics':'delta_native_score','reward_calibrated':False}
                else:
                    threshold = 0.5 if self.config.get("terminal_rule", "argmax") == "argmax" else float(self.config.get("terminal_threshold", 0.5))
                    initial = {"initial_success_probability": probability.item(),
                        "initial_terminal": probability.item() > threshold if self.config.get("terminal_rule", "argmax") == "argmax" else probability.item() >= threshold}
        state = {"backend": "weaver", "version": published_version, "latents": latents,
            "actions": actions[-n:].unsqueeze(0), "instruction": raw["instruction"],
            "request_lineage": raw.get("request_lineage", {}), "predicted_steps": 0,
            "encoder_id": self.schema["encoder_id"], "normalizer_id": self.schema["normalizer_id"], **initial}
        if raw.get('context_padding') is not None:
            state.update(context_padding=dict(raw['context_padding']),synthetic_history=True)
        return state if self.optimizations['resident_rollout_state'] else _cpu(state)

    def _autocast(self):
        if self.compute_dtype in (self.torch.float16, self.torch.bfloat16):
            return self.torch.autocast(device_type=self.device.type, dtype=self.compute_dtype)
        return nullcontext()

    def predict_chunk(self, opaque_state, action_chunk, published_version=None, requested_outputs=None):
        version = self.published_version if published_version is None else published_version
        if opaque_state.get("backend") != "weaver" or opaque_state.get("version") != version or version != self.published_version:
            raise WeaverAdmissionError("Stale WEAVER rollout/cache version; initialize from real context after publication")
        if any(opaque_state.get(key) != self.schema[key] for key in ("encoder_id", "normalizer_id")):
            raise WeaverAdmissionError("Rollout encoder/normalizer identity changed; initialize real context again")
        torch = self.torch
        actions = self._tensor(action_chunk)
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[-1] != self.action_dim or actions.shape[1] < 1:
            raise WeaverAdmissionError("action_chunk must be (H,A) or (1,H,A) with at least one valid action")
        if not torch.isfinite(actions).all():
            raise WeaverAdmissionError("Future actions must be finite")
        h = actions.shape[1]
        future_actions = self._normal(actions, "actions")
        archive = {k: self._tensor(v) for k, v in opaque_state["latents"].items()}
        history_actions = self._tensor(opaque_state["actions"])[:, -self.n_history:]
        context = {k: v[:, -self.n_history:] for k, v in archive.items()}
        self.model.eval()
        with torch.no_grad(), self._autocast():
            memory_tokens = None
            if self.n_memory_frames:
                indices = self._memory_indices(archive["states"].shape[1] - 1)
                memory = {key + "_features": archive[key][:, indices] for key in self.image_keys}
                memory["states"] = archive["states"][:, indices]
                memory_tokens = self.upstream_model.encode_memory_obs(memory)
            generate = self.upstream_model.generate_latent_rollouts_variable_horizon
            if self.optimizations['kv_cache']:
                from weaver_cached_inference import generate_variable_horizon_cached
                from functools import partial
                generate = partial(generate_variable_horizon_cached, self.upstream_model)
            xt = generate(
                context=context, future_actions=future_actions,
                future_states=torch.zeros(1, h, self.state_dim, device=self.device, dtype=self.dtype),
                history_actions=history_actions, memory_tokens=memory_tokens)
            expected = self.n_history + h
            if any(v.shape[1] != expected for v in xt.values()):
                raise RuntimeError("Upstream WEAVER returned an unexpected rollout length")
            future = {k: v[:, self.n_history:] for k, v in xt.items()}
            from weaver_cached_inference import decode_rollout_observations
            decode_mode = ('all' if requested_outputs is not None and 'all_observations' in requested_outputs
                           else self.optimizations['decode_mode'])
            observations = decode_rollout_observations(self.upstream_model, future, self.image_keys,
                chunk_size=int(self.config.get('decode_chunk_size', 4)),
                output_mode=decode_mode)
            result = {"observations": observations.float(), "predicted_states": self._denormal(future["states"], "state").float(),
                      "served_version": version, "encoder_id": self.schema["encoder_id"],
                      "normalizer_id": self.schema["normalizer_id"], "request_lineage": opaque_state.get("request_lineage", {}),
                      "valid_mask": torch.ones(1, h, dtype=torch.bool), "dt": self.schema.get("dt_seconds", self.metadata.get("dt_seconds"))}
            if decode_mode == 'last':
                result['observation_indices'] = [h - 1]
                result['prediction_horizon'] = h
            want_reward = self.config.get("reward_provider") == "world_head" or (requested_outputs is not None and "reward_predictions" in requested_outputs)
            if want_reward:
                task = self._task_features([opaque_state['instruction']])
                all_actions = torch.cat([history_actions, future_actions], dim=1)
                rewards = self.upstream_model.rm(xt, all_actions, task).float()[:, self.n_history:]
                result["reward_predictions"] = rewards
                if self.native_score:
                    if not torch.isfinite(rewards).all() or ((rewards<0)|(rewards>1)).any():
                        raise RuntimeError('Native bounded reward scores are invalid')
                    result['terminal_predictions']=torch.zeros_like(rewards,dtype=torch.bool)
                    result['reward_semantics']='delta_native_score'
                    result['reward_calibrated']=False
                elif self.config.get("reward_provider") == "world_head":
                    head = self.metadata["success_head"]
                    probabilities = rewards.sigmoid() if head["output"] == "logit" else rewards
                    if not torch.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
                        raise RuntimeError("Calibrated success head returned invalid probabilities; refusing silent clipping")
                    result["success_probabilities"] = probabilities
                    terminal = self.metadata["terminal_head"]
                    terminal_prob = probabilities
                    if terminal["source"] == "terminal_head":
                        terminal_prob = self.upstream_model.terminal_head(xt, all_actions, task).float()[:, self.n_history:]
                        if terminal.get("output") == "logit":
                            terminal_prob = terminal_prob.sigmoid()
                        if not torch.isfinite(terminal_prob).all() or ((terminal_prob < 0) | (terminal_prob > 1)).any():
                            raise RuntimeError("Calibrated terminal head returned invalid probabilities")
                    threshold = 0.5 if self.config.get("terminal_rule", "argmax") == "argmax" else float(self.config.get("terminal_threshold", 0.5))
                    result["terminal_predictions"] = terminal_prob > threshold if self.config.get("terminal_rule", "argmax") == "argmax" else terminal_prob >= threshold
            next_archive = {k: torch.cat([archive[k], future[k]], dim=1)[:, -self.required_context_frames:] for k in archive}
            next_actions = torch.cat([self._tensor(opaque_state["actions"]), future_actions], dim=1)[:, -self.required_context_frames:]
            result["next_state"] = {**opaque_state, "latents": next_archive, "actions": next_actions,
                                    "predicted_steps": opaque_state["predicted_steps"] + h}
        if self.optimizations['resident_rollout_state']:
            return {**_cpu({key: value for key, value in result.items() if key != 'next_state'}),
                    'next_state': result['next_state']}
        return _cpu(result)

    def publication_inventory(self):
        return {"parameter_names": [n for n, p in self.model.named_parameters() if p.requires_grad],
                "buffer_names": [n for n, _ in self.model.named_buffers()],
                "backend_identity": {"name": "weaver", "source_commit": PINNED_COMMIT,
                    "encoder_id": self.schema["encoder_id"], "normalizer_id": self.schema["normalizer_id"]},
                "bundle_metadata": {"publication_weights": self.publication_weights, "critic_mode": self.critic_mode,
                    "reward_provider": self.config.get("reward_provider", "external_vlm"),
                    "reward_semantics": self.config.get('reward_semantics','delta_success'),
                    "native_reward_head": self.config.get('native_reward_head'),
                    "success_calibration_id": self.metadata.get("success_head", {}).get("calibration_id"),
                    "terminal_calibration_id": self.metadata.get("terminal_head", {}).get("calibration_id")},
                "cache_version": "weaver-adapter-v1"}

    def publication_tensors(self):
        selected = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        if self.publication_weights == "ema" and not self.loaded_snapshot:
            ema = getattr(getattr(self.upstream_model, "ema", None), "ema", None)
            shadows = getattr(ema, "shadow_params", None)
            if shadows is None or len(shadows) != len(self._ema_parameter_names):
                raise WeaverAdmissionError("EMA publication requires a complete EMA shadow inventory")
            all_shadows = dict(zip(self._ema_parameter_names, shadows))
            selected = {name: all_shadows[name] for name in selected}
            for name, param in self.model.named_parameters():
                if name in selected and tuple(selected[name].shape) != tuple(param.shape):
                    raise WeaverAdmissionError(f"EMA publication shape mismatch: {name}")
        selected.update(dict(self.model.named_buffers()))
        return selected

    def update_ema(self):
        if self.publication_weights == "ema" and not self.loaded_snapshot:
            self.upstream_model.ema.update()
