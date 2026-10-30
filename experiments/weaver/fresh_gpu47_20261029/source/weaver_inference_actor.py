"""Ray-independent actor factory for WEAVER's queued inference service.

The caller supplies its existing colocated transport mixin/communication base
and applies ray.remote. Each dispatch currently executes one trajectory and is
recorded as actual_batch=1; queue depth is not represented as GPU batching.
"""
from __future__ import annotations

import asyncio
import copy
import math
import time
import traceback

from world_backends.runtime import WeaverActorOperations, load_backend_config


def detached_reply_error(error):
    """Do not hand a live worker coroutine's traceback to another Task.

    Consumers such as unittest/Ray may clear received traceback frames. If the
    exact exception still references this long-lived background coroutine,
    clearing it can close the suspended coroutine on Python 3.11. Keep the
    original error for server telemetry and deliver a separate traceback-free
    exception with the original class/message whenever it supports copying.
    """
    try:
        delivered = copy.copy(error)
        if delivered is error:
            raise TypeError('Exception copy retained the original identity')
    except Exception:
        delivered = RuntimeError(type(error).__name__ + ': ' + str(error))
    delivered.__traceback__ = None
    delivered.__cause__ = None
    delivered.__context__ = None
    delivered.origin_traceback = ''.join(traceback.format_exception(type(error), error, error.__traceback__))
    return delivered


def make_weaver_inference_actor(mixin, communication_base, runtime):
    class WeaverWorldInferenceActor(mixin, communication_base):
        def __init__(self, actor_id, stats_actor, args, inference_batch, inference_timeout_ms):
            runtime["setup_debugger"]("weaver_world_inference", actor_id)
            self.actor_id, self.stats_actor = actor_id, stats_actor
            self.world_backend_config = load_backend_config(args)
            self.world_backend_config.validate_admission()
            self.world_frozen = not self.world_backend_config.world_trainable
            options = self.world_backend_config.backend_options
            if self.world_backend_config.backend != "weaver":
                raise ValueError("WeaverWorldInferenceActor requires backend=weaver")
            if self.world_frozen:
                # Reserving Ray's named card resource does not require loading
                # its incompatible Torch/Transformers stack on that GPU. Only
                # the owned foreground child initializes CUDA.
                self.cpu_gpu_bridge = True
                from weaver_subprocess import WeaverProcessClient, WeaverRemoteOperations
                factory = runtime.get("_weaver_process_client", WeaverProcessClient)
                self.worker_client = factory(self.world_backend_config.to_dict(), actor_id)
                self.gpu_binding = self.worker_client.receipt['gpu']
                self.model = self.world_backend = None
                self.device = "cpu"
                self.operations = WeaverRemoteOperations(self.worker_client)
            else:
                super().__init__()
                self.gpu_binding = runtime["verify_bound_gpu"]()
                from world_backends.weaver import WeaverBackend
                loader = runtime.get("_weaver_backend_loader", WeaverBackend.load)
                use_bf16 = args.get("use_bf16", True) if isinstance(args, dict) else getattr(args, "use_bf16", True)
                dtype = "bfloat16" if use_bf16 else "float32"
                if options.get("dtype", dtype) != dtype:
                    raise ValueError("WEAVER inference dtype must match the trainer's use_bf16 setting")
                self.world_backend = loader(options["source_path"], options["checkpoint_dir"],
                    device=options.get("device", "cuda"), dtype=dtype,
                    config=self.world_backend_config, for_training=False)
                self.model = self.world_backend.model
                self.device = self.world_backend.device
                self.operations = WeaverActorOperations(self.world_backend, self.world_backend_config, 0)
            self.capture_model_version = self.model_version = 0
            # Satisfy legacy transport diagnostics while publishing the selected
            # source snapshot directly. Inference owns no EMA shadow or backup.
            self.ema_decay, self.ema_initialized, self.ema_updates = 0.0, False, 0
            self.batch_size = int(inference_batch)
            self.timeout_sec = float(inference_timeout_ms) / 1000.0
            if self.batch_size < 1 or self.timeout_sec < 0:
                raise ValueError("Inference queue size must be positive and timeout nonnegative")
            self.requests, self.promises = [], []
            self.last_process_time = time.time()
            self._bg_task = asyncio.get_event_loop().create_task(self._loop())
            self._bg_task.add_done_callback(self._on_bg_task_done)

        def _sync_publication(self):
            version = int(self.capture_model_version)
            if version != self.operations.published_version:
                self.operations.set_published_version(version)

        def get_frozen_world_state(self):
            """Initial strict-load receipt; no phantom optimizer or publication."""
            if not self.world_frozen:
                raise RuntimeError("This receipt requires a frozen World")
            receipt = dict(self.worker_client.receipt)
            receipt["bridge_cpu_only"] = True
            self.telemetry.event("weaver_bridge_verified", model_version=0, details=receipt)
            self.telemetry.flush()
            return receipt

        def apply_rank_weights(self, *args, **kwargs):
            if self.world_frozen:
                raise RuntimeError("Frozen WEAVER cannot receive training weight publications")
            return super().apply_rank_weights(*args, **kwargs)

        def receive_and_update_weights(self, *args, **kwargs):
            if self.world_frozen:
                raise RuntimeError("Frozen WEAVER cannot receive training weight publications")
            return super().receive_and_update_weights(*args, **kwargs)

        def get_context_geometry(self):
            return self.operations.get_context_geometry()

        def init_rollout(self, real_context, published_version=None):
            self._check_inference_health()
            self._sync_publication()
            return self.operations.init_rollout(real_context, published_version)

        async def request(self, state, actions, instruction=None):
            self._check_inference_health()
            future = asyncio.get_event_loop().create_future()
            self.requests.append((state, actions, instruction))
            self.promises.append(future)
            return await future

        def _on_bg_task_done(self, task):
            if task.cancelled():
                return
            error = task.exception()
            if error is not None:
                self._fail_pending_requests(error)

        async def _loop(self):
            while True:
                if not self.requests:
                    await asyncio.sleep(0.0005)
                    continue
                # Do not wait to assemble a fictitious batch. The model adapter
                # currently supports one rollout per call, including partial H.
                request, promise = self.requests.pop(0), self.promises.pop(0)
                capture = runtime["batch_begin"](self, [request], [promise])
                started = time.time()
                stage = runtime["stage_begin"](self, "isolated_gpu_rpc" if self.world_frozen else "gpu_forward_and_copy",
                                               actual_batch=1, cuda=not self.world_frozen)
                stage_closed = batch_closed = False
                try:
                    self._sync_publication()
                    result = self.operations.request(*request)
                    runtime["stage_end"](self, stage)
                    stage_closed = True
                    runtime["batch_end"](self, capture)
                    batch_closed = True
                    if not promise.done():
                        promise.set_result(result)
                    self.last_process_time = time.time()
                    if not self.world_frozen:
                        runtime["_trim_cuda_cache"]("idle")
                    self.stats_actor.add_timing_metric.remote("Inference/weaver_world_loop_time_s", self.last_process_time - started)
                except BaseException as error:
                    if not stage_closed:
                        runtime["stage_end"](self, stage, error=error)
                    if not batch_closed:
                        runtime["batch_end"](self, capture, error=error)
                    if not promise.done():
                        promise.set_exception(detached_reply_error(error))
                    if "Stale WEAVER rollout/cache version" not in str(error):
                        self._fail_pending_requests(error)
                        raise
                # Allow weight publication/health RPCs between completed chunks.
                await asyncio.sleep(0)

        def evaluate_backend_prediction_metrics(self, backend_samples, seed=0):
            """Actual held-out pixel/state error, with deterministic noise scope.

            This neither substitutes for real-environment success evaluation nor
            claims to establish probability calibration or long-horizon quality.
            """
            self._check_inference_health()
            if self.world_frozen:
                raise RuntimeError("Frozen WEAVER uses its separately audited offline evaluation; no online re-evaluation is scheduled")
            self._sync_publication()
            backend = self.world_backend
            torch = backend.torch
            if not backend_samples:
                raise ValueError("Held-out WEAVER evaluation needs complete real trajectory windows")
            device = backend.device
            devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
            image_errors, state_errors = [], []
            count, frames = 0, 0
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(seed))
                for sample in backend_samples:
                    raw, images, _, _ = backend._raw_sample(sample, backend.training_window_frames, True)
                    n, horizon = backend.required_context_frames, backend.horizon
                    total = n + horizon
                    sample_frames = len(raw.get("obs", raw.get("observations")))
                    start = sample_frames - total
                    context = dict(raw)
                    for key in ("obs", "observations", "states", "actions", "rewards", "reward", "success", "done", "terminated", "truncated", "mask", "valid_mask", "action_mask", "dt"):
                        if key in context and context[key] is not None:
                            context[key] = context[key][start:start + n]
                    state = self.operations.init_rollout(context)
                    future_actions = raw["actions"][start + n:start + total]
                    prediction = self.operations.request(state, future_actions, raw["instruction"])
                    predicted_images = torch.as_tensor(prediction["observations"], device=device).float().add(1).mul(.5)
                    target_images = images[start + n:start + total].float()
                    errors = (predicted_images - target_images).square().flatten(1).mean(1)
                    target_states = torch.as_tensor(raw["states"][start + n:start + total], device=device).float()
                    predicted_states = torch.as_tensor(prediction["predicted_states"], device=device).float()
                    state_error = (predicted_states - target_states).square().mean(1)
                    image_errors.extend(errors.cpu().tolist())
                    state_errors.extend(state_error.cpu().tolist())
                    count += 1
                    frames += horizon
            image_mse = sum(image_errors) / len(image_errors)
            return {"backend": "weaver", "served_version": self.operations.published_version,
                    "windows": count, "predicted_frames": frames,
                    "image_mse": image_mse,
                    "image_psnr_db": -10 * math.log10(max(image_mse, 1e-12)),
                    "state_mse": sum(state_errors) / len(state_errors),
                    "image_value_range": "zero_one", "seed": int(seed),
                    "evaluation": "heldout_fixed_window", "real_environment_success": None}

    return WeaverWorldInferenceActor
