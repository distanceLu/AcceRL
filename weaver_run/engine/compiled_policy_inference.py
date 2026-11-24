"""CUDA Graph for a single live, unpadded OpenVLA rollout request.

The actor owns the original model. Graphs reference its existing parameter storage,
which the publication path updates in place. Sampling, transport and publication
are outside capture. Each instance retains at most one shape and one graph.
"""
from collections import Counter
import time

import torch


def unpadded_action_layout(request, model, prepared_shape):
    labels, attention = request['labels'], request['attention_mask']
    count = model.real_action_count
    if (labels.device.type != 'cpu' or attention.device.type != 'cpu'
            or labels.shape != prepared_shape or attention.shape != prepared_shape or labels.shape[1] < count + 2):
        return False
    # prepare_one_obs produces [ignored prompt, action labels, STOP]. Inspect the
    # source CPU tensors, before batching, without synchronizing the device.
    return bool((attention == 1).all() & (labels[:, :-count-1] == model.real_ignore_index).all()
                & (labels[:, -count-1:-1] > model.real_action_token_begin).all()
                & (labels[:, -1] == model.real_stop_index).all())


def static_action_outputs(logits, hidden, action_count, value_head):
    # The final two predictor positions correspond to STOP and the last unused
    # hidden state. Boolean indexing labels[:, 1:] in the eager path selects
    # the preceding fixed-length suffix when unpadded_action_layout holds.
    selected = slice(-action_count-2, -2)
    return logits[:, selected], value_head(hidden[:, selected].detach())


def _rng_state(device):
    result = [torch.get_rng_state()]
    if device.type == 'cuda':
        result.append(torch.cuda.get_rng_state(device))
    return result


def _same_rng(before, device):
    return all(torch.equal(old, new) for old, new in zip(before, _rng_state(device)))


def _geometry(batch):
    return tuple((name, value.shape, value.dtype, value.device)
                 for name, value in sorted(batch.items()))


def _storage(model):
    return tuple((id(tensor), tensor.data_ptr(), tensor.shape, tensor.dtype)
                 for tensors in (model.parameters(), model.buffers()) for tensor in tensors)


def _headroom(device):
    free, total = torch.cuda.mem_get_info(device)
    return free, max(8 * 2**30, int(total * .15))


class _CapturedForward:
    def __init__(self, model, batch, scope, mask_mode):
        self.scope = scope
        self.model = model
        device = batch['pixel_values'].device
        free, reserve = _headroom(device)
        self.reserve = reserve
        self.free_before = free
        reserved_before = torch.cuda.memory_reserved(device)
        if free < reserve + 2**30:
            raise MemoryError(f'capture free bytes {free} below reserve {reserve} plus 1 GiB')
        keys = ('pixel_values',) if scope == 'vision_projector' else ('input_ids', 'pixel_values', 'attention_mask')
        self.inputs = {key: batch[key].clone() for key in keys}

        def run():
            patches = model.real_action_vision(self.inputs['pixel_values'])
            if scope == 'vision_projector':
                return patches
            return model.real_action_from_patches(self.inputs['input_ids'],
                                                  patches, self.inputs['attention_mask'] if mask_mode == 'original' else None)

        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                run()
        stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.output = run()
        stream.synchronize()
        free, reserve = _headroom(device)
        self.free_after = free
        self.reserved_growth = torch.cuda.memory_reserved(device) - reserved_before
        if free < reserve:
            raise MemoryError(f'retained graph free bytes {free} below reserve {reserve}')

    def replay(self, batch):
        for key, tensor in self.inputs.items():
            # Eligibility checks the CPU source labels and padding every time.
            # Their layout and the all-valid attention mask are static.
            if key in ('pixel_values', 'input_ids'):
                tensor.copy_(batch[key])
        self.graph.replay()
        if self.scope == 'vision_projector':
            return self.model.real_action_from_patches(batch['input_ids'], self.output, batch['attention_mask'])
        # A second call must not overwrite a previously returned result.
        return tuple(output.clone() for output in self.output)


class CompiledPolicyInference:
    def __init__(self, model, emit):
        self.model = model
        self.emit = emit
        self.graph = self.geometry = self.storage = None
        self.seen_version = self.retry_at = None
        self.validated = []
        self.fallbacks, self.replays = Counter(), Counter()
        self.capture_failures = 0
        self.last_failure = None
        self._scope = 'full_logits_value'
        self._mask_mode = 'original'

    def summary(self):
        return dict(backend='cuda_graph', scope=getattr(self.graph, 'scope', None),
                    mask_mode=self._mask_mode if self.graph is not None else None,
                    graph_replays=sum(self.replays.values()), replays_by_scope=dict(self.replays),
                    validated_versions=list(self.validated), fallback_counts=dict(self.fallbacks),
                    capture_failures=self.capture_failures, last_failure=self.last_failure)

    def _unsupported(self, batch, original):
        if batch['input_ids'].shape[0] != 1:
            return 'batch_not_one'
        if batch['pixel_values'].device.type != 'cuda':
            return 'non_cuda'
        if self.model.training or not torch.is_inference_mode_enabled():
            return 'not_eval_inference'
        if set(batch) != {'input_ids', 'labels', 'attention_mask', 'pixel_values'}:
            return 'unsupported_input_fields'
        if (self.model.cfg.use_proprio or self.model.cfg.use_film
                or self.model.cfg.use_diffusion or self.model.cfg.use_l1_regression):
            return 'unsupported_model_config'
        if self.geometry is not None and _geometry(batch) != self.geometry:
            return 'different_static_shape'
        if (batch['input_ids'].shape != batch['labels'].shape
                or batch['input_ids'].shape != batch['attention_mask'].shape):
            return 'invalid_input_geometry'
        if original is None or not unpadded_action_layout(original, self.model, batch['input_ids'].shape):
            return 'padded_or_invalid_action_layout'
        return None

    def _fallback(self, batch, version, reason, reference=None):
        self.fallbacks[reason] += 1
        if self.fallbacks[reason] == 1 or self.fallbacks[reason] % 256 == 0:
            self.emit('compiled_policy_fallback', dict(version=version, reason=reason,
                      count=self.fallbacks[reason], last_failure=self.last_failure))
        return self.model(batch) if reference is None else reference

    def _capture(self, batch):
        return _CapturedForward(self.model, batch, self._scope, self._mask_mode)

    def _failure(self, exc, batch, version):
        self.graph = None
        self.geometry = self.storage = None
        self.capture_failures += 1
        self.last_failure = f'{type(exc).__name__}: {exc}'[:600]
        self.emit('compiled_policy_capture_failed', dict(version=version, scope=self._scope, mask_mode=self._mask_mode,
                  error=self.last_failure))
        device = batch['pixel_values'].device
        if device.type == 'cuda':
            # A damaged CUDA context cannot safely serve an eager fallback.
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()

    def __call__(self, batch, published_version, *, original=None):
        version = int(published_version)
        reason = self._unsupported(batch, original)
        if reason:
            return self._fallback(batch, version, reason)
        new_version = version != self.seen_version
        self.seen_version = version
        if new_version and self.graph is not None and _storage(self.model) != self.storage:
            self.graph = self.geometry = self.storage = None
            self.emit('compiled_policy_storage_changed', dict(version=version))
        if self.graph is None and self.retry_at is not None and version < self.retry_at:
            return self._fallback(batch, version, 'retry_deferred')
        check = self.graph is None or (new_version and
                (len(self.validated) < 2 or version // 32 > self.validated[-1] // 32))
        if not check:
            try:
                output = self.graph.replay(batch)
            except (RuntimeError, MemoryError) as exc:
                self._failure(exc, batch, version)
                self.retry_at = version + 1
                return self._fallback(batch, version, 'replay_failed')
            self.replays[getattr(self.graph, 'scope', self._scope)] += 1
            return output

        device = batch['pixel_values'].device
        devices = [device] if device.type == 'cuda' else []
        previous_rng = _rng_state(device)
        reference = self.model(batch)
        if not _same_rng(previous_rng, device):
            self.graph = self.geometry = self.storage = None
            self.retry_at = version + 32
            return self._fallback(batch, version, 'stochastic_eager_forward', reference)
        with torch.random.fork_rng(devices=devices):
            # Some HF attention backends test a CUDA scalar while processing an
            # explicit padding mask. Only try the elided form for a verified
            # unpadded B=1 and only serve it on exact numerical agreement.
            candidates = [('full_logits_value', 'original'), ('full_logits_value', 'elided_all_valid'),
                          ('vision_projector', 'original')] if self.graph is None else [(self._scope, self._mask_mode)]
            for scope, mask_mode in candidates:
                self._scope, self._mask_mode = scope, mask_mode
                try:
                    if self.graph is None:
                        started = time.monotonic()
                        self.graph = self._capture(batch)
                        self.geometry, self.storage = _geometry(batch), _storage(self.model)
                        self.emit('compiled_policy_capture', dict(version=version, backend='cuda_graph',
                                  scope=scope, mask_mode=mask_mode, shape=list(batch['pixel_values'].shape),
                                  sequence_length=batch['input_ids'].shape[1], elapsed_s=time.monotonic()-started,
                                  free_before_bytes=getattr(self.graph, 'free_before', None),
                                  free_after_bytes=getattr(self.graph, 'free_after', None),
                                  reserve_bytes=getattr(self.graph, 'reserve', None),
                                  allocator_reserved_delta_bytes=getattr(self.graph, 'reserved_growth', None)))
                    output = self.graph.replay(batch)
                    matched = all(a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
                                  and bool(torch.isfinite(b).all()) for a, b in zip(reference, output))
                    unchanged_rng = _same_rng(previous_rng, device)
                    self.emit('compiled_policy_validation', dict(version=version, scope=scope, mask_mode=mask_mode,
                              matched=(matched and unchanged_rng), unchanged_rng=unchanged_rng,
                              max_abs_logits=float((reference[0].float()-output[0].float()).abs().max()),
                              max_abs_value=float((reference[1].float()-output[1].float()).abs().max())))
                    if not matched:
                        raise ValueError('graph differs from current eager logits/value')
                    if not unchanged_rng:
                        raise ValueError('graph capture/replay consumed RNG')
                except (RuntimeError, MemoryError, ValueError) as exc:
                    self._failure(exc, batch, version)
                    if isinstance(exc, (torch.cuda.OutOfMemoryError, MemoryError)):
                        break
                    continue
                if version not in self.validated:
                    self.validated.append(version)
                self.retry_at = None
                self.replays[scope] += 1
                return output
        self.retry_at = version + 32
        return self._fallback(batch, version, 'no_verified_graph', reference)
