"""Frozen-base FP8 linear execution via native PyTorch ``torch._scaled_mm``.

In OpenVLA LoRA RL fine-tuning, ~7.5B base Linear weights (Llama decoder,
DINOv2/SigLIP vision towers, multimodal projector) are frozen
(``requires_grad=False``), while only LoRA adapters, the slim ``lm_head``,
``value_head``, ``attn_pool``, and optional ``proprio_projector`` are trainable.

Replacing only those frozen base ``nn.Linear`` modules with ``FrozenFP8Linear``:
- Keeps all trainable parameters and CUDA IPC weight inventories byte-identical
  (quantized FP8 weights and scales are stored as plain Python tensor
  attributes, never registered as ``nn.Parameter`` or ``nn.Module`` buffers).
- Eliminates GPU BF16 copies of replaced frozen weights: inference retains only
  ``W_fp8`` (~50% weight VRAM reduction) while trainer retains ``W_fp8`` and
  ``W_t_fp8`` (~0% net weight VRAM change, zero runtime weight transpose).
- Leaves LoRA branches, ``lm_head``, ``value_head``, ZeRO-2 optimizer states,
  and PEFT ``save_pretrained`` completely in BF16/FP32.
- Pads ``M % 16 != 0`` inputs with static zero rows so Hopper FP8 Tensor Cores
  work seamlessly under both ``torch.compile`` and ``torch.cuda.CUDAGraph``.
- Runs a startup capability, numerical-parity, and CUDA-Graph self-check before
  converting the model; fails fast if no FP8 recipe passes verification.
"""
from __future__ import annotations

import os
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)
FP8_MAX = 448.0
MIN_SCALE = 1e-24
DEFAULT_MIN_FEATURES = 256
PROBE_MIN_COS_SIM = 0.995
PROBE_MAX_REL_ERR = 0.10
_SCALED_MM_OVERRIDE: Optional[Callable] = None


def _run_scaled_mm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    *,
    out_dtype: torch.dtype,
    use_fast_accum: bool,
    scaled_mm_fn: Optional[Callable] = None,
) -> torch.Tensor:
    op = scaled_mm_fn if scaled_mm_fn is not None else (_SCALED_MM_OVERRIDE or torch._scaled_mm)
    return op(
        mat_a,
        mat_b,
        scale_a=scale_a,
        scale_b=scale_b,
        bias=None,
        out_dtype=out_dtype,
        use_fast_accum=use_fast_accum,
    )


def fp8_base_linear_enabled(env=None) -> bool:
    environ = os.environ if env is None else env
    return environ.get("ACCERL_FP8_BASE_LINEAR") == "1"


def _quantize_fp8_rowwise(x_2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    amax = x_2d.abs().amax(dim=-1, keepdim=True).to(torch.float32)
    scale = (amax / FP8_MAX).clamp(min=MIN_SCALE)
    inv_scale = scale.reciprocal()
    q = (x_2d.to(torch.float32) * inv_scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return q.contiguous(), scale.contiguous()


def _quantize_fp8_tensorwise(x_2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    amax = x_2d.abs().amax().to(torch.float32)
    scale = (amax / FP8_MAX).clamp(min=MIN_SCALE).reshape(1)
    inv_scale = scale.reciprocal()
    q = (x_2d.to(torch.float32) * inv_scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return q.contiguous(), scale.contiguous()


def quantize_frozen_weight(weight_2d: torch.Tensor, scaling_mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D frozen weight matrix ``(out_dim, in_dim)`` to FP8.

    Returns ``(w_fp8, w_scale)`` where ``w_fp8`` is row-major contiguous
    ``(out_dim, in_dim)`` so ``w_fp8.t()`` is column-major ``(in_dim, out_dim)``
    as required by ``torch._scaled_mm``, and ``w_scale`` has shape
    ``(1, out_dim)`` for ``rowwise`` or ``(1,)`` for ``tensorwise``.
    """
    w = weight_2d.detach().contiguous()
    if scaling_mode == "rowwise":
        w_fp8, row_scale = _quantize_fp8_rowwise(w)
        return w_fp8, row_scale.reshape(1, w.shape[0]).contiguous()
    if scaling_mode == "tensorwise":
        return _quantize_fp8_tensorwise(w)
    raise ValueError(f"Unsupported FP8 scaling_mode: {scaling_mode!r}")


def _fp8_mm(
    x_2d: torch.Tensor,
    w_fp8: torch.Tensor,
    w_scale: torch.Tensor,
    *,
    scaling_mode: str,
    use_fast_accum: bool,
    out_dtype: torch.dtype,
    scaled_mm_fn: Optional[Callable] = None,
) -> torch.Tensor:
    m = x_2d.shape[0]
    if m == 0:
        return x_2d.new_empty((0, w_fp8.shape[0]), dtype=out_dtype)
    pad = (-m) % 16
    if pad:
        x_2d = F.pad(x_2d, (0, 0, 0, pad))
    if scaling_mode == "rowwise":
        x_fp8, x_scale = _quantize_fp8_rowwise(x_2d)
    else:
        x_fp8, x_scale = _quantize_fp8_tensorwise(x_2d)
    mm_dtype = out_dtype if out_dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
    # Always pass bias=None so PyTorch 2.12 Inductor can select Hopper TMA FP8 kernels.
    out = _run_scaled_mm(
        x_fp8,
        w_fp8.t(),
        x_scale,
        w_scale,
        out_dtype=mm_dtype,
        use_fast_accum=use_fast_accum,
        scaled_mm_fn=scaled_mm_fn,
    )
    if pad:
        out = out[:m]
    return out if out.dtype == out_dtype else out.to(out_dtype)


class _FrozenFP8LinearFn(torch.autograd.Function):
    """Stateless autograd bridge for frozen-weight FP8 forward & input-grad backward.

    Accepts only tensors and primitive flags (no Python callables) so Dynamo and
    AOTAutograd trace forward and backward with zero graph breaks.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w_fp8: torch.Tensor,
        w_scale: torch.Tensor,
        w_t_fp8: torch.Tensor,
        w_t_scale: torch.Tensor,
        bias: Optional[torch.Tensor],
        scaling_mode: str,
        use_fast_accum: bool,
    ) -> torch.Tensor:
        ctx.save_for_backward(w_t_fp8, w_t_scale)
        ctx.scaling_mode = scaling_mode
        ctx.use_fast_accum = use_fast_accum
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        out = _fp8_mm(
            x_2d,
            w_fp8,
            w_scale,
            scaling_mode=scaling_mode,
            use_fast_accum=use_fast_accum,
            out_dtype=x.dtype,
        )
        if bias is not None:
            out = out + bias.to(out.dtype)
        return out.reshape(*orig_shape[:-1], w_fp8.shape[0])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if not ctx.needs_input_grad[0]:
            return (None,) * 8
        w_t_fp8, w_t_scale = ctx.saved_tensors
        orig_shape = grad_output.shape
        dy_2d = grad_output.reshape(-1, orig_shape[-1])
        dx = _fp8_mm(
            dy_2d,
            w_t_fp8,
            w_t_scale,
            scaling_mode=ctx.scaling_mode,
            use_fast_accum=ctx.use_fast_accum,
            out_dtype=grad_output.dtype,
        )
        return (dx.reshape(*orig_shape[:-1], w_t_fp8.shape[0]),) + (None,) * 7


class _CustomFrozenFP8LinearFn(torch.autograd.Function):
    """Autograd bridge used only when a test injects a custom scaled_mm_fn."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w_fp8: torch.Tensor,
        w_scale: torch.Tensor,
        w_t_fp8: torch.Tensor,
        w_t_scale: torch.Tensor,
        bias: Optional[torch.Tensor],
        scaling_mode: str,
        use_fast_accum: bool,
        scaled_mm_fn: Callable,
    ) -> torch.Tensor:
        ctx.save_for_backward(w_t_fp8, w_t_scale)
        ctx.scaling_mode = scaling_mode
        ctx.use_fast_accum = use_fast_accum
        ctx.scaled_mm_fn = scaled_mm_fn
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        out = _fp8_mm(
            x_2d,
            w_fp8,
            w_scale,
            scaling_mode=scaling_mode,
            use_fast_accum=use_fast_accum,
            out_dtype=x.dtype,
            scaled_mm_fn=scaled_mm_fn,
        )
        if bias is not None:
            out = out + bias.to(out.dtype)
        return out.reshape(*orig_shape[:-1], w_fp8.shape[0])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if not ctx.needs_input_grad[0]:
            return (None,) * 9
        w_t_fp8, w_t_scale = ctx.saved_tensors
        orig_shape = grad_output.shape
        dy_2d = grad_output.reshape(-1, orig_shape[-1])
        dx = _fp8_mm(
            dy_2d,
            w_t_fp8,
            w_t_scale,
            scaling_mode=ctx.scaling_mode,
            use_fast_accum=ctx.use_fast_accum,
            out_dtype=grad_output.dtype,
            scaled_mm_fn=ctx.scaled_mm_fn,
        )
        return (dx.reshape(*orig_shape[:-1], w_t_fp8.shape[0]),) + (None,) * 8


class FrozenFP8Linear(nn.Module):
    """Drop-in replacement for a frozen ``nn.Linear`` using FP8 Tensor Cores.

    Quantized weights and scales live on private Python attributes rather than
    registered parameters or buffers so CUDA IPC weight inventories and PEFT
    ``save_pretrained`` outputs remain byte-identical to BF16.
    """

    def __init__(
        self,
        linear: nn.Linear,
        *,
        for_training: bool,
        scaling_mode: str = "rowwise",
        use_fast_accum: bool = False,
        scaled_mm_fn: Optional[Callable] = None,
    ):
        super().__init__()
        if linear.weight.requires_grad:
            raise ValueError("FrozenFP8Linear only supports frozen weights (requires_grad=False)")
        if linear.bias is not None and linear.bias.requires_grad:
            raise ValueError("FrozenFP8Linear only supports frozen bias (requires_grad=False)")
        if linear.in_features % 16 != 0 or linear.out_features % 16 != 0:
            raise ValueError(
                f"FrozenFP8Linear requires dimensions divisible by 16, got ({linear.out_features}, {linear.in_features})"
            )
        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
        self.for_training = bool(for_training)
        self.scaling_mode = scaling_mode
        self.use_fast_accum = bool(use_fast_accum)
        self._scaled_mm_fn = None if scaled_mm_fn is torch._scaled_mm else scaled_mm_fn

        with torch.no_grad():
            self._w_fp8, self._w_scale = quantize_frozen_weight(linear.weight, scaling_mode)
            if self.for_training:
                w_t = linear.weight.detach().t().contiguous()
                self._w_t_fp8, self._w_t_scale = quantize_frozen_weight(w_t, scaling_mode)
            else:
                self._w_t_fp8 = None
                self._w_t_scale = None
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def fp8_weight_bytes(self) -> int:
        total = self._w_fp8.numel() * self._w_fp8.element_size()
        total += self._w_scale.numel() * self._w_scale.element_size()
        if self._w_t_fp8 is not None:
            total += self._w_t_fp8.numel() * self._w_t_fp8.element_size()
            total += self._w_t_scale.numel() * self._w_t_scale.element_size()
        return int(total)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled() or not x.requires_grad or self._w_t_fp8 is None:
            orig_shape = x.shape
            x_2d = x.reshape(-1, orig_shape[-1])
            out = _fp8_mm(
                x_2d,
                self._w_fp8,
                self._w_scale,
                scaling_mode=self.scaling_mode,
                use_fast_accum=self.use_fast_accum,
                out_dtype=x.dtype,
                scaled_mm_fn=self._scaled_mm_fn,
            )
            if self.bias is not None:
                out = out + self.bias.to(out.dtype)
            return out.reshape(*orig_shape[:-1], self.out_features)
        if self._scaled_mm_fn is None:
            return _FrozenFP8LinearFn.apply(
                x,
                self._w_fp8,
                self._w_scale,
                self._w_t_fp8,
                self._w_t_scale,
                self.bias,
                self.scaling_mode,
                self.use_fast_accum,
            )
        return _CustomFrozenFP8LinearFn.apply(
            x,
            self._w_fp8,
            self._w_scale,
            self._w_t_fp8,
            self._w_t_scale,
            self.bias,
            self.scaling_mode,
            self.use_fast_accum,
            self._scaled_mm_fn,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, scaling={self.scaling_mode}, "
            f"fast_accum={self.use_fast_accum}, for_training={self.for_training}"
        )


def _unwrap_vla_root(model: nn.Module) -> Optional[nn.Module]:
    vla = getattr(model, "vla", None)
    if vla is None:
        return None
    if hasattr(vla, "get_base_model") and callable(vla.get_base_model):
        try:
            inner = vla.get_base_model()
            if inner is not None:
                return inner
        except Exception:
            pass
    return vla


def _eligible_frozen_linears(root: nn.Module, *, min_features: int, require_cuda: bool):
    eligible = []
    skipped = 0
    for name, module in root.named_modules():
        if type(module) is not nn.Linear:
            continue
        if module.weight.requires_grad or (module.bias is not None and module.bias.requires_grad):
            skipped += 1
            continue
        if (
            module.weight.ndim != 2
            or module.in_features < min_features
            or module.out_features < min_features
            or module.in_features % 16 != 0
            or module.out_features % 16 != 0
            or (require_cuda and not module.weight.is_cuda)
        ):
            skipped += 1
            continue
        eligible.append((name, module))
    return eligible, skipped


def _parity_metrics(actual: torch.Tensor, reference: torch.Tensor) -> Tuple[float, float]:
    a = actual.detach().float().reshape(-1)
    r = reference.detach().float().reshape(-1)
    if not torch.isfinite(a).all():
        return float("nan"), float("inf")
    denom = float(r.norm().item())
    rel_err = float((a - r).norm().item() / max(denom, 1e-12))
    cos_sim = float(F.cosine_similarity(a.unsqueeze(0), r.unsqueeze(0), eps=1e-12).item())
    return cos_sim, rel_err


def _probe_fp8_recipe(
    sample_linear: nn.Linear,
    *,
    role: str,
    scaled_mm_fn: Callable,
    candidates=(("rowwise", True), ("rowwise", False), ("tensorwise", True), ("tensorwise", False)),
) -> dict:
    """Verify FP8 capability, numerical parity, and CUDA-Graph safety on a real layer."""
    for_training = role == "trainer"
    device = sample_linear.weight.device
    dtype = sample_linear.weight.dtype if sample_linear.weight.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
    gen = torch.Generator(device="cpu").manual_seed(20261127)
    # Test both an unaligned token count (67, like B=1 inference) and an aligned one (64).
    x_cpu = torch.randn(67, sample_linear.in_features, generator=gen, dtype=torch.float32)
    # Inject a realistic per-token outlier row (common in Llama/ViT hidden states).
    x_cpu[0] *= 8.0
    x_cpu[1].zero_()
    x_base = x_cpu.to(device=device, dtype=dtype)
    dy_base = torch.randn(67, sample_linear.out_features, generator=gen, dtype=torch.float32).to(device=device, dtype=dtype)
    dy_base[2].zero_()

    with torch.no_grad():
        y_ref = F.linear(x_base, sample_linear.weight.to(dtype),
                         sample_linear.bias.to(dtype) if sample_linear.bias is not None else None)
        dx_ref = dy_base @ sample_linear.weight.to(dtype) if for_training else None

    failures = []
    for scaling_mode, use_fast_accum in candidates:
        try:
            probe_mod = FrozenFP8Linear(
                sample_linear,
                for_training=for_training,
                scaling_mode=scaling_mode,
                use_fast_accum=use_fast_accum,
                scaled_mm_fn=scaled_mm_fn,
            )
            if for_training:
                x_in = x_base.detach().clone().requires_grad_(True)
                y_fp8 = probe_mod(x_in)
                y_fp8.backward(dy_base)
                dx_fp8 = x_in.grad
                fwd_cos, fwd_rel = _parity_metrics(y_fp8, y_ref)
                bwd_cos, bwd_rel = _parity_metrics(dx_fp8, dx_ref)
                if not (
                    fwd_cos >= PROBE_MIN_COS_SIM
                    and fwd_rel <= PROBE_MAX_REL_ERR
                    and bwd_cos >= PROBE_MIN_COS_SIM
                    and bwd_rel <= PROBE_MAX_REL_ERR
                ):
                    failures.append(
                        f"{scaling_mode}/fast={use_fast_accum}: parity failed "
                        f"(fwd_cos={fwd_cos:.5f}, fwd_rel={fwd_rel:.4f}, bwd_cos={bwd_cos:.5f}, bwd_rel={bwd_rel:.4f})"
                    )
                    continue
                return {
                    "scaling_mode": scaling_mode,
                    "use_fast_accum": use_fast_accum,
                    "probe_forward_cos_sim": round(fwd_cos, 6),
                    "probe_forward_rel_err": round(fwd_rel, 6),
                    "probe_backward_cos_sim": round(bwd_cos, 6),
                    "probe_backward_rel_err": round(bwd_rel, 6),
                    "probe_cudagraph_ok": None,
                }
            else:
                with torch.inference_mode():
                    y_fp8 = probe_mod(x_base)
                fwd_cos, fwd_rel = _parity_metrics(y_fp8, y_ref)
                if not (fwd_cos >= PROBE_MIN_COS_SIM and fwd_rel <= PROBE_MAX_REL_ERR):
                    failures.append(
                        f"{scaling_mode}/fast={use_fast_accum}: parity failed "
                        f"(fwd_cos={fwd_cos:.5f}, fwd_rel={fwd_rel:.4f})"
                    )
                    continue
                cudagraph_ok = False
                if device.type == "cuda":
                    static_x = x_base.clone()
                    stream = torch.cuda.Stream(device=device)
                    stream.wait_stream(torch.cuda.current_stream(device))
                    with torch.inference_mode(), torch.cuda.stream(stream):
                        for _ in range(2):
                            probe_mod(static_x)
                    stream.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.inference_mode(), torch.cuda.graph(graph, stream=stream):
                        static_out = probe_mod(static_x)
                    stream.synchronize()
                    static_x.copy_(x_base)
                    graph.replay()
                    torch.cuda.synchronize(device)
                    if not torch.isfinite(static_out).all():
                        failures.append(f"{scaling_mode}/fast={use_fast_accum}: CUDA Graph output non-finite")
                        continue
                    if not torch.equal(static_out, y_fp8):
                        max_diff = float((static_out.float() - y_fp8.float()).abs().max().item())
                        failures.append(
                            f"{scaling_mode}/fast={use_fast_accum}: CUDA Graph replay differed from eager FP8 "
                            f"(max_abs_diff={max_diff:.6g})"
                        )
                        continue
                    cg_cos, cg_rel = _parity_metrics(static_out, y_ref)
                    if not (cg_cos >= PROBE_MIN_COS_SIM and cg_rel <= PROBE_MAX_REL_ERR):
                        failures.append(
                            f"{scaling_mode}/fast={use_fast_accum}: CUDA Graph parity failed "
                            f"(cos={cg_cos:.5f}, rel={cg_rel:.4f})"
                        )
                        continue
                    cudagraph_ok = True
                return {
                    "scaling_mode": scaling_mode,
                    "use_fast_accum": use_fast_accum,
                    "probe_forward_cos_sim": round(fwd_cos, 6),
                    "probe_forward_rel_err": round(fwd_rel, 6),
                    "probe_backward_cos_sim": None,
                    "probe_backward_rel_err": None,
                    "probe_cudagraph_ok": cudagraph_ok,
                }
        except Exception as exc:
            failures.append(f"{scaling_mode}/fast={use_fast_accum}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Frozen-base FP8 startup self-check failed for all recipes: " + "; ".join(failures))


def _trainable_and_buffer_inventory(model: nn.Module):
    params = tuple(
        (name, id(p), p.data_ptr(), tuple(p.shape), str(p.dtype))
        for name, p in model.named_parameters(recurse=True)
        if p.requires_grad
    )
    buffers = tuple(
        (name, id(b), b.data_ptr(), tuple(b.shape), str(b.dtype))
        for name, b in model.named_buffers(recurse=True)
    )
    return params, buffers


def _replace_submodule(root: nn.Module, dotted_name: str, replacement: nn.Module) -> None:
    if "." in dotted_name:
        parent_name, attr = dotted_name.rsplit(".", 1)
        parent = root.get_submodule(parent_name)
    else:
        parent, attr = root, dotted_name
    setattr(parent, attr, replacement)


def enable_frozen_base_fp8(
    model: nn.Module,
    *,
    role: str,
    enabled: Optional[bool] = None,
    min_features: int = DEFAULT_MIN_FEATURES,
    require_cuda: bool = True,
    scaled_mm_fn: Callable = torch._scaled_mm,
    candidates=(("rowwise", True), ("rowwise", False), ("tensorwise", True), ("tensorwise", False)),
) -> dict:
    """Convert frozen VLA base Linear layers to ``FrozenFP8Linear`` in place.

    Parameters
    ----------
    model:
        An ``ActorCritic`` instance (or module with ``.vla``).
    role:
        ``"trainer"`` (keeps forward ``W_fp8`` and backward ``W_t_fp8``) or
        ``"inference"`` (keeps forward ``W_fp8`` only, saving ~50% base VRAM).
    """
    if role not in ("trainer", "inference"):
        raise ValueError(f"role must be 'trainer' or 'inference', got {role!r}")
    active = fp8_base_linear_enabled() if enabled is None else bool(enabled)
    receipt = {
        "enabled": active,
        "converted": False,
        "role": role,
        "scaling_mode": None,
        "use_fast_accum": None,
        "layers_converted": 0,
        "layers_skipped": 0,
        "bf16_weight_bytes_replaced": 0,
        "fp8_weight_bytes_allocated": 0,
        "net_weight_bytes_delta": 0,
        "reason": None,
    }
    if not active:
        receipt["reason"] = "disabled"
        return receipt
    if FP8_DTYPE is None or not callable(scaled_mm_fn):
        raise RuntimeError("PyTorch float8_e4m3fn / torch._scaled_mm is unavailable in this runtime")

    vla_root = _unwrap_vla_root(model)
    if vla_root is None:
        raise RuntimeError("Frozen-base FP8 requires an ActorCritic model exposing .vla")

    eligible, skipped = _eligible_frozen_linears(
        vla_root, min_features=min_features, require_cuda=require_cuda
    )
    if not eligible:
        raise RuntimeError(
            f"Frozen-base FP8 found no eligible frozen Linear layers under .vla "
            f"(skipped={skipped}, min_features={min_features}, require_cuda={require_cuda})"
        )

    before_inventory = _trainable_and_buffer_inventory(model)
    # Run the capability, numerical parity, and CUDA-Graph self-check BEFORE replacing layers.
    probe = _probe_fp8_recipe(eligible[0][1], role=role, scaled_mm_fn=scaled_mm_fn, candidates=candidates)

    for_training = role == "trainer"
    bf16_bytes = 0
    fp8_bytes = 0
    for dotted_name, old_linear in eligible:
        bf16_bytes += int(old_linear.weight.numel() * old_linear.weight.element_size())
        replacement = FrozenFP8Linear(
            old_linear,
            for_training=for_training,
            scaling_mode=probe["scaling_mode"],
            use_fast_accum=probe["use_fast_accum"],
            scaled_mm_fn=scaled_mm_fn,
        )
        fp8_bytes += replacement.fp8_weight_bytes()
        _replace_submodule(vla_root, dotted_name, replacement)

    after_inventory = _trainable_and_buffer_inventory(model)
    if before_inventory != after_inventory:
        raise RuntimeError("Frozen-base FP8 unexpectedly altered trainable parameter or buffer inventory")

    if require_cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()

    receipt.update(
        converted=True,
        scaling_mode=probe["scaling_mode"],
        use_fast_accum=probe["use_fast_accum"],
        layers_converted=len(eligible),
        layers_skipped=skipped,
        bf16_weight_bytes_replaced=bf16_bytes,
        fp8_weight_bytes_allocated=fp8_bytes,
        net_weight_bytes_delta=fp8_bytes - bf16_bytes,
        probe_forward_cos_sim=probe["probe_forward_cos_sim"],
        probe_forward_rel_err=probe["probe_forward_rel_err"],
        probe_backward_cos_sim=probe["probe_backward_cos_sim"],
        probe_backward_rel_err=probe["probe_backward_rel_err"],
        probe_cudagraph_ok=probe["probe_cudagraph_ok"],
    )
    return receipt
