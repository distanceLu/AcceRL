"""CPU contract, numerical parity, IPC inventory, and torch.compile tests for frozen-base FP8."""
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

ENGINE = Path(__file__).resolve().parent
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from fp8_base_linear import (
    FrozenFP8Linear,
    enable_frozen_base_fp8,
    fp8_base_linear_enabled,
)
from training_topology import _selected_weights


def _sim_scaled_mm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias=None,
    out_dtype=torch.bfloat16,
    use_fast_accum: bool = False,
) -> torch.Tensor:
    """CPU reference mimicking Hopper cuBLASLt _scaled_mm layout & alignment contracts."""
    if mat_a.dtype != torch.float8_e4m3fn or mat_b.dtype != torch.float8_e4m3fn:
        raise TypeError("Expected float8_e4m3fn operands")
    if mat_a.ndim != 2 or mat_b.ndim != 2:
        raise ValueError("Expected 2D matrices")
    m, k = mat_a.shape
    k2, n = mat_b.shape
    if k != k2 or m % 16 != 0 or k % 16 != 0 or n % 16 != 0:
        raise RuntimeError(f"Alignment violation: ({m}, {k}) x ({k2}, {n})")
    if not mat_a.is_contiguous():
        raise RuntimeError("mat_a must be row-major contiguous")
    if mat_b.stride(0) != 1 or mat_b.stride(1) != k:
        raise RuntimeError(f"mat_b must be column-major, got stride={mat_b.stride()}")
    if bias is not None:
        raise AssertionError("FrozenFP8Linear must pass bias=None to preserve Hopper TMA templates")
    a_f = mat_a.to(torch.float32) * scale_a.to(torch.float32)
    b_f = mat_b.to(torch.float32) * scale_b.to(torch.float32)
    out = a_f @ b_f
    return out.to(out_dtype)


class _ToyLoraLinear(nn.Module):
    def __init__(self, in_f: int, out_f: int, r: int = 8, bias: bool = False):
        super().__init__()
        self.base_layer = nn.Linear(in_f, out_f, bias=bias)
        self.base_layer.weight.requires_grad_(False)
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad_(False)
        self.lora_A = nn.Linear(in_f, r, bias=False)
        self.lora_B = nn.Linear(r, out_f, bias=False)
        nn.init.normal_(self.lora_A.weight, std=0.05)
        nn.init.normal_(self.lora_B.weight, std=0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer(x) + self.lora_B(self.lora_A(x))


class _ToyVLA(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = _ToyLoraLinear(64, 128, bias=True)
        self.fc2 = _ToyLoraLinear(128, 64, bias=False)
        self.tiny_frozen = nn.Linear(16, 16, bias=False)
        self.tiny_frozen.weight.requires_grad_(False)
        self.lm_head = nn.Linear(64, 64, bias=False)  # trainable!
        self.register_buffer("rotary_cos", torch.ones(4, 16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.fc1(x))
        h = self.fc2(h)
        return self.lm_head(h)


class _ToyActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.vla = _ToyVLA()
        self.value_head = nn.Linear(64, 1)


class FrozenBaseFP8Tests(unittest.TestCase):
    def test_enabled_env_helper(self):
        self.assertFalse(fp8_base_linear_enabled({}))
        self.assertFalse(fp8_base_linear_enabled({"ACCERL_FP8_BASE_LINEAR": "0"}))
        self.assertTrue(fp8_base_linear_enabled({"ACCERL_FP8_BASE_LINEAR": "1"}))

    def test_rowwise_and_tensorwise_parity_padding_zero_and_tiny_grad(self):
        torch.manual_seed(17)
        base = nn.Linear(64, 128, bias=True).to(torch.bfloat16)
        base.weight.requires_grad_(False)
        base.bias.requires_grad_(False)

        for scaling in ("rowwise", "tensorwise"):
            for seq_len in (1, 16, 67):
                with self.subTest(scaling=scaling, seq_len=seq_len):
                    fp8_layer = FrozenFP8Linear(
                        base,
                        for_training=True,
                        scaling_mode=scaling,
                        use_fast_accum=False,
                        scaled_mm_fn=_sim_scaled_mm,
                    )
                    x_ref = torch.randn(2, seq_len, 64, dtype=torch.bfloat16)
                    if seq_len > 2:
                        x_ref[0, 0] *= 8.0
                        x_ref[0, 1].zero_()
                    x_ref.requires_grad_(True)
                    x_fp8 = x_ref.detach().clone().requires_grad_(True)

                    y_ref = base(x_ref)
                    y_fp8 = fp8_layer(x_fp8)
                    self.assertEqual(y_fp8.shape, y_ref.shape)
                    self.assertEqual(y_fp8.dtype, torch.bfloat16)
                    self.assertTrue(torch.isfinite(y_fp8).all())

                    dy = torch.randn_like(y_ref) * 1e-18  # exercise tiny gradient scaling
                    if seq_len > 2:
                        dy[0, 2].zero_()
                    y_ref.backward(dy)
                    y_fp8.backward(dy)
                    self.assertTrue(torch.isfinite(x_fp8.grad).all())

                    fwd_cos = F.cosine_similarity(
                        y_fp8.float().reshape(1, -1), y_ref.float().reshape(1, -1), eps=1e-30
                    ).item()
                    bwd_cos = F.cosine_similarity(
                        x_fp8.grad.float().reshape(1, -1), x_ref.grad.float().reshape(1, -1), eps=1e-30
                    ).item()
                    self.assertGreater(fwd_cos, 0.995)
                    self.assertGreater(bwd_cos, 0.995)

    def test_trainer_and_inference_conversion_preserves_ipc_inventory_and_lora_grads(self):
        torch.manual_seed(23)
        for role in ("trainer", "inference"):
            with self.subTest(role=role):
                model = _ToyActorCritic().to(torch.bfloat16)
                before_ipc = [
                    (name, t.data_ptr(), tuple(t.shape), t.dtype)
                    for name, t in _selected_weights(model, "policy")
                ]
                receipt = enable_frozen_base_fp8(
                    model,
                    role=role,
                    enabled=True,
                    min_features=64,
                    require_cuda=False,
                    scaled_mm_fn=_sim_scaled_mm,
                )
                self.assertTrue(receipt["converted"])
                self.assertEqual(receipt["layers_converted"], 2)
                self.assertIsInstance(model.vla.fc1.base_layer, FrozenFP8Linear)
                self.assertIsInstance(model.vla.fc2.base_layer, FrozenFP8Linear)
                # Trainable lm_head, LoRA, value_head, and small frozen layer stay nn.Linear.
                self.assertIsInstance(model.vla.lm_head, nn.Linear)
                self.assertIsInstance(model.vla.tiny_frozen, nn.Linear)
                self.assertIsInstance(model.value_head, nn.Linear)
                if role == "inference":
                    self.assertLess(receipt["net_weight_bytes_delta"], 0)
                    self.assertIsNone(model.vla.fc1.base_layer._w_t_fp8)
                else:
                    self.assertIsNotNone(model.vla.fc1.base_layer._w_t_fp8)

                after_ipc = [
                    (name, t.data_ptr(), tuple(t.shape), t.dtype)
                    for name, t in _selected_weights(model, "policy")
                ]
                self.assertEqual(before_ipc, after_ipc)
                # Ensure no float8 tensors leak into parameters or buffers.
                for _, tensor in list(model.named_parameters()) + list(model.named_buffers()):
                    self.assertNotEqual(tensor.dtype, torch.float8_e4m3fn)

                x = torch.randn(2, 19, 64, dtype=torch.bfloat16)
                if role == "trainer":
                    out = model.vla(x)
                    out.sum().backward()
                    self.assertIsNotNone(model.vla.fc1.lora_A.weight.grad)
                    self.assertIsNotNone(model.vla.lm_head.weight.grad)
                    self.assertTrue(torch.isfinite(model.vla.fc1.lora_A.weight.grad).all())
                else:
                    with torch.inference_mode():
                        out = model.vla(x)
                    self.assertTrue(torch.isfinite(out).all())

    def test_probe_falls_back_from_unsupported_rowwise_fast_accum_and_fails_fast_when_broken(self):
        torch.manual_seed(31)
        model = _ToyActorCritic().to(torch.bfloat16)

        def reject_rowwise_fast(a, b, scale_a, scale_b, bias=None, out_dtype=torch.bfloat16, use_fast_accum=False):
            if scale_a.ndim == 2 and use_fast_accum:
                raise RuntimeError("Rowwise scaling does not support use_fast_accum=True")
            return _sim_scaled_mm(a, b, scale_a, scale_b, bias=bias, out_dtype=out_dtype, use_fast_accum=use_fast_accum)

        receipt = enable_frozen_base_fp8(
            model,
            role="trainer",
            enabled=True,
            min_features=64,
            require_cuda=False,
            scaled_mm_fn=reject_rowwise_fast,
        )
        self.assertEqual(receipt["scaling_mode"], "rowwise")
        self.assertFalse(receipt["use_fast_accum"])

        broken_model = _ToyActorCritic().to(torch.bfloat16)
        def always_nan(a, b, scale_a, scale_b, bias=None, out_dtype=torch.bfloat16, use_fast_accum=False):
            return torch.full((a.shape[0], b.shape[1]), float("nan"), dtype=out_dtype)

        with self.assertRaisesRegex(RuntimeError, "startup self-check failed"):
            enable_frozen_base_fp8(
                broken_model,
                role="trainer",
                enabled=True,
                min_features=64,
                require_cuda=False,
                scaled_mm_fn=always_nan,
            )
        # Model must remain completely unmodified if probe fails.
        self.assertIsInstance(broken_model.vla.fc1.base_layer, nn.Linear)

    def test_fullgraph_compile_traces_frozen_fp8_forward_and_backward(self):
        if sys.version_info >= (3, 14):
            self.skipTest("torch.compile is unsupported on Python 3.14+")
        import fp8_base_linear
        from unittest.mock import patch

        torch.manual_seed(41)
        base = nn.Linear(64, 64, bias=True).to(torch.bfloat16)
        base.weight.requires_grad_(False)
        base.bias.requires_grad_(False)
        layer = FrozenFP8Linear(
            base,
            for_training=True,
            scaling_mode="rowwise",
            use_fast_accum=False,
        )
        with patch.object(fp8_base_linear, "_SCALED_MM_OVERRIDE", _sim_scaled_mm):
            layer.compile(backend="aot_eager", fullgraph=True)
            x = torch.randn(16, 64, dtype=torch.bfloat16, requires_grad=True)
            y = layer(x)
            y.sum().backward()
        self.assertEqual(y.shape, (16, 64))
        self.assertEqual(x.grad.shape, (16, 64))
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_peft_save_pretrained_after_frozen_fp8_swap(self):
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            self.skipTest("peft is not installed in this Python environment")

        class _SmallBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj1 = nn.Linear(64, 128, bias=False)
                self.proj2 = nn.Linear(128, 64, bias=True)

            def forward(self, x):
                return self.proj2(F.silu(self.proj1(x)))

        backbone = _SmallBackbone().to(torch.bfloat16)
        for p in backbone.parameters():
            p.requires_grad_(False)
        peft_vla = get_peft_model(backbone, LoraConfig(r=4, lora_alpha=4, target_modules=["proj1", "proj2"]))
        wrapper = nn.Module()
        wrapper.vla = peft_vla
        receipt = enable_frozen_base_fp8(
            wrapper,
            role="trainer",
            enabled=True,
            min_features=64,
            require_cuda=False,
            scaled_mm_fn=_sim_scaled_mm,
        )
        self.assertEqual(receipt["layers_converted"], 2)
        y = wrapper.vla(torch.randn(2, 17, 64, dtype=torch.bfloat16))
        y.sum().backward()
        with tempfile.TemporaryDirectory() as tmp:
            wrapper.vla.save_pretrained(tmp)
            names = {p.name for p in Path(tmp).iterdir()}
            self.assertIn("adapter_config.json", names)
            self.assertTrue(any(n.startswith("adapter_model.") for n in names))


if __name__ == "__main__":
    unittest.main()
