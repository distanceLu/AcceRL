"""Real native Torch/attention equivalence tests; no weights, GPU or downloads.

Set WEAVER_TEST_UPSTREAM_SOURCE to the audited upstream checkout when it is not
available in this workspace's analysis tree. Optional WEAVER_CPU_REPORT_PATH
writes the measured errors and dependency/source identities after this suite.
The native compile decorator is disabled while importing: production explicitly
unwraps it for use_compile=false, and local Python 3.14 cannot use torch.compile.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True

from weaver_cached_inference import (decode_rollout_observations,
                                     generate_variable_horizon_cached)


def upstream_source():
    configured = os.environ.get("WEAVER_TEST_UPSTREAM_SOURCE")
    if configured:
        source = Path(configured).resolve()
        if not (source / "weaver/wm/model.py").is_file():
            raise RuntimeError("WEAVER_TEST_UPSTREAM_SOURCE is not an upstream checkout")
        return source
    for parent in Path(__file__).resolve().parents:
        source = parent / "analysis/weaver_integration_20261022/upstream"
        if (source / "weaver/wm/model.py").is_file():
            return source
    raise RuntimeError("Set WEAVER_TEST_UPSTREAM_SOURCE for native equivalence tests")


REPORT = {"protocol": "weaver_native_cached_cpu_equivalence_v1", "cases": [],
          "device": "cpu", "production_weights_loaded": False,
          "no_gpu_or_deployment": True}


class NativeCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)
        source = upstream_source()
        sys.path.insert(0, str(source))
        with patch.object(torch, "compile", lambda function, *args, **kwargs: function):
            cls.native = importlib.import_module("weaver.wm.model")
        if Path(cls.native.__file__).resolve() != (source / "weaver/wm/model.py").resolve():
            raise RuntimeError("Tests imported a different WEAVER source")
        REPORT.update(torch_version=torch.__version__, python=sys.version,
                      upstream_source=str(source), compile_enabled=False,
                      source_sha256={name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                     for name in ("weaver/wm/model.py", "weaver/wm/nets.py")})

    def model(self, *, sprint=True, memory=6, forcing=True):
        torch, native = self.torch, self.native
        torch.manual_seed(129)
        # Construct the real native sampler/core without loading encoder assets,
        # a critic, or a duplicate EMA snapshot. Only dimensional scale is tiny.
        model = native.WEAVER.__new__(native.WEAVER)
        torch.nn.Module.__init__(model)
        model._img_keys = ["full_image", "wrist_image"]
        model._n_history, model._n_horizon = 2, 8
        model._n_memory_frames, model._t_memory = memory, 5
        model._use_memory, model._use_precomputed_features = bool(memory), True
        model._spatial_size = 1
        model._inference_steps, model._flow_loss = 16, "v-pred"
        model._diff_forcing = forcing
        model._pyramid_stagger_width, model._pyramid_schedule_type = 1, "linear"
        model._pyramid_schedule_kwargs = {"power": .5, "steepness": 5., "center": .5}
        model._pyramid_schedule_cache = {}
        model.device = "cpu"
        model.wm = native.FlowWM(img_keys=model._img_keys, n_embed=32, n_layers=8,
            n_heads=4, n_spatial=1, n_hidden=24, n_im_feature=4, n_states=8,
            n_actions=7, qk_norm=True, use_linear_proj=True, image_size=(16, 24),
            spatial_size=1, use_sprint=sprint, sprint_drop_ratio=.5,
            sprint_encoder_blocks=1, sprint_decoder_blocks=1, path_drop_prob=.05,
            n_memory_frames=memory)
        original = getattr(model.wm.forward, "_torchdynamo_orig_callable", None)
        if original is not None:
            model.wm.forward = original.__get__(model.wm, type(model.wm))
        model.rm = native.RewardModel(img_keys=model._img_keys, n_embed=32,
            n_hidden=24, n_im_feature=4, n_task_feature=12, n_states=8, n_actions=7)
        model.eval()
        return model

    def inputs(self, model, horizon, *, batch=1):
        torch = self.torch
        context = {key: torch.randn(batch, 2, 6, 4) for key in model._img_keys}
        context["states"] = torch.randn(batch, 2, 8)
        memory = None
        if model._use_memory:
            raw = {key + "_features": torch.randn(batch, model._n_memory_frames, 6, 4)
                   for key in model._img_keys}
            raw["states"] = torch.randn(batch, model._n_memory_frames, 8)
            with torch.no_grad():
                memory = model.encode_memory_obs(raw)
        return dict(context=context, future_actions=torch.randn(batch, horizon, 7),
                    future_states=torch.zeros(batch, horizon, 8),
                    history_actions=torch.randn(batch, 2, 7), memory_tokens=memory)

    def compare(self, horizon, *, sprint=True, memory=6, forcing=True, batch=1,
                autocast=False, history_none=False):
        torch = self.torch
        model = self.model(sprint=sprint, memory=memory, forcing=forcing)
        inputs = self.inputs(model, horizon, batch=batch)
        if history_none:
            inputs["history_actions"] = None
        originals = {key: value.clone() for key, value in inputs["context"].items()}
        original_state = {key: value.clone() for key, value in model.state_dict().items()}
        seed, calls = 18721, []
        native_forward, cached_forward = model.wm.forward, model.wm.forward_cached
        def record_native(*args, **kwargs):
            calls.append("native")
            return native_forward(*args, **kwargs)
        def record_cached(*args, **kwargs):
            calls.append(args[-2])
            return cached_forward(*args, **kwargs)
        model.wm.forward = record_native
        model.wm.forward_cached = record_cached
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            torch.manual_seed(seed)
            expected = model.generate_latent_rollouts_variable_horizon(**inputs)
            expected_rng = torch.get_rng_state().clone()
            expected_calls = list(calls)
            calls.clear()
            torch.manual_seed(seed)
            actual = generate_variable_horizon_cached(model, **inputs)
            actual_rng = torch.get_rng_state().clone()
            task = torch.randn(batch, 12)
            hist = inputs["history_actions"]
            if hist is None:
                hist = torch.zeros(batch, 2, 7)
            actions = torch.cat([hist, inputs["future_actions"]], dim=1)
            expected_score = model.rm(expected, actions, task)
            actual_score = model.rm(actual, actions, task)
        nfe = 16 + horizon - 1 if forcing else 16
        self.assertEqual(expected_calls, ["native"] * nfe)
        self.assertEqual(calls, ["write"] + ["read"] * (nfe - 1))
        self.assertTrue(torch.equal(expected_rng, actual_rng), "Noise RNG consumption changed")
        # Cache reads use an explicit causal mask while full forward uses SDPA's
        # is_causal path. Float32 tolerance allows rounding, not changed semantics.
        atol, rtol = ((.008, .008) if autocast else (3e-6, 3e-6))
        errors = {}
        for key in expected:
            self.assertEqual(actual[key].shape[:2], (batch, 2 + horizon))
            torch.testing.assert_close(actual[key], expected[key], atol=atol, rtol=rtol)
            self.assertTrue(torch.equal(actual[key][:, :2], inputs["context"][key]))
            self.assertFalse(actual[key].requires_grad)
            errors[key] = float((actual[key] - expected[key]).abs().max())
            self.assertTrue(torch.equal(inputs["context"][key], originals[key]))
        torch.testing.assert_close(actual_score, expected_score, atol=atol, rtol=rtol)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, original_state[key]), key)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        REPORT["cases"].append(dict(horizon=horizon, sprint=sprint, memory=memory,
            diff_forcing=forcing, batch=batch, autocast=autocast, history_none=history_none,
            native_nfe=nfe, cached_write=1, cached_reads=nfe-1, identical_rng=True,
            max_abs_errors=errors, score_max_abs_error=float((actual_score-expected_score).abs().max()),
            atol=atol, rtol=rtol))

    def test_production_schedule_every_supported_horizon(self):
        for horizon in range(1, 9):
            with self.subTest(horizon=horizon):
                self.compare(horizon)

    def test_native_no_sprint_no_memory_and_default_history(self):
        for horizon in (1, 4, 8):
            with self.subTest(horizon=horizon):
                self.compare(horizon, sprint=False, memory=0, history_none=True)

    def test_native_lockstep_and_batch_two(self):
        self.compare(4, forcing=False, batch=2)

    def test_bfloat16_autocast_horizons(self):
        for horizon in (1, 4, 8):
            with self.subTest(horizon=horizon):
                self.compare(horizon, autocast=True)

    def test_training_and_invalid_request_rejected_without_rng_change(self):
        torch = self.torch
        model = self.model()
        inputs = self.inputs(model, 4)
        before = torch.get_rng_state().clone()
        model.train()
        with self.assertRaisesRegex(ValueError, "eval"):
            generate_variable_horizon_cached(model, **inputs)
        model.eval()
        for change in (dict(memory_tokens=None), dict(future_states=torch.zeros(1, 3, 8)),
                       dict(history_actions=torch.zeros(1, 1, 7)),
                       dict(future_actions=torch.zeros(1, 9, 7)),
                       dict(future_actions=torch.zeros(1, 0, 7))):
            with self.assertRaises(ValueError):
                generate_variable_horizon_cached(model, **dict(inputs, **change))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_caches_are_per_call_and_weights_can_change_between_calls(self):
        torch = self.torch
        model = self.model()
        inputs = self.inputs(model, 4)
        torch.manual_seed(99)
        first = generate_variable_horizon_cached(model, **inputs)
        with torch.no_grad():
            model.wm.out_prj["states"].weight.add_(.2)
        torch.manual_seed(99)
        second = generate_variable_horizon_cached(model, **inputs)
        torch.manual_seed(99)
        expected = model.generate_latent_rollouts_variable_horizon(**inputs)
        self.assertGreater(float((first["states"]-second["states"]).abs().max()), .01)
        for key in expected:
            torch.testing.assert_close(second[key], expected[key], atol=3e-6, rtol=3e-6)


class NativeSD3DecodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Real Diffusers AutoencoderKL, randomly initialized at a small channel
        # width. This exercises the native 2D conv/groupnorm/attention decoder;
        # no pretrained files, internet/model access, or stubbed pixel decoder.
        import torch
        from diffusers import AutoencoderKL
        cls.torch = torch
        torch.set_num_threads(1)
        torch.manual_seed(731)
        cls.vae = AutoencoderKL(in_channels=3, out_channels=3,
            down_block_types=("DownEncoderBlock2D",) * 4,
            up_block_types=("UpDecoderBlock2D",) * 4,
            block_out_channels=(16, 16, 32, 32), layers_per_block=1,
            latent_channels=16, norm_num_groups=8, sample_size=32,
            scaling_factor=1.5305, shift_factor=.0609,
            use_quant_conv=False, use_post_quant_conv=False).eval()
        # Execute the audited SD3Encoder class unchanged, omitting its module's
        # unrelated CLIP/Transformers imports and its pretrained-loading init.
        import ast
        from typing import Tuple
        import einops
        source = upstream_source()
        parsed = ast.parse((source / "weaver/wm/encoders.py").read_text(encoding="utf-8"))
        definition = next(node for node in parsed.body
                          if isinstance(node, ast.ClassDef) and node.name == "SD3Encoder")
        namespace = dict(torch=torch, nn=torch.nn, rearrange=einops.rearrange,
                         Tuple=Tuple, AutoencoderKL=AutoencoderKL)
        exec(compile(ast.Module(body=[definition], type_ignores=[]),
                     str(source / "weaver/wm/encoders.py"), "exec"), namespace)
        encoder = namespace["SD3Encoder"].__new__(namespace["SD3Encoder"])
        torch.nn.Module.__init__(encoder)
        encoder.vae, encoder.spatial_size = cls.vae, 2
        encoder.latent_h = (4, 4)
        encoder.scaling_factor, encoder.shift_factor = 1.5305, .0609
        encoder.eval()
        sys.path.insert(0, str(source))
        with patch.object(torch, "compile", lambda function, *args, **kwargs: function):
            native = importlib.import_module("weaver.wm.model")
        cls.model = native.WEAVER.__new__(native.WEAVER)
        torch.nn.Module.__init__(cls.model)
        cls.model._img_keys = ["full_image", "wrist_image"]
        cls.model._train_decoder = False
        cls.model.im_encoder = encoder
        cls.model.eval()
        import diffusers
        REPORT["decoder"] = dict(diffusers_version=diffusers.__version__,
            actual_class="diffusers.AutoencoderKL", latent_channels=16,
            temporal_layers=False, source_sd3_wrapper=True, image_size=[32, 32],
            decode_strategy="select_last_from_original_native_decode_chunk", cases=[])

    def test_last_matches_full_for_all_horizons_and_autocast(self):
        torch, model = self.torch, self.model
        for autocast in (False, True):
            for horizon in (1, 4, 8):
                with self.subTest(horizon=horizon, autocast=autocast):
                    future = {key: torch.randn(1, horizon, 4, 64) for key in model._img_keys}
                    future["states"] = torch.randn(1, horizon, 8)
                    originals = {key: value.clone() for key, value in future.items()}
                    calls = []
                    original = self.vae.decode
                    def record(latents, *args, **kwargs):
                        calls.append(latents.shape[0])
                        return original(latents, *args, **kwargs)
                    with patch.object(self.vae, "decode", record), torch.no_grad(), \
                         torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
                        rng = torch.get_rng_state().clone()
                        all_frames = decode_rollout_observations(model, future, model._img_keys,
                                                               chunk_size=2, output_mode="all")
                        full_calls = list(calls)
                        calls.clear()
                        last = decode_rollout_observations(model, future, model._img_keys,
                                                          chunk_size=2, output_mode="last")
                    self.assertEqual(list(last.shape), [1, 1, 2, 3, 32, 32])
                    self.assertEqual(list(all_frames.shape), [1, horizon, 2, 3, 32, 32])
                    self.assertEqual(calls, [1 if horizon == 1 else 2] * 2)
                    self.assertEqual(sum(full_calls), horizon * 2)
                    atol, rtol = 0., 0.
                    torch.testing.assert_close(last, all_frames[:, -1:], atol=atol, rtol=rtol)
                    self.assertEqual(last.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(last).all())
                    self.assertTrue(((last >= -1) & (last <= 1)).all())
                    self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                    for key in future:
                        self.assertTrue(torch.equal(future[key], originals[key]))
                    REPORT["decoder"]["cases"].append(dict(horizon=horizon, autocast=autocast,
                        all_decode_batches=full_calls, last_decode_batches=calls,
                        max_abs_error=float((last-all_frames[:, -1:]).abs().max()), atol=atol, rtol=rtol))

    def test_original_chunk_boundaries_when_batch_and_horizon_do_not_align(self):
        torch, model = self.torch, self.model
        # The legacy flatten order is b0t0,b0t1,b0t2,b1t0,b1t1,b1t2.
        # With chunk2, b0's last frame shares a decode batch with b1's first.
        # Preserve that batch instead of padding, repeating, or changing pixels.
        future = {key: torch.randn(2, 3, 4, 64) for key in model._img_keys}
        future["states"] = torch.randn(2, 3, 8)
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            full = decode_rollout_observations(model, future, model._img_keys, output_mode="all")
            last = decode_rollout_observations(model, future, model._img_keys, output_mode="last")
        self.assertEqual(list(last.shape), [2, 1, 2, 3, 32, 32])
        torch.testing.assert_close(last, full[:, -1:], atol=0., rtol=0.)

    def test_unknown_mode_or_view_order_cannot_silently_change_pixels(self):
        torch, model = self.torch, self.model
        future = {key: torch.zeros(1, 1, 4, 64) for key in model._img_keys}
        future["states"] = torch.zeros(1, 1, 8)
        for kwargs in (dict(output_mode="first"), dict(chunk_size=0)):
            with self.assertRaises(ValueError):
                decode_rollout_observations(model, future, model._img_keys, **kwargs)
        with self.assertRaises(ValueError):
            decode_rollout_observations(model, future, list(reversed(model._img_keys)))


if __name__ == "__main__":
    result = unittest.main(exit=False)
    REPORT.update(successful=result.result.wasSuccessful(), tests_run=result.result.testsRun,
                  failures=len(result.result.failures), errors=len(result.result.errors))
    output = os.environ.get("WEAVER_CPU_REPORT_PATH")
    if output:
        Path(output).write_text(json.dumps(REPORT, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if result.result.wasSuccessful() else 1)
