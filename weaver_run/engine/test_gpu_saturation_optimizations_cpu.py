"""CPU equivalence and safety tests for GPU-saturation training optimizations."""
from copy import deepcopy
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

ENGINE = Path(__file__).resolve().parent
REPO = ENGINE.parents[1]
for p in (str(ENGINE), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from advantage_ema import AdvantageEMA
from fresh_config import FreshConfig
from fresh_policy_replay import SegmentReplay
from jit_gae_runtime import run_jit_policy_update
import jit_gae_runtime
from test_jit_gae_cpu import ToyEngine
from test_model_free_critic_pruning_cpu import TinyPolicy, real_segment
from trainer_compile import compile_policy_trainer, compile_trainer_enabled, configure_compile_cache, eager_inference_context


class _FakeVisionTower(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(3, out_dim, kernel_size=4, stride=4)
        self.blocks = nn.ModuleList([nn.Linear(out_dim, out_dim)])

    def forward(self, x):
        y = self.conv(x).flatten(2).transpose(1, 2)
        for blk in self.blocks:
            y = blk(y)
        return y


class BatchedVisionTests(unittest.TestCase):
    def test_batched_dual_view_matches_serial_loop_forward_and_backward(self):
        from prismatic.extern.hf.modeling_prismatic import PrismaticVisionBackbone

        torch.manual_seed(7)
        vb = object.__new__(PrismaticVisionBackbone)
        nn.Module.__init__(vb)
        vb.use_fused_vision_backbone = True
        vb.num_images_in_input = 2
        vb.featurizer = _FakeVisionTower(6)
        vb.fused_featurizer = _FakeVisionTower(8)

        vb_ref = deepcopy(vb)
        x = torch.randn(3, 12, 8, 8, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)

        actual = vb.forward(x)

        # Reference original serial loop
        images = torch.split(x_ref, [6, 6], dim=1)
        all_patches = []
        for img in images:
            r, f = torch.split(img, [3, 3], dim=1)
            all_patches.append(torch.cat([vb_ref.featurizer(r), vb_ref.fused_featurizer(f)], dim=2))
        expected = torch.cat(all_patches, dim=1)

        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        actual.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-6, atol=1e-6)
        for p_a, p_e in zip(vb.parameters(), vb_ref.parameters()):
            torch.testing.assert_close(p_a.grad, p_e.grad, rtol=1e-6, atol=1e-6)


class ActionOnlyForwardTests(unittest.TestCase):
    def test_prepared_batch_preserves_exact_dict_keys_and_detects_padding(self):
        from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, STOP_INDEX
        from rl.utils import batch_process_obs

        count = ACTION_DIM * NUM_ACTIONS_CHUNK
        prompt_len = 5
        total_len = prompt_len + count + 1
        labels = torch.full((1, total_len), IGNORE_INDEX, dtype=torch.long)
        labels[:, prompt_len : prompt_len + count] = ACTION_TOKEN_BEGIN_IDX + 5
        labels[:, -1] = STOP_INDEX
        item_a = dict(
            input_ids=torch.ones((1, total_len), dtype=torch.long),
            attention_mask=torch.ones((1, total_len), dtype=torch.long),
            labels=labels.clone(),
            pixel_values=torch.zeros((1, 12, 4, 4)),
        )
        item_b = {k: v.clone() for k, v in item_a.items()}
        unpadded = batch_process_obs(0, [item_a, item_b], torch.device("cpu"))
        self.assertIsInstance(unpadded, dict)
        self.assertEqual(set(unpadded.keys()), {"input_ids", "attention_mask", "labels", "pixel_values"})
        self.assertTrue(getattr(unpadded, "unpadded_action_layout", False))

        # Different lengths force padding -> unpadded_action_layout must be False
        item_short = {
            "input_ids": torch.ones((1, total_len - 1), dtype=torch.long),
            "attention_mask": torch.ones((1, total_len - 1), dtype=torch.long),
            "labels": labels[:, 1:].clone(),
            "pixel_values": torch.zeros((1, 12, 4, 4)),
        }
        padded = batch_process_obs(0, [{k: v.clone() for k, v in item_a.items()}, item_short], torch.device("cpu"))
        self.assertFalse(getattr(padded, "unpadded_action_layout", True))

    def test_action_only_fast_path_matches_full_sequence_and_graph_path_exactly(self):
        from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, STOP_INDEX
        from rl.actor_critic_model_discrete import ActorCritic
        from rl.utils import _PreparedBatch

        torch.manual_seed(11)
        dim, bins, patches_n = 16, 12, 4
        action_count = ACTION_DIM * NUM_ACTIONS_CHUNK

        class _ToyCore(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(dim, dim)
                self.seen_attn_masks = []
                self.seen_output_hidden = []

            def forward(self, inputs_embeds=None, attention_mask=None, output_hidden_states=False, **kwargs):
                self.seen_attn_masks.append(attention_mask)
                self.seen_output_hidden.append(output_hidden_states)
                h = torch.tanh(self.proj(inputs_embeds))
                return SimpleNamespace(last_hidden_state=h, hidden_states=(inputs_embeds, h) if output_hidden_states else None)

        class _ToyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = _ToyCore()
                self.lm_head = nn.Linear(dim, bins)
                self.config = SimpleNamespace(pretraining_tp=1)

            def forward(self, inputs_embeds=None, attention_mask=None, output_hidden_states=False, **kwargs):
                out = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                 output_hidden_states=output_hidden_states, **kwargs)
                logits = self.lm_head(out.last_hidden_state).float()
                return SimpleNamespace(logits=logits, hidden_states=out.hidden_states)

        class _ToyVLA(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(64, dim)
                self.vision = nn.Linear(4, dim)
                self.language_model = _ToyLM()
                self.config = SimpleNamespace(output_attentions=False)

            def get_input_embeddings(self):
                return self.embed

            def _process_vision_features(self, pixel_values, language_embeddings=None, use_film=False):
                return self.vision(pixel_values)

            def _process_action_masks(self, labels):
                return (labels > ACTION_TOKEN_BEGIN_IDX) & (labels != STOP_INDEX) & (labels != IGNORE_INDEX)

            def _build_multimodal_attention(self, embedded, patches, attention_mask):
                seq = torch.cat([embedded[:, :1, :], patches, embedded[:, 1:, :]], dim=1)
                if attention_mask is None:
                    return seq, None
                pm = torch.ones((patches.shape[0], patches.shape[1]), dtype=attention_mask.dtype, device=attention_mask.device)
                return seq, torch.cat([attention_mask[:, :1], pm, attention_mask[:, 1:]], dim=1)

        ac = object.__new__(ActorCritic)
        nn.Module.__init__(ac)
        ac.cfg = SimpleNamespace(use_film=False, use_diffusion=False, use_l1_regression=False, use_proprio=False)
        ac.vla = _ToyVLA().to(dtype=torch.bfloat16)
        ac.device = torch.device("cpu")
        ac.model_dtype = torch.bfloat16
        ac.n_action_bins = bins
        ac.attn_pool = nn.Sequential(nn.Linear(dim, 1)).to(dtype=torch.bfloat16)
        ac.value_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1)).to(dtype=torch.bfloat16)
        ac._compute_num_patches = lambda: patches_n

        seq_len = 4 + action_count + 1
        labels = torch.full((2, seq_len), IGNORE_INDEX, dtype=torch.long)
        labels[:, 4 : 4 + action_count] = ACTION_TOKEN_BEGIN_IDX + 3
        labels[:, -1] = STOP_INDEX
        raw_batch = dict(
            input_ids=torch.randint(0, 64, (2, seq_len)),
            attention_mask=torch.ones((2, seq_len), dtype=torch.long),
            labels=labels,
            pixel_values=torch.randn(2, patches_n, 4),
            proprio=None,
        )
        fast_batch = _PreparedBatch(raw_batch, unpadded_action_layout=True)

        # Compare unpadded static path vs general padded path
        logits_fast, val_fast = ac.forward(fast_batch)
        self.assertIsNone(ac.vla.language_model.model.seen_attn_masks[-1])
        self.assertFalse(ac.vla.language_model.model.seen_output_hidden[-1])
        self.assertEqual(logits_fast.dtype, torch.float32)
        self.assertEqual(val_fast.dtype, torch.float32)

        logits_general, val_general = ac.forward(raw_batch)
        self.assertIsNotNone(ac.vla.language_model.model.seen_attn_masks[-1])
        self.assertFalse(ac.vla.language_model.model.seen_output_hidden[-1])

        torch.testing.assert_close(logits_fast, logits_general, rtol=0, atol=0)
        torch.testing.assert_close(val_fast, val_general, rtol=0, atol=0)

        # Also compare single-sample B=1 against real_action_from_patches (CUDA Graph path)
        b1 = _PreparedBatch({k: (v[:1] if isinstance(v, torch.Tensor) else v) for k, v in raw_batch.items()},
                            unpadded_action_layout=True)
        logits_e1, val_e1 = ac.forward(b1)
        patches_1 = ac.vla._process_vision_features(b1["pixel_values"].to(torch.bfloat16), use_film=False)
        with patch("torch.autocast", return_value=eager_inference_context()):
            logits_g1, val_g1 = ac.real_action_from_patches(b1["input_ids"], patches_1, None)
        self.assertEqual(logits_g1.dtype, torch.float32)
        self.assertEqual(val_g1.dtype, torch.float32)
        self.assertTrue(torch.equal(logits_e1, logits_g1))
        self.assertTrue(torch.equal(val_e1, val_g1))


def _make_deferred_vote_fixture(model):
    store = SegmentReplay(256, FreshConfig.from_world_config({"backend": "model_free"}), rng=random.Random(7))
    store.set_versions(5, 0)
    for n in range(6):
        store.add(real_segment(n, 8))
    _, ids = store.reserve(16)
    denoms = dict.fromkeys(ids, 16.0)
    batch, _ = store.materialize(ids, jit_denominators=denoms)
    engine = ToyEngine(model, gas=2)
    ema = AdvantageEMA("cpu")
    ema.commit(torch.tensor([2.0, 0.0, 2.0]), expected_updates=0, optimizer_succeeded=True)
    events = []
    config = SimpleNamespace(
        jit_gae_mode="fused",
        advantage_norm="ema_zscore",
        jit_context_micro=2,
        advantage_ema_beta=0.9,
        advantage_scale_floor=0.001,
        policy_sampling="chunks",
    )

    def verify(eng, kind, before, skips):
        if eng.global_steps - before != 1 or eng.skipped_steps - skips != 0:
            raise RuntimeError("Failed optimizer window")

    owner = SimpleNamespace(
        model=engine,
        base_model=model,
        fresh_config=config,
        global_step=5,
        super_batch_size=16,
        train_batch_size=8,
        rank=0,
        cfg=SimpleNamespace(lora_dropout=0.0),
        capture_policy_metadata=dict(sample_ids=ids, fresh_versions=[5, 0], jit_pending=True),
        vf_coef=0.5,
        ent_coef=0.01,
        kl_coef=0.1,
        clip_mode="gipo",
        clip_eps=0.2,
        policy_train_start_step=0,
        advantage_ema=ema,
        telemetry=SimpleNamespace(event=lambda name, **kw: events.append((name, kw))),
        events=events,
        _verify_optimizer_update=verify,
    )
    return owner, batch


class DeferredForwardVoteTests(unittest.TestCase):
    def test_deferred_vote_runs_once_per_window_and_aborts_before_optimizer_on_nan_or_error(self):
        # 1) Normal fast run: single collective vote instead of per-micro votes
        owner, batch = _make_deferred_vote_fixture(TinyPolicy())
        vote_calls = []
        orig_sum = jit_gae_runtime._sum

        def counting_sum(tensor):
            if tensor.dtype == torch.int32 and tensor.ndim == 0:
                vote_calls.append(int(tensor.item()))
            return orig_sum(tensor)

        with patch.dict(os.environ, ACCERL_FAST_JIT_BOUNDARIES="1", ACCERL_DEFERRED_FORWARD_VOTE="1"), \
             patch.object(jit_gae_runtime, "_sum", side_effect=counting_sum):
            run_jit_policy_update(owner, batch[0], batch[1], batch[3],
                                  micro_begin=lambda *a, **k: None, micro_end=lambda *a, **k: None, trim=lambda _: None)
        # 1 prepare vote + 1 deferred boundary vote (not 1 + 2 micro votes)
        self.assertEqual(len(vote_calls), 2)
        self.assertEqual(vote_calls, [0, 0])
        self.assertTrue(owner.events[-1][1]["details"]["deferred_forward_vote"])
        self.assertEqual(owner.model.global_steps, 1)

        # 2) NaN on micro 0 must still fail before optimizer step
        bad_model = TinyPolicy()
        with torch.no_grad():
            for p in bad_model.parameters():
                p.fill_(float("nan"))
        bad_owner, batch2 = _make_deferred_vote_fixture(bad_model)
        with patch.dict(os.environ, ACCERL_FAST_JIT_BOUNDARIES="1", ACCERL_DEFERRED_FORWARD_VOTE="1"):
            with self.assertRaises(RuntimeError):
                run_jit_policy_update(bad_owner, batch2[0], batch2[1], batch2[3],
                                      micro_begin=lambda *a, **k: None, micro_end=lambda *a, **k: None, trim=lambda _: None)
        self.assertEqual(bad_owner.model.global_steps, 0)
        self.assertEqual(bad_owner.advantage_ema.updates, 1)

        # 3) Exception on micro 0 must still reach boundary vote and fail before optimizer step
        class ExplodingFirstMicro(TinyPolicy):
            def __init__(self):
                super().__init__()
                self.tripped = False

            def forward(self, inputs):
                if torch.is_grad_enabled() and not self.tripped:
                    self.tripped = True
                    raise ValueError("synthetic micro-0 failure")
                return super().forward(inputs)

        err_owner, batch3 = _make_deferred_vote_fixture(ExplodingFirstMicro())
        err_votes = []
        with patch.dict(os.environ, ACCERL_FAST_JIT_BOUNDARIES="1", ACCERL_DEFERRED_FORWARD_VOTE="1"), \
             patch.object(jit_gae_runtime, "_sum", side_effect=lambda t: (err_votes.append(int(t.item())) if t.dtype == torch.int32 and t.ndim == 0 else None) or orig_sum(t)):
            with self.assertRaises(RuntimeError):
                run_jit_policy_update(err_owner, batch3[0], batch3[1], batch3[3],
                                      micro_begin=lambda *a, **k: None, micro_end=lambda *a, **k: None, trim=lambda _: None)
        self.assertEqual(err_votes, [0, 1])  # prepare succeeded (0), deferred boundary reported failure (1)
        self.assertEqual(err_owner.model.global_steps, 0)


class TrainerCompileTests(unittest.TestCase):
    def test_compile_flag_cache_scoping_and_in_place_inventory_preservation(self):
        self.assertFalse(compile_trainer_enabled({"ACCERL_FAST_JIT_BOUNDARIES": "0"}))
        self.assertTrue(compile_trainer_enabled({"ACCERL_FAST_JIT_BOUNDARIES": "1"}))
        self.assertFalse(compile_trainer_enabled({"ACCERL_FAST_JIT_BOUNDARIES": "1", "ACCERL_COMPILE_TRAINER": "0"}))

        with tempfile.TemporaryDirectory() as tmp:
            env = {"ACCERL_EXPERIMENT_ROOT": str(Path(tmp) / "exp")}
            cache = configure_compile_cache(env, rank=2)
            self.assertIn("rank_2", cache["inductor_cache_dir"])
            self.assertIn("rank_2", cache["triton_cache_dir"])
            self.assertTrue(Path(cache["inductor_cache_dir"]).is_dir())
            self.assertTrue(Path(cache["triton_cache_dir"]).is_dir())

        class _DummyPolicy(nn.Module):
            def __init__(self):
                super().__init__()
                self.vla = nn.Module()
                self.vla.language_model = nn.Module()
                self.vla.language_model.model = nn.Module()
                self.vla.language_model.model.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(2)])
                self.vla.vision_backbone = nn.Module()
                self.vla.vision_backbone.featurizer = nn.Module()
                self.vla.vision_backbone.featurizer.blocks = nn.ModuleList([nn.Linear(8, 8)])
                self.vla.vision_backbone.fused_featurizer = nn.Module()
                self.vla.vision_backbone.fused_featurizer.blocks = nn.ModuleList([nn.Linear(8, 8)])
                self.vla.projector = nn.Linear(8, 8)

        # Disabled mode is a clean no-op on every Python version
        off = compile_policy_trainer(_DummyPolicy(), enabled=False)
        self.assertFalse(off["compiled"])
        self.assertEqual(off["reason"], "disabled")

        if sys.version_info >= (3, 14):
            return

        policy = _DummyPolicy()
        before = [(n, tuple(p.shape)) for n, p in policy.named_parameters()]
        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict(os.environ, {"TORCHINDUCTOR_CACHE_DIR": str(Path(tmp) / "ind"),
                                     "TRITON_CACHE_DIR": str(Path(tmp) / "tri")}):
            receipt = compile_policy_trainer(policy, enabled=True, backend="inductor", rank=0)
            after = [(n, tuple(p.shape)) for n, p in policy.named_parameters()]
            self.assertEqual(before, after)
            self.assertTrue(receipt["compiled"])
            self.assertEqual(receipt["llama_layers"], 2)
            self.assertEqual(receipt["vision_blocks"], 2)
            self.assertTrue(receipt["projector_compiled"])

            # Compiled fwd+bwd at B=2 + force_eager variable-batch B=3 must both succeed
            x = torch.randn(2, 8, requires_grad=True)
            y = policy.vla.language_model.model.layers[0](x)
            y.sum().backward()
            self.assertIsNotNone(x.grad)
            with torch.no_grad(), eager_inference_context():
                y_var = policy.vla.language_model.model.layers[0](torch.randn(3, 8))
            self.assertEqual(tuple(y_var.shape), (3, 8))


if __name__ == "__main__":
    unittest.main()
