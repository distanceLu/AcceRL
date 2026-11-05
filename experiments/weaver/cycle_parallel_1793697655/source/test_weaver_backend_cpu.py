"""CPU integration contracts; tiny tensors, no downloaded weights or GPU claims.

The tiny upstream-shaped model executes the actual pinned WEAVER.forward,
_aggregate_losses and variable-horizon helper extracted from the audited file.
It is not a substitute model in the production loader.
"""
import ast
import asyncio
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None

from weaver_checkpoint import (PINNED_COMMIT, WeaverAdmissionError, admit_metadata,
    audit_state_dict, clean_state_dict, environment_report, verify_source)
from world_backends.weaver import WeaverBackend
from weaver_inference_actor import make_weaver_inference_actor

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = ROOT / "analysis/weaver_integration_20261022/upstream"
if os.environ.get('WEAVER_TEST_UPSTREAM'):
    UPSTREAM = Path(os.environ['WEAVER_TEST_UPSTREAM'])
if not UPSTREAM.is_dir():
    # The deployment uses the already-audited pinned checkout, while the local
    # workspace keeps its read-only upstream under analysis/.
    declared = json.loads((Path(__file__).parent/'examples/weaver_frozen_external.json').read_text())
    UPSTREAM = Path(declared['backend_options']['source_path'])


def config(integrated=False, memory=2):
    schema = {"domain": "LIBERO", "action_schema_id": "libero_eef_delta7_v1",
        "state_schema_id": "libero_eef_axisangle_gripper8_v1", "action_dim": 7,
        "state_dim": 8, "encoder_id": "tiny-cpu-test", "normalizer_id": "tiny-stats-v1",
        "alignment": "action_into_frame", "image_keys": ["external", "wrist"],
        "image_size": [4, 4], "image_value_range": "minus_one_one", "dt_seconds": 0.1,
        "n_history": 2, "n_memory_frames": memory, "t_memory": 2}
    metadata = {"checkpoint_id": "tiny-test-only", "code_revision": PINNED_COMMIT,
        "schema": copy.deepcopy(schema), "load_report": {"strict": True, "missing_keys": [],
        "unexpected_keys": [], "evidence_id": "synthetic-test-only"}}
    if integrated:
        metadata.update(success_head={"calibrated": True, "label_semantics": "environment_success",
            "calibration_id": "test-calibration", "source": "rm", "output": "probability"},
            terminal_head={"calibrated": True, "label_semantics": "environment_terminal",
            "calibration_id": "test-terminal", "source": "same_success", "rule": "argmax"})
    return {"backend": "weaver", "reward_provider": "world_head" if integrated else "external_vlm",
        "reward_semantics": "delta_success", "terminal_rule": "argmax", "critic_mode": "disabled",
        "schema": schema, "checkpoint_metadata": metadata,
        "backend_options": {"normalization": {"state": {"mean": [2.] * 8, "std": [2.] * 8},
                                                 "actions": {"mean": [1.] * 7, "std": [2.] * 7}},
                            "publication_weights": "online"}}


if torch is not None:
    def native_methods():
        parsed = ast.parse((UPSTREAM / "weaver/wm/model.py").read_text(encoding="utf-8"))
        model = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == "WEAVER")
        names = {"forward", "_aggregate_losses", "generate_latent_rollouts_variable_horizon"}
        methods = [n for n in model.body if isinstance(n, ast.FunctionDef) and n.name in names]
        ns = {"torch": torch, "Dict": dict, "Optional": __import__("typing").Optional,
              "rearrange": lambda x, pattern: x[:, :, None, None] if pattern == 'b t -> b t 1 1' else x[:, :, None]}
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(UPSTREAM / "weaver/wm/model.py"), "exec"), ns)
        return {name: ns[name] for name in names}

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("scale", torch.tensor(1.))
        def forward(self, x):
            return x.permute(0, 1, 3, 4, 2).flatten(2, 3) * self.scale
        def decode(self, x):
            return x.reshape(*x.shape[:2], 4, 4, 3).permute(0, 1, 4, 2, 3).clamp(0, 1)

    class Flow(torch.nn.Module):
        _n_actions = 7
        def __init__(self):
            super().__init__()
            self.gain = torch.nn.Parameter(torch.tensor(0.25))
            self.register_buffer("counter", torch.tensor(0, dtype=torch.int64))
        def forward(self, obs, actions, t, memory=None):
            return {key: value * self.gain + self.gain for key, value in obs.items()}

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(0.25))
            self.seen_detached = False
        def forward(self, obs, actions, tasks):
            return self.bias.expand(actions.shape[:2])
        def compute_loss(self, obs, actions, tasks, gt_rewards=None, rewards=None):
            self.seen_detached = all(not value.requires_grad for value in obs.values()) and not tasks.requires_grad
            prediction = self.forward(obs, actions, tasks)
            target = gt_rewards if gt_rewards is not None else rewards
            loss = (prediction - target).square().mean()
            return (prediction, loss) if gt_rewards is not None else loss

    class TinyUpstream(torch.nn.Module):
        def __init__(self, memory=2):
            super().__init__()
            self.im_encoder = Encoder()
            self.wm = Flow()
            self.rm = Head()
            self.critic = Head()
            self._img_keys = ["external", "wrist"]
            self._n_history, self._n_horizon, self._n_memory_frames, self._t_memory = 2, 3, memory, 2
            self._use_memory = memory > 0
            self._history_noise_std = 0
            self._use_temporal_loss = self._train_decoder = False
            self._flow_loss = "v-pred"
            self._state_loss_scale = .1
            self._rm_loss_coeff = self._critic_loss_coeff = 1.
            self._pyramid_schedule_cache = {("old",): "cache"}
            self.generate_calls = []
            self.memory_calls = []
        def encode_obs(self, obs):
            return {**{k: obs[k + "_features"] for k in self._img_keys}, "states": obs["states"]}
        def encode_memory_obs(self, memory):
            self.memory_calls.append(memory)
            return memory["states"].mean(dim=-1, keepdim=True)
        def encode_task(self, tasks):
            return tasks.get("features", torch.ones(len(tasks.get("text", [])), 2))
        def sample_noise(self, values):
            return {key: torch.zeros_like(value) for key, value in values.items()}
        def sample_timestep(self, b, t):
            return torch.ones(b, t) * .5
        def interpolate(self, x1, x0, t):
            return {key: value * .5 for key, value in x1.items()}
        def compute_flow_loss(self, x1, x0, pred, t):
            return (pred - x1).square()
        def generate_latent_rollouts(self, x1, actions, memory_tokens=None):
            self.generate_calls.append((x1, actions, memory_tokens))
            result = {key: value.clone() for key, value in x1.items()}
            for key in result:
                result[key][:, self._n_history:] = x1[key][:, self._n_history - 1:self._n_history] + .1
            return result
        def decode_obs(self, latents, chunk_size=0):
            return {**{k: self.im_encoder.decode(latents[k]) for k in self._img_keys}, "states": latents["states"]}

    for method_name, method in native_methods().items():
        setattr(TinyUpstream, method_name, method)


def sample(cfg, frames):
    return {"obs": torch.zeros(frames, 2, 3, 4, 4),
        "states": torch.arange(frames, dtype=torch.float32)[:, None].expand(frames, 8) + 2,
        "actions": torch.ones(frames, 7) * 3, "rewards": torch.zeros(frames),
        "success": torch.zeros(frames), "terminated": [False] * frames, "truncated": [False] * frames,
        "valid_mask": [True] * frames, "action_mask": [True] * frames, "dt": [0.1] * frames,
        "instruction": "pick up the cube", "schema": copy.deepcopy(cfg["schema"]),
        "sample_id": "test", "episode_id": "episode", "request_lineage": {"request_id": "q1"}}


class CheckpointAdmissionTests(unittest.TestCase):
    def test_pinned_real_source_is_checked_and_environment_read_only(self):
        self.assertEqual(verify_source(UPSTREAM), UPSTREAM.resolve())
        self.assertFalse(environment_report()["environment_modified"])

    def test_official_droid_config_cannot_be_labeled_libero(self):
        cfg = config()
        release = {"dataset": {"name": "DROID", "n_actions": 8, "n_states": 8,
                    "img_keys": ["wrist_left", "exterior_1_left"], "image_size": [192, 320]}}
        with self.assertRaisesRegex(WeaverAdmissionError, "DROID weights cannot be relabeled LIBERO"):
            admit_metadata(cfg, cfg["checkpoint_metadata"], release)

    def test_native_progress_never_admitted_as_success(self):
        cfg = config(True)
        cfg["checkpoint_metadata"]["success_head"]["label_semantics"] = "robometer_progress_minus_one"
        with self.assertRaisesRegex(WeaverAdmissionError, "native progress"):
            admit_metadata(cfg, cfg["checkpoint_metadata"])

    def test_missing_implicit_model_fallback_is_rejected(self):
        with self.assertRaisesRegex(WeaverAdmissionError, "Use WeaverBackend.load"):
            WeaverBackend(config())

    def test_checkpoint_normalized_key_collision_rejected(self):
        with self.assertRaisesRegex(WeaverAdmissionError, "collision"):
            clean_state_dict({"module.a": 1, "a": 2})

    def test_production_load_does_not_treat_metadata_as_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(FileNotFoundError, "metadata files are not weights"):
                WeaverBackend.load(UPSTREAM, folder, config=config())


@unittest.skipIf(torch is None, "CPU torch is not installed; no environment modifications allowed")
class BackendCPUIntegrationTests(unittest.TestCase):
    def backend(self, integrated=False, memory=2):
        cfg = config(integrated, memory)
        model = TinyUpstream(memory)
        return cfg, model, WeaverBackend(cfg, model=model)

    def test_strict_checkpoint_reports_names_and_shapes(self):
        expected = {"wm.weight": torch.zeros(2, 3), "rm.bias": torch.zeros(1)}
        actual = {"module.wm.weight": torch.zeros(2, 4), "surprise": torch.zeros(1)}
        with self.assertRaises(WeaverAdmissionError) as error:
            audit_state_dict(expected, actual)
        for value in ("rm.bias", "surprise", "wm.weight", "shape_mismatches"):
            self.assertIn(value, str(error.exception))
        clean, report = audit_state_dict(expected, {"module._orig_mod." + k: v for k, v in expected.items()})
        self.assertEqual(set(clean), set(expected))
        self.assertTrue(report["strict"])

    def test_real_upstream_forward_objectives_and_detach_boundaries(self):
        cfg, upstream, backend = self.backend(True)
        row = sample(cfg, backend.training_window_frames)
        row["success"][-1] = 1
        row["terminated"][-1] = True
        prepared = backend.prepare_train_batch([row, copy.deepcopy(row)])
        self.assertEqual(prepared["actions"].shape, (2, 5, 7))
        self.assertTrue(torch.equal(prepared["actions"], torch.ones(2, 5, 7)))
        self.assertEqual(prepared["memory"]["states"].shape, (2, 2, 8))
        loss, logs = backend.model(prepared)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(upstream.wm.gain.grad)
        self.assertIsNotNone(upstream.rm.bias.grad)
        self.assertIsNone(upstream.critic.bias.grad)
        self.assertTrue(upstream.rm.seen_detached)
        self.assertIn("flow/Total Loss", logs)
        self.assertIn("loss/RM", logs)
        self.assertEqual(float(logs["Total Loss"]), float(loss.detach()))

    def test_external_reward_has_no_hidden_reward_or_critic_optimizer_update(self):
        cfg, upstream, backend = self.backend(False)
        prepared = backend.prepare_train_batch(sample(cfg, backend.training_window_frames))
        self.assertFalse(prepared["update_rm"])
        loss, _ = backend.model(prepared)
        loss.backward()
        self.assertIsNone(upstream.rm.bias.grad)
        self.assertFalse(upstream.rm.bias.requires_grad)
        self.assertIsNone(upstream.critic.bias.grad)

    def test_real_variable_horizon_helper_and_memory_never_advance_padded_steps(self):
        cfg, upstream, backend = self.backend(True)
        state = backend.init_rollout(sample(cfg, backend.required_context_frames), 7)
        self.assertEqual(state["initial_success_probability"], .25)
        out = backend.predict_chunk(state, torch.ones(2, 7) * 3, 7)
        self.assertEqual(out["observations"].shape, (1, 2, 2, 3, 4, 4))
        self.assertEqual(out["predicted_states"].shape, (1, 2, 8))
        self.assertEqual(upstream.generate_calls[-1][1].shape[1], 4)
        self.assertEqual(out["next_state"]["predicted_steps"], 2)
        self.assertTrue(torch.equal(out["success_probabilities"], torch.full((1, 2), .25)))
        self.assertEqual(out["observations"].device.type, "cpu")
        self.assertTrue(torch.allclose(out["predicted_states"], torch.full((1, 2, 8), 7.2)))
        second = backend.predict_chunk(out["next_state"], torch.ones(1, 7) * 3, 7)
        self.assertEqual(second["next_state"]["predicted_steps"], 3)
        self.assertEqual(upstream.generate_calls[-1][1].shape[1], 3)
        self.assertEqual(upstream.memory_calls[-1]["states"].shape[1], 2)

    def test_version_change_invalidates_opaque_history_and_caches(self):
        cfg, upstream, backend = self.backend()
        state = backend.init_rollout(sample(cfg, backend.required_context_frames), 1)
        upstream._pyramid_schedule_cache["cached"] = "value"
        backend.set_published_version(2)
        self.assertEqual(upstream._pyramid_schedule_cache, {})
        with self.assertRaisesRegex(WeaverAdmissionError, "Stale"):
            backend.predict_chunk(state, torch.zeros(1, 7), 2)

    def test_all_mutable_head_parameters_and_buffers_are_published(self):
        _, _, backend = self.backend(True)
        inventory = backend.publication_inventory()
        self.assertIn("upstream.rm.bias", inventory["parameter_names"])
        self.assertIn("upstream.wm.counter", inventory["buffer_names"])
        self.assertIn("actions_std", inventory["buffer_names"])
        selected = backend.publication_tensors()
        self.assertEqual(set(selected), set(inventory["parameter_names"] + inventory["buffer_names"]))

    def test_ema_publication_uses_shadow_references_without_model_swap(self):
        cfg = config(True)
        cfg["backend_options"]["publication_weights"] = "ema"
        upstream = TinyUpstream()
        shadows = [p.detach().clone() + 1 for p in upstream.parameters() if p.requires_grad]
        upstream.ema = SimpleNamespace(ema=SimpleNamespace(shadow_params=shadows))
        original = upstream.wm.gain.detach().clone()
        backend = WeaverBackend(cfg, model=upstream)
        selected = backend.publication_tensors()
        self.assertIs(selected["upstream.wm.gain"], shadows[0])
        self.assertTrue(torch.equal(upstream.wm.gain, original))

    def test_invalid_state_schema_masks_time_and_boundary_fail_before_forward(self):
        cfg, _, backend = self.backend()
        cases = []
        missing = sample(cfg, backend.training_window_frames); missing["states"] = None; cases.append(missing)
        masked = sample(cfg, backend.training_window_frames); masked["valid_mask"][-1] = False; cases.append(masked)
        wrong_dt = sample(cfg, backend.training_window_frames); wrong_dt["dt"][-1] = .2; cases.append(wrong_dt)
        terminal = sample(cfg, backend.training_window_frames); terminal["terminated"][-2] = True; cases.append(terminal)
        wrong_schema = sample(cfg, backend.training_window_frames); wrong_schema["schema"]["state_schema_id"] = "joint_position"; cases.append(wrong_schema)
        for value in cases:
            with self.subTest(value=value.keys()), self.assertRaises(WeaverAdmissionError):
                backend.prepare_train_batch(value)

    def test_out_of_range_success_is_never_clipped_or_silently_reinterpreted(self):
        cfg, upstream, backend = self.backend(True)
        state = backend.init_rollout(sample(cfg, backend.required_context_frames), 1)
        with torch.no_grad():
            upstream.rm.bias.fill_(-.2)
        with self.assertRaisesRegex(RuntimeError, "invalid probabilities"):
            backend.predict_chunk(state, torch.ones(1, 7), 1)

    def test_reset_dt_sentinel_and_terminal_label_supervision(self):
        cfg, _, backend = self.backend(True)
        row = sample(cfg, backend.training_window_frames)
        row["action_mask"][0] = False
        row["dt"][0] = 0
        backend.prepare_train_batch(row)
        row["success"][-1] = 1
        with self.assertRaisesRegex(WeaverAdmissionError, "labels to agree"):
            backend.prepare_train_batch(row)

    def test_native_progress_training_requires_native_labels(self):
        cfg = config()
        cfg["backend_options"]["train_reward_head"] = True
        backend = WeaverBackend(cfg, model=TinyUpstream())
        with self.assertRaisesRegex(WeaverAdmissionError, "sparse environment rewards"):
            backend.prepare_train_batch(sample(cfg, backend.training_window_frames))


@unittest.skipIf(torch is None, "CPU torch is not installed; no environment modifications allowed")
class InferenceActorCPUTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        cfg = config(True)
        cfg["backend_options"].update(source_path="test-only", checkpoint_dir="test-only", horizon=3)
        self.config = cfg
        self.backend = WeaverBackend(cfg, model=TinyUpstream())
        self.events = []
        events = self.events
        class Mixin:
            def _check_inference_health(self):
                if getattr(self, "inference_fatal_error", None) is not None:
                    raise RuntimeError("background failure")
            def _fail_pending_requests(self, error):
                self.inference_fatal_error = error
                for promise in self.promises:
                    if not promise.done(): promise.set_exception(error)
                self.requests, self.promises = [], []
        class Com:
            pass
        runtime = {"setup_debugger": lambda *a: None, "verify_bound_gpu": lambda: {"uuid": "cpu-test"},
            "_weaver_backend_loader": lambda *a, **kw: self.backend,
            "batch_begin": lambda owner, requests, promises: events.append(("batch", len(requests))) or {},
            "batch_end": lambda *a, **kw: events.append(("end", kw.get("error"))),
            "stage_begin": lambda *a, **kw: events.append(("stage", kw["actual_batch"])) or {},
            "stage_end": lambda *a, **kw: events.append(("stage_end", kw.get("error"))),
            "_trim_cuda_cache": lambda *a: None}
        actor_cls = make_weaver_inference_actor(Mixin, Com, runtime)
        stats = SimpleNamespace(add_timing_metric=SimpleNamespace(remote=lambda *a: None))
        self.actor = actor_cls(0, stats, SimpleNamespace(world_backend_config=cfg), 16, 50)

    async def asyncTearDown(self):
        self.actor._bg_task.cancel()
        await asyncio.gather(self.actor._bg_task, return_exceptions=True)

    async def test_queue_reports_actual_single_request_and_returns_cpu_unbatched_results(self):
        state = self.actor.init_rollout(sample(self.config, self.backend.required_context_frames))
        results = await asyncio.gather(self.actor.request(state, torch.ones(2, 7), state["instruction"]),
                                       self.actor.request(state, torch.ones(1, 7), state["instruction"]))
        self.assertEqual([tuple(x["observations"].shape) for x in results], [(2, 2, 3, 4, 4), (1, 2, 3, 4, 4)])
        self.assertEqual([event for event in self.events if event[0] == "batch"], [("batch", 1), ("batch", 1)])
        self.assertEqual(self.actor.get_context_geometry()["horizon"], 3)

    async def test_publication_syncs_operations_and_heldout_evaluation_restores_rng(self):
        self.actor.capture_model_version = 4
        state = self.actor.init_rollout(sample(self.config, self.backend.required_context_frames))
        self.assertEqual(state["version"], 4)
        rng = torch.random.get_rng_state().clone()
        result = self.actor.evaluate_backend_prediction_metrics([sample(self.config, self.backend.training_window_frames)], seed=123)
        self.assertEqual(result["predicted_frames"], 3)
        self.assertEqual(result["served_version"], 4)
        self.assertGreaterEqual(result["image_mse"], 0)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    async def test_stale_queued_state_does_not_kill_service_after_publication(self):
        context = sample(self.config, self.backend.required_context_frames)
        old_state = self.actor.init_rollout(context)
        self.actor.capture_model_version = 1
        with self.assertRaisesRegex(ValueError, "Stale WEAVER"):
            await self.actor.request(old_state, torch.ones(1, 7))
        new_state = self.actor.init_rollout(context)
        output = await self.actor.request(new_state, torch.ones(1, 7))
        self.assertEqual(output["served_version"], 1)
        self.assertFalse(self.actor._bg_task.done())
        self.assertEqual([event[0] for event in self.events],
                         ["batch", "stage", "stage_end", "end"] * 2)
        self.assertIsInstance(self.events[2][1], ValueError)
        self.assertIsNone(self.events[6][1])

    async def test_reply_exception_traceback_never_contains_live_background_coroutine(self):
        import traceback
        context = sample(self.config, self.backend.required_context_frames)
        state = self.actor.init_rollout(context)
        self.actor.capture_model_version = 1
        delivered = None
        try:
            await self.actor.request(state, torch.ones(1, 7))
        except ValueError as error:
            delivered = error
            names = [frame.name for frame in traceback.extract_tb(error.__traceback__)]
            self.assertNotIn('_loop', names)
            self.assertIn('_loop', error.origin_traceback)
            # Reproduce the actual consumer behavior that closed the live
            # coroutine in Python 3.11 when exceptions crossed by identity.
            traceback.clear_frames(error.__traceback__)
        self.assertIsNotNone(delivered)
        new_state = self.actor.init_rollout(context)
        output = await self.actor.request(new_state, torch.ones(1, 7))
        self.assertEqual(output['served_version'], 1)
        self.assertFalse(self.actor._bg_task.done())


if __name__ == "__main__":
    unittest.main()
