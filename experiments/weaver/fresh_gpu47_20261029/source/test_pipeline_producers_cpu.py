"""Execute actual producer/GAE code with bounded fake RPCs and a tiny CPU env.

No Ray/Torch/model/simulator import is required. The fake imagination kernel is
used only to exercise orchestration; actual recipe preservation is checked
against the pre-change world_backends_v1 source saved as test fixtures.
"""
from __future__ import annotations

import ast
from collections import deque
import copy
import hashlib
import math
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np
from pipeline_budget import PipelineBudgetConfig
from fresh_config import FreshConfig
from pipeline_runtime import run_producer


HERE = Path(__file__).resolve().parent


def parse(path):
    return ast.parse(path.read_text(encoding="utf-8"))


def producer_methods():
    cls = next(n for n in parse(HERE / "train.py").body
               if isinstance(n, ast.ClassDef) and n.name == "RolloutWorkerActor")
    return {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}


def baseline_path(kind):
    fixture = HERE / "tests" / "fixtures" / ("baseline_train.py" if kind == "train" else "baseline_world_rollout.py")
    if fixture.exists():
        return fixture
    return HERE.parent / "world_backends_v1" / ("train.py" if kind == "train" else "world_backends/rollout.py")


def baseline_methods():
    cls = next(n for n in parse(baseline_path("train")).body
               if isinstance(n, ast.ClassDef) and n.name == "RolloutWorkerActor")
    return {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}


def fingerprint(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


class FakeRef:
    def __init__(self, name, fn):
        self.name, self.fn = name, fn


class FakeRay:
    def __init__(self, events):
        self.events = events

    def get(self, ref):
        if not isinstance(ref, FakeRef):
            raise AssertionError("test RPC must be explicitly awaited")
        result = ref.fn()
        self.events.append("ack:" + ref.name)
        return result


class Remote:
    def __init__(self, events, name, fn):
        self.events, self.name, self.fn = events, name, fn

    def remote(self, *args, **kwargs):
        self.events.append("submit:" + self.name)
        return FakeRef(self.name, lambda: self.fn(*args, **kwargs))


class TinyEnv:
    max_episode_steps = 2
    task_description = "tiny observed task"

    def __init__(self):
        self.steps = 0

    def observation(self):
        return {"full_image": np.full((8, 8, 3), self.steps, np.uint8),
                "wrist_image": np.full((8, 8, 3), self.steps + 10, np.uint8)}

    def reset(self, seed=None):
        self.steps = 0
        return self.observation(), {}

    def get_name(self):
        return "tiny-env"

    def step(self, action):
        self.steps += 1
        terminal = self.steps >= self.max_episode_steps
        return self.observation(), float(terminal), terminal, False, {"is_success": terminal}


def make_worker(events, replay, wm_replay, *, mode="imagine", backend="ctrl",
                base_must_not_run=False):
    """Compile selected unmodified methods; imports/models in train.py stay out."""
    fake_ray = FakeRay(events)
    class Base:
        def __init__(self, infer, replay, wid, stats_actor, cfg, benchmark_name):
            if base_must_not_run:
                raise AssertionError("imagination initialized a simulator-owning base")
            events.append("base_env_init")
            self.infer, self.replay, self.wid = infer, replay, wid
            self.stats_actor, self.cfg, self.processor = stats_actor, cfg, "processor"
            self.num_tasks, self.envs = 1, [TinyEnv()]
            self.current_env_idx = -1

    def rpc(worker, role, actor, method, *args, **kwargs):
        events.append("inference:" + method)
        if method == "get_num_history":
            return 6
        if method == "get_stride":
            return 4
        if method == "get_context_geometry":
            return {"required_context_frames": 7, "horizon": 4, "training_window_frames": 11}
        if method == "request":
            return None, np.ones((2, 7), np.float32), None, None, 0.
        raise AssertionError(method)

    def align(actions, frames):
        # This external helper is a seam, not the subject of these tests. Frame
        # storage tests separately validate actual canonical action alignment.
        if len(actions) != frames - 1:
            raise AssertionError("collector did not produce one action per transition")
        return np.concatenate([np.zeros((1, actions.shape[1]), np.float32), actions])

    def experience(**kwargs):
        return SimpleNamespace(sample_id=f"{kwargs['episode_id']}:{kwargs['sample_index']}", **kwargs)

    names = ("__init__", "_reset_and_select_env", "get_one_episode", "_process_episode_for_wm", "_process_traj")
    methods = producer_methods()
    node = ast.ClassDef(name="TestProducer", bases=[ast.Name(id="BaseWorkerActor", ctx=ast.Load())],
                        keywords=[], body=[copy.deepcopy(methods[name]) for name in names], decorator_list=[])
    env = dict(BaseWorkerActor=Base, PipelineBudgetConfig=PipelineBudgetConfig, FreshConfig=FreshConfig,
               get_processor=lambda cfg: "processor", setup_debugger=lambda *args: None,
               NUM_ACTIONS_CHUNK=8, deque=deque, np=np, os=os, time=time, math=math,
               WorldBackendConfig=SimpleNamespace(from_mapping=lambda mapping: SimpleNamespace(
                   backend=mapping["backend"], reward_provider="external_vlm", world_trainable=True, schema={"dt_seconds": .05})),
               profile_rpc=rpc, prepare_one_obs=lambda *args: "prepared-observation",
               lineage_begin=lambda *args, **kwargs: None, lineage_end=lambda *args: "lineage",
               lineage_frame_parent=lambda *args: (), lineage_sample=lambda *args, **kwargs: None,
               frame_align_transition_actions=align, ray=fake_ray, Experience=experience)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 str(HERE / "train.py"), "exec"), env)
    stats = SimpleNamespace(add_episode_return=Remote(events, "stats", lambda *args, **kwargs: None))
    worker = env["TestProducer"](infer="existing-policy-actor", replay=replay, wid=4 if mode == "imagine" else 0,
        stats_actor=stats, cfg=SimpleNamespace(), benchmark_name="libero_spatial", num_step_cond=7,
        imagine_horizon=32, torch_dtype="unused-cpu-test", reward_infer="existing-reward-actor",
        ctrl_world_infer="existing-world-actor", reward_scale=1., gamma=.9, lambda_=.8,
        wm_replay_buffer=wm_replay, real_traj_collect_interval=1,
        world_config={"backend": backend}, worker_mode=mode)
    return worker, fake_ray


def trajectory():
    return [("observation", np.array([1]), reward, np.array([.1, .2]), value,
             {"lineage_group_id": "group"}) for reward, value in ((1., .1), (2., .3))]


class ProducerBehaviorTests(unittest.TestCase):
    def harness(self, mode, *, backend="ctrl", error=None, fail_add=False, backpressure=True):
        events, batches, payloads = [], [], []
        jobs = iter(([None] if backpressure else []) + [{"token": "lease", "episode_id": "seed-episode"},
                                                        {"stopped": True}])
        def acquire(_):
            return next(jobs)
        def add(batch):
            if fail_add:
                raise RuntimeError("Replay insertion failed")
            self.assertEqual(getattr(worker, "pipeline_generated_samples", 0), 0)
            batches.append(batch)
        def finish_imagine(token, count):
            self.assertEqual(token, "lease")
            self.assertIn("ack:policy_add", events)
            self.assertEqual(count, 2)
        def finish_real(token, payload):
            self.assertEqual(token, "lease")
            payloads.append(payload)
        wm_replay = SimpleNamespace(
            acquire_real=Remote(events, "acquire_real", acquire),
            acquire_imagine=Remote(events, "acquire_imagine", acquire),
            finish_real_episode=Remote(events, "finish_real", finish_real),
            finish_imagine=Remote(events, "finish_imagine", finish_imagine),
            cancel_real=Remote(events, "cancel_real", lambda token: None),
            cancel_imagine=Remote(events, "cancel_imagine", lambda token: None))
        replay = SimpleNamespace(add_batch=Remote(events, "policy_add", add))
        worker, fake_ray = make_worker(events, replay, wm_replay, mode=mode, backend=backend,
                                       base_must_not_run=(mode == "imagine"))
        def imagine(worker, job):
            events.append("imagine_kernel:" + backend)
            if error:
                raise error
            worker.capture_episode_id = job["episode_id"]
            worker._process_traj(trajectory(), .4)
        fake_kernels = SimpleNamespace(imagine_ctrl_seed=imagine, imagine_weaver_seed=imagine)
        return worker, fake_ray, fake_kernels, events, batches, payloads

    def test_imagination_initialization_owns_neither_simulator_nor_model(self):
        for backend in ("ctrl", "weaver"):
            with self.subTest(backend=backend):
                worker, _, _, events, _, _ = self.harness("imagine", backend=backend)
                self.assertEqual(worker.envs, [])
                self.assertNotIn("base_env_init", events)
                self.assertEqual(worker.infer, "existing-policy-actor")
                self.assertEqual(worker.ctrl_world_infer, "existing-world-actor")
                self.assertFalse(hasattr(worker, "model"))

    def test_real_collection_backpressure_ingestion_ack_then_next_lease(self):
        worker, fake_ray, kernels, events, _, payloads = self.harness("real")
        with patch.dict(sys.modules, {"ray": fake_ray, "pipeline_imagination": kernels}), \
             patch("pipeline_runtime.time.sleep", side_effect=lambda delay: events.append("poll_wait")), \
             patch.dict(os.environ, {"ACCERL_SEED": "17"}):
            run_producer(worker)
        self.assertIn("poll_wait", events)
        self.assertNotIn("submit:acquire_imagine", events)
        self.assertFalse(any(event.startswith("imagine_kernel") for event in events))
        ack = events.index("ack:finish_real")
        self.assertEqual(events[ack + 1], "submit:acquire_real")
        self.assertIsNone(worker.pipeline_lease)
        self.assertEqual(len(payloads), 1)
        payload = payloads[0]
        self.assertEqual(payload["frames"].shape, (3, 2, 3, 224, 224))
        self.assertEqual(payload["frames"].dtype, np.uint8)
        np.testing.assert_array_equal(payload["actions"][0], np.zeros(7))
        np.testing.assert_array_equal(payload["actions"][1:], np.ones((2, 7)))
        np.testing.assert_array_equal(payload["terminated"], [False, False, True])
        np.testing.assert_array_equal(payload["truncated"], [False, False, False])
        np.testing.assert_array_equal(payload["success"], [False, False, True])
        np.testing.assert_array_equal(payload["rewards"], [0., 0., 1.])

    def test_imagination_waits_for_policy_add_ack_before_production_credit(self):
        for backend in ("ctrl", "weaver"):
            with self.subTest(backend=backend):
                worker, fake_ray, kernels, events, batches, _ = self.harness("imagine", backend=backend)
                worker.get_one_episode = lambda: self.fail("imagination called real collector")
                with patch.dict(sys.modules, {"ray": fake_ray, "pipeline_imagination": kernels}), \
                     patch("pipeline_runtime.time.sleep", side_effect=lambda delay: events.append("poll_wait")):
                    run_producer(worker)
                self.assertNotIn("submit:acquire_real", events)
                self.assertIn("poll_wait", events)
                self.assertLess(events.index("ack:policy_add"), events.index("submit:finish_imagine"))
                self.assertEqual(worker.pipeline_generated_samples, 2)
                self.assertIsNone(worker.pipeline_lease)
                self.assertAlmostEqual(batches[0][0].advantage, 2.6532, places=5)
                self.assertAlmostEqual(batches[0][1].advantage, 2.06, places=5)
                self.assertAlmostEqual(batches[0][0].value_target, 2.7532, places=6)
                self.assertAlmostEqual(batches[0][1].value_target, 2.36, places=6)

    def test_imagination_failure_cancels_once_and_preserves_exception(self):
        for error, fail_add in ((RuntimeError("prediction failed"), False), (None, True)):
            worker, fake_ray, kernels, events, _, _ = self.harness("imagine", error=error,
                                                                fail_add=fail_add, backpressure=False)
            with patch.dict(sys.modules, {"ray": fake_ray, "pipeline_imagination": kernels}):
                with self.assertRaisesRegex(RuntimeError, "prediction failed|Replay insertion failed"):
                    run_producer(worker)
            self.assertEqual(events.count("ack:cancel_imagine"), 1)
            self.assertNotIn("submit:finish_imagine", events)
            self.assertEqual(worker.pipeline_generated_samples, 0)
            self.assertIsNone(worker.pipeline_lease)

    def test_environment_exceeding_frame_budget_fails_and_cancels(self):
        worker, fake_ray, kernels, events, _, payloads = self.harness("real", backpressure=False)
        worker.pipeline_max_episode_frames = 2
        with patch.dict(sys.modules, {"ray": fake_ray, "pipeline_imagination": kernels}):
            with self.assertRaisesRegex(RuntimeError, "reserved episode frame budget"):
                run_producer(worker)
        self.assertEqual(payloads, [])
        self.assertIn("ack:cancel_real", events)
        self.assertIsNone(worker.pipeline_lease)

    def test_stop_without_lease_does_no_work(self):
        for mode in ("real", "imagine"):
            events = []
            remote = Remote(events, "stop", lambda wid: {"stopped": True})
            worker = SimpleNamespace(worker_mode=mode, wid=0, wm_replay_buffer=SimpleNamespace(
                acquire_real=remote, acquire_imagine=remote, cancel_real=None, cancel_imagine=None))
            with patch.dict(sys.modules, {"ray": FakeRay(events)}):
                run_producer(worker)
            self.assertEqual(events, ["submit:stop", "ack:stop"])

    def test_cancellation_error_does_not_hide_original_worker_error(self):
        worker, fake_ray, kernels, events, _, _ = self.harness("imagine", error=RuntimeError("original"),
                                                            backpressure=False)
        def broken_cancel(token):
            raise ValueError("lease revoked")
        worker.wm_replay_buffer.cancel_imagine = Remote(events, "cancel_imagine", broken_cancel)
        with patch.dict(sys.modules, {"ray": fake_ray, "pipeline_imagination": kernels}):
            with self.assertRaisesRegex(RuntimeError, "original"):
                run_producer(worker)
        self.assertIsNone(worker.pipeline_lease)


class LearningPreservationTests(unittest.TestCase):
    # Saved from world_backends_v1 before this candidate's edits. AST removes
    # comments/location differences; only explicit transport changes are omitted.
    def test_gae_targets_and_experience_construction_match_prechange_source(self):
        method = producer_methods()["_process_traj"]
        start = next(i for i, statement in enumerate(method.body) if isinstance(statement, ast.Assign)
                     and ast.unparse(statement.targets[0]) == '(rets, advs)')
        class WithoutOptionalSignal(ast.NodeTransformer):
            def visit_If(self, node):
                if ast.unparse(node.test) == 'fresh.signal_telemetry':
                    return []
                return self.generic_visit(node)
        prefix = WithoutOptionalSignal().visit(ast.Module(body=copy.deepcopy(method.body[start:-2]), type_ignores=[]))
        old_prefix = ast.Module(body=baseline_methods()["_process_traj"].body[:-1], type_ignores=[])
        self.assertEqual(fingerprint(prefix), fingerprint(old_prefix))
        self.assertEqual(ast.unparse(method.body[-2]), "ray.get(self.replay.add_batch.remote(batch))")
        self.assertEqual(ast.unparse(method.body[-1]),
                         "self.pipeline_generated_samples = getattr(self, 'pipeline_generated_samples', 0) + len(batch)")

    def test_inference_input_and_reward_recipes_match_prechange_source(self):
        methods, original = producer_methods(), baseline_methods()
        for name in ("predict_next_chunk", "predict_rew_end", "obs2inp"):
            with self.subTest(name=name):
                self.assertEqual(fingerprint(methods[name]), fingerprint(original[name]))

    def test_ctrl_transition_reward_bootstrap_recipe_matches_prechange_source(self):
        function = next(n for n in parse(HERE / "pipeline_imagination.py").body
                        if isinstance(n, ast.FunctionDef) and n.name == "imagine_ctrl_seed")
        index = next(i for i, node in enumerate(function.body) if isinstance(node, ast.Assign)
                     and ast.unparse(node.targets[0]) == "last_succ_prob")
        seed = next(node for node in ast.walk(baseline_methods()["run"])
                    if isinstance(node, ast.For) and ast.unparse(node.target) == "i"
                    and "self.num_step_cond" in ast.unparse(node.iter))
        original_index = next(i for i, node in enumerate(seed.body) if isinstance(node, ast.Assign)
                              and ast.unparse(node.targets[0]) == "last_succ_prob")
        self.assertEqual(fingerprint(ast.Module(body=function.body[index:], type_ignores=[])),
                         fingerprint(ast.Module(body=seed.body[original_index:], type_ignores=[])))

    def test_weaver_transition_recipe_matches_prechange_source_with_seed_return(self):
        # The pinned production v2 uses an owned resident wrapper. Exercise that
        # actual path rather than assuming the pre-resident function body layout.
        from test_integrated_semantics_cpu import ProductionImagine
        calls, batch = ProductionImagine().run_seed()
        self.assertEqual(len(batch), 2)
        self.assertAlmostEqual(batch[1].value_target, .56, places=6)
        self.assertAlmostEqual(batch[0].value_target, .6212, places=6)
        self.assertTrue(any(call[-1] == 'imagined_bootstrap' for call in calls))
        self.assertFalse(any(call[0] == 'reward_infer' for call in calls))



if __name__ == "__main__":
    unittest.main()
