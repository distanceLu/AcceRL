import ast
from dataclasses import replace
import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

from test_world_contracts_cpu import schema
from world_backends.rollout import consume_predicted_chunk, real_context_payload
from world_backends.runtime import service_plan_from_state
from world_backends import build_service_plan


class RolloutBoundaryTests(unittest.TestCase):
    def test_real_context_masks_reset_without_inventing_state(self):
        states = [{"state": [float(i)] * 8} for i in range(4)]
        actions = [[float(i + 1)] * 7 for i in range(3)]
        context = real_context_payload(["first", "second"], states, actions, 0, schema(), "task", "e1", {"seed": "s1"})
        self.assertEqual(context["actions"], [[0.] * 7, [1.] * 7])
        self.assertEqual(context["action_mask"], [False, True])
        self.assertEqual(context["dt"], [0., .05])
        self.assertEqual(context["states"], [[0.] * 8, [1.] * 8])
        context = real_context_payload(["second", "third"], states, actions, 1, schema(), "task", "e1", {})
        self.assertEqual(context["actions"], [[1.] * 7, [2.] * 7])
        self.assertEqual(context["action_mask"], [True, True])

    def test_missing_real_state_is_not_zero_filled(self):
        with self.assertRaisesRegex(ValueError, "measured state"):
            real_context_payload(["first"], [{}], [], 0, schema(), "task", "e1", {})

    def test_integrated_stops_at_first_terminal(self):
        result = {"observations": ["a", "b", "c"], "predicted_states": [1, 2, 3],
                  "success_probabilities": [.3, .8, .1], "terminal_predictions": [False, True, False]}
        reward, probability, terminal, obs, state, consumed = consume_predicted_chunk(result, .2, 2.)
        self.assertAlmostEqual(reward, 1.2)
        self.assertEqual((probability, terminal, obs, state, consumed), (.8, True, "b", 2, 2))

    def test_external_uses_unchanged_success_difference(self):
        seen = []
        def predict(obs):
            seen.append(obs)
            return {"a": (.3, 0), "b": (.5, 0)}[obs]
        result = {"observations": ["a", "b"], "predicted_states": [1, 2]}
        reward, probability, terminal, _, _, consumed = consume_predicted_chunk(result, .2, 1., predict)
        self.assertAlmostEqual(reward, .3)
        self.assertEqual((probability, terminal, consumed, seen), (.5, False, 2, ["a", "b"]))

    def test_saved_plan_rejects_ghost_reward_owner_counts(self):
        plan = build_service_plan({"backend": "weaver", "reward_provider": "world_head"}).to_dict()
        self.assertEqual(service_plan_from_state({"service_plan": plan}).expected_gpu_model_processes, 16)
        plan["expected_gpu_model_processes"] = 24
        with self.assertRaisesRegex(ValueError, "active owners"):
            service_plan_from_state({"service_plan": plan})

    def test_legacy_state_ignores_ambient_config(self):
        self.assertEqual(service_plan_from_state({}).trainer_kinds, ("policy", "ctrl", "reward"))


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("numpy"),
                     "Full rollout seam test requires existing CPU torch and numpy")
class FullWorkerLoopTests(unittest.TestCase):
    """Exercise the real worker GAE and observation/reward methods without Ray."""
    def run_worker(self, *, integrated=True, terminal=False, stale=False, threshold=None, pipeline=False):
        import numpy as np
        import torch
        from test_world_contracts_cpu import weaver_config
        from world_backends.rollout import run_weaver_rollouts

        config = weaver_config(integrated)
        if threshold is not None:
            config = replace(config, terminal_rule="success_probability", terminal_threshold=threshold)
        config.schema.update({"n_history": 2, "n_memory_frames": 0, "t_memory": 1,
                              "image_size": [2, 2]})
        calls, groups, batches, stats = [], [], [], []

        class FinishedTest(Exception):
            pass

        def rpc(worker, role, endpoint, method, *args, **kwargs):
            capture = kwargs.pop("_capture_context", {})
            calls.append((role, method, capture.get("workflow")))
            return getattr(endpoint, method)(*args, **kwargs)

        hooks = types.ModuleType("capture_hooks")
        hooks.profile_rpc = rpc
        hooks.lineage_begin = lambda *args, **kwargs: None
        def end(*args):
            groups.append("group%d" % len(groups))
            return groups[-1]
        hooks.lineage_end = end
        hooks.lineage_frame_parent = lambda frame_groups, frame: ("real-parent",)
        hooks.lineage_sample = lambda *args, **kwargs: None

        class Experience:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
                self.sample_id = "sample-%d" % len(batches)

        # Compile only existing worker methods, avoiding all train.py imports
        # and retaining its actual GAE/returns implementation under test.
        source = Path(__file__).with_name("train.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        worker_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RolloutWorkerActor")
        methods = [node for node in worker_class.body if isinstance(node, ast.FunctionDef) and node.name in
                   ("_process_traj", "predict_rew_end", "obs2inp", "_obs_to_views")]
        namespace = {"np": np, "torch": torch, "Experience": Experience, "profile_rpc": rpc,
                     "ray": types.SimpleNamespace(get=lambda value: value),
                     "lineage_begin": hooks.lineage_begin, "lineage_end": hooks.lineage_end,
                     "lineage_sample": hooks.lineage_sample, "tensor_to_image": lambda value: value,
                     "image_to_tensor": lambda value, device: value,
                     "prepare_one_obs": lambda cfg, processor, obs, task, dtype: obs}
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(Path(__file__).with_name("train.py")), "exec"), namespace)

        class Policy:
            def request(self, inputs, deterministic=False):
                bootstrap = calls[-1][2] == "imagined_bootstrap"
                return None, np.zeros((2, 7), dtype=np.float32), np.array([2]), np.zeros(3), .4 if bootstrap else .1

        class Reward:
            def request(self, inputs):
                p = float(inputs["full_image"][0, 0, 0])
                return torch.tensor([math.log(1. - p), math.log(p)], dtype=torch.float64)

        class World:
            version = 7
            emitted = 0
            rejected_stale = False
            def get_context_geometry(self):
                return {"required_context_frames": 2, "training_window_frames": 4, "horizon": 2}
            def init_rollout(self, context):
                self.emitted = 0
                return {"backend": "weaver", "version": self.version,
                        "initial_success_probability": .2, "instruction": context["instruction"]}
            def request(self, state, actions, instruction):
                if stale and not self.rejected_stale:
                    self.rejected_stale = True
                    self.version = 8
                    raise ValueError("Stale WEAVER rollout/cache version; initialize from real context after publication")
                probabilities = ([.8, .1] if terminal else
                                 ([.3, .4, .5, .6] if integrated else [.25, .3, .35, .4]))
                probabilities = probabilities[self.emitted:self.emitted + len(actions)]
                self.emitted += len(actions)
                observations = torch.stack([torch.full((2, 3, 2, 2), value) for value in probabilities])
                return {"observations": observations, "predicted_states": torch.zeros(len(actions), 8),
                        "next_state": {**state, "version": self.version}, "served_version": self.version,
                        "success_probabilities": probabilities,
                        "terminal_predictions": [bool(terminal and i == 0) for i in range(len(actions))]}

        class Worker:
            pass
        for name in ("_process_traj", "predict_rew_end", "obs2inp", "_obs_to_views"):
            setattr(Worker, name, namespace[name])
        worker = Worker()
        # The production method now explicitly reads FRESH configuration. These
        # legacy-loop fixtures intentionally exercise the disabled FRESH path.
        from fresh_config import FreshConfig
        worker.fresh_config = FreshConfig()
        worker.world_config, worker.num_step_cond, worker.num_policy_chunks = config, 2, 2
        worker.ctrl_world_infer_stride = 2
        worker.real_traj_collect_interval, worker.wid = 1, 0
        worker.gamma, worker.lambda_, worker.reward_scale = .9, .8, 1.
        worker.cfg, worker.processor, worker.torch_dtype = None, None, torch.float32
        worker.infer, worker.reward_infer, worker.ctrl_world_infer = Policy(), Reward(), World()
        worker.local_buffer, worker.episodes, worker.capture_last_versions = [], [], {}
        worker.replay = types.SimpleNamespace(add_batch=types.SimpleNamespace(remote=lambda batch: batches.append(batch)))
        worker.stats_actor = types.SimpleNamespace(add_imagine_reward=types.SimpleNamespace(remote=lambda *args: stats.append(args)))
        collected = []
        def collect():
            if collected:
                raise FinishedTest()
            collected.append(True)
            frames = 4 if stale else 3
            observations = [{"state": np.zeros(8), "full_image": torch.full((3, 2, 2), .2),
                             "wrist_image": torch.full((3, 2, 2), .2)} for _ in range(frames)]
            worker.episodes.append((observations, [0.] * (frames - 1), [False] * (frames - 1),
                                    [np.zeros(7)] * (frames - 1), "task", "real-episode", []))
        worker.get_one_episode = collect
        with patch.dict(sys.modules, {"capture_hooks": hooks}):
            if pipeline:
                from pipeline_imagination import imagine_weaver_seed
                for start in range(2 if stale else 1):
                    imagine_weaver_seed(worker, dict(episode_id='real-episode', start=start,
                        instruction='task', frame_groups=[], states=np.zeros((2,8), np.float32),
                        actions=np.zeros((2,7), np.float32),
                        frames=np.full((2,2,3,2,2), 153, dtype=np.uint8)))
                self.assertEqual(collected, [])
            else:
                with self.assertRaises(FinishedTest):
                    run_weaver_rollouts(worker)
        return calls, batches, stats

    def test_integrated_loop_has_no_reward_rpc_and_uses_policy_bootstrap_and_real_gae(self):
        calls, batches, stats = self.run_worker()
        self.assertFalse(any(role == "reward_infer" for role, _, _ in calls))
        self.assertEqual(sum(workflow == "imagined_bootstrap" for _, _, workflow in calls), 1)
        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0]), 2)
        self.assertAlmostEqual(batches[0][0].advantage, .5212, places=5)
        self.assertAlmostEqual(batches[0][0].value_target, .6212, places=5)
        self.assertAlmostEqual(batches[0][1].value_target, .56, places=5)
        self.assertEqual(batches[0][0].model_versions["world_reward_bundle_version"], 7)
        self.assertEqual(stats[0][2], 2)

    def test_terminal_loop_uses_zero_bootstrap_and_ignores_later_predictions(self):
        calls, batches, _ = self.run_worker(terminal=True)
        self.assertFalse(any(workflow == "imagined_bootstrap" for _, _, workflow in calls))
        self.assertEqual(len(batches[0]), 1)
        self.assertAlmostEqual(batches[0][0].advantage, .5, places=5)
        self.assertAlmostEqual(batches[0][0].value_target, .6, places=5)

    def test_external_loop_calls_existing_classifier_for_seed_and_each_frame(self):
        calls, batches, _ = self.run_worker(integrated=False)
        self.assertEqual(sum(role == "reward_infer" for role, _, _ in calls), 5)
        self.assertEqual(sum(workflow == "imagined_bootstrap" for _, _, workflow in calls), 1)
        self.assertAlmostEqual(batches[0][0].value_target, .4492, places=5)
        self.assertNotIn("world_reward_bundle_version", batches[0][0].model_versions)

    def test_stale_state_discards_unfinished_targets_then_next_seed_succeeds(self):
        calls, batches, stats = self.run_worker(stale=True)
        self.assertEqual(sum(method == "init_rollout" for _, method, _ in calls), 2)
        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0]), 2)
        self.assertTrue(all(sample.model_versions["world_bundle_version"] == 8 for sample in batches[0]))
        self.assertEqual(len(stats), 1)

    def test_external_explicit_probability_rule_is_honored(self):
        calls, batches, _ = self.run_worker(integrated=False, threshold=.3)
        self.assertEqual(sum(role == "reward_infer" for role, _, _ in calls), 3)
        self.assertFalse(any(workflow == "imagined_bootstrap" for _, _, workflow in calls))
        self.assertAlmostEqual(batches[0][0].value_target, .1, places=5)


class PipelineSeedTests(FullWorkerLoopTests):
    """Exercise the actual new single-seed entry with real CPU Torch and GAE."""
    def run_worker(self, **kwargs):
        return super().run_worker(pipeline=True, **kwargs)


if __name__ == "__main__":
    unittest.main()
