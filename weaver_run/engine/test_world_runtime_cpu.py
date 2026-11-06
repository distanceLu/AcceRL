import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from test_world_contracts_cpu import weaver_config
from world_backends.runtime import (CtrlBackendAdapter, WeaverActorOperations, context_geometry,
                                    load_backend_config, prepare_runtime)


class FakeBackend:
    def __init__(self, config):
        self.config = config.to_dict()
        self.required_context_frames, self.training_window_frames, self.horizon = 7, 11, 4
        self.invalidations = []

    def set_published_version(self, version):
        self.invalidations.append(version)

    def init_rollout(self, context, version):
        return {"backend": "weaver", "version": version, "instruction": context.get("instruction", "task"),
                "encoder_id": self.config["schema"]["encoder_id"], "normalizer_id": self.config["schema"]["normalizer_id"],
                "initial_success_probability": .2}

    def predict_chunk(self, state, actions, published_version=None):
        n = len(actions)
        return {"observations": ["frame"] * n, "predicted_states": [[0.] * 8] * n,
                "next_state": dict(state), "served_version": published_version,
                "encoder_id": self.config["schema"]["encoder_id"], "normalizer_id": self.config["schema"]["normalizer_id"],
                "dt": self.config["schema"]["dt_seconds"],
                "success_probabilities": [.7] * n, "terminal_predictions": [True] * n}


class RuntimeTests(unittest.TestCase):
    def test_default_is_legacy_ctrl(self):
        config = load_backend_config(environ={})
        self.assertEqual(config.backend, "ctrl")
        self.assertEqual(prepare_runtime(config).expected_gpu_model_processes, 24)

    def test_json_and_cli_override_without_mutating_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"backend": "weaver", "reward_provider": "world_head"}), encoding="utf-8")
            env = {"ACCERL_WORLD_BACKEND_CONFIG": str(path)}
            original = copy.deepcopy(env)
            self.assertEqual(load_backend_config(environ=env).backend, "weaver")
            override = load_backend_config(SimpleNamespace(world_backend="ctrl", reward_provider="external_vlm"), env)
            self.assertEqual(override.backend, "ctrl")
            self.assertEqual(env, original)

    def test_unverified_weaver_rejected_before_runtime(self):
        with self.assertRaisesRegex(ValueError, "schema is missing"):
            prepare_runtime({"backend": "weaver", "reward_provider": "world_head"})

    def test_context_geometry_uses_declared_memory(self):
        geometry = context_geometry(weaver_config())
        self.assertEqual(geometry["required_context_frames"], 7)
        self.assertEqual(geometry["training_window_frames"], 15)

    def test_ctrl_delegates_without_reinterpreting_state_or_loss(self):
        opaque = object()
        adapter = CtrlBackendAdapter(model=lambda batch: (batch, "loss"), init_fn=lambda context: opaque,
                                     predict_fn=lambda payload, actions, outputs: {"observations": actions, "next_state": payload},
                                     prepare_fn=lambda batch: batch, published_version=3)
        state = adapter.init_rollout({})
        self.assertIs(state["payload"], opaque)
        self.assertEqual(adapter.compute_losses("prepared"), ("prepared", "loss"))
        self.assertIs(adapter.predict_chunk(state, [1])["next_state"]["payload"], opaque)
        self.assertIsNone(adapter.publication_inventory())
        adapter.set_published_version(4)
        with self.assertRaisesRegex(ValueError, "Stale CTRL"):
            adapter.predict_chunk(state, [1])

    def test_rpc_opaque_cache_version_changes_on_publication(self):
        config = weaver_config(True)
        backend = FakeBackend(config)
        ops = WeaverActorOperations(backend, config, published_version=2)
        state = ops.init_rollout({"instruction": "task"})
        result = ops.request(state, [[0.] * 7])
        self.assertEqual(result["served_version"], 2)
        self.assertEqual(ops.get_context_geometry()["horizon"], 4)
        ops.set_published_version(3)
        self.assertEqual(backend.invalidations, [2, 3])
        with self.assertRaisesRegex(ValueError, "Stale WEAVER"):
            ops.request(state, [[0.] * 7])

    def test_task_and_action_contract_rejected(self):
        config = weaver_config()
        ops = WeaverActorOperations(FakeBackend(config), config)
        state = ops.init_rollout({"instruction": "task"})
        with self.assertRaisesRegex(ValueError, "instruction changed"):
            ops.request(state, [[0.] * 7], "different task")
        with self.assertRaisesRegex(ValueError, "action schema"):
            ops.request(state, [[0.] * 8])

    def test_integrated_seed_potential_cannot_be_assumed_zero(self):
        config = weaver_config(True)
        backend = FakeBackend(config)
        original = backend.init_rollout
        def missing_seed(*args):
            result = original(*args)
            del result["initial_success_probability"]
            return result
        backend.init_rollout = missing_seed
        with self.assertRaisesRegex(ValueError, "real seed"):
            WeaverActorOperations(backend, config).init_rollout({})

    def test_templates_are_honest_and_fail_admission(self):
        examples = Path(__file__).parent / "examples"
        self.assertEqual(prepare_runtime(load_backend_config({"world_backend_config": examples / "ctrl_external.json"}, {})).expected_gpu_model_processes, 24)
        for name in ("weaver_external.template.json", "weaver_integrated.template.json"):
            config = load_backend_config({"world_backend_config": examples / name}, {})
            with self.assertRaises(ValueError):
                prepare_runtime(config)


if __name__ == "__main__":
    unittest.main()
