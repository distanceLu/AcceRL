"""Measured replay and actual world-training loop checks, entirely on CPU."""
import ast
from pathlib import Path
import types
import unittest
import math
import time

import numpy as np
import torch
from test_pluggable_main_cpu import online_config
from world_backends import WorldBackendConfig
from world_backends.libero import backend_window_payload, environment_dt

TREE = ast.parse(Path(__file__).with_name("train.py").read_text(encoding="utf-8"))


def trainer_method(name, namespace):
    cls = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "TrainerActor")
    node = next(n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<actual Trainer method>", "exec"), namespace)
    return namespace[name]


class DataTraining(unittest.TestCase):
    def payload(self, start=0):
        value = online_config(True)
        value["schema"]["image_size"] = [2, 3]
        value["checkpoint_metadata"]["schema"]["image_size"] = [2, 3]
        config = WorldBackendConfig.from_mapping(value)
        payload = backend_window_payload(config, np.zeros((5, 2, 3, 2, 3), np.float32),
            np.zeros((5, 8), np.float32), np.zeros((5, 7), np.float32), [0, 0, 0, 0, 1],
            [False]*4 + [True], [False]*4 + [True], [False]*5, start, 5, "pick cup", "ep0", "ep0:w0")
        return payload

    def test_labels_states_reset_and_real_transition_alignment(self):
        reset = self.payload()
        self.assertEqual(reset["states"].shape, (5, 8))
        self.assertEqual(reset["action_mask"].tolist(), [False, True, True, True, True])
        self.assertEqual(reset["dt"][0], 0)
        self.assertEqual(reset["success"].tolist(), [False]*4 + [True])
        self.assertEqual(reset["done"].tolist(), [False]*4 + [True])
        self.assertTrue(self.payload(1)["action_mask"].all())
        self.assertNotIn("obs", reset)  # No second image copy in the metadata.

    def test_real_time_scale_requires_observed_environment_property(self):
        self.assertEqual(environment_dt(types.SimpleNamespace(env=types.SimpleNamespace(control_freq=20))), .05)
        with self.assertRaisesRegex(ValueError, "Cannot verify"):
            environment_dt(types.SimpleNamespace())

    def test_prepare_keeps_raw_payload_on_cpu_until_each_microstep(self):
        prepare = trainer_method("_prepare_wm_batch", {})
        owner = types.SimpleNamespace(training_role="ctrl", world_backend=object(),
                                      world_config=types.SimpleNamespace(backend="weaver"))
        obs = np.zeros((2, 5, 2, 3, 4, 4), dtype=np.float32)
        actions = [object(), object()]
        labels = [0., 1.]
        payloads = [{"sample_id": "a"}, {"sample_id": "b"}]
        output = prepare(owner, obs, actions, labels, ["task", "task"], payloads)
        self.assertIsInstance(output["pixel"][1]["obs"], np.ndarray)
        self.assertTrue(np.shares_memory(output["pixel"][1]["obs"], obs))
        self.assertEqual([x["sample_id"] for x in output["pixel"]], ["a", "b"])
        self.assertIsNone(output["action"])
        with self.assertRaisesRegex(ValueError, "not aligned"):
            prepare(owner, obs, actions, labels, ["task", "task"], payloads[:1])

    def test_actual_train_loop_prepares_per_micro_and_updates_ema_after_success(self):
        calls, events = [], []
        class Engine:
            global_steps, skipped_steps, micros = 0, 0, 0
            optimizer = types.SimpleNamespace(param_groups=[{"lr": 1.}])
            def train(self): pass
            def __call__(self, batch):
                calls.append(("forward", list(batch)))
                return torch.tensor(2., requires_grad=True), {"world": torch.tensor(2.)}
            def backward(self, loss): calls.append(("backward",))
            def is_gradient_accumulation_boundary(self): return self.micros == 2
            def get_global_grad_norm(self): return .25
            def step(self):
                self.micros += 1
                if self.micros == 3: self.global_steps += 1
        class Backend:
            def prepare_train_batch(self, raw):
                calls.append(("prepare", list(raw)))
                return raw
            def update_ema(self): calls.append(("ema",))
        engine = Engine()
        owner = types.SimpleNamespace(global_step=5, ctrl_train_interval=5, ctrl_step=0,
            ctrl_lr=.01, ctrl_warmup_steps=20, total_ctrl_steps=6000, ctrl_world_engine=engine,
            world_backend=Backend(), ctrl_super_batch_size=12, ctrl_batch_size=4, rank=0,
            world_config=types.SimpleNamespace(backend="weaver"),
            telemetry=types.SimpleNamespace(event=lambda *a, **kw: events.append((a, kw))),
            _get_current_lr=lambda *a: .001,
            _verify_optimizer_update=lambda *a: self.assertEqual((engine.global_steps, engine.skipped_steps), (1, 0)))
        namespace = {"time": time, "math": math, "np": np,
            "micro_begin": lambda *a: calls.append(("consume", a[3], a[4], a[5])),
            "micro_end": lambda *a: None, "_trim_cuda_cache": lambda *a: None,
            "successful_updates": lambda before, skips, eng: eng.global_steps-before}
        train = trainer_method("_train_ctrl_world", namespace)
        result = train(owner, list(range(12)), None, None, {})
        self.assertEqual([c[1] for c in calls if c[0] == "prepare"], [list(range(4)), list(range(4, 8)), list(range(8, 12))])
        self.assertEqual([c[1:] for c in calls if c[0] == "consume"], [(4, 0, 4), (4, 4, 8), (4, 8, 12)])
        self.assertEqual(calls[-1], ("ema",))
        self.assertEqual(owner.ctrl_step, 1)
        self.assertEqual(result["epoch_ctrl_losses"], [2., 2., 2.])
        self.assertEqual(len(events), 3)


if __name__ == "__main__":
    unittest.main()
