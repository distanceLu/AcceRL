"""Execute the actual main AST through first fill with inert Ray/Torch stand-ins.

This validates Python startup wiring only. No real actors, models, threads,
object store, tensors or GPU contexts are created.
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import copy
import io
import itertools
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time
import types
import unittest


HERE = Path(__file__).resolve().parent
_options_parser = argparse.ArgumentParser(add_help=False)
_options_parser.add_argument("--source", type=Path, default=HERE)
_options, _remaining = _options_parser.parse_known_args()
sys.argv = [sys.argv[0], *_remaining]
SOURCE = _options.source.resolve()

import ast

TREE = ast.parse((SOURCE / "train.py").read_text(encoding="utf-8"))
LAYOUTS = {"DUAL_LAYOUT": "dual_train_policy_reward", "FOUR_LAYOUT": "four_rank_six_roles",
           "SPLIT_LAYOUT": "four_rank_split_train"}


def definition(tree, name):
    return copy.deepcopy(next(node for node in tree.body
                              if isinstance(node, ast.FunctionDef) and node.name == name))


def execute_definition(node, namespace):
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 "<real startup AST>", "exec"), namespace)
    return namespace[node.name]


def arguments(validate_only=False, overrides=()):
    node = definition(TREE, "parse_args")
    boundary = next(i for i, stmt in enumerate(node.body)
                    if isinstance(stmt, ast.Assign) and any(isinstance(n, ast.Name) and n.id == "args"
                                                           for n in stmt.targets))
    node.body = node.body[:boundary + 1] + [ast.Return(value=ast.Name(id="args", ctx=ast.Load()))]
    argv = []
    for line in (SOURCE / "run.sh").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("--"):
            argv.extend(shlex.split(line.rstrip("\\").strip()))
    if not argv:
        # The managed launcher keeps its arguments in recipe/argv.json. These
        # legacy three-role startup tests exercise the CTRL baseline, before
        # the online-WEAVER overrides appended to that argument list.
        recipe = json.loads((SOURCE.parent / "recipe" / "argv.json").read_text(encoding="utf-8"))
        argv = recipe[:recipe.index("--world-backend-config")]
        argv = ["4,5,6,7" if value == "@ids" else
                "cpu_fixture_" + value[1:] if value.startswith("@") else value
                for value in argv]
    argv.extend(overrides)
    if validate_only:
        argv.append("--validate-only")
    old = sys.argv
    try:
        sys.argv = [str(SOURCE / "train.py"), *argv]
        return execute_definition(node, {"argparse": argparse, "os": os, **LAYOUTS})()
    finally:
        sys.argv = old


class Telemetry:
    def __init__(self, context, actor, actor_id):
        self.context, self.actor, self.actor_id = context, actor, actor_id
        self.events, self.closed = [], False
        context.telemetries.append(self)
    def event(self, kind, **fields):
        if self.closed:
            raise AssertionError("Using closed startup writer")
        self.events.append((kind, fields))
        self.context.actions.append(("event", kind))
    def flush(self):
        pass
    def close(self):
        self.closed = True


class Summary:
    def __init__(self, context, log_dir, tensorboard=True):
        self.log_dir, self.tensorboard = log_dir, object() if tensorboard else None
        self.telemetry = Telemetry(context, "driver", "metrics")
        self.scalars, self.closed = [], False
        context.summaries.append(self)
    def add_scalar(self, *args, **kwargs):
        self.scalars.append((args, kwargs))
    def flush(self):
        pass
    def close(self):
        self.closed = True


class RemoteMethod:
    def __init__(self, actor, name):
        self.actor, self.name = actor, name
    def remote(self, *args, **kwargs):
        return self.actor.call(self.name, args, kwargs)


class Actor:
    def __init__(self, context, name, positional, kwargs, options):
        self.context, self.name, self.positional = context, name, positional
        self.kwargs, self.options = kwargs, options
        self.rank = kwargs.get("rank", kwargs.get("actor_id", 0))
        self.kind = kwargs.get("training_role", {"InferenceActor": "policy", "RewardInferenceActor": "reward",
                                                 "WeaverWorldInferenceActor": "ctrl",
                                                 "CtrlWorldInferenceActor": "ctrl"}.get(name))
        self.pid = 10000 + len(context.actors)
        self.port, self.active = None, {}
        context.actors.append(self)
        context.actions.append(("create", name))
    def __getattr__(self, name):
        return RemoteMethod(self, name)
    def inventory(self, kind):
        return {"kind": kind, "version": 0, "hash": "a" * 64, "bytes": 64, "tensor_count": 1}
    def call(self, method, args, kwargs):
        self.context.calls.append((self.name, self.rank, method, args, kwargs))
        if method == "get_node_ip":
            return "127.0.0.1"
        if method == "setup_deepspeed_group":
            self.port = args[1]
            return True
        if method == "get_model_keys":
            return ["example_parameter"]
        if method == "forward_test":
            return True
        if method == "export_rank_weights":
            kind = args[0]
            if kind != self.kind:
                raise AssertionError("Cross-role source export")
            if kind in ("policy", "ctrl"):
                packet = {"inventory": self.inventory(kind), "token": "%s_%d" % (kind, self.rank)}
                self.active[kind] = packet
                return packet
            return self.inventory(kind)
        if method == "describe_rank_weights":
            return dict(args[0])
        if method == "apply_rank_weights":
            payload, kind = args
            if kind != self.kind:
                raise AssertionError("Cross-role destination apply")
            self.context.actions.append(("ACK", kind, self.rank))
            return dict(payload["inventory"] if kind in ("policy", "ctrl") else payload)
        if method == "finish_ipc_weight_publish":
            kind, token, acknowledgement = args
            packet = self.active.pop(kind)
            if packet["token"] != token or packet["inventory"] != acknowledgement:
                raise AssertionError("Mismatched producer release")
            self.context.actions.append(("release", kind, self.rank))
            return dict(acknowledgement)
        if method == "get_training_state":
            return {"rank": self.rank, "gpu": {"physical_gpu": str(self.context.physical_gpu_ids[self.rank])}, "pid": self.pid,
                    "training_role": self.kind, "world_size": 4, "distributed_world_size": 4,
                    "group_port": self.port, "engines": {self.kind: {"global_steps": 0, "skipped_steps": 0}},
                    "policy_steps": 0, "ctrl_steps": 0, "reward_steps": 0}
        if method == "export_snapshot_component":
            return {"kind": args[0], "fake_cpu_component": True}
        if method == "run":
            return ("pending_worker_run", self.pid)
        if method == "get_renderer_info":
            child = self.options.get('runtime_env', {}).get('env_vars', {})
            return {'backend':child.get('MUJOCO_GL'), 'pid':self.pid,
                    'worker':child.get('ACCERL_RENDER_WORKER')}
        if method == "size":
            return 10000 if self.name == "ReplayBufferActor" else 50000
        if method == "set_training_batch_limit":
            self.context.limits.append((self.name, self.rank, args[0]))
            return args[0]
        raise AssertionError("Unspecified startup method: %s.%s" % (self.name, method))


class ActorClass:
    def __init__(self, context, name, options=None):
        self.context, self.name, self.placement = context, name, options or {}
    def options(self, **kwargs):
        return ActorClass(self.context, self.name, kwargs)
    def remote(self, *args, **kwargs):
        return Actor(self.context, self.name, args, kwargs, self.placement)


class Ray:
    def __init__(self, context):
        self.context, self.init_kwargs = context, None
    def init(self, **kwargs):
        self.init_kwargs = kwargs
        self.context.actions.append(("ray_init",))
    def get(self, value, **kwargs):
        return value
    def remote(self, cls):
        return ActorClass(self.context, getattr(cls, "__name__", "Coordinator"))
    def wait(self, refs, **kwargs):
        return [], refs


class Policy:
    stop_on_target = until_deadline = False
    deadline_unix_seconds = None
    @classmethod
    def from_environment(cls, env):
        return cls()
    def metadata(self):
        return {"manual": True}


class Context:
    def __init__(self, physical_gpu_ids=(4,5,6,7)):
        self.physical_gpu_ids = tuple(physical_gpu_ids)
        self.actors, self.calls, self.telemetries, self.summaries = [], [], [], []
        self.snapshots, self.actions, self.limits = [], [], []
        self.ray = Ray(self)
        self.port_source = itertools.count(20000)

    def namespace(self):
        fake_os = types.SimpleNamespace(environ={"ACCERL_USE_MPS": "1", "ACCERL_TENSORBOARD": "1",
                     "ACCERL_CPU_REPLAY_PREFETCH": "1",
                     "ACCERL_EVAL_TARGET": ".85", "ACCERL_FINAL_CHECKPOINT": "cpu_fixture_checkpoint"},
                     path=types.SimpleNamespace(exists=lambda p: True, isfile=lambda p: True))
        from world_backends import build_service_plan
        from world_backends.runtime import load_backend_config, context_geometry
        from pipeline_budget import PipelineBudgetConfig
        from world_backends.libero import validate_online_schema
        from fresh_config import FreshConfig
        from world_backend_runtime import isolated_online_weaver, validate_online_runtime_arguments
        space = {"load_backend_config": load_backend_config, "build_service_plan": build_service_plan,
                 "FreshConfig": FreshConfig, "isolated_online_weaver": isolated_online_weaver,
                 "validate_online_runtime_arguments": validate_online_runtime_arguments,
                 "PipelineBudgetConfig": PipelineBudgetConfig, "context_geometry": context_geometry,
                 "validate_online_schema": validate_online_schema, "ACTION_DIM": 7,
                 "os": fake_os, "time": time, "math": math, "Path": Path, "json": json,
                 "torch": types.SimpleNamespace(bfloat16="bf16", float32="fp32"),
                 "TrainingStopPolicy": Policy, "EvaluationMilestones": lambda: object(),
                 "ThresholdTracker": lambda **kwargs: types.SimpleNamespace(**kwargs),
                 "now_ns": time.time_ns, "ray": self.ray, "print": lambda *args, **kwargs: None,
                 "serial_enabled": lambda *args, **kwargs: False,
                 "parse_gpu_ids": lambda value: value.split(","),
                 "build_bindings": lambda ids: [{"index": value} for value in ids],
                 "ray_init_options": lambda *args, **kwargs: {"num_gpus": 4},
                 "actor_options": lambda bindings, role, rank=0, **kwargs: {"fake_role": role, "fake_rank": rank},
                 "validate_object_store_budget": lambda value: {"requested": value},
                 "CaptureSummaryWriter": lambda *args, **kwargs: Summary(self, *args, **kwargs),
                 "Telemetry": lambda actor, actor_id: Telemetry(self, actor, actor_id),
                 "GenerateConfig": lambda **kwargs: types.SimpleNamespace(**kwargs),
                 "NUM_ACTIONS_CHUNK": 8, "find_free_port": lambda: next(self.port_source),
                 "GlobalBatchCoordinator": type("GlobalBatchCoordinator", (), {}),
                 "save_split_weights": self.save, "stop_requested": lambda: False,
                 **LAYOUTS}
        for name in ("StatsActor", "ReplayBufferActor", "WMReplayBufferActor", "TrainerActor",
                     "InferenceActor", "RewardInferenceActor", "CtrlWorldInferenceActor", "WeaverWorldInferenceActor",
                     "RolloutWorkerActor", "EvaluationWorkerActor"):
            space[name] = ActorClass(self, name)
        execute_definition(definition(TREE, "build_openvla_cfg"), space)
        execute_definition(definition(TREE, "validate_split_owners"), space)
        return space

    def save(self, components, directory, **kwargs):
        self.snapshots.append((components, directory, kwargs))
        return {"fake_cpu_snapshot": True}


def startup_ast(tree=TREE, mutations=()):
    node = definition(tree, "main")
    # All original statements through first fill+training-phase inference caps.
    # Do not enter any actual training epoch or mutate an internal expression.
    boundary = next(i for i, stmt in enumerate(node.body)
                    if any(isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute)
                           and x.func.attr == "event" and x.args and isinstance(x.args[0], ast.Constant)
                           and x.args[0].value == "training_phase_limits" for x in ast.walk(stmt)))
    node.body = node.body[:boundary + 1]
    for variable in mutations:
        node.body = [stmt for stmt in node.body if not (isinstance(stmt, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == variable for target in stmt.targets))]
    node.body.append(ast.Return(value=ast.Call(func=ast.Name(id="locals", ctx=ast.Load()), args=[], keywords=[])))
    return node


class MainStartupCPU(unittest.TestCase):
    def smoke(self, validate_only=False, mutations=()):
        context = Context()
        function = execute_definition(startup_ast(mutations=mutations), context.namespace())
        result = function(arguments(validate_only=validate_only))
        return context, result

    def test_real_main_creates_all_actors_and_initially_syncs_all_three_models(self):
        context, state = self.smoke()
        counts = Counter(actor.name for actor in context.actors)
        self.assertEqual(counts["StatsActor"], 1)
        self.assertEqual(counts["TrainerActor"], 12)
        self.assertEqual([counts[x] for x in ("InferenceActor", "RewardInferenceActor", "CtrlWorldInferenceActor")], [4, 4, 4])
        args = arguments()
        self.assertEqual((counts["RolloutWorkerActor"], counts["EvaluationWorkerActor"]),
                         (state['pipeline_config'].real_workers + state['pipeline_config'].imagination_workers, args.num_eval_workers))
        self.assertEqual(len(state["all_trainers"]), 12)
        model_initializers = [c for c in context.calls if c[2] == "setup_deepspeed_group"]
        self.assertEqual(len(model_initializers), 12)
        self.assertEqual(len({c[3][1] for c in model_initializers}), 3)
        publications = [fields for t in context.telemetries for kind, fields in t.events
                        if kind == "weight_publication_committed"]
        self.assertEqual([p["details"]["kind"] for p in publications], ["policy", "reward", "ctrl"])
        self.assertEqual([p["model_version"] for p in publications], [0, 0, 0])
        self.assertTrue(all(not actor.active for actor in state["all_trainers"]))
        self.assertEqual(len([x for x in context.actions if x[:1] == ("ACK",)]), 12)
        self.assertEqual(len([x for x in context.actions if x[:1] == ("release",)]), 8)

    def test_tb_stats_and_experiment_exist_before_their_first_uses(self):
        context, state = self.smoke()
        stats = state["stats_actor"]
        consumers = [actor for actor in context.actors if actor.name in
                     ("InferenceActor", "RewardInferenceActor", "CtrlWorldInferenceActor")]
        self.assertTrue(all(actor.kwargs["stats_actor"] is stats for actor in consumers))
        self.assertEqual(len(context.summaries), 1)
        self.assertIn("runs/Libero/libero_spatial/", state["log_dir"])
        fill_writers = [t for t in context.telemetries if t.actor_id == "fill_experiment"]
        self.assertEqual(len(fill_writers), 1)
        self.assertIs(state["experiment"], fill_writers[0])
        self.assertEqual(len([x for x in context.summaries[0].telemetry.events if x[0] == "tensorboard_started"]), 1)

    def test_initial_snapshot_worker_launch_first_fill_and_caps_are_reached(self):
        context, state = self.smoke()
        self.assertEqual(len(context.snapshots), 1)
        self.assertEqual(set(context.snapshots[0][0]), {"policy", "ctrl", "reward"})
        self.assertEqual(context.snapshots[0][2]["reason"], "initial")
        args = arguments()
        self.assertEqual(len(state["worker_run_refs"]), state['pipeline_config'].real_workers + state['pipeline_config'].imagination_workers + args.num_eval_workers)
        producers = [a for a in context.actors if a.name == 'RolloutWorkerActor']
        self.assertEqual(Counter(a.kwargs['worker_mode'] for a in producers), {'real': 4, 'imagine': 4})
        coordinator = next(a for a in context.actors if a.name == 'GlobalBatchCoordinator')
        self.assertTrue(coordinator.kwargs['pipeline_enabled'])
        self.assertEqual((state["min_buffer_size_for_start"], state["wm_min_buffer_size"]), (576, 1024))
        self.assertEqual(len([x for x in state["experiment"].events if x[0] == "replay_ready"]), 1)
        self.assertEqual(Counter((name, cap) for name, rank, cap in context.limits),
            Counter({("InferenceActor", 2): 4, ("CtrlWorldInferenceActor", 3): 4, ("RewardInferenceActor", 2): 4}))

    def test_ray_328_gib_and_cfg_original_model_flags_are_wired(self):
        context, state = self.smoke()
        self.assertEqual(context.ray.init_kwargs["object_store_memory"], 328 * 1024 ** 3)
        self.assertEqual(context.ray.init_kwargs["address"], "local")
        self.assertFalse(state["cfg"].load_in_8bit)
        self.assertFalse(state["cfg"].load_in_4bit)
        self.assertEqual(state["cfg"].num_images_in_input, 2)

    def test_validate_only_also_syncs_and_closes_both_driver_writers(self):
        context, state = self.smoke(validate_only=True)
        self.assertIsNone(state)
        self.assertEqual(len([x for x in context.actors if x.name == "RolloutWorkerActor"]), 0)
        self.assertEqual(len([x for x in context.calls if x[2] == "apply_rank_weights"]), 12)
        self.assertTrue(context.summaries[0].closed)
        fill_writer = next(t for t in context.telemetries if t.actor_id == "fill_experiment")
        self.assertTrue(fill_writer.closed)

    def test_missing_stats_regression_really_fails_this_actual_main(self):
        with self.assertRaises(NameError):
            self.smoke(mutations=("stats_actor",))

    def test_missing_summary_regression_really_fails_this_actual_main(self):
        with self.assertRaises(NameError):
            self.smoke(mutations=("writer",))

    def test_missing_experiment_regression_really_fails_first_publication(self):
        with self.assertRaises(NameError):
            self.smoke(mutations=("experiment",))


if __name__ == "__main__":
    unittest.main(verbosity=2)
