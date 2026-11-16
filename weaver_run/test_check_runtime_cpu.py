"""Actual launcher/parser/admission argument contract without model or GPU imports."""
import argparse
import ast
import copy
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / 'engine')]
if os.name != 'posix':
    sys.modules.setdefault('fcntl', types.SimpleNamespace())
import launch
from model_free_mode import configure_training_mode


def definition(path, name):
    tree = ast.parse(path.read_text(encoding='utf8'))
    return copy.deepcopy(next(node for node in tree.body
                              if isinstance(node, ast.FunctionDef) and node.name == name))


def assignment_index(body, name):
    return next(index for index, node in enumerate(body)
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))


def execute(node, namespace):
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 '<actual CPU admission>', 'exec'), namespace)
    return namespace[node.name]


class RuntimeArgumentAdmissionTests(unittest.TestCase):
    def parsed(self, *, model_free=True, num_eval_workers=None, fixed_validation=None, batch_autotune=False):
        environment = {'ACCERL_EXPERIMENT_ROOT': '/cpu-admission',
                       'ACCERL_BATCH_AUTOTUNE': str(int(batch_autotune)),
                       'ACCERL_PIPELINE_CONFIG': '/cpu-admission/pipeline.json',
                       'ACCERL_ALLOCATED_GPU_IDS': '4,5,6,7',
                       'ACCERL_NUM_EVAL_WORKERS': str(
                           (0 if model_free else 3) if num_eval_workers is None else num_eval_workers)}
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, environment).start()
        work = Path(environment['ACCERL_EXPERIMENT_ROOT'])
        values = dict(launch.settings(), ids='4,5,6,7', checkpoints=str(work / 'checkpoints'),
                      world=str(work / 'world.json'), pipeline=environment['ACCERL_PIPELINE_CONFIG'],
                      agent=str(launch.ROOT / 'envs/config/agent.yaml'),
                      trainer=str(launch.ROOT / 'envs/config/trainer.yaml'))
        argv = launch.training_arguments(values, model_free=model_free,
            num_eval_workers=num_eval_workers, fixed_validation=fixed_validation, batch_autotune=batch_autotune)
        parser = definition(HERE / 'engine/train.py', 'parse_args')
        # Stop before hardware bindings/model admission; retain the real argparse declarations.
        parser.body = parser.body[:assignment_index(parser.body, 'args') + 1]
        parser.body.append(ast.Return(ast.Name('args', ast.Load())))
        namespace = dict(argparse=argparse, os=os, DUAL_LAYOUT='dual_train_policy_reward',
                         FOUR_LAYOUT='four_rank_six_roles', SPLIT_LAYOUT='four_rank_split_train')
        with patch.object(sys, 'argv', ['train.py', *argv]):
            args = execute(parser, namespace)()
        configure_training_mode(args)
        return args

    def admit(self, args):
        check = definition(HERE / 'check_runtime.py', 'main')
        # Execute the production comparison, excluding installed-model checks and simulator reset.
        first = assignment_index(check.body, 'work')
        last = assignment_index(check.body, 'online_eval')
        check.body = check.body[first:last] + [ast.Return(ast.Name('diff', ast.Load()))]
        namespace = dict(vars(launch), args=args, Path=Path, json=json, os=os)
        return execute(check, namespace)()

    def test_default_fixed_validation_launch_is_admitted(self):
        self.assertEqual(self.admit(self.parsed()), {})

    def test_custom_fixed_validation_launch_is_admitted(self):
        args = self.parsed(fixed_validation=dict(interval_env_steps=200000, patience=4,
                           min_env_steps=400000, min_delta_successes=2))
        self.assertEqual(self.admit(args), {})

    def test_online_eval_disables_fixed_validation_and_is_admitted(self):
        self.assertEqual(self.admit(self.parsed(num_eval_workers=4)), {})

    def test_model_based_recipe_remains_admitted(self):
        self.assertEqual(self.admit(self.parsed(model_free=False)), {})

    def test_batch_autotune_launch_is_admitted(self):
        self.assertEqual(self.admit(self.parsed(batch_autotune=True)), {})

    def test_batch_autotune_cannot_drift_from_frozen_launch_environment(self):
        args=self.parsed(batch_autotune=True)
        args.batch_autotune=False
        with self.assertRaisesRegex(ValueError, 'batch_autotune'):
            self.admit(args)

    def test_unrelated_recipe_drift_is_still_rejected(self):
        args = self.parsed()
        args.num_images_in_input = 1
        with self.assertRaisesRegex(ValueError, 'num_images_in_input'):
            self.admit(args)

    def test_invalid_fixed_validation_still_fails_admission(self):
        args = self.parsed(num_eval_workers=4)
        args.fixed_validation_interval_env_steps = 100000
        with self.assertRaisesRegex(ValueError, 'requires --num-eval-workers 0'):
            self.admit(args)


if __name__ == '__main__':
    unittest.main()
