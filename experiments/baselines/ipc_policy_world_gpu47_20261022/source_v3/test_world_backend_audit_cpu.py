"""Actual setup-hook -> engine report -> controller receipt checks, without torch/GPU."""
import ast
import copy
from pathlib import Path
import re
from types import SimpleNamespace
import unittest

import capture_training
from policy_backend_audit import policy_zero2_report
from test_policy_backend_audit_cpu import engine
from world_backend_audit import world_zero2_report, validate_world_zero2_receipts

SOURCE = Path(__file__).resolve().parent


def launcher_argument(name):
    values = re.findall(r'(?m)^\s+--' + re.escape(name) + r'\s+(\d+)\s+\\$',
                        (SOURCE/'run.sh').read_text(encoding='utf-8'))
    if len(values) != 1:
        raise AssertionError('one launcher argument required: ' + name)
    return int(values[0])


def hooked_setup(role, rank, value):
    events = []
    writer = SimpleNamespace(event=lambda kind, **kwargs: events.append((kind, kwargs)),
                             flush=lambda: None, span=lambda *args, **kwargs: None)
    class Trainer:
        def setup_deepspeed_group(self):
            return 'setup_complete'
        def _train_policy(self):
            pass
        def _train_ctrl_world(self):
            pass
        def _train_reward_model(self):
            pass
    capture_training._install_trainer_hooks({'TrainerActor': Trainer})
    actor = Trainer()
    actor.training_role, actor.rank, actor.telemetry = role, rank, writer
    if role == 'ctrl':
        actor.ctrl_world_engine = value
    else:
        actor.model = value
    return actor, events


class WorldBackendAuditTest(unittest.TestCase):
    def receipts(self):
        micro, gas = launcher_argument('ctrl-batch-size'), launcher_argument('ctrl-accumulation-steps')
        self.assertEqual((micro, gas, micro * gas * 4), (4, 64, 1024))
        rows = []
        for rank in range(4):
            value = engine(micro=micro, gas=gas, global_batch=micro * gas * 4)
            actor, events = hooked_setup('ctrl', rank, value)
            self.assertEqual(actor.setup_deepspeed_group(), 'setup_complete')
            kinds = [kind for kind, _ in events]
            self.assertLess(kinds.index('world_zero2_verified'), kinds.index('trainer_ready'))
            details = next(data['details'] for kind, data in events if kind == 'world_zero2_verified')
            rows.append({'actor_id': str(rank), 'pid': 100 + rank, 'details': details})
        return rows

    def test_exact_launcher_reaches_actual_setup_and_controller_acceptance(self):
        rows = self.receipts()
        self.assertTrue(validate_world_zero2_receipts(rows))
        for row in rows:
            self.assertEqual(row['details']['local_batch_size'], 256)

    def test_parent_pair_is_rejected_before_trainer_ready(self):
        actor, events = hooked_setup('ctrl', 0, engine(micro=2, gas=128, global_batch=1024))
        with self.assertRaisesRegex(RuntimeError, 'micro/GAS/global batch'):
            actor.setup_deepspeed_group()
        self.assertFalse(any(kind == 'trainer_ready' for kind, _ in events))

    def test_live_engine_overrides_stale_config_dictionary(self):
        value = engine(micro=2, gas=128, global_batch=1024)
        value._config.train_micro_batch_size_per_gpu = 4
        value._config.gradient_accumulation_steps = 64
        with self.assertRaises(RuntimeError):
            world_zero2_report(value, 0)

    def test_wrong_engine_precision_world_bucket_and_batch_fail(self):
        for name, result in [('zero_optimization_stage', 3), ('train_micro_batch_size_per_gpu', 2),
                ('gradient_accumulation_steps', 128), ('train_batch_size', 512),
                ('bfloat16_enabled', False), ('gradient_clipping', 0.5)]:
            value = engine(micro=4, gas=64, global_batch=1024)
            setattr(value, name, lambda result=result: result)
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                world_zero2_report(value, 0)
        for mutate in (lambda value: setattr(value, 'dp_world_size', 2),
                       lambda value: setattr(value._config.zero_config, 'reduce_bucket_size', 50000000)):
            value = engine(micro=4, gas=64, global_batch=1024)
            mutate(value)
            with self.assertRaises(RuntimeError):
                world_zero2_report(value, 0)

    def test_missing_duplicate_cross_rank_or_stale_receipts_fail(self):
        rows = self.receipts()
        cases = [rows[:3], [rows[0]] * 4]
        for key, value in [('micro_batch_size', 2), ('gradient_accumulation_steps', 128),
                           ('global_batch_size', 512), ('reduce_bucket_size', 50000000)]:
            changed = copy.deepcopy(rows)
            changed[2]['details'][key] = value
            cases.append(changed)
        changed = copy.deepcopy(rows)
        changed[1]['actor_id'] = '2'
        cases.append(changed)
        changed = copy.deepcopy(rows)
        del changed[0]['details']['micro_batch_size']
        cases.append(changed)
        for value in cases:
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                validate_world_zero2_receipts(value)

    def test_policy8_and_reward16_recipe_kept(self):
        self.assertEqual((launcher_argument('train-batch-size'), launcher_argument('accumulation-steps')), (8, 18))
        self.assertEqual((launcher_argument('reward-batch-size'), launcher_argument('reward-accumulation-steps')), (16, 16))
        actor, events = hooked_setup('policy', 0, engine())
        actor.setup_deepspeed_group()
        self.assertTrue(any(kind == 'policy_zero2_verified' for kind, _ in events))
        self.assertFalse(any(kind == 'world_zero2_verified' for kind, _ in events))

    def test_real_controller_acceptance_entry_checks_world_receipts(self):
        tree = ast.parse((SOURCE/'runner.py').read_text(encoding='utf-8'))
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
        call = next(node for node in ast.walk(run) if isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == 'validate_world_zero2_receipts')
        code = compile(ast.fix_missing_locations(ast.Module(body=[call], type_ignores=[])), 'actual_world_receipt_acceptance', 'exec')
        rows = self.receipts()
        exec(code, {'validate_world_zero2_receipts': validate_world_zero2_receipts,
                    'result': {'world_zero2_verified': rows}})
        for result in ({}, {'world_zero2_verified': rows[:3]}):
            with self.assertRaises(RuntimeError):
                exec(code, {'validate_world_zero2_receipts': validate_world_zero2_receipts, 'result': result})
        collect = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'collect')
        self.assertIn('world_zero2_verified', ast.unparse(collect))

    def test_report_does_not_modify_engine_or_optimizer(self):
        value = engine(micro=4, gas=64, global_batch=1024)
        group = value.optimizer.optimizer.param_groups[0]
        before = {key: copy.deepcopy(item) for key, item in group.items() if key != 'params'}
        world_zero2_report(value, 0)
        self.assertEqual(before, {key: item for key, item in group.items() if key != 'params'})


if __name__ == '__main__':
    unittest.main()
