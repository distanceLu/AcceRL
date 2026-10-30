"""CPU checks that backend evidence rejects incorrect live configuration."""
import copy
import unittest
from pathlib import Path
import re
from types import SimpleNamespace

from policy_backend_audit import policy_zero2_report


class Parameter:
    dtype = 'torch.float32'
    requires_grad = True
    def numel(self):
        return 16


class FusedAdam:
    adam_w_mode = True
    def __init__(self):
        self.param_groups = [{'params': [Parameter()], 'name': 'policy', 'lr': 1e-5,
                              'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0.0}]


def engine(*, micro=8, gas=18, global_batch=576, world=4):
    parameter = Parameter()
    module = SimpleNamespace(parameters=lambda: iter([parameter]), modules=lambda: iter([]))
    return SimpleNamespace(module=module, dp_world_size=world,
        zero_optimization_stage=lambda: 2, train_micro_batch_size_per_gpu=lambda: micro,
        gradient_accumulation_steps=lambda: gas, train_batch_size=lambda: global_batch,
        bfloat16_enabled=lambda: True, gradient_clipping=lambda: 1.0,
        _config=SimpleNamespace(zero_config=SimpleNamespace(reduce_bucket_size=500000000,
            allgather_bucket_size=500000000, overlap_comm=True, contiguous_gradients=True,
            reduce_scatter=True, allgather_partitions=True)),
        optimizer=SimpleNamespace(optimizer=FusedAdam()))


class BackendAuditTest(unittest.TestCase):
    def test_exact_frozen_launcher_configuration_reaches_actual_audit_entry(self):
        launcher = (Path(__file__).resolve().parent / 'run.sh').read_text(encoding='utf-8')
        def argument(name):
            values = re.findall(r'(?m)^\s+--' + re.escape(name) + r'\s+(\d+)\s+\\$', launcher)
            self.assertEqual(len(values), 1, name)
            return int(values[0])
        micro, gas = argument('train-batch-size'), argument('accumulation-steps')
        self.assertEqual((micro, gas, micro * gas * 4), (8, 18, 576))
        for rank in range(4):
            report = policy_zero2_report(engine(micro=micro, gas=gas, global_batch=micro * gas * 4), rank)
            self.assertEqual((report['micro_batch_size'], report['gradient_accumulation_steps'],
                              report['global_batch_size'], report['world_size']), (8, 18, 576, 4))
            self.assertTrue(report['no_dtensor_parameters'])

    def test_old_half_micro_pair_is_rejected_despite_equal_global_batch(self):
        with self.assertRaisesRegex(RuntimeError, 'micro/GAS/global batch'):
            policy_zero2_report(engine(micro=4, gas=36, global_batch=576), 0)

    def test_original_configuration_is_read_only(self):
        value = engine()
        group = value.optimizer.optimizer.param_groups[0]
        before = {key: copy.deepcopy(v) for key, v in group.items() if key != 'params'}
        report = policy_zero2_report(value, 0)
        self.assertEqual(report['backend'], 'deepspeed_zero2')
        self.assertEqual(report['global_batch_size'], 576)
        self.assertEqual(before, {key: v for key, v in group.items() if key != 'params'})

    def test_rejects_wrong_backend_batch_and_precision(self):
        for name, result in [('zero_optimization_stage', 3),
                             ('train_micro_batch_size_per_gpu', 4),
                             ('gradient_accumulation_steps', 36), ('train_batch_size', 288),
                             ('bfloat16_enabled', False), ('gradient_clipping', 0.5)]:
            with self.subTest(name=name):
                value = engine()
                setattr(value, name, lambda result=result: result)
                with self.assertRaises(RuntimeError):
                    policy_zero2_report(value, 0)

    def test_rejects_changed_optimizer(self):
        for name, value in [('weight_decay', 0.01), ('eps', 1e-6), ('betas', (0.9, 0.95))]:
            with self.subTest(name=name):
                target = engine()
                target.optimizer.optimizer.param_groups[0][name] = value
                with self.assertRaises(RuntimeError):
                    policy_zero2_report(target, 0)

    def test_rejects_sharded_parameters(self):
        target = engine()
        DTensor = type('DTensor', (Parameter,), {})
        target.module.parameters = lambda: iter([DTensor()])
        with self.assertRaises(RuntimeError):
            policy_zero2_report(target, 0)

    def test_rejects_world_bucket_and_fsdp_module(self):
        target = engine()
        target.dp_world_size = 2
        with self.assertRaises(RuntimeError):
            policy_zero2_report(target, 0)
        target = engine()
        target._config.zero_config.reduce_bucket_size = 50000000
        with self.assertRaises(RuntimeError):
            policy_zero2_report(target, 0)
        target = engine()
        FSDPModule = type('FSDPModule', (), {})
        target.module.modules = lambda: iter([FSDPModule()])
        with self.assertRaises(RuntimeError):
            policy_zero2_report(target, 0)


if __name__ == '__main__':
    unittest.main()
