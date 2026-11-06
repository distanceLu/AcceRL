"""Execute actual installed Engine setter/boundary/step source on CPU doubles.

Fractions test mean-gradient algebra only; no claim of BF16 bitwise identity or
real NCCL/ZeRO buffer correctness. No torch/DeepSpeed imports or GPU allocation.
"""
import ast
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import textwrap
import types
import unittest
from unittest.mock import patch

from policy_joint_batch_mode import (PolicyJointBatchMode, expected_completed_microsteps,
                                    mode_for_step, verify_installed_sources)
import policy_joint_batch_mode as helper

HERE = Path(__file__).resolve().parent
SOURCES = json.loads((HERE / 'installed_deepspeed_boundary_sources.json').read_text())


class DummyOptim:
    pass


class CpuZeroOptimizer:
    def __init__(self):
        self.cpu_offload = False
        self.gradient_accumulation_steps = 18
        self.micro_step_id = -1
        self.all_grad_tensors = {}
        self.averaged_gradients = {}
        self.single_partition_of_fp32_groups = [types.SimpleNamespace(grad=None)]
        self.accumulator = Fraction(0)
        self.committed_gradients = []
        self.is_gradient_accumulation_boundary = False
        self.overflow = False

    def step(self):
        self.committed_gradients.append(self.accumulator)
        self.micro_step_id = -1

    def zero_grad(self):
        self.accumulator = Fraction(0)
        self.all_grad_tensors = {0: None}
        self.averaged_gradients = {0: None}


class CpuEngine:
    def __init__(self):
        self._config = types.SimpleNamespace(train_batch_size=576, train_micro_batch_size_per_gpu=8,
                                             gradient_accumulation_steps=18)
        self.optimizer = CpuZeroOptimizer()
        self.dp_world_size = 4
        self.global_steps = self.micro_steps = self.global_samples = self.skipped_steps = 0
        self._is_gradient_accumulation_boundary = None
        self.zenflow = self.inside_no_sync_ctxt = self.progressive_layer_drop = False
        self.global_rank = self.gas_boundary_ctr = 0
        self.torch_autocast_z0_gradscaler = None
        self.quantizer = False
        self.lr_scheduler = None
        self.checkpoint_engine = types.SimpleNamespace(is_decoupled=lambda: False)
        self.compression_scheduler = types.SimpleNamespace(step=lambda: None)
        self.engine_timers = types.SimpleNamespace(step_timers=[])
        self.tput_timer = types.SimpleNamespace(stop=lambda **kwargs: None)
        self.monitor = types.SimpleNamespace(enabled=False)

    def __getattr__(self, name):
        if name in {'memory_breakdown', 'flops_profiler_enabled', 'eigenvalue_enabled',
                    'autotuning_enabled', 'wall_clock_breakdown', 'fp16_enabled',
                    'amp_enabled'}:
            return lambda: False
        if name in {'_start_timers', '_stop_timers'}:
            return lambda *args: None
        raise AttributeError(name)

    def bfloat16_enabled(self):
        return True

    def zero_optimization(self):
        return True

    def zero_optimization_stage(self):
        return 2

    def gradient_clipping(self):
        return 0.0

    def steps_per_print(self):
        return None

    def backward(self, loss_mean):
        # Mirrors the audited Engine per-GAS scale and boundary propagation.
        self.optimizer.is_gradient_accumulation_boundary = self.is_gradient_accumulation_boundary()
        self.optimizer.accumulator += self._scale_loss_by_gas(loss_mean)
        self.optimizer.micro_step_id += 1
        self.optimizer.all_grad_tensors = {0: self.optimizer.accumulator}


REAL_METHOD_NAMES = {'set_train_batch_size', 'set_train_micro_batch_size', 'train_batch_size',
                     'train_micro_batch_size_per_gpu', 'gradient_accumulation_steps',
                     'is_gradient_accumulation_boundary', 'set_gradient_accumulation_boundary',
                     '_scale_loss_by_gas', '_take_model_step', 'step'}
NAMESPACE = {'torch': types.SimpleNamespace(Tensor=Fraction), 'DummyOptim': DummyOptim,
             'see_memory_usage': lambda *args, **kwargs: None,
             'logger': types.SimpleNamespace(warning=lambda *args, **kwargs: None)}
for item in SOURCES['engine']['methods']:
    if item['name'] not in REAL_METHOD_NAMES:
        continue
    tree = ast.parse(textwrap.dedent(item['source']))
    exec(compile(tree, item['name'] + '_from_installed_Engine', 'exec'), NAMESPACE)
    setattr(CpuEngine, item['name'], NAMESPACE[item['name']])


def trainer(engine):
    return types.SimpleNamespace(model=engine, training_role='policy', world_size=4,
                                 super_batch_size=144, train_batch_size=8,
                                 accumulation_steps=18, ctrl_train_interval=5,
                                 reward_train_interval=5, global_step=0, rank=0)


def run_phase(owner, step):
    engine = owner.model
    sample_gradients = [Fraction(step * 10 + index, 10000) for index in range(144)]
    boundaries, used = [], []
    for index in range(owner.accumulation_steps):
        lo, hi = index * owner.train_batch_size, (index + 1) * owner.train_batch_size
        mini = sample_gradients[lo:hi]
        used.extend(range(lo, hi))
        engine.backward(sum(mini, Fraction(0)) / len(mini))
        if engine.is_gradient_accumulation_boundary():
            boundaries.append(index + 1)
        before = engine.global_steps
        engine.step()  # Actual installed Engine.step source, not a reimplementation.
        owner.global_step += engine.global_steps - before
    return boundaries, used, sum(sample_gradients, Fraction(0)) / len(sample_gradients)


class DynamicBatchTests(unittest.TestCase):
    def test_fresh60_update_pattern_has_exact_final_boundaries_and_mean_gradients(self):
        engine = CpuEngine()
        owner = trainer(engine)
        modes = PolicyJointBatchMode(owner, verify_sources=False)
        for step in range(1, 61):
            descriptor = modes.begin(step)
            boundaries, used, mean_gradient = run_phase(owner, step)
            self.assertEqual(boundaries, [descriptor['gas']])
            self.assertEqual(used, list(range(144)))
            self.assertEqual(engine.optimizer.committed_gradients[-1], mean_gradient)
            modes.complete(step)
        self.assertEqual((engine.global_steps, engine.global_samples, engine.micro_steps), (60, 34560, 1296))
        self.assertEqual(engine.gas_boundary_ctr, 60)
        self.assertEqual(len(engine.optimizer.committed_gradients), 60)

    def test_joint5_10_15_remainders_are_safe_without_counter_rebasing(self):
        self.assertEqual([expected_completed_microsteps(step - 1) for step in (5, 10, 15)], [72, 180, 288])
        self.assertTrue(all(expected_completed_microsteps(step - 1) % 36 == 0 for step in (5, 10, 15)))
        self.assertEqual([expected_completed_microsteps(step) for step in (5, 10, 15)], [108, 216, 324])
        for step in range(1, 100001):
            _, gas = mode_for_step(step)
            self.assertEqual(expected_completed_microsteps(step - 1) % gas, 0)
        for step in (20,25,50,55,30000,30005,1000000):
            k=step//5
            self.assertEqual(expected_completed_microsteps(step-1),108*k-36)
            self.assertEqual(expected_completed_microsteps(step),108*k)

    def test_public_engine_setters_leave_zero_optimizer_gas_stale(self):
        engine = CpuEngine()
        engine.set_train_micro_batch_size(4)
        engine.set_train_batch_size(576)
        self.assertEqual(engine.gradient_accumulation_steps(), 36)
        self.assertEqual(engine.optimizer.gradient_accumulation_steps, 18)

    def test_naive_change_after_one_update_looks_like_one_update_but_commits_halfway(self):
        engine = CpuEngine()
        owner = trainer(engine)
        self.assertEqual(run_phase(owner, 1)[0], [18])
        engine.set_train_micro_batch_size(4)
        engine.set_train_batch_size(576)
        owner.train_batch_size, owner.accumulation_steps = 4, 36
        boundaries, used, full_mean = run_phase(owner, 2)
        self.assertEqual(boundaries, [18])  # Wrong: first72examples commit, next72remain.
        self.assertEqual(engine.global_steps, 2)  # Update-count-only checks miss this.
        self.assertEqual(used, list(range(144)))
        self.assertNotEqual(engine.optimizer.committed_gradients[-1], full_mean)
        self.assertNotEqual(engine.optimizer.micro_step_id, -1)

    def test_no_resume_or_mid_macro_reconstruction_allowed(self):
        engine = CpuEngine()
        owner = trainer(engine)
        run_phase(owner, 1)
        with self.assertRaisesRegex(RuntimeError, 'fresh'):
            PolicyJointBatchMode(owner, verify_sources=False)

    def test_rejects_stale_optimizer_gas_and_live_gradients(self):
        engine = CpuEngine()
        owner = trainer(engine)
        modes = PolicyJointBatchMode(owner, verify_sources=False)
        engine.optimizer.gradient_accumulation_steps = 36
        with self.assertRaisesRegex(RuntimeError, 'stale'):
            modes.begin(1)
        engine.optimizer.gradient_accumulation_steps = 18
        engine.optimizer.all_grad_tensors = {0: Fraction(1)}
        with self.assertRaisesRegex(RuntimeError, 'uncommitted'):
            modes.begin(1)
        engine.optimizer.all_grad_tensors = {0: None}
        engine.optimizer.single_partition_of_fp32_groups[0].grad = Fraction(1)
        with self.assertRaisesRegex(RuntimeError, 'FP32 master'):
            modes.begin(1)

    def test_previous_engine_batch_configuration_cannot_drift(self):
        engine=CpuEngine();owner=trainer(engine)
        modes=PolicyJointBatchMode(owner,verify_sources=False)
        engine.set_train_micro_batch_size(4);engine.set_train_batch_size(576)
        engine.optimizer.gradient_accumulation_steps=36
        with self.assertRaisesRegex(RuntimeError,'drifted'):modes.begin(1)

    def test_override_boundary_or_offload_configuration_rejected(self):
        engine = CpuEngine()
        owner = trainer(engine)
        modes = PolicyJointBatchMode(owner, verify_sources=False)
        engine._is_gradient_accumulation_boundary = False
        with self.assertRaisesRegex(RuntimeError, 'Unsafe'):
            modes.begin(1)
        engine._is_gradient_accumulation_boundary = None
        engine.optimizer.cpu_offload = True
        with self.assertRaisesRegex(RuntimeError, 'GPU-resident'):
            modes.begin(1)

    def test_complete_checks_real_engine_and_actor_counters(self):
        engine = CpuEngine()
        owner = trainer(engine)
        modes = PolicyJointBatchMode(owner, verify_sources=False)
        modes.begin(1)
        with self.assertRaisesRegex(RuntimeError, 'exactly one'):
            modes.complete(1)
        run_phase(owner, 1)
        engine.skipped_steps = 1
        with self.assertRaisesRegex(RuntimeError, 'exactly one'):
            modes.complete(1)

    def test_batch_schedule_and_local_global_cardinality(self):
        for step in range(1, 101):
            micro, gas = mode_for_step(step)
            self.assertEqual(micro * gas, 144)
            self.assertEqual(micro * gas * 4, 576)
        for value in (True, 0, -1, 1.5):
            with self.assertRaises(ValueError):
                mode_for_step(value)


if __name__ == '__main__':
    unittest.main(verbosity=2)
