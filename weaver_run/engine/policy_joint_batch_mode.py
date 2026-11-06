"""Policy ZeRO2 batch switch for this fresh manual run.

No torch import, process control, CUDA synchronization, data sampling, counter
reset or boundary override. This helper must never be injected into a live run.
Freeze a new candidate only after its failed parent has been diagnosed
and all parent GPU processes have exited. Manual stop has no wall deadline.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


GLOBAL_BATCH = 576
LOCAL_BATCH = 144
WORLD_SIZE = 4
INSTALLED_SOURCE_HASHES = {
    'engine': ('/mnt/data/lcx3/envs/merged-env/lib/python3.11/site-packages/deepspeed/runtime/engine.py',
               '310443345985c10ef9b5e8eea4176bb03d981e26fe3671054b331138c7ababdc'),
    'zero': ('/mnt/data/lcx3/envs/merged-env/lib/python3.11/site-packages/deepspeed/runtime/zero/stage_1_and_2.py',
             '478209eee30499ed06aaa199f8aabe760cc176853e251cf929a0ce1333d68b9f'),
}


def mode_for_step(policy_step):
    if isinstance(policy_step, bool) or not isinstance(policy_step, int) or policy_step < 1:
        raise ValueError('A positive integer logical Policy step is required')
    return (4, 36) if policy_step % 5 == 0 else (8, 18)


def expected_completed_microsteps(completed_policy_updates):
    if (isinstance(completed_policy_updates, bool) or not isinstance(completed_policy_updates, int)
            or completed_policy_updates < 0):
        raise ValueError('A nonnegative completed-update counter is required')
    return 18 * completed_policy_updates + 18 * (completed_policy_updates // 5)


def verify_installed_sources():
    for label, (name, expected) in INSTALLED_SOURCE_HASHES.items():
        from importlib.util import find_spec
        package = Path(find_spec('deepspeed').origin).parent
        installed = package / name.split('/deepspeed/', 1)[1]
        if hashlib.sha256(installed.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Re-audit changed DeepSpeed source before dynamic GAS: ' + label)


def _assert_clean_optimizer(engine):
    optimizer = engine.optimizer
    if engine.zero_optimization_stage() != 2 or optimizer.cpu_offload is not False:
        raise RuntimeError('This candidate supports only the audited GPU-resident ZeRO2 optimizer')
    if int(optimizer.micro_step_id) != -1:
        raise RuntimeError('ZeRO optimizer has an unfinished backward accumulation')
    for name in ('all_grad_tensors', 'averaged_gradients'):
        bank = getattr(optimizer, name)
        if not isinstance(bank, dict) or any(value is not None for value in bank.values()):
            raise RuntimeError('ZeRO optimizer still holds uncommitted gradients: ' + name)
    if any(parameter.grad is not None for parameter in optimizer.single_partition_of_fp32_groups):
        raise RuntimeError('ZeRO optimizer still holds an uncommitted FP32 master gradient')


class PolicyJointBatchMode:
    """begin/complete wrap the existing Policy epoch outside learning methods."""

    def __init__(self, trainer, *, writer=None, verify_sources=True):
        if verify_sources:
            verify_installed_sources()
        self.trainer, self.engine, self.writer = trainer, trainer.model, writer
        self.active = None
        if (trainer.training_role != 'policy' or trainer.world_size != WORLD_SIZE
                or trainer.super_batch_size != LOCAL_BATCH
                or trainer.ctrl_train_interval != 5 or trainer.reward_train_interval != 5):
            raise RuntimeError('Dynamic Policy mode requires one Policy owner/world4/local144 and joint interval5')
        if (int(self.engine.global_steps) or int(self.engine.micro_steps)
                or int(self.engine.skipped_steps) or int(trainer.global_step)):
            raise RuntimeError('Construct dynamic Policy mode only at fresh zero-update initialization; no resume')
        if (self.engine.train_micro_batch_size_per_gpu(), self.engine.gradient_accumulation_steps(),
                self.engine.train_batch_size()) != (8, 18, GLOBAL_BATCH):
            raise RuntimeError('Fresh dynamic Policy engine must initialize with Policy8/18/global576')
        if trainer.train_batch_size != 8 or trainer.accumulation_steps != 18:
            raise RuntimeError('Trainer constructor and initial engine batch disagree')
        _assert_clean_optimizer(self.engine)

    def _emit(self, transition, details):
        if self.writer is not None:
            self.writer.event('policy_batch_mode', details={
                'transition': transition, 'rank': self.trainer.rank,
                'world_size': WORLD_SIZE, 'global_batch': GLOBAL_BATCH,
                'local_batch': LOCAL_BATCH, **details})

    def begin(self, policy_step):
        engine, trainer = self.engine, self.trainer
        if self.active is not None:
            raise RuntimeError('Policy batch mode is already active')
        micro, gas = mode_for_step(policy_step)
        completed = policy_step - 1
        if (int(engine.global_steps) != completed or int(trainer.global_step) != completed
                or int(engine.skipped_steps) != 0
                or int(engine.micro_steps) != expected_completed_microsteps(completed)):
            raise RuntimeError('Policy engine/trainer cumulative counters do not match this fresh schedule')
        if (int(engine.micro_steps) % gas != 0 or engine._is_gradient_accumulation_boundary is not None
                or bool(engine.zenflow) or bool(engine.inside_no_sync_ctxt)
                or int(engine.dp_world_size) != WORLD_SIZE):
            raise RuntimeError('Unsafe GAS switch boundary; counters will never be reset or rebased')
        _assert_clean_optimizer(engine)
        old_micro, old_gas = int(engine.train_micro_batch_size_per_gpu()), int(engine.gradient_accumulation_steps())
        expected_old = mode_for_step(completed) if completed else (8, 18)
        if (old_micro, old_gas) != expected_old or int(engine.train_batch_size()) != GLOBAL_BATCH:
            raise RuntimeError('Policy batch configuration drifted before the next scheduled mode')
        if int(engine.optimizer.gradient_accumulation_steps) != old_gas:
            raise RuntimeError('ZeRO optimizer GAS was already stale before switching')
        # These public setters update only Engine config. Their order restores
        # global576 immediately after the micro setter temporarily changes it.
        engine.set_train_micro_batch_size(micro)
        engine.set_train_batch_size(GLOBAL_BATCH)
        engine.optimizer.gradient_accumulation_steps = gas
        trainer.train_batch_size, trainer.accumulation_steps = micro, gas
        if (engine.train_micro_batch_size_per_gpu(), engine.gradient_accumulation_steps(),
                engine.train_batch_size(), trainer.super_batch_size,
                engine.optimizer.gradient_accumulation_steps) != (micro, gas, GLOBAL_BATCH, LOCAL_BATCH, gas):
            raise RuntimeError('Policy micro/GAS/global batch did not synchronize after public setters')
        self.active = {'policy_step': policy_step, 'joint_due': policy_step % 5 == 0,
                       'micro': micro, 'gas': gas, 'old_micro': old_micro, 'old_gas': old_gas,
                       'before_micro_steps': int(engine.micro_steps),
                       'before_global_steps': int(engine.global_steps),
                       'before_global_samples': int(engine.global_samples)}
        self._emit('begin', self.active)
        return dict(self.active)

    def complete(self, policy_step):
        engine, trainer = self.engine, self.trainer
        record = self.active
        if record is None or record['policy_step'] != policy_step:
            raise RuntimeError('Policy completion has no matching batch-mode begin')
        if (int(engine.micro_steps) != record['before_micro_steps'] + record['gas']
                or int(engine.micro_steps) != expected_completed_microsteps(policy_step)
                or int(engine.global_steps) != policy_step or int(trainer.global_step) != policy_step
                or int(engine.skipped_steps) != 0
                or int(engine.global_samples) != record['before_global_samples'] + GLOBAL_BATCH):
            raise RuntimeError('Dynamic Policy mode did not perform exactly one complete update on144 local samples')
        _assert_clean_optimizer(engine)
        self._emit('complete', {**record, 'after_micro_steps': int(engine.micro_steps),
                               'after_global_steps': int(engine.global_steps),
                               'after_global_samples': int(engine.global_samples)})
        self.active = None

