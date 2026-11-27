"""Read the actual reduced ZeRO-2 gradient before clipping/AdamW in a fixed-weight microbatch trial."""
from copy import deepcopy
import hashlib
import json
import math
import re
import time

import torch
import torch.distributed as dist

from batch_autotune_runtime import configure_training
from gpu_trial_replay import _digest
from jit_gae_runtime import _checked, run_balanced_trajectory_update
from paired_gpu_trainer import _validate, layout_receipt
from trainer_compile import compile_counters


def _adam_snapshot(optimizer):
    groups, states = [], []
    for group in optimizer.param_groups:
        if len(group['params']) != 1 or group['lr'] != 0:
            raise ValueError('Expected one fixed-weight FP32 Adam partition per optimizer group')
        master, = group['params']
        state = optimizer.state[master]
        if not {'exp_avg', 'exp_avg_sq', 'step'} <= state.keys():
            raise ValueError('Warm up AdamW before freezing its numerical state')
        groups.append(deepcopy({key: value for key, value in group.items() if key != 'params'}))
        states.append({key: value.detach().to('cpu', copy=True) if torch.is_tensor(value) else deepcopy(value)
                       for key, value in state.items()})
    return groups, states


@torch.no_grad()
def _restore_adam(optimizer, snapshot):
    groups, states = snapshot
    for live, group, saved in zip(optimizer.param_groups, groups, states, strict=True):
        if set(live) != set(group) | {'params'}:
            raise ValueError('AdamW optimizer group schema changed across the paired trial')
        live.update(deepcopy(group))
        master, = live['params']
        state = optimizer.state[master]
        if state.keys() != saved.keys():
            raise ValueError('AdamW moment schema changed across the paired trial')
        for key, value in saved.items():
            if torch.is_tensor(value):
                if state[key].shape != value.shape or state[key].dtype != value.dtype:
                    raise ValueError('AdamW moment geometry changed')
                state[key].copy_(value)
            else:
                state[key] = deepcopy(value)


def _gradient_partition(owner):
    # Audited stage_1_and_2.py independent_gradient_partition_epilogue() stores
    # the accumulated, DP-reduced tensors consumed by step(). Parameter.grad is
    # cleared by this point. Discard only alignment padding at each group's end.
    engine = owner.model
    zero = engine.optimizer
    if (engine.zero_optimization_stage() != 2 or zero.cpu_offload or zero.dynamic_loss_scale
            or zero.loss_scale != 1 or not engine.is_gradient_accumulation_boundary()
            or not zero.is_gradient_accumulation_boundary):
        raise RuntimeError('Expected unscaled and GPU-resident ZeRO-2 accumulated gradients at the GAS boundary')
    named = {id(p): (name, p) for name, p in owner.base_model.named_parameters() if p.requires_grad}
    schema, seen, parts = [], set(), []
    for index, params in enumerate(zero.round_robin_bit16_groups):
        group = zero.real_dp_process_group[index]
        if dist.get_rank(group) != owner.rank or dist.get_world_size(group) != 4:
            raise RuntimeError('ZeRO group rank differs from the trainer/data-parallel rank')
        records = []
        for param in params:
            if id(param) in seen or id(param) not in named:
                raise RuntimeError('Duplicate or unknown ZeRO trainable parameter')
            seen.add(id(param))
            records.append((named[id(param)][0], list(param.shape), str(param.dtype)))
        group_size = sum(param.numel() for param in params)
        partition_size = int(zero.partition_size[index])
        if partition_size != zero.single_partition_of_fp32_groups[index].numel():
            raise RuntimeError('ZeRO master partition geometry changed')
        schema.append((records, group_size, partition_size))
        chunks = zero.averaged_gradients[index]
        if not chunks or sum(x.numel() for x in chunks) != partition_size:
            raise RuntimeError('Missing or incomplete reduced ZeRO gradient partition')
        valid = min(partition_size, max(0, group_size - owner.rank * partition_size))
        output = torch.empty(valid, dtype=torch.float32, device='cpu')
        copied = 0
        for chunk in chunks:
            count = min(valid - copied, chunk.numel())
            if count == 0:
                break
            output[copied:copied + count].copy_(chunk.detach().reshape(-1)[:count])
            copied += count
        parts.append(output)
    if not named or seen != named.keys():
        raise RuntimeError('ZeRO gradient inventory differs from the full trainable model')
    digest = hashlib.sha256(json.dumps(schema, separators=(',', ':')).encode()).hexdigest()
    return digest, sum(param.numel() for _, param in named.values()), parts


def _sums(reference, candidate=None):
    if candidate is not None and reference.shape != candidate.shape:
        raise ValueError('Different gradient shard geometry')
    totals = dict(reference_sq=0.)
    if candidate is not None:
        totals.update(candidate_sq=0., diff_sq=0., dot=0., max_abs_error=0.)
    for start in range(0, reference.numel(), 1024 * 1024):
        section = slice(start, start + 1024 * 1024)
        a = reference[section].double()
        totals['reference_sq'] += a.dot(a).item()
        if candidate is not None:
            b = candidate[section].double()
            delta = b - a
            totals['candidate_sq'] += b.dot(b).item()
            totals['dot'] += a.dot(b).item()
            totals['diff_sq'] += delta.dot(delta).item()
            totals['max_abs_error'] = max(totals['max_abs_error'], delta.abs().max().item())
    if not all(map(math.isfinite, totals.values())):
        raise ValueError('Nonfinite optimizer-input gradient in the shape comparison')
    return totals


def _audit_gradient(owner, micro):
    schema, logical_elements, candidate = _gradient_partition(owner)
    if micro == 16:
        owner._shape_grad_reference = schema, candidate
        groups = [_sums(ref) for ref in candidate]
    else:
        reference_schema, baseline = owner._shape_grad_reference
        if schema != reference_schema:
            raise ValueError('ZeRO gradient partition names, shapes or order changed between arms')
        groups = [_sums(ref, other) for ref, other in zip(baseline, candidate, strict=True)]
        del owner._shape_grad_reference
    totals = {key: (max if key == 'max_abs_error' else sum)(row[key] for row in groups)
              for key in groups[0]}
    return dict(elements=sum(x.numel() for x in candidate), logical_elements=logical_elements,
                schema_sha256=schema, finite=True, groups=groups, **totals)


def _prepare(owner, shard, phase, micro):
    actions, logits = _validate(owner, 'jit', shard)
    version = owner.global_step
    zero = owner.model.optimizer
    for group in zero.param_groups:
        group['lr'] = 0.
    optimizer = zero.optimizer
    identity = (shard['cohort_id'], tuple(shard['sample_ids']), tuple(shard['original_denominators']))
    if version == 0:
        if (phase, micro) != ('warmup', 16):
            raise RuntimeError('Initialize optimizer and advantage EMA with one 16x9 warmup')
        owner._shape_cohort = identity
    else:
        if identity != owner._shape_cohort or owner.advantage_ema.updates != version:
            raise RuntimeError('Paired shape requires the original rank cohort and logical EMA version')
        ema = owner.advantage_ema
        if (phase, micro) == ('audit', 16):
            if version != 1 or hasattr(owner, '_shape_snapshot'):
                raise RuntimeError('Freeze the common EMA and AdamW moments immediately after warmup')
            owner._shape_snapshot = _adam_snapshot(optimizer), ema.mean.clone(), ema.variance.clone()
        else:
            moments, mean, variance = owner._shape_snapshot
            _restore_adam(optimizer, moments)
            ema.mean.copy_(mean)
            ema.variance.copy_(variance)
    return actions, logits


def run_shape_arm(owner, arm, shard, *, micro_begin, micro_end, trim):
    match = re.fullmatch(r'shape_(warmup|audit|time)(16|24)', arm)
    if not match:
        raise ValueError(f'Unknown shape comparison arm {arm!r}')
    phase, micro = match[1], int(match[2])
    device = next(owner.model.parameters()).device
    actions, logits = _checked(lambda: _prepare(owner, shard, phase, micro), device)
    version = owner.global_step
    step = version + 1
    configure_training(owner, step, micro)
    owner._policy_joint_batch_mode.begin(step, micro=micro)
    owner.capture_policy_metadata = dict(sample_ids=list(shard['sample_ids']),
        reservation_id=str(shard['cohort_id']), fresh_versions=[version, 0], jit_pending=True)
    before_steps, before_skips = int(owner.model.global_steps), int(owner.model.skipped_steps)
    compile_before = compile_counters()
    gradient = {}
    def inspect():
        gradient.update(_audit_gradient(owner, micro))

    # The shared autotune meter resets process allocator peak stats at the
    # beginning of each micro. Read the previous high-water marks first.
    allocated = reserved = 0
    def capture_peak():
        nonlocal allocated, reserved
        allocated = max(allocated, torch.cuda.max_memory_allocated(device))
        reserved = max(reserved, torch.cuda.max_memory_reserved(device))

    def begin(*args, **kwargs):
        capture_peak()
        return micro_begin(*args, **kwargs)

    actor_start = owner.policy_train_start_step
    owner.policy_train_start_step = 0
    try:
        torch.cuda.synchronize(device)
        dist.barrier()
        torch.cuda.reset_peak_memory_stats(device)
        start_ns = time.time_ns()
        started = time.perf_counter()
        metrics = run_balanced_trajectory_update(owner, shard['observations'], actions, logits,
            shard['original_denominators'], micro_begin=begin, micro_end=micro_end, trim=trim,
            paired_micro=micro, before_optimizer_step=inspect if phase == 'audit' else None)
        owner._policy_joint_batch_mode.complete(step)
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        end_ns = time.time_ns()
        capture_peak()
    finally:
        owner.policy_train_start_step = actor_start
    compile_after = compile_counters()
    losses = metrics['epoch_losses']
    gas = 144 // micro
    if len(losses) != gas or not all(map(math.isfinite, losses)):
        raise RuntimeError('Incomplete or nonfinite paired shape optimizer window')
    layout = layout_receipt(owner, shard['sample_ids'], version)
    by_id = dict(zip(shard['sample_ids'], shard['original_denominators'], strict=True))
    mean, scale = (float(x) for x in metrics['paired_normalization'])
    memory = owner._autobatch_train_report['memory']
    owner.telemetry.flush(force_gpu=True)
    return dict(arm=arm, rank=owner.rank, micro=micro, gas=gas, cohort_id=shard['cohort_id'],
                version=version, version_after=owner.global_step, input_sample_ids=list(shard['sample_ids']),
                input_order_sha256=_digest(shard['sample_ids']),
                optimizer_steps=int(owner.model.global_steps) - before_steps,
                skipped_steps=int(owner.model.skipped_steps) - before_skips, learning_rate=0.,
                consumed_token_denominators=[int(by_id[sid]) for sid in layout['consumed_sample_ids']],
                normalization=dict(mean=mean, scale=scale), shape_gradient=gradient if phase == 'audit' else None,
                opt_step_s=seconds, timing_start_ns=start_ns, timing_end_ns=end_ns,
                peak_allocated_bytes=allocated, peak_reserved_bytes=reserved,
                sampled_peak_used_bytes=metrics['paired_sampled_peak_used_bytes'],
                estimated_peak_used_bytes=memory['peak_used_bytes'], device_total_bytes=memory['total_bytes'],
                trainer_compile=dict(registered=owner._trainer_compile_receipt, before=compile_before,
                    after=compile_after, delta={k: compile_after[k] - v for k, v in compile_before.items()}),
                **layout)
