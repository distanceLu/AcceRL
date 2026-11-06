"""Read-only verification of the Policy's original DeepSpeed ZeRO-2 engine."""
from __future__ import annotations

import math


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _plain_tensor(value):
    names = {cls.__name__ for cls in type(value).__mro__}
    return 'DTensor' not in names and 'FSDPModule' not in names


def policy_zero2_report(engine, rank):
    """Inspect the live engine without changing state, hooks or optimizer."""
    _require(isinstance(rank, int) and not isinstance(rank, bool) and 0 <= rank < 4,
             'Policy audit requires one of four ranks')
    stage = int(engine.zero_optimization_stage())
    world = int(engine.dp_world_size)
    micro = int(engine.train_micro_batch_size_per_gpu())
    gas = int(engine.gradient_accumulation_steps())
    batch = int(engine.train_batch_size())
    bf16 = bool(engine.bfloat16_enabled())
    clipping = float(engine.gradient_clipping())
    _require(stage == 2 and world == 4, 'Policy must use original four-rank ZeRO-2')
    _require((micro, gas, batch) == (8, 18, 576), 'Unexpected Policy micro/GAS/global batch')
    _require(batch == micro * gas * world, 'Policy global batch arithmetic mismatch')
    _require(bf16 and clipping == 1.0, 'Policy BF16 and clip1 must remain enabled')
    modules = list(engine.module.modules())
    _require(all(_plain_tensor(module) for module in modules), 'FSDP module remains in Policy')
    parameters = list(engine.module.parameters())
    _require(parameters and all(_plain_tensor(p) for p in parameters),
             'Policy parameters must be replicated ordinary tensors, not DTensor')
    zero = engine._config.zero_config
    buckets = {'reduce_bucket_size': int(zero.reduce_bucket_size),
               'allgather_bucket_size': int(zero.allgather_bucket_size)}
    _require(all(size == 500000000 for size in buckets.values()),
             'Policy original 500M communication buckets changed')
    _require(bool(zero.overlap_comm) and bool(zero.contiguous_gradients)
             and bool(zero.reduce_scatter) and bool(zero.allgather_partitions),
             'Policy original ZeRO-2 communication configuration changed')
    optimizer = engine.optimizer
    optimizer_chain = []
    visited = set()
    while id(optimizer) not in visited:
        visited.add(id(optimizer))
        optimizer_chain.append(type(optimizer).__name__)
        nested = getattr(optimizer, 'optimizer', None)
        if nested is None:
            break
        optimizer = nested
    _require('FusedAdam' in {cls.__name__ for cls in type(optimizer).__mro__},
             'Policy must use original DeepSpeed FusedAdam in AdamW mode')
    _require(bool(getattr(optimizer, 'adam_w_mode', False)), 'Policy AdamW mode is disabled')
    groups = []
    for group in optimizer.param_groups:
        betas = tuple(float(value) for value in group['betas'])
        eps = float(group['eps'])
        decay = float(group['weight_decay'])
        lr = float(group['lr'])
        _require(betas == (0.9, 0.999) and eps == 1e-8 and decay == 0.0,
                 'Policy original Adam beta/eps/decay changed')
        _require(math.isfinite(lr) and lr >= 0, 'Invalid live Policy learning rate')
        _require(all(_plain_tensor(p) for p in group['params']),
                 'Policy optimizer contains DTensor/FSDP parameters')
        groups.append({'name': group.get('name'), 'lr': lr, 'betas': list(betas),
                       'eps': eps, 'weight_decay': decay,
                       'master_parameter_dtypes': sorted({str(p.dtype) for p in group['params']})})
    _require(groups, 'Policy optimizer has no parameter groups')
    return {'backend': 'deepspeed_zero2', 'model': 'policy', 'rank': rank,
            'zero_stage': stage, 'world_size': world, 'micro_batch_size': micro,
            'gradient_accumulation_steps': gas, 'global_batch_size': batch,
            'bf16_enabled': bf16, 'gradient_clipping': clipping, **buckets,
            'optimizer_chain': optimizer_chain, 'optimizer_param_groups': groups,
            'parameter_elements': sum(p.numel() for p in parameters),
            'trainable_parameter_elements': sum(p.numel() for p in parameters if p.requires_grad),
            'full_parameter_replicas': True, 'fsdp_or_dtensor_parameters': False,
            'no_dtensor_parameters': True,
            'original_policy_zero2_config_preserved': True}
