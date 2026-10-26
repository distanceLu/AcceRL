"""CPU-only, read-only checks of the live primary World ZeRO-2 configuration."""


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _ordinary(value):
    names = {cls.__name__ for cls in type(value).__mro__}
    return not names.intersection({'DTensor', 'FSDPModule'})


def world_zero2_report(engine, rank):
    """Observe actual engine methods, not launcher/config dictionaries; no CUDA calls."""
    _require(isinstance(rank, int) and not isinstance(rank, bool) and 0 <= rank < 4,
             'World audit requires one of four ranks')
    stage, world = int(engine.zero_optimization_stage()), int(engine.dp_world_size)
    micro = int(engine.train_micro_batch_size_per_gpu())
    gas, batch = int(engine.gradient_accumulation_steps()), int(engine.train_batch_size())
    bf16, clipping = bool(engine.bfloat16_enabled()), float(engine.gradient_clipping())
    _require(stage == 2 and world == 4, 'World must remain four-rank DeepSpeed ZeRO-2')
    _require((micro, gas, batch) == (4, 64, 1024), 'Unexpected World micro/GAS/global batch')
    _require(batch == micro * gas * world, 'World global batch arithmetic mismatch')
    _require(bf16 and clipping == 1.0, 'World BF16 and clip1 must remain enabled')
    parameters = list(engine.module.parameters())
    _require(parameters and all(_ordinary(p) for p in parameters)
             and all(_ordinary(m) for m in engine.module.modules()), 'World contains FSDP/DTensor state')
    zero = engine._config.zero_config
    buckets = {'reduce_bucket_size': int(zero.reduce_bucket_size),
               'allgather_bucket_size': int(zero.allgather_bucket_size)}
    _require(all(size == 500000000 for size in buckets.values()), 'World original 500M buckets changed')
    optimizer, seen, chain = engine.optimizer, set(), []
    while id(optimizer) not in seen:
        seen.add(id(optimizer))
        chain.append(type(optimizer).__name__)
        nested = getattr(optimizer, 'optimizer', None)
        if nested is None:
            break
        optimizer = nested
    groups = [{key: list(value) if key == 'betas' else value
               for key, value in group.items() if key in ('name', 'lr', 'betas', 'eps', 'weight_decay')}
              for group in optimizer.param_groups]
    return {'backend': 'deepspeed_zero2', 'model': 'ctrl', 'rank': rank, 'zero_stage': stage,
            'world_size': world, 'micro_batch_size': micro, 'gradient_accumulation_steps': gas,
            'global_batch_size': batch, 'local_batch_size': micro * gas,
            'bf16_enabled': bf16, 'gradient_clipping': clipping, **buckets,
            'optimizer_chain': chain, 'optimizer_param_groups': groups,
            'no_dtensor_parameters': True, 'global_batch_unchanged': True,
            'micro_and_GAS_changed_from_parent': {'micro_batch_size': 2, 'gradient_accumulation_steps': 128},
            'legacy_world_memory_policy_batch_unchanged_flag_is_not_micro_GAS_evidence': True}


def validate_world_zero2_receipts(rows):
    """Require all four actual World ranks; used by the real controller acceptance path."""
    _require(len(rows) == 4, 'Actual World ZeRO2 receipts require four ranks')
    try:
        ranks = {row['details']['rank'] for row in rows}
        actors = {int(row['actor_id']) for row in rows}
        pids = {row['pid'] for row in rows}
        valid = ranks == actors == set(range(4)) and len(pids) == 4
        for row in rows:
            d = row['details']
            valid = valid and d['rank'] == int(row['actor_id']) and int(row['pid']) > 0
            valid = valid and (d['backend'], d['model'], d['zero_stage'], d['world_size'],
                d['micro_batch_size'], d['gradient_accumulation_steps'], d['global_batch_size']) == (
                'deepspeed_zero2', 'ctrl', 2, 4, 4, 64, 1024)
            valid = valid and d['bf16_enabled'] is True and d['gradient_clipping'] == 1.0
            valid = valid and d['reduce_bucket_size'] == d['allgather_bucket_size'] == 500000000
            valid = valid and d['no_dtensor_parameters'] is True and d['global_batch_unchanged'] is True
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError('Malformed actual World ZeRO2 configuration receipt') from exc
    _require(valid, 'Actual World four-rank ZeRO2 micro4/GAS64/global1024 configuration not verified')
    return True
