"""CPU-only checks for legacy CTRL and the isolated native online WEAVER owner."""

import math


def online_world_identity(config):
    """3200 pretrained updates are provenance; online versions start at zero."""
    from world_backends import WorldBackendConfig
    cfg = config if isinstance(config, WorldBackendConfig) else WorldBackendConfig.from_mapping(config)
    cfg.validate_admission()
    _require(cfg.backend == 'weaver' and cfg.world_trainable is True
             and cfg.backend_options.get('transport') == 'isolated_online_ddp',
             'Online World requires its isolated native DDP configuration')
    _require(cfg.reward_provider == 'world_head' and cfg.reward_semantics == 'delta_native_score'
             and cfg.terminal_rule == 'horizon_only', 'Integrated World requires its uncalibrated native score protocol')
    provenance = cfg.checkpoint_metadata.get('training_provenance', {})
    step = provenance.get('optimizer_steps', provenance.get('global_step', provenance.get('step')))
    _require(type(step) is int and step == 3200
             and cfg.backend_options.get('initial_checkpoint_sha256')
                 == cfg.checkpoint_metadata.get('load_report', {}).get('checkpoint_sha256'),
             'Online World initial provenance must match the evaluated 3200-step native bundle')
    return {'backend': 'weaver', 'world_trainable': True, 'source_optimizer_steps': 3200,
            'source_bundle_sha256': cfg.backend_options['initial_checkpoint_sha256']}


def online_world_asset_report(config, *, verify_checkpoint=False):
    """Read immutable initial assets without relabeling them as online version3200."""
    from world_backends import WorldBackendConfig
    cfg = config if isinstance(config, WorldBackendConfig) else WorldBackendConfig.from_mapping(config)
    identity = online_world_identity(cfg)
    temporary = cfg.to_dict()
    temporary['world_trainable'] = False
    # This reader checks only the immutable initial native bundle documents.
    # Runtime integrated-score semantics are checked separately above; the
    # old fixed-asset reader must not mislabel an uncalibrated RM as success.
    temporary.update(reward_provider='external_vlm', reward_semantics='delta_success', terminal_rule='argmax')
    temporary['backend_options'] = dict(temporary['backend_options'],
        frozen_checkpoint_sha256=identity['source_bundle_sha256'])
    raw = frozen_world_asset_report(temporary, verify_checkpoint=verify_checkpoint)
    return {key: value for key, value in raw.items() if key not in
            ('backend', 'world_trainable', 'fixed_step', 'bundle_sha256')} | identity


def validate_online_world_receipts(rows, config):
    """Require four real native DDP engines, never the CTRL ZeRO2 recipe."""
    identity = online_world_identity(config)
    _require(len(rows) == 4, 'Native WEAVER DDP requires four actual trainer receipts')
    ranks, pids, uuids = set(), set(), set()
    try:
        for row in rows:
            d = row['details']; native = d['native_engine']; rank = d['rank']; pid = int(row['pid'])
            _require(type(rank) is int and rank == int(row['actor_id']) and 0 <= rank < 4,
                     'Native WEAVER trainer rank differs')
            _require(all(type(d.get(k)) is type(v) and d.get(k) == v for k, v in identity.items()),
                     'Native WEAVER trainer initial source differs')
            expected = {'backend': 'weaver', 'backend_engine': 'torch_ddp_native_flow_matching',
                'world_size': 4, 'distributed_world_size': 4, 'micro_batch': 4, 'accumulation_steps': 64, 'global_batch': 1024,
                'optimizer': 'AdamW', 'optimizer_betas': [.9, .99], 'weight_decay': .01,
                'warmup_updates': 20, 'peak_learning_rate': 1e-5,
                'post_warmup_schedule': 'constant', 'max_grad_norm': 1., 'trainable_parameter_count': 921834497,
                'trainable_scope': 'wm+rm', 'optimizer_parameter_tensors': 403}
            _require(all(type(native.get(k)) is type(v) and native.get(k) == v for k, v in expected.items()),
                     'Native WEAVER engine did not verify its actual DDP/flow/optimizer recipe')
            _require(d.get('trainable_parameter_count') == 921834497
                     and d.get('weight_dtype') == 'torch.float32' and d.get('compute_dtype') == 'torch.bfloat16'
                     and type(d.get('activation_checkpoint_blocks')) is int and d['activation_checkpoint_blocks'] > 0,
                     'Native WEAVER precision/checkpoint/trainable inventory differs')
            worker = d['worker_identity']
            _require(worker.get('pid') == pid and worker.get('uid') == 1004
                     and type(worker.get('birth')) is int and worker['birth'] > 0,
                     'Native WEAVER receipt is not its actual isolated GPU child')
            uuid = d['gpu_uuid']
            _require(pid > 0 and isinstance(uuid, str) and uuid.startswith('GPU-'), 'Invalid native process/GPU')
            ranks.add(rank); pids.add(pid); uuids.add(uuid)
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError('Malformed native WEAVER DDP receipt') from error
    _require(ranks == set(range(4)) and len(pids) == len(uuids) == 4,
             'Native WEAVER receipts must identify four distinct GPU owner processes')
    return True


def validate_online_publication(rows, *, version):
    """A committed driver event proves its four ACKs and producer release finished."""
    import re
    _require(type(version) is int and version >= 0 and len(rows) == 1,
             'One final native World publication receipt is required')
    try:
        row = rows[0]; d = row['details']; inventories = d['inventories']; timing = d['driver_timing']
        _require(d.get('kind') == 'ctrl' and d.get('transport') == 'cuda_ipc'
            and type(row.get('model_version')) is int and row['model_version'] == version
            and d.get('receiver_ranks') == [0, 1, 2, 3]
            and len(inventories) == 4 and inventories == [inventories[0]] * 4,
            'Final native World publication does not confirm identical four-rank ACKs')
        inventory = inventories[0]
        _require(inventory.get('kind') == 'ctrl' and type(inventory.get('version')) is int and inventory['version'] == version
            and isinstance(inventory.get('hash'), str) and re.fullmatch('[0-9a-f]{64}', inventory['hash'])
            and type(timing.get('release_ready_ns')) is int and timing['release_ready_ns'] > 0
            and len(timing.get('rank_ACK_observed_ns', [])) == 4
            and all(type(value) is int and value > 0 for value in timing['rank_ACK_observed_ns'])
            and timing['release_ready_ns'] >= max(timing['rank_ACK_observed_ns'])
            and d.get('ack_completed_ns', 0) >= timing['release_ready_ns'],
            'Final native World publication lacks completed producer release')
        publication = inventory.get('publication', {})
        names = publication.get('parameter_names', [])
        _require(publication.get('bundle_metadata', {}).get('reward_provider') == 'world_head'
             and any(name.startswith(('rm.', 'upstream.rm.')) for name in names)
             and any(name.startswith(('wm.', 'upstream.wm.')) for name in names),
             'Native publication must include both WM and RM parameters')
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError('Malformed native World final publication') from error
    return {'version': version, 'publication_hash': inventory['hash'],
            'all_four_receiver_ACKs': True, 'all_four_producer_releases': True}


def validate_online_descriptor(descriptor, config, *, run_id, policy_step, reason=None):
    """Small metadata validation; the dedicated interpreter audits actual tensor bytes."""
    import re
    identity = online_world_identity(config)
    required = {**identity, 'protocol': 'weaver_online_joint_weights_v1', 'run_id': run_id,
        'world_size': 4, 'policy_step': policy_step, 'successful_updates': policy_step // 5,
        'model_version': policy_step // 5, 'weights_only': True,
        'exact_resume_supported': False, 'tensor_count': 866,
        'world_engine_backend': 'torch_ddp_native_flow_matching', 'skipped_steps': 0,
        'source_trainable_scope': 'wm', 'trained_parameter_families': ['wm','rm']}
    _require(isinstance(descriptor, dict) and all(type(descriptor.get(k)) is type(v)
             and descriptor.get(k) == v for k, v in required.items()), 'Native World artifact identity differs')
    _require(reason is None or descriptor.get('reason') == reason, 'Native World artifact reason differs')
    validate_joint_reward_metadata(descriptor, config, updates=policy_step // 5)
    _require(isinstance(descriptor.get('weights_sha256'), str)
             and re.fullmatch('[0-9a-f]{64}', descriptor['weights_sha256'])
             and type(descriptor.get('weights_bytes')) is int and descriptor['weights_bytes'] > 0,
             'Native World artifact has no file checksum/size')
    return identity


def validate_joint_reward_metadata(descriptor, config, *, updates):
    """Head-active updates are a subset of successful joint optimizer updates."""
    from world_backends import WorldBackendConfig
    cfg = config if isinstance(config, WorldBackendConfig) else WorldBackendConfig.from_mapping(config)
    expected = cfg.backend_options.get('native_reward_head')
    _require(isinstance(expected, dict) and expected.get('calibrated') is False
             and descriptor.get('native_reward_head') == expected,
             'Native RM loss/label/uncalibrated-score protocol differs')
    rm_updates = descriptor.get('rm_updates')
    if expected.get('mode') == 'libero_discounted_success_value_v1':
        counts = descriptor.get('reward_target_counts', {})
        _require('reward_class_counts' not in descriptor
             and type(rm_updates) is int and 0 <= rm_updates <= updates
             and counts.get('scope') == 'global_selected_frames_once_per_successful_world_update'
             and all(type(counts.get(key)) is int and counts[key] >= 0
                     for key in ('valid_count', 'masked_count', 'disabled_updates'))
             and counts['disabled_updates'] + rm_updates == updates
             and counts['valid_count'] + counts['masked_count'] == updates * 10240
             and rm_updates <= counts['valid_count'] <= rm_updates * 10240
             and counts['masked_count'] >= counts['disabled_updates'] * 10240
             and type(counts.get('target_sum')) in (int, float)
             and math.isfinite(counts['target_sum']) and 0 <= counts['target_sum'] <= counts['valid_count'],
             'Dense native RM coverage/regression counters differ')
        cohort = counts.get('last_cohort')
        _require((updates == 0 and cohort is None) or (updates > 0 and isinstance(cohort, dict)),
                 'Dense native RM ledger must include its final successful cohort')
        if cohort is not None:
            from fresh_reward_targets import validate_dense_reward_cohort
            validate_dense_reward_cohort(cohort, expected)
            _require(cohort['window_count'] == 1024 and cohort['frame_slice'] == [30, 40]
                 and cohort['global_selected_frames'] == 10240
                 and cohort['valid_count'] <= counts['valid_count']
                 and cohort['masked_count'] <= counts['masked_count']
                 and cohort['target_sum'] <= counts['target_sum']
                 and (cohort['rm_active'] is False or rm_updates > 0),
                 'Dense native final cohort differs from actual selected production geometry/coverage')
        return True
    counts = descriptor.get('reward_class_counts', {})
    _require(type(rm_updates) is int and 0 <= rm_updates <= updates
         and counts.get('scope') == 'global_selected_frames_once_per_successful_world_update'
         and all(type(counts.get(key)) is int and counts[key] >= 0
                 for key in ('positive_count', 'negative_count', 'disabled_updates'))
         and counts['disabled_updates'] + rm_updates == updates,
         'Native RM active/disabled update or global class counters differ')
    _require((updates == 0 and counts.get('last_cohort') is None) or
             (updates > 0 and isinstance(counts.get('last_cohort'), dict)),
             'Native RM class ledger must include its final successful cohort')
    return True


def validate_joint_release_receipts(rows, descriptor):
    """All four real World owners expose the same committed head-active ledger."""
    _require(len(rows) == 4 and {int(row['actor_id']) for row in rows} == set(range(4))
             and len({row['pid'] for row in rows}) == 4,
             'Integrated native publication requires four distinct release owners')
    dense = descriptor.get('native_reward_head', {}).get('mode') == 'libero_discounted_success_value_v1'
    counts = descriptor['reward_target_counts' if dense else 'reward_class_counts']
    for row in rows:
        details = row['details']; native = details.get('native_engine', {})
        _require(details.get('source_refs_released') is True
             and native.get('global_steps') == descriptor['successful_updates']
             and native.get('skipped_steps') == 0 and native.get('gradients_released') is True
             and native.get('rm_updates') == descriptor['rm_updates']
             and native.get('rm_disabled_updates') == counts['disabled_updates']
             and (all(native.get(key) == counts[field] for key, field in (
                     ('reward_target_valid_count', 'valid_count'), ('reward_target_masked_count', 'masked_count'),
                     ('reward_target_sum', 'target_sum'))) if dense else
                  native.get('reward_positive_labels') == counts['positive_count']
                  and native.get('reward_negative_labels') == counts['negative_count'])
             and native.get('last_reward_cohort') == counts['last_cohort'],
             'Integrated native release ledger differs across ranks or from saved head counters')
    return True


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _ordinary(value):
    names = {cls.__name__ for cls in type(value).__mro__}
    return not names.intersection({'DTensor', 'FSDPModule'})


def frozen_world_identity(config):
    """Canonical fixed-World identity; the native bundle hash is not the resume hash."""
    from world_backends import WorldBackendConfig
    cfg = config if isinstance(config, WorldBackendConfig) else WorldBackendConfig.from_mapping(config)
    cfg.validate_admission()
    _require(cfg.backend == 'weaver' and cfg.world_trainable is False,
             'Frozen World identity requires a frozen WEAVER configuration')
    provenance = cfg.checkpoint_metadata['training_provenance']
    step = provenance.get('optimizer_steps', provenance.get('global_step', provenance.get('step')))
    _require(step == 3200, 'This candidate requires the evaluated final 3200-step WEAVER')
    return {'backend': 'weaver', 'world_trainable': False, 'fixed_step': step,
            'bundle_sha256': cfg.backend_options['frozen_checkpoint_sha256']}


def validate_frozen_world_receipts(rows, config):
    """Require four real child-model receipts, never CPU bridge actor placements."""
    expected = frozen_world_identity(config)
    _require(len(rows) == 4, 'Frozen WEAVER receipts require exactly four ranks')
    try:
        ranks, actors, pids, uuids = set(), set(), set(), set()
        for row in rows:
            d = row['details']
            rank, actor, pid = d['rank'], int(row['actor_id']), int(row['pid'])
            _require(type(rank) is int and rank == actor and 0 <= rank < 4 and pid > 0,
                     'Frozen WEAVER rank/actor/PID differs')
            _require(all(d.get(k) == v for k, v in expected.items()),
                     'Frozen WEAVER receipt bundle differs from configuration')
            uuid = d.get('gpu_uuid', d.get('uuid', d.get('gpu_binding', {}).get('uuid')))
            _require(isinstance(uuid, str) and uuid.removeprefix('GPU-') != '',
                     'Frozen WEAVER receipt requires its actual GPU UUID')
            ranks.add(rank); actors.add(actor); pids.add(pid); uuids.add(uuid)
        valid = ranks == actors == set(range(4)) and len(pids) == len(uuids) == 4
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError('Malformed frozen WEAVER model identity receipt') from exc
    _require(valid, 'Frozen WEAVER receipts must cover four unique child models and GPUs')
    return True


def frozen_world_asset_report(config, *, verify_checkpoint=False):
    """Check the immutable native bundle without constructing torch or CUDA state.

    The optional large-file SHA is for one prelaunch/loader audit, never polling.
    """
    import hashlib
    import json
    from pathlib import Path
    from world_backends import WorldBackendConfig
    cfg = config if isinstance(config, WorldBackendConfig) else WorldBackendConfig.from_mapping(config)
    identity = frozen_world_identity(cfg)
    options = cfg.backend_options
    root = Path(options['checkpoint_dir']).resolve()
    filename = options.get('checkpoint_file', 'checkpoint.pt')
    checkpoint = (root / filename).resolve()
    _require(checkpoint.is_relative_to(root) and checkpoint.is_file(),
             'Frozen native checkpoint must exist inside its bundle')
    manifest = json.loads((root / 'bundle_manifest.json').read_text(encoding='utf-8'))
    _require(manifest.get(filename) == identity['bundle_sha256'],
             'Frozen bundle manifest differs from configured checkpoint SHA')
    metadata = json.loads((root / options.get('metadata_file', 'backend_metadata.json')).read_text(encoding='utf-8'))
    _require(metadata == cfg.checkpoint_metadata, 'Frozen bundle disk metadata differs from configuration')
    checked = {}
    for name in ('config.yaml', 'backend_metadata.json', 'norm_stats_relabel.json'):
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        _require(manifest.get(name) == actual, 'Frozen bundle document hash differs: ' + name)
        checked[name] = actual
    norm = json.loads((root / 'norm_stats_relabel.json').read_text(encoding='utf-8'))
    norm_hash = hashlib.sha256(json.dumps(norm, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    _require(metadata.get('normalization_sha256') == norm_hash,
             'Frozen World normalizer differs from its training export')
    if verify_checkpoint:
        digest = hashlib.sha256()
        with checkpoint.open('rb') as source:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                digest.update(block)
        _require(digest.hexdigest() == identity['bundle_sha256'], 'Frozen native weight file SHA differs')
    return {**identity, 'checkpoint': str(checkpoint), 'checkpoint_bytes': checkpoint.stat().st_size,
            'checkpoint_sha256_verified': bool(verify_checkpoint), 'document_sha256': checked,
            'normalization_sha256': norm_hash, 'real_model_loaded': False, 'cuda_initialized': False}


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


def audit_online_native_cpu(request, *, torch_module=None):
    """One mmap/finite/SHA scan in the dedicated WEAVER interpreter, no models."""
    import hashlib
    import json
    import os
    from pathlib import Path
    import sys
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    descriptor = request['descriptor']
    identity = validate_online_descriptor(descriptor, request['config'], run_id=request['run_id'],
        policy_step=request['policy_step'], reason=request['reason'])
    root, path = Path(request['checkpoint_root']), Path(descriptor['weights_file'])
    _require(root.is_absolute() and root.resolve() == root and path.is_absolute()
             and path.resolve() == path and path.is_relative_to(root),
             'Native artifact must remain in this checkpoint root without symlinks')
    _require(descriptor.get('relative_weights_file') == str(path.relative_to(root)),
             'Native relative and absolute artifact paths differ')
    _require(path.is_file() and path.stat().st_size == descriptor['weights_bytes'],
             'Native artifact size differs')
    stat = path.stat()
    stat_identity = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns]
    key = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    cache = root / 'online_world_cpu_audit.json'
    if cache.is_file() and not cache.is_symlink():
        previous = json.loads(cache.read_text(encoding='utf-8'))
        if previous.get('valid') is True and previous.get('request_sha256') == key and previous.get('stat_identity') == stat_identity:
            return {**previous, 'cache_reused': True}
    if torch_module is None:
        import torch as torch_module
    _require(not torch_module.cuda.is_initialized(), 'Native CPU auditor has a CUDA context')
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    _require(digest.hexdigest() == descriptor['weights_sha256'], 'Native artifact SHA256 differs')
    payload = torch_module.load(str(path), map_location='cpu', weights_only=True, mmap=True)
    manifest = payload.get('manifest', {})
    _require(payload.get('format') == 'weaver_online_joint_weights_v1'
             and manifest and all(type(descriptor.get(k)) is type(v) and descriptor.get(k) == v for k, v in manifest.items())
             and set(manifest) == {'protocol', 'run_id', 'backend', 'world_trainable', 'world_size', 'policy_step',
                'successful_updates', 'model_version', 'source_optimizer_steps', 'source_bundle_sha256',
                'weights_only', 'exact_resume_supported', 'reason', 'tensor_count', 'world_engine_backend', 'skipped_steps',
                'source_trainable_scope', 'trained_parameter_families', 'rm_updates', 'native_reward_head',
                ('reward_target_counts' if descriptor.get('native_reward_head', {}).get('mode') ==
                 'libero_discounted_success_value_v1' else 'reward_class_counts')},
             'Native embedded and outer manifests differ')
    model = payload.get('model', {})
    _require(isinstance(model, dict) and len(model) == 866 and all(isinstance(n, str) for n in model),
             'Native World requires its complete 866-tensor inventory')
    for name, tensor in model.items():
        _require(torch_module.is_tensor(tensor) and tensor.device.type == 'cpu',
                 'Native World inventory is not CPU tensors: ' + name)
        _require(not tensor.is_floating_point() or torch_module.isfinite(tensor).all().item(),
                 'Native World contains nonfinite weights: ' + name)
    del payload, model
    after = path.stat()
    _require([after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns] == stat_identity,
             'Native artifact changed during CPU audit')
    _require(not torch_module.cuda.is_initialized(), 'Native CPU auditor initialized CUDA')
    result = {**identity, **descriptor, 'training_steps': descriptor['successful_updates'],
              'valid': True, 'finite': True, 'tensor_count': 866, 'map_location': 'cpu', 'weights_only': True,
              'mmap': True, 'cuda_initialized': False, 'python': sys.executable,
              'request_sha256': key, 'stat_identity': stat_identity, 'cache_reused': False}
    temporary = cache.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(result, handle, sort_keys=True, indent=2)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, cache)
    return result


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(description='CPU-only native WEAVER final artifact audit')
    parser.add_argument('--audit-online-native', required=True)
    args = parser.parse_args()
    try:
        with open(args.audit_online_native, encoding='utf-8') as handle:
            request = json.load(handle)
        result = audit_online_native_cpu(request)
    except Exception as error:
        result = {'valid': False, 'error': repr(error)}
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result['valid'] else 2)
