"""WEAVER-env GPU workers for native online DDP training and IPC inference."""
from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

from weaver_subprocess import cpu_wire, process_identity, receive_frame, send_frame
from weaver_gpu_worker import arm_parent_guard, canonical_uuid


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_backend(raw, *, role, rank):
    from world_backends import WorldBackendConfig
    from weaver_checkpoint import choose_checkpoint
    config = WorldBackendConfig.from_mapping(raw)
    config.validate_admission()
    if (config.backend != 'weaver' or not config.world_trainable or config.reward_provider != 'world_head'
            or config.reward_semantics != 'delta_native_score' or config.terminal_rule != 'horizon_only'):
        raise ValueError('Integrated WEAVER requires the explicit uncalibrated native score and fixed horizon')
    options = config.backend_options
    initial_sha = options.get('initial_checkpoint_sha256', options.get('frozen_checkpoint_sha256'))
    weight_path = choose_checkpoint(Path(options['checkpoint_dir']), options.get('checkpoint_file'))
    if not initial_sha or sha_file(weight_path) != initial_sha:
        raise ValueError('Online WEAVER initial native bundle SHA changed')
    import torch
    expected = os.environ['ACCERL_EXPECTED_GPU_UUID']
    if (os.environ.get('ACCERL_USE_MPS') != '1' or not os.environ.get('CUDA_MPS_PIPE_DIRECTORY')
            or os.environ.get('CUDA_VISIBLE_DEVICES') != expected or torch.cuda.device_count() != 1):
        raise RuntimeError('Online WEAVER requires its exact inherited single UUID and existing MPS pipe')
    torch.cuda.set_device(0)
    actual = canonical_uuid(torch.cuda.get_device_properties(0).uuid)
    if canonical_uuid(expected) != actual:
        raise RuntimeError('Online WEAVER worker GPU binding differs')
    torch.set_num_threads(1)
    torch.manual_seed(int(os.environ.get('ACCERL_SEED', '20261027')) + rank + (1000 if role == 'trainer' else 2000))
    from world_backends.weaver import WeaverBackend
    backend = WeaverBackend.load(options['source_path'], options['checkpoint_dir'], device='cuda',
        dtype='float32', config=config, for_training=role == 'trainer')
    # The new native head is an explicit addition to the World optimizer;
    # encoders and critic remain frozen, with no independent Reward engine.
    for name, parameter in backend.upstream_model.named_parameters():
        parameter.requires_grad_(name.startswith(('wm.','rm.')))
    backend.upstream_model._rm_loss_coeff = 1.
    backend.upstream_model._critic_loss_coeff = 0.
    from weaver_native_reward import install_native_reward_head
    backend.native_reward_controller = install_native_reward_head(backend.upstream_model,options['native_reward_head'])
    backend.compute_dtype = torch.bfloat16
    blocks = 0
    for layer in backend.upstream_model.wm.modules():
        if type(layer).__name__ == 'STAttentionBlock' and hasattr(layer, '_use_checkpoint'):
            layer._use_checkpoint = role == 'trainer'
            blocks += int(role == 'trainer')
    geometry = {'required_context_frames': backend.required_context_frames,
                'training_window_frames': backend.training_window_frames, 'horizon': backend.horizon}
    identity = {'backend': 'weaver', 'world_trainable': True,
                'source_optimizer_steps': 3200, 'source_bundle_sha256': initial_sha}
    receipt = {**identity, 'rank': rank, 'actor_id': rank, 'role': role,
        'model_version': 0, 'worker_identity': process_identity(os.getpid()),
        'gpu': {'physical_gpu': os.environ['ACCERL_PHYSICAL_GPU'], 'uuid': actual, 'cuda_local_device': 0},
        'gpu_uuid': actual, 'geometry': geometry, 'checkpoint_audit': backend.checkpoint_audit,
        'trainable_parameter_count': sum(p.numel() for p in backend.model.parameters() if p.requires_grad),
        'trainable_scope':'wm+rm','source_trainable_scope':'wm',
        'world_parameter_count':sum(p.numel() for name,p in backend.upstream_model.named_parameters() if name.startswith('wm.')),
        'rm_total_parameter_count':sum(p.numel() for name,p in backend.upstream_model.named_parameters() if name.startswith('rm.')),
        'native_reward_head':dict(options['native_reward_head']),
        'activation_checkpoint_blocks': blocks, 'weight_dtype': 'torch.float32',
        'compute_dtype': 'torch.bfloat16', 'python': sys.executable, 'torch_version': torch.__version__,
        'worker_marker': os.environ['WEAVER_GPU_WORKER_ID']}
    if (receipt['world_parameter_count'] != 914935296 or receipt['rm_total_parameter_count'] != 6899201 or
            receipt['trainable_parameter_count'] != 921834497):
        raise ValueError('Integrated native World/RM actual parameter inventory differs from its declared architecture')
    return backend, config, receipt


def snapshot_native(backend, engine, directory, reason, source_identity, run_id):
    import torch
    if engine.rank != 0 or engine.active_update is not None or not engine.publication_idle():
        raise RuntimeError('Only the quiescent rank0 native World owner can publish a snapshot')
    root = Path(os.environ['ACCERL_FINAL_CHECKPOINT']).resolve()
    output = Path(directory).resolve()
    if not output.is_relative_to(root):
        raise ValueError('Native World snapshot must stay within this experiment output')
    output.mkdir(parents=True, exist_ok=True)
    path = output / ('world_p%08d_w%08d_%s.pt' % (engine.policy_step, engine.global_steps, reason))
    if path.exists():
        raise FileExistsError('Native online World snapshots are immutable: ' + str(path))
    model = {name: tensor.detach().cpu().clone() for name, tensor in backend.upstream_model.state_dict().items()}
    if len(model) != 866 or any(not torch.isfinite(t).all().item() for t in model.values() if t.is_floating_point()):
        raise ValueError('Online native World snapshot is incomplete or nonfinite')
    manifest = {'protocol': 'weaver_online_joint_weights_v1', 'run_id': run_id, 'backend': 'weaver',
        'world_trainable': True, 'world_size': 4, 'policy_step': engine.policy_step,
        'successful_updates': engine.global_steps, 'model_version': engine.global_steps,
        'source_optimizer_steps': source_identity['source_optimizer_steps'],
        'source_bundle_sha256': source_identity['source_bundle_sha256'], 'weights_only': True,
        'exact_resume_supported': False, 'reason': str(reason), 'tensor_count': len(model),
        'world_engine_backend': 'torch_ddp_native_flow_matching', 'skipped_steps': engine.skipped_steps,
        'source_trainable_scope':'wm','trained_parameter_families':['wm','rm'],'rm_updates':engine.rm_updates,
        'native_reward_head':dict(engine.native_reward.specification),
        'reward_class_counts':{'scope':'global_selected_frames_once_per_successful_world_update',
            'positive_count':engine.reward_positive_labels,'negative_count':engine.reward_negative_labels,
            'last_cohort':engine.last_reward_cohort,'disabled_updates':engine.rm_disabled_updates}}
    if engine.native_reward.specification.get('mode') == 'libero_discounted_success_value_v1':
        manifest.pop('reward_class_counts')
        manifest['reward_target_counts'] = {
            'scope': 'global_selected_frames_once_per_successful_world_update',
            'valid_count': engine.reward_target_valid_count,
            'masked_count': engine.reward_target_masked_count,
            'target_sum': engine.reward_target_sum,
            'last_cohort': engine.last_reward_cohort,
            'disabled_updates': engine.rm_disabled_updates}
    temporary = path.with_suffix('.tmp')
    try:
        with temporary.open('xb') as handle:
            torch.save({'format': 'weaver_online_joint_weights_v1', 'manifest': manifest, 'model': model}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {**manifest, 'weights_file': str(path), 'relative_weights_file': str(path.relative_to(root)),
            'weights_sha256': sha_file(path), 'weights_bytes': path.stat().st_size,
            'native_cpu_finite_checked': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--role', choices=('trainer', 'inference'), required=True)
    args = parser.parse_args()
    if args.run_id != os.environ.get('ACCERL_RUN_ID') or not 0 <= args.rank < 4:
        raise RuntimeError('Online worker run/rank differs from its launcher')
    arm_parent_guard()
    output = sys.stdout.buffer
    sys.stdout = sys.stderr
    raw = backend = config = engine = operations = resident = exporter = telemetry = receipt = None
    run_id = args.run_id
    try:
        while True:
            try:
                message = receive_frame(sys.stdin.buffer)
            except EOFError:
                return
            try:
                method, values = message['method'], message['args']
                if method == 'load':
                    if raw is not None:
                        raise RuntimeError('Online worker cannot be loaded twice')
                    raw = values[0]
                    if args.role == 'trainer':
                        # World process groups are initialized collectively by
                        # separate setup RPCs after all four CPU actors exist.
                        result = {'worker_identity': process_identity(os.getpid()), 'geometry': {}, 'ready': False}
                    else:
                        backend, config, receipt = load_backend(raw, role=args.role, rank=args.rank)
                        from world_backends.runtime import WeaverActorOperations
                        operations = WeaverActorOperations(backend, config, 0)
                        optimization=getattr(backend,'optimizations',{})
                        if optimization.get('resident_rollout_state',False):
                            from weaver_resident import ResidentRolloutOperations
                            resident = ResidentRolloutOperations(operations,args.run_id,args.rank,
                                process_identity(os.getpid()),
                                max_entries=optimization['resident_max_sessions'],
                                max_bytes=optimization['resident_max_bytes'])
                            receipt=dict(receipt,resident_rollout=resident.report())
                        result = receipt
                elif method == 'setup' and args.role == 'trainer':
                    options = values[0]
                    backend, config, receipt = load_backend(raw, role=args.role, rank=args.rank)
                    import torch
                    from torch.nn.parallel import DistributedDataParallel
                    torch.distributed.init_process_group('nccl',
                        init_method='tcp://%s:%d' % (options['master_addr'], options['master_port']),
                        rank=args.rank, world_size=4, timeout=timedelta(seconds=180))
                    if torch.distributed.get_world_size() != 4 or torch.distributed.get_rank() != args.rank:
                        raise RuntimeError('Native World process group does not have its actual four ranks')
                    model = DistributedDataParallel(backend.model, device_ids=[0],
                        broadcast_buffers=False, find_unused_parameters=True)
                    from weaver_online_engine import WeaverOnlineEngine
                    engine = WeaverOnlineEngine(backend, model=model, rank=args.rank,
                        world_size=4, micro_batch=options['micro_batch'], accumulation_steps=options['accumulation_steps'],
                        learning_rate=1e-5, warmup_updates=20, weight_decay=.01, betas=(.9,.99), max_grad_norm=1.)
                    from cuda_ipc_weights import CudaIPCWeightExporter
                    exporter = CudaIPCWeightExporter(backend.model, 'ctrl', receipt['gpu_uuid'], run_id,
                                                     args.rank, world_backend=backend)
                    engine.publication_idle = lambda: not exporter.active
                    result = dict(receipt, native_engine=engine.state(), group_port=options['master_port'],
                                  distributed_world_size=torch.distributed.get_world_size())
                    receipt = result
                elif method == 'advance' and engine is not None:
                    result = engine.advance_policy(values[0])
                elif method == 'begin_update' and engine is not None:
                    result = engine.begin_update(*values)
                elif method == 'train_micro' and engine is not None:
                    windows, metadata, start, end = values
                    from capture_training import micro_begin, micro_end
                    owner.capture_wm_metadata = metadata
                    owner.global_step = engine.policy_step
                    token = micro_begin(owner, 'ctrl_train', engine, len(windows), start, end)
                    try:
                        result = engine.micro(windows)
                    except BaseException as error:
                        micro_end(owner, token, engine, error)
                        raise
                    micro_end(owner, token, engine)
                    telemetry.event('world_backend_losses', details={'backend': 'weaver', 'rank': args.rank,
                        'update_attempt': engine.global_steps if result['committed'] else engine.global_steps + 1,
                        'micro': engine.micro_steps - 1, 'losses': result['native_losses'],
                        'native_objective': ('flow_matching_v_pred_plus_masked_discounted_success_regression'
                            if engine.native_reward.dense else 'flow_matching_v_pred_plus_native_half_brier'),
                        'reward_training':engine.native_reward.active,'rm_updates':engine.rm_updates,
                        'reward_cohort':engine.last_reward_cohort,
                        'rm_micro':engine.native_reward.last_statistics,'critic_training': False})
                    if result['committed']:
                        import gc
                        import torch
                        telemetry.flush(force_gpu=True)
                        gc.collect()
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        memory = {'allocated_bytes': torch.cuda.memory_allocated(),
                                  'reserved_bytes': torch.cuda.memory_reserved(),
                                  'gradients_released': all(p.grad is None for p in engine.parameters),
                                  'policy_step': engine.policy_step, 'world_step': engine.global_steps}
                        if not memory['gradients_released']:
                            raise RuntimeError('Native World retained gradients beyond its completed update')
                        telemetry.event('weaver_online_update_memory_released', details=memory)
                        result['memory_after_update'] = memory
                elif method == 'state' and engine is not None:
                    result = engine.state()
                elif method == 'export' and exporter is not None:
                    result = exporter.begin(engine.global_steps)
                elif method == 'release' and exporter is not None:
                    result = exporter.finish(*values)
                    import gc
                    import torch
                    gc.collect()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    telemetry.event('weaver_online_publication_released', model_version=result['version'], details={
                        'inventory': result, 'native_engine':engine.state(),'allocated_bytes': torch.cuda.memory_allocated(),
                        'reserved_bytes': torch.cuda.memory_reserved(), 'source_refs_released': True})
                    telemetry.flush()
                elif method == 'snapshot' and engine is not None:
                    result = snapshot_native(backend, engine, *values, receipt, run_id)
                elif method == 'init_rollout' and operations is not None:
                    result = (resident if resident is not None else operations).init_rollout(*values)
                elif method == 'request' and operations is not None:
                    result = (resident if resident is not None else operations).request(*values)
                elif method == 'release_rollout' and operations is not None:
                    result = (resident.release_rollout(*values) if resident is not None else
                              {'released':False,'resident_enabled':False})
                elif method == 'resident_status' and operations is not None:
                    result = resident.report() if resident is not None else {'resident_enabled':False}
                elif method == 'apply' and operations is not None:
                    # No old state survives even a failed/partial publication.
                    if resident is not None:
                        resident.clear()
                    from cuda_ipc_weights import apply_cuda_ipc_weights
                    result = apply_cuda_ipc_weights(backend.model, values[0], 'ctrl', receipt['gpu_uuid'],
                                                    run_id, args.rank, world_backend=backend)
                    (resident if resident is not None else operations).set_published_version(int(result['version']))
                else:
                    raise ValueError('Unknown/unready online worker operation: ' + str(method))
                if telemetry is None and receipt is not None:
                    from capture_telemetry import Telemetry
                    telemetry = Telemetry('ctrl_trainer' if args.role == 'trainer' else 'ctrl_infer', args.rank)
                    telemetry.event('weaver_online_model_verified', model_version=0, details=receipt)
                    if args.role == 'trainer':
                        telemetry.event('weaver_online_trainer_verified', model_version=0, details=receipt)
                    telemetry.event('placement', model_version=0, details={
                        'role': 'ctrl_trainer' if args.role == 'trainer' else 'ctrl_infer',
                        'physical_gpu_uuid': receipt['gpu_uuid'], 'cpu_threads': 1,
                        'physical_gpus': [{'physical_index': int(receipt['gpu']['physical_gpu']), 'uuid': receipt['gpu_uuid']}],
                        'isolated_interpreter': True})
                    if engine is not None:
                        from types import SimpleNamespace
                        owner = SimpleNamespace(telemetry=telemetry, rank=args.rank, world_size=4,
                            training_role='ctrl', global_step=0, active_kinds=('policy','ctrl','reward'),
                            capture_versions={'ctrl': 0})
                    telemetry.flush()
                if method == 'apply':
                    telemetry.event('model_version', model_version=result['version'], details={
                        'meaning': 'successful_source_optimizer_updates', **result})
                    telemetry.flush()
                send_frame(output, {'ok': True, 'result': cpu_wire(result)})
            except BaseException as failure:
                if args.role == 'inference' and isinstance(failure, ValueError) and str(failure).startswith('Stale WEAVER rollout/cache version'):
                    send_frame(output, {'ok': True, 'result': {'_weaver_recoverable_error': 'stale_rollout',
                                                             'message': str(failure)}})
                    continue
                error = traceback.format_exc()
                print(error, file=sys.stderr, flush=True)
                send_frame(output, {'ok': False, 'traceback': error})
                return
    finally:
        if resident is not None:
            resident.clear()
        if telemetry is not None:
            telemetry.close()


if __name__ == '__main__':
    main()
