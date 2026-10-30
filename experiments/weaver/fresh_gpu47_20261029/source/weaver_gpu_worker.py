"""Foreground GPU owner in WEAVER's own environment; no Ray/DeepSpeed imports."""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import os
from pathlib import Path
import signal
import sys
import threading
import time
import traceback
import uuid

from weaver_subprocess import cpu_wire, process_identity, receive_frame, send_frame


def canonical_uuid(raw):
    if isinstance(raw, bytes) and len(raw) == 16:
        return 'GPU-' + str(uuid.UUID(bytes=raw))
    return 'GPU-' + str(uuid.UUID(str(raw).removeprefix('GPU-')))


def arm_parent_guard():
    expected = {key: int(os.environ['WEAVER_PARENT_' + key.upper()])
                for key in ('pid', 'uid', 'birth')}
    if os.getuid() != expected['uid']:
        raise RuntimeError('WEAVER child UID differs from its bridge')
    actual = process_identity(expected['pid'])
    if os.getppid() != expected['pid'] or actual is None or any(
            actual[key] != expected[key] for key in expected):
        raise RuntimeError('WEAVER bridge parent identity changed before worker startup')
    # Arm before importing any CUDA library. Recheck PPID to close the prctl race.
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'PR_SET_PDEATHSIG failed')
    if os.getppid() != expected['pid']:
        raise RuntimeError('WEAVER parent exited while arming parent-death protection')
    def guard():
        while True:
            current = process_identity(expected['pid'])
            if (os.getppid() != expected['pid'] or current is None
                    or any(current[key] != expected[key] for key in expected)):
                os._exit(71)
            time.sleep(1)
    threading.Thread(target=guard, daemon=True).start()
    return expected


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def build_frozen_backend(raw, actor_id):
    from world_backends import WorldBackendConfig
    from world_backends.runtime import WeaverActorOperations
    from world_backend_audit import frozen_world_asset_report
    config = WorldBackendConfig.from_mapping(raw)
    config.validate_admission()
    if config.backend != 'weaver' or config.world_trainable or config.reward_provider != 'external_vlm':
        raise ValueError('This isolated worker only serves frozen WEAVER with external Reward')
    options = config.backend_options
    assets = frozen_world_asset_report(config, verify_checkpoint=True)
    fixed_identity = {key: assets[key] for key in ('backend', 'world_trainable', 'fixed_step', 'bundle_sha256')}
    if fixed_identity['fixed_step'] != 3200:
        raise ValueError('This candidate requires the evaluated step3200 WEAVER bundle')
    import torch
    if os.environ.get('ACCERL_USE_MPS') != '1' or not os.environ.get('CUDA_MPS_PIPE_DIRECTORY'):
        raise RuntimeError('Frozen WEAVER requires the explicitly inherited existing MPS pipe')
    expected_uuid = os.environ['ACCERL_EXPECTED_GPU_UUID']
    if os.environ.get('CUDA_VISIBLE_DEVICES') != expected_uuid or torch.cuda.device_count() != 1:
        raise RuntimeError('Frozen WEAVER must see exactly its one admitted physical UUID')
    torch.cuda.set_device(0)
    actual_uuid = canonical_uuid(torch.cuda.get_device_properties(0).uuid)
    if actual_uuid.removeprefix('GPU-').lower() != expected_uuid.removeprefix('GPU-').lower():
        raise RuntimeError('WEAVER GPU worker UUID differs from its Ray resource reservation')
    torch.set_num_threads(int(os.environ.get('ACCERL_CPU_THREADS', '1')))
    seed = (int(os.environ.get('ACCERL_SEED', '20261010'))
            + int.from_bytes(hashlib.sha256(b'ctrl_infer').digest()[:4], 'big') % 1000000
            + int(actor_id)) % (2 ** 32)
    torch.manual_seed(seed)
    # Follow the already-evaluated FP32 master/BF16 autocast inference recipe.
    if options.get('dtype', 'float32') != 'float32' or options.get('autocast_dtype', 'bfloat16') != 'bfloat16':
        raise ValueError('Frozen WEAVER requires FP32 parameters and BF16 autocast')
    from world_backends.weaver import WeaverBackend
    backend = WeaverBackend.load(options['source_path'], options['checkpoint_dir'],
        device='cuda', dtype='float32', config=config, for_training=False)
    backend.compute_dtype = torch.bfloat16
    backend.model.requires_grad_(False)
    backend.model.eval()
    operations = WeaverActorOperations(backend, config, 0)
    parameters = list(backend.model.parameters())
    if any(p.requires_grad for p in parameters):
        raise RuntimeError('Frozen WEAVER contains trainable parameters')
    receipt = {**fixed_identity, 'frozen_world_identity': fixed_identity,
        'actor_id': int(actor_id), 'rank': int(actor_id), 'model_version': 0, 'optimizer': None,
        'worker_identity': process_identity(os.getpid()),
        'gpu': {'physical_gpu': os.environ['ACCERL_PHYSICAL_GPU'], 'uuid': actual_uuid,
                'cuda_local_device': 0},
        'bridge_parent': process_identity(os.getppid()),
        'checkpoint_id': config.checkpoint_metadata['checkpoint_id'],
        'checkpoint_audit': backend.checkpoint_audit, 'geometry': operations.get_context_geometry(),
        'parameter_count': sum(p.numel() for p in parameters), 'trainable_parameter_count': 0,
        'weight_dtype': str(parameters[0].dtype), 'compute_dtype': str(backend.compute_dtype),
        'worker_marker': os.environ['WEAVER_GPU_WORKER_ID'],
        'gpu_uuid': actual_uuid, 'asset_audit': assets,
        'python': sys.executable, 'torch_version': torch.__version__, 'pid': os.getpid()}
    receipt['seed'] = seed
    return backend, operations, receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--actor-id', required=True, type=int)
    args = parser.parse_args()
    expected_marker = args.run_id + ':weaver:' + str(args.actor_id)
    if not args.run_id or os.environ.get('ACCERL_RUN_ID') != args.run_id or os.environ.get('WEAVER_GPU_WORKER_ID') != expected_marker:
        raise RuntimeError('WEAVER worker run marker differs from its launcher')
    arm_parent_guard()
    protocol_output = sys.stdout.buffer
    # All model/import logging goes to the existing training log. Only our
    # framed responses can reach the private stdout protocol descriptor.
    sys.stdout = sys.stderr
    backend = operations = telemetry = None
    while True:
        try:
            message = receive_frame(sys.stdin.buffer)
        except EOFError:
            return
        try:
            method, values = message['method'], message['args']
            if method == 'load':
                if backend is not None:
                    raise RuntimeError('WEAVER worker cannot load another model')
                backend, operations, result = build_frozen_backend(values[0], args.actor_id)
                from capture_telemetry import Telemetry
                telemetry = Telemetry('ctrl_infer', args.actor_id)
                telemetry.event('weaver_frozen_verified', model_version=0, details=result)
                telemetry.event('placement', model_version=0, details={
                    'role': 'ctrl_infer', 'physical_gpu_uuid': result['gpu_uuid'],
                    'physical_gpus': [{'physical_index': int(result['gpu']['physical_gpu']),
                                      'uuid': result['gpu_uuid']}],
                    'cpu_threads': 1, 'gpu_worker': result['worker_identity'],
                    'isolated_interpreter': True})
                telemetry.flush()
            elif method == 'init_rollout' and operations is not None:
                result = operations.init_rollout(*values)
            elif method == 'request' and operations is not None:
                started_ns = time.time_ns()
                result = operations.request(*values)
                telemetry.event('weaver_isolated_generation', model_version=0,
                    start_ns=started_ns, end_ns=time.time_ns(),
                    details={'predicted_frames': len(values[1]),
                             'timing_scope': 'native_worker_generate_decode_and_CPU_copy_walltime'})
            else:
                raise ValueError('Unknown/unready WEAVER worker method: ' + str(method))
            send_frame(protocol_output, {'ok': True, 'result': cpu_wire(result)})
        except BaseException:
            error = traceback.format_exc()
            print(error, file=sys.stderr, flush=True)
            send_frame(protocol_output, {'ok': False, 'traceback': error})
            # No reset, allocation retry, or silent native-model fallback.
            return


if __name__ == '__main__':
    main()
