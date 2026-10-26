"""CPU-only plan: one manual run, one collector, two bounded NVTX windows."""
from __future__ import annotations
import json
import os
from pathlib import Path
import re
import time

MASTER = Path('/mnt/data/lcx1/dong/need_data/cuda_ipc_weight_sync_gpu47_20261022')
NSYS = Path('/mnt/data/lcx1/dong/need_data/tools/nsys_2026_5_1_user/unpacked/opt/nvidia/nsight-systems-cli/2026.5.1/bin/nsys')
PYTHON = Path('/mnt/data/lcx3/envs/merged-env/bin/python')
CAPTURE_STEPS = (14, 15)
CAPTURE_SECONDS = 30
PREFIX = MASTER / 'nsight/contention_windows'
COLLECTOR_AUTH_ENV = 'ACCERL_CONTENTION_COLLECTOR_TOKEN'


def profile_command(source=None, data=None, prefix=None, session_name=None):
    source = Path(source or MASTER / 'source_v1')
    data, prefix = Path(data or MASTER / 'formal'), Path(prefix or PREFIX)
    if source != MASTER / 'source_v1' or data != MASTER / 'formal' or not prefix.is_relative_to(MASTER / 'nsight'):
        raise ValueError('This one-shot plan can only use the new frozen manual run')
    session_name = session_name or 'accel_contention_20261022'
    if not re.fullmatch(r'accel_contention_[a-zA-Z0-9_]+', session_name):
        raise ValueError('Invalid private collector session name')
    return [str(NSYS), 'profile', '--trace=cuda-sw,nvtx,nccl', '--nccl-trace=api',
        '--session-new=' + session_name,
        '--capture-range=nvtx', '--nvtx-capture=PROFILE_WINDOW',
        '--capture-range-end=repeat-shutdown:2:async', '--kill=none', '--wait=primary',
        '--sample=none', '--cpuctxsw=none', '--cuda-event-trace=false',
        '--gpu-metrics-devices=none', '--cuda-flush-interval=1000', '--export=none',
        '--force-overwrite=false', '--output=' + str(prefix),
        str(PYTHON), str(source / 'runner.py'), '--mode', 'manual', '--data', str(data)]


def training_profile_environment():
    return {'ACCERL_NSIGHT_NVTX': '1', 'ACCERL_NSIGHT_STEPS': '14,15',
            'ACCERL_NSIGHT_SECONDS': '30', 'ACCERL_NSIGHT_MODE': 'timer',
            'NSYS_NVTX_PROFILER_REGISTER_ONLY': '0'}


def process_identity(pid):
    try:
        path = Path('/proc') / str(int(pid))
        stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': int(pid), 'uid': path.stat().st_uid, 'birth_ticks': int(stat[19]),
                'state': stat[0], 'command': (path / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')}
    except (OSError, ValueError):
        return None


def trusted_gpu_injection(inherited, receipt, current):
    """Only forward the new collector's injection after fresh PID/birth proof.

    The outer clears all old NSYS/CUDA_INJECTION before starting nsys. The
    runner gets the token as nsys target; this receipt binds that exact launch.
    CPU guard/sample must continue removing every injection/token key.
    """
    token = inherited.get(COLLECTOR_AUTH_ENV, '')
    expected = receipt.get('collector') or {}
    if (not re.fullmatch(r'[0-9a-f]{32}', token) or token != receipt.get('session_token')
            or receipt.get('source') != str(MASTER / 'source_v1')
            or receipt.get('data') != str(MASTER / 'formal')
            or receipt.get('capture_steps') != [14, 15] or not current
            or any(expected.get(key) != current.get(key) for key in ('pid', 'uid', 'birth_ticks'))
            or current.get('uid') != 1004 or current.get('state') in {'Z', 'X'}
            or receipt.get('session_name') != 'accel_contention_' + token
            or receipt.get('command') != profile_command(session_name=receipt.get('session_name'))):
        raise RuntimeError('GPU child profiler injection is not from this verified fresh collector')
    return {key: value for key, value in inherited.items()
            if key.startswith(('NSYS_', 'CUDA_INJECTION'))}


def load_trusted_gpu_injection(inherited=None):
    inherited = dict(os.environ if inherited is None else inherited)
    if not re.fullmatch(r'[0-9a-f]{32}', inherited.get(COLLECTOR_AUTH_ENV, '')):
        raise RuntimeError('No authorized fresh collector token; old injection must be discarded')
    deadline = time.monotonic() + 10
    while True:
        try:
            receipt = json.loads((MASTER / 'measurement_launch.json').read_text())
            expected = receipt.get('collector') or {}
            if expected:
                return trusted_gpu_injection(inherited, receipt, process_identity(expected['pid']))
        except (OSError, ValueError):
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError('Fresh collector identity receipt unavailable; no arbitrary inherited injection accepted')
        time.sleep(.1)


def export_command(report, output):
    report, output = Path(report), Path(output)
    if not report.is_relative_to(MASTER / 'nsight') or not output.is_relative_to(MASTER / 'nsight'):
        raise ValueError('CPU export paths must stay inside this new master/nsight')
    return [str(NSYS), 'export', '--type=sqlite', '--include-json', 'true',
            '--force-overwrite=false', '--output=' + str(output), str(report)]


def reporting_contract():
    return {'nsight_collection': True, 'capture_steps': [14, 15],
        'phase_names': {'14': 'policy_only', '15': 'joint_three_trainers'},
        'seconds_per_window': 30, 'max_windows': 2,
        'collector_session_shutdown_after_second': True, 'kill_target': False,
        'manual_training_continues_after_measurement': True,
        'measurement_failure_requests_training_stop': False,
        'hardware_counters': 'not collected; RmProfilingAdminOnly=1',
        'roles_identity_required': 24, 'kernel_presence_requires_phase_interpretation': True,
        'full_run_GPU_coverage': False, 'CUPTI_loss_free_claimed': False,
        'P_only_capture_not_expected_to_have_World_or_Reward_learning_kernels': True,
        'automatic_recapture_or_retry': False,
        'collector_only_watchdog_seconds':45,
        'collector_only_watchdog_action':'verified private session nsys stop; no training stop or signals',
        'exclude_from_clean_speed_comparison': ['entire epochs14and15',
            'each window started_ns through nvtx_end_returned_ns', 'collector asyncfinalization CPU work']}
