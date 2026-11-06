#!/usr/bin/env python3
"""CPU-only supervisor for one four-rank MPS run up to the existing TB target.

The reviewed real-fill helper supplies PID/birth/UID/run-id scoped cleanup.
This runner does not create or stop an MPS service and has no wall-clock
deadline. It never imports torch or Ray; the existing launcher owns training.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sqlite3
import struct
import subprocess
import sys
import time
import uuid


ROOT = Path('/mnt/data/lcx1/dong/AcceRL-scheduler')
DATA = Path('/mnt/data/lcx1/dong/need_data/mps_85_20261014')
HELPER = Path('/mnt/data/lcx1/dong/need_data/archive/real_fill_experiment_20261014.py')
MPS_PIPE = '/dev/shm/mps_lcx1/pipe'
MARKER = 'ACCERL_RUN_ID'
METRIC = 'Eval/_Global/Average_Return'
LAYOUT = 'four_rank_six_roles'
TZ = dt.timezone(dt.timedelta(hours=8))
TERMINAL = {'completed', 'completed_without_target', 'failed', 'timed_out', 'cleanup_requires_attention'}
MODELS = {'policy': ('policy_train', 576), 'ctrl': ('ctrl_train', 1024), 'reward': ('reward_train', 1024)}
MILESTONE_TAGS = ('TimeTo85/First_Elapsed_Seconds', 'TimeTo85/First_Completion_Unix_Seconds',
                  'TimeTo85/Consecutive_Two_Elapsed_Seconds',
                  'TimeTo85/Consecutive_Two_Completion_Unix_Seconds')


def load_helper():
    path = HELPER if HELPER.is_file() else Path(__file__).with_name('real_fill_experiment.py')
    if not path.is_file():
        raise FileNotFoundError(f'Reviewed process-safety helper is missing: {HELPER}')
    spec = importlib.util.spec_from_file_location('accerl_reviewed_fill_safety', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


safe = load_helper()
atomic_json = safe.atomic_json
proc_identity = safe.proc_identity
same_process = safe.same_process
remember_owned = safe.remember_owned
cleanup = safe.cleanup


def stamp(unix=None):
    return dt.datetime.fromtimestamp(time.time() if unix is None else unix, TZ).isoformat(timespec='seconds')


def load_status(data):
    try:
        return json.loads((data / 'runnerstatus.json').read_text(encoding='utf-8'))
    except FileNotFoundError:
        return {}


def command_output(command, *, env=None, stdin=None):
    completed = subprocess.run(command, env=env, input=stdin, text=True,
                               capture_output=True, timeout=15, check=True)
    return completed.stdout.strip()


def gpu_snapshot():
    raw = command_output(['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu',
                          '--format=csv,noheader,nounits'])
    result = []
    for line in raw.splitlines():
        columns = [part.strip() for part in line.split(',')]
        if int(columns[0]) in (4, 5, 6, 7):
            result.append({'index': int(columns[0]), 'uuid': columns[1],
                           'total_mib': float(columns[2]), 'used_mib': float(columns[3]),
                           'utilization_percent': float(columns[4])})
    result.sort(key=lambda item: item['index'])
    if [item['index'] for item in result] != [4, 5, 6, 7] or len({item['uuid'] for item in result}) != 4:
        raise RuntimeError('Could not identify exactly physical GPUs 4, 5, 6 and 7')
    return result


def mps_clients(server_pid):
    env = dict(os.environ, CUDA_MPS_PIPE_DIRECTORY=MPS_PIPE)
    raw = command_output(['nvidia-cuda-mps-control'], env=env,
                         stdin=f'get_client_list {int(server_pid)}\n')
    # The existing server returns one numeric client PID per line, or no lines.
    if not raw:
        return []
    fields = raw.split()
    if not all(field.isdigit() for field in fields):
        raise RuntimeError(f'Unexpected read-only MPS client-list response: {raw[:160]}')
    return [int(field) for field in fields]


def preflight():
    cards = gpu_snapshot()
    by_uuid = {item['uuid']: item for item in cards}
    raw = command_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
                          '--format=csv,noheader,nounits'])
    services = {}
    for line in raw.splitlines():
        columns = [part.strip() for part in line.split(',')]
        if columns[0] not in by_uuid:
            continue
        identity = proc_identity(int(columns[1]))
        if identity is None or not safe.protected(identity):
            raise RuntimeError(f'Physical GPU {by_uuid[columns[0]]["index"]} has an existing non-MPS job; no job was stopped')
        if identity['uid'] != os.getuid():
            raise RuntimeError('Selected cards have an MPS server belonging to another UID')
        services[identity['pid']] = identity
    if not services:
        raise RuntimeError('The existing MPS server is not present on GPUs 4-7; this runner never starts a replacement')
    if any(mps_clients(pid) for pid in services):
        raise RuntimeError('The existing MPS service already has clients; no client was stopped')
    if any(item['utilization_percent'] > 0 or item['used_mib'] > 1024 for item in cards):
        raise RuntimeError('Selected GPUs are not idle according to the bounded preflight snapshot')
    return {'cards': cards, 'protected_mps_services': list(services.values()),
            'mps_pipe': MPS_PIPE, 'existing_clients': []}


def event(connection, run_id, kind, *, workload=None, actor_id=None, oldest=False, success=False):
    clauses, args = ['run_id=?', 'workload IS NULL' if workload is None else 'workload=?', 'kind=?'], [run_id]
    if workload is not None:
        args.append(workload)
    args.append(kind)
    if actor_id is not None:
        clauses.append('actor_id=?'); args.append(str(actor_id))
    if success:
        clauses.append("status='ok'")
    direction = 'ASC' if oldest else 'DESC'
    row = connection.execute('SELECT * FROM events WHERE ' + ' AND '.join(clauses)
                             + f' ORDER BY ts_ns {direction} LIMIT 1', args).fetchone()
    if row is None:
        return None
    result = dict(row)
    result['details'] = json.loads(result.get('details_json') or '{}')
    return result


def integer(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def rank_updates(connection, run_id):
    """Read the latest cumulative model version, never a bounded-row sum.

capture_training increments model_version only for successful optimizer steps.
engine_global_steps includes skipped steps, so the difference is cumulative
skips even if the experiment has more than the old reporter's 600 events.
"""
    result = {}
    for model, (workload, batch) in MODELS.items():
        ranks = {}
        for rank in range(4):
            kind = 'optimizer_update' if rank == 0 else 'optimizer_update_replica'
            latest = event(connection, run_id, kind, workload=workload, actor_id=rank)
            last_ok = event(connection, run_id, kind, workload=workload, actor_id=rank, success=True)
            first_ok = event(connection, run_id, kind, workload=workload, actor_id=rank, success=True, oldest=True)
            if latest is None:
                ranks[str(rank)] = {'successful_updates': 0, 'engine_global_steps': 0, 'skipped_steps': 0,
                                    'first_success_ns': None, 'last_success_ns': None, 'global_batch_valid': True}
                continue
            successful = integer(latest['model_version'])
            engine_steps = integer(latest['details'].get('engine_global_steps'))
            valid = successful is not None and engine_steps is not None and 0 <= successful <= engine_steps
            ranks[str(rank)] = {'successful_updates': successful, 'engine_global_steps': engine_steps,
                                'skipped_steps': engine_steps - successful if valid else None,
                                'first_success_ns': first_ok['ts_ns'] if first_ok else None,
                                'last_success_ns': last_ok['ts_ns'] if last_ok else None,
                                'global_batch_valid': latest['actual_batch'] == batch,
                                'cumulative_counter_valid': valid, 'latest_ns': latest['ts_ns']}
        signatures = {(item['successful_updates'], item['engine_global_steps'], item['skipped_steps'])
                      for item in ranks.values()}
        result[model] = {'ranks': ranks, 'rank_counts_match': len(signatures) == 1,
                         'global_updates': ranks['0']['successful_updates'], 'global_batch': batch,
                         'global_batch_valid': all(item['global_batch_valid'] for item in ranks.values()),
                         'counter_source': 'latest model_version; engine_global_steps minus model_version for skips'}
    return result


def target_state(latest_evaluation, reached):
    selected = reached or latest_evaluation
    if not selected:
        return {}
    details = selected['details']
    for key in ('threshold_state', 'target_state', 'state'):
        if isinstance(details.get(key), dict):
            return details[key]
    return details


def checkpoint_evidence(data, state, training, last_complete_step):
    path = data / 'checkpoints/manifest.json'
    if not path.is_file():
        return {'status': 'not_saved_yet', 'covers_last_complete_epoch': False}
    manifest = json.loads(path.read_text(encoding='utf-8'))
    weights = path.parent / manifest.get('weights_file', 'latest_weights.pt')
    current = (manifest.get('metadata', {}).get('run_id') == state['run_id']
               and manifest.get('world_size') == 4 and weights.is_file()
               and manifest.get('saved_at_unix', 0) >= state['started_unix'])
    expected = {model: max(item['ranks'][str(rank)]['successful_updates'] or 0 for rank in range(4))
                for model, item in training.items()}
    coverage = (current and manifest.get('steps', {}).get('policy', -1) >= last_complete_step
                and all(manifest.get('steps', {}).get(model, -1) >= count for model, count in expected.items()))
    return {'status': 'current_run' if current else 'stale_or_invalid',
            'covers_last_complete_epoch': coverage, 'expected_latest_updates': expected,
            'manifest': manifest, 'weights_only': True, 'exact_resume_supported': False}


def double_scalar(proto):
    """Read an actual scalar DT_DOUBLE TensorProto without a TensorFlow import."""
    if proto.dtype != 2:
        raise ValueError('Milestone requires DT_DOUBLE, not a rounded float32 scalar')
    dimensions = [dimension.size for dimension in proto.tensor_shape.dim]
    if dimensions and math.prod(dimensions) != 1:
        raise ValueError('Milestone TensorProto is not scalar')
    if proto.tensor_content:
        if len(proto.tensor_content) != 8:
            raise ValueError('Milestone DT_DOUBLE tensor_content must contain exactly eight bytes')
        value = struct.unpack('<d', proto.tensor_content)[0]
    elif len(proto.double_val) == 1:
        value = proto.double_val[0]
    else:
        raise ValueError('Milestone DT_DOUBLE value is absent or has multiple elements')
    if not math.isfinite(value):
        raise ValueError('Milestone timestamp or elapsed value must be finite')
    return value


def tensorboard_evidence(logdir, *, verify=False):
    if not logdir:
        return {'status': 'not_created_yet'}
    directory = Path(logdir)
    files = sorted(directory.glob('events.out.tfevents.*'))
    result = {'status': 'present' if files else 'missing', 'logdir': str(directory),
              'event_files': [item.name for item in files],
              'bytes': sum(item.stat().st_size for item in files), 'metric_tag': METRIC,
              'readable_verified': False}
    if verify and files:
        # This is the CPU supervisor only, never the training child's environment.
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        accumulator = EventAccumulator(str(directory), size_guidance={'scalars': 10, 'images': 1,
                                                                       'histograms': 1, 'tensors': 1})
        accumulator.Reload()
        result['readable_verified'] = True
        result['scalar_tags'] = accumulator.Tags()['scalars']
        result['tensor_tags'] = accumulator.Tags()['tensors']
        if METRIC in result['scalar_tags']:
            samples = accumulator.Scalars(METRIC)
            if samples:
                latest = max(samples, key=lambda sample: sample.wall_time)
                result['latest_metric'] = {'step': latest.step, 'value': latest.value, 'written_unix': latest.wall_time}
        result['milestones'] = {}
        for tag in MILESTONE_TAGS:
            if tag not in result['tensor_tags']:
                continue
            samples = accumulator.Tensors(tag)
            if not samples:
                continue
            latest = max(samples, key=lambda sample: sample.wall_time)
            try:
                result['milestones'][tag] = {'value': double_scalar(latest.tensor_proto),
                                             'step': latest.step, 'written_unix': latest.wall_time,
                                             'dtype': 'float64'}
            except ValueError as error:
                result['milestones'][tag] = {'error': str(error)}
    return result


def collect_report(data, state, *, verify_tb=False):
    evidence = {'run_id': state['run_id'], 'updated': stamp(), 'training': {},
                'target': {}, 'last_complete_epoch_step': 0, 'completed_epochs': 0}
    connection = safe.sqlite_reader(data)
    if connection is None:
        return evidence
    try:
        with connection:
            driver = {kind: event(connection, state['run_id'], kind) for kind in
                      ('rollout_started', 'replay_ready', 'synchronized_epoch_completed',
                       'evaluation_threshold', 'accuracy_target_reached', 'tensorboard_started', 'experiment_completed')}
            evidence['training'] = rank_updates(connection, state['run_id'])
            evidence['target'] = target_state(driver['evaluation_threshold'], driver['accuracy_target_reached'])
            evidence['target_reached_event'] = bool(driver['accuracy_target_reached'])
            evidence['driver'] = {kind: ({'ts_ns': item['ts_ns'], 'details': item['details']} if item else None)
                                  for kind, item in driver.items()}
            completed = driver['synchronized_epoch_completed']
            if completed:
                evidence['last_complete_epoch_step'] = int(completed['details'].get('global_step', 0))
                evidence['completed_epochs'] = int(completed['details'].get('completed_epochs', 0))
            rollout = driver['rollout_started']
            evidence['rollout_started_ns'] = rollout['ts_ns'] if rollout else None
            tb_start = driver['tensorboard_started']
            logdir = tb_start['details'].get('logdir', tb_start['details'].get('log_dir')) if tb_start else None
            if logdir and not Path(logdir).is_absolute():
                logdir = str(Path(state.get('root', str(ROOT))) / logdir)
            evidence['tensorboard'] = tensorboard_evidence(logdir, verify=verify_tb)
            evidence['placements'] = safe.placement_evidence(connection, state['run_id'])
            evidence['checkpoint'] = checkpoint_evidence(data, state, evidence['training'],
                                                          evidence['last_complete_epoch_step'])
    finally:
        connection.close()
    return evidence


def target_times(target):
    first = integer((target.get('first_qualifying') or {}).get('completed_ns'))
    second = integer((target.get('consecutive_qualifying') or {}).get('completed_ns'))
    return first, second


def write_report(data, state, *, persist_status=True, verify_tb=False):
    try:
        evidence = collect_report(data, state, verify_tb=verify_tb)
        state['report'] = evidence
        state.pop('report_error', None)
    except Exception as error:
        state['report_error'] = repr(error)
        evidence = state.get('report', {'run_id': state['run_id']})
    if persist_status:
        atomic_json(data / 'runnerstatus.json', state)
    atomic_json(data / 'reportstatus.json', {'run_id': state['run_id'], 'updated': stamp(),
                                          'state': state['state'], 'error': state.get('report_error'), **evidence})
    first, second = target_times(evidence.get('target', {}))
    start_ns = int(state.get('started_ns', state['started_unix'] * 1e9))
    time_line = lambda name, ns: (f'{name}：{stamp(ns / 1e9)}；从本轮启动计时 {(ns - start_ns) / 60e9:.2f} 分钟。'
                                  if ns else f'{name}：尚未达标。')
    lines = ['# 四卡 ZeRO-2 + MPS：85% 达标训练', '',
             f'状态：{state["state"]}。本轮：`{state["run_id"]}`。最近更新：{stamp()}。', '',
             '只使用物理 GPU4–7。每张卡驻留 Policy / World / Reward 三训练模型及三个独立推理副本；',
             '共 16 个 GPU 进程、24 个模型角色，四 rank 训练按模型串行执行。使用现有 MPS 服务。', '',
             '评估沿用现有 TensorBoard 口径：`Eval/_Global/Average_Return`，task 0 的最近 1000 个完整评估 episode 滑窗。',
             '阈值 0.85；连续两次统计必须含新增完整 episode，窗口可重叠；不足 1000 个样本不触发结束。',
             '保持在线评估，episode 可能跨模型版本，不能解释为冻结权重的独立评估批次。', '',
             time_line('首次达到 85%', first), time_line('连续第二次达到 85% 的评估统计完成时间', second),
             '计时取 StatsActor 完成评估快照的时间；另保留最新完整 episode 的完成时间，不取 TensorBoard 文件写入时间。', '',
             '无次日或固定小时截止；仅目标达成、原 30000 步学习计划耗尽、程序错误，',
             '或 120 分钟无首次更新 / 60 分钟成功更新停滞时停止。', '',
             '| 模型 | 四 rank 成功更新数 | 四 rank 跳步 | 全局 batch |',
             '| --- | --- | --- | --- |']
    for model, (_, batch) in MODELS.items():
        item = evidence.get('training', {}).get(model, {})
        ranks = item.get('ranks', {})
        counts = '/'.join(str(ranks.get(str(rank), {}).get('successful_updates', '—')) for rank in range(4))
        skips = '/'.join(str(ranks.get(str(rank), {}).get('skipped_steps', '—')) for rank in range(4))
        lines.append(f'| {model} | {counts} | {skips} | {batch} |')
    tb = evidence.get('tensorboard', {})
    lines += ['', f'最后完整同步 epoch：{evidence.get("completed_epochs", 0)}；Policy step：{evidence.get("last_complete_epoch_step", 0)}。',
              f'TensorBoard 本轮日志目录：`{tb.get("logdir", "初始化中")}`。',
              '本轮创建新事件文件，保留用户给出的历史曲线。',
              f'最终权重覆盖完整 epoch：{evidence.get("checkpoint", {}).get("covers_last_complete_epoch", False)}。',
              '权重快照在 checkpoints/latest_weights.pt；只保存权重和 buffers，缺优化器、Replay 与 RNG，不能精确续训。',
              f'本轮训练及 Ray 子进程确认停止：{state.get("training_stopped", False)}。',
              f'结束原因：{state.get("reason", "运行中")}。',
              f'校验问题：{json.dumps(state.get("validation_errors", []), ensure_ascii=False)}。', '',
              '只需查看本 README；异常记录在 runner.log，状态及计数留在 runnerstatus.json、collection.sqlite。']
    if state.get('report_error'):
        lines += ['', '报告暂未刷新：' + state['report_error']]
    temporary = data / f'.README.{os.getpid()}.tmp'
    temporary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    os.replace(temporary, data / 'README.md')
    return evidence


def watchdog_reason(evidence, state, now_ns=None):
    now_ns = time.time_ns() if now_ns is None else now_ns
    policy = evidence.get('training', {}).get('policy', {}).get('ranks', {}).get('0', {})
    first = policy.get('first_success_ns')
    if first:
        last = policy.get('last_success_ns') or first
        return 'successful_update_stalled_60_minutes' if now_ns - last > 3600 * 10**9 else None
    started = evidence.get('rollout_started_ns') or state['started_ns']
    return 'no_first_successful_update_120_minutes' if now_ns - started > 7200 * 10**9 else None


def validate_completion(state):
    evidence = state.get('report', {})
    errors = []
    if not state.get('training_stopped'):
        errors.append('Experiment-owned training or Ray processes still alive')
    if not evidence.get('placements', {}).get('complete_expected_placement'):
        errors.append('Four-card / six-role placement not confirmed')
    for model in MODELS:
        item = evidence.get('training', {}).get(model, {})
        if not item.get('rank_counts_match') or not item.get('global_batch_valid'):
            errors.append(model + ': rank counters or global batch disagree')
        if any(rank.get('skipped_steps') != 0 for rank in item.get('ranks', {}).values()):
            errors.append(model + ': skipped optimizer updates occurred')
    if not evidence.get('checkpoint', {}).get('covers_last_complete_epoch'):
        errors.append('Current-run three-model checkpoint does not cover the final complete epoch')
    tb = evidence.get('tensorboard', {})
    if not tb.get('readable_verified') or METRIC not in tb.get('scalar_tags', []):
        errors.append('The new TensorBoard event directory or existing evaluation tag could not be read')
    if evidence.get('target_reached_event'):
        target = evidence.get('target', {})
        first, second = target_times(target)
        consecutive = integer(target.get('streak')) or 0
        first_snapshot = target.get('first_qualifying') or {}
        second_snapshot = target.get('consecutive_qualifying') or {}
        if (consecutive < 2 or not target.get('should_stop') or not first or not second or second <= first
                or any(snapshot.get('rate', 0) < .85 or snapshot.get('sample_count', 0) < 1000
                       or snapshot.get('fresh_count', 0) < 1 or snapshot.get('policy_step', 0) < 1
                       for snapshot in (first_snapshot, second_snapshot))):
            errors.append('Target-reached state does not contain two ordered qualifying evaluation completions')
        if tb.get('latest_metric', {}).get('value', 0) < .85:
            errors.append('The final existing TensorBoard evaluation tag does not confirm the target')
        started_ns = integer(target.get('run_started_ns'))
        if first and second and started_ns is not None:
            expected = ((first - started_ns) / 1e9, first / 1e9,
                        (second - started_ns) / 1e9, second / 1e9)
            for tag, expected_value in zip(MILESTONE_TAGS, expected):
                marker = tb.get('milestones', {}).get(tag, {})
                actual = marker.get('value')
                if (marker.get('dtype') != 'float64' or actual is None
                        or not math.isfinite(actual) or abs(actual - expected_value) >= 1e-6):
                    errors.append(tag + ': missing float64 milestone or differs from threshold state by >= 1 microsecond')
        else:
            errors.append('Threshold state lacks the run-start timestamp needed to verify TensorBoard milestones')
    elif evidence.get('last_complete_epoch_step', 0) < 30000:
        errors.append('Training ended before the target and before its original 30000-step plan completed')
    return errors


def build_command(root):
    return ['bash', str(root / 'run_oft_ctrl_world_train_wm.sh'), '--cuda-visible-devices', '4,5,6,7',
            '--gpu-layout', LAYOUT, '--num-trainer-gpus', '4', '--num-inference-actors', '4',
            '--num-reward-inference-actors', '4', '--num-ctrl-inference-actors', '4',
            '--accumulation-steps', '18', '--ctrl-accumulation-steps', '64', '--reward-accumulation-steps', '8',
            '--exp-name', 'OpenVLA_DS_gipo_DISCRETE_task0_ctrl_wm_four_zero2_mps']


def build_environment(data, run_id, started_ns, cards):
    env = dict(os.environ)
    for key in ('ACCERL_MEASURE_STOP_AFTER_EPOCHS', 'ACCERL_SOFT_STOP_AT', 'ACCERL_CAPTURE_ONLY',
                'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE', 'CUDA_MPS_PINNED_DEVICE_MEM_LIMIT'):
        env.pop(key, None)
    env.update({MARKER: run_id, 'ACCERL_RUN_STARTED_NS': str(started_ns), 'ACCERL_USE_MPS': '1',
                'ACCERL_CAPTURE_ONLY': '1', 'ACCERL_TENSORBOARD': '1',
                'ACCERL_TELEMETRY_DB': str(data / 'collection.sqlite'), 'ACCERL_CPU_THREADS': '1',
                'CUDA_VISIBLE_DEVICES': ','.join(item['uuid'] for item in cards),
                'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'CUDA_MPS_PIPE_DIRECTORY': MPS_PIPE,
                'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                'TF_FORCE_GPU_ALLOW_GROWTH': 'true', 'PYTHONUNBUFFERED': '1',
                'ACCERL_RAY_TMP': f'/dev/shm/am85_{os.getuid()}_{run_id[-8:]}',
                'ACCERL_RAY_SPILL': f'/mnt/data/lcx1/dong/ray_profile_spill/{run_id}',
                'ACCERL_STOP_FILE': str(data / 'stop.flag'), 'ACCERL_FINAL_CHECKPOINT': str(data / 'checkpoints'),
                'ACCERL_FILL_TIMEOUT_SEC': '7200', 'ACCERL_TRIM_IDLE_CACHE': '1',
                'ACCERL_TRIM_MICRO_CACHE': '1', 'ACCERL_SNAPSHOT_INTERVAL': '25',
                'ACCE_CAPTURE_TRACE_EVERY': '10', 'ACCERL_EVAL_TARGET': '0.85',
                'ACCERL_EVAL_CONSECUTIVE': '2', 'ACCERL_EVAL_MIN_EPISODES': '1000'})
    return env


def start_guard(data, state, log):
    env = dict(os.environ); env.pop(MARKER, None)
    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--mode', 'guard',
                                '--data', str(data), '--run-id', state['run_id']], env=env,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    state['guard'] = proc_identity(process.pid)
    if not state['guard']:
        raise RuntimeError('Could not establish guard PID birth identity')
    return process


def run(data, root):
    if os.name != 'posix' or not Path('/proc').is_dir():
        raise RuntimeError('Run mode requires the Linux training server')
    data.mkdir(parents=True, exist_ok=True)
    with safe.exclusive(data / '.runner.lock'):
        prior = load_status(data)
        if prior and (prior.get('state') not in TERMINAL or not prior.get('training_stopped')):
            raise RuntimeError('This experiment already has an unfinished attempt; no duplicate was started')
        pre = preflight()
        started_ns = time.time_ns()
        run_id = 'mps85_' + dt.datetime.now(TZ).strftime('%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
        command = build_command(root)
        state = {'state': 'starting', 'run_id': run_id, 'started': stamp(started_ns / 1e9),
                 'started_unix': started_ns / 1e9, 'started_ns': started_ns, 'root': str(root),
                 'uid': os.getuid(), 'supervisor': proc_identity(os.getpid()), 'command': command,
                 'minimum_child_birth_ticks': int(float(Path('/proc/uptime').read_text().split()[0]) * os.sysconf('SC_CLK_TCK')),
                 'owned': [], 'training_stopped': False, 'preflight': pre,
                 'previous_attempts': (prior.get('previous_attempts', []) + ([{key: prior.get(key) for key in
                                      ('run_id', 'state', 'started', 'finished', 'reason', 'return_code')}] if prior else []))[-10:],
                 'plan': {'physical_gpus': [4, 5, 6, 7], 'layout': LAYOUT, 'mps': True,
                          'global_batches': {'policy': 576, 'ctrl': 1024, 'reward': 1024},
                          'metric_tag': METRIC, 'threshold': 0.85, 'consecutive': 2, 'min_eval_episodes': 1000,
                          'original_policy_plan_steps': 30000, 'total_deadline': None, 'automatic_restarts': False}}
        sources = [root / 'run_oft_ctrl_world_train_wm.sh'] + [root / 'rl/ctrl_world' / name for name in
                   ('ds_wm_discrete_ctrl_train_wm.py', 'training_topology.py', 'capture_hooks.py',
                    'capture_training.py', 'capture_telemetry.py', 'telemetry.py', 'parallel_checkpoint.py',
                    'shared_inference.py', 'evaluation_threshold.py')]
        state['source_sha256'] = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
        state['runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        env = build_environment(data, run_id, started_ns, pre['cards'])
        state['experiment_environment'] = {key: value for key, value in env.items()
                                           if key.startswith(('ACCERL_', 'ACCE_CAPTURE_', 'CUDA_MPS_'))
                                           or key in ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}
        (data / 'stop.flag').unlink(missing_ok=True)
        atomic_json(data / 'runnerstatus.json', state)
        process = guard_process = None
        with (data / 'runner.log').open('a', encoding='utf-8', buffering=1) as log:
            log.write(f'\n[{stamp()}] START {run_id}; no total deadline; no automatic restart.\n')
            try:
                guard_process = start_guard(data, state, log)
                atomic_json(data / 'runnerstatus.json', state)
                process = subprocess.Popen(command, cwd=root, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
                state['child'] = proc_identity(process.pid); state['state'] = 'running'
                next_report, stopping_since = 0, None
                while process.poll() is None:
                    now = time.monotonic()
                    remember_owned(state)
                    if now >= next_report:
                        evidence = write_report(data, state); next_report = now + 30
                        reason = watchdog_reason(evidence, state)
                        if not same_process(state['guard'], proc_identity(state['guard']['pid'])):
                            reason = 'independent_guard_missing'
                        if reason:
                            state['reason'] = reason
                            safe.requested_stop(data, reason)
                    if (data / 'stop.flag').exists():
                        try:
                            stop_reason = json.loads((data / 'stop.flag').read_text(encoding='utf-8')).get('reason')
                            if stop_reason:
                                state.setdefault('reason', stop_reason)
                        except (OSError, ValueError):
                            pass
                        stopping_since = stopping_since or now
                        # One bounded chance to save at an epoch boundary after a failure/stall.
                        if now - stopping_since > 120:
                            cleanup(state, data); break
                    time.sleep(3)
                if process.poll() is None:
                    process.wait(timeout=10)
                state['return_code'] = process.returncode
                stopped = cleanup(state, data)
                time.sleep(2)
                write_report(data, state, verify_tb=True)
                errors = validate_completion(state)
                state['validation_errors'] = errors
                reached = state.get('report', {}).get('target_reached_event', False)
                if not stopped:
                    state['state'] = 'cleanup_requires_attention'
                elif state.get('reason', '').endswith('minutes'):
                    state['state'] = 'timed_out'
                elif process.returncode == 0 and not errors:
                    state['state'] = 'completed' if reached else 'completed_without_target'
                    state['reason'] = 'two_consecutive_existing_tb_evaluations_reached_85_percent' if reached else 'original_30000_step_learning_plan_exhausted'
                else:
                    state['state'] = 'failed'; state.setdefault('reason', 'training_or_final_validation_failed')
            except BaseException as error:
                state['state'] = 'failed'; state['reason'] = repr(error)
                safe.requested_stop(data, 'supervisor_error')
                if process is not None:
                    cleanup(state, data)
                else:
                    state['training_stopped'] = True
                log.write(f'[{stamp()}] SUPERVISOR ERROR {error!r}\n')
            finally:
                state['finished'] = stamp(); write_report(data, state, verify_tb=state['state'] in TERMINAL)
                log.write(f'[{stamp()}] FINISH {state["state"]}: {state.get("reason")}\n')
                if guard_process is not None:
                    try:
                        guard_process.wait(timeout=12)
                        state['guard_exit_code'] = guard_process.returncode
                    except subprocess.TimeoutExpired:
                        state['guard_exit_code'] = None
                    atomic_json(data / 'runnerstatus.json', state)
        return 0 if state['state'] in ('completed', 'completed_without_target') else 1


def guard(data, run_id):
    """Independent liveness/optimizer watchdog; no schedule or GPU work."""
    while True:
        state = load_status(data)
        if state.get('run_id') != run_id:
            return 0
        if state.get('state') in TERMINAL and state.get('training_stopped'):
            return 0
        parent_alive = same_process(state.get('supervisor'), proc_identity(state.get('supervisor', {}).get('pid', -1)))
        reason = None
        if not parent_alive:
            reason = 'independent_guard_supervisor_exited'
        else:
            try:
                reason = watchdog_reason(collect_report(data, state), state)
            except (sqlite3.Error, FileNotFoundError):
                # Do not interpret a brief SQLite writer-lock window as a training failure.
                pass
        if reason:
            safe.requested_stop(data, reason)
            # Every short wait rereads status: terminal runner cleanup wins.
            for _ in range(24):
                time.sleep(5)
                current = load_status(data)
                if current.get('run_id') != run_id:
                    return 0
                if current.get('state') in TERMINAL and current.get('training_stopped'):
                    return 0
            state = load_status(data)
            if state.get('run_id') != run_id:
                return 0
            cleanup(state, data)
            if not same_process(state.get('supervisor'), proc_identity(state.get('supervisor', {}).get('pid', -1))):
                state['state'] = 'timed_out' if reason.endswith('minutes') else 'failed'
                if not state.get('training_stopped'):
                    state['state'] = 'cleanup_requires_attention'
                state['reason'] = reason; state['finished'] = stamp()
                write_report(data, state, verify_tb=True)
            return 0 if state.get('training_stopped') else 1
        time.sleep(5)


def self_test():
    """No GPU, Ray, torch, MPS service calls, or signalling."""
    import tempfile
    with tempfile.TemporaryDirectory(prefix='mps85-cpu-test-') as directory:
        data = Path(directory)
        connection = sqlite3.connect(data / 'collection.sqlite')
        connection.executescript('''CREATE TABLE events (
            id INTEGER PRIMARY KEY, run_id TEXT, ts_ns INTEGER, actor TEXT, actor_id TEXT,
            pid INTEGER, kind TEXT, workload TEXT, model_version TEXT, actual_batch INTEGER,
            status TEXT, details_json TEXT);
            CREATE INDEX events_run_workload ON events(run_id,workload,kind,ts_ns);''')
        for model, (workload, batch) in MODELS.items():
            for rank in range(4):
                kind = 'optimizer_update' if rank == 0 else 'optimizer_update_replica'
                for step in range(1, 1001):
                    # More than 600 events must still produce exactly 1000 updates.
                    connection.execute('INSERT INTO events(run_id,ts_ns,actor,actor_id,pid,kind,workload,model_version,actual_batch,status,details_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                                       ('test', step * 10**9, 'trainer', str(rank), 999, kind,
                                        workload, str(step), batch, 'ok', json.dumps({'engine_global_steps': step})))
        connection.commit(); connection.row_factory = sqlite3.Row
        counts = rank_updates(connection, 'test')
        assert all(item['global_updates'] == 1000 and item['rank_counts_match'] for item in counts.values())
        connection.execute("UPDATE events SET model_version='999',status='skipped' WHERE workload='ctrl_train' AND ts_ns=?", (1000 * 10**9,)); connection.commit()
        counts = rank_updates(connection, 'test')
        assert counts['ctrl']['global_updates'] == 999
        assert all(rank['skipped_steps'] == 1 for rank in counts['ctrl']['ranks'].values())
        assert counts['policy']['global_updates'] == 1000
        connection.close()
        state = {'started_ns': 10**9}
        assert watchdog_reason({}, state, 7201 * 10**9) is None
        assert watchdog_reason({}, state, 7202 * 10**9) == 'no_first_successful_update_120_minutes'
        progress = {'training': {'policy': {'ranks': {'0': {'first_success_ns': 10**9, 'last_success_ns': 8000 * 10**9}}}}}
        assert watchdog_reason(progress, state, 11500 * 10**9) is None
        assert watchdog_reason(progress, state, 11601 * 10**9) == 'successful_update_stalled_60_minutes'
        # There is deliberately no overall time bound when progress continues.
        assert watchdog_reason(progress, state, 8001 * 10**9) is None
        target = target_state({'details': {'state': {'streak': 2, 'first_qualifying': {'completed_ns': 1},
                                                   'consecutive_qualifying': {'completed_ns': 2}}}}, None)
        assert target_times(target) == (1, 2)
        qualifying = {'rate': .85, 'sample_count': 1000, 'fresh_count': 10, 'policy_step': 10}
        completion = {'training_stopped': True, 'report': {
            'target_reached_event': True,
            'target': {'streak': 2, 'should_stop': True, 'run_started_ns': 0,
                       'first_qualifying': dict(qualifying, completed_ns=100),
                       'consecutive_qualifying': dict(qualifying, completed_ns=200)},
            'placements': {'complete_expected_placement': True},
            'checkpoint': {'covers_last_complete_epoch': True},
            'tensorboard': {'readable_verified': True, 'scalar_tags': [METRIC], 'latest_metric': {'value': .85},
                           'milestones': {tag: {'value': value, 'dtype': 'float64'} for tag, value in
                                          zip(MILESTONE_TAGS, (1e-7, 1e-7, 2e-7, 2e-7))}},
            'training': {model: {'rank_counts_match': True, 'global_batch_valid': True,
                                'ranks': {str(rank): {'skipped_steps': 0} for rank in range(4)}} for model in MODELS}}}
        assert not validate_completion(completion)
        completion['report']['target']['consecutive_qualifying']['completed_ns'] = 100
        assert validate_completion(completion)
        completion['report']['target']['consecutive_qualifying']['completed_ns'] = 200
        completion['report']['tensorboard']['latest_metric']['value'] = .84
        assert validate_completion(completion)
        completion['report']['tensorboard']['latest_metric']['value'] = .85
        completion['report']['tensorboard']['milestones'][MILESTONE_TAGS[3]]['value'] += 2e-6
        assert validate_completion(completion)
        completion['report']['tensorboard']['milestones'][MILESTONE_TAGS[3]]['value'] = 2e-7
        completion['report']['tensorboard']['milestones'].pop(MILESTONE_TAGS[0])
        assert validate_completion(completion)
        from types import SimpleNamespace
        scalar_shape = SimpleNamespace(dim=[])
        precise_time = 1791969027.0793169
        assert double_scalar(SimpleNamespace(dtype=2, tensor_shape=scalar_shape,
                                             tensor_content=struct.pack('<d', precise_time), double_val=[])) == precise_time
        assert double_scalar(SimpleNamespace(dtype=2, tensor_shape=scalar_shape,
                                             tensor_content=b'', double_val=[precise_time])) == precise_time
        for invalid in (SimpleNamespace(dtype=1, tensor_shape=scalar_shape, tensor_content=b'', double_val=[1.]),
                        SimpleNamespace(dtype=2, tensor_shape=scalar_shape, tensor_content=b'1234', double_val=[])):
            try:
                double_scalar(invalid)
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid milestone TensorProto was accepted')
        assert '--train-iters' not in build_command(ROOT)
        env_cards = [{'uuid': f'GPU-test{index}'} for index in range(4)]
        # getuid is not available on Windows; emulate only this environment builder.
        from unittest.mock import patch
        with patch.object(os, 'getuid', return_value=1004, create=True), patch.dict(os.environ, {'ACCERL_CAPTURE_ONLY': '1', 'ACCERL_MEASURE_STOP_AFTER_EPOCHS': '5'}):
            env = build_environment(data, 'test12345678', 123, env_cards)
        assert env['CUDA_VISIBLE_DEVICES'].startswith('GPU-') and env['ACCERL_USE_MPS'] == '1'
        assert env['ACCERL_CAPTURE_ONLY'] == '1' and env['ACCERL_TENSORBOARD'] == '1'
        assert 'ACCERL_MEASURE_STOP_AFTER_EPOCHS' not in env
        assert env['CUDA_MPS_PIPE_DIRECTORY'] == MPS_PIPE
    print('CPU_SELF_TEST_OK: cumulative counters beyond 600, skipped steps, first/stall watchdog, no total deadline, MPS UUID/env, target timestamps')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('run', 'guard', 'report', 'self-test'), default='report')
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--run-id')
    args = parser.parse_args()
    if args.mode == 'self-test':
        return self_test()
    if args.mode == 'run':
        return run(args.data, args.root)
    if args.mode == 'guard':
        if not args.run_id:
            parser.error('--run-id is required for guard mode')
        return guard(args.data, args.run_id)
    state = load_status(args.data)
    if not state:
        parser.error('runnerstatus.json does not yet exist')
    write_report(args.data, state, persist_status=False, verify_tb=state['state'] in TERMINAL)
    print(json.dumps({'run_id': state['run_id'], 'state': state['state'], 'report': str(args.data / 'README.md')}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
