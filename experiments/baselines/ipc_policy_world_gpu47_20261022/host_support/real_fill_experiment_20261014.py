#!/usr/bin/env python3
"""One bounded real-Replay fill experiment; CPU-only supervisor and reporter.

Run on the server with its merged-env Python. This module never imports torch
or Ray. GPU work is launched once through the existing training shell script.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time
import uuid

ROOT = Path('/mnt/data/lcx1/dong/AcceRL-scheduler')
DATA = Path('/mnt/data/lcx1/dong/need_data/real_fill_4wm_20261014')
LAYOUT = 'four_rank_six_roles'
MARKER = 'ACCERL_RUN_ID'
TZ = dt.timezone(dt.timedelta(hours=8))
TERMINAL = {'completed', 'failed', 'timed_out', 'cleanup_requires_attention'}
ROLES = ('policy_train', 'ctrl_train', 'reward_train')
TARGETS = {'policy': 576, 'wm': 1024}


def stamp(value=None):
    return dt.datetime.fromtimestamp(time.time() if value is None else value, TZ).isoformat(timespec='seconds')


def atomic_json(path, value):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_status(data):
    try:
        return json.loads((data / 'runnerstatus.json').read_text(encoding='utf-8'))
    except FileNotFoundError:
        return {}


def proc_identity(pid):
    """Birth ticks, UID and current marker are rechecked before every signal."""
    base = Path('/proc') / str(int(pid))
    try:
        uid = base.stat().st_uid
        raw = (base / 'stat').read_text()
        tail = raw[raw.rfind(')') + 2:].split()
        start_ticks, state = int(tail[19]), tail[0]
        env = dict(item.split(b'=', 1) for item in (base / 'environ').read_bytes().split(b'\0') if b'=' in item)
        command = (base / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        comm = (base / 'comm').read_text().strip()
        return {'pid': int(pid), 'uid': uid, 'birth_ticks': start_ticks,
                'state': state, 'run_id': env.get(MARKER.encode(), b'').decode(errors='replace'),
                'command': command[:600], 'comm': comm}
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
        return None


def same_process(expected, current, *, run_id=None):
    if not expected or not current or current['state'] in ('Z', 'X'):
        return False
    if any(expected.get(key) != current.get(key) for key in ('pid', 'uid', 'birth_ticks')):
        return False
    return run_id is None or current.get('run_id') == run_id


def protected(identity):
    # Existing MPS services are outside this experiment, even if mislabelled.
    value = (identity.get('command', '') + ' ' + identity.get('comm', '')).lower()
    return 'nvidia-cuda-mps' in value


def scan_owned(state):
    run_id, uid = state['run_id'], state['uid']
    minimum_birth = state['minimum_child_birth_ticks']
    found = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        current = proc_identity(int(entry.name))
        if (current and current['uid'] == uid and current['run_id'] == run_id
                and current['birth_ticks'] >= minimum_birth and current['state'] not in ('Z', 'X')
                and not protected(current)):
            found.append(current)
    return found


def remember_owned(state):
    known = {item['pid']: item for item in state.get('owned', [])}
    for item in scan_owned(state):
        old = known.get(item['pid'])
        # A newly observed PID incarnation is recorded only after scan_owned
        # verifies the exact marker, UID and experiment birth lower bound.
        # checked_signal still rereads its birth immediately before signalling.
        if old is None or old['birth_ticks'] != item['birth_ticks']:
            known[item['pid']] = item
    state['owned'] = list(known.values())
    return [item for item in state['owned']
            if same_process(item, proc_identity(item['pid']), run_id=state['run_id']) and not protected(item)]


def checked_signal(expected, state, sig):
    current = proc_identity(expected['pid'])
    if (same_process(expected, current, run_id=state['run_id'])
            and current['uid'] == os.getuid()
            and current['birth_ticks'] >= state['minimum_child_birth_ticks']
            and not protected(current)):
        try:
            os.kill(expected['pid'], sig)
            return True
        except ProcessLookupError:
            pass
    return False


@contextlib.contextmanager
def exclusive(path, *, blocking=False):
    import fcntl
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def cleanup(state, data, grace=15):
    """No process groups, global Ray stops, pkill, or GPU/MPS mutations."""
    with exclusive(data / '.cleanup.lock', blocking=True):
        owned = remember_owned(state)
        child_pid = state.get('child', {}).get('pid')
        # Application workers first, then launcher/driver and this local Ray.
        owned.sort(key=lambda item: item['pid'] == child_pid)
        sent = []
        for item in owned:
            if checked_signal(item, state, signal.SIGTERM):
                sent.append({'pid': item['pid'], 'birth_ticks': item['birth_ticks'], 'signal': 'TERM'})
        end = time.monotonic() + grace
        while time.monotonic() < end and remember_owned(state):
            time.sleep(.5)
        for item in remember_owned(state):
            if checked_signal(item, state, signal.SIGKILL):
                sent.append({'pid': item['pid'], 'birth_ticks': item['birth_ticks'], 'signal': 'KILL'})
        end = time.monotonic() + 5
        while time.monotonic() < end and remember_owned(state):
            time.sleep(.2)
        survivors = remember_owned(state)
        state['cleanup'] = {'at': stamp(), 'signals': sent, 'survivors': survivors,
                            'policy': 'same UID + exact run_id + verified PID birth; MPS excluded'}
        state['training_stopped'] = not survivors
        return not survivors


def sqlite_reader(data):
    database = data / 'collection.sqlite'
    if not database.exists():
        return None
    connection = sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True, timeout=.5)
    connection.row_factory = sqlite3.Row
    deadline = time.monotonic() + 8
    connection.set_progress_handler(lambda: time.monotonic() > deadline, 10000)
    return connection


def query_events(connection, run_id, kind, *, workload=None, limit=600):
    where = 'workload IS NULL' if workload is None else 'workload=?'
    args = [run_id] + ([] if workload is None else [workload]) + [kind, limit]
    rows = connection.execute(f'SELECT * FROM events WHERE run_id=? AND {where} AND kind=? ORDER BY ts_ns DESC LIMIT ?', args).fetchall()
    result = []
    for row in reversed(rows):
        item = dict(row)
        item['details'] = json.loads(item.get('details_json') or '{}')
        result.append(item)
    return result


def replay_evidence(connection, run_id, replay, *, at_ns=None):
    params = [run_id, replay]
    if at_ns is None:
        resident = 'resident=1'
    else:
        resident = '(added_ns<=? AND (evicted_ns IS NULL OR evicted_ns>?))'
        params.extend([at_ns, at_ns])
    # Each row is one sample_id: reservations and duplicate uses never inflate fill.
    row = connection.execute(f'''SELECT COUNT(*) AS added_unique,
        COALESCE(SUM({resident}),0) AS resident_unique,
        COALESCE(SUM(reserved_count),0) AS reservations,
        COALESCE(SUM(use_count),0) AS actual_training_uses,
        COALESCE(SUM(use_count>0),0) AS used_unique,
        COALESCE(SUM(policy_uses),0) AS policy_uses,
        COALESCE(SUM(ctrl_uses),0) AS ctrl_uses,
        COALESCE(SUM(reward_uses),0) AS reward_uses,
        COALESCE(SUM(resident=0 AND eviction_reason='capacity'),0) AS capacity_evicted,
        COALESCE(SUM(eviction_reason='capacity' AND (first_used_ns IS NULL OR first_used_ns>evicted_ns)),0) AS unused_capacity_evicted
        FROM replay_samples WHERE run_id=? AND replay=? AND added_ns IS NOT NULL''',
        # Resident placeholders appear before WHERE placeholders in this query.
        ([at_ns, at_ns, run_id, replay] if at_ns is not None else params)).fetchone()
    result = dict(row)
    result['target'] = TARGETS[replay]
    result['ready'] = result['resident_unique'] >= TARGETS[replay]
    result['reused'] = result['actual_training_uses'] - result['used_unique']
    threshold = connection.execute('''SELECT added_ns FROM replay_samples
        WHERE run_id=? AND replay=? AND added_ns IS NOT NULL
        ORDER BY added_ns LIMIT 1 OFFSET ?''', (run_id, replay, TARGETS[replay] - 1)).fetchone()
    result['nth_unique_added_ns'] = threshold['added_ns'] if threshold else None
    result['threshold_time_note'] = 'Nth unique addition; exact resident threshold requires driver snapshot if prior evictions occurred'
    return result


def rank_training(connection, run_id):
    result = {}
    for role in ROLES:
        rows = (query_events(connection, run_id, 'optimizer_update', workload=role)
                + query_events(connection, run_id, 'optimizer_update_replica', workload=role))
        ranks = {str(rank): {'successful_updates': 0, 'skipped_updates': 0,
                 'engine_global_steps': 0, 'first_success_ns': None, 'last_success_ns': None,
                 'global_batch_valid': True} for rank in range(4)}
        expected_batch = 576 if role == 'policy_train' else 1024
        for event in sorted(rows, key=lambda item: item['ts_ns']):
            d = event['details']; rank = str(d.get('data_parallel_rank', event['actor_id']))
            if rank not in ranks:
                raise ValueError(f'Unexpected training rank {rank}: {role}')
            entry = ranks[rank]
            success, skipped = int(d.get('successful_optimizer_steps', 0)), int(d.get('skipped_steps', 0))
            entry['successful_updates'] += success; entry['skipped_updates'] += skipped
            entry['engine_global_steps'] = int(d.get('engine_global_steps') or 0)
            entry['global_batch_valid'] &= event['actual_batch'] == expected_batch
            if success:
                entry['first_success_ns'] = entry['first_success_ns'] or event['ts_ns']
                entry['last_success_ns'] = event['ts_ns']
        signatures = {(x['successful_updates'], x['skipped_updates'], x['engine_global_steps']) for x in ranks.values()}
        result[role] = {'ranks': ranks, 'rank_counts_match': len(signatures) == 1,
                        'global_successful_updates': ranks['0']['successful_updates'],
                        'global_batch': expected_batch,
                        'global_batch_valid': all(x['global_batch_valid'] for x in ranks.values()),
                        'events_bounded': True, 'event_limit_per_kind': 600,
                        'limit_reached': len(rows) >= 600}
    return result


def checkpoint_evidence(data, state, complete_step):
    path = data / 'checkpoints' / 'manifest.json'
    if not path.exists():
        return {'status': 'not_saved_yet'}
    manifest = json.loads(path.read_text(encoding='utf-8'))
    weights = path.parent / manifest.get('weights_file', 'latest_weights.pt')
    valid = (manifest.get('metadata', {}).get('run_id') == state['run_id']
             and manifest.get('world_size') == 4 and weights.is_file()
             and manifest.get('saved_at_unix', 0) >= state['started_unix'])
    covers = valid and int(manifest.get('steps', {}).get('policy', -1)) >= complete_step
    return {'status': 'current_run' if valid else 'stale_or_invalid',
            'covers_last_complete_epoch': covers, 'manifest': manifest,
            'weights_only': True, 'exact_resume_supported': False}


def sample_summary(values):
    values = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not values:
        return {'n': 0, 'mean': None, 'p50': None, 'p95': None, 'max': None}
    percentile = lambda fraction: values[min(len(values) - 1, int((len(values) - 1) * fraction))]
    return {'n': len(values), 'mean': sum(values) / len(values),
            'p50': percentile(.5), 'p95': percentile(.95), 'max': values[-1]}


def recent_service_evidence(connection, run_id, workload):
    rows = query_events(connection, run_id, 'service', workload=workload, limit=600)
    successful = [row for row in rows if row['status'] == 'ok']
    batches = [row['actual_batch'] for row in successful if row['actual_batch'] is not None]
    peaks = [row['peak_bytes'] / 2**30 for row in successful if row['peak_bytes'] is not None]
    reserved = [row['reserved_bytes'] / 2**30 for row in successful if row['reserved_bytes'] is not None]
    scopes = {}
    for row in successful:
        scope = row['details'].get('peak_scope', 'unavailable')
        scopes[scope] = scopes.get(scope, 0) + 1
    return {'sample_records': len(rows), 'successful_records': len(successful),
            'failed_records': sum(row['status'] == 'error' for row in rows),
            'oldest_ns': rows[0]['ts_ns'] if rows else None,
            'newest_ns': rows[-1]['ts_ns'] if rows else None,
            'actual_batch': sample_summary(batches),
            'batch_distribution': {str(batch): batches.count(batch) for batch in sorted(set(batches))},
            'batch_one_fraction': batches.count(1) / len(batches) if batches else None,
            'wall_seconds': sample_summary([row['duration_ns'] / 1e9 for row in successful if row['duration_ns'] is not None]),
            'gpu_seconds': sample_summary([row['gpu_ms'] / 1000 for row in successful if row['gpu_ms'] is not None]),
            'peak_gib': max(peaks) if peaks else None,
            'reserved_gib': max(reserved) if reserved else None,
            'memory_peak_scopes': scopes, 'record_limit': 600,
            'scope': 'latest up to 600 service spans, successes only for timing/batch/memory; memory is per-process allocator, not whole-card or isolated model'}


def placement_evidence(connection, run_id):
    rows = query_events(connection, run_id, 'placement', limit=600)
    keep = {}
    role_names = {'trainer', 'policy_infer', 'reward_infer', 'ctrl_infer'}
    for row in rows:
        details = row['details']; role = details.get('role', row['actor'])
        if role not in role_names:
            continue
        mapping = details.get('physical_gpus', [])
        physical = mapping[0].get('physical_index') if len(mapping) == 1 else None
        actual_uuid = details.get('physical_gpu_uuid', '')
        mapped_uuid = mapping[0].get('uuid', '') if len(mapping) == 1 else ''
        normalize = lambda value: str(value).removeprefix('GPU-').lower()
        confirmed = bool(actual_uuid and mapped_uuid and normalize(actual_uuid) == normalize(mapped_uuid))
        item = {'role': role, 'actor_id': row['actor_id'], 'pid': row['pid'],
                'physical_gpu': physical, 'uuid': actual_uuid,
                'uuid_confirmed': confirmed, 'cpu_threads': details.get('cpu_threads'),
                'model_roles': 3 if role == 'trainer' else 1}
        # A repeated placement for the same logical actor replaces its old PID.
        keep[(role, row['actor_id'])] = item
    placements = sorted(keep.values(), key=lambda item: (item['physical_gpu'] if item['physical_gpu'] is not None else -1,
                                                        item['role'], item['actor_id']))
    cards = {str(gpu): {'processes': [], 'gpu_process_count': 0, 'model_role_count': 0} for gpu in (4, 5, 6, 7)}
    for item in placements:
        if str(item['physical_gpu']) in cards:
            card = cards[str(item['physical_gpu'])]
            card['processes'].append(item)
            card['gpu_process_count'] += 1
            card['model_role_count'] += item['model_roles']
    complete = (len(placements) == 16 and all(item['uuid_confirmed'] for item in placements)
                and all(card['gpu_process_count'] == 4 and card['model_role_count'] == 6 for card in cards.values())
                and all({item['role'] for item in card['processes']} == role_names for card in cards.values()))
    return {'actors': placements, 'cards': cards, 'observed_gpu_processes': len(placements),
            'observed_model_roles': sum(item['model_roles'] for item in placements),
            'complete_expected_placement': complete, 'record_limit': 600,
            'scope': 'actor initialization UUID evidence, not a current process-aliveness assertion'}


def world_wait_evidence(connection, run_id):
    rows = query_events(connection, run_id, 'client_wait', workload='ctrl_infer', limit=600)
    methods = {}
    for method in ('request', 'init_latent_state'):
        matching = [row for row in rows if row['details'].get('method') == method]
        successful = [row for row in matching if row['status'] == 'ok']
        wall = [row['duration_ns'] / 1e9 for row in successful if row['duration_ns'] is not None]
        methods[method] = {'sample_records': len(matching), 'successful_records': len(successful),
                           'failed_records': sum(row['status'] == 'error' for row in matching),
                           'wall_seconds': sample_summary(wall), 'cumulative_wall_seconds': sum(wall),
                           'unique_callers': len({(row['actor'], row['actor_id']) for row in matching})}
    return {'sample_records': len(rows), 'methods': methods, 'record_limit': 600,
            'scope': 'latest up to 600 completed World client_wait events across methods; Ray submission to deserialized reply; cumulative waits overlap across workers'}


def wm_episode_evidence(connection, run_id, *, at_ns=None):
    time_filter = '' if at_ns is None else ' AND added_ns<=?'
    args = [run_id] + ([] if at_ns is None else [at_ns])
    where = "run_id=? AND replay IN ('wm','wm_eval') AND added_ns IS NOT NULL AND episode_id IS NOT NULL AND episode_id<>''" + time_filter
    rows = connection.execute(f'''SELECT replay, COUNT(DISTINCT episode_id) AS episodes,
            COUNT(*) AS windows FROM replay_samples WHERE {where} GROUP BY replay''', args).fetchall()
    lanes = {row['replay']: {'episodes': row['episodes'], 'windows': row['windows']} for row in rows}
    total = connection.execute(f'SELECT COUNT(DISTINCT episode_id) FROM replay_samples WHERE {where}', args).fetchone()[0]
    return {'added_real_episodes_unique': total,
            'training': lanes.get('wm', {'episodes': 0, 'windows': 0}),
            'held_out': lanes.get('wm_eval', {'episodes': 0, 'windows': 0}),
            'scope': 'unique real episode_id with windows added to WM train or held-out Replay; held-out windows do not satisfy training fill target'}


def collect_report(data, state):
    connection = sqlite_reader(data)
    if connection is None:
        return {'database_status': 'not_created_yet'}
    try:
        run_id = state['run_id']
        starts = query_events(connection, run_id, 'rollout_started', limit=1)
        ready = query_events(connection, run_id, 'replay_ready', limit=1)
        snapshots = query_events(connection, run_id, 'fill_snapshot', limit=300)
        epochs = query_events(connection, run_id, 'synchronized_epoch_completed', limit=30)
        start_ns = starts[0]['ts_ns'] if starts else None
        complete_step = max([int(x['details'].get('global_step', 0)) for x in epochs] or [0])
        evidence = {'database_status': 'ok', 'rollout_started_ns': start_ns,
                    'replay_ready_ns': ready[0]['ts_ns'] if ready else None,
                    'fill_snapshots': [{'ts_ns': x['ts_ns'], 'details': x['details']} for x in snapshots[-3:]],
                    'complete_epochs': [{'ts_ns': x['ts_ns'], 'details': x['details']} for x in epochs],
                    'last_complete_epoch_step': complete_step,
                    'replay': {replay: replay_evidence(connection, run_id, replay) for replay in TARGETS},
                    'training': rank_training(connection, run_id),
                    'recent_workloads': {role: recent_service_evidence(connection, run_id, role)
                                         for role in ('policy_infer', 'ctrl_infer', 'reward_infer') + ROLES},
                    'world_stages': {stage: recent_service_evidence(connection, run_id, 'ctrl_infer_' + stage)
                                     for stage in ('input_prepare', 'gpu_forward_and_copy')},
                    'placements': placement_evidence(connection, run_id),
                    'world_client_wait': world_wait_evidence(connection, run_id),
                    'wm_real_episodes': wm_episode_evidence(connection, run_id)}
        if start_ns and time.time_ns() >= start_ns + 1200 * 10**9:
            cutoff = start_ns + 1200 * 10**9
            evidence['at_twenty_minutes'] = {
                'cutoff_ns': cutoff,
                'resident': {replay: replay_evidence(connection, run_id, replay, at_ns=cutoff)['resident_unique']
                             for replay in TARGETS},
                'driver_snapshot': next(({'ts_ns': x['ts_ns'], 'details': x['details']} for x in snapshots
                                         if x['details'].get('at_twenty_minutes')), None),
                'wm_real_episodes': wm_episode_evidence(connection, run_id, at_ns=cutoff),
                'scope': 'historical resident unique identities reconstructed by addition/eviction timestamps'}
        first = evidence['training']['policy_train']['ranks']['0']['first_success_ns']
        evidence['first_policy_optimizer_ns'] = first
        if 'at_twenty_minutes' in evidence:
            twenty = evidence['at_twenty_minutes']
            twenty['both_replays_ready'] = all(twenty['resident'][key] >= target for key, target in TARGETS.items())
            twenty['first_policy_optimizer_completed'] = bool(first and first <= twenty['cutoff_ns'])
        if start_ns:
            end_ns = (int(dt.datetime.fromisoformat(state['finished']).timestamp() * 1e9)
                      if state.get('state') in TERMINAL and state.get('finished') else time.time_ns())
            evidence['elapsed_since_rollout_seconds'] = (end_ns - start_ns) / 1e9
            evidence['fill_seconds'] = ((evidence['replay_ready_ns'] - start_ns) / 1e9
                                        if evidence['replay_ready_ns'] else None)
            evidence['first_optimizer_seconds'] = ((first - start_ns) / 1e9 if first else None)
        evidence['checkpoint'] = checkpoint_evidence(data, state, complete_step)
        return evidence
    finally:
        connection.close()


def fmt_ns(value):
    return stamp(value / 1e9) if value else '尚未发生'


def fmt_number(value, digits=3):
    return f'{value:.{digits}f}' if isinstance(value, (int, float)) and math.isfinite(value) else '—'


def fmt_elapsed(value):
    return f'{value / 60:.2f}分钟（{value:.1f}秒）' if value is not None else '待测'


def write_report(data, state, *, persist_status=True):
    try:
        evidence = collect_report(data, state)
    except (sqlite3.Error, ValueError, OSError) as exc:
        # A timed-out read must not erase the last good measurement.
        evidence = dict(state.get('report', {}))
        evidence['database_status'] = 'read_pending'; evidence['read_error'] = str(exc)
    state['report'] = evidence
    state['heartbeat'] = stamp()
    if persist_status:
        atomic_json(data / 'runnerstatus.json', state)
    replay, training = evidence.get('replay', {}), evidence.get('training', {})
    twenty = evidence.get('at_twenty_minutes')
    verdict = ('两池已达标' if twenty.get('both_replays_ready') else '两池未达标') if twenty else '尚未到20分钟'
    first_optimizer = evidence.get('first_policy_optimizer_ns')
    lines = ['# 四卡真实 Replay 填池实验', '', f"更新时间：{stamp()}。状态：{state.get('state')}。",
             f"Run ID：`{state['run_id']}`。仅使用物理 GPU4–7。", '',
             f'**20分钟真实填池裁定：{verdict}。**',
             '**首次真实 Policy 优化器更新耗时：' + fmt_elapsed(evidence.get('first_optimizer_seconds'))
             + f'；完成时间：{fmt_ns(first_optimizer)}。**', '',
             '本轮保持真实 rollout、想象配方、全局 batch576/1024/1024、500步 warmup及5:1:1节奏。',
             '四卡各驻留训练 rank 与 Policy/Reward/World 推理副本；三训练角色串行，四路 World 推理并行。',
             '每卡六类模型角色：一个训练进程持有三模型，加三个独立推理进程；四卡共16个GPU进程、24个模型角色。',
             '达到双 Replay 阈值后运行5个完整同步 epoch、完成广播、保存权重并结束。',
             '这5个 Policy epoch含 World/Reward 各1次更新，它们的首步学习率为零；未验证完整 warmup后行为。', '',
             f"模型加载完成后的 rollout 起点：{fmt_ns(evidence.get('rollout_started_ns'))}。",
             f"双池达标：{fmt_ns(evidence.get('replay_ready_ns'))}；填池耗时：{fmt_elapsed(evidence.get('fill_seconds'))}。", '',
             '| Replay | 目标 | 最近记录的 resident 独立样本 | 新增独立样本 | 真实训练曝光 | 预取 reservation |',
             '|---|---:|---:|---:|---:|---:|']
    for role in TARGETS:
        x = replay.get(role, {})
        lines.append(f"| {role} | {TARGETS[role]} | {x.get('resident_unique', '待测')} | {x.get('added_unique', '待测')} | {x.get('actual_training_uses', '待测')} | {x.get('reservations', '待测')} |")
    lines += ['', ('20分钟时真实 resident：' + json.dumps(twenty['resident'], ensure_ascii=False)
                   + '。此前是否完成真实 Policy 更新：' + ('是' if twenty.get('first_policy_optimizer_completed') else '否'))
              if twenty else '20分钟时的双池真实样本快照尚未到时。', '',
              '| 模型 | 四rank成功更新数 | 四rank跳步数 | global_steps | 已记录rank计数一致 |',
              '|---|---|---|---|---|']
    for role in ROLES:
        x = training.get(role, {}); ranks = x.get('ranks', {})
        values = lambda key: '/'.join(str(ranks.get(str(rank), {}).get(key, '?')) for rank in range(4))
        lines.append(f"| {role} | {values('successful_updates')} | {values('skipped_updates')} | {values('engine_global_steps')} | {x.get('rank_counts_match', '待测')} |")
    episodes = evidence.get('wm_real_episodes', {})
    lines += ['', f"WM新增真实 episode：{episodes.get('added_real_episodes_unique', '待测')}条；"
              + f"训练池{episodes.get('training', {}).get('episodes', '待测')}条 / "
              + f"{episodes.get('training', {}).get('windows', '待测')}个滑窗，"
              + f"held-out池{episodes.get('held_out', {}).get('episodes', '待测')}条 / "
              + f"{episodes.get('held_out', {}).get('windows', '待测')}个滑窗。held-out不计入1024训练阈值。", '',
              '下面按每类最近最多600条service采样；批量、耗时、显存只统计成功服务，未发生的训练保留空值。', '',
              '| workload | 成功采样 | 实际batch均值/P95 | wall P50/P95（秒） | GPU interval P50/P95（秒） | 进程peak/reserved（GiB） |',
              '|---|---:|---|---|---|---|']
    for role in ('policy_infer', 'ctrl_infer', 'reward_infer') + ROLES:
        x = evidence.get('recent_workloads', {}).get(role, {})
        batch, wall, gpu = x.get('actual_batch', {}), x.get('wall_seconds', {}), x.get('gpu_seconds', {})
        pair = lambda entry, left, right: fmt_number(entry.get(left)) + '/' + fmt_number(entry.get(right))
        lines.append(f"| {role} | {x.get('successful_records', 0)} | {pair(batch, 'mean', 'p95')} | "
                     + f"{pair(wall, 'p50', 'p95')} | {pair(gpu, 'p50', 'p95')} | "
                     + fmt_number(x.get('peak_gib')) + '/' + fmt_number(x.get('reserved_gib')) + ' |')
    world = evidence.get('recent_workloads', {}).get('ctrl_infer', {})
    one_fraction = world.get('batch_one_fraction')
    if one_fraction is not None:
        lines += ['', f"四路World近期实际batch=1占{one_fraction * 100:.1f}%；实际分布："
                  + json.dumps(world.get('batch_distribution', {}), ensure_ascii=False)
                  + '。当前每路请求数和错峰到达影响组批，不能以设置的触发阈值替代实际batch。']
    lines += ['', '显存是该进程allocator的采样peak/reserved；一个训练进程持有三个模型，不能把表中数字解释为单模型显存，'
              + '也不能将同卡进程peak相加当作整卡同步峰值。GPU interval可能包含CPU enqueue空隙。', '',
              '| World阶段 | 成功采样 | wall P50/P95（秒） | GPU interval P50/P95（秒） |',
              '|---|---:|---|---|']
    for stage in ('input_prepare', 'gpu_forward_and_copy'):
        x = evidence.get('world_stages', {}).get(stage, {})
        wall, gpu = x.get('wall_seconds', {}), x.get('gpu_seconds', {})
        lines.append(f"| {stage} | {x.get('successful_records', 0)} | "
                     + fmt_number(wall.get('p50')) + '/' + fmt_number(wall.get('p95')) + ' | '
                     + fmt_number(gpu.get('p50')) + '/' + fmt_number(gpu.get('p95')) + ' |')
    lines += ['', 'World gpu_forward_and_copy包含denoise前向和D2H输出复制；input_prepare仅覆盖源代码标记范围，'
              + '不能据此把所有CPU、H2D/D2H限制排除。下面将最近最多600条World client_wait按method分开，累计等待会在多个worker之间重叠。', '',
              '| World RPC method | 成功采样 | wall均值/P95（秒） | 采样累计等待（秒） |',
              '|---|---:|---|---:|']
    for method in ('request', 'init_latent_state'):
        x = evidence.get('world_client_wait', {}).get('methods', {}).get(method, {})
        wall = x.get('wall_seconds', {})
        lines.append(f"| {method} | {x.get('successful_records', 0)} | "
                     + fmt_number(wall.get('mean')) + '/' + fmt_number(wall.get('p95')) + ' | '
                     + fmt_number(x.get('cumulative_wall_seconds')) + ' |')
    placement = evidence.get('placements', {})
    lines += ['', 'client_wait是提交到Ray直到回执反序列化的端到端wall；未运行全量join，不能把提交到actor的等待直接当成CPU/H2D开销。', '',
              f"初始化UUID放置实测：{placement.get('observed_gpu_processes', '待测')}个GPU进程 / "
              + f"{placement.get('observed_model_roles', '待测')}类模型角色；预期16进程24角色是否全部核实："
              + ('是' if placement.get('complete_expected_placement') else '尚未核实') + '。', '',
              '| 物理GPU | 训练rank（三模型） | Policy/Reward/World推理actor | GPU进程数 | 模型角色数 |',
              '|---|---|---|---:|---:|']
    for gpu in (4, 5, 6, 7):
        card = placement.get('cards', {}).get(str(gpu), {})
        actors = {item['role']: item['actor_id'] for item in card.get('processes', [])}
        inference = '/'.join(actors.get(role, '—') for role in ('policy_infer', 'reward_infer', 'ctrl_infer'))
        lines.append(f"| {gpu} | {actors.get('trainer', '—')} | {inference} | "
                     + f"{card.get('gpu_process_count', 0)} | {card.get('model_role_count', 0)} |")
    lines += ['', '上述放置取自初始化记录；当前进程是否已停止以下面的本轮清理结果为准。']
    checkpoint = evidence.get('checkpoint', {})
    lines += ['', f"最后完整同步 epoch：{evidence.get('last_complete_epoch_step', 0)}。",
              f"权重快照：{checkpoint.get('status', '待保存')}，覆盖最后完整 epoch：{checkpoint.get('covers_last_complete_epoch', '待验证')}。",
              '快照仅含权重与 buffers；缺优化器、Replay、RNG和预取数据，不能精确续训。', '',
              f"本轮进程确认停止：{state.get('training_stopped', False)}。结束原因：{state.get('reason', '运行中')}。",
              '20分钟判定同时要求 Policy576、WM1024；预取不算真实消费，四rank一次同步只计一次全局更新。',
              '与历史约80分钟对比仅作参考：本次同时复制Policy/Reward/World推理、改变共卡布局；不能把收益全部归因于World。',
              'CPU和数据搬运未排除，四倍实例不保证真实填池线性缩短到20分钟。', '',
              '只需看本 README；原始计数在 collection.sqlite，监督状态在 runnerstatus.json，异常保留在 runner.log。']
    temporary = data / f'.README.{os.getpid()}.tmp'
    temporary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    os.replace(temporary, data / 'README.md')
    return evidence


def requested_stop(data, reason):
    path = data / 'stop.flag'
    if not path.exists():
        atomic_json(path, {'reason': reason, 'at': stamp()})


def start_guard(data, state, log):
    # No training marker on this CPU guard; scoped cleanup cannot kill itself.
    env = dict(os.environ); env.pop(MARKER, None)
    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--mode', 'guard',
                                '--data', str(data), '--run-id', state['run_id']],
                               env=env, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True, close_fds=True)
    state['guard'] = proc_identity(process.pid)


def validate_completion(state):
    evidence = state.get('report', {})
    errors = []
    if evidence.get('last_complete_epoch_step') != 5:
        errors.append('Expected exactly five complete synchronized epochs')
    for role, expected in zip(ROLES, (5, 1, 1)):
        x = evidence.get('training', {}).get(role, {})
        if not x.get('rank_counts_match') or not x.get('global_batch_valid'):
            errors.append(role + ': rank disagreement or global batch mismatch')
        if x.get('limit_reached'):
            errors.append(role + ': bounded event query unexpectedly reached limit')
        for rank in range(4):
            y = x.get('ranks', {}).get(str(rank), {})
            if y.get('successful_updates') != expected or y.get('skipped_updates') != 0:
                errors.append(role + f': rank {rank} counts differ from recipe')
    checkpoint = evidence.get('checkpoint', {})
    manifest = checkpoint.get('manifest', {})
    if (checkpoint.get('status') != 'current_run' or not checkpoint.get('covers_last_complete_epoch')
            or manifest.get('steps') != {'policy': 5, 'ctrl': 1, 'reward': 1}):
        errors.append('Current-run final checkpoint does not cover all three expected step counts')
    if not state.get('training_stopped'):
        errors.append('Experiment-tagged processes still alive')
    return errors


def run(data, root):
    if os.name != 'posix' or not Path('/proc').is_dir():
        raise RuntimeError('Run mode requires the Linux training server')
    data.mkdir(parents=True, exist_ok=True)
    with exclusive(data / '.runner.lock'):
        prior = load_status(data)
        if prior:
            if prior.get('state') not in TERMINAL or not prior.get('training_stopped'):
                raise RuntimeError('This data directory already has an unfinished attempt; do not launch a duplicate')
            # Previous evidence remains in SQLite and the append-only log.
            history = (prior.get('previous_attempts', []) + [{k: prior.get(k) for k in
                ('run_id', 'state', 'started', 'finished', 'reason', 'return_code')}])[-10:]
        else:
            history = []
        started = time.time()
        self_identity = proc_identity(os.getpid())
        run_id = 'real_fill4wm_' + dt.datetime.now(TZ).strftime('%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
        launcher = root / 'run_oft_ctrl_world_train_wm.sh'
        command = ['bash', str(launcher), '--cuda-visible-devices', '4,5,6,7',
                   '--gpu-layout', LAYOUT, '--num-trainer-gpus', '4',
                   '--num-inference-actors', '4', '--num-reward-inference-actors', '4',
                   '--num-ctrl-inference-actors', '4', '--accumulation-steps', '18',
                   '--ctrl-accumulation-steps', '64', '--reward-accumulation-steps', '8']
        state = {'state': 'starting', 'run_id': run_id, 'started': stamp(started), 'started_unix': started,
                 'hard_deadline_unix': started + 10800, 'uid': os.getuid(),
                 'minimum_child_birth_ticks': int(float(Path('/proc/uptime').read_text().split()[0]) * os.sysconf('SC_CLK_TCK')),
                 'supervisor': self_identity, 'command': command, 'owned': [], 'training_stopped': False,
                 'plan': {'layout': LAYOUT, 'physical_gpus': [4, 5, 6, 7], 'fill_timeout_seconds': 7200,
                          'total_timeout_seconds': 10800, 'stop_after_complete_epochs': 5,
                          'global_batches': {'policy': 576, 'ctrl': 1024, 'reward': 1024},
                          'replay_targets': TARGETS, 'automatic_restarts': False},
                 'previous_attempts': history}
        source_files = [launcher] + [root / 'rl/ctrl_world' / name for name in
            ('ds_wm_discrete_ctrl_train_wm.py', 'training_topology.py', 'capture_hooks.py',
             'capture_training.py', 'telemetry.py', 'capture_telemetry.py',
             'parallel_checkpoint.py', 'shared_inference.py')]
        state['source_sha256'] = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in source_files}
        state['runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        stop_path = data / 'stop.flag'
        if stop_path.exists():
            stop_path.unlink()  # Only this explicitly named single file; no tree deletion.
        env = {key: value for key, value in os.environ.items() if not key.startswith('CUDA_MPS_')}
        env.update({MARKER: run_id, 'ACCERL_TELEMETRY_DB': str(data / 'collection.sqlite'),
                    'ACCERL_CAPTURE_ONLY': '1', 'ACCERL_CPU_THREADS': '1',
                    'CUDA_VISIBLE_DEVICES': '4,5,6,7', 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
                    'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                    'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
                    # Ray appends a session name and AF_UNIX socket path (107-byte limit).
                    'ACCERL_RAY_TMP': f'/dev/shm/arf_{os.getuid()}_{run_id[-8:]}',
                    'ACCERL_RAY_SPILL': f'/mnt/data/lcx1/dong/ray_profile_spill/{run_id}',
                    'ACCERL_STOP_FILE': str(stop_path), 'ACCERL_FINAL_CHECKPOINT': str(data / 'checkpoints'),
                    'ACCERL_FILL_TIMEOUT_SEC': '7200', 'ACCERL_TRIM_IDLE_CACHE': '1',
                    'ACCERL_TRIM_MICRO_CACHE': '1', 'ACCERL_MEASURE_STOP_AFTER_EPOCHS': '5',
                    'PYTHONUNBUFFERED': '1', 'ACCE_CAPTURE_TRACE_EVERY': '10'})
        state['experiment_environment'] = {key: value for key, value in env.items()
                                           if key.startswith(('ACCERL_', 'ACCE_CAPTURE_'))
                                           or key in ('PYTHONUNBUFFERED', 'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER',
                                                      'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                                      'TF_FORCE_GPU_ALLOW_GROWTH')}
        state['mps_environment_removed'] = sorted(key for key in os.environ if key.startswith('CUDA_MPS_'))
        atomic_json(data / 'runnerstatus.json', state)
        process = None
        with (data / 'runner.log').open('a', encoding='utf-8', buffering=1) as log:
            log.write(f'\n[{stamp()}] START {run_id}; no automatic next case.\n')
            try:
                # Guard exists before GPU work is launched, so early runner failure cannot orphan it.
                start_guard(data, state, log)
                atomic_json(data / 'runnerstatus.json', state)
                process = subprocess.Popen(command, cwd=root, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
                state['child'] = proc_identity(process.pid); state['state'] = 'running'
                stop_time = None; next_report = 0
                while process.poll() is None:
                    now = time.monotonic()
                    remember_owned(state)
                    if now >= next_report:
                        evidence = write_report(data, state); next_report = now + 15
                        rollout = evidence.get('rollout_started_ns')
                        if rollout and not evidence.get('replay_ready_ns') and time.time_ns() > rollout + 7200 * 10**9:
                            state['reason'] = 'fill_timeout_120_minutes'; requested_stop(data, state['reason'])
                        if time.time() >= state['hard_deadline_unix']:
                            state['reason'] = 'overall_timeout_180_minutes'; requested_stop(data, state['reason'])
                        if not same_process(state.get('guard'), proc_identity(state.get('guard', {}).get('pid', -1))):
                            state['reason'] = 'independent_guard_missing'; requested_stop(data, state['reason'])
                    if stop_path.exists():
                        stop_time = stop_time or now
                        if now - stop_time > 45 or time.time() >= state['hard_deadline_unix']:
                            cleanup(state, data); break
                    time.sleep(3)
                state['return_code'] = process.poll()
                if process.poll() is None:
                    process.wait(timeout=10); state['return_code'] = process.returncode
                stopped = cleanup(state, data)
                # Give outstanding SQLite writer exits a brief chance to finish; no training restarts.
                time.sleep(2)
                write_report(data, state)
                errors = validate_completion(state)
                state['validation_errors'] = errors
                if not stopped:
                    state['state'] = 'cleanup_requires_attention'
                elif state.get('reason', '').endswith('minutes'):
                    state['state'] = 'timed_out'
                elif state['return_code'] == 0 and not errors:
                    state['state'] = 'completed'; state['reason'] = 'five_complete_epochs_saved'
                else:
                    state['state'] = 'failed'; state.setdefault('reason', 'training_or_final_validation_failed')
            except BaseException as exc:
                state['state'] = 'failed'; state['reason'] = repr(exc)
                requested_stop(data, 'supervisor_error')
                if process is not None:
                    cleanup(state, data)
                else:
                    state['training_stopped'] = True
                log.write(f'[{stamp()}] SUPERVISOR ERROR {exc!r}\n')
            finally:
                state['finished'] = stamp(); write_report(data, state)
                log.write(f'[{stamp()}] FINISH {state["state"]}: {state.get("reason")}\n')
        return 0 if state['state'] == 'completed' else 1


def guard(data, run_id):
    """Independent CPU-only deadline watchdog; never launches GPU work."""
    while True:
        state = load_status(data)
        if state.get('run_id') != run_id:
            return 0
        if state.get('state') in TERMINAL and state.get('training_stopped'):
            return 0
        parent_alive = same_process(state.get('supervisor'), proc_identity(state.get('supervisor', {}).get('pid', -1)))
        expired = time.time() >= state['hard_deadline_unix']
        if not parent_alive or expired:
            reason = 'independent_guard_total_timeout' if expired else 'independent_guard_supervisor_exited'
            requested_stop(data, reason)
            if not expired:
                time.sleep(20)  # Only a bounded chance for current code to save at its next safe point.
            state = load_status(data)
            if state.get('run_id') != run_id:
                return 0
            cleanup(state, data)
            # The main runner owns reports if it still lives; otherwise this guard finalizes them.
            if not same_process(state.get('supervisor'), proc_identity(state.get('supervisor', {}).get('pid', -1))):
                state['state'] = 'timed_out' if expired else 'failed'; state['reason'] = reason
                if not state['training_stopped']:
                    state['state'] = 'cleanup_requires_attention'
                state['finished'] = stamp(); write_report(data, state)
            return 0 if state['training_stopped'] else 1
        time.sleep(5)


def self_test():
    """CPU-only synthetic DB tests for accounting, times and signal eligibility."""
    import tempfile
    import importlib.util
    telemetry_path = Path(__file__).resolve().parents[1] / 'scheduler_ddp' / 'telemetry.py'
    if not telemetry_path.is_file():
        telemetry_path = ROOT / 'rl/ctrl_world/telemetry.py'
    spec = importlib.util.spec_from_file_location('fill_test_telemetry', telemetry_path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix='real-fill-selftest-') as raw:
        data = Path(raw); run_id = 'test'; start = time.time_ns() - 1300 * 10**9
        writer = module.Telemetry('driver', db_path=str(data / 'collection.sqlite'), run_id=run_id)
        writer.event('rollout_started', ts_ns=start)
        writer.event('fill_snapshot', ts_ns=start + 1201 * 10**9,
                     details={'at_twenty_minutes': True, 'policy_sizes': [576], 'wm_sizes': [1024]})
        writer.event('replay_ready', ts_ns=start + 1199 * 10**9)
        for replay, size in TARGETS.items():
            for index in range(size):
                writer._queue('INSERT INTO replay_samples(run_id,replay,sample_id,added_ns,resident,reserved_count) VALUES(?,?,?,?,?,?)',
                              (run_id, replay, f'{replay}:{index}', start + (900 + index // 10) * 10**9, 1, 100))
        for epoch in range(1, 6):
            writer.event('synchronized_epoch_completed', details={'global_step': epoch, 'rank_steps': [epoch] * 4})
            for role in ROLES:
                if role != 'policy_train' and epoch != 5:
                    continue
                step = epoch if role == 'policy_train' else 1
                for rank in range(4):
                    writer.actor_id = str(rank)
                    writer.event('optimizer_update' if rank == 0 else 'optimizer_update_replica',
                                 workload=role, actual_batch=576 if role == 'policy_train' else 1024,
                                 successful_optimizer_steps=1, skipped_steps=0, engine_global_steps=step,
                                 data_parallel_rank=rank, status='ok')
        writer._queue("UPDATE replay_samples SET episode_id='real-train-episode' WHERE replay='wm' AND run_id=?", (run_id,))
        writer._queue('INSERT INTO replay_samples(run_id,replay,sample_id,episode_id,added_ns,resident) VALUES(?,?,?,?,?,?)',
                      (run_id, 'wm_eval', 'eval:0', 'real-held-out-episode', start + 1000 * 10**9, 1))
        for rank in range(4):
            for role in ('trainer', 'policy_infer', 'reward_infer', 'ctrl_infer'):
                writer.actor = role; writer.actor_id = str(rank)
                gpu = 4 + rank
                writer.event('placement', details={'role': role, 'physical_gpu_uuid': f'test-uuid-{gpu}',
                             'physical_gpus': [{'physical_index': gpu, 'uuid': f'GPU-test-uuid-{gpu}'}],
                             'cpu_threads': 1})
        writer.actor = 'ctrl_infer'; writer.actor_id = '0'
        for index in range(601):
            batch = 99 if index == 0 else (3 if index % 10 == 0 else 1)
            writer.event('service', workload='ctrl_infer', actual_batch=batch, duration_ns=batch * 10**9,
                         gpu_ms=batch * 900, peak_bytes=(100 if index == 0 else 5) * 2**30,
                         reserved_bytes=6 * 2**30, status='ok', details={'peak_scope': 'span'})
        writer.event('service', workload='ctrl_infer_input_prepare', actual_batch=1, duration_ns=200000, status='ok')
        writer.event('service', workload='ctrl_infer_gpu_forward_and_copy', actual_batch=1,
                     duration_ns=2500000000, gpu_ms=2400, status='ok')
        writer.actor = 'rollout'; writer.actor_id = '0'
        for method, seconds in [('request', 2), ('request', 4), ('init_latent_state', .01), ('init_latent_state', .02)]:
            writer.event('client_wait', workload='ctrl_infer', duration_ns=int(seconds * 1e9),
                         status='ok', details={'method': method})
        writer.actor = 'driver'; writer.actor_id = ''
        writer.flush()
        started = time.time() - 1400
        state = {'run_id': run_id, 'started_unix': started, 'state': 'running', 'training_stopped': True}
        (data / 'checkpoints').mkdir(); (data / 'checkpoints/latest_weights.pt').write_bytes(b'test')
        manifest = {'metadata': {'run_id': run_id}, 'world_size': 4, 'steps': {'policy': 5, 'ctrl': 1, 'reward': 1},
                    'saved_at_unix': time.time(), 'weights_file': 'latest_weights.pt'}
        atomic_json(data / 'checkpoints/manifest.json', manifest)
        x = write_report(data, state)
        assert x['at_twenty_minutes']['resident'] == TARGETS
        assert x['replay']['wm']['actual_training_uses'] == 0
        assert x['replay']['wm']['reservations'] == 102400
        assert abs(x['fill_seconds'] - 1199) < .01
        assert x['recent_workloads']['ctrl_infer']['sample_records'] == 600
        assert x['recent_workloads']['ctrl_infer']['actual_batch']['max'] == 3
        assert x['recent_workloads']['ctrl_infer']['peak_gib'] == 5
        assert x['world_stages']['gpu_forward_and_copy']['gpu_seconds']['p50'] == 2.4
        assert x['world_client_wait']['methods']['request']['wall_seconds']['mean'] == 3
        assert x['world_client_wait']['methods']['init_latent_state']['sample_records'] == 2
        assert x['placements']['complete_expected_placement']
        assert x['placements']['observed_gpu_processes'] == 16
        assert x['placements']['observed_model_roles'] == 24
        assert x['wm_real_episodes']['added_real_episodes_unique'] == 2
        assert x['wm_real_episodes']['training']['windows'] == 1024
        assert x['wm_real_episodes']['held_out']['windows'] == 1
        assert x['at_twenty_minutes']['both_replays_ready']
        assert not x['at_twenty_minutes']['first_policy_optimizer_completed']
        assert not validate_completion(state)
        # A late addition and an early eviction must both be excluded from the 20min count.
        writer._queue('UPDATE replay_samples SET evicted_ns=?,resident=0 WHERE run_id=? AND replay=? AND sample_id=?',
                      (start + 1100 * 10**9, run_id, 'wm', 'wm:0'))
        writer._queue('INSERT INTO replay_samples(run_id,replay,sample_id,added_ns,resident) VALUES(?,?,?,?,?)',
                      (run_id, 'wm', 'wm:late', start + 1250 * 10**9, 1))
        writer.flush(); x = collect_report(data, state)
        assert x['at_twenty_minutes']['resident']['wm'] == 1023
        assert x['replay']['wm']['resident_unique'] == 1024
        current = {'pid': 123, 'uid': 1004, 'birth_ticks': 99, 'state': 'S', 'run_id': run_id, 'command': '', 'comm': ''}
        assert same_process(current, dict(current), run_id=run_id)
        assert not same_process(current, dict(current, birth_ticks=100), run_id=run_id)
        assert not same_process(current, dict(current, uid=1005), run_id=run_id)
        assert not same_process(current, dict(current, run_id='other'), run_id=run_id)
        assert not same_process(current, dict(current, state='Z'), run_id=run_id)
        assert protected(dict(current, command='/usr/bin/nvidia-cuda-mps-server'))
        from unittest.mock import patch
        own_state = {'run_id': run_id, 'uid': 1004, 'minimum_child_birth_ticks': 98}
        this_module = sys.modules[__name__]
        with patch.object(os, 'getuid', return_value=1004, create=True), patch.object(os, 'kill') as kill:
            for observed in (dict(current, birth_ticks=100), dict(current, uid=1005),
                             dict(current, run_id='another-run'), dict(current, state='Z'),
                             dict(current, command='nvidia-cuda-mps-server'), None):
                with patch.object(this_module, 'proc_identity', return_value=observed):
                    assert not checked_signal(current, own_state, signal.SIGTERM)
            assert kill.call_count == 0
            with patch.object(this_module, 'proc_identity', return_value=dict(current)):
                assert checked_signal(current, own_state, signal.SIGTERM)
            kill.assert_called_once_with(123, signal.SIGTERM)
            # An exact identity predating this experiment is also ineligible.
            kill.reset_mock()
            with patch.object(this_module, 'proc_identity', return_value=dict(current)):
                assert not checked_signal(current, dict(own_state, minimum_child_birth_ticks=100), signal.SIGTERM)
            assert kill.call_count == 0
        writer.close()
    print('CPU_SELF_TEST_OK: shared Replay unique/reservation separation, historical20min residency, rank counts, checkpoint identity, PID birth/UID/run_id/MPS guard')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('run', 'report', 'guard', 'self-test'), default='report')
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
    # A manual refresh must not overwrite the supervisor's status with a stale
    # read while it is concurrently completing an epoch or process cleanup.
    write_report(args.data, state, persist_status=False)
    print(json.dumps({'run_id': state['run_id'], 'state': state['state'],
                      'report': str(args.data / 'README.md')}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
