#!/usr/bin/env python3
"""CPU supervisor for serial split-owner training through the user-set deadline.

The reviewed real-fill helper supplies PID/birth/UID/run-id scoped cleanup.
This runner never changes the MPS service. The formal run records 85% milestones
and continues until 2026-10-19 09:00 Asia/Shanghai. The launcher owns training.
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
import re
import sqlite3
import struct
import subprocess
import sys
import time
import uuid

os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Supervisor, report, sampler and guard are CPU-only.

ROOT = Path('/mnt/data/lcx1/dong/AcceRL-scheduler')
PREFLIGHT_DATA = Path('/mnt/data/lcx1/dong/need_data/serial_split_mps_preflight_nsysfix_20261016')
TARGET_DATA = Path('/mnt/data/lcx1/dong/need_data/serial_split_mps_deadline_20261016')
DATA = TARGET_DATA
PREFLIGHT_EPOCHS = 15
DEADLINE_ISO = '2026-10-19T09:00:00+08:00'
DEADLINE_UNIX = int(dt.datetime.fromisoformat(DEADLINE_ISO).timestamp())
DEADLINE_NS = DEADLINE_UNIX * 10**9
DEADLINE_REASON = 'user_deadline_reached'
PROFILER_FINALIZATION_GRACE_SECONDS = 1500  # CPU export + bounded analysis + quiet period
HELPER = Path('/mnt/data/lcx1/dong/need_data/archive/real_fill_experiment_20261014.py')
SAMPLER = Path('/mnt/data/lcx1/dong/need_data/archive/mps_system_profile_20261015.py')
NSYS = Path('/mnt/data/lcx1/dong/need_data/tools/nsys_2026_5_1_user/unpacked/opt/nvidia/nsight-systems-cli/2026.5.1/bin/nsys')
ANALYZER = Path('/mnt/data/lcx1/dong/need_data/archive/nsight_overlap_bounded_20261016.py')
MPS_SERVER_PID, MPS_SERVER_BIRTH, EXPECTED_UID = 1461366, 301739562, 1004
MPS_PIPE = '/dev/shm/mps_lcx1/pipe'
MARKER = 'ACCERL_RUN_ID'
METRIC = 'Eval/_Global/Average_Return'
LAYOUT = 'four_rank_split_train'
TZ = dt.timezone(dt.timedelta(hours=8))
TERMINAL = {'preflight_passed', 'completed_at_deadline', 'failed', 'timed_out', 'cleanup_requires_attention', 'stopped_by_user'}
SUCCESS_STATES = {'preflight_passed', 'completed_at_deadline'}
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


def training_data(data):
    """Outer state is separate from the launcher's freshly created output dir."""
    return Path(data) / 'training'


def request_stop(data, reason):
    directory = training_data(data)
    directory.mkdir(parents=True, exist_ok=True)
    existing = directory / 'stop.flag'
    if reason == DEADLINE_REASON and existing.is_file():
        try:
            prior = json.loads(existing.read_text(encoding='utf-8')).get('reason')
            if prior and prior != DEADLINE_REASON:
                return  # A deadline cannot replace an earlier error/manual stop.
        except (OSError, ValueError):
            pass
    safe.requested_stop(directory, reason)


def user_stop_reason(reason):
    if reason == DEADLINE_REASON:
        return False
    return bool(reason and (str(reason).startswith(('user_', 'manual_', 'requested_by_user'))
                            or reason in ('user_stop', 'user_requested_stop', '用户停止')))


def stop_grace_seconds(reason):
    return 900 if reason == DEADLINE_REASON else 120


def profiler_process(identity):
    """Preserve the profiler/export tools until they finish their trace files."""
    if not identity:
        return False
    words = str(identity.get('command', '')).split()[:2]
    # A bundled tool can have the bare basename "exporter". Preserve it only
    # when its resolved executable is within the selected Nsight installation;
    # unrelated same-named programs do not receive this exemption.
    executable = NSYS.resolve()
    install = next((parent for parent in executable.parents if parent.name.startswith('nsight-systems-')),
                   executable.parent.parent / 'nsight-systems-2025.3.2')
    for word in words:
        candidate = Path(word)
        if candidate.name.lower() == 'exporter':
            try:
                candidate.resolve().relative_to(install)
                return True
            except ValueError:
                pass
    names = [str(identity.get('comm', '')).lower()]
    names += [Path(part).name.lower() for part in words]
    return any(any(token in name for token in ('nsys', 'nsight', 'qdstrm')) for name in names)


def training_driver_liveness(state, root, active, now_ns=None):
    """Discover only this run's exact Python entrypoint, then pin PID birth."""
    now_ns = time.time_ns() if now_ns is None else now_ns
    expected = state.get('training_driver')
    if expected:
        if same_process(expected, proc_identity(expected['pid']), run_id=state['run_id']):
            state.pop('training_driver_disappeared_ns', None)
            return None
        missing_since = state.setdefault('training_driver_disappeared_ns', now_ns)
        return 'python_training_driver_exited_without_completion' if now_ns - missing_since > 10 * 10**9 else None
    entry = (root / 'rl/ctrl_world/ds_wm_discrete_ctrl_train_wm.py').resolve()
    found = []
    for identity in active:
        if (identity.get('uid') != state['uid'] or identity.get('run_id') != state['run_id']
                or identity.get('birth_ticks', -1) < state['minimum_child_birth_ticks']):
            continue
        words = identity.get('command', '').split()
        if len(words) < 2 or not re.fullmatch(r'python(?:\d+(?:\.\d+)*)?', Path(words[0]).name):
            continue
        script = Path(words[1])
        if not script.is_absolute():
            script = root / script
        if script.resolve() == entry:
            found.append(identity)
    if len(found) > 1:
        return 'multiple_owned_python_training_drivers'
    if found:
        state['training_driver'] = found[0]
        state['training_driver_discovered_ns'] = now_ns
        return None
    return 'python_training_driver_not_discovered_120_seconds' if now_ns - state['started_ns'] > 120 * 10**9 else None


def stop_training_keep_profiler(state, data, grace=15):
    """Stop only this run's model/driver/Ray processes; leave CPU trace conversion."""
    with safe.exclusive(data / '.cleanup.lock', blocking=True):
        active = lambda: [item for item in remember_owned(state) if not profiler_process(item)]
        signals = []
        for identity in active():
            if safe.checked_signal(identity, state, safe.signal.SIGTERM):
                signals.append({'pid': identity['pid'], 'birth_ticks': identity['birth_ticks'], 'signal': 'TERM'})
        deadline = time.monotonic() + grace
        while active() and time.monotonic() < deadline:
            time.sleep(.5)
        for identity in active():
            if safe.checked_signal(identity, state, safe.signal.SIGKILL):
                signals.append({'pid': identity['pid'], 'birth_ticks': identity['birth_ticks'], 'signal': 'KILL'})
        state['training_only_cleanup'] = {'at': stamp(), 'signals': signals,
                                         'survivors': active(), 'profiler_preserved': True}
        return not state['training_only_cleanup']['survivors']


def finalize_profiler(data, state, process, log, pending):
    """Bound CPU trace conversion after stopping owned Ray orphan processes."""
    stop_training_keep_profiler(state, data)
    state['state'] = 'finalizing'
    deadline, next_report, quiet_since = time.monotonic() + PROFILER_FINALIZATION_GRACE_SECONDS, 0, None
    while True:
        update_nsight_exports(data, state, pending, log)
        live = [item for item in remember_owned(state) if profiler_process(item)]
        if process.poll() is not None and not live and not pending:
            quiet_since = quiet_since or time.monotonic()
            exports = state.get('nsight_exports', {})
            attempted = all(record.get('analysis') not in ('not_yet_analyzed', 'running')
                            for record in exports.values())
            if attempted and time.monotonic() - quiet_since >= 15:
                state['profiler_finalized'] = True
                return
        else:
            quiet_since = None
        if time.monotonic() > deadline:
            state['profiler_finalized'] = False
            state['profiler_finalization_error'] = 'CPU export and bounded analysis exceeded 1500 seconds'
            return
        if time.monotonic() >= next_report:
            write_report(data, state); next_report = time.monotonic() + 30
        time.sleep(3)


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
    if os.getuid() != EXPECTED_UID:
        raise RuntimeError('This approved run is restricted to UID 1004')
    if not NSYS.is_file() or not SAMPLER.is_file() or not ANALYZER.is_file():
        raise RuntimeError('Required Nsight executable, CPU analyzer or reviewed CPU sampler is missing')
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
    server = services.get(MPS_SERVER_PID)
    if (len(services) != 1 or not server or server['birth_ticks'] != MPS_SERVER_BIRTH
            or not safe.protected(server)):
        raise RuntimeError('Protected MPS server PID/birth identity changed; nothing was stopped')
    if any(mps_clients(pid) for pid in services):
        raise RuntimeError('The existing MPS service already has clients; no client was stopped')
    if any(item['utilization_percent'] > 0 or item['used_mib'] > 1024 for item in cards):
        raise RuntimeError('Selected GPUs are not idle according to the bounded preflight snapshot')
    return {'cards': cards, 'protected_mps_services': list(services.values()),
            'mps_pipe': MPS_PIPE, 'existing_clients': [],
            'nsight_version': command_output([str(NSYS), '--version'])}


def placement_evidence(connection, run_id):
    """Initialization identities for twelve independent trainers + twelve infer actors."""
    roles = {'policy_trainer', 'ctrl_trainer', 'reward_trainer',
             'policy_infer', 'ctrl_infer', 'reward_infer'}
    keep = {}
    for row in safe.query_events(connection, run_id, 'placement', limit=2000):
        details = row['details']; role = details.get('role', row['actor'])
        if role not in roles:
            continue
        mapping = details.get('physical_gpus', [])
        physical = mapping[0].get('physical_index') if len(mapping) == 1 else None
        actual = str(details.get('physical_gpu_uuid', '')).removeprefix('GPU-').lower()
        mapped = str(mapping[0].get('uuid', '')).removeprefix('GPU-').lower() if len(mapping) == 1 else ''
        keep[role, row['actor_id']] = {
            'role': role, 'actor_id': row['actor_id'], 'pid': row['pid'],
            'physical_gpu': physical, 'uuid': details.get('physical_gpu_uuid'),
            'uuid_confirmed': bool(actual and actual == mapped),
            'cpu_threads': details.get('cpu_threads'), 'model_roles': 1}
    actors = sorted(keep.values(), key=lambda item: (str(item['physical_gpu']), item['role']))
    cards = {str(gpu): {'processes': [item for item in actors if item['physical_gpu'] == gpu]}
             for gpu in (4, 5, 6, 7)}
    for card in cards.values():
        card['gpu_process_count'] = len(card['processes'])
        card['model_role_count'] = card['gpu_process_count']
    complete = (len(actors) == 24 and len({item['pid'] for item in actors}) == 24
                and all(item['uuid_confirmed'] and item['cpu_threads'] == 1 for item in actors)
                and all(len(card['processes']) == 6 and {item['role'] for item in card['processes']} == roles
                        for card in cards.values()))
    return {'actors': actors, 'cards': cards, 'observed_gpu_processes': len(actors),
            'observed_model_roles': len(actors), 'complete_expected_placement': complete,
            'scope': 'recorded actor UUID placement; current MPS/aliveness verified separately'}


def replay_evidence(connection, run_id, training):
    rows = connection.execute('''SELECT replay, COUNT(added_ns) added,
        SUM(CASE WHEN resident=1 THEN 1 ELSE 0 END) resident,
        SUM(policy_uses) policy_uses, SUM(ctrl_uses) ctrl_uses, SUM(reward_uses) reward_uses,
        SUM(CASE WHEN evicted_ns IS NOT NULL AND eviction_reason='capacity'
                 AND (first_used_ns IS NULL OR first_used_ns>evicted_ns) THEN 1 ELSE 0 END) unused_at_eviction
        FROM replay_samples WHERE run_id=? GROUP BY replay''', (run_id,)).fetchall()
    result = {row['replay']: dict(row) for row in rows}
    for model, replay in (('policy', 'policy'), ('ctrl', 'wm'), ('reward', 'wm')):
        updates = training.get(model, {}).get('global_updates') or 0
        batch = MODELS[model][1]
        actual = result.get(replay, {}).get(model + '_uses', 0) or 0
        result.setdefault(replay, {})[model + '_balance'] = {
            'actual_sample_exposures': actual, 'completed_updates_times_global_batch': updates * batch,
            'difference_including_inflight_and_sqlite_flush': actual - updates * batch,
            'note': 'prefetch excluded; running differences need in-flight microsteps and async flush reconciliation'}
    return result


def live_mps_evidence(state, placements):
    server = proc_identity(MPS_SERVER_PID)
    valid = bool(server and server['uid'] == EXPECTED_UID and server['birth_ticks'] == MPS_SERVER_BIRTH
                 and safe.protected(server) and server['state'] not in ('Z', 'X'))
    result = {'protected_server_identity_valid': valid, 'expected_clients': 24}
    if valid:
        try:
            clients = sorted(mps_clients(MPS_SERVER_PID))
            expected = sorted(item['pid'] for item in placements.get('actors', []))
            result.update(client_pids=clients, client_count=len(clients),
                          exact_recorded_24_clients=bool(len(expected) == 24 and clients == expected))
        except Exception as error:
            result['read_error'] = repr(error)
    return result


def topology_evidence(record):
    states = (record or {}).get('details', {}).get('states', [])
    errors, ports = [], {}
    expected = {(kind, rank) for kind in MODELS for rank in range(4)}
    if (len(states) != 12 or len({item.get('pid') for item in states}) != 12
            or {(item.get('training_role'), item.get('rank')) for item in states} != expected):
        errors.append('Expected twelve distinct model/rank owners')
    for item in states:
        kind, rank = item.get('training_role'), item.get('rank')
        if (kind not in MODELS or rank not in range(4) or item.get('world_size') != 4
                or item.get('distributed_world_size') != 4 or set(item.get('engines', {})) != {kind}
                or str(item.get('gpu', {}).get('physical_gpu')) != str(rank + 4)):
            errors.append('Engine ownership, world size or physical GPU differs')
        ports.setdefault(kind, set()).add(item.get('group_port'))
    if (set(ports) != set(MODELS) or any(len(values) != 1 or None in values for values in ports.values())
            or len(set().union(*ports.values())) != 3):
        errors.append('Three independent four-rank communicator rendezvous were not confirmed')
    return {'valid': not errors, 'errors': errors, 'states': states,
            'group_ports': {kind: sorted(values, key=str) for kind, values in ports.items()}}


def serial_evidence(connection, run_id, *, limit=4096):
    """Bounded CPU audit of stage barriers, memory and shared WM consumers."""
    rows = connection.execute('''SELECT ts_ns,kind,actor_id,pid,details_json FROM events
        WHERE run_id=? AND workload IS NULL AND kind IN ('split_serial_phase_begin','split_serial_phase_end')
        ORDER BY id DESC LIMIT ?''', (run_id, limit)).fetchall()
    groups, memory, malformed = {}, {}, []
    for row in rows:
        details = json.loads(row['details_json'] or '{}')
        step, model, rank = details.get('policy_step'), details.get('model'), details.get('rank')
        if not isinstance(step, int) or model not in MODELS or rank not in range(4):
            malformed.append({'ts_ns': row['ts_ns'], 'details': details}); continue
        group = groups.setdefault(step, {}).setdefault(model, {'begin': {}, 'end': {}})
        edge = 'begin' if row['kind'].endswith('_begin') else 'end'
        group[edge].setdefault(rank, {'ts_ns': row['ts_ns'], **details})
        summary = memory.setdefault(model, {})
        for key in ('allocated_bytes', 'reserved_bytes', 'peak_allocated_bytes', 'peak_reserved_bytes'):
            value = details.get(key)
            if isinstance(value, int): summary[key + '_max'] = max(summary.get(key + '_max', 0), value)
        if isinstance(details.get('free_bytes'), int):
            summary['minimum_card_free_bytes'] = min(summary.get('minimum_card_free_bytes', details['free_bytes']), details['free_bytes'])
    complete, problems = [], list(malformed)
    release_failures = []
    for step, models in sorted(groups.items()):
        if set(models) != set(MODELS) or any(set(edge) != set(range(4)) for group in models.values() for edge in group.values()):
            continue  # Current in-flight epoch is not a failed barrier.
        complete.append(step)
        for model, group in models.items():
            for rank, end in group['end'].items():
                if (end.get('gpu_batch_refs_released') is not True or end.get('cpu_prefetch_only') is not True
                        or end.get('completion') != 'cuda_sync_and_empty_cache'):
                    release_failures.append({'step': step, 'model': model, 'rank': rank, 'details': end})
        for before, after in (('policy', 'ctrl'), ('ctrl', 'reward')):
            if max(item['ts_ns'] for item in models[before]['end'].values()) > min(item['ts_ns'] for item in models[after]['begin'].values()):
                problems.append({'step': step, 'problem': before + '_to_' + after + '_stage_barrier_overlapped'})
    cohort_rows = connection.execute('''SELECT ts_ns,details_json FROM events WHERE run_id=?
        AND workload IS NULL AND kind='split_wm_cohort_validated' ORDER BY id DESC LIMIT 64''', (run_id,)).fetchall()
    cohorts = []
    for row in cohort_rows:
        detail = json.loads(row['details_json'] or '{}'); consumers = detail.get('consumers', [])
        by_role_rank = {(item.get('model'), item.get('rank')): item for item in consumers}
        expected = {(model, rank) for model in ('ctrl', 'reward') for rank in range(4)}
        issues = []
        if detail.get('expected_consumers') != 8 or len(consumers) != 8 or set(by_role_rank) != expected:
            issues.append('Expected eight distinct World/Reward rank consumers')
        else:
            if len({item.get('reservation_id') for item in consumers}) != 1 or not consumers[0].get('reservation_id'):
                issues.append('Consumers do not share one nonempty Replay reservation')
            for rank in range(4):
                world, reward = by_role_rank['ctrl', rank], by_role_rank['reward', rank]
                for key in ('reservation_id', 'shard_start', 'shard_end', 'sample_ids_sha256'):
                    if world.get(key) != reward.get(key): issues.append(f'rank{rank}: World/Reward {key} differs')
                if world.get('shard_start') != rank * 256 or world.get('shard_end') != (rank + 1) * 256:
                    issues.append(f'rank{rank}: 1024 global WM samples are not split into 256-sample rank shards')
                if not world.get('sample_ids_sha256'): issues.append(f'rank{rank}: sample identity hash absent')
        for key in ('actual_sample_ids_verified', 'rank_shards_disjoint', 'world_reward_ordered_ids_identical'):
            if detail.get(key) is not True:
                issues.append('Runtime actual-ID validation absent or false: ' + key)
        cohorts.append({'policy_step': detail.get('policy_step'), 'valid': not issues,
                        'errors': issues, 'consumers': consumers})
    identity_rows = connection.execute('''SELECT details_json FROM events WHERE run_id=?
        AND workload IS NULL AND kind='split_sample_identity_validated' ORDER BY id DESC LIMIT 1536''', (run_id,)).fetchall()
    verified, identity_errors = set(), []
    for row in identity_rows:
        detail = json.loads(row['details_json'] or '{}')
        step, model = detail.get('policy_step'), detail.get('model')
        if not isinstance(step, int) or model not in MODELS:
            identity_errors.append({'problem': 'Malformed sample identity event', 'details': detail}); continue
        required = model == 'policy' or step % 5 == 0
        if detail.get('required') is not required:
            identity_errors.append({'problem': 'Sample identity required flag differs from 5:1:1 schedule', 'details': detail})
        if required:
            if (detail.get('actual_sample_ids_verified') is not True or detail.get('rank_shards_disjoint') is not True
                    or detail.get('global_samples') != MODELS[model][1]):
                identity_errors.append({'problem': 'Required actual sample IDs/disjoint shards/global sample count were not verified', 'details': detail})
            else:
                verified.add((step, model))
    expected_preflight = {(step, model) for step in range(1, 16) for model in MODELS
                          if model == 'policy' or step % 5 == 0}
    return {'scope': f'latest up to {limit} phase records and 64 WM cohort events; in-flight epochs excluded',
            'phase_records': len(rows), 'complete_policy_steps': complete,
            'phase_order_errors': problems, 'release_failures': release_failures,
            'memory_by_model': memory, 'wm_cohorts': cohorts,
            'sample_identity': {'scope': 'latest 1536 driver sample-identity events',
                                'required_verified': len(verified), 'errors': identity_errors,
                                'recent_complete_steps_covered': bool(complete) and
                                    {(step, model) for step in complete for model in MODELS if model == 'policy' or step % 5 == 0} <= verified,
                                'preflight_all_21_required_events_valid': expected_preflight <= verified and not identity_errors},
            'recent_complete_steps_valid': bool(complete) and not problems and not release_failures,
            'preflight_all_15_steps_valid': set(range(1, 16)) <= set(complete) and not problems and not release_failures,
            'preflight_cohorts_valid': {5, 10, 15} <= {item['policy_step'] for item in cohorts if item['valid']}}


def nsight_evidence(data, state):
    directory = Path(data) / 'nsight'
    raw = sorted(directory.glob('*.nsys-rep')) if directory.exists() else []
    exports = sorted(directory.glob('*.sqlite')) if directory.exists() else []
    diagnostics = []
    runner_log = Path(data) / 'runner.log'
    if runner_log.is_file():
        with runner_log.open('rb') as stream:
            stream.seek(max(0, runner_log.stat().st_size - 128 * 1024))
            diagnostics = [line for line in stream.read().decode(errors='replace').splitlines()
                           if any(term in line.lower() for term in ('cupti', 'dropped', 'incomplete', 'nsight systems'))][-15:]
    analyses = []
    for path in sorted(directory.glob('*.overlap.json')) if directory.exists() else []:
        try:
            analysis = json.loads(path.read_text(encoding='utf-8'))
            analyses.append(summarize_nsight_analysis(analysis, state, path))
        except (OSError, ValueError, KeyError, TypeError) as error:
            analyses.append({'path': str(path), 'valid_real_training_window': False, 'error': repr(error)})
    return {'required': True, 'capture': 'bounded NVTX windows; profiler overhead included in end-to-end time',
            'raw_reports': [{'name': item.name, 'bytes': item.stat().st_size} for item in raw],
            'sqlite_exports': [{'name': item.name, 'bytes': item.stat().st_size} for item in exports],
            'exports': state.get('nsight_exports', {}),
            'analyses': analyses,
            'valid_real_training_windows': sum(bool(item.get('valid_real_training_window')) for item in analyses),
            'valid_learning_microstep_windows': sum(bool(item.get('valid_learning_microstep_window')) for item in analyses),
            'recent_diagnostics': diagnostics,
            'boundary_handling': 'inspect incomplete/dropped kernel diagnostics; analyze interior trimmed by 1 s at each end and state coverage',
            'phase_coverage': 'one complete serial Policy epoch 5 (Policy→World→Reward), bounded by a 600 s safety limit; early warmup only',
            'kernel_scope': 'process-attributed kernels include GPU preparation, verification and training; not strict microstep-only attribution',
            'overlap_claim': 'Nsight process-kernel overlap may be zero; a positive intersection alone does not prove useful-work throughput improvement',
            'counter_limit': 'no hardware-counter privilege; GPU busy is not SM/Tensor Core occupancy or bandwidth saturation'}


def summarize_nsight_analysis(analysis, state, path):
    """Validate real clients and map Nsight gpuId by UUID, never CUDA deviceId."""
    normalize = lambda value: str(value or '').removeprefix('GPU-').lower()
    expected = {normalize(card['uuid']): str(card['index']) for card in state.get('preflight', {}).get('cards', [])}
    physical, errors, micro_errors = {}, [], []
    if analysis.get('status') != 'analyzed':
        errors.append('Analysis is partial or failed')
    if analysis.get('role_map', {}).get('run_id') != state['run_id']:
        errors.append('PID role-map run_id differs')
    if analysis.get('device_mapping', {}).get('unresolved_groups'):
        errors.append('At least one client CUDA-device mapping is unresolved')
    micro_status = analysis.get('microstep_attribution', {})
    if not micro_status.get('available') or micro_status.get('row_limit_reached'):
        micro_errors.append('Exact NVTX/CUDA API/correlation learning microstep attribution unavailable or incomplete')
    for key, gpu in analysis.get('per_gpu', {}).items():
        uuid_value = (gpu.get('gpu_info') or {}).get('uuid')
        index = expected.get(normalize(uuid_value))
        if not index or not gpu.get('physical_gpu_identity_resolved'):
            errors.append('Captured GPU UUID is missing or outside approved cards: ' + str(key))
            continue
        training = gpu.get('training', {}).get('all_kernel', {})
        active = training.get('per_model_union_ns', {})
        counts = gpu.get('counts_by_role_and_category', {})
        observed_counts = {role: sum(int(value) for category, value in counts.items()
                                    if category.startswith(role + ':') and category.split(':', 1)[1] in
                                    ('non_nccl_named_kernel', 'nccl_named_kernel', 'unnamed_kernel'))
                           for role in ('policy_trainer', 'ctrl_trainer', 'reward_trainer')}
        if any(active.get(role, 0) <= 0 or observed_counts[role] <= 0 for role in observed_counts):
            errors.append('Not all three trainer processes have actual kernel records on physical GPU ' + index)
        if (gpu.get('unresolved_kernel_union_ns', 0) > 0
                or any(gpu.get('unresolved_kernel_counts_by_pid', {}).values())):
            errors.append('Unattributed kernel records remain on physical GPU ' + index)
        physical[index] = {'nsight_gpu_id': key, 'uuid': uuid_value,
                           'trainer_kernel_counts': observed_counts, 'training_all_kernel': training,
                           'training_activity_scope': gpu.get('training_activity_scope'),
                           'learning_microstep': gpu.get('learning_microstep', {}),
                           'microstep_attributed_kernel_counts_by_role': gpu.get('microstep_attributed_kernel_counts_by_role', {}),
                           'microstep_attributed_fraction_of_trainer_kernel_count': gpu.get('microstep_attributed_fraction_of_trainer_kernel_count', {}),
                           'training_non_nccl_named_kernel': gpu.get('training', {}).get('non_nccl_named_kernel', {}),
                           'training_nccl_named_kernel': gpu.get('training', {}).get('nccl_named_kernel', {}),
                           'all_captured_kernel_fraction_of_window': gpu.get('all_captured_kernel_fraction_of_window'),
                           'kernel_and_gap_duration_by_role': gpu.get('kernel_and_gap_duration_by_role', {})}
        micro_counts = gpu.get('microstep_attributed_kernel_counts_by_role', {})
        micro_union = gpu.get('learning_microstep', {}).get('all_kernel', {}).get('per_model_union_ns', {})
        if any(micro_counts.get(role, 0) <= 0 or micro_union.get(role, 0) <= 0 for role in observed_counts):
            micro_errors.append('Not all three learning microstep models have attributed kernels on physical GPU ' + index)
    if set(physical) != {'4', '5', '6', '7'}:
        errors.append('Four approved physical GPU UUIDs not all represented')
    return {'path': str(path), 'status': analysis.get('status'), 'valid_real_training_window': not errors,
            'valid_learning_microstep_window': not errors and not micro_errors,
            'learning_microstep_verification': 'observed_three_models' if not errors and not micro_errors else 'unresolved',
            'learning_microstep_errors': micro_errors, 'microstep_attribution': micro_status,
            'validation_errors': errors, 'window': analysis.get('window'), 'physical_gpus': physical,
            'diagnostics': analysis.get('diagnostics'), 'conclusion': analysis.get('conclusion'),
            'useful_work_throughput_improvement_proven': False}


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


def audit_final_weights(data, state, *, torch_module=None, persist=True):
    """One final CPU-only mmap read; never load the initial 7 GB snapshot."""
    result = {'valid': False, 'at': stamp(), 'map_location': 'cpu', 'weights_only': True, 'mmap': True,
              'optimizer_or_replay_restore_executed': False, 'cuda_initialized': False}
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        if torch_module is None:
            import torch as torch_module
        if torch_module.cuda.is_initialized():
            raise RuntimeError('CPU checkpoint verifier unexpectedly has an initialized CUDA context')
        directory = training_data(data) / 'checkpoints'
        external = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        path = directory / external.get('weights_file', 'latest_weights.pt')
        payload = torch_module.load(str(path), map_location='cpu', weights_only=True, mmap=True)
        manifest = payload['manifest']
        external_core = {key: value for key, value in external.items() if key not in ('weights_file', 'weights_bytes')}
        if manifest != external_core:
            raise ValueError('External and embedded weight manifests differ')
        if (manifest.get('metadata', {}).get('run_id') != state['run_id'] or manifest.get('world_size') != 4
                or manifest.get('reason') != 'final' or not manifest.get('weights_only')
                or set(payload.get('models', {})) != set(MODELS)):
            raise ValueError('Final run/world/model/weights-only checkpoint identity differs')
        expected = ({'policy': 15, 'ctrl': 3, 'reward': 3} if state.get('experiment_mode') == 'preflight'
                    else {model: item.get('global_updates') for model, item in state.get('report', {}).get('training', {}).items()})
        if manifest.get('steps') != expected:
            raise ValueError(f'Final checkpoint steps {manifest.get("steps")} differ from {expected}')
        counts = {}
        for model, saved in payload['models'].items():
            tensors = saved.get('state_dict', {})
            if (not tensors or any(not torch_module.is_tensor(tensor) or tensor.device.type != 'cpu' for tensor in tensors.values())
                    or not isinstance(saved.get('trainable_names'), list) or not saved['trainable_names']
                    or not set(saved['trainable_names']) <= set(tensors)
                    or saved.get('full_state') is not (model == 'reward')
                    or len(tensors) != manifest.get('tensor_counts', {}).get(model)):
                raise ValueError(model + ': CPU tensor inventory or trainable identity is incomplete')
            counts[model] = len(tensors)
        if external.get('weights_bytes') != path.stat().st_size:
            raise ValueError('Weights file byte size differs from final manifest')
        result.update(valid=True, path=str(path), steps=manifest['steps'], world_size=4,
                      tensor_counts=counts, bytes=path.stat().st_size, manifest_reason=manifest['reason'])
        del payload
    except Exception as error:
        result['error'] = repr(error)
    state['checkpoint_cpu_audit'] = result
    if persist and (training_data(data) / 'collection.sqlite').is_file():
        connection = sqlite3.connect(training_data(data) / 'collection.sqlite', timeout=5)
        try:
            with connection:
                connection.execute('''INSERT INTO events(run_id,ts_ns,actor,actor_id,pid,kind,status,details_json)
                    VALUES(?,?,?,?,?,?,?,?)''', (state['run_id'], time.time_ns(), 'cpu_supervisor', str(os.getpid()),
                    os.getpid(), 'serial_split_final_weights_cpu_audit', 'ok' if result['valid'] else 'error', json.dumps(result)))
        except sqlite3.Error as error:
            result['persistence_error'] = repr(error)
        finally:
            connection.close()
    return result


def audit_sampled_gpu_peaks(data, state):
    connection = safe.sqlite_reader(training_data(data))
    if connection is None: return {}
    try:
        deadline = time.monotonic() + 60
        connection.set_progress_handler(lambda: time.monotonic() > deadline, 10000)
        row = connection.execute('SELECT COUNT(*),MIN(ts_ns),MAX(ts_ns) FROM system_samples WHERE run_id=?',
                                 (state['run_id'],)).fetchone()
        result = {'samples': row[0], 'first_ns': row[1], 'last_ns': row[2], 'gpus': {},
                  'scope': 'sampled physical-card peak, not an unsampled instantaneous maximum; inference and training share each card'}
        for offset, physical in enumerate((4, 5, 6, 7)):
            item = connection.execute(f'''SELECT MAX(json_extract(payload_json,'$.gpus[{offset}].memory_used_bytes')),
                MAX(json_extract(payload_json,'$.gpus[{offset}].gpu_busy_percent')) FROM system_samples
                WHERE run_id=? AND json_extract(payload_json,'$.gpus[{offset}].physical_gpu')=?''',
                (state['run_id'], physical)).fetchone()
            result['gpus'][str(physical)] = {'sampled_peak_used_bytes': item[0], 'sampled_peak_gpu_busy_percent': item[1]}
        state['sampled_gpu_peaks'] = result
        return result
    except sqlite3.Error as error:
        result = {'status': 'unresolved', 'error': repr(error), 'scope': 'final CPU resource-sample aggregation'}
        state['sampled_gpu_peaks'] = result
        return result
    finally:
        connection.close()


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
    connection = safe.sqlite_reader(training_data(data))
    if connection is None:
        return evidence
    try:
        with connection:
            driver = {kind: event(connection, state['run_id'], kind) for kind in
                      ('rollout_started', 'replay_ready', 'synchronized_epoch_completed',
                       'evaluation_threshold', 'accuracy_target_reached', 'tensorboard_started', 'experiment_completed',
                       'split_training_topology_validated')}
            capture_start = event(connection, state['run_id'], 'nsight_capture_started')
            capture_end = event(connection, state['run_id'], 'nsight_capture_ended')
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
            evidence['placements'] = placement_evidence(connection, state['run_id'])
            evidence['topology'] = topology_evidence(driver['split_training_topology_validated'])
            evidence['serial'] = serial_evidence(connection, state['run_id'])
            evidence['checkpoint_cpu_audit'] = state.get('checkpoint_cpu_audit', {})
            evidence['nsight_capture'] = {'started': capture_start, 'ended': capture_end,
                'boundary_audit': event(connection, state['run_id'], 'nsight_capture_boundary_audited')}
            evidence['replay'] = replay_evidence(connection, state['run_id'], evidence['training'])
            evidence['mps'] = live_mps_evidence(state, evidence['placements'])
            evidence['nsight'] = nsight_evidence(data, state)
            health = connection.execute('SELECT ts_ns,payload_json FROM sampler_health WHERE run_id=?',
                                        (state['run_id'],)).fetchone()
            evidence['sampler'] = ({'ts_ns': health['ts_ns'], **json.loads(health['payload_json'])}
                                   if health else {})
            evidence['checkpoint'] = checkpoint_evidence(training_data(data), state, evidence['training'],
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
    preflight = state.get('experiment_mode') == 'preflight'
    start_ns = int(state.get('started_ns', state['started_unix'] * 1e9))
    time_line = lambda name, ns: (f'{name}：{stamp(ns / 1e9)}；从本轮启动计时 {(ns - start_ns) / 60e9:.2f} 分钟。'
                                  if ns else f'{name}：尚未达标。')
    lines = ['# 独立三训练进程按模型组串行：四卡 ZeRO-2 + MPS + Nsight Systems', '',
             f'状态：{state["state"]}。本轮：`{state["run_id"]}`。最近更新：{stamp()}。', '',
             '只使用物理 GPU4–7。每张卡驻留 Policy / World / Reward 三训练模型及三个独立推理副本；',
             '共 24 个 GPU 进程、每卡六进程；三套独立四 rank ZeRO-2 通信组，按Policy→World→Reward逐组完成。推理进程仍可并发。',
             ('本轮是15完整Policy epoch真实预检，目标更新15/3/3；不是正式到85%计时。预检通过后须由父代理另开全新正式轮。'
              if preflight else f'本轮是新模型/优化器/空Replay正式训练，持续到北京时间{DEADLINE_ISO}；85%只记录，不触发停止。'),
             '沿用首次85%用时14小时35分57秒版本的学习配方；新模型初始化、优化器及空Replay。',
             '全局batch为576/1024/1024，GAS为18/64/8，更新比例5:1:1，三个独立warmup500及原30000步LR horizon保留。',
             '正式轮即使已记录85%或Policy步数达到30000也继续到用户截止，不重置学习率或优化器。',
             'CPU线程1、原缓存策略及训练期推理上限2/3/2保留；自适应额度调度关闭。', '',
             '评估沿用现有 TensorBoard 口径：`Eval/_Global/Average_Return`，task 0 的最近 1000 个完整评估 episode 滑窗。',
             '阈值0.85；连续两次统计必须含新增完整episode，窗口可重叠；不足1000个样本不记录达标。',
             '保持在线评估，episode 可能跨模型版本，不能解释为冻结权重的独立评估批次。', '',
             ('预检不产生正式TimeTo85结论。' if preflight else time_line('首次达到 85%', first)),
             ('预检评估曲线仅为真实学习记录，不得拼接到后续正式达标时间。' if preflight else time_line('连续第二次达到 85% 的评估统计完成时间', second)),
             '计时取 StatsActor 完成评估快照的时间；另保留最新完整 episode 的完成时间，不取 TensorBoard 文件写入时间。', '',
             ('预检在15完整epoch结束；正式轮使用用户明确截止，'
              if preflight else f'正式截止Unix秒为{DEADLINE_UNIX}，对应UTC 2026-10-19 01:00:00；'),
             '截止在完整更新边界停止并保存最终三模型，允许900秒保存宽限；程序错误/卡住仍按120秒安全停止宽限。',
             '首个成功更新120分钟、后续成功更新停滞60分钟异常保护继续生效。',
             'completed_at_deadline仅表示到期正常结束，与是否历史达到85%独立；截止仍未达标时如实保留未达标。', '',
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
              '最终/当前曲线值独立报告：' + str(tb.get('latest_metric') or
                  (evidence.get('driver', {}).get('evaluation_threshold') or {}).get('details', {}).get('rate', '待首次评估')),
              '历史85%里程碑一旦记录即保留；后续成功率跌破85%不改写既有两项达标时间。',
              '本轮创建新事件文件，保留用户给出的历史曲线。',
              f'最终权重覆盖完整 epoch：{evidence.get("checkpoint", {}).get("covers_last_complete_epoch", False)}。',
              '最终一次CPU mmap权重核验：' + json.dumps(state.get('checkpoint_cpu_audit', {}), ensure_ascii=False),
              '权重快照在 training/checkpoints/latest_weights.pt；只保存权重和 buffers，缺优化器、Replay 与 RNG，不能精确续训。',
              f'本轮训练及 Ray 子进程确认停止：{state.get("training_stopped", False)}。',
              f'结束原因：{state.get("reason", "运行中")}。',
              f'校验问题：{json.dumps(state.get("validation_errors", []), ensure_ascii=False)}。', '',
              f'十二训练进程与三套四rank通信组已核验：{evidence.get("topology", {}).get("valid", False)}；'
              f'独立端口：{evidence.get("topology", {}).get("group_ports", {})}。',
              '只需查看本 README；异常记录在 runner.log，状态及计数留在 runnerstatus.json、training/collection.sqlite。', '',
              '## Nsight Systems 验证', '',
              '只采集训练期有界NVTX窗口，保留CUDA/NVTX原始时间线及CPU导出的SQLite。',
              '采样开销计入本轮端到端达标时间；窗口结束后训练继续，不因采样结束停止学习。',
              '当前原始报告：' + json.dumps(evidence.get('nsight', {}).get('raw_reports', []), ensure_ascii=False),
              '当前导出：' + json.dumps(evidence.get('nsight', {}).get('exports', {}), ensure_ascii=False),
              '主采样从第5个到期Policy epoch开始，在完整P→World→Reward返回后关闭；600秒仅为硬安全上限。触及上限则覆盖不足。',
              '该窗口属于早期warmup，不声称覆盖整轮或多个训练阶段；串行训练模型交集预期为0，实测不强行改成正重叠。',
              '进程归因kernel包含准备、校验、训练等GPU工作；不把进程区间直接称为纯microstep计算重叠。',
              '有效重叠：待核对同一物理GPU的跨模型kernel时间线、kernel间隙、NCCL与吞吐，不能仅凭多个客户端或CUDA提交并行宣称提速。',
              '无GPU硬件计数器权限，因此不能由busy百分比推断SM/Tensor Core占用或带宽饱和；CPU计数含忙等，宿主总量含他人。',
              '三个模型各自warmup500；无MPS-off公平对照，不把本轮端到端差异单独归因为重叠技术。', '',
              '## Replay真实消费', '',
              '预取不计消费；四rank样本曝光合计与完成优化器步数×全局batch对账。',
              '运行中差额可能来自在途microstep或SQLite异步flush；淘汰未使用按first_used_ns和evicted_ns时序判定。',
              '```json', json.dumps(evidence.get('replay', {}), ensure_ascii=False, indent=2), '```']
    serial = evidence.get('serial', {})
    lines += ['', '## 串行阶段与显存', '',
              '范围：' + str(serial.get('scope', '等待真实阶段事件')),
              '完整阶段Policy步：' + str(serial.get('complete_policy_steps', [])),
              '阶段顺序问题：' + json.dumps(serial.get('phase_order_errors', []), ensure_ascii=False),
              'GPU批次引用释放问题：' + json.dumps(serial.get('release_failures', []), ensure_ascii=False),
              '各模型进程阶段allocated/reserved/peak：' + json.dumps(serial.get('memory_by_model', {}), ensure_ascii=False),
              '常驻模型/优化器/NCCL等内存可以保留；不把allocated/reserved要求为0。',
              '整卡采样峰值与覆盖：' + json.dumps(state.get('sampled_gpu_peaks', {}), ensure_ascii=False),
              'WM同cohort八消费者：' + json.dumps(serial.get('wm_cohorts', []), ensure_ascii=False),
              '实际sample ID逐组与分片身份核验：' + json.dumps(serial.get('sample_identity', {}), ensure_ascii=False),
              'GPU busy不是SM利用率；采样峰值与Torch进程峰值各自注明范围。']
    for analysis in evidence.get('nsight', {}).get('analyses', []):
        lines += ['', '## Nsight 已分析窗口', '',
                  f'分析结果：`{analysis.get("path")}`；真实三训练进程与物理卡归因有效：{analysis.get("valid_real_training_window", False)}。',
                  '分析限制/错误：' + json.dumps(analysis.get('validation_errors', []), ensure_ascii=False),
                  '窗口及边界覆盖：' + json.dumps(analysis.get('window', {}), ensure_ascii=False),
                  'CUPTI边界/不完整事件诊断：' + json.dumps(analysis.get('diagnostics', {}), ensure_ascii=False), '',
                  '| 物理GPU | Policy/World/Reward真实kernel数 | ≥2训练进程kernel同时执行占分析窗 |',
                  '| --- | --- | --- |']
        for gpu, item in sorted(analysis.get('physical_gpus', {}).items()):
            counts = item.get('trainer_kernel_counts', {})
            count_text = '/'.join(str(counts.get(role, 0)) for role in
                                  ('policy_trainer', 'ctrl_trainer', 'reward_trainer'))
            share = item.get('training_all_kernel', {}).get('at_least_two_fraction_of_window')
            lines.append(f'| {gpu} | {count_text} | {100 * share:.3f}% |' if share is not None
                         else f'| {gpu} | {count_text} | unresolved |')
        lines += ['', '以上是实际kernel时间戳区间交集；NCCL可能含等待，非NCCL命名kernel也不保证都是有效矩阵计算。',
                  '是否提高整体吞吐仍须对照完成更新/真实样本数及可比时间窗，不能由同时执行占比单独推出。']
    if state.get('report_error'):
        lines += ['', '报告暂未刷新：' + state['report_error']]
    prior_start = state.get('experiment_environment', {}).get('ACCERL_PRIOR_STARTUP_FAILURE')
    if prior_start:
        lines += ['', '启动记录：11:42:49 的前次初始化因新增拓扑报告调用早于报告对象创建而退出；',
                  '三个模型均为0次优化器更新，全部本轮模型/Ray/监督进程已退出，MPS保留。',
                  '仅修正记录调用位置后重新启动；学习计算、配方和模型初始化未改。',
                  '本页及TimeTo85计时从当前run_id的新启动开始，未将前次失败启动及修复耗时混入训练用时。',
                  '前次失败原始证据：`' + prior_start + '`。']
    correction = state.get('posthoc_report_correction')
    if correction:
        lines += ['', '## CPU报告复核', '',
                  '本轮训练正常完成15/3/3；旧报告因全局行数截断和NVTX关闭等待计时而判为failed。原始状态及事件保留。',
                  '原始报告备份：`' + correction['original_terminal_reports_backup'] + '`。',
                  f'原始NVTX窗口：{correction["capture_duration_seconds"]:.6f}秒；关闭调用等待/导出：'
                  f'{correction["synchronous_end_return_delay_seconds"]:.6f}秒，仍计入实际端到端耗时。',
                  '已按源表分批完整读取已保存的GPU活动并严格关联学习微步；未重跑训练、未改原评估结果。',
                  '全量读取不等于原始采集无缺失；CUPTI边界丢弃及裁边限制仍保留。']
    temporary = data / f'.README.{os.getpid()}.tmp'
    temporary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    os.replace(temporary, data / 'README.md')
    return evidence


def watchdog_reason(evidence, state, now_ns=None):
    now_ns = time.time_ns() if now_ns is None else now_ns
    if state.get('experiment_mode') == 'target' and now_ns >= DEADLINE_NS:
        return DEADLINE_REASON
    started = evidence.get('rollout_started_ns') or state['started_ns']
    for model in MODELS:
        progress = evidence.get('training', {}).get(model, {}).get('ranks', {}).get('0', {})
        first = progress.get('first_success_ns')
        if first and now_ns - (progress.get('last_success_ns') or first) > 3600 * 10**9:
            return model + '_successful_update_stalled_60_minutes'
        if not first and now_ns - started > 7200 * 10**9:
            return model + '_no_first_successful_update_120_minutes'
    return None


def validate_deadline_completion(evidence):
    completed = evidence.get('driver', {}).get('experiment_completed') or {}
    details = completed.get('details', {})
    errors = []
    if completed.get('ts_ns', 0) < DEADLINE_NS:
        errors.append('The formal training completion event precedes the user deadline or is absent')
    if (details.get('stop_reason') != DEADLINE_REASON or details.get('until_deadline') is not True
            or details.get('stop_on_target') is not False
            or details.get('deadline_unix_seconds') != DEADLINE_UNIX):
        errors.append('The application did not confirm deadline-only normal stopping with the exact user deadline')
    policy = details.get('stop_policy', {})
    if (policy.get('until_deadline') is not True or policy.get('stop_on_target') is not False
            or policy.get('deadline_unix_seconds') != DEADLINE_UNIX):
        errors.append('The recorded runtime stop policy does not match the deadline contract')
    if details.get('global_step') != evidence.get('last_complete_epoch_step'):
        errors.append('The deadline completion event does not cover the final complete Policy epoch')
    if details.get('target_reached') is not bool(evidence.get('target_reached_event')):
        errors.append('The final historical-85% latch disagrees with the durable attainment event')
    if bool(evidence.get('target', {}).get('should_stop')) != bool(evidence.get('target_reached_event')):
        errors.append('The frozen historical threshold state disagrees with the attainment event')
    return errors


def capture_boundary_verified(evidence):
    """Accept native timing, or a separately recorded raw-NVTX CPU audit.

    The latter repairs the old range-end/export timing bug. It never edits the
    old events or infers a boundary from CUDA API duration or kernel extrema.
    """
    record = evidence.get('nsight_capture', {})
    ended = record.get('ended') or {}
    capture = ended.get('details', {})
    if capture.get('capture_mode') != 'epoch' or capture.get('policy_step') != 5:
        return False
    if capture.get('reason') == 'completed_epoch' and capture.get('coverage_complete') is True:
        return True
    if capture.get('reason') != 'completed_epoch_after_deadline':
        return False
    event_audit = record.get('boundary_audit') or {}
    audit = event_audit.get('details', {})
    started = record.get('started') or {}
    if (event_audit.get('status') != 'ok' or audit.get('schema_version') != 1
            or audit.get('original_end_event_id') != ended.get('id')
            or audit.get('original_capture_started_ns') != capture.get('started_ns')
            or audit.get('run_id') != ended.get('run_id')
            or audit.get('source_capture_sha256') != '15a1c537e82f481674b7f86fa0cfc3ce15495d9715d1cadf69f7d47c80b29552'
            or not audit.get('all_four_gpu_three_models_kernel_and_microstep_verified')
            or audit.get('phase_begin_count') != 12 or audit.get('phase_end_count') != 12
            or audit.get('phase_order_and_ranks_verified') is not True):
        return False
    try:
        session = int(audit['session_utc_epoch_ns'])
        left, right = int(audit['nvtx_start_ns']), int(audit['nvtx_end_ns'])
        first, last = int(audit['first_phase_begin_ns']), int(audit['last_phase_end_ns'])
        driver = (int(audit['nvtx_global_tid']) >> 24) & 0xffffff
        limit = float(capture['duration_limit_seconds'])
        return (driver == started.get('pid') == ended.get('pid')
                and 0 <= left < right <= int(audit['diagnostic_stop_ns'])
                and 0 < (right - left) / 1e9 <= limit <= 600
                and session + left <= first < last <= session + right
                and capture.get('completed_ns', 0) >= session + right
                and evidence.get('nsight', {}).get('valid_real_training_windows', 0) > 0
                and evidence.get('nsight', {}).get('valid_learning_microstep_windows', 0) > 0
                and evidence.get('serial', {}).get('preflight_all_15_steps_valid') is True)
    except (KeyError, ValueError, TypeError):
        return False


def validate_completion(state):
    evidence = state.get('report', {})
    errors = []
    if not state.get('training_stopped'):
        errors.append('Experiment-owned training or Ray processes still alive')
    if not evidence.get('placements', {}).get('complete_expected_placement'):
        errors.append('Four-card / six-process placement not confirmed')
    if not evidence.get('topology', {}).get('valid'):
        errors.append('Twelve trainer states and three independent world-4 groups not confirmed')
    for model in MODELS:
        item = evidence.get('training', {}).get(model, {})
        if not item.get('rank_counts_match') or not item.get('global_batch_valid'):
            errors.append(model + ': rank counters or global batch disagree')
        if any(rank.get('skipped_steps') != 0 for rank in item.get('ranks', {}).values()):
            errors.append(model + ': skipped optimizer updates occurred')
    if not evidence.get('checkpoint', {}).get('covers_last_complete_epoch'):
        errors.append('Current-run three-model checkpoint does not cover the final complete epoch')
    if not evidence.get('mps', {}).get('protected_server_identity_valid'):
        errors.append('Protected MPS server identity could not be confirmed')
    if not evidence.get('nsight', {}).get('raw_reports'):
        errors.append('Required Nsight Systems kernel timeline report is absent')
    if not any(item.get('state') == 'complete' and item.get('kernel_table_present')
               for item in evidence.get('nsight', {}).get('exports', {}).values()):
        errors.append('No readable Nsight SQLite export with a CUDA kernel timeline was confirmed')
    if not evidence.get('nsight', {}).get('valid_real_training_windows'):
        errors.append('No analyzed real-training window with valid client/GPU identity and actual kernels from each of three trainer roles on all four GPUs')
    if not evidence.get('nsight', {}).get('valid_learning_microstep_windows'):
        errors.append('Learning microstep kernel attribution is unresolved; process-kernel overlap alone cannot verify learning execution overlap')
    if not capture_boundary_verified(evidence):
        errors.append('Nsight did not confirm a complete epoch-5 serial P→World→Reward capture before its safety limit')
    if state.get('profiler_finalized') is False:
        errors.append('Nsight CPU report conversion did not finish within its bounded finalization grace')
    for model, replay in (('policy', 'policy'), ('ctrl', 'wm'), ('reward', 'wm')):
        balance = evidence.get('replay', {}).get(replay, {}).get(model + '_balance', {})
        if balance.get('difference_including_inflight_and_sqlite_flush') != 0:
            errors.append(model + ': final actual sample exposure balance is not exact')
    tb = evidence.get('tensorboard', {})
    if not tb.get('readable_verified') or METRIC not in tb.get('scalar_tags', []):
        errors.append('The new TensorBoard event directory or existing evaluation tag could not be read')
    if not evidence.get('serial', {}).get('recent_complete_steps_valid'):
        errors.append('Serial model-group phase barriers or released GPU batch references were not verified')
    identity = evidence.get('serial', {}).get('sample_identity', {})
    if identity.get('errors') or not identity.get('recent_complete_steps_covered'):
        errors.append('Actual sample-ID/disjoint-rank validation is missing or false for a required completed model update')
    if not state.get('checkpoint_cpu_audit', {}).get('valid'):
        errors.append('Final three-model weights have not passed the one-time CPU mmap audit')
    if state.get('experiment_mode') == 'preflight':
        if evidence.get('last_complete_epoch_step') != PREFLIGHT_EPOCHS:
            errors.append('Preflight did not finish exactly 15 complete Policy epochs')
        expected = {'policy': 15, 'ctrl': 3, 'reward': 3}
        if any(evidence.get('training', {}).get(model, {}).get('global_updates') != steps for model, steps in expected.items()):
            errors.append('Preflight successful updates differ from 15/3/3')
        if not evidence.get('serial', {}).get('preflight_all_15_steps_valid'):
            errors.append('Preflight lacks all fifteen P→World→Reward barrier cycles')
        if not evidence.get('serial', {}).get('preflight_cohorts_valid'):
            errors.append('Preflight lacks the three complete same-cohort eight-consumer WM audits')
        identity = evidence.get('serial', {}).get('sample_identity', {})
        if not identity.get('preflight_all_21_required_events_valid'):
            errors.append('Preflight actual sample-ID validation is missing/false for one of 15 Policy + 3 World + 3 Reward updates')
        return errors  # A bounded preflight is never a formal TimeTo85 result.
    errors.extend(validate_deadline_completion(evidence))
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
    else:
        # One first qualifying snapshot may exist without a consecutive pair.
        # Verify its two float64 markers without inventing a second attainment.
        target = evidence.get('target', {})
        first, _ = target_times(target)
        started_ns = integer(target.get('run_started_ns'))
        if first and started_ns is not None:
            for tag, expected in zip(MILESTONE_TAGS[:2], ((first - started_ns) / 1e9, first / 1e9)):
                marker = tb.get('milestones', {}).get(tag, {})
                actual = marker.get('value')
                if marker.get('dtype') != 'float64' or actual is None or not math.isfinite(actual) or abs(actual - expected) >= 1e-6:
                    errors.append(tag + ': first-attainment marker missing or inconsistent')
    return errors


def build_command(root, experiment_mode='target'):
    return ['bash', str(root / 'run_oft_ctrl_world_train_wm.sh'), '--cuda-visible-devices', '4,5,6,7',
            '--gpu-layout', LAYOUT, '--num-trainer-gpus', '4', '--num-inference-actors', '4',
            '--num-reward-inference-actors', '4', '--num-ctrl-inference-actors', '4',
            '--accumulation-steps', '18', '--ctrl-accumulation-steps', '64', '--reward-accumulation-steps', '8',
            '--exp-name', 'OpenVLA_DS_gipo_DISCRETE_task0_ctrl_wm_four_zero2_mps_serial_split_nsysfix_deadline_' + experiment_mode]


def profile_command(command, data):
    # The target program keeps running after the bounded ranges. The CLI
    # profiles only this invocation's descendants; no system-wide attachment.
    return [str(NSYS), 'profile', '--trace=cuda-sw,nvtx', '--sample=none', '--cpuctxsw=none',
            '--capture-range=nvtx', '--nvtx-capture=PROFILE_WINDOW', '--capture-range-end=stop',
            '--kill=none', '--wait=all', '--cuda-event-trace=false', '--gpu-metrics-devices=none',
            '--cuda-flush-interval=1000', '--flush-on-cudaprofilerstop=true',
            '--export=sqlite', '--force-overwrite=false',
            '--output=' + str(Path(data) / 'nsight/training')] + command


def build_environment(data, run_id, started_ns, cards, experiment_mode='target'):
    data = training_data(data)
    env = dict(os.environ)
    for key in ('ACCERL_MEASURE_STOP_AFTER_EPOCHS', 'ACCERL_SOFT_STOP_AT', 'ACCERL_CAPTURE_ONLY',
                'ACCERL_STOP_ON_TARGET_85', 'ACCERL_RUN_UNTIL_DEADLINE',
                'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE', 'CUDA_MPS_PINNED_DEVICE_MEM_LIMIT'):
        env.pop(key, None)
    env.update({MARKER: run_id, 'ACCERL_RUN_STARTED_NS': str(started_ns), 'ACCERL_USE_MPS': '1',
                'ACCERL_DATA_DIR': str(data), 'ACCERL_RESOURCE_GOVERNOR': '0',
                'ACCERL_SPLIT_TRAIN_SERIAL': '1',
                'ACCERL_SYSTEM_PROFILE': '1', 'ACCERL_NSIGHT_MODE': 'epoch', 'ACCERL_NSIGHT_STEPS': '5',
                'ACCERL_NSIGHT_SECONDS': '600', 'ACCERL_NSIGHT_NVTX': '1',
                'NSYS_NVTX_PROFILER_REGISTER_ONLY': '0',
                'ACCERL_CAPTURE_ONLY': '1', 'ACCERL_TENSORBOARD': '1',
                'ACCERL_TELEMETRY_DB': str(data / 'collection.sqlite'), 'ACCERL_CPU_THREADS': '1',
                'CUDA_VISIBLE_DEVICES': ','.join(item['uuid'] for item in cards),
                'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'CUDA_MPS_PIPE_DIRECTORY': MPS_PIPE,
                'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                'TF_FORCE_GPU_ALLOW_GROWTH': 'true', 'PYTHONUNBUFFERED': '1',
                'ACCERL_RAY_TMP': f'/dev/shm/as85_{os.getuid()}_{run_id[-8:]}',
                'ACCERL_RAY_SPILL': str(data / 'ray_spill'),
                'ACCERL_STOP_FILE': str(data / 'stop.flag'), 'ACCERL_FINAL_CHECKPOINT': str(data / 'checkpoints'),
                'ACCERL_FILL_TIMEOUT_SEC': '7200', 'ACCERL_TRIM_IDLE_CACHE': '1',
                'ACCERL_TRIM_MICRO_CACHE': '1', 'ACCERL_SNAPSHOT_INTERVAL': '25',
                'ACCE_CAPTURE_TRACE_EVERY': '10', 'ACCERL_EVAL_TARGET': '0.85',
                'ACCERL_EVAL_CONSECUTIVE': '2', 'ACCERL_EVAL_MIN_EPISODES': '1000'})
    if experiment_mode == 'preflight':
        env['ACCERL_MEASURE_STOP_AFTER_EPOCHS'] = str(PREFLIGHT_EPOCHS)
        env['ACCERL_STOP_ON_TARGET_85'] = '0'
        env['ACCERL_RUN_UNTIL_DEADLINE'] = '0'
    elif experiment_mode == 'target':
        env['ACCERL_STOP_ON_TARGET_85'] = '0'
        env['ACCERL_RUN_UNTIL_DEADLINE'] = '1'
        env['ACCERL_SOFT_STOP_AT'] = str(DEADLINE_UNIX)
    elif experiment_mode != 'target':
        raise ValueError('Experiment mode must be preflight or target')
    return env


def start_sampler(data, state, log):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    env.pop(MARKER, None)
    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--mode', 'sample',
                                '--data', str(data), '--run-id', state['run_id']], env=env,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    state['sample'] = proc_identity(process.pid)
    if not state['sample']:
        raise RuntimeError('Could not establish CPU sampler PID birth identity')
    return process


def sample(data, run_id):
    # The reviewed sampler is CPU/NVML-only. Its DB is under training/ while
    # supervisor state is in the outer directory; never copy/rewrite that state.
    directory = training_data(data)
    while not directory.is_dir():
        state = load_status(data)
        if state.get('run_id') != run_id or state.get('state') in TERMINAL:
            return 0
        if not same_process(state.get('supervisor'), proc_identity(state.get('supervisor', {}).get('pid', -1))):
            return 0
        time.sleep(.25)
    spec = importlib.util.spec_from_file_location('split_reviewed_cpu_sampler', SAMPLER)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.load_status = lambda _directory: load_status(data)
    module.TERMINAL = TERMINAL
    module.safe.placement_evidence = placement_evidence
    return module.sample(directory, run_id)


def update_nsight_exports(data, state, pending, log):
    """Inspect Nsight's per-range CPU export; never race its SQLite writer."""
    records = state.setdefault('nsight_exports', {})
    for name, process in list(pending.items()):
        code = process.poll()
        if code is not None:
            records[name].update(analysis='analyzed' if code == 0 else 'incomplete_row_budget' if code == 3 else 'failed',
                                 analysis_return_code=code, analysis_finished=stamp())
            del pending[name]
    directory = Path(data) / 'nsight'
    for source in sorted(directory.glob('*.nsys-rep')) if directory.exists() else []:
        if records.get(source.name, {}).get('state') == 'complete' or source.stat().st_size == 0:
            continue
        destination = source.with_suffix('.sqlite')
        record = records.setdefault(source.name, {'state': 'automatic_export_pending',
                                    'path': str(destination), 'observed': stamp(),
                                    'export_owner': 'nsys --export=sqlite, CPU export per completed bounded window',
                                    'analysis': 'not_yet_analyzed'})
        if (destination.is_file() and time.time() - destination.stat().st_mtime >= 10
                and time.time() - source.stat().st_mtime >= 10):
            try:
                connection = sqlite3.connect(destination.resolve().as_uri() + '?mode=ro', uri=True, timeout=.1)
                tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")]
                connection.close()
                record.update(state='complete', bytes=destination.stat().st_size,
                              kernel_table_present='CUPTI_ACTIVITY_KIND_KERNEL' in tables, finished=stamp())
            except sqlite3.Error as error:
                record['last_read_error'] = repr(error)
    if pending:
        return  # At most one bounded CPU analyzer at a time.
    placements = state.get('report', {}).get('placements', {})
    if not placements.get('complete_expected_placement'):
        return
    role_map = directory / 'roles.json'
    if not role_map.exists():
        atomic_json(role_map, {'run_id': state['run_id'], 'identity_scope': 'recorded UUID/card/rank placements',
                              'processes': [dict(item, rank=integer(item.get('actor_id')))
                                            for item in placements['actors']]})
    for name, record in records.items():
        if record.get('state') != 'complete' or record.get('analysis') != 'not_yet_analyzed':
            continue
        source = Path(record['path']); output = source.with_suffix('.overlap.json')
        if output.exists():
            record.update(analysis='existing_result_requires_review', analysis_output=str(output))
            continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', ACCERL_RUN_ID=state['run_id'])
        command = ['nice', '-n', '19', sys.executable, str(ANALYZER), str(source),
                   '--role-map', str(role_map), '--run-id', state['run_id'], '--phase', source.stem,
                   '--trim-ns', '1000000000', '--batch-rows', '250000',
                   '--max-total-rows', '100000000', '--timeout-seconds', '600',
                   '--max-intervals', '40000000', '--max-correlation-keys', '6000000',
                   '--output', str(output)]
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, close_fds=True)
        pending[name] = process
        record.update(analysis='running', analysis_started=stamp(), analysis_output=str(output),
                      analysis_process=proc_identity(process.pid), analysis_cpu_only=True,
                      analysis_priority='nice 19; one OMP/MKL/OpenBLAS thread', analysis_command=command)
        return


def start_guard(data, state, log):
    env = dict(os.environ); env.pop(MARKER, None)
    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--mode', 'guard',
                                '--data', str(data), '--run-id', state['run_id']], env=env,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    state['guard'] = proc_identity(process.pid)
    if not state['guard']:
        raise RuntimeError('Could not establish guard PID birth identity')
    return process


def run(data, root, experiment_mode='target'):
    if os.name != 'posix' or not Path('/proc').is_dir():
        raise RuntimeError('Run mode requires the Linux training server')
    # Every approved experiment has exactly one fresh outer directory. An
    # existing failed/stopped run is evidence, never an invitation to restart.
    data.mkdir(parents=True, exist_ok=False)
    with safe.exclusive(data / '.runner.lock'):
        prior = load_status(data)
        if prior:
            raise RuntimeError('This directory already contains an attempt; no duplicate was started')
        pre = preflight()
        started_ns = time.time_ns()
        run_id = ('sersplitpre_' if experiment_mode == 'preflight' else 'sersplitdl_') + dt.datetime.now(TZ).strftime('%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
        (data / 'nsight').mkdir()
        command = profile_command(build_command(root, experiment_mode), data)
        state = {'state': 'starting', 'run_id': run_id, 'started': stamp(started_ns / 1e9),
                 'started_unix': started_ns / 1e9, 'started_ns': started_ns, 'root': str(root),
                 'uid': os.getuid(), 'supervisor': proc_identity(os.getpid()), 'command': command,
                 'experiment_mode': experiment_mode,
                 'minimum_child_birth_ticks': int(float(Path('/proc/uptime').read_text().split()[0]) * os.sysconf('SC_CLK_TCK')),
                 'owned': [], 'training_stopped': False, 'preflight': pre,
                 'previous_attempts': (prior.get('previous_attempts', []) + ([{key: prior.get(key) for key in
                                      ('run_id', 'state', 'started', 'finished', 'reason', 'return_code')}] if prior else []))[-10:],
                 'plan': {'physical_gpus': [4, 5, 6, 7], 'layout': LAYOUT, 'mps': True,
                          'gpu_processes': 24, 'trainer_processes': 12, 'training_communicators': 3,
                          'global_batches': {'policy': 576, 'ctrl': 1024, 'reward': 1024},
                          'metric_tag': METRIC, 'threshold': 0.85, 'consecutive': 2, 'min_eval_episodes': 1000,
                          'learning_rate_horizon_policy_steps': 30000, 'stop_at_original_plan_steps': False,
                          'total_deadline': None if experiment_mode == 'preflight' else DEADLINE_ISO,
                          'deadline_unix_seconds': None if experiment_mode == 'preflight' else DEADLINE_UNIX,
                          'stop_on_target_85': False, 'deadline_save_grace_seconds': 900,
                          'error_stop_grace_seconds': 120, 'automatic_restarts': False,
                          'training_dispatch': 'serial_model_groups_policy_then_ctrl_then_reward',
                          'preflight_epochs': PREFLIGHT_EPOCHS if experiment_mode == 'preflight' else None,
                          'expected_preflight_steps': {'policy': 15, 'ctrl': 3, 'reward': 3} if experiment_mode == 'preflight' else None,
                          'nsight_steps': [5], 'nsight_seconds_limit': 600,
                          'nsight_hardware_counters': False, 'nsight_trace_method': 'cuda-sw',
                          'nsight_tool_version': '2026.5.1.161', 'nsight_previous_native_abort': 'CUPTI worker threads / 2025.3.2'}, 'training_data': str(training_data(data))}
        sources = [root / 'run_oft_ctrl_world_train_wm.sh'] + [root / 'rl/ctrl_world' / name for name in
                   ('ds_wm_discrete_ctrl_train_wm.py', 'training_topology.py', 'capture_hooks.py',
                    'capture_training.py', 'capture_telemetry.py', 'telemetry.py', 'parallel_checkpoint.py',
                    'shared_inference.py', 'evaluation_threshold.py', 'nsight_capture.py', 'split_serial.py',
                    'training_stop_policy.py')]
        state['source_sha256'] = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
        state['runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        env = build_environment(data, run_id, started_ns, pre['cards'], experiment_mode)
        state['experiment_environment'] = {key: value for key, value in env.items()
                                           if key.startswith(('ACCERL_', 'ACCE_CAPTURE_', 'CUDA_MPS_', 'NSYS_'))
                                           or key in ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}
        atomic_json(data / 'runnerstatus.json', state)
        process = guard_process = sampler_process = None
        pending_exports = {}
        with (data / 'runner.log').open('a', encoding='utf-8', buffering=1) as log:
            log.write(f'\n[{stamp()}] START {run_id}; mode={experiment_mode}; user deadline={DEADLINE_ISO}; 85% records only; no automatic restart.\n')
            try:
                guard_process = start_guard(data, state, log)
                sampler_process = start_sampler(data, state, log)
                atomic_json(data / 'runnerstatus.json', state)
                process = subprocess.Popen(command, cwd=root, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
                state['child'] = proc_identity(process.pid); state['state'] = 'running'
                next_report, stopping_since, application_done_since = 0, None, None
                while process.poll() is None:
                    now = time.monotonic()
                    active = remember_owned(state)
                    driver_reason = training_driver_liveness(state, root, active)
                    if driver_reason:
                        # Nsight --wait=all can remain alive after Python has
                        # crashed because Ray descendants still exist. Read the
                        # final durable event directly before classifying it.
                        try:
                            latest = collect_report(data, state)
                        except Exception as read_error:
                            latest = {}
                            state['driver_completion_read_error'] = repr(read_error)
                        completed = latest.get('driver', {}).get('experiment_completed')
                        if completed and latest.get('checkpoint', {}).get('covers_last_complete_epoch'):
                            application_done_since = application_done_since or now
                            state['target_program_completed_event'] = completed
                        else:
                            state.setdefault('reason', driver_reason)
                            request_stop(data, driver_reason)
                            stop_training_keep_profiler(state, data)
                            break
                    update_nsight_exports(data, state, pending_exports, log)
                    if now >= next_report:
                        evidence = write_report(data, state); next_report = now + 30
                        completed = evidence.get('driver', {}).get('experiment_completed')
                        if completed and evidence.get('checkpoint', {}).get('covers_last_complete_epoch'):
                            application_done_since = application_done_since or now
                            state['target_program_completed_event'] = completed
                        reason = watchdog_reason(evidence, state)
                        if not same_process(state['guard'], proc_identity(state['guard']['pid'])):
                            reason = 'independent_guard_missing'
                        if sampler_process.poll() is not None:
                            reason = 'cpu_sampler_exited_unexpectedly'
                        if evidence.get('sampler') and time.time_ns() - evidence['sampler']['ts_ns'] > 30 * 10**9:
                            reason = 'cpu_sampler_heartbeat_stale_30_seconds'
                        if not evidence.get('sampler') and time.time_ns() - state['started_ns'] > 120 * 10**9:
                            reason = 'cpu_sampler_no_initial_heartbeat_120_seconds'
                        if reason:
                            state.setdefault('reason', reason)
                            request_stop(data, reason)
                    if application_done_since is not None and now - application_done_since >= 15:
                        # --wait=all includes Ray descendants that can outlive
                        # the driver. Its final durable event + weights are the
                        # application completion boundary, not profiler exit.
                        state['cleanup_after_application_completion'] = True
                        stop_training_keep_profiler(state, data)
                        break
                    if (training_data(data) / 'stop.flag').exists():
                        try:
                            stop_reason = json.loads((training_data(data) / 'stop.flag').read_text(encoding='utf-8')).get('reason')
                            if stop_reason:
                                state.setdefault('reason', stop_reason)
                        except (OSError, ValueError):
                            pass
                        stopping_since = stopping_since or now
                        # One bounded chance to save at an epoch boundary after a failure/stall.
                        if now - stopping_since > stop_grace_seconds(state.get('reason')):
                            stop_training_keep_profiler(state, data); break
                    time.sleep(3)
                finalize_profiler(data, state, process, log, pending_exports)
                update_nsight_exports(data, state, pending_exports, log)
                stopped = cleanup(state, data)
                process.wait(timeout=10)
                state['return_code'] = process.returncode
                time.sleep(2)
                write_report(data, state, verify_tb=True)
                audit_final_weights(data, state)
                audit_sampled_gpu_peaks(data, state)
                errors = validate_completion(state)
                state['validation_errors'] = errors
                reached = state.get('report', {}).get('target_reached_event', False)
                if not stopped:
                    state['state'] = 'cleanup_requires_attention'
                elif user_stop_reason(state.get('reason')):
                    state['state'] = 'stopped_by_user'
                elif state.get('reason', '').endswith('minutes'):
                    state['state'] = 'timed_out'
                elif (process.returncode == 0 or state.get('target_program_completed_event')) and not errors:
                    if experiment_mode == 'preflight':
                        state['state'] = 'preflight_passed'
                        state['reason'] = '15_complete_serial_policy_epochs_and_15_3_3_cpu_verified_final_weights'
                    else:
                        state['state'] = 'completed_at_deadline'
                        state['reason'] = DEADLINE_REASON
                        state['target_85_reached'] = reached
                else:
                    state['state'] = 'failed'; state.setdefault('reason', 'training_or_final_validation_failed')
            except BaseException as error:
                state['state'] = 'failed'; state['reason'] = repr(error)
                request_stop(data, 'supervisor_error')
                if process is not None:
                    try:
                        finalize_profiler(data, state, process, log, pending_exports)
                    except Exception as finalization_error:
                        state['profiler_finalization_error'] = repr(finalization_error)
                    finally:
                        cleanup(state, data)
                        state['state'] = 'failed'
                else:
                    state['training_stopped'] = True
                log.write(f'[{stamp()}] SUPERVISOR ERROR {error!r}\n')
            finally:
                state['finished'] = stamp(); write_report(data, state, verify_tb=state['state'] in TERMINAL)
                log.write(f'[{stamp()}] FINISH {state["state"]}: {state.get("reason")}\n')
                if guard_process is not None:
                    try:
                        guard_process.wait(timeout=20)
                        state['guard_exit_code'] = guard_process.returncode
                    except subprocess.TimeoutExpired:
                        state['guard_exit_code'] = None
                if sampler_process is not None:
                    try:
                        sampler_process.wait(timeout=15)
                        state['sampler_exit_code'] = sampler_process.returncode
                    except subprocess.TimeoutExpired:
                        state['sampler_exit_code'] = None
                if state.get('sample'):
                    state['sampler_stopped'] = not same_process(state['sample'], proc_identity(state['sample']['pid']))
                if state.get('guard'):
                    state['guard_stopped'] = not same_process(state['guard'], proc_identity(state['guard']['pid']))
                write_report(data, state, verify_tb=True)
        return 0 if state['state'] in SUCCESS_STATES else 1


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
        elif state.get('experiment_mode') == 'target' and time.time_ns() >= DEADLINE_NS:
            reason = DEADLINE_REASON  # Deadline enforcement does not depend on SQLite readability.
        else:
            try:
                reason = watchdog_reason(collect_report(data, state), state)
            except (sqlite3.Error, FileNotFoundError):
                # Do not interpret a brief SQLite writer-lock window as a training failure.
                pass
        if reason:
            request_stop(data, reason)
            # Every short wait rereads status: terminal runner cleanup wins.
            for _ in range(stop_grace_seconds(reason) // 5):
                time.sleep(5)
                current = load_status(data)
                if current.get('run_id') != run_id:
                    return 0
                if current.get('state') in TERMINAL and current.get('training_stopped'):
                    return 0
            state = load_status(data)
            if state.get('run_id') != run_id:
                return 0
            stop_training_keep_profiler(state, data)
            # Preserve the profiler's required report conversion after a
            # supervisor failure too. This is a CPU finalization grace, not a
            # deadline on healthy training.
            profiler_deadline = time.monotonic() + PROFILER_FINALIZATION_GRACE_SECONDS
            while time.monotonic() < profiler_deadline:
                current = load_status(data)
                if current.get('run_id') != run_id:
                    return 0
                if current.get('state') in TERMINAL and current.get('training_stopped'):
                    return 0
                if not any(profiler_process(item) for item in remember_owned(state)):
                    break
                time.sleep(3)
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
        assert watchdog_reason({}, state, 7202 * 10**9) == 'policy_no_first_successful_update_120_minutes'
        progress = {'training': {model: {'ranks': {'0': {'first_success_ns': 10**9, 'last_success_ns': 8000 * 10**9}}}
                                 for model in MODELS}}
        assert watchdog_reason(progress, state, 11500 * 10**9) is None
        assert watchdog_reason(progress, state, 11601 * 10**9) == 'policy_successful_update_stalled_60_minutes'
        # There is deliberately no overall time bound when progress continues.
        assert watchdog_reason(progress, state, 8001 * 10**9) is None
        target = target_state({'details': {'state': {'streak': 2, 'first_qualifying': {'completed_ns': 1},
                                                   'consecutive_qualifying': {'completed_ns': 2}}}}, None)
        assert target_times(target) == (1, 2)
        qualifying = {'rate': .85, 'sample_count': 1000, 'fresh_count': 10, 'policy_step': 10}
        completed_detail = {'global_step': 10, 'stop_reason': DEADLINE_REASON, 'until_deadline': True,
                            'stop_on_target': False, 'deadline_unix_seconds': DEADLINE_UNIX, 'target_reached': True,
                            'stop_policy': {'until_deadline': True, 'stop_on_target': False,
                                            'deadline_unix_seconds': DEADLINE_UNIX}}
        completion = {'training_stopped': True, 'experiment_mode': 'target',
                      'checkpoint_cpu_audit': {'valid': True}, 'report': {
            'last_complete_epoch_step': 10,
            'driver': {'experiment_completed': {'ts_ns': DEADLINE_NS + 1, 'details': completed_detail}},
            'target_reached_event': True,
            'target': {'streak': 2, 'should_stop': True, 'run_started_ns': 0,
                       'first_qualifying': dict(qualifying, completed_ns=100),
                       'consecutive_qualifying': dict(qualifying, completed_ns=200)},
            'placements': {'complete_expected_placement': True},
            'topology': {'valid': True},
            'serial': {'recent_complete_steps_valid': True,
                       'sample_identity': {'recent_complete_steps_covered': True, 'errors': []}},
            'nsight_capture': {'ended': {'details': {'capture_mode': 'epoch', 'policy_step': 5,
                                                    'reason': 'completed_epoch', 'coverage_complete': True}}},
            'mps': {'protected_server_identity_valid': True},
            'nsight': {'raw_reports': [{'name': 'training1.nsys-rep', 'bytes': 123}],
                       'valid_real_training_windows': 1,
                       'valid_learning_microstep_windows': 1,
                       'exports': {'training1.nsys-rep': {'state': 'complete', 'kernel_table_present': True}}},
            'replay': {replay: {model + '_balance': {'difference_including_inflight_and_sqlite_flush': 0}
                               for model in models}
                       for replay, models in (('policy', ('policy',)), ('wm', ('ctrl', 'reward')))},
            'checkpoint': {'covers_last_complete_epoch': True},
            'tensorboard': {'readable_verified': True, 'scalar_tags': [METRIC], 'latest_metric': {'value': .85},
                           'milestones': {tag: {'value': value, 'dtype': 'float64'} for tag, value in
                                          zip(MILESTONE_TAGS, (1e-7, 1e-7, 2e-7, 2e-7))}},
            'training': {model: {'rank_counts_match': True, 'global_batch_valid': True,
                                'ranks': {str(rank): {'skipped_steps': 0} for rank in range(4)}} for model in MODELS}}}
        assert not validate_completion(completion)
        from copy import deepcopy
        deadline_completion = deepcopy(completion)
        deadline_completion['report']['tensorboard']['latest_metric']['value'] = .70
        assert not validate_completion(deadline_completion)  # Historical achievement survives a later decline.
        deadline_completion['report']['target_reached_event'] = False
        deadline_completion['report']['target'] = {}
        deadline_completion['report']['tensorboard']['milestones'] = {}
        deadline_completion['report']['driver']['experiment_completed']['details']['target_reached'] = False
        assert not validate_completion(deadline_completion)  # Deadline completion does not fabricate 85%.
        deadline_completion['report']['driver']['experiment_completed']['ts_ns'] = DEADLINE_NS - 1
        assert validate_completion(deadline_completion)  # Neither 85% nor 30000 steps can finish early.
        deadline_completion['report']['driver']['experiment_completed']['ts_ns'] = DEADLINE_NS + 1
        deadline_completion['report']['driver']['experiment_completed']['details']['stop_reason'] = 'consecutive_eval_85'
        assert validate_completion(deadline_completion)
        preflight_completion = deepcopy(completion)
        preflight_completion['experiment_mode'] = 'preflight'
        preflight_completion['report'].update(last_complete_epoch_step=15, target_reached_event=False,
            serial={'recent_complete_steps_valid': True, 'preflight_all_15_steps_valid': True, 'preflight_cohorts_valid': True,
                    'sample_identity': {'recent_complete_steps_covered': True, 'errors': [], 'preflight_all_21_required_events_valid': True}})
        for model, steps in {'policy': 15, 'ctrl': 3, 'reward': 3}.items():
            preflight_completion['report']['training'][model]['global_updates'] = steps
        assert not validate_completion(preflight_completion)
        preflight_completion['report']['training']['ctrl']['global_updates'] = 2
        assert validate_completion(preflight_completion)
        completion['report']['nsight']['valid_learning_microstep_windows'] = 0
        assert validate_completion(completion)
        completion['report']['nsight']['valid_learning_microstep_windows'] = 1
        completion['report']['target']['consecutive_qualifying']['completed_ns'] = 100
        assert validate_completion(completion)
        completion['report']['target']['consecutive_qualifying']['completed_ns'] = 200
        completion['report']['tensorboard']['latest_metric']['value'] = .84
        assert not validate_completion(completion)
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
            env = build_environment(data, 'test12345678', 123, env_cards, 'preflight')
        assert env['CUDA_VISIBLE_DEVICES'].startswith('GPU-') and env['ACCERL_USE_MPS'] == '1'
        assert env['ACCERL_CAPTURE_ONLY'] == '1' and env['ACCERL_TENSORBOARD'] == '1'
        assert env['ACCERL_MEASURE_STOP_AFTER_EPOCHS'] == '15'
        with patch.object(os, 'getuid', return_value=1004, create=True), patch.dict(os.environ, {'ACCERL_MEASURE_STOP_AFTER_EPOCHS': '15'}):
            target_env = build_environment(data, 'test12345678', 123, env_cards, 'target')
        assert 'ACCERL_MEASURE_STOP_AFTER_EPOCHS' not in target_env
        assert target_env['ACCERL_SOFT_STOP_AT'] == '1792371600'
        assert target_env['ACCERL_STOP_ON_TARGET_85'] == '0' and target_env['ACCERL_RUN_UNTIL_DEADLINE'] == '1'
        assert env['ACCERL_STOP_ON_TARGET_85'] == '0' and env['ACCERL_RUN_UNTIL_DEADLINE'] == '0'
        assert 'ACCERL_SOFT_STOP_AT' not in env
        assert env['ACCERL_SPLIT_TRAIN_SERIAL'] == target_env['ACCERL_SPLIT_TRAIN_SERIAL'] == '1'
        assert env['ACCERL_NSIGHT_MODE'] == 'epoch' and env['ACCERL_NSIGHT_SECONDS'] == '600'
        assert 'preflight_passed' in TERMINAL
        assert 'completed_at_deadline' in TERMINAL and 'completed_at_deadline' in SUCCESS_STATES
        assert DEADLINE_UNIX == 1792371600 and DEADLINE_NS == 1792371600000000000
        assert stamp(DEADLINE_UNIX) == '2026-10-19T09:00:00+08:00'
        assert stop_grace_seconds(DEADLINE_REASON) == 900 and stop_grace_seconds('training_error') == 120
        assert PROFILER_FINALIZATION_GRACE_SECONDS >= 2 * 600 + 15
        assert watchdog_reason({}, {'experiment_mode': 'target', 'started_ns': DEADLINE_NS - 1}, DEADLINE_NS) == DEADLINE_REASON
        assert env['CUDA_MPS_PIPE_DIRECTORY'] == MPS_PIPE
        assert env['ACCERL_DATA_DIR'] == str(data / 'training')
        assert env['ACCERL_RAY_SPILL'] == str(data / 'training/ray_spill')
        assert env['ACCERL_RESOURCE_GOVERNOR'] == '0'
        assert env['ACCERL_NSIGHT_STEPS'] == '5'
        assert env['NSYS_NVTX_PROFILER_REGISTER_ONLY'] == '0'
        profiled = profile_command(build_command(ROOT), data)
        assert '--kill=none' in profiled and '--wait=all' in profiled and '--export=sqlite' in profiled
        assert '--capture-range-end=stop' in profiled and '--sample=none' in profiled
        assert '--trace=cuda-sw,nvtx' in profiled and '2026.5.1' in profiled[0]
        assert user_stop_reason('user_requested_stop') and not user_stop_reason('supervisor_error')
        assert not user_stop_reason(DEADLINE_REASON)
        request_stop(data, 'training_error')
        request_stop(data, DEADLINE_REASON)
        assert json.loads((data / 'training/stop.flag').read_text())['reason'] == 'training_error'
        progress['training']['reward']['ranks']['0']['last_success_ns'] = 1
        assert watchdog_reason(progress, state, 8001 * 10**9) == 'reward_successful_update_stalled_60_minutes'
        # Exactly six distinct actors on each physical GPU; a duplicate process
        # cannot masquerade as a split model owner.
        rows = []
        roles = ('policy_trainer', 'ctrl_trainer', 'reward_trainer', 'policy_infer', 'ctrl_infer', 'reward_infer')
        for rank, gpu in enumerate((4, 5, 6, 7)):
            for offset, role in enumerate(roles):
                rows.append({'actor': role, 'actor_id': str(rank), 'pid': 100 + rank * 6 + offset,
                             'details': {'role': role, 'physical_gpus': [{'physical_index': gpu, 'uuid': 'GPU-' + str(gpu)}],
                                         'physical_gpu_uuid': str(gpu), 'cpu_threads': 1}})
        with patch.object(safe, 'query_events', return_value=rows):
            assert placement_evidence(None, 'test')['complete_expected_placement']
            rows[-1]['pid'] = rows[0]['pid']
            assert not placement_evidence(None, 'test')['complete_expected_placement']
        # Evicted before first actual use remains unused at eviction even if a
        # reserved in-flight batch consumes it later.
        ledger = sqlite3.connect(':memory:'); ledger.row_factory = sqlite3.Row
        ledger.execute('''CREATE TABLE replay_samples(run_id,replay,added_ns,resident,policy_uses,
                          ctrl_uses,reward_uses,evicted_ns,eviction_reason,first_used_ns)''')
        ledger.execute("INSERT INTO replay_samples VALUES('test','policy',1,0,576,0,0,3,'capacity',4)")
        balances = replay_evidence(ledger, 'test', {'policy': {'global_updates': 1}})
        assert balances['policy']['unused_at_eviction'] == 1
        assert balances['policy']['policy_balance']['difference_including_inflight_and_sqlite_flush'] == 0
        ledger.close()
        states = [{'pid': 100 + index, 'training_role': kind, 'rank': rank,
                   'world_size': 4, 'distributed_world_size': 4,
                   'gpu': {'physical_gpu': str(4 + rank)}, 'group_port': 30000 + offset,
                   'engines': {kind: {}}}
                  for index, (offset, kind, rank) in enumerate((offset, kind, rank)
                      for offset, kind in enumerate(MODELS) for rank in range(4))]
        assert topology_evidence({'details': {'states': states}})['valid']
        states[-1]['group_port'] = 30000
        assert not topology_evidence({'details': {'states': states}})['valid']
        assert profiler_process({'comm': 'nsys', 'command': '/usr/local/cuda/bin/nsys profile bash train.sh'})
        assert profiler_process({'comm': 'QdstrmImporter', 'command': '/opt/nvidia/QdstrmImporter input.qdstrm'})
        assert not profiler_process({'comm': 'python', 'command': 'python train.py --exp-name=nsys'})
        assert not profiler_process({'comm': 'exporter', 'command': '/tmp/unrelated/exporter input'})
        own_driver = {'uid': 1004, 'run_id': 'driver-test', 'birth_ticks': 100,
                      'pid': 999, 'command': '/opt/env/bin/python rl/ctrl_world/ds_wm_discrete_ctrl_train_wm.py'}
        driver_state = {'uid': 1004, 'run_id': 'driver-test', 'minimum_child_birth_ticks': 90,
                        'started_ns': 0}
        assert training_driver_liveness(driver_state, ROOT, [own_driver], now_ns=10**9) is None
        assert driver_state['training_driver'] == own_driver
        with patch.dict(globals(), {'proc_identity': lambda _pid: None}):
            assert training_driver_liveness(driver_state, ROOT, [], now_ns=2 * 10**9) is None
            assert training_driver_liveness(driver_state, ROOT, [], now_ns=13 * 10**9) == 'python_training_driver_exited_without_completion'
        # Nsight's gpuId is not assumed to be the physical index. UUIDs map
        # captured gpuId 20..23 to approved physical cards 4..7. Zero overlap is
        # a valid measured result when all three real processes have kernels.
        analysis_state = {'run_id': 'test', 'preflight': {'cards': [
            {'index': index, 'uuid': 'GPU-uuid' + str(index)} for index in range(4, 8)]}}
        analyzed = {'status': 'analyzed', 'role_map': {'run_id': 'test'},
                    'device_mapping': {'unresolved_groups': []}, 'per_gpu': {
            str(index + 16): {'physical_gpu_identity_resolved': True,
                             'gpu_info': {'uuid': 'GPU-uuid' + str(index)},
                             'training': {'all_kernel': {'per_model_union_ns': {
                                 role: 10 for role in ('policy_trainer', 'ctrl_trainer', 'reward_trainer')},
                                 'at_least_two_fraction_of_window': 0}},
                             'counts_by_role_and_category': {role + ':non_nccl_named_kernel': 1 for role in
                                 ('policy_trainer', 'ctrl_trainer', 'reward_trainer')}}
            for index in range(4, 8)}}
        assert summarize_nsight_analysis(analyzed, analysis_state, 'test.json')['valid_real_training_window']
        assert not summarize_nsight_analysis(analyzed, analysis_state, 'test.json')['valid_learning_microstep_window']
        analyzed['status'] = 'incomplete_row_budget'
        assert not summarize_nsight_analysis(analyzed, analysis_state, 'test.json')['valid_real_training_window']
        # Full fifteen-epoch stage records plus three globally shared WM draws.
        phases = sqlite3.connect(':memory:'); phases.row_factory = sqlite3.Row
        phases.execute('CREATE TABLE events(id INTEGER PRIMARY KEY,run_id,workload,ts_ns,kind,actor_id,pid,details_json)')
        for step in range(1, 16):
            for model_index, model in enumerate(MODELS):
                for rank in range(4):
                    detail = {'policy_step': step, 'model': model, 'rank': rank,
                              'allocated_bytes': 100, 'reserved_bytes': 200, 'peak_allocated_bytes': 300,
                              'peak_reserved_bytes': 400, 'free_bytes': 500,
                              'gpu_batch_refs_released': True, 'cpu_prefetch_only': True,
                              'completion': 'cuda_sync_and_empty_cache'}
                    for edge, extra in (('begin', 0), ('end', 500_000)):
                        phases.execute('INSERT INTO events(run_id,ts_ns,kind,actor_id,pid,details_json) VALUES(?,?,?,?,?,?)',
                            ('test', step * 10**9 + model_index * 10**6 + rank * 100 + extra,
                             'split_serial_phase_' + edge, str(rank), 100 + model_index * 4 + rank, json.dumps(detail)))
                required = model == 'policy' or step % 5 == 0
                phases.execute('INSERT INTO events(run_id,ts_ns,kind,actor_id,pid,details_json) VALUES(?,?,?,?,?,?)',
                    ('test', step * 10**9 + model_index, 'split_sample_identity_validated', 'driver', 999,
                     json.dumps({'policy_step': step, 'model': model, 'required': required,
                                 'actual_sample_ids_verified': required, 'rank_shards_disjoint': required,
                                 'global_samples': MODELS[model][1] if required else 0})))
            if step % 5 == 0:
                consumers = [{'model': model, 'rank': rank, 'reservation_id': 'wm' + str(step),
                              'shard_start': rank * 256, 'shard_end': (rank + 1) * 256,
                              'sample_ids_sha256': f'{step}-{rank}'}
                             for model in ('ctrl', 'reward') for rank in range(4)]
                phases.execute('INSERT INTO events(run_id,ts_ns,kind,actor_id,pid,details_json) VALUES(?,?,?,?,?,?)',
                    ('test', step * 10**9, 'split_wm_cohort_validated', 'driver', 999,
                     json.dumps({'policy_step': step, 'expected_consumers': 8, 'consumers': consumers,
                                 'actual_sample_ids_verified': True, 'rank_shards_disjoint': True,
                                 'world_reward_ordered_ids_identical': True})))
        serial = serial_evidence(phases, 'test')
        assert serial['preflight_all_15_steps_valid'] and serial['preflight_cohorts_valid']
        assert serial['sample_identity']['preflight_all_21_required_events_valid']
        bad_identity = {'policy_step': 15, 'model': 'policy', 'required': True,
                        'actual_sample_ids_verified': False, 'rank_shards_disjoint': True, 'global_samples': 576}
        phases.execute("UPDATE events SET details_json=? WHERE kind='split_sample_identity_validated' AND ts_ns=?",
                       (json.dumps(bad_identity), 15 * 10**9))
        assert not serial_evidence(phases, 'test')['sample_identity']['preflight_all_21_required_events_valid']
        # A duplicated consumer cannot stand in for a missing Reward rank.
        invalid_cohort = {'policy_step': 15, 'expected_consumers': 8, 'consumers': consumers[:-1] + [consumers[0]]}
        phases.execute("UPDATE events SET details_json=? WHERE kind='split_wm_cohort_validated' AND ts_ns=?",
                       (json.dumps(invalid_cohort), 15 * 10**9))
        assert not serial_evidence(phases, 'test')['preflight_cohorts_valid']
        bad_end = dict(detail, policy_step=15, model='reward', rank=3, gpu_batch_refs_released=False)
        phases.execute("UPDATE events SET details_json=? WHERE kind='split_serial_phase_end' AND pid=111 AND ts_ns>?",
                       (json.dumps(bad_end), 15 * 10**9))
        assert not serial_evidence(phases, 'test')['preflight_all_15_steps_valid']
        phases.execute("UPDATE events SET ts_ns=ts_ns+2000000 WHERE kind='split_serial_phase_end' AND pid=100 AND ts_ns>? AND ts_ns<?",
                       (14 * 10**9, 15 * 10**9))
        assert serial_evidence(phases, 'test')['phase_order_errors']
        phases.close()
        # The verifier performs exactly one CPU-only mmap read of final weights.
        checkpoint = data / 'training/checkpoints'; checkpoint.mkdir(parents=True)
        weights_path = checkpoint / 'latest_weights.pt'; weights_path.write_bytes(b'test')
        manifest = {'metadata': {'run_id': 'cpu-audit'}, 'world_size': 4, 'reason': 'final',
                    'weights_only': True, 'steps': {'policy': 15, 'ctrl': 3, 'reward': 3},
                    'tensor_counts': {model: 1 for model in MODELS}}
        def publish_manifest():
            (checkpoint / 'manifest.json').write_text(json.dumps(dict(manifest, weights_file='latest_weights.pt', weights_bytes=4)))
        publish_manifest()
        class FakeTensor:
            device = SimpleNamespace(type='cpu')
        payload = {'manifest': manifest, 'models': {model: {'state_dict': {'weight': FakeTensor()},
                     'trainable_names': ['weight'], 'full_state': model == 'reward'} for model in MODELS}}
        calls = []
        def fake_load(path, **options):
            calls.append(options)
            assert options == {'map_location': 'cpu', 'weights_only': True, 'mmap': True}
            return payload
        fake_torch = SimpleNamespace(load=fake_load, is_tensor=lambda value: isinstance(value, FakeTensor),
                                     cuda=SimpleNamespace(is_initialized=lambda: False))
        audit_state = {'run_id': 'cpu-audit', 'experiment_mode': 'preflight'}
        assert audit_final_weights(data, audit_state, torch_module=fake_torch, persist=False)['valid']
        assert len(calls) == 1
        manifest['steps']['ctrl'] = 2; publish_manifest()
        assert not audit_final_weights(data, audit_state, torch_module=fake_torch, persist=False)['valid']
    print('CPU_SELF_TEST_OK: exact Beijing deadline; no early 85%/30000 completion; historical float64 milestones survive final decline; 900s deadline/120s error grace; serial phase/cohort/actual-ID audit; final CPU mmap weights; independent environment/guard/terminal states')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('run', 'guard', 'sample', 'report', 'self-test'), default='report')
    parser.add_argument('--data', type=Path)
    parser.add_argument('--experiment-mode', choices=('preflight', 'target'), default='target')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--run-id')
    args = parser.parse_args()
    args.data = args.data or (PREFLIGHT_DATA if args.experiment_mode == 'preflight' else TARGET_DATA)
    if args.mode == 'self-test':
        return self_test()
    if args.mode == 'run':
        return run(args.data, args.root, args.experiment_mode)
    if args.mode == 'guard':
        if not args.run_id:
            parser.error('--run-id is required for guard mode')
        return guard(args.data, args.run_id)
    if args.mode == 'sample':
        if not args.run_id:
            parser.error('--run-id is required for sample mode')
        return sample(args.data, args.run_id)
    state = load_status(args.data)
    if not state:
        parser.error('runnerstatus.json does not yet exist')
    write_report(args.data, state, persist_status=False, verify_tb=state['state'] in TERMINAL)
    print(json.dumps({'run_id': state['run_id'], 'state': state['state'], 'report': str(args.data / 'README.md')}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
