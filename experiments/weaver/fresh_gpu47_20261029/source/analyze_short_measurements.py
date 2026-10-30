"""CPU-only, snapshot-consistent service analysis; never imports the training runtime.

Input SQLite is mode=ro/query_only. Rows stream in bounded batches; exact duration
quantiles and interval unions use a separate temporary SQLite database on disk.
No request-ID joins, CUDA-event attribution, kernel overlap, or SM claims are made.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
import json
import math
from pathlib import Path
import sqlite3

TRAIN = {'policy_train': 'policy', 'ctrl_train': 'ctrl', 'reward_train': 'reward'}
MACRO = {name + '_update': model for name, model in TRAIN.items()}
INFER = {'policy_infer', 'ctrl_infer', 'reward_infer'}
SERVICE_WORKLOADS = tuple(TRAIN) + tuple(MACRO) + tuple(sorted(INFER))
GLOBAL_BATCH = {'policy': 576, 'ctrl': 1024, 'reward': 1024}
EVENT_COLUMNS = {'id', 'run_id', 'ts_ns', 'actor', 'actor_id', 'pid', 'kind', 'workload',
                 'start_ns', 'end_ns', 'actual_batch', 'duration_ns', 'status', 'details_json'}
SELECT_EVENT = 'id,ts_ns,actor,actor_id,pid,kind,workload,start_ns,end_ns,actual_batch,duration_ns,status,details_json'


def rows(connection, sql, arguments=()):
    cursor = connection.execute(sql, arguments)
    while True:
        batch = cursor.fetchmany(2048)
        if not batch:
            return
        yield from batch


def columns(connection, table):
    return {row[1] for row in connection.execute('PRAGMA table_info(' + table + ')')}


def details(row, diagnostics):
    try:
        value = json.loads(row['details_json'] or '{}')
        if not isinstance(value, dict):
            raise ValueError('details must be an object')
        return value
    except (ValueError, TypeError):
        diagnostics['malformed_details_json'] += 1
        return {}


def integer(value, minimum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if minimum is None or value >= minimum else None


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


class Mask:
    def __init__(self, intervals=()):
        self.intervals = merge_intervals(intervals)
        self.starts = [x[0] for x in self.intervals]

    def point(self, timestamp):
        index = bisect_right(self.starts, timestamp) - 1
        return index >= 0 and timestamp < self.intervals[index][1]

    def overlap(self, start, end):
        index = max(0, bisect_right(self.starts, start) - 1)
        while index < len(self.intervals) and self.intervals[index][0] < end:
            if self.intervals[index][1] > start:
                return True
            index += 1
        return False

    def unmasked_length(self, start, end):
        if end <= start:
            return 0
        removed = sum(max(0, min(end, right) - max(start, left))
                      for left, right in self.intervals if right > start and left < end)
        return end - start - removed


def profile_intervals(connection, run_id, diagnostics):
    started, ended, windows = {}, {}, []
    for row in rows(connection, 'SELECT ' + SELECT_EVENT + ' FROM events WHERE run_id=? '
                    "AND kind IN ('nsight_capture_started','nsight_capture_ended') ORDER BY id", (run_id,)):
        data = details(row, diagnostics)
        start = integer(data.get('started_ns'), 0)
        if start is None:
            diagnostics['profile_missing_exact_start'] += 1
            continue
        key = (row['actor'], row['actor_id'], row['pid'], start)
        target = started if row['kind'] == 'nsight_capture_started' else ended
        if key in target:
            diagnostics['profile_duplicate_boundary'] += 1
        target[key] = (row, data)
    for key in sorted(set(started) & set(ended), key=lambda k: k[-1]):
        begin_row, begin = started[key]
        end_row, end = ended[key]
        start = key[-1]
        returned = integer(end.get('nvtx_end_returned_ns', end.get('completed_ns')), 0)
        close = integer(end.get('close_requested_ns'), 0)
        if returned is None or returned < start or (close is not None and not start <= close <= returned):
            diagnostics['profile_invalid_exact_interval'] += 1
            continue
        windows.append({'actor': key[0], 'actor_id': key[1], 'pid': key[2],
                        'started_event_id': begin_row['id'], 'ended_event_id': end_row['id'],
                        'start_ns': start, 'range_close_requested_ns': close,
                        'nvtx_end_returned_ns': returned,
                        'capture_mode': end.get('capture_mode', begin.get('capture_mode')),
                        'policy_step': end.get('policy_step', begin.get('policy_step')),
                        'range_ns': [start, close if close is not None else returned],
                        'synchronous_export_ns': [close, returned] if close is not None else None,
                        'mask_ns': [start, returned],
                        'range_export_split_known': close is not None})
    unmatched = len(set(started) ^ set(ended))
    diagnostics['profile_unmatched_boundaries'] += unmatched
    invalid = diagnostics['profile_missing_exact_start'] + diagnostics['profile_invalid_exact_interval']
    return windows, bool(unmatched or invalid or diagnostics['profile_duplicate_boundary'])


def measurement_window(connection, run_id):
    bounds = connection.execute('SELECT MIN(ts_ns),MAX(ts_ns),MIN(start_ns),MAX(end_ns),COUNT(*) '
                                'FROM events WHERE run_id=?', (run_id,)).fetchone()
    if not bounds[4]:
        raise ValueError('No events for requested run_id')
    event_start = min(x for x in (bounds[0], bounds[2]) if x is not None)
    event_end = max(x for x in (bounds[1], bounds[3]) if x is not None)
    run = None
    if {'run_id', 'started_ns', 'finished_ns'} <= columns(connection, 'runs'):
        run = connection.execute('SELECT started_ns,finished_ns FROM runs WHERE run_id=?', (run_id,)).fetchone()
    start = min(event_start, run[0]) if run and run[0] is not None else event_start
    end = max(event_end, run[1]) if run and run[1] is not None else event_end
    return {'start_ns': start, 'end_ns': end, 'wall_seconds': (end - start) / 1e9,
            'source': 'envelope of runs timestamps and recorded event/service endpoints',
            'runs_started_ns': run[0] if run else None, 'runs_finished_ns': run[1] if run else None,
            'snapshot_event_count': bounds[4],
            'limitation': 'Includes recorded initialization/prefill/idle/finalization. Not an outer supervisor clock or a fixed checkpoint benchmark.'}


def role_for(row, data):
    model = TRAIN.get(row['workload']) or MACRO.get(row['workload'])
    if model:
        return str(data.get('training_role') or model) + '_trainer'
    return row['workload']


class Scratch:
    """Own temporary on-disk store, separate from and never attached to input DB."""
    def __init__(self):
        self.connection = sqlite3.connect('')
        self.connection.execute('CREATE TABLE observations(g INTEGER,ok INTEGER,duration INTEGER,start INTEGER,end INTEGER,batch INTEGER)')
        self.groups, self.buffer = {}, []

    def group(self, key, identity):
        if key not in self.groups:
            self.groups[key] = dict(identity, group_id=len(self.groups), records=0, errors=0,
                                    invalid_interval=0, invalid_duration=0, invalid_batch=0,
                                    excluded_profile_overlap=0, statuses=Counter(), ranks=Counter())
        return self.groups[key]

    def add(self, group, row, data, mask):
        group['records'] += 1
        group['statuses'][str(row['status'])] += 1
        rank = integer(data.get('data_parallel_rank'), 0)
        if rank is not None:
            group['ranks'][rank] += 1
        start, end = integer(row['start_ns'], 0), integer(row['end_ns'], 0)
        if start is None or end is None or end < start:
            group['invalid_interval'] += 1
            return
        if mask.overlap(start, end):
            group['excluded_profile_overlap'] += 1
            return
        duration = integer(row['duration_ns'], 0)
        if duration is None:
            group['invalid_duration'] += 1
            return
        batch = integer(row['actual_batch'], 0)
        if batch is None:
            group['invalid_batch'] += 1
            return
        ok = row['status'] == 'ok'
        if not ok:
            group['errors'] += 1
        self.buffer.append((group['group_id'], int(ok), duration, start, end, batch))
        if len(self.buffer) >= 2048:
            self.flush()

    def flush(self):
        if self.buffer:
            self.connection.executemany('INSERT INTO observations VALUES(?,?,?,?,?,?)', self.buffer)
            self.buffer.clear()

    def quantiles(self, gid, only_success=False):
        where = 'g=?' + (' AND ok=1' if only_success else '')
        n, minimum, maximum = self.connection.execute('SELECT COUNT(*),MIN(duration),MAX(duration) FROM observations WHERE ' + where, (gid,)).fetchone()
        if not n:
            return {'count': 0, 'min_ms': None, 'median_ms': None, 'p95_ms': None, 'max_ms': None}

        def percentile(fraction):
            position = (n - 1) * fraction
            lower, upper = math.floor(position), math.ceil(position)
            lo = self.connection.execute('SELECT duration FROM observations WHERE ' + where + ' ORDER BY duration LIMIT 1 OFFSET ?', (gid, lower)).fetchone()[0]
            hi = self.connection.execute('SELECT duration FROM observations WHERE ' + where + ' ORDER BY duration LIMIT 1 OFFSET ?', (gid, upper)).fetchone()[0]
            return (lo + (hi - lo) * (position - lower)) / 1e6
        return {'count': n, 'min_ms': minimum / 1e6, 'median_ms': percentile(.5),
                'p95_ms': percentile(.95), 'max_ms': maximum / 1e6}

    def summarize(self, wall_seconds):
        self.flush()
        self.connection.execute('CREATE INDEX observations_group_duration ON observations(g,ok,duration)')
        self.connection.execute('CREATE INDEX observations_group_interval ON observations(g,ok,start,end)')
        summaries = []
        for group in self.groups.values():
            gid = group['group_id']
            successful, units = self.connection.execute('SELECT COUNT(*),COALESCE(SUM(batch),0) FROM observations WHERE g=? AND ok=1', (gid,)).fetchone()
            batch_distribution = {str(batch): count for batch, count in self.connection.execute(
                'SELECT batch,COUNT(*) FROM observations WHERE g=? AND ok=1 GROUP BY batch ORDER BY batch', (gid,))}
            active, left, right = 0, None, None
            for start, end in rows(self.connection, 'SELECT start,end FROM observations WHERE g=? AND ok=1 ORDER BY start,end', (gid,)):
                if left is None:
                    left, right = start, end
                elif start <= right:
                    right = max(right, end)
                else:
                    active += right - left
                    left, right = start, end
            if left is not None:
                active += right - left
            unit = group['unit']
            result = {k: v for k, v in group.items() if k != 'group_id'}
            result.update(ranks={str(k): v for k, v in sorted(group['ranks'].items())},
                          statuses=dict(group['statuses']), eligible_successful_records=successful,
                          duration=self.quantiles(gid), successful_duration=self.quantiles(gid, True),
                          actual_batch_distribution_successful=batch_distribution,
                          successful_batch_units_sum=units, active_service_union_seconds=active / 1e9,
                          units_per_active_service_second=units / (active / 1e9) if active else None,
                          units_per_measurement_wall_second=units / wall_seconds if wall_seconds > 0 else None,
                          numerator_semantics=('actual sample exposures' if unit == 'train_micro' else
                                               'successfully served requests' if unit == 'inference_batch' else
                                               'local update-span batch sizes; NOT additional exposures'),
                          timing_semantics='CPU/perf service duration; wall interval includes preparation/enqueue/output-copy or synchronization. Not pure GPU kernel time.')
            summaries.append(result)
        return sorted(summaries, key=lambda g: (g['scope'], g['workload'], str(g['actor_id']), str(g['pid'])))


def analyze_services(connection, scratch, run_id, mask, wall_seconds, diagnostics):
    query = 'SELECT ' + SELECT_EVENT + ' FROM events WHERE run_id=? AND kind=\'service\' AND workload IN (' + ','.join('?' for _ in SERVICE_WORKLOADS) + ')'
    for row in rows(connection, query, (run_id,) + SERVICE_WORKLOADS):
        data = details(row, diagnostics)
        workload = row['workload']
        unit = 'train_micro' if workload in TRAIN else 'train_macro' if workload in MACRO else 'inference_batch'
        role = role_for(row, data)
        for scope, actor_id, pid in [('process', row['actor_id'], row['pid']), ('workload_all_processes', None, None)]:
            key = (scope, role, actor_id, pid, workload, unit)
            group = scratch.group(key, {'scope': scope, 'role': role, 'actor_id': actor_id, 'pid': pid,
                                        'actor': row['actor'] if scope == 'process' else None,
                                        'workload': workload, 'unit': unit})
            scratch.add(group, row, data, mask)
        if workload in INFER and isinstance(data.get('request_ids'), list) and len(data['request_ids']) != row['actual_batch']:
            diagnostics['inference_request_id_count_mismatch'] += 1
    return scratch.summarize(wall_seconds)


def analyze_optimizer(connection, run_id, mask, wall_seconds, diagnostics):
    models = {}
    for model in TRAIN.values():
        models[model] = {'global_successful_updates': 0, 'global_skipped_updates': 0,
                         'global_count_source': 'optimizer_update with explicit data_parallel_rank=0 only',
                         'rank_events': {}, 'global_batch_distribution': Counter(),
                         'excluded_profile_points': 0, 'invalid_global_rank_events': 0,
                         'missing_explicit_success_counter': 0}
    for row in rows(connection, 'SELECT ' + SELECT_EVENT + ' FROM events WHERE run_id=? AND kind IN '
                    "('optimizer_update','optimizer_update_replica') AND workload IN ('policy_train','ctrl_train','reward_train')", (run_id,)):
        model = models[TRAIN[row['workload']]]
        if mask.point(row['ts_ns']):
            model['excluded_profile_points'] += 1
            continue
        data = details(row, diagnostics)
        rank = integer(data.get('data_parallel_rank'), 0)
        success = integer(data.get('successful_optimizer_steps'), 0)
        skipped = integer(data.get('skipped_steps'), 0)
        if success is None:
            model['missing_explicit_success_counter'] += 1
            continue
        if rank is not None:
            key = (rank, row['actor_id'], row['pid'])
            rank_record = model['rank_events'].setdefault(key, {'rank': rank, 'actor_id': row['actor_id'],
                       'pid': row['pid'], 'successful_updates': 0, 'skipped_updates': 0, 'event_count': 0})
            rank_record['successful_updates'] += success
            rank_record['skipped_updates'] += skipped or 0
            rank_record['event_count'] += 1
        if row['kind'] == 'optimizer_update':
            if rank != 0:
                model['invalid_global_rank_events'] += 1
                continue
            model['global_successful_updates'] += success
            model['global_skipped_updates'] += skipped or 0
            model['global_batch_distribution'][str(row['actual_batch'])] += 1
    for model in models.values():
        model['rank_events'] = sorted(model['rank_events'].values(), key=lambda x: (x['rank'], x['pid']))
        model['global_batch_distribution'] = dict(model['global_batch_distribution'])
        model['global_updates_per_measurement_wall_second'] = model['global_successful_updates'] / wall_seconds if wall_seconds > 0 else None
        ranks = {record['rank'] for record in model['rank_events']}
        model['four_ranks_observed'] = ranks == {0, 1, 2, 3}
        model['exclusion_semantics'] = 'Update boundary timestamp is masked; this does not prove the entire accumulated update was free of profiling.'
    return models


def analyze_episode_production(connection, run_id, mask, wall_seconds, diagnostics):
    counts = {'real_rollout_postprocessing_completed': 0, 'evaluation_completed': 0}
    successes = 0
    for row in rows(connection, 'SELECT ' + SELECT_EVENT + ' FROM events WHERE run_id=? AND '
                    "(kind='eval_episode_completed' OR (kind='service' AND actor='rollout' AND workload='cpu_wm_windows'))", (run_id,)):
        when = row['end_ns'] if row['kind'] == 'service' else row['ts_ns']
        if when is None or mask.point(when):
            continue
        if row['kind'] == 'eval_episode_completed':
            data = details(row, diagnostics)
            counts['evaluation_completed'] += 1
            successes += int(data.get('success') == 1)
        elif row['status'] == 'ok':
            counts['real_rollout_postprocessing_completed'] += 1
    result = {name: {'count': count, 'per_measurement_wall_hour': count * 3600 / wall_seconds if wall_seconds > 0 else None}
              for name, count in counts.items()}
    result['real_rollout_postprocessing_completed']['semantics'] = 'Known rollout _process_episode_for_wm service, once after a completed real episode; a postprocessing-completion proxy, not the exact environment termination timestamp. Zero may mean missing instrumentation.'
    result['evaluation_completed'].update(success_count=successes, semantics='Explicit eval_episode_completed events; not replay data, imagined rollouts, or 1000-window snapshots.')
    result['wm_episode_ingestion'] = {'status': 'unavailable'}
    required = {'run_id', 'replay', 'episode_id', 'added_ns'}
    if required <= columns(connection, 'replay_samples'):
        ingestion = {}
        for replay, episode_id, added in rows(connection, "SELECT replay,episode_id,MIN(added_ns) FROM replay_samples WHERE run_id=? AND replay IN ('wm','wm_eval') AND episode_id IS NOT NULL GROUP BY replay,episode_id", (run_id,)):
            if added is not None and not mask.point(added):
                ingestion[replay] = ingestion.get(replay, 0) + 1
        result['wm_episode_ingestion'] = {
            'status': 'available', 'counts_by_replay': ingestion,
            'per_measurement_wall_hour_by_replay': {key: count * 3600 / wall_seconds if wall_seconds > 0 else None for key, count in ingestion.items()},
            'semantics': 'Distinct real source episode IDs first added to wm training or wm_eval holdout replay. Ingestion is separate from rollout completion, eval episodes and imaginary sample production.'}
    return result


def analyze_prefetch(connection, run_id, mask, window, diagnostics):
    """Only consume an explicit schema; never infer queue occupancy from exposure counts."""
    fields = ('ready_batches', 'pending_fetches', 'in_use_batches', 'empty_waiters')
    owners = {}
    for row in rows(connection, 'SELECT ' + SELECT_EVENT + ' FROM events WHERE run_id=? '
                    "AND kind='cpu_replay_prefetch_state' ORDER BY ts_ns,id", (run_id,)):
        data = details(row, diagnostics)
        model, rank = data.get('model'), integer(data.get('rank'), 0)
        if model not in {'policy', 'ctrl', 'reward'} or rank is None:
            diagnostics['prefetch_invalid_owner_identity'] += 1
            continue
        key = (model, rank, row['actor_id'], row['pid'])
        owner = owners.setdefault(key, {'model': model, 'rank': rank, 'actor_id': row['actor_id'],
                  'pid': row['pid'], 'events': 0, 'invalid_state_events': 0, 'last': None,
                  'first_event_ns': row['ts_ns'], 'last_event_ns': row['ts_ns'],
                  'coverage_ns': 0, 'time_ns': Counter(), 'transitions': Counter(),
                  'empty_wait_ns_first': None, 'empty_wait_ns_last': None,
                  'observer_errors_max': 0, 'failed_observed': False, 'closed_observed': False})
        previous = owner['last']
        if previous is not None and not previous['closed']:
            left, right = max(previous['ts'], window['start_ns']), min(row['ts_ns'], window['end_ns'])
            elapsed = mask.unmasked_length(left, right)
            owner['coverage_ns'] += elapsed
            for field in fields:
                owner['time_ns'][field] += elapsed * previous[field]
        owner['events'] += 1
        owner['last_event_ns'] = row['ts_ns']
        owner['transitions'][str(data.get('transition'))] += 1
        valid = data.get('capacity_batches') == 1 and all(integer(data.get(field), 0) in (0, 1) for field in fields)
        if not valid or not isinstance(data.get('closed'), bool):
            owner['invalid_state_events'] += 1
            owner['last'] = None
            continue
        state = {field: data[field] for field in fields}
        state.update(ts=row['ts_ns'], closed=data['closed'])
        owner['last'] = state
        owner['closed_observed'] |= data['closed']
        owner['failed_observed'] |= data.get('failed') is True
        owner['observer_errors_max'] = max(owner['observer_errors_max'], integer(data.get('observer_errors'), 0) or 0)
        cumulative = integer(data.get('empty_wait_ns'), 0)
        if cumulative is not None:
            if owner['empty_wait_ns_first'] is None:
                owner['empty_wait_ns_first'] = cumulative
            if owner['empty_wait_ns_last'] is not None and cumulative < owner['empty_wait_ns_last']:
                diagnostics['prefetch_empty_wait_counter_regressed'] += 1
            owner['empty_wait_ns_last'] = cumulative
    summaries = []
    for owner in owners.values():
        previous = owner['last']
        if previous is not None and not previous['closed']:
            elapsed = mask.unmasked_length(max(previous['ts'], window['start_ns']), window['end_ns'])
            owner['coverage_ns'] += elapsed
            for field in fields:
                owner['time_ns'][field] += elapsed * previous[field]
        coverage = owner['coverage_ns']
        result = {key: value for key, value in owner.items()
                  if key not in {'last', 'time_ns', 'coverage_ns', 'transitions'}}
        result.update(coverage_seconds=coverage / 1e9,
                      state_seconds={field: owner['time_ns'][field] / 1e9 for field in fields},
                      state_fraction_of_observed_lifecycle={field: owner['time_ns'][field] / coverage if coverage else None for field in fields},
                      fraction_of_measurement_wall_covered=coverage / (window['effective_wall_seconds'] * 1e9) if window['effective_wall_seconds'] > 0 else None,
                      transitions=dict(owner['transitions']), right_censored=not owner['closed_observed'])
        first, last = owner['empty_wait_ns_first'], owner['empty_wait_ns_last']
        result['cumulative_empty_wait_delta_ns_unmasked_run_scope'] = last - first if first is not None and last is not None and last >= first else None
        summaries.append(result)
    resident = []
    if {'run_id', 'ts_ns', 'replay', 'actor_id', 'resident_count', 'capacity'} <= columns(connection, 'replay_snapshots'):
        summaries_by_replay = {}
        for timestamp, replay, actor_id, count, capacity in rows(connection,
                'SELECT ts_ns,replay,actor_id,resident_count,capacity FROM replay_snapshots WHERE run_id=? ORDER BY ts_ns,id', (run_id,)):
            if mask.point(timestamp):
                continue
            item = summaries_by_replay.setdefault((replay, actor_id), {'replay': replay, 'actor_id': actor_id,
                                      'snapshots': 0, 'minimum_resident': count, 'maximum_resident': count})
            item['snapshots'] += 1
            item['minimum_resident'] = min(item['minimum_resident'], count)
            item['maximum_resident'] = max(item['maximum_resident'], count)
            item.update(last_resident=count, last_capacity=capacity, last_snapshot_ns=timestamp)
        resident = list(summaries_by_replay.values())
    return {'status': 'available' if owners else 'not_collected',
            'event_kind': 'cpu_replay_prefetch_state',
            'owners': sorted(summaries, key=lambda item: (item['model'], item['rank'], item['pid'])),
            'resident_snapshots': resident,
            'resident_snapshot_semantics': 'Recorded Replay resident item snapshots only; no exact resident time-occupancy is inferred.',
            'semantics': [
                'State transition timestamps are Telemetry wallclock observations, carried forward until the next event or explicit close. Missing time before the first event is not filled with zero.',
                'Ready is one raw CPU batch slot; Pending is an awaited replay/coordinator fetch, not proof of missing resident Replay data.',
                'Ready, pending fetch, in-use and empty-wait indicators can overlap; their fractions are not mutually exclusive and must not be summed to 100%.',
                'In-use lasts from take through trainer-phase release and is not actual microstep consumption. Training exposures come only from successful micro service actual_batch.',
                'No periodic ticks occur while synchronous training blocks the event loop; the last explicit state remains in force. A final unclosed state is marked right-censored.',
                'Profile masking subtracts exact wall time from state intervals, using all transition events to retain boundary state. Cumulative monotonic empty_wait_ns deltas are separately reported for the unmasked-run scope and are never prorated.']}


def analyze(db, run_id, exclude_profiled=False):
    db = Path(db).resolve()
    connection = sqlite3.connect(db.as_uri() + '?mode=ro', uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    scratch = Scratch()
    diagnostics = Counter()
    try:
        connection.execute('PRAGMA query_only=ON')
        connection.execute('BEGIN')
        missing = EVENT_COLUMNS - columns(connection, 'events')
        if missing:
            raise ValueError('Missing events columns: ' + ','.join(sorted(missing)))
        window = measurement_window(connection, run_id)
        profile, incomplete = profile_intervals(connection, run_id, diagnostics)
        if exclude_profiled and incomplete:
            raise ValueError('Cannot exclude profiling: exact started/ended boundaries are missing, duplicate or invalid')
        mask = Mask([row['mask_ns'] for row in profile] if exclude_profiled else [])
        effective = mask.unmasked_length(window['start_ns'], window['end_ns']) / 1e9
        window.update(profile_exclusion=bool(exclude_profiled), effective_wall_seconds=effective,
                      removed_profile_and_export_wall_seconds=window['wall_seconds'] - effective)
        service = analyze_services(connection, scratch, run_id, mask, effective, diagnostics)
        optimizer = analyze_optimizer(connection, run_id, mask, effective, diagnostics)
        exposure = {}
        for group in service:
            if group['scope'] == 'workload_all_processes' and group['unit'] == 'train_micro':
                model = TRAIN[group['workload']]
                exposure[model] = {'actual_exposures_all_rank_micro_batches': group['successful_batch_units_sum'],
                                   'observed_data_parallel_ranks': sorted(int(rank) for rank in group['ranks']),
                                   'global_batch_expected': GLOBAL_BATCH[model],
                                   'macro_and_microstep_events_not_added': True,
                                   'semantics': 'Sum of actual_batch for status=ok micro service records across recorded ranks. Does not mean unique replay items or count optimizer updates.'}
        return {
            'schema_version': 1, 'run_id': run_id, 'db': str(db),
            'input_access': {'mode': 'ro', 'query_only': True, 'single_read_snapshot': True,
                             'stream_fetch_rows': 2048, 'request_id_joins': False,
                             'quantiles': 'Exact linear interpolation at (n-1)*q, using separate temporary SQLite storage'},
            'measurement_window': window, 'profile_windows': profile,
            'profile_filter': {'requested': exclude_profiled, 'exact_complete': not incomplete,
                              'merged_mask_ns': mask.intervals,
                              'service_policy': 'Discard the whole service if its wall interval overlaps [start, NVTX end returned); never prorate batch samples or durations.',
                              'point_policy': 'Optimizer/episode points use half-open masks; an update point outside the mask may still accumulate profiled microsteps.',
                              'contamination': ['Mask includes the recorded NVTX range and synchronous range-close/export wait.',
                                                'Outer profiler/controller overhead, initialization, export after these recorded boundaries and perturbation of online trajectories remain unquantified.',
                                                'Excluding capture intervals does not establish an unprofiled causal baseline.']},
            'services': service, 'training_exposures': exposure, 'optimizer_updates': optimizer,
            'episode_production': analyze_episode_production(connection, run_id, mask, effective, diagnostics),
            'prefetch': analyze_prefetch(connection, run_id, mask, window, diagnostics),
            'diagnostics': dict(diagnostics),
            'limitations': ['Service and active-union times include CPU preparation, enqueue gaps, output copies and selected synchronizations; neither is pure GPU kernel time.',
                            'All-process active throughput divides by the union of eligible service wall intervals, rather than the sum of parallel rank seconds.',
                            'Only exact service workloads are counted; client RPC, batch_dispatch, microstep and macro-update spans are not additional inference requests or training exposures.',
                            'No SM Active, occupancy or actual kernel overlap is derived from this SQLite analysis; those require independently verified hardware/Nsight evidence.',
                            'Results describe this snapshot and measurement window. Missing events, invalid spans and profiling-crossing batches are disclosed; they are not repaired.']}
    finally:
        connection.rollback()
        connection.close()
        scratch.connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exclude-profiled', action='store_true')
    args = parser.parse_args()
    if args.db.resolve() == args.output.resolve():
        parser.error('Output must differ from input SQLite database')
    result = analyze(args.db, args.run_id, args.exclude_profiled)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output.resolve()), 'run_id': args.run_id,
                      'service_groups': len(result['services']),
                      'effective_wall_seconds': result['measurement_window']['effective_wall_seconds']}))


if __name__ == '__main__':
    main()
