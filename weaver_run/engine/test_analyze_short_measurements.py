"""Schema fixtures exercise counting semantics, exact masking and time denominators."""
from pathlib import Path
import hashlib
import json
import sqlite3
import tempfile
import unittest

from analyze_short_measurements import analyze

NS = 1_000_000_000
RUN = 'fixture_run'
SCHEMA = '''
CREATE TABLE runs(run_id TEXT PRIMARY KEY,started_ns INTEGER,finished_ns INTEGER);
CREATE TABLE events(id INTEGER PRIMARY KEY AUTOINCREMENT,run_id TEXT,ts_ns INTEGER,
 actor TEXT,actor_id TEXT,pid INTEGER,kind TEXT,workload TEXT,start_ns INTEGER,end_ns INTEGER,
 actual_batch INTEGER,duration_ns INTEGER,status TEXT,details_json TEXT);
CREATE INDEX events_run_workload ON events(run_id,workload,kind,ts_ns);
CREATE TABLE replay_samples(run_id TEXT,replay TEXT,episode_id TEXT,added_ns INTEGER);
CREATE TABLE replay_snapshots(id INTEGER PRIMARY KEY AUTOINCREMENT,run_id TEXT,ts_ns INTEGER,
 replay TEXT,actor_id TEXT,resident_count INTEGER,capacity INTEGER);
'''


class AnalysisFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.db = Path(self.temporary.name) / 'fixture.sqlite'
        self.connection = sqlite3.connect(self.db)
        self.connection.executescript(SCHEMA)
        self.connection.execute('INSERT INTO runs VALUES(?,?,?)', (RUN, 0, 100 * NS))

    def tearDown(self):
        self.connection.close()
        self.temporary.cleanup()

    def event(self, kind, *, workload=None, timestamp=1, start=None, end=None,
              batch=None, duration=None, actor='trainer', actor_id='0', pid=100,
              status=None, data=None, run_id=RUN):
        self.connection.execute('''INSERT INTO events(run_id,ts_ns,actor,actor_id,pid,kind,workload,
         start_ns,end_ns,actual_batch,duration_ns,status,details_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)''',
         (run_id, int(timestamp * NS), actor, str(actor_id), pid, kind, workload,
          int(start * NS) if start is not None else None,
          int(end * NS) if end is not None else None, batch,
          int(duration * NS) if duration is not None else None, status,
          json.dumps(data or {})))

    def service(self, workload, start, end, batch, *, rank=0, status='ok', pid=None,
                duration=None, actor=None):
        actor = actor or ('trainer' if '_train' in workload else workload)
        data = {'training_role': workload.split('_')[0], 'data_parallel_rank': rank,
                'data_parallel_world_size': 4} if '_train' in workload else {}
        self.event('service', workload=workload, start=start, end=end, timestamp=end,
                   duration=end - start if duration is None else duration, batch=batch,
                   status=status, actor=actor, actor_id=rank, pid=pid or 100 + rank, data=data)

    def update(self, rank, timestamp=15, *, kind=None, success=1, skipped=0):
        self.event(kind or ('optimizer_update' if rank == 0 else 'optimizer_update_replica'),
                   workload='policy_train', timestamp=timestamp, batch=576, actor_id=rank, pid=100 + rank,
                   status='ok', data={'data_parallel_rank': rank, 'data_parallel_world_size': 4,
                                      'successful_optimizer_steps': success, 'skipped_steps': skipped,
                                      'logical_optimizer_steps': success if rank == 0 else 0})

    def profile(self, start=40, close=50, returned=55):
        self.event('nsight_capture_started', timestamp=start + 1, actor='driver', pid=1,
                   data={'started_ns': start * NS, 'policy_step': 10, 'capture_mode': 'timer'})
        self.event('nsight_capture_ended', timestamp=returned + 1, actor='driver', pid=1,
                   data={'started_ns': start * NS, 'close_requested_ns': close * NS,
                         'nvtx_end_returned_ns': returned * NS, 'completed_ns': returned * NS,
                         'policy_step': 10, 'capture_mode': 'timer'})

    def report(self, exclude=False):
        self.connection.commit()
        return analyze(self.db, RUN, exclude)

    def group(self, report, workload, scope='workload_all_processes', pid=None):
        matches = [g for g in report['services'] if g['workload'] == workload and g['scope'] == scope
                   and (pid is None or g['pid'] == pid)]
        self.assertEqual(len(matches), 1)
        return matches[0]

    def test_four_rank_micro_exposures_macro_and_global_updates_are_distinct(self):
        for rank in range(4):
            self.service('policy_train', 10, 12, 8, rank=rank)
            self.service('policy_train', 12, 14, 8, rank=rank)
            self.service('policy_train_update', 9, 15, 16, rank=rank)
            self.update(rank)
        self.service('policy_train', 14, 16, 8, status='error')
        self.event('microstep', workload='policy_train', batch=8000, timestamp=14)
        self.event('replay_use', workload='policy_train', batch=8000, timestamp=14)
        self.event('service', workload='policy_train_client', start=1, end=2, duration=1,
                   batch=9999, status='ok', timestamp=2)
        self.event('service', workload='policy_train', start=1, end=2, duration=1,
                   batch=9999, status='ok', timestamp=2, run_id='other_run')
        report = self.report()
        self.assertEqual(report['training_exposures']['policy']['actual_exposures_all_rank_micro_batches'], 64)
        micro = self.group(report, 'policy_train')
        macro = self.group(report, 'policy_train_update')
        self.assertEqual(micro['unit'], 'train_micro')
        self.assertEqual(macro['unit'], 'train_macro')
        self.assertEqual(micro['eligible_successful_records'], 8)
        self.assertEqual(micro['errors'], 1)
        self.assertEqual(micro['active_service_union_seconds'], 4)
        self.assertEqual(micro['units_per_active_service_second'], 16)
        self.assertEqual(micro['units_per_measurement_wall_second'], .64)
        self.assertEqual(macro['successful_batch_units_sum'], 64)
        optimizer = report['optimizer_updates']['policy']
        self.assertEqual(optimizer['global_successful_updates'], 1)
        self.assertEqual(sum(x['successful_updates'] for x in optimizer['rank_events']), 4)
        self.assertTrue(optimizer['four_ranks_observed'])

    def test_inference_success_requests_use_batch_size_and_union_active_time(self):
        self.service('policy_infer', 20, 22, 2, pid=200)
        self.service('policy_infer', 21, 23, 4, pid=200)
        self.service('policy_infer', 23, 24, 9, pid=200, status='error')
        self.service('policy_infer', 30, 31, 1, pid=201)
        self.event('batch_dispatch', workload='policy_infer', batch=1000, timestamp=20)
        self.event('request_reply', workload='policy_infer', batch=1000, timestamp=20)
        report = self.report()
        process = self.group(report, 'policy_infer', 'process', 200)
        total = self.group(report, 'policy_infer')
        self.assertEqual(process['successful_batch_units_sum'], 6)
        self.assertEqual(process['active_service_union_seconds'], 3)
        self.assertEqual(process['units_per_active_service_second'], 2)
        self.assertEqual(total['successful_batch_units_sum'], 7)
        self.assertEqual(total['active_service_union_seconds'], 4)
        self.assertEqual(total['units_per_measurement_wall_second'], .07)
        self.assertEqual(total['actual_batch_distribution_successful'], {'1': 1, '2': 1, '4': 1})

    def test_exact_quantiles_have_documented_interpolation(self):
        for index, duration in enumerate([1, 2, 10, 20]):
            self.service('reward_infer', index * 21, index * 21 + duration, 1)
        duration = self.group(self.report(), 'reward_infer')['duration']
        self.assertEqual(duration['median_ms'], 6000)
        self.assertAlmostEqual(duration['p95_ms'], 18500)
        self.assertEqual(duration['min_ms'], 1000)
        self.assertEqual(duration['max_ms'], 20000)

    def test_exact_profile_and_export_mask_discards_whole_crossing_service(self):
        self.profile()
        self.service('ctrl_infer', 39, 41, 2)
        self.service('ctrl_infer', 52, 54, 3)
        self.service('ctrl_infer', 60, 62, 4)
        self.update(0, timestamp=50)
        self.update(0, timestamp=55)
        report = self.report(True)
        self.assertEqual(report['profile_filter']['merged_mask_ns'], [[40 * NS, 55 * NS]])
        self.assertEqual(report['profile_windows'][0]['synchronous_export_ns'], [50 * NS, 55 * NS])
        self.assertEqual(report['measurement_window']['effective_wall_seconds'], 85)
        group = self.group(report, 'ctrl_infer')
        self.assertEqual(group['excluded_profile_overlap'], 2)
        self.assertEqual(group['successful_batch_units_sum'], 4)
        self.assertAlmostEqual(group['units_per_measurement_wall_second'], 4 / 85)
        self.assertEqual(report['optimizer_updates']['policy']['global_successful_updates'], 1)
        self.assertEqual(report['optimizer_updates']['policy']['excluded_profile_points'], 1)

    def test_incomplete_profile_refuses_false_clean_baseline(self):
        self.event('nsight_capture_started', actor='driver', data={'started_ns': 40 * NS}, timestamp=40)
        self.service('policy_infer', 60, 62, 1)
        self.assertFalse(self.report()['profile_filter']['exact_complete'])
        with self.assertRaisesRegex(ValueError, 'Cannot exclude profiling'):
            self.report(True)

    def test_global_optimizer_rank_must_be_explicit_zero(self):
        self.update(0)
        self.update(2, kind='optimizer_update')
        self.event('optimizer_update', workload='policy_train', timestamp=20, batch=576,
                   data={'successful_optimizer_steps': 1})
        result = self.report()['optimizer_updates']['policy']
        self.assertEqual(result['global_successful_updates'], 1)
        self.assertEqual(result['invalid_global_rank_events'], 2)

    def test_episode_categories_and_unique_wm_ingestion_remain_separate(self):
        self.service('cpu_wm_windows', 5, 6, 1, actor='rollout')
        self.event('eval_episode_completed', actor='stats', timestamp=7, data={'success': 1})
        self.event('eval_episode_completed', actor='stats', timestamp=8, data={'success': 0})
        self.connection.executemany('INSERT INTO replay_samples VALUES(?,?,?,?)',
            [(RUN, 'wm', 'episode_a', 6 * NS), (RUN, 'wm', 'episode_a', 6 * NS + 1),
             (RUN, 'wm_eval', 'episode_b', 9 * NS)])
        result = self.report()['episode_production']
        self.assertEqual(result['real_rollout_postprocessing_completed']['count'], 1)
        self.assertEqual(result['evaluation_completed']['count'], 2)
        self.assertEqual(result['evaluation_completed']['success_count'], 1)
        self.assertEqual(result['wm_episode_ingestion']['counts_by_replay'], {'wm': 1, 'wm_eval': 1})

    def prefetch(self, timestamp, transition, ready=0, pending=0, in_use=0, wait=0, closed=False, cumulative=0):
        self.event('cpu_replay_prefetch_state', timestamp=timestamp, actor='trainer',
                   data={'model': 'policy', 'rank': 0, 'world_size': 4, 'transition': transition,
                         'capacity_batches': 1, 'expected_local_batch_size': 144,
                         'ready_batches': ready, 'pending_fetches': pending,
                         'in_use_batches': in_use, 'empty_waiters': wait, 'closed': closed,
                         'empty_wait_ns': cumulative * NS, 'observer_errors': 0, 'failed': False})

    def test_prefetch_state_integral_uses_explicit_transitions_not_event_frequency(self):
        self.prefetch(0, 'created')
        self.prefetch(10, 'fetch_started', pending=1)
        self.prefetch(20, 'empty_wait_started', pending=1, wait=1)
        self.prefetch(30, 'ready', ready=1, wait=1, cumulative=10)
        self.prefetch(31, 'empty_wait_ended', ready=1, cumulative=11)
        self.prefetch(40, 'taken', in_use=1, cumulative=11)
        self.prefetch(60, 'released_after_training_phase', cumulative=11)
        self.prefetch(70, 'closed', closed=True, cumulative=11)
        result = self.report()['prefetch']
        self.assertEqual(result['status'], 'available')
        owner = result['owners'][0]
        self.assertEqual(owner['coverage_seconds'], 70)
        self.assertEqual(owner['state_seconds'], {'ready_batches': 10, 'pending_fetches': 20,
                                                  'in_use_batches': 20, 'empty_waiters': 11})
        self.assertAlmostEqual(owner['state_fraction_of_observed_lifecycle']['ready_batches'], 1 / 7)
        self.assertEqual(owner['cumulative_empty_wait_delta_ns_unmasked_run_scope'], 11 * NS)
        self.assertFalse(owner['right_censored'])

    def test_prefetch_mask_splits_time_while_preserving_boundary_state_and_raw_counter(self):
        self.profile(20, 30, 35)
        self.prefetch(0, 'created')
        self.prefetch(10, 'fetch_started', pending=1)
        self.prefetch(20, 'empty_wait_started', pending=1, wait=1)
        self.prefetch(30, 'ready', ready=1, wait=1, cumulative=10)
        self.prefetch(31, 'empty_wait_ended', ready=1, cumulative=11)
        self.prefetch(40, 'taken', in_use=1, cumulative=11)
        self.prefetch(60, 'released_after_training_phase', cumulative=11)
        self.prefetch(70, 'closed', closed=True, cumulative=11)
        owner = self.report(True)['prefetch']['owners'][0]
        self.assertEqual(owner['coverage_seconds'], 55)
        self.assertEqual(owner['state_seconds']['ready_batches'], 5)
        self.assertEqual(owner['state_seconds']['pending_fetches'], 10)
        self.assertEqual(owner['state_seconds']['empty_waiters'], 0)
        self.assertEqual(owner['cumulative_empty_wait_delta_ns_unmasked_run_scope'], 11 * NS)

    def test_missing_prefetch_is_not_reported_as_zero_occupancy(self):
        self.service('policy_train', 10, 12, 8)
        self.event('data_sample', timestamp=4, data={'ready': True})
        result = self.report()['prefetch']
        self.assertEqual(result['status'], 'not_collected')
        self.assertEqual(result['owners'], [])

    def test_database_is_read_only_and_missing_schema_rejected(self):
        self.service('policy_infer', 10, 12, 8)
        self.connection.commit()
        before = hashlib.sha256(self.db.read_bytes()).hexdigest()
        self.report()
        self.assertEqual(before, hashlib.sha256(self.db.read_bytes()).hexdigest())
        bad = Path(self.temporary.name) / 'bad.sqlite'
        connection = sqlite3.connect(bad)
        try:
            connection.execute('CREATE TABLE events(run_id TEXT)')
            connection.commit()
        finally:
            connection.close()
        with self.assertRaisesRegex(ValueError, 'Missing events columns'):
            analyze(bad, RUN)


if __name__ == '__main__':
    unittest.main()
