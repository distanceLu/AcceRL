"""CPU lifecycle contracts; no torch, Ray, or simulator imports."""
import ast
import argparse
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rl.legacy_baseline_control import LegacyBaselineControl, actor_placement, validate_rank_steps


class Clock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now


class LegacyBaselineControlTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.clock = Clock()

    def control(self, seconds=20, stop_file=None):
        return LegacyBaselineControl(self.root, seconds, stop_file, clock=self.clock)

    def test_deadline_starts_after_initialization_and_only_stops_at_boundary(self):
        control = self.control()
        self.clock.now = 500
        control.start_training()
        self.clock.now = 519
        self.assertFalse(control.after_update(1, train_seconds=2, publication_seconds=1, samples=576))
        self.clock.now = 520
        self.assertTrue(control.after_update(2, train_seconds=2, publication_seconds=1, samples=576))
        self.assertEqual(control.reason, 'deadline')
        records = [json.loads(line) for line in (self.root / 'updates.jsonl').read_text().splitlines()]
        self.assertEqual(records[-1]['consumed_training_samples'], 1152)
        self.assertEqual(records[-1]['training_seconds'], 20)

    def test_zero_deadline_disabled_and_stop_flag_stops_at_boundary(self):
        stop_file = self.root / 'stop'
        control = self.control(seconds=0, stop_file=stop_file)
        control.start_training()
        self.clock.now += 10000
        self.assertFalse(control.after_update(1, train_seconds=1, publication_seconds=1, samples=576))
        stop_file.touch()
        self.assertTrue(control.after_update(2, train_seconds=1, publication_seconds=1, samples=576))
        self.assertEqual(control.reason, 'stop_file')

    def test_completion_receipt_follows_successful_save_only(self):
        control = self.control()
        control.start_training()
        checkpoint = self.root / 'checkpoints' / 'latest' / 'agent_checkpoint_epoch_1'

        def save():
            self.assertFalse((self.root / 'driver_complete.json').exists())
            checkpoint.mkdir(parents=True)
            return checkpoint

        receipt = control.complete(1, save)
        self.assertEqual(receipt['checkpoint_dir'], str(checkpoint.resolve()))
        self.assertEqual(receipt['checkpoint_kind'], 'inference_weights_only')
        self.assertFalse(receipt['exact_resume'])
        self.assertEqual(json.loads((self.root / 'driver_complete.json').read_text())['status'], 'completed')

    def test_save_exception_cannot_create_completion_receipt(self):
        control = self.control()
        control.start_training()

        def save():
            raise RuntimeError('save failed')

        with self.assertRaisesRegex(RuntimeError, 'save failed'):
            control.complete(1, save)
        self.assertFalse((self.root / 'driver_complete.json').exists())

    def test_metadata_records_actual_batch_and_separate_startup_fill_save_times(self):
        control = self.control()
        self.clock.now += 5
        args = dict(train_batch_size=12, accumulation_steps=16, num_trainer_gpus=3)
        placements = [dict(role='trainer', index=0, ray_gpu_ids=[0])]
        control.record_metadata(args, placements)
        self.clock.now += 20
        control.record_metadata(args, placements, replay_fill_seconds=20)
        control.start_training()
        self.clock.now += 8

        def save():
            self.clock.now += 3
            return self.root

        receipt = control.complete(1, save)
        metadata = json.loads((self.root / 'driver_metadata.json').read_text())
        self.assertEqual(metadata['global_batch_size'], 576)
        self.assertEqual(metadata['placements'], placements)
        self.assertEqual(receipt['initialization_seconds'], 5)
        self.assertEqual(receipt['replay_fill_seconds'], 20)
        self.assertEqual(receipt['training_seconds'], 8)
        self.assertEqual(receipt['final_save_seconds'], 3)

    def test_legacy_parser_defaults_and_operational_arguments(self):
        source = Path(__file__).with_name('ds_libero_ppo_discrete.py').read_text(encoding='utf-8')
        function = next(node for node in ast.parse(source).body
                        if isinstance(node, ast.FunctionDef) and node.name == 'parse_args')
        namespace = dict(argparse=argparse, math=math, os=os, json=json)
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<parser>', 'exec'), namespace)
        with patch('sys.argv', ['driver']), patch.dict(os.environ):
            args = namespace['parse_args']()
        self.assertEqual((args.max_train_seconds, args.baseline_output_dir, args.ray_temp_dir, args.stop_file),
                         (0, None, '/dev/shm', None))
        self.assertEqual((args.target_success_rate, args.target_success_window), (0, 1000))
        for argv in (['--max-train-seconds', '20'], ['--max-train-seconds', 'nan']):
            with patch('sys.argv', ['driver'] + argv), patch('sys.stderr'):
                with self.assertRaises(SystemExit):
                    namespace['parse_args']()
        invalid_targets = [
            ['--target-success-rate', '.99'],
            ['--baseline-output-dir', str(self.root), '--target-success-rate', 'nan'],
            ['--baseline-output-dir', str(self.root), '--target-success-rate', '1.1'],
            ['--baseline-output-dir', str(self.root), '--target-success-rate', '-.1'],
            ['--baseline-output-dir', str(self.root), '--target-success-rate', '.99'],
            ['--baseline-output-dir', str(self.root), '--target-success-rate', '.99', '--task-ids', '0', '--target-success-window', '0'],
        ]
        for argv in invalid_targets:
            with patch('sys.argv', ['driver'] + argv), patch('sys.stderr'):
                with self.assertRaises(SystemExit):
                    namespace['parse_args']()
        with patch('sys.argv', ['driver', '--baseline-output-dir', str(self.root),
                               '--target-success-rate', '.99', '--task-ids', '0']), patch.dict(os.environ):
            args = namespace['parse_args']()
        self.assertEqual(args.target_success_rate, .99)
        self.assertEqual(args.max_train_seconds, 0)

    def test_bounded_import_preserves_parent_tmpdir_legacy_import_keeps_default(self):
        source = Path(__file__).with_name('ds_libero_ppo_discrete.py').read_text(encoding='utf-8')
        prefix = source.split('import time', 1)[0]
        for supplied, expected in (({'WEAVER_LEGACY_GIT_COMMIT': 'commit', 'TMPDIR': '/data/run/tmp'}, '/data/run/tmp'),
                                   ({'WEAVER_LEGACY_GIT_COMMIT': 'commit'}, '/dev/shm'),
                                   ({'TMPDIR': '/data/other'}, '/dev/shm')):
            with patch.dict(os.environ, supplied, clear=True):
                exec(compile(prefix, '<driver-import>', 'exec'), {})
                self.assertEqual(os.environ['TMPDIR'], expected)

    def test_invalid_duration_and_existing_completion_are_rejected(self):
        for duration in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                self.control(seconds=duration)
        (self.root / 'driver_complete.json').write_text('{}')
        with self.assertRaises(FileExistsError):
            self.control()

    def test_placement_queries_existing_actor_allocation(self):
        from types import SimpleNamespace
        torch = SimpleNamespace(cuda=SimpleNamespace(current_device=lambda: 0,
                                get_device_name=lambda device: 'existing GPU'))
        ray = SimpleNamespace(get_gpu_ids=lambda: [2])
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '6'}):
            placement = actor_placement(torch, ray, role='trainer', index=2)
        self.assertEqual(placement['torch_device'], 'cuda:0')
        self.assertEqual(placement['ray_gpu_ids'], [2])
        self.assertEqual(placement['cuda_visible_devices'], '6')

    def test_rank_step_validation_requires_one_matching_update_before_counting(self):
        def results(*steps):
            return [(None,) * 6 + (step,) for step in steps]

        self.assertEqual(validate_rank_steps(results(8, 8, 8), 7), 8)
        control = self.control()
        control.start_training()
        for steps in ((8, 9, 8), (7, 7, 7), (9, 9, 9)):
            with self.assertRaisesRegex(RuntimeError, 'rank.*step'):
                validate_rank_steps(results(*steps), 7)
            self.assertEqual(control.completed_updates, 0)
            self.assertFalse((self.root / 'driver_complete.json').exists())
        with self.assertRaises(RuntimeError):
            validate_rank_steps([], 7)

    def test_driver_hooks_follow_collective_and_publication(self):
        source = Path(__file__).with_name('ds_libero_ppo_discrete.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        loop = next(node for node in ast.walk(main) if isinstance(node, ast.While)
                    and 'global_step < args.train_iters' in ast.unparse(node.test))
        loop_source = ast.unparse(loop)
        self.assertLess(loop_source.index('results = ray.get(train_tasks)'), loop_source.index('sync_time ='))
        self.assertLess(loop_source.index('validate_rank_steps(results, global_step)'),
                        loop_source.index('broadcast_task ='))
        self.assertLess(loop_source.index('ray.get([broadcast_task] + receive_tasks)'),
                        loop_source.index('baseline.after_update'))
        self.assertLess(loop_source.index('ray.get([broadcast_task] + receive_tasks)'),
                        loop_source.index('stats_actor.get_success_snapshot.remote'))
        main_source = ast.unparse(main)
        self.assertLess(main_source.index('baseline.complete'), main_source.index('ray.shutdown()'))
        self.assertIn('raise FileNotFoundError', main_source)
        self.assertIn("args.num_eval_workers > 0", main_source)
        self.assertIn("os.path.join(args.baseline_output_dir, 'tensorboard')", main_source)

    def target_control(self, **kwargs):
        return LegacyBaselineControl(self.root, clock=self.clock, wall_clock=lambda: 1000 + self.clock.now,
                                     target_success_rate=.99, target_success_window=1000, **kwargs)

    def snapshot(self, successes=990, episodes=1000, **kwargs):
        result = dict(window_episodes=episodes, window_successes=successes, window_size=1000,
                      success_rate=successes / episodes if episodes else 0.,
                      total_episodes_processed=episodes + 50, total_env_steps=123000,
                      observed_unix=1000 + self.clock.now, **kwargs)
        result['crossings'] = {str(threshold): dict(result) for threshold in (98, 99)
                               if episodes == 1000 and successes * 100 >= threshold * 1000}
        return result

    def target_update(self, control, step, snapshot):
        return control.after_update(step, train_seconds=1, publication_seconds=.1,
                                    samples=576, success_snapshot=snapshot)

    def test_target_requires_full_window_and_exact_counts(self):
        control = self.target_control()
        control.start_training()
        self.assertFalse(self.target_update(control, 1, self.snapshot(99, 99)))
        self.assertFalse(control.milestones)
        self.assertFalse(self.target_update(control, 2, self.snapshot(989)))
        self.assertIn('98', control.milestones)
        self.assertNotIn('99', control.milestones)
        self.assertTrue(self.target_update(control, 3, self.snapshot(990)))
        self.assertEqual(control.reason, 'success_target_reached')

    def test_first_crossing_milestones_are_immutable_and_include_launch_time(self):
        with patch.dict(os.environ, {'WEAVER_LEGACY_STARTED_UNIX': '1000'}):
            control = self.target_control()
        self.clock.now += 5
        control.start_training()
        self.clock.now += 8
        self.assertTrue(self.target_update(control, 7, self.snapshot(995)))
        first = json.loads(json.dumps(control.milestones))
        self.assertEqual(first['98'], first['99'])
        self.assertEqual(first['99']['policy_step'], 7)
        self.assertEqual(first['99']['elapsed_seconds'], 13)
        self.assertEqual(first['99']['training_seconds'], 8)
        self.assertEqual(first['99']['launch_elapsed_seconds'], 23)
        self.clock.now += 100
        self.assertTrue(self.target_update(control, 8, self.snapshot(900)))
        self.assertEqual(control.milestones, first)
        with self.assertRaises(RuntimeError):
            control.complete(8, lambda: self.root)
        receipt = control.complete(7, lambda: self.root)
        self.assertEqual(receipt['reason'], 'success_target_reached')
        self.assertEqual(receipt['milestones'], first)
        status = json.loads((self.root / 'driver_status.json').read_text())
        self.assertEqual(status['reason'], 'success_target_reached')
        updates = [json.loads(line) for line in (self.root / 'updates.jsonl').read_text().splitlines()]
        self.assertEqual(updates[0]['reason'], 'success_target_reached')
        self.assertEqual(len(updates), 1)

    def test_malformed_snapshot_cannot_count_an_update_or_complete(self):
        control = self.target_control()
        control.start_training()
        for bad in (dict(success_rate=float('nan')), dict(window_successes=1001),
                    dict(window_episodes=True), dict(window_successes=990.0),
                    dict(success_rate=.989), dict(window_size=999),
                    dict(total_episodes_processed=1), dict(total_env_steps=0),
                    dict(observed_unix=float('inf'))):
            snapshot = self.snapshot()
            snapshot.update(bad)
            with self.assertRaises(ValueError):
                self.target_update(control, 1, snapshot)
        with self.assertRaises(ValueError):
            self.target_update(control, 1, None)
        self.assertEqual(control.completed_updates, 0)
        self.assertFalse((self.root / 'driver_complete.json').exists())

    def test_manual_stop_precedes_target_and_target_precedes_deadline(self):
        stop = self.root / 'stop'
        control = self.target_control(max_train_seconds=1, stop_file=stop)
        control.start_training()
        self.clock.now += 2
        stop.touch()
        self.assertTrue(self.target_update(control, 1, self.snapshot()))
        self.assertEqual(control.reason, 'stop_file')
        self.assertIn('99', control.milestones)
        with tempfile.TemporaryDirectory() as other:
            control = LegacyBaselineControl(other, max_train_seconds=1, target_success_rate=.99,
                                            clock=self.clock)
            control.start_training()
            self.clock.now += 2
            self.assertTrue(self.target_update(control, 1, self.snapshot()))
            self.assertEqual(control.reason, 'success_target_reached')

    def test_target_step_cap_is_truthful_without_full_success_window(self):
        control = self.target_control()
        control.start_training()
        self.target_update(control, 30000, self.snapshot(980))
        receipt = control.complete(30000, lambda: self.root)
        self.assertEqual(receipt['reason'], 'train_iters')
        self.assertNotIn('99', receipt['milestones'])
        self.assertEqual(json.loads((self.root / 'driver_status.json').read_text())['reason'], 'train_iters')

    def test_stats_snapshot_excludes_eval_and_refuses_multiple_task_windows(self):
        from collections import deque, defaultdict
        from typing import Dict, Optional
        import time
        source = Path(__file__).with_name('ds_libero_ppo_discrete.py').read_text(encoding='utf-8')
        actor = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)
                     and node.name == 'StatsActor')
        actor.decorator_list = []
        namespace = dict(deque=deque, defaultdict=defaultdict, Dict=Dict, Optional=Optional, time=time)
        exec(compile(ast.Module(body=[actor], type_ignores=[]), '<stats>', 'exec'), namespace)
        stats = namespace['StatsActor'](window_size=2, success_window_size=3)
        empty = stats.get_success_snapshot(3)
        self.assertEqual((empty['window_episodes'], empty['window_successes'], empty['success_rate']), (0, 0, 0.))
        with self.assertRaises(ValueError):
            stats.get_success_snapshot(4)
        with self.assertRaises(ValueError):
            stats.add_episode_return('task0', 1., .1, 10, float('nan'))
        for success in (1., 0., 1., 1.):
            stats.add_episode_return('task0', 1., .1, 10, success)
        stats.add_episode_return('eval_task0', 1., .1, 10, 1.)
        snapshot = stats.get_success_snapshot(3)
        self.assertEqual(snapshot['window_successes'], 2)
        self.assertEqual(snapshot['window_episodes'], 3)
        self.assertEqual(snapshot['total_episodes_processed'], 4)
        self.assertEqual(snapshot['total_env_steps'], 40)
        stats.add_episode_return('task1', 1., .1, 10, 1.)
        with self.assertRaises(ValueError):
            stats.get_success_snapshot(3)

    def test_target_metadata_keeps_first_milestones_and_exact_window(self):
        control = self.target_control()
        control.record_metadata(dict(train_batch_size=12, accumulation_steps=16, num_trainer_gpus=3), [])
        control.start_training()
        self.target_update(control, 1, self.snapshot(985))
        first = json.loads((self.root / 'driver_metadata.json').read_text())
        self.assertEqual(first['milestones']['98']['policy_step'], 1)
        self.assertEqual(first['success_snapshot']['window_successes'], 985)
        self.target_update(control, 2, self.snapshot())
        final = json.loads((self.root / 'driver_metadata.json').read_text())
        self.assertEqual(final['milestones']['98'], first['milestones']['98'])
        self.assertEqual(final['milestones']['99']['policy_step'], 2)

    def test_latched_crossing_stops_after_regression_with_actual_event_times(self):
        with patch.dict(os.environ, {'WEAVER_LEGACY_STARTED_UNIX': '1000'}):
            control = self.target_control()
        crossing = self.snapshot()
        crossing['observed_unix'] = 1012.
        crossing = {key: value for key, value in crossing.items() if key != 'crossings'}
        self.clock.now = 15
        control.start_training()
        self.clock.now = 25
        latest = self.snapshot(989)
        latest['crossings'] = {'98': dict(crossing), '99': dict(crossing)}
        self.assertTrue(self.target_update(control, 9, latest))
        milestone = control.milestones['99']
        self.assertEqual(milestone['observed_unix'], 1012.)
        self.assertEqual(milestone['launch_elapsed_seconds'], 12.)
        self.assertEqual(milestone['elapsed_seconds'], 2.)
        self.assertEqual(milestone['training_seconds'], 0.)
        self.assertEqual(milestone['training_time_scope'], 'before_training_loop')
        self.assertEqual(milestone['boundary_policy_step'], 9)
        self.assertEqual(milestone['boundary_observed_unix'], 1025.)
        receipt = control.complete(9, lambda: self.root)
        self.assertEqual(receipt['success_snapshot']['window_successes'], 990)
        self.assertEqual(receipt['current_success_snapshot']['window_successes'], 989)

    def test_training_crossing_time_excludes_delay_until_safe_boundary(self):
        control = self.target_control()
        self.clock.now = 15
        control.start_training()
        event = {key: value for key, value in self.snapshot().items() if key != 'crossings'}
        event['observed_unix'] = 1018.
        self.clock.now = 25
        latest = self.snapshot(989)
        latest['crossings'] = {'98': dict(event), '99': dict(event)}
        self.assertTrue(self.target_update(control, 5, latest))
        self.assertEqual(control.milestones['99']['training_seconds'], 3.)
        self.assertEqual(control.milestones['99']['training_time_scope'], 'training_loop')
        self.assertEqual(control.milestones['99']['boundary_observed_unix'], 1025.)

    def test_stats_latches_first_990_before_decline_between_queries(self):
        from collections import deque, defaultdict
        from typing import Dict, Optional
        from types import SimpleNamespace
        source = Path(__file__).with_name('ds_libero_ppo_discrete.py').read_text(encoding='utf-8')
        actor = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)
                     and node.name == 'StatsActor')
        actor.decorator_list = []
        namespace = dict(deque=deque, defaultdict=defaultdict, Dict=Dict, Optional=Optional,
                         time=SimpleNamespace(time=lambda: 1000 + self.clock.now))
        exec(compile(ast.Module(body=[actor], type_ignores=[]), '<stats>', 'exec'), namespace)
        stats = namespace['StatsActor'](window_size=2, success_window_size=1000)
        for success in [1] * 990 + [0] * 10:
            stats.add_episode_return('task0', 1., .1, 10, success)
        first = stats.get_success_snapshot(1000)['crossings']
        self.assertEqual(first['99']['window_successes'], 990)
        self.clock.now += 10
        stats.add_episode_return('eval_task0', 1., .1, 10, 1)
        stats.add_episode_return('task0', 1., .1, 10, 0)
        latest = stats.get_success_snapshot(1000)
        self.assertEqual(latest['window_successes'], 989)
        self.assertEqual(latest['crossings'], first)
        self.assertEqual(first['99']['total_episodes_processed'], 1000)
        control = self.target_control()
        control.start_training()
        self.assertTrue(self.target_update(control, 1, latest))


if __name__ == '__main__':
    unittest.main()
