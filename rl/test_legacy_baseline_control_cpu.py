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
        for argv in (['--max-train-seconds', '20'], ['--max-train-seconds', 'nan']):
            with patch('sys.argv', ['driver'] + argv), patch('sys.stderr'):
                with self.assertRaises(SystemExit):
                    namespace['parse_args']()

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
        main_source = ast.unparse(main)
        self.assertLess(main_source.index('baseline.complete'), main_source.index('ray.shutdown()'))
        self.assertIn('raise FileNotFoundError', main_source)
        self.assertIn("args.num_eval_workers > 0", main_source)
        self.assertIn("os.path.join(args.baseline_output_dir, 'tensorboard')", main_source)


if __name__ == '__main__':
    unittest.main()
