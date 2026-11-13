"""Real configuration and StatsActor methods, without importing Ray or CUDA."""
import ast
from collections import defaultdict, deque
import copy
import json
from pathlib import Path
import time
import unittest

import numpy as np
import evaluation_threshold as threshold
import test_main_startup_cpu as startup

HERE = Path(__file__).resolve().parent


class OnlineEvaluationTests(unittest.TestCase):
    def resolve(self, workers=3, replicas=4, mapping='1,2,3', minimum=None):
        resolver = getattr(threshold, 'resolve_online_evaluation', None)
        self.assertTrue(callable(resolver), 'Online Eval needs validated routing and its own window')
        return resolver(workers, replicas, mapping, minimum)

    def test_three_workers_use_replicas_one_two_three_and_300_samples(self):
        plan = self.resolve()
        self.assertEqual(plan['replica_ids'], [1, 2, 3])
        self.assertEqual(plan['window_size'], 300)
        self.assertEqual(plan['min_samples'], 300)

    def test_legacy_round_robin_and_workers_times_100_remain_available(self):
        self.assertEqual(self.resolve(10, 4, None)['replica_ids'], [0,1,2,3,0,1,2,3,0,1])
        self.assertEqual(self.resolve(10, 4, None)['window_size'], 1000)
        self.assertEqual(self.resolve(4, 4, None)['window_size'], 400)
        self.assertEqual(self.resolve(2, 4, None)['window_size'], 200)

    def test_zero_eval_workers_create_no_targets(self):
        self.assertEqual(self.resolve(0, 4, None)['replica_ids'], [])

    def test_managed_cli_can_disable_eval_with_an_explicit_empty_route(self):
        args = startup.arguments(overrides=('--num-eval-workers','0','--eval-inference-replicas',''))
        context = startup.Context()
        fn = startup.execute_definition(startup.startup_ast(),context.namespace())
        state = fn(args)
        self.assertEqual(state['eval_workers'],[])
        self.assertFalse(any(a.name=='EvaluationWorkerActor' for a in context.actors))

    def test_invalid_routes_fail_before_actor_creation(self):
        for mapping in ('', '0,1', '1,2,4', '-1,2,3', '1,,3', '1,2,true'):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                self.resolve(mapping=mapping)

    def test_unreachable_1000_gate_is_rejected_for_300_window(self):
        with self.assertRaises(ValueError):
            self.resolve(minimum='1000')

    def test_auto_gate_fills_at300_and_preserves_freshness_check(self):
        plan = self.resolve(minimum='auto')
        tracker = threshold.ThresholdTracker(target=.85, run_started_ns=0,
                                             min_samples=plan['min_samples'])
        before = tracker.update(1, .96, 1, 299, 20, evaluation_watermark=299)
        self.assertEqual(before['reason'], 'insufficient_samples')
        full = tracker.update(2, .96, 2, 300, 21, evaluation_watermark=300)
        self.assertTrue(full['qualified'])
        repeat = tracker.update(3, .96, 3, 300, 22, evaluation_watermark=300)
        self.assertEqual(repeat['reason'], 'no_fresh_episodes')

    def test_actual_stats_keep_eval300_and_real_imag1000(self):
        tree = ast.parse((HERE/'train.py').read_text(encoding='utf8'))
        node = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=='StatsActor'))
        node.decorator_list = []
        init = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name=='__init__')
        self.assertIn('eval_window_size', [a.arg for a in init.args.args])
        ns = dict(defaultdict=defaultdict, deque=deque, np=np, time=time,
                  setup_debugger=lambda *a:None, now_ns=time.time_ns)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])), '<actual StatsActor>', 'exec'), ns)
        stats = ns['StatsActor'](window_size=1000, eval_window_size=300)
        for i in range(1001):
            value = float(i>=701)
            stats.add_episode_return('eval_libero',value,1,20,value,completed_ns=i+1)
            stats.add_episode_return('libero',value,1,20,value)
            stats.add_imagine_reward(value,0)
            stats.add_timing_metric('test',value)
        result = stats.get_stats()
        self.assertEqual(result['_global_eval_']['window_episodes'],300)
        self.assertEqual(result['_global_eval_']['avg_return'],1.)
        self.assertEqual(result['_global_eval_']['total_episodes_processed'],1001)
        self.assertEqual(result['libero']['num_episodes_in_avg'],1000)
        self.assertAlmostEqual(result['_global_rollout_']['avg_return'],.3)
        self.assertAlmostEqual(result['_global_rollout_']['avg_imagine_reward'],.3)
        self.assertAlmostEqual(result['_timings_']['test'],.3)

    def test_real_main_routes_recipe_workers_without_changing_worker_ids(self):
        context, state = startup.MainStartupCPU().smoke()
        workers = [a for a in context.actors if a.name=='EvaluationWorkerActor']
        targets = [a.positional[0].kwargs['actor_id'] for a in workers]
        self.assertEqual(targets,[1,2,3])
        self.assertEqual([a.positional[1] for a in workers],[0,1,2])
        stats = next(a for a in context.actors if a.name=='StatsActor')
        self.assertEqual(stats.kwargs,dict(window_size=1000,eval_window_size=300))
        config = next(fields['details'] for t in context.telemetries for kind,fields in t.events if kind=='online_evaluation_config')
        self.assertEqual(config['physical_gpus'],[5,6,7])
        self.assertEqual(config['window_size'],300)
        self.assertEqual(state['threshold_tracker'].min_samples,300)

    def test_launch_preflight_reference_matches_evaluation_arguments(self):
        actual = vars(startup.arguments())
        expected = json.loads((HERE.parent/'recipe/parsed-args.json').read_bytes())
        for key in ('num_eval_workers','eval_inference_replicas','moving_avg_window'):
            with self.subTest(key=key):
                self.assertEqual(expected.get(key),actual[key])

    def test_portable_allocation_routes_within_the_selected_four_cards(self):
        context = startup.Context(physical_gpu_ids=(0,1,2,3))
        args = startup.arguments()
        args.cuda_visible_devices = '0,1,2,3'
        fn = startup.execute_definition(startup.startup_ast(), context.namespace())
        fn(args)
        config = next(fields['details'] for t in context.telemetries for kind,fields in t.events if kind=='online_evaluation_config')
        self.assertEqual(config['physical_gpus'],[1,2,3])

    def test_invalid_worker_count_and_threshold_are_rejected(self):
        for count in (-1, True, 2.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.resolve(workers=count)
        for minimum in (0, -1, True, 1.5, '0', 'bad'):
            with self.subTest(minimum=minimum), self.assertRaises(ValueError):
                self.resolve(minimum=minimum)


if __name__=='__main__':
    unittest.main()
