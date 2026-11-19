"""Episode-level first crossings, with the real StatsActor and no Ray runtime."""
import ast
from collections import defaultdict, deque
import copy
import json
from pathlib import Path
import time
import unittest


def stats_actor(clock):
    tree = ast.parse(Path(__file__).with_name('train.py').read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'StatsActor')
    node.decorator_list = []
    namespace = dict(defaultdict=defaultdict, deque=deque, setup_debugger=lambda *args: None,
                     now_ns=clock, time=time)
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'StatsActor', 'exec'), namespace)
    return namespace['StatsActor']


class ModelFreeSuccessTests(unittest.TestCase):
    def setUp(self):
        self.start = 123_000_000_000
        self.clock = self.start
        self.actor = stats_actor(lambda: self.clock)(
            window_size=3, model_free_run_started_ns=self.start)

    def episode(self, index, success, length=150):
        self.clock += 1_000_000_000
        self.actor.add_episode_return(f'task{index % 4}', 3., .1, length, float(success),
                                      actor_id=index % 3, step_num=1)

    def crossings(self):
        from model_free_success import validate_model_free_success
        snapshot = self.actor.get_model_free_success()
        self.assertEqual(snapshot, json.loads(json.dumps(snapshot, allow_nan=False)))
        self.assertIs(validate_model_free_success(snapshot), snapshot)
        return snapshot

    def test_records_first_crossings_between_reads_in_one_global_real_window(self):
        self.assertEqual(self.crossings()['window_episodes'], 0)
        for i in range(999):
            self.episode(i, i >= 20)
        self.actor.add_episode_return('eval_task3', 1., 0., 150, 1.)
        self.actor.add_imagine_reward(1., 0)
        self.assertEqual(self.crossings()['crossings'], {})
        # Roll the oldest failures out while four tasks and three workers interleave.
        for i in range(999, 1010):
            self.episode(i, True, length=3)
        # Force the window back below 99% before the next driver read.
        for i in range(1010, 1030):
            self.episode(i, False)
        latest = self.crossings()
        low, high = latest['crossings']['98'], latest['crossings']['99']
        self.assertEqual((low['window_successes'], low['total_episodes_processed']), (980, 1000))
        self.assertEqual((high['window_successes'], high['total_episodes_processed']), (990, 1010))
        self.assertEqual(high['total_env_steps'], 999 * 150 + 11 * 3)
        self.assertEqual(high['observed_ns'], self.start + 1010 * 1_000_000_000)
        self.assertEqual(high['elapsed_seconds'], 1010.)
        self.assertEqual(latest['success_rate'], .98)
        self.assertEqual(latest['window_episodes'], 1000)
        high.clear()
        self.assertEqual(self.crossings()['crossings']['99']['window_successes'], 990)

    def test_rejects_rounded_or_inconsistent_evidence_and_partial_claims(self):
        from model_free_success import validate_model_free_success
        for i in range(1000):
            self.episode(i, True)
        original = self.crossings()
        for key, value in [('success_rate', .99), ('window_successes', 1000.), ('window_episodes', 999),
                           ('elapsed_seconds', 0.), ('observed_ns', 1), ('total_env_steps', 5)]:
            sample = copy.deepcopy(original)
            sample['crossings']['99'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_model_free_success(sample)
        for key in ('98', '99'):
            sample = copy.deepcopy(original)
            del sample['crossings'][key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                validate_model_free_success(sample)

    def test_nonbinary_outcome_is_rejected_before_counting(self):
        with self.assertRaises(ValueError):
            self.episode(0, .99)
        self.assertEqual(self.crossings()['total_episodes_processed'], 0)
        self.assertEqual(self.actor.stats.get('task0'), None)


if __name__ == '__main__':
    unittest.main()
