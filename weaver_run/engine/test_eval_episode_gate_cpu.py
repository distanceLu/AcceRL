import threading
import unittest
from eval_episode_gate import admission_decision, wait_for_episode

class EpisodeGateTests(unittest.TestCase):
    def test_pending_and_guard(self):
        for remaining in (1, 4, 2):
            self.assertFalse(admission_decision(True, remaining, 359)['admitted'])
        self.assertEqual(admission_decision(True, 2, 360)['reason'], 'guard_timeout')
        self.assertTrue(admission_decision(True, 0, 1)['admitted'])
        self.assertTrue(admission_decision(False, 2, 0)['admitted'])

    def test_wait_release_stop_and_failure(self):
        stop = threading.Event()
        calls = []
        def query(elapsed):
            calls.append(elapsed)
            return admission_decision(True, int(len(calls) == 1), elapsed)
        self.assertTrue(wait_for_episode(query, stop))
        self.assertEqual(len(calls), 2)
        self.assertGreaterEqual(calls[1], .09)
        stop.set()
        self.assertFalse(wait_for_episode(query, stop))
        stop.clear()
        def failure(elapsed):
            raise RuntimeError('broker failed')
        with self.assertRaises(RuntimeError):
            wait_for_episode(failure, stop)

class BrokerGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_nested_release_stop_failure_telemetry(self):
        from cost_scheduler_runtime import CostAdmissionBroker
        from test_cost_scheduler_runtime_cpu import settings
        events = []
        broker = CostAdmissionBroker(settings(eval_refresh_gate=True), 'test', 'a'*64,
            emit=lambda *a, **kw: events.append((a, kw)))
        await broker.refresh_demand(2)
        await broker.refresh_demand(3)
        self.assertFalse((await broker.eval_episode_admission('e', 0))['admitted'])
        await broker.eval_episode_admission('e', 1)
        self.assertEqual(sum(a[0] == 'eval_episode_admission' for a, _ in events), 1)
        await broker.refresh_demand(-2)
        self.assertFalse((await broker.eval_episode_admission('e', 359))['admitted'])
        self.assertTrue((await broker.eval_episode_admission('e', 360))['guard_triggered'])
        await broker.refresh_demand(-3)
        self.assertEqual((await broker.eval_episode_admission('e', 361))['reason'], 'refresh_finished')
        await broker.stop_eval_admission()
        self.assertTrue((await broker.eval_episode_admission('e', 0))['stopped'])
        broker.eval_admission_stopped = False
        broker.failed = 'failure'
        with self.assertRaisesRegex(RuntimeError, 'failure'): await broker.eval_episode_admission('e', 0)

    def test_worker_reset_boundaries(self):
        import ast
        from pathlib import Path
        tree = ast.parse(Path(__file__).with_name('train.py').read_text(encoding='utf8'))
        worker = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'EvaluationWorkerActor')
        run = next(n for n in worker.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
        resets = 0
        for node in ast.walk(run):
            for value in vars(node).values():
                if not isinstance(value, list): continue
                for i, stmt in enumerate(value):
                    if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Attribute) and stmt.value.func.attr == '_reset_and_select_env':
                        self.assertIsInstance(value[i-1], ast.If)
                        self.assertIn('_admit_episode', ast.unparse(value[i-1]))
                        resets += 1
        self.assertEqual(resets, 2)

    def test_stop_during_pending_wait(self):
        stop = threading.Event()
        def query(elapsed):
            stop.set()
            return admission_decision(True, 1, elapsed)
        self.assertFalse(wait_for_episode(query, stop))


class WorkerEpisodeBoundaryTests(unittest.TestCase):
    def run_worker(self, admit_first):
        """Execute the real worker run method without Ray, models or a GPU."""
        import ast
        import os
        import random
        import sys
        import time
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        tree = ast.parse(Path(__file__).with_name('train.py').read_text(encoding='utf8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'EvaluationWorkerActor')
        run = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
        events = []
        active_refresh = False
        def admit():
            events.append('admit')
            return admit_first and not active_refresh
        def reset(seed):
            events.append('reset')
            return {}, {}
        def step(action):
            nonlocal active_refresh
            active_refresh = True
            events.append('step')
            completed = events.count('step') == 2
            if events.count('step') > 2:
                raise AssertionError('A new episode ran during active refresh')
            return {}, float(completed), completed, False, {'is_success': completed}
        preview = SimpleNamespace(add=lambda *a, **k: None, finish=lambda **k: events.append('finish'))
        visual = SimpleNamespace(begin=lambda *a, **k: preview, env_metadata=lambda *a: {})
        worker = SimpleNamespace(wid=0, task_description='task', env=SimpleNamespace(step=step),
            cfg=None, processor=None, torch_dtype=None, infer=None, current_env_name='task',
            capture_last_versions={'policy': 1}, _admit_episode=admit, _reset_and_select_env=reset,
            stats_actor=SimpleNamespace(add_episode_return=SimpleNamespace(remote=lambda *a, **k: events.append('stats'))))
        namespace = dict(os=os, random=random, time=time, now_ns=time.time_ns,
            prepare_one_obs=lambda *a: {}, profile_rpc=lambda *a, **k: (None, [0, 1], None, None, None))
        exec(compile(ast.Module(body=[run], type_ignores=[]), 'actual_eval_run', 'exec'), namespace)
        with patch.dict(sys.modules, {'visualization': visual}), patch.dict(os.environ, {'ACCERL_SEED': '1'}):
            namespace['run'](worker)
        return events

    def test_denied_first_episode_never_resets_or_submits_inference(self):
        self.assertEqual(self.run_worker(False), ['admit'])

    def test_refresh_start_does_not_interrupt_admitted_episode(self):
        self.assertEqual(self.run_worker(True), ['admit', 'reset', 'step', 'step', 'finish', 'stats', 'admit'])


if __name__ == '__main__': unittest.main()
