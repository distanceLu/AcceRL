"""CPU-only controller checks. AST isolates functions; no runner/safety module import."""
import ast
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from deadline_guard import guard_decision
from training_stop_policy import TrainingStopPolicy

SOURCE = Path(__file__).resolve().parent
DEADLINE = 2000000000


def source_tree(name):
    return ast.parse((SOURCE / name).read_text(encoding='utf-8'), filename=name)


def load_function(filename, name, namespace):
    """Compile one function only, so archived safety/CUDA imports never execute."""
    tree = source_tree(filename)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, filename, 'exec'), namespace)
    return namespace[name]


def formal_environment(extra=None):
    inherited = {'PATH': '/cpu/test', 'ACCERL_SHORT_TEST_DEADLINE_UNIX': str(DEADLINE)}
    inherited.update(extra or {})
    namespace = {'os': SimpleNamespace(environ=inherited), 'DEADLINE_UNIX': DEADLINE,
                 'ROOT': Path('/repo')}
    function = load_function('runner.py', 'formal_environment', namespace)
    return function({'run_id': 'w4pre_unit', 'started_ns': 123}, Path('/master/preflight'),
                    {'cards': [{'uuid': str(i)} for i in range(4, 8)],
                     'mps_pipe': '/tmp/pipe', 'mps_log': '/tmp/log'})


class ShortPreflightTest(unittest.TestCase):
    def test_environment_clears_pollution_and_sets_exact_short_contract(self):
        env = formal_environment({
            'ACCERL_SOFT_STOP_AT': '1', 'ACCERL_MEASURE_STOP_AFTER_EPOCHS': '999',
            'ACCERL_RUN_UNTIL_DEADLINE': '1', 'ACCERL_RUN_UNTIL_MANUAL_STOP': '1',
            'ACCERL_STOP_ON_TARGET_85': '1', 'ACCERL_SHORT_TEST_DEADLINE_UNIX': '1',
            'ACCERL_MPS_POLICY_PERCENT': '40', 'ACCERL_RESOURCE_GOVERNOR': '1',
            'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE': '20', 'CUDA_MPS_PINNED_DEVICE_MEM_LIMIT': '1G',
            'CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING': '1'})
        for key in ['ACCERL_MPS_POLICY_PERCENT', 'ACCERL_RESOURCE_GOVERNOR',
                    'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE', 'CUDA_MPS_PINNED_DEVICE_MEM_LIMIT',
                    'CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING']:
            self.assertNotIn(key, env)
        for key in ['ACCERL_RUN_UNTIL_DEADLINE', 'ACCERL_RUN_UNTIL_MANUAL_STOP',
                    'ACCERL_STOP_ON_TARGET_85']:
            self.assertEqual(env[key], '0')
        self.assertEqual(env['ACCERL_NSIGHT_NVTX'], '1')
        self.assertEqual(env['ACCERL_NSIGHT_MODE'], 'timer')
        self.assertEqual(env['ACCERL_NSIGHT_STEPS'], '10')
        self.assertEqual(env['ACCERL_NSIGHT_SECONDS'], '30')
        self.assertEqual(env['ACCERL_MEASURE_STOP_AFTER_EPOCHS'], '15')
        self.assertEqual(env['ACCERL_SOFT_STOP_AT'], str(DEADLINE))
        self.assertEqual(env['ACCERL_SHORT_TEST_DEADLINE_UNIX'], str(DEADLINE))
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '4,5,6,7')
        self.assertEqual(env['ACCERL_RUN_ID'], 'w4pre_unit')
        self.assertEqual(env['PATH'], '/cpu/test')

    def test_fifteen_epoch_bound_and_deadline_use_existing_stop_policy(self):
        env = formal_environment()
        policy = TrainingStopPolicy.from_environment(env)
        self.assertFalse(policy.until_manual_stop)
        self.assertFalse(policy.until_deadline)
        self.assertFalse(policy.stop_on_target)
        self.assertEqual(policy.deadline_unix_seconds, DEADLINE)
        args = dict(global_step=14, train_iters=30000, target_reached=True,
                    requested_stop=False, completed_epochs=14,
                    measure_stop_epochs=int(env['ACCERL_MEASURE_STOP_AFTER_EPOCHS']))
        self.assertTrue(policy.should_continue(**args, now=DEADLINE - 1))
        self.assertFalse(policy.should_continue(**dict(args, completed_epochs=15), now=DEADLINE - 1))
        self.assertFalse(policy.should_continue(**args, now=DEADLINE))

    def test_deadline_is_required_integer_without_importing_safety(self):
        tree = source_tree('gpu47_safety.py')
        assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'DEADLINE_UNIX' for t in n.targets))
        following = tree.body[tree.body.index(assignment) + 1]
        module = ast.fix_missing_locations(ast.Module(body=[assignment, following], type_ignores=[]))
        code = compile(module, 'deadline_assignment_only', 'exec')
        for environ, error in [({}, KeyError), ({'ACCERL_SHORT_TEST_DEADLINE_UNIX': '1.5'}, ValueError),
                               ({'ACCERL_SHORT_TEST_DEADLINE_UNIX': '0'}, ValueError),
                               ({'ACCERL_SHORT_TEST_DEADLINE_UNIX': '-1'}, ValueError)]:
            with self.assertRaises(error):
                exec(code, {'os': SimpleNamespace(environ=environ)})
        namespace = {'os': SimpleNamespace(environ={'ACCERL_SHORT_TEST_DEADLINE_UNIX': str(DEADLINE)})}
        exec(code, namespace)
        self.assertEqual(namespace['DEADLINE_UNIX'], DEADLINE)
        self.assertIsInstance(namespace['DEADLINE_UNIX'], int)

    def test_cpu_helper_environment_and_call_sites_preserve_same_deadline(self):
        namespace = {'os': SimpleNamespace(environ={'ACCERL_RUN_ID': 'stale',
                     'CUDA_VISIBLE_DEVICES': 'old', 'ACCERL_SHORT_TEST_DEADLINE_UNIX': '1'}),
                     'DEADLINE_UNIX': DEADLINE}
        helper = load_function('gpu47_safety.py', 'cpu_helper_environment', namespace)
        env = helper()
        self.assertEqual(env['ACCERL_SHORT_TEST_DEADLINE_UNIX'], str(DEADLINE))
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
        self.assertNotIn('ACCERL_RUN_ID', env)
        runner = ast.unparse(source_tree('runner.py'))
        safety = ast.unparse(source_tree('gpu47_safety.py'))
        self.assertIn('clean_env = safe.cpu_helper_environment()', runner)
        self.assertIn('env=clean_env', runner)
        self.assertIn('env = cpu_helper_environment()', safety)
        self.assertIn('env=env', safety)

    def test_collect_reports_bounded_contract_without_database(self):
        namespace = {'stamp': lambda: 'cpu', 'SOURCE': Path('/source'),
                     'DEADLINE_UNIX': DEADLINE}
        load_function('runner.py', 'measurement_configuration', namespace)
        collect = load_function('runner.py', 'collect', namespace)
        with tempfile.TemporaryDirectory() as temporary:
            report = collect(Path(temporary), {'run_id': 'w4pre_unit', 'state': 'running', 'mode': 'preflight'})
        self.assertFalse(report['until_manual_stop'])
        self.assertEqual(report['deadline_unix'], DEADLINE)
        self.assertFalse(report['stop_on_target'])

    def test_deadline_window_requests_stop_and_keeps_fixed_plus900_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            (data / 'training').mkdir()
            stop = Mock(side_effect=lambda folder, reason: (folder / 'training/stop.flag').write_text(reason))
            namespace = {'DEADLINE_UNIX': DEADLINE, 'stop': stop,
                         'SaveGraceExpired': type('SaveGraceExpired', (RuntimeError,), {})}
            check = load_function('runner.py', 'enforce_stop_window', namespace)
            child = Mock()
            child.poll.return_value = None
            state = {}
            check(data, state, child, now=DEADLINE - 1)
            stop.assert_not_called()
            check(data, state, child, now=DEADLINE)
            stop.assert_called_once_with(data, 'user_deadline_reached')
            self.assertEqual(state['deadline_requested_ns'], DEADLINE * 1_000_000_000)
            check(data, state, child, now=DEADLINE + 899)
            with self.assertRaisesRegex(RuntimeError, 'deadline GPU save grace'):
                check(data, state, child, now=DEADLINE + 900)
            child.poll.return_value = 0
            check(data, state, child, now=DEADLINE + 900)

    def test_error_wait_budget_cannot_extend_save_limit(self):
        budget = load_function('runner.py', 'error_wait_budget', {'DEADLINE_UNIX': DEADLINE})
        self.assertEqual(budget({}, now=DEADLINE - 1), 120)
        self.assertEqual(budget({}, now=DEADLINE + 850), 50)
        self.assertEqual(budget({}, now=DEADLINE + 900), 0)
        state = {'stop_requested_ns': (DEADLINE - 100) * 1_000_000_000}
        self.assertEqual(budget(state, now=DEADLINE + 750), 50)
        self.assertEqual(budget(state, now=DEADLINE + 800), 0)

    def test_actual_error_handler_skips_or_bounds_wait_without_running_controller(self):
        run = next(n for n in source_tree('runner.py').body if isinstance(n, ast.FunctionDef) and n.name == 'run')
        handler = next(n for n in ast.walk(run) if isinstance(n, ast.ExceptHandler)
                       and isinstance(n.type, ast.Name) and n.type.id == 'Exception' and n.name == 'exc')
        function = ast.parse('def isolated_handler(exc, state, data, child):\n    pass\n').body[0]
        function.body = handler.body
        module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        expired = type('SaveGraceExpired', (RuntimeError,), {})
        namespace = {'SaveGraceExpired': expired, 'error_wait_budget': Mock(return_value=50),
                     'stop': Mock(), 'safe': SimpleNamespace(atomic_json=Mock())}
        exec(compile(module, 'isolated_exception_handler', 'exec'), namespace)
        child = Mock()
        namespace['isolated_handler'](expired('limit'), {}, Path('/cpu/test'), child)
        child.wait.assert_not_called()
        namespace['isolated_handler'](RuntimeError('failure'), {}, Path('/cpu/test'), child)
        child.wait.assert_called_once_with(timeout=50)

    def test_guard_deadline_and_heartbeat_error_graces(self):
        state = {'deadline_unix': 1000, 'heartbeat_ns': 999_000_000_000}
        self.assertIsNone(guard_decision(state, 999, True)['stop_reason'])
        state['heartbeat_ns'] = 1000_000_000_000
        decision = guard_decision(state, 1000, True)
        self.assertEqual(decision['stop_reason'], 'user_deadline_reached')
        self.assertFalse(decision['cleanup'])
        for now, expected in [(1899, False), (1900, True)]:
            state['heartbeat_ns'] = now * 1_000_000_000
            self.assertEqual(guard_decision(state, now, True)['cleanup'], expected)
        state = {'deadline_unix': DEADLINE, 'heartbeat_ns': 1000_000_000_000}
        self.assertIsNone(guard_decision(state, 1120, True)['stop_reason'])
        first = guard_decision(state, 1121, True)
        self.assertEqual(first['stop_reason'], 'supervisor_missing_or_stale')
        self.assertFalse(first['cleanup'])
        self.assertFalse(guard_decision(state, 1240, True, first['failure_at'])['cleanup'])
        self.assertTrue(guard_decision(state, 1241, True, first['failure_at'])['cleanup'])
        missing = guard_decision(state, 1000, False)
        self.assertEqual(missing['stop_reason'], 'supervisor_missing_or_stale')
        self.assertTrue(guard_decision(state, 1120, False, missing['failure_at'])['cleanup'])

    def test_mode_gate_and_receipts_statically_without_calling_run_or_preflight(self):
        tree = source_tree('runner.py')
        run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
        first = run.body[0]
        self.assertIsInstance(first, ast.If)
        condition = compile(ast.Expression(first.test), 'run_mode_gate_only', 'eval')
        for mode in ['target', 'handoff', 'restart', 'manual']:
            self.assertTrue(eval(condition, {'mode': mode, 'stop_on_target': False}))
        self.assertTrue(eval(condition, {'mode': 'preflight', 'stop_on_target': True}))
        self.assertFalse(eval(condition, {'mode': 'preflight', 'stop_on_target': False}))
        self.assertIsInstance(first.body[0], ast.Raise)
        code = ast.unparse(tree)
        self.assertIn("choices=['preflight', 'guard', 'report', 'sample']", code)
        self.assertIn("['micro_batch_size'] != 8", code)
        self.assertIn("['gradient_accumulation_steps'] != 18", code)
        self.assertIn("['global_batch_size'] != 576", code)
        self.assertIn("steps != {'policy': 15, 'ctrl': 3, 'reward': 3}", code)
        self.assertIn('deadline_unix=DEADLINE_UNIX, until_manual_stop=False', code)
        self.assertIn('automatic_restart=False', code)
        self.assertIn('enforce_stop_window(data, state, child)', code)
        pre = next(n for n in source_tree('gpu47_safety.py').body
                   if isinstance(n, ast.FunctionDef) and n.name == 'preflight')
        self.assertIn('time.time() >= DEADLINE_UNIX', ast.unparse(pre.body[0]))

    def test_stop_request_checks_uid_run_and_supervisor_birth_with_mocks(self):
        with tempfile.TemporaryDirectory() as temporary:
            master = Path(temporary)
            data = master / 'preflight'
            (data / 'training').mkdir(parents=True)
            identity = {'pid': 123, 'uid': 1004, 'birth_ticks': 456}
            state = {'uid': 1004, 'run_id': 'w4pre_unit', 'mode': 'preflight',
                     'until_manual_stop': False, 'deadline_unix': DEADLINE,
                     'state': 'running', 'supervisor': dict(identity)}
            safe = SimpleNamespace(MASTER=master, DEADLINE_UNIX=DEADLINE,
                                   proc_identity=Mock(return_value=dict(identity)),
                                   same_process=Mock(return_value=True))
            fake_runner = SimpleNamespace(load=Mock(return_value=state), stop=Mock())
            function = load_function('request_stop.py', 'request',
                                     {'os': SimpleNamespace(getuid=lambda: 1004),
                                      'safe': safe, 'runner': fake_runner})
            result = function(data, 'w4pre_unit')
            self.assertTrue(result['stop_requested'])
            fake_runner.stop.assert_called_once_with(data, 'user_requested_stop')
            fake_runner.stop.reset_mock()
            with self.assertRaisesRegex(RuntimeError, 'state identity'):
                function(data, 'another_run')
            safe.proc_identity.return_value['birth_ticks'] = 457
            with self.assertRaisesRegex(RuntimeError, 'birth identity'):
                function(data, 'w4pre_unit')
            fake_runner.stop.assert_not_called()


if __name__ == '__main__':
    unittest.main()
