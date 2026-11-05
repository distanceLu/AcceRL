"""Error-stop regression: real wrapper loop/files, fake clock and child only.

No torch, CUDA, process signals, SSH, or training data are used by these tests.
"""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import training_entry as q


RID = 'weaverintegrated_error_fixture'
ERROR = "RuntimeError('ctrl successful-update watchdog expired')"


def flag(reason=ERROR, stamp=1000_000_000_000):
    return {'run_id': RID, 'uid': 1004, 'reason': reason, 'requested_ns': stamp}


def error_state():
    return {'run_id': RID, 'uid': 1004, 'state': 'stopping_error',
            'error': ERROR, 'error_started_ns': 1000_000_000_000}


class StopEvidenceTests(unittest.TestCase):
    def test_runner_error_is_not_manual_and_original_flag_bytes_stay_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp); (data / 'training').mkdir()
            target = data / 'training/stop.flag'; raw = json.dumps(flag()).encode()
            target.write_bytes(raw)
            self.assertEqual(q.read_stop_evidence(data, RID, error_state())['kind'], 'runner_error')
            with self.assertRaises(ValueError): q.read_manual_stop(data, RID)
            self.assertEqual(target.read_bytes(), raw)

    def test_write_order_race_is_bounded_nonmanual_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp); (data / 'training').mkdir()
            (data / 'training/stop.flag').write_text(json.dumps(flag()))
            stale = dict(error_state(), state='running', error=None)
            self.assertEqual(q.read_stop_evidence(data, RID, stale)['kind'], 'same_run_nonmanual_stop')
            self.assertEqual(q.read_stop_evidence(data, RID, error_state())['kind'], 'runner_error')

    def test_foreign_and_malformed_flags_rejected_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp); (data / 'training').mkdir(); target = data / 'training/stop.flag'
            for mutation in ({'run_id': RID + '_foreign'}, {'uid': 0}, {'uid': True},
                             {'requested_ns': True}, {'requested_ns': -1}, {'reason': ''}):
                value = dict(flag(), **mutation); raw = json.dumps(value).encode(); target.write_bytes(raw)
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    q.read_stop_evidence(data, RID, error_state())
                self.assertEqual(target.read_bytes(), raw)

    def test_manual_still_exact(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp); (data / 'training').mkdir(); target = data / 'training/stop.flag'
            self.assertIsNone(q.read_stop_evidence(data, RID))
            value = flag('user_requested_stop'); target.write_text(json.dumps(value))
            self.assertEqual(q.read_manual_stop(data, RID), value)
            self.assertEqual(q.read_stop_evidence(data, RID)['kind'], 'manual')


class StopDeadlineTests(unittest.TestCase):
    def deadlines(self, ev, state, now=1000, mono=10, prior=None):
        return q.stop_deadlines(ev, state, now_ns=int(now * 1e9), now_monotonic=mono,
                                post_stop_grace_seconds=1800, prior=prior)

    def test_error_reserves_120_for_work_and_separate_90_for_teardown(self):
        result = self.deadlines({'kind': 'runner_error', 'flag': flag()}, error_state())
        self.assertEqual(result['work_deadline'], 130)
        self.assertEqual(result['supervisor_deadline'], 220)

    def test_reobservations_or_clock_rollback_never_renew_error_window(self):
        ev = {'kind': 'same_run_nonmanual_stop', 'flag': flag()}
        first = self.deadlines(ev, {})
        changed = dict(error_state(), error_started_ns=1100_000_000_000)
        second = self.deadlines(ev, changed, now=1100, mono=110, prior=first)
        self.assertEqual(second['work_deadline'], first['work_deadline'])
        rolled = self.deadlines(ev, changed, now=990, mono=120, prior=second)
        self.assertEqual(rolled['supervisor_deadline'], first['supervisor_deadline'])

    def test_manual_900_and_error_120_take_min_not_sum(self):
        ev = {'kind': 'manual', 'flag': flag('user_requested_stop')}
        first = self.deadlines(ev, {})
        state = dict(error_state(), error_started_ns=1890_000_000_000,
                     stop_requested_ns=1000_000_000_000)
        result = self.deadlines(ev, state, now=1890, mono=900, prior=first)
        self.assertEqual(result['work_deadline'], 910)  # Only ten seconds of GPU-save time remain.
        self.assertEqual(result['supervisor_deadline'], 1000)

    def test_expired_error_deadline_has_no_new_grace(self):
        result = self.deadlines({'kind': 'runner_error', 'flag': flag()}, error_state(), now=1300, mono=310)
        self.assertLess(result['supervisor_deadline'], 310)

    def test_future_flag_cannot_allocate_extra_time(self):
        future = flag(stamp=9999_000_000_000)
        result = self.deadlines({'kind': 'same_run_nonmanual_stop', 'flag': future}, {})
        self.assertEqual(result['work_deadline'], 130)

    def test_error_before_flag_and_manual_flag_during_error_are_bounded(self):
        absent = self.deadlines(None, error_state())
        manual = self.deadlines({'kind': 'manual', 'flag': flag('user_requested_stop')}, error_state())
        self.assertEqual(absent['work_deadline'], 130)
        self.assertEqual(manual['work_deadline'], 130)


class WrapperLoopTests(unittest.TestCase):
    def exercise(self, *, finishes_after=125, race=False, manual_then_error=False, foreign=False,
                 emergency_race=False):
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary).resolve(); source = temp / 'source'; source.mkdir()
            root = temp / 'root'; root.mkdir(); data = root / 'formal_v3'; data.mkdir()
            (data / 'training').mkdir()
            identity = {'pid': 1234, 'uid': 1004, 'birth_ticks': 4567}
            clock = SimpleNamespace(elapsed=0.)
            clock.monotonic = lambda: 100 + clock.elapsed
            clock.time_ns = lambda: int((1000 + clock.elapsed) * 1e9)
            clock.sleep = lambda seconds: setattr(clock, 'elapsed', clock.elapsed + seconds)
            state = dict(error_state(), candidate='weaver_integrated_v1', source_path=str(source),
                         experiment_root=str(root), mode='manual', until_manual_stop=True,
                         world_backend_config={'backend_options': {'fresh': dict(q.FRESH_OPTIONS)}},
                         stop_on_target=False, deadline_unix=None, supervisor=identity,
                         manual_completion=None, checkpoint_cpu_audit=None)
            stop = flag('user_requested_stop' if manual_then_error else ERROR)
            if manual_then_error:
                stop['requested_ns'] = 110_000_000_000
                state['stop_requested_ns'] = 110_000_000_000
            if foreign: stop['run_id'] += '_foreign'
            original_flag = json.dumps(stop).encode()
            if not emergency_race:
                (data / 'training/stop.flag').write_bytes(original_flag)
            else:
                state.update(state='running', error=None)

            def save_state():
                value = copy.deepcopy(state)
                if race and clock.elapsed < 3:
                    value.update(state='running', error=None)
                    value.pop('error_started_ns', None)
                (data / 'runnerstatus.json').write_text(json.dumps(value))

            save_state()
            (data / 'reportstatus.json').write_text(json.dumps({'run_id': RID, 'state': 'failed', 'source': str(source)}))

            class Child:
                pid = 1234
                returncode = None
                def poll(self):
                    if self.returncode is None and clock.elapsed >= finishes_after:
                        self.returncode = 1
                        state.update(state='failed', training_stopped=True, cleanup={'survivors': []})
                    save_state()
                    return self.returncode
                def wait(self, timeout):
                    if self.poll() is not None: return self.returncode
                    clock.sleep(min(timeout, max(0, finishes_after - clock.elapsed)))
                    if self.poll() is None: raise subprocess.TimeoutExpired('fixture', timeout)
                    return self.returncode
            child = Child()
            cleanup = Mock()
            def clean(*args):
                cleanup(*args)
                child.returncode = -15
                state.update(state='failed', cleanup={'survivors': []})
                save_state()
                return {'signals': [{'fixture': True}], 'survivors': []}
            def request_racing_error(*args):
                state.update(error_state())
                (data / 'training/stop.flag').write_bytes(original_flag)
                save_state()
                raise subprocess.CalledProcessError(1, ['request_stop.py'])
            request = Mock(side_effect=request_racing_error if emergency_race else
                           AssertionError('Existing error/foreign flag must never be replaced'))
            check_cost = Mock(side_effect=RuntimeError('fixture wrapper observation failure')) if emergency_race else lambda *args: True
            env = {'EXPERIMENT_QUEUE_JOB_ID': 'cpu-error-regression', 'EXPERIMENT_QUEUE_MANUAL': '1',
                   'EXPERIMENT_QUEUE_TIMEOUT_SECONDS': '0', 'CUDA_VISIBLE_DEVICES': ','.join(q.GPUS),
                   'CUDA_MPS_PIPE_DIRECTORY': q.PIPE}
            args = SimpleNamespace(source=source, root=root, manifest_sha256='a' * 64,
                                   post_stop_grace_seconds=1800, native_gpu_preflight=True, validate_only=False)
            with contextlib.ExitStack() as stack:
                for name, replacement in {
                    'validate': lambda *args: {}, 'verify_manifest': lambda *args: {},
                    'check_cost_scheduler_state': check_cost, 'wait_idle': lambda *args: None,
                    'run_native_preflight': lambda *args, **kwargs: {'cpu_fixture': True},
                    'synchronized_epochs': lambda *args: 30, 'proc_identity': lambda pid: identity,
                    'marked_children': lambda *args: [identity] if child.poll() is None else [],
                    'scoped_cleanup': clean, 'request_stop': request, 'time': clock,
                }.items(): stack.enter_context(patch.object(q, name, replacement))
                stack.enter_context(patch.object(q.sys, 'platform', 'linux'))
                stack.enter_context(patch.object(q.os, 'getuid', return_value=1004, create=True))
                stack.enter_context(patch.object(q.subprocess, 'Popen', return_value=child))
                stack.enter_context(patch.object(q.signal, 'signal'))
                stack.enter_context(patch.dict(os.environ, env))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                result = q.run(args)
            final = json.loads((root / 'queue_receipt.json').read_text())
            self.assertEqual(result, 1)
            self.assertEqual(final['state'], 'failed')
            self.assertFalse(final['acceptance']['passed'])
            self.assertEqual((data / 'training/stop.flag').read_bytes(), original_flag)
            # Assert only the count; failure output must not print launch env.
            self.assertEqual(request.call_count, 1 if emergency_race else 0)
            return final, cleanup, clock.elapsed

    def test_real_wrapper_loop_waits_full_error_work_then_observes_runner_cleanup(self):
        final, cleanup, elapsed = self.exercise()
        self.assertEqual(elapsed, 125)
        cleanup.assert_not_called()
        self.assertEqual(final['runner_exit_code'], 1)
        self.assertNotIn('wrapper_error', final)
        self.assertEqual(final['runner_error'], ERROR)
        self.assertEqual(final['runner_cleanup'], {'survivors': []})
        self.assertEqual(final['stop_observation']['kind'], 'runner_error')
        self.assertNotIn('final_audit_unavailable', final['acceptance']['reasons'])

    def test_wrapper_loop_tolerates_flag_before_runner_state_write(self):
        final, cleanup, elapsed = self.exercise(race=True)
        cleanup.assert_not_called()
        self.assertEqual(elapsed, 125)
        self.assertEqual(final['stop_trigger']['reason'], 'same_run_nonmanual_stop')
        self.assertEqual(final['stop_observation']['kind'], 'runner_error')

    def test_hung_runner_is_cleaned_after_bounded_envelope_without_reset(self):
        final, cleanup, elapsed = self.exercise(finishes_after=10000)
        cleanup.assert_called_once()
        self.assertEqual(elapsed, 210)
        self.assertIn('bounded stop/teardown', final['wrapper_error'])

    def test_error_at_end_of_manual_save_has_no_added_120_seconds(self):
        final, cleanup, elapsed = self.exercise(finishes_after=10000, manual_then_error=True)
        cleanup.assert_called_once()
        self.assertEqual(elapsed, 100)  # Ten remaining save seconds plus ninety teardown.
        self.assertFalse(final['acceptance']['passed'])

    def test_foreign_flag_never_requests_manual_replacement(self):
        final, cleanup, elapsed = self.exercise(foreign=True)
        cleanup.assert_called_once()
        self.assertLess(elapsed, 1)
        self.assertIn('UID/run', final['wrapper_error'])

    def test_wrapper_fault_racing_runner_error_preserves_existing_error_and_waits(self):
        final, cleanup, elapsed = self.exercise(emergency_race=True)
        cleanup.assert_not_called()
        self.assertEqual(elapsed, 125)
        self.assertIn('fixture wrapper observation failure', final['wrapper_error'])
        self.assertEqual(final['runner_error'], ERROR)
        self.assertNotIn('emergency_stop_error', final)
        self.assertEqual(final['stop_evidence'], flag())


if __name__ == '__main__':
    unittest.main()
