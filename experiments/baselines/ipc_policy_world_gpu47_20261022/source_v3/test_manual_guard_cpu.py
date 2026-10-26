"""Independent standard-library tests; no controller, safety, GPU or process work."""
import importlib.util
from pathlib import Path
import unittest


SOURCE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('manual_guard_under_test', SOURCE/'deadline_guard.py')
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


def running(now=1000, **changes):
    state = {'state': 'running', 'until_manual_stop': True,
             'heartbeat_ns': int(now * 1e9)}
    state.update(changes)
    return state


class ManualGuardCPU(unittest.TestCase):
    def test_healthy_manual_run_has_no_deadline_even_with_legacy_fields(self):
        for now in (1000, 1000000, 2000000000):
            state = running(now, deadline_unix=1, short_deadline_unix=1,
                            completed_epochs=30001, global_step=30001,
                            target_reached=True)
            result = guard.guard_decision(state, now, True)
            self.assertIsNone(result['stop_reason'])
            self.assertIsNone(result['cleanup_at'])
            self.assertFalse(result['cleanup'])

    def test_user_stop_keeps_exact_first_request_plus900(self):
        first_request = 1000
        for now, cleanup in ((1000, False), (1899.999, False), (1900, True), (2100, True)):
            state = running(now, stop_requested_ns=first_request * 1000000000)
            result = guard.guard_decision(state, now, True)
            self.assertEqual(result['stop_reason'], 'user_requested_stop')
            self.assertEqual(result['cleanup_at'], 1900)
            self.assertEqual(result['cleanup'], cleanup)
            self.assertIsNone(result['failure_at'])

    def test_missing_supervisor_immediately_requests_stop_then120(self):
        result = guard.guard_decision(running(), 1000, False)
        self.assertEqual(result['failure_at'], 1000)
        self.assertEqual(result['stop_reason'], 'supervisor_missing_or_stale')
        self.assertFalse(result['cleanup'])
        for now, cleanup in ((1119.999, False), (1120, True)):
            follow = guard.guard_decision(running(now), now, False, result['failure_at'])
            self.assertEqual(follow['cleanup'], cleanup)
            self.assertEqual(follow['cleanup_at'], 1120)

    def test_heartbeat_boundary_is_strictly_over120_seconds(self):
        state = running()
        self.assertIsNone(guard.guard_decision(state, 1120, True)['failure_at'])
        result = guard.guard_decision(state, 1120.001, True)
        self.assertEqual(result['failure_at'], 1120.001)
        self.assertFalse(result['cleanup'])

    def test_missing_or_invalid_heartbeat_fails_closed(self):
        for heartbeat in (None, 0, -1, '1000', True, float('nan')):
            result = guard.guard_decision(running(heartbeat_ns=heartbeat), 1000, True)
            self.assertEqual(result['failure_at'], 1000)
            self.assertEqual(result['cleanup_at'], 1120)

    def test_first_failure_is_latched_after_heartbeat_recovers(self):
        first = guard.guard_decision(running(), 1000, False)
        follow = guard.guard_decision(running(1120), 1120, True, first['failure_at'])
        self.assertEqual(follow['failure_at'], 1000)
        self.assertEqual(follow['cleanup_at'], 1120)
        self.assertTrue(follow['cleanup'])

    def test_explicit_error_uses120_seconds_with_fresh_heartbeat(self):
        for changes in ({'state': 'stopping_error'}, {'error': 'training exited1'}):
            first = guard.guard_decision(running(**changes), 1000, True)
            self.assertEqual(first['stop_reason'], 'supervisor_error')
            self.assertEqual(first['failure_at'], 1000)
            self.assertEqual(first['cleanup_at'], 1120)
            self.assertFalse(first['cleanup'])
            end = guard.guard_decision(running(1120, **changes), 1120, True, first['failure_at'])
            self.assertTrue(end['cleanup'])

    def test_persisted_error_anchor_and_future_anchor_cannot_extend_window(self):
        prior = guard.guard_decision(running(state='stopping_error', error_started_ns=950000000000), 1000, True)
        self.assertEqual(prior['failure_at'], 950)
        self.assertEqual(prior['cleanup_at'], 1070)
        future = guard.guard_decision(running(state='stopping_error', error_started_ns=1100000000000), 1000, True)
        self.assertEqual(future['failure_at'], 1000)
        self.assertEqual(future['cleanup_at'], 1120)

    def test_late_failure_never_adds120_after_manual_save_window(self):
        state = running(1890, stop_requested_ns=1000000000000, state='stopping_error')
        first = guard.guard_decision(state, 1890, True)
        self.assertEqual(first['failure_at'], 1890)
        self.assertEqual(first['cleanup_at'], 1900)
        self.assertEqual(first['stop_reason'], 'user_requested_stop')
        state['heartbeat_ns'] = 1900000000000
        self.assertTrue(guard.guard_decision(state, 1900, True, first['failure_at'])['cleanup'])

    def test_earlier_error_window_is_not_replaced_by_later_user_stop(self):
        first = guard.guard_decision(running(), 1000, False)
        state = running(1050, stop_requested_ns=1050000000000)
        result = guard.guard_decision(state, 1050, True, first['failure_at'])
        self.assertEqual(result['cleanup_at'], 1120)
        self.assertFalse(result['cleanup'])
        self.assertTrue(guard.guard_decision(running(1120, stop_requested_ns=1050000000000),
                                             1120, True, first['failure_at'])['cleanup'])

    def test_error_during_manual_save_can_only_shorten_never_extend(self):
        state = running(1100, stop_requested_ns=1000000000000, error='training error')
        result = guard.guard_decision(state, 1100, True)
        self.assertEqual(result['cleanup_at'], 1220)
        self.assertEqual(result['stop_reason'], 'supervisor_error')

    def test_all_manual_terminal_states_are_quiet_even_after_expired_windows(self):
        for terminal in ('completed', 'stopped_by_user', 'failed'):
            state = running(5000, state=terminal, stop_requested_ns=1000000000000,
                            error='old error', deadline_unix=1)
            result = guard.guard_decision(state, 5000, False, failure_at=1000)
            self.assertIsNone(result['stop_reason'])
            self.assertIsNone(result['cleanup_at'])
            self.assertFalse(result['cleanup'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
