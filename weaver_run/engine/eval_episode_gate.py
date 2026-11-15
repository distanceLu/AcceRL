"""Episode boundary admission; no resources or episode leases are held."""
import time
import os

MAX_WAIT_SECONDS = 360.0
POLL_SECONDS = 0.1


def admission_decision(enabled, remaining, waited):
    reason = ('disabled' if not enabled else 'refresh_finished' if remaining == 0
              else 'guard_timeout' if waited >= MAX_WAIT_SECONDS else 'refresh_pending')
    return dict(admitted=reason != 'refresh_pending', reason=reason,
                waited_seconds=max(0., waited), guard_triggered=reason == 'guard_timeout')


def wait_for_episode(query, stop, *, clock=time.monotonic):
    started = clock()
    while not stop.is_set():
        stop_file = os.environ.get('ACCERL_STOP_FILE')
        if stop_file and os.path.exists(stop_file):
            return False
        decision = query(clock() - started)
        if decision.get('stopped'):
            return False
        if decision['admitted']:
            return not stop.is_set()
        if stop.wait(POLL_SECONDS):
            break
    return False
