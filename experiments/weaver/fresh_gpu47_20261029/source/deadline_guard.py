"""Pure CPU decisions for a manual-stop run, with no wall-clock run deadline.

The user's first stop request fixes its save window at request + 900 seconds.
The first observed supervisor failure fixes an independent 120 second window.
Cleanup uses the earlier endpoint; neither window is reset or appended to the
other. This module never reads environment variables or signals a process.
"""
from __future__ import annotations

import math


SAVE_GRACE_SECONDS = 900
ERROR_GRACE_SECONDS = 120
HEARTBEAT_STALE_SECONDS = 120
TERMINAL = frozenset({'completed', 'stopped_by_user', 'failed'})


def _seconds_from_ns(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return value / 1e9


def guard_decision(state, now, supervisor_alive, failure_at=None):
    """Return a stop/cleanup decision; retain ``failure_at`` between polls.

    ``error_started_ns`` is optional: the supervisor may persist it when entering
    ``stopping_error`` so both processes use the same failure origin. An invalid
    or future error timestamp never postpones the first observed failure.
    Legacy deadline fields are deliberately unused in this manual controller.
    """
    if state.get('state') in TERMINAL:
        return {'failure_at': failure_at, 'stop_reason': None, 'cleanup': False,
                'cleanup_at': None, 'window_reason': None}

    heartbeat = _seconds_from_ns(state.get('heartbeat_ns'))
    stale = (not supervisor_alive or heartbeat is None
             or now - heartbeat > HEARTBEAT_STALE_SECONDS)
    explicit_error = state.get('state') == 'stopping_error' or bool(state.get('error'))
    if (stale or explicit_error) and failure_at is None:
        error_started = _seconds_from_ns(state.get('error_started_ns')) if explicit_error else None
        failure_at = min(now, error_started) if error_started is not None else now

    windows = []
    stopped = _seconds_from_ns(state.get('stop_requested_ns'))
    if stopped is not None:
        windows.append((stopped + SAVE_GRACE_SECONDS, 'user_requested_stop'))
    if failure_at is not None:
        reason = 'supervisor_error' if explicit_error else 'supervisor_missing_or_stale'
        windows.append((failure_at + ERROR_GRACE_SECONDS, reason))
    if not windows:
        return {'failure_at': failure_at, 'stop_reason': None, 'cleanup': False,
                'cleanup_at': None, 'window_reason': None}

    cleanup_at, reason = min(windows, key=lambda item: item[0])
    return {'failure_at': failure_at, 'stop_reason': reason,
            'cleanup': now >= cleanup_at, 'cleanup_at': cleanup_at,
            'window_reason': reason}
