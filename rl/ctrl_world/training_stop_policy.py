"""CPU-only run termination and one-time milestone publication policy.

The learning-rate horizon belongs to the unchanged learning methods. Selecting
an explicit deadline changes only when the outer run stops, not that horizon.
ThresholdTracker remains latched after success; live evaluation freshness is
therefore tracked independently from the milestone tracker.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from numbers import Integral
import time


def _flag(environment, name, default):
    value = environment.get(name, "1" if default else "0")
    if value not in ("0", "1"):
        raise ValueError(name + " must be 0 or 1")
    return value == "1"


@dataclass(frozen=True)
class TrainingStopPolicy:
    configured_stop_on_target: bool = True
    until_deadline: bool = False
    deadline_unix_seconds: float | None = None

    def __post_init__(self):
        if not isinstance(self.configured_stop_on_target, bool) or not isinstance(self.until_deadline, bool):
            raise ValueError("Stop policy flags must be booleans")
        deadline = self.deadline_unix_seconds
        if deadline is not None and (isinstance(deadline, bool) or not isinstance(deadline, (int, float))
                                     or not math.isfinite(deadline) or deadline <= 0):
            raise ValueError("Deadline must be a positive finite Unix timestamp")
        if self.until_deadline and deadline is None:
            raise ValueError("Deadline mode requires an explicit deadline")

    @classmethod
    def from_environment(cls, environment):
        raw_deadline = environment.get("ACCERL_SOFT_STOP_AT")
        deadline = float(raw_deadline) if raw_deadline else None
        if deadline is not None and (not math.isfinite(deadline) or deadline <= 0):
            raise ValueError("ACCERL_SOFT_STOP_AT must be a positive finite Unix timestamp")
        until_deadline = _flag(environment, "ACCERL_RUN_UNTIL_DEADLINE", False)
        if until_deadline and deadline is None:
            raise ValueError("Deadline mode requires explicit ACCERL_SOFT_STOP_AT")
        return cls(_flag(environment, "ACCERL_STOP_ON_TARGET_85", True), until_deadline, deadline)

    @property
    def stop_on_target(self):
        return self.configured_stop_on_target and not self.until_deadline

    def deadline_reached(self, now=None):
        return (self.deadline_unix_seconds is not None
                and (time.time() if now is None else now) >= self.deadline_unix_seconds)

    def target_stop_active(self, target_reached):
        return bool(self.stop_on_target and target_reached)

    def should_continue(self, *, global_step, train_iters, target_reached,
                        requested_stop, completed_epochs=0, measure_stop_epochs=0,
                        now=None):
        if requested_stop or self.deadline_reached(now) or self.target_stop_active(target_reached):
            return False
        if self.until_deadline:
            # Historical measurement caps and the LR plan are not run deadlines.
            return True
        return (global_step < train_iters
                and (measure_stop_epochs <= 0 or completed_epochs < measure_stop_epochs))

    def stop_reason(self, *, global_step, train_iters, target_reached,
                    requested_stop, completed_epochs=0, measure_stop_epochs=0,
                    now=None):
        if self.deadline_reached(now):
            return "user_deadline_reached"
        if self.target_stop_active(target_reached):
            return "consecutive_eval_85"
        if requested_stop:
            return "scheduled_stop"
        if not self.until_deadline and measure_stop_epochs > 0 and completed_epochs >= measure_stop_epochs:
            return "training_plan_completed"
        if not self.until_deadline and global_step >= train_iters:
            return "training_plan_completed"
        return "unexpected_outer_loop_exit"

    def metadata(self):
        return {"stop_on_target": self.stop_on_target,
                "configured_stop_on_target": self.configured_stop_on_target,
                "until_deadline": self.until_deadline,
                "deadline_unix_seconds": self.deadline_unix_seconds,
                "deadline_utc": datetime.fromtimestamp(self.deadline_unix_seconds, timezone.utc).isoformat()
                    if self.deadline_unix_seconds is not None else None,
                "stop_boundary": "completed synchronized epoch or pre-update replay-fill boundary; preserve final weights",
                "learning_rate_horizon_unchanged": True}


class EvaluationMilestones:
    """Separate live episode observations from a latched attainment tracker."""
    def __init__(self):
        self.last_watermark = 0
        self.published = set()
        self.target_announced = False

    def observe_watermark(self, watermark):
        if (isinstance(watermark, bool) or not isinstance(watermark, Integral)
                or watermark < self.last_watermark):
            raise ValueError("Live evaluation watermark must be a nondecreasing integer")
        fresh = int(watermark) - self.last_watermark
        self.last_watermark = int(watermark)
        return fresh

    def new_marks(self, tracker):
        result = []
        for label, value in (("First", tracker.first_qualifying),
                             ("Consecutive_Two", tracker.consecutive_qualifying)):
            if value is not None and label not in self.published:
                self.published.add(label)
                result.append((label, value))
        return result

    def announce_target(self, target_reached):
        if not target_reached or self.target_announced:
            return False
        self.target_announced = True
        return True
