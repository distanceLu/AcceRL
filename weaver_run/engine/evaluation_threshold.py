"""Track a success threshold without changing the TensorBoard evaluation metric.

The caller owns the evaluation window and supplies the same success rate used
by its existing TensorBoard tag. Polling that window again is not another
evaluation: every observation needs a new ID and evidence of fresh episodes.
This module imports no training, CUDA, NumPy, or TensorBoard dependencies.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from numbers import Integral
import os
from pathlib import Path
import tempfile
import time


def _count(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _json_copy(value):
    # Besides making independent snapshots, reject unserializable values and
    # NaNs in metadata before mutating the tracker.
    return json.loads(json.dumps(value, allow_nan=False))


def _iso_ns(value):
    seconds, ns = divmod(value, 1_000_000_000)
    return datetime.fromtimestamp(seconds, timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S"
    ) + f".{ns:09d}Z"


class ThresholdTracker:
    """Record first attainment and stop at consecutive qualifying evaluations.

    ``evaluation_watermark`` is a cumulative count of completed evaluation
    episodes; it must increase between accepted observations. Alternatively,
    ``fresh_count`` can be the caller's independently checked count of new
    episodes in this evaluation. Supplying both performs both checks. They can
    be keyword arguments or fields in ``metadata``. Neither field changes the
    caller's original window, averaging method, or evaluation sample count.

    Identity and clock errors raise ValueError, with no state mutation. Invalid
    rates, empty windows, missing fresh data, and initialization-only policy
    step 0 are skipped: they do not qualify or reset a streak. A valid fresh
    evaluation below the threshold resets only the consecutive streak.
    """

    def __init__(self, target=0.85, run_started_ns=None, required_consecutive=2,
                 min_samples=1):
        target = float(target)
        if not math.isfinite(target) or not 0 <= target <= 1:
            raise ValueError("target must be finite and between 0 and 1")
        self.target = target
        self.run_started_ns = _count(
            time.time_ns() if run_started_ns is None else run_started_ns,
            "run_started_ns",
        )
        self.required_consecutive = _count(
            required_consecutive, "required_consecutive", minimum=1,
        )
        self.min_samples = _count(min_samples, "min_samples", minimum=1)
        self.streak = 0
        self.first_qualifying = None
        self.consecutive_qualifying = None
        self.latest_evaluation = None
        self.last_evaluation_id = None
        self.last_completed_ns = None
        self.last_evaluation_watermark = None

    @property
    def should_stop(self):
        return self.consecutive_qualifying is not None

    def update(self, evaluation_id, rate, completed_ns, sample_count, policy_step,
               *, fresh_count=None, evaluation_watermark=None,
               evaluation_started_ns=None, metadata=None):
        """Observe one completed evaluation and return a JSON-safe decision.

        The second attainment snapshot is locked after success. Later calls
        return ``reason='already_achieved'`` without replacing either timestamp.
        ``accepted`` means eligible to count as a real threshold evaluation;
        ``qualified`` means its unchanged rate is >= the target.
        """
        if self.should_stop:
            return self._decision(False, False, "already_achieved")

        evaluation_id = _count(evaluation_id, "evaluation_id")
        completed_ns = _count(completed_ns, "completed_ns")
        sample_count = _count(sample_count, "sample_count")
        policy_step = _count(policy_step, "policy_step")
        metadata = _json_copy({} if metadata is None else metadata)
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a JSON object")
        if fresh_count is None:
            fresh_count = metadata.get("fresh_count")
        if evaluation_watermark is None:
            evaluation_watermark = metadata.get("evaluation_watermark")
        if fresh_count is not None:
            fresh_count = _count(fresh_count, "fresh_count")
        if evaluation_watermark is not None:
            evaluation_watermark = _count(evaluation_watermark, "evaluation_watermark")
        if evaluation_started_ns is not None:
            evaluation_started_ns = _count(evaluation_started_ns, "evaluation_started_ns")
            if not self.run_started_ns <= evaluation_started_ns <= completed_ns:
                raise ValueError("evaluation start must be between run start and completion")
        if self.last_evaluation_id is not None and evaluation_id <= self.last_evaluation_id:
            raise ValueError("evaluation_id must strictly increase; duplicate evaluations cannot count")
        if completed_ns < self.run_started_ns:
            raise ValueError("evaluation completion cannot precede run start")
        if self.last_completed_ns is not None and completed_ns < self.last_completed_ns:
            raise ValueError("evaluation completion time cannot move backwards")

        try:
            rate = float(rate)
        except (ValueError, TypeError, OverflowError):
            rate = math.nan
        valid_rate = math.isfinite(rate) and 0 <= rate <= 1
        watermark_fresh = (evaluation_watermark is not None and
                           evaluation_watermark > (self.last_evaluation_watermark or 0))
        fresh = watermark_fresh if fresh_count is None else fresh_count > 0
        if evaluation_watermark is not None:
            fresh = fresh and watermark_fresh

        snapshot = {
            "evaluation_id": evaluation_id,
            "rate": rate if valid_rate else None,
            "completed_ns": completed_ns,
            "completed_iso": _iso_ns(completed_ns),
            "elapsed_ns": completed_ns - self.run_started_ns,
            "elapsed_seconds": (completed_ns - self.run_started_ns) / 1_000_000_000,
            "sample_count": sample_count,
            "policy_step": policy_step,
            "fresh_count": fresh_count,
            "evaluation_watermark": evaluation_watermark,
            "evaluation_started_ns": evaluation_started_ns,
            "evaluation_duration_seconds": (
                (completed_ns - evaluation_started_ns) / 1_000_000_000
                if evaluation_started_ns is not None else None
            ),
            "metadata": metadata,
        }
        self.last_evaluation_id = evaluation_id
        self.last_completed_ns = completed_ns
        # Consume observed episodes even for a skipped initial checkpoint or
        # invalid metric. They cannot be reclassified as fresh after an update.
        if watermark_fresh:
            self.last_evaluation_watermark = evaluation_watermark
        self.latest_evaluation = snapshot

        if not valid_rate:
            return self._decision(False, False, "invalid_rate")
        if sample_count == 0:
            return self._decision(False, False, "empty_evaluation")
        if sample_count < self.min_samples:
            return self._decision(False, False, "insufficient_samples")
        if not fresh:
            return self._decision(False, False, "no_fresh_episodes")
        if policy_step == 0:
            return self._decision(False, False, "no_policy_update")

        qualified = rate >= self.target
        if qualified:
            self.streak += 1
            if self.first_qualifying is None:
                self.first_qualifying = _json_copy(snapshot)
            if self.streak >= self.required_consecutive:
                self.consecutive_qualifying = _json_copy(snapshot)
        else:
            self.streak = 0
        return self._decision(True, qualified, "qualified" if qualified else "below_target")

    def _decision(self, accepted, qualified, reason):
        return {
            "accepted": accepted,
            "qualified": qualified,
            "reason": reason,
            "streak": self.streak,
            "should_stop": self.should_stop,
            "first_qualifying": _json_copy(self.first_qualifying),
            "consecutive_qualifying": _json_copy(self.consecutive_qualifying),
            "evaluation": _json_copy(self.latest_evaluation),
        }

    def as_dict(self):
        return _json_copy({
            "schema_version": 1,
            "target": self.target,
            "required_consecutive": self.required_consecutive,
            "min_samples": self.min_samples,
            "run_started_ns": self.run_started_ns,
            "run_started_iso": _iso_ns(self.run_started_ns),
            "streak": self.streak,
            "should_stop": self.should_stop,
            "first_qualifying": self.first_qualifying,
            "consecutive_qualifying": self.consecutive_qualifying,
            "latest_evaluation": self.latest_evaluation,
            "last_evaluation_id": self.last_evaluation_id,
            "last_completed_ns": self.last_completed_ns,
            "last_evaluation_watermark": self.last_evaluation_watermark,
        })

    def save(self, path):
        """Atomically replace a single JSON state file in its own directory."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp",
                                                 dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(self.as_dict(), handle, ensure_ascii=False, allow_nan=False,
                          indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
