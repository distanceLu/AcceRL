"""Opt-in Nsight windows bounded by completion and a deadline; no CUDA sync."""
from __future__ import annotations

import os
import threading
import time


def nvtx_begin(label):
    if os.environ.get("ACCERL_NSIGHT_NVTX", "0") != "1":
        return None
    import torch
    return torch.cuda.nvtx.range_start(str(label))


def nvtx_end(handle):
    if handle is not None:
        import torch
        torch.cuda.nvtx.range_end(handle)


class NsightCaptureWindows:
    """CPU-driver NVTX triggers with explicit completion/timeout coverage.

    Nsight traces the launched process tree, so the driver range also gates
    training and inference clients. A range is a collection trigger, never
    evidence that kernels overlap. No worker is paused or rescheduled here.
    ``timer`` preserves the original fixed-window behavior. ``epoch`` closes
    after the selected epoch's P/World/Reward ACKs, or the deadline. ``stages``
    opens one range per serial model phase, sharing the same <=600 s budget;
    use Nsight ``--capture-range-end=repeat:3`` for one selected epoch. Phase
    completion is bookkeeping, not proof of GPU kernels or successful tracing.
    """
    def __init__(self, writer, *, steps=None, seconds=None, mode=None, begin=None, end=None,
                 timer_factory=None, clock_ns=None, monotonic=None):
        self.writer = writer
        self.steps = tuple(int(value) for value in (
            steps if steps is not None else
            os.environ.get("ACCERL_NSIGHT_STEPS", "").split(",")) if str(value).strip())
        self.seconds = float(seconds if seconds is not None else
                             os.environ.get("ACCERL_NSIGHT_SECONDS", "90"))
        self.mode = str(mode if mode is not None else os.environ.get("ACCERL_NSIGHT_MODE", "timer"))
        if self.mode not in ("timer", "epoch", "stages"):
            raise ValueError("Nsight mode must be timer, epoch or stages")
        if self.seconds <= 0 or self.seconds > 600:
            raise ValueError("Nsight capture windows must be in (0, 600] seconds")
        if any(step < 1 for step in self.steps) or len(set(self.steps)) != len(self.steps):
            raise ValueError("Nsight steps must be unique positive integers")
        self.begin, self.end = begin or nvtx_begin, end or nvtx_end
        self.timer_factory = timer_factory or threading.Timer
        self.clock_ns = clock_ns or time.time_ns
        self.monotonic = monotonic or time.monotonic
        self.lock = threading.RLock()
        self.active = None
        self.timer = None
        self.started = set()
        self.token = 0
        self.stage_started = set()
        self.stage_results = {}
        self.stage_spent = {}

    def before_epoch(self, step):
        step = int(step)
        with self.lock:
            if step not in self.steps or step in self.started:
                return
            skipped = self.active is not None
            self.started.add(step)
            if self.mode == "stages" and not skipped:
                self.stage_spent[step] = 0.0
                self.stage_results[step] = {}
                return
            details = self._open_locked(step, None, self.seconds) if not skipped else None
        # SQLite can wait for another writer. Never hold the capture lock while
        # logging, or that wait could prevent the timer from closing the range.
        if skipped:
            self.writer.event("nsight_capture_skipped", details={
                "policy_step": step, "reason": "previous_window_still_active"})
            return
        self._started_event(details)

    def _open_locked(self, step, model, seconds):
        handle = self.begin("PROFILE_WINDOW")
        if handle is None:
            raise RuntimeError("Nsight windows configured without NVTX enabled")
        self.token += 1
        token = self.token
        started_ns, started_monotonic = self.clock_ns(), self.monotonic()
        self.active = {"step": step, "model": model, "handle": handle,
                       "started_ns": started_ns, "started_monotonic": started_monotonic,
                       "token": token, "limit_seconds": seconds}
        try:
            self.timer = self.timer_factory(seconds, lambda: self.finish("duration_limit", token))
            self.timer.daemon = True
            self.timer.start()
        except BaseException:
            self.active = None
            self.end(handle)
            raise
        return {"policy_step": step, "model": model, "duration_limit_seconds": seconds,
                "capture_mode": self.mode, "capture_range": "PROFILE_WINDOW",
                "scope": "launched_process_tree", "started_ns": started_ns,
                "deadline_ns": started_ns + int(seconds * 1e9),
                "kernel_timeline_required": True}

    def _started_event(self, details):
        self.writer.event("nsight_capture_started", details=details)
        self.writer.flush()

    def before_stage(self, step, model):
        """Optional split-window trigger before a serial Policy/World/Reward phase."""
        if self.mode != "stages":
            return
        step, model = int(step), str(model)
        if model not in ("policy", "ctrl", "reward"):
            raise ValueError("Unknown serial training model")
        with self.lock:
            if step not in self.steps or (step, model) in self.stage_started:
                return
            if step not in self.stage_results:
                raise RuntimeError("before_epoch must select a staged capture first")
            self.stage_started.add((step, model))
            remaining = max(0.0, self.seconds - self.stage_spent[step])
            skipped = self.active is not None or remaining <= 0
            details = None if skipped else self._open_locked(step, model, remaining)
        if skipped:
            self.writer.event("nsight_capture_skipped", details={"policy_step": step,
                "model": model, "reason": "previous_window_still_active" if self.active else "shared_duration_budget_exhausted"})
            return
        self._started_event(details)

    def after_stage(self, step, model):
        if self.mode != "stages":
            return
        with self.lock:
            active = self.active
            token = active["token"] if active and (active["step"], active["model"]) == (int(step), str(model)) else None
        if token is not None:
            self.finish("completed_stage", token)

    def after_epoch(self, step):
        """Call only after all required serial model groups completed their ACKs."""
        step = int(step)
        if self.mode == "timer" or step not in self.steps:
            return
        with self.lock:
            active = self.active
            token = active["token"] if active and active["step"] == step else None
        if token is not None:
            self.finish("completed_epoch" if self.mode == "epoch" else "epoch_closed_incomplete_stage", token)
        if self.mode == "stages":
            outcomes = dict(self.stage_results.get(step, {}))
            self.writer.event("nsight_capture_epoch_coverage", details={"policy_step": step,
                "capture_mode": self.mode, "stage_results": outcomes,
                "coverage_complete": all(outcomes.get(model) == "completed_stage" for model in ("policy", "ctrl", "reward")),
                "captured_seconds": self.stage_spent.get(step, 0.0),
                "duration_limit_seconds": self.seconds,
                "kernel_timeline_required": True})
            self.writer.flush()

    def finish(self, reason="manual", expected_token=None):
        with self.lock:
            if self.active is None or (expected_token is not None and self.active["token"] != expected_token):
                return
            active = self.active
            self.active = None
            close_requested_ns = self.clock_ns()
            elapsed = max(0.0, self.monotonic() - active["started_monotonic"])
            if self.timer is not None:
                self.timer.cancel()
            if elapsed > active["limit_seconds"] and reason in ("completed_epoch", "completed_stage"):
                reason += "_after_deadline"
            if self.mode == "stages":
                step = active["step"]
                self.stage_spent[step] += elapsed
                self.stage_results[step][active["model"]] = reason
        # Nsight may synchronously flush/export inside nvtxRangeEnd. Measure
        # the request boundary before that call, and report its blocking time
        # separately. Remove the active token first so a cancelled timer cannot
        # close the same range again while conversion blocks the caller.
        self.end(active["handle"])
        end_returned_ns = self.clock_ns()
        end_call_seconds = max(0.0, self.monotonic() - active["started_monotonic"] - elapsed)
        self.writer.event("nsight_capture_ended", details={
            "policy_step": active["step"], "model": active["model"],
            "started_ns": active["started_ns"], "completed_ns": end_returned_ns,
            "close_requested_ns": close_requested_ns,
            "nvtx_end_returned_ns": end_returned_ns,
            "nvtx_end_call_seconds": end_call_seconds,
            "elapsed_scope": "range close request; excludes synchronous flush/export inside nvtxRangeEnd",
            "capture_mode": self.mode, "reason": reason, "elapsed_seconds": elapsed,
            "duration_limit_seconds": active["limit_seconds"],
            "coverage_complete": reason in ("completed_epoch", "completed_stage"),
            "deadline_overrun_seconds": max(0.0, elapsed - active["limit_seconds"]),
            "kernel_timeline_required": True})
        self.writer.flush()

    def close(self):
        with self.lock:
            if self.timer is not None:
                self.timer.cancel()
        self.finish("close")
