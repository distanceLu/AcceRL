"""CPU-only one-slot Replay prefetch instrumentation.

This class preserves CpuReplayPrefetch's fetch/take ordering and cancellation.
It does not sample, consume, clone, or inspect tensor contents and never imports
torch. Counts describe raw-prefetch task state, not Replay resident population
or successful microbatch exposures. The separate existing Replay ledger remains
the source of truth for actual training consumption.
"""
from __future__ import annotations

import asyncio
import time

from split_serial import CpuReplayPrefetch, assert_cpu_batch


def _sample_count(record):
    """Read existing CPU identity metadata without retaining batch references."""
    try:
        metadata = record["raw"][-1]
        if "capture_meta" in metadata:
            metadata = metadata["capture_meta"]
        identities = metadata.get("sample_ids")
        return len(identities) if isinstance(identities, (tuple, list)) else None
    except (KeyError, TypeError, IndexError, AttributeError):
        return None


class ObservedCpuReplayPrefetch(CpuReplayPrefetch):
    """The original bounded producer plus scalar state-transition telemetry.

    ``pending`` means one fetch coroutine is awaiting Replay/coordinator; it is
    not a second ready slot. ``in_use`` lasts from take until the trainer phase
    releases it, including preparation/learning. It is independent of the ready
    slot. No background sampling timer or extra scheduling yield is introduced.
    """

    def __init__(self, fetch, *, writer, model, rank, world_size, local_batch_size,
                 clock_ns=None):
        super().__init__(fetch)
        self.writer = writer
        self.model, self.rank = str(model), int(rank)
        self.world_size, self.local_batch_size = int(world_size), int(local_batch_size)
        self.clock_ns = clock_ns or time.monotonic_ns
        self.pending = self.in_use = self.empty_wait = False
        self.ready_sample_count = self.in_use_sample_count = None
        self.fetches_started = self.fetches_completed = self.batches_taken = 0
        self.batches_released = self.observer_errors = self.empty_wait_ns = 0
        self._wait_started_ns = None
        self._emit("created")

    def snapshot(self):
        waiting_ns = (max(0, self.clock_ns() - self._wait_started_ns)
                      if self._wait_started_ns is not None else 0)
        return {
            "scope": "raw_cpu_prefetch_slot_not_replay_resident_or_consumption",
            "model": self.model, "rank": self.rank, "world_size": self.world_size,
            "replay": "policy" if self.model == "policy" else "wm",
            "capacity_batches": 1, "expected_local_batch_size": self.local_batch_size,
            "ready_batches": int(self.ready is not None), "pending_fetches": int(self.pending),
            "in_use_batches": int(self.in_use), "empty_waiters": int(self.empty_wait),
            "ready_sample_count": self.ready_sample_count,
            "in_use_sample_count": self.in_use_sample_count,
            "fetches_started": self.fetches_started, "fetches_completed": self.fetches_completed,
            "batches_taken": self.batches_taken, "batches_released": self.batches_released,
            "empty_wait_ns": self.empty_wait_ns + waiting_ns,
            "closed": self.closed, "failed": self.error is not None,
            "observer_errors": self.observer_errors,
            "reserved_batches_are_not_consumed": True,
        }

    def _emit(self, transition):
        try:
            self.writer.event("cpu_replay_prefetch_state", details={
                "transition": transition, **self.snapshot()})
        except Exception:
            # Observer failure cannot redraw data or change learning. The error
            # counter is attached to every later successfully recorded state.
            self.observer_errors += 1

    def start(self):
        task = super().start()
        self._emit("started")
        return task

    async def _produce(self):
        # Same synchronization and fetch order as the base implementation.
        try:
            while True:
                async with self.condition:
                    await self.condition.wait_for(lambda: self.closed or self.ready is None)
                    if self.closed:
                        return
                self.pending = True
                self.fetches_started += 1
                self._emit("fetch_started")
                value = await self.fetch()
                assert_cpu_batch(value)
                async with self.condition:
                    if self.closed:
                        return
                    if self.ready is not None:
                        raise RuntimeError("CPU Replay prefetch exceeded its one-slot bound")
                    self.ready = value
                    value = None
                    self.pending = False
                    self.fetches_completed += 1
                    self.ready_sample_count = _sample_count(self.ready)
                    self._emit("ready")
                    self.condition.notify_all()
        except asyncio.CancelledError:
            self.pending = False
            self._emit("fetch_cancelled")
            raise
        except BaseException as error:
            async with self.condition:
                self.pending = False
                self.error = error
                self._emit("fetch_failed")
                self.condition.notify_all()

    async def take(self):
        async with self.condition:
            waiting = self.ready is None and self.error is None and not self.closed
            if waiting:
                self.empty_wait = True
                self._wait_started_ns = self.clock_ns()
                self._emit("empty_wait_started")
            try:
                await self.condition.wait_for(
                    lambda: self.ready is not None or self.error is not None or self.closed)
            finally:
                if waiting:
                    self.empty_wait_ns += max(0, self.clock_ns() - self._wait_started_ns)
                    self._wait_started_ns = None
                    self.empty_wait = False
                    self._emit("empty_wait_ended")
            if self.error is not None:
                raise RuntimeError("CPU Replay prefetch failed; this cohort is not redrawn") from self.error
            if self.closed:
                raise RuntimeError("CPU Replay prefetch is closed")
            value, self.ready = self.ready, None
            self.in_use = True
            self.in_use_sample_count, self.ready_sample_count = self.ready_sample_count, None
            self.batches_taken += 1
            self._emit("taken")
            self.condition.notify_all()
            return value

    def release_in_use(self):
        """Call after phase completion; this is not a Replay consume receipt."""
        if not self.in_use:
            return
        self.in_use = False
        self.in_use_sample_count = None
        self.batches_released += 1
        self._emit("released_after_training_phase")

    async def close(self):
        await super().close()
        self.pending = self.in_use = False
        self.ready_sample_count = self.in_use_sample_count = None
        self._emit("closed")
