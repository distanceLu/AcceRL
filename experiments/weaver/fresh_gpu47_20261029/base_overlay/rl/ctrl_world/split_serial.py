"""CPU-only helpers for optional, explicit P -> World -> Reward stage handoff.

The producer stores raw CPU/Ray Replay shards only. GPU preparation is owned by
the consuming stage, so an inactive model cannot allocate its next GPU batch.
There is deliberately no torch, Ray, model eviction, or optimizer mutation here.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping


MODEL_ORDER = ("policy", "ctrl", "reward")


def assert_cpu_batch(value):
    """Reject accidentally GPU-backed Replay values without importing torch."""
    if getattr(value, "is_cuda", False) is True:
        raise RuntimeError("Serial CPU prefetch received a GPU tensor")
    if isinstance(value, Mapping):
        for item in value.values():
            assert_cpu_batch(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            assert_cpu_batch(item)


def serial_enabled(layout, *, environment):
    return layout == "four_rank_split_train" and environment.get("ACCERL_SPLIT_TRAIN_SERIAL") == "1"


def epoch_groups(training_groups, owners, enabled):
    """One group means the legacy concurrent path; three mean ordered handoff."""
    return [training_groups[kind] for kind in MODEL_ORDER] if enabled else [owners]


def validate_sample_group(records, model, step, world_size, local_size, *, required=True):
    """Check actual CPU identities, including partial overlaps hashes cannot detect."""
    if len(records) != world_size or {item.get("rank") for item in records} != set(range(world_size)):
        raise RuntimeError("Serial sample identity audit requires every distinct rank")
    seen, reservations, hashes = set(), set(), {}
    for item in records:
        rank, cohort = item["rank"], item.get("cohort")
        if item.get("model") != model or item.get("policy_step") != step:
            raise RuntimeError("Serial sample identity belongs to another model or step")
        if not required:
            if cohort is not None:
                raise RuntimeError("Off-schedule phase unexpectedly claims a sample cohort")
            continue
        if not isinstance(cohort, dict):
            raise RuntimeError("Active serial phase is missing its sample identities")
        identities = cohort.get("sample_ids")
        if (not isinstance(identities, list) or len(identities) != local_size
                or any(not isinstance(identity, str) or not identity for identity in identities)
                or len(set(identities)) != local_size):
            raise RuntimeError("Serial sample shard has missing or duplicate actual identities")
        if (cohort.get("shard_start") != rank * local_size
                or cohort.get("shard_end") != (rank + 1) * local_size):
            raise RuntimeError("Serial sample identities do not match their rank shard")
        if seen.intersection(identities):
            raise RuntimeError("Actual sample identities overlap across training ranks")
        seen.update(identities)
        reservation = cohort.get("reservation_id")
        if not isinstance(reservation, str) or not reservation:
            raise RuntimeError("Serial sample reservation is missing")
        reservations.add(reservation)
        hashes[rank] = hashlib.sha256(json.dumps(identities, separators=(",", ":")).encode()).hexdigest()
    if required and len(reservations) != 1:
        raise RuntimeError("Training ranks used different sample reservations")
    return {"policy_step": step, "model": model, "world_size": world_size,
            "required": required, "actual_sample_ids_verified": True,
            "rank_shards_disjoint": True, "global_samples": len(seen),
            "reservation_id": next(iter(reservations)) if reservations else None,
            "rank_sample_ids_sha256": hashes}


def validate_shared_wm_samples(world_records, reward_records):
    """Both independent owners must consume the same ordered raw shard per rank."""
    world = {item["rank"]: item for item in world_records}
    reward = {item["rank"]: item for item in reward_records}
    if set(world) != set(reward):
        raise RuntimeError("World and Reward sample rank identities differ")
    for rank in world:
        if (world[rank]["policy_step"] != reward[rank]["policy_step"]
                or world[rank]["cohort"] != reward[rank]["cohort"]):
            raise RuntimeError("World and Reward actual ordered sample identities differ")
    return True


class SerialPhase:
    """Reject duplicate/out-of-order or overlapping RPCs before GPU preparation."""
    def __init__(self, model, rank, world_size):
        if model not in MODEL_ORDER or not 0 <= rank < world_size:
            raise ValueError("Invalid serial model/rank")
        self.model, self.rank, self.world_size = model, rank, world_size
        self.completed_step = 0
        self.active = None
        self.failure = None

    def begin(self, step):
        if self.failure is not None:
            raise RuntimeError("Serial owner is poisoned after a failed stage") from self.failure
        if (isinstance(step, bool) or not isinstance(step, int)
                or step != self.completed_step + 1 or self.active is not None):
            raise ValueError("Serial stage requires one common next Policy boundary")
        self.active = (step, self.model)
        return self.active

    def complete(self, key):
        if self.active != key:
            raise RuntimeError("Serial completion does not match its active stage")
        self.completed_step = key[0]
        self.active = None

    def fail(self, error):
        self.failure = error
        self.active = None


class CpuReplayPrefetch:
    """One raw CPU shard, fetched ahead without holding any prepared GPU value.

    Every World/Reward owner starts this producer before its stage is allowed;
    thus all eight coordinator claims can proceed while GPU work stays serial.
    Errors wake consumers. Cancellation never causes a retry or redraw.
    """
    def __init__(self, fetch):
        self.fetch = fetch
        self.condition = asyncio.Condition()
        self.ready = None
        self.error = None
        self.closed = False
        self.task = None

    def start(self):
        if self.task is not None:
            raise RuntimeError("CPU prefetch was already started")
        self.task = asyncio.create_task(self._produce())
        return self.task

    async def _produce(self):
        try:
            while True:
                async with self.condition:
                    await self.condition.wait_for(lambda: self.closed or self.ready is None)
                    if self.closed:
                        return
                value = await self.fetch()
                assert_cpu_batch(value)
                async with self.condition:
                    if self.closed:
                        return
                    if self.ready is not None:
                        raise RuntimeError("CPU Replay prefetch exceeded its one-slot bound")
                    self.ready = value
                    value = None  # The producer retains no extra previous batch reference.
                    self.condition.notify_all()
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            async with self.condition:
                self.error = error
                self.condition.notify_all()

    async def take(self):
        async with self.condition:
            await self.condition.wait_for(
                lambda: self.ready is not None or self.error is not None or self.closed)
            if self.error is not None:
                raise RuntimeError("CPU Replay prefetch failed; this cohort is not redrawn") from self.error
            if self.closed:
                raise RuntimeError("CPU Replay prefetch is closed")
            value, self.ready = self.ready, None
            self.condition.notify_all()
            return value

    async def close(self):
        async with self.condition:
            self.closed, self.ready = True, None
            self.condition.notify_all()
        if self.task is not None and self.task is not asyncio.current_task():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass


async def finish_preparation(future):
    """A cancelled asyncio waiter must not leave executor GPU work unobserved.

    Python cannot cancel a running executor thread. Shield it, then drain that
    exact future on cancellation before permitting the failure to propagate.
    The caller owns stage failure and never ACKs a cancelled stage.
    """
    cancelled = False
    while not future.done():
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError:
            # Repeated cancellation still cannot stop the executor thread.
            cancelled = True
        except BaseException:
            if not cancelled:
                raise
            break
    if cancelled:
        try:
            future.result()  # Observe an exception too; never retry the batch.
        except BaseException:
            pass
        raise asyncio.CancelledError()
    return future.result()
