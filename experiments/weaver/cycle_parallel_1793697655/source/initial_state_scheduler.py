"""Persistent, CPU-only global initial-state allocation and simulator-step budget.

Exactly one serialized owner calls this class (normally the WM Replay actor).
Allocation consumes one position of a seeded, shuffled permutation regardless of
worker count or episode duration. A cancelled allocation is never silently put
back into that permutation. The full episode reservation includes settling
steps; only a reported, validated terminal result refunds unused steps.

An interrupted episode cannot safely be rerun from its initial state: some of
its reserved simulator steps may already have executed. Reloaded in-flight
leases therefore require explicit ``recover`` before their worker may acquire
again. They remain fully reserved meanwhile. Saving model weights does not
replace the latest allocation journal with an older snapshot.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any


class InitialStateSchedulerError(RuntimeError):
    pass


class CheckpointMismatchError(InitialStateSchedulerError):
    pass


class UnresolvedLeaseError(InitialStateSchedulerError):
    pass


class BudgetExceededError(InitialStateSchedulerError):
    pass


def _integer(name: str, value: Any, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _tuple_tree(value: Any) -> Any:
    return tuple(_tuple_tree(item) for item in value) if isinstance(value, list) else value


def _json_copy(value: Any) -> Any:
    # Reject non-finite metadata and ensure the exact persisted type is returned.
    return json.loads(json.dumps(value, allow_nan=False))


class InitialStateScheduler:
    """Global without-replacement state queue with conservative budget leases.

    ``acquire`` returns a JSON-serializable lease or ``None`` when no useful
    reservation fits. ``finish`` accepts actual action/reset step counts and
    refunds the rest. ``cancel`` without exact counts charges the full lease as
    unknown consumption; explicitly supplying both counts allows a proven
    partial refund. Neither terminal operation reassigns the initial-state slot.

    Existing journals are loaded automatically and must match *all* config
    fields, including the initial-state asset SHA256. ``snapshot`` can be saved
    with model weights. ``from_snapshot`` is for explicit journal restoration,
    not for rolling the latest journal back to an older model checkpoint.
    """
    SCHEMA_VERSION = 1

    def __init__(self, state_count: int = 50, seed: int = 20261026,
                 max_env_steps: int = 11000, checkpoint_path: str | Path | None = None,
                 state_asset_sha256: str | None = None, *, snapshot: dict | None = None):
        _integer("state_count", state_count, 1)
        _integer("seed", seed)
        _integer("max_env_steps", max_env_steps, 1)
        if state_asset_sha256 is not None:
            if (not isinstance(state_asset_sha256, str) or len(state_asset_sha256) != 64
                    or any(char not in "0123456789abcdef" for char in state_asset_sha256)):
                raise ValueError("state_asset_sha256 must be a lowercase SHA256 hex digest")
        self.config = dict(state_count=state_count, seed=seed, max_env_steps=max_env_steps,
                           state_asset_sha256=state_asset_sha256)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self._rng = random.Random(seed)
        env_seed_material = hashlib.sha256(f"initial-state-env-seed:{seed}".encode()).digest()
        self._env_rng = random.Random(int.from_bytes(env_seed_material, "big"))
        self._permutation = list(range(state_count))
        self._rng.shuffle(self._permutation)
        self._cycle = 0
        self._cursor = 0
        self._allocations: dict[str, dict] = {}
        self._inflight: dict[str, dict] = {}
        self._terminal: dict[str, dict] = {}
        self._events: list[dict] = []
        self._last_block_reason: str | None = None
        if self.checkpoint_path is not None and self.checkpoint_path.exists():
            if snapshot is not None:
                raise ValueError("provide either an existing journal or an explicit snapshot, not both")
            self._load(json.loads(self.checkpoint_path.read_text(encoding="utf-8")))
        elif snapshot is not None:
            self._load(snapshot)
        self.persist()

    @classmethod
    def from_snapshot(cls, snapshot: dict, checkpoint_path: str | Path | None = None,
                      **expected_config: Any) -> "InitialStateScheduler":
        if not isinstance(snapshot, dict) or not isinstance(snapshot.get("config"), dict):
            raise CheckpointMismatchError("snapshot has no scheduler config")
        config = dict(snapshot["config"])
        for name, value in expected_config.items():
            if name not in config or config[name] != value:
                raise CheckpointMismatchError(f"snapshot config mismatch: {name}")
        return cls(**config, checkpoint_path=checkpoint_path, snapshot=snapshot)

    def _load(self, state: dict) -> None:
        if not isinstance(state, dict) or state.get("schema_version") != self.SCHEMA_VERSION:
            raise CheckpointMismatchError("unsupported initial-state journal schema")
        if state.get("config") != self.config:
            raise CheckpointMismatchError("initial-state pool or scheduler config differs from journal")
        try:
            permutation = state["permutation"]
            if (not isinstance(permutation, list) or len(permutation) != self.config["state_count"]
                    or any(type(value) is not int for value in permutation)
                    or sorted(permutation) != list(range(self.config["state_count"]))):
                raise ValueError("invalid permutation")
            cycle = _integer("cycle", state["cycle"])
            cursor = _integer("cursor", state["cursor"])
            if cursor > self.config["state_count"]:
                raise ValueError("cursor beyond permutation")
            allocations = state["allocations"]
            if not isinstance(allocations, list):
                raise ValueError("allocations must be a list")
            allocated = {}
            for ordinal, lease in enumerate(allocations, 1):
                self._validate_lease(lease)
                expected_token = f"initial-state:{ordinal:012d}"
                if lease["token"] != expected_token or lease["allocation_id"] != ordinal:
                    raise ValueError("allocation sequence is not contiguous")
                if lease["cycle"] != (ordinal - 1) // self.config["state_count"]:
                    raise ValueError("allocation cycle mismatch")
                if lease["index"] != (ordinal - 1) % self.config["state_count"]:
                    raise ValueError("allocation permutation index mismatch")
                allocated[lease["token"]] = deepcopy(lease)
            expected_cycle = (len(allocated) - 1) // self.config["state_count"] if allocated else 0
            expected_cursor = (len(allocated) - 1) % self.config["state_count"] + 1 if allocated else 0
            if (cycle, cursor) != (expected_cycle, expected_cursor):
                raise ValueError("journal cursor disagrees with allocation ledger")
            by_cycle: dict[int, list[int]] = {}
            for lease in allocated.values():
                by_cycle.setdefault(lease["cycle"], []).append(lease["initial_state_id"])
            if any(len(ids) != len(set(ids)) for ids in by_cycle.values()):
                raise ValueError("initial-state repeated inside a permutation cycle")
            if by_cycle.get(cycle, []) != permutation[:cursor]:
                raise ValueError("current permutation disagrees with allocation ledger")
            terminal = state["terminal"]
            inflight_list = state["inflight"]
            if not isinstance(terminal, dict) or not isinstance(inflight_list, list):
                raise ValueError("invalid terminal/in-flight ledger")
            inflight = {}
            workers = set()
            for lease in inflight_list:
                self._validate_lease(lease)
                token = lease["token"]
                if token not in allocated or token in inflight or lease != allocated[token]:
                    raise ValueError("in-flight lease differs from allocation ledger")
                if lease["worker_id"] in workers:
                    raise ValueError("worker owns multiple in-flight leases")
                workers.add(lease["worker_id"])
                inflight[token] = deepcopy(lease)
            if set(inflight) & set(terminal) or set(inflight) | set(terminal) != set(allocated):
                raise ValueError("allocation has missing or duplicate terminal status")
            for token, result in terminal.items():
                if result.get("token") != token or result.get("status") not in ("completed", "cancelled"):
                    raise ValueError("invalid terminal result")
                self._validate_consumption(allocated[token], result["action_steps"], result["reset_steps"],
                                           allow_unknown=result["status"] == "cancelled")
                expected_charge = (allocated[token]["reserved_env_steps"] if result["action_steps"] is None
                                   else result["action_steps"] + result["reset_steps"])
                if result.get("charged_env_steps") != expected_charge:
                    raise ValueError("terminal budget charge disagrees with consumed steps")
                if result["status"] == "completed" and type(result.get("success")) is not bool:
                    raise ValueError("completed result is missing boolean success")
            self._rng.setstate(_tuple_tree(state["rng_state"]))
            self._env_rng.setstate(_tuple_tree(state["env_rng_state"]))
            if not isinstance(state["events"], list):
                raise ValueError("events must be a list")
            self._permutation, self._cycle, self._cursor = list(permutation), cycle, cursor
            self._allocations, self._terminal = allocated, deepcopy(terminal)
            self._inflight, self._events = inflight, deepcopy(state["events"])
            # This process cannot know whether an interrupted worker stepped.
            for lease in self._inflight.values():
                lease["recovery_required"] = True
                self._allocations[lease["token"]]["recovery_required"] = True
            if self.status()["committed_env_steps"] > self.config["max_env_steps"]:
                raise ValueError("journal exceeds global simulator-step budget")
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointMismatchError(f"invalid initial-state journal: {exc}") from exc

    def _validate_lease(self, lease: dict) -> None:
        if not isinstance(lease, dict):
            raise ValueError("lease must be an object")
        state_id = _integer("initial_state_id", lease["initial_state_id"])
        if state_id >= self.config["state_count"]:
            raise ValueError("initial_state_id outside state pool")
        _integer("max_action_steps", lease["max_action_steps"], 1)
        _integer("reset_steps", lease["reset_steps"])
        _integer("env_seed", lease["env_seed"])
        if lease["reserved_env_steps"] != lease["max_action_steps"] + lease["reset_steps"]:
            raise ValueError("reservation disagrees with action/reset limits")
        if not isinstance(lease["worker_id"], str) or not lease["worker_id"]:
            raise ValueError("worker_id must be nonempty")
        if type(lease.get("recovery_required")) is not bool:
            raise ValueError("lease recovery_required must be boolean")

    def _validate_consumption(self, lease: dict, action_steps: int | None,
                              reset_steps: int | None, allow_unknown: bool) -> None:
        if action_steps is None or reset_steps is None:
            if allow_unknown and action_steps is None and reset_steps is None:
                return
            raise ValueError("both consumed action_steps and reset_steps are required")
        _integer("action_steps", action_steps)
        _integer("reset_steps", reset_steps)
        if action_steps > lease["max_action_steps"] or reset_steps > lease["reset_steps"]:
            raise BudgetExceededError("reported simulator steps exceed their episode reservation")

    def acquire(self, worker_id: str, max_action_steps: int = 220,
                reset_steps: int = 10) -> dict | None:
        if not isinstance(worker_id, str) or not worker_id:
            raise ValueError("worker_id must be a nonempty string")
        _integer("max_action_steps", max_action_steps, 1)
        _integer("reset_steps", reset_steps)
        for lease in self._inflight.values():
            if lease["worker_id"] == worker_id:
                if lease["recovery_required"]:
                    raise UnresolvedLeaseError(f"lease {lease['token']} requires explicit recovery")
                # A prior persistence failure may have raised before this lease
                # was handed out. An idempotent retry must make it durable too.
                self.persist()
                return deepcopy(lease)
        available = self.status()["available_env_steps"]
        action_limit = min(max_action_steps, available - reset_steps)
        if action_limit <= 0:
            self._last_block_reason = ("inflight_reservations" if self._inflight
                                       else "insufficient_budget_for_reset_and_one_action")
            return None
        if self._cursor == self.config["state_count"]:
            self._cycle += 1
            self._cursor = 0
            self._permutation = list(range(self.config["state_count"]))
            self._rng.shuffle(self._permutation)
        ordinal = len(self._allocations) + 1
        token = f"initial-state:{ordinal:012d}"
        lease = dict(token=token, allocation_id=ordinal, worker_id=worker_id,
                     initial_state_id=self._permutation[self._cursor],
                     env_seed=self._env_rng.randrange(0, 2**31),
                     max_action_steps=action_limit, reset_steps=reset_steps,
                     reserved_env_steps=action_limit + reset_steps,
                     cycle=self._cycle, index=self._cursor, allocated_at_ns=time.time_ns(),
                     recovery_required=False)
        self._cursor += 1
        self._allocations[token] = deepcopy(lease)
        self._inflight[token] = lease
        self._events.append(dict(event="allocated", ts_ns=lease["allocated_at_ns"], **deepcopy(lease)))
        self._last_block_reason = None
        self.persist()  # Durable assignment precedes handing it to a worker.
        return deepcopy(lease)

    def finish(self, token: str, action_steps: int, reset_steps: int,
               success: bool, **metadata: Any) -> dict:
        if type(success) is not bool:
            raise ValueError("success must be boolean")
        return self._finalize(token, "completed", action_steps, reset_steps,
                              success=success, metadata=metadata, allow_recovery=False)

    def cancel(self, token: str, *, action_steps: int | None = None,
               reset_steps: int | None = None, reason: str = "cancelled", **metadata: Any) -> dict:
        return self._finalize(token, "cancelled", action_steps, reset_steps,
                              success=None, metadata=dict(reason=reason, **metadata), allow_recovery=False)

    def recover(self, token: str, *, mode: str = "cancel", action_steps: int | None = None,
                reset_steps: int | None = None, success: bool | None = None,
                reason: str = "explicit_recovery", **metadata: Any) -> dict:
        """Explicitly reconcile an interrupted lease; default charges it in full.

        A recovered slot is terminal and never rerun. The worker's next acquire
        advances to the next global permutation slot. Proven completion may be
        reported using mode='finish' and exact counts plus boolean success.
        """
        if mode == "finish":
            if type(success) is not bool:
                raise ValueError("successful recovery needs boolean episode success")
            return self._finalize(token, "completed", action_steps, reset_steps, success=success,
                                  metadata=metadata, allow_recovery=True)
        if mode != "cancel":
            raise ValueError("recovery mode must be 'finish' or 'cancel'")
        return self._finalize(token, "cancelled", action_steps, reset_steps, success=None,
                              metadata=dict(reason=reason, **metadata), allow_recovery=True)

    def _finalize(self, token: str, status: str, action_steps: int | None,
                  reset_steps: int | None, *, success: bool | None,
                  metadata: dict, allow_recovery: bool) -> dict:
        if token not in self._allocations:
            raise InitialStateSchedulerError(f"unknown initial-state lease: {token}")
        lease = self._allocations[token]
        self._validate_consumption(lease, action_steps, reset_steps, allow_unknown=status == "cancelled")
        metadata = _json_copy(metadata)
        payload = dict(token=token, status=status, action_steps=action_steps, reset_steps=reset_steps,
                       success=success, metadata=metadata)
        if token in self._terminal:
            old = self._terminal[token]
            if any(old.get(key) != value for key, value in payload.items()):
                raise InitialStateSchedulerError("terminal lease cannot be completed/cancelled differently")
            self.persist()
            return deepcopy(old)
        if self._inflight[token]["recovery_required"] and not allow_recovery:
            raise UnresolvedLeaseError(f"lease {token} requires explicit recovery")
        charged = lease["reserved_env_steps"] if action_steps is None else action_steps + reset_steps
        result = dict(**payload, charged_env_steps=charged,
                      refunded_env_steps=lease["reserved_env_steps"] - charged,
                      charge_mode="conservative_reservation" if action_steps is None else "exact_consumption",
                      initial_state_id=lease["initial_state_id"], worker_id=lease["worker_id"],
                      cycle=lease["cycle"], index=lease["index"], finalized_at_ns=time.time_ns())
        del self._inflight[token]
        self._terminal[token] = result
        self._events.append(dict(event=status, **deepcopy(result)))
        self.persist()
        return deepcopy(result)

    def status(self) -> dict:
        completed = [result for result in self._terminal.values() if result["status"] == "completed"]
        cancelled = [result for result in self._terminal.values() if result["status"] == "cancelled"]
        known_action = sum(result["action_steps"] for result in self._terminal.values()
                           if result["action_steps"] is not None)
        known_reset = sum(result["reset_steps"] for result in self._terminal.values()
                          if result["reset_steps"] is not None)
        unknown = sum(result["charged_env_steps"] for result in self._terminal.values()
                      if result["action_steps"] is None)
        reserved = sum(lease["reserved_env_steps"] for lease in self._inflight.values())
        used = known_action + known_reset + unknown
        assigned_counts = Counter(lease["initial_state_id"] for lease in self._allocations.values())
        completed_counts = Counter(result["initial_state_id"] for result in completed)
        assigned_ids, completed_ids = set(assigned_counts), set(completed_counts)
        all_ids = set(range(self.config["state_count"]))
        return dict(
            **self.config, cycle=self._cycle, cursor=self._cursor,
            episodes_assigned=len(self._allocations), episodes_completed=len(completed),
            episodes_cancelled=len(cancelled), episodes_inflight=len(self._inflight),
            successful_episodes=sum(result["success"] for result in completed),
            known_action_steps=known_action, known_reset_steps=known_reset,
            conservative_unknown_env_steps=unknown, consumed_env_steps=used,
            reserved_env_steps=reserved, committed_env_steps=used + reserved,
            available_env_steps=self.config["max_env_steps"] - used - reserved,
            budget_exhausted=used + reserved >= self.config["max_env_steps"],
            recovery_required_tokens=sorted(token for token, lease in self._inflight.items()
                                            if lease["recovery_required"]),
            assigned_initial_state_ids=sorted(assigned_ids),
            completed_initial_state_ids=sorted(completed_ids),
            inflight_initial_state_ids=sorted({lease["initial_state_id"] for lease in self._inflight.values()}),
            assigned_not_completed_initial_state_ids=sorted(assigned_ids - completed_ids),
            unassigned_initial_state_ids=sorted(all_ids - assigned_ids),
            uncompleted_initial_state_ids=sorted(all_ids - completed_ids),
            assigned_state_counts={str(key): assigned_counts.get(key, 0) for key in sorted(all_ids)},
            completed_state_counts={str(key): completed_counts.get(key, 0) for key in sorted(all_ids)},
            assigned_coverage_count=len(assigned_ids), completed_coverage_count=len(completed_ids),
            last_block_reason=self._last_block_reason)

    def snapshot(self) -> dict:
        return _json_copy(dict(
            schema_version=self.SCHEMA_VERSION, config=self.config,
            permutation=self._permutation, cycle=self._cycle, cursor=self._cursor,
            rng_state=self._rng.getstate(), env_rng_state=self._env_rng.getstate(),
            allocations=list(self._allocations.values()), inflight=list(self._inflight.values()),
            terminal=self._terminal, events=self._events, counters=self.status()))

    def persist(self) -> dict:
        """Atomically replace the journal after every state mutation."""
        state = self.snapshot()
        if self.checkpoint_path is None:
            return state
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                             prefix=f".{self.checkpoint_path.name}.", suffix=".tmp",
                                             dir=self.checkpoint_path.parent, delete=False) as handle:
                temporary = handle.name
                json.dump(state, handle, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.checkpoint_path)
            temporary = None
            # On POSIX, also durably commit the directory entry when supported.
            if os.name != "nt":
                directory_fd = os.open(self.checkpoint_path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
        return state
