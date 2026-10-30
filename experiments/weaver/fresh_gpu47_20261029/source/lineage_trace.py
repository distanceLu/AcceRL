"""Bounded CPU-only request/group lineage records for offline DAG auditing.

This recorder does not infer causality from timestamps or model versions. The
caller explicitly supplies the groups that produced each action, frame window,
transition, GAE suffix and sample. Cross-actor or old parent IDs are accepted
only with an explicit caller attestation, which is retained in the event. A
parent ID alone never means that its full historical request chain was recorded.

Closed groups and successful requests live in telemetry, not in an ever-growing
Python registry. Only active groups plus a bounded recent-ID cache are retained.
``emit`` has signature ``emit(kind, details)``; it can wrap Telemetry.event.
"""
from __future__ import annotations

from collections import deque
import json
import threading
import uuid


SCHEMA_VERSION = 1


def _text(value, name, limit=1024):
    if not isinstance(value, str) or not value or len(value) > limit:
        raise ValueError("Invalid lineage " + name)
    return value


def _positive(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise ValueError("Invalid lineage " + name)
    return value


def _ids(values, name, limit):
    if isinstance(values, (str, bytes)):
        raise ValueError("Lineage " + name + " must be an iterable of IDs")
    result, seen = [], set()
    for value in values:
        _text(value, name)
        if value not in seen:
            if len(result) >= limit:
                raise ValueError("Lineage " + name + " exceeds configured bound")
            seen.add(value)
            result.append(value)
    return result


def _metadata(value, maximum_bytes):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Lineage metadata must be a JSON object")
    try:
        encoded = json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError("Lineage metadata must contain finite JSON values") from error
    if len(encoded.encode("utf-8")) > maximum_bytes:
        raise ValueError("Lineage metadata exceeds configured byte bound")
    # Copy now: a later mutation by the caller must not change queued lineage.
    return json.loads(encoded)


class LineageRecorder:
    """Emit direct request edges and group edges with strictly bounded live state."""

    def __init__(self, run_id, namespace, emit=None, *, emit_request_events=True,
                 max_active_groups=32, max_requests_per_group=1024,
                 max_parents=64, max_recent_closed=512, max_metadata_bytes=65536):
        self.run_id = _text(run_id, "run_id", 256)
        self.namespace = _text(namespace, "namespace", 256)
        if emit is not None and not callable(emit):
            raise ValueError("Lineage emit must be callable or None")
        if not isinstance(emit_request_events, bool):
            raise ValueError("emit_request_events must be bool")
        self.emit = emit
        self.emit_request_events = emit_request_events
        self.max_active_groups = _positive(max_active_groups, "max_active_groups")
        self.max_requests_per_group = _positive(max_requests_per_group, "max_requests_per_group")
        self.max_parents = _positive(max_parents, "max_parents")
        self.max_recent_closed = _positive(max_recent_closed, "max_recent_closed", allow_zero=True)
        self.max_metadata_bytes = _positive(max_metadata_bytes, "max_metadata_bytes")
        self._instance = uuid.uuid4().hex
        self._next_group = 0
        self._active = {}
        self._recent_order = deque()
        self._recent = set()
        self._lock = threading.RLock()

    def _emit(self, kind, details):
        if self.emit is not None:
            self.emit(kind, {"schema_version": SCHEMA_VERSION, "run_id": self.run_id,
                             "namespace": self.namespace, **details})

    def record_successful_request(self, request_id, role, model_version, workflow,
                                  *, success=True, **metadata):
        """Record an actual successful RPC receipt; never retain request history.

        If the existing client_wait event already records these four fields,
        disable duplicate request events using ``emit_request_events=False``.
        ``add_request`` then references those externally retained receipts. This
        method cannot independently verify a remote request's success/version.
        """
        _text(request_id, "request_id")
        _text(role, "role", 128)
        _text(workflow, "workflow", 128)
        _positive(model_version, "model_version", allow_zero=True)
        if success is not True:
            raise ValueError("Failed or incomplete RPCs cannot be successful lineage dependencies")
        values = _metadata(metadata, self.max_metadata_bytes)
        if self.emit_request_events:
            self._emit("lineage_request_succeeded", {
                "request_id": request_id, "role": role, "model_version": model_version,
                "workflow": workflow, "success": True, "metadata": values})
        return request_id

    def _validate_references(self, groups, known_by_caller):
        if not isinstance(known_by_caller, bool):
            raise ValueError("parents_known_by_caller must be bool")
        active = [group for group in groups if group in self._active]
        if active:
            raise ValueError("Lineage may reference only completed groups, not active groups")
        recent = [group for group in groups if group in self._recent]
        external = [group for group in groups if group not in self._recent]
        if external and not known_by_caller:
            raise ValueError("Old/cross-actor lineage references require explicit caller attestation")
        return {"recent_local_group_ids": recent, "caller_attested_group_ids": external,
                "scope": "bounded_local_validation_only; full DAG requires retained telemetry"}

    def begin_group(self, stage, parent_groups=(), metadata=None, *, parents_known_by_caller=False):
        """Create a small DAG node; parents are references, not flattened ancestry."""
        _text(stage, "stage", 128)
        parents = _ids(parent_groups, "parent_group_ids", self.max_parents)
        values = _metadata(metadata, self.max_metadata_bytes)
        with self._lock:
            if len(self._active) >= self.max_active_groups:
                raise RuntimeError("Too many active lineage groups; close or abort existing groups")
            validation = self._validate_references(parents, parents_known_by_caller)
            self._next_group += 1
            group_id = "lineage:%s:%d" % (self._instance, self._next_group)
            self._active[group_id] = {"group_id": group_id, "stage": stage,
                                      "parent_group_ids": parents, "request_ids": [],
                                      "request_set": set(), "metadata": values,
                                      "parent_validation": validation}
            return group_id

    def add_request(self, group_id, request_id):
        """Add a direct successful-RPC reference once; caller supplies its receipt."""
        _text(group_id, "group_id")
        _text(request_id, "request_id")
        with self._lock:
            if group_id not in self._active:
                raise ValueError("Lineage group is unknown or already closed")
            group = self._active[group_id]
            if request_id not in group["request_set"]:
                if len(group["request_ids"]) >= self.max_requests_per_group:
                    raise RuntimeError("Lineage direct request count exceeds configured group bound")
                group["request_set"].add(request_id)
                group["request_ids"].append(request_id)
            return group_id

    def _remember_closed(self, group_id):
        if not self.max_recent_closed:
            return
        while len(self._recent_order) >= self.max_recent_closed:
            self._recent.remove(self._recent_order.popleft())
        self._recent_order.append(group_id)
        self._recent.add(group_id)

    def end_group(self, group_id):
        """Emit one finalized group then forget its requests and metadata."""
        with self._lock:
            if group_id not in self._active:
                raise ValueError("Lineage group is unknown or already closed")
            group = self._active[group_id]
            details = {key: value for key, value in group.items() if key != "request_set"}
            self._emit("lineage_group", details)
            del self._active[group_id]
            self._remember_closed(group_id)
            return group_id

    def abort_group(self, group_id, reason):
        """Drop a failed group without making it usable as a sample dependency."""
        _text(reason, "abort reason")
        with self._lock:
            if group_id not in self._active:
                raise ValueError("Lineage group is unknown or already closed")
            group = self._active[group_id]
            self._emit("lineage_group_aborted", {"group_id": group_id, "stage": group["stage"],
                                                 "reason": reason})
            del self._active[group_id]

    def sample_producer(self, sample_id, stage_groups, metadata=None, *, parents_known_by_caller=False):
        """Emit sample->stage-group edges without retaining a sample registry.

        Example stages are state_prefix/action/world/reward/gae_suffix/bootstrap.
        Their meaning and coverage are explicit integration responsibilities.
        """
        _text(sample_id, "sample_id")
        if not isinstance(stage_groups, dict) or not stage_groups or len(stage_groups) > self.max_parents:
            raise ValueError("Sample lineage requires a bounded stage->group-ID mapping")
        normalized = {}
        for stage, groups in stage_groups.items():
            _text(stage, "sample stage", 128)
            normalized[stage] = _ids([groups] if isinstance(groups, str) else groups,
                                     "sample group IDs", self.max_parents)
        all_groups = _ids([group for values in normalized.values() for group in values],
                          "sample group IDs", self.max_parents)
        values = _metadata(metadata, self.max_metadata_bytes)
        with self._lock:
            validation = self._validate_references(all_groups, parents_known_by_caller)
            self._emit("lineage_sample_producer", {"sample_id": sample_id,
                       "stage_groups": normalized, "metadata": values,
                       "group_validation": validation})
        return sample_id

    def cache_state(self):
        """Cheap bounded-state diagnostics; does not traverse historical events."""
        with self._lock:
            return {"active_groups": len(self._active),
                    "active_request_references": sum(len(group["request_ids"]) for group in self._active.values()),
                    "recent_closed_ids": len(self._recent),
                    "successful_request_registry_size": 0,
                    "sample_registry_size": 0,
                    "total_groups_started": self._next_group}
