"""Cooperative admission by estimated reference-device execution seconds.

This is a submission scheduler, not a GPU preemptor.  Submitted work is never
cancelled, and resources are explicit logical lanes rather than whole-card
locks.  Independent replicas charge the same service account.  Queue delay is
never an execution-cost observation.  Wall-time observations remain labelled
as proxies; this module does not initialize CUDA or measure hardware counters.
"""
from __future__ import annotations

from collections import deque, OrderedDict
from dataclasses import dataclass, field, asdict
from copy import deepcopy
import json
import math
from typing import Callable, Iterable, Mapping


COST_UNIT = "reference_device_seconds"


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'} and finite")
    return value


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 1024:
        raise ValueError(f"{name} must be a nonempty bounded string")
    return value


def _count(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def normalize_execution_cost(execution_seconds, device_count=1, *, capacity_scale=1.0):
    """Normalize execution only, not enqueue-to-reply duration.

    Equal reference devices have capacity_scale=1.  For a gang, callers may sum
    per-rank execution intervals instead of multiplying the maximum rank time.
    The result is a resource-cost proxy unless hardware events were measured.
    """
    return (_number(execution_seconds, "execution_seconds")
            * _count(device_count, "device_count")
            * _number(capacity_scale, "capacity_scale", positive=True))


@dataclass(frozen=True)
class WorkItem:
    work_id: str
    service: str
    cost_key: str
    units: float = 1.0
    replica: str | None = None
    dependencies: tuple[str, ...] = ()
    resources: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in ("work_id", "service", "cost_key"):
            _identifier(getattr(self, name), name)
        object.__setattr__(self, "units", _number(self.units, "units", positive=True))
        if self.replica is not None:
            _identifier(self.replica, "replica")
        for name in ("dependencies", "resources"):
            values = getattr(self, name)
            if not isinstance(values, (list, tuple)):
                raise ValueError(f"{name} must be a list or tuple")
            values = tuple(_identifier(x, name) for x in values)
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates")
            object.__setattr__(self, name, values)
        if self.work_id in self.dependencies:
            raise ValueError("A work item cannot depend on itself")
        if type(self.metadata) is not dict:
            raise ValueError("metadata must be a JSON object")
        try:
            json.dumps(self.metadata, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must be finite JSON data") from exc
        object.__setattr__(self, "metadata", deepcopy(self.metadata))


@dataclass(frozen=True)
class ServiceConfig:
    weight: float = 1.0
    max_inflight: int = 1

    def __post_init__(self):
        object.__setattr__(self, "weight", _number(self.weight, "weight", positive=True))
        _count(self.max_inflight, "max_inflight")


@dataclass(frozen=True)
class WorkGrant:
    item: WorkItem
    estimated_cost: float
    dispatch_weight: float
    virtual_finish: float
    dispatch_index: int
    cost_unit: str = COST_UNIT

    @property
    def work_id(self):
        return self.item.work_id

    @property
    def service(self):
        return self.item.service


class CostModel:
    """EWMA execution cost per *completed work unit*, keyed by shape/phase.

    Different batch shapes and backend phases must use distinct cost keys.
    Queue wait must not be supplied to update().  provenance is reported as
    supplied; wall proxies are deliberately not called measured GPU time.
    """
    def __init__(self, initial_costs=None, *, default_unit_cost=1.0, alpha=0.2,
                 min_unit_cost=1e-9, max_unit_cost=1e9):
        self.default_unit_cost = _number(default_unit_cost, "default_unit_cost", positive=True)
        self.alpha = _number(alpha, "alpha", positive=True)
        if self.alpha > 1:
            raise ValueError("alpha must be <= 1")
        self.min_unit_cost = _number(min_unit_cost, "min_unit_cost", positive=True)
        self.max_unit_cost = _number(max_unit_cost, "max_unit_cost", positive=True)
        if self.max_unit_cost < self.min_unit_cost:
            raise ValueError("Cost-model bounds are reversed")
        self._costs = {}
        self._observations = {}
        for key, value in dict(initial_costs or {}).items():
            self._costs[_identifier(key, "cost_key")] = self._bounded(_number(value, "unit_cost", positive=True))

    def _bounded(self, value):
        return max(self.min_unit_cost, min(self.max_unit_cost, value))

    def estimate(self, cost_key, units=1.0):
        _identifier(cost_key, "cost_key")
        result = self._costs.get(cost_key, self.default_unit_cost) * _number(units, "units", positive=True)
        return _number(result, "estimated_cost", positive=True)

    def update(self, cost_key, observed_cost, units=1.0, *, provenance="execution_wall_proxy"):
        _identifier(cost_key, "cost_key")
        _identifier(provenance, "provenance")
        observed = _number(observed_cost, "observed_cost")
        units = _number(units, "units", positive=True)
        unit_cost = self._bounded(observed / units)
        previous = self._costs.get(cost_key, self.default_unit_cost)
        self._costs[cost_key] = self._bounded((1-self.alpha)*previous + self.alpha*unit_cost)
        old = self._observations.get(cost_key, {"count": 0, "total_cost": 0.0, "total_units": 0.0})
        self._observations[cost_key] = {"count": old["count"]+1,
            "total_cost": old["total_cost"]+observed, "total_units": old["total_units"]+units,
            "last_provenance": provenance, "last_observed_unit_cost": unit_cost}
        return self._costs[cost_key]

    def snapshot(self):
        return {"cost_unit": COST_UNIT, "default_unit_cost": self.default_unit_cost,
                "alpha": self.alpha, "unit_costs": dict(self._costs),
                "observations": deepcopy(self._observations)}


class WeightedCostScheduler:
    """Weighted virtual-finish admission with completion feedback.

    A single owner (e.g. one Ray actor) must serialize calls to this object.
    Selection uses projected finish F + estimated_cost / current_weight.
    Dispatch reserves that amount immediately, including concurrent replicas.
    Completion corrects (observed-estimated)/dispatch_weight exactly once.
    A dependency only becomes satisfied after successful full completion.

    Terminal payload/history retention is bounded. Work IDs are globally unique
    for this scheduler lifetime, including after retirement. This calling
    contract is mandatory for arbitrary IDs: bounded recent history cannot
    provide exact all-history deduplication of arbitrary strings. Callers that
    need enforcement can register a fixed identity namespace and enqueue its
    canonical ``namespace:sequence`` IDs with a monotonic sequence watermark.
    Retired completions are rejected, never treated as new work completion.

    If a completion will be a dependency of work enqueued much later, explicitly
    retain its lightweight token with ``retire_completed(...,
    keep_dependency=True)`` and release it once that dependency is no longer
    needed. Automatic history eviction never invents dependency satisfaction.
    """
    def __init__(self, *, cost_model=None, resource_limits=None,
                 max_pending=10000, max_inflight_total=None, max_replica_inflight=1,
                 max_terminal_history=1024, max_dependency_tokens=4096,
                 max_identity_namespaces=128):
        self.cost_model = cost_model if cost_model is not None else CostModel()
        self.resource_limits = {_identifier(k, "resource"): _count(v, "capacity")
                                for k, v in dict(resource_limits or {}).items()}
        self.max_pending = _count(max_pending, "max_pending")
        self.max_inflight_total = (None if max_inflight_total is None else
                                   _count(max_inflight_total, "max_inflight_total"))
        self.max_replica_inflight = _count(max_replica_inflight, "max_replica_inflight")
        self.max_terminal_history = _count(max_terminal_history, "max_terminal_history")
        self.max_dependency_tokens = _count(max_dependency_tokens, "max_dependency_tokens")
        self.max_identity_namespaces = _count(max_identity_namespaces, "max_identity_namespaces")
        self._services, self._queues, self._work, self._state = {}, {}, {}, {}
        self._inflight, self._completed, self._resources, self._replicas = {}, {}, {}, {}
        self._order = {}; self._sequence = 0; self._virtual_time = 0.0
        self._terminal_order = OrderedDict()
        self._dependents = {}
        self._dependency_tokens = set()
        self._identity_watermarks = {}
        self._retired_count = 0
        self._resource_reservation = None

    def register_service(self, service, config=None):
        _identifier(service, "service")
        if service in self._services:
            raise ValueError("Service already registered")
        config = ServiceConfig() if config is None else config
        if not isinstance(config, ServiceConfig):
            raise ValueError("config must be ServiceConfig")
        self._services[service] = {"config": config, "finish": self._virtual_time,
            "eligible": False, "inflight": 0, "dispatched": 0, "completed": 0,
            "failed": 0, "observed_cost": 0.0, "completed_units": 0.0,
            "estimated_cost": 0.0, "correction": 0.0}
        self._queues[service] = deque()

    def register_identity_namespace(self, namespace):
        """Register a fixed producer's exact monotonic work-ID namespace.

        Namespaces should identify long-lived producer replicas, not individual
        requests. Their count is bounded separately from terminal history.
        """
        _identifier(namespace, "identity_namespace")
        if namespace in self._identity_watermarks:
            raise ValueError("Identity namespace already registered")
        if len(self._identity_watermarks) >= self.max_identity_namespaces:
            raise RuntimeError("Identity namespace bound reached")
        self._identity_watermarks[namespace] = -1

    def enqueue(self, item, *, identity_namespace=None, identity_sequence=None):
        if not isinstance(item, WorkItem):
            raise ValueError("item must be WorkItem")
        if item.service not in self._services:
            raise ValueError("Unknown service")
        if item.work_id in self._work or item.work_id in self._dependency_tokens:
            raise ValueError("Work identity has already been used")
        if (identity_namespace is None) != (identity_sequence is None):
            raise ValueError("Identity namespace and sequence must be supplied together")
        if identity_namespace is not None:
            if identity_namespace not in self._identity_watermarks:
                raise ValueError("Identity namespace must be registered before use")
            if type(identity_sequence) is not int or identity_sequence < 0:
                raise ValueError("Identity sequence must be a canonical nonnegative integer")
            if item.work_id != f"{identity_namespace}:{identity_sequence}":
                raise ValueError("Work ID does not match its exact namespace and sequence")
            if identity_sequence <= self._identity_watermarks[identity_namespace]:
                raise ValueError("Identity sequence was already used or regressed")
        elif any(item.work_id.startswith(namespace + ':') for namespace in self._identity_watermarks):
            raise ValueError("Registered identity namespaces require their monotonic sequence")
        if sum(len(q) for q in self._queues.values()) >= self.max_pending:
            raise RuntimeError("Pending work bound reached")
        if self.resource_limits and any(x not in self.resource_limits for x in item.resources):
            raise ValueError("Unknown logical resource")
        def reaches_self(work_id, visited):
            if work_id == item.work_id:
                return True
            if work_id in visited or work_id not in self._work:
                return False
            visited.add(work_id)
            return any(reaches_self(x, visited) for x in self._work[work_id].dependencies)
        if any(reaches_self(dep, set()) for dep in item.dependencies):
            raise ValueError("Dependency cycle")
        item = deepcopy(item)
        state = self._services[item.service]
        if not self._queues[item.service] and state["inflight"] == 0:
            state["finish"] = max(state["finish"], self._virtual_time)
        self._sequence += 1
        self._work[item.work_id] = item; self._state[item.work_id] = "pending"
        self._order[item.work_id] = self._sequence
        self._queues[item.service].append(item.work_id)
        for dep in item.dependencies:
            self._dependents.setdefault(dep, set()).add(item.work_id)
        if identity_namespace is not None:
            self._identity_watermarks[identity_namespace] = identity_sequence
        return item.work_id

    def reserve_pending_group(self, work_ids, *, gate_resource):
        """Reserve one future lane per resource for a fully arrived rank group.

        Existing executions drain normally. Spare capacity remains available;
        no reservation is made while another group owns the training gate.
        """
        if self._state.get(self._resource_reservation) == 'pending':
            return self._resource_reservation
        self._resource_reservation = None
        if self._resources.get(gate_resource, 0) >= self.resource_limits[gate_resource]:
            return None
        candidates = [self._work[wid] for wid in work_ids if self._state.get(wid) == 'pending']
        if candidates:
            def priority(item):
                state = self._services[item.service]
                return (state['finish'] + self.cost_model.estimate(item.cost_key, item.units)
                        / state['config'].weight, self._order[item.work_id])
            self._resource_reservation = min(candidates, key=priority).work_id
        return self._resource_reservation

    def _can_run(self, item, ready, available_resources):
        if any(self._state.get(dep) != "completed" and dep not in self._dependency_tokens
               for dep in item.dependencies):
            return False
        if item.replica is not None and self._replicas.get(item.replica, 0) >= self.max_replica_inflight:
            return False
        for resource in item.resources:
            limit = self.resource_limits.get(resource)
            reserved = self._work.get(self._resource_reservation)
            if reserved is not None and item.work_id != reserved.work_id and resource in reserved.resources:
                limit = max(0, limit-1) if limit is not None else limit
            if limit is not None and self._resources.get(resource, 0) >= limit:
                return False
            if available_resources is not None:
                if isinstance(available_resources, Mapping):
                    if available_resources.get(resource, 0) <= 0:
                        return False
                elif resource not in available_resources:
                    return False
        return ready is None or bool(ready(deepcopy(item)))

    def claim_next(self, *, ready: Callable[[WorkItem], bool] | None = None,
                   available_resources: Iterable[str] | Mapping[str, int] | None = None):
        if self.max_inflight_total is not None and len(self._inflight) >= self.max_inflight_total:
            return None
        if ready is not None and not callable(ready):
            raise ValueError("ready must be callable")
        if available_resources is not None and not isinstance(available_resources, Mapping):
            available_resources = frozenset(available_resources)
        candidates = []
        for name, state in self._services.items():
            config = state["config"]
            eligible_item = None
            if state["inflight"] < config.max_inflight:
                eligible_item = next((self._work[key] for key in self._queues[name]
                                     if self._can_run(self._work[key], ready, available_resources)), None)
            if eligible_item is None:
                state["eligible"] = False
                continue
            # Only enqueue() places genuinely idle services at the floor.
            # Resource/max-inflight blocking must not erase completion refunds.
            state["eligible"] = True
            cost = self.cost_model.estimate(eligible_item.cost_key, eligible_item.units)
            finish = state["finish"] + cost/config.weight
            candidates.append((finish, self._order[eligible_item.work_id], name, eligible_item, cost))
        if not candidates:
            return None
        finish, _, name, item, cost = min(candidates, key=lambda x:
            (x[3].work_id != self._resource_reservation, x[0], x[1]))
        state = self._services[name]; config = state["config"]
        self._queues[name].remove(item.work_id)
        self._sequence += 1
        grant = WorkGrant(deepcopy(item), cost, config.weight, finish, self._sequence)
        self._inflight[item.work_id] = grant; self._state[item.work_id] = "inflight"
        if self._resource_reservation == item.work_id:
            self._resource_reservation = None
        state["finish"] = finish; state["inflight"] += 1; state["dispatched"] += 1
        state["estimated_cost"] += cost
        for resource in item.resources:
            self._resources[resource] = self._resources.get(resource, 0)+1
        if item.replica is not None:
            self._replicas[item.replica] = self._replicas.get(item.replica, 0)+1
        active = [s["finish"] for s in self._services.values() if s["eligible"] or s["inflight"]]
        if active:
            self._virtual_time = max(self._virtual_time, min(active))
        return deepcopy(grant)

    def complete(self, work_id, observed_cost, *, completed_units=None,
                 provenance="execution_wall_proxy", success=True):
        _identifier(work_id, "work_id"); _identifier(provenance, "provenance")
        if type(success) is not bool:
            raise ValueError("success must be bool")
        if work_id not in self._inflight:
            raise ValueError("Only inflight work may complete; duplicate completion is forbidden")
        observed_cost = _number(observed_cost, "observed_cost")
        grant = self._inflight[work_id]; item = grant.item
        units = item.units if completed_units is None and success else (0.0 if completed_units is None else completed_units)
        units = _number(units, "completed_units")
        if units > item.units or (success and units != item.units):
            raise ValueError("Successful work must complete its full declared units")
        state = self._services[item.service]
        correction = (observed_cost-grant.estimated_cost)/grant.dispatch_weight
        state["finish"] = max(0.0, state["finish"]+correction)
        state["correction"] += correction; state["observed_cost"] += observed_cost
        state["completed_units"] += units; state["inflight"] -= 1
        state["completed" if success else "failed"] += 1
        self._state[work_id] = "completed" if success else "failed"
        del self._inflight[work_id]
        for resource in item.resources:
            self._resources[resource] -= 1
            if self._resources[resource] == 0:
                del self._resources[resource]
        if item.replica is not None:
            self._replicas[item.replica] -= 1
            if self._replicas[item.replica] == 0:
                del self._replicas[item.replica]
        if success:
            self.cost_model.update(item.cost_key, observed_cost, item.units, provenance=provenance)
        record = {"work_id": work_id, "service": item.service, "status": self._state[work_id],
            "cost_unit": COST_UNIT, "provenance": provenance, "estimated_cost": grant.estimated_cost,
            "observed_cost": observed_cost, "dispatch_weight": grant.dispatch_weight,
            "virtual_correction": correction, "completed_units": units,
            "dispatch_index": grant.dispatch_index}
        self._completed[work_id] = record
        self._detach_dependencies(work_id)
        self._remember_terminal(work_id)
        return deepcopy(record)

    def cancel(self, work_id):
        """Withdraw pending work only. Already submitted work must finish."""
        _identifier(work_id, "work_id")
        if self._state.get(work_id) != "pending":
            raise ValueError("Only pending work may be cancelled; inflight work is not preemptible")
        item = self._work[work_id]; self._queues[item.service].remove(work_id)
        self._state[work_id] = "cancelled"
        if self._resource_reservation == work_id:
            self._resource_reservation = None
        self._detach_dependencies(work_id)
        self._remember_terminal(work_id)
        return True

    def _detach_dependencies(self, work_id):
        for dep in self._work[work_id].dependencies:
            users = self._dependents.get(dep)
            if users is not None:
                users.discard(work_id)
                if not users:
                    del self._dependents[dep]

    def _remember_terminal(self, work_id):
        self._terminal_order[work_id] = None
        # Terminal work with pending/inflight dependents is pinned until they
        # finish/cancel. Extra pinned entries are bounded by the live dependency
        # graph, not by total historical request count.
        if len(self._terminal_order) > self.max_terminal_history:
            for key in tuple(self._terminal_order):
                if len(self._terminal_order) <= self.max_terminal_history:
                    break
                if not self._dependents.get(key):
                    self.retire_completed(key)

    def retire_completed(self, work_id, *, keep_dependency=False):
        """Release one terminal payload without changing aggregate accounting.

        Returns False if already absent or pinned by pending/inflight children.
        Active/pending work raises: it must never be retired or preempted.
        keep_dependency retains a successful completion token for future work;
        it is explicit and bounded, never an automatically evicted tombstone.
        """
        _identifier(work_id, "work_id")
        if type(keep_dependency) is not bool:
            raise ValueError("keep_dependency must be bool")
        state = self._state.get(work_id)
        if state is None:
            return False
        if state not in ("completed", "failed", "cancelled"):
            raise ValueError("Only terminal work may retire; pending/inflight work must continue")
        if self._dependents.get(work_id):
            return False
        if keep_dependency:
            if state != "completed":
                raise ValueError("Only successful completion may become a dependency token")
            if work_id not in self._dependency_tokens and len(self._dependency_tokens) >= self.max_dependency_tokens:
                raise RuntimeError("Retained dependency-token bound reached")
            self._dependency_tokens.add(work_id)
        self._terminal_order.pop(work_id, None)
        self._completed.pop(work_id, None)
        self._order.pop(work_id, None)
        self._state.pop(work_id, None)
        self._work.pop(work_id, None)
        self._retired_count += 1
        return True

    def release_dependency(self, work_id):
        """Drop an explicitly retained token after every live user has finished."""
        _identifier(work_id, "work_id")
        if self._dependents.get(work_id):
            raise ValueError("A live pending/inflight dependency still needs this token")
        if work_id not in self._dependency_tokens:
            return False
        self._dependency_tokens.remove(work_id)
        return True

    def set_weights(self, weights):
        if not isinstance(weights, Mapping) or not weights:
            raise ValueError("weights must be a nonempty service mapping")
        validated = {}
        for name, weight in weights.items():
            if name not in self._services:
                raise ValueError("Unknown service")
            validated[name] = _number(weight, "weight", positive=True)
        for name, weight in validated.items():
            old = self._services[name]["config"]
            self._services[name]["config"] = ServiceConfig(weight, old.max_inflight)

    def snapshot(self):
        return {"cost_unit": COST_UNIT, "virtual_time": self._virtual_time,
            "resource_reservation": self._resource_reservation,
            "services": {name: {**{k: v for k, v in state.items() if k != "config"},
                                      "config": asdict(state["config"]),
                                      "pending": len(self._queues[name])}
                         for name, state in self._services.items()},
            "inflight": {key: asdict(value) for key, value in self._inflight.items()},
            "states": dict(self._state), "resource_inflight": dict(self._resources),
            "replica_inflight": dict(self._replicas), "cost_model": self.cost_model.snapshot(),
            "completed": deepcopy(self._completed),
            "retention": {"max_terminal_history": self.max_terminal_history,
                "terminal_history": len(self._terminal_order),
                "pinned_terminal_count": sum(bool(self._dependents.get(key)) for key in self._terminal_order),
                "retired_count": self._retired_count,
                "retained_dependency_tokens": sorted(self._dependency_tokens),
                "max_dependency_tokens": self.max_dependency_tokens,
                "identity_watermarks": dict(self._identity_watermarks),
                "max_identity_namespaces": self.max_identity_namespaces,
                "identity_contract": "Arbitrary work IDs are globally unique for scheduler lifetime; retired IDs cannot be reused. Registered namespaces enforce exact monotonic IDs."},
            "scope": "Cooperative bounded admission; no preemption, GPU partitioning, or full-card drain."}
