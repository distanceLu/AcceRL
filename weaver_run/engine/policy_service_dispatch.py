"""Four-class cooperative Policy admission, independent of Torch and Ray.

The actor owns one model. Refresh jobs preserve ordered K+1 observations and a
published-version pin, but yield after each bounded chunk. Global accounting
is performed by the broker; this queue only chooses the next local class.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from copy import deepcopy
import math
import uuid

from cost_scheduler import CostModel


CLASSES = ("real", "imagined", "refresh", "eval")
WORKFLOW_CLASS = {"real_rollout": "real", "imagined_action": "imagined",
                  "imagined_bootstrap": "imagined", "evaluation": "eval"}


def request_class(metadata):
    if not isinstance(metadata, dict) or metadata.get("workflow") not in WORKFLOW_CLASS:
        raise ValueError("Scheduled Policy actions require an explicit known workflow")
    return WORKFLOW_CLASS[metadata["workflow"]]


def dispatch_settings(settings):
    if not isinstance(settings, dict):
        raise ValueError("Policy dispatch settings must be a mapping")
    out = dict(settings)
    allowed = {"enabled", "refresh_micro", "action_micro", "class_weights", "max_pending"}
    if set(out)-allowed:
        raise ValueError("Unknown Policy dispatch settings")
    if type(out.get("enabled", True)) is not bool:
        raise ValueError("enabled must be bool")
    out.setdefault("enabled", True)
    for key in ("refresh_micro", "action_micro"):
        out.setdefault(key, 2)
        if type(out[key]) is not int or not 1 <= out[key] <= 2:
            raise ValueError("Policy dispatch micro limits must be 1 or 2")
    out.setdefault("max_pending", 4096)
    if type(out["max_pending"]) is not int or not 1 <= out["max_pending"] <= 65536:
        raise ValueError("Policy pending bound must be 1..65536")
    weights = dict(out.get("class_weights", {k:1.0 for k in CLASSES}))
    if set(weights) != set(CLASSES):
        raise ValueError("All four Policy request classes need weights")
    for key, value in weights.items():
        if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Policy class weights must be positive finite numbers")
    out["class_weights"] = {k:float(v) for k,v in weights.items()}
    return out


def refresh_batch_sizes(count, micro, batch_sizes=None):
    if type(count) is not int or not 1 <= count <= 4096 or type(micro) is not int or micro not in (1,2):
        raise ValueError('Refresh batch geometry is outside the reviewed bound')
    if batch_sizes is None:
        return tuple(min(micro,count-start) for start in range(0,count,micro))
    if (not isinstance(batch_sizes,(list,tuple)) or not batch_sizes
            or any(type(n) is not int or not 1 <= n <= micro for n in batch_sizes)
            or sum(batch_sizes) != count):
        raise ValueError('Explicit refresh batches must cover exactly the bounded input list')
    return tuple(batch_sizes)


@dataclass
class ActionJob:
    inputs: object
    deterministic: bool
    metadata: dict
    future: object


@dataclass
class RefreshJob:
    job_id: str
    inputs: tuple
    version: int
    future: object
    batch_sizes: tuple = ()
    batch_index: int = 0
    offset: int = 0
    values: list = field(default_factory=list)
    ready: bool = False
    reported_remaining: int = 0


@dataclass
class DispatchBatch:
    batch_id: str
    request_class: str
    inputs: tuple
    estimated_cost: float
    weight: float
    actions: tuple = ()
    refresh_job: RefreshJob | None = None
    refresh_start: int = 0

    @property
    def units(self):
        return len(self.inputs)

    @property
    def metadata(self):
        return [x.metadata for x in self.actions]


class PolicyServiceDispatch:
    def __init__(self, settings):
        self.settings = dispatch_settings(settings)
        self.weights = self.settings["class_weights"]
        self.queues = {k:deque() for k in CLASSES}
        self.cost_model = CostModel(default_unit_cost=.1)
        self.finish = {k:0.0 for k in CLASSES}
        self.virtual_time = 0.0
        self.active = None
        self._request_ids = set()
        self._refresh = {}

    @property
    def pending_count(self):
        return sum(len(q) for q in self.queues.values())

    @property
    def refresh_pending(self):
        return bool(self._refresh)

    def assert_publication_allowed(self):
        if self.refresh_pending:
            raise RuntimeError("Policy publication is forbidden while current-critic refresh is pending")

    def _capacity(self):
        if self.pending_count >= self.settings["max_pending"]:
            raise RuntimeError("Policy class queues reached their configured bound")

    def enqueue_action(self, inputs, deterministic, metadata, future):
        cls = request_class(metadata)
        if type(deterministic) is not bool:
            raise ValueError("deterministic must be bool")
        rid = metadata.get("request_id")
        if not isinstance(rid,str) or not rid or rid in self._request_ids:
            raise ValueError("Scheduled Policy requests need a unique request_id")
        self._capacity()
        if not self.queues[cls]:
            self.finish[cls] = max(self.finish[cls], self.virtual_time)
        job = ActionJob(inputs, deterministic, deepcopy(metadata), future)
        self.queues[cls].append(job); self._request_ids.add(rid)
        return rid

    def owns_request(self, request_id):
        return request_id in self._request_ids

    def cancel_queued_action(self, request_id):
        """Cancellation withdraws queued work, never an already selected batch."""
        for cls in ("real", "imagined", "eval"):
            for job in self.queues[cls]:
                if job.metadata['request_id'] == request_id:
                    self.queues[cls].remove(job); self._request_ids.remove(request_id)
                    return True
        return False

    def enqueue_refresh(self, inputs, version, future, *, batch_sizes=None):
        if type(version) is not int or version < 0 or not isinstance(inputs,(list,tuple)) or not 1 <= len(inputs) <= 4096:
            raise ValueError("Refresh needs a nonnegative exact version and 1..4096 observations")
        sizes = refresh_batch_sizes(len(inputs), self.settings['refresh_micro'], batch_sizes)
        self._capacity()
        if any(job.version != version for job in self._refresh.values()):
            raise ValueError("Queued refresh jobs cannot mix published Policy versions")
        job = RefreshJob('refresh:'+uuid.uuid4().hex, tuple(inputs), version, future, batch_sizes=sizes)
        if not self.queues['refresh']:
            self.finish['refresh'] = max(self.finish['refresh'], self.virtual_time)
        self.queues['refresh'].append(job); self._refresh[job.job_id] = job
        return job

    def next_batch(self, version, *, action_limit=None):
        if self.active is not None:
            raise RuntimeError("An actor may execute only one selected Policy batch")
        limit = min(self.settings['action_micro'], action_limit or self.settings['action_micro'])
        candidates = []
        for index, cls in enumerate(CLASSES):
            q = self.queues[cls]
            if not q or (cls == 'refresh' and not q[0].ready):
                continue
            if cls == 'refresh':
                job = q[0]
                if job.version != version:
                    raise RuntimeError("Policy version changed during a pinned refresh job")
                count = job.batch_sizes[job.batch_index]
            else:
                count = min(limit,len(q))
            cost = self.cost_model.estimate(cls, count)
            candidates.append((self.finish[cls]+cost/self.weights[cls],index,cls,count,cost))
        if not candidates:
            return None
        finish,_,cls,count,cost = min(candidates)
        self.finish[cls] = finish
        if cls == 'refresh':
            job = self.queues[cls][0]
            batch = DispatchBatch('policy_batch:'+uuid.uuid4().hex,cls,
                job.inputs[job.offset:job.offset+count],cost,self.weights[cls],
                refresh_job=job,refresh_start=job.offset)
        else:
            jobs = tuple(self.queues[cls].popleft() for _ in range(count))
            batch = DispatchBatch('policy_batch:'+uuid.uuid4().hex,cls,
                tuple(x.inputs for x in jobs),cost,self.weights[cls],actions=jobs)
        self.active = batch
        return batch

    def complete_batch(self, batch, observed_cost):
        if self.active is not batch:
            raise RuntimeError("Policy batch completion identity mismatch")
        if isinstance(observed_cost,bool) or not isinstance(observed_cost,(int,float)) or not math.isfinite(observed_cost) or observed_cost < 0:
            raise ValueError("Policy execution cost must be finite and nonnegative")
        cls = batch.request_class
        self.finish[cls] = max(0.,self.finish[cls]+(observed_cost-batch.estimated_cost)/batch.weight)
        self.cost_model.update(cls,observed_cost,batch.units,provenance='execution_wall_proxy')
        for job in batch.actions:
            self._request_ids.remove(job.metadata['request_id'])
        self.active = None
        # Advance the idle-service floor from corrected execution progress,
        # not a potentially overestimated first dispatch reservation. Otherwise
        # a newly arrived action can inherit an entire long-refresh estimate.
        active = [self.finish[k] for k,q in self.queues.items() if q or k == cls]
        self.virtual_time = max(self.virtual_time,min(active))

    def finish_refresh(self, batch, values, version):
        job = batch.refresh_job
        if job is None or job.job_id not in self._refresh or job.offset != batch.refresh_start or version != job.version:
            raise RuntimeError("Refresh chunk identity, order, or Policy version mismatch")
        if len(values) != batch.units or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in values):
            raise ValueError("Refresh chunk must return finite per-observation values")
        if batch.units != job.batch_sizes[job.batch_index]:
            raise ValueError('Refresh completion changed an explicit micro boundary')
        job.values.extend(float(x) for x in values); job.offset += len(values)
        job.batch_index += 1
        if job.offset == len(job.inputs):
            if self.queues['refresh'][0] is not job:
                raise RuntimeError("Refresh completion order mismatch")
            self.queues['refresh'].popleft();del self._refresh[job.job_id]
            if not job.future.done():
                job.future.set_result({'policy_version':job.version,'values':job.values})
            return True
        return False

    def fail_all(self, error):
        demand = sum(job.reported_remaining for job in self._refresh.values())
        futures = [job.future for cls,q in self.queues.items() for job in q]
        if self.active is not None:
            futures.extend(job.future for job in self.active.actions)
        for job in self._refresh.values():
            job.reported_remaining = 0
        for future in futures:
            if not future.done():
                future.set_exception(error)
        self.queues = {k:deque() for k in CLASSES}
        self._refresh.clear();self._request_ids.clear();self.active = None
        return demand

    def snapshot(self):
        return {'pending_by_class':{k:len(q) for k,q in self.queues.items()},
                'refresh_observations_remaining':sum(len(x.inputs)-x.offset for x in self._refresh.values()),
                'active_class':None if self.active is None else self.active.request_class,
                'virtual_finish':dict(self.finish),'cost_unit':'reference_device_seconds',
                'cost_provenance':'execution_wall_proxy','max_refresh_micro':self.settings['refresh_micro']}
