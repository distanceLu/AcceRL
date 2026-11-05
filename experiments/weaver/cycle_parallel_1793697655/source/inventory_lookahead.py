"""CPU-only, read-only capacity forecasting and expiring production advice.

This estimates feasible future cohorts, NOT the cohorts that random Replay will
actually draw. It never reserves samples or changes their learning attributes.
The production owner may spend existing credits early; it cannot mint credits.
"""
from collections import Counter, deque
import asyncio
import math
import time


def _flow(groups, quota, horizon, new_count):
    """Integral max flow for groups of identical per-sample use/age capacity.

    A group has C samples with U remaining uses. Its total capacity is C*U,
    but each future batch can take at most C distinct samples from the group.
    No sample is assigned here: this is only a capacity-envelope calculation.
    """
    groups = list(groups.items()) + [((horizon, (True,) * horizon), new_count)]
    end = 1 + len(groups) + horizon
    edges = [[] for _ in range(end + 1)]

    def edge(a, b, capacity):
        edges[a].append([b, capacity, len(edges[b])])
        edges[b].append([a, 0, len(edges[a]) - 1])

    for index, ((uses, mask), count) in enumerate(groups, 1):
        edge(0, index, count * min(uses, horizon))
        for step, eligible in enumerate(mask):
            if eligible:
                edge(index, 1 + len(groups) + step, count)
    for step in range(horizon):
        edge(1 + len(groups) + step, end, quota)
    total = 0
    while True:
        parents = {0: None}
        queue = deque([0])
        while queue and end not in parents:
            node = queue.popleft()
            for index, (target, cap, _) in enumerate(edges[node]):
                if cap and target not in parents:
                    parents[target] = (node, index)
                    queue.append(target)
        if end not in parents:
            return total
        node, amount = end, quota * horizon
        while node:
            previous, index = parents[node]
            amount = min(amount, edges[previous][index][1])
            node = previous
        node = end
        while node:
            previous, index = parents[node]
            target, _, reverse = edges[previous][index]
            edges[previous][index][1] -= amount
            edges[target][reverse][1] += amount
            node = previous
        total += amount


def forecast(store, horizon, *, batch_size=576, world_interval=5):
    """Forecast only the admitted fixed FRESH recipe and aligned publications."""
    if type(horizon) is not int or not 1 <= horizon <= 4:
        raise ValueError('lookahead horizon must be an integer in [1,4]')
    cfg = store.config
    if (batch_size != 576 or world_interval != 5 or not cfg.replay_age_limits
            or not cfg.real_policy_replay or cfg.real_fraction != .25
            or cfg.max_sample_uses != 4 or cfg.max_policy_age != 16
            or cfg.max_world_age != 3):
        raise ValueError('lookahead requires the reviewed global576, 5:1 FRESH recipe')
    if store.versions is None:
        return {'ready': False, 'reason': 'unpublished'}
    policy, world = store.versions
    if world != policy // world_interval:
        return {'ready': False, 'reason': 'publication_in_progress'}
    future = [(policy + step, (policy + step) // world_interval)
              for step in range(1, horizon + 1)]
    from trajectory_sampling import enabled, forecast_groups, UNIT
    unit = UNIT if enabled(cfg) else 1
    if unit != 1:
        groups, eligible_now = forecast_groups(store, future)
    else:
        groups, eligible_now = Counter(), 0
        for sample in store.samples.values():
            segment = store.segments[sample['segment_id']]
            if segment.source != 'imagined' or not store._eligible(sample):
                continue
            eligible_now += 1
            versions = segment.versions[sample['index']]
            born_world = versions.get('world_reward_bundle_version', versions.get('world_bundle_version'))
            mask = tuple(0 <= p - versions['policy'] <= cfg.max_policy_age
                         and 0 <= w - born_world <= cfg.max_world_age for p, w in future)
            uses = cfg.max_sample_uses - sample['uses'] - sample['reserved']
            if any(mask):
                groups[(uses, mask)] += 1
    quota = 432//unit
    capacity = _flow(groups, quota, horizon, 0)
    low, high = 0, quota
    while low < high:
        mid = (low + high) // 2
        if _flow(groups, quota, horizon, mid) == quota * horizon:
            high = mid
        else:
            low = mid + 1
    return dict(ready=True, versions=[policy, world], horizon=horizon,
                future_versions=future, imagined_per_update=quota*unit,
                eligible_now=eligible_now*unit, feasible_exposures=capacity*unit,
                required_exposures=quota * horizon*unit, new_chunks_needed=low*unit,
                sampling_unit_chunks=unit,
                eligible_complete_trajectories=eligible_now if unit == UNIT else None,
                new_trajectories_needed=low if unit == UNIT else None,
                outstanding_reservations=len(store.reservations),
                meaning='feasible_capacity_envelope_not_a_random_draw_guarantee')


class LookaheadAdmission:
    """A version-bound, two-second advisory; source-owned revisions fence RPCs."""
    def __init__(self, horizon, clock=time.monotonic):
        self.horizon, self.clock = horizon, clock
        self.revision, self.snapshot = -1, None

    def clear(self):
        self.snapshot = None

    def receive(self, snapshot, versions):
        if not self.horizon:
            raise ValueError('lookahead admission is disabled')
        if not isinstance(snapshot, dict):
            raise ValueError('lookahead snapshot must be a mapping')
        for name in ('revision', 'new_chunks_needed', 'admitted_imagined_chunks'):
            if type(snapshot.get(name)) is not int or snapshot[name] < 0:
                raise ValueError('lookahead needs exact nonnegative ' + name)
        if (snapshot['new_chunks_needed'] > 432 or type(snapshot.get('horizon')) is not int
                or snapshot['horizon'] != self.horizon):
            raise ValueError('lookahead exceeds the reviewed horizon/cohort bound')
        version = snapshot.get('versions')
        if (not isinstance(version, (list, tuple)) or len(version) != 2
                or any(type(v) is not int or v < 0 for v in version)):
            raise ValueError('lookahead requires exact published versions')
        observed = snapshot.get('observed_monotonic')
        if type(observed) not in (float, int) or not math.isfinite(observed):
            raise ValueError('lookahead requires a finite host monotonic clock')
        if snapshot['revision'] <= self.revision:
            return False
        self.revision = snapshot['revision']
        if tuple(version) != versions or not 0 <= self.clock() - observed < 2.:
            self.clear()
            return False
        self.snapshot = dict(snapshot, versions=tuple(version), expires_at=observed + 2.)
        return True

    def status(self, versions, added, reserved, credits, seed_size, stopped):
        row = self.snapshot
        active = bool(not stopped and row and row['versions'] == versions
                      and self.clock() < row['expires_at'])
        target = row['new_chunks_needed'] if active else 0
        supplied = max(0, added - row['admitted_imagined_chunks']) if active else 0
        remaining = max(0, target - supplied - reserved)
        return dict(enabled=bool(self.horizon), revision=self.revision, active=active,
                    target_new_chunks=target, supplied_since_snapshot=supplied,
                    reserved_policy_samples=reserved, remaining_new_chunks=remaining,
                    can_spend_existing_credit=bool(remaining and credits >= seed_size),
                    credit_refill_performed=False)


class LookaheadPump:
    """One bounded feedback RPC at a time, with owned cleanup and visible failure.

    The async Replay actor must yield during critic/optimizer execution. This
    background task performs no CUDA, sampling, publication, or collective work.
    """
    def __init__(self, owner, horizon):
        self.owner, self.horizon = owner, horizon
        self.task = None
        self.error = None
        self.revision = 0
        self.last_log = -float('inf')
        self.stopped = False

    async def send(self, plan):
        from fresh_replay_pressure import bounded_feedback_rpc
        self.revision += 1
        snapshot = dict(plan, revision=self.revision, horizon=self.horizon,
                        observed_monotonic=time.monotonic(),
                        admitted_imagined_chunks=self.owner.fresh_admitted_chunks['imagined'])
        return await bounded_feedback_rpc(self.owner,
            self.owner.fresh_production_budget.set_fresh_lookahead.remote(snapshot))

    async def tick(self):
        started = time.monotonic()
        plan = forecast(self.owner.fresh_store, self.horizon)
        if not plan['ready']:
            # Producer publication already revokes the old version's advice.
            return
        receipt = await self.send(plan)
        if receipt.get('failure'):
            raise RuntimeError('lookahead production failed: ' + repr(receipt))
        if receipt.get('stopped'):
            self.stopped = True
        if started - self.last_log >= 10.:
            self.owner.telemetry.event('fresh_inventory_lookahead', details={
                **plan, 'revision': self.revision, 'production_receipt': receipt,
                'forecast_and_rpc_wall_s': time.monotonic() - started})
            self.last_log = started

    async def run(self):
        try:
            while not self.stopped:
                await self.tick()
                await asyncio.sleep(.5)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.error = repr(error)
            self.owner.telemetry.event('fresh_inventory_lookahead_failed', details={'error': self.error})

    def check(self):
        if self.error:
            raise RuntimeError('inventory lookahead failed; refusing silent fallback: ' + self.error)

    async def stop(self):
        self.stopped = True
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        versions = self.owner.fresh_store.versions
        if versions is not None:
            await self.send(dict(versions=list(versions), new_chunks_needed=0))
        return {'stopped': True, 'revision': self.revision}


def check_health(owner):
    pump = getattr(owner, 'inventory_lookahead', None)
    if pump is not None:
        pump.check()
