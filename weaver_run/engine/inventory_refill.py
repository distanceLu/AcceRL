"""CPU metadata control for bounded, version-aware inventory refill.

Forecasts are capacity envelopes, never sampling decisions or credit grants.
One serialized producer still owns all spend and in-flight accounting.
"""
from collections import Counter, deque
import asyncio
import math
import time

PLAN_TTL = 6.


class ProductionTiming:
    """Bounded ACK-side latency history; never GPU execution-time accounting."""
    def __init__(self, workers):
        self.workers = workers
        self.rows = deque(maxlen=512)
        self.cohorts = deque(maxlen=4)
        self.seed_wait = 0.

    def idle(self):
        self.rows.clear()

    def completed(self, started, finished, chunks):
        self.rows.append((started, finished, chunks))
        total, earliest = 0, finished
        for begin, end, count in reversed(self.rows):
            earliest = min(earliest, begin)
            total += count
            if total >= 432:
                self.cohorts.append(finished-earliest)
                break

    def status(self):
        if self.cohorts:
            lead, method = max(self.cohorts), 'completed_cohort_wall'
        elif self.rows:
            # Explicit bootstrap proxy, replaced by actual complete cohorts.
            latency = max(end-start for start, end, _ in self.rows)
            lead = latency*math.ceil(108/self.workers)
            method = 'seed_latency_scaled_proxy'
        else:
            lead, method = None, 'unmeasured'
        return dict(cohort_lead_seconds=lead,
                    lead_method=method, measured_seed_wait_seconds=self.seed_wait)


class DemandTiming:
    """Reserve-to-publication wall time, excluding preceding stockout waits."""
    def __init__(self):
        self.work = {'ordinary': deque(maxlen=16), 'joint': deque(maxlen=16)}
        self.active = None

    def observe(self, context, seconds):
        if context not in self.work or type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Expected positive finite ordinary/joint work seconds')
        self.work[context].append(float(seconds))

    def estimate(self, context):
        values = self.work[context]
        # An early deadline is conservative for refill; this is not an SLA.
        return min(values) if values else None

    def reserved(self, versions, now):
        self.active = (tuple(versions), now)

    def published(self, versions, now):
        if self.active is None:
            return
        (policy, world), started = self.active
        if tuple(versions) == (policy+1, (policy+1)//5):
            if now > started:
                self.observe('joint' if (policy+1)%5 == 0 else 'ordinary', now-started)
            self.active = None
        elif versions[0] == policy and versions[1] == world:
            # A skipped/failed update supplies no completed-work observation.
            self.active = None


def project_refill(store, timing, producer, now):
    """Find the first infeasible prefix through the next joint+first draw.

    Prefix flow enforces per-draw distinctness and total legal use capacity.
    It is an optimistic matching bound, not a guarantee for random sampling.
    New output is evaluated at its *current* generation versions. Unknown
    timing requests bounded immediate refill; it never invents spare credit.
    """
    cfg = store.config
    if (not cfg.replay_age_limits or not cfg.real_policy_replay or cfg.real_fraction != .25
            or (cfg.max_sample_uses, cfg.max_policy_age, cfg.max_world_age) != (4, 16, 3)):
        raise ValueError('Refill requires the reviewed 144:432, age16/3, usecap4 recipe')
    versions = store.versions
    if versions is None or versions[1] != versions[0]//5:
        return {'ready':False, 'reason':'publication_in_progress'}
    policy, world = versions
    active = timing.active if timing.active and timing.active[0] == tuple(versions) else None
    first = 1 if active else 0
    boundary = (policy//5+1)*5
    steps = list(range(first, boundary-policy+1))
    future = [(policy+step, (policy+step)//5) for step in steps]
    deadlines, deadline_known, at = [], [], now
    known = True
    for step in range(boundary-policy+1):
        if step >= first:
            deadlines.append(at)
            deadline_known.append(known)
        duration = timing.estimate('joint' if (policy+step+1)%5 == 0 else 'ordinary')
        known &= duration is not None
        remaining = duration or 0.
        if step == 0 and active:
            remaining = max(0., remaining-(now-active[1]))
        at += remaining
    groups = Counter()
    for sample in store.samples.values():
        segment = store.segments[sample['segment_id']]
        if segment.source != 'imagined' or not store._eligible(sample):
            continue
        born = segment.versions[sample['index']]
        bw = born.get('world_reward_bundle_version', born.get('world_bundle_version'))
        mask = tuple(0 <= p-born['policy'] <= 16 and 0 <= w-bw <= 3 for p, w in future)
        groups[(4-sample['uses']-sample['reserved'], mask)] += 1
    from inventory_lookahead import _flow
    lead = producer.get('cohort_lead_seconds')
    if lead is not None and (type(lead) not in (int, float) or not math.isfinite(lead) or lead < 0):
        raise ValueError('Invalid measured cohort lead')
    base = dict(ready=True, versions=list(versions), forecast_draws=len(future),
        new_chunks_needed=0, timing_measured=known and lead is not None,
        cohort_lead_seconds=lead,
        meaning='feasible_capacity_bound_not_a_random_draw_guarantee')
    for index, target in enumerate(future):
        prefix = Counter()
        for (uses, mask), count in groups.items():
            prefix[(uses, mask[:index+1])] += count
        capacity = _flow(prefix, 432, index+1, 0)
        if capacity == 432*(index+1):
            continue
        generation_mask = tuple(0 <= p-policy <= 16 and 0 <= w-world <= 3
                                for p, w in future[:index+1])
        low, high = 0, 432
        while low < high:
            mid = (low+high)//2
            augmented = prefix.copy()
            augmented[(4, generation_mask)] += mid
            if _flow(augmented, 432, index+1, 0) == 432*(index+1):
                high = mid
            else:
                low = mid+1
        due = (not deadline_known[index] or lead is None or deadlines[index] <= now+lead+2.)
        can_prefill = generation_mask[-1]
        return dict(base, target_versions=list(target), deadline_monotonic=deadlines[index],
            timing_measured=deadline_known[index] and lead is not None,
            projected_distinct=sum(count for (_, mask), count in groups.items() if mask[index]),
            projected_remaining_uses=sum(count*uses for (uses, mask), count in groups.items() if mask[index]),
            feasible_prefix_exposures=capacity, required_prefix_exposures=432*(index+1),
            earliest_deficit_chunks=low, new_chunks_needed=low if due and can_prefill else 0,
            reason=('generation_version_too_old' if not can_prefill else
                    'refill_due' if due else 'outside_lead_window'))
    return dict(base, reason='inventory_covers_horizon')


class InventoryRefillPump:
    """One sequential CPU control loop on the existing async Replay actor.

    A timed-out owned RPC is never retried while it is still outstanding.
    Exceptions close speculation by TTL, report degradation, and leave the
    original reactive shortage path available. No GPU operations occur here.
    """
    def __init__(self, owner):
        self.owner = owner
        self.timing = DemandTiming()
        self.task = None
        self.stopped = False
        self.revision = 0
        self.wake = asyncio.Event()
        self.last_plan = None
        self.error = None
        self.last_log = -float('inf')
        self.lock = asyncio.Lock()

    def notify(self):
        self.wake.set()

    async def rpc(self, awaitable):
        from fresh_replay_pressure import bounded_feedback_rpc
        return await bounded_feedback_rpc(self.owner, awaitable)

    async def tick(self):
        async with self.lock:
            return await self._tick()

    async def _tick(self):
        started = time.monotonic()
        if getattr(self.owner, '_fresh_feedback_pending', ()):
            return 'transport_pending'
        sink = self.owner.fresh_production_budget
        state = await self.rpc(sink.production_admission.remote())
        # Snapshot only after the transport await: Replay may have published or
        # admitted output while waiting. Producer versions fence delayed plans.
        plan = project_refill(self.owner.fresh_store, self.timing, state, time.monotonic())
        self.last_plan = plan
        if not plan['ready']:
            return plan['reason']
        self.revision += 1
        receipt = await self.rpc(sink.set_refill_plan.remote(dict(plan,
            revision=self.revision, observed_monotonic=time.monotonic(),
            admitted_imagined_chunks=self.owner.fresh_admitted_chunks['imagined'])))
        if getattr(self.owner, 'cost_scheduler_enabled', False):
            await self.rpc(self.owner.cost_scheduler_broker.update_production.remote(receipt))
        if time.monotonic()-self.last_log >= 10. or self.error is not None:
            self.owner.telemetry.event('fresh_inventory_refill', details=dict(plan,
                revision=self.revision, production=receipt, recovered=self.error is not None,
                control_wall_seconds=time.monotonic()-started))
            self.last_log = time.monotonic()
        self.error = None
        return receipt

    async def run(self):
        try:
            await self.rpc(self.owner.fresh_production_budget.arm_inventory_refill.remote())
            while not self.stopped:
                try:
                    await self.tick()
                except Exception as error:
                    self.degraded(error)
                self.wake.clear()
                try:
                    await asyncio.wait_for(self.wake.wait(), 2.)
                except asyncio.TimeoutError:
                    pass
        except Exception as error:
            self.error = repr(error)
            self.owner.telemetry.event('fresh_inventory_refill_degraded',
                details={'error':self.error, 'fallback':'bootstrap_or_reactive_only'})

    def degraded(self, error):
        if self.error != repr(error):
            self.owner.telemetry.event('fresh_inventory_refill_degraded',
                details={'error':repr(error), 'fallback':'reactive_only_after_plan_expiry'})
        self.error = repr(error)

    def start(self):
        if self.task is None:
            self.task = asyncio.create_task(self.run())

    async def stop(self):
        self.stopped = True
        self.wake.set()
        if self.task is not None:
            await asyncio.shield(self.task)
        if not getattr(self.owner, '_fresh_feedback_pending', ()):
            versions = self.owner.fresh_store.versions
            if versions is not None:
                self.revision += 1
                try:
                    await self.rpc(self.owner.fresh_production_budget.set_refill_plan.remote(
                        dict(revision=self.revision, versions=list(versions), new_chunks_needed=0,
                             admitted_imagined_chunks=self.owner.fresh_admitted_chunks['imagined'],
                             observed_monotonic=time.monotonic())))
                except Exception as error:
                    self.owner.telemetry.event('fresh_inventory_refill_stop_degraded',
                        details={'error':repr(error), 'fallback':'expires_by_sender_clock'})
