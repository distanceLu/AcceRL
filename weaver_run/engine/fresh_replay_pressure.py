"""Read-only Replay shortage feedback; never mints production credits.

A paused producer may spend credit already earned by successful global draws.
Eligibility, source quotas, reservations and successful-use accounting remain
owned by SegmentReplay. The producer receives a two-second advisory lease.
"""
import asyncio
import time
import uuid
from fresh_policy_replay import FreshReplayNotReady

POLL_SECONDS = .25
PRESSURE_SECONDS = 1.0
DIAGNOSTIC_SECONDS = 30.0
CREDIT_SHORTFALL_SECONDS = 30.0
RPC_TIMEOUT_SECONDS = 5.0


async def bounded_feedback_rpc(owner, awaitable):
    """Bound control transport without cancelling already-submitted Ray work.

    A late pressure cannot outlive its sender-clock TTL. Keep the owned task
    referenced and consume its eventual outcome, even after the caller fails.
    """
    task = asyncio.ensure_future(awaitable)
    pending = getattr(owner, '_fresh_feedback_pending', None)
    if pending is None:
        pending = owner._fresh_feedback_pending = set()
    pending.add(task)
    def finished(item):
        pending.discard(item)
        if not item.cancelled():
            item.exception()
    task.add_done_callback(finished)
    deadline = time.monotonic()+RPC_TIMEOUT_SECONDS
    try:
        return await asyncio.wait_for(asyncio.shield(task), RPC_TIMEOUT_SECONDS)
    except asyncio.TimeoutError as error:
        raise RuntimeError('FRESH feedback RPC timed out; advisory lease expires without retry') from error
    except asyncio.CancelledError:
        # A cancellation may race acceptance. Reconcile only within the same
        # bounded window, then let the clear/revision/TTL protocol retire it.
        while not task.done() and time.monotonic() < deadline:
            try:
                await asyncio.wait_for(asyncio.shield(task), max(.001,deadline-time.monotonic()))
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        raise


def shortage(store, count):
    if type(count) is not int or count < 1:
        raise ValueError('FRESH cohort size must be positive')
    eligible = {'real': 0, 'imagined': 0}
    for sample in store.samples.values():
        if store._eligible(sample):
            eligible[store.segments[sample['segment_id']].source] += 1
    if store.config.real_policy_replay:
        if count < 2:
            raise ValueError('Mixed-source FRESH cohorts require at least two samples')
        real = max(1, min(count-1, int(round(count*store.config.real_fraction))))
        quotas = {'real': real, 'imagined': count-real}
    else:
        # Pressure integration is enabled only for the fixed mixed FRESH recipe.
        # For other callers reserve() remains the sole admission authority.
        quotas = {'real': 0, 'imagined': 0}
    return dict(versions=list(store.versions or (0, 0)), eligible=eligible,
                quotas=quotas, missing={key:max(0, quotas[key]-eligible[key]) for key in quotas})


async def reserve_with_production_pressure(owner, count):
    store = owner.fresh_store
    sink = getattr(owner, 'fresh_production_budget', None)
    if not store.config.real_policy_replay:
        sink = None
    request_id = 'fresh-demand:'+uuid.uuid4().hex
    started = time.monotonic()
    last_progress = started
    last_progress_key = None
    next_pressure = next_diagnostic = started
    reported = False
    result = None

    async def send(snapshot):
        owner.fresh_pressure_revision = getattr(owner, 'fresh_pressure_revision', 0)+1
        snapshot = dict(snapshot, request_id=request_id, revision=owner.fresh_pressure_revision)
        return await bounded_feedback_rpc(owner, sink.set_fresh_replay_pressure.remote(snapshot))

    try:
        while True:
            try:
                result = store.reserve(count)
                return result
            except FreshReplayNotReady:
                now = time.monotonic()
                if now >= next_pressure:
                    status = dict(shortage(store, count), observed_monotonic=now,
                                  admitted_imagined_chunks=owner.fresh_admitted_chunks['imagined'])
                    # Keep the macro controller informed while reserve() is
                    # blocked. This is also its safe probe-timeout observation
                    # point; retain its authoritative revision/version fields.
                    if getattr(owner, 'cost_scheduler_enabled', False):
                        await bounded_feedback_rpc(owner,
                            owner.cost_scheduler_broker.update_inventory.remote(owner.scheduler_inventory()))
                    budget = {}
                    if sink is not None:
                        reported = True  # cancellation may race remote acceptance
                        budget = await send(status)
                        if budget.get('stopped') or budget.get('failure'):
                            raise RuntimeError('FRESH production is stopped or failed: '+repr(budget))
                    # Real collection can continue while imagined supply is
                    # permanently unfunded. It must not reset this diagnosis.
                    key = (status['eligible']['imagined'], budget.get('policy_added'),
                           budget.get('policy_sampled'), budget.get('reserved_policy_samples'))
                    if key != last_progress_key:
                        last_progress, last_progress_key = now, key
                    evidence = dict(request_id=request_id, batch_size=count, **status,
                                    production_budget=budget, waited_seconds=now-started,
                                    seconds_without_supply_progress=now-last_progress,
                                    credit_refill_performed=False)
                    if now >= next_diagnostic:
                        owner.telemetry.event('fresh_replay_wait', details=evidence)
                        next_diagnostic = now+DIAGNOSTIC_SECONDS
                    # A complete next cohort cannot replenish its own missing
                    # supply. Fail explicitly after a bounded observation when
                    # there is no funded seed and no outstanding production.
                    shortfall = (sink is not None and status['missing']['imagined'] > 0
                        and budget.get('active_imagine') == 0
                        and budget.get('reserved_policy_samples') == 0
                        and budget.get('available_policy_credits', float('inf'))
                            < budget.get('policy_chunks_per_seed', 0))
                    if shortfall and now-last_progress >= CREDIT_SHORTFALL_SECONDS:
                        owner.telemetry.event('fresh_replay_credit_shortfall', details=evidence)
                        raise RuntimeError('FRESH next cohort cannot fund an imagined seed from earned credits; '
                                           'refusing to mint credits or relax eligibility: '+repr(evidence))
                    next_pressure = now+PRESSURE_SECONDS
                await asyncio.sleep(POLL_SECONDS)
    finally:
        if sink is not None and reported:
            clear = dict(versions=list(store.versions or (0, 0)), missing={'real':0,'imagined':0},
                         observed_monotonic=time.monotonic(),
                         admitted_imagined_chunks=owner.fresh_admitted_chunks['imagined'])
            try:
                await send(clear)
            except BaseException:
                # If a reservation was made but its producer stop cannot be
                # reconciled, do not leak its use lease to a caller without ACK.
                if result is not None:
                    store.cancel(result[0])
                raise
