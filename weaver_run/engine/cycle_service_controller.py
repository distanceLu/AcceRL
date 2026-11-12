"""CPU-only, advisory controller for a complete FRESH 5:1 cycle.

This module changes service *weights*, never learning rules or Replay state.
Responses are measured at a complete weight vector and execution context.  A
response at weight 1 is deliberately not multiplied by a candidate weight: GPU
co-runners, batching and actor queues make that extrapolation unsafe.

The forecast is a measured end-to-end progress-capacity envelope. Responses
share wall-clock windows, so overlapping progress durations are combined by
maximum, never added as if they were isolated execution stages.
An unmeasured candidate is returned as a bounded probe recommendation; callers
must explicitly allow a probe to apply one.  No Ray, Torch or GPU import occurs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
import math


SERVICES = ('policy_train', 'world_train', 'policy_infer', 'world_infer')
PROGRESS_SERVICE = {
    'policy_training': 'policy_train', 'world_training': 'world_train',
    'critic_observations': 'policy_infer', 'policy_action_observations': 'policy_infer',
    'real_chunks': 'policy_infer',
    'imagined_chunks': 'world_infer',
}


def _number(name, value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError('%s must be finite' % name)
    if value < 0 or (positive and value == 0):
        raise ValueError('%s must be %s' % (name, 'positive' if positive else 'nonnegative'))
    return float(value)


def _integer(name, value, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('%s must be an integer >= %s' % (name, minimum))
    return value


def _weights(value):
    if not isinstance(value, Mapping) or set(value) != set(SERVICES):
        raise ValueError('Weights must contain exactly the four service budgets')
    return {s: _number('weight '+s, value[s], positive=True) for s in SERVICES}


def _key(weights):
    return tuple(round(weights[s], 9) for s in SERVICES)


@dataclass(frozen=True)
class InventoryBucket:
    """A disjoint set of sample IDs with identical eligibility metadata.

    ``distinct_chunks`` counts IDs, not remaining uses.  One sample can appear
    only once in the next cohort.  The runtime must aggregate each ID into
    exactly one bucket; duplicate bucket identities are rejected here.
    """
    bucket_id: str
    source: str
    distinct_chunks: int
    generation_policy_version: int
    generation_world_version: int | None = None
    successful_uses: int = 0
    reserved_uses: int = 0
    expires_at: float | None = None

    def __post_init__(self):
        if not isinstance(self.bucket_id, str) or not self.bucket_id:
            raise ValueError('Inventory bucket needs an identity')
        if self.source not in ('real', 'imagined'):
            raise ValueError('Inventory source must be real or imagined')
        _integer('distinct_chunks', self.distinct_chunks)
        _integer('generation_policy_version', self.generation_policy_version)
        if self.generation_world_version is not None:
            _integer('generation_world_version', self.generation_world_version)
        if self.source == 'imagined' and self.generation_world_version is None:
            raise ValueError('Imagined inventory needs its generating World/RM version')
        _integer('successful_uses', self.successful_uses)
        _integer('reserved_uses', self.reserved_uses)
        if self.successful_uses + self.reserved_uses > 4:
            raise ValueError('Successful plus reserved uses cannot exceed four')
        if self.expires_at is not None:
            _number('expires_at', self.expires_at)


@dataclass(frozen=True)
class CycleSnapshot:
    """Read-only projection of remaining work to a synchronized cycle boundary.

    Versions are the synchronized published boundary, not an uncommitted owner
    counter. Training units are the caller's backend work units, not requests.
    The same units must be supplied in ResponseSample.  Remaining critic work
    is the current batch; future batches require an observed/estimated count
    in ``critic_observations_per_future_update`` or the forecast stays unknown.
    """
    now: float
    policy_version: int
    world_version: int
    inventory: tuple[InventoryBucket, ...] = ()
    critic_observations_remaining: float = 0.0
    policy_training_units_remaining: float = 1.0
    world_training_units_remaining: float = 0.0
    publication_seconds_remaining: float = 0.0
    publication_blocked: bool = False
    remaining_policy_updates: int | None = None
    phase: str = 'needs_batch'
    context: str = 'ordinary'
    critic_observations_per_future_update: float | None = None
    policy_training_units_per_future_update: float = 1.0
    world_training_units_per_future_update: float = 1.0
    real_production_ready: bool = True
    imagination_ready: bool = True
    world_batch_ready: bool = True
    policy_action_observations_per_real_chunk: float | None = None
    policy_action_observations_per_imagined_chunk: float | None = None
    production_state_required: bool = False
    production_state_expires_at: float = 0.
    production_inflight: bool = False
    production_refill_chunks: int = 0
    production_seed_expires_at: float = 0.
    production_inflight_chunks: int = 0

    def __post_init__(self):
        _number('now', self.now)
        _integer('policy_version', self.policy_version)
        _integer('world_version', self.world_version)
        if self.world_version not in (self.policy_version // 5, max(0, self.policy_version // 5 - 1)):
            raise ValueError('Snapshot is outside a possible 5:1 update boundary')
        if self.policy_version % 5 and self.world_version != self.policy_version // 5:
            raise ValueError('A pending World update cannot be skipped by another Policy update')
        if self.phase not in ('needs_batch', 'refresh', 'train', 'publish'):
            raise ValueError('Unknown update phase')
        if not isinstance(self.context, str) or not self.context:
            raise ValueError('Execution context must be explicit')
        for name in ('publication_blocked', 'real_production_ready', 'imagination_ready', 'world_batch_ready',
                     'production_state_required', 'production_inflight'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(name+' must be boolean')
        _number('production_state_expires_at', self.production_state_expires_at)
        _integer('production_refill_chunks', self.production_refill_chunks)
        _number('production_seed_expires_at', self.production_seed_expires_at)
        _integer('production_inflight_chunks', self.production_inflight_chunks)
        for name in ('critic_observations_remaining', 'policy_training_units_remaining',
                     'world_training_units_remaining', 'publication_seconds_remaining',
                     'policy_training_units_per_future_update', 'world_training_units_per_future_update'):
            _number(name, getattr(self, name))
        if self.critic_observations_per_future_update is not None:
            _number('critic_observations_per_future_update', self.critic_observations_per_future_update)
        for name in ('policy_action_observations_per_real_chunk', 'policy_action_observations_per_imagined_chunk'):
            if getattr(self,name) is not None:
                _number(name,getattr(self,name))
        if self.remaining_policy_updates is not None:
            _integer('remaining_policy_updates', self.remaining_policy_updates)
            if self.remaining_policy_updates > 5 or (self.policy_version + self.remaining_policy_updates) % 5:
                raise ValueError('Planning target must be the next complete 5:1 cycle boundary')
            if self.remaining_policy_updates == 0 and self.world_version == self.policy_version // 5:
                raise ValueError('No remaining cycle work')
        if any(not isinstance(x, InventoryBucket) for x in self.inventory):
            raise ValueError('Inventory must contain InventoryBucket records')
        if len({x.bucket_id for x in self.inventory}) != len(self.inventory):
            raise ValueError('Duplicate inventory bucket identity')
        object.__setattr__(self,'inventory',tuple(self.inventory))

    @property
    def updates_remaining(self):
        if self.remaining_policy_updates is not None:
            return self.remaining_policy_updates
        if self.world_version < self.policy_version // 5:
            return 0
        return 5 - self.policy_version % 5

    @property
    def target_policy_version(self):
        return self.policy_version + self.updates_remaining

    @property
    def current_critic_work(self):
        if not self.updates_remaining:
            return 0.0
        if self.phase == 'needs_batch' and self.critic_observations_remaining == 0:
            # Not enqueued does not mean finished. Until a cohort is selected,
            # use the observed per-update estimate or keep the work unknown.
            return self.critic_observations_per_future_update
        return self.critic_observations_remaining


@dataclass(frozen=True)
class ResponseSample:
    """Actual progress in a stable-weight, stable-context wall-clock window.

    ``execution_seconds`` is the observation window, including time in which
    this service made no progress because co-runners were using the resource.
    It must NOT be a single request's active latency: that is not an observed
    system throughput response. It is not GPU occupancy or GPU execution time.
    Units count successfully completed work only. Samples never combine windows
    across changed weight vectors, execution contexts or incompatible units.
    """
    weights: Mapping[str, float]
    service: str
    completed_units: float
    execution_seconds: float
    context: str = 'ordinary'
    progress_kind: str = 'work'
    observed_at: float = 0.0
    sample_id: str | None = None

    def __post_init__(self):
        object.__setattr__(self,'weights',_weights(self.weights))
        if self.service not in SERVICES:
            raise ValueError('Unknown service')
        if self.progress_kind != 'work' and PROGRESS_SERVICE.get(self.progress_kind) != self.service:
            raise ValueError('Progress kind does not belong to its service')
        _number('completed_units', self.completed_units)
        _number('execution_seconds', self.execution_seconds, positive=True)
        _number('observed_at', self.observed_at)
        if not isinstance(self.context, str) or not self.context:
            raise ValueError('Response needs an execution context')
        if self.sample_id is not None and (not isinstance(self.sample_id, str) or not self.sample_id):
            raise ValueError('sample_id must be nonempty when supplied')


@dataclass(frozen=True)
class ControllerConfig:
    total_weight: float = 4.0
    min_weight: float = 0.25
    max_weight: float = 2.5
    search_step: float = 0.25
    max_l1_change: float = 1.0
    beta_seconds_per_weight: float = 5.0
    min_relative_improvement: float = 0.02
    min_absolute_improvement_seconds: float = 0.1
    cooldown_seconds: float = 30.0
    response_max_age_seconds: float = 3600.0
    ewma_alpha: float = 0.25
    min_response_samples: int = 2
    max_response_contexts: int = 256

    def __post_init__(self):
        for name in ('total_weight', 'min_weight', 'max_weight', 'search_step', 'max_l1_change',
                     'response_max_age_seconds', 'ewma_alpha'):
            _number(name, getattr(self, name), positive=True)
        for name in ('beta_seconds_per_weight', 'min_relative_improvement',
                     'min_absolute_improvement_seconds', 'cooldown_seconds'):
            _number(name, getattr(self, name))
        if not 0 < self.ewma_alpha <= 1 or self.min_relative_improvement >= 1:
            raise ValueError('Invalid EWMA or hysteresis fraction')
        if not 4*self.min_weight <= self.total_weight <= 4*self.max_weight:
            raise ValueError('Weight bounds cannot contain the total')
        _integer('min_response_samples', self.min_response_samples, minimum=1)
        _integer('max_response_contexts', self.max_response_contexts, minimum=1)


@dataclass(frozen=True)
class WeightDecision:
    weights: dict[str, float]
    changed: bool
    reason: str
    predicted_cycle_seconds: float | None
    score: float | None
    demands: dict
    critical_path: str | None
    ready_services: tuple[str, ...]
    exploration_candidate: dict[str, float] | None = None
    prediction_scope: str = 'end-to-end progress-capacity envelope; overlapping wall windows are not additive'
    baseline_cycle_seconds: float | None = None
    probe_comparison: dict | None = None


class CycleServiceController:
    def __init__(self, initial_weights=None, config=None):
        self.config = config or ControllerConfig()
        if not isinstance(self.config, ControllerConfig):
            raise ValueError('config must be ControllerConfig')
        self.weights = _weights(initial_weights or {s: self.config.total_weight/4 for s in SERVICES})
        self._validate_weights(self.weights)
        self.responses = {}
        self._sample_ids = {}
        self.last_change_at = None
        self.last_decision_at = None

    def _validate_weights(self, weights):
        if not math.isclose(sum(weights.values()), self.config.total_weight, abs_tol=1e-8):
            raise ValueError('Service weight total changed')
        if any(not self.config.min_weight <= v <= self.config.max_weight for v in weights.values()):
            raise ValueError('Service weight outside configured bounds')

    def observe(self, sample):
        if not isinstance(sample, ResponseSample):
            raise ValueError('Expected completed ResponseSample')
        weights = _weights(sample.weights);self._validate_weights(weights)
        if sample.sample_id is not None:
            prior = self._sample_ids.get(sample.sample_id)
            if prior is not None:
                if prior != sample:
                    raise ValueError('Conflicting duplicate progress sample')
                return False
        k = (_key(weights), sample.context, sample.service, sample.progress_kind)
        rate = sample.completed_units/sample.execution_seconds
        old = self.responses.get(k)
        if old is not None and sample.observed_at < old['observed_at']:
            raise ValueError('Out-of-order response sample')
        if old is None:
            self.responses[k] = {'rate':rate, 'count':1, 'observed_at':sample.observed_at}
        else:
            a = self.config.ewma_alpha
            self.responses[k] = {'rate':a*rate+(1-a)*old['rate'], 'count':old['count']+1,
                                 'observed_at':sample.observed_at}
        if sample.sample_id is not None:
            self._sample_ids[sample.sample_id] = sample
        # Bound the observation model and de-duplication receipts.
        while len(self.responses) > self.config.max_response_contexts:
            del self.responses[min(self.responses, key=lambda key:self.responses[key]['observed_at'])]
        while len(self._sample_ids) > self.config.max_response_contexts*16:
            del self._sample_ids[next(iter(self._sample_ids))]
        return True

    def inventory(self, snapshot, *, consume_at=None, policy_version=None, world_version=None):
        """Eligibility projection, without reserving or modifying any sample."""
        at = snapshot.now if consume_at is None else _number('consume_at', consume_at)
        p = snapshot.policy_version if policy_version is None else _integer('policy_version', policy_version)
        w = snapshot.world_version if world_version is None else _integer('world_version', world_version)
        distinct = {'real':0, 'imagined':0};exposure = dict(distinct);rejected = {}
        for bucket in snapshot.inventory:
            reason = None
            uses = 4-bucket.successful_uses-bucket.reserved_uses
            if bucket.expires_at is not None and bucket.expires_at <= at:
                reason = 'expired'
            elif not 0 <= p-bucket.generation_policy_version <= 16:
                reason = 'policy_age'
            elif bucket.source == 'imagined' and not 0 <= w-bucket.generation_world_version <= 3:
                reason = 'world_age'
            elif not uses:
                reason = 'use_cap'
            if reason is not None:
                rejected[reason] = rejected.get(reason, 0)+bucket.distinct_chunks
            else:
                distinct[bucket.source] += bucket.distinct_chunks
                exposure[bucket.source] += bucket.distinct_chunks*uses
        return {'eligible_distinct_chunks':distinct, 'remaining_exposure_credits':exposure,
                'excluded_distinct_chunks':rejected, 'evaluated_at':at, 'policy_version':p, 'world_version':w}

    def demands(self, snapshot, *, consume_at=None):
        if not isinstance(snapshot, CycleSnapshot):
            raise ValueError('Expected CycleSnapshot')
        stock = self.inventory(snapshot, consume_at=consume_at)
        n = snapshot.updates_remaining
        needs_current_batch = n > 0 and snapshot.phase == 'needs_batch'
        distinct = stock['eligible_distinct_chunks']
        next_missing = {s:max(0, quota-distinct[s]) if needs_current_batch else 0
                        for s, quota in [('real',144),('imagined',432)]}
        # For cycle capacity, project every batch's age and successful-use cap.
        # Each sample contributes at most once at each future update boundary.
        cycle_capacity = {'real':0, 'imagined':0}
        current_reserved = 0 if needs_current_batch else min(1,n)
        for bucket in snapshot.inventory:
            uses = 4-bucket.successful_uses-bucket.reserved_uses
            if not uses:
                continue
            eligible_batches = 0
            for step in range(current_reserved,n):
                version = snapshot.policy_version+step
                world = snapshot.world_version  # no mid-cycle World publication before the final fifth update
                if bucket.expires_at is not None and bucket.expires_at <= (consume_at or snapshot.now):
                    continue
                if not 0 <= version-bucket.generation_policy_version <= 16:
                    continue
                if bucket.source == 'imagined' and not 0 <= world-bucket.generation_world_version <= 3:
                    continue
                eligible_batches += 1
            cycle_capacity[bucket.source] += bucket.distinct_chunks*min(uses, eligible_batches)
        future_batches = max(0,n-current_reserved)
        cycle_missing = {s:max(0,quota*future_batches-cycle_capacity[s])
                         for s,quota in [('real',144),('imagined',432)]}
        critic = snapshot.current_critic_work
        if n > 1:
            critic = None if critic is None or snapshot.critic_observations_per_future_update is None else (
                critic+(n-1)*snapshot.critic_observations_per_future_update)
        target_world = snapshot.target_policy_version//5
        world_updates = target_world-snapshot.world_version
        if world_updates not in (0,1):
            raise ValueError('Forecast must finish exactly the next 5:1 boundary')
        world_units = snapshot.world_training_units_remaining
        if world_updates and world_units == 0 and snapshot.phase != 'publish':
            world_units = snapshot.world_training_units_per_future_update
        policy_units = snapshot.policy_training_units_remaining if n else 0
        policy_units += max(0,n-1)*snapshot.policy_training_units_per_future_update
        # Production rates count NEW chunks, while the ledger counts successful
        # consumption exposures. A new sample is distinct in each batch and may
        # be reused at most four times across the remaining batches. This is a
        # necessary supply bound, not a guarantee of an exact future matching.
        new_sample_capacity = max(1,min(4,future_batches))
        cycle_missing_chunks = {source:max(next_missing[source],
            math.ceil(cycle_missing[source]/new_sample_capacity)) for source in ('real','imagined')}
        # The producer may already be refilling the first post-publication
        # batch even when the old cycle-only demand is zero.
        if not snapshot.production_state_required or snapshot.now < snapshot.production_state_expires_at:
            unstarted = (snapshot.production_refill_chunks if not snapshot.production_state_required
                         or snapshot.now < snapshot.production_seed_expires_at else 0)
            cycle_missing_chunks['imagined'] = max(cycle_missing_chunks['imagined'],
                                                   unstarted+snapshot.production_inflight_chunks)
        action_work = 0.0
        action_work_known = True
        for source, multiplier in [('real',snapshot.policy_action_observations_per_real_chunk),
                                   ('imagined',snapshot.policy_action_observations_per_imagined_chunk)]:
            if cycle_missing_chunks[source] and multiplier is None:
                action_work_known = False
            elif multiplier is not None:
                action_work += cycle_missing_chunks[source]*multiplier
        return dict(stock, next_batch_missing=next_missing, cycle_missing_exposures=cycle_missing,
                    minimum_new_chunks=cycle_missing_chunks,
                    cycle_existing_capacity=cycle_capacity, remaining_policy_updates=n,
                    remaining_world_updates=world_updates, critic_observations=critic,
                    policy_action_observations=action_work if action_work_known else None,
                    policy_training=policy_units, world_training=world_units,
                    publication_seconds=snapshot.publication_seconds_remaining,
                    invariant={'real_per_batch':144,'imagined_per_batch':432,'global_batch':576,
                               'max_successful_uses':4,'max_policy_age':16,'max_world_age':3,'policy_to_world':5})

    def ready_services(self, snapshot, demands=None):
        d = demands or self.demands(snapshot)
        if snapshot.publication_blocked:
            return ()
        ready = []
        state_fresh = not snapshot.production_state_required or snapshot.now < snapshot.production_state_expires_at
        if state_fresh and snapshot.real_production_ready and d['cycle_missing_exposures']['real']:
            ready.append('policy_infer')
        imag_ready = (snapshot.imagination_ready and state_fresh
            and (not snapshot.production_state_required or snapshot.production_inflight
                 or snapshot.now < snapshot.production_seed_expires_at))
        if imag_ready and (d['cycle_missing_exposures']['imagined']
                or snapshot.production_refill_chunks or snapshot.production_inflight):
            ready.extend(('policy_infer','world_infer'))
        if not any(d['next_batch_missing'].values()):
            if snapshot.current_critic_work is None or snapshot.current_critic_work > 0:
                ready.append('policy_infer')
            elif snapshot.phase == 'train' and d['remaining_policy_updates'] > 0 and snapshot.policy_training_units_remaining > 0:
                ready.append('policy_train')
        world_due_now = (snapshot.world_version < snapshot.policy_version//5 or
                         (d['remaining_policy_updates'] > 0 and (snapshot.policy_version+1)%5 == 0))
        if world_due_now and snapshot.world_batch_ready and d['world_training'] > 0:
            ready.append('world_train')
        return tuple(s for s in SERVICES if s in ready)

    def _rate(self, snapshot, weights, service, progress, *, context=None):
        r = self.responses.get((_key(weights), context or snapshot.context, service, progress))
        if r is None or r['count'] < self.config.min_response_samples:
            return None
        if not 0 <= snapshot.now-r['observed_at'] <= self.config.response_max_age_seconds:
            return None
        return r['rate']

    def forecast(self, snapshot, weights):
        weights = _weights(weights);self._validate_weights(weights)
        d = self.demands(snapshot)
        if d['critic_observations'] is None:
            return None, d, 'future_critic_work_unknown'
        if d['policy_action_observations'] is None:
            return None,d,'production_policy_observation_conversion_unknown'
        def duration(service, progress, work, *, context=None):
            if work == 0:
                return 0.0
            rate = self._rate(snapshot,weights,service,progress,context=context)
            return None if rate is None else (work/rate if rate > 0 else math.inf)
        def contextual_work(service, progress, current, per_future):
            n = snapshot.updates_remaining
            if n == 0:
                return 0.0
            pieces = [(current,snapshot.context)]
            if n > 1 and snapshot.context in ('ordinary','joint'):
                # The last remaining Policy update is the fifth/joint update.
                pieces.extend([(max(0,n-2)*per_future,'ordinary'),(per_future,'joint')])
            elif n > 1:
                pieces.append(((n-1)*per_future,snapshot.context))
            total = 0.0
            for work, context in pieces:
                seconds = duration(service,progress,work,context=context)
                if seconds is None:
                    return None
                total += seconds
            return total
        # First pass computes when this batch could be consumed; expiring stock
        # must be rechecked at that time, rather than only when it was observed.
        paths = None
        at = snapshot.now
        for _ in range(min(len(snapshot.inventory)+2,32)):
            d = self.demands(snapshot,consume_at=at)
            if d['policy_action_observations'] is None:
                return None,d,'production_policy_observation_conversion_unknown'
            real = duration('policy_infer','real_chunks',d['minimum_new_chunks']['real'])
            imag = duration('world_infer','imagined_chunks',d['minimum_new_chunks']['imagined'])
            first_real = duration('policy_infer','real_chunks',d['next_batch_missing']['real'])
            first_imag = duration('world_infer','imagined_chunks',d['next_batch_missing']['imagined'])
            critic = contextual_work('policy_infer','critic_observations',
                snapshot.current_critic_work,snapshot.critic_observations_per_future_update or 0)
            actions = duration('policy_infer','policy_action_observations',d['policy_action_observations'])
            policy = contextual_work('policy_train','policy_training',
                snapshot.policy_training_units_remaining,snapshot.policy_training_units_per_future_update)
            world = duration('world_train','world_training',d['world_training'],
                context='joint' if snapshot.context in ('ordinary','joint') else snapshot.context)
            if any(x is None for x in (real,imag,first_real,first_imag,critic,actions,policy,world)):
                return None,d,'unmeasured_response'
            if any(not math.isfinite(x) for x in (real,imag,first_real,first_imag,critic,actions,policy,world)):
                return None,d,'observed_no_progress'
            fresh = not snapshot.production_state_required or snapshot.now < snapshot.production_state_expires_at
            imag_ready = fresh and snapshot.imagination_ready and (not snapshot.production_state_required
                or snapshot.production_inflight or snapshot.now < snapshot.production_seed_expires_at)
            if (real > 0 and not (fresh and snapshot.real_production_ready)) or (imag > 0 and not imag_ready):
                return None,d,'production_dependency_not_ready'
            if world > 0 and not snapshot.world_batch_ready:
                return None,d,'world_batch_dependency_not_ready'
            # Every response rate already includes co-runner waits, refresh,
            # training and publication in its full update window. These are
            # capacity constraints, not separate phase latency estimates.
            delay = snapshot.publication_seconds_remaining
            paths = {'real_supply':real,'imagined_supply':imag,
                     'policy_refresh_and_train':max(first_real,first_imag,critic,policy),
                     'shared_policy_production_and_refresh':max(actions,critic),
                     'world_joint_training':world}
            seconds = max(delay,max(paths.values()))
            next_at = snapshot.now+max(delay,first_real,first_imag,critic)
            if next_at <= at:
                break
            at = next_at
        else:
            return None,d,'expiry_projection_did_not_converge'
        return {'seconds':seconds,'paths_seconds':paths,'critical_path':max(paths,key=paths.get),
                'inventory_projected_at':at},d,None

    def candidates(self):
        values = [dict(self.weights)]
        # A small, deterministic pairwise transfer search; submitted work is not
        # cancelled or drained when a service weight changes.
        for donor in SERVICES:
            for receiver in SERVICES:
                if donor == receiver:
                    continue
                trial = dict(self.weights)
                trial[donor] -= self.config.search_step
                trial[receiver] += self.config.search_step
                try:self._validate_weights(trial)
                except ValueError:continue
                if sum(abs(trial[s]-self.weights[s]) for s in SERVICES) <= self.config.max_l1_change+1e-9:
                    values.append(trial)
        return values

    def assess_probe(self, snapshot, baseline_weights):
        """Close a measured probe against its unchanged baseline definition.

        Both vectors are evaluated against the SAME current demand snapshot.
        Missing/expired responses cannot justify retaining an experiment or
        chaining another probe. A rollback is conservative, not a speed claim.
        Runtime calls this only after the complete observation-cycle boundary.
        """
        baseline_weights = _weights(baseline_weights)
        self._validate_weights(baseline_weights)
        baseline,d,baseline_missing = self.forecast(snapshot,baseline_weights)
        current,_,current_missing = self.forecast(snapshot,self.weights)
        change = sum(abs(self.weights[s]-baseline_weights[s]) for s in SERVICES)
        penalty = self.config.beta_seconds_per_weight*change
        threshold = None if baseline is None else max(
            self.config.min_absolute_improvement_seconds,
            baseline['seconds']*self.config.min_relative_improvement)
        keep = (baseline is not None and current is not None and
                baseline['seconds']-(current['seconds']+penalty) >= threshold)
        comparison = {'baseline_weights':dict(baseline_weights),
            'candidate_weights':dict(self.weights),
            'baseline_seconds':None if baseline is None else baseline['seconds'],
            'candidate_seconds':None if current is None else current['seconds'],
            'baseline_missing':baseline_missing,'candidate_missing':current_missing,
            'l1_penalty_seconds':penalty,'required_improvement_seconds':threshold,
            'accepted':keep,'same_demand_snapshot':True}
        reason = ('probe_measured_cycle_improvement' if keep else
                  'probe_rollback_inconclusive' if baseline is None or current is None else
                  'probe_rollback_no_measured_improvement')
        selected = dict(self.weights if keep else baseline_weights)
        changed = selected != self.weights
        self.weights = selected
        self.last_change_at = snapshot.now
        self.last_decision_at = snapshot.now
        estimate = current if keep else baseline
        return WeightDecision(selected,changed,reason,
            None if estimate is None else estimate['seconds'],
            None if estimate is None else estimate['seconds']+(penalty if keep else 0),
            d,None if estimate is None else estimate['critical_path'],
            self.ready_services(snapshot,d),
            baseline_cycle_seconds=None if baseline is None else baseline['seconds'],
            probe_comparison=comparison)

    def choose(self, snapshot, *, allow_probe=False):
        if not isinstance(snapshot,CycleSnapshot) or type(allow_probe) is not bool:
            raise ValueError('Expected a snapshot and explicit boolean probe flag')
        if self.last_decision_at is not None and snapshot.now < self.last_decision_at:
            raise ValueError('Controller clock moved backwards')
        self.last_decision_at = snapshot.now
        base,d,missing = self.forecast(snapshot,self.weights)
        ready = self.ready_services(snapshot,d)
        if snapshot.publication_blocked:
            return WeightDecision(dict(self.weights),False,'publication_dependency',None,None,d,None,ready,None)
        base_score = None if base is None else base['seconds']
        best = None;unknown = []
        for candidate in self.candidates()[1:]:
            estimate,_,why = self.forecast(snapshot,candidate)
            if estimate is None:
                unknown.append(candidate);continue
            change = sum(abs(candidate[s]-self.weights[s]) for s in SERVICES)
            score = estimate['seconds']+self.config.beta_seconds_per_weight*change
            if best is None or score < best[0]:best=(score,candidate,estimate)
        # Prefer an exploration direction along a measured/existing critical
        # dependency. This recommendation is never an invented throughput rate.
        target = next((s for s in ('world_infer','policy_infer','world_train','policy_train') if s in ready),None)
        probe = next((x for x in unknown if target is not None and x[target] > self.weights[target]),None)
        cooldown = self.last_change_at is not None and snapshot.now-self.last_change_at < self.config.cooldown_seconds
        if cooldown:
            reason='cooldown'
        elif base is not None and best is not None and base_score-best[0] >= max(
                self.config.min_absolute_improvement_seconds,base_score*self.config.min_relative_improvement):
            self.weights=dict(best[1]);self.last_change_at=snapshot.now
            return WeightDecision(dict(self.weights),True,'measured_cycle_improvement',best[2]['seconds'],best[0],d,
                                  best[2]['critical_path'],ready,probe)
        elif allow_probe and base is not None and probe is not None and ready:
            self.weights=dict(probe);self.last_change_at=snapshot.now
            return WeightDecision(dict(self.weights),True,'bounded_probe_without_speed_prediction',None,None,d,None,ready,None,
                                  baseline_cycle_seconds=base_score)
        else:
            reason=missing if base is None else 'hysteresis_or_no_measured_improvement'
        return WeightDecision(dict(self.weights),False,reason,None if base is None else base['seconds'],
                              base_score,d,None if base is None else base['critical_path'],ready,probe)

    def snapshot(self):
        return {'weights':dict(self.weights),'response_points':len(self.responses),
                'last_change_at':self.last_change_at,'last_decision_at':self.last_decision_at,
                'learning_rules_changed':False,'gpu_sampling':False}
