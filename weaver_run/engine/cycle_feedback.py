"""Bounded weight experiments measured in complete successful 5:1 cycles.

Backpressure is part of end-to-end time, not a missing per-service capacity.
No GPU imports, assumed weight-to-throughput scaling, or learning-rule changes.
"""
from collections import OrderedDict
from dataclasses import dataclass, asdict
import math

from cycle_service_controller import SERVICES, WeightDecision


@dataclass(frozen=True)
class CycleFeedbackConfig:
    enabled: bool = False
    warmup_policy_updates: int = 20
    observation_cycles: int = 2
    timeout_seconds: float = 7200.
    inventory_tolerance: float = .1
    max_cycle_spread: float = .5

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('Cycle feedback enabled must be boolean')
        if (type(self.warmup_policy_updates) is not int or self.warmup_policy_updates < 0
                or self.warmup_policy_updates % 5):
            raise ValueError('Warmup must end at a complete cycle')
        if type(self.observation_cycles) is not int or not 2 <= self.observation_cycles <= 20:
            raise ValueError('At least two bounded cycles per observation')
        for name in ('timeout_seconds', 'inventory_tolerance', 'max_cycle_spread'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError('Invalid '+name)
        if self.timeout_seconds <= 0 or self.inventory_tolerance > 1 or self.max_cycle_spread > 1:
            raise ValueError('Invalid feedback bounds')


def key(weights):
    return tuple(round(weights[s], 9) for s in SERVICES)


class CompleteCycleFeedback:
    def __init__(self, config):
        self.config = config
        self.last = None
        self.anchor = None
        self.records = []
        self.active_probe = None
        self.last_result = None
        self.last_change_at = None
        self.tried = OrderedDict()
        self.last_observation = None

    @staticmethod
    def coverage(controller, snapshot):
        stock = controller.inventory(snapshot)
        # Bounded diagnostic reserve; actual cohort matching stays in Replay/H2.
        return {s: min(1., stock['eligible_distinct_chunks'][s]/quota,
                       stock['remaining_exposure_credits'][s]/(2*quota))
                for s, quota in [('real', 144), ('imagined', 432)]}

    def observe(self, snapshot, weights, admitted, controller):
        p, w = snapshot.policy_version, snapshot.world_version
        row = dict(policy=p, world=w, now=snapshot.now, weights=dict(weights),
                   admitted=dict(admitted), coverage=self.coverage(controller, snapshot))
        if set(admitted) != {'real', 'imagined'} or any(type(x) is not int or x < 0 for x in admitted.values()):
            raise ValueError('Cycle measurements need exact cumulative admission counters')
        if self.last is not None:
            old = self.last
            if p < old['policy'] or w < old['world'] or snapshot.now < old['now']:
                raise ValueError('Cycle observation regressed')
            if any(admitted[s] < old['admitted'][s] for s in admitted):
                raise ValueError('Cycle admission counters regressed')
            if (p, w) == (old['policy'], old['world']):
                return None  # retries cannot shorten or double-count an interval
            if (p != old['policy']+1 or w != p//5 or old['weights'] != weights):
                self.anchor = None
        self.last = row
        if p < self.config.warmup_policy_updates or w != p//5:
            self.anchor = None
            return None
        if p % 5:
            return None
        result = None
        if self.anchor is not None and p-self.anchor['policy'] == 5:
            start = self.anchor
            seconds = snapshot.now-start['now']
            if start['weights'] == weights and seconds > 0:
                result = dict(start_policy=start['policy'], end_policy=p,
                    started_at=start['now'], finished_at=snapshot.now, seconds=seconds,
                    weights=dict(weights), successful_policy_updates=5, successful_world_updates=1,
                    new_chunks={s: admitted[s]-start['admitted'][s] for s in admitted},
                    start_coverage=start['coverage'], end_coverage=row['coverage'],
                    scope='complete synchronized cycle wall time, including production backpressure')
                self.records.append(result)
                self.records = self.records[-64:]
        self.anchor = row
        self.last_observation = result
        return result

    def window(self, weights, *, since=0.):
        result = []
        for row in reversed(self.records):
            if row['weights'] != weights or row['started_at'] < since:
                break
            if result and row['end_policy'] != result[-1]['start_policy']:
                break
            result.append(row)
            if len(result) == self.config.observation_cycles:
                return list(reversed(result))
        return []

    def stable(self, rows):
        if not rows:
            return False
        values = [x['seconds'] for x in rows]
        return (max(values)-min(values))/(sum(values)/len(values)) <= self.config.max_cycle_spread

    def rebase(self, snapshot, weights, admitted, controller):
        # Rejected or mixed-weight partial cycles must never fit the next vector.
        self.anchor = self.last = None
        self.records = []
        self.observe(snapshot, weights, admitted, controller)

    def decide(self, snapshot, weights, controller, *, permit_changes, allow_probe, safe_boundary):
        demands = controller.demands(snapshot)
        ready = controller.ready_services(snapshot, demands)
        baseline_rows = self.window(weights)
        baseline_seconds = (sum(x['seconds'] for x in baseline_rows)/len(baseline_rows)
                            if baseline_rows else None)

        def decision(reason, selected=None, comparison=None):
            selected = dict(weights if selected is None else selected)
            return WeightDecision(selected, selected != weights, reason, None, None, demands, None,
                ready, prediction_scope='measured complete-cycle comparison; no throughput prediction',
                baseline_cycle_seconds=baseline_seconds, probe_comparison=comparison)

        if not permit_changes:
            return decision('cycle_feedback_advisory')
        if snapshot.publication_blocked or not safe_boundary:
            return decision('cycle_feedback_dependency_hold')
        if self.active_probe and snapshot.now >= self.active_probe['deadline_at']:
            return self.rollback(snapshot, weights, controller, 'cycle_probe_rollback_timeout')
        if snapshot.policy_version % 5 or snapshot.world_version != snapshot.policy_version//5:
            return decision('cycle_feedback_wait_complete_boundary')
        if self.active_probe:
            probe = self.active_probe
            rows = self.window(weights, since=probe['started_at'])
            if not rows:
                return decision('cycle_probe_observation_hold')
            base = probe['baseline_seconds']
            trial = sum(x['seconds'] for x in rows)/len(rows)
            penalty = controller.config.beta_seconds_per_weight*sum(
                abs(weights[s]-probe['baseline_weights'][s]) for s in SERVICES)
            threshold = max(controller.config.min_absolute_improvement_seconds,
                            base*controller.config.min_relative_improvement)
            stock_ok = all(rows[-1]['end_coverage'][s]+self.config.inventory_tolerance >=
                           probe['start_coverage'][s] for s in ('real', 'imagined'))
            keep = self.stable(rows) and stock_ok and base-trial-penalty >= threshold
            result = dict(accepted=keep, baseline_seconds=base, candidate_seconds=trial,
                penalty_seconds=penalty, threshold_seconds=threshold, inventory_reserve_preserved=stock_ok,
                baseline_weights=probe['baseline_weights'], candidate_weights=dict(weights),
                baseline_cycles=probe['baseline_cycles'], candidate_cycles=rows,
                complete_observation_window=True, completed_at=snapshot.now,
                reason='cycle_probe_accepted' if keep else 'cycle_probe_rollback',
                scope='within-run sequential probe; not randomized causal evidence')
            self.last_result = result
            self.active_probe = None
            self.last_change_at = snapshot.now
            return decision(result['reason'], weights if keep else result['baseline_weights'], result)
        if snapshot.policy_version <= self.config.warmup_policy_updates:
            return decision('cycle_feedback_warmup')
        if not allow_probe:
            return decision('cycle_feedback_probe_disabled')
        if not baseline_rows or not self.stable(baseline_rows):
            return decision('cycle_feedback_collect_baseline')
        if self.last_change_at is not None and snapshot.now-self.last_change_at < controller.config.cooldown_seconds:
            return decision('cycle_feedback_cooldown')
        # Ready training with full inventory is a valid direction at needs_batch.
        # When replenishment is needed, the producer gets the direction instead.
        targets = [s for s in ('world_infer', 'policy_infer', 'policy_train', 'world_train') if s in ready]
        # A cycle-boundary experiment spans the upcoming fifth update too.
        # World need not be immediately runnable to be a future ready consumer.
        if (demands['remaining_world_updates'] and snapshot.world_batch_ready
                and (targets or not any(demands['next_batch_missing'].values()))
                and 'world_train' not in targets):
            targets.append('world_train')
        candidates = controller.candidates()[1:]
        trial = None
        for target in targets:
            options = [c for c in candidates if c[target] > weights[target]
                       and (key(weights), key(c)) not in self.tried]
            # Avoid funding a producer by reducing its companion producer first.
            options.sort(key=lambda c: (sum(weights[s]-c[s] for s in ready if c[s]<weights[s]), key(c)))
            if options:
                trial = options[0]
                break
        if trial is None:
            return decision('cycle_feedback_no_untried_ready_direction')
        signature = (key(weights), key(trial))
        self.tried[signature] = snapshot.now
        while len(self.tried) > 128:
            self.tried.popitem(last=False)
        self.active_probe = dict(baseline_weights=dict(weights), candidate_weights=dict(trial),
            baseline_seconds=baseline_seconds, baseline_cycles=baseline_rows,
            start_coverage=self.coverage(controller, snapshot), started_at=snapshot.now,
            deadline_at=snapshot.now+self.config.timeout_seconds)
        self.last_change_at = snapshot.now
        return decision('complete_cycle_bounded_probe', trial)

    def rollback(self, snapshot, weights, controller, reason):
        probe = self.active_probe
        self.active_probe = None
        self.last_change_at = snapshot.now
        self.last_result = dict(reason=reason, accepted=False, baseline_weights=probe['baseline_weights'],
            candidate_weights=dict(weights), baseline_seconds=probe['baseline_seconds'],
            candidate_seconds=None, complete_observation_window=False, completed_at=snapshot.now)
        return WeightDecision(dict(probe['baseline_weights']), weights != probe['baseline_weights'], reason,
            None, None, controller.demands(snapshot), None, controller.ready_services(snapshot),
            probe_comparison=self.last_result,
            prediction_scope='timeout rollback; no candidate speed estimate')

    def snapshot(self):
        return dict(config=asdict(self.config), records=self.records,
            active_probe=self.active_probe, last_result=self.last_result,
            last_change_at=self.last_change_at, tried_count=len(self.tried))
