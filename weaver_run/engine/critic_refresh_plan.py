"""Place critic segments, then optionally prune unused original micro batches.

The full plan retains K+1 observations per segment. The dependency plan preserves
that placement and micro geometry, evaluating every required current Value once
at the caller's pinned Policy version. Neither path adds a model or training gang.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
import os


def dependency_pruning_enabled():
    value = os.environ.get('ACCERL_CRITIC_DEPENDENCY_PRUNING', '0')
    if value not in ('0', '1'):
        raise ValueError('ACCERL_CRITIC_DEPENDENCY_PRUNING must be 0 or 1')
    return value == '1'


@dataclass(frozen=True)
class RefreshedSuffix:
    """Only a current-critic suffix; terminal endpoint0 is a masked placeholder."""
    segment_id: str
    first_index: int
    values: tuple
    terminal: bool

    def for_segment(self, segment, selected):
        if (self.segment_id != segment.segment_id or type(self.first_index) is not int
                or self.first_index != min(selected) or not 0 <= self.first_index < segment.count
                or type(self.terminal) is not bool or self.terminal != bool(segment.terminated[-1])
                or len(self.values) != segment.count - self.first_index + 1
                or any(isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x) for x in self.values)
                or (self.terminal and self.values[-1] != 0.)):
            raise ValueError('Current critic suffix does not cover the selected GAE dependencies')
        return self.first_index, self.values


@dataclass(frozen=True)
class DependencyRefreshGroup:
    inputs: tuple
    positions: tuple
    batch_sizes: tuple
    segments: tuple
    original_observations: int
    original_batches: int
    required_observations: int

    def restore(self, values):
        if (not isinstance(values, (list, tuple)) or len(values) != len(self.positions)
                or any(isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x) for x in values)):
            raise ValueError('Sparse critic response must contain every executed finite Value exactly once')
        if len(set(self.positions)) != len(self.positions):
            raise ValueError('Duplicate sparse critic result positions')
        measured = dict(zip(self.positions, values))
        result = {}
        for sid, first, count, terminal in self.segments:
            needed = [(sid, i) for i in range(first, count + int(not terminal))]
            if any(position not in measured for position in needed):
                raise ValueError('Sparse critic response is missing a current GAE dependency')
            suffix = tuple(float(measured[p]) for p in needed) + ((0.,) if terminal else ())
            result[sid] = RefreshedSuffix(sid, first, suffix, terminal)
        return result


def balanced_refresh_groups(segments, replicas):
    if type(replicas) is not int or replicas < 1:
        raise ValueError('Critic refresh needs a positive number of independent replicas')
    items = list(segments)
    identities = [sid for sid, _ in items]
    if len(set(identities)) != len(identities):
        raise ValueError('Critic refresh segment identities must be unique')
    if any(not len(observations) for _, observations in items):
        raise ValueError('Critic refresh cannot omit a complete K+1 segment')
    original = [[] for _ in range(replicas)]
    planned = [[] for _ in range(replicas)]
    loads = [0] * replicas
    for index, item in enumerate(items):
        original[index % replicas].append((index, item))
    # Longest first, stable identity/order ties, then least loaded replica.
    for index in sorted(range(len(items)), key=lambda i: (-len(items[i][1]), i)):
        rank = min(range(replicas), key=lambda r: (loads[r], r))
        planned[rank].append((index, items[index]))
        loads[rank] += len(items[index][1])
    def peak(groups):
        return max((sum(len(item[1]) for _, item in group) for group in groups), default=0)
    # A heuristic may lose on an unusual input. Never increase the old plan's
    # maximum observation count; ties keep the original placement.
    chosen = planned if peak(planned) < peak(original) else original
    return [[item for _, item in sorted(group)] for group in chosen]


def refresh_plan_receipt(groups):
    return {'strategy': 'complete_segments_balanced_by_observation_count',
            'observations_by_replica': [sum(len(obs) for _, obs in g) for g in groups],
            'segments_by_replica': [len(g) for g in groups],
            'all_k_plus_one_observations_preserved': True,
            'independent_replicas': True}


def dependency_refresh_groups(segments, selected_indices, replicas, micro=2):
    """Filter whole ORIGINAL micro blocks, never repack or rebalance retained rows."""
    if type(micro) is not int or micro not in (1, 2):
        raise ValueError('Reviewed critic dependency micro must be 1 or 2')
    items = list(segments)
    ids = [s.segment_id for s in items]
    if len(set(ids)) != len(ids) or set(ids) != set(selected_indices):
        raise ValueError('Selected dependencies must identify each segment exactly once')
    by_id = {s.segment_id:s for s in items}
    first = {}
    for s in items:
        indices = selected_indices[s.segment_id]
        if (not indices or any(type(i) is not int or not 0 <= i < s.count for i in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError('Selected dependency indices must be unique valid chunk indices')
        first[s.segment_id] = min(indices)
    original = balanced_refresh_groups([(s.segment_id, s.observations) for s in items], replicas)
    result = []
    for group in original:
        positions = [(sid,i) for sid, obs in group for i in range(len(obs))]
        kept, inputs, sizes = [], [], []
        def required(position):
            sid, i = position
            s = by_id[sid]
            return i >= first[sid] and (i < s.count or not s.terminated[-1])
        for offset in range(0, len(positions), micro):
            batch = positions[offset:offset+micro]
            if any(required(p) for p in batch):
                kept.extend(batch)
                inputs.extend(by_id[sid].observations[i] for sid, i in batch)
                sizes.append(len(batch))
        result.append(DependencyRefreshGroup(tuple(inputs), tuple(kept), tuple(sizes),
            tuple((sid,first[sid],by_id[sid].count,bool(by_id[sid].terminated[-1])) for sid,obs in group),
            len(positions), (len(positions)+micro-1)//micro, sum(required(p) for p in positions)))
    return tuple(result)


def dependency_refresh_receipt(groups):
    evaluated = [len(g.inputs) for g in groups]
    original = [g.original_observations for g in groups]
    return {'strategy':'original_micro_pairs_dependency_pruned',
            'observations_by_replica':evaluated,
            'original_observations_by_replica':original,
            'segments_by_replica':[len(g.segments) for g in groups],
            'required_observations_by_replica':[g.required_observations for g in groups],
            'original_batches_by_replica':[g.original_batches for g in groups],
            'executed_batches_by_replica':[len(g.batch_sizes) for g in groups],
            'stored_k_plus_one_preserved':True, 'required_dependencies_covered':True,
            'all_k_plus_one_observations_preserved':evaluated == original,
            'original_micro_pairs_preserved':True, 'independent_replicas':True}
