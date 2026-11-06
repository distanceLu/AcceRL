"""Place complete K+1 critic segments by observation work, without changing data.

Every segment stays ordered and is evaluated exactly once at the caller's pinned
Policy version. This only balances independent replicas; it adds no model or gang.
"""
from __future__ import annotations


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
