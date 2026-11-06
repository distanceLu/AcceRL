"""CPU dependency plans preserving the random cohort and original rank shards.

Only selected chunks incur loss/uses. Each rank may duplicate context shared with
another rank; that overhead is explicit, never hidden by resampling episodes.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Node:
    segment: str
    index: int
    selected: int | None = None
    bootstrap: bool = False


@dataclass(frozen=True)
class Bundle:
    nodes: tuple
    selected: tuple


def build_plan(observations, micro, version):
    if type(micro) is not int or micro < 1 or not observations or len(observations) % micro:
        raise ValueError('JIT requires a full equal-size accumulation window')
    segments, selected = {}, {}
    for pos, obs in enumerate(observations):
        segment = obs['_jit_segment']; sid = segment['segment_id']; index = obs['_jit_index']
        if obs['_jit_version'] != version or type(index) is not int or not 0 <= index < len(segment['rewards']):
            raise ValueError('JIT context version or selected index mismatch')
        if sid in segments and segments[sid]['count'] != segment['count']:
            raise ValueError('Inconsistent JIT segment geometry')
        segments[sid] = segment
        key = (sid, index)
        if key in selected:
            raise ValueError('Duplicate selected chunk')
        selected[key] = pos
    nodes = []
    for sid, segment in segments.items():
        first = min(index for name, index in selected if name == sid)
        count = segment['count']
        if (type(count) is not int or count < 1 or len(segment['observations']) != count+1 or len(segment['rewards']) != count
                or len(segment['terminated']) != count or len(segment['truncated']) != count
                or any(segment['terminated'][:-1]) or any(segment['truncated'][:-1])):
            raise ValueError('JIT cannot cross an episode or branch boundary')
        if segment.get('sampling') == 'imagined_trajectories' and segment.get('source') == 'imagined':
            if count != 4 or {i for name,i in selected if name == sid} != set(range(4)):
                raise ValueError('JIT rank must own every chunk of its imagined trajectory')
        if not segment['terminated'][-1]:
            nodes.append(Node(sid, count, bootstrap=True))
        nodes.extend(Node(sid, i, selected.get((sid, i))) for i in range(count-1, first-1, -1))
    bundles, pending, chosen = [], [], []
    for node in nodes:
        pending.append(node)
        if node.selected is not None:
            chosen.append(node.selected)
        if len(chosen) == micro:
            bundles.append(Bundle(tuple(pending), tuple(chosen)))
            pending, chosen = [], []
    if pending or chosen:
        raise ValueError('Incomplete reverse dependency batch')
    if sorted(i for b in bundles for i in b.selected) != list(range(len(observations))):
        raise ValueError('JIT planner changed the sampled cohort')
    return segments, tuple(bundles)
