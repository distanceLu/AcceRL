"""CPU-only, bounded seed stratification; no value-based ranking or model calls.

Positions refer to measured frame indices in the complete real episode. Early
contexts can have a negative *context origin*, but their anchor is a real frame.
The negative positions are never claimed to be observed physical history.
"""
from __future__ import annotations

from collections import Counter
import random


POSITIONS = ('early', 'middle', 'late')
BUCKETS = tuple(outcome + ':' + position for outcome in ('success', 'failure')
                for position in POSITIONS)


def _integer(name, value, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError('%s must be an integer >= %d' % (name, minimum))
    return value


def context_seed_starts(frame_count, context_frames, limit, *, include_early=False):
    """Evenly cover real anchors with a following transition, without duplicates.

    With padding disabled, this is exactly the legacy uniform start formula.
    Enabling padding extends the domain left to real anchor zero without adding
    more than ``limit`` candidates or changing any admission/resource quota.
    """
    _integer('frame_count', frame_count, 0)
    _integer('context_frames', context_frames, 1)
    _integer('limit', limit, 1)
    if type(include_early) is not bool:
        raise ValueError('include_early must be boolean')
    first_anchor = 0 if include_early else context_frames - 1
    last_anchor = frame_count - 2  # A real following frame must exist.
    count = max(0, last_anchor - first_anchor + 1)
    if count == 0:
        return []
    selected = min(count, limit)
    anchors = ([first_anchor] if selected == 1 else
               [first_anchor + i * (count - 1) // (selected - 1)
                for i in range(selected)])
    return [anchor - context_frames + 1 for anchor in anchors]


def seed_bucket(success, anchor_frame_index, frame_count):
    """Outcome x thirds of measured episode positions, not predicted utility."""
    if type(success) is not bool:
        raise ValueError('Stratification requires the completed real episode outcome')
    _integer('frame_count', frame_count, 2)
    _integer('anchor_frame_index', anchor_frame_index, 0)
    if anchor_frame_index >= frame_count - 1:
        raise ValueError('A seed anchor must have a real following frame')
    position = POSITIONS[min(2, anchor_frame_index * 3 // (frame_count - 1))]
    return ('success' if success else 'failure') + ':' + position


class FreshSeedScheduler:
    """Choose from the caller's finite pending set with a private RNG.

    Balanced mode cycles through available strata; unavailable strata are
    skipped rather than fabricating episodes. Candidate/episode order supplied
    by the caller breaks within-stratum ties. Random mode samples pending
    candidates uniformly and is an explicit ablation of stratification.
    """
    def __init__(self, mode='balanced', seed=20261028):
        if mode not in ('balanced', 'random'):
            raise ValueError('seed scheduler mode must be balanced or random')
        _integer('seed', seed, 0)
        self.mode = mode
        self.seed = seed
        self._rng = random.Random(seed)
        self._cursor = 0
        self.admitted = Counter()
        self.completed = Counter()
        self.cancelled = Counter()
        self.dropped = Counter()

    def choose(self, candidates):
        """Return one existing ``(episode_id, start, bucket)`` candidate."""
        if not candidates:
            raise ValueError('Cannot dispatch from an empty seed set')
        if any(item[2] not in BUCKETS for item in candidates):
            raise ValueError('Unknown seed bucket')
        if self.mode == 'random':
            selected = self._rng.choice(candidates)
        else:
            selected = None
            for offset in range(len(BUCKETS)):
                index = (self._cursor + offset) % len(BUCKETS)
                selected = next((item for item in candidates if item[2] == BUCKETS[index]), None)
                if selected is not None:
                    self._cursor = (index + 1) % len(BUCKETS)
                    break
        self.admitted[selected[2]] += 1
        return selected

    def stats(self, pending):
        return {'protocol': 'fresh_seed_strata_v1', 'mode': self.mode, 'seed': self.seed,
                'bucket_definition': 'real_outcome_x_measured_episode_position_thirds',
                'pending': {key: int(pending.get(key, 0)) for key in BUCKETS},
                'admitted': {key: int(self.admitted[key]) for key in BUCKETS},
                'completed': {key: int(self.completed[key]) for key in BUCKETS},
                'cancelled': {key: int(self.cancelled[key]) for key in BUCKETS},
                'dropped': {key: int(self.dropped[key]) for key in BUCKETS},
                'utility_or_success_probability_used': False}
