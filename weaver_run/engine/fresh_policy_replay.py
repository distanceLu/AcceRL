"""Ordered FRESH segments, conservative use leases, and current-critic GAE.

This module is CPU-only. Recomputing values does not refresh generated rewards
or model transitions, and behaviour logits are immutable sampling-time data.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
import random
import time
import uuid

import numpy as np


class FreshReplayNotReady(RuntimeError):
    """The required fresh cohort is unavailable; callers must apply backpressure."""


def generalized_advantages(rewards, values, terminated, truncated, gamma, lambda_):
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    terminal = np.asarray(terminated, dtype=bool)
    truncated = np.asarray(truncated, dtype=bool)
    if (rewards.ndim != 1 or values.shape != (len(rewards) + 1,)
            or terminal.shape != rewards.shape or truncated.shape != rewards.shape
            or not np.isfinite(rewards).all() or not np.isfinite(values).all()
            or not 0 <= gamma <= 1 or not 0 <= lambda_ <= 1):
        raise ValueError('FRESH GAE requires finite ordered K rewards/K+1 values and explicit masks')
    advantages = np.zeros_like(rewards)
    carry = 0.
    for index in range(len(rewards) - 1, -1, -1):
        bootstrap = 0. if terminal[index] else values[index + 1]
        # Time-limit and branch-horizon endpoints bootstrap, but never leak GAE
        # across a new episode or an explicitly truncated segment boundary.
        delta = rewards[index] + gamma * bootstrap - values[index]
        continuation = not (terminal[index] or truncated[index])
        carry = delta + gamma * lambda_ * float(continuation) * carry
        advantages[index] = carry
    targets = advantages + values[:-1]
    if not np.isfinite(advantages).all() or not np.isfinite(targets).all():
        raise ValueError('FRESH GAE overflows its finite target contract')
    return advantages.astype(np.float32), targets.astype(np.float32)


def _array(value, dtype, shape=None):
    array = np.asarray(value, dtype=dtype)
    if (shape is not None and array.shape != shape) or not np.isfinite(array).all():
        raise ValueError('FRESH segment has an invalid shape or nonfinite data')
    # bytes backing makes storage immutable even if the caller mutates its input.
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class PolicySegment:
    """Learning boundaries; source env termination flags remain in versions.

    A complete finite LIBERO horizon is a learning terminal even when the
    simulator's original flag is ``truncated``. Earlier budget censoring and
    imagined branch horizons retain their bootstrap boundary instead.
    """
    segment_id: str
    episode_id: str
    source: str
    observations: tuple
    action_tokens: np.ndarray
    behaviour_logits: np.ndarray
    rewards: np.ndarray
    generation_values: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    action_mask: np.ndarray
    versions: tuple
    gamma: float
    lambda_: float
    phi: tuple
    reward_source: str
    created_ns: int
    seed_anchor: int | None = None

    @classmethod
    def create(cls, *, segment_id, episode_id, source, observations, action_tokens,
               behaviour_logits, rewards, generation_values, terminated, truncated,
               versions, gamma, lambda_, action_mask=None, phi=None,
               reward_source=None, created_ns=None, seed_anchor=None):
        count = len(rewards)
        if (source not in ('real', 'imagined') or not segment_id or not episode_id
                or not 1 <= count <= 1024 or len(observations) != count + 1
                or len(versions) != count or not math.isfinite(gamma)
                or not math.isfinite(lambda_)):
            raise ValueError('FRESH segment requires bounded ordered transitions and provenance')
        tokens = _array(action_tokens, np.int64)
        logits = _array(behaviour_logits, np.float32)
        if (tokens.ndim != 3 or tokens.shape[0] != count or logits.ndim != 4
                or logits.shape[:-1] != tokens.shape or logits.shape[-1] < 2
                or (tokens < 0).any() or (tokens >= logits.shape[-1]).any()):
            raise ValueError('Behaviour token and categorical-logit geometry differs')
        masks = np.ones(tokens.shape, dtype=bool) if action_mask is None else np.asarray(action_mask)
        if masks.dtype != np.bool_ or masks.shape != tokens.shape or not masks.any(axis=(1, 2)).all():
            raise ValueError('Every FRESH transition needs at least one actually executed action token')
        terminal = _array(terminated, bool, (count,))
        truncation = _array(truncated, bool, (count,))
        if (terminal[:-1] | truncation[:-1]).any():
            raise ValueError('A stored FRESH segment cannot cross an episode/branch endpoint')
        copied_versions = []
        for version in versions:
            row = dict(version)
            if type(row.get('policy')) is not int or row['policy'] < 0:
                raise ValueError('FRESH transitions require the actual behaviour Policy version')
            if source == 'imagined':
                world = row.get('world_reward_bundle_version', row.get('world_bundle_version'))
                if type(world) is not int or world < 0:
                    raise ValueError('Imagined transitions require their actual World/RM version')
            copied_versions.append(row)
        scores = tuple([None] * (count + 1) if phi is None else phi)
        if len(scores) != count + 1 or any(v is not None and (not math.isfinite(float(v)) or not 0 <= float(v) <= 1) for v in scores):
            raise ValueError('FRESH Phi must be aligned bounded scores or explicit missing values')
        raw_rewards = _array(rewards, np.float32, (count,))
        values = _array(generation_values, np.float32, (count + 1,))
        generalized_advantages(raw_rewards, values, terminal, truncation, gamma, lambda_)
        return cls(str(segment_id), str(episode_id), source, tuple(observations), tokens, logits,
                   raw_rewards, values, terminal, truncation, _array(masks, bool), tuple(copied_versions),
                   float(gamma), float(lambda_), scores,
                   reward_source or ('environment_success' if source == 'real' else 'generation_time_native_score_delta'),
                   int(time.time_ns() if created_ns is None else created_ns), seed_anchor)

    @property
    def count(self):
        return len(self.rewards)

    def sample_id(self, index):
        return '%s:chunk%d' % (self.segment_id, index)


class SegmentReplay:
    """Capacity is measured in chunks; segments remain ordered and indivisible.

    Outstanding reservations count against the use cap. A failed update may
    release its reservation, but never creates unrecorded extra training uses.
    """
    def __init__(self, capacity, config, *, rng=None, on_add=None, on_evict=None):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('FRESH Replay capacity must be positive')
        self.capacity, self.config = capacity, config
        self.rng = rng or random
        self.on_add, self.on_evict = on_add, on_evict
        self.segments = OrderedDict()
        self.samples, self.reservations = {}, {}
        self.versions = None

    def set_versions(self, policy, world):
        if any(type(v) is not int or v < 0 for v in (policy, world)):
            raise ValueError('FRESH versions must be successful published update counts')
        if self.versions and (policy < self.versions[0] or world < self.versions[1]):
            raise ValueError('FRESH publication versions cannot move backwards')
        self.versions = policy, world

    def _evict(self, segment_id, reason):
        segment = self.segments[segment_id]
        if any(self.samples[segment.sample_id(i)]['reserved'] for i in range(segment.count)):
            return False
        for index in range(segment.count):
            sample_id = segment.sample_id(index)
            del self.samples[sample_id]
            if self.on_evict:
                self.on_evict(sample_id, reason)
        del self.segments[segment_id]
        return True

    def add(self, segment):
        if not isinstance(segment, PolicySegment) or segment.count > self.capacity:
            raise ValueError('A complete segment must fit FRESH Replay capacity')
        if segment.segment_id in self.segments or any(segment.sample_id(i) in self.samples for i in range(segment.count)):
            raise ValueError('Duplicate FRESH segment/sample identity')
        while len(self.samples) + segment.count > self.capacity:
            if not any(self._evict(key, 'fresh_capacity') for key in list(self.segments)):
                raise FreshReplayNotReady('FRESH Replay capacity is pinned by in-flight training')
        self.segments[segment.segment_id] = segment
        for index in range(segment.count):
            sample_id = segment.sample_id(index)
            self.samples[sample_id] = {'segment_id':segment.segment_id, 'index':index, 'uses':0, 'reserved':0}
            if self.on_add:
                self.on_add(sample_id, segment.episode_id, index, model_versions=segment.versions[index])

    def _eligible(self, sample):
        if not self.versions:
            return False
        segment = self.segments[sample['segment_id']]
        if not self.config.replay_age_limits:
            return True
        index = sample['index']; versions = segment.versions[index]
        policy_age = self.versions[0] - versions['policy']
        if not 0 <= policy_age <= self.config.max_policy_age:
            return False
        if segment.source == 'imagined':
            world = versions.get('world_reward_bundle_version', versions.get('world_bundle_version'))
            if not 0 <= self.versions[1] - world <= self.config.max_world_age:
                return False
        return sample['uses'] + sample['reserved'] < self.config.max_sample_uses

    def reserve(self, count):
        if type(count) is not int or count < 1:
            raise ValueError('FRESH cohort size must be positive')
        eligible = [(sid, value) for sid, value in self.samples.items() if self._eligible(value)]
        if self.config.real_policy_replay:
            real_count = max(1, min(count - 1, int(round(count * self.config.real_fraction))))
            if count < 2:
                raise ValueError('Mixed-source FRESH cohorts require at least two samples')
            groups = {source:[sid for sid, value in eligible if self.segments[value['segment_id']].source == source]
                      for source in ('real', 'imagined')}
            if len(groups['real']) < real_count or len(groups['imagined']) < count - real_count:
                raise FreshReplayNotReady('Fresh source quotas unavailable; no stale or alternate-source fallback')
            selected = self.rng.sample(groups['real'], real_count) + self.rng.sample(groups['imagined'], count - real_count)
            self.rng.shuffle(selected)
        else:
            if len(eligible) < count:
                raise FreshReplayNotReady('Fresh cohort unavailable; no stale Replay fallback')
            selected = self.rng.sample([sid for sid, _ in eligible], count)
        token = 'fresh-reservation:' + uuid.uuid4().hex
        for sid in selected:
            self.samples[sid]['reserved'] += 1
        self.reservations[token] = set(selected)
        return token, selected

    def cancel(self, token):
        selected = self.reservations.pop(token)
        for sid in selected:
            self.samples[sid]['reserved'] -= 1

    def commit(self, token, sample_ids):
        pending = self.reservations.get(token)
        sample_ids = list(sample_ids)
        if pending is None or len(set(sample_ids)) != len(sample_ids) or not set(sample_ids) <= pending:
            raise ValueError('FRESH commit must match one unconsumed rank shard exactly once')
        # Nothing mutates until all identities pass the common version check.
        for sid in sample_ids:
            sample = self.samples[sid]
            if not self._eligible({**sample, 'reserved':sample['reserved']-1}):
                raise ValueError('FRESH sample exceeded its age/use contract before actual consumption')
        for sid in sample_ids:
            self.samples[sid]['reserved'] -= 1
            self.samples[sid]['uses'] += 1
            pending.remove(sid)
        if not pending:
            del self.reservations[token]

    def materialize(self, sample_ids, refreshed_values=None):
        from critic_refresh_plan import RefreshedSuffix
        targets = {}
        selected = {}
        for sid in sample_ids:
            sample = self.samples[sid]
            selected.setdefault(sample['segment_id'], []).append(sample['index'])
        for sid in sample_ids:
            segment = self.segments[self.samples[sid]['segment_id']]
            if segment.segment_id not in targets:
                values = (refreshed_values[segment.segment_id] if refreshed_values is not None else segment.generation_values)
                first = 0
                if isinstance(values, RefreshedSuffix):
                    first, values = values.for_segment(segment, selected[segment.segment_id])
                adv, target = generalized_advantages(segment.rewards[first:], values,
                    segment.terminated[first:], segment.truncated[first:], segment.gamma, segment.lambda_)
                targets[segment.segment_id] = first, adv, target
        observations, tokens, advantages, logits, returns, records = [], [], [], [], [], []
        for sid in sample_ids:
            sample = self.samples[sid];index = sample['index'];segment = self.segments[sample['segment_id']]
            first, adv, target = targets[segment.segment_id]
            offset = index - first
            observation = dict(segment.observations[index])
            observation['_fresh_action_mask'] = segment.action_mask[index].copy()
            observations.append(observation);tokens.append(segment.action_tokens[index])
            advantages.append(adv[offset]);logits.append(segment.behaviour_logits[index]);returns.append(target[offset])
            records.append(dict(sample_id=sid, source=segment.source, segment_id=segment.segment_id,
                reward=float(segment.rewards[index]), advantage=float(adv[offset]), value_target=float(target[offset]),
                phi_before=segment.phi[index], phi_after=segment.phi[index+1],
                generation_policy_version=segment.versions[index]['policy'],
                generation_world_version=segment.versions[index].get('world_bundle_version'),
                generation_rm_version=segment.versions[index].get('world_reward_bundle_version'),
                refresh_policy_version=self.versions[0] if refreshed_values is not None else None,
                reward_source=segment.reward_source, rewards_refreshed=False,
                learning_terminal=bool(segment.terminated[index]),learning_truncated=bool(segment.truncated[index]),
                env_terminated=segment.versions[index].get('env_terminated'),
                env_truncated=segment.versions[index].get('env_truncated'),
                termination_reason=segment.versions[index].get('termination_reason'),
                seed_anchor=segment.seed_anchor, created_ns=segment.created_ns))
        return ((observations, np.stack(tokens), np.asarray(advantages, np.float32), np.stack(logits),
                 np.asarray(returns, np.float32)), records)
