"""CPU-only admission and demand budgets for the decoupled data pipeline.

One serialized owner (the WM replay actor) calls this object. No Ray, Torch,
background threads, or model state lives here. Frame bytes are canonical uint8
bytes; the caller supplies total resident bytes when enforcing the frame store
limit. A full Policy replay is not a stop signal: replay sampling does not remove
items, so *successful global sample draws* replenish bounded production credits.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import time
from typing import Any, Callable


class PipelineBudgetError(RuntimeError):
    pass


class LeaseExpiredError(PipelineBudgetError):
    pass


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class PipelineBudgetConfig:
    """Conservative initial budgets, configurable rather than measured optima.

    Watermarks describe the bounded *fresh production budget*, not physical
    replay length. Consuming one replay sample grants ``generation_per_sample``
    new samples. Credits cannot accumulate beyond the high watermark.
    """
    real_workers: int = 4
    imagination_workers: int = 4
    max_real_in_flight: int = 4
    max_imagine_in_flight: int = 4
    max_pending_episodes: int = 32
    seeds_per_episode: int = 16
    max_episode_frames: int = 1025
    bytes_per_frame: int = 2 * 3 * 224 * 224
    max_resident_frame_bytes: int = 8 * 1024**3
    policy_high_watermark: int = 4096
    policy_low_watermark: int = 2048
    generation_per_sample: float = 0.25
    poll_seconds: float = 0.25
    lease_timeout_seconds: float = 3600.0
    replace_old_pending: bool = True
    initial_state_enabled: bool = False
    initial_state_count: int = 50
    initial_state_seed: int = 20261026
    real_env_step_budget: int = 11000
    # Real collection may stop before the task horizon. This is a censored
    # prefix, not a failed task; evaluation and imagination keep their horizons.
    real_collection_action_limit: int | None = None

    def __post_init__(self) -> None:
        integer_fields = (
            "real_workers", "imagination_workers", "max_real_in_flight", "max_imagine_in_flight", "max_pending_episodes",
            "seeds_per_episode", "max_episode_frames", "bytes_per_frame",
            "max_resident_frame_bytes", "policy_high_watermark",
        )
        for name in integer_fields:
            _positive_int(name, getattr(self, name))
        _positive_int("initial_state_count", self.initial_state_count)
        _nonnegative_int("initial_state_seed", self.initial_state_seed)
        _positive_int("real_env_step_budget", self.real_env_step_budget)
        if type(self.initial_state_enabled) is not bool:
            raise ValueError("initial_state_enabled must be boolean")
        if self.real_collection_action_limit is not None:
            _positive_int("real_collection_action_limit", self.real_collection_action_limit)
            if not self.initial_state_enabled:
                raise ValueError("real_collection_action_limit requires initial_state_enabled")
        _nonnegative_int("policy_low_watermark", self.policy_low_watermark)
        if self.policy_low_watermark >= self.policy_high_watermark:
            raise ValueError("policy_low_watermark must be below policy_high_watermark")
        for name in ("generation_per_sample", "poll_seconds", "lease_timeout_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.replace_old_pending, bool):
            raise ValueError("replace_old_pending must be boolean")
        if self.max_episode_bytes > self.max_resident_frame_bytes:
            raise ValueError("max_resident_frame_bytes cannot fit one maximum episode")

    @property
    def max_episode_bytes(self) -> int:
        return self.max_episode_frames * self.bytes_per_frame

    @property
    def initial_policy_credit(self) -> int:
        return self.policy_high_watermark

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, source: str | Path | dict[str, Any] | None) -> "PipelineBudgetConfig":
        if source is None:
            return cls()
        if isinstance(source, cls):
            return source
        if isinstance(source, dict):
            raw = dict(source)
        else:
            raw = json.loads(Path(source).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("pipeline budget JSON must be an object")
        unknown = set(raw) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"unknown pipeline budget fields: {sorted(unknown)}")
        return cls(**raw)


def uniform_seed_starts(frame_count: int, context_frames: int, limit: int) -> list[int]:
    """Evenly cover valid starts, including both ends, without RNG or duplicates.

    Match the existing context-window contract: at least one following frame is
    required, so the final start is ``frame_count - context_frames - 1``.
    """
    _nonnegative_int("frame_count", frame_count)
    _positive_int("context_frames", context_frames)
    _positive_int("limit", limit)
    count = max(0, frame_count - context_frames)
    if count == 0:
        return []
    selected = min(count, limit)
    if selected == 1:
        return [0]
    return [i * (count - 1) // (selected - 1) for i in range(selected)]


@dataclass
class _Episode:
    episode_id: str
    starts: deque
    frame_bytes: int
    active: int = 0
    success: bool | None = None
    frame_count: int = 0
    context_frames: int = 0


class PipelineBudget:
    """Bounded FIFO episode admission with round-robin seed dispatch.

    Every acquisition has exactly one completion/cancellation. The owner must
    keep ``pinned_episode_ids()`` resident, drain released IDs, and surface stale
    or failed leases as a fatal pipeline error rather than retrying a bad seed.
    """
    def __init__(self, config: PipelineBudgetConfig, policy_chunks_per_seed: int,
                 clock: Callable[[], float] = time.monotonic, *, fresh=None):
        _positive_int("policy_chunks_per_seed", policy_chunks_per_seed)
        if policy_chunks_per_seed > config.policy_high_watermark:
            raise ValueError("one seed must fit the policy production high watermark")
        self.config = config
        self.policy_chunks_per_seed = policy_chunks_per_seed
        self._clock = clock
        from fresh_config import FreshConfig
        self.fresh = FreshConfig.from_mapping(fresh)
        self.seed_scheduler = None
        self._seed_outcome_censored = 0
        if self.fresh.seed_balancing:
            from fresh_seed_scheduler import FreshSeedScheduler
            self.seed_scheduler = FreshSeedScheduler(
                getattr(self.fresh, 'seed_scheduler_mode', 'balanced'),
                getattr(self.fresh, 'seed_scheduler_seed', 20261028))
        self._episodes: OrderedDict[str, _Episode] = OrderedDict()
        self._dispatch: deque[str] = deque()
        self._real: dict[str, dict] = {}
        self._imagine: dict[str, dict] = {}
        self._released: deque[str] = deque()
        self._next_token = 0
        self._credits = float(config.policy_high_watermark)
        self._paused = False
        self._replay_pressure = None
        self._pressure_revision = -1
        self._production_versions = None
        self._stopped = False
        self._failure: str | None = None
        self._last_block_reason: str | None = None
        # Residency is owned by the separate Policy replay actor. An omitted
        # observation is unknown, not evidence that its queue is empty.
        self.policy_size = None
        self.policy_added = 0
        self.policy_sampled = 0
        self._counts = dict(real_admitted=0, real_completed=0, real_cancelled=0,
                            episodes_dropped=0, seeds_dropped=0, episodes_drained=0,
                            imagine_admitted=0, imagine_completed=0, imagine_cancelled=0)

    def _token(self, kind: str) -> str:
        self._next_token += 1
        return f"{kind}:{self._next_token}"

    def _guard(self) -> None:
        if self._failure is not None:
            raise PipelineBudgetError(self._failure)
        self.check_stale_leases()

    def check_stale_leases(self) -> None:
        now = self._clock()
        stale = [lease["token"] for lease in (*self._real.values(), *self._imagine.values())
                 if now - lease["acquired_at"] > self.config.lease_timeout_seconds]
        if stale:
            message = f"pipeline leases expired; refusing silent retry: {stale}"
            self.fail(message)
            raise LeaseExpiredError(message)

    @property
    def reserved_policy_samples(self) -> int:
        return sum(lease["reserved_samples"] for lease in self._imagine.values())

    @property
    def reserved_frame_bytes(self) -> int:
        return sum(lease["reserved_bytes"] for lease in self._real.values())

    def _return_credits(self, value: float) -> None:
        cap = self.config.policy_high_watermark - self.reserved_policy_samples
        self._credits = min(float(cap), self._credits + value)
        self._refresh_pause()

    def _refresh_pause(self) -> None:
        if self._paused:
            refill = max(self.policy_chunks_per_seed,
                         self.config.policy_high_watermark - self.config.policy_low_watermark)
            if self._credits >= refill:
                self._paused = False
        elif self._credits < self.policy_chunks_per_seed:
            self._paused = True

    def set_production_versions(self, policy, world):
        versions = (policy, world)
        if any(type(value) is not int or value < 0 for value in versions):
            raise ValueError('Production versions must be nonnegative integers')
        if self._production_versions is not None and any(a < b for a,b in zip(versions,self._production_versions)):
            raise ValueError('Production versions cannot move backwards')
        if versions != self._production_versions:
            self._production_versions = versions
            self._replay_pressure = None
        return dict(production_versions=list(versions), pressure_cleared=self._replay_pressure is None)

    def set_replay_pressure(self, snapshot):
        """A short-lived next-cohort shortage may use existing credits early.

        This does not replenish credits, alter hysteresis, or accept stale data.
        Replay supplies eligibility; the production owner keeps all lease caps.
        A monotonic revision rejects delayed updates after clear/cancellation.
        """
        self._guard()
        if not isinstance(snapshot, dict):
            raise ValueError('Replay pressure must be a mapping')
        revision = snapshot.get('revision')
        if type(revision) is not int or revision < 0:
            raise ValueError('Replay pressure requires a nonnegative revision')
        versions = snapshot.get('versions')
        if not isinstance(versions, (tuple, list)) or len(versions) != 2 or any(type(v) is not int or v < 0 for v in versions):
            raise ValueError('Replay pressure requires exact published versions')
        missing = snapshot.get('missing')
        if not isinstance(missing, dict) or set(missing) != {'real', 'imagined'} or any(type(v) is not int or v < 0 for v in missing.values()):
            raise ValueError('Replay pressure requires exact source deficits')
        if not isinstance(snapshot.get('request_id'), str) or not snapshot['request_id']:
            raise ValueError('Replay pressure requires its cohort identity')
        baseline = snapshot.get('admitted_imagined_chunks')
        if type(baseline) is not int or baseline < 0:
            raise ValueError('Replay pressure requires its cumulative admitted imagined counter')
        observed = snapshot.get('observed_monotonic')
        if type(observed) not in (float, int) or not math.isfinite(observed):
            raise ValueError('Replay pressure requires its observation clock')
        if revision <= self._pressure_revision:
            return self.pressure_status()
        self._pressure_revision = revision
        # All actors of this admitted topology share one host monotonic clock.
        # Receipt time must not extend an already-expired advisory RPC.
        if tuple(versions) != self._production_versions or not 0 <= self._clock()-observed < 2.0:
            self._replay_pressure = None
            return dict(self.pressure_status(), pressure_rejected=True)
        self._replay_pressure = dict(snapshot, versions=list(versions), missing=dict(missing),
                                     policy_added_at_snapshot=baseline,
                                     expires_at=observed+2.0)
        return self.pressure_status()

    def pressure_status(self):
        pressure = self._replay_pressure
        active = pressure is not None and self._clock() < pressure['expires_at']
        missing = pressure['missing']['imagined'] if active else 0
        supplied = max(0, self.policy_added-pressure['policy_added_at_snapshot']) if active else 0
        remaining = max(0, missing-supplied-self.reserved_policy_samples)
        return dict(available_policy_credits=self._credits,
                    reserved_policy_samples=self.reserved_policy_samples,
                    policy_chunks_per_seed=self.policy_chunks_per_seed,
                    seed_granularity_overshoot_max=self.policy_chunks_per_seed-1,
                    policy_paused=self._paused, active_imagine=len(self._imagine),
                    pending_seed_count=sum(len(ep.starts) for ep in self._episodes.values()),
                    pressure_missing_imagined=missing,
                    pressure_supplied_since_snapshot=supplied,
                    pressure_remaining_imagined=remaining,
                    pressure_revision=self._pressure_revision,
                    production_versions=self._production_versions,
                    pressure_active=bool(active and missing),
                    pressure_can_spend_existing_credit=bool(remaining > 0
                        and self._credits >= self.policy_chunks_per_seed),
                    policy_added=self.policy_added, policy_sampled=self.policy_sampled,
                    stopped=self._stopped, failure=self._failure,
                    credit_refill_performed=False)

    def policy_drawn(self, global_samples: int, policy_size: int | None = None) -> dict:
        """Call once after a successful global draw, not once per training rank."""
        self._guard()
        _positive_int("global_samples", global_samples)
        if policy_size is not None:
            self.policy_size = _nonnegative_int("policy_size", policy_size)
        self.policy_sampled += global_samples
        self._return_credits(global_samples * self.config.generation_per_sample)
        return self.stats()

    def update_demand(self, policy_size: int, policy_added: int, policy_sampled: int) -> dict:
        """Optional cumulative-counter synchronization; do not double-report draws.

        Normally use ``policy_drawn`` and ``finish_imagine``. External production
        is charged here only for counts not already recorded by completion.
        """
        self._guard()
        for name, value in (("policy_size", policy_size), ("policy_added", policy_added),
                            ("policy_sampled", policy_sampled)):
            _nonnegative_int(name, value)
        if policy_added < self.policy_added or policy_sampled < self.policy_sampled:
            raise ValueError("policy production/consumption counters must be monotonic")
        generated = policy_added - self.policy_added
        sampled = policy_sampled - self.policy_sampled
        self.policy_size, self.policy_added, self.policy_sampled = policy_size, policy_added, policy_sampled
        self._return_credits(sampled * self.config.generation_per_sample)
        self._credits -= generated
        self._refresh_pause()
        return self.stats()

    def _drop_episode(self, episode_id: str, *, dropped: bool) -> None:
        episode = self._episodes[episode_id]
        if episode.active:
            raise PipelineBudgetError("attempted to release an episode with active imagination")
        del self._episodes[episode_id]
        self._dispatch = deque(eid for eid in self._dispatch if eid != episode_id)
        self._released.append(episode_id)
        if dropped:
            self._counts["episodes_dropped"] += 1
            self._counts["seeds_dropped"] += len(episode.starts)
            if self.seed_scheduler is not None:
                from fresh_seed_scheduler import seed_bucket
                for start in episode.starts:
                    self.seed_scheduler.dropped[seed_bucket(episode.success,
                        start + episode.context_frames - 1, episode.frame_count)] += 1
        else:
            self._counts["episodes_drained"] += 1

    def _evict_old_pending(self) -> bool:
        for eid, episode in self._episodes.items():
            if episode.active == 0:
                self._drop_episode(eid, dropped=True)
                return True
        return False

    def drop_oldest_pending(self) -> str | None:
        """Let the owner shed one imagination pin under total frame pressure.

        Active episodes are protected. The owner must release its corresponding
        cache reference and remeasure total resident bytes before admission.
        """
        self._guard()
        if not self.config.replace_old_pending:
            return None
        for eid, episode in self._episodes.items():
            if episode.active == 0:
                self._drop_episode(eid, dropped=True)
                return eid
        return None

    def acquire_real(self, worker_id: Any, resident_bytes: int | None = None) -> dict | None:
        self._guard()
        if self._stopped:
            return None
        worker_id = str(worker_id)
        if any(lease["worker_id"] == worker_id for lease in self._real.values()):
            raise PipelineBudgetError(f"real worker {worker_id} already owns a lease")
        if len(self._real) >= self.config.max_real_in_flight:
            self._last_block_reason = "real_in_flight"
            return None
        if resident_bytes is not None:
            _nonnegative_int("resident_bytes", resident_bytes)
        # Eviction frees imagination's pin only. Caller may have training/eval
        # pins too; supplied total resident bytes therefore remain conservative.
        while len(self._episodes) + len(self._real) >= self.config.max_pending_episodes:
            if not self.config.replace_old_pending or not self._evict_old_pending():
                self._last_block_reason = "pending_episodes"
                return None
        tracked_bytes = sum(episode.frame_bytes for episode in self._episodes.values())
        resident = tracked_bytes if resident_bytes is None else max(resident_bytes, tracked_bytes)
        while resident + self.reserved_frame_bytes + self.config.max_episode_bytes > self.config.max_resident_frame_bytes:
            if resident_bytes is not None or not self.config.replace_old_pending or not self._evict_old_pending():
                self._last_block_reason = "resident_frames"
                return None
            resident = sum(episode.frame_bytes for episode in self._episodes.values())
        token = self._token("real")
        lease = dict(token=token, worker_id=worker_id, acquired_at=self._clock(),
                     reserved_bytes=self.config.max_episode_bytes)
        self._real[token] = lease
        self._counts["real_admitted"] += 1
        self._last_block_reason = None
        return dict(lease)

    def complete_real(self, token: str, episode_id: str, frame_count: int,
                      context_frames: int, frame_bytes: int, *, success=None,
                      outcome_censored=False) -> dict:
        self._guard()
        if token not in self._real:
            raise PipelineBudgetError(f"unknown or completed real lease: {token}")
        _nonnegative_int("frame_count", frame_count)
        _nonnegative_int("frame_bytes", frame_bytes)
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("episode_id must be a nonempty string")
        if episode_id in self._episodes:
            raise PipelineBudgetError(f"episode is already pinned: {episode_id}")
        if frame_count > self.config.max_episode_frames:
            raise ValueError("episode exceeds max_episode_frames; truncation is forbidden")
        if frame_bytes > self._real[token]["reserved_bytes"]:
            raise ValueError("episode exceeds its reserved canonical frame byte budget")
        if type(outcome_censored) is not bool:
            raise ValueError('outcome_censored must be boolean')
        if self.seed_scheduler is not None and not outcome_censored and type(success) is not bool:
            raise ValueError('FRESH seed balancing needs a measured completed episode outcome')
        if self.seed_scheduler is not None and outcome_censored:
            starts = []
            self._seed_outcome_censored += 1
        elif self.fresh.early_seed_context:
            from fresh_seed_scheduler import context_seed_starts
            starts = context_seed_starts(frame_count, context_frames,
                self.config.seeds_per_episode, include_early=True)
        else:
            starts = uniform_seed_starts(frame_count, context_frames, self.config.seeds_per_episode)
        del self._real[token]
        self._counts["real_completed"] += 1
        if starts:
            self._episodes[episode_id] = _Episode(episode_id, deque(starts), frame_bytes,
                success=success, frame_count=frame_count, context_frames=context_frames)
            self._dispatch.append(episode_id)
        else:
            self._released.append(episode_id)
        result = dict(episode_id=episode_id, starts=starts, seed_count=len(starts),
                      pinned=bool(starts))
        if self.seed_scheduler is not None and outcome_censored:
            result['seed_skip_reason'] = 'censored_real_episode_outcome'
        return result

    def cancel_real(self, token: str) -> None:
        if token not in self._real:
            raise PipelineBudgetError(f"unknown or completed real lease: {token}")
        del self._real[token]
        self._counts["real_cancelled"] += 1

    def acquire_imagine(self, worker_id: Any) -> dict | None:
        self._guard()
        if self._stopped:
            return None
        worker_id = str(worker_id)
        if any(lease["worker_id"] == worker_id for lease in self._imagine.values()):
            raise PipelineBudgetError(f"imagination worker {worker_id} already owns a lease")
        self._refresh_pause()
        if len(self._imagine) >= self.config.max_imagine_in_flight:
            self._last_block_reason = "imagine_in_flight"
            return None
        if self._paused and not self.pressure_status()['pressure_can_spend_existing_credit']:
            self._last_block_reason = "policy_demand"
            return None
        if not self._dispatch:
            self._last_block_reason = "no_seeds"
            return None
        bucket = None
        if self.seed_scheduler is None:
            episode_id = self._dispatch.popleft()
            episode = self._episodes[episode_id]
            start = episode.starts.popleft()
        else:
            from fresh_seed_scheduler import seed_bucket
            candidates = [(eid, start, seed_bucket(self._episodes[eid].success,
                          start + self._episodes[eid].context_frames - 1,
                          self._episodes[eid].frame_count))
                          for eid in self._dispatch for start in self._episodes[eid].starts]
            episode_id, start, bucket = self.seed_scheduler.choose(candidates)
            self._dispatch.remove(episode_id)
            episode = self._episodes[episode_id]
            episode.starts.remove(start)
        episode.active += 1
        if episode.starts:
            self._dispatch.append(episode_id)
        token = self._token("imagine")
        lease = dict(token=token, worker_id=worker_id, acquired_at=self._clock(),
                     episode_id=episode_id, start=start,
                     reserved_samples=self.policy_chunks_per_seed)
        if self.fresh.seed_balancing or self.fresh.early_seed_context:
            lease.update(seed_anchor=start + episode.context_frames - 1,
                         seed_bucket=bucket,
                         seed_scheduler_mode=self.seed_scheduler.mode if self.seed_scheduler else 'legacy_round_robin')
        self._credits -= self.policy_chunks_per_seed
        self._imagine[token] = lease
        self._counts["imagine_admitted"] += 1
        self._refresh_pause()
        self._last_block_reason = None
        return dict(lease)

    def _finish_imagine(self, token: str, actual_samples: int, cancelled: bool) -> dict:
        self._guard()
        if token not in self._imagine:
            raise PipelineBudgetError(f"unknown or completed imagination lease: {token}")
        _nonnegative_int("actual_samples", actual_samples)
        lease = self._imagine[token]
        if actual_samples > lease["reserved_samples"]:
            raise ValueError("produced samples exceed the seed's prior demand reservation")
        del self._imagine[token]
        self.policy_added += actual_samples
        self._return_credits(lease["reserved_samples"] - actual_samples)
        episode = self._episodes[lease["episode_id"]]
        episode.active -= 1
        released = not episode.starts and episode.active == 0
        if released:
            self._drop_episode(episode.episode_id, dropped=False)
        self._counts["imagine_cancelled" if cancelled else "imagine_completed"] += 1
        if self.seed_scheduler is not None:
            counts = self.seed_scheduler.cancelled if cancelled else self.seed_scheduler.completed
            counts[lease['seed_bucket']] += 1
        return dict(episode_id=episode.episode_id,
                    released_episode_id=episode.episode_id if released else None,
                    actual_samples=actual_samples)

    def finish_imagine(self, token: str, actual_samples: int) -> dict:
        """After Replay add is ACKed; reserve covers the seed's worst case."""
        return self._finish_imagine(token, actual_samples, cancelled=False)

    def cancel_imagine(self, token: str) -> dict:
        """Cancel an unpublished seed; cancellation does not silently requeue it."""
        return self._finish_imagine(token, 0, cancelled=True)

    def pinned_episode_ids(self) -> list[str]:
        return list(self._episodes)

    def drain_released_episode_ids(self) -> list[str]:
        released = list(self._released)
        self._released.clear()
        return released

    def stop(self) -> dict:
        """Revoke outstanding admissions and release all coordinator pins."""
        self._stopped = True
        self._released.extend(self._episodes)
        self._episodes.clear()
        self._dispatch.clear()
        self._counts["real_cancelled"] += len(self._real)
        self._counts["imagine_cancelled"] += len(self._imagine)
        if self.seed_scheduler is not None:
            for lease in self._imagine.values():
                self.seed_scheduler.cancelled[lease['seed_bucket']] += 1
        returned = self.reserved_policy_samples
        self._real.clear()
        self._imagine.clear()
        self._return_credits(float(returned))
        return self.stats()

    def fail(self, reason: str) -> dict:
        self._failure = str(reason)
        return self.stop()

    def stats(self) -> dict:
        result = dict(config=asdict(self.config), policy_chunks_per_seed=self.policy_chunks_per_seed,
                    policy_size=self.policy_size, policy_added=self.policy_added,
                    policy_sampled=self.policy_sampled, available_policy_credits=self._credits,
                    reserved_policy_samples=self.reserved_policy_samples,
                    reserved_frame_bytes=self.reserved_frame_bytes,
                    pending_episode_count=len(self._episodes),
                    pending_seed_count=sum(len(episode.starts) for episode in self._episodes.values()),
                    pinned_frame_bytes=sum(episode.frame_bytes for episode in self._episodes.values()),
                    active_real=len(self._real), active_imagine=len(self._imagine),
                    policy_paused=self._paused, stopped=self._stopped, failure=self._failure,
                    last_block_reason=self._last_block_reason, **self._counts)
        result['replay_pressure'] = self.pressure_status()
        if self.seed_scheduler is not None:
            from collections import Counter
            from fresh_seed_scheduler import seed_bucket
            pending = Counter(seed_bucket(episode.success, start + episode.context_frames - 1,
                               episode.frame_count) for episode in self._episodes.values()
                               for start in episode.starts)
            result['fresh_seed_scheduler'] = self.seed_scheduler.stats(pending)
            result['fresh_seed_scheduler']['outcome_censored_episodes'] = self._seed_outcome_censored
        return result
