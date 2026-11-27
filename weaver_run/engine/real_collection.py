"""Serialized, metadata-only collection leases for Policy-only training.

The scheduler owns the same global shuffled initial-state queue and conservative
simulator-step accounting used by model-based collection. No frame store, WM
replay, windows, or imagination inventory is created by this coordinator.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import time

from initial_state_scheduler import InitialStateScheduler
from pipeline_budget import PipelineBudgetConfig, PipelineBudgetError, LeaseExpiredError
from real_episode_sampling import REAL_BLOCK_SAMPLING


def _episode_rate(value):
    rate = float(value)
    if not math.isfinite(rate) or rate < 0 or (rate and not math.isfinite(60. / rate)):
        raise ValueError('Real episode rate must be finite, nonnegative and have a finite interval')
    return rate


def configured_episode_rate(value, *, model_free, real_workers, trajectory_jit, real_sampling, inventory_gate):
    rate = _episode_rate(value)
    if rate and (not model_free or real_workers != 8 or not trajectory_jit
                 or real_sampling not in REAL_BLOCK_SAMPLING or inventory_gate):
        raise ValueError('Fixed episode rate requires eight-worker model-free block sampling JIT '
                         'recipe with inventory gating disabled')
    return rate


class RealCollectionCoordinator:
    def __init__(self, pipeline_config, *, real_episodes_per_minute=0.):
        self.pipeline_config = PipelineBudgetConfig.from_json(pipeline_config)
        if not self.pipeline_config.model_free:
            raise ValueError('RealCollectionCoordinator requires model_free collection')
        self.initial_state_scheduler = None
        self.initial_state_pool = None
        self.pipeline_stopped = False
        self._paused = False
        self._gate_revision = 0
        self._gate_closed_since = None
        self._gate_closed_seconds = 0.
        self._gate_deferred_polls = 0
        self._gate_transitions = 0
        self._episode_rate = _episode_rate(real_episodes_per_minute)
        self._episode_interval = 60. / self._episode_rate if self._episode_rate else 0.
        self._next_episode_at = float('-inf')
        self._rate_deferred_polls = 0
        self._rate_idle_budget = None
        self._leases = {}
        self._next_token = 0
        self._completed_ids = set()
        self._counts = dict(real_admitted=0, real_completed=0, real_cancelled=0)
        self._versions = None
        self._failure = None

    def _initial_state_owner(self, pool):
        if not self.pipeline_config.initial_state_enabled:
            return None
        if not isinstance(pool, dict):
            raise ValueError('Explicit initial-state allocation requires simulator pool metadata')
        if pool.get('count') != self.pipeline_config.initial_state_count:
            raise ValueError('Loaded initial-state pool count differs from the experiment configuration')
        for name, minimum in (('max_action_steps', 1), ('reset_steps', 0)):
            if type(pool.get(name)) is not int or pool[name] < minimum:
                raise ValueError('Initial-state pool has invalid ' + name)
        horizon = self.pipeline_config.real_training_action_horizon
        if horizon is not None and pool['max_action_steps'] != horizon:
            raise ValueError('Real workers must apply the configured training horizon before reporting their pool')
        if self.initial_state_pool is not None and pool != self.initial_state_pool:
            raise ValueError('Real workers loaded different initial-state pools or horizons')
        if self.initial_state_scheduler is None:
            if pool['count'] == 500:
                from initial_state_env import spatial_pool_digest
                if (pool.get('task_suite') != 'libero_spatial' or
                        spatial_pool_digest(pool.get('task_names'), pool.get('task_pools')) != pool['sha256']):
                    raise ValueError('Spatial collection requires its task-indexed initial-state registry')
            # JSON metadata is the only pool material admitted; state arrays stay
            # in each simulator, and the asset digest identifies their contents.
            copied_pool = json.loads(json.dumps(pool, allow_nan=False))
            directory = os.environ.get('ACCERL_DATA_DIR')
            checkpoint = Path(directory) / 'initial_state_queue.json' if directory else None
            self.initial_state_scheduler = InitialStateScheduler(
                state_count=pool['count'], seed=self.pipeline_config.initial_state_seed,
                max_env_steps=self.pipeline_config.real_env_step_budget,
                checkpoint_path=checkpoint, state_asset_sha256=pool['sha256'])
            self.initial_state_pool = copied_pool
        return self.initial_state_scheduler

    def _check_leases(self):
        if self._failure is not None:
            raise PipelineBudgetError(self._failure)
        stale = [row['token'] for row in self._leases.values()
                 if time.monotonic() - row['acquired_at'] > self.pipeline_config.lease_timeout_seconds]
        if stale:
            self._failure = 'Real collection leases expired; refusing silent retry: %s' % stale
            raise LeaseExpiredError(self._failure)

    def initialize_initial_state_pool(self, pool):
        """Register validated simulator metadata without allocating a real episode."""
        self._check_leases()
        if not isinstance(pool, dict):
            raise ValueError('Explicit initial-state allocation requires simulator pool metadata')
        training_pool = deepcopy(pool)
        horizon = self.pipeline_config.real_training_action_horizon
        if horizon is not None:
            benchmark_horizon = training_pool.get('benchmark_action_horizon',
                                                 training_pool.get('max_action_steps'))
            if type(benchmark_horizon) is not int or horizon > benchmark_horizon:
                raise ValueError('Real training horizon exceeds the validated benchmark horizon')
            # Match initial_state_pool_metadata after the real worker applies
            # set_real_training_action_horizon, retaining benchmark provenance.
            training_pool['benchmark_action_horizon'] = benchmark_horizon
            training_pool['max_action_steps'] = horizon
        self._initial_state_owner(training_pool)
        return self.initial_state_checkpoint()

    def acquire_real(self, worker_id, initial_state_pool=None):
        if self.pipeline_stopped:
            return {'stopped': True}
        self._check_leases()
        owner = self._initial_state_owner(initial_state_pool)
        worker_id = str(worker_id)
        for row in self._leases.values():
            if row['worker_id'] == worker_id:
                return deepcopy(row)
        if self._paused:
            return None
        if self._episode_interval and time.monotonic() < self._next_episode_at:
            # 等待期间不占用初态或预算。最后一个回合结束后仍须向采集端报告预算耗尽。
            if not self._leases and owner is not None and self._rate_idle_budget is None:
                status = owner.status()
                self._rate_idle_budget = status if self._budget_exhausted(status) else {}
            if self._rate_idle_budget:
                return {'budget_exhausted': True, 'status': self._rate_idle_budget}
            self._rate_deferred_polls += 1
            return None
        if owner is not None:
            status = owner.status()
            if self._budget_exhausted(status):
                return {'budget_exhausted': True, 'status': status}
        active_count = status['episodes_inflight'] if owner is not None else len(self._leases)
        if active_count >= self.pipeline_config.max_real_in_flight:
            return None
        # 只挡新回合。已分配的初态、在途动作和预算收尾照常处理。
        if self._gate_closed_since is not None:
            self._gate_deferred_polls += 1
            return None
        initial_state = None
        if owner is not None:
            action_limit = initial_state_pool['max_action_steps']
            if self.pipeline_config.real_collection_action_limit is not None:
                action_limit = min(action_limit, self.pipeline_config.real_collection_action_limit)
            initial_state = owner.acquire(worker_id, max_action_steps=action_limit,
                                          reset_steps=initial_state_pool['reset_steps'])
            if initial_state is None:
                return None
        self._next_token += 1
        lease = dict(token='real:%d' % self._next_token, worker_id=worker_id,
                     acquired_at=time.monotonic())
        if initial_state is not None:
            lease['initial_state'] = initial_state
        self._leases[lease['token']] = lease
        self._counts['real_admitted'] += 1
        if self._episode_interval:
            self._next_episode_at = lease['acquired_at'] + self._episode_interval
            self._rate_idle_budget = None
        return deepcopy(lease)

    def finish_real_episode(self, token, payload):
        if self.pipeline_stopped:
            return {'stopped': True, 'discarded_episode': payload['episode_id']}
        self._check_leases()
        if not isinstance(payload, dict) or set(payload) != {'episode_id', 'metadata'}:
            raise ValueError('Real collection completion accepts episode identity and metadata only')
        episode_id, metadata = payload['episode_id'], payload['metadata']
        if not isinstance(episode_id, str) or not episode_id or episode_id in self._completed_ids:
            raise ValueError('Real collection requires a unique nonempty episode identity')
        if not isinstance(metadata, dict):
            raise ValueError('Real collection completion requires metadata')
        assigned = self._leases[token].get('initial_state')
        if assigned is not None:
            digest = metadata.get('initial_state_hash')
            reason = metadata.get('termination_reason')
            versions = metadata.get('policy_versions', [])
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(char not in '0123456789abcdef' for char in digest)):
                raise ValueError('Initial-state completion must retain a SHA256 identity')
            if not isinstance(reason, str) or not reason or len(reason) > 128:
                raise ValueError('Termination reason must be a short string')
            if not isinstance(versions, list) or any(type(value) is not int or value < 0 for value in versions):
                raise ValueError('Policy publication evidence must be a list of nonnegative versions')
            for name, field in (('initial_state_id', 'initial_state_id'), ('env_seed', 'env_seed'),
                                ('reset_env_steps', 'reset_steps')):
                if type(metadata.get(name)) is not int or metadata[name] != assigned[field]:
                    raise ValueError('Collected %s does not match its global assignment' % name)
            task_metadata = {}
            if self.initial_state_pool.get('task_suite') == 'libero_spatial':
                from initial_state_env import spatial_task_state
                task, local = spatial_task_state(assigned['initial_state_id'])
                if (metadata.get('task_suite') != 'libero_spatial'
                        or type(metadata.get('task_id')) is not int or metadata['task_id'] != task
                        or type(metadata.get('task_initial_state_id')) is not int
                        or metadata['task_initial_state_id'] != local):
                    raise ValueError('Collected task or local initial state does not match its global lease')
                task_metadata = dict(task_suite='libero_spatial', task_id=task, task_initial_state_id=local)
            # Whitelist the same small completion evidence as EpisodeReplayMixin.
            # The scheduler validates measured consumption before any lease is freed.
            self.initial_state_scheduler.finish(assigned['token'],
                action_steps=metadata['action_steps'], reset_steps=metadata['reset_env_steps'],
                success=metadata['success'], episode_id=episode_id,
                initial_state_hash=metadata['initial_state_hash'],
                termination_reason=metadata['termination_reason'],
                policy_versions=metadata.get('policy_versions', []), **task_metadata)
        del self._leases[token]
        self._completed_ids.add(episode_id)
        self._counts['real_completed'] += 1
        return {'episode_id': episode_id}

    def cancel_real(self, token):
        if self.pipeline_stopped:
            return {'stopped': True}
        assigned = self._leases[token].get('initial_state')
        if assigned is not None:
            self.initial_state_scheduler.cancel(assigned['token'], reason='producer_failure')
        del self._leases[token]
        self._counts['real_cancelled'] += 1
        return self.pipeline_status()

    def initial_state_checkpoint(self):
        return dict(enabled=self.pipeline_config.initial_state_enabled,
                    pool=deepcopy(self.initial_state_pool),
                    scheduler=self.initial_state_scheduler.snapshot() if self.initial_state_scheduler else None)

    def pipeline_status(self):
        self._check_leases()
        states = self.initial_state_scheduler.status() if self.initial_state_scheduler else {}
        return dict(budget=dict(config=self.pipeline_config.to_dict(),
                    active_real=len(self._leases), active_imagine=0,
                    reserved_frame_bytes=0, pending_episode_count=0, pending_seed_count=0,
                    stopped=self.pipeline_stopped, failure=self._failure, paused=self._paused,
                    budget_exhausted=self._budget_exhausted(states),
                    reset_steps=self.initial_state_pool['reset_steps'] if self.initial_state_pool else None,
                    rollout_gate_closed=self._gate_closed_since is not None,
                    rollout_gate_closed_seconds=self._gate_closed_seconds + (
                        time.monotonic() - self._gate_closed_since if self._gate_closed_since is not None else 0.),
                    rollout_gate_transitions=self._gate_transitions,
                    rollout_gate_deferred_polls=self._gate_deferred_polls,
                    real_episodes_per_minute=self._episode_rate,
                    real_rate_deferred_polls=self._rate_deferred_polls,
                    real_rate_wait_seconds=max(0., self._next_episode_at - time.monotonic()),
                    **self._counts), initial_states=states,
                    frames={'resident_frame_bytes': 0}, train_windows=0, eval_windows=0, pending_draws=0)

    def _budget_exhausted(self, states):
        return bool(states and self.initial_state_pool and not self._failure
            and not self._leases and not states['episodes_inflight']
            and not states['reserved_env_steps'] and not states['episodes_cancelled']
            and not states['conservative_unknown_env_steps']
            and not states['recovery_required_tokens']
            and states['available_env_steps'] <= self.initial_state_pool['reset_steps'])

    def set_rollout_gate(self, revision, closed):
        # Replay 是唯一发布方。迟到的旧通知不能覆盖较新的准入状态。
        if self._episode_rate:
            raise ValueError('The fixed real episode rate and Replay inventory gate are mutually exclusive')
        if revision <= self._gate_revision:
            return
        self._gate_revision = revision
        if closed == (self._gate_closed_since is not None):
            return
        self._gate_transitions += 1
        now = time.monotonic()
        if closed:
            self._gate_closed_since = now
        else:
            self._gate_closed_seconds += now - self._gate_closed_since
            self._gate_closed_since = None

    def set_paused(self, paused):
        """Pause new admissions; existing leases retain identity and can finish."""
        if type(paused) is not bool:
            raise ValueError('paused must be boolean')
        self._check_leases()
        self._paused = paused
        return self.pipeline_status()

    def stop_pipeline(self):
        # A stop cannot prove how far an active simulator stepped. Charge every
        # remaining reservation conservatively before discarding its local lease.
        for token in list(self._leases):
            self.cancel_real(token)
        self.pipeline_stopped = True
        return self.pipeline_status()

    def set_fresh_replay_versions(self, policy, world=0):
        if type(policy) is not int or policy < 0 or type(world) is not int or world != 0:
            raise ValueError('Model-free publication requires a nonnegative Policy version and no World version')
        if self._versions is not None and policy < self._versions[0]:
            raise ValueError('Policy publication versions cannot move backwards')
        self._versions = policy, 0
        return {'production_versions': [policy, 0]}
