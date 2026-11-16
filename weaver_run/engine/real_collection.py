"""Serialized, metadata-only collection leases for Policy-only training.

The scheduler owns the same global shuffled initial-state queue and conservative
simulator-step accounting used by model-based collection. No frame store, WM
replay, windows, or imagination inventory is created by this coordinator.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import time

from initial_state_scheduler import InitialStateScheduler
from pipeline_budget import PipelineBudgetConfig, PipelineBudgetError, LeaseExpiredError


class RealCollectionCoordinator:
    def __init__(self, pipeline_config):
        self.pipeline_config = PipelineBudgetConfig.from_json(pipeline_config)
        if not self.pipeline_config.model_free:
            raise ValueError('RealCollectionCoordinator requires model_free collection')
        self.initial_state_scheduler = None
        self.initial_state_pool = None
        self.pipeline_stopped = False
        self._paused = False
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
        if owner is not None:
            status = owner.status()
            if self._budget_exhausted(status):
                return {'budget_exhausted': True, 'status': status}
        active_count = owner.status()['episodes_inflight'] if owner is not None else len(self._leases)
        if active_count >= self.pipeline_config.max_real_in_flight:
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
            # Whitelist the same small completion evidence as EpisodeReplayMixin.
            # The scheduler validates measured consumption before any lease is freed.
            self.initial_state_scheduler.finish(assigned['token'],
                action_steps=metadata['action_steps'], reset_steps=metadata['reset_env_steps'],
                success=metadata['success'], episode_id=episode_id,
                initial_state_hash=metadata['initial_state_hash'],
                termination_reason=metadata['termination_reason'],
                policy_versions=metadata.get('policy_versions', []))
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
                    **self._counts), initial_states=states,
                    frames={'resident_frame_bytes': 0}, train_windows=0, eval_windows=0, pending_draws=0)

    def _budget_exhausted(self, states):
        return bool(states and self.initial_state_pool and not self._failure
            and not self._leases and not states['episodes_inflight']
            and not states['reserved_env_steps'] and not states['episodes_cancelled']
            and not states['conservative_unknown_env_steps']
            and not states['recovery_required_tokens']
            and states['available_env_steps'] <= self.initial_state_pool['reset_steps'])

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
