"""CPU episode ownership and bounded producer integration for the Ray actors.

Only the WM Replay actor owns episode records. RPC payloads contain a projected
rank shard or one context, never a copy of the entire episode cache.
"""
from __future__ import annotations

import random
import os
from pathlib import Path
import time
import uuid
import numpy as np


class EpisodeReplayMixin:
    def init_pipeline(self, config, world_config, context_frames, window_frames,
                      policy_chunks_per_seed, minimum_train_windows):
        from episode_frame_store import EpisodeFrameStore
        from pipeline_budget import PipelineBudgetConfig, PipelineBudget
        from fresh_config import FreshConfig
        self.pipeline_config = PipelineBudgetConfig.from_json(config)
        self.fresh = FreshConfig.from_world_config(world_config)
        if self.fresh.early_seed_context and (world_config or {}).get('backend') != 'weaver':
            raise ValueError('Explicit reset-padded early contexts require WEAVER mask admission')
        self.pipeline = PipelineBudget(self.pipeline_config, policy_chunks_per_seed,
                                       fresh=self.fresh)
        self.frame_store = EpisodeFrameStore()
        self.pipeline_world_config = world_config
        self.context_frames = int(context_frames)
        self.window_frames = int(window_frames)
        self.minimum_train_windows = int(minimum_train_windows)
        self.imagination_records = {}
        self.window_tickets = {}
        self.pipeline_stopped = False
        self.initial_state_scheduler = None
        self.initial_state_pool = None
        self.initial_state_leases = {}
        self.initial_state_exhaustion_logged = False

    def _initial_state_owner(self, pool):
        if not self.pipeline_config.initial_state_enabled:
            return None
        if pool is None:
            raise ValueError('Explicit initial-state allocation requires actual simulator pool metadata')
        if pool['count'] != self.pipeline_config.initial_state_count:
            raise ValueError('Loaded initial-state pool count differs from the experiment configuration')
        if self.initial_state_pool is not None and pool != self.initial_state_pool:
            raise ValueError('Real workers loaded different initial-state pools or horizons')
        if self.initial_state_scheduler is None:
            from initial_state_scheduler import InitialStateScheduler
            directory = os.environ.get('ACCERL_DATA_DIR')
            checkpoint = Path(directory) / 'initial_state_queue.json' if directory else None
            self.initial_state_scheduler = InitialStateScheduler(
                state_count=pool['count'], seed=self.pipeline_config.initial_state_seed,
                max_env_steps=self.pipeline_config.real_env_step_budget,
                checkpoint_path=checkpoint, state_asset_sha256=pool['sha256'])
            self.initial_state_pool = dict(pool)
            self._pipeline_event('initial_state_queue_ready', pool=pool,
                                 status=self.initial_state_scheduler.status())
        return self.initial_state_scheduler

    def initial_state_checkpoint(self):
        return {'enabled': self.pipeline_config.initial_state_enabled,
                'pool': self.initial_state_pool,
                'scheduler': self.initial_state_scheduler.snapshot() if self.initial_state_scheduler else None}

    def _pipeline_event(self, name, **details):
        writer = getattr(self, 'telemetry', None) or getattr(self, 'capture_telemetry', None)
        if writer is not None:
            writer.event(name, details=details)

    def _prune_imagination_records(self):
        pinned = self.pipeline.pinned_episode_ids()
        for episode_id in list(self.imagination_records):
            if episode_id not in pinned:
                del self.imagination_records[episode_id]
        # Counters are retained by the scheduler; consume its finite notices.
        self.pipeline.drain_released_episode_ids()

    def reserve_windows(self, global_size):
        if self.window_tickets:
            raise RuntimeError('Only one outstanding WM global draw is allowed')
        batch = random.sample(self.buffer, global_size)
        episode_ids = {b.episode_id for b in batch}
        fresh = episode_ids - self.sampled_episode_ids
        self.sampled_episode_ids.update(episode_ids)
        ticket_id = uuid.uuid4().hex
        ledger = getattr(self, 'capture_ledger', None)
        metadata = (ledger.reserve([b.sample_id for b in batch]) if ledger else
                    {'reservation_id': ticket_id, 'sample_ids': [b.sample_id for b in batch],
                     'replay': 'wm', 'consumer': 'training', 'reserved_ns': time.time_ns()})
        if (self.pipeline_world_config or {}).get('reward_semantics') == 'delta_native_score':
            from replay_projection import native_reward_cohort
            specification = (self.pipeline_world_config or {}).get('backend_options', {}).get('native_reward_head', {})
            if specification.get('mode') == 'libero_discounted_success_value_v1':
                metadata['reward_cohort'] = native_reward_cohort(batch, specification)
            else:
                metadata['reward_cohort'] = native_reward_cohort(batch)
        stats = {'unique_episodes': len(episode_ids),
                 'fresh_episode_ratio': len(fresh) / max(1, len(episode_ids)),
                 'capture_meta': metadata}
        self.window_tickets[ticket_id] = (batch, stats)
        self._pipeline_event('wm_window_reservation', ticket_id=ticket_id,
                             global_batch_size=global_size, pixels_materialized=False)
        return {'ticket_id': ticket_id, 'global_batch_size': global_size}

    def project_windows(self, ticket_id, rank, local_size, consumer):
        from replay_projection import materialize_window_shard
        batch, stats = self.window_tickets[ticket_id]
        world_size, remainder = divmod(len(batch), local_size)
        if remainder:
            raise ValueError('Nonintegral rank partition')
        result = materialize_window_shard(batch, stats, rank, local_size, world_size, consumer)
        self._pipeline_event('wm_projection', ticket_id=ticket_id, rank=rank,
                             consumer=consumer, shape=list(result[0].shape),
                             pixel_bytes=int(result[0].nbytes), dtype=str(result[0].dtype))
        return result

    def describe_window_shard(self, ticket_id, rank, local_size, consumer="ctrl"):
        from replay_projection import describe_window_shard
        batch, stats = self.window_tickets[ticket_id]
        world_size, remainder = divmod(len(batch), local_size)
        if remainder:
            raise ValueError("Nonintegral streaming World rank partition")
        return describe_window_shard(batch, stats, rank, local_size, world_size, consumer)

    def project_window_microshard(self, ticket_id, rank, local_size, start, end):
        from replay_projection import materialize_window_microshard
        batch, stats = self.window_tickets[ticket_id]
        world_size, remainder = divmod(len(batch), local_size)
        if remainder:
            raise ValueError("Nonintegral streaming World micro partition")
        result = materialize_window_microshard(batch, stats, rank, local_size, world_size, start, end)
        self._pipeline_event('wm_micro_projection', ticket_id=ticket_id, rank=rank,
            consumer='ctrl', micro_start=start, micro_end=end, shape=list(result[0].shape),
            pixel_bytes=int(result[0].nbytes), dtype=str(result[0].dtype), consumption_recorded=False)
        return result

    def release_windows(self, ticket_id):
        del self.window_tickets[ticket_id]
        self._pipeline_event('wm_window_release', ticket_id=ticket_id)
        return True

    def policy_drawn(self, global_samples):
        self.pipeline.policy_drawn(global_samples)
        return True

    def _resident_frame_bytes(self):
        return self.frame_store.stats()['image_bytes']

    def acquire_real(self, worker_id, initial_state_pool=None):
        if self.pipeline_stopped:
            return {'stopped': True}
        state_owner = self._initial_state_owner(initial_state_pool)
        if state_owner is not None:
            status = state_owner.status()
            if status['available_env_steps'] <= initial_state_pool['reset_steps'] and not status['episodes_inflight']:
                if not self.initial_state_exhaustion_logged:
                    self._pipeline_event('real_env_budget_exhausted', status=status)
                    self.initial_state_exhaustion_logged = True
                return {'budget_exhausted': True, 'status': status}
        self.pipeline.check_stale_leases()
        self._prune_imagination_records()
        # Byte pressure can evict old training windows, but preserves enough
        # windows for a full global draw. Eval and active reservations stay pinned.
        limit = self.pipeline_config.max_resident_frame_bytes
        reserve = self.pipeline_config.max_episode_bytes + self.pipeline.reserved_frame_bytes
        while self._resident_frame_bytes() + reserve > limit:
            if self.pipeline.drop_oldest_pending() is not None:
                self._prune_imagination_records()
            elif len(self.buffer) <= self.minimum_train_windows:
                break
            else:
                sample = self.buffer.popleft()
                ledger = getattr(self, 'capture_ledger', None)
                if ledger is not None:
                    ledger.evict(sample.sample_id, 'episode_frame_byte_budget')
                del sample
        lease = self.pipeline.acquire_real(worker_id, resident_bytes=self._resident_frame_bytes())
        self._prune_imagination_records()
        if lease is not None and state_owner is not None:
            state_lease = state_owner.acquire(str(worker_id),
                max_action_steps=initial_state_pool['max_action_steps'], reset_steps=initial_state_pool['reset_steps'])
            if state_lease is None:
                self.pipeline.cancel_real(lease['token'])
                return None
            self.initial_state_leases[lease['token']] = state_lease
            lease = dict(lease, initial_state=state_lease)
            self._pipeline_event('initial_state_assigned', pipeline_token=lease['token'],
                                 allocation=state_lease, status=state_owner.status())
        return lease

    def finish_real_episode(self, token, payload):
        if self.pipeline_stopped:
            return {'stopped': True, 'discarded_episode': payload['episode_id']}
        from episode_frame_store import EpisodeRecord
        if payload['episode_id'] in self.frame_store:
            raise ValueError('Duplicate episode identity')
        record = EpisodeRecord(**payload, backend_config=self.pipeline_world_config)
        if self.initial_state_scheduler is not None:
            assigned = self.initial_state_leases[token]
            metadata = dict(record.metadata)
            if metadata.get('initial_state_id') != assigned['initial_state_id']:
                raise ValueError('Collected initial state does not match its global assignment')
            if metadata.get('env_seed') != assigned['env_seed']:
                raise ValueError('Collected environment seed does not match its global assignment')
            if metadata.get('reset_env_steps') != assigned['reset_steps']:
                raise ValueError('Completed episode must account for all reserved reset settling steps')
            completion = self.initial_state_scheduler.finish(assigned['token'],
                action_steps=len(record.frames) - 1, reset_steps=metadata['reset_env_steps'],
                success=bool(record.success[-1]), episode_id=record.episode_id,
                initial_state_hash=metadata['initial_state_hash'],
                termination_reason=metadata['termination_reason'],
                policy_versions=metadata.get('policy_versions', []))
            del self.initial_state_leases[token]
            self._pipeline_event('initial_state_completed', completion=completion,
                                 status=self.initial_state_scheduler.status())
        # Scheduler validation precedes registration and all Replay mutations.
        kwargs = {}
        if self.fresh.seed_balancing:
            if record.success is None:
                raise ValueError('FRESH seed outcome must come from measured real success labels')
            censored = (record.metadata.get('budget_censored') is True
                        or not bool(record.terminated[-1] or record.truncated[-1])
                        or record.metadata.get('termination_reason') in
                        ('real_env_budget', 'incomplete', 'user_requested_stop', 'producer_failure', 'cancelled'))
            kwargs.update(success=bool(record.success[-1]), outcome_censored=censored)
        outcome = self.pipeline.complete_real(token, record.episode_id, len(record.frames),
                    self.context_frames, record.frames.nbytes, **kwargs)
        self.frame_store.register(record)
        self.imagination_records[record.episode_id] = record
        windows = [record.window(start, self.window_frames)
                   for start in range(max(0, len(record.frames) - self.window_frames + 1))]
        self.add_batch(windows)
        self._prune_imagination_records()
        self._pipeline_event('episode_frames_added', episode_id=record.episode_id,
                             frames=len(record.frames), frame_bytes=int(record.frames.nbytes),
                             windows=len(windows), budget=outcome)
        return {'episode_id': record.episode_id, 'windows': len(windows)}

    def cancel_real(self, token):
        if self.pipeline_stopped:
            return {'stopped': True}
        if token in self.initial_state_leases:
            assigned = self.initial_state_leases.pop(token)
            cancellation = self.initial_state_scheduler.cancel(assigned['token'], reason='producer_failure')
            self._pipeline_event('initial_state_cancelled', cancellation=cancellation,
                                 status=self.initial_state_scheduler.status())
        result = self.pipeline.cancel_real(token)
        self._prune_imagination_records()
        return result

    def acquire_imagine(self, worker_id):
        if self.pipeline_stopped:
            return {'stopped': True}
        self.pipeline.check_stale_leases()
        lease = self.pipeline.acquire_imagine(worker_id)
        if lease is None:
            return None
        record = self.imagination_records[lease['episode_id']]
        start = lease['start']
        end = start + self.context_frames
        job = dict(lease, frames=record.frames[start:end], actions=record.actions[start:end],
                   instruction=record.instruction, frame_groups=record.frame_groups)
        if self.fresh.seed_balancing or self.fresh.early_seed_context:
            from episode_frame_store import materialize_imagination_context
            job.update(materialize_imagination_context(record, start, self.context_frames,
                       self.pipeline_world_config['schema']['dt_seconds']))
            job.update(seed_episode_success=bool(record.success[-1]),
                       seed_episode_action_steps=record.frame_count - 1,
                       seed_episode_termination_reason=record.metadata.get('termination_reason'))
        job['real_anchor_terminal']=bool(record.terminated[end-1] or record.truncated[end-1]
                                         or record.success[end-1])
        if record.states is not None:
            if not (self.fresh.seed_balancing or self.fresh.early_seed_context):
                job['states'] = record.states[start:end]
        self._pipeline_event('imagination_lease', token=lease['token'],
                             episode_id=record.episode_id, start=start,
                             context_bytes=int(job['frames'].nbytes),
                             **({'fresh_seed': {key: job.get(key) for key in
                                 ('seed_anchor', 'seed_bucket', 'seed_scheduler_mode', 'context_padding',
                                  'frame_indices', 'valid_mask', 'action_mask', 'dt')}}
                                if self.fresh.seed_balancing or self.fresh.early_seed_context else {}))
        return job

    def finish_imagine(self, token, actual_samples):
        if self.pipeline_stopped:
            return {'stopped': True}
        result = self.pipeline.finish_imagine(token, actual_samples)
        self._prune_imagination_records()
        self._pipeline_event('imagination_completed', token=token, actual_samples=actual_samples)
        return result

    def cancel_imagine(self, token):
        if self.pipeline_stopped:
            return {'stopped': True}
        result = self.pipeline.cancel_imagine(token)
        self._prune_imagination_records()
        return result

    def pipeline_status(self):
        return {'budget': self.pipeline.stats(), 'frames': self.frame_store.stats(),
                'initial_states': self.initial_state_scheduler.status() if self.initial_state_scheduler else {},
                'train_windows': len(self.buffer), 'eval_windows': len(self.eval_buffer),
                'pending_draws': len(self.window_tickets)}

    def stop_pipeline(self):
        self.pipeline_stopped = True
        self.pipeline.stop()
        self._prune_imagination_records()
        return self.pipeline_status()


def run_producer(worker):
    """One lease and at most one synchronous model RPC per CPU producer."""
    import ray
    replay = worker.wm_replay_buffer
    real = worker.worker_mode == 'real'
    acquire = replay.acquire_real if real else replay.acquire_imagine
    cancel = replay.cancel_real if real else replay.cancel_imagine
    while True:
        if real and getattr(worker, 'initial_state_enabled', False):
            job = ray.get(acquire.remote(worker.wid, worker.initial_state_pool))
        else:
            job = ray.get(acquire.remote(worker.wid))
        if job is None:
            time.sleep(worker.pipeline_poll_seconds)
            continue
        if job.get('stopped'):
            return
        if job.get('budget_exhausted'):
            time.sleep(max(worker.pipeline_poll_seconds, 1.0))
            continue
        worker.pipeline_lease = job['token']
        worker.pipeline_job = job
        try:
            if real:
                worker.get_one_episode()
            else:
                worker.pipeline_generated_samples = 0
                if worker.world_config.backend == 'weaver':
                    from pipeline_imagination import imagine_weaver_seed
                    imagine_weaver_seed(worker, job)
                else:
                    from pipeline_imagination import imagine_ctrl_seed
                    imagine_ctrl_seed(worker, job)
                # _process_traj has awaited Policy Replay insertion before this
                # acknowledgement replenishes unused in-flight reservations.
                ray.get(replay.finish_imagine.remote(job['token'], worker.pipeline_generated_samples))
        except BaseException:
            try:
                ray.get(cancel.remote(job['token']))
            except Exception:
                pass  # Preserve the actual producer failure for driver fail-fast.
            raise
        finally:
            worker.pipeline_lease = None
