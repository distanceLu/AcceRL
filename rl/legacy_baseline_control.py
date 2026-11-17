"""Opt-in, CPU-only bookkeeping for the original PPO driver's lifecycle.

This module does not schedule training, change replay/GAE, or save optimizer
state. The driver invokes the boundary hook only after every training rank and
the original inference publication have finished.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time
from fractions import Fraction


def validate_success_snapshot(snapshot, window_size):
    """Accept exact episode counts, never a rounded or partial success claim."""
    if not isinstance(snapshot, dict):
        raise ValueError('A success snapshot is required for success-target mode')
    integer_fields = ('window_episodes', 'window_successes', 'window_size',
                      'total_episodes_processed', 'total_env_steps')
    for name in integer_fields:
        value = snapshot.get(name)
        if type(value) is not int or value < 0:
            raise ValueError(f'Invalid success snapshot count: {name}')
    episodes = snapshot['window_episodes']
    successes = snapshot['window_successes']
    if (snapshot['window_size'] != window_size or episodes > window_size
            or successes > episodes or snapshot['total_episodes_processed'] < episodes
            or snapshot['total_env_steps'] < snapshot['total_episodes_processed']):
        raise ValueError('Inconsistent success snapshot counts or window size')
    for name in ('success_rate', 'observed_unix'):
        value = snapshot.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'Invalid success snapshot number: {name}')
    expected = successes / episodes if episodes else 0.
    if not math.isclose(snapshot['success_rate'], expected, rel_tol=0., abs_tol=1e-12):
        raise ValueError('Success rate does not match exact episode counts')
    return {name: snapshot[name] for name in integer_fields + ('success_rate', 'observed_unix')}


def validate_rank_steps(results, previous_step):
    """Require exactly one completed update on every rank before publication."""
    steps = [result[6] for result in results]
    expected = previous_step + 1
    if not steps or any(step != expected for step in steps):
        raise RuntimeError(f'Training rank global steps must all advance to {expected}; got {steps}')
    return expected


class LegacyBaselineControl:
    def __init__(self, output_dir, max_train_seconds=0, stop_file=None, *,
                 clock=time.monotonic, wall_clock=time.time,
                 target_success_rate=0, target_success_window=1000):
        if not math.isfinite(max_train_seconds) or max_train_seconds < 0:
            raise ValueError('max_train_seconds must be finite and nonnegative')
        if not math.isfinite(target_success_rate) or not 0 <= target_success_rate <= 1:
            raise ValueError('target_success_rate must be finite and between 0 and 1')
        if type(target_success_window) is not int or target_success_window <= 0:
            raise ValueError('target_success_window must be a positive integer')
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if (self.output_dir / 'driver_complete.json').exists():
            raise FileExistsError('Refusing to overwrite an existing completion receipt')
        self.max_train_seconds = max_train_seconds
        self.stop_file = Path(stop_file).resolve() if stop_file else None
        self.clock = clock
        self.wall_clock = wall_clock
        self.started = clock()
        self.started_unix = wall_clock()
        self.training_started = None
        self.training_started_unix = None
        self.training_finished = None
        self.completed_updates = 0
        self.consumed_training_samples = 0
        self.reason = None
        self.target_success_rate = target_success_rate
        self.target_success_window = target_success_window
        self.success_snapshot = None
        self.current_success_snapshot = None
        self.milestones = {}
        self.metadata = None
        launch_unix = os.environ.get('WEAVER_LEGACY_STARTED_UNIX')
        self.launch_started_unix = float(launch_unix) if launch_unix is not None else None
        if self.launch_started_unix is not None and (not math.isfinite(self.launch_started_unix)
                                                     or self.launch_started_unix < 0):
            raise ValueError('WEAVER_LEGACY_STARTED_UNIX must be finite and nonnegative')
        self.initialization_seconds = None
        self.replay_fill_seconds = None
        self.identity = {'run_id': os.environ.get('ACCERL_RUN_ID'),
                         'git_commit': os.environ.get('WEAVER_LEGACY_GIT_COMMIT')}
        self.status('startup')

    def _write(self, name, value):
        path = self.output_dir / name
        temporary = path.with_suffix(path.suffix + '.tmp')
        with temporary.open('w', encoding='utf-8') as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def _times(self):
        now = self.clock()
        return {'elapsed_seconds': now - self.started,
                'training_seconds': 0 if self.training_started is None else
                    (now if self.training_finished is None else self.training_finished) - self.training_started,
                'training_started_unix': self.training_started_unix,
                'launch_elapsed_seconds': None if self.launch_started_unix is None else
                    self.wall_clock() - self.launch_started_unix,
                'initialization_seconds': self.initialization_seconds,
                'replay_fill_seconds': self.replay_fill_seconds}

    def _success_fields(self):
        return {'target_success_rate': self.target_success_rate,
                'target_success_window': self.target_success_window,
                'success_snapshot': self.success_snapshot, 'milestones': self.milestones,
                'current_success_snapshot': self.current_success_snapshot,
                'reason': self.reason}

    def _crossing_times(self, observed_unix):
        return {'elapsed_seconds': max(0., observed_unix - self.started_unix),
                'training_seconds': max(0., observed_unix - self.training_started_unix),
                'training_time_scope': 'before_training_loop' if observed_unix < self.training_started_unix
                                       else 'training_loop',
                'launch_elapsed_seconds': None if self.launch_started_unix is None else
                    observed_unix - self.launch_started_unix}

    def status(self, phase, **fields):
        result = {'schema_version': 1, **self.identity, 'phase': phase,
                  'updated_unix': self.wall_clock(), **self._times(),
                  'completed_updates': self.completed_updates,
                  'consumed_training_samples': self.consumed_training_samples,
                  'reason': self.reason,
                  **self._success_fields(),
                  **fields}
        self._write('driver_status.json', result)
        return result

    def record_metadata(self, args, placements, *, replay_fill_seconds=None):
        if self.initialization_seconds is None:
            self.initialization_seconds = self.clock() - self.started
        self.replay_fill_seconds = replay_fill_seconds
        self.metadata = {
            'schema_version': 1, **self.identity, 'args': args,
            'placements': placements,
            'global_batch_size': args['train_batch_size'] * args['accumulation_steps'] * args['num_trainer_gpus'],
            'checkpoint_kind': 'inference_weights_only', 'exact_resume': False,
            'initialization_seconds': self.initialization_seconds,
            'replay_fill_seconds': replay_fill_seconds,
            **self._success_fields(),
        }
        self._write('driver_metadata.json', self.metadata)

    def start_training(self):
        if self.training_started is not None:
            raise RuntimeError('Training clock already started')
        self.training_started = self.clock()
        self.training_started_unix = self.wall_clock()
        self.status('training', global_step=0)

    def after_update(self, global_step, *, train_seconds, publication_seconds, samples, success_snapshot=None):
        if self.training_started is None:
            raise RuntimeError('Training clock has not started')
        if self.reason is not None:
            return True
        if self.target_success_rate:
            snapshot = validate_success_snapshot(success_snapshot, self.target_success_window)
            raw_crossings = success_snapshot.get('crossings')
            if not isinstance(raw_crossings, dict) or any(key not in ('98', '99', 'target') for key in raw_crossings):
                raise ValueError('Success snapshot requires per-episode first crossing records')
            crossings = {}
            for key, raw in raw_crossings.items():
                event = validate_success_snapshot(raw, self.target_success_window)
                threshold = Fraction(str(self.target_success_rate)) if key == 'target' else Fraction(int(key), 100)
                if (event['window_episodes'] != self.target_success_window
                        or Fraction(event['window_successes'], self.target_success_window) < threshold
                        or event['observed_unix'] > snapshot['observed_unix']
                        or event['total_episodes_processed'] > snapshot['total_episodes_processed']
                        or event['total_env_steps'] > snapshot['total_env_steps']):
                    raise ValueError('Invalid first success crossing record')
                crossings[key] = event
            boundary = {'policy_step': global_step, 'boundary_policy_step': global_step,
                        'boundary_observed_unix': self.wall_clock(),
                        'policy_step_semantics': 'completed_update_boundary_observing_crossing'}
            self.current_success_snapshot = dict(snapshot, **boundary)
            for key in ('98', '99'):
                if key in crossings:
                    event = crossings[key]
                    self.milestones.setdefault(key, dict(event, **boundary, **self._crossing_times(event['observed_unix'])))
            target_key = '99' if self.target_success_rate == .99 else '98' if self.target_success_rate == .98 else 'target'
            target_crossing = crossings.get(target_key)
            self.success_snapshot = (dict(target_crossing, **boundary) if target_crossing is not None
                                     else dict(self.current_success_snapshot))
        self.completed_updates += 1
        self.consumed_training_samples += samples
        record = {'schema_version': 1, **self.identity, 'global_step': global_step,
                  'completed_updates': self.completed_updates,
                  'update_seconds': train_seconds,
                  'publication_seconds': publication_seconds,
                  'total_seconds': train_seconds + publication_seconds,
                  'samples_per_update': samples,
                  'consumed_training_samples': self.consumed_training_samples,
                  **self._success_fields(),
                  **self._times()}
        if self.stop_file is not None and self.stop_file.exists():
            self.reason = 'stop_file'
        elif self.target_success_rate and target_crossing is not None:
            self.reason = 'success_target_reached'
        elif self.max_train_seconds and record['training_seconds'] >= self.max_train_seconds:
            self.reason = 'deadline'
        record['reason'] = self.reason
        with (self.output_dir / 'updates.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, allow_nan=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        if self.metadata is not None and self.target_success_rate:
            self.metadata.update(self._success_fields())
            self._write('driver_metadata.json', self.metadata)
        self.status('update_complete', global_step=global_step, reason=self.reason)
        return self.reason is not None

    def complete(self, global_step, save):
        """Write success only after the caller's synchronous weights save returns."""
        if self.reason == 'success_target_reached' and global_step != self.success_snapshot['policy_step']:
            raise RuntimeError('Final checkpoint step must match the first success-target boundary')
        self.reason = self.reason or 'train_iters'
        if self.metadata is not None and self.target_success_rate:
            self.metadata.update(self._success_fields())
            self._write('driver_metadata.json', self.metadata)
        self.training_finished = self.clock()
        self.status('final_save', global_step=global_step, reason=self.reason or 'train_iters')
        checkpoint_dir = Path(save()).resolve()
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f'Final checkpoint directory missing: {checkpoint_dir}')
        result = {'schema_version': 1, **self.identity, 'status': 'completed',
                  'global_step': global_step, 'reason': self.reason or 'train_iters',
                  'checkpoint_dir': str(checkpoint_dir),
                  'checkpoint_kind': 'inference_weights_only', 'exact_resume': False,
                  'completed_updates': self.completed_updates,
                  'consumed_training_samples': self.consumed_training_samples,
                  **self._success_fields(),
                  'final_save_seconds': self.clock() - self.training_finished,
                  'completed_unix': self.wall_clock(), **self._times()}
        self._write('driver_complete.json', result)
        self.status('completed', global_step=global_step, checkpoint_dir=str(checkpoint_dir))
        return result


def actor_placement(torch, ray, *, role, index):
    """Query an existing actor's allocation; never launch GPU helper services."""
    device = torch.cuda.current_device()
    return {'role': role, 'index': index, 'pid': os.getpid(),
            'ray_gpu_ids': ray.get_gpu_ids(),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'torch_device': f'cuda:{device}', 'torch_device_name': torch.cuda.get_device_name(device)}
