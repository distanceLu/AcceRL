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


def validate_rank_steps(results, previous_step):
    """Require exactly one completed update on every rank before publication."""
    steps = [result[6] for result in results]
    expected = previous_step + 1
    if not steps or any(step != expected for step in steps):
        raise RuntimeError(f'Training rank global steps must all advance to {expected}; got {steps}')
    return expected


class LegacyBaselineControl:
    def __init__(self, output_dir, max_train_seconds=0, stop_file=None, *,
                 clock=time.monotonic, wall_clock=time.time):
        if not math.isfinite(max_train_seconds) or max_train_seconds < 0:
            raise ValueError('max_train_seconds must be finite and nonnegative')
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if (self.output_dir / 'driver_complete.json').exists():
            raise FileExistsError('Refusing to overwrite an existing completion receipt')
        self.max_train_seconds = max_train_seconds
        self.stop_file = Path(stop_file).resolve() if stop_file else None
        self.clock = clock
        self.wall_clock = wall_clock
        self.started = clock()
        self.training_started = None
        self.training_started_unix = None
        self.training_finished = None
        self.completed_updates = 0
        self.consumed_training_samples = 0
        self.reason = None
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
                'initialization_seconds': self.initialization_seconds,
                'replay_fill_seconds': self.replay_fill_seconds}

    def status(self, phase, **fields):
        result = {'schema_version': 1, **self.identity, 'phase': phase,
                  'updated_unix': self.wall_clock(), **self._times(),
                  'completed_updates': self.completed_updates,
                  'consumed_training_samples': self.consumed_training_samples,
                  **fields}
        self._write('driver_status.json', result)
        return result

    def record_metadata(self, args, placements, *, replay_fill_seconds=None):
        if self.initialization_seconds is None:
            self.initialization_seconds = self.clock() - self.started
        self.replay_fill_seconds = replay_fill_seconds
        self._write('driver_metadata.json', {
            'schema_version': 1, **self.identity, 'args': args,
            'placements': placements,
            'global_batch_size': args['train_batch_size'] * args['accumulation_steps'] * args['num_trainer_gpus'],
            'checkpoint_kind': 'inference_weights_only', 'exact_resume': False,
            'initialization_seconds': self.initialization_seconds,
            'replay_fill_seconds': replay_fill_seconds,
        })

    def start_training(self):
        if self.training_started is not None:
            raise RuntimeError('Training clock already started')
        self.training_started = self.clock()
        self.training_started_unix = self.wall_clock()
        self.status('training', global_step=0)

    def after_update(self, global_step, *, train_seconds, publication_seconds, samples):
        if self.training_started is None:
            raise RuntimeError('Training clock has not started')
        self.completed_updates += 1
        self.consumed_training_samples += samples
        record = {'schema_version': 1, **self.identity, 'global_step': global_step,
                  'completed_updates': self.completed_updates,
                  'update_seconds': train_seconds,
                  'publication_seconds': publication_seconds,
                  'total_seconds': train_seconds + publication_seconds,
                  'samples_per_update': samples,
                  'consumed_training_samples': self.consumed_training_samples,
                  **self._times()}
        with (self.output_dir / 'updates.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, allow_nan=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        if self.stop_file is not None and self.stop_file.exists():
            self.reason = 'stop_file'
        elif self.max_train_seconds and record['training_seconds'] >= self.max_train_seconds:
            self.reason = 'deadline'
        self.status('update_complete', global_step=global_step, reason=self.reason)
        return self.reason is not None

    def complete(self, global_step, save):
        """Write success only after the caller's synchronous weights save returns."""
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
