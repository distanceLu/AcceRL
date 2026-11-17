"""Driver-only admission and finite validation; no model state is reconstructed."""
from pathlib import Path
import time

from model_free_control import ModelFreeControl, collection_budget


class ValidationInterrupted(Exception):
    pass


class ModelFreeRuntime:
    def __init__(self, *, ray, collector, replay, inference_pool, worker_refs, cfg,
                 benchmark, dtype, render_bindings, environment, config, writer,
                 experiment, requested_stop, global_batch_size):
        self.ray, self.collector, self.replay = ray, collector, replay
        self.inference_pool, self.worker_refs = inference_pool, worker_refs
        self.cfg, self.benchmark, self.dtype = cfg, benchmark, dtype
        self.render_bindings, self.environment = render_bindings, environment
        self.control = ModelFreeControl(config)
        self.writer, self.experiment = writer, experiment
        self.requested_stop, self.global_batch_size = requested_stop, global_batch_size
        self.sleep = time.sleep
        self.last_budget, self.last_result = None, None

    def status(self):
        status = self.ray.get(self.collector.pipeline_status.remote(), timeout=15)
        self.last_budget = collection_budget(status)
        return status

    def check_workers(self):
        if self.worker_refs:
            ready, _ = self.ray.wait(self.worker_refs, timeout=0, num_returns=1)
            if ready:
                self.ray.get(ready[0])
                raise RuntimeError('A real collection worker exited at the pre-update boundary')

    def completion_fields(self):
        self.status()
        return dict(model_free=True, budget_status=self.last_budget,
                    fixed_validation=self.control.summary())

    def before_update(self, policy_step):
        """Only return None after one full cohort is atomically reserved."""
        started = time.monotonic()
        while True:
            if self.requested_stop():
                return 'requested_stop'
            self.check_workers()
            self.status()
            if self.control.target_reached:
                return 'fixed_validation_target_reached'
            if self.last_budget['budget_exhausted']:
                return 'real_budget_exhausted'
            validation_started = time.monotonic()
            try:
                self.validate(policy_step)
            except ValidationInterrupted:
                return 'requested_stop'
            # Collection is deliberately paused for validation; its independent
            # drain/evaluation deadlines must not consume the Replay supply grace.
            started += time.monotonic() - validation_started
            if self.requested_stop():
                return 'requested_stop'
            if self.control.target_reached:
                return 'fixed_validation_target_reached'
            if self.control.plateau_reached:
                return 'fixed_validation_plateau'
            # Validation may have drained the last budget leases.
            self.status()
            if self.last_budget['budget_exhausted']:
                return 'real_budget_exhausted'
            admission = self.ray.get(
                self.replay.admit_model_free_batch.remote(self.global_batch_size), timeout=30)
            if admission['ready']:
                self.experiment.event('model_free_update_admitted', details={
                    'policy_step': policy_step + 1, **admission})
                return None
            if time.monotonic() - started > 3600:
                raise TimeoutError('Model-free Replay did not supply a complete eligible cohort in 3600 seconds')
            self.sleep(.25)

    def validate(self, policy_step, *, force=False):
        self.status()
        consumed = self.last_budget['consumed_env_steps']
        if not self.control.should_validate(consumed, policy_step, force=force):
            return self.last_result
        if self.requested_stop():
            raise ValidationInterrupted()
        from environment_rendering import render_worker_options, render_backend
        from frozen_policy_eval import atomic_json
        from periodic_fixed_eval import PeriodicFixedEvalWorker
        if render_backend(self.environment) != 'egl':
            raise ValueError('Periodic model-free fixed validation requires EGL')
        data_dir = self.environment.get('ACCERL_DATA_DIR')
        if not data_dir:
            raise ValueError('Periodic fixed validation requires a managed data directory')
        directory = Path(data_dir)
        output = directory / 'fixed_validation'
        output.mkdir(parents=True, exist_ok=True)
        started = time.time()
        activity = dict(run_id=self.environment.get('ACCERL_RUN_ID'), active=True,
            policy_step=policy_step, started_unix_seconds=started,
            deadline_unix_seconds=started + 5400, phase='drain')
        atomic_json(directory / 'fixed_validation_activity.json', activity)
        actor = None
        self.experiment.event('fixed_validation_started', details=activity)
        self.experiment.flush()
        self.ray.get(self.collector.set_paused.remote(True), timeout=15)
        completed = False
        try:
            drain_deadline = time.monotonic() + 1800
            while True:
                if self.requested_stop():
                    raise ValidationInterrupted()
                self.check_workers()
                self.status()
                if all(self.last_budget[key] == 0 for key in
                       ('active_real', 'episodes_inflight', 'reserved_env_steps')):
                    break
                if time.monotonic() >= drain_deadline:
                    raise TimeoutError('Existing real episodes did not drain for fixed validation')
                self.sleep(.25)
            # No Real requests, no online Eval, and no update/critic refresh can
            # submit inference while this driver owns the validation boundary.
            self.ray.get([infer.set_training_batch_limit.remote(1)
                          for infer in self.inference_pool], timeout=15)
            options = render_worker_options(self.render_bindings, 0, 'eval-999', environ=self.environment)
            actor = self.ray.remote(PeriodicFixedEvalWorker).options(max_concurrency=2, **options).remote(
                self.inference_pool[0], self.cfg, self.benchmark, self.dtype)
            activity['phase'] = 'evaluate'
            atomic_json(directory / 'fixed_validation_activity.json', activity)
            pool = self.ray.get(self.collector.initial_state_checkpoint.remote(), timeout=15).get('pool')
            evaluation = actor.run.remote(policy_step, timeout_seconds=3600, expected_pool=pool)
            deadline = time.monotonic() + 3600
            while True:
                if self.requested_stop():
                    self.ray.get(actor.stop.remote(), timeout=15)
                    raise ValidationInterrupted()
                self.check_workers()
                ready, _ = self.ray.wait([evaluation], timeout=1, num_returns=1)
                if ready:
                    result = self.ray.get(evaluation)
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError('Periodic fixed validation exceeded 3600 seconds')
            if (result.get('complete') is not True or result.get('completed_cases') != 50
                    or result.get('num_cases') != 50 or result.get('policy_version') != policy_step):
                raise RuntimeError('Incomplete or wrong-version fixed validation cannot drive early stopping')
            if self.last_result is not None and result['protocol_sha256'] != self.last_result['protocol_sha256']:
                raise RuntimeError('Fixed validation case protocol changed between checkpoints')
            if self.last_result is not None and any(
                    result['initial_state_pool'][key] != self.last_result['initial_state_pool'][key]
                    for key in ('count', 'shape', 'sha256')):
                raise RuntimeError('Fixed validation initial-state asset changed between checkpoints')
            self.ray.get(self.collector.initialize_initial_state_pool.remote(
                result['initial_state_pool']), timeout=15)
            self.status()
            summary = self.control.observe(result['success_count'], policy_step=policy_step,
                consumed_env_steps=self.last_budget['consumed_env_steps'], cases=50)
            self.last_result = result
            record = dict(result, control=summary, budget_status=self.last_budget,
                          validation_seconds=time.time() - started)
            atomic_json(output / ('P%d.json' % policy_step), record)
            atomic_json(output / 'latest.json', record)
            self.experiment.event('fixed_validation_completed', details=record)
            for name, value in (('Success_Rate', result['success_rate']),
                                ('Success_Count', result['success_count']),
                                ('Budget_Env_Steps', self.last_budget['consumed_env_steps']),
                                ('Duration_Seconds', record['validation_seconds'])):
                self.writer.add_scalar('FixedValidation/' + name, value, policy_step)
            self.writer.flush()
            self.experiment.flush()
            completed = True
            return result
        finally:
            # This actor owns only a temporary simulator. No trainer is killed,
            # no optimizer/Replay is reloaded, and no active collective exists.
            try:
                if actor is not None:
                    self.ray.kill(actor, no_restart=True)
                self.ray.get([infer.set_training_batch_limit.remote(2)
                              for infer in self.inference_pool], timeout=15)
            finally:
                resume = (completed and not force and not self.control.target_reached
                          and not self.control.plateau_reached and not self.requested_stop())
                self.ray.get(self.collector.set_paused.remote(not resume), timeout=15)
                activity.update(active=False, finished_unix_seconds=time.time())
                atomic_json(directory / 'fixed_validation_activity.json', activity)

    def finish(self, policy_step):
        if not self.requested_stop():
            try:
                self.validate(policy_step, force=True)
            except ValidationInterrupted:
                pass
        return self.completion_fields()
