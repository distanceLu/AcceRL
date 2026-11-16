"""Budget/stop decisions happen before collective-producing epoch RPCs."""
from types import SimpleNamespace
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from model_free_runtime import ModelFreeRuntime
from model_free_control import ModelFreeControlConfig


class Method:
    def __init__(self, fn): self.remote = fn


class RuntimeTests(unittest.TestCase):
    def runtime(self, *, exhausted=False, ready=True):
        self.events, self.admissions = [], []
        scheduler = dict(max_env_steps=100, consumed_env_steps=100 if exhausted else 10,
            available_env_steps=0 if exhausted else 90, reserved_env_steps=0,
            episodes_inflight=0, conservative_unknown_env_steps=0, episodes_cancelled=0)
        status = dict(budget=dict(active_real=0, failure=None, paused=False, stopped=False,
                                  reset_steps=10, budget_exhausted=exhausted),
                      initial_states=scheduler, initial_state_pool=dict(reset_steps=10))
        # Coordinator supplies its canonical normalized accounting receipt.
        status['collection_budget'] = dict(initialized=True, **scheduler, active_real=0,
            reset_steps=10, budget_exhausted=exhausted, paused=False, failure=None, stopped=False)
        collector = SimpleNamespace(pipeline_status=Method(lambda: status))
        def admit(count):
            self.admissions.append(count)
            return {'ready': ready}
        replay = SimpleNamespace(admit_model_free_batch=Method(admit))
        ray = SimpleNamespace(get=lambda value, **kw: value, wait=lambda refs, **kw: ([], refs))
        runtime = ModelFreeRuntime(ray=ray, collector=collector, replay=replay,
            inference_pool=[], worker_refs=[], cfg=None, benchmark='libero_spatial', dtype=None,
            render_bindings=[], environment={}, config=ModelFreeControlConfig(interval_env_steps=0),
            writer=SimpleNamespace(add_scalar=lambda *a: None, flush=lambda: None),
            experiment=SimpleNamespace(event=lambda *a, **k: self.events.append((a, k)), flush=lambda: None),
            requested_stop=lambda: False, global_batch_size=576)
        return runtime

    def test_exhaustion_never_admits_an_epoch(self):
        runtime = self.runtime(exhausted=True)
        self.assertEqual(runtime.before_update(7), 'real_budget_exhausted')
        self.assertEqual(self.admissions, [])

    def test_manual_stop_never_admits_an_epoch(self):
        runtime = self.runtime()
        runtime.requested_stop = lambda: True
        self.assertEqual(runtime.before_update(7), 'requested_stop')
        self.assertEqual(self.admissions, [])

    def test_complete_reserved_batch_allows_next_epoch(self):
        runtime = self.runtime()
        self.assertIsNone(runtime.before_update(7))
        self.assertEqual(self.admissions, [576])

    def test_stop_is_polled_while_replay_has_no_cohort(self):
        runtime = self.runtime(ready=False)
        runtime.sleep = lambda seconds: None
        runtime.requested_stop = lambda: bool(self.admissions)
        self.assertEqual(runtime.before_update(7), 'requested_stop')
        self.assertEqual(self.admissions, [576])

    def test_validation_time_does_not_spend_replay_supply_timeout(self):
        runtime = self.runtime(ready=False)
        clock = [0.]
        runtime.sleep = lambda seconds: None
        runtime.validate = lambda step: clock.__setitem__(0, clock[0] + 4000.)
        runtime.requested_stop = lambda: bool(self.admissions)
        with patch('model_free_runtime.time.monotonic', side_effect=lambda: clock[0]):
            self.assertEqual(runtime.before_update(7), 'requested_stop')

    def validation_fixture(self, directory, result=None):
        from model_free_control import ModelFreeControl
        runtime = self.runtime()
        runtime.control = ModelFreeControl(ModelFreeControlConfig())
        runtime.environment = {'ACCERL_DATA_DIR': directory, 'ACCERL_RUN_ID': 'test',
                               'MUJOCO_GL': 'egl'}
        self.pauses, self.limits, self.killed = [], [], []
        runtime.collector.set_paused = Method(lambda value: self.pauses.append(value))
        runtime.collector.initial_state_checkpoint = Method(lambda: {'pool': None})
        runtime.inference_pool = [SimpleNamespace(set_training_batch_limit=Method(
            lambda value: self.limits.append(value)))]
        result = result or dict(complete=True, num_cases=50, completed_cases=50,
            policy_version=0, success_count=48, success_rate=.96, protocol_sha256='a'*64,
            initial_state_pool=dict(count=50, shape=[50,45], sha256='b'*64))
        actor = SimpleNamespace(run=Method(lambda *args, **kw: result), stop=Method(lambda: None))
        factory = SimpleNamespace(options=lambda **kw: SimpleNamespace(remote=lambda *args: actor))
        runtime.ray.remote = lambda cls: factory
        runtime.ray.wait = lambda refs, **kw: (refs[:1], refs[1:])
        runtime.ray.kill = lambda owner, **kw: self.killed.append(owner)
        return runtime

    def test_finite_validation_restores_batch_cap_and_records_one_complete_score(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                'environment_rendering.render_worker_options', return_value={}):
            runtime = self.validation_fixture(directory)
            runtime.validate(0)
            self.assertEqual(self.pauses, [True, False])
            self.assertEqual(self.limits, [1, 2])
            self.assertEqual(len(self.killed), 1)
            self.assertEqual(runtime.control.summary()['completed_validations'], 1)
            record = json.loads((Path(directory)/'fixed_validation/P0.json').read_text())
            self.assertEqual(record['success_count'], 48)
            activity = json.loads((Path(directory)/'fixed_validation_activity.json').read_text())
            self.assertFalse(activity['active'])
            runtime.validate(0, force=True)
            self.assertEqual(len(self.killed), 1)  # no duplicate patience observation

    def test_partial_validation_is_fatal_and_does_not_spend_patience(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                'environment_rendering.render_worker_options', return_value={}):
            runtime = self.validation_fixture(directory, dict(complete=False, completed_cases=49))
            with self.assertRaisesRegex(RuntimeError, 'Incomplete'):
                runtime.validate(0)
            self.assertEqual(runtime.control.summary()['completed_validations'], 0)
            self.assertEqual(self.pauses, [True, True])  # failure must not restart producers
            self.assertEqual(self.limits, [1, 2])
            self.assertEqual(len(self.killed), 1)
            self.assertFalse((Path(directory)/'fixed_validation/latest.json').exists())

    def test_final_validation_keeps_collection_paused(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                'environment_rendering.render_worker_options', return_value={}):
            runtime = self.validation_fixture(directory)
            runtime.validate(0, force=True)
            self.assertEqual(self.pauses, [True, True])


if __name__ == '__main__': unittest.main()
