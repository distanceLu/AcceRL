"""CPU fakes: IPC validation disables every profiler path and preserves manual stop."""
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

from deadline_guard import guard_decision
import nsight_capture
from nsight_capture import NsightCaptureWindows
from training_stop_policy import TrainingStopPolicy
from test_manual_controller_cpu import env,RUN


class ManualMeasurementIndependenceCPU(unittest.TestCase):
    def test_disabled_service_nvtx_never_calls_torch_for_trusted_or_untrusted_environment(self):
        for verified in (True,False):
            with self.subTest(verified=verified):
                environment=env({'ACCERL_NSIGHT_NVTX':'1','ACCERL_NSIGHT_STEPS':'14,15',
                    'NSYS_NVTX_PROFILER_REGISTER_ONLY':'0','CUDA_INJECTION64_PATH':'old',
                    'NVTX_INJECTION64_PATH':'old'},verified=verified)
                start,end=Mock(side_effect=AssertionError('disabled NVTX')),Mock(side_effect=AssertionError('disabled NVTX'))
                fake=SimpleNamespace(cuda=SimpleNamespace(nvtx=SimpleNamespace(range_start=start,range_end=end)))
                with patch.dict(sys.modules,{'torch':fake}),patch.dict(nsight_capture.os.environ,environment,clear=True):
                    self.assertIsNone(nsight_capture.nvtx_begin('service_span'))
                    self.assertIs(nsight_capture.nvtx_end(None),True)
                start.assert_not_called();end.assert_not_called()

    def test_inherited_collector_never_creates_stop_or_triggers_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            data=Path(temporary);(data/'training').mkdir()
            for verified in (True,False):
                with self.subTest(verified=verified):
                    state={'run_id':RUN,'started_ns':1,'state':'running','heartbeat_ns':1_000_000_000}
                    environment=env({'ACCERL_SHORT_TEST_DEADLINE_UNIX':'2','SOFT_STOP_AT':'1',
                        'MEASURE_STOP_AFTER_EPOCHS':'15','ACCERL_RUN_UNTIL_DEADLINE':'1',
                        'ACCERL_CONTENTION_COLLECTOR_TOKEN':'old'},verified=verified,state=state)
                    self.assertFalse((data/'training/stop.flag').exists())
                    self.assertEqual(state['state'],'running')
                    self.assertFalse(state['measurement_injection_verified'])
                    self.assertNotIn('error',state)
                    self.assertIsNone(guard_decision(state,now=2,supervisor_alive=True)['stop_reason'])
                    policy=TrainingStopPolicy.from_environment(environment)
                    self.assertIsNone(policy.deadline_unix_seconds)
                    self.assertTrue(policy.should_continue(
                        global_step=50000,train_iters=30000,target_reached=True,requested_stop=False,
                        completed_epochs=50000,measure_stop_epochs=15,now=9999999999))

    def test_no_window_timer_marker_or_writer_runs_even_at_old_selected_epochs(self):
        for verified in (True,False):
            with self.subTest(verified=verified),tempfile.TemporaryDirectory() as temporary:
                data=Path(temporary);(data/'training').mkdir()
                environment=env({'ACCERL_NSIGHT_STEPS':'14,15','ACCERL_NSIGHT_NVTX':'1',
                    'LD_PRELOAD':'/tools/nsight/libToolsInjection64.so'},verified=verified)
                callbacks=[Mock(side_effect=AssertionError('Profiler path must not execute')) for _ in range(6)]
                begin,end,marker,timer,event,flush=callbacks
                with patch.dict(os.environ,environment,clear=True):
                    capture=NsightCaptureWindows(SimpleNamespace(event=event,flush=flush),
                        begin=begin,end=end,marker=marker,timer_factory=timer)
                    for step in (1,14,15,25,30001,50000):
                        capture.before_epoch(step)
                        for model in ('policy','ctrl','reward'):
                            capture.before_stage(step,model)
                            capture.after_stage(step,model)
                        capture.after_epoch(step)
                    capture.close()
                self.assertEqual(capture.steps,())
                self.assertFalse(capture.measurement_errors)
                self.assertIsNone(capture.active);self.assertIsNone(capture.timer)
                for callback in callbacks:callback.assert_not_called()
                self.assertFalse((data/'training/stop.flag').exists())

    def test_collector_outcomes_do_not_change_manual_guard_or_user_window(self):
        base={'state':'running','heartbeat_ns':1_000_000_000,'stop_requested_ns':1_000_000_000}
        expected=guard_decision(base,now=2,supervisor_alive=True)
        self.assertEqual(expected['cleanup_at'],901)
        for collector in ('running','completed','failed','missing','CPU_export_timeout'):
            state=dict(base,measurement_status=collector,measurement_configuration_error='capture failed')
            self.assertEqual(guard_decision(state,now=2,supervisor_alive=True),expected)
            later=guard_decision(state,now=901,supervisor_alive=True)
            self.assertTrue(later['cleanup'])
            self.assertEqual(later['cleanup_at'],901)


if __name__=='__main__':unittest.main()
