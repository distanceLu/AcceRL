"""CPU-only checks; no CUDA/profiler process is started."""
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parents[1] / 'scheduler_ddp'))
import profile_plan as plan
from nsight_capture import NsightCaptureWindows
from test_manual_controller_cpu import load


class Writer:
    def __init__(self): self.events = []
    def event(self, kind, **fields): self.events.append((kind, fields))
    def flush(self): pass


class Timer:
    def __init__(self, seconds, callback):
        self.seconds, self.callback, self.cancelled = seconds, callback, False
    def start(self): pass
    def cancel(self): self.cancelled = True


class ProfilePlanTests(unittest.TestCase):
    def test_one_fixed_process_tree_and_no_caps(self):
        cmd = plan.profile_command()
        self.assertIn('--trace=cuda-sw,nvtx,nccl', cmd)
        self.assertIn('--nccl-trace=api', cmd)
        self.assertIn('--kill=none', cmd)
        self.assertIn('--capture-range-end=repeat-shutdown:2:async', cmd)
        self.assertIn('--wait=primary',cmd)
        self.assertIn('--gpu-metrics-devices=none', cmd)
        self.assertNotIn('system-wide', ' '.join(cmd))
        self.assertNotIn('sudo', cmd)
        self.assertEqual(cmd[-1],str(plan.MASTER/'formal'))
        self.assertEqual(cmd[-3],'manual')
        self.assertIn(str(plan.MASTER/'source_v1/runner.py'),cmd)
        self.assertIn('--output='+str(plan.MASTER/'nsight/contention_windows'),cmd)
        self.assertFalse(any(value.startswith(('--duration','--delay')) for value in cmd))
        self.assertNotIn('MPS_ACTIVE_THREAD', str(plan.training_profile_environment()))

    def test_paths_cannot_escape_new_master(self):
        with self.assertRaises(ValueError):plan.profile_command(source=Path('/tmp/wrong'))
        with self.assertRaises(ValueError):plan.profile_command(data=Path('/tmp/wrong'))
        with self.assertRaises(ValueError):plan.profile_command(prefix=Path('/tmp/wrong'))
        with self.assertRaises(ValueError):plan.export_command('/tmp/wrong','/tmp/wrong.sqlite')

    def test_counter_denial_is_unavailable_even_exit_zero(self):
        result=load('runner.py','measurement_configuration',{})()['hardware_metrics']
        self.assertEqual(result['status'],'not_collected_permission_restricted')
        self.assertIsNone(result['sm_active'])
        self.assertIsNone(result['occupancy'])
        self.assertFalse(result['enabled'])
        self.assertIn('--gpu-metrics-devices=none',plan.profile_command())

    def test_unknown_probe_cannot_silently_claim_metrics(self):
        self.assertFalse(plan.reporting_contract()['CUPTI_loss_free_claimed'])
        self.assertFalse(plan.reporting_contract()['full_run_GPU_coverage'])
        self.assertIn('not collected',plan.reporting_contract()['hardware_counters'])

    def test_timer_windows_exactly_once_at_fourteen_and_fifteen(self):
        writer, opened, closed, markers = Writer(), [], [], []
        capture = NsightCaptureWindows(writer, steps=plan.CAPTURE_STEPS, seconds=30, mode='timer',
            begin=lambda label: opened.append(label) or len(opened), end=closed.append,
            timer_factory=Timer,marker=markers.append)
        for step in range(1, 14): capture.before_epoch(step)
        self.assertEqual(opened, [])
        for step in (14,15):
            capture.before_epoch(step)
            self.assertEqual(capture.timer.seconds,30)
            capture.after_epoch(step)
            self.assertIsNotNone(capture.active)
            capture.timer.callback()
        for step in (14,15,20,25,30001):capture.before_epoch(step)
        capture.close()
        self.assertEqual(opened,['PROFILE_WINDOW','PROFILE_WINDOW'])
        self.assertEqual(closed,[1,2])
        self.assertEqual(markers,['ACCERL_CONTENTION|policy_only|14','ACCERL_CONTENTION|joint_three_trainers|15'])
        ended = [v['details'] for k, v in writer.events if k == 'nsight_capture_ended']
        self.assertEqual([value['policy_step'] for value in ended],[14,15])
        self.assertTrue(all(value['reason']=='duration_limit' and not value['coverage_complete'] for value in ended))

    def test_env_and_reporting_contract(self):
        env = plan.training_profile_environment()
        self.assertEqual(env['ACCERL_NSIGHT_STEPS'], '14,15')
        self.assertEqual(env['ACCERL_NSIGHT_SECONDS'], '30')
        self.assertEqual(env['ACCERL_NSIGHT_MODE'], 'timer')
        self.assertTrue(plan.reporting_contract()['manual_training_continues_after_measurement'])
        self.assertFalse(plan.reporting_contract()['measurement_failure_requests_training_stop'])
        self.assertFalse(plan.reporting_contract()['automatic_recapture_or_retry'])


if __name__ == '__main__': unittest.main()
