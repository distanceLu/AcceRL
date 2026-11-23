"""CPU checks for the single-host P189 action admission pulse."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from bench_action_gate import RolloutPulseGate, write_rollout_pulse
from capture_hooks import profile_rpc


class Recorder:
    def __init__(self):
        self.rows = []

    def event(self, kind, **fields):
        self.rows.append((kind, fields))


class GateTest(unittest.TestCase):
    def test_only_explicit_benchmark_plan_enables_gate(self):
        self.assertIsNone(RolloutPulseGate.from_environment({}))
        with tempfile.TemporaryDirectory() as d:
            plan = Path(d) / 'plan.json'
            env = {'ACCERL_JIT_BENCH_PLAN': str(plan), 'ACCERL_JIT_BENCH_OUTPUT': d}
            plan.write_text(json.dumps({'comparison': 'episode_sampling'}))
            self.assertIsNone(RolloutPulseGate.from_environment(env))
            plan.write_text(json.dumps({'comparison': 'admission_pulse'}))
            self.assertIsInstance(RolloutPulseGate.from_environment(env), RolloutPulseGate)

    def test_all_new_requests_share_one_expiry(self):
        with tempfile.TemporaryDirectory() as d:
            clock = [1_000_000_000]
            def sleep(sec):
                clock[0] += round(sec * 1e9)
            with mock.patch('bench_action_gate.time.monotonic_ns', side_effect=lambda: clock[0]), \
                 mock.patch('bench_action_gate.time.sleep', side_effect=sleep):
                gate = RolloutPulseGate(Path(d))
                self.assertEqual(gate.wait(), (None, 0))
                pulse = write_rollout_pulse(d, 'block0-rollout', 5)
                self.assertEqual(pulse['until_ns']-pulse['issued_ns'], 5_000_000_000)
                clock[0] += 2_000_000_000
                self.assertEqual(gate.wait(), ('block0-rollout', 3_000_000_000))
                self.assertEqual(gate.wait(), (None, 0))

    def test_only_real_policy_request_waits_and_has_end_to_end_telemetry(self):
        recorder = Recorder()
        gate = mock.Mock()
        gate.wait.return_value = ('pulse', 50)
        owner = SimpleNamespace(bench_action_gate=gate, telemetry=recorder, wid=3)
        actor = SimpleNamespace(request=SimpleNamespace(remote=lambda *args, **kwargs: 'action'))
        with mock.patch.dict(sys.modules, ray=SimpleNamespace(get=lambda x: x)):
            for workflow in ('real_rollout', 'evaluation'):
                self.assertEqual(profile_rpc(owner, 'policy_infer', actor, 'request', 4,
                                             _capture_context={'workflow': workflow}), 'action')
        gate.wait.assert_called_once()
        ready, = [f for k, f in recorder.rows if k == 'bench_action_ready']
        done, = [f for k, f in recorder.rows if k == 'bench_action_request']
        self.assertEqual(done['request_id'], ready['request_id'])
        self.assertLessEqual(ready['ts_ns'], done['submitted_ns'])
        self.assertLessEqual(done['submitted_ns'], done['end_ns'])
        self.assertEqual(done['duration_ns'], done['end_ns'] - ready['ts_ns'])
        self.assertEqual(done['details']['gate_wait_ns'], 50)
        self.assertEqual(done['details']['arm_id'], 'pulse')

    def test_pulse_during_rpc_leaves_inflight_request_alone(self):
        recorder = Recorder()
        with tempfile.TemporaryDirectory() as directory:
            owner = SimpleNamespace(wid=0, telemetry=recorder,
                                    bench_action_gate=RolloutPulseGate(directory))
            actor = SimpleNamespace(request=SimpleNamespace(remote=lambda **kw: 'inflight'))
            clock = [0]
            def reply(ref):
                write_rollout_pulse(directory, 'arm', 5)
                return 'reply'
            with mock.patch('bench_action_gate.time.monotonic_ns', side_effect=lambda: clock[0]), \
                 mock.patch('bench_action_gate.time.sleep') as sleep, \
                 mock.patch.dict(sys.modules, ray=SimpleNamespace(get=reply)):
                profile_rpc(owner, 'policy_infer', actor, 'request', _capture_context={'workflow': 'real_rollout'})
                sleep.assert_not_called()
            completed, = [f for k, f in recorder.rows if k == 'bench_action_request']
            self.assertIsNone(completed['details']['arm_id'])


if __name__ == '__main__':
    unittest.main()
