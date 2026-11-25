"""Single-host control, clock calibration and real action join for the critical JIT trial."""
import json
from contextlib import ExitStack
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bench_action_gate import RolloutPulseGate
from capture_hooks import request_begin, batch_begin, batch_end, stage_begin, stage_end, profile_rpc


class CriticalProbeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        plan = self.out / 'plan.json'
        plan.write_text(json.dumps({'comparison': 'critical_jit'}))
        self.env = {'ACCERL_JIT_BENCH_PLAN': str(plan), 'ACCERL_JIT_BENCH_OUTPUT': str(self.out)}

    def test_matching_new_action_expires_and_earlier_actions_other_cards_and_sham_pass(self):
        from critical_probe import configure_probe, GateWindow
        mono = [1_000_000_000]
        def advance(seconds):
            mono[0] += round(seconds * 1e9)
        clock = ExitStack()
        self.addCleanup(clock.close)
        clock.enter_context(patch('critical_probe.time.monotonic_ns', side_effect=lambda: mono[0]))
        clock.enter_context(patch('critical_probe.time.sleep', side_effect=advance))
        spec = configure_probe(self.out, step=1, arm_id='early', condition='P-other',
                               target_rank=2, gate_rank=1, target_bundle=3, ttl_s=.012)
        a = RolloutPulseGate.from_environment(self.env, worker_id=5)
        b = RolloutPulseGate.from_environment(self.env, worker_id=2)
        self.addCleanup(a.actions.close)
        self.addCleanup(b.actions.close)
        gate = GateWindow(self.out, spec)
        record = gate.open()
        self.assertEqual(b.wait(time.time_ns()), (None, 0))
        self.assertEqual(a.wait(record['gate_issued_ns'] - 1), (None, 0))
        arm, waited = a.wait(time.time_ns())
        self.assertEqual(arm, 'early')
        self.assertGreater(waited, 0)
        self.assertEqual(a.wait(time.time_ns()), (None, 0))
        self.assertGreaterEqual(gate.close(), record['gate_issued_ns'])
        spec = configure_probe(self.out, step=2, arm_id='sham', condition='N',
                               target_rank=1, target_bundle=3, ttl_s=1.)
        gate = GateWindow(self.out, spec)
        gate.open()
        self.assertEqual(a.wait(time.time_ns()), ('sham', 0))
        gate.close()
        self.assertEqual(a.wait(time.time_ns()), (None, 0))

    def test_clock_is_bracketed_by_existing_synchronized_boundaries(self):
        from critical_probe import align_marker
        # The local event is 0.5ms before the vote. Its direct host bounds are
        # wide. The required final readback places its completion within 20us.
        lo, mid, hi = align_marker(-500_000, [(0, 900_000, 1_200_000),
                                              (1_000_000, 2_000_000, 2_020_000)],
                                   400_000, 1_200_000)
        self.assertLessEqual(lo, 500_000)
        self.assertGreaterEqual(hi, 520_000)
        self.assertLess(hi - lo, 30_000)
        self.assertLessEqual(lo, mid)
        self.assertLessEqual(mid, hi)

    def test_real_jit_window_emits_nine_local_ready_vote_and_backward_marks(self):
        from critical_probe import configure_probe, read_probe
        from jit_gae_runtime import run_jit_policy_update
        from test_paired_gpu_trainer_cpu import Engine
        import numpy as np
        import torch
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

        class CpuEvent:
            def __init__(self, enable_timing): pass
            def record(self): self.timestamp = time.monotonic_ns()
            def elapsed_time(self, last): return (last.timestamp - self.timestamp) / 1e6

        configure_probe(self.out, step=1, arm_id='sham', condition='N',
                        target_rank=2, target_bundle=4)
        config = SimpleNamespace(jit_gae_mode='fused', advantage_norm='ema_zscore', jit_context_micro=2,
                                 advantage_ema_beta=.9, advantage_scale_floor=.001, policy_sampling='chunks')
        observations = []
        for i in range(9):
            sid = f'seg-{i}'
            segment = dict(segment_id=sid, count=1, observations=[{'state': i}, {}], rewards=[.1],
                           terminated=[True], truncated=[False], gamma=.99, lambda_=.9)
            observations.append(dict(state=i, _jit_sample_id=sid, _jit_segment=segment, _jit_index=0,
                                     _jit_version=0, _fresh_action_mask=np.ones((2, 1), bool)))
        for rank in range(4):
            engine = Engine()
            owner = SimpleNamespace(rank=rank, gpu_binding=dict(uuid=f'gpu-{rank}', physical_gpu=rank+4),
                global_step=0, model=engine, base_model=engine.module, fresh_config=config, cfg=SimpleNamespace(),
                train_batch_size=1, super_batch_size=9, vf_coef=.5, ent_coef=.01, kl_coef=.1,
                clip_mode='gipo', clip_eps=.2, policy_train_start_step=0,
                telemetry=SimpleNamespace(event=lambda *a, **k: None), _verify_optimizer_update=lambda *a: None,
                capture_policy_metadata=dict(sample_ids=[o['_jit_sample_id'] for o in observations],
                                             fresh_versions=[0, 0], jit_pending=True))
            with patch.dict(os.environ, self.env), patch.object(torch.cuda, 'Event', CpuEvent), \
                    patch('critical_probe.time.time_ns', time.monotonic_ns):
                result = run_jit_policy_update(owner, observations, np.zeros((9, 2, 1), np.int64),
                    np.zeros((9, 2, 1, 3), np.float32), micro_begin=lambda *a, **k: None,
                    micro_end=lambda *a, **k: None, trim=lambda _: None)
            self.assertEqual((len(result['epoch_losses']), owner.global_step), (9, 1))
        for rank, result in enumerate(read_probe(self.out, 1)):
            self.assertEqual(result['gpu_uuid'], f'gpu-{rank}')
            self.assertEqual([r['bundle'] for r in result['bundles']], list(range(9)))
            for row in result['bundles']:
                self.assertEqual('gate_closed_ns' in row, (rank, row['bundle']) == (2, 4))
                stamps = [row[f'{p}_ns'] for p in
                          ('forward_start', 'forward_ready', 'vote_done', 'backward_only_done', 'backward_done')]
                self.assertEqual(stamps, sorted(stamps))

    def test_actor_service_records_distinguish_queue_batch_and_gpu_stage(self):
        from critical_probe import ActionTrace, read_actions
        class Span:
            def __enter__(self): return self
            def __exit__(self, *exc): pass
        writer = SimpleNamespace(event=lambda *args, **kw: None, span=lambda *args, **kw: Span())
        owner = SimpleNamespace(actor_id=1, capture_role='policy_infer', telemetry=writer,
                                critical_actions=ActionTrace(self.out, 1, gpu_uuid='GPU-abc', physical_gpu=6))
        self.addCleanup(owner.critical_actions.close)
        request_begin(owner, dict(request_id='real', caller_id='5', workflow='real_rollout'))
        request_begin(owner, dict(request_id='other', caller_id='1', workflow='evaluation'))
        batch = batch_begin(owner, [1, 2])
        core = stage_begin(owner, 'gpu_forward_and_copy')
        stage_end(owner, core)
        batch_end(owner, batch)
        rows = read_actions(self.out)
        self.assertEqual([r['kind'] for r in rows], ['enqueue', 'service_start', 'gpu_start', 'gpu_end', 'service_end'])
        self.assertTrue(all((r['rank'], r['worker_id'], r['gpu_uuid'], r['request_id']) ==
                            (1, 5, 'GPU-abc', 'real') for r in rows))
        self.assertEqual(len({r['batch_id'] for r in rows[1:]}), 1)

    def test_worker_readiness_check_and_remote_submission_precede_action_log_io(self):
        from critical_probe import read_actions
        gate = RolloutPulseGate.from_environment(self.env, worker_id=4)
        self.addCleanup(gate.actions.close)
        writer = SimpleNamespace(event=lambda *args, **kw: None)
        owner = SimpleNamespace(wid=4, telemetry=writer, critical_actions=gate.actions, bench_action_gate=gate)
        boundaries = []
        def submit(**kwargs):
            boundaries.append(time.time_ns())
            self.assertEqual(read_actions(self.out), [])
            return 'ref'
        def reply(ref):
            self.assertEqual([r['kind'] for r in read_actions(self.out)], ['ready', 'dispatch'])
            return 'action'
        actor = SimpleNamespace(request=SimpleNamespace(remote=submit))
        with patch.dict(sys.modules, ray=SimpleNamespace(get=reply)):
            profile_rpc(owner, 'policy_infer', actor, 'request', _capture_context={'workflow': 'real_rollout'})
        ready, dispatch, response = read_actions(self.out)
        self.assertEqual(response['request_id'], ready['request_id'])
        self.assertLessEqual(dispatch['ts_ns'], boundaries[0])
        self.assertLessEqual(boundaries[0], dispatch['remote_returned_ns'])
        self.assertLessEqual(dispatch['remote_returned_ns'], response['ts_ns'])

    def test_gate_requires_physical_rank_consistent_condition_and_ttl(self):
        from critical_probe import configure_probe
        for condition, gate, ttl in [('P-target', 3, 1.), ('P-other', 1, 1.), ('N', 1, 5.01)]:
            with self.subTest(condition=condition), self.assertRaises(ValueError):
                configure_probe(self.out, step=1, arm_id='trial', condition=condition,
                                target_rank=1, gate_rank=gate, target_bundle=8, ttl_s=ttl)


if __name__ == '__main__':
    unittest.main()
