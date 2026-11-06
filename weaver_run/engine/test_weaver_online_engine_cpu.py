"""Actual native flow/Adam update and isolation tests; no GPU allocations."""
from __future__ import annotations

import copy
import unittest
from unittest import mock

try:
    import torch
except ImportError:
    torch = None

from weaver_online_engine import WeaverOnlineEngine


@unittest.skipIf(torch is None, 'CPU Torch required')
class NativeEngine(unittest.TestCase):
    def backend(self):
        from test_weaver_backend_cpu import TinyUpstream, config, sample
        from world_backends.weaver import WeaverBackend
        cfg = config(memory=2)
        backend = WeaverBackend(cfg, model=TinyUpstream(memory=2))
        backend.compute_dtype = torch.bfloat16
        rows = [sample(cfg, backend.training_window_frames) for _ in range(2)]
        return backend, rows

    def engine(self, **options):
        backend, rows = self.backend()
        engine = WeaverOnlineEngine(backend, world_size=1, micro_batch=2,
                                     accumulation_steps=3, **options)
        for step in range(1, 6):
            engine.advance_policy(step)
        return engine, rows

    def test_native_flow_updates_world_only_and_exact_micro_boundaries(self):
        engine, rows = self.engine()
        before = {name: value.detach().clone() for name, value in engine.module.named_parameters()}
        engine.begin_update(5)
        for index in range(3):
            result = engine.micro(rows)
            self.assertEqual(result['committed'], index == 2)
            self.assertEqual(engine.global_steps, int(index == 2))
        self.assertEqual(engine.micro_steps, 3)
        self.assertEqual(engine.skipped_steps, 0)
        self.assertEqual(engine.last_result['global_samples'], 6)
        self.assertTrue(all(p.grad is None for p in engine.parameters))
        self.assertFalse(torch.equal(before['upstream.wm.gain'], engine.module.upstream.wm.gain))
        self.assertTrue(torch.equal(before['upstream.rm.bias'], engine.module.upstream.rm.bias))
        self.assertTrue(torch.equal(before['upstream.critic.bias'], engine.module.upstream.critic.bias))
        self.assertEqual(engine.state()['backend_engine'], 'torch_ddp_native_flow_matching')
        self.assertEqual(engine.state()['optimizer_betas'], [.9, .99])
        self.assertEqual(engine.state()['weight_decay'], .01)

    def test_accumulated_native_gradient_matches_reference_full_batch_adam(self):
        engine, rows = self.engine(warmup_updates=0)
        reference_backend, _ = self.backend()
        reference_backend.model.load_state_dict(engine.module.state_dict())
        parameters = [p for p in reference_backend.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=1e-5, betas=(.9,.99), weight_decay=.01)
        # Same deterministic native Tiny WEAVER.forward, not a mirror loss.
        engine.begin_update(5)
        for _ in range(3):
            engine.micro(rows)
        prepared = reference_backend.prepare_train_batch(rows * 3)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            loss = reference_backend.model(prepared)[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        for actual, expected in zip(engine.parameters, parameters):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-7)

    def test_warmup_matches_actual_offline_lambdalr_then_remains_constant(self):
        parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.AdamW([parameter], lr=1e-5, betas=(.9,.99), weight_decay=.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1., (step+1)/20))
        expected = []
        for _ in range(22):
            expected.append(optimizer.param_groups[0]['lr'])
            parameter.grad = torch.zeros_like(parameter)
            optimizer.step()
            scheduler.step()
        engine, rows = self.engine()
        actual = []
        for update in range(22):
            if update:
                for step in range(update * 5 + 1, update * 5 + 6):
                    engine.advance_policy(step)
            actual.append(engine.begin_update((update+1)*5)['learning_rate'])
            for _ in range(3):
                engine.micro(rows)
        self.assertEqual(actual, expected)
        self.assertAlmostEqual(actual[0], 5e-7)
        self.assertEqual(actual[19:], [1e-5] * 3)

    def test_publication_hold_and_wrong_schedule_fail_before_learning(self):
        engine, rows = self.engine()
        engine.publication_idle = lambda: False
        with self.assertRaisesRegex(RuntimeError, 'IPC'):
            engine.begin_update(5)
        engine.publication_idle = lambda: True
        with self.assertRaisesRegex(ValueError, 'fifth'):
            engine.begin_update(4)
        with self.assertRaisesRegex(RuntimeError, 'active'):
            engine.micro(rows)

    def test_nonfinite_loss_has_no_optimizer_commit(self):
        engine, rows = self.engine()
        engine.begin_update(5)
        engine.model = lambda prepared: torch.tensor(float('nan'), requires_grad=True)
        with self.assertRaisesRegex(FloatingPointError, 'Nonfinite native'):
            engine.micro(rows)
        self.assertEqual(engine.global_steps, 0)
        self.assertEqual(engine.micro_steps, 0)


if __name__ == '__main__':
    unittest.main()
