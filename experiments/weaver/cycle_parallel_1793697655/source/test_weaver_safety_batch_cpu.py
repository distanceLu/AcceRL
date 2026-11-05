"""Batched finite checks retain commit/publication failure boundaries."""
import copy
import unittest
import torch
import test_weaver_online_engine_cpu as fixtures


class BatchedSafetyTests(unittest.TestCase):
    def pair(self):
        fixture = fixtures.NativeEngine()
        old, rows = fixture.engine()
        new, _ = fixture.engine()
        new.module.load_state_dict(copy.deepcopy(old.module.state_dict()))
        new.backend.optimizations['safety_checks'] = 'batched'
        return old, new, rows

    def test_same_actual_adam_update(self):
        old, new, rows = self.pair()
        for engine in (old, new):
            engine.begin_update(5)
            for _ in range(3):
                engine.micro(rows)
        for name, value in old.module.state_dict().items():
            torch.testing.assert_close(value, new.module.state_dict()[name], atol=0, rtol=0)
        self.assertEqual(old.global_steps, new.global_steps)
        self.assertEqual(old.last_result['grad_norm'], new.last_result['grad_norm'])
        self.assertEqual(old.last_result['loss'], new.last_result['loss'])

    def test_nan_gradient_never_commits(self):
        _, engine, rows = self.pair()
        parameter = engine.parameters[0]
        handle = parameter.register_hook(lambda gradient: torch.full_like(gradient, float('nan')))
        engine.begin_update(5)
        with self.assertRaisesRegex(FloatingPointError, 'gradients'):
            for _ in range(3):
                engine.micro(rows)
        handle.remove()
        self.assertEqual(engine.global_steps, 0)
        self.assertEqual(len(engine.optimizer.state), 0)
        self.assertTrue(all(parameter.grad is None for parameter in engine.parameters))

    def test_nonfinite_post_step_weights_forbid_publication_counter(self):
        _, engine, rows = self.pair()
        actual_step = engine.optimizer.step
        def broken_step():
            actual_step()
            with torch.no_grad():
                engine.parameters[0].fill_(float('inf'))
        engine.optimizer.step = broken_step
        engine.begin_update(5)
        with self.assertRaisesRegex(FloatingPointError, 'publication forbidden'):
            for _ in range(3):
                engine.micro(rows)
        self.assertEqual(engine.global_steps, 0)

    def test_no_grad_parameters_are_skipped_without_skipping_other_elements(self):
        _, engine, _ = self.pair()
        self.assertTrue(engine._all_true(engine._finite_tensors([None, torch.tensor([1., 2.])])) )
        self.assertFalse(engine._all_true(engine._finite_tensors([None, torch.tensor([1., float('nan')])])))


if __name__ == '__main__':
    unittest.main()
