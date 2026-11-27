"""ZeRO-2 pre-step layout, fixed optimizer moments and GAE parity failure paths."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from paired_shape_trainer import _adam_snapshot, _audit_gradient, _restore_adam


def owner_at_boundary(rank):
    model = torch.nn.Linear(2, 3)
    groups = [[model.weight], [model.bias]]
    # Two groups aligned to four ranks. Values in the extra padding are poison.
    chunks = {i: [torch.tensor([float(j + 1) if j < p.numel() else float('nan') for j in
                               range(rank * size, (rank + 1) * size)])]
              for i, (p, size) in enumerate(((model.weight, 2), (model.bias, 1)))}
    zero = SimpleNamespace(cpu_offload=False, dynamic_loss_scale=False, loss_scale=1.,
        is_gradient_accumulation_boundary=True, round_robin_bit16_groups=groups,
        partition_size=[2., 1.], real_dp_process_group=[None, None], averaged_gradients=chunks,
        single_partition_of_fp32_groups=[torch.empty(2), torch.empty(1)])
    engine = SimpleNamespace(optimizer=zero, zero_optimization_stage=lambda: 2,
                             is_gradient_accumulation_boundary=lambda: True)
    return SimpleNamespace(rank=rank, model=engine, base_model=model)


class ShapeTrainerTests(unittest.TestCase):
    def test_partition_capture_counts_real_elements_once_and_excludes_padding(self):
        baseline, candidate = [], []
        for rank in range(4):
            owner = owner_at_boundary(rank)
            with patch('paired_shape_trainer.dist.get_rank', return_value=rank), \
                    patch('paired_shape_trainer.dist.get_world_size', return_value=4):
                baseline.append(_audit_gradient(owner, 16))
                if rank == 1:
                    owner.model.optimizer.averaged_gradients[0][0][1] = 5.  # replace 4 by 5
                candidate.append(_audit_gradient(owner, 24))
        self.assertEqual(sum(row['elements'] for row in candidate), 9)
        self.assertEqual({row['logical_elements'] for row in baseline}, {9})
        self.assertEqual(len({row['schema_sha256'] for row in baseline + candidate}), 1)
        self.assertEqual(sum(row['reference_sq'] for row in baseline), 105.)
        self.assertEqual(sum(row['candidate_sq'] for row in candidate), 114.)
        self.assertEqual(sum(row['dot'] for row in candidate), 109.)
        self.assertEqual(sum(row['diff_sq'] for row in candidate), 1.)
        self.assertEqual(max(row['max_abs_error'] for row in candidate), 1.)

    def test_incomplete_partition_or_nonfinite_unpadded_gradient_fails_audit(self):
        with patch('paired_shape_trainer.dist.get_rank', return_value=0), \
                patch('paired_shape_trainer.dist.get_world_size', return_value=4):
            owner = owner_at_boundary(0)
            owner.model.optimizer.averaged_gradients[1][0][0] = torch.nan
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                _audit_gradient(owner, 16)
            owner = owner_at_boundary(0)
            owner.model.optimizer.averaged_gradients[0] = [torch.ones(1)]
            with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                _audit_gradient(owner, 16)

    def test_restore_adam_moments_in_place_at_fixed_weights(self):
        master = torch.nn.Parameter(torch.tensor([3., -2.]))
        optimizer = torch.optim.AdamW([master], lr=0.)
        for value in (1., 2.):
            master.grad = torch.full_like(master, value)
            optimizer.step()
        snapshot = _adam_snapshot(optimizer)
        state = optimizer.state[master]
        identities = {k: id(v) for k, v in state.items()}
        master.grad.fill_(-3.)
        optimizer.step()
        self.assertFalse(torch.equal(state['exp_avg'], snapshot[1][0]['exp_avg']))
        _restore_adam(optimizer, snapshot)
        for key, value in state.items():
            self.assertEqual(id(value), identities[key])
            torch.testing.assert_close(value, snapshot[1][0][key], rtol=0, atol=0)
        torch.testing.assert_close(master, torch.tensor([3., -2.]), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
