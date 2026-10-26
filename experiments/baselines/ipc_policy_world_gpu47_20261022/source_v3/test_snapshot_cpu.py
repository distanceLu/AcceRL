"""Inference cache broadcasts must not leak non-persistent buffers to snapshots."""
import types
import unittest
import torch
from fsdp_engine import PolicyFSDPEngine


class SnapshotTest(unittest.TestCase):
    def test_inference_caches_are_not_restore_keys(self):
        model = torch.nn.Linear(2, 1)
        model.register_buffer('persistent', torch.ones(2))
        model.register_buffer('rotary_cache', torch.zeros(5), persistent=False)
        values = dict(model.named_parameters()) | dict(model.named_buffers())
        owner = types.SimpleNamespace(module=model, kind='policy', trainable_names=['weight', 'bias'],
            _cpu_export={'version': 2, 'tensors': values})
        result = PolicyFSDPEngine.snapshot_weights(owner, 2)
        self.assertEqual(set(result['state_dict']), set(model.state_dict()))
        self.assertIn('rotary_cache', owner._cpu_export['tensors'])
        model.load_state_dict(result['state_dict'], strict=True)
        with self.assertRaises(RuntimeError):
            PolicyFSDPEngine.snapshot_weights(owner, 3)

    def test_reward_full_state_includes_frozen_parameters(self):
        model = torch.nn.Linear(2, 1)
        model.weight.requires_grad_(False)
        model.register_buffer('rotary_cache', torch.zeros(5), persistent=False)
        owner = types.SimpleNamespace(module=model, kind='reward', trainable_names=['bias'],
            _cpu_export={'version': 1, 'tensors': {'bias': model.bias}},
            _cpu_full_state=dict(model.named_parameters()) | dict(model.named_buffers()))
        result = PolicyFSDPEngine.snapshot_weights(owner, 1)
        self.assertTrue(result['full_state'])
        self.assertEqual(set(result['state_dict']), set(model.state_dict()))
        self.assertEqual(result['trainable_names'], ['bias'])
        model.load_state_dict(result['state_dict'], strict=True)
        owner._cpu_full_state.pop('weight')
        with self.assertRaises(RuntimeError):
            PolicyFSDPEngine.snapshot_weights(owner, 1)


if __name__ == '__main__':
    unittest.main()
