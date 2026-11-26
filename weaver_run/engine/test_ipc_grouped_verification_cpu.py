"""Bitwise checks for the grouped policy publication path, using CPU tensors."""
import unittest

import torch

import cuda_ipc_weights as ipc
from training_topology import _weight_hash


class GroupedVerification(unittest.TestCase):
    def test_historical_sha_and_raw_copy_comparison(self):
        weights = [('empty', torch.empty(0)), ('scalar', torch.tensor(-0.0)),
                   ('strided', torch.arange(24).view(4, 6).t()),
                   ('values', torch.tensor([1, -2, float('nan')], dtype=torch.bfloat16))]
        source = [value.clone() for _, value in weights]
        for limit in (1, 32, 1 << 20):
            self.assertEqual(ipc._grouped_policy_hash(weights, limit), _weight_hash(weights, 'policy'))
            self.assertTrue(ipc._copied_bytes_match(weights, source, limit))
        source[1].fill_(0.0)  # Different bit representation despite equal numerical value
        self.assertFalse(ipc._copied_bytes_match(weights, source, 32))
        source[1].fill_(-0.0)
        source[2][-1, -1] += 1
        self.assertFalse(ipc._copied_bytes_match(weights, source, 32))

    def test_changed_control_record_is_rejected(self):
        payload = {'kind': 'policy', 'verification': ipc.GPU_BYTE_COMPARE, 'inventory': {'hash': 'old'},
                   'tensors': [{'args': [b'handle']}]}
        payload['control_sha256'] = ipc.metadata_fingerprint(payload)
        self.assertEqual(ipc._checked_fingerprint(payload), payload['control_sha256'])
        for key, changed in [('inventory', {'hash': 'changed'}), ('tensors', [])]:
            with self.assertRaises(ValueError):
                ipc._checked_fingerprint(dict(payload, **{key: changed}))


if __name__ == '__main__':
    unittest.main()
