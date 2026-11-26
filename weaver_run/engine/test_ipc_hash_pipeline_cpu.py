"""Check DMA slot scheduling and digest bytes without requiring a GPU."""
from contextlib import nullcontext
import unittest
from unittest.mock import Mock, patch

import torch

import cuda_ipc_weights as ipc
from training_topology import _weight_hash


class _Source:
    device = torch.device('cuda:0')

    def __init__(self, tensor):
        self.tensor = tensor

    def __getattr__(self, name):
        return getattr(self.tensor, name)


class GroupedHashPipelineTests(unittest.TestCase):
    def test_groups_packed_on_demand_and_digest_preserves_historical_bytes(self):
        weights = [(str(n), torch.tensor([n, -n], dtype=torch.int32)) for n in range(15)]
        weights.extend([('empty', torch.empty(0)), ('oversize', torch.arange(17))])
        on_device = [(name, _Source(t)) for name, t in weights]
        pack = ipc._packed_bytes
        allocate = torch.empty
        events = []
        packed = []

        def tracked_pack(tensors):
            packed.append(len(tensors))
            self.assertLessEqual(len(packed), 2 + sum(e.synchronize.call_count for e in events))
            return pack(tensors)

        def event():
            events.append(Mock())
            return events[-1]

        with patch.object(ipc, '_packed_bytes', side_effect=tracked_pack), \
             patch.object(torch.cuda, 'Stream'), patch.object(torch.cuda, 'current_stream'), \
             patch.object(torch.cuda, 'stream', return_value=nullcontext()), \
             patch.object(torch.cuda, 'Event', side_effect=event), \
             patch.object(torch, 'empty', side_effect=lambda n, dtype, pin_memory: allocate(n, dtype=dtype)):
            actual = ipc._grouped_policy_hash(on_device, 16)
        self.assertEqual(actual, _weight_hash(weights, 'policy'))
        self.assertEqual(len(events), 2)
        self.assertTrue(all(e.record.call_count == e.synchronize.call_count for e in events))


if __name__ == '__main__':
    unittest.main()
