"""Synthetic CPU descriptors for the official Torch2.7/2.12 CUDA IPC envelopes.

No GPU export/import is performed. These opaque fixture bytes were constructed
from the documented envelope layout, not captured from the failed GPU run.
"""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
FROZEN=HERE
sys.path.insert(0,str(FROZEN))


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


repaired=load(HERE/'cuda_ipc_weights.py','cpu_repaired_ipc')


def descriptor(handle=b'\x01c'+b'a'*64):
    return {'name':'rm.net.weight','dtype':'torch.float32','size':[2,3],'stride':[3,1],
            'args':['torch.Tensor',[2,3],[3,1],0,'torch.storage.TypedStorage','torch.float32',
                    0,handle,24,0,False,b'/torch_ref',0,b'e'*64,True]}


def empty():
    item=descriptor(None)
    item.update(size=[0,3])
    item['args'][1]=[0,3]
    item['args'][8]=0
    for index in (11,13,14):item['args'][index]=None
    return item


class Envelope(unittest.TestCase):
    def test_documented_torch27_envelope_is_accepted(self):
        value=descriptor()
        self.assertIs(repaired.validate_reduction_descriptor(value),value)

    def test_both_documented_versions_and_legacy_remain_byte_exact(self):
        for handle in (b'a'*64,b'\x01c'+b'a'*64,b'\x02c'+b'a'*64):
            with self.subTest(prefix=handle[:2]):
                value=descriptor(handle)
                saved=copy.deepcopy(value)
                repaired.validate_reduction_descriptor(value)
                self.assertEqual(value,saved)
                self.assertIs(value['args'][7],handle)

    def test_unknown_versions_and_expandable_envelopes_are_rejected(self):
        for prefix in (b'\x00c',b'\x03c',b'\xffc',b'\x01e',b'\x02e',b'xx'):
            with self.subTest(prefix=prefix),self.assertRaises(ValueError):
                repaired.validate_reduction_descriptor(descriptor(prefix+b'a'*64))

    def test_malformed_lengths_and_types_remain_rejected(self):
        for handle in (b'',b'a',b'a'*63,b'a'*65,b'a'*67,b'a'*128,'a'*64,bytearray(b'a'*64)):
            with self.subTest(handle_type=type(handle)),self.assertRaises(ValueError):
                repaired.validate_reduction_descriptor(descriptor(handle))

    def test_empty_none_storage_keeps_original_strict_branch(self):
        value=empty()
        self.assertIs(repaired.validate_reduction_descriptor(value),value)
        self.assertIsNone(value['args'][7])
        self.assertIsNone(value['args'][14])
        for index,bad in ((8,4),(11,b'ref'),(13,b'e'*64),(14,False)):
            corrupted=empty();corrupted['args'][index]=bad
            with self.subTest(index=index),self.assertRaises(ValueError):
                repaired.validate_reduction_descriptor(corrupted)

    def test_nonempty_cannot_use_null_storage_or_null_event(self):
        with self.assertRaises(ValueError):repaired.validate_reduction_descriptor(descriptor(None))
        for event in (None,b'',b'e'*63,b'e'*65):
            value=descriptor();value['args'][13]=event
            with self.subTest(event=event),self.assertRaises(ValueError):
                repaired.validate_reduction_descriptor(value)

    def test_invalid_device_shape_stride_or_rebuild_flags_remain_rejected(self):
        for index,bad in ((6,1),(10,True),(3,-1),(2,[2,1])):
            value=descriptor();value['args'][index]=bad
            with self.subTest(index=index),self.assertRaises(ValueError):
                repaired.validate_reduction_descriptor(value)

    def test_cpu_decoder_never_opens_cuda_and_preserves_envelope(self):
        try:import torch
        except ImportError:self.skipTest('CPU Torch needed only for metadata dtype decoding')
        self.assertFalse(torch.cuda.is_initialized())
        for handle in (b'\x01c'+b'a'*64,b'\x02c'+b'b'*64,b'c'*64):
            value=descriptor(handle)
            decoded=repaired._decode_reduction(torch,value)
            self.assertEqual(decoded[7],handle)
        decoded=repaired._decode_reduction(torch,empty())
        self.assertIsNone(decoded[7])
        self.assertFalse(torch.cuda.is_initialized())

    def test_actual_frozen_smoke_tiny_inventory_has_no_zero_sized_buffers(self):
        try:import torch
        except ImportError:self.skipTest('CPU Torch needed for synthetic tiny inventory')
        from weaver_online_smoke import tiny_backend
        backend=tiny_backend(torch,'cpu')
        values=dict(backend.model.named_parameters())
        self.assertEqual(len(values),4)
        self.assertEqual(list(backend.model.named_buffers()),[])
        self.assertEqual(sum(value.numel() for value in values.values()),9)
        self.assertTrue(all(value.numel()>0 for value in values.values()))
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main(verbosity=2)
