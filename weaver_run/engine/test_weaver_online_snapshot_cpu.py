"""Immutable native file handoff remains inside the declared checkpoint root."""
import hashlib
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

try:
    import torch
except ImportError:
    torch = None

from weaver_online_gpu_worker import snapshot_native


@unittest.skipIf(torch is None,'Actual CPU Torch serialization required')
class NativeSnapshot(unittest.TestCase):
    def engine(self):
        from weaver_online_smoke import reward_specification
        return SimpleNamespace(rank=0,active_update=None,publication_idle=lambda:True,
            policy_step=5,global_steps=1,skipped_steps=0,rm_updates=0,rm_disabled_updates=1,
            native_reward=SimpleNamespace(specification=reward_specification()),
            reward_positive_labels=0,reward_negative_labels=10240,last_reward_cohort=None)

    def test_full_native_descriptor_relative_to_checkpoint_directory_and_strict_cpu_load(self):
        tensors = {'tensor_%04d'%index:torch.ones(1)*index for index in range(866)}
        backend = SimpleNamespace(upstream_model=SimpleNamespace(state_dict=lambda:tensors))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'checkpoints'
            with mock.patch.dict(os.environ,{'ACCERL_FINAL_CHECKPOINT':str(root)}):
                descriptor = snapshot_native(backend,self.engine(),root/'checkpoints_world','final',
                    {'source_optimizer_steps':3200,'source_bundle_sha256':'a'*64},'weaverintegrated_cpu')
            self.assertEqual(Path(descriptor['relative_weights_file']).parts[0],'checkpoints_world')
            self.assertEqual(descriptor['tensor_count'],866)
            self.assertEqual(descriptor['successful_updates'],1)
            self.assertEqual(descriptor['protocol'],'weaver_online_joint_weights_v1')
            self.assertEqual(descriptor['trained_parameter_families'],['wm','rm'])
            path = Path(descriptor['weights_file'])
            self.assertEqual(descriptor['weights_sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
            loaded = torch.load(path,map_location='cpu',weights_only=True,mmap=True)
            self.assertEqual(len(loaded['model']),866)
            self.assertEqual(loaded['manifest']['policy_step'],5)
            with mock.patch.dict(os.environ,{'ACCERL_FINAL_CHECKPOINT':str(root)}),self.assertRaises(FileExistsError):
                snapshot_native(backend,self.engine(),root/'checkpoints_world','final',
                    {'source_optimizer_steps':3200,'source_bundle_sha256':'a'*64},'weaverintegrated_cpu')

    def test_outside_directory_rejected_before_materializing_model(self):
        backend = SimpleNamespace(upstream_model=SimpleNamespace(state_dict=lambda: (_ for _ in ()).throw(AssertionError())))
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict(os.environ,{
                'ACCERL_FINAL_CHECKPOINT':str(Path(folder)/'checkpoints')}):
            with self.assertRaises(ValueError):
                snapshot_native(backend,self.engine(),folder,'final',{},'weaverintegrated_cpu')


if __name__ == '__main__':
    unittest.main()
