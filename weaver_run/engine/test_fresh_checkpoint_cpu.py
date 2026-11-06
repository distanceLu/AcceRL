"""Dense RM ledger is auditable in actual CPU-serialized native artifacts."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import torch

from fresh_reward_targets import dense_reward_specification
from weaver_online_gpu_worker import snapshot_native
from world_backend_audit import (audit_online_native_cpu, validate_joint_reward_metadata,
                                  validate_joint_release_receipts)


class DenseCheckpoint(unittest.TestCase):
    def specification(self):
        return dense_reward_specification()

    def config(self):
        return json.loads((Path(__file__).parent / 'examples/fresh_01_dense_signal.json').read_text())

    def cohort(self):
        spec = self.specification()
        return {'protocol': 'weaver_discounted_success_value_cohort_v1', 'mode': spec['mode'],
            'gamma_per_action': spec['gamma_per_action'], 'discount_unit': 'action_steps',
            'label_source': 'libero_env_step', 'window_count': 1024, 'frame_slice': [30, 40],
            'label_frames_per_window': 10, 'global_selected_frames': 10240,
            'valid_count': 10240, 'masked_count': 0, 'target_sum': 0., 'target_square_sum': 0.,
            'success_episode_valid_count': 0, 'failure_episode_valid_count': 10240,
            'rm_active': True, 'normalization': 'global_valid_frames', 'balance': 'none'}

    def descriptor(self):
        return {'native_reward_head': self.specification(), 'rm_updates': 1,
            'reward_target_counts': {'scope': 'global_selected_frames_once_per_successful_world_update',
                'valid_count': 10240, 'masked_count': 0, 'target_sum': 0.,
                'last_cohort': self.cohort(), 'disabled_updates': 0}}

    def test_dense_ledger_rejects_false_updates_and_wrong_geometry(self):
        descriptor = self.descriptor()
        self.assertTrue(validate_joint_reward_metadata(descriptor, self.config(), updates=1))
        for change in ('updates', 'geometry', 'target'):
            bad = copy.deepcopy(descriptor)
            if change == 'updates':
                bad['rm_updates'] = 0; bad['reward_target_counts']['disabled_updates'] = 1
            if change == 'geometry': bad['reward_target_counts']['last_cohort']['frame_slice'] = [0, 10]
            if change == 'target': bad['reward_target_counts']['target_sum'] = float('nan')
            with self.assertRaises((RuntimeError, ValueError)):
                validate_joint_reward_metadata(bad, self.config(), updates=1)

    def test_dense_cpu_snapshot_and_independent_cpu_audit(self):
        config = self.config()
        engine = SimpleNamespace(rank=0, active_update=None, publication_idle=lambda: True,
            policy_step=5, global_steps=1, skipped_steps=0, rm_updates=1, rm_disabled_updates=0,
            native_reward=SimpleNamespace(specification=self.specification()),
            reward_positive_labels=0, reward_negative_labels=0, reward_target_valid_count=10240,
            reward_target_masked_count=0, reward_target_sum=0., last_reward_cohort=self.cohort())
        # Tiny serialization fixture: 866 tensors, not production model weights.
        backend = SimpleNamespace(upstream_model=SimpleNamespace(state_dict=lambda: {
            'fixture%04d' % i: torch.tensor([float(i)]) for i in range(866)}))
        identity = {'source_optimizer_steps': 3200,
                    'source_bundle_sha256': config['backend_options']['initial_checkpoint_sha256']}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            with mock.patch.dict(os.environ, {'ACCERL_FINAL_CHECKPOINT': str(root)}):
                descriptor = snapshot_native(backend, engine, root / 'checkpoints_world', 'final',
                                             identity, 'weaverintegrated_fresh_cpu')
            self.assertNotIn('reward_class_counts', descriptor)
            request = {'descriptor': descriptor, 'config': config, 'run_id': descriptor['run_id'],
                       'policy_step': 5, 'reason': 'final', 'checkpoint_root': str(root)}
            audited = audit_online_native_cpu(request)
            self.assertTrue(audited['valid'])
            rows = [{'actor_id': str(i), 'pid': 100 + i, 'details': {
                'source_refs_released': True, 'native_engine': {'global_steps': 1,
                'skipped_steps': 0, 'gradients_released': True, 'rm_updates': 1,
                'rm_disabled_updates': 0, 'reward_target_valid_count': 10240,
                'reward_target_masked_count': 0, 'reward_target_sum': 0.,
                'last_reward_cohort': self.cohort()}}} for i in range(4)]
            self.assertTrue(validate_joint_release_receipts(rows, descriptor))
            rows[3]['details']['native_engine']['reward_target_valid_count'] = 0
            with self.assertRaises(RuntimeError): validate_joint_release_receipts(rows, descriptor)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__': unittest.main()
