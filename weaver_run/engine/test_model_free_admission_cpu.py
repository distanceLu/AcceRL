"""Actual Replay admission must precede all distributed training work."""
import asyncio
import unittest
from test_fresh_policy_replay_cpu import runtime_replay_class, segment, Ledger, Writer


class ModelFreeAdmissionTests(unittest.TestCase):
    def owner(self):
        owner = runtime_replay_class()(8, dict(real_only=True, real_policy_replay=True,
            real_fraction=1., replay_age_limits=True, max_policy_age=16, max_sample_uses=1))
        owner.capture_ledger, owner.telemetry = Ledger(), Writer()
        owner.configure_fresh([object()])
        owner.set_fresh_versions(4, 0)
        return owner

    def test_shortage_returns_without_wait_or_reservation(self):
        owner = self.owner()
        result = owner.admit_model_free_batch(2)
        self.assertFalse(result['ready'])
        self.assertEqual(result['missing']['real'], 2)
        self.assertEqual(owner.fresh_store.reservations, {})

    def test_reserved_cohort_is_protected_and_sample_consumes_exact_admission(self):
        owner = self.owner()
        owner.add_segment(segment('real', source='real', count=2))
        first = owner.admit_model_free_batch(2)
        self.assertTrue(first['ready'])
        self.assertEqual(owner.admit_model_free_batch(2), first)
        self.assertFalse(owner.fresh_store._evict('real', 'test'))
        self.assertEqual(len(owner.fresh_store.reservations), 1)
        result = asyncio.run(owner.sample(2))
        self.assertEqual(result[-1]['fresh_reservation_token'], first['reservation_token'])
        self.assertIsNone(owner._model_free_admission)

    def test_stopping_after_admission_releases_unconsumed_samples(self):
        owner = self.owner()
        owner.add_segment(segment('real', source='real', count=2))
        first = owner.admit_model_free_batch(2)
        owner.cancel_model_free_admission(first['reservation_token'])
        self.assertEqual(owner.fresh_store.reservations, {})
        self.assertTrue(owner.admit_model_free_batch(2)['ready'])

    def test_cannot_change_count_or_version_of_admitted_cohort(self):
        owner = self.owner()
        owner.add_segment(segment('real', source='real', count=2))
        owner.admit_model_free_batch(2)
        with self.assertRaisesRegex(ValueError, 'size'):
            owner.admit_model_free_batch(1)
        with self.assertRaisesRegex(RuntimeError, 'admitted'):
            owner.set_fresh_versions(5, 0)

    def test_legacy_replay_cannot_use_model_free_admission(self):
        owner = runtime_replay_class()(8, {})
        with self.assertRaisesRegex(ValueError, 'real-only'):
            owner.admit_model_free_batch(2)


if __name__ == '__main__':
    unittest.main()
