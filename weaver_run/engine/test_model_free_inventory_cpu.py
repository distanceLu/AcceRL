"""CPU checks for real collection admission against the live Replay use ledger."""
import random
import unittest

from fresh_config import FreshConfig
from fresh_policy_replay import SegmentReplay
from test_fresh_policy_replay_cpu import segment


def real_store(capacity=32, max_uses=4):
    cfg = FreshConfig(real_policy_replay=True, real_only=True, real_fraction=1.,
                      replay_age_limits=True, max_policy_age=16, max_sample_uses=max_uses)
    store = SegmentReplay(capacity, cfg, rng=random.Random(22))
    store.set_versions(16, 0)
    return store


class ModelFreeInventoryTests(unittest.TestCase):
    def test_ages_at_current_and_next_publications(self):
        store = real_store()
        store.add(segment('expires', 'real', policy=0, count=5))
        store.add(segment('last_next', 'real', policy=1, count=3))
        store.add(segment('young', 'real', policy=16, count=2))
        snapshot = store.model_free_inventory(5)
        self.assertEqual(snapshot['eligible_now'], 10)
        self.assertEqual(snapshot['eligible_next'], 5)
        self.assertTrue(snapshot['next_batch_ready'])
        store.set_versions(17, 0)
        self.assertEqual(store.model_free_inventory(5)['eligible_next'], 2)
        self.assertFalse(store.model_free_inventory(5)['next_batch_ready'])

    def test_debit_reserved_use_once_across_rank_commit_and_cancellation(self):
        store = real_store(max_uses=1)
        store.add(segment('a', 'real', policy=16, count=4))
        self.assertTrue(store.model_free_inventory(4)['next_batch_ready'])
        token, ids = store.reserve(2)
        self.assertEqual(store.model_free_inventory(4)['eligible_next'], 2)
        store.commit(token, ids[:1])
        self.assertEqual(store.model_free_inventory(4)['eligible_next'], 2)
        store.cancel(token)
        self.assertEqual(store.model_free_inventory(4)['eligible_next'], 3)

    def test_evicted_and_replaced_segments_change_projected_inventory(self):
        store = real_store(capacity=3)
        store.add(segment('young', 'real', policy=16, count=3))
        self.assertTrue(store.model_free_inventory(3)['next_batch_ready'])
        store.add(segment('old', 'real', policy=0, count=2))
        self.assertEqual(store.model_free_inventory(3)['eligible_next'], 0)

    def test_two_batches_need_distinct_ids_per_version_and_two_available_uses(self):
        store = real_store()
        store.add(segment('a', 'real', policy=16, count=2))
        # Two IDs with four uses each can fill at most two positions in each draw.
        snapshot = store.model_free_inventory(3)
        self.assertEqual(snapshot['next_two_batch_slots'], 4)
        self.assertFalse(snapshot['next_two_versions_ready'])
        self.assertTrue(store.model_free_inventory(2)['next_two_versions_ready'])
        for _ in range(3):
            token, ids = store.reserve(2)
            store.commit(token, ids)
        self.assertEqual(store.model_free_inventory(2)['next_two_batch_slots'], 2)
        self.assertFalse(store.model_free_inventory(2)['next_two_versions_ready'])

    def test_survivors_and_combined_use_slots_must_both_cover_two_draws(self):
        store = real_store(max_uses=2)
        store.add(segment('survivors', 'real', policy=16, count=3))
        token, ids = store.reserve(3)
        store.add(segment('expire_after_next', 'real', policy=1, count=2))
        # At v+2 there are three eligible IDs, but using one in v+1 spends its last use.
        snapshot = store.model_free_inventory(3)
        self.assertEqual(snapshot['next_two_batch_slots'], 7)
        self.assertFalse(snapshot['next_two_versions_ready'])
        store.commit(token, ids[:2])
        self.assertFalse(store.model_free_inventory(3)['next_two_versions_ready'])
        store.cancel(token)
        self.assertTrue(store.model_free_inventory(3)['next_two_versions_ready'])

    def test_two_version_capacity_excludes_ids_expiring_between_draws(self):
        store = real_store(max_uses=1)
        store.add(segment('usable_only_once', 'real', policy=1, count=4))
        store.add(segment('survivor', 'real', policy=16, count=1))
        snapshot = store.model_free_inventory(2)
        self.assertEqual(snapshot['next_two_batch_slots'], 5)
        self.assertTrue(snapshot['next_two_batches_ready'])
        self.assertFalse(snapshot['next_two_versions_ready'])
        store.add(segment('second_survivor', 'real', policy=16, count=1))
        self.assertTrue(store.model_free_inventory(2)['next_two_versions_ready'])

    def test_unpublished_future_rows_and_empty_replay_do_not_admit(self):
        cfg = FreshConfig.from_world_config(dict(backend='model_free'))
        store = SegmentReplay(8, cfg)
        store.add(segment('future', 'real', policy=17, count=2))
        self.assertFalse(store.model_free_inventory(2)['next_batch_ready'])
        store.set_versions(16, 0)
        self.assertFalse(store.model_free_inventory(2)['next_batch_ready'])
        store.set_versions(17, 0)
        self.assertTrue(store.model_free_inventory(2)['next_batch_ready'])


if __name__ == '__main__':
    unittest.main()
