"""Admission, inclusion probabilities and use accounting for real-only sampling."""
import random
import unittest
from collections import Counter

from fresh_config import FreshConfig
from fresh_policy_replay import FreshReplayNotReady, SegmentReplay
from test_fresh_policy_replay_cpu import segment


def make_store(lengths, *, max_uses=4, rng=None):
    config = FreshConfig(real_policy_replay=True, real_only=True, real_fraction=1.,
                         replay_age_limits=True, max_sample_uses=max_uses)
    store = SegmentReplay(sum(lengths), config, rng=rng or random.Random(7))
    for number, length in enumerate(lengths):
        store.add(segment(str(number), 'real', policy=0, count=length, terminal=True))
    store.set_versions(0, 0)
    return store


class FixedOffset(random.Random):
    """Enumerate all starts for one fixed ordering of the episodes."""
    offset = 0

    def shuffle(self, sequence):
        pass

    def randrange(self, stop):
        assert 0 <= self.offset < stop
        return self.offset


class EpisodeSamplingTests(unittest.TestCase):
    def test_exact_packs_keep_episodes_together_and_lease_only_selected_chunks(self):
        store = make_store([9] * 80 + [10])
        token, ids = store.reserve(576, real_sampling='whole_episodes')
        audit = store.reservation_sampling[token]
        self.assertEqual((audit['requested'], audit['actual']), ('whole_episodes', 'whole_episodes'))
        self.assertIsNone(audit['fallback_reason'])
        self.assertEqual(audit['rank_split_segments'], 0)
        self.assertEqual(len(set(ids)), 576)
        touched = set()
        for rank in range(4):
            shard = ids[144*rank:144*(rank+1)]
            episodes = {store.samples[s]['segment_id'] for s in shard}
            self.assertTrue(touched.isdisjoint(episodes))
            touched.update(episodes)
            for sid in episodes:
                episode = store.segments[sid]
                self.assertTrue({episode.sample_id(i) for i in range(episode.count)} <= set(shard))
        self.assertEqual(sum(s['reserved'] for s in store.samples.values()), 576)
        store.commit(token, ids[:144])
        self.assertIn(token, store.reservation_sampling)
        store.cancel(token)
        self.assertNotIn(token, store.reservation_sampling)
        self.assertEqual(sum(s['uses'] for s in store.samples.values()), 144)
        self.assertEqual(sum(s['reserved'] for s in store.samples.values()), 0)

    def test_uniform_blocks_include_every_eligible_chunk_equally_despite_holes(self):
        rng = FixedOffset(7)
        store = make_store([200, 200, 180], rng=rng)
        # The two holes are still resident and must keep their original physical indices.
        store.samples['0:chunk10']['uses'] = 4
        store.samples['1:chunk6']['reserved'] = 4
        candidates = set(store.samples) - {'0:chunk10', '1:chunk6'}
        hits = Counter()
        for offset in range(len(candidates)):
            rng.offset = offset
            token, ids = store.reserve(576, real_sampling='uniform_blocks')
            self.assertEqual(len(set(ids)), 576)
            self.assertEqual(store.reservation_sampling[token]['eligible_chunks'], 578)
            hits.update(ids)
            store.cancel(token)
        self.assertEqual(hits, dict.fromkeys(candidates, 576))

    def test_explicit_block_fallback_and_no_quota_bypass(self):
        store = make_store([145]*4, max_uses=1)
        store.samples['0:chunk0']['uses'] = 1
        token, ids = store.reserve(576, real_sampling='whole_episodes')
        audit = store.reservation_sampling[token]
        self.assertEqual(audit['actual'], 'uniform_blocks')
        self.assertTrue(audit['fallback_reason'])
        self.assertNotIn('0:chunk0', ids)
        with self.assertRaises(FreshReplayNotReady):
            store.reserve(576, real_sampling='whole_episodes')
        store.commit(token, ids)
        self.assertNotIn(token, store.reservation_sampling)
        self.assertEqual(sum(s['uses'] for s in store.samples.values()), 577)

    def test_first_use_blocks_choose_higher_coverage_on_the_same_ring(self):
        store = make_store([300, 302])
        # Seed 7 keeps the episode order and proposes starts 154 and 404
        # on the 600 eligible chunks. Both windows wrap around the ring.
        store.samples['1:chunk300']['uses'] = 4
        store.samples['1:chunk301']['reserved'] = 4
        for index in range(80, 104):
            store.samples[f'1:chunk{index}']['uses'] = 1
        # Already claimed by an outstanding reservation, still eligible for reuse.
        store.samples['1:chunk78']['reserved'] = 1
        token, ids = store.reserve(576, real_sampling='first_use_blocks')
        audit = store.reservation_sampling[token]
        self.assertEqual(audit['actual'], 'first_use_blocks')
        self.assertEqual(audit['candidate_first_use_chunks'], [551, 575])
        self.assertEqual(audit['selected_first_use_chunks'], 575)
        self.assertEqual(ids, [f'1:chunk{i}' for i in range(104, 300)] +
                         [f'0:chunk{i}' for i in range(300)] +
                         [f'1:chunk{i}' for i in range(80)])
        store.cancel(token)
        self.assertEqual(store.samples['1:chunk78']['reserved'], 1)

    def test_first_use_tie_keeps_the_uniform_first_proposal(self):
        uniform = make_store([300, 302])
        priority = make_store([300, 302])
        uniform_token, expected = uniform.reserve(576, real_sampling='uniform_blocks')
        token, ids = priority.reserve(576, real_sampling='first_use_blocks')
        self.assertEqual(ids, expected)
        self.assertEqual(priority.reservation_sampling[token]['candidate_first_use_chunks'], [576, 576])
        self.assertNotIn('candidate_first_use_chunks', uniform.reservation_sampling[uniform_token])

    def test_age_limit_and_unchanged_default_uniform_draw(self):
        store = make_store([200]*3)
        expected = random.Random(7).sample(list(store.samples), 576)
        token, ids = store.reserve(576)
        self.assertEqual(ids, expected)
        store.cancel(token)
        store.set_versions(17, 0)
        with self.assertRaises(FreshReplayNotReady):
            store.reserve(576, real_sampling='uniform_blocks')
        self.assertFalse(store.reservations)

    def test_protocol_rejects_wrong_mode_source_and_shape_before_reserving(self):
        store = make_store([200]*3)
        with self.assertRaises(ValueError):
            store.reserve(576, real_sampling='misspelt')
        with self.assertRaises(ValueError):
            store.reserve(575, real_sampling='uniform_blocks')
        mixed = SegmentReplay(1000, FreshConfig(real_policy_replay=True))
        with self.assertRaises(ValueError):
            mixed.reserve(576, real_sampling='whole_episodes')
        self.assertFalse(store.reservations)
        self.assertFalse(mixed.reservations)


if __name__ == '__main__':
    unittest.main()
