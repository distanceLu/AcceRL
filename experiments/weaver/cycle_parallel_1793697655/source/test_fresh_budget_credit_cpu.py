"""Exercise actual four-rank coordinator credits with mixed FRESH sources."""
import asyncio
from types import SimpleNamespace
import unittest

from training_topology import GlobalBatchCoordinator
from test_replay_projection_cpu import TicketReplay


class SourceCredits(unittest.IsolatedAsyncioTestCase):
    def coordinator(self, counts):
        async def sample(size):
            ids = [str(i) for i in range(size)]
            return ids, ids, ids, ids, ids, {'sample_ids': ids,
                'reservation_id': 'actual-ledger-reservation',
                'fresh_reservation_token': 'separate-use-lease', 'source_counts': counts}
        replay = TicketReplay()
        policy = SimpleNamespace(sample=SimpleNamespace(remote=sample))
        return GlobalBatchCoordinator(policy, replay, 4, pipeline_enabled=True), replay

    async def test_real_quota_does_not_grant_imaginary_production_credits(self):
        coordinator, replay = self.coordinator({'real': 2, 'imagined': 6})
        batches = await asyncio.gather(*[coordinator.sample_policy(i, 2) for i in range(4)])
        self.assertEqual(replay.policy_credits, 6)
        self.assertEqual(sum(len(row[0]) for row in batches), 8)
        self.assertTrue(all(row[-1]['reservation_id'] == 'actual-ledger-reservation' for row in batches))
        self.assertEqual(len({sid for row in batches for sid in row[-1]['sample_ids']}), 8)

    async def test_all_imagined_fresh_cohort_retains_original_credit_total(self):
        coordinator, replay = self.coordinator({'real': 0, 'imagined': 8})
        await asyncio.gather(*[coordinator.sample_policy(i, 2) for i in range(4)])
        self.assertEqual(replay.policy_credits, 8)

    async def test_missing_or_inconsistent_source_counts_cannot_issue_credits(self):
        for counts in ({}, {'real': 2, 'imagined': 8}, {'real': True, 'imagined': 7}):
            coordinator, replay = self.coordinator(counts)
            with self.assertRaises(ValueError): await coordinator.sample_policy(0, 2)
            self.assertEqual(replay.policy_credits, 0)


if __name__ == '__main__': unittest.main()
