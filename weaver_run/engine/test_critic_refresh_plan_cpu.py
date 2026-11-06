import random
import unittest
from types import SimpleNamespace
import numpy as np
from critic_refresh_plan import balanced_refresh_groups, refresh_plan_receipt


class RefreshPlanTests(unittest.TestCase):
    def test_skewed_whole_segments_reduce_tail_without_changing_any_observation(self):
        sizes = [29, 5, 5, 5] * 20
        segments = [(str(i), tuple((i, j) for j in range(n))) for i, n in enumerate(sizes)]
        result = balanced_refresh_groups(segments, 4)
        counts = refresh_plan_receipt(result)['observations_by_replica']
        self.assertLess(max(counts), sum(sizes[::4]))
        self.assertEqual(dict(segments), dict(item for group in result for item in group))
        self.assertEqual(sum(map(len, result)), len(segments))

    def test_random_shapes_never_worsen_round_robin_peak_and_are_deterministic(self):
        rng = random.Random(20261030)
        for _ in range(200):
            sizes = [rng.randint(2, 1025) for _ in range(rng.randint(1, 576))]
            segments = [(str(i), tuple(range(n))) for i, n in enumerate(sizes)]
            before = max(sum(sizes[r::4]) for r in range(4))
            state = random.getstate()
            groups = balanced_refresh_groups(segments, 4)
            self.assertEqual(random.getstate(), state)
            self.assertEqual(groups, balanced_refresh_groups(segments, 4))
            self.assertLessEqual(max(refresh_plan_receipt(groups)['observations_by_replica']), before)
            flattened = [item for group in groups for item in group]
            self.assertEqual(dict(flattened), dict(segments))
            self.assertEqual(len(flattened), len(segments))
            for sid, observations in flattened:
                self.assertIs(observations, segments[int(sid)][1])

    def test_empty_and_single_segment(self):
        self.assertEqual(balanced_refresh_groups([], 4), [[], [], [], []])
        one = [('segment', (1, 2))]
        self.assertEqual(balanced_refresh_groups(one, 4), [one, [], [], []])

    def test_invalid_identity_and_replica_count_fail_before_dispatch(self):
        for n in [0, -1, True, 1.5]:
            with self.assertRaises(ValueError):balanced_refresh_groups([], n)
        with self.assertRaises(ValueError):balanced_refresh_groups([('x', (1,)), ('x', (2,))], 4)
        with self.assertRaises(ValueError):balanced_refresh_groups([('x', ())], 4)


class ActualReplayPlacementTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_replay_keeps_full_versioned_segments_and_materialized_targets(self):
        from test_fresh_policy_replay_cpu import runtime_replay_class, Ledger, Writer, segment
        from fresh_config import FreshConfig
        sizes = [28, 4, 4, 4] * 5
        replay = runtime_replay_class()(512, FreshConfig(refresh_gae=True,
            replay_age_limits=True, signal_telemetry=True).to_dict())
        replay.capture_ledger = Ledger()
        replay.telemetry = Writer()
        calls = [[] for _ in range(4)]
        def critic(rank):
            async def evaluate(inputs, version):
                calls[rank].append((inputs, version))
                return {'policy_version': version, 'values': [o['value'] + 2. for o in inputs]}
            return SimpleNamespace(evaluate_values=SimpleNamespace(remote=evaluate))
        replay.configure_fresh([critic(rank) for rank in range(4)])
        replay.set_fresh_versions(5, 1)
        for i, size in enumerate(sizes):
            replay.add_segment(segment(str(i), count=size))
        result = await replay.sample(sum(sizes))
        ids = result[-1]['sample_ids']
        unique = list(dict.fromkeys(replay.fresh_store.samples[sid]['segment_id'] for sid in ids))
        segments = [(sid, replay.fresh_store.segments[sid].observations) for sid in unique]
        planned = balanced_refresh_groups(segments, 4)
        for rank, group in enumerate(planned):
            expected = [obs for _, observations in group for obs in observations]
            self.assertEqual(len(calls[rank]), bool(expected))
            if expected:
                self.assertEqual(calls[rank][0][1], 5)
                self.assertEqual(calls[rank][0][0], expected)
        values = {sid: [o['value'] + 2. for o in observations] for sid, observations in segments}
        expected, _ = replay.fresh_store.materialize(ids, values)
        for index in (1, 2, 3, 4):
            np.testing.assert_array_equal(result[index], expected[index])
        plans = [kw['details'] for kind, kw in replay.telemetry.events if kind == 'fresh_critic_refresh_plan']
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0]['policy_version'], 5)
        self.assertEqual(sum(plans[0]['observations_by_replica']), sum(sizes) + len(sizes))
        replay.commit_fresh_use(result[-1]['fresh_reservation_token'], ids)


if __name__ == '__main__':
    unittest.main()
