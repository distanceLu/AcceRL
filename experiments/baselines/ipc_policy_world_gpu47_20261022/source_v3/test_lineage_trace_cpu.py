"""Meaningful CPU contracts for bounded causal DAG telemetry."""
import math
import unittest

from lineage_trace import LineageRecorder


class LineageCPU(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.recorder = LineageRecorder("run_test", "actor:0", lambda kind, details: self.events.append((kind, details)),
                                        max_active_groups=2, max_requests_per_group=3,
                                        max_parents=4, max_recent_closed=2, max_metadata_bytes=256)

    def close(self, stage, parents=(), requests=(), **kwargs):
        group = self.recorder.begin_group(stage, parents, **kwargs)
        for request in requests:
            self.recorder.add_request(group, request)
        self.recorder.end_group(group)
        return group

    def test_success_receipt_has_actual_role_version_workflow(self):
        self.recorder.record_successful_request("request1", "policy", 12, "imagined_action", batch_id="batch1")
        kind, record = self.events[-1]
        self.assertEqual(kind, "lineage_request_succeeded")
        self.assertEqual((record["request_id"], record["role"], record["model_version"], record["workflow"]),
                         ("request1", "policy", 12, "imagined_action"))
        self.assertEqual(record["metadata"], {"batch_id": "batch1"})
        self.assertEqual(self.recorder.cache_state()["successful_request_registry_size"], 0)

    def test_failed_or_unknown_version_not_successful_dependency(self):
        for version, success in ((0, False), (-1, True), (True, True), (None, True)):
            with self.subTest(version=version, success=success):
                with self.assertRaises(ValueError):
                    self.recorder.record_successful_request("r", "policy", version, "evaluation", success=success)
        self.assertEqual(self.events, [])

    def test_direct_requests_deduplicate_and_prefix_stays_an_edge(self):
        parent = self.close("seed", requests=["real_action1"])
        child = self.close("imagined_transition", [parent, parent], ["action2", "world2", "action2"])
        record = self.events[-1][1]
        self.assertEqual(record["group_id"], child)
        self.assertEqual(record["parent_group_ids"], [parent])
        self.assertEqual(record["request_ids"], ["action2", "world2"])
        self.assertNotIn("real_action1", record["request_ids"])
        self.assertEqual(self.recorder.cache_state()["active_request_references"], 0)

    def test_sample_links_shared_prefix_and_suffix_without_flattening(self):
        prefix = self.close("causal_prefix", requests=["action"])
        suffix = self.close("gae_suffix", [prefix], ["bootstrap"])
        self.recorder.sample_producer("sample1", {"state_prefix": prefix, "gae_suffix": [suffix, suffix]},
                                      {"seed_episode_id": "real_episode2", "coverage": "recorded_groups_only"})
        record = self.events[-1][1]
        self.assertEqual(record["stage_groups"], {"state_prefix": [prefix], "gae_suffix": [suffix]})
        self.assertEqual(self.recorder.cache_state()["sample_registry_size"], 0)

    def test_unknown_historical_parent_requires_attestation(self):
        with self.assertRaises(ValueError):
            self.recorder.begin_group("seed", ["other_actor:group1"])
        group = self.close("seed_reference", ["other_actor:group1"], parents_known_by_caller=True,
                           metadata={"coverage": "reference_only"})
        record = self.events[-1][1]
        self.assertEqual(record["parent_validation"]["caller_attested_group_ids"], ["other_actor:group1"])
        self.assertEqual(record["parent_validation"]["recent_local_group_ids"], [])
        self.assertEqual(group, record["group_id"])

    def test_active_parent_cannot_be_used_before_finalization(self):
        group = self.recorder.begin_group("active")
        with self.assertRaises(ValueError):
            self.recorder.begin_group("bad", [group], parents_known_by_caller=True)
        with self.assertRaises(ValueError):
            self.recorder.sample_producer("sample", {"state": group}, parents_known_by_caller=True)

    def test_requests_and_active_groups_are_bounded(self):
        group = self.recorder.begin_group("a")
        self.recorder.begin_group("b")
        with self.assertRaises(RuntimeError):
            self.recorder.begin_group("c")
        for request in ("r1", "r2", "r3"):
            self.recorder.add_request(group, request)
        self.recorder.add_request(group, "r1")
        with self.assertRaises(RuntimeError):
            self.recorder.add_request(group, "r4")
        self.assertEqual(self.recorder.cache_state()["active_request_references"], 3)

    def test_long_run_cache_remains_bounded(self):
        parent = None
        for i in range(10000):
            self.recorder.record_successful_request("request:%d" % i, "ctrl", i, "imagined_generation")
            parent = self.close("prefix", [] if parent is None else [parent], ["request:%d" % i])
            self.recorder.sample_producer("sample:%d" % i, {"transition": parent})
        self.assertEqual(self.recorder.cache_state(), {"active_groups": 0, "active_request_references": 0,
                         "recent_closed_ids": 2, "successful_request_registry_size": 0,
                         "sample_registry_size": 0, "total_groups_started": 10000})

    def test_evicted_local_id_is_not_false_claimed_as_verified(self):
        old = self.close("a")
        self.close("b")
        self.close("c")
        with self.assertRaises(ValueError):
            self.recorder.sample_producer("sample", {"seed": old})
        self.recorder.sample_producer("sample", {"seed": old}, parents_known_by_caller=True)
        self.assertEqual(self.events[-1][1]["group_validation"]["caller_attested_group_ids"], [old])

    def test_abort_does_not_produce_a_closed_dependency(self):
        group = self.recorder.begin_group("will_fail")
        self.recorder.abort_group(group, "RPC failed")
        self.assertEqual(self.recorder.cache_state()["active_groups"], 0)
        self.assertEqual(self.recorder.cache_state()["recent_closed_ids"], 0)
        self.assertEqual(self.events[-1][0], "lineage_group_aborted")

    def test_duplicate_request_events_can_use_existing_client_receipts(self):
        recorder = LineageRecorder("run", "actor", lambda *args: self.events.append(args), emit_request_events=False)
        recorder.record_successful_request("r", "reward", 2, "imagined_generation")
        self.assertEqual(self.events, [])

    def test_metadata_is_finite_bounded_and_copied(self):
        value = {"frame_window": [4, 8]}
        group = self.recorder.begin_group("window", metadata=value)
        value["frame_window"][0] = 100
        self.recorder.end_group(group)
        self.assertEqual(self.events[-1][1]["metadata"]["frame_window"], [4, 8])
        for invalid in ({"v": math.nan}, {"v": object()}, {"v": "x" * 300}):
            with self.assertRaises(ValueError):
                self.recorder.begin_group("invalid", metadata=invalid)


if __name__ == "__main__":
    unittest.main(verbosity=2)
