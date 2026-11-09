"""CPU-only exact-process cleanup and four-worker result admission tests."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import parallel_eval as supervisor
import plan_contract as contract
from test_parallel_contract_cpu import BASE, TOKEN, make_plan, worker_ledgers


def completion_reports(plan, shards):
    return {worker["worker_index"]: {
        "plan_sha256": plan["sha256"], "shards_sha256": shards["sha256"], "run_token": TOKEN,
        "worker_index": worker["worker_index"], "gpu_uuid": worker["gpu_uuid"],
        "complete": True, "errors": [], "exit_code": 0,
        "expected_episode_records": worker["expected_episode_records"],
        "completed_episode_records": worker["expected_episode_records"],
    } for worker in shards["workers"]}


class ExactProcessIdentityTests(unittest.TestCase):
    def setUp(self):
        self.identity = {"pid": 81234, "uid": 1004, "birth": 24133742, "run_token": TOKEN}

    def test_same_exact_process_and_execution_token_matches(self):
        self.assertTrue(supervisor.same_identity(self.identity, dict(self.identity), TOKEN))

    def test_reused_pid_different_uid_birth_or_token_does_not_match(self):
        for field, value in (("pid", 81235), ("uid", 1005), ("birth", 24133743),
                             ("run_token", "another-execution")):
            actual = dict(self.identity)
            actual[field] = value
            with self.subTest(field=field):
                self.assertFalse(supervisor.same_identity(self.identity, actual, TOKEN))

    def test_unavailable_or_partial_identity_fails_closed(self):
        for actual in (None, {}, {"pid": 81234}, {"pid": 81234, "uid": 1004, "birth": 24133742}):
            with self.subTest(actual=actual):
                self.assertFalse(supervisor.same_identity(self.identity, actual, TOKEN))

    def test_expected_identity_is_not_allowed_to_omit_an_exact_field(self):
        for field in self.identity:
            expected = dict(self.identity)
            expected.pop(field)
            with self.subTest(field=field):
                self.assertFalse(supervisor.same_identity(expected, self.identity, TOKEN))

    def test_bool_float_or_string_process_numbers_cannot_alias_integer_identity(self):
        for field in ("pid", "uid", "birth"):
            for value in (True, float(self.identity[field]), str(self.identity[field])):
                actual = dict(self.identity)
                actual[field] = value
                with self.subTest(field=field, value=value):
                    self.assertFalse(supervisor.same_identity(self.identity, actual, TOKEN))

    def test_empty_or_foreign_cleanup_token_fails_closed(self):
        for token in ("", None, "foreign-token"):
            with self.subTest(token=token):
                self.assertFalse(supervisor.same_identity(self.identity, self.identity, token))

    def test_signal_exact_sends_only_to_the_confirmed_pid(self):
        calls = []
        probes = []

        def probe(pid):
            probes.append(pid)
            return dict(self.identity)

        def send(pid, number):
            calls.append((pid, number))

        self.assertTrue(supervisor.signal_exact(self.identity, TOKEN, probe, send, 15))
        self.assertEqual(probes, [81234])
        self.assertEqual(calls, [(81234, 15)])

    def test_signal_exact_refuses_changed_identity_without_sending(self):
        for change in ({"birth": 999}, {"uid": 999}, {"run_token": "foreign"}):
            actual = {**self.identity, **change}
            sent = []
            with self.subTest(change=change):
                self.assertFalse(supervisor.signal_exact(self.identity, TOKEN, lambda pid: actual,
                                                          lambda pid, number: sent.append((pid, number)), 9))
                self.assertEqual(sent, [])

    def test_signal_exact_refuses_missing_identity_without_sending(self):
        sent = []
        self.assertFalse(supervisor.signal_exact(self.identity, TOKEN, lambda pid: None,
                                                  lambda pid, number: sent.append((pid, number)), 15))
        self.assertEqual(sent, [])

    def test_signal_exact_reprobes_before_each_signal_to_reject_pid_reuse(self):
        sent = []
        actual = dict(self.identity)

        def probe(pid):
            return dict(actual)

        self.assertTrue(supervisor.signal_exact(self.identity, TOKEN, probe,
                                                lambda pid, number: sent.append((pid, number)), 15))
        actual["birth"] += 1
        self.assertFalse(supervisor.signal_exact(self.identity, TOKEN, probe,
                                                 lambda pid, number: sent.append((pid, number)), 9))
        self.assertEqual(sent, [(81234, 15)])


class CompletionAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.plan = make_plan()
        self.shards = contract.make_shards(self.plan)
        self.rows = worker_ledgers(self.plan, self.shards)
        self.completions = completion_reports(self.plan, self.shards)
        self.codes = {index: 0 for index in range(4)}
        self.cleanup = {"survivors": [], "errors": []}

    def admit(self, **kwargs):
        values = {"plan": self.plan, "shards": self.shards, "worker_rows": self.rows,
                  "worker_completions": self.completions, "exit_codes": self.codes,
                  "cleanup": self.cleanup, "run_token": TOKEN, "base": BASE}
        values.update(kwargs)
        return supervisor.completion_acceptance(**values)

    def assert_incomplete(self, **kwargs):
        result = self.admit(**kwargs)
        self.assertIs(result["complete"], False)
        self.assertTrue(result["errors"], result)
        return result

    def test_complete_requires_all_four_reports_all_two_hundred_rows_and_clean_exit(self):
        result = self.admit()
        self.assertIs(result["complete"], True)
        self.assertEqual(result["errors"], [])

    def test_missing_worker_completion_cannot_pass_despite_all_episode_rows(self):
        self.completions.pop(2)
        self.assert_incomplete()

    def test_one_nonzero_process_exit_fails_even_if_report_claims_success(self):
        self.codes[1] = 1
        self.assert_incomplete()

    def test_missing_process_exit_is_unknown_not_success(self):
        self.codes.pop(3)
        self.assert_incomplete()

    def test_exit_zero_does_not_prove_full_coverage(self):
        self.rows[0].pop()
        self.assert_incomplete()

    def test_report_with_wrong_plan_shard_execution_or_gpu_identity_fails(self):
        for field, value in (("plan_sha256", "0" * 64), ("shards_sha256", "1" * 64),
                             ("run_token", "another-execution"), ("worker_index", 2),
                             ("gpu_uuid", self.shards["workers"][1]["gpu_uuid"])):
            reports = copy.deepcopy(self.completions)
            reports[0][field] = value
            with self.subTest(field=field):
                self.assert_incomplete(worker_completions=reports)

    def test_false_incomplete_or_error_completion_report_fails(self):
        for field, value in (("complete", False), ("errors", [{"error": "worker_failure"}]),
                             ("exit_code", 1), ("expected_episode_records", 0),
                             ("completed_episode_records", 51)):
            reports = copy.deepcopy(self.completions)
            reports[0][field] = value
            with self.subTest(field=field):
                self.assert_incomplete(worker_completions=reports)

    def test_boolean_counts_and_exit_status_do_not_alias_integer_values(self):
        for field, value in (("worker_index", False), ("exit_code", False)):
            reports = copy.deepcopy(self.completions)
            reports[0][field] = value
            with self.subTest(field=field):
                self.assert_incomplete(worker_completions=reports)

    def test_complete_string_cannot_claim_success(self):
        self.completions[0]["complete"] = "true"
        self.assert_incomplete()

    def test_missing_required_report_field_fails(self):
        for field in tuple(self.completions[0]):
            reports = copy.deepcopy(self.completions)
            reports[0].pop(field)
            with self.subTest(field=field):
                self.assert_incomplete(worker_completions=reports)

    def test_cleanup_survivor_or_error_prevents_completion(self):
        for cleanup in ({"survivors": [{"pid": 81234}], "errors": []},
                        {"survivors": [], "errors": [{"error": "identity_unknown"}]}):
            with self.subTest(cleanup=cleanup):
                self.assert_incomplete(cleanup=cleanup)

    def test_missing_or_unavailable_cleanup_evidence_cannot_be_clean(self):
        for cleanup in ({}, None):
            with self.subTest(cleanup=cleanup):
                self.assert_incomplete(cleanup=cleanup)

    def test_duplicate_corrupt_foreign_or_wrong_device_row_fails_without_throwing(self):
        for mode in ("duplicate", "corrupt", "foreign", "wrong_device"):
            rows = copy.deepcopy(self.rows)
            if mode == "duplicate":
                rows[0].append(copy.deepcopy(rows[0][0]))
            elif mode == "corrupt":
                rows[0][0]["success"] = "true"
            elif mode == "foreign":
                rows[0][0]["execution_run_token"] = "foreign"
            else:
                rows[0][0]["execution_gpu_uuid"] = self.shards["workers"][1]["gpu_uuid"]
            with self.subTest(mode=mode):
                self.assert_incomplete(worker_rows=rows)

    def test_paired_initial_state_hash_mismatch_cannot_pass_completion(self):
        row = next(value for value in self.rows[0] if value["checkpoint"] == "final")
        row["initial_state_hash"] = "f" * 64
        self.assert_incomplete()

    def test_no_worker_row_dictionary_entry_can_be_silently_omitted(self):
        self.rows.pop(1)
        self.assert_incomplete()

    def test_supervisor_admission_does_not_mutate_input_evidence(self):
        before = copy.deepcopy((self.plan, self.shards, self.rows, self.completions, self.codes, self.cleanup))
        self.admit()
        self.assertEqual(before, (self.plan, self.shards, self.rows, self.completions, self.codes, self.cleanup))


class CoordinatorCollectionTests(unittest.TestCase):
    def setUp(self):
        self.plan = make_plan()
        self.shards = contract.make_shards(self.plan)
        self.rows = worker_ledgers(self.plan, self.shards)
        self.reports = completion_reports(self.plan, self.shards)

    def write_evidence(self, output):
        for index in range(4):
            directory = output / "workers" / ("worker_" + str(index))
            directory.mkdir(parents=True)
            rows = self.rows[index]
            if index == 0:
                rows = rows[:-1]
            (directory / "episodes.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            BASE.atomic_json(directory / "completed.json", self.reports[index])
        ledger = output / "workers" / "worker_0" / "episodes.jsonl"
        with ledger.open("ab") as handle:
            handle.write(b'{"interrupted_last_row":')
        return ledger

    @staticmethod
    def children(*, running=False):
        return {index: SimpleNamespace(returncode=None if running and index == 0 else (2 if index == 0 else 0),
                                       poll=lambda index=index: None if running and index == 0 else (2 if index == 0 else 0))
                for index in range(4)}

    def test_exited_workers_allow_only_partial_tail_repair_preserving_durable_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            ledger = self.write_evidence(output)
            result = supervisor.collect(BASE, output, self.plan, self.shards, TOKEN,
                                        self.children(), {"survivors": [], "errors": []}, [])
            self.assertIs(result["complete"], False)
            self.assertEqual(result["completed_episode_records"], 199)
            self.assertEqual(len(BASE.read_rows(ledger, self.plan)), 51)
            self.assertTrue(ledger.read_bytes().endswith(b"\n"))
            self.assertEqual(len(BASE.read_rows(output / "episodes.jsonl", self.plan)), 199)

    def test_cleanup_survivors_prevent_any_worker_ledger_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            ledger = self.write_evidence(output)
            original = ledger.read_bytes()
            result = supervisor.collect(BASE, output, self.plan, self.shards, TOKEN,
                                        self.children(), {"survivors": [{"pid": 81234}], "errors": []}, [])
            self.assertIs(result["complete"], False)
            self.assertEqual(ledger.read_bytes(), original)

    def test_running_child_prevents_partial_tail_repair_even_with_empty_cleanup_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            ledger = self.write_evidence(output)
            original = ledger.read_bytes()
            result = supervisor.collect(BASE, output, self.plan, self.shards, TOKEN,
                                        self.children(running=True), {"survivors": [], "errors": []}, [])
            self.assertIs(result["complete"], False)
            self.assertEqual(ledger.read_bytes(), original)


class StoppedAdmissionTests(unittest.TestCase):
    @staticmethod
    def adapter(marked):
        receipt = {"run_id": "cpu-stopped-run", "state": "succeeded", "all_owned_exited": True,
                   "job_id":"cpu-reservation", "runner_exit_code":0,
                   "final_source_validation":{"git_commit":"a"*40},
                   "acceptance": {"passed": True,"reasons":[]}, "checkpoint_cpu_audit": {"valid": True}}
        identities = {100: {"pid": 100, "ppid": 200}, 200: {"pid": 200, "ppid": 1}}
        return SimpleNamespace(
            read_json=lambda path: receipt if path.name == "training_queue_receipt.json" else {},
            validate_mps_receipt=mock.Mock(), proc_identity=lambda pid: identities.get(pid),
            marked_children=mock.Mock(return_value=marked), launch_environment=mock.Mock(return_value={}),
            wait_idle=mock.Mock())

    def test_only_current_coordinator_and_verified_ancestors_are_excluded(self):
        adapter = self.adapter([{"pid": 100}, {"pid": 200}])
        with mock.patch.object(supervisor.os, "getuid", return_value=1004, create=True), \
                mock.patch.object(supervisor.os, "getpid", return_value=100), \
                mock.patch.dict(os.environ, {"EXPERIMENT_QUEUE_JOB_ID": "cpu-reservation"}, clear=True):
            env = supervisor.stopped_admission(adapter, Path("/master"), Path("/source"), "cpu-stopped-run", "a"*40)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], ",".join(contract.APPROVED_GPUS))
        adapter.wait_idle.assert_called_once()

    def test_any_nonancestor_training_or_evaluation_process_blocks_admission(self):
        adapter = self.adapter([{"pid": 100}, {"pid": 200}, {"pid": 900}])
        with mock.patch.object(supervisor.os, "getuid", return_value=1004, create=True), \
                mock.patch.object(supervisor.os, "getpid", return_value=100), \
                mock.patch.dict(os.environ, {"EXPERIMENT_QUEUE_JOB_ID": "cpu-reservation"}, clear=True), \
                self.assertRaisesRegex(ValueError, "processes remain"):
            supervisor.stopped_admission(adapter, Path("/master"), Path("/source"), "cpu-stopped-run", "a"*40)
        adapter.wait_idle.assert_not_called()
        adapter.launch_environment.assert_not_called()

    def test_resume_refuses_live_execution_token_before_launching_more_workers(self):
        plan = make_plan()
        shards = contract.make_shards(plan)
        adapter = SimpleNamespace(marked_children=mock.Mock(return_value=[{"pid": 900}]))
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "parallel"
            output.mkdir()
            BASE.atomic_json(output / "started.json", {
                "deadline_unix": time.time() + 120, "started_ns": time.time_ns(), "run_token": TOKEN,
                "run_id": "cpu-stopped-run", "queue_job_id": "cpu-reservation",
                "plan_sha256": plan["sha256"], "shards_sha256": shards["sha256"]})
            with mock.patch.object(supervisor, "load_base", return_value=BASE), \
                    mock.patch.object(supervisor, "prepare_plan", return_value=plan), \
                    mock.patch.object(supervisor, "load_adapter", return_value=adapter), \
                    mock.patch.object(supervisor, "stopped_admission", return_value={}), \
                    mock.patch.object(supervisor, "run_workers") as launch, \
                    mock.patch.dict(os.environ, {"EXPERIMENT_QUEUE_JOB_ID": "cpu-reservation"}, clear=True), \
                    self.assertRaisesRegex(ValueError, "still alive"):
                supervisor.main(["--source", tmp, "--queue-wrapper", tmp, "--master", tmp,
                                 "--output-root", str(output), "--run-id", "cpu-stopped-run",
                                 "--git-commit", "a"*40, "--resume"])
            launch.assert_not_called()
            adapter.marked_children.assert_called_once_with(TOKEN, marker_name=supervisor.RUN_MARKER)
            self.assertFalse((output / "coordinator.lock").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
