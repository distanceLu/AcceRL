"""CPU-only worker tests with a fake model and simulator; no Ray/CUDA work."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

from plan_contract import APPROVED_GPUS, load_base, make_shards
import parallel_worker as worker_module
from parallel_worker import RUN_MARKER, assignment, execute_assignment, read_worker_rows, validate_visibility

SOURCE = Path(__file__).resolve().parents[2] / "engine"
BASE = load_base(SOURCE)
TOKEN = "cpu-test-execution-v1"


def payload():
    names = ["vla.lora_A.weight", "vla.lm_head.weight", "attn_pool.0.weight", "value_head.0.weight",
             "proprio_projector.linear.weight"]
    return {"format": BASE.CHECKPOINT_FORMAT,
            "manifest": {"format": BASE.CHECKPOINT_FORMAT, "weights_only": True, "steps": {"policy": 132},
                         "tensor_counts": {"policy": len(names)},
                         "metadata": {"run_id": "test_run", "args": {
                             "pretrained_checkpoint": "/weights/base", "checkpoint2": None,
                             "benchmark": "libero_spatial", "num_images_in_input": 2,
                             "use_proprio": True, "use_bf16": True}}},
            "models": {"policy": {"state_dict": dict.fromkeys(names, "tensor-placeholder"),
                                  "trainable_names": names, "full_state": False}}}


def make_fixture(repeats=1):
    metadata = BASE.validate_checkpoint_payload(payload())
    checkpoints = [{"name": name, "path": "/" + name + "/latest_weights.pt", "sha256": code * 64,
                    "metadata": copy.deepcopy(metadata)} for name, code in (("baseline", "a"), ("final", "b"))]
    plan = BASE.make_plan(checkpoints, repeats=repeats)
    return plan, make_shards(plan)


def row_for(plan, checkpoint, case):
    return {"protocol": BASE.PROTOCOL, "plan_sha256": plan["sha256"], "checkpoint": checkpoint["name"],
            "checkpoint_sha256": checkpoint["sha256"], "policy_version": checkpoint["metadata"]["policy_step"],
            **case, "initial_state_hash": ("%064x" % case["initial_state_id"]), "success": True,
            "steps": 36, "episode_return": 1.0, "termination": "success"}


class FakeRuntime:
    def __init__(self, fail_at=None, close_error=False, load_error=False, mutate=None, prepare_error=False):
        self.fail_at, self.close_error, self.load_error = fail_at, close_error, load_error
        self.mutate, self.prepare_error = mutate, prepare_error
        self.prepared, self.closed, self.releases = 0, 0, 0
        self.loaded, self.calls, self.deadlines = [], [], []

    def prepare(self, plan, deadline, directory, identity):
        self.prepared += 1
        self.deadlines.append(deadline)
        if self.prepare_error:
            raise RuntimeError("fixture prepare failure")

    def load(self, checkpoint, deadline):
        self.loaded.append(checkpoint["name"])
        if self.load_error:
            raise RuntimeError("fixture model restoration failure")

    def run_case(self, case, checkpoint, plan, deadline):
        if len(self.calls) == self.fail_at:
            raise RuntimeError("fixture mid-shard failure")
        self.calls.append((checkpoint["name"], copy.deepcopy(case)))
        row = row_for(plan, checkpoint, case)
        if self.mutate:
            self.mutate(row, plan, checkpoint, case)
        return row

    def release(self):
        self.releases += 1

    def close(self):
        self.closed += 1
        if self.close_error:
            raise RuntimeError("fixture close failure")


class ParallelWorkerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.plan, self.shards = make_fixture()
        self.env = patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": APPROVED_GPUS[0], RUN_MARKER: TOKEN})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def run_worker(self, runtime=None, index=0, deadline=None):
        runtime = runtime or FakeRuntime()
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": APPROVED_GPUS[index]}):
            return execute_assignment(base=BASE, plan=self.plan, shards=self.shards, worker_index=index,
                                      run_token=TOKEN, output_root=self.root,
                                      deadline_unix=deadline if deadline is not None else time.time() + 3600,
                                      runtime=runtime)

    def worker_dir(self, index=0):
        return self.root / "workers" / ("worker_" + str(index))

    def ledger(self, index=0):
        return read_worker_rows(BASE, self.worker_dir(index) / "episodes.jsonl", self.plan,
                                self.shards, index, TOKEN)

    def test_import_has_no_torch_libero_or_ray(self):
        script = ("import sys; import parallel_worker; "
                  "assert not any(m == 'torch' or m.startswith(('torch.', 'ray', 'rl.libero_env')) "
                  "for m in sys.modules); print('CPU_ONLY_IMPORT')")
        result = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).parent,
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CPU_ONLY_IMPORT", result.stdout)

    def test_visibility_requires_one_exact_approved_uuid_and_marker(self):
        work = self.shards["workers"][0]
        for visible in ("", "4", "0", ",".join(APPROVED_GPUS), APPROVED_GPUS[1], APPROVED_GPUS[0] + " "):
            with self.subTest(visible=visible), self.assertRaises(ValueError):
                validate_visibility(work, TOKEN, {"CUDA_VISIBLE_DEVICES": visible, RUN_MARKER: TOKEN})
        for marker in ("", TOKEN + "foreign"):
            with self.subTest(marker=marker), self.assertRaises(ValueError):
                validate_visibility(work, TOKEN, {"CUDA_VISIBLE_DEVICES": APPROVED_GPUS[0], RUN_MARKER: marker})
        self.assertEqual(validate_visibility(work, TOKEN), APPROVED_GPUS[0])

    def test_invalid_worker_indices_are_rejected(self):
        for index in (-1, 4, True, 0.0, "0"):
            with self.subTest(index=index), self.assertRaises(ValueError):
                assignment(self.plan, self.shards, index)

    def test_shared_deadline_must_be_finite(self):
        for deadline in (True, None, float("nan"), float("inf")):
            with self.subTest(deadline=deadline), self.assertRaises(ValueError):
                self.run_worker(deadline=deadline if deadline is not None else "invalid")

    def test_worker_loads_each_checkpoint_once_and_preserves_all_case_fields(self):
        fake = FakeRuntime()
        result = self.run_worker(fake)
        self.assertTrue(result["complete"], result)
        self.assertEqual(result["expected_episode_records"], 52)
        self.assertEqual(fake.loaded, ["baseline", "final"])
        self.assertEqual(fake.prepared, 1)
        self.assertEqual(fake.closed, 1)
        self.assertEqual(fake.releases, 2)
        assigned = self.shards["workers"][0]["assigned_pair_ids"]
        expected = [(checkpoint["name"], case) for checkpoint in self.plan["checkpoints"]
                    for case in self.plan["cases"] if case["pair_id"] in assigned]
        self.assertEqual(fake.calls, expected)
        rows = self.ledger()
        self.assertEqual(len(rows), 52)
        for row in rows:
            self.assertEqual(row["execution_worker_index"], 0)
            self.assertEqual(row["execution_gpu_uuid"], APPROVED_GPUS[0])
            self.assertEqual(row["execution_run_token"], TOKEN)
        for name in ("started.json", "progress.json", "summary.json", "completed.json"):
            value = json.loads((self.worker_dir() / name).read_text())
            self.assertEqual(value["plan_sha256"], self.plan["sha256"])
            self.assertEqual(value["shards_sha256"], self.shards["sha256"])

    def test_four_ledgers_cover_all200_without_cross_worker_case_or_pairing_changes(self):
        all_rows = []
        for index in range(4):
            result = self.run_worker(index=index)
            self.assertTrue(result["complete"], result)
            all_rows.extend(self.ledger(index))
        self.assertEqual(len(all_rows), 200)
        self.assertEqual(len({BASE.row_key(r) for r in all_rows}), 200)
        for state_id in range(50):
            self.assertEqual(len({r["execution_worker_index"] for r in all_rows if r["initial_state_id"] == state_id}), 1)
        self.assertFalse((self.root / "episodes.jsonl").exists())

    def test_complete_reuse_does_not_prepare_model_env_or_change_ledger(self):
        first = self.run_worker()
        self.assertTrue(first["complete"], first)
        before = (self.worker_dir() / "episodes.jsonl").read_bytes()
        fake = FakeRuntime(prepare_error=True)
        result = self.run_worker(fake, deadline=time.time() - 1)
        self.assertTrue(result["reused_complete"])
        self.assertTrue(result["complete"])
        self.assertEqual(fake.prepared, 0)
        self.assertEqual(fake.loaded, [])
        self.assertEqual((self.worker_dir() / "episodes.jsonl").read_bytes(), before)

    def test_midshard_error_flushes_partial_and_resume_runs_only_missing(self):
        failed = FakeRuntime(fail_at=3)
        result = self.run_worker(failed)
        self.assertFalse(result["complete"])
        self.assertEqual(result["exit_code"], 2)
        self.assertEqual(len(self.ledger()), 3)
        self.assertIn("mid-shard", result["errors"][0]["error"])
        self.assertEqual(failed.closed, 1)
        resumed = FakeRuntime()
        complete = self.run_worker(resumed)
        self.assertTrue(complete["complete"])
        self.assertEqual(len(resumed.calls), 49)
        self.assertFalse(set((name, case["pair_id"]) for name, case in failed.calls)
                         & set((name, case["pair_id"]) for name, case in resumed.calls))

    def test_resume_cannot_extend_original_absolute_deadline(self):
        original = time.time() + 300
        self.run_worker(FakeRuntime(fail_at=1), deadline=original)
        fake = FakeRuntime()
        result = self.run_worker(fake, deadline=time.time() + 3600)
        self.assertTrue(result["complete"])
        self.assertEqual(result["deadline_unix"], original)
        self.assertEqual(fake.deadlines, [original])

    def test_expired_partial_does_not_create_runtime(self):
        self.run_worker(FakeRuntime(fail_at=1))
        fake = FakeRuntime()
        result = self.run_worker(fake, deadline=time.time() - 1)
        self.assertFalse(result["complete"])
        self.assertEqual(len(self.ledger()), 1)
        self.assertEqual(fake.prepared, 0)
        self.assertEqual(result["errors"][0]["type"], "TimeoutError")

    def test_prepare_and_load_failures_close_and_publish_incomplete_report(self):
        for kind in ("prepare", "load"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                original = self.root
                self.root = Path(directory)
                fake = FakeRuntime(prepare_error=kind == "prepare", load_error=kind == "load")
                try:
                    result = self.run_worker(fake)
                    self.assertFalse(result["complete"])
                    self.assertEqual(fake.closed, 1)
                    self.assertTrue((self.worker_dir() / "completed.json").exists())
                finally:
                    self.root = original

    def test_environment_close_failure_cannot_claim_complete_or_silently_resume(self):
        result = self.run_worker(FakeRuntime(close_error=True))
        self.assertEqual(len(self.ledger()), 52, result)
        self.assertFalse(result["complete"])
        self.assertEqual(result["errors"][0]["phase"], "close")
        fake = FakeRuntime()
        resumed = self.run_worker(fake)
        self.assertFalse(resumed["complete"])
        self.assertEqual(fake.prepared, 0)

    def test_partial_last_line_repaired_only_in_own_ledger(self):
        self.run_worker(FakeRuntime(fail_at=2))
        path = self.worker_dir() / "episodes.jsonl"
        with path.open("ab") as handle:
            handle.write(b'{"interrupted":')
        other_dir = self.worker_dir(1)
        other_dir.mkdir()
        other = other_dir / "episodes.jsonl"
        other.write_bytes(b'OTHER_WORKER_UNTOUCHED')
        self.assertTrue(self.run_worker()["complete"])
        self.assertEqual(len(self.ledger()), 52)
        self.assertEqual(other.read_bytes(), b'OTHER_WORKER_UNTOUCHED')

    def test_foreign_execution_and_duplicate_rows_reject_before_gpu_work(self):
        self.run_worker(FakeRuntime(fail_at=1))
        path = self.worker_dir() / "episodes.jsonl"
        original = path.read_text()
        for mutation in (lambda r: r.update(execution_worker_index=1),
                         lambda r: r.update(execution_gpu_uuid=APPROVED_GPUS[1]),
                         lambda r: r.update(execution_run_token="another-run"),
                         lambda r: r.update(checkpoint_sha256="f" * 64)):
            row = json.loads(original)
            mutation(row)
            path.write_text(json.dumps(row) + "\n")
            fake = FakeRuntime()
            result = self.run_worker(fake)
            self.assertFalse(result["complete"])
            self.assertEqual(fake.prepared, 0)
        path.write_text(original + original)
        self.assertFalse(self.run_worker()["complete"])

    def test_existing_output_identity_cannot_be_relabelled(self):
        self.run_worker(FakeRuntime(fail_at=1))
        path = self.worker_dir() / "started.json"
        original = path.read_text()
        value = json.loads(original)
        for key in ("run_token", "gpu_uuid", "plan_sha256", "shards_sha256"):
            changed = dict(value, **{key: "foreign"})
            path.write_text(json.dumps(changed))
            fake = FakeRuntime()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.run_worker(fake)
            self.assertEqual(fake.prepared, 0)
        path.write_text(original)

    def test_unowned_ledger_is_never_adopted(self):
        self.worker_dir().mkdir(parents=True)
        (self.worker_dir() / "episodes.jsonl").write_text("")
        with self.assertRaises(ValueError):
            self.run_worker()

    def test_runtime_returning_another_assigned_case_is_rejected(self):
        def change_case(row, plan, checkpoint, case):
            other = next(c for c in plan["cases"] if c["pair_id"] in self.shards["workers"][0]["assigned_pair_ids"]
                         and c["pair_id"] != case["pair_id"])
            row.update(other)
        result = self.run_worker(FakeRuntime(mutate=change_case))
        self.assertFalse(result["complete"])
        self.assertEqual(len(self.ledger()), 0)

    def test_pair_initial_state_mismatch_is_rejected_before_append(self):
        def change_hash(row, plan, checkpoint, case):
            if checkpoint["name"] == "final":
                row["initial_state_hash"] = "f" * 64
        result = self.run_worker(FakeRuntime(mutate=change_hash))
        self.assertFalse(result["complete"])
        self.assertEqual(len(self.ledger()), 26, result)
        self.assertIn("same initial state", result["errors"][0]["error"])

    def test_runtime_preserves_original_episode_call_including_batch_one(self):
        runtime = worker_module.Runtime(BASE, SOURCE, APPROVED_GPUS[0])
        runtime.model, runtime.cfg, runtime.dtype, runtime.env = "model", "config", "dtype", "env"
        case, checkpoint = self.plan["cases"][0], self.plan["checkpoints"][0]
        with patch.object(BASE, "run_episode", return_value={"result": "unchanged"}) as call:
            self.assertEqual(runtime.run_case(case, checkpoint, self.plan, 123), {"result": "unchanged"})
            call.assert_called_once_with("model", "config", "dtype", "env", case, checkpoint, self.plan, 123)
        self.assertEqual(self.plan["inference_batch_size"], 1)

    def test_runtime_rejects_changed_snapshot_before_gpu_probe(self):
        checkpoint = self.plan["checkpoints"][0]
        actual = {key: checkpoint[key] for key in ("path", "sha256", "metadata")}
        for key in ("path", "sha256", "metadata"):
            wrong = copy.deepcopy(actual)
            wrong[key] = "changed"
            runtime = worker_module.Runtime(BASE, SOURCE, APPROVED_GPUS[0])
            with self.subTest(key=key), patch.object(os, "getuid", return_value=1004, create=True), \
                    patch.object(BASE, "load_snapshot", return_value=(payload(), wrong)), \
                    patch.object(worker_module.subprocess, "check_output") as gpu_probe:
                with self.assertRaises(ValueError):
                    runtime.prepare(self.plan, time.time() + 100, self.root, {})
                gpu_probe.assert_not_called()

    def runtime_fakes(self, missing_pool=False, wrong_pool=False, baseline_pool_none=False):
        pool = {"count": 50, "shape": [50, 7], "sha256": "c" * 64}
        example = payload()
        if not missing_pool:
            example["manifest"]["metadata"]["initial_state_queue"] = {"pool": copy.deepcopy(pool)}
            if wrong_pool:
                example["manifest"]["metadata"]["initial_state_queue"]["pool"]["sha256"] = "d" * 64
        env = types.SimpleNamespace(close=lambda: None)
        cuda_calls = []
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(device_count=lambda: 1,
                                           set_device=lambda value: cuda_calls.append(("set_device", value)),
                                           empty_cache=lambda: cuda_calls.append(("empty_cache",)))
        libero = types.ModuleType("rl.libero_env")
        libero.LiberoEnvWrapper = lambda **kwargs: env
        initial = types.ModuleType("initial_state_env")
        initial.initial_state_pool_metadata = lambda value: pool
        def snapshot(path, deadline):
            checkpoint = next(c for c in self.plan["checkpoints"] if Path(c["path"]) == path)
            result = copy.deepcopy(example)
            if baseline_pool_none and checkpoint["name"] == "baseline":
                result["manifest"]["metadata"]["initial_state_queue"] = {"pool": None}
            return result, {**checkpoint, "bytes": 1}
        return pool, torch, libero, initial, snapshot, cuda_calls

    def test_runtime_scopes_gpu_query_and_preserves_both_checkpoint_pool_metadata(self):
        pool, torch, libero, initial, snapshot, calls = self.runtime_fakes()
        runtime = worker_module.Runtime(BASE, SOURCE, APPROVED_GPUS[0])
        with patch.object(os, "getuid", return_value=1004, create=True), \
                patch.object(BASE, "load_snapshot", side_effect=snapshot), \
                patch.object(worker_module.subprocess, "check_output", return_value=APPROVED_GPUS[0] + "\n") as probe, \
                patch.dict(sys.modules, {"torch": torch, "rl.libero_env": libero, "initial_state_env": initial}):
            runtime.prepare(self.plan, time.time() + 100, self.root, {})
            self.assertEqual(sys.path[0], str(SOURCE.resolve()))
            command = probe.call_args.args[0]
            self.assertIn("--id=" + APPROVED_GPUS[0], command)
            self.assertEqual(command[2:], ["--query-gpu=uuid", "--format=csv,noheader,nounits"])
            self.assertEqual(calls, [("set_device", 0)])
            self.assertEqual(json.loads((self.root / "initial_state_pool.json").read_text()), pool)
            self.assertEqual(set(runtime.payloads), {"baseline", "final"})
            runtime.close()
            self.assertEqual(runtime.payloads, {})
            self.assertIsNone(runtime.env)

    def test_runtime_rejects_missing_or_different_saved_pool(self):
        for missing, wrong in ((True, False), (False, True)):
            pool, torch, libero, initial, snapshot, calls = self.runtime_fakes(missing, wrong)
            runtime = worker_module.Runtime(BASE, SOURCE, APPROVED_GPUS[0])
            with self.subTest(missing=missing, wrong=wrong), \
                    patch.object(os, "getuid", return_value=1004, create=True), \
                    patch.object(BASE, "load_snapshot", side_effect=snapshot), \
                    patch.object(worker_module.subprocess, "check_output", return_value=APPROVED_GPUS[0] + "\n"), \
                    patch.dict(sys.modules, {"torch": torch, "rl.libero_env": libero, "initial_state_env": initial}):
                with self.assertRaisesRegex(ValueError, "evidenced"):
                    runtime.prepare(self.plan, time.time() + 100, self.root, {})
                runtime.close()
                self.assertEqual(runtime.payloads, {})

    def test_baseline_pool_none_before_first_rollout_is_legal_with_valid_final_pool(self):
        pool, torch, libero, initial, snapshot, calls = self.runtime_fakes(baseline_pool_none=True)
        runtime = worker_module.Runtime(BASE, SOURCE, APPROVED_GPUS[0])
        with patch.object(os, "getuid", return_value=1004, create=True), \
                patch.object(BASE, "load_snapshot", side_effect=snapshot), \
                patch.object(worker_module.subprocess, "check_output", return_value=APPROVED_GPUS[0] + "\n"), \
                patch.dict(sys.modules, {"torch": torch, "rl.libero_env": libero, "initial_state_env": initial}):
            runtime.prepare(self.plan, time.time() + 100, self.root, {})
            self.assertIsNone(runtime.payloads["baseline"]["manifest"]["metadata"]["initial_state_queue"]["pool"])
            self.assertEqual(json.loads((self.root / "initial_state_pool.json").read_text()), pool)
            runtime.close()

    def test_missing_final_pool_rejected_before_gpu_probe(self):
        pool, torch, libero, initial, snapshot, calls = self.runtime_fakes(missing_pool=True)
        runtime = worker_module.Runtime(BASE, SOURCE, APPROVED_GPUS[0])
        with patch.object(os, "getuid", return_value=1004, create=True), \
                patch.object(BASE, "load_snapshot", side_effect=snapshot), \
                patch.object(worker_module.subprocess, "check_output") as probe:
            with self.assertRaisesRegex(ValueError, "final checkpoint"):
                runtime.prepare(self.plan, time.time() + 100, self.root, {})
            probe.assert_not_called()
            runtime.close()

    def test_runtime_delegates_original_checkpoint_signature_restoration(self):
        runtime = worker_module.Runtime(BASE, SOURCE, APPROVED_GPUS[0])
        runtime.payloads["baseline"] = payload()
        checkpoint = self.plan["checkpoints"][0]
        with patch.object(BASE, "restore_policy", return_value=("model", "cfg", "dtype")) as restore:
            runtime.load(checkpoint, 123)
            restore.assert_called_once_with(runtime.payloads["baseline"], 123)
        self.assertEqual((runtime.model, runtime.cfg, runtime.dtype), ("model", "cfg", "dtype"))
        with self.assertRaises(ValueError):
            runtime.load(checkpoint, 123)

    def test_runtime_releases_cpu_references_even_if_environment_close_fails(self):
        runtime = worker_module.Runtime(BASE, SOURCE, APPROVED_GPUS[0])
        runtime.model, runtime.payloads = "model", {"baseline": payload()}
        def fail_close():
            raise RuntimeError("simulator close fixture")
        runtime.env = types.SimpleNamespace(close=fail_close)
        with self.assertRaises(RuntimeError):
            runtime.close()
        self.assertIsNone(runtime.env)
        self.assertIsNone(runtime.model)
        self.assertEqual(runtime.payloads, {})

    def test_exclusive_worker_claim_prevents_duplicate_writers_and_releases_normally(self):
        with worker_module.worker_claim(self.root, 0, TOKEN) as claim:
            self.assertTrue(claim.exists())
            with self.assertRaises(FileExistsError):
                with worker_module.worker_claim(self.root, 0, TOKEN):
                    self.fail("Duplicate writer entered the same ledger")
        self.assertFalse(claim.exists())

    def test_stale_worker_claim_is_refused_and_preserved(self):
        directory = self.worker_dir()
        directory.mkdir(parents=True)
        claim = directory / "active_worker.json"
        claim.write_text('{"stale":"owner-is-unknown"}')
        with self.assertRaises(FileExistsError):
            with worker_module.worker_claim(self.root, 0, TOKEN):
                self.fail("Stale claim must not be automatically removed")
        self.assertEqual(claim.read_text(), '{"stale":"owner-is-unknown"}')

    def test_worker_claim_released_on_handled_exception(self):
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with worker_module.worker_claim(self.root, 0, TOKEN) as claim:
                raise RuntimeError("fixture")
        self.assertFalse(claim.exists())

    @staticmethod
    def replace_error(path, winerror=5, *, second_path=True):
        error = PermissionError("fixture Windows replace sharing failure")
        error.winerror = winerror
        error.filename = str(path.with_name(path.name + ".tmp"))
        error.filename2 = str(path) if second_path else None
        return error

    def test_windows_replace_retry_is_bounded_and_retains_atomic_result(self):
        path = self.root / "progress.json"
        path.write_text('{"previous":true}')
        original = BASE.atomic_json
        attempts = []
        def write(target, value):
            attempts.append(str(target))
            if len(attempts) < 3:
                raise self.replace_error(target, 32)
            original(target, value)
        with patch.object(worker_module, "WINDOWS_REPLACE_RETRY", True), \
                patch.object(BASE, "atomic_json", side_effect=write), \
                patch.object(worker_module.time, "sleep") as sleep:
            self.assertEqual(worker_module.atomic_worker_json(BASE, path, {"final": True}), 2)
        self.assertEqual(len(attempts), 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [0.01, 0.03])
        self.assertEqual(json.loads(path.read_text()), {"final": True})

    def test_persistent_windows_replace_error_is_not_swallowed(self):
        path = self.root / "progress.json"
        error = self.replace_error(path)
        with patch.object(worker_module, "WINDOWS_REPLACE_RETRY", True), \
                patch.object(BASE, "atomic_json", side_effect=error) as write, \
                patch.object(worker_module.time, "sleep") as sleep:
            with self.assertRaises(PermissionError) as raised:
                worker_module.atomic_worker_json(BASE, path, {"value": 1})
        self.assertIs(raised.exception, error)
        self.assertEqual(write.call_count, 4)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [0.01, 0.03, 0.09])

    def test_non_windows_permission_error_is_never_retried(self):
        path = self.root / "progress.json"
        with patch.object(worker_module, "WINDOWS_REPLACE_RETRY", False), \
                patch.object(BASE, "atomic_json", side_effect=self.replace_error(path)) as write, \
                patch.object(worker_module.time, "sleep") as sleep:
            with self.assertRaises(PermissionError):
                worker_module.atomic_worker_json(BASE, path, {})
        self.assertEqual(write.call_count, 1)
        sleep.assert_not_called()

    def test_open_and_unrelated_windows_errors_are_never_retried(self):
        path = self.root / "progress.json"
        for error in (PermissionError("unknown permission failure"),
                      self.replace_error(path, second_path=False),
                      self.replace_error(path, winerror=87),
                      self.replace_error(self.root / "another.json")):
            with self.subTest(error=repr(error)), \
                    patch.object(worker_module, "WINDOWS_REPLACE_RETRY", True), \
                    patch.object(BASE, "atomic_json", side_effect=error) as write, \
                    patch.object(worker_module.time, "sleep") as sleep:
                with self.assertRaises(PermissionError):
                    worker_module.atomic_worker_json(BASE, path, {})
                self.assertEqual(write.call_count, 1)
                sleep.assert_not_called()

    def test_transient_progress_replace_error_does_not_repeat_episode(self):
        original = BASE.atomic_json
        progress_failures = []
        def write(path, value):
            if path.name == "progress.json" and not progress_failures:
                progress_failures.append(str(path))
                raise self.replace_error(path)
            original(path, value)
        fake = FakeRuntime()
        with patch.object(worker_module, "WINDOWS_REPLACE_RETRY", True), \
                patch.object(BASE, "atomic_json", side_effect=write), \
                patch.object(worker_module.time, "sleep"):
            result = self.run_worker(fake)
        self.assertTrue(result["complete"], result)
        self.assertEqual(len(progress_failures), 1)
        self.assertEqual(len(fake.calls), 52)
        self.assertEqual(len(self.ledger()), 52)
        self.assertEqual(len(set(BASE.row_key(row) for row in self.ledger())), 52)

    def test_persistent_progress_replace_error_keeps_durable_prefix_and_fails(self):
        original = BASE.atomic_json
        progress_attempts = []
        def write(path, value):
            if path.name == "progress.json" and value["completed_episode_records"] == 19:
                progress_attempts.append(str(path))
                raise self.replace_error(path)
            original(path, value)
        fake = FakeRuntime()
        with patch.object(worker_module, "WINDOWS_REPLACE_RETRY", True), \
                patch.object(BASE, "atomic_json", side_effect=write), \
                patch.object(worker_module.time, "sleep"):
            result = self.run_worker(fake)
        self.assertFalse(result["complete"])
        self.assertEqual(result["errors"][0]["type"], "PermissionError")
        self.assertEqual(len(progress_attempts), 4)
        self.assertEqual(len(fake.calls), 19)
        self.assertEqual(len(self.ledger()), 19)
        self.assertEqual(fake.closed, 1)

    def test_non_permission_write_failures_are_not_retried(self):
        path = self.root / "progress.json"
        for error in (TypeError("fixture serialization"), OSError("fixture fsync")):
            with self.subTest(error=repr(error)), \
                    patch.object(BASE, "atomic_json", side_effect=error) as write, \
                    patch.object(worker_module.time, "sleep") as sleep:
                with self.assertRaises(type(error)):
                    worker_module.atomic_worker_json(BASE, path, {})
                self.assertEqual(write.call_count, 1)
                sleep.assert_not_called()

    def test_main_can_load_external_reference_and_keeps_actual_source_first(self):
        plan_path, shards_path = self.root / "plan.json", self.root / "shards.json"
        plan_path.write_text(json.dumps(self.plan), encoding="utf-8")
        shards_path.write_text(json.dumps(self.shards), encoding="utf-8")
        reference = self.root / "reference"
        reference.mkdir()
        for external_reference in (False, True):
            args = ["--source", str(SOURCE), "--plan", str(plan_path), "--shards", str(shards_path),
                    "--worker-index", "0", "--output-root", str(self.root),
                    "--deadline-unix", str(time.time() + 300), "--run-token", TOKEN]
            if external_reference:
                args += ["--reference-source", str(reference)]
            result = {"worker_index": 0, "complete": True, "completed_episode_records": 52,
                      "expected_episode_records": 52, "errors": []}
            with self.subTest(external_reference=external_reference), \
                    patch.object(worker_module, "load_base", return_value=BASE) as load, \
                    patch.object(worker_module, "Runtime") as runtime, \
                    patch.object(worker_module, "execute_assignment", return_value=result) as execute, \
                    patch.object(worker_module.signal, "signal"), \
                    patch.object(sys, "path", list(sys.path)), patch("builtins.print"):
                self.assertEqual(worker_module.main(args), 0)
                load.assert_called_once_with(reference if external_reference else SOURCE)
                runtime.assert_called_once_with(BASE, SOURCE, APPROVED_GPUS[0])
                self.assertEqual(execute.call_args.kwargs["base"], BASE)
                self.assertEqual(sys.path[0], str(SOURCE.resolve()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
