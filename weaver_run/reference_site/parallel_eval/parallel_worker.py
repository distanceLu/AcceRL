"""One independent GPU worker for the immutable paired Policy evaluation plan.

Importing this module uses only the standard library.  Torch, Policy and LIBERO
are loaded by Runtime.prepare(), after the shard and execution identities pass.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import traceback

from plan_contract import APPROVED_GPUS, load_base, validate_shards, validate_worker_row

WORKER_PROTOCOL = "accerl_parallel_eval_worker_v1"
RUN_MARKER = "PARALLEL_POLICY_EVAL_RUN_TOKEN"
WINDOWS_REPLACE_RETRY = os.name == "nt"
REPLACE_RETRY_DELAYS = (0.01, 0.03, 0.09)


def atomic_worker_json(base, path, value):
    """Retry only bounded Windows sharing/replace failures from the atomic rename.

    The worker owns its output claim.  A Windows reader can briefly prevent
    replacing an existing progress file even after our temporary file is closed.
    Do not retry serialization/open/fsync failures, arbitrary PermissionError,
    or any Linux error.  Persistent rename failures remain evaluation failures.
    """
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    for attempt in range(len(REPLACE_RETRY_DELAYS) + 1):
        try:
            base.atomic_json(path, value)
            return attempt
        except PermissionError as exc:
            replace_failed = (WINDOWS_REPLACE_RETRY and getattr(exc, "winerror", None) in (5, 32, 33)
                              and exc.filename is not None and exc.filename2 is not None
                              and os.path.normcase(os.fspath(exc.filename)) == os.path.normcase(str(temporary))
                              and os.path.normcase(os.fspath(exc.filename2)) == os.path.normcase(str(path)))
            if not replace_failed or attempt == len(REPLACE_RETRY_DELAYS):
                raise
            time.sleep(REPLACE_RETRY_DELAYS[attempt])


def assignment(plan, shards, worker_index):
    validate_shards(plan, shards)
    if type(worker_index) is not int or not 0 <= worker_index < 4:
        raise ValueError("Worker index must be an integer from 0 through 3")
    return shards["workers"][worker_index]


def validate_visibility(worker, run_token, environ=None):
    """CPU admission, before any CUDA module import or device discovery."""
    environ = os.environ if environ is None else environ
    if not isinstance(run_token, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", run_token) is None:
        raise ValueError("A bounded, explicit execution run token is required")
    gpu_uuid = worker["gpu_uuid"]
    if gpu_uuid not in APPROVED_GPUS or environ.get("CUDA_VISIBLE_DEVICES") != gpu_uuid:
        raise ValueError("CUDA_VISIBLE_DEVICES must equal this worker's one approved GPU UUID")
    if environ.get(RUN_MARKER) != run_token:
        raise ValueError("Worker execution marker differs from its coordinator run token")
    return gpu_uuid


def process_identity():
    identity = {"pid": os.getpid(), "ppid": os.getppid(),
                "uid": os.getuid() if hasattr(os, "getuid") else None,
                "hostname": socket.gethostname(), "birth_ticks": None,
                "boot_id": None, "marker": os.environ.get(RUN_MARKER)}
    stat_path = Path("/proc/self/stat")
    if stat_path.exists():
        # The executable name can contain spaces or parentheses.
        identity["birth_ticks"] = int(stat_path.read_text().rsplit(") ", 1)[1].split()[19])
        identity["boot_id"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    return identity


@contextmanager
def worker_claim(output_root, worker_index, run_token):
    """A stale or concurrent writer is refused; only its creator releases a claim."""
    directory = Path(output_root) / "workers" / ("worker_" + str(worker_index))
    for candidate in (Path(output_root), Path(output_root) / "workers", directory):
        if candidate.is_symlink():
            raise ValueError("Worker output directories must not be symlinks")
    directory.mkdir(parents=True, exist_ok=True)
    claim = directory / "active_worker.json"
    value = {"run_token": run_token, "worker_index": worker_index,
             "process": process_identity(), "created_ns": time.time_ns()}
    # Exclusive creation is the guard; checking existence before writing races.
    with claim.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    original_inode = claim.stat().st_ino
    try:
        yield claim
    finally:
        if (claim.is_symlink() or not claim.exists() or claim.stat().st_ino != original_inode
                or json.loads(claim.read_text(encoding="utf-8")) != value):
            raise ValueError("Worker claim changed; refusing to remove another process's claim")
        claim.unlink()


def _identity_fields(plan, shards, worker, run_token):
    return {"protocol": WORKER_PROTOCOL, "worker_index": worker["worker_index"],
            "gpu_uuid": worker["gpu_uuid"], "run_token": run_token,
            "plan_sha256": plan["sha256"], "shards_sha256": shards["sha256"]}


def _same_identity(value, expected):
    if not isinstance(value, dict) or any(value.get(k) != v for k, v in expected.items()):
        raise ValueError("Existing worker output belongs to another plan, shard, GPU or execution")


def _finite_deadline(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("A finite absolute shared deadline is required")
    return float(value)


def _check_pairs(rows):
    initial_hashes = {}
    for row in rows:
        previous = initial_hashes.setdefault(row["pair_id"], row["initial_state_hash"])
        if previous != row["initial_state_hash"]:
            raise ValueError("Paired checkpoints did not reset to the same initial state")


def read_worker_rows(base, path, plan, shards, worker_index, run_token, repair_partial_tail=False):
    rows = base.read_rows(path, plan, repair_partial_tail=repair_partial_tail)
    for row in rows:
        validate_worker_row(row, plan, shards, worker_index, run_token, base=base)
    _check_pairs(rows)
    return rows


def execute_assignment(*, base, plan, shards, worker_index, run_token, output_root,
                       deadline_unix, runtime):
    """Execute only this shard; runtime is injectable for CPU contract tests.

    Runtime methods: prepare(plan, deadline, worker_dir, identity), load(checkpoint,
    deadline), run_case(case, checkpoint, plan, deadline), release(), close().
    Each method may fail; partial JSONL rows are already fsynced when it does.
    """
    worker = assignment(plan, shards, worker_index)
    validate_visibility(worker, run_token)
    fields = _identity_fields(plan, shards, worker, run_token)
    deadline = _finite_deadline(deadline_unix)
    directory = Path(output_root) / "workers" / ("worker_" + str(worker_index))
    # Never follow a worker directory or ledger symlink into another output.
    for candidate in (Path(output_root), Path(output_root) / "workers", directory):
        if candidate.is_symlink():
            raise ValueError("Worker output directories must not be symlinks")
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("started.json", "progress.json", "summary.json", "completed.json", "episodes.jsonl"):
        if (directory / name).is_symlink():
            raise ValueError("Worker output files must not be symlinks")
    started_path, rows_path = directory / "started.json", directory / "episodes.jsonl"
    completed_path = directory / "completed.json"
    process = process_identity()
    if started_path.exists():
        started = json.loads(started_path.read_text(encoding="utf-8"))
        _same_identity(started, fields)
        deadline = min(deadline, _finite_deadline(started.get("deadline_unix")))
    else:
        if rows_path.exists() or completed_path.exists():
            raise ValueError("Existing worker ledger has no admitted execution identity")
        started = {**fields, "started_ns": time.time_ns(), "deadline_unix": deadline,
                   "process": process}
        atomic_worker_json(base, started_path, started)

    previous_completion = None
    if completed_path.exists():
        previous_completion = json.loads(completed_path.read_text(encoding="utf-8"))
        _same_identity(previous_completion, fields)

    assigned = set(worker["assigned_pair_ids"])
    expected_keys = {(checkpoint["name"], pair_id) for checkpoint in plan["checkpoints"] for pair_id in assigned}
    rows, errors = [], []
    attempted_runtime = False
    phase = "read_ledger"
    try:
        rows = read_worker_rows(base, rows_path, plan, shards, worker_index, run_token,
                                repair_partial_tail=True)
        completed_keys = {base.row_key(row) for row in rows}
        if previous_completion and previous_completion.get("complete") is True:
            if (completed_keys != expected_keys or previous_completion.get("errors")
                    or previous_completion.get("deadline_unix") != started["deadline_unix"]
                    or previous_completion.get("completed_episode_records") != len(expected_keys)):
                raise ValueError("Completed worker claim differs from its durable assigned ledger")
            # A completed shard remains reusable after the evaluation deadline.
            return {**previous_completion, "reused_complete": True}
        if completed_keys == expected_keys and previous_completion and previous_completion.get("errors"):
            raise ValueError("Full ledger has a previous failed completion; review that failure before reuse")
        phase = "prepare"
        base.check_deadline(deadline, 10)
        if completed_keys != expected_keys:
            attempted_runtime = True
            runtime.prepare(plan, deadline, directory, fields)
        for checkpoint in plan["checkpoints"]:
            pending = [case for case in plan["cases"]
                       if case["pair_id"] in assigned
                       and (checkpoint["name"], case["pair_id"]) not in completed_keys]
            if not pending:
                continue
            phase = "restore_" + checkpoint["name"]
            base.check_deadline(deadline, 30)
            try:
                runtime.load(checkpoint, deadline)
                for case in pending:
                    phase = "episode_" + checkpoint["name"] + "_" + case["pair_id"]
                    base.check_deadline(deadline, 5)
                    row = dict(runtime.run_case(case, checkpoint, plan, deadline))
                    if base.row_key(row) != (checkpoint["name"], case["pair_id"]):
                        raise ValueError("Runtime returned another checkpoint or case instead of the dispatched episode")
                    row.update(execution_worker_index=worker_index,
                               execution_gpu_uuid=worker["gpu_uuid"], execution_run_token=run_token)
                    validate_worker_row(row, plan, shards, worker_index, run_token, base=base)
                    _check_pairs([*rows, row])
                    base.append_row(rows_path, row, plan, completed_keys)
                    rows.append(row)
                    atomic_worker_json(base, directory / "progress.json", {
                        **fields, "completed_episode_records": len(rows),
                        "expected_episode_records": len(expected_keys), "deadline_unix": deadline,
                        "last_checkpoint": checkpoint["name"], "last_pair_id": case["pair_id"],
                        "updated_ns": time.time_ns(), "process": process})
                    # Keep completed rows even if a slow final chunk crossed the cap.
                    base.check_deadline(deadline)
            finally:
                phase_before_release = phase
                phase = "release_" + checkpoint["name"]
                runtime.release()
                phase = phase_before_release
    except BaseException as exc:
        errors.append({"phase": phase, "type": type(exc).__name__, "error": str(exc),
                       "traceback": traceback.format_exc()})
    finally:
        if attempted_runtime:
            try:
                runtime.close()
            except BaseException as exc:
                errors.append({"phase": "close", "type": type(exc).__name__, "error": str(exc),
                               "traceback": traceback.format_exc()})

    actual_keys = {base.row_key(row) for row in rows}
    complete = actual_keys == expected_keys and not errors
    summary = {**fields, "complete": complete, "exit_code": 0 if complete else 2,
               "started_ns": started["started_ns"], "finished_ns": time.time_ns(),
               "deadline_unix": deadline, "completed_episode_records": len(rows),
               "expected_episode_records": len(expected_keys),
               "global_expected_episode_records": plan["expected_episode_records"],
               "assigned_pair_ids": worker["assigned_pair_ids"],
               "missing_episode_keys": [list(key) for key in sorted(expected_keys - actual_keys)],
               "checkpoint_sha256": {c["name"]: c["sha256"] for c in plan["checkpoints"]},
               "errors": errors, "process": process,
               "coverage_scope": "one worker assignment; only the coordinator validates global coverage"}
    atomic_worker_json(base, directory / "summary.json", summary)
    atomic_worker_json(base, completed_path, summary)
    return summary


class Runtime:
    """GPU and simulator implementation; all external imports are deferred."""
    def __init__(self, base, source, gpu_uuid):
        self.base, self.source, self.gpu_uuid = base, Path(source), gpu_uuid
        self.payloads = {}
        self.env = self.torch = self.model = self.cfg = self.dtype = None

    def prepare(self, plan, deadline, worker_dir, identity):
        self.base.check_deadline(deadline, 10)
        if not hasattr(os, "getuid") or os.getuid() != 1004:
            raise ValueError("The admitted server evaluation must run as UID1004")
        for name, value in (("MUJOCO_GL", "osmesa"), ("PYOPENGL_PLATFORM", "osmesa"),
                            ("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1"),
                            ("TOKENIZERS_PARALLELISM", "false")):
            os.environ.setdefault(name, value)
        repository = Path(os.environ.get("ACCERL_REPO_ROOT", "/mnt/data/lcx1/dong/AcceRL-scheduler"))
        if str(repository) not in sys.path:
            sys.path.insert(0, str(repository))
        # The pinned reference may live outside the training source after the
        # evaluator entry is replaced with a dispatch shim.  Training support
        # modules must still come from the actual frozen training source.
        source_path = str(self.source.resolve(strict=True))
        if source_path in sys.path:
            sys.path.remove(source_path)
        sys.path.insert(0, source_path)
        for checkpoint in plan["checkpoints"]:
            payload, actual = self.base.load_snapshot(Path(checkpoint["path"]), deadline)
            if any(actual.get(key) != checkpoint.get(key) for key in ("path", "sha256", "metadata")):
                raise ValueError("Checkpoint path, SHA or training metadata differs from the global plan")
            if "bytes" in checkpoint and actual["bytes"] != checkpoint["bytes"]:
                raise ValueError("Checkpoint byte count differs from the global plan")
            saved_pool = payload["manifest"]["metadata"].get("initial_state_queue", {}).get("pool")
            # The initial snapshot is legitimately saved before the first real
            # worker registers its pool.  Final training must have registered it.
            if saved_pool is None and checkpoint["name"] == "baseline":
                pass
            elif (not isinstance(saved_pool, dict) or saved_pool.get("count") != 50
                  or not isinstance(saved_pool.get("sha256"), str)
                  or re.fullmatch(r"[0-9a-f]{64}", saved_pool["sha256"]) is None
                  or "shape" not in saved_pool):
                raise ValueError("Initial-state pool must be evidenced by the final checkpoint and any populated baseline")
            self.payloads[checkpoint["name"]] = payload
        recipes = [checkpoint["metadata"]["args"] for checkpoint in plan["checkpoints"]]
        if recipes[0] != recipes[1] or recipes[0]["benchmark"] != "libero_spatial":
            raise ValueError("Paired checkpoint model or environment recipes differ")
        self.base.check_deadline(deadline, 10)
        # Query only this explicitly admitted UUID, without an all-device inventory.
        inventory = subprocess.check_output([
            "nvidia-smi", "--id=" + self.gpu_uuid, "--query-gpu=uuid",
            "--format=csv,noheader,nounits"], text=True,
            timeout=min(10.0, max(0.1, deadline - time.time())))
        if inventory.strip().splitlines() != [self.gpu_uuid]:
            raise ValueError("The selected physical GPU UUID differs from the shard")
        import torch
        self.torch = torch
        if torch.cuda.device_count() != 1:
            raise ValueError("Worker must see exactly one admitted GPU")
        torch.cuda.set_device(0)
        from rl.libero_env import LiberoEnvWrapper
        from initial_state_env import initial_state_pool_metadata
        self.env = LiberoEnvWrapper(benchmark_name="libero_spatial", task_id=0,
                                   image_size=224, render_mode="rgb_array")
        pool = initial_state_pool_metadata(self.env)
        if pool["count"] != 50:
            raise ValueError("Live initial-state pool differs from the fixed 50-state protocol")
        for payload in self.payloads.values():
            saved_pool = payload["manifest"]["metadata"].get("initial_state_queue", {}).get("pool")
            if saved_pool is not None and any(saved_pool.get(key) != pool.get(key)
                                               for key in ("count", "shape", "sha256")):
                raise ValueError("Live initial-state pool differs from an evidenced training checkpoint")
        pool_path = worker_dir / "initial_state_pool.json"
        if pool_path.is_symlink():
            raise ValueError("Worker initial-state pool file must not be a symlink")
        if pool_path.exists() and json.loads(pool_path.read_text(encoding="utf-8")) != pool:
            raise ValueError("Initial-state pool changed across worker resume")
        atomic_worker_json(self.base, pool_path, pool)

    def load(self, checkpoint, deadline):
        if self.model is not None:
            raise ValueError("Previous checkpoint model has not been released")
        self.model, self.cfg, self.dtype = self.base.restore_policy(self.payloads[checkpoint["name"]], deadline)

    def run_case(self, case, checkpoint, plan, deadline):
        return self.base.run_episode(self.model, self.cfg, self.dtype, self.env,
                                     case, checkpoint, plan, deadline)

    def release(self):
        self.model = self.cfg = self.dtype = None
        if self.torch is not None:
            self.torch.cuda.empty_cache()

    def close(self):
        try:
            self.release()
        finally:
            try:
                if self.env is not None:
                    self.env.close()
            finally:
                self.env = None
                self.payloads.clear()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference-source", type=Path)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--run-token", required=True)
    args = parser.parse_args(argv)
    base = load_base(args.reference_source or args.source)
    source_path = str(args.source.resolve(strict=True))
    if source_path in sys.path:
        sys.path.remove(source_path)
    sys.path.insert(0, source_path)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    shards = json.loads(args.shards.read_text(encoding="utf-8"))
    worker = assignment(plan, shards, args.worker_index)
    validate_visibility(worker, args.run_token)
    # Reproduce the canonical original plan instead of accepting a rehashed edit.
    canonical = base.make_plan(plan["checkpoints"], initial_state_count=plan["initial_state_count"],
                               seed=plan["seed"], modes=plan["modes"], repeats=plan["repeats"])
    if canonical != plan:
        raise ValueError("Global evaluation plan differs from the unchanged reference protocol")
    def interrupted(number, frame):
        raise InterruptedError("Evaluation worker received supervisor signal " + str(number))
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, interrupted)
    with worker_claim(args.output_root, args.worker_index, args.run_token):
        result = execute_assignment(base=base, plan=plan, shards=shards,
                                    worker_index=args.worker_index, run_token=args.run_token,
                                    output_root=args.output_root, deadline_unix=args.deadline_unix,
                                    runtime=Runtime(base, args.source, worker["gpu_uuid"]))
    print(json.dumps({key: result[key] for key in ("worker_index", "complete", "completed_episode_records",
                                                   "expected_episode_records", "errors")}, ensure_ascii=False))
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
