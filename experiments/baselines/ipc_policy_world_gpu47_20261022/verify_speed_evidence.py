"""Recompute the published 31--100 comparison using only Python's standard library.

No training imports, GPU access, network access, subprocesses or source mutation.
The compact evidence is a projection of recorded driver events, not a new run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


EXPECTED_RUNS = {
    "ipc": "ipcpwmanual_1022_161523_17963ef5",
    "baseline": "jointp4manual_1022_104329_229b879f",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def recompute(document):
    if document["schema_version"] != 1 or set(document["runs"]) != set(EXPECTED_RUNS):
        raise ValueError("Unexpected evidence schema or run set")
    results = {}
    for label, run in document["runs"].items():
        if run["run_id"] != EXPECTED_RUNS[label]:
            raise ValueError("Run identity mismatch")
        rows = run["epochs"]
        if digest(rows) != run["projected_rows_sha256"]:
            raise ValueError("Projected evidence digest mismatch")
        if [row["completed_epochs"] for row in rows] != list(range(30, 101)):
            raise ValueError("Missing, duplicated or reordered synchronized epoch")
        by_step = {row["completed_epochs"]: row for row in rows}
        previous_ns = None
        for step, row in by_step.items():
            if row["global_step"] != step or row["rank_steps"] != [step] * 4:
                raise ValueError("Four-rank completed-update receipt mismatch")
            if row["trained_ctrl"] is not (step % 5 == 0) or row["trained_reward"] is not (step % 5 == 0):
                raise ValueError("Not a complete 5:1:1 schedule")
            timestamp = row["ts_ns"]
            if type(timestamp) is not int or (previous_ns is not None and timestamp <= previous_ns):
                raise ValueError("Invalid or non-monotonic integer timestamp")
            previous_ns = timestamp
            if not all(math.isfinite(row[key]) and row[key] >= 0
                       for key in ("train_seconds", "sync_seconds")):
                raise ValueError("Non-finite or negative host phase duration")
        cycles = []
        for begin in range(31, 101, 5):
            end = begin + 4
            members = [by_step[step] for step in range(begin, end + 1)]
            elapsed_ns = by_step[end]["ts_ns"] - by_step[begin - 1]["ts_ns"]
            wall = elapsed_ns / 1e9
            train = sum(row["train_seconds"] for row in members)
            sync = sum(row["sync_seconds"] for row in members)
            cycles.append({"start": begin, "end": end, "wall_ns": elapsed_ns,
                           "wall_seconds": wall, "train_seconds": train,
                           "sync_seconds": sync, "other_seconds": wall - train - sync})
        total_wall = sum(cycle["wall_seconds"] for cycle in cycles)
        results[label] = {
            "run_id": run["run_id"], "cycles": cycles, "ncycles": len(cycles),
            "policy_updates": len(cycles) * 5,
            "world_updates": len(cycles), "reward_updates": len(cycles),
            "sum_cycle_wall_seconds": total_wall,
            "policy_updates_per_hour": len(cycles) * 5 * 3600 / total_wall,
            "mean_cycle_parts_seconds": {
                key: statistics.mean(cycle[key] for cycle in cycles)
                for key in ("wall_seconds", "train_seconds", "sync_seconds", "other_seconds")},
        }
    old, new = results["baseline"], results["ipc"]
    baseline_parts, ipc_parts = old["mean_cycle_parts_seconds"], new["mean_cycle_parts_seconds"]
    wall_saved = baseline_parts["wall_seconds"] - ipc_parts["wall_seconds"]
    sync_saved = baseline_parts["sync_seconds"] - ipc_parts["sync_seconds"]
    comparison = {
        "baseline_updates_per_hour": old["policy_updates_per_hour"],
        "ipc_updates_per_hour": new["policy_updates_per_hour"],
        "throughput_gain_percent": (new["policy_updates_per_hour"] / old["policy_updates_per_hour"] - 1) * 100,
        "cycle_seconds_saved": wall_saved,
        "sync_reduction_percent": sync_saved / baseline_parts["sync_seconds"] * 100,
        "share_of_saved_seconds_from_sync_percent": sync_saved / wall_saved * 100,
    }
    for key, expected in document["expected_headline"].items():
        if not math.isclose(comparison[key], expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Headline disagreement for {key}: {comparison[key]} vs {expected}")
    return {"passed": True, "access": "CPU standard-library-only recomputation of compact event projection",
            "comparison": comparison, "runs": results,
            "limitations": document["limitations"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("speed_evidence_compact.json"))
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = recompute(json.loads(arguments.input.read_text(encoding="utf-8")))
    result["input_sha256"] = hashlib.sha256(arguments.input.read_bytes()).hexdigest()
    result["verifier_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if arguments.output:
        arguments.output.write_text(rendered, encoding="utf-8")
    print(json.dumps({"passed": result["passed"], **result["comparison"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
