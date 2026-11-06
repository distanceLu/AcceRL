"""CPU-only tests for paired plans, strict admission and durable result accounting."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from frozen_policy_eval import (
    CHECKPOINT_FORMAT, PROTOCOL, append_row, make_plan, read_rows,
    summarize, validate_checkpoint_payload, validate_row,
)


def payload():
    names = ["vla.lora_A.weight", "vla.lm_head.weight", "attn_pool.0.weight", "value_head.0.weight",
             "proprio_projector.linear.weight"]
    return {"format": CHECKPOINT_FORMAT,
            "manifest": {"format": CHECKPOINT_FORMAT, "weights_only": True, "steps": {"policy": 132},
                         "tensor_counts": {"policy": len(names)},
                         "metadata": {"run_id": "test_run", "args": {
                             "pretrained_checkpoint": "/weights/base", "checkpoint2": None,
                             "benchmark": "libero_spatial", "num_images_in_input": 2,
                             "use_proprio": True, "use_bf16": True}}},
            "models": {"policy": {"state_dict": dict.fromkeys(names, "tensor-placeholder"),
                                  "trainable_names": names, "full_state": False}}}


def checkpoints():
    metadata = validate_checkpoint_payload(payload())
    return [{"name": name, "path": "/" + name + "/latest_weights.pt", "sha256": code * 64,
             "metadata": copy.deepcopy(metadata)} for name, code in (("baseline", "a"), ("final", "b"))]


def result(plan, name="baseline", case_index=0, success=True):
    case = plan["cases"][case_index]
    checkpoint = next(item for item in plan["checkpoints"] if item["name"] == name)
    return {"protocol": PROTOCOL, "plan_sha256": plan["sha256"], "checkpoint": name,
            "checkpoint_sha256": checkpoint["sha256"], "policy_version": checkpoint["metadata"]["policy_step"],
            **case, "initial_state_hash": ("%064x" % case["initial_state_id"]), "success": success,
            "steps": 36 if success else 220, "episode_return": float(success),
            "termination": "success" if success else "horizon"}


class FrozenPolicyEvalTests(unittest.TestCase):
    def test_balanced_reproducible_plan_is_paired_and_not_prefix(self):
        plan = make_plan(checkpoints(), repeats=2)
        self.assertEqual(plan, make_plan(checkpoints(), repeats=2))
        self.assertEqual(plan["expected_episode_records"], 400)
        for repeat in range(2):
            for mode in ("greedy", "stochastic"):
                ids = [case["initial_state_id"] for case in plan["cases"]
                       if case["repeat"] == repeat and case["mode"] == mode]
                self.assertEqual(sorted(ids), list(range(50)))
                self.assertNotEqual(ids, list(range(50)))
        first, second = plan["cases"][:2]
        self.assertEqual(first["env_seed"], second["env_seed"])
        self.assertEqual(first["action_seed"], second["action_seed"])
        self.assertNotEqual(first["env_seed"], first["action_seed"])

    def test_formal_plan_rejects_partial_or_invalid_modes(self):
        for kwargs in ({"initial_state_count": 8}, {"repeats": 0}, {"seed": -1},
                       {"modes": ["greedy", "greedy"]}, {"modes": []}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_plan(checkpoints(), **kwargs)

    def test_recipe_and_all_trained_policy_families_are_required(self):
        original = payload()
        self.assertEqual(validate_checkpoint_payload(original)["policy_step"], 132)
        for field in ("pretrained_checkpoint", "checkpoint2", "use_proprio", "use_bf16", "benchmark"):
            bad = copy.deepcopy(original)
            del bad["manifest"]["metadata"]["args"][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_checkpoint_payload(bad)
        for family in ("lora_", "lm_head", "attn_pool", "value_head", "proprio_projector"):
            bad = copy.deepcopy(original)
            removed = [name for name in bad["models"]["policy"]["trainable_names"] if family in name]
            for name in removed:
                bad["models"]["policy"]["trainable_names"].remove(name)
                del bad["models"]["policy"]["state_dict"][name]
            bad["manifest"]["tensor_counts"]["policy"] -= len(removed)
            with self.subTest(family=family), self.assertRaises(ValueError):
                validate_checkpoint_payload(bad)

    def test_metadata_rejects_missing_or_contradictory_identity(self):
        for mutation in (
            lambda p: p["manifest"]["metadata"].pop("run_id"),
            lambda p: p["manifest"]["steps"].update(policy=True),
            lambda p: p["manifest"]["tensor_counts"].update(policy=0),
            lambda p: p["manifest"].update(weights_only=False),
            lambda p: p["models"]["policy"]["state_dict"].pop("value_head.0.weight"),
        ):
            bad = payload()
            mutation(bad)
            with self.assertRaises(ValueError):
                validate_checkpoint_payload(bad)

    def test_same_number_steps_does_not_make_different_checkpoint_interchangeable(self):
        plan = make_plan(checkpoints())
        row = result(plan)
        row["checkpoint_sha256"] = "c" * 64
        with self.assertRaises(ValueError):
            validate_row(row, plan)

    def test_row_rejects_changed_pairing_and_contradictory_outcome(self):
        plan = make_plan(checkpoints())
        for field, changed in (("action_seed", -1), ("initial_state_id", 55), ("policy_version", 133),
                               ("steps", 221), ("termination", "horizon"), ("episode_return", float("nan"))):
            row = result(plan)
            row[field] = changed
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_row(row, plan)

    def test_append_resume_is_idempotent_and_does_not_rerun_completed_keys(self):
        plan = make_plan(checkpoints())
        row = result(plan)
        with tempfile.TemporaryDirectory() as folder:
            path, keys = Path(folder) / "episodes.jsonl", set()
            append_row(path, row, plan, keys)
            loaded = read_rows(path, plan)
            self.assertEqual(loaded, [row])
            restored = {(item["checkpoint"], item["pair_id"]) for item in loaded}
            self.assertIn((row["checkpoint"], row["pair_id"]), restored)
            with self.assertRaises(ValueError):
                append_row(path, row, plan, restored)
            self.assertEqual(len(read_rows(path, plan)), 1)

    def test_interrupted_tail_repaired_without_discarding_completed_episode(self):
        plan = make_plan(checkpoints())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "episodes.jsonl"
            path.write_text(json.dumps(result(plan)) + '\n{"partial":', encoding="utf-8")
            self.assertEqual(len(read_rows(path, plan, repair_partial_tail=True)), 1)
            self.assertTrue(path.read_bytes().endswith(b"\n"))
            self.assertEqual(len(read_rows(path, plan)), 1)

    def test_corrupt_interior_and_duplicate_rows_are_never_silently_repaired(self):
        plan = make_plan(checkpoints())
        line = json.dumps(result(plan)) + "\n"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "episodes.jsonl"
            for text in (line + "broken\n" + line, line + line):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    read_rows(path, plan, repair_partial_tail=True)

    def test_missing_newline_is_repaired_before_append(self):
        plan = make_plan(checkpoints())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "episodes.jsonl"
            path.write_text(json.dumps(result(plan)))
            rows = read_rows(path, plan, repair_partial_tail=True)
            keys = {(r["checkpoint"], r["pair_id"]) for r in rows}
            append_row(path, result(plan, case_index=1), plan, keys)
            self.assertEqual(len(read_rows(path, plan)), 2)

    def test_partial_summary_only_pairs_identical_cases_and_reports_missing_states(self):
        plan = make_plan(checkpoints(), modes=["greedy"])
        rows = [result(plan, case_index=0, success=False), result(plan, "final", 0, True),
                result(plan, case_index=1, success=True)]
        summary = summarize(plan, rows)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["paired"]["greedy"]["paired_episodes"], 1)
        self.assertEqual(summary["paired"]["greedy"]["final_minus_baseline_success_rate"], 1.0)
        self.assertEqual(summary["paired"]["greedy"]["unpaired_baseline_episodes"], 1)
        self.assertEqual(len(summary["groups"]["final:greedy"]["missing_initial_state_ids"]), 49)

    def test_paired_state_hash_difference_fails_summary(self):
        plan = make_plan(checkpoints(), modes=["greedy"])
        rows = [result(plan), result(plan, "final")]
        rows[1]["initial_state_hash"] = "f" * 64
        with self.assertRaises(ValueError):
            summarize(plan, rows)

    def test_complete_requires_every_expected_record_and_no_errors(self):
        plan = make_plan(checkpoints())
        rows = [result(plan, name, i, success=i % 2 == 0)
                for name in ("baseline", "final") for i in range(len(plan["cases"]))]
        self.assertTrue(summarize(plan, rows)["complete"])
        self.assertFalse(summarize(plan, rows, errors=[{"error": "shutdown"}])["complete"])
        self.assertFalse(summarize(plan, rows[:-1])["complete"])
        self.assertEqual(summarize(plan, rows)["paired"]["stochastic"]["paired_episodes"], 50)


if __name__ == "__main__":
    unittest.main()
