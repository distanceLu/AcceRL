"""CPU-only admission, sharding and strict-merge tests for four GPU evaluation.

No training module, CUDA runtime, network connection or model is initialized.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import plan_contract as contract

SOURCE = HERE.parents[1] / "engine"
BASE = contract.load_base(SOURCE)
TOKEN = "cpu-contract-execution-token"


def checkpoints():
    names = ["vla.lora_A.weight", "vla.lm_head.weight", "attn_pool.0.weight",
             "value_head.0.weight", "proprio_projector.linear.weight"]
    payload = {"format": BASE.CHECKPOINT_FORMAT,
               "manifest": {"format": BASE.CHECKPOINT_FORMAT, "weights_only": True,
                            "steps": {"policy": 132}, "tensor_counts": {"policy": len(names)},
                            "metadata": {"run_id": "cpu_contract_run", "args": {
                                "pretrained_checkpoint": "/weights/base", "checkpoint2": None,
                                "benchmark": "libero_spatial", "num_images_in_input": 2,
                                "use_proprio": True, "use_bf16": True}}},
               "models": {"policy": {"state_dict": dict.fromkeys(names, "cpu-placeholder"),
                                     "trainable_names": names, "full_state": False}}}
    metadata = BASE.validate_checkpoint_payload(payload)
    return [{"name": name, "path": "/" + name + "/latest_weights.pt", "sha256": char * 64,
             "metadata": copy.deepcopy(metadata)} for name, char in (("baseline", "a"), ("final", "b"))]


def make_plan(**kwargs):
    return BASE.make_plan(checkpoints(), **kwargs)


def episode_row(plan, shards, worker_index, name, case, *, success=True, token=TOKEN):
    checkpoint = next(value for value in plan["checkpoints"] if value["name"] == name)
    worker = shards["workers"][worker_index]
    return {"protocol": BASE.PROTOCOL, "plan_sha256": plan["sha256"],
            "checkpoint": name, "checkpoint_sha256": checkpoint["sha256"],
            "policy_version": checkpoint["metadata"]["policy_step"], **case,
            "initial_state_hash": "%064x" % case["initial_state_id"],
            "success": success, "steps": 36 if success else 220,
            "episode_return": float(success), "termination": "success" if success else "horizon",
            "execution_worker_index": worker_index, "execution_gpu_uuid": worker["gpu_uuid"],
            "execution_run_token": token}


def worker_ledgers(plan, shards, *, token=TOKEN):
    assigned = {pair: worker["worker_index"] for worker in shards["workers"]
                for pair in worker["assigned_pair_ids"]}
    result = {index: [] for index in range(4)}
    for checkpoint in plan["checkpoints"]:
        for case in plan["cases"]:
            index = assigned[case["pair_id"]]
            result[index].append(episode_row(plan, shards, index, checkpoint["name"], case,
                                            success=checkpoint["name"] == "final", token=token))
    return result


def rehash(value):
    value["sha256"] = contract.digest({key: item for key, item in value.items() if key != "sha256"})
    return value


class ShardContractTests(unittest.TestCase):
    def setUp(self):
        self.plan = make_plan()
        self.shards = contract.make_shards(self.plan)

    def test_four_shards_are_stable_balanced_and_cover_exact_global_pairs(self):
        original = copy.deepcopy(self.plan)
        self.assertEqual(self.shards, contract.make_shards(self.plan))
        self.assertEqual(self.plan, original)
        self.assertEqual([worker["expected_episode_records"] for worker in self.shards["workers"]],
                         [52, 52, 48, 48])
        pairs = [pair for worker in self.shards["workers"] for pair in worker["assigned_pair_ids"]]
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertEqual(set(pairs), {case["pair_id"] for case in self.plan["cases"]})
        self.assertEqual(sum(worker["expected_episode_records"] for worker in self.shards["workers"]), 200)

    def test_same_initial_state_and_repeat_keeps_both_modes_and_checkpoints_on_one_gpu(self):
        ownership = {pair: worker["worker_index"] for worker in self.shards["workers"]
                     for pair in worker["assigned_pair_ids"]}
        grouped = {}
        for case in self.plan["cases"]:
            grouped.setdefault((case["repeat"], case["initial_state_id"]), set()).add(ownership[case["pair_id"]])
        self.assertEqual(len(grouped), 50)
        self.assertTrue(all(len(indices) == 1 for indices in grouped.values()))
        ledgers = worker_ledgers(self.plan, self.shards)
        for rows in ledgers.values():
            self.assertEqual({row["checkpoint"] for row in rows}, {"baseline", "final"})

    def test_multiple_repeats_keep_full_coverage_and_continue_round_robin(self):
        plan = make_plan(repeats=2)
        shards = contract.make_shards(plan)
        self.assertEqual([worker["expected_episode_records"] for worker in shards["workers"]], [100] * 4)
        self.assertEqual(sum(len(worker["assigned_pair_ids"]) for worker in shards["workers"]), 200)

    def test_single_mode_preserves_requested_global_plan(self):
        plan = make_plan(modes=("greedy",))
        shards = contract.make_shards(plan)
        self.assertEqual([worker["expected_episode_records"] for worker in shards["workers"]], [26, 26, 24, 24])
        self.assertEqual(shards["plan_sha256"], plan["sha256"])

    def test_explicit_approved_gpu_order_is_respected(self):
        order = tuple(reversed(contract.APPROVED_GPUS))
        shards = contract.make_shards(self.plan, order)
        self.assertEqual(tuple(worker["gpu_uuid"] for worker in shards["workers"]), order)
        contract.validate_shards(self.plan, shards)

    def test_gpu_admission_rejects_missing_duplicate_or_foreign_devices(self):
        cases = [(), contract.APPROVED_GPUS[:3], (*contract.APPROVED_GPUS, contract.APPROVED_GPUS[0]),
                 (contract.APPROVED_GPUS[0],) * 4,
                 (*contract.APPROVED_GPUS[:3], "GPU-unapproved")]
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                contract.make_shards(self.plan, value)

    def test_invalid_plan_digest_is_rejected_before_sharding(self):
        changed = copy.deepcopy(self.plan)
        changed["cases"][0]["env_seed"] += 1
        with self.assertRaises(ValueError):
            contract.make_shards(changed)

    def test_rehashed_modified_case_seed_or_determinism_is_not_canonical(self):
        for field, value in (("env_seed", 17), ("action_seed", 19), ("deterministic", False)):
            changed = copy.deepcopy(self.plan)
            changed["cases"][0][field] = value
            rehash(changed)
            with self.subTest(field=field), self.assertRaises(ValueError):
                contract.make_shards(changed)

    def test_modified_coverage_and_formal_runtime_contract_are_rejected(self):
        for field, value in (("initial_state_count", 49), ("task_id", 1),
                             ("horizon_action_steps", 221), ("reset_settling_steps", 0),
                             ("inference_batch_size", 4), ("policy_updates_during_evaluation", True)):
            changed = copy.deepcopy(self.plan)
            changed[field] = value
            rehash(changed)
            with self.subTest(field=field), self.assertRaises(ValueError):
                contract.make_shards(changed)

    def test_shard_digest_and_every_assignment_are_admitted_exactly(self):
        bad_hash = copy.deepcopy(self.shards)
        bad_hash["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            contract.validate_shards(self.plan, bad_hash)
        for index, mutate in enumerate((
                lambda obj: obj.update(plan_sha256="0" * 64),
                lambda obj: obj["workers"][0].update(expected_episode_records=999),
                lambda obj: obj["workers"][0]["assigned_pair_ids"].pop(),
                lambda obj: obj["workers"][0]["assigned_pair_ids"].append(obj["workers"][1]["assigned_pair_ids"][0]),
                lambda obj: obj["workers"][0].update(worker_index=2))):
            changed = copy.deepcopy(self.shards)
            mutate(changed)
            rehash(changed)
            with self.subTest(mutation=index), self.assertRaises(ValueError):
                contract.validate_shards(self.plan, changed)

    def test_reviewed_evaluator_is_loaded_without_torch_or_cuda(self):
        self.assertEqual(BASE.PROTOCOL, "accerl_frozen_policy_paired_v1")
        self.assertNotIn("torch", sys.modules)

    def test_modified_reference_evaluator_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "frozen_policy_eval.py").write_text("raise AssertionError('must not execute')\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                contract.load_base(directory)


class WorkerRowsAndMergeTests(unittest.TestCase):
    def setUp(self):
        self.plan = make_plan()
        self.shards = contract.make_shards(self.plan)
        self.ledgers = worker_ledgers(self.plan, self.shards)

    def test_all_original_case_fields_and_execution_identity_are_preserved(self):
        row = self.ledgers[0][0]
        original = copy.deepcopy(row)
        self.assertEqual(contract.validate_worker_row(row, self.plan, self.shards, 0, TOKEN, BASE), row)
        self.assertEqual(row, original)

    def test_invalid_worker_index_cannot_alias_a_valid_integer(self):
        for index in (-1, 4, True, False, "0", 0.0):
            with self.subTest(index=index), self.assertRaises(ValueError):
                contract.validate_worker_row(self.ledgers[0][0], self.plan, self.shards, index, TOKEN, BASE)

    def test_bool_execution_worker_index_cannot_alias_rank_one(self):
        row = copy.deepcopy(self.ledgers[1][0])
        row["execution_worker_index"] = True
        with self.assertRaises(ValueError):
            contract.validate_worker_row(row, self.plan, self.shards, 1, TOKEN, BASE)

    def test_missing_reviewed_base_is_rejected(self):
        with self.assertRaises(ValueError):
            contract.validate_worker_row(self.ledgers[0][0], self.plan, self.shards, 0, TOKEN)

    def test_foreign_shard_pair_is_rejected_even_with_forged_execution_tags(self):
        row = copy.deepcopy(self.ledgers[1][0])
        row["execution_worker_index"] = 0
        row["execution_gpu_uuid"] = self.shards["workers"][0]["gpu_uuid"]
        with self.assertRaises(ValueError):
            contract.validate_worker_row(row, self.plan, self.shards, 0, TOKEN, BASE)

    def test_wrong_device_token_or_execution_index_is_rejected(self):
        for field, value in (("execution_gpu_uuid", self.shards["workers"][1]["gpu_uuid"]),
                             ("execution_run_token", "foreign-execution"), ("execution_worker_index", 1)):
            row = copy.deepcopy(self.ledgers[0][0])
            row[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                contract.validate_worker_row(row, self.plan, self.shards, 0, TOKEN, BASE)

    def test_original_episode_admission_remains_strict(self):
        for field, value in (("checkpoint_sha256", "c" * 64), ("action_seed", 0),
                             ("success", "true"), ("steps", 221), ("episode_return", float("nan"))):
            row = copy.deepcopy(self.ledgers[0][0])
            row[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                contract.validate_worker_row(row, self.plan, self.shards, 0, TOKEN, BASE)

    def test_merge_is_deterministic_and_complete_only_for_all_two_hundred_unique_episodes(self):
        rows = contract.merge_rows(self.plan, self.shards, self.ledgers, TOKEN, BASE)
        shuffled = {index: list(reversed(self.ledgers[index])) for index in reversed(range(4))}
        self.assertEqual(rows, contract.merge_rows(self.plan, self.shards, shuffled, TOKEN, BASE))
        expected = [(checkpoint["name"], case["pair_id"]) for checkpoint in self.plan["checkpoints"]
                    for case in self.plan["cases"]]
        self.assertEqual([(row["checkpoint"], row["pair_id"]) for row in rows], expected)
        summary = BASE.summarize(self.plan, rows)
        self.assertTrue(summary["complete"])
        self.assertEqual(len(rows), 200)
        for value in summary["groups"].values():
            self.assertEqual(value["episodes"], 50)

    def test_missing_worker_ledger_cannot_be_silently_omitted(self):
        del self.ledgers[3]
        with self.assertRaises(ValueError):
            contract.merge_rows(self.plan, self.shards, self.ledgers, TOKEN, BASE)

    def test_empty_or_partial_worker_ledger_cannot_produce_complete_summary(self):
        for trim in (1, len(self.ledgers[3])):
            partial = copy.deepcopy(self.ledgers)
            partial[3] = partial[3][:-trim]
            rows = contract.merge_rows(self.plan, self.shards, partial, TOKEN, BASE)
            self.assertFalse(BASE.summarize(self.plan, rows)["complete"])

    def test_duplicate_within_worker_is_rejected(self):
        self.ledgers[0].append(copy.deepcopy(self.ledgers[0][0]))
        with self.assertRaises(ValueError):
            contract.merge_rows(self.plan, self.shards, self.ledgers, TOKEN, BASE)

    def test_row_copied_to_another_worker_cannot_be_counted_twice(self):
        row = copy.deepcopy(self.ledgers[0][0])
        row["execution_worker_index"] = 1
        row["execution_gpu_uuid"] = self.shards["workers"][1]["gpu_uuid"]
        self.ledgers[1].append(row)
        with self.assertRaises(ValueError):
            contract.merge_rows(self.plan, self.shards, self.ledgers, TOKEN, BASE)

    def test_foreign_run_ledger_is_rejected(self):
        self.ledgers[2][0]["execution_run_token"] = "another-run"
        with self.assertRaises(ValueError):
            contract.merge_rows(self.plan, self.shards, self.ledgers, TOKEN, BASE)

    def test_initial_state_hash_mismatch_still_invalidates_paired_summary(self):
        rows = contract.merge_rows(self.plan, self.shards, self.ledgers, TOKEN, BASE)
        row = next(value for value in rows if value["checkpoint"] == "final")
        row["initial_state_hash"] = "f" * 64
        with self.assertRaises(ValueError):
            BASE.summarize(self.plan, rows)

    def test_worker_failure_is_preserved_even_when_all_episode_rows_exist(self):
        rows = contract.merge_rows(self.plan, self.shards, self.ledgers, TOKEN, BASE)
        self.assertFalse(BASE.summarize(self.plan, rows, errors=[{"worker": 2, "error": "failed"}])["complete"])


class GlobalDeadlineTests(unittest.TestCase):
    def test_all_workers_share_one_absolute_deadline(self):
        self.assertEqual(contract.choose_deadline(1000, 7200), 8200)
        self.assertEqual(contract.choose_deadline(1000, 7200, deadline_unix=2000), 2000)

    def test_resume_and_explicit_override_can_only_shorten_original_window(self):
        prior = {"deadline_unix": 1800}
        self.assertEqual(contract.choose_deadline(1200, 7200, prior_started=prior), 1800)
        self.assertEqual(contract.choose_deadline(1200, 7200, deadline_unix=1600, prior_started=prior), 1600)
        self.assertEqual(contract.choose_deadline(1200, 10, prior_started=prior), 1210)

    def test_elapsed_original_deadline_cannot_be_restarted(self):
        with self.assertRaises(TimeoutError):
            contract.choose_deadline(2000, 7200, prior_started={"deadline_unix": 1800})

    def test_invalid_budget_or_nonfinite_time_is_rejected(self):
        for seconds in (0, -1, 7201, True, "7200", float("inf"), float("nan")):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                contract.choose_deadline(1000, seconds)
        for value in (True, float("inf"), float("nan"), "2000"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                contract.choose_deadline(1000, 60, deadline_unix=value)

    def test_resume_requires_finite_original_deadline(self):
        for prior in ({}, {"deadline_unix": True}, {"deadline_unix": float("nan")}):
            with self.subTest(prior=prior), self.assertRaises(ValueError):
                contract.choose_deadline(1000, 60, prior_started=prior)


if __name__ == "__main__":
    unittest.main(verbosity=2)
