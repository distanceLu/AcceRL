"""CPU-only contracts for frozen RM reports; all expected errors are hand derived."""
import copy
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from .rm_report import rm_state_report


def episode(eid, targets, *, success=True, state=None, model="perfect", predictions=None, length=None):
    predictions = targets if predictions is None else predictions
    return [dict(model=model, episode_id=eid, initial_state_id=eid if state is None else state,
                 frame_index=i + 1, episode_length=len(targets) if length is None else length,
                 success=success, target=y, prediction=p)
            for i, (y, p) in enumerate(zip(targets, predictions))]


def panel():
    base = (episode("cal-ok", [0., .5, 1.]) + episode("cal-fail", [0., 0., 0.], success=False)
            + episode("test-ok", [0., .5, 1.]) + episode("test-fail", [0., 0., 0.], success=False))
    return [dict(r, model=m, prediction=(r["target"] if m == "perfect" else
                                       1. - r["target"] if m == "reversed" else .25))
            for m in ("perfect", "constant", "reversed") for r in base]


def report(rows=None, **kwargs):
    return rm_state_report(panel() if rows is None else rows, ["cal-ok", "cal-fail"],
                           ["test-ok", "test-fail"], **kwargs)


class RMReportTests(unittest.TestCase):
    def test_perfect_constant_reversed_and_progression(self):
        result = report()
        perfect, constant, reverse = (result["models"][m] for m in ("perfect", "constant", "reversed"))
        for weighting in ("frame_weighted", "equal_episode"):
            self.assertEqual(perfect["headline"][weighting]["mse"], 0.)
            self.assertEqual(perfect["headline"][weighting]["skill_vs_constant"], 1.)
            self.assertAlmostEqual(constant["headline"][weighting]["mse"], 7 / 48)
            self.assertAlmostEqual(constant["headline"][weighting]["constant_mse"], 7 / 48)
            self.assertAlmostEqual(constant["headline"][weighting]["skill_vs_constant"], 0.)
            self.assertAlmostEqual(reverse["headline"][weighting]["mse"], 5 / 6)
            self.assertAlmostEqual(reverse["headline"][weighting]["bias"], .5)
            self.assertEqual(perfect["progression"][weighting]["centered_mse"], 0.)
            self.assertEqual(constant["progression"][weighting]["skill_vs_flat"], 0.)
            self.assertAlmostEqual(reverse["progression"][weighting]["skill_vs_flat"], -3.)
        self.assertEqual(perfect["success_failure"]["episode_mean_auc"], 1.)
        self.assertEqual(constant["success_failure"]["episode_mean_auc"], .5)
        self.assertEqual(reverse["success_failure"]["episode_mean_auc"], 0.)
        self.assertEqual(len(result["raw_rows"]), 36)
        json.dumps(result, allow_nan=False)

    def test_unequal_lengths_fit_distinct_calibration_constants(self):
        rows = (episode("c-short", [0.], success=False, predictions=[.5])
                + episode("c-long", [1., 1., 1.], predictions=[.5] * 3)
                + episode("t-short", [0.], success=False, predictions=[.5])
                + episode("t-long", [1., 1., 1.], predictions=[.5] * 3))
        result = rm_state_report(rows, ["c-short", "c-long"], ["t-short", "t-long"])
        self.assertEqual(result["constant_reference"]["frame_weighted"], .75)
        self.assertEqual(result["constant_reference"]["equal_episode"], .5)
        h = result["models"]["perfect"]["headline"]
        self.assertEqual(h["frame_weighted"]["bias"], -.25)
        self.assertEqual(h["equal_episode"]["bias"], 0.)
        self.assertEqual(h["frame_weighted"]["constant_mse"], .1875)
        self.assertEqual(h["equal_episode"]["constant_mse"], .25)

    def test_test_labels_cannot_fit_constant_and_oracle_is_opt_in(self):
        rows = episode("c", [0., 0.], success=False) + episode("t", [1., 1.])
        result = rm_state_report(rows, ["c"], ["t"])
        self.assertEqual(result["constant_reference"]["frame_weighted"], 0.)
        self.assertEqual(result["models"]["perfect"]["headline"]["frame_weighted"]["constant_mse"], 1.)
        self.assertNotIn("held_out_oracle", result)
        oracle = rm_state_report(rows, ["c"], ["t"], include_oracle=True)["held_out_oracle"]
        self.assertFalse(oracle["deployable"])
        self.assertEqual(oracle["scope"], "constant_class_lower_bound")
        self.assertEqual(oracle["frame_weighted"], {"value": 1., "mse": 0.})

    def test_prediction_distribution_is_weighted_and_keeps_quantiles(self):
        rows = (episode("c", [0.]) + episode("short", [0.], success=False)
                + episode("long", [1., 1., 1.]))
        d = rm_state_report(rows, ["c"], ["short", "long"])["models"]["perfect"]["prediction_distribution"]
        self.assertEqual(d["frame_weighted"]["mean"], .75)
        self.assertAlmostEqual(d["frame_weighted"]["std"], np.sqrt(.1875))
        self.assertEqual(d["equal_episode"]["mean"], .5)
        self.assertEqual(d["equal_episode"]["std"], .5)
        self.assertEqual(d["frame_weighted"]["quantiles"]["p05"], 0.)
        self.assertEqual(d["frame_weighted"]["quantiles"]["p50"], 1.)
        self.assertEqual(d["frame_weighted"]["quantiles"]["p95"], 1.)

    def test_equal_episode_mse_and_groups_do_not_refit_constants(self):
        rows = (episode("c", [0.], success=False)
                + episode("short", [0.], success=False, predictions=[0.])
                + episode("long", [1., 1., 1.], predictions=[0.] * 3))
        m = rm_state_report(rows, ["c"], ["short", "long"])["models"]["perfect"]
        self.assertEqual(m["headline"]["frame_weighted"]["mse"], .75)
        self.assertEqual(m["headline"]["equal_episode"]["mse"], .5)
        success = next(g for g in m["groups"] if g["outcome"] == "success" and g["phase"] == "all")
        self.assertEqual(success["frame_weighted"]["constant_mse"], 1.)
        self.assertEqual(m["success_failure"]["status"], "limited_support")

    def test_decimal_flat_trace_has_no_spurious_progression_skill(self):
        rows = episode("c", [.1] * 3) + episode("t", [.1] * 3, predictions=[.3] * 3)
        m = rm_state_report(rows, ["c"], ["t"])["models"]["perfect"]
        for weighting in ("frame_weighted", "equal_episode"):
            self.assertEqual(m["progression"][weighting]["centered_target_variance"], 0.)
            self.assertEqual(m["progression"][weighting]["centered_mse"], 0.)
            self.assertIsNone(m["progression"][weighting]["skill_vs_flat"])
            self.assertIsNone(m["headline"][weighting]["skill_vs_constant"])
        self.assertEqual(m["progression"]["status"], "no_support")

    def test_constant_scores_tie_in_auc_despite_unequal_trace_lengths(self):
        rows = (episode("c", [.1]) + episode("positive", [.1, .2, .3], predictions=[.1] * 3)
                + episode("negative", [0.], success=False, predictions=[.1]))
        d = rm_state_report(rows, ["c"], ["positive", "negative"])["models"]["perfect"]["success_failure"]
        self.assertEqual(d["episode_mean_auc"], .5)
        self.assertEqual(d["episode_mean_score_gap"], 0.)

    def test_groups_use_actual_frame_index_not_sample_rank(self):
        rows = episode("c", [0.]) + episode("t", [.2, .8], length=12)
        rows[1]["frame_index"], rows[2]["frame_index"] = 1, 12
        groups = rm_state_report(rows, ["c"], ["t"])["models"]["perfect"]["groups"]
        lookup = {(g["outcome"], g["phase"]): g for g in groups}
        self.assertEqual(lookup[("all", "early")]["support"]["frames"], 1)
        self.assertEqual(lookup[("all", "late")]["support"]["frames"], 1)
        for key in (("all", "middle"), ("failure", "all"), ("failure", "early")):
            self.assertEqual(lookup[key]["status"], "no_support")
            self.assertIsNone(lookup[key]["frame_weighted"]["mse"])

    def test_progression_removes_episode_offsets(self):
        rows = (episode("c", [.5]) + episode("t1", [.1, .3], predictions=[.3, .5])
                + episode("t2", [.5, .7], predictions=[.3, .5]))
        m = rm_state_report(rows, ["c"], ["t1", "t2"])["models"]["perfect"]
        self.assertAlmostEqual(m["headline"]["equal_episode"]["mse"], .04)
        self.assertAlmostEqual(m["progression"]["equal_episode"]["centered_mse"], 0.)
        self.assertAlmostEqual(m["progression"]["equal_episode"]["skill_vs_flat"], 1.)

    def test_absent_outcomes_flat_targets_and_singletons_have_explicit_support(self):
        rows = episode("c", [.5]) + episode("t", [.5])
        m = rm_state_report(rows, ["c"], ["t"])["models"]["perfect"]
        self.assertEqual(m["success_failure"]["status"], "no_support")
        self.assertIn("failure", m["success_failure"]["reason"])
        self.assertIsNone(m["success_failure"]["episode_mean_auc"])
        self.assertEqual(m["progression"]["status"], "no_support")
        self.assertIsNone(m["headline"]["frame_weighted"]["skill_vs_constant"])
        rows = episode("c", [.5]) + episode("t", [.5, .5])
        p = rm_state_report(rows, ["c"], ["t"])["models"]["perfect"]["progression"]
        self.assertEqual(p["frame_weighted"]["centered_mse"], 0.)
        self.assertIsNone(p["frame_weighted"]["skill_vs_flat"])
        self.assertIn("variation", p["reason"])

    def test_explicit_initial_state_split_groups_policy_panels_and_records_mapping(self):
        rows = [dict(r, policy_panel=policy) for model in ("orange", "refill")
                for policy in ("orange150", "refill") for state in (0, 3, 6, 8)
                for r in episode(f"{policy}:state{state}", [.3, 1.], state=state, model=model)]
        before = copy.deepcopy(rows)
        result = rm_state_report(rows, calibration_initial_states=[0, 6], test_initial_states=[3, 8])
        self.assertEqual(result["split"]["calibration_initial_states"], [0, 6])
        self.assertEqual(set(result["split"]["test_ids"]),
                         {f"{p}:state{s}" for p in ("orange150", "refill") for s in (3, 8)})
        self.assertEqual(result["models"]["orange"]["support"]["episodes"], 4)
        self.assertEqual(result["models"]["orange"]["support"]["initial_states"], 2)
        self.assertEqual(result["split"]["episodes_by_initial_state"][1]["episode_ids"],
                         ["orange150:state3", "refill:state3"])
        self.assertEqual(rows, before)
        self.assertEqual(result, rm_state_report(list(reversed(rows)),
                                                calibration_initial_states=[6, 0], test_initial_states=[8, 3]))
        self.assertTrue(all("policy_panel" in r for r in result["raw_rows"]))

    def test_rejects_episode_or_initial_state_leakage(self):
        with self.assertRaisesRegex(ValueError, "disjoint"):
            rm_state_report(panel(), ["cal-ok", "test-ok"], ["test-ok"])
        rows = episode("c", [.5], state=1) + episode("t", [.5], state=1)
        with self.assertRaisesRegex(ValueError, "initial.state"):
            rm_state_report(rows, ["c"], ["t"])
        with self.assertRaisesRegex(ValueError, "disjoint"):
            rm_state_report(rows, calibration_initial_states=[1], test_initial_states=[1])

    def test_split_must_cover_exactly_known_episodes_and_use_one_mode(self):
        rows = episode("c", [.5]) + episode("t", [.5])
        for cal, test in (([], ["t"]), (["c"], []), (["c", "ghost"], ["t"]), (["c"], ["other"])):
            with self.subTest(cal=cal, test=test), self.assertRaises(ValueError):
                rm_state_report(rows, cal, test)
        with self.assertRaises(ValueError):
            rm_state_report(rows, ["c"], ["t"], calibration_initial_states=["c"], test_initial_states=["t"])
        with self.assertRaises(ValueError):
            rm_state_report(rows)
        with self.assertRaises(ValueError):
            rm_state_report([], ["c"], ["t"])

    def test_duplicate_pairs_and_missing_model_frames_fail(self):
        rows = panel()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            report(rows + [rows[0]])
        with self.assertRaisesRegex(ValueError, "panel"):
            report(rows[:-1])

    def test_inconsistent_targets_metadata_or_masks_across_models_fail(self):
        for field, value in (("target", .7), ("success", True), ("episode_length", 12),
                             ("initial_state_id", "other"), ("policy_panel", "different")):
            rows = panel()
            rows[-1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                report(rows)
        mask = np.ones(36, dtype=bool)
        mask[-1] = False
        with self.assertRaisesRegex(ValueError, "mask"):
            report(valid=mask)

    def test_bad_masks_and_nonfinite_or_out_of_range_values_fail(self):
        for mask in ([1] * 36, np.ones((36, 1), dtype=bool), [True] * 35, [False] * 36):
            with self.subTest(mask=str(mask)[:30]), self.assertRaises(ValueError):
                report(valid=mask)
        for field in ("target", "prediction"):
            for value in (float("nan"), float("inf"), -.1, 1.1, "0.5", True):
                rows = panel()
                rows[0][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    report(rows)

    def test_masked_nonfinite_rows_are_retained_as_null_with_common_mask(self):
        rows = panel()
        mask = np.ones(len(rows), dtype=bool)
        for i in (0, 12, 24):
            mask[i] = False
            rows[i]["prediction"] = float("nan")
            rows[i]["target"] = float("nan")
        result = report(rows, valid=mask)
        self.assertEqual(result["split"]["masked_pairs"], 1)
        raw = [r for r in result["raw_rows"] if not r["valid"]]
        self.assertEqual(len(raw), 3)
        self.assertTrue(all(r["prediction"] is None and r["target"] is None for r in raw))
        json.dumps(result, allow_nan=False)

    def test_entire_masked_episode_fails_instead_of_changing_panel(self):
        rows = panel()
        mask = np.array([r["episode_id"] != "cal-ok" for r in rows])
        with self.assertRaisesRegex(ValueError, "support"):
            report(rows, valid=mask)

    def test_invalid_row_metadata_is_rejected(self):
        for field, value in (("frame_index", -1), ("frame_index", 4), ("frame_index", .5),
                             ("episode_length", 0), ("episode_length", True), ("success", 1),
                             ("episode_id", []), ("model", ""), ("initial_state_id", None)):
            rows = panel()
            rows[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                report(rows)
        rows = panel()
        del rows[0]["target"]
        with self.assertRaisesRegex(ValueError, "target"):
            report(rows)

    def test_import_and_report_do_not_load_torch(self):
        code = ("import sys; from weaver_run.diagnostics.rm_report import rm_state_report; "
                "from weaver_run.diagnostics.test_rm_report import report; report(); "
                "assert 'torch' not in sys.modules")
        completed = subprocess.run([sys.executable, "-B", "-c", code],
                                   cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
