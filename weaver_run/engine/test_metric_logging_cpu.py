"""Execute the real writer and driver logging AST without importing GPU code."""
import ast
import copy
from contextlib import closing
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest

from telemetry import Telemetry
from training_topology import combine_wm_metrics


SOURCE = Path(__file__).with_name("train.py")
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))
VISIBLE = {
    "Eval/_Global/Average_Return", "Eval/_Global/Average_Episode_Length",
    "Rollout/_Global/Average_Return", "Rollout/_Global/Average_Episode_Length",
    "Loss/Policy", "Loss/Value", "Loss/CtrlWorld", "Loss/Reward_Model",
    "Metrics/Entropy", "Metrics/KL_Divergence",
    "Performance/policy_train_time", "Performance/ctrl_train_time", "Performance/sync_time",
    "System/Replay_Buffer_Size_Total", "System/WM_Replay_Buffer_Size_Total",
}


class FakeTensorBoard:
    def __init__(self, path, flush_secs):
        self.scalars, self.flushed, self.closed = [], False, False

    def add_scalar(self, tag, value, step, **kwargs):
        self.scalars.append((tag, value, step, kwargs))

    def flush(self):
        self.flushed = True

    def close(self):
        self.closed = True


class FakeTelemetry:
    def __init__(self, actor, actor_id):
        self.events, self.flushed, self.closed = [], False, False

    def config(self, value):
        self.configuration = value

    def event(self, kind, **fields):
        self.events.append((kind, fields))

    def flush(self):
        self.flushed = True

    def close(self):
        self.closed = True


def make_writer(enabled=True, telemetry=FakeTelemetry):
    node = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "CaptureSummaryWriter")
    namespace = dict(SummaryWriter=FakeTensorBoard, Telemetry=telemetry,
                     args=SimpleNamespace(ent_coef=0), __file__=str(SOURCE), Path=Path,
                     os=os, sys=sys, torch=SimpleNamespace(__version__="fake"),
                     ray=SimpleNamespace(__version__="fake"))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["CaptureSummaryWriter"]("unused", tensorboard=enabled)


def has_tag(node, tag):
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == "add_scalar" and n.args
               and isinstance(n.args[0], ast.Constant) and n.args[0].value == tag
               for n in ast.walk(node))


def driver_logging_code():
    """Keep the real logging guards/expressions; exclude actor RPCs and training.

    Include World logging both inside and outside the periodic log guard, so
    moving it back under that guard would fail the sparse-update regression.
    """
    main = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    periodic = next(n for n in ast.walk(main) if isinstance(n, ast.If)
                    and any(isinstance(c, ast.Attribute) and c.attr == "log_interval_seconds"
                            for c in ast.walk(n.test)))
    first = next(i for i, n in enumerate(periodic.body) if has_tag(n, "Loss/Total"))
    end = next(i for i, n in enumerate(periodic.body) if has_tag(n, "Rollout/_Global/Average_Return"))
    log = copy.deepcopy(periodic)
    log.body, log.orelse = log.body[first:end], []
    periodic_ids = {id(n) for n in ast.walk(periodic)}
    outside = [n for n in ast.walk(main) if isinstance(n, ast.If) and id(n) not in periodic_ids
               and has_tag(n, "Loss/CtrlWorld") and not has_tag(n, "Loss/Policy")]
    # Keep only outermost blocks (their nested guards execute normally).
    outside = [n for n in outside if not any(n is not parent and n in ast.walk(parent) for parent in outside)]
    return compile(ast.Module(body=sorted([*outside, log], key=lambda n: n.lineno), type_ignores=[]),
                   str(SOURCE), "exec")


def log_epoch(writer, wm_metrics, due=True, step=5, performance=None):
    perf = performance if performance is not None else {
        "policy_sample_time": .1, "policy_prep_time": .2, "policy_train_time": 2.,
        "policy_batch_wait_time": .3, "wm_sample_time": 0., "wm_prep_time": 0.,
        "wm_batch_wait_time": 0., "ctrl_train_time": 0., "reward_train_time": 0.,
    }
    namespace = dict(writer=writer, wm_metrics=wm_metrics, global_step=step,
                     perf_metrics_list=[perf], results=[(None,) * 9 + (perf, wm_metrics)],
                     np=SimpleNamespace(mean=lambda values: sum(values) / len(values)), math=math,
                     current_time=10 if due else 0, last_log_time=0,
                     args=SimpleNamespace(log_interval_seconds=1),
                     total_losses=[3.], p_losses=[1.], v_losses=[2.], e_losses=[0.], kl_losses=[.01],
                     ents=[.5], avg_kl_divs=[.2], training_speed_steps_per_sec=1., timing_stats={},
                     train_time=2., sync_time=.2, t_train_start=0, time=SimpleNamespace(time=lambda: 10))
    exec(driver_logging_code(), namespace)


class MetricLoggingTests(unittest.TestCase):
    def test_main_selects_loss_slots_from_the_configured_backend(self):
        main = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        start = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'writer' for t in n.targets))
        end = next(i for i in range(start + 1, len(main.body)) if isinstance(main.body[i], ast.If))
        code = compile(ast.Module(body=main.body[start:end], type_ignores=[]), str(SOURCE), 'exec')
        for online, provider, expected in (
            (False, 'external_vlm', {'Loss/CtrlWorld', 'Loss/Reward_Model'}),
            (True, 'external_vlm', {'Loss/World_Flow', 'Loss/Reward_Model'}),
            (True, 'world_head', {'Loss/World_Flow', 'Loss/World_RM'}),
        ):
            writer = make_writer()
            exec(code, dict(CaptureSummaryWriter=lambda *a, **kw: writer, log_dir='unused', os=os,
                            online_weaver=online, world_config=SimpleNamespace(reward_provider=provider)))
            for tag in ('Loss/CtrlWorld', 'Loss/Reward_Model', 'Loss/World_Flow', 'Loss/World_RM'):
                writer.add_scalar(tag, 1., 5)
            self.assertEqual({row[0] for row in writer.tensorboard.scalars}, expected)

    def test_native_losses_replace_legacy_slots_without_expanding_compact_view(self):
        for native_reward in (False, True):
            writer = make_writer()
            writer.native_world_losses = True
            writer.native_reward_losses = native_reward
            for tag in VISIBLE | {"Loss/World_Flow", "Loss/World_RM"}:
                writer.add_scalar(tag, .5, 5)
            expected = (VISIBLE - {"Loss/CtrlWorld"}) | {"Loss/World_Flow"}
            if native_reward:
                expected = (expected - {"Loss/Reward_Model"}) | {"Loss/World_RM"}
            self.assertEqual({row[0] for row in writer.tensorboard.scalars}, expected)
            self.assertEqual(len(writer.tensorboard.scalars), 15)
            self.assertEqual(len(writer.telemetry.events), 17)

    def test_native_components_are_rank_means_and_only_emitted_on_world_updates(self):
        writer = make_writer()
        writer.native_world_losses = writer.native_reward_losses = True
        # Rank 0 has no valid targets; the already scaled RM contributions must
        # be averaged equally, not taken from rank 0 or reweighted by masks.
        ranks = [dict(trained_ctrl=True, trained_reward=False, ctrl_loss=flow + rm,
                      native_flow_loss=flow, native_half_mse_loss=rm)
                 for flow, rm in ((1., 0.), (2., .2), (3., .6), (4., 0.))]
        combined = combine_wm_metrics(ranks)
        self.assertAlmostEqual(combined['native_half_mse_loss'], .2)
        self.assertAlmostEqual(combined['native_flow_loss'], 2.5)
        self.assertAlmostEqual(combined['ctrl_loss'], 2.7)
        log_epoch(writer, combined, due=False)
        log_epoch(writer, dict(combined, trained_ctrl=False), due=True, step=6)
        visible = {row[0]: row[1] for row in writer.tensorboard.scalars if row[0].startswith('Loss/World_')}
        self.assertEqual(visible, {'Loss/World_Flow': 2.5, 'Loss/World_RM': .2})
        self.assertNotIn('Loss/CtrlWorld', {row[0] for row in writer.tensorboard.scalars})
        raw = [fields for _, fields in writer.telemetry.events]
        self.assertEqual([r['step'] for r in raw if r['tag'] == 'Loss/World_RM'], [5])
        self.assertEqual([r['value'] for r in raw if r['tag'] == 'Loss/CtrlWorld'], [2.7])

    def test_optional_native_components_require_all_ranks_and_preserve_zero(self):
        for key in ('native_half_mse_loss', 'native_half_brier_loss', 'native_flow_loss'):
            self.assertIsNone(combine_wm_metrics([{key: 1.}, {key: None}]).get(key))
            self.assertIsNone(combine_wm_metrics([{key: 1.}, {}]).get(key))
            self.assertEqual(combine_wm_metrics([{key: 0.}, {key: 0.}])[key], 0.)
        for key in ('native_half_mse_loss', 'native_half_brier_loss'):
            writer = make_writer()
            writer.native_world_losses = writer.native_reward_losses = True
            log_epoch(writer, {'trained_ctrl': True, 'trained_reward': False,
                               'ctrl_loss': 1., 'native_flow_loss': 1., key: 0.}, due=False)
            self.assertIn(('Loss/World_RM', 0., 5), [row[:3] for row in writer.tensorboard.scalars])

    def test_compact_allowlist_keeps_raw_events_and_unknown_high_diagnostics(self):
        writer = make_writer()
        hidden = {"Loss/Total", "Loss/Entropy", "Loss/KL", "Train/Learning_Rate/Policy",
                  "System/MPS_Enabled", "Eval/Target85/Threshold", "Eval/task0/Success_Rate",
                  "TimeTo85/First_Elapsed_Seconds", "WorldModel/Diagnostics/Ctrl_Grad_Norm",
                  "Pipeline/new_high_cardinality/999", "Performance/new_diagnostic"}
        for tag in sorted(VISIBLE | hidden):
            writer.add_scalar(tag, 1e12, 7, walltime=123.25, double_precision=True)
        self.assertEqual({row[0] for row in writer.tensorboard.scalars}, VISIBLE)
        self.assertEqual({fields["tag"] for _, fields in writer.telemetry.events}, VISIBLE | hidden)
        self.assertTrue(all(fields["ts_ns"] == 123250000000 and fields["step"] == 7
                            and fields["value"] == 1e12 for _, fields in writer.telemetry.events))
        self.assertTrue(all(row[3] == {"walltime": 123.25, "new_style": True, "double_precision": True}
                            for row in writer.tensorboard.scalars))

    def test_flat_warmup_and_single_allowed_points_are_not_dropped(self):
        writer = make_writer()
        for step in range(30):
            writer.add_scalar("Metrics/KL_Divergence", 0., step)
        writer.add_scalar("Eval/_Global/Average_Return", .85, 30)
        self.assertEqual(len(writer.tensorboard.scalars), 31)

    def test_capture_only_and_writer_lifecycle(self):
        for enabled in (False, True):
            writer = make_writer(enabled)
            writer.add_scalar("Loss/Total", 3., 1)
            writer.flush()
            writer.close()
            self.assertEqual(writer.telemetry.events, [("metric", {"tag": "Loss/Total", "value": 3., "step": 1})])
            self.assertTrue(writer.telemetry.flushed and writer.telemetry.closed)
            if enabled:
                self.assertTrue(writer.tensorboard.flushed and writer.tensorboard.closed)
                self.assertEqual(writer.tensorboard.scalars, [])
            else:
                self.assertIsNone(writer.tensorboard)

    def test_sqlite_retains_hidden_scalars_and_anomaly_events(self):
        with tempfile.TemporaryDirectory() as folder:
            db = str(Path(folder) / "events.sqlite")
            writer = make_writer(telemetry=lambda actor, actor_id: Telemetry(
                actor, actor_id, db_path=db, run_id="metric-test"))
            try:
                writer.add_scalar("Loss/Entropy", 0., 7, walltime=123.25)
                writer.add_scalar("Unexpected/Very_Large_Value", 1e12, 7)
                writer.telemetry.event("optimizer_update_failed", details={"reason": "overflow", "step": 7})
                writer.flush()
            finally:
                writer.close()
            with closing(sqlite3.connect(db)) as connection:
                events = [(kind, stamp, json.loads(fields)) for kind, stamp, fields in connection.execute(
                    "SELECT kind,ts_ns,details_json FROM events ORDER BY id")]
            metrics = [row for row in events if row[0] == "metric"]
            self.assertEqual(metrics[0], ("metric", 123250000000,
                                         {"tag": "Loss/Entropy", "value": 0., "step": 7}))
            self.assertEqual(metrics[1][2], {"tag": "Unexpected/Very_Large_Value", "value": 1e12, "step": 7})
            self.assertIn(("optimizer_update_failed", {"reason": "overflow", "step": 7}),
                          [(kind, fields) for kind, _, fields in events])
            self.assertEqual(writer.tensorboard.scalars, [])

    def test_ordinary_update_has_no_world_zero_or_stale_observations(self):
        for stale in ({}, {"ctrl_loss": 9., "ctrl_grad_norm": 2., "ctrl_clip_ratio": 1.,
                           "reward_loss": 8., "reward_grad_norm": 3., "reward_clip_ratio": 1.,
                           "ctrl_sample_unique_episodes": 6, "ctrl_sample_fresh_episode_ratio": .5}):
            writer = make_writer()
            log_epoch(writer, {"trained_ctrl": False, "trained_reward": False, **stale})
            tags = {fields["tag"] for _, fields in writer.telemetry.events}
            self.assertFalse({tag for tag in tags if tag.startswith("WorldModel/")
                              or tag in {"Loss/CtrlWorld", "Loss/Reward_Model"}
                              or tag.startswith(("Performance/wm_", "Performance/ctrl_", "Performance/reward_"))}, tags)
            self.assertIn("Loss/Policy", tags)
            self.assertIn("Loss/Entropy", tags)  # Still evidence even with ent_coef=0.

    def test_world_updates_survive_periodic_log_throttle_and_keep_real_zero(self):
        for due in (False, True):
            writer = make_writer()
            log_epoch(writer, {"trained_ctrl": True, "trained_reward": False,
                               "ctrl_loss": 0., "ctrl_grad_norm": 0., "ctrl_clip_ratio": 0.,
                               "ctrl_sample_unique_episodes": 4, "ctrl_sample_fresh_episode_ratio": 0.}, due=due)
            raw = [fields for _, fields in writer.telemetry.events]
            self.assertEqual([r["value"] for r in raw if r["tag"] == "Loss/CtrlWorld"], [0.])
            self.assertEqual([r["value"] for r in raw if r["tag"] == "WorldModel/Diagnostics/Fresh_Episode_Ratio"], [0.])
            self.assertEqual([r["value"] for r in raw if r["tag"] == "WorldModel/Diagnostics/Ctrl_Grad_Norm"], [0.])
            self.assertNotIn("Loss/Reward_Model", {r["tag"] for r in raw})
            self.assertEqual([r[:3] for r in writer.tensorboard.scalars if r[0] == "Loss/CtrlWorld"],
                             [("Loss/CtrlWorld", 0., 5)])

    def test_missing_world_measurements_are_not_filled_with_zero(self):
        writer = make_writer()
        log_epoch(writer, {"trained_ctrl": True, "trained_reward": False, "ctrl_loss": 2.},
                  performance={"policy_sample_time": .1, "policy_prep_time": .2, "policy_train_time": 2.})
        tags = {fields["tag"] for _, fields in writer.telemetry.events}
        self.assertNotIn("WorldModel/Diagnostics/Sampled_Unique_Episodes", tags)
        self.assertNotIn("WorldModel/Diagnostics/Fresh_Episode_Ratio", tags)
        self.assertNotIn("Performance/wm_sample_time", tags)

    def test_reward_only_update_keeps_reward_evidence_without_ctrl_observations(self):
        writer = make_writer()
        log_epoch(writer, {"trained_ctrl": False, "trained_reward": True, "reward_loss": .4,
                           "reward_grad_norm": .5, "reward_clip_ratio": 0., "reward_tp": 3,
                           "reward_tn": 2, "reward_fp": 1, "reward_fn": 1, "reward_pos_ratio_batch": .6,
                           "ctrl_sample_unique_episodes": 8, "ctrl_sample_fresh_episode_ratio": .5}, due=False)
        raw = {fields["tag"]: fields["value"] for _, fields in writer.telemetry.events}
        self.assertEqual(raw.get("Loss/Reward_Model"), .4)
        self.assertEqual(raw.get("WorldModel/Reward_Pos_Acc"), .75)
        self.assertEqual(raw.get("WorldModel/Diagnostics/Reward_Grad_Norm"), .5)
        self.assertNotIn("Loss/CtrlWorld", raw)
        self.assertNotIn("WorldModel/Diagnostics/Sampled_Unique_Episodes", raw)
        self.assertNotIn("Performance/ctrl_train_time", raw)


if __name__ == "__main__":
    unittest.main()
