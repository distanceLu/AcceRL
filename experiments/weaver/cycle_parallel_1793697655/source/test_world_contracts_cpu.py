"""Standard-library tests; no torch, Ray, model weights, or GPU required."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from world_backends import (DeltaSuccessRewardProvider, EpisodeWindow, RolloutResult,
                            WorldBackendConfig, build_reward_provider, build_service_plan)


def schema():
    return {"domain": "libero", "action_schema_id": "libero_ee_delta7_v1",
            "state_schema_id": "libero_ee_axis_angle_gripper8_v1", "encoder_id": "adapted_encoder_v1",
            "normalizer_id": "libero_train_stats_v1", "action_dim": 7, "state_dim": 8,
            "alignment": "action_into_frame", "image_keys": ["agentview", "wrist"],
            "image_size": [192, 320], "image_value_range": "minus_one_one", "dt_seconds": 0.05,
            "n_history": 2, "n_memory_frames": 1, "t_memory": 5}


def weaver_config(integrated=False):
    config_schema = schema()
    metadata = {"checkpoint_id": "synthetic_test_checkpoint", "code_revision": "synthetic_test_revision",
                "schema": copy.deepcopy(config_schema), "load_report": {
                    "strict": True, "missing_keys": [], "unexpected_keys": [], "evidence_id": "synthetic_load_test"},
                "success_head": {"calibrated": True, "label_semantics": "environment_success",
                                 "calibration_id": "synthetic_success_test", "source": "rm", "output": "logit"},
                "terminal_head": {"calibrated": True, "label_semantics": "environment_terminal",
                                  "calibration_id": "synthetic_terminal_test", "source": "same_success", "rule": "argmax"}}
    return WorldBackendConfig(backend="weaver", reward_provider="world_head" if integrated else "external_vlm",
                              schema=config_schema, checkpoint_metadata=metadata)


def window():
    return EpisodeWindow(observations=[object(), object(), object()], actions=[[0.] * 7] * 3,
                         states=[[0.] * 8] * 3, rewards=[0., 0., 1.], success=[False, False, True],
                         terminated=[False, False, True], truncated=[False] * 3, valid_mask=[True] * 3,
                         dt=[0., .05, .05], action_mask=[False, True, True], episode_id="episode-1",
                         sample_id="window-1", task_id="task-0", instruction="move the cup", schema=schema(),
                         label_source="libero_environment")


class ConfigTests(unittest.TestCase):
    def test_default_ctrl_admission_and_json(self):
        WorldBackendConfig().validate_admission()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "world.json"
            path.write_text(json.dumps({"backend": "ctrl"}), encoding="utf-8")
            self.assertEqual(WorldBackendConfig.from_json(path), WorldBackendConfig())

    def test_unknown_keys_fail_instead_of_ignoring_typos(self):
        with self.assertRaisesRegex(ValueError, "Unknown"):
            WorldBackendConfig.from_mapping({"reward_provier": "world_head"})

    def test_validated_weaver_requires_complete_checkpoint_evidence(self):
        config = weaver_config()
        config.validate_admission()
        config.checkpoint_metadata["schema"]["action_schema_id"] = "droid_joint8_v1"
        with self.assertRaisesRegex(ValueError, "action_schema_id"):
            config.validate_admission()

    def test_checkpoint_missing_keys_fail(self):
        config = weaver_config()
        config.checkpoint_metadata["load_report"]["missing_keys"] = ["rm.layer.weight"]
        with self.assertRaisesRegex(ValueError, "missing or unexpected"):
            config.validate_admission()

    def test_official_progress_cannot_be_success(self):
        config = weaver_config(True)
        config.checkpoint_metadata["success_head"]["label_semantics"] = "robometer_progress_minus_one"
        with self.assertRaisesRegex(ValueError, "native progress"):
            config.validate_admission()

    def test_terminal_calibration_is_independent_requirement(self):
        config = weaver_config(True)
        config.checkpoint_metadata["terminal_head"]["calibrated"] = False
        with self.assertRaisesRegex(ValueError, "terminal_head"):
            config.validate_admission()

    def test_integrated_head_and_native_progress_modes(self):
        config = weaver_config(True)
        self.assertTrue(config.capabilities().provides_terminal)
        with self.assertRaises(NotImplementedError):
            build_reward_provider({"backend": "weaver", "reward_semantics": "native_progress"})
        with self.assertRaises(NotImplementedError):
            WorldBackendConfig(reward_semantics="native_progress").validate_admission()


class ServicePlanTests(unittest.TestCase):
    def test_external_and_integrated_counts_and_schedule(self):
        external = build_service_plan()
        integrated = build_service_plan({"backend": "weaver", "reward_provider": "world_head"})
        self.assertEqual((external.expected_gpu_model_processes, integrated.expected_gpu_model_processes), (24, 16))
        self.assertEqual((external.expected_wm_consumers, integrated.expected_wm_consumers), (8, 4))
        self.assertEqual(external.wm_consumers, ("ctrl", "reward"))
        self.assertEqual(integrated.wm_consumers, ("ctrl",))
        self.assertEqual(integrated.expected_optimizer_steps(10), {"policy": 10, "ctrl": 2})
        self.assertEqual(external.expected_optimizer_steps(10), {"policy": 10, "ctrl": 2, "reward": 2})
        self.assertEqual(external.logical_update_ratio, "5:1:1")
        self.assertEqual(integrated.logical_update_ratio, "5:1")
        self.assertEqual(integrated.due_kinds(4), ("policy",))
        self.assertEqual(integrated.due_kinds(5), ("policy", "ctrl"))
        self.assertNotIn("reward", integrated.checkpoint_kinds)

    def test_four_ranks_are_a_runtime_constraint(self):
        plan = build_service_plan(world_size=2)
        self.assertEqual(plan.expected_gpu_model_processes, 12)
        with self.assertRaisesRegex(ValueError, "four ranks"):
            plan.validate_runtime()

    def test_optional_receipts_require_every_active_role_once(self):
        plan = build_service_plan({"backend": "weaver", "reward_provider": "world_head"})
        plan.validate_runtime()
        receipts = [{"phase": phase, "kind": kind, "rank": rank} for phase in ("trainer", "inference")
                    for kind in plan.active_kinds for rank in range(4)]
        plan.validate_runtime(role_receipts=receipts)
        with self.assertRaisesRegex(ValueError, "exactly once"):
            plan.validate_runtime(role_receipts=receipts + [receipts[0]])
        with self.assertRaisesRegex(ValueError, "exactly once"):
            plan.validate_runtime(role_receipts=receipts[:-1])


class DataContractTests(unittest.TestCase):
    def test_frame_aligned_state_reward_and_identity(self):
        sample = window()
        self.assertIs(sample.validate(weaver_config()), sample)
        self.assertEqual(EpisodeWindow.from_mapping(sample.to_dict()).episode_id, "episode-1")

    def test_missing_state_and_action_mismatch_rejected(self):
        sample = window()
        sample.states = None
        with self.assertRaisesRegex(ValueError, "Missing measured state"):
            sample.validate(weaver_config())
        sample = window()
        sample.actions = sample.actions[:-1]
        with self.assertRaisesRegex(ValueError, "Action count"):
            sample.validate(weaver_config())

    def test_wrong_state_semantics_and_time_scale_rejected(self):
        sample = window()
        sample.schema["state_schema_id"] = "droid_joint_position8"
        with self.assertRaisesRegex(ValueError, "state_schema_id"):
            sample.validate(weaver_config())
        sample = window()
        sample.dt[-1] = .1
        with self.assertRaisesRegex(ValueError, "time scale"):
            sample.validate(weaver_config())

    def test_scalar_tail_reward_not_promoted_to_frame_labels(self):
        sample = window()
        sample.rewards = 1.0
        with self.assertRaisesRegex(ValueError, "frame sequence"):
            sample.validate()

    def test_actions_and_padding_masks_align(self):
        sample = window()
        sample.valid_mask[-1] = False
        with self.assertRaisesRegex(ValueError, "padded observation"):
            sample.validate()

    def test_rollout_state_is_opaque_but_outputs_are_aligned(self):
        opaque = object()
        result = RolloutResult(["frame"], opaque, 5, "encoder", "normalizer", {"request_id": "r1"},
                               predicted_state=[[0.] * 8], success_probabilities=[.2], terminal_predictions=[False], dt=[.05])
        result.validate(1, require_reward=True, require_state=True)
        self.assertIs(result.to_dict()["opaque_next_state"], opaque)
        result.success_probabilities = [1.2]
        with self.assertRaisesRegex(ValueError, "not native progress"):
            result.validate()


class RewardTests(unittest.TestCase):
    def test_original_softmax_difference_argmax_and_tie(self):
        reward = DeltaSuccessRewardProvider(reward_scale=2.0).from_logits([0., 0.], [[0., math.log(3.)], [0., 0.]])
        self.assertAlmostEqual(reward.rewards[0], .5)
        self.assertAlmostEqual(reward.rewards[1], -.5)
        self.assertEqual(reward.terminated, (True, False))
        self.assertFalse(reward.bootstrap_allowed)
        self.assertEqual(sum(reward.rewards), 0.)

    def test_explicit_probability_terminal_rule(self):
        reward = DeltaSuccessRewardProvider(terminal_rule="success_probability", terminal_threshold=.8)
        result = reward.from_probabilities(.2, [.5, .8])
        self.assertEqual(result.terminated, (False, True))
        with self.assertRaisesRegex(ValueError, "argmax semantics"):
            DeltaSuccessRewardProvider().from_probabilities(.2, [.5])

    def test_native_progress_and_nonfinite_logits_rejected(self):
        with self.assertRaisesRegex(ValueError, "not progress"):
            DeltaSuccessRewardProvider(terminal_rule="success_probability").from_probabilities(0., [-.2])
        with self.assertRaisesRegex(ValueError, "finite logits"):
            DeltaSuccessRewardProvider().from_logits([0., 0.], [[0., float("nan")]])


if __name__ == "__main__":
    unittest.main()
