"""Exercise actual orchestration AST for both service graphs, without GPU actors."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest

from test_main_startup_cpu import (TREE, SOURCE, Context, arguments, definition,
                                  execute_definition, startup_ast)
from test_world_contracts_cpu import weaver_config
from world_backends import WorldBackendConfig
from world_backends.libero import ACTION_SCHEMA, STATE_SCHEMA, validate_online_schema


def online_config(integrated):
    value = weaver_config(integrated).to_dict()
    value["schema"].update(action_schema_id=ACTION_SCHEMA, state_schema_id=STATE_SCHEMA,
                           image_keys=["full_image", "wrist_image"])
    value["checkpoint_metadata"]["schema"] = copy.deepcopy(value["schema"])
    return value


class PluggableMain(unittest.TestCase):
    def smoke(self, integrated):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "world.json"
            config_path.write_text(json.dumps(online_config(integrated)), encoding="utf-8")
            args = arguments()
            args.world_backend_config = str(config_path)
            from world_backends.runtime import context_geometry
            args.num_step_cond = context_geometry(WorldBackendConfig.from_mapping(online_config(integrated)))['required_context_frames']
            args.ctrl_world_ckpt = None  # No CTRL dependencies/checkpoint for WEAVER.
            args.num_reward_inference_actors = 0 if integrated else 4
            if integrated:
                # An absent service must not impose its learning schedule on startup.
                args.reward_warmup_steps = args.train_iters
                args.reward_train_interval = 0
            context = Context()
            result = execute_definition(startup_ast(), context.namespace())(args)
            return context, result

    def test_integrated_creates_16_models_and_four_replay_consumers(self):
        context, result = self.smoke(True)
        models = [a for a in context.actors if a.kind is not None]
        self.assertEqual(len(models), 16)
        self.assertEqual(set(result["training_groups"]), {"policy", "ctrl"})
        self.assertEqual(result["reward_inference_pool"], [])
        coordinator = next(a for a in context.actors if a.name == "GlobalBatchCoordinator")
        self.assertEqual(coordinator.kwargs["wm_consumers"], ("ctrl",))
        self.assertEqual(len(result["all_trainers"]), 8)
        self.assertEqual(len([x for x in context.actions if x[:1] == ("ACK",)]), 8)
        self.assertEqual(len([x for x in context.actions if x[:1] == ("release",)]), 8)
        self.assertEqual(set(context.snapshots[0][0]), {"policy", "ctrl"})
        self.assertEqual(context.snapshots[0][2]["active_kinds"], ("policy", "ctrl"))
        self.assertTrue(all(a.kwargs["world_config"]["reward_provider"] == "world_head"
                            for a in context.actors if a.name == "RolloutWorkerActor"))
        self.assertFalse(any(a.name in ("RewardInferenceActor", "CtrlWorldInferenceActor") for a in context.actors))

    def test_external_keeps24_and_real_reward_publication(self):
        context, result = self.smoke(False)
        self.assertEqual(len([a for a in context.actors if a.kind is not None]), 24)
        self.assertEqual(len(result["reward_inference_pool"]), 4)
        self.assertEqual(len([x for x in context.actions if x[:2] == ("ACK", "reward")]), 4)
        self.assertEqual(set(context.snapshots[0][0]), {"policy", "ctrl", "reward"})

    def test_native_droid_schema_rejected_before_actor_creation(self):
        config = online_config(False)
        config["schema"]["action_schema_id"] = "droid_joint_delta8"
        with self.assertRaisesRegex(ValueError, "incompatible robot schema"):
            validate_online_schema(WorldBackendConfig.from_mapping(config))

    def test_integrated_epoch_merge_requires_both_real_groups(self):
        merge = execute_definition(definition(TREE, "merge_split_epoch_results"), {})
        policy = [(0, 0, 0, 0, 0, {}, 5, 0, 0, {}, {}) for _ in range(4)]
        world = [{"kind": "ctrl", "rank": i, "policy_step": 5, "model_step": 1,
                  "metrics": {"trained_ctrl": True, "trained_reward": False,
                              "ctrl_sample_unique_episodes": 2, "ctrl_sample_fresh_episode_ratio": .5},
                  "learning_rates": {"ctrl": .001}, "wm_cohort": {"id": 1},
                  "performance": {"wm_sample_time": 1, "wm_prep_time": 2, "wm_batch_wait_time": 0}}
                 for i in range(4)]
        result = merge(policy + world, 4, 5, 5, 5, ("policy", "ctrl"))
        self.assertEqual(len(result), 4)
        self.assertTrue(all(row[10]["trained_ctrl"] and not row[10]["trained_reward"] for row in result))
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            merge(policy + world[:-1], 4, 5, 5, 5, ("policy", "ctrl"))
        world[2]["model_step"] = 0
        with self.assertRaisesRegex(RuntimeError, "diverged"):
            merge(policy + world, 4, 5, 5, 5, ("policy", "ctrl"))

    def test_policy_value_gae_ctrl_loss_and_real_eval_bodies_preserved(self):
        baseline = SOURCE / "tests/fixtures/baseline_train.py"
        old_tree = ast.parse(baseline.read_text(encoding="utf-8"))
        def method(tree, cls, name):
            owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
            node = next(n for n in owner.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
            return ast.dump(node, include_attributes=False)
        # GAE arithmetic is checked separately in test_pipeline_producers_cpu;
        # its producer publication now waits for the bounded Replay ACK.
        for cls, name in (("AcceRLCtrlWorld", "forward"),
                          ("EvaluationWorkerActor", "run"),
                          ("TrainerActor", "_train_policy")):
            self.assertEqual(method(old_tree, cls, name), method(TREE, cls, name), (cls, name))


if __name__ == "__main__":
    unittest.main()
