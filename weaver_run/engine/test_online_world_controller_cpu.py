"""Native World orchestration admission and split artifact tests, no ML load."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import types
import tempfile
import unittest
from unittest import mock
from world_backend_runtime import (isolated_online_weaver, save_online_split_snapshot,
                                  prune_online_world_snapshots, validate_online_runtime_arguments)


def load_contracts():
    path=Path(__file__).parent/'world_backends'/'contracts.py'
    spec=importlib.util.spec_from_file_location('online_controller_candidate_contracts',path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


class AdmissionTests(unittest.TestCase):
    def config(self):
        return json.loads((Path(__file__).parent/'examples'/'weaver_integrated_internal.json').read_text(encoding='utf-8'))
    def test_exact_joint_config_accepts_native_ddp_and_integrated_reward_scope(self):
        module=load_contracts(); config=module.WorldBackendConfig.from_mapping(self.config())
        config.validate_admission()
        self.assertTrue(isolated_online_weaver(config))
        self.assertEqual(config.backend_options['online_training']['backend'],'torch_ddp')
        self.assertEqual(config.backend_options['online_training']['global_batch_size'],1024)
        self.assertEqual(config.backend_options['online_training']['trainable_scope'],'wm+rm')
        self.assertEqual(config.reward_provider,'world_head')
    def test_changed_world_recipe_or_source_is_rejected(self):
        module=load_contracts()
        for mutate in (lambda c:c['backend_options']['online_training'].update(micro_batch_size=8),
                       lambda c:c['backend_options']['online_training'].update(optimizer='ctrl_legacy'),
                       lambda c:c['backend_options'].update(initial_checkpoint_sha256='0'*64),
                       lambda c:c['schema'].update(n_history=3)):
            raw=self.config(); mutate(raw)
            with self.assertRaises(ValueError): module.WorldBackendConfig.from_mapping(raw).validate_admission()
    def test_online_in_process_native_load_is_rejected_before_actor_creation(self):
        config=types.SimpleNamespace(backend='weaver',world_trainable=True,backend_options={})
        with self.assertRaisesRegex(ValueError,'dedicated-interpreter'): isolated_online_weaver(config)
    def test_online_micro_gas_and_five_to_one_clock_admitted_before_actors(self):
        config=types.SimpleNamespace(backend='weaver',world_trainable=True,
            reward_provider='external_vlm',reward_semantics='delta_success',
            backend_options={'transport':'isolated_online_ddp'})
        values={'num_trainer_gpus':4,'ctrl_batch_size':4,'ctrl_accumulation_steps':64,
                'ctrl_train_interval':5,'reward_train_interval':5}
        validate_online_runtime_arguments(config,types.SimpleNamespace(**values))
        for key in values:
            changed=dict(values);changed[key]=True if key=='num_trainer_gpus' else values[key]+1
            with self.assertRaisesRegex(ValueError,'before actor construction'):
                validate_online_runtime_arguments(config,types.SimpleNamespace(**changed))


class SnapshotTests(unittest.TestCase):
    def setup_snapshot(self, policy_step=10):
        config=types.SimpleNamespace(backend='weaver',world_trainable=True,
            reward_provider='external_vlm',reward_semantics='delta_success',
            backend_options={'transport':'isolated_online_ddp','initial_checkpoint_sha256':'a'*64})
        groups={}
        for kind in ('policy','reward'):
            component={'kind':kind,'policy_step':policy_step,'model_step':policy_step if kind=='policy' else policy_step//5}
            groups[kind]=[types.SimpleNamespace(export_snapshot_component=types.SimpleNamespace(remote=mock.Mock(return_value=component)))]
        descriptor={'protocol':'weaver_online_weights_v1','backend':'weaver','world_trainable':True,
            'run_id':'this_run','world_size':4,'policy_step':policy_step,'successful_updates':policy_step//5,
            'model_version':policy_step//5,'tensor_count':866,'source_optimizer_steps':3200,
            'source_bundle_sha256':'a'*64,'weights_only':True,'weights_file':'/output/native.pt','weights_sha256':'b'*64}
        native=mock.Mock(return_value=descriptor)
        groups['ctrl']=[types.SimpleNamespace(export_native_snapshot=types.SimpleNamespace(remote=native))]
        return config,groups,descriptor,native
    def test_same_boundary_saves_only_policy_reward_and_exact_native_descriptor(self):
        config,groups,descriptor,native=self.setup_snapshot()
        save=mock.Mock(return_value={'ok':True})
        result=save_online_split_snapshot(groups,config,'/output','final',{'run_id':'this_run'},
                    ray_get=lambda x:x,save_split_weights=save)
        self.assertTrue(result['ok'])
        self.assertEqual(result['native_retention']['pruned'],[])
        self.assertEqual(set(save.call_args.args[0]),{'policy','reward'})
        self.assertEqual(save.call_args.kwargs['active_kinds'],('policy','reward'))
        self.assertIs(save.call_args.kwargs['metadata']['online_world'],descriptor)
        native.assert_called_once_with(str(Path('/output')/'checkpoints_world'),'final',10)
    def test_zero_update_initial_artifact_is_allowed_without_claimed_update(self):
        config,groups,descriptor,native=self.setup_snapshot(0)
        save=mock.Mock()
        save_online_split_snapshot(groups,config,'/output','initial',{'run_id':'this_run'},
            ray_get=lambda x:x,save_split_weights=save)
        self.assertEqual(descriptor['successful_updates'],0)
        self.assertEqual(descriptor['source_optimizer_steps'],3200)
    def test_mixed_policy_boundary_never_requests_native_snapshot(self):
        config,groups,descriptor,native=self.setup_snapshot()
        groups['reward'][0].export_snapshot_component.remote.return_value['policy_step']=9
        save=mock.Mock()
        with self.assertRaisesRegex(RuntimeError,'boundaries differ'):
            save_online_split_snapshot(groups,config,'/output','final',{'run_id':'this_run'},
                ray_get=lambda x:x,save_split_weights=save)
        native.assert_not_called();save.assert_not_called()
    def test_foreign_run_version_source_or_tensor_count_cannot_publish_monolith(self):
        for field,value in [('run_id','old'),('world_size',2),('successful_updates',1),('model_version',0),
                            ('source_bundle_sha256','c'*64),('source_optimizer_steps',3199),('tensor_count',865)]:
            config,groups,descriptor,native=self.setup_snapshot(); descriptor[field]=value
            save=mock.Mock()
            with self.assertRaisesRegex(RuntimeError,'Native WEAVER artifact'):
                save_online_split_snapshot(groups,config,'/output','final',{'run_id':'this_run'},
                    ray_get=lambda x:x,save_split_weights=save)
            save.assert_not_called()
    def test_actual_main_uses_online_factories_streaming_and_separate_snapshot_helper(self):
        source=(Path(__file__).parent/'train.py').read_text(encoding='utf-8')
        ast.parse(source)
        for actual in ('world_streaming=online_weaver','make_weaver_online_trainer(globals())',
                       'WeaverOnlineTrainerActor if online_weaver and training_role == "ctrl"',
                       'WeaverOnlineWorldInferenceActor if online_weaver',
                       'save_online_split_snapshot(training_groups, world_config'):
            self.assertIn(actual,source)
    def test_retention_keeps_initial_current_latest_two_and_never_unknown_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();folder=root/'checkpoints_world';folder.mkdir()
            names=['world_p00000000_w00000000_initial.pt']+[
                f'world_p{p:08d}_w{p//5:08d}_periodic.pt' for p in (25,50,75,100)]
            for name in names:(folder/name).write_bytes(b'immutable fixture')
            unknown=folder/'do_not_touch.pt';unknown.write_bytes(b'unknown')
            current=folder/names[-1]
            descriptor={'weights_file':str(current),'weights_sha256':'f'*64}
            # New native file alone gives no permission to prune old artifacts.
            self.assertEqual(prune_online_world_snapshots(root,descriptor)['pruned'],[])
            (root/'manifest.json').write_text(json.dumps({'metadata':{'online_world':descriptor}}))
            result=prune_online_world_snapshots(root,descriptor)
            self.assertEqual(set(result['pruned']),set(names[1:3]))
            self.assertTrue((folder/names[0]).is_file())
            self.assertTrue((folder/names[-2]).is_file())
            self.assertTrue(current.is_file());self.assertTrue(unknown.is_file())
    def test_failed_new_monolith_save_cannot_delete_previous_native_artifacts(self):
        config,groups,descriptor,native=self.setup_snapshot()
        with mock.patch('world_backend_runtime.prune_online_world_snapshots') as prune:
            with self.assertRaisesRegex(OSError,'disk'):
                save_online_split_snapshot(groups,config,'/output','final',{'run_id':'this_run'},
                    ray_get=lambda x:x,save_split_weights=mock.Mock(side_effect=OSError('disk full')))
            prune.assert_not_called()


if __name__=='__main__':unittest.main()
