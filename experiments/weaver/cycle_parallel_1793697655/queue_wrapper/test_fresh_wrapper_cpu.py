"""Dedicated full-FRESH admission and final audit; no CUDA or remote execution."""
import copy
import json
import os
from pathlib import Path
import unittest

import training_entry as train
from test_manual_wrapper import good_state


def admitted_config():
    # Use the actual candidate profile rather than a copy of wrapper constants.
    candidate = Path(os.environ.get('FRESH_CANDIDATE_SOURCE',
        str(Path(__file__).resolve().parents[3] / 'weaver_fresh_v1')))
    path = candidate / 'examples/trajectory_01_imagined.json'
    backend = json.loads(path.read_text(encoding='utf8'))
    admission = {'configuration_admitted':True,'assets_checked':True,'errors':[],
        'environment':{'compatible':True},'online_training_environment':{
            'python':backend['backend_options']['training_python'],'compatible':True,
            'distributed_available':True,'nccl_available':True,
            'cuda_initialized':False,'gpu_access_performed':False}}
    return backend, dict(train.PIPELINE_RECIPE), admission


class FreshWrapperTests(unittest.TestCase):
    def test_actual_full_profile_is_admitted(self):
        backend, pipeline, admission = admitted_config()
        self.assertTrue(train.validate_backend_admission(backend,pipeline,admission))
        self.assertEqual(train.BACKEND_CONFIG,'examples/trajectory_01_imagined.json')

    def test_binary_reward_cannot_enter_full_fresh_job(self):
        backend, pipeline, admission = admitted_config()
        backend['backend_options']['native_reward_head'] = {
            'mode':'libero_success_score_v1','link':'sigmoid','loss':'brier',
            'label_source':'libero_env_step','calibrated':False,'terminal':'horizon_only',
            'balance':'global_cohort_inverse_frequency','loss_coefficient':1.0}
        with self.assertRaisesRegex(ValueError,'reward target/loss'):
            train.validate_backend_admission(backend,pipeline,admission)

    def test_each_of_seven_flags_is_required_and_boolean(self):
        names=('dense_reward','signal_telemetry','refresh_gae','replay_age_limits',
               'seed_balancing','early_seed_context','real_policy_replay')
        for name in names:
            for replacement in (False,1,None):
                backend,pipeline,admission=admitted_config()
                backend['backend_options']['fresh'][name]=replacement
                with self.subTest(name=name,replacement=replacement),self.assertRaises(ValueError):
                    train.validate_backend_admission(backend,pipeline,admission)
            backend,pipeline,admission=admitted_config()
            del backend['backend_options']['fresh'][name]
            with self.subTest(missing=name),self.assertRaises(ValueError):
                train.validate_backend_admission(backend,pipeline,admission)

    def test_changed_caps_quota_scheduler_or_unknown_options_rejected(self):
        for name,value in (('max_policy_age',17),('max_world_age',4),('max_sample_uses',5),
                           ('max_policy_age',True),('max_sample_uses',4.),('real_fraction',.5),
                           ('seed_scheduler_mode','random'),('seed_scheduler_seed',20261029),
                           ('unknown_future_option',True)):
            backend,pipeline,admission=admitted_config()
            backend['backend_options']['fresh'][name]=value
            with self.subTest(name=name,value=value),self.assertRaises(ValueError):
                train.validate_backend_admission(backend,pipeline,admission)

    def test_dense_gamma_units_loss_and_exact_types_required(self):
        for name,value in (('gamma_per_action',.99),('discount_unit','policy_chunks'),
                           ('balance','global_cohort_inverse_frequency'),('loss','brier'),
                           ('loss_coefficient',True),('calibrated',0)):
            backend,pipeline,admission=admitted_config()
            backend['backend_options']['native_reward_head'][name]=value
            with self.subTest(name=name,value=value),self.assertRaises(ValueError):
                train.validate_backend_admission(backend,pipeline,admission)

    def test_dense_final_audit_requires_target_not_binary_counts(self):
        state,_,_=good_state(15)
        world=state['checkpoint_cpu_audit']['online_world']
        self.assertTrue(train.validate_online_world_audit(world,state['run_id'],15,3)['passed'])
        for edit in (lambda row:row.pop('reward_target_counts'),
                     lambda row:row.update(reward_class_counts={'positive_count':1}),
                     lambda row:row['reward_target_counts'].update(disabled_updates=2),
                     lambda row:row['reward_target_counts'].update(masked_count=0),
                     lambda row:row['reward_target_counts'].update(valid_count=True)):
            changed=copy.deepcopy(world);edit(changed)
            self.assertFalse(train.validate_online_world_audit(changed,state['run_id'],15,3)['passed'])


if __name__=='__main__':unittest.main()
