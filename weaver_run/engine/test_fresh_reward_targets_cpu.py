"""Observed-outcome target/censor/time-index contracts; no ML runtime required."""
import copy
from types import SimpleNamespace
import unittest

import numpy as np

from fresh_reward_targets import (dense_reward_specification, derive_episode_reward_targets,
    reward_target_payload, validate_reward_target_payload, dense_reward_cohort,
    validate_dense_reward_cohort)


def episode(outcome='success',count=41,identity='ep'):
    labels=[np.zeros(count,dtype=bool) for _ in range(3)]
    success,terminated,truncated=labels
    if outcome=='success':success[-1]=terminated[-1]=True
    elif outcome in ('failure','censored'):truncated[-1]=True
    reasons={'success':'success','failure':'task_horizon','censored':'real_env_budget','incomplete':'incomplete'}
    return SimpleNamespace(episode_id=identity,success=success,terminated=terminated,truncated=truncated,
        metadata={'label_source':'libero_env_step','budget_censored':outcome=='censored',
                  'termination_reason':reasons[outcome],'action_steps':count-1})


class DenseTargetContract(unittest.TestCase):
    def setUp(self):self.spec=dense_reward_specification()

    def derive(self,ep):
        return derive_episode_reward_targets(ep.success,ep.terminated,ep.truncated,ep.metadata,self.spec)

    def test_complete_success_frame_indices_and_chunk_discount_unit(self):
        ep=episode();result=self.derive(ep);target=result['reward_targets']
        self.assertEqual(target.dtype,np.float32)
        self.assertEqual(target[-1],1.)
        self.assertAlmostEqual(float(target[-9]),.99,places=7)
        self.assertAlmostEqual(float(target[0]),.99**5,places=7)
        self.assertTrue(result['reward_target_mask'].all())
        np.testing.assert_array_equal(ep.success,np.arange(41)==40)

    def test_failure_and_budget_truncation_have_different_supervision(self):
        failure=self.derive(episode('failure'));censored=self.derive(episode('censored'))
        self.assertFalse(failure['reward_targets'].any());self.assertFalse(censored['reward_targets'].any())
        self.assertTrue(failure['reward_target_mask'].all());self.assertFalse(censored['reward_target_mask'].any())
        self.assertFalse(self.derive(episode('incomplete'))['reward_target_mask'].any())

    def test_true_environment_failure_is_valid_without_timeout(self):
        ep=episode('failure');ep.truncated[-1]=False;ep.terminated[-1]=True
        ep.metadata['termination_reason']='env_terminated'
        self.assertTrue(self.derive(ep)['reward_target_mask'].all())

    def test_unknown_truncation_and_false_success_metadata_are_rejected(self):
        ep=episode('failure');ep.metadata['termination_reason']='unknown'
        with self.assertRaises(ValueError):self.derive(ep)
        ep=episode('failure');ep.metadata['termination_reason']='success'
        with self.assertRaises(ValueError):self.derive(ep)
        ep=episode('censored');ep.metadata['budget_censored']=False
        with self.assertRaises(ValueError):self.derive(ep)

    def test_cannot_guess_censoring_or_source_or_frame_alignment(self):
        for change in ({'budget_censored':None},{'label_source':'imagined'},{'action_steps':41}):
            ep=episode();ep.metadata.update(change)
            with self.assertRaises(ValueError):self.derive(ep)
        ep=episode();ep.success[15]=True
        with self.assertRaises(ValueError):self.derive(ep)

    def test_payload_uses_absolute_episode_time_not_window_end(self):
        ep=episode(count=72)
        payload=reward_target_payload(ep,13,53,self.spec)
        payload['episode_id']=ep.episode_id
        targets,mask,metadata=validate_reward_target_payload(payload,self.spec,expected_count=40)
        self.assertAlmostEqual(float(targets[-1]),self.spec['gamma_per_action']**19,places=7)
        self.assertTrue(mask.all());self.assertEqual(metadata['terminal_frame_index'],71)

    def test_payload_formula_mask_source_and_identity_tampering_rejected(self):
        payload=reward_target_payload(episode(),1,41,self.spec);payload['episode_id']='ep'
        for key,value in [('reward_targets',np.zeros(40,np.float32)),
                          ('reward_target_mask',np.zeros(40,bool)),('episode_id','different')]:
            bad=copy.deepcopy(payload);bad[key]=value
            with self.assertRaises(ValueError):validate_reward_target_payload(bad,self.spec,expected_count=40)

    def test_cohort_moments_and_masked_coverage_are_fixed_on_global_draw(self):
        records=[SimpleNamespace(episode=episode(outcome,identity=str(i)),start=0,length=40)
                 for i,outcome in enumerate(('success','failure','censored','success'))]
        result=dense_reward_cohort(records,self.spec)
        self.assertEqual((result['global_selected_frames'],result['valid_count'],result['masked_count']),(40,30,10))
        self.assertEqual((result['success_episode_valid_count'],result['failure_episode_valid_count']),(20,10))
        self.assertTrue(result['rm_active']);self.assertEqual(result,validate_dense_reward_cohort(result,self.spec))
        bad=dict(result,valid_count=29)
        with self.assertRaises(ValueError):validate_dense_reward_cohort(bad,self.spec)

    def test_all_failures_still_update_but_all_censored_skip(self):
        for outcome,active in [('failure',True),('success',True),('censored',False)]:
            result=dense_reward_cohort([SimpleNamespace(episode=episode(outcome),start=0,length=40)],self.spec)
            self.assertIs(result['rm_active'],active)

    def test_invalid_discount_and_spec_are_rejected(self):
        for gamma in (0,-1,1.01,float('nan'),True):
            with self.assertRaises(ValueError):dense_reward_specification(gamma)
        ep=episode()
        with self.assertRaises(ValueError):
            derive_episode_reward_targets(ep.success,ep.terminated,ep.truncated,ep.metadata,
                                          dict(self.spec,balance='global_cohort_inverse_frequency'))


if __name__=='__main__':unittest.main()
