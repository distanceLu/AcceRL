"""Dense regression uses the actual native forward and preserves old RM mode."""
import copy
from types import SimpleNamespace
import unittest

import numpy as np
try:
    import torch
except ImportError:
    torch=None

from fresh_reward_targets import dense_reward_specification,dense_reward_cohort,reward_target_payload
from test_fresh_reward_targets_cpu import episode
from weaver_native_reward import install_native_reward_head,validate_specification
from weaver_online_engine import WeaverOnlineEngine
from training_topology import combine_wm_metrics


@unittest.skipIf(torch is None,'Actual CPU Torch required')
class DenseNativeRegression(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def backend(self):
        from test_weaver_backend_cpu import TinyUpstream,config,sample
        from world_backends.weaver import WeaverBackend
        cfg=config();spec=dense_reward_specification()
        # Explicit CPU injection follows the existing native RM tests. It does
        # not fabricate strict866 production admission for the tiny fixture.
        backend=WeaverBackend(cfg,model=TinyUpstream())
        controller=install_native_reward_head(backend.upstream_model,spec)
        backend.upstream_model.native_reward_controller=controller
        backend.native_reward_controller=controller
        backend.reward_training=backend.native_score=True
        backend.config.update(reward_provider='world_head',reward_semantics='delta_native_score',native_reward_head=spec)
        def row(outcome,index):
            raw=sample(cfg,backend.training_window_frames)
            ep=episode(outcome,count=backend.training_window_frames,identity='dense-%d'%index)
            raw.update(episode_id=ep.episode_id,label_source='libero_env_step',
                       success=ep.success,terminated=ep.terminated,truncated=ep.truncated)
            raw.update(reward_target_payload(ep,0,ep.metadata['action_steps']+1,spec))
            return raw,SimpleNamespace(episode=ep,start=0,length=backend.training_window_frames)
        return backend,row

    def cohort(self,backend,records):
        width=backend.n_history+backend.horizon
        return dense_reward_cohort(records,backend.native_reward_controller.specification,
            frame_slice=(backend.training_window_frames-width,backend.training_window_frames))

    def test_actual_native_augmentation_has_exact_masked_half_mse(self):
        backend,row=self.backend();pairs=[row('success',0),row('censored',1)]
        backend.native_reward_controller.set_cohort(self.cohort(backend,[x[1] for x in pairs]))
        prepared=backend.prepare_train_batch([x[0] for x in pairs]);loss,components=backend.model(prepared)
        stat=backend.native_reward_controller.last_statistics
        target=prepared['gt_rewards'][0]
        prediction=torch.sigmoid(backend.upstream_model.rm.bias)
        expected=.5*(prediction-target).square().mean()
        torch.testing.assert_close(components['loss/RM'],expected)
        self.assertEqual(stat['valid_targets_with_native_augmentation'],2*len(target))
        self.assertEqual(stat['masked_targets_with_native_augmentation'],2*len(target))
        self.assertIsNone(backend.native_reward_controller.micro_targets)
        loss.backward()
        self.assertIsNotNone(backend.upstream_model.rm.bias.grad)
        self.assertTrue(all(not parameter.requires_grad for parameter in backend.upstream_model.critic.parameters()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_uneven_micro_masks_match_one_global_valid_mean_not_micro_means(self):
        backend,row=self.backend();pairs=[row(outcome,i) for i,outcome in enumerate(('success','censored','failure','failure'))]
        cohort=self.cohort(backend,[x[1] for x in pairs]);backend.native_reward_controller.set_cohort(cohort)
        losses=[];expected_errors=[]
        for index in (0,2):
            prepared=backend.prepare_train_batch([x[0] for x in pairs[index:index+2]])
            prediction=torch.sigmoid(backend.upstream_model.rm.bias)
            expected_errors.extend((prediction-prepared['gt_rewards'])[prepared['gt_reward_mask']].square().unbind())
            _,components=backend.model(prepared);losses.append(components['loss/RM'])
        torch.testing.assert_close(sum(losses)/2,.5*torch.stack(expected_errors).mean())

    def test_all_failure_updates_rm_all_censored_preserves_adam_and_decay(self):
        backend,row=self.backend();pairs=[row('failure',i) for i in range(2)]
        engine=WeaverOnlineEngine(backend,world_size=1,micro_batch=2,accumulation_steps=1)
        for step in range(1,6):engine.advance_policy(step)
        engine.begin_update(5,self.cohort(backend,[x[1] for x in pairs]));engine.micro([x[0] for x in pairs])
        self.assertEqual((engine.rm_updates,engine.rm_disabled_updates),(1,0))
        before={name:value.detach().clone() for name,value in backend.upstream_model.rm.named_parameters()}
        states={parameter:{name:value.clone() for name,value in engine.optimizer.state[parameter].items()}
                for parameter in backend.upstream_model.rm.parameters()}
        pairs=[row('censored',i+2) for i in range(2)]
        for step in range(6,11):engine.advance_policy(step)
        engine.begin_update(10,self.cohort(backend,[x[1] for x in pairs]));result=engine.micro([x[0] for x in pairs])
        self.assertEqual((engine.global_steps,engine.rm_updates,engine.rm_disabled_updates),(2,1,1))
        for name,parameter in backend.upstream_model.rm.named_parameters():
            self.assertTrue(torch.equal(parameter,before[name]))
            for key,value in engine.optimizer.state[parameter].items():self.assertTrue(torch.equal(value,states[parameter][key]))
        self.assertIn('native_half_mse_loss',result['result']);self.assertNotIn('native_half_brier_loss',result['result'])

    def test_actual_consumption_moments_mismatch_rejects_before_optimizer(self):
        backend,row=self.backend();pairs=[row('success',i) for i in range(2)]
        engine=WeaverOnlineEngine(backend,world_size=1,micro_batch=2,accumulation_steps=1)
        for step in range(1,6):engine.advance_policy(step)
        cohort=self.cohort(backend,[x[1] for x in pairs]);cohort['target_sum']+=.001
        engine.begin_update(5,cohort)
        with self.assertRaises(RuntimeError):engine.micro([x[0] for x in pairs])
        self.assertEqual(engine.global_steps,0)

    def test_four_rank_two_gas_reduction_matches_global_target_gradient(self):
        backend,row=self.backend()
        outcomes=('success','censored','failure','censored','failure','failure','success','success')*2
        pairs=[row(outcome,i) for i,outcome in enumerate(outcomes)]
        backend.native_reward_controller.set_cohort(self.cohort(backend,[x[1] for x in pairs]))
        losses=[];reported=[];errors=[]
        for index in range(0,len(pairs),2):
            prepared=backend.prepare_train_batch([x[0] for x in pairs[index:index+2]])
            prediction=torch.sigmoid(backend.upstream_model.rm.bias)
            errors.extend((prediction-prepared['gt_rewards'])[prepared['gt_reward_mask']].square().unbind())
            loss,components=backend.model(prepared);losses.append(loss);reported.append(components['loss/RM'])
        averaged=sum(losses)/(4*2)  # DDP rank mean times GAS mean, same model parameters.
        reference=.5*torch.stack(errors).mean()
        torch.testing.assert_close(sum(reported)/(4*2),reference)
        actual_grad=torch.autograd.grad(averaged,backend.upstream_model.rm.bias,retain_graph=True)[0]
        expected_grad=torch.autograd.grad(reference,backend.upstream_model.rm.bias)[0]
        torch.testing.assert_close(actual_grad,expected_grad)

    def test_reported_flow_and_rm_match_actual_objective_with_uneven_rank_masks(self):
        from unittest import mock
        backend,row=self.backend()
        outcomes=('censored',)*4 + ('success','failure','failure','failure') + \
                 ('success','censored','censored','censored') + ('success','success','failure','censored')
        pairs=[row(outcome,i) for i,outcome in enumerate(outcomes)]
        for index,(raw,_) in enumerate(pairs):
            raw['states']=raw['states']+index  # Distinct micro flow losses catch last-only logging.
        cohort=self.cohort(backend,[x[1] for x in pairs])
        errors=[];ranks=[];flow_means=[];conversions=[]

        class CountedScalar:
            def __init__(self,value):self.value=value
            def numel(self):return 1
            def detach(self):return self
            def item(self):
                conversions.append('flow')
                return self.value.item()

        for rank in range(4):
            backend,_=self.backend()
            engine=WeaverOnlineEngine(backend,rank=rank,world_size=4,micro_batch=2,accumulation_steps=2)
            for step in range(1,6):engine.advance_policy(step)
            engine.begin_update(5,cohort)
            def count_conversion(module,inputs,output):
                loss,logs=output
                return loss,dict(logs,**{'flow/Total Loss':CountedScalar(logs['flow/Total Loss'])})
            handle=backend.model.register_forward_hook(count_conversion)
            micro_flows=[]
            try:
                with mock.patch.object(backend.model,'forward',wraps=backend.model.forward) as forward:
                    for offset in (0,2):
                        rows=[x[0] for x in pairs[rank*4+offset:rank*4+offset+2]]
                        prepared=backend.prepare_train_batch(rows)
                        prediction=torch.sigmoid(backend.upstream_model.rm.bias).detach()
                        errors.extend((prediction-prepared['gt_rewards'])[prepared['gt_reward_mask']].square().unbind())
                        result=engine.micro(rows)
                        logs=result['native_losses']
                        self.assertAlmostEqual(result['loss'],logs['flow/Total Loss']+logs['RM loss'],places=5)
                        micro_flows.append(logs['flow/Total Loss'])
                    self.assertEqual(forward.call_count,2)
            finally:
                handle.remove()
            final=result['result']
            self.assertAlmostEqual(final.get('native_flow_loss',-1.),sum(micro_flows)/2.,places=6)
            flow_means.append(sum(micro_flows)/2.)
            ranks.append(dict(ctrl_loss=final['loss'],native_flow_loss=final['native_flow_loss'],
                              native_half_mse_loss=final['native_half_mse_loss']))
        combined=combine_wm_metrics(ranks)
        expected_rm=.5*torch.stack(errors).mean().item()
        self.assertEqual(ranks[0]['native_half_mse_loss'],0.)
        self.assertGreater(expected_rm,0.)
        self.assertAlmostEqual(combined['native_half_mse_loss'],expected_rm,places=6)
        self.assertAlmostEqual(combined['native_flow_loss'],sum(flow_means)/4.,places=6)
        self.assertAlmostEqual(combined['ctrl_loss'],combined['native_flow_loss']+expected_rm,places=5)
        self.assertEqual(conversions,['flow']*8)  # Reuse the original per-micro conversion once.
        self.assertFalse(torch.cuda.is_initialized())

    def test_legacy_spec_remains_exact_and_calibrated_true_is_rejected(self):
        from weaver_online_smoke import reward_specification
        self.assertEqual(validate_specification(reward_specification()),reward_specification())
        with self.assertRaises(ValueError):validate_specification(dict(dense_reward_specification(),calibrated=True))


if __name__=='__main__':unittest.main()
