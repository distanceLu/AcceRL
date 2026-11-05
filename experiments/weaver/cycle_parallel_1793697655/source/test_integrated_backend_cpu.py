"""Actual adapter construction/native forward with an explicitly injected tiny model."""
import copy
import unittest
try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'CPU Torch unavailable')
class JointBackend(unittest.TestCase):
    def backend(self):
        from test_weaver_backend_cpu import config,TinyUpstream
        from test_integrated_semantics_cpu import config as admitted
        from world_backends.weaver import WeaverBackend
        cfg=config();base=admitted()
        cfg.update(reward_provider='world_head',reward_semantics='delta_native_score',terminal_rule='horizon_only',world_trainable=True)
        cfg['checkpoint_metadata']['success_head']={'calibrated':False,'trained':False}
        cfg['checkpoint_metadata']['terminal_head']={'calibrated':False,'trained':False}
        cfg['checkpoint_metadata']['load_report'].update(tensor_count=866,checkpoint_sha256='a'*64,shape_mismatches=[])
        cfg['backend_options'].update(initial_checkpoint_sha256='a'*64,
            train_reward_head=True,native_reward_head=base['backend_options']['native_reward_head'])
        backend=WeaverBackend(cfg,model=TinyUpstream())
        return cfg,backend
    def test_actual_constructor_is_idempotent_and_native_forward_trains_wm_rm_not_critic(self):
        from test_weaver_backend_cpu import sample
        from weaver_native_reward import install_native_reward_head
        cfg,backend=self.backend();upstream=backend.upstream_model
        self.assertIs(backend.native_reward_controller,install_native_reward_head(upstream,cfg['backend_options']['native_reward_head']))
        self.assertEqual(upstream._rm_loss_coeff,1.);self.assertEqual(upstream._critic_loss_coeff,0.)
        row=sample(cfg,backend.training_window_frames)
        row['label_source']='libero_env_step';row['success'][-1]=1
        row['terminated'][-1]=False;row['truncated'][-1]=True
        backend.native_reward_controller.set_cohort({'protocol':'weaver_reward_cohort_v1',
            'window_count':1,'label_frames_per_window':5,'positive_count':1,'negative_count':4,
            'positive_weight':4.,'negative_weight':1.,'rm_active':True,
            'label_source':'libero_env_step','frame_slice':[4,9]})
        prepared=backend.prepare_train_batch([row])
        self.assertTrue(prepared['update_rm']);self.assertEqual(float(prepared['gt_rewards'].sum()),1.)
        loss,logs=backend.compute_losses(prepared);loss.backward()
        self.assertIsNotNone(upstream.wm.gain.grad);self.assertIsNotNone(upstream.rm.bias.grad)
        self.assertIsNone(upstream.critic.bias.grad)
        self.assertTrue(all(not p.requires_grad for p in upstream.critic.parameters()))
        self.assertTrue(torch.isfinite(loss));self.assertIn('loss/RM',logs)
    def test_actual_inference_returns_score_fields_and_never_a_success_terminal(self):
        from test_weaver_backend_cpu import sample
        cfg,backend=self.backend();row=sample(cfg,backend.required_context_frames)
        initial=backend.init_rollout(row,0)
        self.assertNotIn('initial_success_probability',initial)
        self.assertGreater(initial['initial_reward_score'],.5)
        self.assertFalse(initial['initial_terminal'])
        result=backend.predict_chunk(initial,torch.zeros(3,7),published_version=0)
        self.assertNotIn('success_probabilities',result)
        self.assertFalse(result['terminal_predictions'].any())
        self.assertEqual(tuple(result['reward_predictions'].shape),(1,3))
        self.assertFalse(result['reward_calibrated'])
        self.assertTrue(((result['reward_predictions']>=0)&(result['reward_predictions']<=1)).all())


if __name__=='__main__':unittest.main()
