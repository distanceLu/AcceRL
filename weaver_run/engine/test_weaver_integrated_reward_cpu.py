"""Actual pinned native RM/ScoreBlock/MLP and native WEAVER.forward on CPU.

Only dimensions are small; class implementations and the flow detach path are
extracted from the audited upstream sources, with no encoder downloads or CUDA.
"""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
from typing import List,Optional
import unittest

try:
    import torch
except ImportError:
    torch=None

from weaver_native_reward import install_native_reward_head,validate_reward_cohort
from weaver_online_engine import WeaverOnlineEngine
from weaver_online_smoke import reward_specification


def actual_reward_class():
    from test_weaver_backend_cpu import UPSTREAM
    def rearrange(value,pattern,**dimensions):
        if pattern=='b t d -> (b t) d':
            return value.flatten(0,1)
        if pattern=="(b t) 1 -> b t 1":
            return value.reshape(dimensions['b'],-1,1)
        raise ValueError(pattern)
    def reduce(value,pattern,reduction):
        if pattern!='b t n d -> b t d' or reduction!='mean':
            raise ValueError(pattern)
        return value.mean(dim=2)
    namespace={'torch':torch,'nn':torch.nn,'List':List,'Optional':Optional,
               'rearrange':rearrange,'reduce':reduce}
    for relative,names in [('weaver/wm/nets.py',{'MLP'}),('weaver/wm/model.py',{'ScoreBlock','RewardModel'})]:
        path=UPSTREAM/relative
        nodes=[node for node in ast.parse(path.read_text(encoding='utf-8')).body
               if isinstance(node,ast.ClassDef) and node.name in names]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace['RewardModel']


def cohort(windows,width,positive,start=0):
    negative=windows*width-positive
    return {'protocol':'weaver_reward_cohort_v1','window_count':windows,'label_source':'libero_env_step',
            'frame_slice':[start,start+width],'positive_count':positive,'negative_count':negative,
            'positive_weight':negative/positive if positive and negative else 0.,'negative_weight':1.,
            'rm_active':bool(positive and negative)}


@unittest.skipIf(torch is None,'Actual CPU Torch required')
class NativeReward(unittest.TestCase):
    def test_actual_full_size_native_rm_count_and_20_state_keys(self):
        cls=actual_reward_class()
        rm=cls(img_keys=['full_image','wrist_image'],n_embed=1536,n_hidden=512,n_im_feature=64,
               n_task_feature=768,n_states=8,n_actions=7)
        self.assertEqual(sum(parameter.numel() for parameter in rm.parameters()),6899201)
        before=list(rm.state_dict())
        upstream=SimpleNamespace(rm=rm,named_parameters=lambda:[('rm.'+n,p) for n,p in rm.named_parameters()])
        controller=install_native_reward_head(upstream,reward_specification())
        self.assertEqual(before,list(rm.state_dict()))
        self.assertEqual(len(before),20)
        obs={'full_image':torch.zeros(1,1,1,64),'wrist_image':torch.zeros(1,1,1,64),
             'states':torch.zeros(1,1,8)}
        score=rm(obs,torch.zeros(1,1,7),torch.zeros(1,768))
        self.assertTrue(torch.isfinite(score).all())
        self.assertTrue(((score>=0)&(score<=1)).all())
        self.assertEqual(len(controller.parameters),20)

    def model(self):
        torch.manual_seed(729)
        cls=actual_reward_class()
        return cls(img_keys=['external','wrist'],n_embed=4,n_hidden=8,n_im_feature=3,
                   n_task_feature=2,n_states=8,n_actions=7)

    def native_backend(self):
        from test_weaver_backend_cpu import TinyUpstream,config,sample
        from world_backends.weaver import WeaverBackend
        cfg=config()
        upstream=TinyUpstream()
        upstream.rm=self.model()
        backend=WeaverBackend(cfg,model=upstream)
        controller=install_native_reward_head(upstream,reward_specification())
        backend.native_reward_controller=controller
        backend.reward_training=True
        backend.native_score=True
        backend.config['reward_provider']='world_head'
        backend.metadata['success_head']={'calibrated':False,'trained':False}
        rows=[sample(cfg,backend.training_window_frames) for _ in range(2)]
        for index,row in enumerate(rows):
            row['sample_id']='actual-native-rm-%d'%index
            row['label_source']='libero_env_step'
            row['success'][-1]=1
            row['terminated'][-1]=True
        return backend,rows

    def test_actual_native_class_link_half_brier_exact_and_state_names_unchanged(self):
        rm=self.model()
        upstream=SimpleNamespace(rm=rm,named_parameters=lambda:[('rm.'+n,p) for n,p in rm.named_parameters()])
        before={name:value.clone() for name,value in rm.state_dict().items()}
        obs={'external':torch.zeros(2,2,3,3),'wrist':torch.ones(2,2,3,3),
             'states':torch.ones(2,2,8)}
        actions=torch.ones(2,2,7)
        tasks=torch.ones(2,2)
        raw=rm(obs,actions,tasks).detach()
        controller=install_native_reward_head(upstream,reward_specification())
        self.assertEqual(len(before),20)
        self.assertEqual(set(before),set(rm.state_dict()))
        controller.set_cohort(cohort(2,2,1))
        target=torch.tensor([[0.,0.],[0.,1.]])
        prediction,loss=rm.compute_loss(obs,actions,tasks,target)
        torch.testing.assert_close(prediction,raw.sigmoid())
        expected=.5*((prediction-target).square()*torch.where(target.bool(),3.,1.)).mean()
        torch.testing.assert_close(loss,expected)
        loss.backward()
        self.assertTrue(all(parameter.grad is not None for parameter in rm.parameters()))
        self.assertFalse(controller.report()['calibrated'])
        self.assertIs(controller,install_native_reward_head(upstream,reward_specification()))

    def test_actual_native_forward_rm_loss_detaches_world_and_critic(self):
        backend,rows=self.native_backend()
        second,other_rows=self.native_backend()
        second.model.load_state_dict(backend.model.state_dict())
        start=backend.training_window_frames-backend.n_history-backend.horizon
        metadata=cohort(2,backend.n_history+backend.horizon,2,start)
        backend.native_reward_controller.set_cohort(metadata)
        second.native_reward_controller.set_cohort(metadata)
        second.upstream_model._rm_loss_coeff=0.
        loss,_=backend.model(backend.prepare_train_batch(rows))
        flow_only,_=second.model(second.prepare_train_batch(other_rows))
        loss.backward();flow_only.backward()
        torch.testing.assert_close(backend.upstream_model.wm.gain.grad,second.upstream_model.wm.gain.grad)
        self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum()>0
                            for parameter in backend.upstream_model.rm.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in backend.upstream_model.critic.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in backend.upstream_model.critic.parameters()))

    def test_missing_class_skips_rm_adam_momentum_and_decay_but_world_continues(self):
        backend,rows=self.native_backend()
        engine=WeaverOnlineEngine(backend,world_size=1,micro_batch=2,accumulation_steps=1)
        width=backend.n_history+backend.horizon
        start=backend.training_window_frames-width
        for policy in range(1,6):engine.advance_policy(policy)
        engine.begin_update(5,cohort(2,width,2,start));engine.micro(rows)
        frozen={name:parameter.detach().clone() for name,parameter in backend.upstream_model.rm.named_parameters()}
        states={parameter:{key:value.clone() for key,value in engine.optimizer.state[parameter].items()}
                for parameter in backend.upstream_model.rm.parameters()}
        for row in rows:row['success'].zero_();row['terminated'][-1]=False
        for policy in range(6,11):engine.advance_policy(policy)
        engine.begin_update(10,cohort(2,width,0,start));engine.micro(rows)
        self.assertEqual((engine.global_steps,engine.rm_updates,engine.rm_disabled_updates),(2,1,1))
        for name,parameter in backend.upstream_model.rm.named_parameters():
            self.assertTrue(torch.equal(parameter,frozen[name]))
            for key,value in engine.optimizer.state[parameter].items():
                self.assertTrue(torch.equal(value,states[parameter][key]))
        self.assertTrue(all(parameter.requires_grad for parameter in backend.upstream_model.rm.parameters()))
        self.assertTrue(engine.state()['gradients_released'])

    def test_real_consumed_class_counts_must_match_cohort_before_optimizer(self):
        backend,rows=self.native_backend()
        engine=WeaverOnlineEngine(backend,world_size=1,micro_batch=2,accumulation_steps=1)
        width=backend.n_history+backend.horizon;start=backend.training_window_frames-width
        for policy in range(1,6):engine.advance_policy(policy)
        engine.begin_update(5,cohort(2,width,1,start))
        before={name:value.clone() for name,value in backend.model.state_dict().items()}
        with self.assertRaisesRegex(RuntimeError,'Actually consumed'):
            engine.micro(rows)
        self.assertEqual(engine.global_steps,0)
        self.assertTrue(all(torch.equal(value,before[name]) for name,value in backend.model.state_dict().items()))

    def test_pseudo_label_source_and_inconsistent_global_weights_are_rejected(self):
        bad=cohort(1024,10,1,30);bad['label_source']='demonstration_last_frame'
        with self.assertRaises(ValueError):validate_reward_cohort(bad)
        bad=cohort(1024,10,1,30);bad['positive_weight']=1.
        with self.assertRaises(ValueError):validate_reward_cohort(bad)
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
