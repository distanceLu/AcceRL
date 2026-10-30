"""Execute the candidate's actual partial-chunk loss and real collector on CPU."""
import ast
from collections import deque
import math
import os
from pathlib import Path
import time
from types import MethodType, SimpleNamespace
import unittest

import numpy as np
import torch
from torch.distributions import kl

from fresh_config import FreshConfig
from fresh_policy_replay import PolicySegment
from fresh_signal import record_signal


TREE=ast.parse((Path(__file__).parent/'train.py').read_text(encoding='utf8'))


def actual_method(class_name, method, namespace):
    cls=next(node for node in TREE.body if isinstance(node,ast.ClassDef) and node.name==class_name)
    node=next(node for node in cls.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name==method)
    module=ast.Module(body=[node],type_ignores=[])
    exec(compile(module,'actual '+method,'exec'),namespace)
    return namespace[method]


class PolicyLossTests(unittest.TestCase):
    def run_loss(self, unused_logit, masked):
        namespace=dict(torch=torch,np=np,kl=kl,time=time,NUM_ACTIONS_CHUNK=2,ACTION_DIM=1,
            micro_begin=lambda *a:None,micro_end=lambda *a:None,_trim_cuda_cache=lambda *a:None,
            successful_updates=lambda before,skips,engine:engine.global_steps-before)
        run=actual_method('TrainerActor','_train_policy',namespace)
        class Model:
            global_steps=0;skipped_steps=0
            def __init__(self):
                self.logits=torch.nn.Parameter(torch.tensor([[[0.,0.],[unused_logit,-unused_logit]]]))
                self.value=torch.nn.Parameter(torch.zeros(1))
            def forward(self,inputs):
                if '_fresh_action_mask' in inputs:raise AssertionError('Mask leaked into model inputs')
                return self.logits,self.value
            def backward(self,loss):loss.backward()
            def step(self):self.global_steps+=1
        model=Model()
        owner=SimpleNamespace(model=model,super_batch_size=1,train_batch_size=1,vf_coef=1.,
            global_step=1,policy_train_start_step=0,clip_mode='ppo',clip_eps=.2,kl_coef=.1,ent_coef=.01,
            _verify_optimizer_update=lambda *a:None)
        inputs={'pixels':torch.zeros((1,1))}
        if masked:inputs['_fresh_action_mask']=torch.tensor([[[True],[False]]])
        result=run(owner,inputs,torch.zeros((1,2,1),dtype=torch.long),torch.ones(1),
                   torch.zeros((1,2,1,2)),torch.zeros(1),torch.tensor(0.),torch.tensor(1.),{})
        return result['epoch_losses'][0],model.logits.grad

    def test_unexecuted_tail_does_not_affect_surrogate_entropy_or_kl(self):
        left,grad=self.run_loss(0.,True);right,_=self.run_loss(5.,True)
        self.assertAlmostEqual(left,right,places=7)
        self.assertTrue(torch.equal(grad[:,1],torch.zeros_like(grad[:,1])))
        old_left,_=self.run_loss(0.,False);old_right,_=self.run_loss(5.,False)
        self.assertNotAlmostEqual(old_left,old_right,places=4)


class CriticEndpointTests(unittest.TestCase):
    def test_actual_value_endpoint_preserves_sampling_rng_and_checks_version(self):
        endpoint=actual_method('InferenceActor','evaluate_values',dict(torch=torch,np=np))
        class Model:
            def prepare_inputs_batch(self,rows):return torch.tensor([row['value'] for row in rows])
            def __call__(self,inputs):return None,inputs+2
            def post_process(self,*args,**kwargs):raise AssertionError('Critic refresh sampled actions')
        owner=SimpleNamespace(model=Model(),capture_model_version=5,training_batch_limit=2,
            _check_inference_health=lambda:None,telemetry=SimpleNamespace(event=lambda *a,**k:None))
        before=torch.random.get_rng_state().clone()
        result=endpoint(owner,[{'value':float(i)} for i in range(5)],5)
        self.assertEqual(result['values'],[2.,3.,4.,5.,6.])
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        with self.assertRaises(ValueError):endpoint(owner,[{'value':0.}],4)


class RealCollectorTests(unittest.TestCase):
    def collect(self, *, steps=10, success=True, budget_cut=False):
        segments=[];observed=[];critic_calls=[]
        namespace=dict(np=np,math=math,os=os,time=time,FreshConfig=FreshConfig,PolicySegment=PolicySegment,
            NUM_ACTIONS_CHUNK=8,ACTION_DIM=7,new_id=lambda prefix:prefix+':one',
            prepare_one_obs=lambda cfg,processor,obs,*a:{'frame':obs['frame']},
            environment_dt=lambda env:.05,ray=SimpleNamespace(get=lambda value:value),
            lineage_begin=lambda *a,**k:None,lineage_end=lambda *a:'actual-real-group',
            lineage_sample=lambda *a,**k:None,record_signal=record_signal)
        def rpc(owner,*args,**kwargs):
            owner.capture_last_versions={'policy':4}
            return None,np.ones((8,7),np.float32),np.full((8,7),2),np.zeros((8,7,4)),.3
        namespace['profile_rpc']=rpc
        collect=actual_method('RolloutWorkerActor','get_one_episode',namespace)
        process=actual_method('RolloutWorkerActor','_process_traj',namespace)
        class Env:
            def __init__(self):self.n=0
            def step(self,action):
                self.n+=1;done=self.n==steps
                return {'frame':self.n},float(done and success),bool(done and success),bool(done and not success),{'is_success':bool(done and success)}
        env=Env()
        def critic(rows):
            critic_calls.append(rows)
            return {'policy_version':4,'values':[.7]}
        assigned={'env_seed':3,'allocation_id':1,'cycle':0,'max_action_steps':steps if budget_cut else 220}
        owner=SimpleNamespace(wid=0,initial_state_enabled=True,pipeline_job={'initial_state':assigned},
            env=env,cfg=None,processor=None,infer=SimpleNamespace(evaluate_values=SimpleNamespace(remote=critic)),
            task_description='task',torch_dtype=None,world_config=SimpleNamespace(backend='weaver',schema={'dt_seconds':.05}),
            pipeline_max_episode_frames=221,fresh_config=FreshConfig(real_policy_replay=True),
            _reset_and_select_env=lambda **kwargs:({'frame':0},{'task_action_horizon':220}),reward_scale=1.,
            env_outcome=[deque()],current_env_idx=0,current_env_name='task',gamma=.9,lambda_=.8,
            stats_actor=SimpleNamespace(add_episode_return=SimpleNamespace(remote=lambda *a,**k:None)),
            replay=SimpleNamespace(add_segment=SimpleNamespace(remote=lambda item:segments.append(item))),
            _process_episode_for_wm=lambda *a,**k:observed.append((a,k)))
        owner._process_traj=MethodType(process,owner)
        collect(owner)
        self.assertEqual(len(segments),1)
        return segments[0],observed,owner,critic_calls

    def test_actual_real_collection_retains_behavior_and_masks_partial_last_chunk(self):
        item,observed,owner,calls=self.collect()
        self.assertEqual(item.source,'real');self.assertEqual(item.reward_source,'environment_success')
        self.assertEqual(item.action_tokens.shape,(2,8,7));self.assertTrue((item.action_tokens==2).all())
        self.assertEqual(item.action_mask[0,:,0].tolist(),[True]*8)
        self.assertEqual(item.action_mask[1,:,0].tolist(),[True,True]+[False]*6)
        self.assertEqual(item.rewards.tolist(),[0.,1.]);self.assertEqual(item.generation_values[-1],0.)
        self.assertTrue(item.terminated[-1]);self.assertEqual(len(item.observations),3)
        self.assertEqual(item.versions[0]['policy'],4);self.assertEqual(len(observed),1)

    def test_complete_task_horizon_is_learning_terminal_without_fabricated_env_term(self):
        item,observed,owner,calls=self.collect(steps=220,success=False)
        self.assertEqual(calls,[])
        self.assertEqual(item.generation_values[-1],0.)
        self.assertTrue(item.terminated[-1]);self.assertFalse(item.truncated[-1])
        self.assertTrue(item.versions[-1]['env_truncated']);self.assertFalse(item.versions[-1]['env_terminated'])
        self.assertEqual(item.versions[-1]['termination_reason'],'task_horizon')
        self.assertTrue(owner.collection_metadata['termination_reason']=='task_horizon')
        terminated,truncated,success=observed[0][1]['frame_labels']
        self.assertFalse(terminated[-1]);self.assertTrue(truncated[-1]);self.assertFalse(success[-1])
        self.assertTrue((item.rewards==0).all())

    def test_artificial_budget_censoring_bootstraps_and_retains_partial_action_mask(self):
        item,observed,owner,calls=self.collect(steps=10,success=False,budget_cut=True)
        self.assertEqual(len(calls),1)
        self.assertAlmostEqual(float(item.generation_values[-1]),.7,places=6)
        self.assertFalse(item.terminated[-1]);self.assertTrue(item.truncated[-1])
        self.assertTrue(item.versions[-1]['env_truncated']);self.assertFalse(item.versions[-1]['env_terminated'])
        self.assertEqual(item.versions[-1]['termination_reason'],'real_env_budget')
        self.assertEqual(item.action_mask[-1,:,0].tolist(),[True,True]+[False]*6)
        self.assertTrue(owner.collection_metadata['budget_censored'])
        terminated,truncated,success=observed[0][1]['frame_labels']
        self.assertFalse(terminated[-1]);self.assertTrue(truncated[-1]);self.assertFalse(success[-1])



if __name__=='__main__':unittest.main()
