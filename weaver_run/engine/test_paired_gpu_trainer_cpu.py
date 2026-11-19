"""Real JIT planner/loss/GAE through warmup and paired steady windows on a CPU toy."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

from advantage_ema import AdvantageEMA
from jit_gae_runtime import run_jit_policy_update
from paired_gpu_trainer import weight_digest


class SmallPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1,7)

    def prepare_inputs_batch(self, rows):
        return torch.tensor([[row['state']/100.] for row in rows])

    def forward(self, rows):
        output = self.layer(rows)
        return output[:,:6].reshape(-1,2,1,3), output[:,6]


class Engine(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module = SmallPolicy()
        self.optimizer = torch.optim.AdamW(self.module.parameters(), lr=0.)
        self.global_steps = self.skipped_steps = self.micro_steps = 0

    def forward(self, rows):
        return self.module(rows)

    def backward(self, loss):
        (loss/9).backward()

    def step(self):
        self.micro_steps += 1
        if self.micro_steps % 9 == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.global_steps += 1


class PairedWeightsTests(unittest.TestCase):
    def test_orig_eight_group_weights_survive_scrambling_and_zero_lr_optimizer(self):
        torch.set_num_threads(1)
        engine = Engine()
        ids = [f's:{i}' for i in range(144)]
        owner = SimpleNamespace(model=engine, base_model=engine.module, rank=0, global_step=2,
            telemetry=SimpleNamespace(event=lambda *a,**kw:None), cfg=SimpleNamespace(lora_dropout=0.),
            fresh_config=SimpleNamespace(jit_gae_mode='fused',advantage_norm='ema_zscore',jit_context_micro=2,
                advantage_ema_beta=.9, advantage_scale_floor=.001,policy_sampling='chunks'),
            train_batch_size=16, super_batch_size=144, vf_coef=.5, clip_mode='gipo', clip_eps=.2,
            ent_coef=.01, kl_coef=.1, policy_train_start_step=0, _verify_optimizer_update=lambda *a:None)
        owner.advantage_ema = AdvantageEMA('cpu')
        owner.advantage_ema.commit(torch.tensor([1.,0.,1.]), expected_updates=0, optimizer_succeeded=True)
        masks = np.ones((144,2,1),bool)
        masks[:72,1] = False
        denoms = np.repeat(masks.reshape(18,8,2,1).sum(axis=(1,2,3)),8)
        # Each of the first 16-sample training groups mixes disparate mask denominators.
        order = np.stack((np.arange(72),np.arange(72,144)),axis=1).ravel()
        observations = []
        for i in order:
            seg = dict(segment_id=ids[i],count=2,observations=[{'state':int(i)}, {'state':int(i)+1}, {}],
                       rewards=[0.,.1],terminated=[False,True],truncated=[False,False],gamma=.99,lambda_=.9)
            observations.append(dict(state=int(i),_jit_sample_id=ids[i],_jit_segment=seg,_jit_index=0,
                                     _jit_version=2,_fresh_action_mask=masks[i]))
        owner.capture_policy_metadata = dict(sample_ids=[ids[i] for i in order], fresh_versions=[2,0], jit_pending=True)
        before = weight_digest(owner)['sha256']
        from jit_gae import masked_policy_loss
        seen = []
        def weighted(*args,**kwargs):
            seen.extend(args[7].tolist())
            return masked_policy_loss(*args,**kwargs)
        actions = np.zeros((144,2,1),np.int64)
        with patch('jit_gae_runtime.masked_policy_loss', side_effect=weighted):
            result = run_jit_policy_update(owner,observations,actions,np.zeros((144,2,1,3),np.float32),
                micro_begin=lambda *a,**kw:None, micro_end=lambda *a,**kw:None, trim=lambda mode:None,
                original_denominators=denoms[order])
        np.testing.assert_array_equal(seen, (.5/denoms[order]).astype(np.float32))
        self.assertEqual(len(result['epoch_losses']),9)
        self.assertEqual(engine.global_steps,1)
        self.assertEqual(before,weight_digest(owner)['sha256'])
        self.assertTrue(all(torch.isfinite(p).all() for p in engine.parameters()))
