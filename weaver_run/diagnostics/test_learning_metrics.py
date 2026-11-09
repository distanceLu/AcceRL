import unittest
import numpy as np
import torch
from .learning_metrics import gradient_geometry, source_losses, loss_gradients, paired_rm_metrics, paired_wm_metrics
from ..engine.jit_gae import masked_policy_loss


class MetricsTests(unittest.TestCase):
    def test_wm_physical_units_mask_and_alignment_shape(self):
        images=np.zeros((3,2,3,4,4)); states=np.zeros((3,8));pred=states.copy()
        pred[:,0]=.03;pred[:,1]=.04;pred[2]=float('nan')
        d=paired_wm_metrics(images,images,pred,states,np.array([1,1,0],bool))
        self.assertAlmostEqual(d['eef_xyz_l2_mean'],.05)
        self.assertEqual(d['image_mse_full'],0.)
        with self.assertRaises(ValueError):paired_wm_metrics(images,images,pred[:2],states,np.ones(3,bool))

    def test_geometry_known_vectors_and_zero(self):
        r={'p':torch.tensor([1.,0.])};i={'p':torch.tensor([-2.,0.])}
        d=gradient_geometry(r,i)
        self.assertEqual(d['angle_degrees'],180.)
        self.assertEqual(d['real_dot_joint'],-1.)
        self.assertEqual(d['joint_norm'],1.)
        self.assertIsNone(gradient_geometry(r,{'p':torch.zeros(2)})['cosine'])

    def test_sum_gradients_not_mean_micro_cosine(self):
        d=gradient_geometry({'x':torch.tensor([3.,1.])},{'x':torch.tensor([3.,-1.])})
        self.assertAlmostEqual(d['cosine'],.8)
        self.assertAlmostEqual(d['real_projection_fraction']+d['imagined_projection_fraction'],1.)

    def test_rm_mask_and_episode_weighting(self):
        d=paired_rm_metrics([1.,1.,1.,0.,float('nan')],[0.,0.,0.,0.,float('nan')],
                            np.array([1,1,1,1,0],bool),['a','a','a','b','c'])
        self.assertEqual(d['mse_frame_weighted'],.75)
        self.assertEqual(d['mse_episode_weighted'],.5)
        self.assertEqual(d['masked_frames'],1)

    def test_nonfinite_or_missing_gradient_rejected(self):
        with self.assertRaises(ValueError):gradient_geometry({'x':torch.ones(1)},{'y':torch.ones(1)})
        with self.assertRaises(ValueError):gradient_geometry({'x':torch.ones(1)},{'x':torch.tensor([float('nan')])})

    def test_actual_objective_and_gradient_additivity(self):
        torch.manual_seed(43)
        for mode in ('gipo','ppo','sapo'):
            for actor in (True,False):
                for codes in ([0,1,1,1],[1,1,1,1]):
                    with self.subTest(mode=mode,actor=actor,codes=codes):
                        logits=torch.randn(4,2,3,7,requires_grad=True)
                        value=torch.randn(4,requires_grad=True)
                        actions=torch.randint(0,7,(4,2,3)); old=torch.randn_like(logits)
                        adv=torch.randn(4);targets=torch.randn(4)
                        masks=torch.tensor([[[1,1,1],[1,0,0]],[[1,1,1],[1,1,1]],[[1,1,1],[0,0,0]],[[1,1,0],[0,0,0]]],dtype=torch.bool)
                        weights=torch.ones(4)/masks.sum()
                        args=(logits,value,actions,old,adv,targets,masks,weights)
                        cfg=dict(vf_coef=.5,ent_coef=.03,kl_coef=.1,clip_mode=mode,clip_eps=.2,actor_enabled=actor)
                        joint=masked_policy_loss(*args,**cfg)[0]
                        split=source_losses(*args,torch.tensor(codes),**cfg)
                        summed=sum(v for part in split.values() for v in part.values())
                        torch.testing.assert_close(summed,joint)
                        params=[('logits',logits),('value',value)]
                        r=loss_gradients(sum(split['real'].values()),params,retain_graph=True)
                        i=loss_gradients(sum(split['imagined'].values()),params,retain_graph=True)
                        g=loss_gradients(joint,params)
                        for n in g:torch.testing.assert_close(r[n]+i[n],g[n],atol=1e-6,rtol=1e-5)
                        self.assertIsNone(logits.grad);self.assertIsNone(value.grad)

if __name__=='__main__':unittest.main()
