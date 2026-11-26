"""Check that optimized policy forwards preserve the model's configured behavior."""
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from prismatic.extern.hf.modeling_prismatic import PrismaticVisionBackbone
from rl.actor_critic_model_discrete import ActorCritic


class VisionTower(nn.Sequential):
    def forward(self, pixels):
        return super().forward(pixels).flatten(2).transpose(1, 2)


class ActionForwardContractTests(unittest.TestCase):
    def test_training_views_preserve_statistics_rng_and_evaluation_batching(self):
        vb = PrismaticVisionBackbone.__new__(PrismaticVisionBackbone)
        nn.Module.__init__(vb)
        vb.use_fused_vision_backbone = True
        vb.num_images_in_input = 2
        vb.featurizer = VisionTower(nn.BatchNorm2d(3))
        vb.fused_featurizer = VisionTower(nn.Dropout(.5))
        pixels = torch.arange(2*12*4*4, dtype=torch.float32).reshape(2, 12, 4, 4)
        serial = deepcopy(vb)
        def original():
            patches = []
            for view in pixels.split(6, dim=1):
                regular, fused = view.split(3, dim=1)
                patches.append(torch.cat((serial.featurizer(regular), serial.fused_featurizer(fused)), dim=2))
            return torch.cat(patches, dim=1)
        torch.manual_seed(37)
        expected = original()
        expected_rng = torch.get_rng_state()
        torch.manual_seed(37)
        actual = vb(pixels)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        self.assertTrue(torch.equal(vb.featurizer[0].running_mean, serial.featurizer[0].running_mean))
        vb.eval()
        serial.eval()
        with patch.object(vb.featurizer, 'forward', wraps=vb.featurizer.forward) as calls:
            torch.testing.assert_close(vb(pixels), original())
            self.assertEqual(calls.call_count, 1)

    def test_action_projection_respects_attention_backend_and_stochastic_head(self):
        policy = ActorCritic.__new__(ActorCritic)
        nn.Module.__init__(policy)
        policy.cfg = SimpleNamespace(use_proprio=False)
        lm = nn.Module()
        lm.model = nn.Identity()
        lm.lm_head = nn.Sequential(nn.Dropout(.5), nn.Linear(3, 5))
        lm.config = SimpleNamespace(pretraining_tp=1, output_attentions=False)
        policy.vla = SimpleNamespace(language_model=lm, config=SimpleNamespace(output_attentions=False))
        batch = {'input_ids': torch.zeros(1, 8), 'pixel_values': torch.zeros(1, 12, 4, 4)}
        self.assertFalse(policy._can_use_action_only_fast_path(batch))
        lm.eval()
        self.assertTrue(policy._can_use_action_only_fast_path(batch))
        policy.vla.config.output_attentions = True
        self.assertFalse(policy._can_use_action_only_fast_path(batch))


if __name__ == '__main__':
    unittest.main()
