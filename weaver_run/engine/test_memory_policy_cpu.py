import copy
import unittest
import torch
from test_reward_features_cpu import RewardModel
from reward_features import install_reward_features_only
from memory_policy import install_reward_checkpointing


class RecomputeTest(unittest.TestCase):
    def check_model(self, peft):
        torch.manual_seed(107)
        original = RewardModel(peft).train()
        install_reward_features_only(original)
        changed = copy.deepcopy(original)
        keys = set(changed.state_dict())
        ids = [id(p) for p in changed.parameters()]
        install_reward_checkpointing(changed)
        self.assertEqual(keys, set(changed.state_dict()))
        self.assertEqual(ids, [id(p) for p in changed.parameters()])
        # Check recomputation preserves stochastic attention too.
        for module in [original, changed]:
            for child in module.modules():
                if type(child).__name__ in {'LlamaAttention', 'LlamaSdpaAttention', 'LlamaFlashAttention2'}:
                    child.attention_dropout = .1
        args = (torch.tensor([[1, 3, 4, 5], [1, 4, 5, 0]]),
                torch.randn(2, 2, 5), torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]))
        outputs = []
        for module in [original, changed]:
            torch.manual_seed(111)
            value = module(*args)
            value.square().mean().backward()
            outputs.append(value)
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        for (name, a), (other, b) in zip(original.named_parameters(), changed.named_parameters()):
            self.assertEqual(name, other)
            if a.grad is None:
                self.assertIsNone(b.grad)
            else:
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0, msg=name)

    def test_plain(self):
        self.check_model(False)

    def test_peft(self):
        self.check_model(True)

    def test_vision_intermediate_feature_path(self):
        from timm.models.vision_transformer import VisionTransformer
        class Vision(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.featurizer = VisionTransformer(img_size=8, patch_size=4, embed_dim=16,
                    depth=2, num_heads=4, num_classes=0, mlp_ratio=2,
                    attn_drop_rate=.1, proj_drop_rate=.1, drop_path_rate=.1)
                self.fused_featurizer = copy.deepcopy(self.featurizer)
        torch.manual_seed(149)
        original = RewardModel().train()
        original.vla.vision_backbone = Vision()
        install_reward_features_only(original)
        changed = copy.deepcopy(original)
        keys = set(changed.state_dict())
        ids = [id(p) for p in changed.parameters()]
        metadata = install_reward_checkpointing(changed)
        self.assertEqual(metadata['vision_block_count'], 4)
        self.assertEqual(keys, set(changed.state_dict()))
        self.assertEqual(ids, [id(p) for p in changed.parameters()])
        image = torch.randn(2, 3, 8, 8)
        outputs = []
        for module in [original, changed]:
            torch.manual_seed(177)
            vision = module.vla.vision_backbone
            a = vision.featurizer.get_intermediate_layers(image, n={0})[0]
            b = vision.fused_featurizer.get_intermediate_layers(image, n={0})[0]
            pixels = (a + b)[:, :2, :5]
            value = module(torch.tensor([[1, 3, 4, 5], [1, 4, 5, 0]]), pixels,
                           torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]))
            value.square().mean().backward()
            outputs.append(value)
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        for (name, a), (other, b) in zip(original.named_parameters(), changed.named_parameters()):
            self.assertEqual(name, other)
            if a.grad is None:
                self.assertIsNone(b.grad)
            else:
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0, msg=name)


if __name__ == '__main__':
    unittest.main()
