"""Output/gradient equivalence and hook preservation for the actual HF decoder."""
import copy
import unittest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from peft import LoraConfig, get_peft_model
from reward_features import install_reward_features_only


class VLA(torch.nn.Module):
    def __init__(self, peft=False):
        super().__init__()
        cfg = LlamaConfig(vocab_size=48, hidden_size=32, intermediate_size=64,
                          num_hidden_layers=3, num_attention_heads=4,
                          num_key_value_heads=4, attention_dropout=0.0, use_cache=True)
        self.language_model = LlamaForCausalLM(cfg)
        if peft:
            self.language_model = get_peft_model(self.language_model, LoraConfig(
                r=4, lora_alpha=4, lora_dropout=0.0, target_modules=['q_proj', 'v_proj']))
        self.visual = torch.nn.Linear(5, 32)

    def forward(self, ids, pixels, mask):
        text = self.language_model.get_input_embeddings()(ids)
        images = self.visual(pixels)
        embeds = torch.cat([text[:, :1], images, text[:, 1:]], dim=1)
        multimodal_mask = torch.cat([mask[:, :1], torch.ones_like(mask[:, :2]), mask[:, 1:]], dim=1)
        return self.language_model(inputs_embeds=embeds, attention_mask=multimodal_mask,
                                   position_ids=None, labels=None, use_cache=True,
                                   output_hidden_states=True, return_dict=True)


class RewardModel(torch.nn.Module):
    def __init__(self, peft=False):
        super().__init__()
        self.vla = VLA(peft)
        self.reward_head = torch.nn.Linear(32, 2)

    def forward(self, *inputs):
        output = self.vla(*inputs)
        return self.reward_head(output.hidden_states[-1][:, 1])


class RewardFeatureTests(unittest.TestCase):
    def check_equivalent(self, peft, training):
        torch.manual_seed(53)
        old = RewardModel(peft).train(training)
        new = copy.deepcopy(old)
        keys, ids = set(new.state_dict()), [id(p) for p in new.parameters()]
        install_reward_features_only(new)
        self.assertEqual(keys, set(new.state_dict()))
        self.assertEqual(ids, [id(p) for p in new.parameters()])
        args = (torch.tensor([[1, 3, 4, 5], [1, 4, 5, 0]]),
                torch.randn(2, 2, 5), torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]))
        counts = {}
        handles = []
        for label, model in [('old', old), ('new', new)]:
            lm = model.vla.language_model
            core = lm.get_base_model() if hasattr(lm, 'get_base_model') else lm
            for name, child in [('language', lm), ('decoder', core.model), ('lm_head', core.lm_head)]:
                key = label + '_' + name
                counts[key] = 0
                def hook(module, inputs, output, key=key):
                    counts[key] += 1
                handles.append(child.register_forward_hook(hook))
        a, b = old(*args), new(*args)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        labels = torch.tensor([0, 1])
        loss_a = torch.nn.functional.cross_entropy(a, labels)
        loss_b = torch.nn.functional.cross_entropy(b, labels)
        torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=0)
        loss_a.backward()
        loss_b.backward()
        for (name, p), (other_name, q) in zip(old.named_parameters(), new.named_parameters()):
            self.assertEqual(name, other_name)
            self.assertEqual(p.grad is None, q.grad is None, name)
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0, msg=name)
        self.assertEqual(counts, {'old_language': 1, 'old_decoder': 1, 'old_lm_head': 1,
                                  'new_language': 1, 'new_decoder': 1, 'new_lm_head': 0})
        for h in handles:
            h.remove()
        # Other model instances and the library class still run vocabulary projection.
        self.assertNotIn('forward', old.vla.language_model.__dict__)

    def test_plain_training(self):
        self.check_equivalent(False, True)

    def test_plain_eval(self):
        self.check_equivalent(False, False)

    def test_peft_training(self):
        self.check_equivalent(True, True)

    def test_unsupported_lm_loss_rejected(self):
        model = RewardModel()
        install_reward_features_only(model)
        with self.assertRaises(ValueError):
            model.vla.language_model(input_ids=torch.ones(1, 3, dtype=torch.long),
                                     labels=torch.ones(1, 3, dtype=torch.long))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
