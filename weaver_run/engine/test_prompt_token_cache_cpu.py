"""Compare the opt-in single-observation path against the original on CPU."""
import ast
from collections import OrderedDict, namedtuple
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[2]


def observation_helpers():
    # rl.utils imports the robot, TensorFlow, and CUDA training stacks at module scope.
    # Run the actual preprocessing definitions with their lightweight dependencies.
    names = {'PromptTokenCache', 'prepare_one_obs', 'process_one_obs',
             'prepare_input_for_action_prediction', 'prepare_labels_for_action_prediction'}
    tree = ast.parse((ROOT / 'rl/utils.py').read_text(encoding='utf8'))
    tree.body = [n for n in tree.body if getattr(n, 'name', '') in names]
    env = dict(torch=torch, OrderedDict=OrderedDict, namedtuple=namedtuple, Any=object, Dict=dict,
               prepare_images_for_vla=lambda images, cfg: images,
               IGNORE_INDEX=-100, STOP_INDEX=2, ACTION_TOKEN_BEGIN_IDX=31743,
               ACTION_DIM=7, NUM_ACTIONS_CHUNK=8)
    exec(compile(tree, 'rl/utils.py', 'exec'), env)
    return env


class Batch(dict):
    def to(self, *, dtype):
        for key, tensor in self.items():
            if tensor.is_floating_point():
                self[key] = tensor.to(dtype=dtype)
        return self


class Processor:
    def __init__(self):
        self.tokenized = []
        self.images = []

    def image_processor(self, image, *, return_tensors):
        assert return_tensors == 'pt'
        self.images.append(image)
        return Batch(pixel_values=torch.tensor([[[[image]]]], dtype=torch.float32))

    def __call__(self, prompt, image):
        self.tokenized.append(prompt)
        pixels = self.image_processor(image, return_tensors='pt')
        # Some tokenizer configurations include a type ID; these must round-trip too.
        suffix = 29871 if 'cup' in prompt else 43
        ids = torch.tensor([[1, len(prompt), suffix]])
        return Batch(input_ids=ids, attention_mask=torch.ones_like(ids),
                     token_type_ids=torch.zeros_like(ids), **pixels)


class PromptTokenCacheTests(unittest.TestCase):
    def setUp(self):
        funcs = observation_helpers()
        self.prepare = funcs['prepare_one_obs']
        self.processor = Processor()
        self.reference = Processor()
        self.cache = funcs['PromptTokenCache'](self.processor, maxsize=2)
        self.cfg = SimpleNamespace(num_images_in_input=2, use_proprio=True)

    def check_obs(self, label, primary, wrist, dtype=torch.bfloat16):
        obs = dict(full_image=primary, wrist_image=wrist, state=(primary, wrist))
        baseline = self.prepare(self.cfg, self.reference, obs, label, dtype)
        cached = self.prepare(self.cfg, self.processor, obs, label, dtype, prompt_cache=self.cache)
        self.assertEqual(type(baseline), type(cached))
        self.assertEqual(baseline.keys(), cached.keys())
        for key in baseline:
            if key == 'proprio':
                self.assertEqual(cached[key], baseline[key])
            else:
                torch.testing.assert_close(cached[key], baseline[key], rtol=0, atol=0, check_dtype=True)
        return cached

    def test_image_and_prompt_changes_and_no_mutable_aliases(self):
        first = self.check_obs('Lift cup', 0.25, 0.75)
        for key in ('input_ids', 'attention_mask', 'token_type_ids', 'labels', 'pixel_values'):
            first[key].zero_()
        second = self.check_obs('Lift cup', 0.25, 3.5)
        second['labels'].fill_(17)
        self.check_obs('Lift cup', 2.25, 3.5)
        self.check_obs('Move bowl', 2.25, 3.5)
        self.assertEqual(self.processor.images, [0.25, 0.75, 0.25, 3.5, 2.25, 3.5, 2.25, 3.5])
        self.assertEqual(len(self.processor.tokenized), 2)
        self.assertEqual(len(self.reference.tokenized), 8)
        self.assertEqual((self.cache.cache_info().hits, self.cache.cache_info().misses), (2, 2))

    def test_lru_eviction_single_image_and_dtype(self):
        self.cfg.num_images_in_input = 1
        self.cfg.use_proprio = False
        for task in ('first', 'second', 'first', 'third', 'second'):
            self.check_obs(task, 1.125, 2.375)
        self.check_obs('second', 1.125, 2.375, torch.float32)
        self.assertEqual(self.cache.cache_info().currsize, 2)
        self.assertEqual(self.processor.images, [1.125] * 6)
        self.assertEqual(self.cache.cache_info().hits, 1)


if __name__ == '__main__':
    unittest.main()
