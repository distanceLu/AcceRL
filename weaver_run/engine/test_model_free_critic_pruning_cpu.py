"""Paired full/pruned forwards on an admitted real-only Replay cohort and CPU weights."""
import asyncio
import os
import random
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.distributions import kl

from batch_autotune_runtime import DeviceMemoryMeter, configure_inference, inference_report
from fresh_policy_replay import PolicySegment
from test_batch_autotune_runtime_cpu import Cuda
from test_fresh_policy_replay_cpu import Ledger, Writer, runtime_replay_class
from test_fresh_policy_runtime_cpu import actual_method
from training_topology import successful_updates


LENGTHS = (11, 22, 15, 31, 18, 27)


def real_segment(episode, length):
    # The last token of the last chunk is padding. Alternate learning terminals
    # and time-limit truncations, giving the latter a nonzero current bootstrap.
    tokens = np.arange(length * 2).reshape(length, 2, 1) % 3
    mask = np.ones_like(tokens, dtype=bool)
    mask[-1, 1] = False
    terminal = episode % 2 == 0
    return PolicySegment.create(
        segment_id=str(episode), episode_id=str(episode), source='real',
        observations=[dict(episode=episode, step=i) for i in range(length + 1)],
        action_tokens=tokens, action_mask=mask,
        behaviour_logits=np.cos(tokens[..., None] + np.arange(3)) * .2,
        rewards=np.linspace(-.2, .4, length),
        generation_values=np.full(length + 1, -17.),
        terminated=[False] * (length - 1) + [terminal],
        truncated=[False] * (length - 1) + [not terminal],
        versions=[dict(policy=4)] * length, gamma=.9, lambda_=.8)


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Linear(2, 6)
        self.critic = torch.nn.Linear(2, 1)
        self.calls = []

    def prepare_inputs_batch(self, rows):
        rows = rows.copy()  # OpenVLA's batch preparer edits a list copy.
        self.calls.append([(r['episode'], r['step']) for r in rows])
        return {'features': torch.tensor([[r['episode'] / 5., r['step'] / 30.] for r in rows])}

    def forward(self, inputs):
        features = inputs['features']
        return self.actor(features).reshape(-1, 2, 1, 3), self.critic(features).flatten()


class AccumulatingPolicy(TinyPolicy):
    """Retain gradients at the boundary of two equally sized training micros."""
    global_steps = skipped_steps = micro_steps = 0

    def backward(self, loss):
        (loss / 2).backward()

    def step(self):
        self.micro_steps += 1
        if self.micro_steps == 2:
            self.global_steps += 1


def loss_and_gradients(weights, batch):
    model = AccumulatingPolicy()
    model.load_state_dict(weights)
    owner = SimpleNamespace(model=model, base_model=model, super_batch_size=16, train_batch_size=8,
        vf_coef=.5, kl_coef=.02, ent_coef=.01, global_step=5, policy_train_start_step=0,
        clip_mode='gipo', _verify_optimizer_update=lambda *args: None)
    # GPU capture hooks and the distributed optimizer are external to this CPU check.
    namespace = dict(torch=torch, np=np, kl=kl, time=time, NUM_ACTIONS_CHUNK=2, ACTION_DIM=1,
        micro_begin=lambda *args: None, micro_end=lambda *args: None,
        _trim_cuda_cache=lambda *args: None, successful_updates=successful_updates)
    prepare = actual_method('TrainerActor', '_prepare_policy_batch', namespace)
    train = actual_method('TrainerActor', '_train_policy', namespace)
    prepared = prepare(owner, *batch[:-1])
    adv = prepared[2]
    mean = adv.mean()
    std = (adv.square().mean() - mean.square()).clamp_min(1e-12).sqrt()
    metrics = train(owner, *prepared, mean, std, {})
    return metrics['epoch_losses'], {name: p.grad for name, p in model.named_parameters()}


class RealOnlyPruningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        cls.rng_state = torch.get_rng_state()
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(41)
            cls.weights = TinyPolicy().state_dict()
        cls.replay_actor = runtime_replay_class()

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)
        torch.set_rng_state(cls.rng_state)

    async def sample(self, *, prune, micro, autotuned, legacy_config=False):
        with patch.dict(os.environ, ACCERL_CRITIC_DEPENDENCY_PRUNING=str(int(prune))):
            replay = self.replay_actor(200, dict(real_only=True, real_policy_replay=True,
                real_fraction=1., refresh_gae=True, replay_age_limits=True, max_sample_uses=1))
        replay.capture_ledger, replay.telemetry = Ledger(), Writer()
        replicas, endpoints = [], []
        for rank in range(4):
            model = TinyPolicy()
            model.load_state_dict(self.weights)
            owner = SimpleNamespace(model=model, telemetry=Writer(), actor_id=rank,
                capture_model_version=5, _check_inference_health=lambda: None, _policy_dispatch=None)
            if autotuned:
                configure_inference(owner, 6, micro)
                owner._autobatch_inference['meter'] = DeviceMemoryMeter(Cuda())
                method = actual_method('InferenceActor', 'autotuned_evaluate_values', {})
            else:
                method = actual_method('InferenceActor', 'evaluate_values', dict(torch=torch, np=np))

            async def remote(*args, owner=owner, method=method, **kwargs):
                result = method(owner, *args, **kwargs)
                return await result if autotuned else result

            endpoint = SimpleNamespace(remote=remote)
            endpoints.append(SimpleNamespace(**{method.__name__: endpoint}))
            replicas.append(owner)
        replay.configure_fresh(endpoints)
        replay.set_fresh_versions(5, 0)
        for episode, length in enumerate(LENGTHS):
            replay.add_segment(real_segment(episode, length))
        replay.fresh_store.rng = random.Random(413)
        admission = replay.admit_model_free_batch(16)
        self.assertTrue(admission['ready'])
        if legacy_config:
            replay.configure_batch_autotune(6)
        elif autotuned:
            replay.configure_batch_autotune(6, micro)
        batch = await replay.sample(16)
        metadata = batch[-1]
        self.assertEqual(metadata['fresh_reservation_token'], admission['reservation_token'])
        self.assertEqual(metadata['source_counts'], dict(real=16, imagined=0))
        replay.commit_fresh_use(admission['reservation_token'], metadata['sample_ids'])
        self.assertFalse(replay.fresh_store.reservations)
        calls = [o.model.calls for o in replicas]
        receipt = next(event['details'] for kind, event in replay.telemetry.events
                       if kind == 'fresh_critic_refresh_plan')
        counts = [sum(map(len, blocks)) for blocks in calls]
        self.assertEqual(receipt['observations_by_replica'], counts)
        if autotuned:
            self.assertEqual([inference_report(o, 6)['refresh_observations'] for o in replicas], counts)
        return batch, calls

    def compare(self, micro, autotuned):
        full, original = asyncio.run(self.sample(prune=False, micro=micro, autotuned=autotuned))
        pruned, executed = asyncio.run(self.sample(prune=True, micro=micro, autotuned=autotuned))
        ids = full[-1]['sample_ids']
        self.assertEqual(ids, pruned[-1]['sample_ids'])
        selected = [(row['episode'], row['step']) for row in full[0]]
        self.assertEqual([(r['episode'], r['step']) for r in pruned[0]], selected)
        for index in (1, 2, 3, 4):
            np.testing.assert_array_equal(full[index], pruned[index])
        for left, right in zip(full[0], pruned[0]):
            np.testing.assert_array_equal(left['_fresh_action_mask'], right['_fresh_action_mask'])
        terminal_row, truncated_row = selected.index((0, 10)), selected.index((1, 21))
        self.assertEqual(full[0][terminal_row]['_fresh_action_mask'].tolist(), [[True], [False]])
        self.assertAlmostEqual(full[4][terminal_row], .4, places=6)
        with torch.no_grad():
            model = TinyPolicy()
            model.load_state_dict(self.weights)
            _, bootstrap = model(model.prepare_inputs_batch([dict(episode=1, step=22)]))
        self.assertAlmostEqual(full[4][truncated_row], .4 + .9 * bootstrap.item(), places=6)
        full_loss, full_grad = loss_and_gradients(self.weights, full)
        sparse_loss, sparse_grad = loss_and_gradients(self.weights, pruned)
        self.assertEqual(sparse_loss, full_loss)
        for name, grad in full_grad.items():
            self.assertGreater(grad.abs().max().item(), 0, name)
            torch.testing.assert_close(sparse_grad[name], grad, rtol=0, atol=0)
        self.assertEqual(sum(len(block) for replica in original for block in replica),
                         sum(n + 1 for n in LENGTHS))
        required = {(episode, j) for episode, i in selected
                    for j in range(i, LENGTHS[episode] + int(episode % 2 == 1))}
        wanted = [[block for block in replica if required.intersection(block)] for replica in original]
        self.assertLess(sum(map(len, wanted)), sum(map(len, original)))
        self.assertEqual(executed, wanted)

    def test_direct_refresh_keeps_sample_targets_gradients_and_original_batches(self):
        self.compare(micro=2, autotuned=False)

    def test_autotuned_refresh_keeps_sample_targets_gradients_and_original_batches(self):
        for micro in (2, 4, 8):
            with self.subTest(micro=micro):
                self.compare(micro=micro, autotuned=True)

    def test_legacy_autotune_without_replay_micro_keeps_full_batches(self):
        full, original = asyncio.run(self.sample(prune=False, micro=4, autotuned=True))
        legacy, executed = asyncio.run(self.sample(prune=True, micro=4, autotuned=True,
                                                  legacy_config=True))
        self.assertEqual(executed, original)
        np.testing.assert_array_equal(legacy[2], full[2])
        np.testing.assert_array_equal(legacy[4], full[4])


if __name__ == '__main__':
    unittest.main()
