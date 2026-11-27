"""Compare packed JIT with a full-episode GAE reference on identical sample IDs."""
from copy import deepcopy
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import os

import numpy as np
import torch

from advantage_ema import AdvantageEMA
from fresh_config import FreshConfig
from fresh_policy_replay import SegmentReplay
from jit_gae import masked_policy_loss
from jit_gae_runtime import run_jit_policy_update
from test_jit_gae_cpu import ToyEngine
from test_model_free_critic_pruning_cpu import TinyPolicy, real_segment


class PackedObjectiveTests(unittest.TestCase):
    def test_episode_samplers_preserve_full_gae_and_actually_accumulated_gradients(self):
        torch.set_num_threads(1)
        for mode in ('whole_episodes', 'uniform_blocks', 'first_use_blocks'):
            for fast in (False, True):
                with self.subTest(mode=mode, fast=fast), torch.random.fork_rng():
                    torch.manual_seed(1729)
                    self.check_update(mode, fast)

    def test_paired_micro24_matches_full_episode_objective_before_step(self):
        with torch.random.fork_rng():
            torch.manual_seed(1729)
            self.check_update('uniform_blocks', True, paired_micro=24)

    def test_skipped_shape_update_does_not_commit_ema_or_emit_consumption(self):
        with torch.random.fork_rng():
            torch.manual_seed(1729)
            self.check_update('uniform_blocks', True, paired_micro=24, skip=True)

    def check_update(self, mode, fast, paired_micro=None, skip=False):
        micro = paired_micro or 16
        gas = 144 // micro
        store = SegmentReplay(2400, FreshConfig.from_world_config({'backend': 'model_free'}),
                              rng=random.Random(123))
        store.set_versions(5, 0)
        lengths = (8, 9, 10, 16, 17, 19)
        for n in range(90):
            segment = real_segment(n, lengths[n % len(lengths)])
            store.add(segment)
            if n % 7 == 0:
                store.samples[segment.sample_id(3)]['uses'] = 4
        token, ids = store.reserve(576, real_sampling=mode)
        self.assertEqual(store.reservation_sampling[token]['actual'], mode)

        model = TinyPolicy()
        with torch.no_grad():
            reference_values = {s.segment_id: model(model.prepare_inputs_batch(list(s.observations)))[1].numpy()
                                for s in store.segments.values()}
        for rank in range(4):
            selected = ids[rank*144:(rank+1)*144]
            reference_batch, _ = store.materialize(selected, refreshed_values=reference_values)
            masks = torch.tensor(np.stack([obs['_fresh_action_mask'] for obs in reference_batch[0]]))
            denominators = masks.reshape(18, 8, -1).sum((1, 2)).repeat_interleave(8)
            packed, _ = store.materialize(selected, jit_denominators=dict(zip(selected, denominators.tolist())))
            active = deepcopy(model)
            reference = deepcopy(model)
            engine = ToyEngine(active, gas=gas, skip=skip)
            grads = []
            engine.optimizer.register_step_pre_hook(
                lambda *_: grads.extend(p.grad.detach().clone() for p in active.parameters()))
            ema = AdvantageEMA('cpu')
            ema.commit(torch.tensor([2., 0., 2.]), expected_updates=0, optimizer_succeeded=True)
            config = SimpleNamespace(jit_gae_mode='fused', advantage_norm='ema_zscore', jit_context_micro=2,
                                     advantage_ema_beta=.9, advantage_scale_floor=.001, policy_sampling='chunks')
            owner = SimpleNamespace(model=engine, base_model=active, fresh_config=config, global_step=5,
                super_batch_size=144, train_batch_size=micro, rank=rank, cfg=SimpleNamespace(lora_dropout=0.),
                capture_policy_metadata=dict(sample_ids=selected, fresh_versions=[5, 0], jit_pending=True),
                vf_coef=.5, ent_coef=.01, kl_coef=.1, clip_mode='gipo', clip_eps=.2, policy_train_start_step=0,
                advantage_ema=ema, telemetry=SimpleNamespace(event=lambda *args, **kwargs: None),
                _verify_optimizer_update=lambda *args: None)
            events = []
            owner.telemetry = SimpleNamespace(event=lambda name, details: events.append(details))
            def verify(engine, _, before, skips):
                if engine.global_steps - before != 1 or engine.skipped_steps != skips:
                    raise RuntimeError('Skipped or incomplete update')
            owner._verify_optimizer_update = verify
            captured = []
            def capture():
                self.assertEqual((engine.micro_steps, engine.global_steps), (gas - 1, 0))
                captured.extend(p.grad.detach().clone() for p in active.parameters())

            kwargs = dict(paired_micro=micro, before_optimizer_step=capture) if paired_micro else {}
            with patch.dict(os.environ, ACCERL_FAST_JIT_BOUNDARIES=str(int(fast))):
                def update(**extra):
                    return run_jit_policy_update(owner, packed[0], packed[1], packed[3],
                        micro_begin=lambda *a, **k: None, micro_end=lambda *a, **k: None, trim=lambda _: None,
                        original_denominators=denominators.numpy(), **extra)
                if paired_micro:
                    with self.assertRaisesRegex(RuntimeError, 'preparation'):
                        update()  # Formal training still rejects an implicit 24x6 recipe.
                if skip:
                    with self.assertRaisesRegex(RuntimeError, 'Skipped'):
                        update(**kwargs)
                    self.assertIsNone(owner._jit_receipts)
                    self.assertEqual((ema.updates, owner.global_step, engine.skipped_steps), (1, 5, 1))
                    return
                result = update(**kwargs)
            stats = events[-1]
            self.assertEqual(stats['batched_window_context'], fast)
            if fast:
                self.assertEqual(stats['no_grad_forward_calls'], (stats['context_only_states'] + micro - 1) // micro)
            rows, actions, advantages, old_logits, targets = reference_batch
            receipt = {row['sample_id']: row for row in owner._jit_receipts}
            self.assertEqual(set(receipt), set(selected))
            np.testing.assert_allclose([receipt[s]['advantage'] for s in selected], advantages, atol=3e-6)
            np.testing.assert_allclose([receipt[s]['value_target'] for s in selected], targets, atol=3e-6)
            clean = [{k: v for k, v in row.items() if k != '_fresh_action_mask'} for row in rows]
            logits, values = reference(reference.prepare_inputs_batch(clean))
            # 9 equal micro losses; token weights retain the original groups of eight.
            loss, _ = masked_policy_loss(logits, values, torch.tensor(actions), torch.tensor(old_logits),
                torch.tensor(advantages), torch.tensor(targets), masks, .5 / denominators / 9,
                vf_coef=.5, ent_coef=.01, kl_coef=.1, clip_mode='gipo', clip_eps=.2, actor_enabled=True)
            loss.backward()
            self.assertEqual(len(grads), len(list(reference.parameters())))
            for actual, parameter in zip(grads, reference.parameters()):
                torch.testing.assert_close(actual, parameter.grad, rtol=3e-5, atol=5e-6)
            if paired_micro:
                self.assertEqual(len(captured), len(grads))
                for observed, actual in zip(captured, grads):
                    torch.testing.assert_close(observed, actual, rtol=0, atol=0)
                self.assertEqual(tuple(float(x) for x in result['paired_normalization']), (0., 1.))
                independent = AdvantageEMA('cpu')
                independent.commit(torch.tensor([2., 0., 2.]), expected_updates=0, optimizer_succeeded=True)
                adv = torch.tensor(advantages, dtype=torch.float64)
                independent.commit(torch.stack((adv.new_tensor(144.), adv.sum(), adv.square().sum())),
                                   expected_updates=1, optimizer_succeeded=True)
                torch.testing.assert_close(ema.mean, independent.mean, atol=3e-7, rtol=3e-6)
                torch.testing.assert_close(ema.variance, independent.variance, atol=3e-7, rtol=3e-6)
            self.assertEqual((engine.global_steps, engine.micro_steps, ema.updates), (1, gas, 2))
            store.commit(token, selected)
        self.assertNotIn(token, store.reservations)


if __name__ == '__main__':
    unittest.main()
