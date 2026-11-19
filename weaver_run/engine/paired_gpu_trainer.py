"""Explicit four-rank short-test endpoints. Called only by the paired diagnostic."""
import hashlib
import math
import struct
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist

from batch_autotune_runtime import begin_training_micro, configure_training, end_training_micro, finish_training
from jit_gae_runtime import _checked, run_jit_policy_update


def weight_digest(owner):
    """Bitwise digest of the tensors supplied by the trainable-weights checkpoint."""
    digest = hashlib.sha256()
    elements = parameters = 0
    for name, parameter in sorted(owner.base_model.named_parameters(), key=lambda p: p[0]):
        if not parameter.requires_grad:
            continue
        header = str((name, str(parameter.dtype), tuple(parameter.shape))).encode()
        digest.update(struct.pack('>Q', len(header)))
        digest.update(header)
        raw = parameter.detach().reshape(-1).contiguous().view(torch.uint8)
        for block in raw.split(8 * 1024 * 1024):
            digest.update(memoryview(block.cpu().numpy()))
        elements += parameter.numel()
        parameters += 1
    if not elements:
        raise ValueError('No trainable Policy tensors to verify')
    return dict(rank=owner.rank, sha256=digest.hexdigest(), elements=elements, parameters=parameters)


class ContextTimer:
    """Record events on the trainer stream, resolving them once the arm has finished."""
    def __init__(self):
        self.calls = []

    @staticmethod
    def event():
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def begin(self, bundle, count):
        return dict(bundle=bundle, count=count, start=self.event(), host_start=time.perf_counter())

    def model_begin(self, marker):
        marker['model_start'] = self.event()

    def model_end(self, marker):
        marker['model_end'] = self.event()

    def end(self, marker):
        marker['end'] = self.event()
        marker['host_s'] = time.perf_counter() - marker['host_start']
        self.calls.append(marker)

    def report(self):
        rows = [dict(bundle=m['bundle'], count=m['count'], host_s=m['host_s'],
                     cuda_service_s=m['start'].elapsed_time(m['end']) / 1000.,
                     cuda_model_s=m['model_start'].elapsed_time(m['model_end']) / 1000.)
                for m in self.calls]
        n = len(rows)
        host = sum(r['host_s'] for r in rows)
        service = sum(r['cuda_service_s'] for r in rows)
        model = sum(r['cuda_model_s'] for r in rows)
        return dict(calls=n, observations=sum(r['count'] for r in rows),
                    host_service_sum_s=host, cuda_service_sum_s=service, cuda_model_sum_s=model,
                    mean_host_s=host/n if n else None, mean_cuda_service_s=service/n if n else None,
                    calls_detail=rows)


def _validate(owner, arm, shard):
    if (arm not in ('pruned','jit') or owner.training_role != 'policy'
            or owner.world_config.backend != 'model_free' or not owner._batch_autotune_enabled
            or owner.world_size != 4 or owner.super_batch_size != 144 or owner.clip_mode != 'gipo'):
        raise ValueError('The paired GPU measurement requires four model-free GIPO/autobatch ranks')
    ids = shard['sample_ids']
    if (shard['rank'] != owner.rank or shard['version'] != owner.global_step
            or int(owner.model.global_steps) != owner.global_step
            or len(ids) != 144 or len(set(ids)) != 144 or len(shard['observations']) != 144):
        raise ValueError('Paired trainer received the wrong ranked sample cohort or published version')
    actions, old = np.asarray(shard['actions']), np.asarray(shard['old_logits'])
    if actions.shape != (144, 8, 7) or old.shape[:-1] != actions.shape:
        raise ValueError('Paired trainer received malformed action-token or behavior-logit geometry')
    return actions, old


def run_arm(owner, arm, shard, *, micro_begin, micro_end, trim):
    device = next(owner.model.parameters()).device
    actions, old_logits = _checked(lambda: _validate(owner, arm, shard), device)
    version = int(owner.global_step)
    step = version + 1
    configure_training(owner, step, 16)
    owner._policy_joint_batch_mode.begin(step, micro=16)
    # Execute the real AdamW/ZeRO2 work while leaving the P189 numerical
    # weights fixed for the collectors and both arms. No training checkpoint is written.
    for group in owner.model.optimizer.param_groups:
        group['lr'] = 0.
    owner.capture_policy_metadata = dict(sample_ids=list(shard['sample_ids']),
        reservation_id=str(shard['cohort_id']), fresh_versions=[version,0], jit_pending=arm == 'jit')
    before = int(owner.model.global_steps)
    before_skips = int(owner.model.skipped_steps)
    original_actor_start = owner.policy_train_start_step
    original_config = owner.fresh_config
    owner.policy_train_start_step = 0
    timer = ContextTimer()
    initialization_prepass = arm == 'jit' and not getattr(getattr(owner, 'advantage_ema', None), 'updates', 0)
    started = time.perf_counter()
    try:
        if arm == 'pruned':
            prepared = owner._prepare_policy_batch(shard['observations'], actions, shard['advantages'],
                                                   old_logits, shard['targets'])
            advantages = prepared[2]
            moments = torch.stack((advantages.sum(), advantages.square().sum(),
                                   advantages.new_tensor(advantages.numel())))
            dist.all_reduce(moments)
            center = moments[0] / moments[2]
            scale = (moments[1] / moments[2] - center.square()).clamp_min(1e-12).sqrt()
            metrics = owner._train_policy(*prepared, center, scale, {})
            del prepared
        else:
            owner.fresh_config = SimpleNamespace(jit_gae_mode='fused', advantage_norm='ema_zscore',
                jit_context_micro=2, advantage_ema_beta=.9, advantage_scale_floor=.001, policy_sampling='chunks')

            def begin(rank_owner, workload, engine, local_micro, start, end, *, sample_indices):
                token = micro_begin(rank_owner, workload, engine, local_micro, start, end)
                token['sample_ids'] = [shard['sample_ids'][i] for i in sample_indices]
                begin_training_micro(rank_owner, torch)
                return token

            def end(rank_owner, token, engine, error=None):
                micro_end(rank_owner, token, engine, error=error)
                if error is None:
                    end_training_micro(rank_owner)

            metrics = run_jit_policy_update(owner, shard['observations'], actions, old_logits,
                micro_begin=begin, micro_end=end, trim=trim,
                original_denominators=shard['original_denominators'], context_timer=timer)
            finish_training(owner, metrics['policy_train_time'])
    finally:
        owner.policy_train_start_step = original_actor_start
        owner.fresh_config = original_config
    owner._policy_joint_batch_mode.complete(step)
    torch.cuda.synchronize(device)
    owner.telemetry.flush(force_gpu=True)
    torch.cuda.empty_cache()
    seconds = time.perf_counter() - started
    losses = [float(x) for x in metrics['epoch_losses']]
    if len(losses) != 9 or not all(math.isfinite(x) for x in losses):
        raise RuntimeError('The short test needs nine finite real GIPO training micros on every rank')
    context = timer.report()
    if arm == 'jit':
        expected = shard['bundle_context_counts']
        counts = [sum(r['count'] for r in context['calls_detail'] if r['bundle'] == j)
                  for j in range(9)]
        if counts != list(expected):
            raise RuntimeError('GPU context observations differ from the planned per-bundle suffixes')
    return dict(arm=arm, rank=owner.rank, version=version, version_after=owner.global_step,
                cohort_id=shard['cohort_id'], sample_ids_sha256=shard['sample_ids_sha256'],
                optimizer_delta=int(owner.model.global_steps)-before,
                skipped_delta=int(owner.model.skipped_steps)-before_skips,
                actor_loss_enabled=True, learning_rate=0., train_micro=16, gas=9,
                initialization_prepass=initialization_prepass, context=context,
                trainer_wall_s=seconds, mean_loss=sum(losses)/len(losses))
