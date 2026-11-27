"""Explicit four-rank short-test endpoints. Called only by the paired diagnostic."""
import hashlib
import math
import os
import struct
import time

import numpy as np
import torch
import torch.distributed as dist

from batch_autotune_runtime import configure_training
from jit_gae_runtime import _checked, run_balanced_trajectory_update


def prepare_layout_ema(owner, cohort, version, first, *, repeated=False):
    ema = owner.advantage_ema
    if version != owner.global_step or ema.updates != version or version < (1 if repeated else 2):
        raise ValueError('Layout EMA preparation requires the completed warmup and current optimizer step')
    # 两臂都在计时前读写并同步这两个标量。
    current = torch.stack((ema.mean,ema.variance)).tolist()
    if first:
        owner._layout_ema_snapshot = (cohort,version,current)
    prior = getattr(owner,'_layout_ema_snapshot',None)
    if not prior or prior[:2] != (cohort,version if first else version-1):
        raise ValueError('Missing the initial EMA moments for this layout pair')
    if repeated:
        owner._layout_ema_snapshot = (cohort,version,prior[2])
    ema.mean.fill_(prior[2][0])
    ema.variance.fill_(prior[2][1])
    mean,variance = torch.stack((ema.mean,ema.variance)).tolist()
    if [mean,variance]!=prior[2] or not all(map(math.isfinite,(mean,variance))) or variance<0:
        raise ValueError('Invalid or unrecovered layout EMA moments')
    return dict(rank=owner.rank, version=version, updates=ema.updates, mean=mean,
                variance=variance, scale_floor=ema.scale_floor)


def read_layout_normalization(owner, version):
    computed_version,mean,scale = owner._layout_normalization
    if computed_version!=version or owner.global_step!=version+1:
        raise ValueError('Layout normalization receipt belongs to a different optimizer update')
    values = torch.stack((mean,scale)).tolist()
    owner._layout_normalization = None
    return dict(rank=owner.rank, version=version, mean=values[0], scale=values[1])


def layout_receipt(owner, ids, version):
    from gpu_trial_replay import _digest

    receipts = owner._jit_receipts
    consumed = [row['sample_id'] for row in receipts]
    if len(consumed)!=144 or set(consumed)!=set(ids) or any(row['refresh_policy_version']!=version for row in receipts):
        raise ValueError('JIT exposure receipts do not cover the layout rank at its current version')
    # 启动脚本提供默认值，训练器仅在开关为 1 时清理。
    cache_trim = {phase: '1' if os.environ.get('ACCERL_TRIM_'+phase.upper()+'_CACHE')=='1' else '0'
                  for phase in ('micro','idle')}
    return dict(consumed_sample_ids=consumed, consumed_sample_ids_sha256=_digest(consumed),cache_trim=cache_trim,
                advantage_value_targets=[[row['advantage'],row['value_target']] for row in receipts])


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
    if arm.startswith('shape_'):
        from paired_shape_trainer import run_shape_arm
        return run_shape_arm(owner, arm, shard, micro_begin=micro_begin, micro_end=micro_end, trim=trim)
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
    owner.policy_train_start_step = 0
    timer = ContextTimer()
    timer.capture_layout_normalization = 'layout_arm' in shard
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
            metrics = run_balanced_trajectory_update(owner, shard['observations'], actions, old_logits,
                shard['original_denominators'], micro_begin=micro_begin, micro_end=micro_end,
                trim=trim, context_timer=timer)
    finally:
        owner.policy_train_start_step = original_actor_start
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
    layout = {}
    if 'layout_arm' in shard:
        layout = layout_receipt(owner, shard['sample_ids'], version)
        owner._layout_normalization = (version,*timer.layout_normalization)
        denominators = dict(zip(shard['sample_ids'],shard['original_denominators']))
        layout.update(layout_arm=shard['layout_arm'],layout_planning_s=shard['layout_planning_s'],
            consumed_token_denominators=[denominators[sid] for sid in layout['consumed_sample_ids']],
            planned_bundle_context_counts=list(shard['bundle_context_counts']))
    return dict(arm=arm, rank=owner.rank, version=version, version_after=owner.global_step,
                sampled_device_memory=owner._autobatch_train_report['memory'] if arm=='jit' else None,
                cohort_id=shard['cohort_id'], sample_ids_sha256=shard['sample_ids_sha256'],
                optimizer_delta=int(owner.model.global_steps)-before,
                skipped_delta=int(owner.model.skipped_steps)-before_skips,
                actor_loss_enabled=True, learning_rate=0., train_micro=16, gas=9,
                initialization_prepass=initialization_prepass, context=context,
                trainer_wall_s=seconds, mean_loss=sum(losses)/len(losses), **layout)
