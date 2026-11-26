"""Trainer-owned JIT GAE. ZeRO2 optimizer-update quantum, bounded activations.

No Ray calls on the micro path. Context-only forward uses the replicated ZeRO2
module (no engine hooks/step); selected forward uses the normal engine. Inputs
still originate on CPU. This is not a claim of a fully GPU-resident Replay.
"""
from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
import os
import time
import numpy as np
import torch
import torch.distributed as dist

from advantage_ema import AdvantageEMA, moments, raw_sums
from jit_gae import ReverseGAE, masked_policy_loss
from jit_gae_plan import build_plan


def _distributed():
    return dist.is_available() and dist.is_initialized()


def _sum(tensor):
    if _distributed():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def _checked(fn, device, valid=None, probe=None, timings=None):
    """All ranks reach the same failure vote before a backward collective.

This does not promise recovery from a broken CUDA context/NCCL process. Such a
failure remains a failed job; no further optimizer window is admitted.
"""
    error, result = None, None
    try:
        result = fn()
        failed = (~valid(result)).to(torch.int32) if valid is not None else torch.zeros((), dtype=torch.int32, device=device)
    except Exception as exc:
        error = exc
        failed = torch.ones((), dtype=torch.int32, device=device)
    if probe is not None:
        probe.mark('forward_ready')
    if timings is not None:
        vote_started = time.monotonic()
    _sum(failed)
    if probe is not None:
        voted = probe.mark('vote_done')
    failure = bool(failed)
    if timings is not None:
        timings['forward_vote_host_wait_s'] += time.monotonic() - vote_started
    if probe is not None:
        probe.vote_complete(voted)
    if failure:
        raise RuntimeError('JIT rank preparation/forward failed or nonfinite; stop the rank group') from error
    return result


def _same_state(state):
    if _distributed():
        rows = [torch.empty_like(state) for _ in range(dist.get_world_size())]
        dist.all_gather(rows, state)
        if not bool((torch.stack(rows) == state).all()):
            raise RuntimeError('JIT rank version, EMA or accumulation recipe differs')


def _deterministic_module(module):
    for child in module.modules():
        if child.training and isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
            raise ValueError('Stateful training BatchNorm is not admitted by fused JIT')
        if child.training and isinstance(child, torch.nn.modules.dropout._DropoutNd) and child.p:
            raise ValueError('Active dropout requires a separate reviewed JIT RNG contract')
        if child.training:
            for name in ('attention_dropout', 'dropout_p', 'drop_prob', 'dropout'):
                value = getattr(child, name, 0.)
                if isinstance(value, (int, float)) and value:
                    raise ValueError('Active attention dropout is not admitted by fused JIT')


@contextmanager
def _prepass_state(module, device):
    """Extra initialization forward must not advance RNG or mutable buffers."""
    cpu_rng = torch.get_rng_state()
    gpu_rng = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
    buffers = {name: value.detach().clone() for name,value in module.named_buffers()}
    try:
        yield
        if (not torch.equal(torch.get_rng_state(),cpu_rng)
                or (gpu_rng is not None and not torch.equal(torch.cuda.get_rng_state(device),gpu_rng))):
            raise RuntimeError('JIT prepass consumed RNG; stochastic forward needs a reviewed contract')
        current = dict(module.named_buffers())
        if set(current) != set(buffers) or any(not torch.equal(current[k], v) for k,v in buffers.items()):
            raise RuntimeError('JIT prepass mutated model buffers; candidate fails closed')
    finally:
        torch.set_rng_state(cpu_rng)
        if gpu_rng is not None:
            torch.cuda.set_rng_state(gpu_rng, device)


def _inputs(owner, observations):
    # Dependency dictionaries and masks must never enter the model tokenizer.
    clean = [{k:v for k,v in obs.items() if not k.startswith('_jit_') and k != '_fresh_action_mask'}
             for obs in observations]
    started=time.monotonic()
    result=owner.base_model.prepare_inputs_batch(clean)
    owner._jit_times['input_prepare_host_call_wall_s']+=time.monotonic()-started
    return result


def _forward_values(owner, nodes, segments, micro, counters, *, timer=None, bundle=None):
    from trainer_compile import eager_inference_context

    values = {}
    with torch.no_grad(), eager_inference_context():
        for offset in range(0, len(nodes), micro):
            group = nodes[offset:offset+micro]
            marker = timer.begin(bundle, len(group)) if timer is not None else None
            inputs = _inputs(owner, [segments[n.segment]['observations'][n.index] for n in group])
            if timer is not None:
                timer.model_begin(marker)
            start=time.monotonic()
            logits, value = owner.base_model(inputs)
            if timer is not None:
                timer.model_end(marker)
            owner._jit_times['no_grad_forward_host_call_wall_s']+=time.monotonic()-start
            value = value.float().reshape(-1)
            if len(value) != len(group):
                raise ValueError('JIT value geometry differs from observations')
            for i,n in enumerate(group):
                values[(n.segment,n.index)] = value[i].detach()
            counters['context_only_states' if all(n.selected is None for n in group)
                     else 'prepass_selected_and_context_states'] += len(group)
            counters['no_grad_forward_calls'] += 1
            del inputs, logits, value
            if timer is not None:
                timer.end(marker)
    return values


def _precompute(owner, segments, bundles, device, counters, context_micro):
    gae = ReverseGAE(segments, device)
    result = []
    with _prepass_state(owner.base_model, device):
        for bundle in bundles:
            values = _forward_values(owner, bundle.nodes, segments, context_micro, counters)
            start=time.monotonic();result.append(gae.consume(bundle, values))
            owner._jit_times['gae_host_call_wall_s']+=time.monotonic()-start
    return result


def run_jit_policy_update(owner, observations, actions_np, old_logits_np, *, micro_begin, micro_end, trim,
                          original_denominators=None, context_timer=None, config=None):
    from cost_scheduler_runtime import scheduled_policy_update
    with scheduled_policy_update(owner):
        return _run(owner, observations, actions_np, old_logits_np,
                    micro_begin=micro_begin, micro_end=micro_end, trim=trim,
                    original_denominators=original_denominators, context_timer=context_timer,
                    config=owner.fresh_config if config is None else config)


def _run(owner, observations, actions_np, old_logits_np, *, micro_begin, micro_end, trim,
         original_denominators, context_timer, config):
    started = time.monotonic()
    device = next(owner.model.parameters()).device
    version = int(owner.global_step)
    micro = owner.train_batch_size
    fast = os.environ.get('ACCERL_FAST_JIT_BOUNDARIES') == '1'
    owner._jit_receipts = None
    owner._jit_times=dict.fromkeys(('input_prepare_host_call_wall_s','no_grad_forward_host_call_wall_s',
        'selected_forward_host_call_wall_s','gae_host_call_wall_s','backward_step_host_call_wall_s',
        'forward_vote_host_wait_s'),0.)

    def prepare():
        if config.jit_gae_mode not in ('trainer_separate', 'fused'):
            raise ValueError('Trainer JIT mode is not enabled')
        _deterministic_module(owner.base_model)
        if owner.model.module is not owner.base_model:
            raise ValueError('Context and selected forwards must use the same trainer parameters')
        if getattr(owner.cfg, 'lora_dropout', 0.):
            raise ValueError('JIT requires explicit zero LoRA dropout')
        metadata = owner.capture_policy_metadata
        if (metadata.get('fresh_versions', [None])[0] != version or not metadata.get('jit_pending')
                or metadata['sample_ids'] != [obs['_jit_sample_id'] for obs in observations]
                or len(observations) != owner.super_batch_size):
            raise ValueError('JIT cohort/version differs from original reserved rank shard')
        from trajectory_sampling import enabled, validate_rank
        if enabled(config):
            validate_rank(observations, owner.rank,
                dist.get_world_size() if _distributed() else getattr(owner, 'world_size', None))
        elif any(obs['_jit_segment'].get('sampling') == 'imagined_trajectories' for obs in observations):
            raise ValueError('Trajectory transport received by a chunk-sampling trainer')
        segments, bundles = build_plan(observations, micro, version)
        masks = np.stack([obs['_fresh_action_mask'] for obs in observations])
        if (masks.dtype != np.bool_ or masks.shape != actions_np.shape
                or old_logits_np.shape[:-1] != actions_np.shape
                or not masks.any(axis=(1,2)).all()):
            raise ValueError('JIT action masks/old logits must preserve the original cohort geometry')
        if original_denominators is None:
            weights = np.empty(len(masks), np.float32)
            for start in range(0, len(masks), micro):
                weights[start:start+micro] = 1. / masks[start:start+micro].sum()
        else:
            # Original autobatch averages the means of two eight-sample groups
            # before GAS scales a 16-sample backward. Carry that denominator
            # with each sample even when its rank and training group change.
            denominators = np.asarray(original_denominators)
            if (micro != 16 or denominators.shape != (len(masks),)
                    or denominators.dtype.kind not in 'iu' or (denominators <= 0).any()
                    or (denominators > 8 * masks.shape[1] * masks.shape[2]).any()):
                raise ValueError('Invalid original eight-sample token denominator')
            weights = np.float32(.5) / denominators.astype(np.float32)
        tokens = np.asarray(actions_np, dtype=np.int64)
        if fast and (tokens.min() < 0 or tokens.max() >= old_logits_np.shape[-1]):
            raise ValueError('Action token outside categorical support')
        # Stage the complete gather order here. Each micro uses a device view.
        order = [b.selected for b in bundles]
        return (segments, bundles, masks, weights,
                torch.as_tensor(tokens, device=device), torch.as_tensor(old_logits_np, device=device, dtype=torch.float32),
                torch.as_tensor(masks, device=device), torch.as_tensor(weights, device=device),
                torch.tensor(order, device=device, dtype=torch.long))

    (segments, bundles, masks, weights, action_device, old_device,
     mask_device, weight_device, order_device) = _checked(prepare, device)
    if not hasattr(owner, 'advantage_ema'):
        owner.advantage_ema = AdvantageEMA(device, config.advantage_ema_beta, config.advantage_scale_floor)
    ema = owner.advantage_ema
    expected = ema.updates
    before_steps, before_skips = int(owner.model.global_steps), int(owner.model.skipped_steps)
    recipe_state = ema.mean.new_tensor([expected, version, len(bundles), micro, before_steps, before_skips,
        ema.beta, ema.scale_floor, ('optimizer_window','ema_zscore','ema_rms').index(config.advantage_norm),
        ('trainer_separate','fused').index(config.jit_gae_mode), owner.capture_policy_metadata['fresh_versions'][1],
        config.policy_sampling == 'imagined_trajectories', fast])
    _same_state(torch.cat((torch.stack((ema.mean, ema.variance)), recipe_state)))
    counters = dict(context_only_states=0, prepass_selected_and_context_states=0,
                    no_grad_forward_calls=0, selected_grad_states=0, selected_grad_forward_calls=0)
    separate = config.jit_gae_mode == 'trainer_separate'
    need_prepass = separate or config.advantage_norm == 'optimizer_window' or not ema.updates
    targets = _checked(lambda: _precompute(owner,segments,bundles,device,counters,config.jit_context_micro),
                       device) if need_prepass else None
    initial_sums = (_sum(sum((raw_sums(a) for a,_ in targets), torch.zeros(3,device=device,dtype=torch.float64)))
                    if need_prepass else None)
    if initial_sums is not None and not bool(torch.isfinite(initial_sums).all()):
        raise ValueError('Nonfinite initial global advantage moments')
    mean, scale = ema.frozen(config.advantage_norm, initial_sums)
    if getattr(context_timer,'capture_layout_normalization',False):
        # 保留 loss 实际使用的 FP64 均值和尺度，发布后再回读。
        context_timer.layout_normalization = (mean,scale)
    # Freeze once for the entire GAS window; do not mutate EMA inside the loop.
    gae = ReverseGAE(segments, device)
    batched_context = fast and not need_prepass and context_timer is None
    window_context = {}
    sums = torch.zeros(3, device=device, dtype=torch.float64)
    metrics, receipt_tensors, receipt_indices = [], [], []
    probe = None
    if os.environ.get('ACCERL_JIT_BENCH_PLAN'):
        from critical_probe import TrainerProbe
        probe = TrainerProbe.for_update(owner, version + 1)
    # ZeRO-2 can reduce gradients in every backward, including non-boundary
    # micros. All ranks must agree the forward succeeded before entering it.
    distributed_backward = _distributed() and dist.get_world_size() > 1
    deferred_vote = (fast and probe is None and not distributed_backward
                     and os.environ.get('ACCERL_DEFERRED_FORWARD_VOTE', '1') != '0')
    deferred_failed = torch.zeros((), dtype=torch.int32, device=device) if deferred_vote else None
    deferred_error = None
    last_bundle = len(bundles) - 1
    prepass_s = time.monotonic()-started
    for number,bundle in enumerate(bundles):
        indices = list(bundle.selected)
        capture = micro_begin(owner, 'policy_train', owner.model, micro, number*micro,
                              (number+1)*micro, sample_indices=indices)
        try:
            if probe is not None:
                probe.begin(number)
            def forward():
                if batched_context:
                    # No parameter update occurs until all GAS micros complete.
                    # Keep detached values only; reuse each original suffix node
                    # in its original reverse-GAE order.
                    if number == 0:
                        context_nodes = [n for b in bundles for n in b.nodes if n.selected is None]
                        window_context.update(_forward_values(owner, context_nodes, segments, micro, counters))
                    values = {(n.segment,n.index): window_context[(n.segment,n.index)]
                              for n in bundle.nodes if n.selected is None}
                else:
                    values = {} if separate else _forward_values(owner,
                        [n for n in bundle.nodes if n.selected is None],segments,config.jit_context_micro,counters,
                        timer=context_timer, bundle=number)
                inputs = _inputs(owner,[observations[i] for i in indices])
                start=time.monotonic();logits,value = owner.model.forward(inputs)
                owner._jit_times['selected_forward_host_call_wall_s']+=time.monotonic()-start
                value = value.float().reshape(-1)
                if len(value) != micro:
                    raise ValueError('Selected value shape mismatch')
                for j,i in enumerate(indices):
                    obs = observations[i]
                    values[(obs['_jit_segment']['segment_id'],obs['_jit_index'])] = value[j].detach()
                start=time.monotonic();advantage,target = targets[number] if separate else gae.consume(bundle,values)
                owner._jit_times['gae_host_call_wall_s']+=time.monotonic()-start
                normalized = ((advantage.double()-mean)/scale).float().detach()
                index = order_device[number]
                actions=action_device.index_select(0,index)
                old=old_device.index_select(0,index)
                mask=mask_device.index_select(0,index)
                weight=weight_device.index_select(0,index)
                loss,metric = masked_policy_loss(logits,value,actions,old,normalized,target,mask,weight,
                    vf_coef=owner.vf_coef, ent_coef=owner.ent_coef, kl_coef=owner.kl_coef,
                    clip_mode=owner.clip_mode, clip_eps=owner.clip_eps,
                    actor_enabled=version >= owner.policy_train_start_step, action_tokens_checked=fast)
                return loss,metric,advantage,target
            if deferred_vote:
                loss = metric = advantage = target = None
                if deferred_error is None:
                    try:
                        loss,metric,advantage,target = forward()
                        valid = torch.stack([torch.isfinite(t).all() for t in (loss,metric,advantage,target)]).all()
                        deferred_failed = deferred_failed | (~valid).to(torch.int32)
                    except Exception as exc:
                        deferred_error = exc
                        deferred_failed = torch.ones((), dtype=torch.int32, device=device)
                if number == last_bundle:
                    vote_started = time.monotonic()
                    _sum(deferred_failed)
                    any_failed = bool(deferred_failed)
                    owner._jit_times['forward_vote_host_wait_s'] += time.monotonic() - vote_started
                    if any_failed:
                        raise RuntimeError('JIT rank preparation/forward failed or nonfinite; stop the rank group') from deferred_error
                elif deferred_error is not None:
                    micro_end(owner,capture,owner.model)
                    continue
            else:
                loss,metric,advantage,target = _checked(forward, device,
                    valid=lambda rows:torch.stack([torch.isfinite(t).all() for t in rows]).all(), probe=probe,
                    timings=owner._jit_times)
            sums += raw_sums(advantage)
            metrics.append(metric)
            receipt_tensors.append(torch.stack((advantage,target),dim=1))
            receipt_indices.extend(indices)
            backward_start=time.monotonic();owner.model.backward(loss)
            if probe is not None:
                probe.mark('backward_only_done')
            old_steps,old_skips = int(owner.model.global_steps),int(owner.model.skipped_steps)
            owner.model.step()
            if probe is not None:
                probe.mark('backward_done')
            owner._jit_times['backward_step_host_call_wall_s']+=time.monotonic()-backward_start
            successful = max(0,int(owner.model.global_steps)-old_steps-int(owner.model.skipped_steps)+old_skips)
            counters['selected_grad_states'] += micro
            counters['selected_grad_forward_calls'] += 1
            del loss,advantage,target,metric
        except BaseException as error:
            micro_end(owner,capture,owner.model,error=error)
            raise
        else:
            micro_end(owner,capture,owner.model)
            owner.global_step += successful  # Capture hook labels the completed boundary itself.
        trim('micro')  # Preserve the parent allocator policy as an independent factor.
    owner._verify_optimizer_update(owner.model,'policy',before_steps,before_skips)
    dependency_states=sum(len(b.nodes) for b in bundles)
    totals=_sum(torch.cat((sums,sums.new_tensor([dependency_states,
        counters['context_only_states']+counters['prepass_selected_and_context_states'],
        counters['selected_grad_states']]))))
    sums=totals[:3]
    expected_count = len(observations)*(dist.get_world_size() if _distributed() else 1)
    ema.commit(sums, expected_updates=expected, optimizer_succeeded=True, expected_count=expected_count)
    batch_mean, batch_variance = moments(sums)
    # One packed result readback after the successful optimizer boundary. Required
    # finite/error votes above remain explicit synchronization points.
    packed_device = torch.cat((torch.stack(metrics).reshape(-1),torch.cat(receipt_tensors).reshape(-1),
                        totals[3:].float(),torch.stack((mean,scale,ema.mean,ema.variance,
                                                       batch_mean,batch_variance.sqrt())).float())).detach()
    if probe is not None:
        anchor = probe.mark('packed')
    readback_start = time.monotonic()
    packed = packed_device.cpu().numpy()
    packed_readback_s = time.monotonic() - readback_start
    if probe is not None:
        probe.finish(anchor, time.time_ns())
    del packed_device
    split = len(bundles)*7
    metric_rows = packed[:split].reshape(-1,7)
    receipt_rows = packed[split:-9].reshape(-1,2)
    owner._jit_receipts = [dict(sample_id=observations[i]['_jit_sample_id'],advantage=float(row[0]),
        value_target=float(row[1]),refresh_policy_version=version) for i,row in zip(receipt_indices,receipt_rows)]
    trim_start = time.monotonic()
    trim('idle')
    idle_trim_s = time.monotonic() - trim_start
    elapsed = time.monotonic()-started
    owner.telemetry.event('jit_gae_optimizer_window',details=dict(rank=owner.rank,policy_version=version,
        mode=config.jit_gae_mode,normalization=config.advantage_norm,ema_updates=ema.updates,
        ema_beta=ema.beta,ema_scale_floor=ema.scale_floor,normalization_mean=float(packed[-6]),
        normalization_scale=float(packed[-5]),ema_mean=float(packed[-4]),ema_variance=float(packed[-3]),
        raw_advantage_mean=float(packed[-2]),raw_advantage_std=float(packed[-1]),
        selected_chunks=len(observations),unique_rank_segments=len(segments),
        sampling_protocol=owner.capture_policy_metadata.get('real_sampling', config.policy_sampling),
        whole_imagined_segments=sum(s.get('source') == 'imagined' for s in segments.values()),
        dependency_states=dependency_states,global_rank_dependency_states=int(packed[-9]),
        global_no_grad_states=int(packed[-8]),global_selected_grad_states=int(packed[-7]),
        global_unique_suffix_states=owner.capture_policy_metadata.get('jit_global_suffix_states'),
        original_shared_full_refresh_states=owner.capture_policy_metadata.get('jit_full_refresh_states'),
        initialization_prepass=bool(need_prepass), fast_jit_boundaries=fast, batched_window_context=batched_context,
        deferred_forward_vote=deferred_vote,
        packed_readback_host_wall_s=packed_readback_s, idle_trim_host_wall_s=idle_trim_s,
        context_forward_batch_limit=micro if batched_context else config.jit_context_micro,
        prepass_and_setup_s=prepass_s,wall_s=elapsed,weights_only_resume='reset_ema',
        timing_scope='host call wall, may enqueue GPU work or include implicit waits; not GPU kernel duration',
        selected_aux_h2d_payload_bytes=int(actions_np.nbytes+old_logits_np.nbytes+masks.nbytes+weights.nbytes),
        auxiliary_transfer_scope='logical selected actions/logits/masks/weights only; excludes images, context, protocol and copies',
        packed_report_d2h_payload_bytes=int(packed.nbytes),**owner._jit_times,**counters))
    keys=('epoch_losses','epoch_p_losses','epoch_v_losses','epoch_e_losses','epoch_kl_losses','epoch_ent','epoch_kl_divs')
    return dict(zip(keys,metric_rows.T.tolist()),policy_train_time=elapsed)


def run_balanced_trajectory_update(owner, observations, actions, old_logits, denominators, *,
                                   micro_begin, micro_end, trim, context_timer=None):
    """The same trajectory-local optimizer window for formal training and paired timing."""
    from batch_autotune_runtime import begin_training_micro, end_training_micro, finish_training

    def begin(rank_owner, workload, engine, local_micro, start, end, *, sample_indices):
        token = micro_begin(rank_owner, workload, engine, local_micro, start, end)
        ids = rank_owner.capture_policy_metadata['sample_ids']
        token['sample_ids'] = [ids[i] for i in sample_indices]
        begin_training_micro(rank_owner, torch)
        return token

    def end(rank_owner, token, engine, error=None):
        micro_end(rank_owner, token, engine, error=error)
        if error is None:
            end_training_micro(rank_owner)

    recipe = SimpleNamespace(jit_gae_mode='fused', advantage_norm='ema_zscore', jit_context_micro=2,
                             advantage_ema_beta=.9, advantage_scale_floor=.001, policy_sampling='chunks')
    result = run_jit_policy_update(owner, observations, actions, old_logits, micro_begin=begin, micro_end=end,
                                   trim=trim, original_denominators=denominators,
                                   context_timer=context_timer, config=recipe)
    finish_training(owner, result['policy_train_time'])
    return result
