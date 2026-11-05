"""Trainer-owned JIT GAE. ZeRO2 optimizer-update quantum, bounded activations.

No Ray calls on the micro path. Context-only forward uses the replicated ZeRO2
module (no engine hooks/step); selected forward uses the normal engine. Inputs
still originate on CPU. This is not a claim of a fully GPU-resident Replay.
"""
from __future__ import annotations

from contextlib import contextmanager
import time
import numpy as np
import torch
import torch.distributed as dist

from advantage_ema import AdvantageEMA, raw_sums
from jit_gae import ReverseGAE, masked_policy_loss
from jit_gae_plan import build_plan


def _distributed():
    return dist.is_available() and dist.is_initialized()


def _sum(tensor):
    if _distributed():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def _checked(fn, device, valid=None):
    """All ranks reach the same failure vote before a backward collective.

This does not promise recovery from a broken CUDA context/NCCL process. Such a
failure remains a failed job; no further optimizer window is admitted.
"""
    error, result = None, None
    try:
        result = fn()
    except Exception as exc:
        error = exc
    failed = torch.tensor(int(error is not None), device=device)
    if error is None and valid is not None:
        failed = (~valid(result)).to(failed.dtype)
    _sum(failed)
    if bool(failed):
        raise RuntimeError('JIT rank preparation/forward failed or nonfinite; stop the rank group') from error
    return result


def _same_state(state):
    if _distributed():
        rows = [torch.empty_like(state) for _ in range(dist.get_world_size())]
        dist.all_gather(rows, state)
        if any(not torch.equal(state, row) for row in rows):
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


def _forward_values(owner, nodes, segments, micro, counters):
    values = {}
    with torch.no_grad():
        for offset in range(0, len(nodes), micro):
            group = nodes[offset:offset+micro]
            inputs = _inputs(owner, [segments[n.segment]['observations'][n.index] for n in group])
            start=time.monotonic()
            logits, value = owner.base_model(inputs)
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
    return values


def _precompute(owner, segments, bundles, device, counters):
    gae = ReverseGAE(segments, device)
    result = []
    with _prepass_state(owner.base_model, device):
        for bundle in bundles:
            values = _forward_values(owner, bundle.nodes, segments,
                                     owner.fresh_config.jit_context_micro, counters)
            start=time.monotonic();result.append(gae.consume(bundle, values))
            owner._jit_times['gae_host_call_wall_s']+=time.monotonic()-start
    return result


def run_jit_policy_update(owner, observations, actions_np, old_logits_np, *, micro_begin, micro_end, trim):
    from cost_scheduler_runtime import scheduled_policy_update
    with scheduled_policy_update(owner):
        return _run(owner, observations, actions_np, old_logits_np,
                    micro_begin=micro_begin, micro_end=micro_end, trim=trim)


def _run(owner, observations, actions_np, old_logits_np, *, micro_begin, micro_end, trim):
    started = time.monotonic()
    config = owner.fresh_config
    device = next(owner.model.parameters()).device
    version = int(owner.global_step)
    micro = owner.train_batch_size
    owner._jit_receipts = None
    owner._jit_times=dict.fromkeys(('input_prepare_host_call_wall_s','no_grad_forward_host_call_wall_s',
        'selected_forward_host_call_wall_s','gae_host_call_wall_s','backward_step_host_call_wall_s'),0.)

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
        weights = np.empty(len(masks), np.float32)
        for start in range(0, len(masks), micro):
            weights[start:start+micro] = 1. / masks[start:start+micro].sum()
        return segments, bundles, masks, weights

    segments,bundles,masks,weights = _checked(prepare, device)
    # These small, immutable selected-cohort tensors cross H2D once per rank
    # window. Images and variable-length context inputs remain micro-bounded.
    action_device,old_device,mask_device,weight_device = _checked(lambda: (
        torch.tensor(actions_np,device=device,dtype=torch.long),
        torch.tensor(old_logits_np,device=device,dtype=torch.float32),
        torch.tensor(masks,device=device,dtype=torch.bool),
        torch.tensor(weights,device=device,dtype=torch.float32)),device)
    if not hasattr(owner, 'advantage_ema'):
        owner.advantage_ema = AdvantageEMA(device, config.advantage_ema_beta, config.advantage_scale_floor)
    ema = owner.advantage_ema
    expected = ema.updates
    before_steps, before_skips = int(owner.model.global_steps), int(owner.model.skipped_steps)
    _same_state(torch.stack([ema.mean, ema.variance, ema.mean.new_tensor(expected),
        ema.mean.new_tensor(version), ema.mean.new_tensor(len(bundles)),
        ema.mean.new_tensor(micro), ema.mean.new_tensor(before_steps), ema.mean.new_tensor(before_skips),
        ema.mean.new_tensor(ema.beta),ema.mean.new_tensor(ema.scale_floor),
        ema.mean.new_tensor(('optimizer_window','ema_zscore','ema_rms').index(config.advantage_norm)),
        ema.mean.new_tensor(('trainer_separate','fused').index(config.jit_gae_mode)),
        ema.mean.new_tensor(owner.capture_policy_metadata['fresh_versions'][1]),
        ema.mean.new_tensor(config.policy_sampling == 'imagined_trajectories')]))
    counters = dict(context_only_states=0, prepass_selected_and_context_states=0,
                    no_grad_forward_calls=0, selected_grad_states=0, selected_grad_forward_calls=0)
    separate = config.jit_gae_mode == 'trainer_separate'
    need_prepass = separate or config.advantage_norm == 'optimizer_window' or not ema.updates
    targets = _checked(lambda: _precompute(owner,segments,bundles,device,counters), device) if need_prepass else None
    initial_sums = (_sum(sum((raw_sums(a) for a,_ in targets), torch.zeros(3,device=device,dtype=torch.float64)))
                    if need_prepass else None)
    if initial_sums is not None and not bool(torch.isfinite(initial_sums).all()):
        raise ValueError('Nonfinite initial global advantage moments')
    mean, scale = ema.frozen(config.advantage_norm, initial_sums)
    # Freeze once for the entire GAS window; do not mutate EMA inside the loop.
    gae = ReverseGAE(segments, device)
    sums = torch.zeros(3, device=device, dtype=torch.float64)
    metrics, receipt_tensors, receipt_indices = [], [], []
    prepass_s = time.monotonic()-started
    for number,bundle in enumerate(bundles):
        indices = list(bundle.selected)
        capture = micro_begin(owner, 'policy_train', owner.model, micro, number*micro,
                              (number+1)*micro, sample_indices=indices)
        try:
            def forward():
                values = {} if separate else _forward_values(owner,
                    [n for n in bundle.nodes if n.selected is None],segments,config.jit_context_micro,counters)
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
                index=torch.tensor(indices,device=device,dtype=torch.long)
                actions=action_device.index_select(0,index)
                old=old_device.index_select(0,index)
                mask=mask_device.index_select(0,index)
                weight=weight_device.index_select(0,index)
                loss,metric = masked_policy_loss(logits,value,actions,old,normalized,target,mask,weight,
                    vf_coef=owner.vf_coef, ent_coef=owner.ent_coef, kl_coef=owner.kl_coef,
                    clip_mode=owner.clip_mode, clip_eps=owner.clip_eps,
                    actor_enabled=version >= owner.policy_train_start_step)
                return loss,metric,advantage,target
            loss,metric,advantage,target = _checked(forward, device,
                valid=lambda rows:torch.stack([torch.isfinite(t).all() for t in rows]).all())
            sums += raw_sums(advantage)
            metrics.append(metric)
            receipt_tensors.append(torch.stack((advantage,target),dim=1))
            receipt_indices.extend(indices)
            backward_start=time.monotonic();owner.model.backward(loss)
            old_steps,old_skips = int(owner.model.global_steps),int(owner.model.skipped_steps)
            owner.model.step()
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
    if not bool(sums[0] == expected_count):
        raise RuntimeError('EMA statistics must cover selected chunks only, on every rank')
    ema.commit(sums,expected_updates=expected,optimizer_succeeded=True)
    # One packed result readback after the successful optimizer boundary. Required
    # finite/error votes above remain explicit synchronization points.
    packed = torch.cat((torch.stack(metrics).reshape(-1),torch.cat(receipt_tensors).reshape(-1),
                        totals[3:].float(),torch.stack((mean,scale,ema.mean,ema.variance)).float())).detach().cpu().numpy()
    split = len(bundles)*7
    metric_rows = packed[:split].reshape(-1,7)
    receipt_rows = packed[split:-7].reshape(-1,2)
    owner._jit_receipts = [dict(sample_id=observations[i]['_jit_sample_id'],advantage=float(row[0]),
        value_target=float(row[1]),refresh_policy_version=version) for i,row in zip(receipt_indices,receipt_rows)]
    trim('idle')
    elapsed = time.monotonic()-started
    owner.telemetry.event('jit_gae_optimizer_window',details=dict(rank=owner.rank,policy_version=version,
        mode=config.jit_gae_mode,normalization=config.advantage_norm,ema_updates=ema.updates,
        ema_beta=ema.beta,ema_scale_floor=ema.scale_floor,normalization_mean=float(packed[-4]),
        normalization_scale=float(packed[-3]),ema_mean=float(packed[-2]),ema_variance=float(packed[-1]),
        selected_chunks=len(observations),unique_rank_segments=len(segments),
        sampling_protocol=config.policy_sampling,
        whole_imagined_segments=sum(s.get('source') == 'imagined' for s in segments.values()),
        dependency_states=dependency_states,global_rank_dependency_states=int(packed[-7]),
        global_no_grad_states=int(packed[-6]),global_selected_grad_states=int(packed[-5]),
        global_unique_suffix_states=owner.capture_policy_metadata.get('jit_global_suffix_states'),
        original_shared_full_refresh_states=owner.capture_policy_metadata.get('jit_full_refresh_states'),
        initialization_prepass=bool(need_prepass),
        prepass_and_setup_s=prepass_s,wall_s=elapsed,weights_only_resume='reset_ema',
        timing_scope='host call wall, may enqueue GPU work or include implicit waits; not GPU kernel duration',
        selected_aux_h2d_payload_bytes=int(actions_np.nbytes+old_logits_np.nbytes+masks.nbytes+weights.nbytes),
        auxiliary_transfer_scope='logical selected actions/logits/masks/weights only; excludes images, context, protocol and copies',
        packed_report_d2h_payload_bytes=int(packed.nbytes),**owner._jit_times,**counters))
    keys=('epoch_losses','epoch_p_losses','epoch_v_losses','epoch_e_losses','epoch_kl_losses','epoch_ent','epoch_kl_divs')
    return dict(zip(keys,metric_rows.T.tolist()),policy_train_time=elapsed)
