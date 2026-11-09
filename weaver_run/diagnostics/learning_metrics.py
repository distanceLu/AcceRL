"""Offline learning diagnostics. Never imported by the training entry point.

The caller supplies actual tensors from one frozen model and a fixed replay
panel. Metadata, advantage means and training losses cannot replace these inputs.
"""
from __future__ import annotations

import math
import numpy as np


def paired_wm_metrics(predicted_images, true_images, predicted_states, true_states, valid):
    """One episode/noise pair; images in [0,1], states in physical LIBERO units.

    Shapes: [T,2,3,H,W], [T,8], and [T]. State schema is explicitly
    eef-position(3), axis-angle(3), gripper(2). Only XYZ gets a geometric L2;
    axis-angle coordinates are not treated as a rotation geodesic distance.
    Inputs must have identical action-into-frame alignment BEFORE calling.
    """
    pi, ti, ps, ts = (np.asarray(x, dtype=np.float64) for x in
                       (predicted_images, true_images, predicted_states, true_states))
    mask = np.asarray(valid)
    if (pi.shape != ti.shape or pi.ndim != 5 or pi.shape[1:3] != (2,3)
        or ps.shape != ts.shape or ps.shape != (len(pi),8)
        or mask.dtype != bool or mask.shape != (len(pi),) or not mask.any()):
        raise ValueError('Need aligned WM prediction/ground-truth tensors and valid-frame mask')
    for x in (pi,ti,ps,ts):
        if not np.isfinite(x[mask]).all():raise ValueError('Nonfinite valid WM pair')
    if any(np.any((x[mask] < 0) | (x[mask] > 1)) for x in (pi,ti)):
        raise ValueError('WM image errors require explicit [0,1] pixels')
    mse=np.mean((pi[mask]-ti[mask])**2,axis=(2,3,4))
    errors=ps[mask]-ts[mask]
    return {'valid_frames':int(mask.sum()),
            'image_mse_full':float(mse[:,0].mean()), 'image_mse_wrist':float(mse[:,1].mean()),
            'eef_xyz_l2_mean':float(np.linalg.norm(errors[:,:3],axis=1).mean()),
            'eef_xyz_l2_max':float(np.linalg.norm(errors[:,:3],axis=1).max()),
            'axis_angle_coordinate_mse':float(np.mean(errors[:,3:6]**2)),
            'gripper_mse':float(np.mean(errors[:,6:]**2)),
            'scope':'paired fixed real actions; no policy divergence and no training-loss proxy'}


def paired_rm_metrics(predictions, targets, valid, episode_ids):
    """Score the RM's discounted-success regression target, NOT a binary reward.

    Results include equal-episode weighting as well as frame weighting, so long
    failed episodes do not silently change the evaluation estimand.
    """
    p, y = (np.asarray(x, dtype=np.float64) for x in (predictions, targets))
    mask = np.asarray(valid)
    ids = np.asarray(episode_ids)
    if p.ndim != 1 or any(x.shape != p.shape for x in (y, mask, ids)) or mask.dtype != bool:
        raise ValueError('Aligned one-dimensional predictions/targets/mask/episode IDs required')
    if not mask.any():
        raise ValueError('No observed outcome pairs')
    if not np.isfinite(p[mask]).all() or not np.isfinite(y[mask]).all():
        raise ValueError('Nonfinite valid RM pair')
    if np.any((p[mask] < 0) | (p[mask] > 1) | (y[mask] < 0) | (y[mask] > 1)):
        raise ValueError('RM score and target must be in [0,1]')
    err = p[mask] - y[mask]
    by_episode = []
    for eid in np.unique(ids[mask]):
        e = p[mask & (ids == eid)] - y[mask & (ids == eid)]
        by_episode.append({'episode_id': str(eid), 'n': len(e), 'mse': float(np.mean(e*e)),
                           'mae': float(np.mean(abs(e))), 'bias': float(np.mean(e))})
    return {'valid_frames': int(mask.sum()), 'masked_frames': int((~mask).sum()),
            'episodes': len(by_episode), 'mse_frame_weighted': float(np.mean(err*err)),
            'mae_frame_weighted': float(np.mean(abs(err))), 'bias_frame_weighted': float(np.mean(err)),
            'mse_episode_weighted': float(np.mean([r['mse'] for r in by_episode])),
            'by_episode': by_episode, 'target': 'discounted_observed_success_not_immediate_binary_reward'}


def gradient_geometry(real, imagined):
    """Exact full-parameter geometry for already weighted source contributions.

    Accumulate after all micros/ranks in the same parameter coordinates. Calling
    this separately per micro and averaging cosines is mathematically different.
    A zero vector has an undefined angle, represented by None.
    """
    import torch
    if set(real) != set(imagined) or not real:
        raise ValueError('Both gradients must cover identical named parameters')
    rr = ii = ri = 0.0
    for name in sorted(real):
        r, i = real[name], imagined[name]
        if r.shape != i.shape:
            raise ValueError('Mismatched gradient shape: '+name)
        r, i = (v.detach().reshape(-1).to(device='cpu', dtype=torch.float64) for v in (r, i))
        if not torch.isfinite(r).all() or not torch.isfinite(i).all():
            raise ValueError('Nonfinite gradient: '+name)
        rr += float(torch.dot(r, r)); ii += float(torch.dot(i, i)); ri += float(torch.dot(r, i))
    total2 = max(0.0, rr+ii+2*ri)
    cos = max(-1.0, min(1.0, ri/math.sqrt(rr*ii))) if rr and ii else None
    return {'real_norm': math.sqrt(rr), 'imagined_norm': math.sqrt(ii),
            'dot': ri, 'cosine': cos, 'angle_degrees': math.degrees(math.acos(cos)) if cos is not None else None,
            'joint_norm': math.sqrt(total2), 'imagined_to_real_norm': math.sqrt(ii/rr) if rr else None,
            'real_dot_joint': rr+ri, 'imagined_dot_joint': ii+ri,
            'real_projection_fraction': (rr+ri)/total2 if total2 else None,
            'imagined_projection_fraction': (ii+ri)/total2 if total2 else None,
            'interpretation': 'Euclidean gradient geometry; not Adam parameter-update geometry or success causality'}


def source_losses(logits, values, actions, old_logits, advantages, targets,
                  masks, token_weights, sources, *, vf_coef, ent_coef, kl_coef,
                  clip_mode, clip_eps, actor_enabled=True):
    """Split the actual JIT micro objective without renormalizing each source.

    sources: 0=real, 1=imagined. Preserve ORIGINAL micro token_weights, action
    masks, detached targets and the one fixed advantage normalization snapshot.
    The caller applies the same GAS and DP reduction factor to every contribution.
    This API is for the restored JIT recipe; the legacy current-critic trainer
    requires its own loss adapter and must not silently use this one.
    """
    import torch
    from ..engine.jit_gae import masked_policy_loss
    n = values.numel()
    if not n or sources.ndim != 1 or len(sources) != n or not torch.all((sources == 0) | (sources == 1)):
        raise ValueError('Every chunk must have a known real/imagined source')
    logits = logits.reshape(*actions.shape, logits.shape[-1])
    result = {}
    for label, code in (('real', 0), ('imagined', 1)):
        select = sources == code
        count = int(select.sum())
        if not count:
            # Keep both actor and value paths for a valid zero-gradient source.
            zero = logits.sum()*0 + values.sum()*0
            result[label] = dict(actor=zero, critic=zero, regularizer=zero)
            continue
        args = (logits[select], values.reshape(-1)[select], actions[select], old_logits[select],
                advantages[select], targets[select], masks[select], token_weights[select])
        common = dict(clip_mode=clip_mode, clip_eps=clip_eps)
        actor = masked_policy_loss(*args, vf_coef=0., ent_coef=0., kl_coef=0.,
                                   actor_enabled=actor_enabled, **common)[0]
        critic = masked_policy_loss(*args, vf_coef=vf_coef*count/n, ent_coef=0., kl_coef=0.,
                                    actor_enabled=False, **common)[0]
        reg_args = (*args[:4], torch.zeros_like(args[4]), *args[5:])
        regularizer = masked_policy_loss(*reg_args, vf_coef=0., ent_coef=ent_coef, kl_coef=kl_coef,
                                         actor_enabled=actor_enabled, **common)[0]
        result[label] = dict(actor=actor, critic=critic, regularizer=regularizer)
    return result


def loss_gradients(loss, named_parameters, *, retain_graph=False, storage_device='cpu', storage_dtype=None):
    """No optimizer step and no writes to parameter.grad. Full coordinates."""
    import torch
    pairs = [(n, p) for n, p in named_parameters if p.requires_grad]
    if not pairs or loss.ndim:
        raise ValueError('Need trainable named parameters and a scalar loss')
    grads = torch.autograd.grad(loss, [p for _, p in pairs], allow_unused=True, retain_graph=retain_graph)
    dtype=torch.float64 if storage_dtype is None else storage_dtype
    return {n: (g.detach().to(device=storage_device, dtype=dtype) if g is not None
                else torch.zeros(p.shape, device=storage_device, dtype=dtype)) for (n, p), g in zip(pairs, grads)}
