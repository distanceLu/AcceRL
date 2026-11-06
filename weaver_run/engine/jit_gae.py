"""Detached reverse-time GAE on the trainer device; no host scalar reads."""
from __future__ import annotations

import torch


class ReverseGAE:
    def __init__(self, segments, device):
        self.segments = segments
        self.rewards = {sid:torch.tensor(s['rewards'], dtype=torch.float64, device=device)
                        for sid,s in segments.items()}
        self.zero = torch.zeros((), device=device, dtype=torch.float64)
        self.carry, self.next_value = {}, {}

    def consume(self, bundle, values):
        advantages, returns = {}, {}
        for node in bundle.nodes:
            s = self.segments[node.segment]
            value = values[(node.segment, node.index)].detach().reshape(()).to(torch.float64)
            if node.bootstrap:
                self.next_value[node.segment] = value
                self.carry[node.segment] = self.zero
                continue
            terminal = bool(s['terminated'][node.index])
            truncated = bool(s['truncated'][node.index])
            nxt = self.zero if terminal else self.next_value[node.segment]
            carry = self.carry.get(node.segment, self.zero)
            delta = self.rewards[node.segment][node.index] + s['gamma']*nxt - value
            adv = delta + (0. if terminal or truncated else s['gamma']*s['lambda_'])*carry
            self.carry[node.segment], self.next_value[node.segment] = adv.detach(), value
            if node.selected is not None:
                advantages[node.selected] = adv.float().detach()
                returns[node.selected] = (adv+value).float().detach()
        return (torch.stack([advantages[i] for i in bundle.selected]),
                torch.stack([returns[i] for i in bundle.selected]))


def masked_policy_loss(logits, value, actions, old_logits, advantages, targets,
                       masks, token_weights, *, vf_coef, ent_coef, kl_coef,
                       clip_mode, clip_eps, actor_enabled=True):
    """Preserve original micro token weighting even when reverse order regroups it.

    token_weights[i] = 1 / valid_tokens_in_original_micro(i). DeepSpeed's fixed
    GAS scaling and DP averaging remain unchanged; value loss is chunk-weighted.
    """
    value = value.float().reshape(-1)
    logits = logits.reshape(*actions.shape, logits.shape[-1])
    vloss = vf_coef*(value-targets).square().mean()
    zero = vloss.new_zeros(())
    if not actor_enabled:
        return vloss, torch.stack((vloss.detach(), zero, vloss.detach(), zero, zero, zero, zero))
    dist = torch.distributions.Categorical(logits=logits)
    old = torch.distributions.Categorical(logits=old_logits.detach())
    ratio = (dist.log_prob(actions)-old.log_prob(actions)).exp()
    adv = advantages.detach()[:, None, None]
    if clip_mode == 'gipo':
        coeff = (-.5*ratio.detach().clamp_min(1e-9).log().square()).exp()
        surrogate = ratio*adv*coeff
    elif clip_mode == 'ppo':
        surrogate = torch.minimum(ratio*adv, ratio.clamp(1-clip_eps, 1+clip_eps)*adv)
    elif clip_mode == 'sapo':
        tau = torch.where(adv > 0, 1., 2.)
        surrogate = (tau*(ratio.clamp(1e-6, 1e6)-1)).sigmoid()*(4/tau)*adv
    else:
        raise ValueError('Unknown policy objective')
    weight = masks.to(logits.dtype)*token_weights[:, None, None]
    ploss = -(surrogate*weight).sum()
    entropy = (dist.entropy()*weight).sum()
    divergence = (torch.distributions.kl_divergence(old, dist)*weight).sum()
    eloss, kloss = -ent_coef*entropy, kl_coef*divergence
    loss = ploss+vloss+eloss+kloss
    return loss, torch.stack([t.detach() for t in (loss,ploss,vloss,eloss,kloss,entropy,divergence)])
