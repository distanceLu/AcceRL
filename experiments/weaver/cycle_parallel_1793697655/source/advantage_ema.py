"""GPU-resident, transactional optimizer-window advantage moments."""
from __future__ import annotations

import math
import torch


def moments(sums):
    count, total, square = sums.unbind()
    mean = total / count.clamp_min(1)
    return mean.detach(), (square / count.clamp_min(1) - mean.square()).clamp_min(0).detach()


def raw_sums(values):
    values = values.detach().to(torch.float64)
    return torch.stack((values.new_tensor(values.numel()), values.sum(), values.square().sum()))


class AdvantageEMA:
    def __init__(self, device, beta=.9, scale_floor=1e-3):
        if not math.isfinite(beta) or not 0 <= beta < 1:
            raise ValueError('EMA beta must be finite in [0,1)')
        if not math.isfinite(scale_floor) or scale_floor <= 0:
            raise ValueError('EMA scale floor must be positive and finite')
        self.beta, self.scale_floor = float(beta), float(scale_floor)
        self.mean = torch.zeros((), dtype=torch.float64, device=device)
        self.variance = torch.zeros_like(self.mean)
        self.updates = 0

    def frozen(self, mode, initialization_sums=None):
        if mode not in ('optimizer_window', 'ema_zscore', 'ema_rms'):
            raise ValueError('Unknown advantage normalization')
        if mode == 'optimizer_window' or not self.updates:
            if initialization_sums is None:
                raise ValueError('Exact first-window initialization required')
            mean, variance = moments(initialization_sums)
        else:
            mean, variance = self.mean.detach().clone(), self.variance.detach().clone()
        if mode == 'ema_rms':
            return torch.zeros_like(mean), (variance + mean.square()).sqrt().clamp_min(self.scale_floor)
        # Exact baseline matches the parent's population std convention.
        if mode == 'optimizer_window':
            return mean, variance.clamp_min(1e-12).sqrt() + 1e-8
        return mean, variance.sqrt().clamp_min(self.scale_floor)

    def commit(self, sums, *, expected_updates, optimizer_succeeded):
        if expected_updates != self.updates:
            raise RuntimeError('Duplicate or out-of-order EMA commit')
        if not optimizer_succeeded:
            return False
        if sums.shape != (3,) or not bool(torch.isfinite(sums).all()) or not bool(sums[0] > 0):
            raise ValueError('Invalid global advantage moments')
        mean, variance = moments(sums)
        if self.updates:
            delta = mean - self.mean
            variance = (self.beta*self.variance + (1-self.beta)*variance
                        + self.beta*(1-self.beta)*delta.square())
            mean = self.beta*self.mean + (1-self.beta)*mean
        self.mean.copy_(mean)
        self.variance.copy_(variance.clamp_min(0))
        self.updates += 1
        return True

    def state_dict(self):
        return dict(schema=1, beta=self.beta, scale_floor=self.scale_floor,
                    updates=self.updates, mean=self.mean.detach().clone(),
                    variance=self.variance.detach().clone())

    def load_state_dict(self, state):
        if (state.get('schema') != 1 or state.get('beta') != self.beta
                or state.get('scale_floor') != self.scale_floor
                or type(state.get('updates')) is not int or state['updates'] < 0):
            raise ValueError('EMA recovery metadata mismatch')
        mean = torch.as_tensor(state['mean'], device=self.mean.device, dtype=torch.float64)
        variance = torch.as_tensor(state['variance'], device=self.mean.device, dtype=torch.float64)
        if (mean.shape != () or variance.shape != () or not bool(torch.isfinite(mean))
                or not bool(torch.isfinite(variance)) or not bool(variance >= 0)):
            raise ValueError('Invalid recovered EMA moments')
        self.mean.copy_(mean); self.variance.copy_(variance); self.updates = state['updates']
