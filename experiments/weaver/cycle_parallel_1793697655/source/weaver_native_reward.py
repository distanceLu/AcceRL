"""Native WEAVER RM adapted to a bounded, explicitly uncalibrated LIBERO score.

The native head parameters and state-dict keys remain unchanged. A sigmoid link
and the original half-MSE use genuine environment success labels. The fixed
global-cohort class weights are plans, not proof of sample consumption.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from types import MethodType

from fresh_reward_targets import (DENSE_MODE, validate_dense_specification,
                                  validate_dense_reward_cohort)


MODE = 'libero_success_score_v1'
COHORT_PROTOCOL = 'weaver_reward_cohort_v1'
LOSS_DEFINITION = '0.5*mean(weight*(sigmoid(raw)-label)^2)'


def validate_specification(specification):
    if not isinstance(specification,Mapping):
        raise ValueError('Native reward requires an explicit specification mapping')
    if specification.get('mode') == DENSE_MODE:
        return validate_dense_specification(specification)
    required = {'mode':MODE,'link':'sigmoid','loss':'brier','label_source':'libero_env_step',
                'calibrated':False,'terminal':'horizon_only',
                'balance':'global_cohort_inverse_frequency','loss_coefficient':1.0}
    if (dict(specification) != required or specification.get('calibrated') is not False or
            isinstance(specification.get('loss_coefficient'),bool)):
        raise ValueError('Native reward requires the explicit uncalibrated LIBERO sigmoid/half-Brier contract')
    return required


def validate_reward_cohort(value):
    if isinstance(value,dict) and value.get('mode') == DENSE_MODE:
        return validate_dense_reward_cohort(value)
    if not isinstance(value,dict) or value.get('protocol') != COHORT_PROTOCOL:
        raise ValueError('Native RM requires its actual global reward cohort metadata')
    if value.get('label_source') != 'libero_env_step':
        raise ValueError('Native RM rejects pseudo rewards and non-environment success labels')
    positive,negative = value.get('positive_count'),value.get('negative_count')
    if any(type(number) is not int or number < 0 for number in (positive,negative)) or positive+negative < 1:
        raise ValueError('Reward global class counts must be nonnegative integers with real coverage')
    frames = value.get('frame_slice')
    if (not isinstance(frames,(list,tuple)) or len(frames)!=2 or
            any(type(number) is not int for number in frames) or not 0 <= frames[0] < frames[1]):
        raise ValueError('Reward selected frame slice is invalid')
    enabled = positive>0 and negative>0
    weight = negative/positive if enabled else 0.
    if (type(value.get('rm_active')) is not bool or value['rm_active'] != enabled or
            value.get('negative_weight') != 1. or isinstance(value.get('negative_weight'),bool) or
            isinstance(value.get('positive_weight'),bool) or
            not isinstance(value.get('positive_weight'),(int,float)) or
            not math.isfinite(value['positive_weight']) or
            not math.isclose(value['positive_weight'],weight,rel_tol=1e-12,abs_tol=1e-12)):
        raise ValueError('Reward active flag or weights differ from the fixed global class counts')
    if value.get('global_selected_frames',positive+negative) != positive+negative:
        raise ValueError('Reward class counts disagree with selected global frame count')
    return copy.deepcopy(value)


class NativeRewardController:
    def __init__(self,upstream,specification):
        self.upstream,self.rm = upstream,upstream.rm
        self.specification = validate_specification(specification)
        self.raw_forward = self.rm.forward
        self.cohort = None
        self.last_statistics = None
        self.micro_targets = self.micro_mask = None
        self.parameters = list(self.rm.parameters())
        if not self.parameters:
            raise ValueError('Native RM has no parameters')

    def set_cohort(self,metadata):
        self.cohort = validate_reward_cohort(metadata)
        if self.dense:
            self.cohort = validate_dense_reward_cohort(metadata,self.specification)
        elif self.cohort.get('mode') == DENSE_MODE:
            raise ValueError('Legacy native RM cannot consume a dense regression cohort')
        self.last_statistics = None
        return self.cohort

    @property
    def dense(self):
        return self.specification['mode'] == DENSE_MODE

    def set_micro_targets(self,targets,mask):
        if not self.dense or self.micro_targets is not None:
            raise RuntimeError('Dense micro target context must have one explicit owner')
        import torch
        if (not isinstance(targets,torch.Tensor) or not isinstance(mask,torch.Tensor)
                or targets.ndim != 2 or mask.shape != targets.shape or mask.dtype != torch.bool):
            raise ValueError('Dense native micro requires aligned target and boolean mask matrices')
        self.micro_targets,self.micro_mask=targets,mask

    def clear_micro_targets(self):
        self.micro_targets=self.micro_mask=None

    @property
    def active(self):
        return bool(self.cohort and self.cohort['rm_active'])

    def forward(self,obs,actions,tasks):
        return self.raw_forward(obs,actions,tasks).float().sigmoid()

    def compute_loss(self,obs,actions,tasks,gt_rewards):
        if self.cohort is None:
            raise RuntimeError('Native RM training requires begin_update with a fixed global reward cohort')
        import torch
        prediction = self.rm(obs,actions,tasks)
        targets = gt_rewards.float()
        if self.dense:
            return self._dense_loss(prediction,targets)
        if targets.shape != prediction.shape or not torch.isfinite(targets).all().item() or (
                (targets != 0) & (targets != 1)).any().item():
            raise ValueError('Native RM accepts only aligned true binary LIBERO success labels')
        weights = torch.where(targets.bool(),float(self.cohort['positive_weight']),1.)
        brier = (prediction-targets).square()
        weighted = (weights*brier).mean()
        loss = .5*weighted if self.active else 0.*weighted
        self.last_statistics = {'positive_labels_with_native_augmentation':int(targets.sum().item()),
            'negative_labels_with_native_augmentation':int(targets.numel()-targets.sum().item()),
            'unweighted_brier':float(brier.detach().mean()),
            'weighted_brier':float(weighted.detach()),'native_half_brier_loss':float(loss.detach()),
            'rm_active':self.active,'label_source':'libero_env_step','calibrated':False}
        return prediction,loss

    def _dense_loss(self,prediction,targets):
        import torch
        base,mask=self.micro_targets,self.micro_mask
        if base is None or mask is None:
            raise RuntimeError('Dense native RM requires the measured micro target/mask context')
        expected=torch.cat([base,base],dim=0).float()
        paired_mask=torch.cat([mask,mask],dim=0)
        if (prediction.shape != expected.shape or targets.shape != expected.shape
                or not torch.equal(targets,expected) or not torch.isfinite(targets).all().item()
                or ((targets<0)|(targets>1)).any().item()):
            raise ValueError('Native dense augmentation must preserve both aligned copies of its targets')
        error=(prediction-targets).square()
        valid=int(paired_mask.sum().item())
        masked=paired_mask.numel()-valid
        total=self.cohort['global_selected_frames']
        global_valid=self.cohort['valid_count']
        # DDP averages ranks and the engine averages GAS. Every micro retains
        # its full denominator; multiplying by global_selected/global_valid
        # yields one global valid-frame mean even when masks differ per rank.
        scale=total/global_valid if global_valid else 0.
        normalized=.5*(error*paired_mask).mean()*scale
        loss=normalized if self.active else 0.*normalized
        actual=targets[paired_mask].detach().double()
        local_mse=float(error.detach()[paired_mask].mean().item()) if valid else 0.
        self.last_statistics={
            'valid_targets_with_native_augmentation':valid,
            'masked_targets_with_native_augmentation':masked,
            'target_sum_with_native_augmentation':float(actual.sum().item()),
            'target_square_sum_with_native_augmentation':float(actual.square().sum().item()),
            'masked_target_mse':local_mse,'native_half_mse_loss':float(loss.detach().item()),
            'rm_active':self.active,'label_source':'libero_env_step',
            'target_mode':DENSE_MODE,'calibrated':False}
        return prediction,loss

    def report(self):
        definition=('0.5*global_valid_mean((sigmoid(raw)-discounted_success_target)^2)'
                    if self.dense else LOSS_DEFINITION)
        return {**self.specification,'loss_definition':definition,
                'score_semantics':('bounded_uncalibrated_discounted_success_value' if self.dense
                                   else 'bounded_uncalibrated_success_score'),
                'cohort':copy.deepcopy(self.cohort),'last_micro':copy.deepcopy(self.last_statistics)}


def install_native_reward_head(upstream,specification):
    existing = getattr(upstream,'native_reward_controller',None)
    if existing is not None:
        if existing.specification != validate_specification(specification):
            raise ValueError('Native RM cannot be reconfigured after installation')
        return existing
    controller = NativeRewardController(upstream,specification)
    def forward(rm,obs,actions,tasks):
        return controller.forward(obs,actions,tasks)
    def compute_loss(rm,obs,actions,tasks,gt_rewards):
        return controller.compute_loss(obs,actions,tasks,gt_rewards)
    upstream.rm.forward = MethodType(forward,upstream.rm)
    upstream.rm.compute_loss = MethodType(compute_loss,upstream.rm)
    upstream.native_reward_controller = controller
    upstream._rm_loss_coeff = float(specification['loss_coefficient'])
    upstream._critic_loss_coeff = 0.
    for name,parameter in upstream.named_parameters():
        parameter.requires_grad_(name.startswith(('wm.','rm.')))
    return controller
