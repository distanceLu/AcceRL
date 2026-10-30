"""Native WEAVER flow-matching updates, with bounded streamed microbatches.

This engine is torch DDP, not DeepSpeed/ZeRO. It preserves the previously
validated offline World AdamW/FP32/BF16 objective; no CTRL objective or reward
head optimizer is imported. Optional model/backend injection is CPU-test only.
"""
from __future__ import annotations

from contextlib import nullcontext
import math
import time


class WeaverOnlineEngine:
    def __init__(self, backend, *, model=None, rank=0, world_size=4, micro_batch=4,
                 accumulation_steps=64, learning_rate=1e-5, warmup_updates=20,
                 weight_decay=.01, betas=(.9, .99), max_grad_norm=1., telemetry=None):
        import torch
        self.torch, self.backend = torch, backend
        self.module = backend.model
        self.model = self.module if model is None else model
        self.rank, self.world_size = int(rank), int(world_size)
        self.micro_batch, self.accumulation_steps = int(micro_batch), int(accumulation_steps)
        if min(self.micro_batch, self.accumulation_steps, self.world_size) < 1:
            raise ValueError('World micro/GAS/world size must be positive')
        self.peak_lr, self.warmup_updates = float(learning_rate), int(warmup_updates)
        self.max_grad_norm = float(max_grad_norm)
        if self.peak_lr <= 0 or self.warmup_updates < 0 or self.max_grad_norm != 1.:
            raise ValueError('Native World learning rate/warmup/clip is invalid')
        self.parameters = [p for p in self.module.parameters() if p.requires_grad]
        if not self.parameters:
            raise ValueError('Native World optimizer has no trainable parameters')
        self.optimizer = torch.optim.AdamW(self.parameters, lr=self.peak_lr,
                                         betas=tuple(betas), weight_decay=float(weight_decay))
        self.global_steps = self.skipped_steps = self.micro_steps = 0
        self.policy_step = 0
        self.telemetry = telemetry
        self.device = self.parameters[0].device
        self.active_update = None
        self.last_result = None
        self.publication_idle = lambda: True
        self.native_reward = getattr(backend,'native_reward_controller',
            getattr(getattr(backend,'upstream_model',None),'native_reward_controller',None))
        self.rm_updates = self.rm_disabled_updates = 0
        self.reward_positive_labels = self.reward_negative_labels = 0
        self.reward_target_valid_count = self.reward_target_masked_count = 0
        self.reward_target_sum = self.reward_target_square_sum = 0.
        self.last_reward_cohort = None

    def train_batch_size(self):
        return self.micro_batch * self.accumulation_steps * self.world_size

    def is_gradient_accumulation_boundary(self):
        return (self.micro_steps + 1) % self.accumulation_steps == 0

    def _all_true(self, condition):
        torch = self.torch
        flag = (condition.detach().to(device=self.device, dtype=torch.int32).clone()
                if isinstance(condition, torch.Tensor) else
                torch.tensor(int(bool(condition)), dtype=torch.int32, device=self.device))
        if flag.ndim != 0:
            raise ValueError('Distributed safety decision must be a scalar')
        dist = torch.distributed
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    def _finite_tensors(self, values):
        """Check every element while doing one host readback per boundary."""
        checks = [self.torch.isfinite(value.detach()).all() for value in values if value is not None]
        return self.torch.stack(checks).all() if checks else self.torch.ones((), dtype=self.torch.bool, device=self.device)

    def advance_policy(self, policy_step):
        if type(policy_step) is not int or policy_step != self.policy_step + 1:
            raise ValueError('World owner requires the next common Policy boundary')
        if self.active_update is not None or not self.publication_idle():
            raise RuntimeError('World cannot advance while update or publication is outstanding')
        self.policy_step = policy_step
        return self.state()

    def begin_update(self, policy_step, reward_cohort=None):
        if policy_step != self.policy_step or policy_step % 5:
            raise ValueError('Native World updates run only at each fifth Policy boundary')
        if self.global_steps != policy_step // 5 - 1 or self.active_update is not None:
            raise RuntimeError('Native World optimizer update counters diverged')
        if self.micro_steps % self.accumulation_steps or not self.publication_idle():
            raise RuntimeError('Native World update has unfinished gradients or IPC references')
        if self.native_reward is not None:
            self.last_reward_cohort = self.native_reward.set_cohort(reward_cohort)
            width = self.last_reward_cohort['frame_slice'][1]-self.last_reward_cohort['frame_slice'][0]
            if hasattr(self.backend,'training_window_frames'):
                expected_end = self.backend.training_window_frames
                expected_start = expected_end-self.backend.n_history-self.backend.horizon
                if list(self.last_reward_cohort['frame_slice']) != [expected_start,expected_end]:
                    raise ValueError('Reward cohort labels do not match the native selected history/future frames')
            selected=(self.last_reward_cohort['global_selected_frames'] if self.native_reward.dense else
                      self.last_reward_cohort['positive_count']+self.last_reward_cohort['negative_count'])
            if (selected !=
                    self.train_batch_size()*width):
                raise ValueError('Reward cohort does not cover the actual complete global World batch')
            dist = self.torch.distributed
            if dist.is_available() and dist.is_initialized():
                import hashlib,json
                fingerprint = hashlib.sha256(json.dumps(self.last_reward_cohort,sort_keys=True,
                    separators=(',',':')).encode()).hexdigest()
                gathered = [None]*dist.get_world_size()
                dist.all_gather_object(gathered,fingerprint)
                if len(set(gathered)) != 1:
                    raise RuntimeError('Four native World ranks received different RM cohort weights')
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        lr = self.peak_lr * (min(1., (self.global_steps + 1) / self.warmup_updates)
                             if self.warmup_updates else 1.)
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        self.active_update = {'policy_step': policy_step, 'micro': 0, 'losses': [],
                              'started': time.monotonic(), 'lr': lr,'rm_losses':[],
                              'rm_positive_augmented':0,'rm_negative_augmented':0,
                              'rm_valid_augmented':0,'rm_masked_augmented':0,
                              'rm_target_sum_augmented':0.,'rm_target_square_sum_augmented':0.,
                              'rm_active':bool(self.native_reward and self.native_reward.active)}
        return {'policy_step': policy_step, 'model_version_before': self.global_steps,
                'learning_rate': lr, 'expected_microbatches': self.accumulation_steps}

    def prepare_micro(self, windows):
        """Normalize only the present uint8 microbatch, in this interpreter."""
        torch = self.torch
        if len(windows) != self.micro_batch:
            raise ValueError('A streamed World microbatch must contain exactly micro windows')
        if getattr(self.backend, 'optimizations', {}).get('validate_on_cpu'):
            with torch.no_grad(), torch.autocast(self.device.type, dtype=torch.bfloat16):
                return self.backend.prepare_train_batch(windows)
        normalized = []
        for source in windows:
            row = dict(source)
            image = torch.as_tensor(row['obs'], device=self.device)
            if image.dtype == torch.uint8:
                image = image.float().div(255.).mul(2.).sub(1.)
            else:
                image = image.float()
            row['obs'] = image
            normalized.append(row)
        with torch.no_grad(), torch.autocast(self.device.type, dtype=torch.bfloat16):
            return self.backend.prepare_train_batch(normalized)

    def micro(self, windows):
        update = self.active_update
        if update is None or update['micro'] >= self.accumulation_steps:
            raise RuntimeError('World microbatch has no active native update')
        last = update['micro'] + 1 == self.accumulation_steps
        sync = (self.model.no_sync() if hasattr(self.model, 'no_sync') and not last else nullcontext())
        with sync:
            prepared = self.prepare_micro(windows)
            with self.torch.autocast(self.device.type, dtype=self.torch.bfloat16):
                result = self.model(prepared)
                loss = result[0] if isinstance(result, tuple) else result
            if not isinstance(loss, self.torch.Tensor) or loss.ndim != 0:
                raise ValueError('WEAVER must return its native scalar flow-matching loss')
            if not self._all_true(self.torch.isfinite(loss.detach())):
                raise FloatingPointError('Nonfinite native WEAVER loss; no optimizer update committed')
            (loss / self.accumulation_steps).backward()
        if self.native_reward is not None and self.native_reward.last_statistics is not None:
            statistics = self.native_reward.last_statistics
            if self.native_reward.dense:
                update['rm_losses'].append(statistics['native_half_mse_loss'])
                update['rm_valid_augmented'] += statistics['valid_targets_with_native_augmentation']
                update['rm_masked_augmented'] += statistics['masked_targets_with_native_augmentation']
                update['rm_target_sum_augmented'] += statistics['target_sum_with_native_augmentation']
                update['rm_target_square_sum_augmented'] += statistics['target_square_sum_with_native_augmentation']
            else:
                update['rm_losses'].append(statistics['native_half_brier_loss'])
                update['rm_positive_augmented'] += statistics['positive_labels_with_native_augmentation']
                update['rm_negative_augmented'] += statistics['negative_labels_with_native_augmentation']
        value = float(loss.detach().item())
        update['losses'].append(value)
        update['micro'] += 1
        self.micro_steps += 1
        if last:
            if self.native_reward is not None:
                augmented = ([update['rm_valid_augmented'],update['rm_masked_augmented']] if self.native_reward.dense else
                             [update['rm_positive_augmented'],update['rm_negative_augmented']])
                if any(count%2 for count in augmented):
                    raise RuntimeError('Native real/predicted RM label branches are not paired')
                counts = self.torch.tensor([number//2 for number in augmented],dtype=self.torch.int64,device=self.device)
                dist = self.torch.distributed
                initialized = dist.is_available() and dist.is_initialized()
                if initialized:
                    dist.all_reduce(counts,op=dist.ReduceOp.SUM)
                if initialized or self.world_size==1:
                    expected=([self.last_reward_cohort['valid_count'],self.last_reward_cohort['masked_count']]
                              if self.native_reward.dense else
                              [self.last_reward_cohort['positive_count'],self.last_reward_cohort['negative_count']])
                    if counts.cpu().tolist()!=expected:
                        raise RuntimeError('Actually consumed native RM labels differ from the global cohort class weights')
                if self.native_reward.dense:
                    moments=self.torch.tensor([update['rm_target_sum_augmented']/2.,
                        update['rm_target_square_sum_augmented']/2.],dtype=self.torch.float64,device=self.device)
                    if initialized:
                        dist.all_reduce(moments,op=dist.ReduceOp.SUM)
                    if initialized or self.world_size==1:
                        expected=[self.last_reward_cohort['target_sum'],self.last_reward_cohort['target_square_sum']]
                        if any(not math.isclose(actual,wanted,rel_tol=1e-9,abs_tol=1e-8)
                               for actual,wanted in zip(moments.cpu().tolist(),expected)):
                            raise RuntimeError('Actually consumed soft target moments differ from the fixed global cohort')
            if self.native_reward is not None and not self.native_reward.active:
                # A zero gradient would still advance Adam momentum and apply
                # weight decay. None makes this a genuine head-update skip.
                for parameter in self.native_reward.parameters:
                    parameter.grad = None
            batched_checks = getattr(self.backend, 'optimizations', {}).get('safety_checks') == 'batched'
            finite = (self._finite_tensors(p.grad for p in self.parameters) if batched_checks else
                      all(p.grad is None or self.torch.isfinite(p.grad).all().item() for p in self.parameters))
            if not self._all_true(finite):
                self.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError('Nonfinite native WEAVER gradients; no optimizer update committed')
            norm = self.torch.nn.utils.clip_grad_norm_(self.parameters, self.max_grad_norm,
                                                      error_if_nonfinite=True)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            finite_weights = (self._finite_tensors(self.parameters) if batched_checks else
                              all(self.torch.isfinite(p.detach()).all().item() for p in self.parameters))
            if not self._all_true(finite_weights):
                raise FloatingPointError('Nonfinite native World weights after AdamW; publication forbidden')
            self.global_steps += 1
            if self.native_reward is not None:
                self.rm_updates += int(self.native_reward.active)
                self.rm_disabled_updates += int(not self.native_reward.active)
                if self.native_reward.dense:
                    self.reward_target_valid_count += self.last_reward_cohort['valid_count']
                    self.reward_target_masked_count += self.last_reward_cohort['masked_count']
                    self.reward_target_sum += self.last_reward_cohort['target_sum']
                    self.reward_target_square_sum += self.last_reward_cohort['target_square_sum']
                else:
                    self.reward_positive_labels += self.last_reward_cohort['positive_count']
                    self.reward_negative_labels += self.last_reward_cohort['negative_count']
            self.last_result = {'successful_updates': self.global_steps, 'policy_step': self.policy_step,
                'loss': sum(update['losses']) / len(update['losses']), 'grad_norm': float(norm),
                'learning_rate': update['lr'], 'microbatches': update['micro'],
                'local_samples': self.micro_batch * self.accumulation_steps,
                'global_samples': self.train_batch_size(), 'skipped_steps': 0,
                'rm_updates':self.rm_updates,'rm_active':update['rm_active'],
                'update_wall_seconds': time.monotonic() - update['started']}
            self.last_result['native_half_mse_loss' if self.native_reward is not None and self.native_reward.dense
                             else 'native_half_brier_loss']=(sum(update['rm_losses'])/len(update['rm_losses'])
                                                           if update['rm_losses'] else None)
            self.active_update = None
        components = result[1] if isinstance(result, tuple) and len(result) > 1 and isinstance(result[1], dict) else {}
        logs = {key: float(item.detach().item()) if hasattr(item, 'detach') else float(item)
                for key, item in components.items() if isinstance(item, (int, float)) or
                (hasattr(item, 'numel') and item.numel() == 1)}
        return {'loss': value, 'native_losses': logs, 'micro_steps': self.micro_steps, 'successful_updates': self.global_steps,
                'committed': last, 'result': self.last_result if last else None}

    def state(self):
        dist = self.torch.distributed
        distributed_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        return {'backend': 'weaver', 'backend_engine': 'torch_ddp_native_flow_matching',
                'policy_steps': self.policy_step, 'ctrl_steps': self.global_steps,
                'global_steps': self.global_steps, 'skipped_steps': self.skipped_steps,
                'micro_steps': self.micro_steps, 'world_size': self.world_size,
                'distributed_world_size': distributed_size,
                'micro_batch': self.micro_batch, 'accumulation_steps': self.accumulation_steps,
                'global_batch': self.train_batch_size(), 'active_update': self.active_update is not None,
                'optimizer': 'AdamW', 'optimizer_betas': list(self.optimizer.defaults['betas']),
                'weight_decay': self.optimizer.defaults['weight_decay'], 'warmup_updates': self.warmup_updates,
                'max_grad_norm': self.max_grad_norm,
                'peak_learning_rate': self.peak_lr, 'post_warmup_schedule': 'constant',
                'trainable_parameter_count': sum(p.numel() for p in self.parameters),
                'trainable_scope':'wm+rm' if self.native_reward is not None else 'wm',
                'world_parameter_count':sum(p.numel() for name,p in self.module.named_parameters() if
                    name.startswith(('upstream.wm.','wm.')) and p.requires_grad),
                'rm_total_parameter_count':sum(p.numel() for p in self.native_reward.parameters) if self.native_reward else 0,
                'optimizer_parameter_tensors': len(self.parameters),
                'optimizer_state_entries': len(self.optimizer.state),
                'rm_updates':self.rm_updates,'rm_disabled_updates':self.rm_disabled_updates,
                'reward_positive_labels':self.reward_positive_labels,'reward_negative_labels':self.reward_negative_labels,
                'reward_target_valid_count':self.reward_target_valid_count,
                'reward_target_masked_count':self.reward_target_masked_count,
                'reward_target_sum':self.reward_target_sum,
                'reward_target_square_sum':self.reward_target_square_sum,
                'last_reward_cohort':self.last_reward_cohort,
                'native_reward_head':self.native_reward.report() if self.native_reward is not None else None,
                'gradients_released': all(p.grad is None for p in self.parameters),
                'parameter_dtype': str(self.parameters[0].dtype), 'autocast_dtype': 'torch.bfloat16'}
