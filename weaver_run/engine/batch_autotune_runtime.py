"""Opt-in model-free calibration adapters; never retry a partial CUDA update.

Only the driver changes the configuration, between full four-rank updates.
Critic inference yields between chunks; no extra Replay sampling or action RNG.
"""
from collections import deque
import asyncio
import math
from pathlib import Path
import time


class DeviceMemoryMeter:
    """Process activation growth plus sampled whole-device use, including peers.

    This is a conservative estimate from allocator peaks and device snapshots,
    not a proof against OOM or a continuous NVML physical-memory trace.
    """
    def __init__(self, cuda):
        self.cuda = cuda
        self.total = self.peak_used = self.activation = 0

    def begin_micro(self):
        self.base = int(self.cuda.memory_allocated())
        self.cuda.reset_peak_memory_stats()
        self._sample()

    def _sample(self):
        free, total = self.cuda.mem_get_info()
        # Account for our own peak reservations already released at this point.
        released = max(0, int(self.cuda.max_memory_reserved()) - int(self.cuda.memory_reserved()))
        self.total = int(total)
        self.peak_used = max(self.peak_used, int(total-free) + released)

    def end_micro(self):
        self.activation = max(self.activation, int(self.cuda.max_memory_allocated()) - self.base)
        self._sample()

    def report(self):
        return dict(total_bytes=self.total, peak_used_bytes=self.peak_used,
                    activation_bytes=max(0,self.activation))


def configure_training(owner, policy_step, micro):
    if (not getattr(owner, '_batch_autotune_enabled', False)
            or owner.training_role != 'policy' or owner.world_config.backend != 'model_free'):
        raise ValueError('Batch autotuning is restricted to opted-in model-free Policy owners')
    if owner._policy_joint_batch_mode.active is not None:
        raise RuntimeError('Cannot change a batch during an active update')
    if type(policy_step) is not int or policy_step != owner.global_step+1 or type(micro) is not int or micro not in (8,16,24):
        raise ValueError('Batch choice must name the next update and a supported micro')
    owner._autobatch_next = (policy_step,micro)
    owner._autobatch_train_report = None
    return dict(rank=owner.rank,policy_step=policy_step,micro=micro)


def begin_training_micro(owner, torch_module):
    meter = getattr(owner,'_autobatch_train_meter',None)
    if meter is None:
        meter = owner._autobatch_train_meter = DeviceMemoryMeter(torch_module.cuda)
    meter.begin_micro()


def end_training_micro(owner):
    owner._autobatch_train_meter.end_micro()


def finish_training(owner, seconds):
    step,micro = owner._autobatch_next
    owner._autobatch_train_report = dict(rank=owner.rank,policy_step=step,micro=micro,
        train_seconds=float(seconds),memory=owner._autobatch_train_meter.report())
    owner._autobatch_train_meter = None


def training_report(owner, policy_step):
    report = getattr(owner,'_autobatch_train_report',None)
    if not report or report['policy_step'] != policy_step or owner.global_step != policy_step:
        raise RuntimeError('Autotune statistics requested before the complete training update')
    return report


def configure_inference(owner, policy_step, refresh_micro):
    if (type(policy_step) is not int or policy_step != owner.capture_model_version+1
            or type(refresh_micro) is not int or refresh_micro not in (2,4,8)):
        raise ValueError('Critic batch must name the next update and supported refresh batch')
    if owner._policy_dispatch is not None or getattr(owner,'_autobatch_refresh_active',False):
        raise RuntimeError('Autotuning requires the model-free idle refresh boundary')
    owner._autobatch_inference = dict(rank=int(owner.actor_id),policy_step=policy_step,
        refresh_micro=refresh_micro,refresh_seconds=0.,refresh_observations=0,
        meter=None)
    if not hasattr(owner,'_autobatch_action_latencies'):
        owner._autobatch_action_latencies = deque(maxlen=4096)
    return dict(rank=int(owner.actor_id),policy_step=policy_step,refresh_micro=refresh_micro)


def action_timer(owner, metadata):
    stats = getattr(owner,'_autobatch_inference',None)
    if stats is None or not isinstance(metadata,dict) or metadata.get('workflow') != 'real_rollout':
        return None
    submitted = metadata.get('submitted_ns')
    return owner, (int(submitted) if type(submitted) is int and submitted > 0 else time.time_ns())


def finish_action(token):
    if token is not None:
        owner,submitted = token
        # A completion belongs to the next report, even when its request began
        # before the previous report/configuration boundary. Do not lose tails.
        owner._autobatch_action_latencies.append(max(0., (time.time_ns()-submitted)/1e9))


async def refresh_values(owner, inputs, expected_version, *, torch_module=None):
    owner._check_inference_health()
    stats = getattr(owner,'_autobatch_inference',None)
    if (stats is None or type(expected_version) is not int
            or expected_version != owner.capture_model_version
            or expected_version != stats['policy_step']-1
            or not isinstance(inputs,(list,tuple)) or not 1 <= len(inputs) <= 4096):
        raise ValueError('Autotuned refresh needs bounded observations and the exact published version')
    if getattr(owner,'_autobatch_refresh_active',False):
        raise RuntimeError('Only one critic cohort per replica may be refreshed')
    if torch_module is None:
        import torch as torch_module
    micro = stats['refresh_micro']
    meter = stats['meter']
    if meter is None:
        meter = stats['meter'] = DeviceMemoryMeter(torch_module.cuda)
    owner._autobatch_refresh_active = True
    started = time.perf_counter()
    values = []
    try:
        for offset in range(0,len(inputs),micro):
            owner._check_inference_health()
            if owner.capture_model_version != expected_version:
                raise RuntimeError('Critic version changed during autotuned refresh')
            meter.begin_micro()
            with torch_module.inference_mode():
                prepared = owner.model.prepare_inputs_batch(inputs[offset:offset+micro])
                _,value = owner.model(prepared)
                current = value.detach().float().reshape(-1).cpu().tolist()
            meter.end_micro()
            if len(current) != min(micro,len(inputs)-offset) or any(not math.isfinite(v) for v in current):
                raise ValueError('Critic refresh must return finite ordered per-observation Values')
            values.extend(current)
            # Release chunk tensors before letting the action loop use this model.
            del prepared,value,_
            await asyncio.sleep(0)
        if owner.capture_model_version != expected_version:
            raise RuntimeError('Critic version changed before refresh completion')
        elapsed = time.perf_counter()-started
        stats['refresh_seconds'] += elapsed
        stats['refresh_observations'] += len(values)
        owner.telemetry.event('fresh_critic_refresh',actual_batch=len(values),model_version=expected_version,
            details=dict(action_sampling=False,maximum_micro_batch=micro,batch_autotune=True,
                         cooperatively_scheduled=True,refresh_seconds=elapsed))
        return dict(policy_version=expected_version,values=values)
    finally:
        owner._autobatch_refresh_active = False


def inference_report(owner, policy_step):
    stats = getattr(owner,'_autobatch_inference',None)
    if stats is None or stats['policy_step'] != policy_step or getattr(owner,'_autobatch_refresh_active',False):
        raise RuntimeError('Critic report requested outside its completed update')
    latencies=list(owner._autobatch_action_latencies)
    owner._autobatch_action_latencies.clear()
    return {**{k:v for k,v in stats.items() if k != 'meter'},
            'action_latencies':latencies,
            'memory':None if stats['meter'] is None else stats['meter'].report()}


def aggregate_sample(policy_step, update_seconds, trainers, inference):
    for reports in (trainers,inference):
        if (len(reports) != 4 or sorted(x['rank'] for x in reports) != list(range(4))
                or any(x['policy_step'] != policy_step for x in reports)):
            raise ValueError('Autotune requires four complete same-update rank reports')
    if len({x['micro'] for x in trainers}) != 1 or len({x['refresh_micro'] for x in inference}) != 1:
        raise ValueError('Autotune ranks disagree on the measured configuration')
    infers = {x['rank']:x for x in inference}
    cards=[]
    for trainer in trainers:
        infer = infers[trainer['rank']]
        a,b = trainer['memory'],infer['memory']
        if a is None or b is None:
            cards=[];break  # Missing measurements refuse expansion in the controller.
        if a['total_bytes'] != b['total_bytes']:
            raise ValueError('Colocated trainer and inference device memory differs')
        cards.append(dict(rank=trainer['rank'],total_bytes=a['total_bytes'],
            peak_used_bytes=max(a['peak_used_bytes'],b['peak_used_bytes']),
            train_activation_bytes=a['activation_bytes'],refresh_activation_bytes=b['activation_bytes']))
    latencies=sorted(t for item in inference for t in item['action_latencies'])
    return dict(policy_step=policy_step,update_seconds=float(update_seconds),
        train_seconds=max(x['train_seconds'] for x in trainers),
        refresh_seconds=max(x['refresh_seconds'] for x in inference),
        refresh_observations=sum(x['refresh_observations'] for x in inference),
        action_p95_seconds=latencies[max(0,math.ceil(len(latencies)*.95)-1)] if latencies else None,
        cards=cards)


class BatchAutotuneRuntime:
    def __init__(self, *, ray, trainers, inference_pool, replay, writer, experiment, directory):
        from batch_autotune import BatchAutotuneConfig, BatchAutotuner
        self.controller = BatchAutotuner(BatchAutotuneConfig(enabled=True))
        self.ray,self.trainers,self.inference,self.replay = ray,trainers,inference_pool,replay
        self.writer,self.experiment,self.directory = writer,experiment,Path(directory)
        self.started = None

    def before_update(self, policy_step):
        chosen = self.controller.current
        # Finish all configuration RPCs before any collective-producing epoch.
        results = self.ray.get([actor.configure_batch_autotune.remote(policy_step,chosen['train_micro'])
                               for actor in self.trainers],timeout=30)
        if (sorted(x['rank'] for x in results) != list(range(4))
                or any((x['policy_step'],x['micro']) != (policy_step,chosen['train_micro']) for x in results)):
            raise RuntimeError('Four training ranks failed to acknowledge one batch choice')
        results = self.ray.get([actor.configure_batch_autotune.remote(policy_step,chosen['refresh_micro'])
                               for actor in self.inference],timeout=30)
        if (sorted(x['rank'] for x in results) != list(range(4))
                or any((x['policy_step'],x['refresh_micro']) != (policy_step,chosen['refresh_micro']) for x in results)):
            raise RuntimeError('Four critic replicas failed to acknowledge one batch choice')
        self.ray.get(self.replay.configure_batch_autotune.remote(policy_step),timeout=30)
        self.experiment.event('batch_autotune_applied',details=dict(policy_step=policy_step,**chosen))
        self.started = time.perf_counter()

    def after_update(self, policy_step):
        seconds = time.perf_counter()-self.started
        training = self.ray.get([x.get_batch_autotune_stats.remote(policy_step) for x in self.trainers],timeout=30)
        inference = self.ray.get([x.get_batch_autotune_stats.remote(policy_step) for x in self.inference],timeout=30)
        sample = aggregate_sample(policy_step,seconds,training,inference)
        chosen = dict(self.controller.current)
        decision = self.controller.observe(sample)
        receipt = dict(policy_step=policy_step,sample=sample,decision=decision,
                       measurement_scope='post_admission_to_weight_publication; validation and supply wait excluded')
        self.experiment.event('batch_autotune_observation',details=receipt)
        from frozen_policy_eval import atomic_json
        atomic_json(self.directory/'batch_autotune.json',dict(receipt,calibration=self.controller.summary()))
        values = dict(Train_Micro=chosen['train_micro'],Gradient_Accumulation=144//chosen['train_micro'],
            Refresh_Micro=chosen['refresh_micro'],Update_Seconds=seconds,
            Train_Seconds=sample['train_seconds'],Refresh_Seconds=sample['refresh_seconds'],
            Refresh_Observations=sample['refresh_observations'],Action_P95_Seconds=sample['action_p95_seconds'],
            Sampled_Peak_Memory_GiB=max((x['peak_used_bytes'] for x in sample['cards']),default=0)/2**30,
            Frozen=int(decision['frozen']))
        for name,value in values.items():
            if value is not None:self.writer.add_scalar('BatchAutotune/'+name,value,policy_step)
        self.writer.flush();self.experiment.flush()
        self.started = None
        return receipt
