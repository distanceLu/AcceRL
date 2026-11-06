"""Cooperative Ray adapter. Logical admission lanes are not SM partitions.

No CUDA/Ray import at module import. Queue time never becomes execution cost.
Rank stages are conservatively whole optimizer updates; independent inference
replicas share accounts but never become a gang. Existing submitted work drains.
"""
from __future__ import annotations

import asyncio
from collections import Counter, OrderedDict
from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import time

from cost_scheduler import CostModel, ServiceConfig, WorkItem, WeightedCostScheduler
from collective_phase_gate import BoundaryAttestation, PhaseSpec, GangPhaseGate
from cycle_service_controller import (CycleServiceController, ControllerConfig,
    CycleSnapshot, InventoryBucket, ResponseSample, SERVICES)
from cycle_feedback import CycleFeedbackConfig, CompleteCycleFeedback

CLASSES = ('real', 'imagined', 'refresh', 'eval')


async def finish_owned_rpc(task):
    """Finish an owned control RPC despite cancellation of its caller."""
    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                return task.result()


async def acquire_inference(broker, spec):
    """Keep ownership until admission is delivered or its unused grant released.

    Cancelling a local await need not cancel the remote Ray method. Shield that
    method, reconcile its reply, and release without claiming successful work.
    """
    task = asyncio.ensure_future(broker.acquire.remote(spec))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        receipt = await finish_owned_rpc(task)
        cleanup = asyncio.ensure_future(broker.complete.remote(receipt['work_id'], 0.,
            completed_units=0., success=False))
        await finish_owned_rpc(cleanup)
        raise


async def publish_after_refresh(owner, publish):
    """Close refresh admission, drain pinned jobs, then apply atomically."""
    if not hasattr(owner, '_policy_publication_lock'):
        owner._policy_publication_lock = asyncio.Lock()
    async with owner._policy_publication_lock:
        owner._policy_publication_pending = True
        try:
            while owner._policy_dispatch.refresh_pending:
                owner._check_inference_health()
                await asyncio.sleep(.0005)
            owner._policy_dispatch.assert_publication_allowed()
            return publish()
        finally:
            owner._policy_publication_pending = False


def load_settings(path=None):
    path = path or os.environ.get('ACCERL_COST_SCHEDULER_CONFIG')
    if not path:
        return {'enabled': False}
    raw = json.loads(Path(path).read_text(encoding='utf8'))
    allowed = {'schema', 'enabled', 'weights', 'class_weights', 'refresh_micro',
        'action_micro', 'max_pending', 'card_inflight', 'training_gangs',
        'macro_enabled', 'allow_probe', 'probe_min_cycles', 'controller', 'initial_unit_costs', 'execution',
        'cycle_feedback'}
    if set(raw)-allowed or raw.get('schema') != 1 or type(raw.get('enabled')) is not bool:
        raise ValueError('Unknown cost scheduler schema/options')
    for key, default, maximum in [('refresh_micro', 2, 2), ('action_micro', 2, 2),
            ('max_pending', 4096, 65536), ('card_inflight', 2, 8), ('training_gangs', 1, 2)]:
        value = raw.setdefault(key, default)
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError('Invalid bounded scheduler option: '+key)
    for key in ('macro_enabled', 'allow_probe'):
        if type(raw.setdefault(key, False)) is not bool:
            raise ValueError(key+' must be boolean')
    if type(raw.setdefault('probe_min_cycles',2)) is not int or not 2<=raw['probe_min_cycles']<=1000:
        raise ValueError('probe_min_cycles must be a bounded integer >=2')
    for key, names in [('weights', SERVICES), ('class_weights', CLASSES)]:
        weights = raw.setdefault(key, dict.fromkeys(names, 1.0))
        if set(weights) != set(names) or any(type(v) not in (int,float) or not math.isfinite(v) or v <= 0 for v in weights.values()):
            raise ValueError('Scheduler weights must be positive finite named budgets')
    from execution_control import execution_settings
    raw['execution'] = execution_settings(raw.get('execution'))
    ControllerConfig(**raw.get('controller', {}))
    CycleFeedbackConfig(**raw.get('cycle_feedback', {}))
    if raw['training_gangs'] == 2 and raw['card_inflight'] < 3:
        raise ValueError('Two training groups require a spare per-card inference lane')
    return raw


def source_fingerprint(root):
    """Use the Git revision as the shared identity in the 64-hex rank protocol."""
    import subprocess
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    return hashlib.sha256(("git-commit:" + commit).encode("ascii")).hexdigest()


class CostAdmissionBroker:
    def __init__(self, settings, run_id, source_sha256, *, clock=None, emit=None):
        self.settings, self.run_id, self.source_sha256 = settings, run_id, source_sha256
        from execution_control import execution_settings
        self.execution = execution_settings(settings.get('execution'))
        self.clock, self.emit = clock or time.monotonic, emit or (lambda *args,**kwargs:None)
        if emit is None and settings.get('telemetry'):
            from capture_telemetry import Telemetry
            self.telemetry=Telemetry('cost_scheduler','0')
            self.emit=self.telemetry.event
        self.weights = dict(settings.get('weights', dict.fromkeys(SERVICES,1.)))
        self.controller = CycleServiceController(self.weights, ControllerConfig(**settings.get('controller',{})))
        feedback_config = CycleFeedbackConfig(**settings.get('cycle_feedback', {}))
        self.cycle_feedback = CompleteCycleFeedback(feedback_config) if feedback_config.enabled else None
        if settings.get('training_gangs', 1) == 2 and settings.get('card_inflight', 2) < 3:
            raise ValueError('Two training groups require a spare per-card inference lane')
        limits = {'card:'+str(i):settings.get('card_inflight',2) for i in tuple(map(int, os.environ.get('ACCERL_ALLOCATED_GPU_IDS', '4,5,6,7').split(',')))}
        limits['training_lane'] = settings.get('training_gangs',1)
        self.scheduler = WeightedCostScheduler(cost_model=CostModel(settings.get('initial_unit_costs',{})),
            resource_limits=limits,max_pending=settings.get('max_pending',4096),max_inflight_total=16)
        for service in ('policy_train','world_train','world_infer'):
            self.scheduler.register_service(service,ServiceConfig(self.weights[service],4 if service=='world_infer' else 1))
        if self.execution['parent_policy_budget']:
            self.scheduler.register_service('policy_infer', ServiceConfig(self.weights['policy_infer'], 4))
        else:
            for kind, weight in self._child_weights().items():
                self.scheduler.register_service('policy_infer.'+kind,ServiceConfig(weight,4))
        self.waiters, self.gates, self.phase_records = {}, {}, {}
        self.active_items = {}
        self.refresh_remaining = 0
        self.refresh_completed = 0
        self.action_observations_completed = 0
        self.publication_blocked = False
        self.last_cycle = None
        self.stable_windows = 0
        self.skip_response_window = False
        self.last_decision = None
        self.latest_cycle_snapshot = None
        self.probe_anchor_cycle = None
        self.probe_windows_at_anchor = 0
        self.probe_contexts = set()
        self.active_probe = None
        self.last_probe_result = None
        self.unpublished_training = {}
        self.latest_admitted = {'real':0,'imagined':0}
        self.last_partial_response_window = None
        self.response_window_sequence = 0
        self.skip_response_reason = None
        self.latest_inventory = None
        self.inventory_revision = -1
        self.failed = None
        self.production = None
        self.production_revision = -1
        self.production_blocked_window = set()
        self.refresh_settlements = OrderedDict()

    def _child_weights(self):
        cw = self.settings.get('class_weights',dict.fromkeys(CLASSES,1.))
        return {k:self.weights['policy_infer']*cw[k]/sum(cw.values()) for k in CLASSES}

    def _scheduler_weights(self):
        weights = {k:self.weights[k] for k in ('policy_train','world_train','world_infer')}
        if self.execution['parent_policy_budget']:
            weights['policy_infer'] = self.weights['policy_infer']
        else:
            weights.update({'policy_infer.'+k:v for k,v in self._child_weights().items()})
        return weights

    def _pump(self):
        while self.failed is None:
            self.scheduler.reserve_pending_group(
                (wid for wid, record in self.phase_records.items() if record['enqueued']),
                gate_resource='training_lane')
            grant = self.scheduler.claim_next()
            if grant is None:
                return
            wid = grant.work_id
            self.active_items[wid]=grant.item
            record = self.phase_records.get(wid)
            result = {'work_id':wid,'estimated_cost':grant.estimated_cost,
                'dispatch_weight':grant.dispatch_weight,'cost_unit':grant.cost_unit}
            if record is not None:
                record['group_grant'] = self.gates[record['spec'].group_id].grant(record['spec'])
                record['admitted_at'] = self.clock()
                result['group_grant'] = asdict(record['group_grant'])
                peers = [x for x in self.active_items.values()
                         if x.work_id != wid and x.service in ('policy_train','world_train')]
                self.emit('cost_training_execution_admitted', details={
                    'work_id':wid, 'service':grant.service,
                    'policy_version':record['spec'].policy_version,
                    'world_version':record['spec'].world_version,
                    'coadmitted_training': [x.work_id for x in peers],
                    'training_gangs':self.settings.get('training_gangs',1),
                    'scope':'coexisting whole-update submission leases; not CUDA kernel overlap'})
            future = self.waiters[wid]
            if not future.done():future.set_result(result)
            self.emit('cost_work_admitted',details={**result,'service':grant.service})

    async def acquire(self, spec):
        if self.failed is not None:raise RuntimeError(self.failed)
        item = WorkItem(**spec)
        if item.service not in ('world_infer',)+tuple('policy_infer.'+x for x in CLASSES):
            raise ValueError('Training requires collective phase admission')
        if item.metadata.get('bounded_refresh_lease'):
            if (item.service != 'policy_infer.refresh' or self.execution['refresh_lease_micros'] <= 1
                    or item.units > 2*self.execution['refresh_lease_micros']):
                raise ValueError('Bounded refresh grants cannot be used for actions or unbounded work')
        if item.replica is None or len(item.resources)!=1 or item.resources[0] not in {'card:'+str(i) for i in tuple(map(int, os.environ.get('ACCERL_ALLOCATED_GPU_IDS', '4,5,6,7').split(',')))}:
            raise ValueError('Independent inference needs one exact replica and card lane')
        if self.execution['parent_policy_budget'] and item.service.startswith('policy_infer.'):
            item = replace(item, service='policy_infer', metadata={**item.metadata,
                'request_class':item.service.split('.')[1]})
        future = asyncio.get_running_loop().create_future()
        self.scheduler.enqueue(item);self.waiters[item.work_id]=future;self._pump()
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            # No receipt has been delivered by this coroutine. Even if a
            # grant raced with cancellation, the caller cannot have submitted
            # GPU work under that receipt yet.
            if item.work_id in self.active_items:
                await self.complete(item.work_id, 0., completed_units=0., success=False)
            else:
                self.scheduler.cancel(item.work_id)
                self.scheduler.retire_completed(item.work_id)
                self.waiters.pop(item.work_id, None)
                future.cancel()
                self._pump()
            raise

    async def complete(self, work_id, observed_cost, *, completed_units=None, success=True):
        # Capture the class before retirement. This is service progress, never
        # used as a surrogate for admitted Replay chunks.
        item = self.active_items.get(work_id)
        if not success:
            completed_units=0.
        result = self.scheduler.complete(work_id,observed_cost,completed_units=completed_units,
            provenance='execution_wall_proxy',success=success,
            allow_partial=bool(item and item.metadata.get('bounded_refresh_lease')))
        service = (('policy_infer.'+item.metadata['request_class']) if item is not None
            and item.service == 'policy_infer' else None if item is None else item.service)
        result['request_service'] = service
        if success and service in ('policy_infer.real','policy_infer.imagined'):
            self.action_observations_completed += completed_units if completed_units is not None else item.units
        if success and service == 'policy_infer.refresh':
            self.refresh_completed += result['completed_units']
        self.waiters.pop(work_id,None)
        self.active_items.pop(work_id,None)
        if work_id in self.phase_records:
            del self.phase_records[work_id]
        if hasattr(self.scheduler,'retire_completed'):
            self.scheduler.retire_completed(work_id)
        self._pump()
        self.emit('cost_work_completed',details=result)
        return result

    async def settle_refresh(self, work_id, observed_cost, completed_units):
        """Atomic bounded block settlement plus demand progress, idempotent."""
        signature = (observed_cost, completed_units)
        if work_id in self.refresh_settlements:
            old_signature, receipt = self.refresh_settlements[work_id]
            if signature != old_signature:
                raise ValueError('Refresh settlement retry differs')
            return receipt
        item = self.active_items.get(work_id)
        if (item is None or not item.metadata.get('bounded_refresh_lease')
                or type(completed_units) is not int or not 0 < completed_units <= item.units
                or completed_units > self.refresh_remaining):
            raise ValueError('Invalid refresh lease settlement')
        receipt = await self.complete(work_id, observed_cost, completed_units=completed_units)
        await self.refresh_demand(-completed_units)
        self.refresh_settlements[work_id] = (signature, receipt)
        while len(self.refresh_settlements) > 256:
            self.refresh_settlements.popitem(last=False)
        return receipt

    async def update_production(self, raw):
        if not self.execution['authoritative_production']:
            return False
        raw = dict(raw)
        if (type(raw.get('revision')) is not int or raw['revision'] < 0
                or any(type(raw.get(k)) is not bool for k in ('real_production_ready','imagination_ready'))
                or len(raw.get('versions', ())) != 2
                or any(type(v) is not int or v < 0 for v in raw['versions'])):
            raise ValueError('Production needs canonical authority revision, versions and readiness')
        if raw['revision'] <= self.production_revision:
            return False
        self.production_revision = raw['revision']
        self.production = raw
        for field, progress in [('real_production_ready','real_chunks'),('imagination_ready','imagined_chunks')]:
            if not raw[field]:
                self.production_blocked_window.add(progress)
        self.emit('cost_production_authority', details=raw)
        return True

    async def refresh_demand(self, delta):
        if type(delta) is not int or self.refresh_remaining+delta<0:
            raise ValueError('Refresh progress cannot underflow')
        previous=self.refresh_remaining
        self.refresh_remaining += delta
        if (previous==0)!=(self.refresh_remaining==0):
            self._reconsider_phase('refresh_started' if self.refresh_remaining else 'refresh_finished',phase='refresh')
        return self.refresh_remaining

    async def acquire_training(self, rank, raw):
        if self.failed is not None:raise RuntimeError(self.failed)
        group_id = raw.get('group_id') if isinstance(raw,dict) else None
        service = raw.get('service') if isinstance(raw,dict) else None
        existing_group = next((gid for gid,gate in self.gates.items() if gate.service==service),None)
        try:
            raw = dict(raw);att = BoundaryAttestation(**raw.pop('attestation'))
            if att.source_sha256 != self.source_sha256 or att.world_size != 4:
                raise ValueError('Collective source/rank identity differs from this candidate')
            spec = PhaseSpec(attestation=att,**raw)
            expected = {'policy_train':'policy_zero2','world_train':'native_ddp'}
            if expected.get(spec.service)!=att.backend:
                raise ValueError('No reviewed service/backend adapter')
            if existing_group is not None and existing_group!=spec.group_id:
                raise ValueError('A communicating service cannot change its exact rank group')
            gate = self.gates.setdefault(spec.group_id,GangPhaseGate(spec.group_id,spec.service,tuple(range(4))))
            gate.arrive(rank,spec,previous_collective_complete=True,own_stream_complete=True)
            wid = f'{spec.service}:{spec.group_id}:{spec.sequence}'
            if wid not in self.waiters:
                self.waiters[wid]=asyncio.get_running_loop().create_future()
                self.phase_records[wid]={'spec':spec,'enqueued':False}
            record=self.phase_records[wid]
            if gate.ready and not record['enqueued']:
                if self.publication_blocked:
                    raise ValueError('Cannot admit a training stage during weight publication')
                # Distinct communicators may overlap only at the same published
                # model boundary. Every group's own four-rank protocol remains.
                if self.settings.get('training_gangs',1) == 2:
                    for other_id, other in self.phase_records.items():
                        if other_id == wid or not other['enqueued']:
                            continue
                        peer = other['spec']
                        if ((peer.policy_version, peer.world_version) != (spec.policy_version, spec.world_version)
                                or peer.service == spec.service):
                            raise ValueError('Concurrent training groups require distinct services at one version boundary')
                self._reconsider_phase(spec.service+'_ranks_ready',
                    phase='train' if spec.service=='policy_train' else None)
                record['enqueued']=True
                cost_key=(f'{spec.service}:{att.backend}:micro{att.micro_batch}'
                          f':gas{att.accumulation_steps}:global{att.global_batch}')
                self.scheduler.enqueue(WorkItem(wid,spec.service,cost_key,
                    units=1.,resources=tuple('card:'+str(i) for i in tuple(map(int, os.environ.get('ACCERL_ALLOCATED_GPU_IDS', '4,5,6,7').split(','))))+('training_lane',),
                    metadata={'phase':asdict(spec),'cost_scope':'sum_rank_execution_wall_proxy'}))
                self._pump()
            receipt=await asyncio.shield(self.waiters[wid])
            gate.start(rank,record['group_grant'])
            return receipt
        except BaseException as error:
            # A contradictory fourth rank must wake the three ranks waiting for
            # its boundary ACK. Failing the gate never revokes an already
            # committed grant: those ranks still join and drain communication.
            failed_group = existing_group or group_id
            if failed_group in self.gates:
                await self.fail_training(failed_group,repr(error))
            raise

    async def complete_training(self, rank, work_id, execution_seconds):
        record=self.phase_records[work_id];gate=self.gates[record['spec'].group_id]
        completion=gate.complete(rank,record['group_grant'],max(float(execution_seconds),1e-9),
            collective_complete=True,own_stream_complete=True,cost_source='execution_wall_proxy')
        if completion is not None:
            # A completed gang is not yet a published version boundary.
            self.unpublished_training[record['spec'].service]=record['spec'].policy_version+1
            await self.complete(work_id,completion.actual_cost,completed_units=1.)
            return {'group_complete':True,'actual_cost':completion.actual_cost}
        return {'group_complete':False}

    async def fail_training(self, group_id, reason):
        # Never revoke a granted cohort: already granted ranks still join/drain.
        self.failed='collective phase failed: '+str(reason)
        if group_id in self.gates:self.gates[group_id].fail(str(reason))
        for wid,future in self.waiters.items():
            if not future.done():future.set_exception(RuntimeError(self.failed))
        return {'new_admission_blocked':True,'submitted_work_revoked':False}

    def _window_progress(self, snapshot, admitted):
        last=self.last_cycle
        return {'policy_training':snapshot.policy_version-last['policy'],
            'world_training':snapshot.world_version-last['world'],
            'critic_observations':self.refresh_completed-last['refresh'],
            'policy_action_observations':self.action_observations_completed-last['actions'],
            'real_chunks':admitted.get('real',0)-last['admitted'].get('real',0),
            'imagined_chunks':admitted.get('imagined',0)-last['admitted'].get('imagined',0)}

    def _record_partial_window(self, snapshot, admitted, reason):
        """Preserve progress without fitting a mixed-window throughput rate."""
        if self.last_cycle is None:return
        progress=self._window_progress(snapshot,admitted)
        if any(v<0 for v in progress.values()):
            raise ValueError('Partial response counters cannot move backwards')
        self.response_window_sequence+=1
        self.last_partial_response_window={
            'sequence':self.response_window_sequence,'started_at':self.last_cycle['at'],
            'finished_at':snapshot.now,'weights':dict(self.last_cycle['weights']),
            'context':self.last_cycle['context'],
            'start_versions':[self.last_cycle['policy'],self.last_cycle['world']],
            'end_versions':[snapshot.policy_version,snapshot.world_version],
            'progress':progress,'reason':reason,'valid_response_sample':False}
        self.emit('cost_response_window_partial',details=self.last_partial_response_window)

    def _expire_probe_if_safe(self, snapshot, admitted, *, inventory_versions):
        """Rollback at an owned control boundary; never revoke a granted gang.

        Authoritative inventory notifications keep this deadline active while
        Replay is backpressured. A finished but unpublished optimizer update
        still belongs to the old measurement vector and defers the rollback.
        """
        probe = self.cycle_feedback.active_probe if self.cycle_feedback is not None else self.active_probe
        if probe is None or snapshot.now < probe['deadline_at']:return False
        blockers=[]
        if self.publication_blocked:blockers.append('publication')
        if self.unpublished_training:blockers.append('training_not_published')
        if any(item.service in ('policy_train','world_train') for item in self.active_items.values()):
            blockers.append('training_grant_inflight')
        if tuple(inventory_versions)!=(snapshot.policy_version,snapshot.world_version):
            blockers.append('inventory_versions_not_published')
        if blockers:
            probe['timeout_pending']=True;probe['timeout_blockers']=blockers
            return False
        if self.cycle_feedback is not None:
            decision = self.cycle_feedback.rollback(snapshot, self.weights, self.controller,
                                                    'cycle_probe_rollback_timeout')
            self._apply_cycle_feedback(decision, snapshot, admitted)
            return True
        self._record_partial_window(snapshot,admitted,'probe_deadline_before_complete_window')
        baseline=dict(probe['baseline_weights']);candidate=dict(self.weights)
        self.weights=baseline;self.controller.weights=dict(baseline)
        self.controller.last_change_at=snapshot.now
        self.scheduler.set_weights(self._scheduler_weights())
        self.last_probe_result={'baseline_weights':baseline,'candidate_weights':candidate,
            'reason':'probe_rollback_timeout','accepted':False,
            'started_at':probe['started_at'],'deadline_at':probe['deadline_at'],
            'completed_at':snapshot.now,'complete_observation_window':False,
            'partial_window_sequence':self.response_window_sequence}
        self.active_probe=None
        self.probe_anchor_cycle=(snapshot.policy_version+4)//5
        self.probe_windows_at_anchor=self.stable_windows
        self.probe_contexts.clear()
        # Rebase cumulative counters, never clear them. Record the remainder
        # separately too: neither half is a full stable-weight update window.
        self.last_cycle={'at':snapshot.now,'policy':snapshot.policy_version,'world':snapshot.world_version,
            'refresh':self.refresh_completed,'actions':self.action_observations_completed,
            'admitted':dict(admitted),'weights':dict(self.weights),'context':snapshot.context}
        self.skip_response_window=True
        self.skip_response_reason='probe_timeout_interrupted_update_remainder'
        self.emit('cost_probe_result',details=self.last_probe_result)
        return True

    def _choose_cycle(self, snapshot, *, published=True, permit_changes=True):
        if self.cycle_feedback is not None:
            safe = (self.failed is None and published and not self.publication_blocked and not self.unpublished_training
                    and not any(x.service in ('policy_train','world_train') for x in self.active_items.values()))
            decision = self.cycle_feedback.decide(snapshot, self.weights, self.controller,
                permit_changes=bool(permit_changes and self.settings.get('macro_enabled')),
                allow_probe=bool(self.settings.get('allow_probe')), safe_boundary=safe)
            self._apply_cycle_feedback(decision, snapshot, self.latest_admitted)
            return decision
        prior_controller_weights=dict(self.controller.weights)
        prior_change_at=self.controller.last_change_at
        apply_enabled=bool(self.settings.get('macro_enabled') and permit_changes)
        if self.probe_anchor_cycle is None:
            # Ignore a partial first cycle. Count only complete synchronized
            # 5:1 cycles observed at one unchanged admission weight vector.
            self.probe_anchor_cycle=(snapshot.policy_version+4)//5
            self.probe_windows_at_anchor=self.stable_windows
        synchronized_cycle=min(snapshot.policy_version//5,snapshot.world_version)
        complete_probe_window=(synchronized_cycle-self.probe_anchor_cycle >= self.settings.get('probe_min_cycles',2)
            and self.stable_windows-self.probe_windows_at_anchor>=2 and 'joint' in self.probe_contexts
            and snapshot.policy_version % 5 == 0 and snapshot.world_version == snapshot.policy_version//5)
        # A probe must be settled before its result becomes another baseline.
        # Reaching the observation boundary with expired/missing evidence rolls
        # back; it must not wait forever for observations under an inactive vector.
        if self.active_probe is not None and apply_enabled and published and complete_probe_window:
            decision=self.controller.assess_probe(snapshot,self.active_probe['baseline_weights'])
            self.last_probe_result={**decision.probe_comparison,'reason':decision.reason,
                'started_at':self.active_probe['started_at'],'completed_at':snapshot.now,
                'completed_cycle':synchronized_cycle}
            self.active_probe=None
            self.weights=dict(decision.weights)
            self.scheduler.set_weights(self._scheduler_weights())
            self.probe_anchor_cycle=synchronized_cycle
            self.probe_windows_at_anchor=self.stable_windows
            self.probe_contexts.clear()
            return decision
        probe_ready=(complete_probe_window
            and all(self.controller._rate(snapshot,self.weights,service,progress,context=context) is not None
                for context,service,progress in (
                    ('ordinary','policy_train','policy_training'),
                    ('joint','policy_train','policy_training'),
                    ('joint','world_train','world_training'))))
        if probe_ready:
            probe_ready=self.controller.forecast(snapshot,self.weights)[0] is not None
        holding_probe=self.active_probe is not None or (self.probe_windows_at_anchor>0 and not probe_ready)
        apply_enabled=apply_enabled and not holding_probe
        decision=self.controller.choose(snapshot,allow_probe=bool(apply_enabled and
            self.settings.get('allow_probe') and probe_ready and published))
        if not apply_enabled:
            self.controller.weights=prior_controller_weights
            self.controller.last_change_at=prior_change_at
            if decision.changed:
                decision=replace(decision,weights=dict(self.weights),changed=False,
                    reason='macro_disabled_advisory' if not self.settings.get('macro_enabled')
                           else 'probe_observation_hold' if holding_probe and permit_changes
                           else 'phase_advisory_until_measurement_closed',
                    predicted_cycle_seconds=None,score=None,critical_path=None,
                    exploration_candidate=dict(decision.weights))
        if apply_enabled and decision.changed:
            if decision.reason=='bounded_probe_without_speed_prediction':
                self.active_probe={'baseline_weights':dict(self.weights),
                    'candidate_weights':dict(decision.weights),'started_at':snapshot.now,
                    'deadline_at':snapshot.now+self.controller.config.response_max_age_seconds,
                    'baseline_seconds_at_start':decision.baseline_cycle_seconds}
            self.weights=dict(decision.weights)
            self.scheduler.set_weights(self._scheduler_weights())
            self.probe_anchor_cycle=(snapshot.policy_version+4)//5
            self.probe_windows_at_anchor=self.stable_windows
            self.probe_contexts.clear()
        return decision

    def _apply_cycle_feedback(self, decision, snapshot, admitted):
        if decision.changed:
            previous = dict(self.weights)
            self.weights = dict(decision.weights)
            self.controller.weights = dict(self.weights)
            self.controller.last_change_at = snapshot.now
            self.scheduler.set_weights(self._scheduler_weights())
            self.emit('cost_weights_applied', details={'previous':previous,
                'weights':dict(self.weights),'reason':decision.reason,
                'policy_version':snapshot.policy_version,'world_version':snapshot.world_version,
                'scope':'future cooperative admission; submitted grants keep their dispatch weight'})
        if decision.changed or decision.probe_comparison is not None:
            self.cycle_feedback.rebase(snapshot, self.weights, admitted, self.controller)
            if decision.reason == 'cycle_probe_rollback_timeout':
                # Even p%5==0 may be mid-wait for the next update. Exclude that
                # remainder; anchor only at the next fully published boundary.
                self.cycle_feedback.anchor = None
        if decision.probe_comparison is not None:
            self.last_probe_result = decision.probe_comparison
            self.emit('cost_probe_result', details=decision.probe_comparison)

    def _reconsider_phase(self, trigger, *, phase=None, published=True,
                          policy_version=None, world_version=None, permit_changes=False):
        """Replan once per dependency edge; never reset progress windows.

        Callers do not use this on every microbatch. The cached inventory is an
        advisory projection; Replay remains the authority for actual draws.
        """
        prior=self.latest_cycle_snapshot
        if prior is None:
            return None
        updates={'now':self.clock(),'critic_observations_remaining':self.refresh_remaining,
                 'publication_blocked':self.publication_blocked}
        if phase is not None:updates['phase']=phase
        if policy_version is not None:
            updates.update(policy_version=policy_version,world_version=world_version,
                remaining_policy_updates=None,policy_training_units_remaining=1.,
                world_training_units_remaining=float((policy_version+1)%5==0),
                context='joint' if (policy_version+1)%5==0 else 'ordinary')
        versions=(updates.get('policy_version',prior.policy_version),
                  updates.get('world_version',prior.world_version))
        if self.latest_inventory is not None and self.latest_inventory['versions']==versions:
            updates['inventory']=self.latest_inventory['inventory']
        if self.execution['authoritative_production']:
            matched = self.production is not None and tuple(self.production['versions']) == versions
            for field in ('real_production_ready','imagination_ready'):
                updates[field] = bool(matched and self.production[field])
        snapshot=replace(prior,**updates)
        self.latest_cycle_snapshot=snapshot
        # Phase edges are advisory. Changes use update_cycle(), or a bounded
        # probe timeout reconciled by authoritative update_inventory().
        decision=self._choose_cycle(snapshot,published=published,permit_changes=False)
        self.last_decision={**asdict(decision),'trigger':trigger,'phase':snapshot.phase,
            'current_critic_observations_remaining':snapshot.critic_observations_remaining,
            'publication_blocked':snapshot.publication_blocked,
            'inventory_scope':'last read-only Replay snapshot; actual draws remain Replay-validated'}
        self.emit('cost_cycle_decision',details=self.last_decision)
        return self.last_decision

    async def update_inventory(self, raw):
        """Accept monotonic Replay reserve/commit/cancel snapshots, read-only."""
        revision=raw['inventory_revision']
        versions=tuple(raw['inventory_versions'])
        if type(revision) is not int or revision<0 or len(versions)!=2 or any(type(v) is not int or v<0 for v in versions):
            raise ValueError('Inventory needs a canonical revision and version pair')
        refresh_owner=raw.get('critic_refresh_owner',getattr(self,'critic_refresh_owner','inference'))
        if refresh_owner not in ('inference','trainer'):raise ValueError('Unknown critic owner')
        inventory=tuple(InventoryBucket(**x) for x in raw['inventory'])
        admitted=raw.get('admitted_chunks')
        if admitted is not None and (set(admitted)-{'real','imagined'} or
                any(type(v) is not int or v<0 for v in admitted.values())):
            raise ValueError('Production progress requires exact nonnegative counters')
        if revision<=self.inventory_revision:
            return False
        self.inventory_revision=revision
        self.critic_refresh_owner=refresh_owner
        self.latest_inventory={'versions':versions,'inventory':inventory}
        if admitted is not None:self.latest_admitted=dict(admitted)
        prior=self.latest_cycle_snapshot
        if prior is not None and versions==(prior.policy_version,prior.world_version):
            self.latest_cycle_snapshot=replace(prior,inventory=inventory)
        if self.latest_cycle_snapshot is not None and admitted is not None:
            snapshot=replace(self.latest_cycle_snapshot,now=self.clock(),
                critic_observations_remaining=self.refresh_remaining,
                publication_blocked=self.publication_blocked)
            self._expire_probe_if_safe(snapshot,admitted,inventory_versions=versions)
        return True

    async def update_cycle(self, raw):
        now=self.clock();raw=dict(raw);admitted=raw.pop('admitted_chunks',{})
        refresh_owner=raw.pop('critic_refresh_owner',getattr(self,'critic_refresh_owner','inference'))
        revision=raw.pop('inventory_revision',None)
        versions=raw.pop('inventory_versions',None)
        if revision is not None:
            if tuple(versions or ())!=(raw['policy_version'],raw['world_version']):
                raise ValueError('Decision inventory differs from the published versions')
            await self.update_inventory(dict(inventory_revision=revision,
                inventory_versions=versions,inventory=raw.get('inventory',()),admitted_chunks=admitted,
                critic_refresh_owner=refresh_owner))
        published=raw.pop('published',True)
        completed_context=raw.pop('completed_context',raw.get('context','ordinary'))
        if self.last_cycle is not None:
            delta=raw['policy_version']-self.last_cycle['policy']
            if delta>0:
                raw.setdefault('critic_observations_per_future_update',
                    (self.refresh_completed-self.last_cycle['refresh'])/delta)
        if refresh_owner == 'trainer':
            if self.refresh_remaining:raise RuntimeError('Trainer JIT cannot have inference refresh debt')
            raw['critic_observations_per_future_update']=0.
        raw['inventory']=tuple(InventoryBucket(**x) for x in raw.get('inventory',()))
        raw['now']=now;raw['critic_observations_remaining']=self.refresh_remaining
        if self.execution['authoritative_production']:
            production=self.production
            matching=production is not None and tuple(production['versions'])==(raw['policy_version'],raw['world_version'])
            for key in ('real_production_ready','imagination_ready'):
                raw[key]=bool(matching and production[key])
            if not matching:self.production_blocked_window.update(('real_chunks','imagined_chunks'))
        snapshot=CycleSnapshot(**raw)
        self.latest_admitted=dict(admitted)
        self._expire_probe_if_safe(snapshot,admitted,
            inventory_versions=(snapshot.policy_version,snapshot.world_version))
        if self.last_cycle is not None:
            last=self.last_cycle;elapsed=now-last['at']
            if (elapsed>0 and not self.skip_response_window and not self.publication_blocked
                    and completed_context==last['context'] and last['weights']==self.weights):
                progress=self._window_progress(snapshot,admitted)
                service={'policy_training':'policy_train','world_training':'world_train',
                    'critic_observations':'policy_infer','real_chunks':'policy_infer','imagined_chunks':'world_infer'}
                service['policy_action_observations']='policy_infer'
                for kind,units in progress.items():
                    if self.execution['authoritative_production'] and kind in self.production_blocked_window:
                        continue
                    if units<0:raise ValueError('Service progress counters cannot move backwards')
                    self.controller.observe(ResponseSample(last['weights'],service[kind],units,elapsed,
                        context=last['context'],progress_kind=kind,observed_at=now))
                self.stable_windows+=1
                if progress['world_training']>0 and completed_context=='joint':
                    self.probe_contexts.add('joint')
                elif completed_context=='ordinary':
                    self.probe_contexts.add('ordinary')
            elif self.skip_response_window and self.skip_response_reason:
                self._record_partial_window(snapshot,admitted,self.skip_response_reason)
            self.skip_response_window=False
            self.skip_response_reason=None
        self.production_blocked_window.clear()
        if self.execution['authoritative_production']:
            for field,kind in [('real_production_ready','real_chunks'),('imagination_ready','imagined_chunks')]:
                if not getattr(snapshot,field):self.production_blocked_window.add(kind)
        self.latest_cycle_snapshot=snapshot
        if self.cycle_feedback is not None and published and not self.publication_blocked:
            measured = self.cycle_feedback.observe(snapshot, self.weights, admitted, self.controller)
            if measured is not None:
                self.emit('cost_complete_cycle_measured', details=measured)
        decision=self._choose_cycle(snapshot,published=published)
        self.last_cycle={'at':now,'policy':snapshot.policy_version,'world':snapshot.world_version,
            'refresh':self.refresh_completed,'actions':self.action_observations_completed,
            'admitted':dict(admitted),'weights':dict(self.weights),'context':snapshot.context}
        self.last_decision=asdict(decision)
        self.emit('cost_cycle_decision',details=self.last_decision)
        return self.last_decision

    async def publication(self, blocked, *, policy_version=None, world_version=None):
        if type(blocked) is not bool:raise ValueError('Publication state must be boolean')
        if (policy_version is None)!=(world_version is None):
            raise ValueError('Publication requires both Policy and World versions')
        if policy_version is not None:
            if (type(policy_version) is not int or type(world_version) is not int or
                    policy_version<0 or world_version!=policy_version//5):
                raise ValueError('Publication versions must be a synchronized 5:1 boundary')
            prior=self.latest_cycle_snapshot
            if prior is not None and (policy_version<prior.policy_version or world_version<prior.world_version):
                raise ValueError('Published versions cannot move backwards')
        changed=self.publication_blocked!=blocked
        self.publication_blocked=blocked
        if not blocked and policy_version is not None:
            self.unpublished_training={service:target for service,target in self.unpublished_training.items()
                                       if target>policy_version}
        if changed:
            self._reconsider_phase('publication_blocked' if blocked else 'publication_released',
                phase='publish' if blocked or policy_version is None else 'needs_batch',
                published=not blocked,policy_version=policy_version if not blocked else None,
                world_version=world_version if not blocked else None,
                permit_changes=False)
        self.emit('cost_publication_dependency',details={'blocked':blocked})
        return blocked

    def snapshot(self):
        return {'run_id':self.run_id,'source_sha256':self.source_sha256,
            'cycle_feedback':None if self.cycle_feedback is None else self.cycle_feedback.snapshot(),
            'execution':dict(self.execution),'production_authority':self.production,
            'weights':dict(self.weights),'failed':self.failed,'refresh_remaining':self.refresh_remaining,
            'scheduler':self.scheduler.snapshot(),'controller':self.controller.snapshot(),
            'phase_gates':{k:v.snapshot() for k,v in self.gates.items()},
            'last_decision':self.last_decision,'cost_provenance':'execution_wall_proxy',
            'probe_hold':{'anchor_cycle':self.probe_anchor_cycle,
                'minimum_complete_cycles':self.settings.get('probe_min_cycles',2),
                'stable_windows_since_change':self.stable_windows-self.probe_windows_at_anchor,
                'observed_contexts':sorted(self.probe_contexts),
                'active_probe':self.active_probe,'last_result':self.last_probe_result,
                'unpublished_training':dict(self.unpublished_training),
                'last_partial_response_window':self.last_partial_response_window},
            'gpu_partitioning':False,'global_gpu_drain':False}


class TrainingAdmissionClient:
    def __init__(self, broker, settings, source_sha256, *, ray_module=None):
        self.broker,self.settings,self.source_sha256=broker,settings,source_sha256
        if ray_module is None:import ray as ray_module
        self.ray=ray_module

    def phase(self, service, rank, policy_step, micro, gas, *, world_version=None):
        if service not in ('policy_train','world_train'):
            raise ValueError('No reviewed training service backend')
        if type(rank) is not int or not 0<=rank<4:
            raise ValueError('Training rank must be one exact rank0..3')
        if any(type(x) is not int or x<1 for x in (policy_step,micro,gas)):
            raise ValueError('Training step/micro/GAS must be canonical positive integers')
        if service=='world_train' and policy_step%5:
            raise ValueError('World update is only ready at a fifth Policy boundary')
        expected_world=(policy_step-1)//5
        if world_version is not None and (type(world_version) is not int or world_version!=expected_world):
            raise ValueError('Training phase must bind the preceding published World version')
        if micro*gas*4 != (576 if service=='policy_train' else 1024):
            raise ValueError('Training service global batch changed')
        if not isinstance(self.settings.get('run_id'),str) or not self.settings['run_id']:
            raise ValueError('Training phase requires its exact run identity')
        backend='policy_zero2' if service=='policy_train' else 'native_ddp'
        sequence=policy_step-1 if service=='policy_train' else policy_step//5-1
        recipe={'backend':backend,'world_size':4,'micro_batch':micro,'accumulation_steps':gas,
            'global_batch':micro*gas*4}
        spec={'service':service,'group_id':service+':'+self.settings['run_id'],'sequence':sequence,
            'policy_version':policy_step-1,'world_version':((policy_step-1)//5 if world_version is None else world_version),
            'attestation':dict(recipe,source_sha256=self.source_sha256,
                recipe_sha256=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest(),
                proof_id='source_review_whole_optimizer_update_v1',boundary='optimizer_update')}
        return spec


def _validate_native_boundary(owner, receipt, expected_steps):
    if type(expected_steps) is not int or expected_steps<0:
        raise ValueError('Native expected update count must be canonical nonnegative integer')
    if (not isinstance(receipt,dict) or receipt.get('safe') is not True or
            receipt.get('backend')!='native_ddp' or receipt.get('boundary')!='optimizer_update' or
            type(receipt.get('rank')) is not int or type(owner.rank) is not int or
            receipt['rank']!=owner.rank or not 0<=owner.rank<4 or
            type(receipt.get('world_size')) is not int or receipt['world_size']!=4 or
            type(getattr(owner,'world_size',None)) is not int or owner.world_size!=4 or
            type(receipt.get('global_steps')) is not int or receipt['global_steps']!=expected_steps or
            receipt.get('own_stream_complete') is not True or receipt.get('global_gpu_drain') is not False):
        raise RuntimeError('Native World lacks an exact completed rank/backend/update/stream boundary receipt')
    return dict(receipt)


async def scheduled_world_begin(owner, policy_step):
    client=getattr(owner,'cost_admission',None)
    if client is None:return None
    spec=client.phase('world_train',owner.rank,policy_step,owner.micro,owner.gas)
    try:
        boundary=owner.worker_client.call('scheduler_boundary')
        _validate_native_boundary(owner,boundary,(policy_step-1)//5)
        receipt=await client.broker.acquire_training.remote(owner.rank,spec)
    except BaseException as error:
        await client.broker.fail_training.remote(spec['group_id'],repr(error))
        raise
    return {'receipt':receipt,'spec':spec,'policy_step':policy_step,'started':time.monotonic()}


async def scheduled_world_end(owner, token, error=None):
    if token is None:return
    client=owner.cost_admission
    if error is not None:
        await client.broker.fail_training.remote(token['spec']['group_id'],repr(error))
        return
    try:
        boundary=owner.worker_client.call('scheduler_boundary')
        _validate_native_boundary(owner,boundary,token['policy_step']//5)
        await client.broker.complete_training.remote(owner.rank,token['receipt']['work_id'],time.monotonic()-token['started'])
    except BaseException as failure:
        await client.broker.fail_training.remote(token['spec']['group_id'],repr(failure))
        raise


@contextmanager
def scheduled_policy_update(owner):
    client=getattr(owner,'cost_admission',None)
    if client is None:
        yield
        return
    import torch
    # Only the owner's stream is awaited; unrelated inference keeps running.
    torch.cuda.current_stream().synchronize()
    spec=client.phase('policy_train',owner.rank,owner.global_step+1,owner.train_batch_size,owner.accumulation_steps)
    receipt=client.ray.get(client.broker.acquire_training.remote(owner.rank,spec))
    started=time.monotonic()
    try:
        yield
        torch.cuda.current_stream().synchronize()
    except BaseException as error:
        client.ray.get(client.broker.fail_training.remote(spec['group_id'],repr(error)))
        raise
    else:
        client.ray.get(client.broker.complete_training.remote(owner.rank,receipt['work_id'],time.monotonic()-started))


def replay_inventory(store, admitted=None):
    from trajectory_sampling import enabled, inventory_buckets
    trajectory = enabled(getattr(store, 'config', None))
    buckets=Counter()
    for sample in store.samples.values():
        segment=store.segments[sample['segment_id']];version=segment.versions[sample['index']]
        if trajectory and segment.source == 'imagined':
            continue
        key=(segment.source,version['policy'],version.get('world_reward_bundle_version',version.get('world_bundle_version')),
            sample['uses'],sample['reserved'])
        buckets[key]+=1
    inventory=[]
    for index,(key,count) in enumerate(buckets.items()):
        source,pv,wv,uses,reserved=key
        inventory.append(dict(bucket_id=str(index),source=source,distinct_chunks=count,
            generation_policy_version=pv,generation_world_version=wv,
            successful_uses=uses,reserved_uses=reserved))
    if trajectory:
        inventory.extend(inventory_buckets(store))
    result={'inventory':inventory,'admitted_chunks':dict(admitted or {})}
    if getattr(getattr(store,'config',None),'jit_gae_mode','off') != 'off':
        result['critic_refresh_owner']='trainer'
    return result


async def acquire_world_request(owner, *, cost_key='world_infer:generate_chunk'):
    broker=getattr(owner,'cost_broker',None)
    if broker is None:return None
    owner.cost_sequence=getattr(owner,'cost_sequence',0)+1
    work_id=f"world:{owner.actor_id}:{owner.cost_sequence}"
    receipt=await acquire_inference(broker,dict(work_id=work_id,service='world_infer',
        cost_key=cost_key,units=1.,replica=f'world:{owner.actor_id}',
        resources=('card:'+str(int(owner.gpu_binding['physical_gpu'])),),
        metadata={'world_version':owner.capture_model_version}))
    return {'work_id':receipt['work_id'],'started':time.monotonic()}


async def complete_world_request(owner, token, success):
    if token is not None:
        task=asyncio.ensure_future(owner.cost_broker.complete.remote(token['work_id'],
            max(time.monotonic()-token['started'],1e-9),completed_units=1.,success=success))
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await finish_owned_rpc(task)
            raise
