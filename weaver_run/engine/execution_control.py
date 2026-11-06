"""Opt-in execution controls; importable without Ray or CUDA.

The first GPU experiment changes control granularity, not micro shape, GAE,
Replay sampling, reward labels, reuse or publication safety.
"""
from __future__ import annotations

import asyncio
import math
import time


DEFAULTS = dict(parent_policy_budget=False, authoritative_production=False,
                refresh_lease_micros=1, refresh_lease_seconds=0.8,
                refresh_trim='legacy', dependency_priority=False,
                action_wait_seconds=1.0)


def execution_settings(raw=None):
    raw = dict(raw or {})
    if set(raw) - set(DEFAULTS):
        raise ValueError('Unknown execution control settings')
    out = dict(DEFAULTS, **raw)
    for key in ('parent_policy_budget', 'authoritative_production', 'dependency_priority'):
        if type(out[key]) is not bool:
            raise ValueError(key + ' must be boolean')
    if type(out['refresh_lease_micros']) is not int or not 1 <= out['refresh_lease_micros'] <= 4:
        raise ValueError('Refresh leases allow 1..4 unchanged microbatches')
    for key in ('refresh_lease_seconds', 'action_wait_seconds'):
        if type(out[key]) not in (int, float) or not math.isfinite(out[key]) or not .01 <= out[key] <= 5:
            raise ValueError(key + ' must be finite and bounded')
    if out['refresh_trim'] not in ('legacy', 'block', 'disabled'):
        raise ValueError('Unknown refresh allocator experiment')
    return out


async def wait_event(event):
    """Yield the CPU actor loop; never synchronize every process on the card."""
    while not event.query():
        await asyncio.sleep(.001)


async def refresh_block(owner, batch, torch, *, clock=time.perf_counter):
    """Keep at most one GPU micro outstanding and copy only one value block.

The returned prefix may be shorter than the reservation. Its unused allowance
is refunded at settlement. A pinned job is not advanced until settlement.
"""
    dispatcher = owner._policy_dispatch
    cfg = dispatcher.execution
    limit = min(dispatcher.settings['refresh_micro'],
                max(1, int(getattr(owner, 'training_batch_limit', 2))))
    values = []
    started = clock()
    event = None
    trim_seconds = 0.0
    try:
        for offset in range(0, batch.units, limit):
            if owner.capture_model_version != batch.refresh_job.version:
                raise RuntimeError('Policy changed during pinned refresh block')
            rows = batch.inputs[offset:offset + limit]
            with torch.inference_mode():
                prepared = owner.model.prepare_inputs_batch(list(rows))
                _, value = owner.model(prepared)
                value = value.detach().float().reshape(-1)
                if value.numel() != len(rows):
                    raise ValueError('Refresh micro returned wrong observation count')
                values.append(value)
                if value.is_cuda:
                    event = torch.cuda.Event()
                    event.record(torch.cuda.current_stream(value.device))
            if event is not None:
                await wait_event(event)
                event = None
            del prepared, value
            if cfg['refresh_trim'] == 'legacy' and values[-1].is_cuda:
                import os
                if os.environ.get('ACCERL_TRIM_IDLE_CACHE') == '1':
                    trim_started = clock()
                    torch.cuda.empty_cache()
                    trim_seconds += clock() - trim_started
            await asyncio.sleep(0)
            if (dispatcher.queues['real'] or dispatcher.queues['imagined']
                    or clock() - started >= cfg['refresh_lease_seconds']):
                break
        # A single bounded D2H instead of one per micro. CPU checks gate all
        # delivery/consumption; an invalid block never reaches training.
        array = torch.cat(values).cpu().numpy()
        return array, dict(micros=len(values), observations=len(array),
                          reserved_observations=batch.units, value_d2h_calls=1,
                          trim_seconds=trim_seconds, execution_seconds=clock()-started,
                          cost_provenance='execution_wall_proxy')
    finally:
        # Cancellation cannot release a version pin while submitted GPU work
        # still reads those weights. Complete only this stream's event.
        if event is not None:
            from cost_scheduler_runtime import finish_owned_rpc
            await finish_owned_rpc(asyncio.ensure_future(wait_event(event)))


def production_snapshot(owner):
    """Read the actual production gate, including pressure/lookahead bypass.

Real service readiness is observed from admitted episode leases. Starting a
new real episode additionally needs allocation/eviction and simulator budget
checks, so it is explicitly not inferred from credit or Replay size.
"""
    gate = owner.pipeline.imagination_admission()
    owner.production_revision = getattr(owner, 'production_revision', 0) + 1
    active_real = len(owner.pipeline._real)
    active_imagined = len(owner.pipeline._imagine)
    return dict(revision=owner.production_revision,
                versions=list(owner.pipeline._production_versions or (0, 0)),
                real_production_ready=bool(active_real),
                imagination_ready=bool(active_imagined or gate['can_start']),
                active_real=active_real, active_imagined=active_imagined,
                imagination_can_start=gate['can_start'], imagination_block_reason=gate['reason'],
                real_new_episode_requires_allocation=True,
                available_credits=owner.pipeline._credits,
                pending_seeds=sum(len(x.starts) for x in owner.pipeline._episodes.values()))
