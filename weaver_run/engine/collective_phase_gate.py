"""Cooperative, group-wide admission at reviewed collective boundaries.

The initial adapters expose a *whole optimizer update*, not arbitrary microbatches.
Policy ZeRO2 invokes DeepSpeed backward/step for every microbatch; native WEAVER
DDP invokes a finite-loss all-reduce even inside ``no_sync()``.  Neither fact is
evidence that independently pausing a rank between arbitrary calls is safe.

This module does no CUDA, process, or network operations.  Completion receipts
must be supplied by backend adapters after their own execution stream and all
phase communication have completed.  It does not require unrelated GPU work to
drain, and a granted cohort is never revoked while ranks still need to join it.
Source-review attestations describe a contract; CPU tests do not certify a real
GPU backend or communicator implementation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
import threading
from types import MappingProxyType
import uuid


class PhaseGateError(RuntimeError):
    """A collective stage cannot safely be admitted or acknowledged."""


class PhaseNotReady(PhaseGateError):
    """Not every rank has arrived at the same verified boundary."""


class UnsupportedBoundary(ValueError):
    """No reviewed backend adapter exists for the requested boundary."""


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be a canonical integer >= {minimum}')
    return value


def _name(value, name):
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f'{name} must be a nonempty trimmed string')
    return value


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a finite positive number')
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a finite positive number')
    return value


# Conservative allowlist.  FSDP/TP and arbitrary microbatch boundaries need their
# own reviewed communication adapter; they are deliberately absent.
BACKEND_BOUNDARIES = MappingProxyType({
    'policy_zero2': ('optimizer_update',),
    'native_ddp': ('optimizer_update',),
})


@dataclass(frozen=True)
class BoundaryAttestation:
    backend: str
    source_sha256: str
    recipe_sha256: str
    world_size: int
    micro_batch: int
    accumulation_steps: int
    global_batch: int
    proof_id: str
    boundary: str = 'optimizer_update'

    def __post_init__(self):
        if self.backend not in BACKEND_BOUNDARIES or self.boundary not in BACKEND_BOUNDARIES[self.backend]:
            raise UnsupportedBoundary(f'Unreviewed backend/boundary: {self.backend}/{self.boundary}')
        for name in ('source_sha256', 'recipe_sha256'):
            if not isinstance(getattr(self, name), str) or re.fullmatch('[0-9a-f]{64}', getattr(self, name)) is None:
                raise ValueError(f'{name} must be an exact lowercase SHA256')
        for name in ('world_size', 'micro_batch', 'accumulation_steps', 'global_batch'):
            _integer(getattr(self, name), name, 1)
        if self.micro_batch * self.accumulation_steps * self.world_size != self.global_batch:
            raise ValueError('The attested micro/GAS/rank recipe does not equal global_batch')
        _name(self.proof_id, 'proof_id')


@dataclass(frozen=True)
class PhaseSpec:
    service: str
    group_id: str
    sequence: int
    policy_version: int
    world_version: int
    attestation: BoundaryAttestation

    def __post_init__(self):
        _name(self.service, 'service')
        _name(self.group_id, 'group_id')
        for name in ('sequence', 'policy_version', 'world_version'):
            _integer(getattr(self, name), name)
        if not isinstance(self.attestation, BoundaryAttestation):
            raise ValueError('PhaseSpec requires a BoundaryAttestation')


@dataclass(frozen=True)
class GroupGrant:
    grant_id: str
    phase: PhaseSpec
    ranks: tuple[int, ...]
    must_drain: bool = True

    @property
    def sequence(self):
        return self.phase.sequence

    @property
    def work_id(self):
        return f'{self.phase.service}:{self.phase.group_id}:{self.sequence}'


@dataclass(frozen=True)
class GroupCompletion:
    grant_id: str
    phase: PhaseSpec
    rank_costs: tuple[tuple[int, float], ...]
    cost_sources: tuple[tuple[int, str], ...]
    actual_cost: float
    max_rank_cost: float
    completed_phases: int = 1

    @property
    def execution_cost(self):
        return self.actual_cost


class GangPhaseGate:
    """One ordered stream of group grants; independent services keep running.

    A single instance is shared by all ranks of one communicating service.
    ``grant`` authorizes the entire cohort.  After this commit point, ``fail``
    blocks future grants but permits all members to start/finish the granted
    phase, because revoking a late rank could strand an in-flight collective.
    A real process death must be handled by the backend's communicator failure
    procedure; the gate never fabricates completion or sends process signals.

    ``execution_cost`` is already normalized reference-device-seconds for that
    rank.  The one group completion reports their sum, exactly once.  Wall-clock
    proxies must remain labelled as proxies, not GPU event measurements.
    """
    def __init__(self, group_id, service, ranks, *, max_rank_skew=1, initial_sequence=0):
        self.group_id = _name(group_id, 'group_id')
        self.service = _name(service, 'service')
        ranks = tuple(ranks)
        if not ranks or any(type(r) is not int or r < 0 for r in ranks) or len(set(ranks)) != len(ranks):
            raise ValueError('ranks must be a nonempty unique set of canonical nonnegative integers')
        self.ranks = tuple(sorted(ranks))
        self.max_rank_skew = _integer(max_rank_skew, 'max_rank_skew', 1)
        self.next_sequence = _integer(initial_sequence, 'initial_sequence')
        self._phase = None
        self._arrived = set()
        self._grant = None
        self._started = set()
        self._finished = {}
        self._failure = None
        self._issued_count = self._completed_count = 0
        self._last_completion = None
        self._last_phase = None
        self._lock = threading.RLock()

    def _rank(self, rank):
        if type(rank) is not int or rank not in self.ranks:
            raise ValueError('The acknowledgement rank does not belong to this exact group')

    def _violation(self, message):
        self._failure = self._failure or message
        raise PhaseGateError(message)

    def _match(self, spec):
        if not isinstance(spec, PhaseSpec):
            raise ValueError('A PhaseSpec is required')
        if spec.group_id != self.group_id or spec.service != self.service:
            self._violation('Phase service/group identity differs from the gate')
        if spec.attestation.world_size != len(self.ranks):
            self._violation('Attested rank count differs from the exact group')
        expected_backend = {'policy_train': 'policy_zero2', 'world_train': 'native_ddp'}.get(self.service)
        if expected_backend and spec.attestation.backend != expected_backend:
            self._violation('Training service does not match its reviewed backend')
        if spec.sequence != self.next_sequence:
            self._violation('Ranks must submit the same next scheduling sequence')
        if self._last_phase is not None and (
                spec.policy_version < self._last_phase.policy_version or
                spec.world_version < self._last_phase.world_version):
            self._violation('Published versions cannot regress between group phases')
        if self._phase is not None and spec != self._phase:
            self._violation('Ranks disagree on phase, versions, source, or recipe attestation')

    @staticmethod
    def _boundary_ack(collective_complete, own_stream_complete):
        if collective_complete is not True or own_stream_complete is not True:
            raise PhaseGateError('The backend must acknowledge completed phase collectives and its own execution stream')

    def arrive(self, rank, spec, *, previous_collective_complete, own_stream_complete):
        """Record a rank at the preceding safe boundary; return whole-group ready."""
        with self._lock:
            self._rank(rank)
            if self._failure:
                raise PhaseGateError(f'Gate failed; no new stage admission: {self._failure}')
            self._boundary_ack(previous_collective_complete, own_stream_complete)
            self._match(spec)
            if self._grant is not None:
                self._violation('An admitted phase must finish before any next phase arrival')
            self._phase = spec
            self._arrived.add(rank)
            return self.ready

    @property
    def ready(self):
        with self._lock:
            return self._failure is None and self._phase is not None and self._grant is None and self._arrived == set(self.ranks)

    def grant(self, spec=None):
        """Commit one gang dispatch; repeated reads return the same active grant."""
        with self._lock:
            if self._grant is not None:
                if spec is not None and spec != self._grant.phase:
                    self._violation('Attempted to replace a submitted cohort')
                return self._grant
            if self._failure:
                raise PhaseGateError(f'Gate failed: {self._failure}')
            if spec is not None:
                self._match(spec)
            if not self.ready:
                raise PhaseNotReady('All exact ranks must arrive at the same safe phase boundary')
            self._grant = GroupGrant(uuid.uuid4().hex, self._phase, self.ranks)
            self._issued_count += 1
            return self._grant

    def _active(self, grant):
        if not isinstance(grant, GroupGrant) or grant != self._grant:
            raise PhaseGateError('Grant is not this group\'s exact active dispatch')

    def start(self, rank, grant):
        """ACK local launch.  A failure never revokes peers' committed launch."""
        with self._lock:
            self._rank(rank)
            self._active(grant)
            if rank in self._started:
                return False
            # One cohort is active at a time.  Rank phase progress can differ by
            # at most one, which is stricter than any configured bound >= 1.
            proposed = [self.next_sequence + int(r in self._started or r == rank) for r in self.ranks]
            if max(proposed) - min(proposed) > self.max_rank_skew:
                self._violation('Rank submission progress exceeds its attested skew bound')
            self._started.add(rank)
            return True

    def complete(self, rank, grant, execution_cost, *, collective_complete,
                 own_stream_complete, cost_source='execution_wall_proxy'):
        """ACK one local completion; emit exactly one all-rank cost receipt."""
        with self._lock:
            self._rank(rank)
            self._active(grant)
            self._boundary_ack(collective_complete, own_stream_complete)
            cost = _positive(execution_cost, 'execution_cost')
            if cost_source not in ('execution_wall_proxy', 'gpu_event_device_seconds', 'cpu_test_proxy'):
                raise ValueError('Unknown execution-cost provenance')
            if rank not in self._started:
                raise PhaseGateError('A rank cannot complete a phase it has not started')
            if rank in self._finished:
                if self._finished[rank] != (cost, cost_source):
                    self._violation('A duplicate completion changed the accounted execution cost')
                return None
            self._finished[rank] = (cost, cost_source)
            if len(self._finished) != len(self.ranks):
                return None
            costs = tuple((r, self._finished[r][0]) for r in self.ranks)
            sources = tuple((r, self._finished[r][1]) for r in self.ranks)
            try:
                total_cost = math.fsum(v for _, v in costs)
            except OverflowError:
                self._violation('Total group execution cost overflowed')
            if not math.isfinite(total_cost):
                self._violation('Total group execution cost must remain finite')
            result = GroupCompletion(grant.grant_id, grant.phase, costs, sources,
                                     total_cost, max(v for _, v in costs))
            self._last_completion = result
            self._last_phase = grant.phase
            self._completed_count += 1
            self.next_sequence += 1
            self._phase = self._grant = None
            self._arrived.clear()
            self._started.clear()
            self._finished.clear()
            return result

    def fail(self, reason):
        with self._lock:
            reason = _name(reason, 'reason')
            self._failure = self._failure or reason
            return self.snapshot()

    def snapshot(self):
        with self._lock:
            progress = {r: self.next_sequence + int(r in self._started) for r in self.ranks}
            return {'group_id': self.group_id, 'service': self.service,
                    'ranks': list(self.ranks), 'next_sequence': self.next_sequence,
                    'phase': asdict(self._phase) if self._phase else None,
                    'ready': self.ready, 'grant_id': self._grant.grant_id if self._grant else None,
                    'arrived_ranks': sorted(self._arrived), 'started_ranks': sorted(self._started),
                    'completed_ranks': sorted(self._finished), 'rank_submission_progress': progress,
                    'rank_submission_skew': max(progress.values()) - min(progress.values()),
                    'max_rank_skew': self.max_rank_skew, 'failure': self._failure,
                    'must_drain': self._grant is not None,
                    'remaining_rank_completions': [r for r in self.ranks if r not in self._finished] if self._grant else [],
                    'issued_group_grants': self._issued_count,
                    'completed_group_phases': self._completed_count,
                    'global_gpu_drain_required': False,
                    'last_completion': asdict(self._last_completion) if self._last_completion else None}
