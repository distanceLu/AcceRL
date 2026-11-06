"""CPU protocol checks; no claim of CUDA/backend boundary certification."""
from dataclasses import replace
import json
import math
import unittest

from collective_phase_gate import (
    BACKEND_BOUNDARIES, BoundaryAttestation, GangPhaseGate, GroupGrant,
    PhaseGateError, PhaseNotReady, PhaseSpec, UnsupportedBoundary,
)


def attestation(backend='policy_zero2', **changes):
    fields = dict(backend=backend, source_sha256='a'*64, recipe_sha256='b'*64,
                  world_size=4, micro_batch=8 if backend == 'policy_zero2' else 4,
                  accumulation_steps=18 if backend == 'policy_zero2' else 64,
                  global_batch=576 if backend == 'policy_zero2' else 1024,
                  proof_id='source-review/whole-update/cpu-protocol-only')
    fields.update(changes)
    return BoundaryAttestation(**fields)


def phase(sequence=0, *, service='policy_train', **changes):
    fields = dict(service=service, group_id='group47', sequence=sequence,
                  policy_version=0, world_version=0,
                  attestation=attestation('native_ddp' if service == 'world_train' else 'policy_zero2'))
    fields.update(changes)
    return PhaseSpec(**fields)


def arrive_all(gate, spec):
    for rank in gate.ranks:
        gate.arrive(rank, spec, previous_collective_complete=True, own_stream_complete=True)


def complete_rank(gate, rank, grant, cost=1., **changes):
    fields = dict(collective_complete=True, own_stream_complete=True,
                  cost_source='cpu_test_proxy')
    fields.update(changes)
    return gate.complete(rank, grant, cost, **fields)


class BoundaryAttestationTests(unittest.TestCase):
    def test_only_conservative_native_ddp_and_zero2_whole_updates(self):
        self.assertEqual(BACKEND_BOUNDARIES, {'policy_zero2': ('optimizer_update',),
                                            'native_ddp': ('optimizer_update',)})
        self.assertEqual(attestation().global_batch, 576)
        self.assertEqual(attestation('native_ddp').global_batch, 1024)

    def test_arbitrary_microbatch_is_not_safe_no_sync_evidence(self):
        for boundary in ['microbatch', 'ddp_no_sync_micro', 'forward', 'backward', 'step']:
            for backend in ['policy_zero2', 'native_ddp']:
                with self.subTest(backend=backend, boundary=boundary):
                    with self.assertRaises(UnsupportedBoundary):
                        attestation(backend, boundary=boundary)

    def test_backend_allowlist_is_immutable(self):
        with self.assertRaises(TypeError):
            BACKEND_BOUNDARIES['fsdp'] = ('optimizer_update',)

    def test_unknown_tp_fsdp_fail_closed(self):
        for backend in ['fsdp', 'fsdp2', 'tensor_parallel', 'tp', 'ddp', '']:
            with self.subTest(backend=backend):
                with self.assertRaises(UnsupportedBoundary):
                    attestation(backend)

    def test_recipe_must_preserve_global_batch(self):
        for changes in [dict(global_batch=575), dict(micro_batch=4),
                        dict(accumulation_steps=36), dict(world_size=8)]:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    attestation(**changes)

    def test_joint_micro4_gas36_is_same576(self):
        self.assertEqual(attestation(micro_batch=4, accumulation_steps=36).global_batch, 576)

    def test_canonical_recipe_integers(self):
        for field in ['micro_batch', 'world_size', 'accumulation_steps', 'global_batch']:
            for invalid in [True, '4', 4., 0, -1]:
                with self.subTest(field=field, invalid=invalid):
                    with self.assertRaises(ValueError):
                        attestation(**{field: invalid})

    def test_exact_source_recipe_hash_and_proof_required(self):
        for field, invalid in [('source_sha256', 'A'*64), ('recipe_sha256', 'b'*63),
                               ('source_sha256', ''), ('proof_id', '')]:
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    attestation(**{field: invalid})

    def test_phase_versions_sequence_not_coerced(self):
        for field in ['sequence', 'policy_version', 'world_version']:
            for invalid in [True, '0', 0., -1]:
                with self.subTest(field=field, invalid=invalid):
                    with self.assertRaises(ValueError):
                        phase(**{field: invalid})


class GangPhaseGateTests(unittest.TestCase):
    def setUp(self):
        self.gate = GangPhaseGate('group47', 'policy_train', range(4))
        self.spec = phase()

    def grant(self):
        arrive_all(self.gate, self.spec)
        return self.gate.grant(self.spec)

    def finish(self, grant, costs=(1., 2., 3., 4.)):
        result = None
        for rank in self.gate.ranks:
            self.gate.start(rank, grant)
            result = complete_rank(self.gate, rank, grant, costs[rank])
            if rank < 3:
                self.assertIsNone(result)
        return result

    def test_only_all_exact_ranks_ready(self):
        for rank in [0, 1, 2]:
            self.assertFalse(self.gate.arrive(rank, self.spec,
                previous_collective_complete=True, own_stream_complete=True))
        with self.assertRaises(PhaseNotReady):
            self.gate.grant(self.spec)
        self.assertTrue(self.gate.arrive(3, self.spec,
            previous_collective_complete=True, own_stream_complete=True))
        self.assertTrue(self.gate.ready)

    def test_duplicate_arrivals_never_substitute_missing_rank(self):
        for _ in range(4):
            self.assertFalse(self.gate.arrive(0, self.spec,
                previous_collective_complete=True, own_stream_complete=True))
        self.assertEqual(self.gate.snapshot()['arrived_ranks'], [0])

    def test_receipt_must_be_explicit_own_stream_not_global_gpu(self):
        with self.assertRaises(TypeError):
            self.gate.arrive(0, self.spec)
        for kwargs in [dict(previous_collective_complete=False, own_stream_complete=True),
                       dict(previous_collective_complete=True, own_stream_complete=False),
                       dict(previous_collective_complete=1, own_stream_complete=True)]:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(PhaseGateError):
                    self.gate.arrive(0, self.spec, **kwargs)
        self.assertEqual(self.gate.snapshot()['arrived_ranks'], [])
        self.assertFalse(self.gate.snapshot()['global_gpu_drain_required'])

    def test_unknown_rank_and_bool_rank_rejected(self):
        for rank in [4, True, '0', 0.0, -1]:
            with self.subTest(rank=rank):
                with self.assertRaises(ValueError):
                    self.gate.arrive(rank, self.spec,
                        previous_collective_complete=True, own_stream_complete=True)

    def test_divergent_sequence_fails_group_closed(self):
        self.gate.arrive(0, self.spec, previous_collective_complete=True, own_stream_complete=True)
        with self.assertRaises(PhaseGateError):
            self.gate.arrive(1, phase(1), previous_collective_complete=True, own_stream_complete=True)
        self.assertIsNotNone(self.gate.snapshot()['failure'])
        with self.assertRaises(PhaseGateError):
            self.gate.grant()

    def test_divergent_versions_recipe_and_source_rejected(self):
        for changed in [dict(policy_version=1), dict(world_version=1),
                        dict(attestation=attestation(recipe_sha256='c'*64)),
                        dict(attestation=attestation(source_sha256='c'*64)),
                        dict(attestation=attestation(micro_batch=4, accumulation_steps=36))]:
            with self.subTest(changed=changed):
                gate = GangPhaseGate('group47', 'policy_train', range(4))
                gate.arrive(0, self.spec, previous_collective_complete=True, own_stream_complete=True)
                with self.assertRaises(PhaseGateError):
                    gate.arrive(1, phase(**changed), previous_collective_complete=True, own_stream_complete=True)

    def test_wrong_group_service_or_rank_count_rejected(self):
        for spec in [phase(group_id='other'), phase(service='other'),
                     phase(attestation=attestation(world_size=2, accumulation_steps=36)),
                     phase(attestation=attestation('native_ddp'))]:
            with self.subTest(spec=spec):
                gate = GangPhaseGate('group47', 'policy_train', range(4))
                with self.assertRaises(PhaseGateError):
                    gate.arrive(0, spec, previous_collective_complete=True, own_stream_complete=True)

    def test_one_group_grant_not_four_service_debits(self):
        grant = self.grant()
        self.assertIs(grant, self.gate.grant(self.spec))
        self.assertEqual(self.gate.snapshot()['issued_group_grants'], 1)
        result = self.finish(grant)
        self.assertEqual(result.actual_cost, 10.)
        self.assertEqual(result.max_rank_cost, 4.)
        self.assertEqual(result.completed_phases, 1)
        self.assertEqual(len(result.rank_costs), 4)
        self.assertEqual(self.gate.snapshot()['completed_group_phases'], 1)
        self.assertEqual(self.gate.snapshot()['next_sequence'], 1)

    def test_rank_progress_difference_is_at_most_one(self):
        grant = self.grant()
        self.gate.start(0, grant)
        self.assertEqual(self.gate.snapshot()['rank_submission_skew'], 1)
        complete_rank(self.gate, 0, grant)
        self.assertEqual(self.gate.snapshot()['next_sequence'], 0)
        with self.assertRaises(PhaseGateError):
            self.gate.arrive(0, phase(1), previous_collective_complete=True, own_stream_complete=True)
        self.assertTrue(self.gate.snapshot()['must_drain'])

    def test_failure_before_grant_does_not_launch(self):
        arrive_all(self.gate, self.spec)
        self.gate.fail('dependency unavailable')
        with self.assertRaises(PhaseGateError):
            self.gate.grant()
        self.assertEqual(self.gate.snapshot()['issued_group_grants'], 0)

    def test_failure_after_partial_launch_does_not_revoke_peers(self):
        grant = self.grant()
        self.gate.start(0, grant)
        failed = self.gate.fail('observer lost an acknowledgement')
        self.assertTrue(failed['must_drain'])
        self.assertIs(self.gate.grant(), grant)
        complete_rank(self.gate, 0, grant)
        result = None
        for rank in [1, 2, 3]:
            self.assertTrue(self.gate.start(rank, grant))
            result = complete_rank(self.gate, rank, grant)
        self.assertEqual(result.actual_cost, 4.)
        self.assertFalse(self.gate.snapshot()['must_drain'])
        with self.assertRaises(PhaseGateError):
            self.gate.arrive(0, phase(1), previous_collective_complete=True, own_stream_complete=True)

    def test_no_fabricated_completion_for_missing_or_dead_rank(self):
        grant = self.grant()
        for rank in [0, 1, 2]:
            self.gate.start(rank, grant)
            self.assertIsNone(complete_rank(self.gate, rank, grant))
        self.gate.fail('rank3 died; communicator error handling required')
        state = self.gate.snapshot()
        self.assertEqual(state['remaining_rank_completions'], [3])
        self.assertEqual(state['completed_group_phases'], 0)
        self.assertTrue(state['must_drain'])

    def test_completion_requires_started_and_safe_boundary(self):
        grant = self.grant()
        with self.assertRaises(PhaseGateError):
            complete_rank(self.gate, 0, grant)
        self.gate.start(0, grant)
        with self.assertRaises(TypeError):
            self.gate.complete(0, grant, 1.)
        with self.assertRaises(PhaseGateError):
            complete_rank(self.gate, 0, grant, collective_complete=False)
        self.assertEqual(self.gate.snapshot()['completed_ranks'], [])

    def test_duplicate_launch_and_completion_do_not_double_count(self):
        grant = self.grant()
        self.assertTrue(self.gate.start(0, grant))
        self.assertFalse(self.gate.start(0, grant))
        self.assertIsNone(complete_rank(self.gate, 0, grant))
        self.assertIsNone(complete_rank(self.gate, 0, grant))
        self.assertEqual(self.gate.snapshot()['completed_ranks'], [0])
        with self.assertRaises(PhaseGateError):
            complete_rank(self.gate, 0, grant, cost=2.)

    def test_forged_or_old_grant_rejected(self):
        grant = self.grant()
        forged = replace(grant, grant_id='wrong')
        with self.assertRaises(PhaseGateError):
            self.gate.start(0, forged)
        self.finish(grant)
        with self.assertRaises(PhaseGateError):
            self.gate.start(0, grant)
        with self.assertRaises(PhaseGateError):
            complete_rank(self.gate, 0, grant)

    def test_cost_validation_and_provenance(self):
        grant = self.grant()
        self.gate.start(0, grant)
        for cost in [0., -1., True, float('nan'), float('inf'), '1']:
            with self.subTest(cost=cost):
                with self.assertRaises(ValueError):
                    complete_rank(self.gate, 0, grant, cost)
        with self.assertRaises(ValueError):
            complete_rank(self.gate, 0, grant, cost_source='SM_active_times_wall')
        self.assertEqual(self.gate.snapshot()['completed_ranks'], [])

    def test_next_whole_update_only_after_all_rank_ack(self):
        grant = self.grant()
        self.finish(grant)
        next_spec = phase(1, policy_version=1)
        arrive_all(self.gate, next_spec)
        next_grant = self.gate.grant()
        self.assertNotEqual(grant.grant_id, next_grant.grant_id)
        self.assertEqual(next_grant.sequence, 1)

    def test_published_versions_cannot_regress_after_complete_group(self):
        for previous, following in [(dict(policy_version=2), dict(policy_version=1)),
                                    (dict(world_version=2), dict(world_version=1))]:
            with self.subTest(previous=previous):
                gate = GangPhaseGate('group47', 'policy_train', range(4))
                arrive_all(gate, phase(**previous))
                grant = gate.grant()
                for rank in range(4):
                    gate.start(rank, grant)
                    complete_rank(gate, rank, grant)
                with self.assertRaises(PhaseGateError):
                    gate.arrive(0, phase(1, **following),
                        previous_collective_complete=True, own_stream_complete=True)

    def test_total_cost_must_not_overflow_into_scheduler_accounting(self):
        grant = self.grant()
        for rank in range(4):
            self.gate.start(rank, grant)
        for rank in range(3):
            complete_rank(self.gate, rank, grant, 1e308)
        with self.assertRaises(PhaseGateError):
            complete_rank(self.gate, 3, grant, 1e308)
        self.assertEqual(self.gate.snapshot()['completed_group_phases'], 0)

    def test_independent_service_groups_have_no_global_barrier(self):
        stuck = self.grant()
        other = GangPhaseGate('world', 'world_train', [0, 1, 2, 3])
        spec = phase(service='world_train', group_id='world', policy_version=4)
        arrive_all(other, spec)
        other_grant = other.grant()
        self.assertEqual(other_grant.phase.service, 'world_train')
        self.assertTrue(self.gate.snapshot()['must_drain'])
        self.assertEqual(self.gate.snapshot()['completed_group_phases'], 0)

    def test_snapshot_serializable_and_cost_provenance_retained(self):
        grant = self.grant()
        self.finish(grant)
        encoded = json.loads(json.dumps(self.gate.snapshot()))
        self.assertEqual(encoded['last_completion']['cost_sources'][0][1], 'cpu_test_proxy')
        self.assertFalse(encoded['global_gpu_drain_required'])

    def test_invalid_group_rank_or_skew_configuration(self):
        for ranks in [[], [0, 0], [True, 1], ['0'], [-1], [0.]]:
            with self.subTest(ranks=ranks):
                with self.assertRaises(ValueError):
                    GangPhaseGate('group47', 'policy_train', ranks)
        for skew in [0, -1, True, 1.]:
            with self.subTest(skew=skew):
                with self.assertRaises(ValueError):
                    GangPhaseGate('group47', 'policy_train', range(4), max_rank_skew=skew)


if __name__ == '__main__':
    unittest.main()
