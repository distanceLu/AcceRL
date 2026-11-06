"""Contract and fairness tests; no Ray, Torch, GPU, or sleeping."""
import json
import math
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cost_scheduler import (COST_UNIT, CostModel, ServiceConfig, WorkItem,
                            WeightedCostScheduler, normalize_execution_cost)


class CostModelTests(unittest.TestCase):
    def test_normalizes_execution_resource_cost(self):
        self.assertEqual(normalize_execution_cost(2, 4), 8)
        self.assertEqual(normalize_execution_cost(2, 4, capacity_scale=.5), 4)

    def test_ewma_uses_work_units_not_request_count(self):
        c = CostModel({'refresh/shape': 2}, alpha=.5)
        self.assertEqual(c.estimate('refresh/shape', 4), 8)
        self.assertEqual(c.update('refresh/shape', 12, 4), 2.5)
        self.assertEqual(c.estimate('refresh/shape', 4), 10)
        self.assertEqual(c.snapshot()['observations']['refresh/shape']['total_units'], 4)

    def test_provenance_is_explicit_and_json_serializable(self):
        c = CostModel()
        c.update('x', 3, 2, provenance='execution_wall_proxy')
        data = c.snapshot()
        self.assertEqual(data['cost_unit'], COST_UNIT)
        self.assertEqual(data['observations']['x']['last_provenance'], 'execution_wall_proxy')
        json.dumps(data, allow_nan=False)

    def test_rejects_bad_numbers_and_shapes(self):
        for value in [True, float('nan'), float('inf'), -1, '2']:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    CostModel().estimate('x', value)
        for alpha in [0, 2, float('nan')]:
            with self.assertRaises(ValueError):
                CostModel(alpha=alpha)
        with self.assertRaises(ValueError):
            normalize_execution_cost(1, True)


class AdmissionTests(unittest.TestCase):
    def make(self, **kwargs):
        s = WeightedCostScheduler(**kwargs)
        s.register_service('a', ServiceConfig(1, 8))
        s.register_service('b', ServiceConfig(1, 8))
        return s

    def add(self, scheduler, name, service='a', **kwargs):
        scheduler.enqueue(WorkItem(name, service, kwargs.pop('cost_key', service), **kwargs))

    def test_equal_weights_balance_cost_not_number_of_requests(self):
        s = self.make(cost_model=CostModel({'a':10, 'b':1}))
        for i in range(1000):
            self.add(s, f'a{i}')
            self.add(s, f'b{i}', 'b')
        counts = {'a':0, 'b':0}; costs = {'a':0., 'b':0.}
        for _ in range(550):
            g = s.claim_next()
            counts[g.service] += 1; costs[g.service] += g.estimated_cost
            s.complete(g.work_id, g.estimated_cost)
        self.assertLessEqual(abs(costs['a']-costs['b']), 10)
        self.assertGreater(counts['b'], counts['a']*8)

    def test_weights_control_cost_share_for_unequal_jobs(self):
        s = self.make(cost_model=CostModel({'a':5, 'b':1}))
        s.set_weights({'a':2, 'b':1})
        for i in range(1000):
            self.add(s, f'a{i}'); self.add(s, f'b{i}', 'b')
        consumed = {'a':0., 'b':0.}
        for _ in range(700):
            g = s.claim_next(); consumed[g.service] += g.estimated_cost
            s.complete(g.work_id, g.estimated_cost)
        self.assertLess(abs(consumed['a']/consumed['b']-2), .05)

    def test_parallel_replicas_share_one_budget_without_gang(self):
        s = self.make()
        self.add(s, 'a0', replica='replica0'); self.add(s, 'a1', replica='replica1')
        self.add(s, 'b0', 'b', replica='replica2')
        grants = [s.claim_next(), s.claim_next(), s.claim_next()]
        self.assertEqual({g.work_id for g in grants}, {'a0','a1','b0'})
        self.assertEqual(s.snapshot()['services']['a']['finish'], 2)
        self.assertEqual(s.snapshot()['services']['a']['inflight'], 2)
        self.assertEqual(s.snapshot()['services']['b']['finish'], 1)

    def test_shared_replica_is_not_concurrently_called_across_classes(self):
        s = self.make()
        self.add(s, 'action', replica='policy0'); self.add(s, 'refresh', 'b', replica='policy0')
        g = s.claim_next(); self.assertIsNone(s.claim_next())
        s.complete(g.work_id, 1)
        self.assertEqual(s.claim_next().work_id, 'refresh')

    def test_feedback_corrects_once_with_dispatch_weight(self):
        s = self.make(cost_model=CostModel({'a':10}))
        s.set_weights({'a':2}); self.add(s, 'x')
        g = s.claim_next(); self.assertEqual(g.virtual_finish, 5)
        s.set_weights({'a':4})
        record = s.complete('x', 14)
        self.assertEqual(record['dispatch_weight'], 2)
        self.assertEqual(record['virtual_correction'], 2)
        self.assertEqual(s.snapshot()['services']['a']['finish'], 7)
        before = s.snapshot()
        with self.assertRaises(ValueError): s.complete('x', 14)
        self.assertEqual(before, s.snapshot())

    def test_invalid_completion_does_not_release_or_mutate_work(self):
        s = self.make(); self.add(s, 'x', units=2); s.claim_next()
        before = s.snapshot()
        for cost, units in [(float('nan'),2),(2,1),(-1,2),(2,3)]:
            with self.assertRaises(ValueError): s.complete('x',cost,completed_units=units)
            self.assertEqual(before, s.snapshot())
        s.complete('x',2,completed_units=2)

    def test_dependency_waits_for_successful_completion_not_submission(self):
        s = self.make(); self.add(s, 'child','b',dependencies=('parent',))
        self.add(s, 'parent'); self.assertEqual(s.claim_next().work_id, 'parent')
        self.assertIsNone(s.claim_next()); s.complete('parent',1)
        self.assertEqual(s.claim_next().work_id, 'child')

    def test_failed_work_is_charged_but_not_dependency_progress(self):
        s = self.make(); self.add(s, 'x'); self.add(s,'child','b',dependencies=('x',))
        s.claim_next(); s.complete('x',.4,completed_units=.25,success=False)
        a = s.snapshot()['services']['a']
        self.assertEqual(a['observed_cost'], .4); self.assertEqual(a['completed_units'], .25)
        self.assertEqual(a['failed'],1); self.assertIsNone(s.claim_next())
        self.assertNotIn('a',s.snapshot()['cost_model']['observations'])

    def test_readiness_can_skip_independent_blocked_items(self):
        s = self.make(); self.add(s,'blocked',dependencies=('future',));self.add(s,'ready')
        self.assertEqual(s.claim_next().work_id,'ready')
        self.assertIsNone(s.claim_next(ready=lambda w:False))

    def test_dependency_cycle_is_rejected_when_future_work_arrives(self):
        s = self.make(); self.add(s,'x',dependencies=('y',))
        with self.assertRaises(ValueError): self.add(s,'y',dependencies=('x',))

    def test_resource_lanes_do_not_require_a_global_drain(self):
        s = self.make(resource_limits={'lane0':1,'lane1':1})
        self.add(s,'x',resources=('lane0',)); self.add(s,'y',resources=('lane0',))
        self.add(s,'other','b',resources=('lane1',))
        self.assertEqual(s.claim_next().work_id,'x')
        self.assertEqual(s.claim_next().work_id,'other')
        self.assertIsNone(s.claim_next())
        with self.assertRaises(ValueError):s.cancel('x')
        s.complete('x',1)
        self.assertEqual(s.claim_next().work_id,'y')

    def test_availability_is_checked_without_claiming_unavailable_lanes(self):
        s = self.make();self.add(s,'x',resources=('l0',));self.add(s,'y','b',resources=('l1',))
        self.assertEqual(s.claim_next(available_resources={'l1'}).work_id,'y')
        self.assertIsNone(s.claim_next(available_resources={'l0':0}))

    def test_max_inflight_limits_submission_not_running_execution(self):
        s = self.make(max_inflight_total=2)
        for i in range(3):self.add(s,str(i),replica=f'r{i}')
        one,two=s.claim_next(),s.claim_next();self.assertIsNone(s.claim_next())
        self.assertEqual(s.snapshot()['states'][one.work_id],'inflight')
        s.complete(two.work_id,1);self.assertIsNotNone(s.claim_next())

    def test_pending_cancellation_is_free_and_never_satisfies_dependency(self):
        s = self.make();self.add(s,'x');self.add(s,'child','b',dependencies=('x',))
        self.assertTrue(s.cancel('x'));self.assertIsNone(s.claim_next())
        self.assertEqual(s.snapshot()['services']['a']['estimated_cost'],0)
        with self.assertRaises(ValueError):s.cancel('x')

    def test_idle_service_does_not_hoard_free_credit(self):
        s = self.make()
        for i in range(20):
            self.add(s,f'a{i}');g=s.claim_next();s.complete(g.work_id,1)
        self.add(s,'a_next');self.add(s,'new','b')
        self.assertGreaterEqual(s.snapshot()['services']['b']['finish'],19)
        ids=[]
        for _ in range(2):g=s.claim_next();ids.append(g.work_id);s.complete(g.work_id,1)
        self.assertEqual(set(ids),{'a_next','new'})

    def test_weights_are_validated_atomically(self):
        s = self.make();before=s.snapshot()
        for weights in [{'a':2,'b':0},{'a':2,'missing':1},{'a':True}]:
            with self.assertRaises(ValueError):s.set_weights(weights)
            self.assertEqual(before,s.snapshot())

    def test_pending_bound_and_work_identity(self):
        s = self.make(max_pending=1);self.add(s,'x')
        with self.assertRaises(RuntimeError):self.add(s,'y')
        with self.assertRaises(ValueError):self.add(s,'x')
        s.claim_next();s.complete('x',1)
        with self.assertRaises(ValueError):self.add(s,'x')

    def test_work_items_defensively_copy_metadata(self):
        data={'phase':{'id':1}};w=WorkItem('x','a','a',metadata=data)
        data['phase']['id']=9;s=self.make();s.enqueue(w);w.metadata['phase']['id']=8
        g=s.claim_next();self.assertEqual(g.item.metadata['phase']['id'],1)
        g.item.metadata['phase']['id']=7
        self.assertEqual(s.snapshot()['inflight']['x']['item']['metadata']['phase']['id'],1)

    def test_work_item_validation(self):
        for kwargs in [{'units':0},{'units':True},{'dependencies':('x',)},
                       {'dependencies':('z','z')},{'resources':'gpu0'},
                       {'metadata':{'bad':math.nan}},{'replica':''}]:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):WorkItem('x','a','cost',**kwargs)


class RetentionTests(unittest.TestCase):
    def make(self, **kwargs):
        s = WeightedCostScheduler(**kwargs)
        s.register_service('policy', ServiceConfig(1, 4))
        return s

    def submit(self, s, work_id, *, dependencies=(), replica='policy0', **enqueue_kwargs):
        s.enqueue(WorkItem(work_id, 'policy', 'policy/shape2', replica=replica,
                           dependencies=dependencies, resources=('policy_lane',),
                           metadata={'bounded_payload': 'x'*256}), **enqueue_kwargs)
        return s.claim_next()

    def test_10000_automatic_completions_bound_all_terminal_payload_tables(self):
        s = self.make(max_terminal_history=16)
        for i in range(10000):
            g = self.submit(s, f'globally-unique-{i}', replica=f'ephemeral-replica-{i}')
            s.complete(g.work_id, 1.)
        self.assertEqual(len(s._work), 16)
        self.assertEqual(len(s._state), 16)
        self.assertEqual(len(s._order), 16)
        self.assertEqual(len(s._completed), 16)
        self.assertEqual(len(s._terminal_order), 16)
        self.assertEqual(s._replicas, {})
        self.assertEqual(s._resources, {})
        self.assertEqual(s._dependents, {})
        self.assertEqual(s.snapshot()['services']['policy']['completed'], 10000)
        self.assertEqual(s.snapshot()['services']['policy']['observed_cost'], 10000.)
        self.assertEqual(s.snapshot()['retention']['retired_count'], 9984)
        self.assertEqual(s.cost_model.snapshot()['observations']['policy/shape2']['count'], 10000)

    def test_10000_explicit_retirements_keep_only_aggregate_ledgers(self):
        s = self.make(max_terminal_history=8)
        for i in range(10000):
            g = self.submit(s, f'globally-unique-{i}')
            receipt = s.complete(g.work_id, 1.)
            self.assertEqual(receipt['observed_cost'], 1.)
            self.assertTrue(s.retire_completed(g.work_id))
        self.assertEqual(s._work, {})
        self.assertEqual(s._state, {})
        self.assertEqual(s._order, {})
        self.assertEqual(s._completed, {})
        self.assertEqual(s._terminal_order, {})
        self.assertEqual(s.snapshot()['services']['policy']['completed_units'], 10000.)

    def test_late_completion_after_retirement_never_charges_again(self):
        s = self.make(max_terminal_history=1)
        for work_id in ['old', 'new']:
            g = self.submit(s, work_id)
            s.complete(g.work_id, 1.)
        self.assertNotIn('old', s.snapshot()['states'])
        before = s.snapshot()
        with self.assertRaises(ValueError):
            s.complete('old', 2.)
        self.assertEqual(s.snapshot(), before)

    def test_retirement_rejects_pending_and_inflight(self):
        s = self.make()
        s.enqueue(WorkItem('x', 'policy', 'policy'))
        with self.assertRaises(ValueError):
            s.retire_completed('x')
        s.claim_next()
        with self.assertRaises(ValueError):
            s.retire_completed('x')

    def test_pending_and_inflight_dependencies_pin_terminal_parent(self):
        s = self.make(max_terminal_history=1)
        self.submit(s, 'parent')
        s.enqueue(WorkItem('child', 'policy', 'policy', dependencies=('parent',)))
        s.complete('parent', 1.)
        self.assertFalse(s.retire_completed('parent'))
        child = s.claim_next()
        self.assertEqual(child.work_id, 'child')
        self.assertFalse(s.retire_completed('parent'))
        s.complete('child', 1.)
        self.assertNotIn('parent', s.snapshot()['states'])
        self.assertEqual(s._dependents, {})

    def test_auto_eviction_never_drops_pinned_parent_for_unrelated_work(self):
        s = self.make(max_terminal_history=1)
        self.submit(s, 'parent')
        s.enqueue(WorkItem('child', 'policy', 'policy', dependencies=('parent', 'future')))
        s.complete('parent', 1.)
        for i in range(50):
            g = self.submit(s, f'independent{i}')
            s.complete(g.work_id, 1.)
        self.assertEqual(s.snapshot()['states']['parent'], 'completed')
        self.assertEqual(s.snapshot()['states']['child'], 'pending')
        self.assertIsNone(s.claim_next())
        s.cancel('child')
        self.assertNotIn('parent', s.snapshot()['states'])

    def test_explicit_lightweight_token_supports_future_dependency(self):
        s = self.make(max_terminal_history=1)
        g = self.submit(s, 'parent')
        s.complete(g.work_id, 1.)
        self.assertTrue(s.retire_completed('parent', keep_dependency=True))
        self.assertNotIn('parent', s._work)
        for i in range(10):
            g = self.submit(s, f'independent{i}')
            s.complete(g.work_id, 1.)
        child = self.submit(s, 'child', dependencies=('parent',))
        self.assertEqual(child.work_id, 'child')
        with self.assertRaises(ValueError):
            s.release_dependency('parent')
        s.complete('child', 1.)
        self.assertTrue(s.release_dependency('parent'))
        self.assertFalse(s.release_dependency('parent'))

    def test_retiring_unretained_parent_does_not_invent_future_progress(self):
        s = self.make()
        g = self.submit(s, 'parent')
        s.complete(g.work_id, 1.)
        s.retire_completed('parent')
        self.assertIsNone(self.submit(s, 'child', dependencies=('parent',)))
        self.assertEqual(s.snapshot()['states']['child'], 'pending')

    def test_dependency_tokens_are_bounded_and_never_auto_evicted(self):
        s = self.make(max_dependency_tokens=1)
        for work_id in ['first', 'second']:
            g = self.submit(s, work_id)
            s.complete(g.work_id, 1.)
        s.retire_completed('first', keep_dependency=True)
        before = s.snapshot()
        with self.assertRaises(RuntimeError):
            s.retire_completed('second', keep_dependency=True)
        self.assertEqual(before, s.snapshot())
        with self.assertRaises(ValueError):
            s.enqueue(WorkItem('first', 'policy', 'policy'))

    def test_failed_cancelled_work_cannot_satisfy_token_dependency(self):
        s = self.make()
        g = self.submit(s, 'failed')
        s.complete(g.work_id, .5, success=False)
        s.enqueue(WorkItem('cancelled', 'policy', 'policy'))
        s.cancel('cancelled')
        for work_id in ['failed', 'cancelled']:
            with self.subTest(work_id=work_id):
                with self.assertRaises(ValueError):
                    s.retire_completed(work_id, keep_dependency=True)
                self.assertTrue(s.retire_completed(work_id))

    def test_retirement_releases_child_dependency_references_after_cancel(self):
        s = self.make()
        g = self.submit(s, 'parent')
        s.enqueue(WorkItem('child', 'policy', 'policy', dependencies=('parent',)))
        s.complete(g.work_id, 1.)
        self.assertFalse(s.retire_completed('parent'))
        s.cancel('child')
        self.assertTrue(s.retire_completed('parent'))
        self.assertEqual(s._dependents, {})

    def test_monotonic_namespace_watermark_rejects_retired_identity_reuse(self):
        s = self.make(max_terminal_history=1)
        s.register_identity_namespace('actor0')
        for seq in range(100):
            g = self.submit(s, f'actor0:{seq}', identity_namespace='actor0', identity_sequence=seq)
            s.complete(g.work_id, 1.)
        with self.assertRaises(ValueError):
            self.submit(s, 'actor0:0', identity_namespace='actor0', identity_sequence=0)
        with self.assertRaises(ValueError):
            self.submit(s, 'actor0:0')
        with self.assertRaises(ValueError):
            self.submit(s, 'actor0:0', identity_namespace='actor0', identity_sequence=100)
        self.assertEqual(s.snapshot()['retention']['identity_watermarks'], {'actor0': 99})

    def test_namespace_validation_is_atomic_and_count_bounded(self):
        s = self.make(max_identity_namespaces=1)
        s.register_identity_namespace('actor0')
        before = s.snapshot()
        for fields in [dict(identity_namespace='missing', identity_sequence=0),
                       dict(identity_namespace='actor0', identity_sequence=True),
                       dict(identity_namespace='actor0', identity_sequence=-1),
                       dict(identity_namespace='actor0'), dict(identity_sequence=0)]:
            with self.subTest(fields=fields):
                with self.assertRaises(ValueError):
                    self.submit(s, 'actor0:0', **fields)
                self.assertEqual(before, s.snapshot())
        with self.assertRaises(RuntimeError):
            s.register_identity_namespace('actor1')

    def test_retention_configuration_requires_positive_canonical_counts(self):
        for field in ['max_terminal_history', 'max_dependency_tokens', 'max_identity_namespaces']:
            for value in [0, -1, True, 1.]:
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        self.make(**{field:value})


if __name__ == '__main__':
    unittest.main()
