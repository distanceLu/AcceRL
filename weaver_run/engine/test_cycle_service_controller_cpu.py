"""Behavioral CPU contracts for empirical cycle-level service allocation."""
from dataclasses import replace
import math
import unittest

from cycle_service_controller import (
    SERVICES, InventoryBucket, CycleSnapshot, ResponseSample, ControllerConfig,
    CycleServiceController,
)


BASE = {s:1.0 for s in SERVICES}


def stock(real=144, imagined=432, **kwargs):
    return (InventoryBucket('r','real',real,4,**kwargs),
            InventoryBucket('i','imagined',imagined,4,0,**kwargs))


def snapshot(**kwargs):
    args = dict(now=100,policy_version=4,world_version=0,inventory=stock(),
                critic_observations_remaining=10,policy_training_units_remaining=1,
                world_training_units_remaining=1,
                policy_action_observations_per_real_chunk=1.0,
                policy_action_observations_per_imagined_chunk=1.25)
    args.update(kwargs)
    return CycleSnapshot(**args)


def teach(controller, weights=BASE, *, now=100, context='ordinary', rates=None):
    data = {'policy_training':1.0, 'world_training':1.0, 'critic_observations':10.0,
            'policy_action_observations':10.0, 'real_chunks':1.0, 'imagined_chunks':1.0}
    data.update(rates or {})
    services = {'policy_training':'policy_train','world_training':'world_train',
                'critic_observations':'policy_infer','policy_action_observations':'policy_infer',
                'real_chunks':'policy_infer','imagined_chunks':'world_infer'}
    for kind,rate in data.items():
        for n in range(controller.config.min_response_samples):
            controller.observe(ResponseSample(weights,services[kind],rate,1.0,context,kind,now))
            if kind=='world_training' and context=='ordinary':
                controller.observe(ResponseSample(weights,services[kind],rate,1.0,'joint',kind,now))


class InventoryContracts(unittest.TestCase):
    def setUp(self):self.c=CycleServiceController()

    def test_distinct_is_not_remaining_exposures(self):
        d=self.c.demands(snapshot(inventory=stock(36,108)))
        self.assertEqual(d['eligible_distinct_chunks'],{'real':36,'imagined':108})
        self.assertEqual(d['remaining_exposure_credits'],{'real':144,'imagined':432})
        self.assertEqual(d['next_batch_missing'],{'real':108,'imagined':324})
        self.assertEqual(d['cycle_missing_exposures'],{'real':108,'imagined':324})

    def test_missing_source_never_uses_other_source(self):
        d=self.c.demands(snapshot(inventory=stock(10000,0)))
        self.assertEqual(d['next_batch_missing'],{'real':0,'imagined':432})
        self.assertEqual(d['cycle_missing_exposures']['imagined'],432)

    def test_expired_uses_and_versions_fail_closed(self):
        inv=(InventoryBucket('expired','real',10,20,expires_at=100),
             InventoryBucket('old-policy','real',11,0),
             InventoryBucket('old-world','imagined',12,20,0),
             InventoryBucket('used','real',13,20,successful_uses=4),
             InventoryBucket('reserved','real',14,20,successful_uses=3,reserved_uses=1))
        s=snapshot(policy_version=20,world_version=4,inventory=inv)
        d=self.c.inventory(s)
        self.assertEqual(d['eligible_distinct_chunks'],{'real':0,'imagined':0})
        self.assertEqual(d['excluded_distinct_chunks'],{'expired':10,'policy_age':11,
            'world_age':12,'use_cap':27})

    def test_real_inventory_does_not_need_world_version(self):
        s=snapshot(inventory=(InventoryBucket('r','real',144,4),))
        self.assertEqual(self.c.inventory(s)['eligible_distinct_chunks']['real'],144)

    def test_cycle_capacity_counts_once_per_update_and_ages_out(self):
        s=CycleSnapshot(100,16,3,(InventoryBucket('age16','real',144,0),),
                        critic_observations_per_future_update=10,
                        policy_action_observations_per_real_chunk=1,
                        policy_action_observations_per_imagined_chunk=1.25)
        d=self.c.demands(s)
        self.assertEqual(d['remaining_policy_updates'],4)
        self.assertEqual(d['cycle_existing_capacity']['real'],144)
        self.assertEqual(d['cycle_missing_exposures']['real'],3*144)

    def test_inflight_current_batch_does_not_consume_future_capacity_twice(self):
        s=CycleSnapshot(100,16,3,(InventoryBucket('old','real',144,0),),phase='refresh',
                        critic_observations_per_future_update=10,
                        policy_action_observations_per_real_chunk=1,
                        policy_action_observations_per_imagined_chunk=1.25)
        d=self.c.demands(s)
        self.assertEqual(d['cycle_existing_capacity']['real'],0)
        self.assertEqual(d['cycle_missing_exposures']['real'],3*144)

    def test_no_inventory_mutation(self):
        buckets=stock();before=repr(buckets);s=snapshot(inventory=buckets)
        self.c.demands(s);self.c.choose(s)
        self.assertEqual(repr(buckets),before)

    def test_duplicate_buckets_and_future_versions_rejected_or_excluded(self):
        with self.assertRaises(ValueError):snapshot(inventory=(stock()[0],stock()[0]))
        s=snapshot(inventory=(InventoryBucket('future','real',144,5),))
        self.assertEqual(self.c.inventory(s)['eligible_distinct_chunks']['real'],0)

    def test_original_learning_invariants_are_explicit(self):
        self.assertEqual(self.c.demands(snapshot())['invariant'],
            {'real_per_batch':144,'imagined_per_batch':432,'global_batch':576,
             'max_successful_uses':4,'max_policy_age':16,'max_world_age':3,'policy_to_world':5})

    def test_exposure_credits_and_new_production_chunks_are_different_units(self):
        d=self.c.demands(snapshot(policy_version=0,inventory=(),
            critic_observations_per_future_update=10))
        self.assertEqual(d['cycle_missing_exposures'],{'real':720,'imagined':2160})
        self.assertEqual(d['minimum_new_chunks'],{'real':180,'imagined':540})
        self.assertEqual(d['policy_action_observations'],180+540*1.25)


class ForecastContracts(unittest.TestCase):
    def test_complete_joint_cycle_has_parallel_not_additive_world_path(self):
        c=CycleServiceController();teach(c,rates={'world_training':0.1})
        f,d,why=c.forecast(snapshot(),BASE)
        self.assertIsNone(why)
        self.assertEqual(f['critical_path'],'world_joint_training')
        self.assertEqual(f['seconds'],10)
        self.assertEqual(d['remaining_world_updates'],1)

    def test_expiry_at_predicted_consumption_is_rechecked(self):
        c=CycleServiceController();teach(c)
        s=snapshot(inventory=stock(expires_at=100.5))
        f,d,why=c.forecast(s,BASE)
        self.assertIsNone(why)
        self.assertEqual(d['eligible_distinct_chunks'],{'real':0,'imagined':0})
        self.assertEqual(d['next_batch_missing'],{'real':144,'imagined':432})
        self.assertGreaterEqual(f['seconds'],432)

    def test_critic_demand_is_not_a_cheap_cpu_phase(self):
        c=CycleServiceController();teach(c)
        f,d,_=c.forecast(snapshot(critic_observations_remaining=1353),BASE)
        # Both rates describe overlapping end-to-end windows, not additive phases.
        self.assertAlmostEqual(f['paths_seconds']['policy_refresh_and_train'],135.3)
        self.assertEqual(c.ready_services(snapshot()),('world_train','policy_infer'))
        self.assertIn('policy_train',c.ready_services(snapshot(phase='train',critic_observations_remaining=0)))

    def test_missing_future_refresh_work_keeps_forecast_unknown(self):
        c=CycleServiceController();teach(c)
        s=snapshot(policy_version=0,inventory=())
        decision=c.choose(s)
        self.assertEqual(decision.reason,'future_critic_work_unknown')
        self.assertIsNone(decision.predicted_cycle_seconds)
        self.assertFalse(decision.changed)

    def test_policy_production_conversion_must_be_explicit(self):
        c=CycleServiceController();teach(c)
        d=c.choose(snapshot(inventory=(),policy_action_observations_per_imagined_chunk=None))
        self.assertEqual(d.reason,'production_policy_observation_conversion_unknown')
        self.assertFalse(d.changed)

    def test_action_demand_has_real_imagination_and_bootstrap_units(self):
        c=CycleServiceController()
        d=c.demands(snapshot(inventory=()))
        self.assertEqual(d['policy_action_observations'],144+432*1.25)

    def test_publication_blocks_dependents_without_gpu_drain(self):
        c=CycleServiceController();teach(c)
        s=snapshot(publication_blocked=True,publication_seconds_remaining=3)
        d=c.choose(s,allow_probe=True)
        self.assertEqual(d.reason,'publication_dependency')
        self.assertEqual(d.ready_services,())
        self.assertFalse(d.changed)

    def test_world_batch_not_ready_is_not_scheduled(self):
        c=CycleServiceController();teach(c)
        s=snapshot(world_batch_ready=False)
        f,d,why=c.forecast(s,BASE)
        self.assertIsNone(f);self.assertEqual(why,'world_batch_dependency_not_ready')
        self.assertNotIn('world_train',c.ready_services(s))

    def test_missing_source_dependency_does_not_fallback(self):
        c=CycleServiceController();teach(c)
        s=snapshot(inventory=stock(10000,0),imagination_ready=False)
        f,_,why=c.forecast(s,BASE)
        self.assertIsNone(f);self.assertEqual(why,'production_dependency_not_ready')

    def test_pending_world_boundary_does_not_start_new_policy_cycle(self):
        c=CycleServiceController();teach(c)
        s=CycleSnapshot(100,5,0,world_training_units_remaining=1)
        d=c.demands(s)
        self.assertEqual(d['remaining_policy_updates'],0)
        self.assertEqual(d['remaining_world_updates'],1)
        self.assertEqual(d['policy_training'],0)
        self.assertEqual(c.ready_services(s),('world_train',))

    def test_future_joint_work_is_planned_but_not_ready_on_ordinary_update(self):
        c=CycleServiceController()
        s=snapshot(policy_version=1,inventory=(),critic_observations_per_future_update=10)
        self.assertEqual(c.demands(s)['remaining_world_updates'],1)
        self.assertNotIn('world_train',c.ready_services(s))

    def test_future_joint_work_requires_its_measured_context(self):
        c=CycleServiceController();teach(c)
        s=snapshot(policy_version=3,inventory=(),critic_observations_per_future_update=10)
        f,_,why=c.forecast(s,BASE)
        self.assertIsNone(f);self.assertEqual(why,'unmeasured_response')
        teach(c,context='joint',rates={'policy_training':0.5,'critic_observations':5})
        f,d,why=c.forecast(s,BASE)
        self.assertIsNone(why)
        self.assertGreater(f['seconds'],0)


class EmpiricalControlContracts(unittest.TestCase):
    def test_unknown_response_is_not_scaled_linearly(self):
        c=CycleServiceController();teach(c)
        trial=dict(BASE,world_infer=1.25,policy_train=0.75)
        estimate,_,why=c.forecast(snapshot(inventory=()),trial)
        self.assertIsNone(estimate);self.assertEqual(why,'unmeasured_response')
        d=c.choose(snapshot(inventory=()))
        self.assertFalse(d.changed)
        self.assertIsNotNone(d.exploration_candidate)

    def test_measured_nonlinear_candidate_beats_more_weight(self):
        cfg=ControllerConfig(beta_seconds_per_weight=0,cooldown_seconds=0)
        c=CycleServiceController(config=cfg)
        teach(c)
        bad=dict(BASE,world_infer=1.25,policy_train=0.75)
        good=dict(BASE,world_infer=0.75,policy_infer=1.25)
        teach(c,bad,rates={'imagined_chunks':0.5})
        teach(c,good,rates={'imagined_chunks':2})
        d=c.choose(snapshot(inventory=()))
        self.assertTrue(d.changed)
        self.assertEqual(d.weights,good)
        self.assertEqual(d.reason,'measured_cycle_improvement')
        self.assertLess(d.predicted_cycle_seconds,300)

    def test_l1_penalty_prevents_small_unhelpful_switch(self):
        cfg=ControllerConfig(beta_seconds_per_weight=100,min_relative_improvement=0)
        c=CycleServiceController(config=cfg);teach(c)
        trial=dict(BASE,policy_train=1.25,world_train=0.75)
        teach(c,trial,rates={'policy_training':2})
        d=c.choose(snapshot())
        self.assertFalse(d.changed)
        self.assertEqual(d.weights,BASE)

    def test_explicit_probe_obeys_bounds_total_change_and_cooldown(self):
        c=CycleServiceController()
        teach(c)
        d=c.choose(snapshot(inventory=()),allow_probe=True)
        self.assertTrue(d.changed)
        self.assertIsNone(d.predicted_cycle_seconds)
        self.assertAlmostEqual(sum(d.weights.values()),4)
        self.assertLessEqual(sum(abs(d.weights[s]-BASE[s]) for s in SERVICES),c.config.max_l1_change)
        self.assertTrue(all(c.config.min_weight<=v<=c.config.max_weight for v in d.weights.values()))
        again=c.choose(snapshot(now=101,inventory=()),allow_probe=True)
        self.assertFalse(again.changed);self.assertEqual(again.reason,'cooldown')

    def test_response_context_and_age_are_not_transferred(self):
        c=CycleServiceController();teach(c)
        self.assertEqual(c.choose(snapshot(context='joint')).reason,'unmeasured_response')
        self.assertEqual(c.choose(snapshot(now=3701)).reason,'unmeasured_response')

    def test_ewma_and_completed_sample_identity(self):
        c=CycleServiceController(config=ControllerConfig(min_response_samples=1,ewma_alpha=0.5))
        x=ResponseSample(BASE,'policy_train',1,1,progress_kind='policy_training',observed_at=99,sample_id='x')
        self.assertTrue(c.observe(x));self.assertFalse(c.observe(x))
        with self.assertRaises(ValueError):c.observe(replace(x,completed_units=2))
        c.observe(replace(x,completed_units=3,observed_at=100,sample_id='y'))
        self.assertEqual(c._rate(snapshot(),BASE,'policy_train','policy_training'),2)

    def test_zero_progress_is_recorded_not_infinite_throughput(self):
        c=CycleServiceController();teach(c,rates={'world_training':0})
        d=c.choose(snapshot())
        self.assertEqual(d.reason,'observed_no_progress');self.assertIsNone(d.predicted_cycle_seconds)

    def test_response_model_is_bounded(self):
        c=CycleServiceController(config=ControllerConfig(max_response_contexts=2))
        for i in range(6):
            c.observe(ResponseSample(BASE,'policy_train',1,1,context=str(i),observed_at=i))
        self.assertEqual(c.snapshot()['response_points'],2)

    def test_bad_values_and_inconsistent_units_fail(self):
        for value in [math.nan,math.inf,-1,True]:
            with self.subTest(value=value),self.assertRaises(ValueError):
                ResponseSample(BASE,'policy_train',1,value)
        with self.assertRaises(ValueError):ResponseSample(BASE,'policy_train',1,1,progress_kind='imagined_chunks')
        with self.assertRaises(ValueError):CycleSnapshot(100,6,0)
        with self.assertRaises(ValueError):CycleSnapshot(100,4,0,remaining_policy_updates=2)
        with self.assertRaises(ValueError):InventoryBucket('bad','imagined',1,4)
        with self.assertRaises(ValueError):InventoryBucket('bad','real',1,4,successful_uses=4,reserved_uses=1)
        with self.assertRaises(ValueError):ControllerConfig(min_weight=2)
        with self.assertRaises(ValueError):CycleServiceController(initial_weights=dict(BASE,policy_train=2))


if __name__=='__main__':unittest.main()
