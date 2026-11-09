"""Regression: P10->P20 took 3621.8 seconds against a 3600-second baseline TTL.

Drive the actual broker window settlement and controller, with no Ray/Torch/GPU.
"""
import unittest
import asyncio
from copy import deepcopy

from cost_scheduler_runtime import CostAdmissionBroker, load_settings
from cycle_service_controller import CycleServiceController
from test_cost_scheduler_runtime_cpu import Clock, SOURCE, cycle, phase
from test_cycle_service_controller_cpu import BASE, snapshot, teach
from pathlib import Path


class ProbeTransactionTests(unittest.IsolatedAsyncioTestCase):
    async def make_probe(self):
        clock=Clock()
        cfg=load_settings(Path(__file__).parent/'examples/cost_03_adaptive.json')
        broker=CostAdmissionBroker(cfg,'probe-regression',SOURCE,clock=clock)
        await broker.update_cycle(cycle())
        for p in range(1,11):
            decision=await self.advance(broker,clock,p,100.)
        self.assertEqual(decision['reason'],'bounded_probe_without_speed_prediction')
        self.assertIsNotNone(broker.active_probe)
        self.assertEqual(broker.active_probe['baseline_weights'],BASE)
        return broker,clock

    async def advance(self,broker,clock,p,elapsed):
        clock.advance(elapsed)
        broker.refresh_completed+=100
        broker.action_observations_completed+=1000
        return await broker.update_cycle(cycle(policy=p,world=p//5,
            context='joint' if (p+1)%5==0 else 'ordinary',
            completed_context='joint' if p%5==0 else 'ordinary',
            admitted={'real':p,'imagined':p*4}))

    async def test_actual_3621_second_probe_expired_baseline_rolls_back_without_chaining(self):
        broker,clock=await self.make_probe()
        probe=deepcopy(broker.weights)
        for p in range(11,21):
            decision=await self.advance(broker,clock,p,362.18)
            if p<20:self.assertEqual(broker.weights,probe)
        self.assertEqual(broker.last_probe_result['reason'],'probe_rollback_timeout')
        self.assertFalse(broker.last_probe_result['accepted'])
        self.assertEqual(broker.weights,BASE)
        self.assertIsNone(broker.active_probe)
        self.assertEqual(broker.probe_anchor_cycle,4)

    async def test_same_observed_response_rolls_back_instead_of_next_blind_probe(self):
        broker,clock=await self.make_probe()
        for p in range(11,21):decision=await self.advance(broker,clock,p,100.)
        self.assertEqual(decision['reason'],'probe_rollback_no_measured_improvement')
        self.assertEqual(broker.weights,BASE)
        self.assertFalse(decision['probe_comparison']['accepted'])

    async def test_measured_gain_keeps_candidate_and_restarts_observation_hold(self):
        broker,clock=await self.make_probe();probe=deepcopy(broker.weights)
        for p in range(11,21):decision=await self.advance(broker,clock,p,50.)
        self.assertEqual(decision['reason'],'probe_measured_cycle_improvement')
        self.assertFalse(decision['changed'])  # keep; no new admission-vector change
        self.assertTrue(decision['probe_comparison']['accepted'])
        self.assertEqual(broker.weights,probe)
        self.assertIsNone(broker.active_probe)
        self.assertEqual(broker.probe_anchor_cycle,4)

    async def test_expiry_during_phase_calls_does_not_change_weights_mid_window(self):
        broker,clock=await self.make_probe();probe=deepcopy(broker.weights)
        clock.advance(4000.)
        await broker.publication(True)
        await broker.publication(False,policy_version=10,world_version=2)
        await broker.refresh_demand(2);await broker.refresh_demand(-2)
        self.assertEqual(broker.weights,probe)
        self.assertIsNotNone(broker.active_probe)
        # The first subsequent authoritative inventory/update boundary is safe
        # to end the expired probe, even before two complete cycles.
        await self.advance(broker,clock,11,50.)
        self.assertEqual(broker.last_probe_result['reason'],'probe_rollback_timeout')
        self.assertEqual(broker.weights,BASE)

    async def inventory(self,broker,policy,world):
        return await broker.update_inventory(dict(inventory_revision=broker.inventory_revision+1,
            inventory_versions=[policy,world],inventory=[],
            admitted_chunks={'real':policy,'imagined':policy*4}))

    async def test_backpressure_inventory_ends_expired_probe_without_two_cycles(self):
        broker,clock=await self.make_probe();events=[]
        broker.emit=lambda name,**kw:events.append((name,kw))
        before=deepcopy(broker.controller.responses)
        broker.refresh_completed+=25
        clock.advance(3601.)
        await self.inventory(broker,10,2)
        self.assertEqual(broker.weights,BASE)
        self.assertIsNone(broker.active_probe)
        self.assertEqual(broker.controller.responses,before)
        partial=broker.last_partial_response_window
        self.assertFalse(partial['valid_response_sample'])
        self.assertEqual(partial['progress']['critic_observations'],25)
        self.assertEqual(partial['end_versions'],[10,2])
        await self.inventory(broker,10,2)
        self.assertEqual(sum(name=='cost_probe_result' for name,_ in events),1)

    async def test_inflight_collective_and_unpublished_completion_both_defer_timeout(self):
        broker,clock=await self.make_probe();probe=deepcopy(broker.weights)
        spec=phase();spec.update(policy_version=10,world_version=2)
        grants=await asyncio.gather(*(broker.acquire_training(r,deepcopy(spec)) for r in range(4)))
        clock.advance(3601.)
        await self.inventory(broker,10,2)
        self.assertEqual(broker.weights,probe)
        self.assertIn('training_grant_inflight',broker.active_probe['timeout_blockers'])
        for rank in range(3):await broker.complete_training(rank,grants[0]['work_id'],1.)
        await self.inventory(broker,10,2)
        self.assertEqual(broker.weights,probe)
        await broker.complete_training(3,grants[0]['work_id'],1.)
        await self.inventory(broker,10,2)
        self.assertEqual(broker.weights,probe)
        self.assertIn('training_not_published',broker.active_probe['timeout_blockers'])
        await broker.publication(True)
        await self.inventory(broker,10,2)
        self.assertEqual(broker.weights,probe)
        await broker.publication(False,policy_version=11,world_version=2)
        await self.inventory(broker,11,2)
        self.assertEqual(broker.weights,BASE)
        self.assertEqual(broker.last_partial_response_window['progress']['policy_training'],1)
        self.assertEqual(broker.unpublished_training,{})

    async def test_partial_joint_progress_is_recorded_then_next_joint_is_valid(self):
        broker,clock=await self.make_probe();events=[]
        broker.emit=lambda name,**kw:events.append((name,kw))
        for p in range(11,15):await self.advance(broker,clock,p,100.)
        broker.refresh_completed+=25
        clock.advance(3601.)
        await self.inventory(broker,14,2)
        self.assertTrue(broker.skip_response_window)
        before=deepcopy(broker.controller.responses)
        await self.advance(broker,clock,15,100.)
        partial=[kw['details'] for name,kw in events if name=='cost_response_window_partial']
        self.assertEqual(len(partial),2)
        self.assertEqual(sum(w['progress']['world_training'] for w in partial),1)
        self.assertEqual(sum(w['progress']['policy_training'] for w in partial),1)
        self.assertEqual(sum(w['progress']['critic_observations'] for w in partial),125)
        self.assertTrue(all(w['valid_response_sample'] is False for w in partial))
        self.assertEqual(broker.controller.responses,before)
        for p in range(16,21):await self.advance(broker,clock,p,100.)
        key=(tuple(BASE[s] for s in BASE),'joint','world_train','world_training')
        self.assertGreater(broker.controller.responses[key]['count'],before[key]['count'])
        self.assertEqual(broker.weights,BASE)

    async def test_missing_candidate_measurements_also_terminates_probe(self):
        broker,clock=await self.make_probe()
        for p in range(11,20):await self.advance(broker,clock,p,100.)
        # Model capacity bound/expiry can remove a required response; do not
        # indefinitely hold an experiment while waiting for a complete forecast.
        broker.controller.responses.clear()
        decision=await self.advance(broker,clock,20,100.)
        self.assertEqual(decision['reason'],'probe_rollback_inconclusive')
        self.assertIsNone(decision['probe_comparison']['candidate_seconds'])
        self.assertEqual(broker.weights,BASE)

    def test_no_baseline_measurements_cannot_start_blind_probe(self):
        controller=CycleServiceController()
        decision=controller.choose(snapshot(inventory=()),allow_probe=True)
        self.assertFalse(decision.changed)
        self.assertEqual(controller.weights,BASE)

    def test_fresh_baseline_can_propose_bounded_probe(self):
        controller=CycleServiceController();teach(controller)
        decision=controller.choose(snapshot(inventory=()),allow_probe=True)
        self.assertTrue(decision.changed)
        self.assertIsNotNone(decision.baseline_cycle_seconds)
        self.assertIsNone(decision.predicted_cycle_seconds)


if __name__=='__main__':unittest.main()
