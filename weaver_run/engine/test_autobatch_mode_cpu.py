"""CPU execution of audited DeepSpeed methods and adaptive receipt invariants."""
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest

from policy_joint_batch_mode import PolicyJointBatchMode
from policy_batch_receipts import validate_policy_batch_mode_receipts, validate_policy_batch_modes_from_db
from test_policy_joint_batch_mode_cpu import CpuEngine, trainer, run_phase


def adaptive_rows(schedules):
    rows = []
    for rank, schedule in enumerate(schedules):
        cumulative, previous = 0, (8, 18)
        for step, micro in enumerate(schedule, 1):
            gas = 144 // micro
            record = dict(autotune=True, policy_step=step, rank=rank, world_size=4,
                global_batch=576, local_batch=144, joint_due=False, micro=micro, gas=gas,
                old_micro=previous[0], old_gas=previous[1], before_micro_steps=cumulative,
                before_global_steps=step-1, before_global_samples=576*(step-1))
            for transition in ('begin', 'complete'):
                details = dict(record, transition=transition)
                if transition == 'complete':
                    details.update(after_micro_steps=cumulative+gas,
                        after_global_steps=step, after_global_samples=576*step)
                rows.append(dict(run_id='adaptive', actor='policy_trainer',
                    actor_id=str(rank), pid=1000+rank, details=details))
            cumulative += gas
            previous = micro, gas
    # Rank-major records intentionally test constant-memory cross-rank checking.
    return rows


class AdaptiveModeTests(unittest.TestCase):
    def make(self):
        engine = CpuEngine()
        owner = trainer(engine)
        owner.world_config = SimpleNamespace(backend='model_free')
        return owner, PolicyJointBatchMode(owner, verify_sources=False, autotune=True)

    def test_six_update_windows_keep_complete_mean_gradient_and_counter_truth(self):
        owner, modes = self.make()
        schedule = [8]*6 + [16]*6 + [24]*6 + [8]*6
        for step, micro in enumerate(schedule, 1):
            receipt = modes.begin(step, micro=micro)
            boundaries, used, mean_gradient = run_phase(owner, step)
            self.assertEqual(boundaries, [144//micro])
            self.assertEqual(used, list(range(144)))
            self.assertEqual(owner.model.optimizer.committed_gradients[-1], mean_gradient)
            self.assertTrue(receipt['autotune'])
            modes.complete(step)
        self.assertEqual(owner.model.micro_steps, 306)
        self.assertEqual(owner.model.global_samples, 13824)
        self.assertEqual(owner.model.gas_boundary_ctr, 24)

    def test_unsafe_cumulative_switch_is_rejected_without_mutating_engine(self):
        owner, modes = self.make()
        for step, micro in enumerate((8,16),1):
            modes.begin(step, micro=micro)
            run_phase(owner, step)
            modes.complete(step)
        self.assertEqual(owner.model.micro_steps,27)
        with self.assertRaisesRegex(RuntimeError, 'Unsafe'):
            modes.begin(3, micro=24)
        self.assertEqual(owner.model.train_micro_batch_size_per_gpu(),16)
        self.assertEqual(owner.model.micro_steps,27)

    def test_only_opted_in_model_free_and_reviewed_micros(self):
        plain = trainer(CpuEngine())
        with self.assertRaises(ValueError):
            PolicyJointBatchMode(plain, verify_sources=False, autotune=True)
        owner, modes = self.make()
        for micro in (True, 0, 12, 32, 8.0):
            with self.assertRaises(ValueError):
                modes.begin(1,micro=micro)
        legacy = PolicyJointBatchMode(plain, verify_sources=False)
        with self.assertRaises(ValueError):
            legacy.begin(1,micro=16)

    def test_adaptive_counter_or_previous_configuration_drift_is_rejected(self):
        owner, modes = self.make()
        modes.begin(1,micro=8)
        run_phase(owner,1)
        modes.complete(1)
        owner.model.micro_steps = 36
        with self.assertRaisesRegex(RuntimeError,'cumulative'):
            modes.begin(2,micro=16)


class AdaptiveReceiptTests(unittest.TestCase):
    def check(self, rows, steps):
        return validate_policy_batch_mode_receipts(iter(rows),run_id='adaptive',steps=steps,
            model_free=True,autotune=True)

    def test_records_actual_cumulative_not_eighteen_times_updates(self):
        schedule=[8]*6+[16]*6+[24]*6+[8]*6
        report=self.check(adaptive_rows([schedule]*4),24)
        self.assertTrue(report['valid'])
        self.assertEqual(report['micro_steps_per_rank'],306)
        self.assertEqual(report['global_exposures'],13824)

    def test_rank_divergence_even_with_same_final_counter_is_rejected(self):
        schedules=[[8]*6+[16]*6,[16]*6+[8]*6,[8]*6+[16]*6,[8]*6+[16]*6]
        with self.assertRaisesRegex(RuntimeError,'rank|schedule'):
            self.check(adaptive_rows(schedules),12)

    def test_unsafe_boundaries_and_forged_complete_or_mode_are_rejected(self):
        with self.assertRaises(RuntimeError):
            self.check(adaptive_rows([[8,16,24]]*4),3)
        for key, value in [('micro',24),('after_micro_steps',999),('autotune',False)]:
            rows=adaptive_rows([[8,16,16]]*4)
            rows[1]['details'][key]=value
            with self.assertRaises(RuntimeError):
                self.check(rows,3)
        with self.assertRaises(ValueError):
            validate_policy_batch_mode_receipts([],run_id='adaptive',steps=0,autotune=True)

    def test_legacy_audit_cannot_accept_adaptive_receipts(self):
        with self.assertRaises(RuntimeError):
            validate_policy_batch_mode_receipts(adaptive_rows([[8]*6]*4),
                run_id='adaptive',steps=6,model_free=True)

    def test_sqlite_entry_point_requires_opt_in_and_checks_entire_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'collection.sqlite'
            connection=sqlite3.connect(path)
            connection.execute('CREATE TABLE events(id INTEGER PRIMARY KEY,run_id TEXT,ts_ns INTEGER,'
                'actor TEXT,actor_id TEXT,pid INTEGER,kind TEXT,workload TEXT,details_json TEXT)')
            rows=adaptive_rows([[8]*6+[16]*6+[24]*6]*4)
            for index,row in enumerate(rows,1):
                connection.execute('INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?)',
                    (index,'adaptive',index,row['actor'],row['actor_id'],row['pid'],
                     'policy_batch_mode',None,json.dumps(row['details'])))
            connection.commit()
            report=validate_policy_batch_modes_from_db(path,run_id='adaptive',steps=18,
                model_free=True,autotune=True)
            self.assertEqual(report['micro_steps_per_rank'],198)
            self.assertEqual(report['final_policy_micro_GAS'],[24,6])
            with self.assertRaises(RuntimeError):
                validate_policy_batch_modes_from_db(path,run_id='adaptive',steps=18,model_free=True)
            forged=dict(rows[-1]['details'],after_global_samples=999)
            connection.execute('UPDATE events SET details_json=? WHERE id=?',
                (json.dumps(forged),len(rows)))
            connection.commit();connection.close()
            before=path.read_bytes()
            with self.assertRaises(RuntimeError):
                validate_policy_batch_modes_from_db(path,run_id='adaptive',steps=18,
                    model_free=True,autotune=True)
            self.assertEqual(path.read_bytes(),before)

    def test_empty_stop_is_valid(self):
        result=self.check([],0)
        self.assertEqual(result['micro_steps_per_rank'],0)
        self.assertTrue(result['stopped_before_first_update'])


if __name__ == '__main__':
    unittest.main()
