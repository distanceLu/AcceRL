import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from policy_batch_receipts import validate_policy_batch_mode_receipts, validate_policy_batch_modes_from_db
from policy_joint_batch_mode import mode_for_step, expected_completed_microsteps


def receipts(steps=15):
    rows = []
    for step in range(1, steps+1):
        micro, gas = mode_for_step(step)
        old_micro, old_gas = mode_for_step(step - 1) if step > 1 else (8, 18)
        for rank in range(4):
            base = {'policy_step': step, 'rank': rank, 'world_size': 4, 'global_batch': 576,
                    'local_batch': 144, 'joint_due': step % 5 == 0, 'micro': micro, 'gas': gas,
                    'old_micro': old_micro, 'old_gas': old_gas,
                    'before_micro_steps': expected_completed_microsteps(step - 1),
                    'before_global_steps': step - 1, 'before_global_samples': (step - 1) * 576}
            for transition in ['begin', 'complete']:
                details = dict(base, transition=transition)
                if transition == 'complete':
                    details.update(after_micro_steps=expected_completed_microsteps(step),
                                   after_global_steps=step, after_global_samples=step * 576)
                rows.append({'run_id': 'fixture', 'actor': 'policy_trainer',
                             'actor_id': str(rank), 'pid': 1000 + rank, 'details': details})
    return rows


class ReceiptTests(unittest.TestCase):
    def test_all120_receipts_and324_real_microsteps(self):
        value = validate_policy_batch_mode_receipts(receipts(), run_id='fixture',steps=15)
        self.assertTrue(value['valid'])
        self.assertEqual(value['mode_receipts'], 120)
        self.assertEqual(value['micro_steps_per_rank'], 324)
        self.assertEqual(value['global_exposures'], 8640)

    def test_missing_duplicate_foreign_and_changed_rank_identity_rejected(self):
        correct = receipts()
        for invalid in [correct[:-1], correct + [correct[0]],
                        [{**correct[0], 'run_id': 'other'}, *correct[1:]],
                        [{**correct[0], 'actor': 'ctrl_trainer'}, *correct[1:]],
                        [{**correct[0], 'pid': 1001}, *correct[1:]]]:
            with self.assertRaises(RuntimeError):
                validate_policy_batch_mode_receipts(invalid, run_id='fixture',steps=15)

    def test_joint_must_be4_36_and_ordinary8_18_with_cumulative_truth(self):
        for index, field, wrong in [(32, 'micro', 8), (33, 'after_micro_steps', 90),
                                    (0, 'gas', 36), (0, 'local_batch', 72),
                                    (0, 'before_global_samples', 576)]:
            rows = receipts()
            rows[index]['details'][field] = wrong
            with self.assertRaises(RuntimeError):
                validate_policy_batch_mode_receipts(rows, run_id='fixture',steps=15)

    def test_manual_longer_than_200_receipts_and_zero_update_fill_stop(self):
        value=validate_policy_batch_mode_receipts(iter(receipts(57)),run_id='fixture',steps=57)
        self.assertEqual(value['mode_receipts'],456)
        self.assertEqual(value['micro_steps_per_rank'],1224)
        zero=validate_policy_batch_mode_receipts(iter(()),run_id='fixture',steps=0)
        self.assertTrue(zero['valid'])
        self.assertEqual(zero['mode_receipts'],0)
        with self.assertRaises(RuntimeError):
            validate_policy_batch_mode_receipts(iter(receipts(1)),run_id='fixture',steps=0)

    def test_streaming_sqlite_reads_all_and_catches_bad_receipt_after_old_cap(self):
        with tempfile.TemporaryDirectory() as raw:
            db=Path(raw)/'collection.sqlite'
            c=sqlite3.connect(db)
            c.execute('CREATE TABLE events(id INTEGER PRIMARY KEY,run_id TEXT,ts_ns INTEGER,actor TEXT,actor_id TEXT,pid INTEGER,kind TEXT,workload TEXT,details_json TEXT)')
            for index,row in enumerate(receipts(57),1):
                c.execute('INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?)',(index,'fixture',index,row['actor'],row['actor_id'],row['pid'],'policy_batch_mode',None,json.dumps(row['details'])))
            c.commit()
            result=validate_policy_batch_modes_from_db(db,run_id='fixture',steps=57)
            self.assertEqual(result['mode_receipts'],456)
            row=json.loads(c.execute('SELECT details_json FROM events WHERE id=450').fetchone()[0]);row['gas']=36
            c.execute('UPDATE events SET details_json=? WHERE id=450',(json.dumps(row),));c.commit();c.close()
            before=db.read_bytes()
            with self.assertRaises(RuntimeError):validate_policy_batch_modes_from_db(db,run_id='fixture',steps=57)
            self.assertEqual(db.read_bytes(),before)

    def test_reordered_and_incomplete_macro_fail(self):
        rows=receipts(2)
        rows[0],rows[1]=rows[1],rows[0]
        with self.assertRaises(RuntimeError):validate_policy_batch_mode_receipts(rows,run_id='fixture',steps=2)
        with self.assertRaises(RuntimeError):validate_policy_batch_mode_receipts(receipts(2)[:-1],run_id='fixture',steps=2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
