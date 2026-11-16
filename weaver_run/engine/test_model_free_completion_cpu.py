"""Final CPU audits and ordinary Policy batches for model-free runs."""
import copy
import json
import os
from pathlib import Path
import sqlite3
from types import SimpleNamespace
import tempfile
import unittest

import torch

from parallel_checkpoint import export_split_component, save_split_weights
from policy_batch_receipts import validate_policy_batch_mode_receipts, validate_policy_batch_modes_from_db
from policy_joint_batch_mode import PolicyJointBatchMode
from test_manual_controller_cpu import load
from test_model_free_resources_cpu import CONFIG, PLAN, TinyPolicy, metadata
from test_policy_joint_batch_mode_cpu import CpuEngine, trainer, run_phase


def ordinary_rows(steps=6):
    rows = []
    for rank in range(4):
        owner = trainer(CpuEngine())
        owner.rank = rank
        owner.world_config = SimpleNamespace(backend='model_free')
        def emit(kind, *, details, rank=rank):
            rows.append(dict(run_id='cpu', actor='policy_trainer', actor_id=str(rank),
                             pid=1000+rank, details=details))
        helper = PolicyJointBatchMode(owner, writer=SimpleNamespace(event=emit), verify_sources=False)
        for step in range(1, steps+1):
            descriptor = helper.begin(step)
            if (descriptor['micro'], descriptor['gas'], descriptor['joint_due']) != (8, 18, False):
                raise AssertionError('Model-free Policy changed its ordinary 8/18 batch at step '+str(step))
            boundaries, used, mean = run_phase(owner, step)
            if boundaries != [18] or used != list(range(144)) or owner.model.optimizer.committed_gradients[-1] != mean:
                raise AssertionError('Ordinary Policy update lost its full cohort or mean-gradient boundary')
            helper.complete(step)
        if owner.model.micro_steps != steps * 18:
            raise AssertionError('Model-free accumulated phantom joint microsteps')
    return rows


class OrdinaryPolicyBatch(unittest.TestCase):
    def test_step_five_stays_ordinary_and_actual_helper_receipts_pass_cpu_audit(self):
        rows = ordinary_rows()
        checked = validate_policy_batch_mode_receipts(rows, run_id='cpu', steps=6, model_free=True)
        self.assertEqual(checked['micro_steps_per_rank'], 108)
        self.assertEqual(checked['mode_receipts'], 48)
        self.assertEqual(checked['global_exposures'], 3456)
        bad = copy.deepcopy(rows)
        bad[8]['details']['joint_due'] = True
        with self.assertRaises(RuntimeError):
            validate_policy_batch_mode_receipts(bad, run_id='cpu', steps=6, model_free=True)
        bad = copy.deepcopy(rows)
        bad[9]['details']['after_micro_steps'] = 108
        with self.assertRaises(RuntimeError):
            validate_policy_batch_mode_receipts(bad, run_id='cpu', steps=6, model_free=True)

    def test_database_audit_keeps_the_explicit_model_free_schedule(self):
        rows = ordinary_rows()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'receipts.sqlite'
            conn = sqlite3.connect(path)
            conn.execute('CREATE TABLE events(id INTEGER PRIMARY KEY,run_id TEXT,ts_ns INTEGER,actor TEXT,actor_id TEXT,pid INTEGER,kind TEXT,workload TEXT,details_json TEXT)')
            for index, row in enumerate(rows, 1):
                conn.execute('INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?)',
                    (index, 'cpu', index, row['actor'], row['actor_id'], row['pid'],
                     'policy_batch_mode', None, json.dumps(row['details'])))
            conn.commit()
            conn.close()
            self.assertEqual(validate_policy_batch_modes_from_db(path, run_id='cpu', steps=6,
                             model_free=True)['micro_steps_per_rank'], 108)
            with self.assertRaises(RuntimeError):
                validate_policy_batch_modes_from_db(path, run_id='cpu', steps=6)


class ModelFreeFinalAudit(unittest.TestCase):
    def state(self, steps):
        return dict(run_id='model-free-cpu', service_plan=copy.deepcopy(PLAN),
            world_backend_config=copy.deepcopy(CONFIG),
            report={'training': {'policy': {'global_updates': steps}}})

    def save(self, root, steps=6):
        owner = SimpleNamespace(rank=0, world_size=4, training_role='policy',
                                global_step=steps, model=TinyPolicy())
        meta = metadata()
        meta['stop_policy'] = dict(until_manual_stop=True, until_deadline=False,
                                  deadline_unix_seconds=None, stop_on_target=False)
        return save_split_weights({'policy': export_split_component(owner, 'policy')},
            root / 'training/checkpoints', reason='final', metadata=meta, active_kinds=('policy',))

    def audit(self, root, state):
        namespace = {'os': os, 'json': json, 'legacy': SimpleNamespace(stamp=lambda: 'cpu')}
        audit = load('gpu47_safety.py', 'audit_final_weights', namespace)
        return audit(root, state, torch_module=torch, persist=False)

    def test_real_cpu_artifact_accepts_zero_and_ordinary_steps_without_a_world_artifact(self):
        for steps in (0, 6):
            with self.subTest(steps=steps), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.save(root, steps)
                checked = self.audit(root, self.state(steps))
                self.assertTrue(checked['valid'], checked)
                self.assertEqual(checked['steps'], {'policy': steps})
                self.assertEqual(checked['monolith_steps'], {'policy': steps})
                self.assertEqual(set(checked['tensor_counts']), {'policy'})
                self.assertNotIn('frozen_world', checked)
                self.assertNotIn('online_world', checked)

    def test_final_audit_rejects_world_metadata_or_nonfinite_policy(self):
        for mutation in ('world', 'plan', 'config', 'nonfinite'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.save(root)
                folder = root / 'training/checkpoints'
                path = folder / 'latest_weights.pt'
                payload = torch.load(path, map_location='cpu', weights_only=True)
                meta = payload['manifest']['metadata']
                if mutation == 'world': meta['frozen_world'] = {'version': 0}
                if mutation == 'plan': meta['service_plan']['wm_consumers'] = ['ctrl']
                if mutation == 'config': meta['world_backend_config']['world_trainable'] = True
                if mutation == 'nonfinite':
                    next(iter(payload['models']['policy']['state_dict'].values())).fill_(float('nan'))
                torch.save(payload, path)
                manifest = dict(payload['manifest'], weights_file='latest_weights.pt', weights_bytes=path.stat().st_size)
                (folder / 'manifest.json').write_text(json.dumps(manifest))
                self.assertFalse(self.audit(root, self.state(6))['valid'])

    def test_placement_requires_eight_policy_processes_and_no_ghost_world(self):
        rows = []
        for rank in range(4):
            for offset, role in enumerate(('policy_trainer', 'policy_infer')):
                rows.append(dict(actor=role, actor_id=str(rank), pid=100+2*rank+offset, details={
                    'role': role, 'physical_gpus': [{'physical_index': rank+4, 'uuid': f'GPU-{rank}'}],
                    'physical_gpu_uuid': f'GPU-{rank}', 'cpu_threads': 1}))
        check = load('gpu47_safety.py', 'placement_evidence',
                     {'safe': SimpleNamespace(query_events=lambda *a, **kw: rows)})
        self.assertTrue(check(None, 'cpu', service_plan=PLAN)['complete_expected_placement'])
        rows.append(dict(actor='ctrl_trainer', actor_id='0', pid=999, details={
            'role': 'ctrl_trainer', 'physical_gpus': [{'physical_index': 4, 'uuid': 'GPU-0'}],
            'physical_gpu_uuid': 'GPU-0', 'cpu_threads': 1}))
        self.assertFalse(check(None, 'cpu', service_plan=PLAN)['complete_expected_placement'])


if __name__ == '__main__':
    unittest.main()
