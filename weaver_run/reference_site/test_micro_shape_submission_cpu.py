import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from weaver_run.reference_site import jit_bench_queue as queue
import launch
from weaver_run.reference_site.dispatcher.store import QueueStore


class MicroShapeSubmissionTest(unittest.TestCase):
    def test_pinned_job_and_actual_managed_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cli = Mock(EXPERIMENTS=base/'experiments', PARENT=base, HOME=base, QUEUE=base/'queue',
                       CHECKOUT=queue.REPO, RUNTIME=base/'active-runtime', PYTHON=sys.executable)
            cli.EXPERIMENTS.mkdir()
            cli.LABEL = re.compile(r'[a-z0-9-]+')
            cli.require = launch.require
            cli.lookup.return_value = None
            cli.published.return_value = 'a'*40
            cli.create.return_value = {'id':'test-id','state':'queued'}
            cli.PROFILES = {'restore': {'mps_receipt': hashlib.sha256(b'{}').hexdigest()}}
            (base/'mps.json').write_bytes(b'{}')

            def pin(source, runtime, commit):
                recipe = runtime/'weaver_run/recipe'
                recipe.mkdir(parents=True)
                (recipe/'visualization.json').write_bytes(b'{}')
            cli.pin_worktree.side_effect = pin
            label = 'mf8-micro16-24-p189-1127-a'
            cli.offline_check.return_value = dict(cpu_admission='passed',isolated_queue='passed',production_submission=False)
            with patch.object(queue,'verified_checkpoint') as checkpoint, patch(
                    'weaver_run.reference_site.dispatcher.store.QueueStore',
                    side_effect=lambda root: QueueStore(root,allowed_cwd_root=base)):
                dry = queue.submit(cli, label, shapes=True, check_only=True)
                self.assertFalse(dry['production_submission'])
                self.assertEqual(dry['benchmark_queue'],'passed')
                self.assertFalse((cli.EXPERIMENTS/label).exists())
                cli.create.assert_not_called()
                cli.pin_worktree.assert_not_called()
                receipt = queue.submit(cli, label, shapes=True)
                self.assertEqual(checkpoint.call_count,2)
            self.assertEqual(receipt['submission_id'],'weaver-bench-shapes-'+label)
            cli.ready.assert_not_called()
            root = cli.EXPERIMENTS/label
            job = json.loads((root/'experiment.json').read_bytes())['job']
            self.assertEqual(Path(job['cwd']),base/'library/jit-bench-code'/('a'*40))
            self.assertEqual(job['argv'][-1],str(root/'plan.json'))
            plan = json.loads((root/'plan.json').read_bytes())
            self.assertEqual(plan['comparison'],'micro_shapes')
            self.assertEqual(plan['train_shapes'],[[16,9],[24,6]])
            self.assertEqual([plan[x] for x in ('parity_pairs','warmup_pairs','measured_pairs')],[1,1,4])
            self.assertEqual(plan['global_batch'],576)
            env = {'ACCERL_JIT_BENCH_PLAN':str(root/'plan.json'),'ACCERL_MODEL_FREE_REAL_WORKERS':'8',
                   'ACCERL_FINAL_CHECKPOINT':str(root/'checkpoints'),'ACCERL_PIPELINE_CONFIG':str(root/'git-pipeline.json')}
            with patch.dict(os.environ,env,clear=True), patch.object(os,'chdir'), patch.object(os,'execve') as train:
                launch.exec_managed_driver(queue.benchmark_argv(plan)[2:])
            _, argv, effective = train.call_args.args
            parser = argparse.ArgumentParser(add_help=False)
            expected={'--policy-lr':0.,'--value-lr':0.,'--num-rollout-workers':8,
                      '--num-trainer-gpus':4,'--num-eval-workers':0}
            for key in expected:
                parser.add_argument(key,type=float)
            opts, _ = parser.parse_known_args(argv[2:])
            self.assertEqual(vars(opts),{key[2:].replace('-','_'):value for key,value in expected.items()})
            for key, value in {'ACCERL_COMPILE_TRAINER':'1','ACCERL_TRAINER_MPS_PERCENT':'90',
                               'ACCERL_COMPILE_ACTION_INFERENCE':'1','ACCERL_FAST_JIT_BOUNDARIES':'1',
                               'ACCERL_TRIM_MICRO_CACHE':'0','ACCERL_TRIM_IDLE_CACHE':'0',
                               'ACCERL_REAL_SAMPLING':'uniform_blocks'}.items():
                self.assertEqual(effective[key],value)

    def test_hardware_receipt_requires_four_physical_pairs_and_inductor_tf32(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'fixture.sqlite'
            trainers=[dict(uuid=str(i),physical_gpu=str(i),visible_sm_count=70,mps_client_percent=90)
                      for i in range(4,8)]
            inference=[dict(row,visible_sm_count=78,mps_client_percent=100) for row in trainers]
            binding=dict(trainer_mps_percent=90,trainers=trainers,policy_inference=inference)
            compiled=dict(enabled=True,compiled=True,backend='inductor',tf32_enabled=True,dynamic=False,
                          llama_layers=32,vision_blocks=51,projector_compiled=True)
            with closing(sqlite3.connect(path)) as db, db:
                db.execute('CREATE TABLE events(run_id,kind,details_json)')
                rows=[('trial','trainer_mps_verified',json.dumps(binding))]
                rows.extend(('trial','policy_trainer_compiled',json.dumps(dict(compiled,rank=i))) for i in range(4))
                db.executemany('INSERT INTO events VALUES(?,?,?)',rows)
            receipt=queue.verified_micro_shape_hardware(path,'trial')
            self.assertEqual(receipt['mps'],binding)
            with closing(sqlite3.connect(path)) as db, db:
                db.execute("UPDATE events SET details_json=? WHERE kind='policy_trainer_compiled' AND rowid=2",
                           [json.dumps(dict(compiled,tf32_enabled=False,rank=0))])
            with self.assertRaisesRegex(RuntimeError,'Inductor/TF32'):
                queue.verified_micro_shape_hardware(path,'trial')


if __name__ == '__main__':
    unittest.main()
