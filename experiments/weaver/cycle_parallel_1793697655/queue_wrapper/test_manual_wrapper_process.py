"""Real CPU subprocess controller test; never loads torch or queries a GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import training_entry as train
from test_manual_wrapper import mps_authorization, mps_receipt


@unittest.skipUnless(sys.platform == 'linux' and getattr(os, 'getuid', lambda: -1)() == 1004,
                     'production Linux UID1004 birth checks required')
class RealManualControllerTests(unittest.TestCase):
    def test_one_time_preflight_real_cpu_controller_eight_children_and_no_gpu_queries(self):
        # Only the smoke workload/GPU observations are replaced. The real
        # private marker, PID births, subprocesses, persistent claim, lifecycle
        # and duplicate-launch rejection execute unmodified.
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve(); source=root/'source';source.mkdir();output=root/'output';output.mkdir()
            (source/'examples').mkdir()
            (source/train.BACKEND_CONFIG).write_text(json.dumps({
                'backend_options':{'training_python':sys.executable}}))
            (output/'mps.json').write_text(json.dumps(mps_receipt()))
            wrapper=str(Path(__file__).resolve().parent)
            smoke=r'''
import argparse,json,os,subprocess,sys
from pathlib import Path
sys.path.insert(0,WRAPPER)
from test_manual_wrapper import preflight_report
p=argparse.ArgumentParser();p.add_argument('--plan',type=Path);a=p.parse_args()
plan=json.loads(a.plan.read_text());a.output_root=Path(plan['output']);a.run_id=plan['run_id'];a.output_root.mkdir()
processes=[subprocess.Popen([sys.executable,'-c','import time;time.sleep(.15)']) for _ in range(8)]
rows=[]
for index,process in enumerate(processes):
 raw=(Path('/proc')/str(process.pid)/'stat').read_text().rsplit(')',1)[1].split()
 rows.append({'pid':process.pid,'uid':os.getuid(),'birth':int(raw[19]),'role':'trainer' if index<4 else 'inference','rank':index%4})
for process in processes:process.wait()
report=preflight_report(a.run_id)
for actual,old in zip(rows,report['child_identities']):actual['gpu_uuid']=old['gpu_uuid']
report['child_identities']=rows;report['scope']='CPU fixture, mocked GPU workload and observations'
for name in ('report.json','completed.json'):(a.output_root/name).write_text(json.dumps(report))
'''.replace('WRAPPER',repr(wrapper))
            (source/'weaver_online_smoke.py').write_text(textwrap.dedent(smoke))
            args=SimpleNamespace(source=source,root=output)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',EXPERIMENT_QUEUE_JOB_ID='native-preflight-cpu')
            idle=Mock()
            with patch.object(train,'wait_idle',idle), patch.object(train,'mps_authorization',return_value=mps_authorization()):
                receipt=train.run_native_preflight(args,env)
            self.assertEqual(receipt['state'],'passed')
            self.assertTrue(receipt['all_owned_exited'])
            self.assertEqual(receipt['idle_checks_after_preflight'],2)
            self.assertEqual(idle.call_count,2)
            self.assertLess(receipt['elapsed_seconds'],180)
            with self.assertRaises(FileExistsError):train.run_native_preflight(args,env)

    def test_real_child_waits_for_user_flag_and_accepts_boundary_final(self):
        # Admission and manifest checks are replaced with CPU fixture checks;
        # process creation, file polling, stop authentication, final receipts,
        # and exact UID/PID/birth observation use the real implementation.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source'
            source.mkdir()
            output = root / 'out'
            output.mkdir()
            # Extend the legacy process fixture with an immutable scheduler
            # profile, preserving the real new scheduler state-admission check.
            import hashlib
            candidate = Path(os.environ.get('COST_CANDIDATE_SOURCE', os.environ.get(
                'FRESH_CANDIDATE_SOURCE', Path(__file__).resolve().parents[1] / 'staged_v1/weaver_integrated_v1')))
            raw = (candidate / train.COST_SCHEDULER_CONFIG).read_bytes()
            (source / 'examples').mkdir()
            (source / train.COST_SCHEDULER_CONFIG).write_bytes(raw)
            (source / 'source_manifest.json').write_text(json.dumps({'files': {
                train.COST_SCHEDULER_CONFIG: hashlib.sha256(raw).hexdigest()}}))
            runner = r'''
import argparse,json,os,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--mode');p.add_argument('--data',type=Path);p.add_argument('--pipeline-config');p.add_argument('--world-backend-config');p.add_argument('--cost-scheduler-config',type=Path);a=p.parse_args()
data=a.data;data.mkdir();(data/'training').mkdir()
f=(Path('/proc')/str(os.getpid())/'stat').read_text().rsplit(')',1)[1].split()
who={'pid':os.getpid(),'uid':os.getuid(),'birth_ticks':int(f[19])}
rid='weaverintegrated_cpu_fixture';steps={'policy':0,'ctrl':0}
state={'candidate':'weaver_integrated_v1','source_path':str(Path.cwd()),'experiment_root':str(data.parent),'run_id':rid,'uid':1004,'mode':'manual','until_manual_stop':True,'stop_on_target':False,'deadline_unix':None,'supervisor':who,'state':'running'}
import hashlib
raw=a.cost_scheduler_config.read_bytes();digest=hashlib.sha256(raw).hexdigest()
(data/'cost_scheduler_config.json').write_bytes(raw)
state['cost_scheduler']={'profile':'03_feedback_parallel','source_path':str(a.cost_scheduler_config),'source_sha256':digest,'configuration_path':str(data/'cost_scheduler_config.json'),'configuration_sha256':digest,'settings':json.loads(raw)}
(data/'runnerstatus.json').write_text(json.dumps(state));(data/'cpu_ready').write_text('ready')
while not (data/'training/stop.flag').exists(): time.sleep(.02)
world={'protocol':'weaver_online_joint_weights_v1','backend':'weaver','world_trainable':True,'source_optimizer_steps':3200,'source_bundle_sha256':'6b45705115034914803224c3653533e30236fffdac544b46400b5af68bfd3443','run_id':rid,'world_size':4,'policy_step':0,'successful_updates':0,'training_steps':0,'model_version':0,'rm_updates':0,'weights_only':True,'map_location':'cpu','mmap':True,'valid':True,'finite':True,'tensor_count':866,'weights_sha256':'a'*64,'cuda_initialized':False,'python':'/test/independent/python'}
world.update(native_reward_head=FRESH_NATIVE_REWARD, reward_target_counts={'scope':'global_selected_frames_once_per_successful_world_update','valid_count':0,'masked_count':0,'disabled_updates':0,'target_sum':0.})
state.update(state='stopped_by_user',training_stopped=True,cleanup={'survivors':[]},manual_completion={'steps':steps,'checkpoint_reason':'final_before_first_update'},checkpoint_cpu_audit={'valid':True,'steps':steps,'monolith_steps':{'policy':0},'manifest_reason':'final_before_first_update','online_world':world})
report={'run_id':rid,'state':'stopped_by_user','source':str(Path.cwd()),'training':{'policy':{'global_updates':0},'ctrl':{'global_updates':0}},'experiment_completed':[{'details':{'global_step':0}}]}
(data/'runnerstatus.json').write_text(json.dumps(state));(data/'reportstatus.json').write_text(json.dumps(report))
'''
            (source / 'runner.py').write_text(textwrap.dedent(runner).replace('FRESH_NATIVE_REWARD',repr(train.NATIVE_REWARD)))
            driver = root / 'driver.py'
            wrapper = Path(__file__).resolve().parent
            driver.write_text(textwrap.dedent(f'''
                import sys
                from pathlib import Path
                from types import SimpleNamespace
                sys.path.insert(0,{str(wrapper)!r})
                import training_entry as q
                q.validate=lambda *args: {{'cpu_fixture':True}}
                q.verify_manifest=lambda *args: {{'cpu_fixture':True}}
                q.wait_idle=lambda *args: None
                q.run_native_preflight=lambda *args,**kwargs: {{'cpu_fixture':True,'state':'passed'}}
                raise SystemExit(q.run(SimpleNamespace(source=Path({str(source)!r}),root=Path({str(output)!r}),manifest_sha256='a'*64,post_stop_grace_seconds=900,validate_only=False,native_gpu_preflight=True)))
            '''))
            env = dict(os.environ, EXPERIMENT_QUEUE_JOB_ID='manual-cpu-fixture',
                       EXPERIMENT_QUEUE_MANUAL='1', EXPERIMENT_QUEUE_TIMEOUT_SECONDS='0',
                       CUDA_VISIBLE_DEVICES=','.join((
                           'GPU-c43d6671-0979-b38d-9eae-4fe0a9d71c8c',
                           'GPU-fe0edfe4-b193-1af7-6a98-cbff40214f74',
                           'GPU-8c9bc264-7f09-989e-1ec7-cecd6cde782c',
                           'GPU-07ae0bbc-8923-bd0c-199e-c726821e655d')),
                       CUDA_MPS_PIPE_DIRECTORY='/dev/shm/mps_lcx1_gpu47_pzero2_20261020/pipe')
            process = subprocess.Popen([sys.executable, str(driver)], env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                ready = output / 'formal_v3/cpu_ready'
                until = time.monotonic() + 8
                while not ready.exists() and process.poll() is None and time.monotonic() < until:
                    time.sleep(.03)
                self.assertTrue(ready.exists())
                time.sleep(1.2)
                self.assertIsNone(process.poll())
                receipt = json.loads((output / 'queue_receipt.json').read_text())
                self.assertIsNone(receipt['deadline_ns'])
                self.assertIsNone(receipt['stop_trigger'])
                flag = {'run_id': 'weaverintegrated_cpu_fixture', 'uid': 1004,
                        'reason': 'user_requested_stop', 'requested_ns': time.time_ns()}
                (output / 'formal_v3/training/stop.flag').write_text(json.dumps(flag))
                stdout, stderr = process.communicate(timeout=10)
                self.assertEqual(process.returncode, 0, (stdout, stderr))
                final = json.loads((output / 'queue_receipt.json').read_text())
                self.assertEqual(final['state'], 'succeeded')
                self.assertEqual(final['stop_evidence'], flag)
                self.assertTrue(final['all_owned_exited'])
                self.assertTrue(final['acceptance']['stopped_before_first_update'])
                self.assertNotIn('stop_ack', final)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.communicate(timeout=5)


if __name__ == '__main__':
    unittest.main()
