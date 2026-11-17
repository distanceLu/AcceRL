"""CPU contracts for the isolated legacy queue path; never contacts live queue/GPU."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import legacy_baseline as baseline
import legacy_queue_entry as entry


class Contracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.record = baseline.make_record((self.root/'frozen/code').resolve(), self.root, 'test-a', 'a'*40, 7200)
        self.audit_patch = patch.object(entry,'audit_weight_tensors',return_value={'finite':True})
        self.audit_patch.start()

    def tearDown(self):
        self.audit_patch.stop()
        self.temp.cleanup()

    def checkpoint(self):
        folder = self.root/'training/checkpoints/latest/agent_checkpoint_epoch_1'
        (folder/'agent_lora').mkdir(parents=True)
        (folder/'agent_lora/adapter_model.safetensors').write_bytes(b'weights')
        (folder/'agent_lora/adapter_config.json').write_text('{}')
        (folder/'agent_extra_layers.pt').write_bytes(b'extra')
        complete = dict(schema_version=1,status='completed',global_step=1,reason='deadline',
                        checkpoint_dir=str(folder),checkpoint_kind='inference_weights_only',
                        exact_resume=False,completed_updates=1,
                        consumed_training_samples=576,
                        run_id=self.record['run_id'],git_commit=self.record['git_commit'])
        entry.atomic(self.root/'training/driver_complete.json', complete)
        identity = {k:self.record[k] for k in ('run_id','git_commit')}
        entry.atomic(self.root/'training/driver_metadata.json',dict(identity,global_batch_size=576,
            placements=[dict(role='trainer',index=i,pid=10+i) for i in range(3)]+[
                        dict(role='inference',index=0,pid=13)]))
        (self.root/'training/updates.jsonl').write_text(json.dumps(dict(identity,global_step=1,
            samples_per_update=576))+'\n')
        return folder, complete

    def test_exact_configuration(self):
        argv = self.record['driver_argv']
        for key, value in {'--num-trainer-gpus':'3','--num-inference-actors':'1',
                           '--num-rollout-workers':'10','--num-eval-workers':'0',
                           '--train-batch-size':'12','--accumulation-steps':'16',
                           '--clip-mode':'gipo','--sigma':'1.0','--task-ids':'0',
                           '--num-images-in-input':'2','--cuda-visible-devices':','.join(baseline.GPUS)}.items():
            self.assertEqual(argv[argv.index(key)+1], value)
        self.assertNotIn('--use-proprio', argv)
        self.assertNotIn('--world-backend-config', argv)
        self.assertEqual(self.record['configuration']['global_batch'],576)
        self.assertEqual(self.record['configuration']['physical_gpus'],[4,5,6,7])
        self.assertEqual(self.record['configuration']['cuda_visible_gpu_uuids'],list(baseline.GPUS))

    def test_reject_invalid_duration_and_label(self):
        for seconds in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                baseline.make_record((self.root/'frozen').resolve(),self.root,'ok','a'*40,seconds)
        with self.assertRaises(ValueError):
            baseline.make_record((self.root/'frozen').resolve(),self.root,'../bad','a'*40,7200)

    def test_ray_socket_path_fits_unix_limit_for_maximum_pid_width(self):
        ray_temp = self.record['ray_temp_dir']
        session = 'session_2026-11-17_23-59-59_999999_4294967295'
        socket_path = ray_temp+'/'+session+'/sockets/plasma_store'
        self.assertLessEqual(len(socket_path.encode()),107)
        argv = self.record['driver_argv']
        self.assertEqual(argv[argv.index('--ray-temp-dir')+1],ray_temp)
        self.assertEqual(Path(ray_temp).parent,baseline.RAY_TEMP_BASE)

    def test_ray_temp_identity_changes_with_label_commit_or_output(self):
        first = self.record['ray_temp_dir']
        variants = [baseline.make_record(Path(self.record['checkout']),self.root,label,commit)['ray_temp_dir']
                    for label,commit in [('test-b','a'*40),('test-a','b'*40)]]
        variants.append(baseline.make_record(Path(self.record['checkout']),self.root/'other',
                                             'test-a','a'*40)['ray_temp_dir'])
        self.assertEqual(len({first,*variants}),4)
        self.assertEqual(baseline.make_record(Path(self.record['checkout']),self.root,
                                              'test-a','a'*40)['ray_temp_dir'],first)

    def test_existing_identity_cannot_change(self):
        baseline.match_record(self.record,self.record)
        changed = dict(self.record,training_seconds=1)
        with self.assertRaises(ValueError):
            baseline.match_record(self.record,changed)

    def test_payload_cannot_change(self):
        row = dict(self.record['job'])
        row['argv_json'] = json.dumps(row.pop('argv'))
        baseline.match_job(self.record,row)
        row['cwd'] = '/other'
        with self.assertRaises(ValueError):
            baseline.match_job(self.record,row)

    def test_checkpoint_audits_nonempty_artifacts(self):
        folder, _ = self.checkpoint()
        result = entry.audit_completion(self.record)
        self.assertEqual(len(result['files']),3)
        (folder/'agent_extra_layers.pt').write_bytes(b'')
        with self.assertRaises(ValueError):
            entry.audit_completion(self.record)

    def test_checkpoint_escape_rejected(self):
        _, complete = self.checkpoint()
        complete['checkpoint_dir'] = str(self.root.parent)
        entry.atomic(self.root/'training/driver_complete.json',complete)
        with self.assertRaises(ValueError):
            entry.audit_completion(self.record)

    def test_checkpoint_with_mismatched_update_ledger_rejected(self):
        self.checkpoint()
        (self.root/'training/updates.jsonl').write_text('')
        with self.assertRaises(ValueError):
            entry.audit_completion(self.record)

    def test_completion_source_identity_rejected(self):
        _, complete = self.checkpoint()
        complete['git_commit'] = 'b'*40
        entry.atomic(self.root/'training/driver_complete.json',complete)
        with self.assertRaises(ValueError):
            entry.audit_completion(self.record)

    def test_environment_removes_other_run_controls(self):
        from unittest.mock import patch
        with patch.dict(os.environ,{'ACCERL_MODEL_FREE':'1','RAY_ADDRESS':'other',
            'EXPERIMENT_QUEUE_JOB_ID':'my-job','CUDA_VISIBLE_DEVICES':'0',
            'LD_PRELOAD':'injection','WEAVER_LIBERO_ROOT':'/libero'}):
            env = entry.launch_environment(self.record,{'pipe':'/mps/pipe'})
        self.assertNotIn('ACCERL_MODEL_FREE',env)
        self.assertNotIn('RAY_ADDRESS',env)
        self.assertNotIn('LD_PRELOAD',env)
        self.assertEqual(env['EXPERIMENT_QUEUE_JOB_ID'],'my-job')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],','.join(baseline.GPUS))
        self.assertEqual(env['MUJOCO_GL'],'osmesa')
        self.assertEqual(env['WEAVER_LEGACY_GIT_COMMIT'],self.record['git_commit'])

    def fake_run(self, code=0, survivors=False, timeout=False, training=False):
        self.checkpoint()
        if training:
            self.record['training_seconds'] = 2
            entry.atomic(self.root/'training/driver_status.json',dict(
                run_id=self.record['run_id'],git_commit=self.record['git_commit'],training_started_unix=1))
        class Child:
            pid = 123
            def poll(self): return None if timeout else code
            def wait(self,timeout=None): return code
        class Safety:
            @staticmethod
            def proc_identity(pid): return dict(pid=pid,uid=os.getuid() if hasattr(os,'getuid') else 1004,birth_ticks=1,state='S')
            @staticmethod
            def marked_children(job,known=()): return [dict(pid=321)] if survivors else []
            @staticmethod
            def scoped_cleanup(job,known=()): return {'survivors':[321] if survivors else [],'signals':[]}
        ticks = iter(range(100))
        return entry.supervise(self.record,{'EXPERIMENT_QUEUE_JOB_ID':'fake-job'},Safety,
            popen=lambda *a,**kw:Child(),clock=lambda:next(ticks),sleep=lambda _:None,
            startup_seconds=1,grace_seconds=1)

    def test_fake_child_success_after_exit_and_audit(self):
        result = self.fake_run()
        self.assertEqual(result['state'],'succeeded')
        self.assertTrue(result['all_owned_exited'])

    def test_nonzero_exit_with_weights_still_fails(self):
        result = self.fake_run(code=2)
        self.assertEqual(result['state'],'failed')
        self.assertIn('2',result['error'])

    def test_surviving_owned_process_blocks_success(self):
        result = self.fake_run(survivors=True)
        self.assertEqual(result['state'],'failed')
        self.assertFalse(result['all_owned_exited'])

    def test_startup_timeout_blocks_success(self):
        result = self.fake_run(timeout=True)
        self.assertEqual(result['state'],'failed')
        self.assertIn('startup',result['error'])

    def test_training_save_grace_timeout_blocks_success(self):
        result = self.fake_run(timeout=True,training=True)
        self.assertEqual(result['state'],'failed')
        self.assertIn('boundary/final save',result['error'])


class TensorAudit(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ImportError:
            self.skipTest('CPU torch unavailable; run tensor fixtures in server control Python')
        self.torch = torch
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        (self.directory/'agent_lora').mkdir()
        torch.save({'adapter':torch.ones(2)},self.directory/'agent_lora/adapter_model.bin')
        torch.save({key:{'weight':torch.ones(2)} for key in ('value_head','attn_pool','lm_head')},
                   self.directory/'agent_extra_layers.pt')

    def tearDown(self):
        if hasattr(self,'temp'):
            self.temp.cleanup()

    def test_real_cpu_weights_load_and_are_finite(self):
        result = entry.audit_weight_tensors(self.directory)
        self.assertTrue(result['finite'])
        self.assertEqual(result['device'],'cpu')
        self.assertEqual(result['tensor_count'],4)

    def test_nonfinite_lora_is_rejected(self):
        self.torch.save({'adapter':self.torch.tensor([float('nan')])},
                        self.directory/'agent_lora/adapter_model.bin')
        with self.assertRaisesRegex(ValueError,'Non-finite'):
            entry.audit_weight_tensors(self.directory)

    def test_truncated_extra_weights_are_rejected(self):
        (self.directory/'agent_extra_layers.pt').write_bytes(b'truncated')
        with self.assertRaises(Exception):
            entry.audit_weight_tensors(self.directory)


if __name__ == '__main__':
    unittest.main()
