import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from weaver_run.reference_site import jit_bench_queue as queue
from weaver_run.reference_site.dispatcher.store import QueueStore
import launch


class CriticalJitSubmissionTest(unittest.TestCase):
    def test_queue_behind_active_job_uses_commit_directory_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cli = Mock(EXPERIMENTS=base/'experiments', PARENT=base, HOME=base, QUEUE=base/'queue',
                       CHECKOUT=queue.REPO, RUNTIME=base/'active-runtime', PYTHON=sys.executable)
            cli.EXPERIMENTS.mkdir()
            cli.LABEL = re.compile(r'[a-z0-9-]+')
            cli.require = launch.require
            store = QueueStore(cli.QUEUE, allowed_cwd_root=base)
            store.create(name='active training', argv=[sys.executable], cwd=str(cli.RUNTIME),
                         manual=True, timeout_seconds=0, submission_id='active-training')
            active = store.claim_next()['id']
            cli.lookup.side_effect = lambda sid: next((r for r in store.list() if r['submission_id']==sid),None)
            cli.create.side_effect = lambda job, location: store.create(**job)
            commit = 'a'*40
            cli.published.return_value = commit
            cli.PROFILES = {'restore': {'mps_receipt': hashlib.sha256(b'{}').hexdigest()}}
            (base/'mps.json').write_bytes(b'{}')

            def pin(source, runtime, commit):
                recipe = runtime/'weaver_run/recipe'
                recipe.mkdir(parents=True)
                (recipe/'visualization.json').write_bytes(b'{}')
            cli.pin_worktree.side_effect = pin
            receipt = queue.submit(cli, 'critical-test', critical=True)
            runtime = base/'library/jit-bench-code'/commit
            cli.pin_worktree.assert_called_once_with(cli.CHECKOUT, runtime, commit)
            cli.ready.assert_not_called()
            self.assertEqual(store.get(active)['state'], 'running')
            self.assertIsNone(store.claim_next())
            root = cli.EXPERIMENTS/'critical-test'
            plan = json.loads((root/'plan.json').read_bytes())
            job = json.loads((root/'experiment.json').read_bytes())['job']
            self.assertEqual(job['cwd'], str(runtime))
            self.assertEqual(job['argv'][-2:], [str(runtime/'weaver_run/reference_site/jit_bench_queue.py'),str(root/'plan.json')])
            self.assertEqual((receipt['submission_id'], plan['comparison'], plan['git_commit']),
                             ('weaver-bench-critical-critical-test','critical_jit',commit))
            self.assertEqual((plan['checkpoint_policy_step'],plan['checkpoint_sha256']), (189,queue.WEIGHTS_SHA256))
            self.assertEqual((plan['real_workers'], plan['online_eval_workers'], plan['learning_rate']), (8,0,0))
            self.assertEqual((plan['gpu_ids'],plan['rank_size'],plan['train_micro'],plan['context_micro']),([4,5,6,7],144,16,2))
            self.assertEqual((plan['global_batch'],plan['real_sampling']), (576,'uniform_blocks'))
            self.assertEqual((plan['warmup_updates'],plan['measured_blocks'],plan['noise_blocks'],plan['updates_per_arm']), (1,6,2,1))
            self.assertEqual((plan['minimum_replay_chunks'],plan['high_next_chunks'],plan['gate_ttl_s'],plan['protocol_seed']),
                             (2753,1152,5.,20261125))
            self.assertTrue(plan['no_micro_cache_trim'])
            self.assertFalse(plan['compile_action_inference'] or plan['fixed_validation'])
            pipeline = json.loads((root/'git-pipeline.json').read_bytes())
            self.assertEqual((pipeline['real_workers'], pipeline['max_real_in_flight']), (8,8))

            for status in (True,False):
                response = queue.submit(cli, 'critical-test', critical=True, status=status)
                self.assertEqual((response['job_id'],response['state']), (receipt['job_id'],'queued'))
            cli.create.assert_called_once()
            cli.lookup.assert_called_with(receipt['submission_id'])
            unrelated = queue.submit(cli, 'critical-test', status=True, layout=True)
            self.assertNotIn('submission.json', unrelated)

    def test_managed_driver_receives_fixed_training_and_source_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights = root/'weights.pt'
            weights.write_bytes(b'fixture')
            sha = hashlib.sha256(weights.read_bytes()).hexdigest()
            plan = dict(comparison='critical_jit', real_sampling='uniform_blocks', real_workers=8,
                        no_micro_cache_trim=True, learning_rate=0., label='critical-test',
                        checkpoint=str(weights), checkpoint_sha256=sha, git_commit='a'*40, output=str(root))
            queue.put(root/'plan.json',plan)
            queue.put(root/'manifest.json',dict(steps={'policy':189}, metadata={'run_id':queue.SOURCE_RUN},
                                               weights_bytes=weights.stat().st_size, tensor_counts={'policy':2}))
            queue.put(root/'result.json',{'status':'completed'})
            for rank in range(4):
                queue.put(root/f'checkpoint_rank_{rank}.json',{'policy_step':189,'sha256':sha})
            child = Mock(returncode=0)
            child.wait.return_value = 0
            safe = Mock()
            safe.cpu_helper_environment.return_value = {'PYTHONPATH':'/legacy-checkout'}
            safe.preflight.return_value = dict(cards=[dict(uuid='GPU-'+str(i)) for i in range(4,8)],
                                               mps_pipe='/pipe', mps_log='/log')
            safe.proc_identity.return_value = {'birth_ticks':1}
            def cleaned(state,data):
                state['training_stopped'] = True
                return True
            safe.cleanup.side_effect = cleaned
            with patch.dict(sys.modules,{'gpu47_safety':safe}), patch.dict(os.environ,clear=True), \
                    patch.object(queue,'revision'), patch.object(queue,'WEIGHTS_SHA256',sha), \
                    patch.object(queue.signal,'signal'), patch.object(queue.subprocess,'Popen',return_value=child) as start:
                import runner
                with patch.object(runner,'safe',safe):
                    queue.run(root/'plan.json')
            env = start.call_args.kwargs['env']
            self.assertEqual(env['ACCERL_TRAJECTORY_JIT_INITIAL_CHECKPOINT'],str(weights))
            self.assertEqual(start.call_args.kwargs['cwd'],queue.REPO)
            self.assertEqual(json.loads((root/'benchmark_queue.json').read_bytes())['state'],'completed')
            env['ACCERL_FINAL_CHECKPOINT'] = str(root/'formal_v3/training/checkpoints')  # set by engine/run.sh
            with patch.dict(os.environ,env,clear=True), patch.object(os,'chdir') as chdir, patch.object(os,'execve') as run:
                launch.exec_managed_driver(start.call_args.args[0][2:])
        _, argv, effective = run.call_args.args
        parser = argparse.ArgumentParser(add_help=False)
        for option in ('--policy-lr','--value-lr','--num-rollout-workers','--num-eval-workers','--num-trainer-gpus'):
            parser.add_argument(option,type=float)
        options, _ = parser.parse_known_args(argv[2:])
        self.assertEqual((options.policy_lr, options.value_lr), (0,0))
        self.assertEqual((options.num_rollout_workers,options.num_eval_workers,options.num_trainer_gpus),(8,0,4))
        expected = {'ACCERL_TRAJECTORY_JIT':'1','ACCERL_REAL_SAMPLING':'uniform_blocks',
                    'ACCERL_COMPILE_ACTION_INFERENCE':'0','ACCERL_TRIM_MICRO_CACHE':'0',
                    'ACCERL_TRIM_IDLE_CACHE':'1','ACCERL_MODEL_FREE_REAL_WORKERS':'8',
                    'ACCERL_REPO_ROOT':str(queue.REPO),'ACCERL_JIT_BENCH_CHECKPOINT':str(weights)}
        self.assertEqual({k:effective[k] for k in expected},expected)
        self.assertEqual(effective['PYTHONPATH'].split(os.pathsep)[:2],[str(queue.ENGINE),str(queue.REPO)])
        chdir.assert_called_once_with(queue.REPO)

    def test_public_and_site_commands_route_submission_and_status(self):
        from weaver_run.reference_site import cli
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp)
            (site/'.owner').write_text('WEAVER server catalog v1\n')
            for command in ('bench-critical-jit','bench-critical-jit-status'):
                args = [command,'--id','critical-test']
                with self.subTest(command=command), patch.object(sys,'argv',['run.sh',*args]), \
                        patch.object(launch.socket,'gethostname',return_value=launch.REFERENCE_SERVER), \
                        patch.object(os,'execv',side_effect=InterruptedError) as run:
                    with self.assertRaises(InterruptedError):
                        launch.main()
                    self.assertEqual(run.call_args.args[1][-3:],args)
                with patch.object(sys,'argv',['cli.py',*args]), patch.object(cli,'HOME',site), \
                        patch.object(cli,'EXPERIMENTS',site), patch.object(cli.socket,'gethostname',return_value=launch.REFERENCE_SERVER), \
                        patch.object(os,'getuid',return_value=1004), patch.object(sys,'stdout',io.StringIO()), \
                        patch.object(queue,'submit',return_value={}) as submit:
                    cli.main()
                    self.assertEqual(submit.call_args.args,(cli,'critical-test'))
                    self.assertTrue(submit.call_args.kwargs['critical'])
                    self.assertEqual(submit.call_args.kwargs['status'],command.endswith('-status'))


if __name__ == '__main__':
    unittest.main()
