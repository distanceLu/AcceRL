"""No-CUDA contracts for optimized launch configuration and per-boot MPS binding."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import training_entry as train
import queue_entry as queue
from test_manual_wrapper import mps_authorization, mps_receipt


class OptimizedWrapperTests(unittest.TestCase):
    def test_every_config_path_is_explicitly_optimized(self):
        source = Path('/frozen/source')
        environment = train.launch_environment(Path('/new/output'), source, {
            'ACCERL_WORLD_BACKEND_CONFIG': '/old/internal.json'})
        self.assertEqual(environment['ACCERL_WORLD_BACKEND_CONFIG'], str(source / train.BACKEND_CONFIG))
        self.assertEqual(train.BACKEND_CONFIG, 'examples/trajectory_01_imagined.json')
        self.assertEqual(environment['ACCERL_FRESH_PROFILE'], '04_real')
        wrapper = Path(train.__file__).read_text()
        self.assertNotIn('examples/weaver_integrated_internal.json', wrapper)

    def test_optimization_options_are_exact_and_type_strict(self):
        backend={'backend':'weaver','reward_provider':'world_head','reward_semantics':'delta_native_score',
            'terminal_rule':'horizon_only','critic_mode':'disabled','world_trainable':True,
            'backend_options':{'training_python':'/native/python','optimizations':dict(train.OPTIMIZATIONS),
                'native_reward_head':dict(train.NATIVE_REWARD), 'fresh':dict(train.FRESH_OPTIONS)},
            'checkpoint_metadata':{'load_report':{'checkpoint_sha256':train.INITIAL_WORLD['source_bundle_sha256']}}}
        admitted={'configuration_admitted':True,'assets_checked':True,'errors':[],
            'environment':{'compatible':True},'online_training_environment':{
                'python':backend['backend_options']['training_python'],'compatible':True,
                'distributed_available':True,'nccl_available':True,
                'cuda_initialized':False,'gpu_access_performed':False}}
        self.assertTrue(train.validate_backend_admission(backend,train.PIPELINE_RECIPE,admitted))
        for key,value in (('kv_cache',False),('selected_frame_encoding',False),('kv_cache',1),
                          ('resident_max_sessions',True),('decode_mode','all'),('safety_checks','legacy')):
            changed=copy.deepcopy(backend);changed['backend_options']['optimizations'][key]=value
            with self.subTest(key=key,value=value),self.assertRaisesRegex(ValueError,'optimizations'):
                train.validate_backend_admission(changed,train.PIPELINE_RECIPE,admitted)
        changed=copy.deepcopy(backend);changed['backend_options'].pop('optimizations')
        with self.assertRaisesRegex(ValueError,'optimizations'):
            train.validate_backend_admission(changed,train.PIPELINE_RECIPE,admitted)

    def test_receipt_rejects_old_boot_pid_birth_and_scope(self):
        expected = mps_authorization()
        with mock.patch.object(train, 'mps_authorization', return_value=expected):
            self.assertEqual(train.validate_mps_receipt(mps_receipt()), expected)
            for edit in (
                lambda r:r.update(boot_id='00000000-0000-0000-0000-000000000000'),
                lambda r:r['server'].update(pid=691086),
                lambda r:r['server'].update(birth_ticks=1),
                lambda r:r['control'].update(uid=1005),
                lambda r:r.update(scope=[0,1,2,3]),
                lambda r:r['cards'][0].update(uuid='GPU-other'),
            ):
                value=copy.deepcopy(mps_receipt());edit(value)
                with self.assertRaises(ValueError):train.validate_mps_receipt(value)

    def test_live_authorization_checks_actual_boot_birth_and_executable(self):
        original_text, original_bytes = Path.read_text, Path.read_bytes
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);(folder/'mps_authorization.json').write_text(json.dumps(mps_authorization()))
            current_boot=[mps_authorization()['boot_id']]
            role_commands={'server': b'/usr/bin/nvidia-cuda-mps-server\0',
                           'control': b'/usr/bin/nvidia-cuda-mps-control\0-d\0'}
            def text(path,*args,**kwargs):
                return current_boot[0] if path.as_posix()=='/proc/sys/kernel/random/boot_id' else original_text(path,*args,**kwargs)
            def byte_content(path,*args,**kwargs):
                for role in ('server','control'):
                    if path.as_posix()==f"/proc/{mps_authorization()[role]['pid']}/cmdline":return role_commands[role]
                return original_bytes(path,*args,**kwargs)
            identities={x['pid']:{**x,'state':'S'} for x in (mps_authorization()['server'],mps_authorization()['control'])}
            with mock.patch.object(train,'__file__',str(folder/'training_entry.py')), \
                 mock.patch.object(train,'proc_identity',side_effect=lambda pid:identities.get(pid)), \
                 mock.patch.object(Path,'read_text',text), mock.patch.object(Path,'read_bytes',byte_content):
                self.assertEqual(train.mps_authorization(),mps_authorization())
                current_boot[0]='00000000-0000-0000-0000-000000000000'
                with self.assertRaisesRegex(ValueError,'rebooted'):train.mps_authorization()
                current_boot[0]=mps_authorization()['boot_id']
                identities[410001]['birth_ticks']+=1
                with self.assertRaisesRegex(ValueError,'no longer exists'):train.mps_authorization()
                identities[410001]['birth_ticks']-=1
                role_commands['control']=b'/usr/bin/unrelated\0'
                with self.assertRaisesRegex(ValueError,'executable'):train.mps_authorization()

    def test_existing_claim_rejected_before_mps_or_plan_admission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            claim=root/'native_gpu_preflight_claim.json'
            original=b'{"state":"unknown_outcome","run_id":"original-attempt"}\n'
            claim.write_bytes(original)
            args=SimpleNamespace(root=root,source=root/'unavailable-source')
            with mock.patch.object(train,'native_preflight_plan',side_effect=AssertionError('must not read a new plan')) as plan:
                with self.assertRaisesRegex(FileExistsError,'refusing a second launch'):
                    train.run_native_preflight(args,{'CUDA_VISIBLE_DEVICES':''})
                plan.assert_not_called()
            self.assertEqual(claim.read_bytes(),original)
            self.assertFalse((root/'native_preflight_plan.json').exists())

    def test_validate_only_builds_explicit_plan_without_writing_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'source';output=root/'output';source.mkdir();output.mkdir()
            (source/'examples').mkdir()
            (source/train.BACKEND_CONFIG).write_text(json.dumps({'backend_options':{'training_python':'/native/python'}}))
            (output/'mps.json').write_text(json.dumps(mps_receipt()))
            args=SimpleNamespace(source=source,root=output,manifest_sha256='a'*64,validate_only=True,
                native_gpu_preflight=True,post_stop_grace_seconds=1800,evaluation_seconds=7200,
                evaluation_modes='greedy,stochastic',evaluation_seed=20261026,evaluation_repeats=1,
                baseline_checkpoint=None)
            before={p.relative_to(root).as_posix() for p in root.rglob('*')}
            with mock.patch.object(train,'validate',return_value={'fixture':True}), \
                 mock.patch.object(train,'mps_authorization',return_value=mps_authorization()), \
                 mock.patch.object(train,'log') as log:
                self.assertEqual(queue.run(args),0)
                logged=log.call_args.kwargs
                self.assertFalse(logged['outputs_created'])
                self.assertEqual(logged['native_preflight_plan']['mps']['server']['birth'],123456)
                self.assertEqual(logged['native_preflight_argv'][2:4],['--plan',str(output/'native_preflight_plan.json')])
                self.assertTrue(logged['native_preflight_argv'][-1]=='--plan-only')
            self.assertEqual(before,{p.relative_to(root).as_posix() for p in root.rglob('*')})


if __name__=='__main__':unittest.main()
