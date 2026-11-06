"""Actual online controller/auditor paths with bounded CPU-only inventories."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from unittest.mock import patch

from test_manual_controller_cpu import load, result as legacy_result, SOURCE
from world_backends import build_service_plan
from world_backend_audit import (online_world_identity, validate_online_world_receipts,
    validate_online_publication, validate_online_descriptor, audit_online_native_cpu, validate_joint_reward_metadata,
    online_world_asset_report)

CONFIG = json.loads((SOURCE/'examples/weaver_integrated_internal.json').read_text())
PLAN = build_service_plan(CONFIG).to_dict()
RUN = 'weaverintegrated_cpu_acceptance'


def descriptor(p=27, reason='final'):
    updates=p//5; rm_updates=max(0,updates-1)
    return {**online_world_identity(CONFIG), 'protocol': 'weaver_online_joint_weights_v1',
        'run_id': RUN, 'world_size': 4, 'policy_step': p, 'successful_updates': p//5,
        'model_version': p//5, 'weights_only': True, 'exact_resume_supported': False,
        'world_engine_backend': 'torch_ddp_native_flow_matching', 'skipped_steps': 0,
        'source_trainable_scope':'wm','trained_parameter_families':['wm','rm'],'rm_updates':rm_updates,
        'native_reward_head':copy.deepcopy(CONFIG['backend_options']['native_reward_head']),
        'reward_class_counts':{'scope':'global_selected_frames_once_per_successful_world_update',
            'positive_count':updates*4,'negative_count':updates*6,'disabled_updates':updates-rm_updates,
            'last_cohort':{'positive_count':4,'negative_count':6} if updates else None},
        'reason': reason, 'tensor_count': 866, 'weights_file': '/tmp/checkpoints/world/native.pt',
        'relative_weights_file': 'world/native.pt', 'weights_sha256': 'a'*64,
        'weights_bytes': 7, 'native_cpu_finite_checked': True}


def engine():
    return {'backend': 'weaver', 'backend_engine': 'torch_ddp_native_flow_matching',
        'world_size': 4, 'distributed_world_size': 4, 'micro_batch': 4, 'accumulation_steps': 64, 'global_batch': 1024,
        'optimizer': 'AdamW', 'optimizer_betas': [.9, .99], 'weight_decay': .01,
        'warmup_updates': 20, 'peak_learning_rate': 1e-5, 'post_warmup_schedule': 'constant',
        'max_grad_norm': 1., 'trainable_parameter_count': 921834497,
        'trainable_scope':'wm+rm','optimizer_parameter_tensors':403}


def receipts():
    return [{'actor_id': str(rank), 'pid': 400+rank, 'details': {
        **online_world_identity(CONFIG), 'rank': rank, 'native_engine': engine(),
        'worker_identity': {'pid': 400+rank, 'uid': 1004, 'birth': 10000+rank},
        'gpu_uuid': 'GPU-'+str(rank), 'trainable_parameter_count': 921834497,
        'weight_dtype': 'torch.float32', 'compute_dtype': 'torch.bfloat16',
        'activation_checkpoint_blocks': 24}} for rank in range(4)]


def publication(version):
    inventory = {'kind': 'ctrl', 'version': version, 'hash': 'b'*64,
        'publication':{'parameter_names':['upstream.wm.weight','upstream.rm.weight'],
                       'bundle_metadata':{'reward_provider':'world_head'}}}
    return [{'model_version': version, 'details': {'kind': 'ctrl', 'transport': 'cuda_ipc',
        'inventories': [copy.deepcopy(inventory) for _ in range(4)], 'receiver_ranks': [0,1,2,3],
        'ack_completed_ns': 50, 'driver_timing': {'rank_ACK_observed_ns': [20,21,22,23],
                                               'release_ready_ns': 40}}}]


def fixture(p=27):
    value = legacy_result(p)
    value['training'].pop('reward')
    value['fsdp_sharding_verified']=[];value['reward_features_only_enabled']=[];value['reward_checkpointing_verified']=[]
    value['replay'][1][5]=0
    value.update(world_zero2_verified=[], world_memory_policy_verified=[],
        weaver_online_trainer_verified=receipts(), world_publication_committed=publication(p//5))
    value['policy_batch_mode_validation']['run_id'] = RUN
    value['experiment_completed'][0]['details']['service_plan'] = PLAN
    value['split_wm_cohort_validated'] = ([{'details': {'wm_consumers': ['ctrl'],
                                        'expected_wm_consumers': 4}}] if p>=5 else [])
    reason = 'final' if p else 'final_before_first_update'
    value['checkpoint'].update(steps={'policy': p}, online_world=descriptor(p, reason))
    native=descriptor(p,reason);counts=native['reward_class_counts']
    value['weaver_online_publication_released']=[{'actor_id':str(rank),'pid':400+rank,'details':{
        'source_refs_released':True,'native_engine':{'global_steps':p//5,'skipped_steps':0,'gradients_released':True,
            'rm_updates':native['rm_updates'],'rm_disabled_updates':counts['disabled_updates'],
            'reward_positive_labels':counts['positive_count'],'reward_negative_labels':counts['negative_count'],
            'last_reward_cohort':counts['last_cohort']}}} for rank in range(4)]
    return value


def state(p=27):
    return {'run_id': RUN, 'started_ns': 100, 'service_plan': PLAN, 'world_backend_config': CONFIG,
            'report': {'training': {kind: {'global_updates': count} for kind, count in
                                   {'policy': p, 'ctrl': p//5}.items()}}}


def check(value):
    validate = load('runner.py', 'validate_manual_completion', {})
    return validate(value, state(), {'run_id': RUN, 'uid': 1004,
                                    'reason': 'user_requested_stop', 'requested_ns': 1000})


class Tensor:
    device = SimpleNamespace(type='cpu')
    def is_floating_point(self): return True


def fake_torch(payload, *, finite=True):
    return SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False),
        load=Mock(return_value=payload), is_tensor=lambda t: isinstance(t, Tensor),
        isfinite=lambda _: SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: finite)))


class OnlineControllerAcceptanceCPU(unittest.TestCase):
    def test_initial_asset_reader_does_not_require_a_frozen_integrated_reward_service(self):
        immutable={'backend':'weaver','world_trainable':False,'fixed_step':3200,
                   'bundle_sha256':CONFIG['backend_options']['initial_checkpoint_sha256'],'cuda_initialized':False}
        with patch('world_backend_audit.frozen_world_asset_report',return_value=immutable) as read:
            value=online_world_asset_report(CONFIG)
        initial_configuration=read.call_args.args[0]
        self.assertFalse(initial_configuration['world_trainable'])
        self.assertEqual(initial_configuration['reward_provider'],'external_vlm')
        self.assertTrue(value['world_trainable'])
        self.assertNotIn('fixed_step',value)

    def test_policy_only_checkpoint_preserves_baseline_and_admits_two_hundred_paired_eval_cases(self):
        try: import torch
        except ImportError:self.skipTest('Real torch CPU checkpoint roundtrip requires the server environment')
        from parallel_checkpoint import save_split_weights
        from checkpoint_evaluation import preserve_initial_checkpoint
        from frozen_policy_eval import validate_checkpoint_payload, make_plan
        names=['lora_weight','lm_head_weight','attn_pool_weight','value_head_weight']
        saved={'state_dict':{name:torch.tensor([1.]) for name in names},'trainable_names':names,'full_state':False}
        recipe={'pretrained_checkpoint':'/checkpoint/base','checkpoint2':None,'benchmark':'libero_10',
                'num_images_in_input':2,'use_proprio':False,'use_bf16':True}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve()/'checkpoints'
            def publish(p,reason):
                metadata={'run_id':RUN,'args':recipe,'service_plan':PLAN,'online_world':descriptor(p,reason)}
                component={'kind':'policy','rank':0,'world_size':4,'policy_step':p,'model_step':p,'model':saved}
                return save_split_weights({'policy':component},root,reason=reason,metadata=metadata,
                                          torch_module=torch,active_kinds=('policy',))
            publish(0,'initial');initial=preserve_initial_checkpoint(root)
            final=publish(27,'final')
            checkpoints=[]
            for name,path in (('baseline',Path(initial['initial_checkpoint'])),('final',root/'latest_weights.pt')):
                payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
                self.assertEqual(set(payload['models']),{'policy'})
                admitted=validate_checkpoint_payload(payload)
                self.assertEqual(admitted['policy_step'],0 if name=='baseline' else 27)
                checkpoints.append({'name':name,'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                                    'metadata':admitted})
            self.assertEqual(final['steps'],{'policy':27})
            plan=make_plan(checkpoints)
            self.assertEqual(len(plan['cases'])*len(plan['checkpoints']),200)
            self.assertEqual(plan['expected_episode_records'],200)

    def test_missing_class_reward_updates_may_be_less_than_world_without_fake_reward_optimizer(self):
        value=descriptor()
        self.assertEqual(value['successful_updates'],5)
        self.assertEqual(value['rm_updates'],4)
        self.assertTrue(validate_joint_reward_metadata(value,CONFIG,updates=5))
        for mutate in (
            lambda d:d.update(rm_updates=6),
            lambda d:d['reward_class_counts'].update(disabled_updates=0),
            lambda d:d['native_reward_head'].update(calibrated=True),
            lambda d:d['reward_class_counts'].update(positive_count=-1),
        ):
            bad=copy.deepcopy(value);mutate(bad)
            with self.assertRaises(RuntimeError):validate_joint_reward_metadata(bad,CONFIG,updates=5)

    def test_actual_snapshot_export_to_actual_torch_mmap_audit_schema(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Torch absent locally; run the real snapshot/audit regression in the server CPU environment')
        from weaver_online_gpu_worker import snapshot_native
        model=torch.nn.Module()
        for number in range(866):model.register_buffer('native_%04d'%number,torch.tensor([float(number)]))
        engine_state=SimpleNamespace(rank=0,active_update=None,publication_idle=lambda:True,
            policy_step=27,global_steps=5,skipped_steps=0,rm_updates=4,rm_disabled_updates=1,
            reward_positive_labels=20,reward_negative_labels=30,last_reward_cohort={'positive_count':4,'negative_count':6},
            native_reward=SimpleNamespace(specification=CONFIG['backend_options']['native_reward_head']))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve()
            with patch.dict(os.environ,{'ACCERL_FINAL_CHECKPOINT':str(root)}):
                d=snapshot_native(SimpleNamespace(upstream_model=model),engine_state,
                    str(root/'checkpoints_world'),'final',online_world_identity(CONFIG),RUN)
            request={'descriptor':d,'config':CONFIG,'run_id':RUN,'policy_step':27,'reason':'final',
                     'checkpoint_root':str(root)}
            audited=audit_online_native_cpu(request,torch_module=torch)
            self.assertTrue(audited['valid'] and audited['finite'])
            self.assertEqual(audited['tensor_count'],866)
            self.assertEqual(audited['world_engine_backend'],'torch_ddp_native_flow_matching')
            self.assertEqual(audited['skipped_steps'],0)
            self.assertEqual(audited['weights_sha256'],d['weights_sha256'])
            self.assertTrue(audit_online_native_cpu(request,torch_module=torch)['cache_reused'])

    def test_integrated_service_plan_has_two_trainers_and_sixteen_gpu_models(self):
        self.assertEqual(PLAN['trainer_kinds'], ['policy','ctrl'])
        self.assertEqual(PLAN['expected_gpu_model_processes'], 16)
        self.assertEqual(check(fixture())['steps'], {'policy': 27, 'ctrl': 5})

    def test_stop_zero_requires_native_initial_receipts_and_version0(self):
        value = fixture(0)
        self.assertTrue(check(value)['stopped_before_first_update'])
        value['world_publication_committed'] = []
        with self.assertRaises(RuntimeError): check(value)

    def test_real_ddp_receipts_reject_legacy_zero_recipe_and_wrong_optimizer(self):
        self.assertTrue(validate_online_world_receipts(receipts(), CONFIG))
        for edit in (
            lambda r: r.pop(),
            lambda r: r[0]['details']['native_engine'].update(backend_engine='deepspeed_zero2'),
            lambda r: r[0]['details']['native_engine'].update(distributed_world_size=1),
            lambda r: r[0]['details']['native_engine'].update(max_grad_norm=0.),
            lambda r: r[0]['details']['native_engine'].update(accumulation_steps=32),
            lambda r: r[0]['details']['native_engine'].update(global_batch=512),
            lambda r: r[0]['details']['native_engine'].update(optimizer_betas=[.9,.999]),
            lambda r: r[0]['details']['native_engine'].update(warmup_updates=500),
            lambda r: r[0]['details']['worker_identity'].update(uid=1000),
            lambda r: r[0]['details']['worker_identity'].update(birth=0),
            lambda r: r[0].update(pid=r[1]['pid']),
            lambda r: r[0]['details'].update(gpu_uuid=r[1]['details']['gpu_uuid']),
        ):
            rows=receipts(); edit(rows)
            with self.assertRaises(RuntimeError): validate_online_world_receipts(rows, CONFIG)

    def test_completion_requires_skip0_four_rank_world_and_exact_four_consumers(self):
        for edit in (
            lambda r: r['training']['ctrl']['ranks'].pop('3'),
            lambda r: r['training']['ctrl']['ranks']['0'].update(successful_updates=4),
            lambda r: r['training']['ctrl']['ranks']['0'].update(skipped_steps=1),
            lambda r: r['split_wm_cohort_validated'][0]['details'].update(expected_wm_consumers=8),
            lambda r: r['split_wm_cohort_validated'][0]['details'].update(wm_consumers=['reward']),
            lambda r: r.update(world_zero2_verified=[{}]),
            lambda r: r.update(weaver_frozen_verified=[{}]),
        ):
            value=fixture(); edit(value)
            with self.assertRaises(RuntimeError): check(value)

    def test_final_native_descriptor_rejects_frozen_or_wrong_policy_boundary(self):
        for key, wrong in (('world_trainable',False),('policy_step',25),('successful_updates',3205),
                           ('source_optimizer_steps',0),('tensor_count',383),('run_id','old')):
            value=fixture(); value['checkpoint']['online_world'][key]=wrong
            with self.assertRaises(RuntimeError): check(value)

    def test_publication_requires_same_four_hashes_and_finished_release(self):
        self.assertTrue(validate_online_publication(publication(5),version=5)['all_four_producer_releases'])
        for edit in (
            lambda r: r[0].update(model_version=4),
            lambda r: r[0]['details']['inventories'][3].update(hash='c'*64),
            lambda r: r[0]['details'].update(receiver_ranks=[0,1,2]),
            lambda r: r[0]['details']['driver_timing'].pop('release_ready_ns'),
            lambda r: r[0]['details']['driver_timing'].update(release_ready_ns=20),
            lambda r: r[0]['details']['driver_timing'].update(rank_ACK_observed_ns=[20,21,22]),
        ):
            rows=publication(5); edit(rows)
            with self.assertRaises(RuntimeError): validate_online_publication(rows,version=5)

    def native_fixture(self, root):
        root=Path(root); path=root/'world/native.pt'; path.parent.mkdir()
        path.write_bytes(b'fixture')
        d=descriptor(); d.update(weights_file=str(path.resolve()), relative_weights_file=str(path.relative_to(root)),
                                weights_sha256=hashlib.sha256(b'fixture').hexdigest())
        manifest={k:v for k,v in d.items() if k not in ('weights_file','relative_weights_file',
            'weights_sha256','weights_bytes','native_cpu_finite_checked')}
        payload={'format':'weaver_online_joint_weights_v1','manifest':manifest,
                 'model':{str(n):Tensor() for n in range(866)}}
        request={'descriptor':d,'config':CONFIG,'run_id':RUN,'policy_step':27,'reason':'final',
                 'checkpoint_root':str(root.resolve())}
        return request,payload,path

    def test_native_cpu_mmap_finite_checksum_scan_once_with_cache(self):
        with tempfile.TemporaryDirectory() as root:
            request,payload,path=self.native_fixture(root); torch=fake_torch(payload)
            first=audit_online_native_cpu(request,torch_module=torch)
            second=audit_online_native_cpu(request,torch_module=torch)
            self.assertTrue(first['valid'] and first['finite'])
            self.assertEqual(first['training_steps'],5)
            self.assertFalse(first['cuda_initialized'])
            self.assertTrue(second['cache_reused'])
            torch.load.assert_called_once_with(str(path),map_location='cpu',weights_only=True,mmap=True)

    def test_native_cpu_rejects_changed_file_nan_or_incomplete_native_inventory(self):
        for mode in ('checksum','nan','inventory','embedded'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as root:
                request,payload,path=self.native_fixture(root)
                if mode=='checksum': path.write_bytes(b'changed')
                if mode=='inventory': payload['model'].pop('865')
                if mode=='embedded': payload['manifest']['run_id']='other'
                with self.assertRaises(RuntimeError):
                    audit_online_native_cpu(request,torch_module=fake_torch(payload,finite=mode!='nan'))

    def test_cached_native_audit_cannot_mask_post_audit_file_change(self):
        with tempfile.TemporaryDirectory() as root:
            request,payload,path=self.native_fixture(root)
            audit_online_native_cpu(request,torch_module=fake_torch(payload))
            path.write_bytes(b'changed')
            with self.assertRaises(RuntimeError):
                audit_online_native_cpu(request,torch_module=fake_torch(payload))

    def test_native_artifact_path_must_stay_in_checkpoint_root(self):
        with tempfile.TemporaryDirectory() as root:
            request,payload,path=self.native_fixture(root)
            request['checkpoint_root']=str((Path(root)/'another').resolve())
            with self.assertRaises(RuntimeError): audit_online_native_cpu(request,torch_module=fake_torch(payload))

    def test_parent_audit_aggregates_joint_world_but_loads_policy_only_in_merged(self):
        with tempfile.TemporaryDirectory() as root:
            data=Path(root); directory=data/'training/checkpoints'; directory.mkdir(parents=True)
            path=directory/'latest_weights.pt';path.write_bytes(b'fixture')
            native={**descriptor(), 'valid':True,'finite':True,'training_steps':5,
                    'cuda_initialized':False,'python':CONFIG['backend_options']['training_python']}
            m={'metadata':{'run_id':RUN,'service_plan':PLAN,'online_world':descriptor(),
                'stop_policy':{'until_manual_stop':True,'until_deadline':False,
                               'deadline_unix_seconds':None,'stop_on_target':False}},
               'world_size':4,'reason':'final','steps':{'policy':27},'weights_only':True,
               'tensor_counts':{'policy':1}}
            (directory/'manifest.json').write_text(json.dumps(dict(m,weights_file=path.name,weights_bytes=7)))
            payload={'manifest':m,'models':{kind:{'state_dict':{'w':Tensor()},'trainable_names':['w'],
                'full_state':False} for kind in ('policy',)}}
            process=SimpleNamespace(run=Mock(return_value=SimpleNamespace(stdout=json.dumps(native))))
            namespace={'os':os,'legacy':SimpleNamespace(stamp=lambda:'CPU'),'json':json,
                'Path':Path,'SOURCE':SOURCE,'subprocess':process,'cpu_helper_environment':lambda:{'CUDA_VISIBLE_DEVICES':''}}
            audit=load('gpu47_safety.py','audit_final_weights',namespace)
            value=audit(data,state(),torch_module=fake_torch(payload),persist=False)
            self.assertTrue(value['valid'],value)
            self.assertEqual(value['steps'],{'policy':27,'ctrl':5})
            self.assertEqual(value['monolith_steps'],{'policy':27})
            self.assertEqual(value['tensor_counts']['ctrl'],866)
            self.assertEqual(process.run.call_args.args[0][0],CONFIG['backend_options']['training_python'])
            self.assertIn('--audit-online-native',process.run.call_args.args[0])


if __name__=='__main__':unittest.main()
