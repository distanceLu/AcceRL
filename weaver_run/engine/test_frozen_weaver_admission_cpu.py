"""CPU-only frozen WEAVER admission, placements and immutable bundle tests."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import training_topology as topology
from world_backends import WorldBackendConfig, build_service_plan
from world_backend_audit import frozen_world_identity, frozen_world_asset_report, validate_frozen_world_receipts

CONFIG = Path(__file__).parent / 'examples/weaver_frozen_external.json'


def configuration():
    return json.loads(CONFIG.read_text(encoding='utf-8'))


class FrozenContracts(unittest.TestCase):
    def test_exact_evaluated_bundle_and_topology(self):
        cfg = WorldBackendConfig.from_mapping(configuration())
        cfg.validate_admission()
        plan = build_service_plan(cfg)
        self.assertEqual(plan.active_kinds, ('policy', 'reward'))
        self.assertEqual(plan.inference_kinds, ('policy', 'ctrl', 'reward'))
        self.assertEqual(plan.wm_consumers, ('reward',))
        self.assertEqual(plan.expected_wm_consumers, 4)
        self.assertEqual(plan.expected_gpu_model_processes, 20)
        self.assertEqual(plan.publication_kinds, ('policy', 'reward'))
        self.assertEqual(plan.checkpoint_kinds, ('policy', 'reward'))
        self.assertEqual(plan.logical_update_ratio, '5:0:1')
        self.assertEqual(plan.expected_optimizer_steps(27), {'policy': 27, 'reward': 5})
        self.assertEqual(plan.due_kinds(5), ('policy', 'reward'))
        identity = frozen_world_identity(cfg)
        self.assertEqual(identity['fixed_step'], 3200)
        self.assertEqual(identity['bundle_sha256'], '6b45705115034914803224c3653533e30236fffdac544b46400b5af68bfd3443')

    def test_metadata_relabel_or_initial_snapshot_rejected(self):
        changes = [lambda c:c['backend_options'].update(frozen_checkpoint_sha256='f'*64),
                   lambda c:c['checkpoint_metadata']['training_provenance'].update(optimizer_steps=0),
                   lambda c:c['checkpoint_metadata']['training_provenance'].pop('run_id'),
                   lambda c:c.update(world_trainable='false'),
                   lambda c:c.update(reward_provider='world_head'),
                   lambda c:c['backend_options'].update(publication_weights='ema')]
        for change in changes:
            cfg = configuration(); change(cfg)
            with self.subTest(config=cfg), self.assertRaises((ValueError, RuntimeError)):
                frozen_world_identity(cfg)
        cfg=configuration();cfg['checkpoint_metadata']['training_provenance']['optimizer_steps']=3100
        with self.assertRaisesRegex(RuntimeError, '3200'):
            frozen_world_identity(cfg)

    def test_two_trainers_and_three_inference_roles_require_five_slots_per_card(self):
        cfg=WorldBackendConfig.from_mapping(configuration()); plan=build_service_plan(cfg)
        bindings=[{'resource':f'gpu{i}','index':str(i),'uuid':f'uuid{i}'} for i in range(4)]
        with patch.dict('os.environ', {'ACCERL_USE_MPS':'0'}):
            init=topology.ray_init_options(bindings,topology.SPLIT_LAYOUT,plan.active_kinds,plan.inference_kinds)
            self.assertEqual(sum(init['resources'].values()),20)
            for role in ('policy_trainer','reward_trainer','policy','ctrl','reward'):
                self.assertEqual(topology.actor_options(bindings,role,0,topology.SPLIT_LAYOUT,plan.active_kinds)['resources'],{'gpu0':1.0})
            with self.assertRaises(ValueError):
                topology.actor_options(bindings,'ctrl_trainer',0,topology.SPLIT_LAYOUT,plan.active_kinds)
            with self.assertRaises(ValueError):
                topology.ray_init_options(bindings,topology.SPLIT_LAYOUT,plan.active_kinds,('policy','reward'))

    def test_runtime_receipts_must_include_real_five_roles(self):
        plan=build_service_plan(configuration())
        rows=[{'phase':phase,'kind':kind,'rank':rank} for phase,kinds in
              [('trainer',plan.trainer_kinds),('inference',plan.inference_kinds)]
              for kind in kinds for rank in range(4)]
        plan.validate_runtime(role_receipts=rows)
        for bad in (rows[:-1], rows+[{'phase':'trainer','kind':'ctrl','rank':0}]):
            with self.assertRaises(ValueError):plan.validate_runtime(role_receipts=bad)

    def test_frozen_child_identity_receipts_are_four_distinct_gpu_models(self):
        cfg=configuration();identity=frozen_world_identity(cfg)
        rows=[{'actor_id':str(i),'pid':100+i,'details':dict(identity,rank=i,gpu_uuid=f'GPU-{i}')} for i in range(4)]
        self.assertTrue(validate_frozen_world_receipts(rows,cfg))
        for change in (lambda r:r[0]['details'].update(bundle_sha256='0'*64),
                       lambda r:r[0].update(pid=r[1]['pid']),
                       lambda r:r[0]['details'].update(gpu_uuid=r[1]['details']['gpu_uuid']),
                       lambda r:r[0]['details'].update(world_trainable=True)):
            bad=copy.deepcopy(rows);change(bad)
            with self.assertRaises(RuntimeError):validate_frozen_world_receipts(bad,cfg)

    def test_bundle_documents_normalizer_and_weight_hash_are_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); cfg=configuration()
            weights=b'exact native weights CPU fixture'; sha=hashlib.sha256(weights).hexdigest()
            cfg['backend_options'].update(checkpoint_dir=str(root),frozen_checkpoint_sha256=sha)
            metadata=cfg['checkpoint_metadata'];metadata['load_report']['checkpoint_sha256']=sha
            norm={'state':{'mean':[0.]*8,'std':[1.]*8},'actions':{'mean':[0.]*7,'std':[1.]*7}}
            metadata['normalization_sha256']=hashlib.sha256(json.dumps(norm,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            files={'checkpoint.pt':weights,'config.yaml':b'{}','backend_metadata.json':json.dumps(metadata).encode(),
                   'norm_stats_relabel.json':json.dumps(norm).encode()}
            for name,data in files.items():(root/name).write_bytes(data)
            (root/'bundle_manifest.json').write_text(json.dumps({n:hashlib.sha256(v).hexdigest() for n,v in files.items()}))
            report=frozen_world_asset_report(cfg,verify_checkpoint=True)
            self.assertTrue(report['checkpoint_sha256_verified'])
            self.assertFalse(report['real_model_loaded'])
            (root/'checkpoint.pt').write_bytes(b'tampered')
            with self.assertRaisesRegex(RuntimeError,'weight file SHA'):
                frozen_world_asset_report(cfg,verify_checkpoint=True)
            (root/'norm_stats_relabel.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'document hash'):
                frozen_world_asset_report(cfg)


if __name__=='__main__':unittest.main()
