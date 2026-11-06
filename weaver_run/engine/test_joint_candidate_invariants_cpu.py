"""CPU-only immutable learning/inference/recipe checks for the fresh manual source."""
import ast,hashlib,json
from pathlib import Path
import unittest
HERE=Path(__file__).resolve().parent
EXPECTED_NODES={'ColocatedInferenceMixin': 'b456ed0e1759b429c5deffd5403472f8daa36c61db65cee5657351dc6207bf56', 'InferenceActor': 'd24701b2d3bcde4f8e8b687bb44c8ad2973144e0b0a38eb1e817d58a7f6988dc', 'RewardInferenceActor': '7217b84c77a1cab08894e1a3a87f60e7751fc701a18e8e02ef894591039758fc', 'CtrlWorldInferenceActor': '90f91cb997ae7eadc73980832ae9f1f804a13889d5b21c1ab6534594a55f29c2', '_verify_optimizer_update': 'ace3faf68ac391fef70ad854fd7b5779078e9f98561f7ba75847d22c58b3b2d3', '_get_current_lr': '40ff2905c227da6e1c4244962ae7ee1b379e45348691e78dbfe6a3e3222c84dc', '_train_policy': 'f286e60303e645b26635bf3915df613f4f24f49d877973a91c92b2b23d06ff3a', '_train_ctrl_world': '8f56dc5257300a57367fd7dcf4f50a133cc91660bedd029881a1c6446b8a1f88', '_train_reward_model': 'a75d8dc25ec8fff178afd161bdb3d46d0f748a24943774b4fef4922d2ea3eb96'}
EXPECTED_FILES={'run.sh': '48f0eec3863b6aba16c3c299f686c5e772bf2bbd41bcb7757115bbd8f76160d7', 'split_serial.py': '73c597b5f582f40ab6c11dd8c7864f023ef5a266d46eb4b5cbee0290d4034540', 'shared_inference.py': '3fe741ab4f14e66c2742f13cb2d8261fc54f16034e1b63cbaaf6a21367e0e5e6', 'memory_policy.py': '7b3ad59d661a092d7427deab8350fd24937fa21e7c7068378586f331b7cdedb8', 'reward_features.py': 'e16fb3d886a801f4e9a1e4834b7ef3568bf4675d331f820be9c1d4cf93128e43', 'fsdp_engine.py': 'f0b79ae819550131ec02d2af82468e8662c11c4b0349f73238d3c322e00e55ef', 'parallel_checkpoint.py': '983bea482fb473bcdc5983294f7685a0c93ddcf8029d30ddc99a225074153c30', 'policy_backend_audit.py': 'b2aac854dedc01f03025dd0de49400777812c4d526fc5edcf28952bb81184f89', 'world_backend_audit.py': '6c0192a7e0657e89095e3bfa5f46cb56b4ca2e51f4c0a29f31fb3719b5653042', 'training_stop_policy.py': '95f2ee2df1b3d247a95c631ff476b3ee9dbfcc324e0dcc987d532ea910df772c'}
# The mixin's publication method is intentionally rewritten for IPC.
EXPECTED_NODES.pop('ColocatedInferenceMixin')
EXPECTED_FILES.pop('run.sh')

def normalize(value):
    if isinstance(value,ast.AST):return [type(value).__name__,{key:normalize(item) for key,item in ast.iter_fields(value) if key!='type_params'}]
    if isinstance(value,list):return [normalize(item) for item in value]
    return value
class Invariants(unittest.TestCase):
    def test_six_workflow_metadata_tags_are_preserved(self):
        tree=ast.parse((HERE/'train.py').read_text(encoding='utf-8'))
        tags=[]
        for node in ast.walk(tree):
            if not isinstance(node,ast.Call):continue
            for keyword in list(node.keywords):
                if keyword.arg=='_capture_context':
                    self.assertIsInstance(node.func,ast.Name)
                    self.assertEqual(node.func.id,'profile_rpc')
                    tags.append(ast.literal_eval(keyword.value)['workflow'])
                    node.keywords.remove(keyword)
        self.assertCountEqual(tags,['imagined_action','imagined_bootstrap','real_rollout',
                                   'imagined_generation','imagined_generation','evaluation'])
        # IPC changes only the publication/control path; learning nodes below retain exact AST hashes.

    def test_five_core_methods_and_inference_classes_match_parent(self):
        tree=ast.parse((HERE/'train.py').read_text(encoding='utf-8'))
        actual={node.name:hashlib.sha256(json.dumps(normalize(node),sort_keys=True).encode()).hexdigest()
                for node in ast.walk(tree) if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in EXPECTED_NODES}
        self.assertEqual(actual,EXPECTED_NODES)
    def test_world_reward_recipe_inference_and_launcher_bytes_match_parent(self):
        self.assertEqual({name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in EXPECTED_FILES},EXPECTED_FILES)
        launch=(HERE/'run.sh').read_text(encoding='utf-8')
        original=launch.replace('export ACCERL_NSIGHT_NVTX=0\nexport ACCERL_NSIGHT_STEPS=\n','')
        for model,steps in [('policy',100),('value',100),('ctrl',20),('reward',20)]:
            self.assertEqual(launch.count(f'--{model}-warmup-steps {steps} '),1)
            original=original.replace(f'--{model}-warmup-steps {steps} ',f'--{model}-warmup-steps 500 ')
        self.assertEqual(hashlib.sha256(original.encode()).hexdigest(),
                         '48f0eec3863b6aba16c3c299f686c5e772bf2bbd41bcb7757115bbd8f76160d7')
    def test_mode_hook_wraps_whole_policy_phase_outside_learning(self):
        tree=ast.parse((HERE/'train.py').read_text(encoding='utf-8'))
        phase=next(node for node in ast.walk(tree) if isinstance(node,ast.AsyncFunctionDef) and node.name=='_run_serial_epoch')
        calls=[ast.unparse(node.func) for node in ast.walk(phase) if isinstance(node,ast.Call)]
        self.assertEqual(calls.count('self._policy_joint_batch_mode.begin'),1)
        self.assertEqual(calls.count('self._policy_joint_batch_mode.complete'),1)
        code=ast.unparse(phase)
        self.assertLess(code.index('self._policy_joint_batch_mode.begin'),code.index('self._run_policy_epoch'))
        self.assertGreater(code.index('self._policy_joint_batch_mode.complete'),code.index('self._run_policy_epoch'))
    def test_manual_controller_has_uncapped_final_receipt_audit_and_bounded_live_tail(self):
        runner=(HERE/'runner.py').read_text(encoding='utf-8')
        helper=(HERE/'policy_batch_receipts.py').read_text(encoding='utf-8')
        self.assertIn('LIMIT 80',runner)
        self.assertNotIn('limit 200',runner.lower())
        self.assertIn('validate_policy_batch_modes_from_db',runner)
        self.assertNotIn('steps=15',helper)
        self.assertIn('mode=ro',helper)
        self.assertIn('ipcpwmanual_',runner)
        self.assertIn('cuda_ipc_weight_sync_gpu47_20261022',(HERE/'gpu47_safety.py').read_text())
if __name__=='__main__':unittest.main()
