"""AST isolation and fake RPC only; no Ray/Torch/training imports."""
import ast,collections,copy,hashlib,json,sys
from pathlib import Path
import types,unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'candidate'
PATCHED=HERE/'workflow_runtime' if (HERE/'workflow_runtime/train.py').is_file() else HERE
BASELINE=json.loads((HERE/'workflow_baseline.json').read_text())
def canonical(value):
    if isinstance(value,ast.AST):return [type(value).__name__,{key:canonical(item) for key,item in ast.iter_fields(value) if key!='type_params'}]
    if isinstance(value,list):return [canonical(item) for item in value]
    return value
def function(name,namespace):
    tree=ast.parse((PATCHED/'capture_hooks.py').read_text(encoding='utf-8'))
    node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),name,'exec'),namespace)
    return namespace[name]

class WorkflowTests(unittest.TestCase):
    def test_removing_six_metadata_kwargs_restores_the_worker_method_AST(self):
        current=ast.parse((PATCHED/'train.py').read_text(encoding='utf-8'));tags=[]
        for node in ast.walk(current):
            if not isinstance(node,ast.Call):continue
            for key in list(node.keywords):
                if key.arg=='_capture_context':tags.append(ast.literal_eval(key.value)['workflow']);node.keywords.remove(key)
        self.assertCountEqual(tags,['real_rollout','evaluation','imagined_action','imagined_bootstrap','imagined_generation','imagined_generation'])
        actual={cls.name+'.'+fn.name:hashlib.sha256(json.dumps(canonical(fn),sort_keys=True).encode()).hexdigest()
            for cls in current.body if isinstance(cls,ast.ClassDef) for fn in cls.body
            if isinstance(fn,(ast.FunctionDef,ast.AsyncFunctionDef)) and cls.name+'.'+fn.name in BASELINE['worker_method_AST_sha256']}
        self.assertEqual(actual,BASELINE['worker_method_AST_sha256'])

    def test_rpc_context_is_consumed_and_real_remote_arguments_and_payload_stay_same(self):
        events=[];calls=[]
        writer=types.SimpleNamespace(event=lambda kind,**kw:events.append((kind,kw)))
        owner=types.SimpleNamespace(wid=1,capture_episode_id='episode')
        actor=types.SimpleNamespace(request=types.SimpleNamespace(remote=lambda *a,**kw:calls.append((a,kw)) or {'_capture_payload':'payload','versions':{'policy':7}}))
        namespace={'_role':lambda _:'rollout','now_ns':lambda:1000,'new_id':lambda _:'request-id',
            '_writer':lambda _:writer,'time':types.SimpleNamespace(perf_counter_ns=lambda:100),
            'json':json,'MODEL_KEY':{'policy_infer':'policy'}}
        rpc=function('profile_rpc',namespace)
        with patch.dict(sys.modules,{'ray':types.SimpleNamespace(get=lambda x:x)}):
            result=rpc(owner,'policy_infer',actor,'request','input',deterministic=False,_capture_context={'workflow':'imagined_action'})
        self.assertEqual(result,'payload')
        args,kwargs=calls[0]
        self.assertEqual(args,('input',));self.assertIs(kwargs['deterministic'],False)
        self.assertNotIn('_capture_context',kwargs)
        self.assertEqual(kwargs['telemetry_meta']['workflow'],'imagined_action')
        self.assertEqual(events[0][1]['details']['workflow'],'imagined_action')
        self.assertEqual(events[-1][1]['details']['workflow'],'imagined_action')

    def test_mixed_batch_workflows_survive_dispatch_and_service_with_unknown(self):
        events=[];spans=[]
        def span(*args,**kwargs):
            spans.append((args,kwargs));return types.SimpleNamespace(__enter__=lambda:None)
        writer=types.SimpleNamespace(event=lambda kind,**kw:events.append((kind,kw)),span=span)
        owner=types.SimpleNamespace(capture_pending_meta=collections.deque([
            {'request_id':'a','submitted_ns':1,'arrival_ns':2,'workflow':'real_rollout'},
            {'request_id':'b','submitted_ns':1,'arrival_ns':2,'workflow':'evaluation'},
            {'request_id':'c','submitted_ns':1,'arrival_ns':2}]),capture_served_versions={},batch_size=3)
        namespace={'collections':collections,'now_ns':lambda:10,'new_id':lambda _:'unused',
            '_writer':lambda _:writer,'_role':lambda _:'policy_infer','_version':lambda _:3,
            'MODEL_KEY':{'policy_infer':'policy'}}
        function('batch_begin',namespace)(owner,[1,2,3])
        dispatch=events[0][1];service=spans[0][1]
        expected={'real_rollout':1,'evaluation':1,'unknown':1}
        self.assertEqual(dispatch['details']['workflow_counts'],expected)
        self.assertEqual(service['workflow_counts'],expected)
        self.assertEqual(sum(expected.values()),dispatch['actual_batch'])
        self.assertEqual(len(service['request_ids']),service['actual_batch'])

    def test_no_new_wait_sync_rng_or_model_operations_in_hook_patch(self):
        after=ast.parse((PATCHED/'capture_hooks.py').read_text(encoding='utf-8'))
        before=collections.Counter();actual=collections.Counter()
        for name,counts in BASELINE['hook_calls'].items():before.update(counts)
        for node in after.body:
            if isinstance(node,ast.FunctionDef) and node.name in BASELINE['hook_calls']:
                self.assertFalse(any(isinstance(n,ast.Await) for n in ast.walk(node)))
                actual.update(ast.unparse(n.func) for n in ast.walk(node) if isinstance(n,ast.Call))
        extra=actual-before
        self.assertEqual(set(extra),{'dict','collections.Counter','str','item.get','metadata.get'})

if __name__=='__main__':unittest.main()
