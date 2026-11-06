"""CPU-only cloudpickle export smoke check. Never imports or starts Ray."""
from pathlib import Path
import argparse
import ast
import asyncio
import contextvars
import hashlib
import json
import pickle
import sys
import time
from types import SimpleNamespace


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cloudpickle-root',required=True,
        help='Existing isolated serializer tool path; no dependencies are installed')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    sys.path.insert(0,str(Path(args.cloudpickle_root).resolve()))
    import cloudpickle
    import capture_hooks
    from cost_scheduler_runtime import CostAdmissionBroker
    checks={}
    # Confirm this serializer actually rejects the dangerous captured object.
    context=contextvars.ContextVar('negative_export_fixture')
    def bad_factory():
        async def bad():return context.get(None)
        return bad
    try:cloudpickle.dumps(bad_factory())
    except TypeError:checks['rejects_captured_contextvar']=True
    else:checks['rejects_captured_contextvar']=False

    async def original(owner):
        from capture_hooks import current_request_metadata
        return current_request_metadata()
    wrapper=capture_hooks._request_wrapper(original)
    restored=pickle.loads(cloudpickle.dumps(wrapper))
    checks['actual_dynamic_request_wrapper_roundtrip']=callable(restored)
    events=[]
    owner=SimpleNamespace(capture_role='policy_infer',capture_model_version=3,
        telemetry=SimpleNamespace(event=lambda *a,**kw:events.append((a,kw))))
    result=asyncio.run(restored(owner,telemetry_meta={'request_id':'serialize-request',
        'workflow':'real_rollout','submitted_ns':time.time_ns()}))
    checks['restored_wrapper_preserves_workflow_and_id']=(result['request_id']=='serialize-request'
        and result['_capture_payload']['workflow']=='real_rollout')
    checks['context_reset_after_return']=capture_hooks.current_request_metadata() is None
    exported=type('DynamicCostBroker',(CostAdmissionBroker,),{'acquire':CostAdmissionBroker.acquire})
    checks['dynamic_broker_class_roundtrip']=issubclass(pickle.loads(cloudpickle.dumps(exported)),CostAdmissionBroker)

    # Serialize the actual actor method bodies without importing train.py (which
    # would require a real Ray/model environment). This is not a Ray runtime test.
    source=ast.parse((root/'train.py').read_text('utf8'))
    cls=next(x for x in source.body if isinstance(x,ast.ClassDef) and x.name=='InferenceActor')
    selected=('configure_cost_scheduler','scheduled_evaluate_values','_scheduled_policy_step','request','apply_rank_weights')
    for name in selected:
        node=next(x for x in cls.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name==name)
        namespace={'__name__':'__main__','asyncio':asyncio,'time':time}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'actual InferenceActor.'+name,'exec'),namespace)
        checks['actual_actor_method_export:'+name]=callable(pickle.loads(cloudpickle.dumps(namespace[name])))
    module_path=Path(cloudpickle.__file__).resolve().parent/'cloudpickle.py'
    report={'protocol':'cost_scheduler_cpu_serialization_v1','observed_ns':time.time_ns(),
        'cloudpickle_version':cloudpickle.__version__,'cloudpickle_source':str(module_path),
        'cloudpickle_sha256':hashlib.sha256(module_path.read_bytes()).hexdigest(),
        'checks':checks,'passed':all(checks.values()),'actual_ray_started':False,
        'gpu_used':False,'scope':'actual dynamic function/class CPU serialization, not Ray actor execution'}
    (root/'validation/serializer_validation.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
