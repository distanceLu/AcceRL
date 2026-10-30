"""Actual private-pipe CPU fixture. It never loads a model, Torch or CUDA."""
import os
from types import SimpleNamespace
import numpy as np


class FixtureOperations:
    def __init__(self, backend=None, config=None, published_version=0):
        self.published_version=published_version
    def set_published_version(self, version):
        self.published_version=version
    def init_resident_rollout(self, context, version):
        return {'backend':'weaver','version':version,'instruction':context.get('instruction','task'),
                'latent':np.arange(65536,dtype=np.float32),'step':0,
                'initial_reward_score':.25,'initial_terminal':False,'reward_semantics':'delta_native_score'}
    def request_wire(self, state, actions, instruction=None):
        if instruction is not None and instruction!=state['instruction']:
            raise ValueError('Rollout task instruction changed')
        state=dict(state,step=state['step']+len(actions))
        return {'served_version':self.published_version,'next_state':state,
                'observations':np.full((1,2,3,2,2),state['step'],dtype=np.float32),
                'predicted_states':np.full((len(actions),8),state['step'],dtype=np.float32),
                'prediction_horizon':len(actions),'observation_indices':[len(actions)-1]}
    init_rollout=init_resident_rollout
    request=request_wire


class NullTelemetry:
    def __init__(self,*args): pass
    def event(self,*args,**kwargs): pass
    def flush(self,*args,**kwargs): pass
    def close(self): pass


def main():
    # Exercise the real worker's inference protocol branches. Production parent
    # guards remain unchanged; Linux has independent guard/lifetime tests.
    import weaver_online_gpu_worker as worker
    import world_backends.runtime as runtime
    import capture_telemetry
    import sys
    identity=worker.process_identity(os.getpid()) or {'pid':os.getpid(),'uid':0,'birth':0}
    worker.process_identity=lambda pid:identity if pid==os.getpid() else None
    worker.arm_parent_guard=lambda:None
    runtime.WeaverActorOperations=FixtureOperations
    capture_telemetry.Telemetry=NullTelemetry
    sys.modules['cuda_ipc_weights']=SimpleNamespace(apply_cuda_ipc_weights=lambda model,payload,*a,**kw:
        {'version':payload['version']})
    def load(raw,role,rank):
        opts=raw.get('backend_options',{}).get('optimizations',{})
        return (SimpleNamespace(model=object(),optimizations=opts),SimpleNamespace(backend_options={}),
            {'worker_identity':identity,'gpu_uuid':'cpu-fixture','gpu':{'physical_gpu':'4'},'geometry':{'horizon':8}})
    worker.load_backend=load
    worker.main()


if __name__=='__main__': main()
