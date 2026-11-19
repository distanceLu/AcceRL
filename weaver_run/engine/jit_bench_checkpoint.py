"""Apply the verified weights-only P189 overlay before ZeRO creates optimizer masters."""
import json
import os
from pathlib import Path
from types import SimpleNamespace


def restore(model, rank):
    from parallel_checkpoint import apply_parallel_weights
    root=Path(os.environ['ACCERL_JIT_BENCH_OUTPUT'])
    plan=json.loads(Path(os.environ['ACCERL_JIT_BENCH_PLAN']).read_bytes())
    verified=json.loads((root/'checkpoint_verified.json').read_bytes())
    snapshot=Path(os.environ['ACCERL_JIT_BENCH_CHECKPOINT'])
    if (str(snapshot)!=plan['checkpoint'] or verified['sha256']!=plan['checkpoint_sha256']
            or verified['bytes']!=snapshot.stat().st_size):
        raise ValueError('P189 snapshot differs from the queue-verified archive')
    manifest=apply_parallel_weights(SimpleNamespace(model=model),snapshot,active_kinds=('policy',))
    if (manifest['steps']!={'policy':189} or manifest['metadata']['run_id']!=verified['run_id']
            or manifest['tensor_counts']['policy']!=verified['tensors']):
        raise ValueError('Loaded Policy overlay differs from the P189 receipt')
    receipt=dict(verified,rank=rank,loaded_before_deepspeed=True,optimizer_restored=False)
    dest=root/f'checkpoint_rank_{rank}.json'
    temp=dest.with_suffix('.tmp')
    temp.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    temp.replace(dest)
