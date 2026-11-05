"""Four real CPU Gloo ranks around the actual JIT runtime and a toy optimizer.

Exercises unequal context work, global EMA, and collective failure votes.
This does not instantiate DeepSpeed, Ray, OpenVLA, NCCL or CUDA.
"""
from datetime import timedelta
import json
import os
from pathlib import Path
import tempfile
import time

os.environ['CUDA_VISIBLE_DEVICES']=''


def worker(rank, directory):
    import torch
    import torch.distributed as dist
    from advantage_ema import raw_sums, moments
    from test_jit_gae_cpu import cohort, ToyModel, owner_for, execute
    torch.set_num_threads(1);torch.manual_seed(47)
    store=dist.FileStore(str(Path(directory)/'rendezvous'),4)
    dist.init_process_group('gloo',store=store,rank=rank,world_size=4,timeout=timedelta(seconds=150))
    try:
        replay,batch,_,ids=cohort(rank=rank)
        # Rank-specific context length: later selected index trims more prefix,
        # while the selected shard size and GAS count remain identical.
        ids[0]=replay.segments['a'+str(rank)].sample_id(1+rank)
        batch,_=replay.materialize(ids,jit=True)
        for seg in replay.segments.values():
            for obs in seg.observations:obs['value']+=rank*.7
        owner=owner_for(ToyModel(),batch,ids,rank=rank)
        def verify(engine,kind,before,skips):
            local=torch.tensor([engine.global_steps-before,engine.skipped_steps-skips])
            rows=[torch.empty_like(local) for _ in range(4)];dist.all_gather(rows,local)
            if any(row.tolist()!=[1,0] for row in rows):raise RuntimeError('Failed optimizer rank group')
        owner._verify_optimizer_update=verify
        execute(owner,batch)
        sums=raw_sums(torch.tensor([r['advantage'] for r in owner._jit_receipts]))
        dist.all_reduce(sums)
        mean,var=moments(sums)
        torch.testing.assert_close(owner.advantage_ema.mean,mean)
        torch.testing.assert_close(owner.advantage_ema.variance,var)
        assert sums[0].item()==16 and owner.advantage_ema.updates==1
        state=torch.stack((owner.advantage_ema.mean,owner.advantage_ema.variance))
        gathered=[torch.empty_like(state) for _ in range(4)];dist.all_gather(gathered,state)
        assert all(torch.equal(state,row) for row in gathered)
        initial=state.clone()
        next_batch=list(batch);next_batch[0]=[dict(obs,_jit_version=6) for obs in batch[0]]
        owner.capture_policy_metadata['fresh_versions']=[6,1]
        owner.model.skip=(rank==2)
        try:execute(owner,next_batch)
        except RuntimeError as error:assert 'Failed optimizer rank group' in str(error)
        else:raise AssertionError('One-rank skipped optimizer was accepted')
        assert owner._jit_receipts is None and owner.advantage_ema.updates==1
        torch.testing.assert_close(torch.stack((owner.advantage_ema.mean,owner.advantage_ema.variance)),initial)
        # A rank-local identity fault must stop every rank before another forward.
        owner.global_step=6
        if rank==1:owner.capture_policy_metadata['sample_ids']=ids[::-1]
        before=owner.model.micro_steps
        try:execute(owner,next_batch)
        except RuntimeError as error:assert 'rank preparation/forward failed' in str(error)
        else:raise AssertionError('Rank-local bad cohort accepted')
        assert owner.model.micro_steps==before
        row=dict(rank=rank,passed=True,backend=dist.get_backend(),global_selected_count=16,
                 ema_mean=float(mean),ema_variance=float(var),dependency_states=owner.events[0][1]['details']['dependency_states'],
                 successful_window=True,one_rank_skip_rejected=True,one_rank_identity_rejected=True,
                 cuda_initialized=torch.cuda.is_initialized())
        (Path(directory)/f'rank{rank}.json').write_text(json.dumps(row,indent=2),encoding='utf8')
    finally:
        dist.destroy_process_group()


def main():
    import torch.multiprocessing as mp
    import traceback
    root=Path(__file__).resolve().parent
    started=time.time_ns();out=root/'validation';out.mkdir(exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix='jit_gloo_') as directory:
            mp.spawn(worker,args=(directory,),nprocs=4,join=True)
            ranks=[json.loads((Path(directory)/f'rank{i}.json').read_text()) for i in range(4)]
    except BaseException:
        report=dict(started_ns=started,completed_ns=time.time_ns(),passed=False,error=traceback.format_exc(),
                    actual_deepspeed=False,actual_ray=False,actual_cuda=False)
        (out/'jit_gloo_cpu_failed.json').write_text(json.dumps(report,indent=2),encoding='utf8')
        raise
    report=dict(started_ns=started,completed_ns=time.time_ns(),passed=all(r['passed'] for r in ranks),
                ranks=ranks,actual_gloo_processes=4,actual_deepspeed=False,actual_ray=False,actual_cuda=False)
    (out/'jit_gloo_cpu.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
