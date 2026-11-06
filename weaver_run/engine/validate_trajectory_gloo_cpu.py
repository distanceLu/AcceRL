"""Actual four-process CPU Gloo validation of the 576-chunk trajectory recipe.

No Ray, ZeRO2, OpenVLA, NCCL or CUDA. Uses the candidate JIT runtime and toy SGD.
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
    from test_trajectory_sampling_cpu import replay, owner_for_rank
    from test_jit_gae_cpu import execute
    torch.set_num_threads(1);torch.manual_seed(47)
    store=dist.FileStore(str(Path(directory)/'rendezvous'),4)
    dist.init_process_group('gloo',store=store,rank=rank,world_size=4,timeout=timedelta(seconds=120))
    try:
        r=replay(108);token,global_ids=r.reserve(576)
        ids=global_ids[rank*144:(rank+1)*144]
        owner,batch,records=owner_for_rank(r,ids,rank)
        def verify(engine,kind,before,skips):
            local=torch.tensor([engine.global_steps-before,engine.skipped_steps-skips])
            rows=[torch.empty_like(local) for _ in range(4)];dist.all_gather(rows,local)
            if any(row.tolist()!=[1,0] for row in rows):raise RuntimeError('Failed optimizer rank group')
        owner._verify_optimizer_update=verify
        for window in range(2):
            if window:
                owner.capture_policy_metadata['fresh_versions']=[6,1]
                for obs in batch[0]:obs['_jit_version']=6
            previous_mean=owner.advantage_ema.mean.clone() if window else None
            previous_var=owner.advantage_ema.variance.clone() if window else None
            execute(owner,batch)
            sums=raw_sums(torch.tensor([row['advantage'] for row in owner._jit_receipts]));dist.all_reduce(sums)
            mean,var=moments(sums)
            assert sums[0].item()==576
            if window:
                expected_mean=.9*previous_mean+.1*mean
                expected_var=.9*previous_var+.1*var+.09*(mean-previous_mean).square()
            else:expected_mean,expected_var=mean,var
            torch.testing.assert_close(owner.advantage_ema.mean,expected_mean)
            torch.testing.assert_close(owner.advantage_ema.variance,expected_var)
            detail=owner.events[-1][1]['details']
            assert detail['whole_imagined_segments']==27 and detail['selected_grad_states']==144
            assert detail['initialization_prepass'] == (window==0)
            if window:
                assert detail['prepass_selected_and_context_states']==0
                assert abs(detail['normalization_mean']-float(previous_mean))<1e-6
        imag={row['segment_id'] for row in records if row['source']=='imagined'}
        gathered=[None]*4;dist.all_gather_object(gathered,sorted(imag))
        assert len(set(s for rows in gathered for s in rows))==108
        r.commit(token,ids);r.cancel(token)
        assert sum(s['uses'] for s in r.samples.values())==144
        state=torch.stack((owner.advantage_ema.mean,owner.advantage_ema.variance)).clone()
        owner.capture_policy_metadata['fresh_versions']=[7,1]
        for obs in batch[0]:obs['_jit_version']=7
        owner.model.skip=(rank==2)
        try:execute(owner,batch)
        except RuntimeError as error:assert 'Failed optimizer rank group' in str(error)
        else:raise AssertionError('One-rank skip was accepted')
        assert owner.advantage_ema.updates==2 and owner._jit_receipts is None
        torch.testing.assert_close(torch.stack((owner.advantage_ema.mean,owner.advantage_ema.variance)),state)
        # One rank swaps a chunk with another complete group. Same count/quota,
        # but incomplete trajectory -> every rank fails before another forward.
        owner.global_step=7
        owner.model.skip=False
        if rank==1:
            outsider=next(s for s in r.segments.values() if s.source=='imagined' and s.segment_id not in imag)
            bad,_=r.materialize([outsider.sample_id(0)],jit=True)
            pos=next(i for i,o in enumerate(batch[0]) if o['_jit_segment']['source']=='imagined')
            batch[0][pos]=dict(bad[0][0],_jit_version=7)
            owner.capture_policy_metadata['sample_ids']=list(ids)
            owner.capture_policy_metadata['sample_ids'][pos]=outsider.sample_id(0)
        before=owner.model.micro_steps
        try:execute(owner,batch)
        except RuntimeError as error:assert 'rank preparation/forward failed' in str(error)
        else:raise AssertionError('Incomplete trajectory accepted')
        assert owner.model.micro_steps==before
        row=dict(rank=rank,passed=True,selected=144,real=36,imagined=108,whole_imagined_segments=27,
            global_ema_selected_count=576,dependency_states=detail['dependency_states'],
            two_successful_windows=True,one_rank_skip_rejected=True,one_rank_fragment_rejected=True,
            ema_mean=float(state[0]),ema_variance=float(state[1]),cuda_initialized=torch.cuda.is_initialized())
        (Path(directory)/f'rank{rank}.json').write_text(json.dumps(row,indent=2),encoding='utf8')
    finally:dist.destroy_process_group()


def main():
    import torch.multiprocessing as mp
    import traceback
    root=Path(__file__).resolve().parent;out=root/'validation';out.mkdir(exist_ok=True)
    started=time.time_ns()
    try:
        with tempfile.TemporaryDirectory(prefix='trajectory_gloo_') as directory:
            mp.spawn(worker,args=(directory,),nprocs=4,join=True)
            ranks=[json.loads((Path(directory)/f'rank{i}.json').read_text()) for i in range(4)]
    except BaseException:
        report=dict(started_ns=started,completed_ns=time.time_ns(),passed=False,error=traceback.format_exc())
        (out/'trajectory_gloo_cpu_failed.json').write_text(json.dumps(report,indent=2),encoding='utf8')
        raise
    report=dict(started_ns=started,completed_ns=time.time_ns(),passed=True,ranks=ranks,
        actual_gloo_processes=4,actual_deepspeed=False,actual_ray=False,actual_openvla=False,actual_cuda=False)
    (out/'trajectory_gloo_cpu.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
