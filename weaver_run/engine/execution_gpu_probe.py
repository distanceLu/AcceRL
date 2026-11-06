"""Bounded synthetic CUDA qualification inside the existing owned preflight.

This checks the actual refresh helper on each authorized card, not full-model
speed, real Ray deployment or learning quality. No process is launched here.
"""
def run(torch):
    import asyncio
    from types import SimpleNamespace
    from execution_control import refresh_block
    from policy_service_dispatch import PolicyServiceDispatch

    class Model:
        def prepare_inputs_batch(self, rows):
            return torch.tensor(rows, device='cuda', dtype=torch.float32)
        def __call__(self, rows):
            return None, rows * 2 + 1

    async def check():
        settings = dict(enabled=True, refresh_micro=2, action_micro=2,
            execution=dict(refresh_lease_micros=4, refresh_lease_seconds=5., refresh_trim='disabled'))
        dispatch = PolicyServiceDispatch(settings)
        owner = SimpleNamespace(_policy_dispatch=dispatch, model=Model(),
            capture_model_version=7, training_batch_limit=2)
        future = asyncio.get_running_loop().create_future()
        job = dispatch.enqueue_refresh(list(range(17)),7,future);job.ready=True
        before = torch.cuda.get_rng_state().clone()
        receipts=[]
        while not future.done():
            batch=dispatch.next_batch(7)
            values, receipt=await refresh_block(owner,batch,torch)
            batch.inputs=batch.inputs[:len(values)]
            dispatch.complete_batch(batch,receipt['execution_seconds'])
            dispatch.finish_refresh(batch,[float(x) for x in values],7)
            receipts.append(receipt)
        result=await future
        assert result['values']==[float(x*2+1) for x in range(17)]
        assert torch.equal(before,torch.cuda.get_rng_state())
        dispatch.assert_publication_allowed()
        assert sum(x['observations'] for x in receipts)==17
        return dict(passed=True, exact_values=True, rng_unchanged=True, version_pin_released=True,
                    micro_batch=2, max_block_micros=4, receipts=receipts,
                    scope='actual bounded refresh helper with synthetic CUDA tensors; not full model or Ray')
    return asyncio.run(check())
