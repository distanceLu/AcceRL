"""A ZeRO-2 rank may communicate on backward before the GAS boundary."""
from datetime import timedelta
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from test_jit_gae_cpu import execute
from test_gpu_saturation_optimizations_cpu import TinyPolicy, _make_deferred_vote_fixture


def _run_rank(rank, store_uri):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=store_uri, rank=rank, world_size=2,
                            timeout=timedelta(seconds=20))
    try:
        for fault in ('exception', 'nan'):
            class FaultOnFirstForward(TinyPolicy):
                def forward(self, inputs):
                    logits, value = super().forward(inputs)
                    if rank == 0 and torch.is_grad_enabled():
                        if fault == 'exception':
                            raise ValueError('injected rank-local forward error')
                        value = value * float('nan')
                    return logits, value

            owner, batch = _make_deferred_vote_fixture(FaultOnFirstForward())
            owner.rank = rank
            backward_calls = []

            def backward(loss):
                backward_calls.append(1)
                # Deliberately communicate at micro zero, as a ZeRO-2 engine can.
                dist.all_reduce(torch.ones((), dtype=torch.int32))
                raise AssertionError('Entered backward before all ranks validated forward')

            owner.model.backward = backward
            with patch.dict(os.environ, ACCERL_FAST_JIT_BOUNDARIES='1', ACCERL_DEFERRED_FORWARD_VOTE='1'):
                try:
                    execute(owner, batch)
                except RuntimeError as error:
                    if 'JIT rank preparation/forward failed' not in str(error):
                        raise
                else:
                    raise AssertionError('Invalid forward did not fail both ranks')
            assert not backward_calls and owner.model.global_steps == 0
            assert owner.advantage_ema.updates == 1 and owner._jit_receipts is None
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(os.name == 'posix' and dist.is_available() and dist.is_gloo_available(), 'POSIX CPU Gloo required')
class DistributedForwardFailureTests(unittest.TestCase):
    def test_failed_rank_votes_before_peers_run_backward(self):
        with tempfile.TemporaryDirectory() as folder:
            store_uri = (Path(folder) / 'gloo_store').as_uri()
            mp.spawn(_run_rank, args=(store_uri,), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
