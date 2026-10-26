"""Bounded GPU4-7 ZeRO-2 check of Policy GAS, replicas and CPU exports.

The reference is another original ZeRO-2 engine processing the same 576
samples in one accumulation step. BF16 accumulation orders differ; this
checks the declared 0.002 absolute tolerance, not bitwise equivalence.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

PIPE = '/dev/shm/mps_lcx1_gpu47_pzero2_20261020/pipe'
EXPECTED = [
    'GPU-c43d6671-0979-b38d-9eae-4fe0a9d71c8c',
    'GPU-fe0edfe4-b193-1af7-6a98-cbff40214f74',
    'GPU-8c9bc264-7f09-989e-1ec7-cecd6cde782c',
    'GPU-07ae0bbc-8923-bd0c-199e-c726821e655d',
]


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def cards_from_receipt(path):
    receipt = json.loads(path.read_text(encoding='utf-8'))
    assert receipt['scope'] == [4, 5, 6, 7], 'MPS receipt scope must be physical GPU4-7'
    cards = sorted(receipt['cards'], key=lambda card: int(card['index']))
    assert [int(card['index']) for card in cards] == [4, 5, 6, 7]
    assert [card['uuid'] for card in cards] == EXPECTED, 'GPU4-7 physical UUIDs changed'
    return cards


def worker(args):
    cards = cards_from_receipt(args.receipt)
    card = cards[args.rank]
    os.environ.update(CUDA_VISIBLE_DEVICES=card['uuid'], CUDA_MPS_PIPE_DIRECTORY=PIPE,
        ACCERL_USE_MPS='1', ACCERL_PHYSICAL_GPU=str(card['index']),
        ACCERL_EXPECTED_GPU_UUID=card['uuid'], LOCAL_RANK='0', RANK=str(args.rank),
        WORLD_SIZE='4', MASTER_ADDR='127.0.0.1', MASTER_PORT=str(args.port),
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    import torch
    import torch.distributed as dist
    import deepspeed
    from training_topology import verify_bound_gpu, export_cpu_weights
    from parallel_checkpoint import _weights
    from policy_backend_audit import policy_zero2_report
    torch.set_num_threads(1)
    binding = verify_bound_gpu()
    deepspeed.init_distributed(dist_backend='nccl')

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.frozen = torch.nn.Linear(16, 16)
            self.frozen.requires_grad_(False)
            self.adapter = torch.nn.Linear(16, 16, bias=False)
            self.value = torch.nn.Linear(16, 1)

        def forward(self, x):
            return self.value((self.frozen(x) + self.adapter(x)).tanh())

        def groups(self):
            return [{'params': list(self.adapter.parameters()), 'name': 'policy', 'lr': 1e-3},
                    {'params': list(self.value.parameters()), 'name': 'value', 'lr': 2e-3}]

    def config(micro, gas):
        return {'train_micro_batch_size_per_gpu': micro, 'gradient_accumulation_steps': gas,
            'optimizer': {'type': 'AdamW', 'params': {}}, 'bf16': {'enabled': True},
            'zero_optimization': {'stage': 2, 'allgather_partitions': True,
                'allgather_bucket_size': 500000000, 'reduce_scatter': True,
                'reduce_bucket_size': 500000000, 'overlap_comm': True,
                'contiguous_gradients': True}, 'gradient_clipping': 1.0,
            'steps_per_print': 1000000}

    torch.manual_seed(82)
    model = Toy().cuda().bfloat16()
    reference = copy.deepcopy(model)
    initial = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
    engine, _, _, _ = deepspeed.initialize(model=model, config=config(4, 36),
                                          model_parameters=model.groups())
    ref_engine, _, _, _ = deepspeed.initialize(model=reference, config=config(144, 1),
                                              model_parameters=reference.groups())
    report = policy_zero2_report(engine, args.rank)
    errors = []
    hashes_by_update = []
    for update in range(2):
        generator = torch.Generator(device='cpu').manual_seed(1000 + args.rank * 100 + update)
        x = torch.randn(144, 16, generator=generator).cuda().bfloat16()
        y = (x[:, :1].float() * 0.3 - x[:, 1:2].float() * 0.2).bfloat16()
        for micro in range(36):
            start = micro * 4
            loss = (engine(x[start:start + 4]).float() - y[start:start + 4].float()).square().mean()
            engine.backward(loss)
            engine.step()
            expected_updates = update + int(micro == 35)
            assert int(engine.global_steps) == expected_updates, 'Wrong GAS update boundary'
            assert int(engine.skipped_steps) == 0
        reference_loss = (ref_engine(x).float() - y.float()).square().mean()
        ref_engine.backward(reference_loss)
        ref_engine.step()
        assert int(ref_engine.global_steps) == update + 1 and int(ref_engine.skipped_steps) == 0
        payload = export_cpu_weights(engine.module, 'policy', update + 1)
        hashes = [None] * 4
        dist.all_gather_object(hashes, payload['hash'])
        assert len(set(hashes)) == 1, 'Policy ZeRO-2 replicas disagree'
        hashes_by_update.append(hashes)
        reference_state = ref_engine.module.state_dict()
        error = max((value.float() - reference_state[name].detach().cpu().float()).abs().max().item()
                    for name, value in payload['tensors'].items())
        errors.append(error)
        assert error <= 0.002, f'Declared BF16 reference tolerance exceeded: {error}'
    assert hashes_by_update[0][0] != hashes_by_update[1][0], 'Policy did not change on second update'
    state = engine.module.state_dict()
    for name, parameter in engine.module.named_parameters():
        if not parameter.requires_grad:
            assert torch.equal(initial[name], parameter.detach().cpu()), 'Frozen base changed'
    assert any(not torch.equal(initial[name], state[name].detach().cpu()) for name in initial
               if name.startswith('adapter.')), 'Policy adapter did not update'
    assert any(not torch.equal(initial[name], state[name].detach().cpu()) for name in initial
               if name.startswith('value.')), 'Value head did not update'
    snapshot = _weights(engine.module)
    expected_names = {name for name, value in engine.module.named_parameters() if value.requires_grad}
    assert set(snapshot['state_dict']) == expected_names
    assert all(value.device.type == 'cpu' and bool(torch.isfinite(value).all())
               for value in snapshot['state_dict'].values())
    snapshot_path = args.ranks / f'rank_{args.rank}_weights.pt'
    torch.save(snapshot, snapshot_path)
    loaded = torch.load(snapshot_path, map_location='cpu', weights_only=True, mmap=True)
    assert set(loaded['state_dict']) == expected_names
    assert all(torch.equal(value, loaded['state_dict'][name])
               for name, value in snapshot['state_dict'].items())
    atomic_json(args.ranks / f'rank_{args.rank}.json', {
        'rank': args.rank, 'pid': os.getpid(), 'binding': binding,
        'updates': int(engine.global_steps), 'skip': int(engine.skipped_steps),
        'global_batch': int(engine.train_batch_size()), 'microsteps': 72,
        'samples_consumed': 288, 'actual_engine': report,
        'reference': 'original_deepspeed_zero2_micro144_gas1_same_global_samples',
        'reference_max_abs_errors': errors, 'reference_absolute_tolerance': 0.002,
        'reference_bitwise_equivalence_claimed': False,
        'rank_hashes_by_update': hashes_by_update, 'adapter_and_value_changed': True,
        'frozen_base_unchanged': True, 'cpu_snapshot_tensor_count': len(expected_names),
        'cpu_snapshot_weights_only_read_verified': True})
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--rank', type=int)
    parser.add_argument('--port', type=int)
    parser.add_argument('--ranks', type=Path)
    args = parser.parse_args()
    if args.rank is not None:
        worker(args)
        return
    assert args.output is not None and args.output.is_absolute()
    assert not args.output.exists(), 'GPU test result must be a new path'
    cards_from_receipt(args.receipt)
    assert Path(PIPE).is_dir(), 'Expected independent GPU4-7 MPS pipe is missing'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ranks = args.output.with_name(args.output.stem + '_ranks')
    ranks.mkdir(exist_ok=False)
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    procs = []
    logs = []
    started = time.monotonic()
    try:
        for rank in range(4):
            log = (ranks / f'rank_{rank}.log').open('w', encoding='utf-8')
            logs.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', ACCERL_CPU_THREADS='1')
            procs.append(subprocess.Popen([sys.executable, '-u', __file__, '--receipt', str(args.receipt),
                '--rank', str(rank), '--port', str(port), '--ranks', str(ranks)],
                env=env, stdout=log, stderr=subprocess.STDOUT))
        deadline = started + 180
        while any(proc.poll() is None for proc in procs):
            if any(proc.poll() not in (None, 0) for proc in procs):
                raise RuntimeError('Policy ZeRO-2 GPU check failed; inspect rank logs')
            if time.monotonic() > deadline:
                raise TimeoutError('Policy ZeRO-2 GPU check exceeded its 180-second bound')
            time.sleep(1)
        assert all(proc.returncode == 0 for proc in procs)
        reports = [json.loads((ranks / f'rank_{rank}.json').read_text()) for rank in range(4)]
        assert all(row['updates'] == 2 and row['skip'] == 0 for row in reports)
        atomic_json(args.output, {'passed': True, 'elapsed_seconds': time.monotonic() - started,
            'scope': 'four_rank_policy_zero2_smoke_not_six_service_preflight',
            'rank_reports': reports})
    except BaseException as exc:
        atomic_json(args.output, {'passed': False, 'error': repr(exc),
                                  'elapsed_seconds': time.monotonic() - started})
        raise
    finally:
        # Only direct children created above; never global Ray/MPS cleanup.
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()
