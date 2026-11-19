"""Versioned starting Policy shared by the original-site pruned and JIT experiments."""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace


def reference_policy(experiments=Path("/mnt/data/lcx1/dong/WEAVER/experiments")):
    checkpoint = (Path(experiments) / 'model-free-autobatch-prune-20261118-a' / 'formal_v3' /
                  'training' / 'checkpoints_initial' / 'latest_weights.pt')
    return dict(checkpoint=str(checkpoint), run_id='weaverintegrated_1119_001039_efef204a',
                sha256='1050accec71095b3d4a59c9f6973236d0f51426a86ab7799a11ad594f213204e',
                bytes=289371721, tensors=891, policy_step=0)


def validate_manifest(manifest, initial):
    if (manifest['steps'] != {'policy': 0} or manifest['metadata']['run_id'] != initial['run_id'] or
            manifest['tensor_counts'] != {'policy': initial['tensors']}):
        raise ValueError('Saved Policy does not match the pruned reference at P0')


def verify_reference_policy(root):
    initial = reference_policy()
    if json.loads((root/'experiment.json').read_bytes()).get('initial_policy') != initial:
        raise ValueError('Queued JIT initial Policy differs from the reference experiment')
    path = Path(initial['checkpoint'])
    manifest = json.loads((path.parent/'manifest.json').read_bytes())
    validate_manifest(manifest, initial)
    checksum = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    if path.stat().st_size != initial['bytes'] or checksum.hexdigest() != initial['sha256']:
        raise ValueError('Pruned P0 weights archive failed checksum verification')
    return initial


def restore(model, rank):
    from parallel_checkpoint import apply_parallel_weights
    root = Path(os.environ['ACCERL_EXPERIMENT_ROOT'])
    initial = reference_policy()
    checkpoint = Path(os.environ['ACCERL_TRAJECTORY_JIT_INITIAL_CHECKPOINT'])
    if (os.environ.get('ACCERL_TRAJECTORY_JIT') != '1' or str(checkpoint) != initial['checkpoint']
            or checkpoint.stat().st_size != initial['bytes']):
        raise ValueError('Unexpected initial Policy for trajectory JIT')
    manifest = apply_parallel_weights(SimpleNamespace(model=model), checkpoint, active_kinds=('policy',))
    validate_manifest(manifest, initial)
    receipt = dict(initial, rank=rank, loaded_before_deepspeed=True, optimizer_restored=False)
    output = root/f'initial_policy_rank_{rank}.json'
    temp = output.with_suffix('.tmp')
    temp.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf8')
    temp.replace(output)
