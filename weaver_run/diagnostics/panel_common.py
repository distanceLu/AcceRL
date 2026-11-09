"""Frozen-panel protocol and CPU-only episode contracts."""
import hashlib
import json
from pathlib import Path
import random
import numpy as np

PROTOCOL='weaver_frozen_learning_panel_v1'
STATES=sorted(random.Random(20261109).sample(range(50),16))
GPUS=('GPU-c43d6671-0979-b38d-9eae-4fe0a9d71c8c','GPU-fe0edfe4-b193-1af7-6a98-cbff40214f74',
      'GPU-8c9bc264-7f09-989e-1ec7-cecd6cde782c','GPU-07ae0bbc-8923-bd0c-199e-c726821e655d')

def save(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:4],'big')

def spec():
    data='/mnt/data/lcx1/dong/need_data/'
    models={}
    for key,dirname,run,p,w in (
        ('original','weaver_cycle_parallel_gpu47_20261103_v1','weaverintegrated_1103_172031_a7557d7d',451,90),
        ('restore','weaver_cycle_restore_gpu47_20261105_v1','weaverintegrated_1105_140654_70115c34',457,91)):
        root=Path(data+dirname)/'formal_v3'
        models[key]=dict(run_id=run,policy_step=p,world_step=w,
            policy=str(root/'training/checkpoints/latest_weights.pt'),
            world=str(root/f'training/checkpoints/checkpoints_world/world_p{p:08d}_w{w:08d}_final.pt'),
            config=str(root/'world_backend_config.json'),db=str(root/'training/collection.sqlite'))
    return dict(protocol=PROTOCOL,models=models,initial_states=STATES,noise_seeds=[19,47],
        anchors=[0,31],horizons=[8,16,32],max_episode_actions=220,seed=20261109,
        max_seconds=14400,real_chunks=144,imagined_chunks=432,imagined_trajectory_chunks=4,
        gradient_scope='fresh frozen-model panel, NOT reconstruction of historical asynchronous gradients',
        normalization='recorded global EMA after the checkpoint optimizer update; no EMA update in diagnostics',
        insufficient_real_chunks='use largest available positive multiple of four, keep 1:3 ratio, report actual count; never duplicate real chunks',
        action_alignment='action_into_frame; frame0 has zero incoming action',
        no_optimizer_steps=True)

def validate_episode(ep):
    n=len(ep['actions'])
    if not 1<=n<=220 or ep['frames'].shape!=(n+1,2,3,224,224) or ep['frames'].dtype!=np.uint8:
        raise ValueError('Invalid measured frame/action geometry')
    if ep['states'].shape!=(n+1,8) or ep['actions'].shape!=(n,7):raise ValueError('Missing N+1 states')
    if not np.isfinite(ep['states']).all() or not np.isfinite(ep['actions']).all():raise ValueError('Nonfinite measured episode')
    if ep['initial_state_id'] not in STATES or type(ep['success']) is not bool:raise ValueError('Unknown fixed episode')
    if not ep['success'] and n!=220:raise ValueError('Incomplete episode cannot be scored as failure')
    return n

def context(ep,anchor,cfg):
    from episode_frame_store import EpisodeRecord,materialize_imagination_context
    n=validate_episode(ep)
    incoming=np.concatenate([np.zeros((1,7),np.float32),ep['actions']],axis=0)
    record=EpisodeRecord(ep['episode_id'],ep['instruction'],ep['frames'],incoming,
                         np.zeros(n+1),states=ep['states'])
    d=materialize_imagination_context(record,anchor-31,32,cfg['schema'].get('dt_seconds',.05))
    value=dict(obs=d['frames'],states=d['states'],actions=d['actions'],dt=d['dt'],
        valid_mask=d['valid_mask'],action_mask=d['action_mask'],frame_indices=d['frame_indices'],
        frame_start=d['frame_start'],schema=cfg['schema'],instruction=ep['instruction'],
        episode_id=ep['episode_id'],sample_id=ep['episode_id']+':'+str(anchor),
        request_lineage=dict(seed_anchor=anchor,context_start=anchor-31))
    if 'context_padding' in d:value['context_padding']=d['context_padding']
    return value

def discounted_targets(ep,indices):
    n=validate_episode(ep);indices=np.asarray(indices)
    if np.any((indices<0)|(indices>n)):raise ValueError('Target lies outside actual episode')
    return np.power(.99**(1/8),n-indices) if ep['success'] else np.zeros(len(indices))

def flatten_segments(segments):
    return [(j,i) for j,s in enumerate(segments) for i in range(len(s['rewards']))]
