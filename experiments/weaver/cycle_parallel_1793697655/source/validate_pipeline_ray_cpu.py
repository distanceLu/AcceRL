"""Isolated real-Ray CPU smoke; no model imports, training, or GPU resources."""
from __future__ import annotations

import ast
from collections import deque
import json
import os
from pathlib import Path
import random
import tempfile
import time
from typing import List
import gc

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
import ray
from pipeline_runtime import EpisodeReplayMixin
from pipeline_budget import PipelineBudgetConfig
from training_topology import GlobalBatchCoordinator


class SyntheticWindowOccurrence:
    """Distinct test occurrence IDs share four real window views, not pixel copies.

    These are explicit CPU transport fixtures, never production Replay samples.
    Their underlying EpisodeRecord retains canonical images and label geometry.
    """
    def __init__(self,window,ordinal):
        self.window,self.ordinal=window,ordinal
    def __getattr__(self,name):return getattr(object.__getattribute__(self,'window'),name)
    @property
    def sample_id(self):return 'synthetic-ray-occurrence-%04d'%self.ordinal
    @property
    def backend_payload(self):return dict(self.window.backend_payload,sample_id=self.sample_id)


def inject_integrated_fixture(self,world_config):
    from episode_frame_store import EpisodeRecord
    count=43
    frames=np.broadcast_to(np.arange(count,dtype=np.uint8)[:,None,None,None,None],
                           (count,2,3,224,224)).copy()
    actions=np.broadcast_to(np.arange(count,dtype=np.float32)[:,None],(count,7)).copy()
    states=np.broadcast_to(np.arange(count,dtype=np.float32)[:,None],(count,8)).copy()
    success=np.zeros(count,bool);success[-1]=True
    record=EpisodeRecord('synthetic-ray-integrated','fixture task',frames,actions,
        success.astype(np.float64),states=states,terminated=success,truncated=np.zeros(count,bool),
        success=success,metadata={'label_source':'libero_env_step','test_fixture':True},
        backend_config=world_config)
    self.frame_store.register(record)
    self.buffer=deque([SyntheticWindowOccurrence(record.window(i%4,40),i) for i in range(1024)],maxlen=1024)
    return {'synthetic_fixture':True,'distinct_occurrence_ids':1024,'distinct_source_windows':4,
            'canonical_pixel_bytes':frames.nbytes,'frames':count,'cuda_initialized':False}


def load_actor_class(integrated_fixture=False):
    tree = ast.parse(Path(__file__).with_name('train.py').read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WMReplayBufferActor')
    node.decorator_list = []
    ns = {'__name__': '__main__', 'EpisodeReplayMixin': EpisodeReplayMixin,
          'deque': deque, 'random': random, 'np': np, 'List': List,
          'WMExperience': object, 'setup_debugger': lambda *args: None,
          # Fixture episodes are all train. The real holdout routing is exercised
          # separately with the actual split function in the integration suite.
          'stable_episode_fraction': lambda episode_id: 1.0}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual WM actor>', 'exec'), ns)
    cls=ns['WMReplayBufferActor']
    if integrated_fixture:cls.inject_integrated_fixture=inject_integrated_fixture
    return cls


def check_integrated_roundtrip(actors):
    """Four actual Ray streams and bounded real40-frame micro payloads.

    Full64-micro/update completion is covered by the CPU cohort suite. This
    finite RPC check fetches one micro per rank, then deliberately rejects the
    incomplete stream and verifies fail-closed ticket release. It trains nothing.
    """
    world_config=json.loads(Path(__file__).with_name('examples').joinpath('weaver_integrated_internal.json').read_text())
    budget=PipelineBudgetConfig(max_episode_frames=48,max_pending_episodes=2,
        max_resident_frame_bytes=64*1024**2,seeds_per_episode=2)
    wm=ray.remote(num_cpus=0)(load_actor_class(integrated_fixture=True)).remote(
        capacity=1024,ctrl_eval_capacity=2,pipeline_config=budget.to_dict(),world_config=world_config,
        context_frames=32,window_frames=40,policy_chunks_per_seed=4,minimum_train_windows=1024)
    actors.append(wm)
    fixture=ray.get(wm.inject_integrated_fixture.remote(world_config),timeout=30)
    assert fixture['canonical_pixel_bytes']<16*1024**2
    coordinator=ray.remote(num_cpus=0)(GlobalBatchCoordinator).remote(None,wm,4,
        wm_consumers=('ctrl',),pipeline_enabled=True,world_streaming=True)
    actors.append(coordinator)
    streams=ray.get([coordinator.reserve_wm_stream.remote(rank,256) for rank in range(4)],timeout=30)
    expected=streams[0]['capture_meta']['reward_cohort']
    assert expected['positive_count']==256 and expected['negative_count']==9984
    assert expected['positive_weight']==39. and expected['rm_active'] is True
    assert all(s['capture_meta']['reward_cohort']==expected for s in streams)
    all_ids=[sample_id for stream in streams for sample_id in stream['capture_meta']['sample_ids']]
    assert len(all_ids)==len(set(all_ids))==1024
    peak_bytes=0
    for rank,stream in enumerate(streams):
        result=ray.get(coordinator.sample_wm_stream.remote(stream['stream_id'],0,4),timeout=30)
        images,actions,rewards,text,metadata=result
        assert images.shape==(4,40,2,3,224,224) and images.dtype==np.uint8
        assert actions.shape==(4,40,7)
        assert metadata['capture_meta']['sample_ids']==stream['capture_meta']['sample_ids'][:4]
        assert metadata['capture_meta']['reward_cohort']==expected
        assert metadata['capture_meta']['global_batch_size']==1024
        assert metadata['capture_meta']['data_parallel_rank']==rank
        for i,payload in enumerate(metadata['backend_samples']):
            assert payload['label_source']=='libero_env_step' and payload['states'].shape==(40,8)
            np.testing.assert_array_equal(images[i,:,0,0,0,0],actions[i,:,0])
            np.testing.assert_array_equal(payload['states'][:,0],actions[i,:,0])
        peak_bytes=max(peak_bytes,int(images.nbytes))
        # Do not retain four48MB arrays or a1024-window global pixel batch.
        del result,images,actions,rewards,text,metadata,payload
        gc.collect()
    try:
        ray.get(coordinator.complete_wm_stream.remote(streams[0]['stream_id']),timeout=30)
    except Exception as error:
        assert 'partially projected rank stream' in str(error)
    else:raise AssertionError('A one-micro stream was incorrectly accepted as complete')
    status=ray.get(wm.pipeline_status.remote(),timeout=15)
    assert status['pending_draws']==0
    ray.get(wm.stop_pipeline.remote(),timeout=15)
    return {'passed':True,'scope':'synthetic repeated-window occurrence fixture, real Ray CPU projection only',
        'fixture':fixture,'world_consumers_reserved':4,'global_unique_occurrence_ids':1024,
        'reward_cohort':expected,'micro_rpc_count':4,'micro_shape':[4,40,2,3,224,224],
        'max_pixel_payload_bytes':peak_bytes,'partial_stream_rejected_and_released':True,
        'optimizer_update_completed':False,'gpu_used':False}


def main():
    started = time.time()
    run_dir = Path(tempfile.mkdtemp(prefix='dp_', dir='/tmp'))
    config = PipelineBudgetConfig(max_episode_frames=32, max_pending_episodes=4,
                                 max_resident_frame_bytes=64*1024**2, seeds_per_episode=2)
    ray.init(address='local', num_cpus=2, num_gpus=0, include_dashboard=False,
             object_store_memory=256*1024**2, _temp_dir=str(run_dir / 'ray'),
             runtime_env={'env_vars': {'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
                                      'PYTHONPATH': str(Path(__file__).resolve().parent)}})
    actors = []
    try:
        wm = ray.remote(num_cpus=0)(load_actor_class()).remote(capacity=8, ctrl_eval_capacity=2,
                pipeline_config=config.to_dict(), context_frames=7, window_frames=11,
                policy_chunks_per_seed=4, minimum_train_windows=8)
        actors.append(wm)
        for i in range(3):
            lease = ray.get(wm.acquire_real.remote(i), timeout=15)
            assert lease is not None
            frames = np.arange(14*2*3*224*224, dtype=np.uint8).reshape(14,2,3,224,224)
            labels = np.zeros(14, dtype=bool)
            payload = dict(episode_id=f'cpu-{i}', instruction='fixture task', frames=frames,
                           actions=np.zeros((14,7), np.float32), rewards=np.zeros(14, np.float64),
                           terminated=labels, truncated=labels, success=labels,
                           frame_groups=[{'group_id': f'group{i}', 'start_frame': 0, 'end_frame': 13}])
            receipt = ray.get(wm.finish_real_episode.remote(lease['token'], payload), timeout=15)
            assert receipt['windows'] == 4
        coordinator = ray.remote(num_cpus=0)(GlobalBatchCoordinator).remote(None, wm, 4,
                                      wm_consumers=('ctrl','reward'), pipeline_enabled=True)
        actors.append(coordinator)
        requests = [(rank, kind, coordinator.sample_wm.remote(rank,2,kind))
                    for rank in (3,1,0,2) for kind in ('reward','ctrl')]
        results = {(rank, kind): ray.get(ref, timeout=30) for rank,kind,ref in requests}
        ids = []
        for rank in range(4):
            world, reward = results[rank,'ctrl'], results[rank,'reward']
            assert world[-1]['capture_meta'] == reward[-1]['capture_meta']
            np.testing.assert_array_equal(world[0][:,-1,0], reward[0])
            np.testing.assert_array_equal(world[2], reward[2])
            ids.extend(world[-1]['capture_meta']['sample_ids'])
            assert reward[0].shape == (2,3,224,224) and reward[1].shape == (2,0)
        assert len(set(ids)) == 8
        job = ray.get(wm.acquire_imagine.remote('imag0'), timeout=15)
        assert job['frames'].shape == (7,2,3,224,224)
        ray.get(wm.finish_imagine.remote(job['token'],3), timeout=15)
        ray.get(wm.policy_drawn.remote(576), timeout=15)
        status = ray.get(wm.pipeline_status.remote(), timeout=15)
        assert status['pending_draws'] == 0
        assert status['budget']['policy_added'] == 3
        assert status['budget']['policy_sampled'] == 576
        stop = ray.get(wm.stop_pipeline.remote(), timeout=15)
        assert stop['budget']['active_real'] == 0 and stop['budget']['active_imagine'] == 0
        assert ray.get(wm.acquire_real.remote('stopped'), timeout=15) == {'stopped': True}
        explicit_config = PipelineBudgetConfig(initial_state_enabled=True, real_workers=1,
            max_real_in_flight=1, real_env_step_budget=23, max_episode_frames=32,
            max_pending_episodes=4, seeds_per_episode=2, max_resident_frame_bytes=64*1024**2)
        explicit = ray.remote(num_cpus=0)(load_actor_class()).options(
            runtime_env={'env_vars': {'ACCERL_DATA_DIR': str(run_dir / 'explicit')}}).remote(
                capacity=16, ctrl_eval_capacity=2, pipeline_config=explicit_config.to_dict(),
                context_frames=2, window_frames=3, policy_chunks_per_seed=4, minimum_train_windows=1)
        actors.append(explicit)
        pool = {'count': 50, 'shape': [50, 92], 'sha256': 'e'*64,
                'max_action_steps': 8, 'reset_steps': 2}
        assigned_ids = []
        for ordinal in range(3):
            lease = ray.get(explicit.acquire_real.remote(0, pool), timeout=20)
            assigned = lease['initial_state']
            assigned_ids.append(assigned['initial_state_id'])
            action_steps = min(6, assigned['max_action_steps'])
            count = action_steps + 1
            labels = np.zeros(count, dtype=bool)
            truncated = labels.copy()
            truncated[-1] = True
            payload = dict(episode_id=f'explicit-{ordinal}', instruction='fixture task',
                frames=np.zeros((count,2,3,224,224), np.uint8), actions=np.zeros((count,7), np.float32),
                rewards=np.zeros(count, np.float64), terminated=labels, truncated=truncated, success=labels,
                metadata={'initial_state_id': assigned['initial_state_id'], 'env_seed': assigned['env_seed'],
                    'reset_env_steps': 2, 'initial_state_hash': 'f'*64, 'termination_reason': 'fixture_horizon'})
            ray.get(explicit.finish_real_episode.remote(lease['token'], payload), timeout=20)
        exhausted = ray.get(explicit.acquire_real.remote(0, pool), timeout=20)
        assert exhausted['budget_exhausted'] is True
        checkpoint = ray.get(explicit.initial_state_checkpoint.remote(), timeout=20)
        counters = checkpoint['scheduler']['counters']
        assert counters['consumed_env_steps'] == 23 and counters['episodes_completed'] == 3
        assert len(set(assigned_ids)) == 3 and counters['completed_coverage_count'] == 3
        assert (run_dir / 'explicit/initial_state_queue.json').is_file()
        integrated=check_integrated_roundtrip(actors)
        report = {'status':'passed', 'scope':'real Ray CPU actor/RPC, synthetic episodes, no models',
                  'gpu_used':False, 'world_shape':list(results[0,'ctrl'][0].shape),
                  'reward_shape':list(results[0,'reward'][0].shape),
                  'world_pixel_bytes_per_rank':results[0,'ctrl'][0].nbytes,
                  'reward_pixel_bytes_per_rank':results[0,'reward'][0].nbytes,
                  'unique_sample_ids':len(set(ids)), 'same_eight_consumer_cohort':True,
                  'pipeline_status':status, 'initial_state_queue': counters,
                  'explicit_assignment_rpc_and_checkpoint': True,'integrated_world_roundtrip':integrated,
                  'elapsed_seconds':time.time()-started}
        Path(__file__).with_name('validation_ray_cpu.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
    finally:
        for actor in actors:
            ray.kill(actor, no_restart=True)
        ray.shutdown()


if __name__ == '__main__':
    main()
