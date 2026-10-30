"""Bounded real projection/cohort tests; no Ray, Torch, CUDA or model load."""
import asyncio
import types
import json
from pathlib import Path
import unittest
import numpy as np
from replay_projection import describe_window_shard, materialize_window_microshard, materialize_window_shard
from training_topology import GlobalBatchCoordinator


class Window:
    reads = 0
    peak_micro = 0
    def __init__(self, i):
        self.sample_id = str(i)
        self.instruction = 'task'
        self.rew = i % 2
        self.i = i
    @property
    def world_observations(self):
        Window.reads += 1
        return np.full((40, 2, 3, 2, 2), self.i % 256, np.uint8)
    @property
    def reward_primary(self):
        return np.full((3, 2, 2), self.i % 256, np.uint8)
    @property
    def act(self): return np.full((40, 7), self.i, np.float32)
    @property
    def backend_payload(self): return {'sample_id': self.sample_id, 'states': np.zeros((40, 8), np.float32)}


class Replay:
    def __init__(self):
        self.records = [Window(i) for i in range(1024)]
        self.stats = {'unique_episodes': 3, 'capture_meta': {'sample_ids': [r.sample_id for r in self.records],
                       'reservation_id': 'r1', 'reserved_ns': 123}}
        self.calls = self.micro_calls = self.releases = 0
        self.poison = False
        for name in ('reserve_windows', 'describe_window_shard', 'project_window_microshard',
                     'project_windows', 'release_windows'):
            setattr(self, name, types.SimpleNamespace(remote=getattr(self, '_' + name)))
    async def _reserve_windows(self, size):
        assert size == 1024
        self.calls += 1
        return {'ticket_id': 'ticket', 'global_batch_size': 1024}
    async def _describe_window_shard(self, ticket, rank, size, consumer):
        return describe_window_shard(self.records, self.stats, rank, size, 4, consumer)
    async def _project_window_microshard(self, ticket, rank, size, start, end):
        self.micro_calls += 1
        result = materialize_window_microshard(self.records, self.stats, rank, size, 4, start, end)
        if self.poison: result[-1]['capture_meta']['sample_ids'] = ['bad'] * 4
        return result
    async def _project_windows(self, ticket, rank, size, consumer):
        return materialize_window_shard(self.records, self.stats, rank, size, 4, consumer)
    async def _release_windows(self, ticket): self.releases += 1


class ProjectionTests(unittest.TestCase):
    def setUp(self): Window.reads = 0
    def test_reserve_has_no_pixels_and_only_micro_four_is_materialized(self):
        replay = Replay()
        description = describe_window_shard(replay.records, replay.stats, 2, 256, 4)
        self.assertEqual(Window.reads, 0)
        self.assertEqual(description['capture_meta']['sample_ids'][0], '512')
        batch = materialize_window_microshard(replay.records, replay.stats, 2, 256, 4, 12, 16)
        self.assertEqual(Window.reads, 4)
        self.assertEqual(batch[0].shape, (4, 40, 2, 3, 2, 2))
        self.assertEqual(batch[0].dtype, np.uint8)
        self.assertEqual(batch[-1]['capture_meta']['sample_ids'], ['524','525','526','527'])
        self.assertEqual(batch[-1]['capture_meta']['shard_start'], 524)
        self.assertEqual(batch[-1]['capture_meta']['global_batch_size'], 1024)
        self.assertEqual([p['sample_id'] for p in batch[-1]['backend_samples']], ['524','525','526','527'])
        np.testing.assert_array_equal(batch[1][:,0,0], [524,525,526,527])
    def test_reward_only_uses_last_primary_even_when_world_is_forty_frames(self):
        replay = Replay()
        reward = materialize_window_shard(replay.records,replay.stats,3,256,4,'reward')
        self.assertEqual(Window.reads, 0)
        self.assertEqual(reward[0].shape, (256,3,2,2))
        self.assertEqual(reward[1].shape, (256,0))
        self.assertNotIn('backend_samples', reward[-1])
    def test_real_episode_reference_keeps_native_pixels_and_action_state_alignment(self):
        from episode_frame_store import EpisodeRecord
        cfg=json.loads((Path(__file__).parent/'examples/weaver_integrated_internal.json').read_text(encoding='utf-8'))
        count=42
        pixels=np.arange(count*2*3*224*224,dtype=np.uint8).reshape(count,2,3,224,224)
        actions=np.arange(count*7,dtype=np.float32).reshape(count,7);actions[0]=0
        states=np.arange(count*8,dtype=np.float32).reshape(count,8)
        rewards=np.arange(count,dtype=np.float32);rewards[0]=0
        done=np.zeros(count,bool);done[-1]=True
        record=EpisodeRecord('real','pick',pixels,actions,rewards,states=states,
            terminated=done,truncated=np.zeros(count,bool),success=done,backend_config=cfg)
        window=record.window(1,40)
        # Accessing these paths works in a CPU interpreter without Torch; it
        # must not enter the old full-window float resize implementation.
        self.assertEqual(window.world_observations.dtype,np.uint8)
        self.assertEqual(window.world_observations.shape,(40,2,3,224,224))
        self.assertEqual(window.reward_primary.shape,(3,224,224))
        np.testing.assert_array_equal(window.reward_primary,pixels[40,0])
        np.testing.assert_array_equal(window.act,actions[1:41])
        np.testing.assert_array_equal(window.backend_payload['states'],states[1:41])
        self.assertTrue(window.backend_payload['action_mask'].all())
        self.assertEqual(record.window(0,40).backend_payload['action_mask'][0],False)


class CohortTests(unittest.IsolatedAsyncioTestCase):
    def setup_cohort(self):
        r=Replay()
        c=GlobalBatchCoordinator(None,r,4,('ctrl','reward'),pipeline_enabled=True,world_streaming=True)
        return r,c
    async def test_all_eight_consumers_share_exact_rank_order_and_release_after_streams(self):
        r,c=self.setup_cohort()
        streams=[await c.reserve_wm_stream(rank,256) for rank in range(4)]
        self.assertEqual(Window.reads,0)
        rewards=[await c.sample_wm(rank,256,'reward') for rank in range(4)]
        self.assertEqual(r.releases,0)
        ids=[]
        for rank,stream in enumerate(streams):
            used=[]
            for start in range(0,256,4):
                batch=await c.sample_wm_stream(stream['stream_id'],start,start+4)
                used.extend(batch[-1]['capture_meta']['sample_ids'])
            self.assertEqual(used,rewards[rank][-1]['capture_meta']['sample_ids'])
            ids.extend(used)
            receipt=await c.complete_wm_stream(stream['stream_id'])
            self.assertEqual(receipt['micro_projections'],64)
            self.assertEqual(r.releases,int(rank==3))
        self.assertEqual(len(set(ids)),1024)
        self.assertEqual(r.calls,1)
        self.assertEqual(r.micro_calls,256)
        self.assertEqual(c.wm_streams,{})
        self.assertEqual(c.claimed['wm'],set())
    async def test_full_world_shard_is_disabled(self):
        r,c=self.setup_cohort()
        with self.assertRaisesRegex(ValueError,'stream micro4'): await c.sample_wm(0,256,'ctrl')
        self.assertEqual(r.calls,0)
    async def test_out_of_order_micro_poison_releases_once_and_never_redraws(self):
        r,c=self.setup_cohort(); s=await c.reserve_wm_stream(0,256)
        with self.assertRaisesRegex(ValueError,'exactly once'): await c.sample_wm_stream(s['stream_id'],4,8)
        self.assertEqual(r.releases,1)
        with self.assertRaisesRegex(RuntimeError,'cohort failed'): await c.reserve_wm_stream(1,256)
        self.assertEqual(r.calls,1)
    async def test_partial_stream_cannot_be_completed(self):
        r,c=self.setup_cohort(); s=await c.reserve_wm_stream(0,256)
        await c.sample_wm_stream(s['stream_id'],0,4)
        with self.assertRaisesRegex(RuntimeError,'partially'): await c.complete_wm_stream(s['stream_id'])
        self.assertEqual(r.releases,1)
    async def test_corrupt_micro_identity_is_fatal(self):
        r,c=self.setup_cohort(); s=await c.reserve_wm_stream(0,256); r.poison=True
        with self.assertRaisesRegex(ValueError,'changed'): await c.sample_wm_stream(s['stream_id'],0,4)
        self.assertEqual(r.releases,1)
    async def test_next_world_stream_waits_until_reward_has_claimed(self):
        r,c=self.setup_cohort()
        for rank in range(4):
            s=await c.reserve_wm_stream(rank,256)
            for start in range(0,256,4): await c.sample_wm_stream(s['stream_id'],start,start+4)
            await c.complete_wm_stream(s['stream_id'])
        next_stream=asyncio.create_task(c.reserve_wm_stream(0,256))
        await asyncio.sleep(0)
        self.assertFalse(next_stream.done())
        for rank in range(4): await c.sample_wm(rank,256,'reward')
        await asyncio.wait_for(next_stream,1)
        self.assertEqual(r.calls,2)
        self.assertEqual(r.releases,1)


if __name__=='__main__': unittest.main()
