"""The optional legacy backend argument remains usable without native-label bypass."""
import unittest
from pipeline_runtime import EpisodeReplayMixin
from test_episode_frame_store_cpu import record


class Owner(EpisodeReplayMixin):
    def _pipeline_event(self,*args,**kwargs):pass


class OptionalBackend(unittest.TestCase):
    def owner(self,config=None):
        obj=Owner();episode=record(count=14)
        obj.buffer=[episode.window(i,11) for i in range(4)]
        obj.sampled_episode_ids=set();obj.window_tickets={};obj.pipeline_world_config=config
        return obj
    def test_actual_default_none_reserves_projects_all_ranks_and_releases(self):
        owner=self.owner()
        ticket=owner.reserve_windows(4)
        ids=[]
        for rank in range(4):
            result=owner.project_windows(ticket['ticket_id'],rank,1,'ctrl')
            self.assertEqual(result[0].shape,(1,11,2,3,224,224))
            ids.extend(result[-1]['capture_meta']['sample_ids'])
            self.assertNotIn('reward_cohort',result[-1]['capture_meta'])
        self.assertEqual(len(set(ids)),4)
        owner.release_windows(ticket['ticket_id'])
        self.assertEqual(owner.window_tickets,{})
    def test_explicit_native_mode_does_not_turn_missing_labels_into_legacy(self):
        owner=self.owner({'reward_semantics':'delta_native_score'})
        with self.assertRaisesRegex(ValueError,'complete global1024'):
            owner.reserve_windows(4)
        self.assertEqual(owner.window_tickets,{})


if __name__=='__main__':unittest.main()
