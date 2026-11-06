"""Owned-device seam and explicit last-frame decoding without CUDA allocation."""
import copy
import json
from pathlib import Path
import unittest
import numpy as np

from world_backends import WorldBackendConfig
from world_backends.runtime import WeaverActorOperations
from world_backends.rollout import consume_predicted_chunk, _resident_seed_lifetime
from types import SimpleNamespace
from test_world_contracts_cpu import weaver_config


class DeviceOnly:
    def __init__(self,device):self.device=device


class Backend:
    def __init__(self,config,*,sparse=True,gpu_state=True):
        self.config=config.to_dict();self.device='cuda:0';self.sparse=sparse;self.gpu_state=gpu_state
        self.mutate=lambda r:None
    def set_published_version(self,version):self.version=version
    def init_rollout(self,context,version):
        return {'backend':'weaver','version':version,'instruction':'task',
            'encoder_id':self.config['schema']['encoder_id'],
            'normalizer_id':self.config['schema']['normalizer_id'],
            'initial_reward_score':.2,'initial_terminal':False,
            'latents':DeviceOnly(self.device) if self.gpu_state else np.zeros(3)}
    def predict_chunk(self,state,actions,published_version=None):
        n=len(actions)
        frames=np.arange(n*24,dtype=np.float32).reshape(1,n,2,3,2,2)
        result={'observations':frames[:,-1:] if self.sparse else frames,
            'predicted_states':np.arange(n*8,dtype=np.float32).reshape(1,n,8),
            'served_version':published_version,'next_state':dict(state),
            'reward_predictions':np.linspace(.3,.7,n,dtype=np.float32)[None,:],
            'terminal_predictions':np.zeros((1,n),bool),'valid_mask':np.ones((1,n),bool),
            'reward_semantics':'delta_native_score','reward_calibrated':False,'dt':.05,
            'encoder_id':self.config['schema']['encoder_id'],
            'normalizer_id':self.config['schema']['normalizer_id']}
        if self.sparse:result.update(observation_indices=[n-1],prediction_horizon=n)
        self.mutate(result)
        return result


def native_config():
    return WorldBackendConfig.from_mapping(json.loads((Path(__file__).parent/'examples/weaver_integrated_internal.json').read_text(encoding='utf8')))


class OwnedRuntime(unittest.TestCase):
    def make(self,**kwargs):
        cfg=native_config();backend=Backend(cfg,**kwargs)
        return backend,WeaverActorOperations(backend,cfg)
    def test_owned_gpu_state_is_private_but_public_entrypoints_remain_cpu(self):
        b,o=self.make();state=o.init_resident_rollout({'instruction':'task'})
        self.assertEqual(state['latents'].device,'cuda:0')
        with self.assertRaisesRegex(ValueError,'CPU'):o.init_rollout({'instruction':'task'})
        with self.assertRaisesRegex(ValueError,'CPU'):o.request(state,np.zeros((3,7)))
        result=o.request_wire(state,np.zeros((3,7)),'task')
        self.assertEqual(result['next_state']['latents'].device,'cuda:0')
        self.assertEqual(result['observations'].shape,(1,2,3,2,2))
        self.assertEqual(result['predicted_states'].shape,(3,8))
        self.assertEqual(len(result['reward_predictions']),3)
        self.assertEqual(result['dt'],[.05]*3)
    def test_owned_state_foreign_device_and_gpu_public_output_are_rejected(self):
        b,o=self.make();state=o.init_resident_rollout({})
        state['latents']=DeviceOnly('cuda:1')
        with self.assertRaisesRegex(ValueError,'another device'):o.request_wire(state,np.zeros((3,7)))
        state=o.init_resident_rollout({});b.mutate=lambda r:r.update(last_observation=DeviceOnly('cuda:0'))
        with self.assertRaisesRegex(ValueError,'CPU'):o.request_wire(state,np.zeros((3,7)))
    def test_owned_state_does_not_weaken_actions_instruction_or_version(self):
        b,o=self.make();state=o.init_resident_rollout({})
        for actions,instruction in [(np.zeros((3,8)),None),(np.full((3,7),np.nan),None),(np.zeros((3,7)),'other')]:
            with self.assertRaises(ValueError):o.request_wire(state,actions,instruction)
        o.set_published_version(1)
        with self.assertRaisesRegex(ValueError,'Stale'):o.request_wire(state,np.zeros((3,7)))
    def test_full_and_sparse_native_outputs_consume_identical_h_scores_and_state(self):
        fullb,fullo=self.make(sparse=False);sparseb,sparseo=self.make()
        full=fullo.request_wire(fullo.init_resident_rollout({}),np.zeros((8,7)))
        sparse=sparseo.request_wire(sparseo.init_resident_rollout({}),np.zeros((8,7)))
        a=consume_predicted_chunk(full,.2,2.,reward_semantics='delta_native_score')
        b=consume_predicted_chunk(sparse,.2,2.,reward_semantics='delta_native_score')
        self.assertAlmostEqual(a[0],b[0]);self.assertEqual(a[1:3],b[1:3]);self.assertEqual(a[-1],8);self.assertEqual(b[-1],8)
        np.testing.assert_array_equal(a[3],b[3]);np.testing.assert_array_equal(a[4],b[4])
    def test_sparse_geometry_masks_dt_scores_and_terminal_cannot_be_shortened(self):
        mutations=[lambda r:r.update(observation_indices=[0]),lambda r:r.update(prediction_horizon=2),
            lambda r:r.update(prediction_horizon=True),lambda r:r.update(valid_mask=np.ones((1,2),bool)),
            lambda r:r.update(valid_mask=np.array([[True,False,True]])),
            lambda r:r.update(dt=[.05,.05]),lambda r:r.update(reward_predictions=np.array([[.3,2.,.7]])),
            lambda r:r.update(terminal_predictions=np.array([[False,True,False]]))]
        for mutate in mutations:
            b,o=self.make();state=o.init_resident_rollout({});b.mutate=mutate
            with self.assertRaises(ValueError):o.request_wire(state,np.zeros((3,7)))
    def test_non_native_sparse_output_is_rejected(self):
        cfg=weaver_config(True);b=Backend(cfg,gpu_state=False);o=WeaverActorOperations(b,cfg)
        # This calibrated legacy fixture uses a valid seed; sparse geometry is
        # still forbidden even if a backend attaches otherwise aligned scores.
        original=b.init_rollout
        b.init_rollout=lambda *a:dict(original(*a),initial_success_probability=.2)
        with self.assertRaisesRegex(ValueError,'Sparse'):o.request(o.init_rollout({}),np.zeros((3,7)))
    def test_cpu_full_legacy_device_contract_unchanged(self):
        b,o=self.make(sparse=False,gpu_state=False)
        out=o.request(o.init_rollout({}),np.zeros((3,7)))
        self.assertEqual(out['observations'].shape,(3,2,3,2,2))
    def test_sparse_consumer_checks_every_intermediate_score_and_required_fields(self):
        b,o=self.make();out=o.request_wire(o.init_resident_rollout({}),np.zeros((3,7)))
        for key,value in [('reward_predictions',[.3,2.,.7]),('valid_mask',[True,True]),
                          ('dt',[.05,0,.05]),('terminal_predictions',[False,True,False])]:
            bad=copy.copy(out);bad[key]=value
            with self.assertRaises(ValueError):consume_predicted_chunk(bad,.2,reward_semantics='delta_native_score')
        with self.assertRaises(ValueError):consume_predicted_chunk(out,.2,external_reward=lambda x:(.5,False))


class SeedLifetime(unittest.TestCase):
    def test_latest_handle_released_for_normal_continue_and_break(self):
        seen=[];worker=SimpleNamespace(ctrl_world_infer=object())
        def rpc(worker,role,actor,method,state,**kw):seen.append((method,state['_resident']))
        for mode in ('normal','continue','break'):
            with _resident_seed_lifetime(worker,rpc) as latest:
                latest[0]={'_resident':mode+'-initial'}
                latest[0]={'_resident':mode+'-latest'}
                if mode=='continue':continue
                if mode=='break':break
        self.assertEqual(seen,[('release_rollout',m+'-latest') for m in ('normal','continue','break')])
    def test_primary_exception_is_preserved_and_stale_cleanup_is_expected(self):
        worker=SimpleNamespace(ctrl_world_infer=object())
        def bad(*a,**kw):raise RuntimeError('release failed')
        original=ValueError('primary')
        with self.assertRaises(ValueError) as found:
            with _resident_seed_lifetime(worker,bad) as latest:
                latest[0]={'_resident':'cap'};raise original
        self.assertIs(found.exception,original)
        def stale(*a,**kw):raise ValueError('Stale WEAVER rollout/cache version')
        with _resident_seed_lifetime(worker,stale) as latest:latest[0]={'_resident':'cap'}
        with self.assertRaisesRegex(RuntimeError,'release failed'):
            with _resident_seed_lifetime(worker,bad) as latest:latest[0]={'_resident':'cap'}


if __name__=='__main__':unittest.main()
