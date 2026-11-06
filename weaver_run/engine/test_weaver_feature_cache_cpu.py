"""Actual CPU tensor tests for selected frames, strict admission and CPU LRU."""
from __future__ import annotations

import copy
import ast
import gc
import hashlib
import os
from pathlib import Path
import unittest
import weakref

import numpy as np
import torch

from weaver_feature_cache import (FeatureNamespace, FrameFeatureKey, FrozenEncoderGuard,
    FrozenFeatureCache, build_frame_keys, encode_feature_batch, fingerprint,
    training_frame_selection, validate_cpu_window)


CAMERAS = ["full_image", "wrist_image"]
SCHEMA = {"domain":"libero", "action_schema_id":"libero_eef_delta_7_v1",
    "state_schema_id":"libero_eef_pos_axisangle_gripper_v1", "encoder_id":"frozen_real_encoder",
    "normalizer_id":"train_only_stats", "action_dim":7,"state_dim":8,
    "alignment":"action_into_frame", "image_keys":CAMERAS,"image_size":[8,12],
    "image_value_range":"minus_one_one","dt_seconds":.05}


def pinned_encoder_source():
    """Locate an explicitly supplied frozen upstream before local fallback."""
    for variable in ("WEAVER_TEST_UPSTREAM_SOURCE","WEAVER_TEST_UPSTREAM"):
        value=os.environ.get(variable)
        if value:
            root=Path(value)
            candidates=(root,root/"weaver/wm/encoders.py",root/"wm/encoders.py")
            for path in candidates:
                if path.is_file():return path
            raise AssertionError(variable+" does not resolve to the pinned encoder source: "+value)
    return Path(__file__).resolve().parents[4]/"analysis/weaver_integration_20261022/upstream/weaver/wm/encoders.py"


def sample(episode="real_episode_4", start=0, count=40):
    rng = np.random.default_rng(510 + start)
    action_mask = np.ones(count,dtype=bool)
    if start == 0:
        action_mask[0] = False
    dt = np.full(count,.05,dtype=np.float32)
    dt[~action_mask] = 0.
    return {"obs":rng.integers(0,256,size=(count,2,3,8,12),dtype=np.uint8),
        "states":rng.normal(size=(count,8)).astype(np.float32),
        "actions":rng.normal(size=(count,7)).astype(np.float32),
        "success":np.zeros(count,dtype=bool), "done":np.zeros(count,dtype=bool),
        "truncated":np.zeros(count,dtype=bool),"mask":np.ones(count,dtype=bool),
        "action_mask":action_mask,"dt":dt,"schema":copy.deepcopy(SCHEMA),
        "episode_id":episode,"sample_id":episode+":window"+str(start),
        "instruction":"put the bowl on the plate","label_source":"libero_env_step"}


def namespace(run="verified_run", encoder="a"*64, preprocess="b"*64):
    return FeatureNamespace(run,encoder,preprocess)


class Posterior:
    def __init__(self,mean):
        self.mean=mean
    def mode(self):
        return self.mean
    def sample(self):
        raise AssertionError("Deterministic feature path must not sample posterior noise")


class TinyFrozenEncoder(torch.nn.Module):
    """Actual frozen convolution+posterior.mode, independent per input frame."""
    def __init__(self):
        super().__init__()
        self.conv=torch.nn.Conv2d(3,4,1,bias=True)
        with torch.no_grad():
            self.conv.weight.copy_(torch.arange(12,dtype=torch.float32).reshape(4,3,1,1)/20)
            self.conv.bias.copy_(torch.arange(4,dtype=torch.float32)/10)
        self.requires_grad_(False)
        self.calls=0
        self.images=0
        self.batches=[]
    def forward(self,images):
        self.calls+=1
        batch,frames=images.shape[:2]
        self.images+=batch*frames
        self.batches.append(batch*frames)
        latent=Posterior(self.conv(images.flatten(0,1))).mode()
        return latent.flatten(2).transpose(1,2).reshape(batch,frames,-1,4)


class SelectionAndValidationTests(unittest.TestCase):
    def test_actual_geometry_exactly16_real_positions(self):
        plan=training_frame_selection(40,2,8,6,5)
        self.assertEqual(plan["memory_indices"],[1,6,11,16,21,26])
        self.assertEqual(plan["main_indices"],list(range(30,40)))
        self.assertEqual(len(plan["selected_indices"]),16)
        self.assertEqual([plan["selected_indices"][i] for i in plan["main_positions"]],list(range(30,40)))
        self.assertEqual([plan["selected_indices"][i] for i in plan["memory_positions"]],[1,6,11,16,21,26])
    def test_overlapping_memory_main_frames_encoded_once(self):
        plan=training_frame_selection(10,2,2,2,1)
        self.assertEqual(plan["selected_indices"],list(range(5,10)))
        self.assertEqual(plan["memory_positions"],[0,1])
    def test_insufficient_geometry_rejected(self):
        with self.assertRaises(ValueError): training_frame_selection(8,2,8,6,5)
    def test_entire40_window_checked_before_selection(self):
        raw=sample()
        result=validate_cpu_window(raw,SCHEMA,40,True,require_real_labels=True)
        self.assertEqual(result["frame_count"],40)
        self.assertIs(result["observations"],raw["obs"])
        for field in ("states","actions"):
            bad=copy.deepcopy(raw);bad[field][0,0]=np.nan
            with self.subTest(field=field),self.assertRaises(ValueError):
                validate_cpu_window(bad,SCHEMA,40,True)
        bad=copy.deepcopy(raw);bad["obs"]=bad["obs"].astype(np.float32)/255*2-1
        bad["obs"][0,0,0,0,0]=np.nan
        with self.assertRaises(ValueError): validate_cpu_window(bad,SCHEMA,40,True)
    def test_full_semantic_constraints_preserved(self):
        for mutate in (lambda x:x["mask"].__setitem__(0,False),
                       lambda x:x["done"].__setitem__(20,True),
                       lambda x:x["truncated"].__setitem__(0,True),
                       lambda x:x["dt"].__setitem__(1,.1),
                       lambda x:x["action_mask"].__setitem__(9,False),
                       lambda x:x["schema"].__setitem__("normalizer_id","wrong"),
                       lambda x:x.__setitem__("label_source","imagined")):
            bad=sample();mutate(bad)
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):
                validate_cpu_window(bad,SCHEMA,40,True,require_real_labels=True)
    def test_only_final_terminal_allowed(self):
        raw=sample();raw["done"][-1]=True
        validate_cpu_window(raw,SCHEMA,40,True)
    def test_reset_dt_explicit_and_nonreset_window(self):
        raw=sample(start=10)
        validate_cpu_window(raw,SCHEMA,40,True)
        bad=sample();bad["dt"][0]=.05
        with self.assertRaises(ValueError):validate_cpu_window(bad,SCHEMA,40,True)
    def test_floating_image_range_rejected(self):
        raw=sample();raw["obs"]=raw["obs"].astype(np.float32)
        with self.assertRaises(ValueError): validate_cpu_window(raw,SCHEMA,40,True)
    def test_finite_float64_state_overflow_rejected_before_gpu(self):
        raw=sample();raw['states']=raw['states'].astype(np.float64);raw['states'][0,0]=1e300
        with self.assertRaises(ValueError):validate_cpu_window(raw,SCHEMA,40,True)


class IdentityAndLRUTests(unittest.TestCase):
    def test_real_frame_identity_overlapping_windows_matches(self):
        first=sample(start=5);second=sample(start=6)
        second["obs"][0]=first["obs"][1]
        a=build_frame_keys(first,namespace(),[1])[0]
        b=build_frame_keys(second,namespace(),[0])[0]
        self.assertEqual(a,b)
        self.assertEqual(a[0].frame_index,6)
        self.assertNotEqual(a[0],a[1])
    def test_namespace_run_encoder_and_preprocessing_isolation(self):
        a=build_frame_keys(sample(),namespace(),[1])[0][0]
        for ns in (namespace(run="next_run"),namespace(encoder="c"*64),namespace(preprocess="c"*64)):
            b=build_frame_keys(sample(),ns,[1])[0][0]
            self.assertNotEqual(a.namespace_sha256,b.namespace_sha256)
            cache=FrozenFeatureCache(namespace())
            with self.assertRaises(ValueError):cache.get(b)
    def test_changed_retained_identity_content_is_error(self):
        raw=sample();key=build_frame_keys(raw,namespace(),[1])[0][0]
        cache=FrozenFeatureCache(namespace());cache.put(key,torch.ones(2,3))
        raw["obs"][1,0,0,0,0]^=1
        other=build_frame_keys(raw,namespace(),[1])[0][0]
        with self.assertRaises(ValueError):cache.get(other)
    def test_bad_or_guessed_sample_identity_rejected(self):
        for mutate in (lambda x:x.__setitem__("sample_id","some_counter_5"),
                       lambda x:x.__setitem__("episode_id",""),
                       lambda x:x.__setitem__("frame_start",99)):
            raw=sample();mutate(raw)
            with self.assertRaises(ValueError):build_frame_keys(raw,namespace(),[1])
    def test_explicit_actual_frame_start_supported(self):
        raw=sample();raw["frame_start"]=17;raw.pop("sample_id")
        key=build_frame_keys(raw,namespace(),[1])[0][0]
        self.assertEqual(key.frame_index,18)
    def test_canonical_seed_matches_actual_window_frame_and_lineage(self):
        raw=sample(start=6)
        window=build_frame_keys(raw,namespace(),[1],require_window=True)
        raw["sample_id"]=raw["episode_id"]+":seed6"
        raw["request_lineage"]={"episode_id":raw["episode_id"],"seed_start":6}
        self.assertEqual(window,build_frame_keys(raw,namespace(),[1]))
        with self.assertRaises(ValueError):build_frame_keys(raw,namespace(),[1],require_window=True)
        raw["request_lineage"]["seed_start"]=7
        with self.assertRaises(ValueError):build_frame_keys(raw,namespace(),[1])
    def test_seed_identity_strict_fullmatch_rejects_prefix_guesses(self):
        for identity in ("real_episode_4:seed01","real_episode_4:seed2extra","other_episode:seed2"):
            raw=sample();raw["sample_id"]=identity
            with self.assertRaises(ValueError):build_frame_keys(raw,namespace(),[1])
    def test_owned_clones_and_no_autograd_or_input_alias(self):
        key=build_frame_keys(sample(),namespace(),[1])[0][0]
        cache=FrozenFeatureCache(namespace())
        value=torch.ones(2,3)
        cache.put(key,value);value.zero_()
        hit=cache.get(key);hit.add_(3)
        self.assertTrue(torch.equal(cache.get(key),torch.ones(2,3)))
        ref=weakref.ref(value);del value;gc.collect();self.assertIsNone(ref())
        self.assertFalse(hit.requires_grad)
        self.assertEqual(hit.device.type,"cpu")
    def test_strict_byte_and_entry_bound_lru(self):
        keys=build_frame_keys(sample(),namespace(),[1,6,11])
        unit=FrozenFeatureCache._entry_bytes(keys[0][0],torch.ones(2,3))
        cache=FrozenFeatureCache(namespace(),max_bytes=unit*2,max_entries=2)
        cache.put(keys[0][0],torch.ones(2,3));cache.put(keys[1][0],torch.ones(2,3))
        cache.get(keys[0][0])
        cache.put(keys[2][0],torch.ones(2,3))
        self.assertLessEqual(cache.resident_bytes,unit*2)
        self.assertEqual(cache.feature_payload_bytes,48)
        self.assertIsNone(cache.get(keys[1][0]))
        self.assertIsNotNone(cache.get(keys[0][0]))
        self.assertEqual(cache.report()["entries"],2)
        self.assertEqual(cache.report()["evictions"],1)
    def test_oversized_zero_and_nonfinite_features_never_retained(self):
        key=build_frame_keys(sample(),namespace(),[1])[0][0]
        for cap in (0,4):
            cache=FrozenFeatureCache(namespace(),max_bytes=cap)
            self.assertFalse(cache.put(key,torch.ones(2,3)))
            self.assertEqual(cache.resident_bytes,0)
        cache=FrozenFeatureCache(namespace())
        for value in (torch.tensor([float("nan")]),torch.ones(2,requires_grad=True),torch.ones(2,dtype=torch.int64)):
            with self.assertRaises(ValueError):cache.put(key,value)
    def test_rebind_clears_prior_encoder_features(self):
        key=build_frame_keys(sample(),namespace(),[1])[0][0]
        cache=FrozenFeatureCache(namespace());cache.put(key,torch.ones(2))
        cache.rebind(namespace(encoder="d"*64))
        self.assertEqual(cache.resident_bytes,0);self.assertEqual(cache.report()["generation"],1)
        with self.assertRaises(ValueError):cache.get(key)
    def test_bfloat16_cache_preserves_values_and_bytes(self):
        key=build_frame_keys(sample(),namespace(),[1])[0][0]
        cache=FrozenFeatureCache(namespace());value=torch.linspace(-1,1,10).bfloat16()
        cache.put(key,value)
        self.assertTrue(torch.equal(value,cache.get(key)))
        self.assertEqual(cache.feature_payload_bytes,20)
        self.assertGreater(cache.resident_bytes,20)
    def test_metadata_fingerprint_finite_only(self):
        self.assertEqual(fingerprint({"a":1,"b":1.}),fingerprint({"b":1.,"a":1}))
        for value in (float("nan"),float("inf"),torch.ones(1),object()):
            with self.assertRaises(ValueError):fingerprint({"invalid":value})


class ActualTensorFeatureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.encoder=TinyFrozenEncoder()
        self.guard=FrozenEncoderGuard(self.encoder,namespace().encoder_sha256)
        self.plan=training_frame_selection(40,2,8,6,5)
        self.sources=[sample(episode="e"+str(i)) for i in range(4)]
        self.full=torch.from_numpy(np.stack([row["obs"] for row in self.sources])).float()/255.
        self.images=self.full[:,self.plan["selected_indices"]]
        self.keys=[build_frame_keys(row,namespace(),self.plan["selected_indices"]) for row in self.sources]
    def test_selected16_matches_full40_main_memory_and_gradient(self):
        full=encode_feature_batch(self.encoder,self.full,None,CAMERAS,guard=self.guard)
        selected=encode_feature_batch(self.encoder,self.images,None,CAMERAS,guard=self.guard)
        for key in full:
            torch.testing.assert_close(selected[key],full[key][:,self.plan["selected_indices"]],rtol=0,atol=0)
            torch.testing.assert_close(selected[key][:,self.plan["main_positions"]],full[key][:,30:],rtol=0,atol=0)
            torch.testing.assert_close(selected[key][:,self.plan["memory_positions"]],full[key][:,self.plan["memory_indices"]],rtol=0,atol=0)
        # Real autograd checks downstream trainable weights, with deterministic
        # frozen image features and identical selected state/action/labels.
        w1=torch.nn.Parameter(torch.linspace(-.2,.3,4));w2=torch.nn.Parameter(w1.detach().clone())
        baseline=sum((full[k][:,30:]@w1).square().mean() for k in full)
        optimized=sum((selected[k][:,self.plan["main_positions"]]@w2).square().mean() for k in selected)
        baseline.backward();optimized.backward()
        torch.testing.assert_close(optimized,baseline,rtol=0,atol=0)
        torch.testing.assert_close(w2.grad,w1.grad,rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in self.encoder.parameters()))
    def test_exact160_to64_image_exposures_per_view_and_cache_hit(self):
        cache=FrozenFeatureCache(namespace())
        first=encode_feature_batch(self.encoder,self.images,self.keys,CAMERAS,cache=cache,guard=self.guard)
        self.assertEqual(self.encoder.images,4*16*2)
        self.assertEqual(self.encoder.calls,8)
        self.assertLessEqual(max(self.encoder.batches),4*4)
        before=self.encoder.images
        second=encode_feature_batch(self.encoder,self.images,self.keys,CAMERAS,cache=cache,guard=self.guard)
        self.assertEqual(self.encoder.images,before)
        self.assertEqual(cache.report()["hits"],4*16*2)
        for key in first:torch.testing.assert_close(first[key],second[key],rtol=0,atol=0)
        self.assertLessEqual(cache.resident_bytes,cache.max_bytes)
    def test_partial_hits_and_eviction_reencode_same_values(self):
        size=8*12*4*4
        cache=FrozenFeatureCache(namespace(),max_bytes=size*20,max_entries=20)
        actual=encode_feature_batch(self.encoder,self.images,self.keys,CAMERAS,cache=cache,guard=self.guard)
        again=encode_feature_batch(self.encoder,self.images,self.keys,CAMERAS,cache=cache,guard=self.guard)
        for key in actual:torch.testing.assert_close(actual[key],again[key],rtol=0,atol=0)
        self.assertGreater(cache.evictions,0);self.assertLessEqual(cache.resident_bytes,size*20)
    def test_no_rng_consumption_with_mode_and_cache(self):
        cache=FrozenFeatureCache(namespace())
        before=torch.random.get_rng_state().clone()
        encode_feature_batch(self.encoder,self.images,self.keys,CAMERAS,cache=cache,guard=self.guard)
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        encode_feature_batch(self.encoder,self.images,self.keys,CAMERAS,cache=cache,guard=self.guard)
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
    def test_mutated_frozen_weights_and_trainable_encoder_rejected(self):
        cache=FrozenFeatureCache(namespace())
        encode_feature_batch(self.encoder,self.images,self.keys,CAMERAS,cache=cache,guard=self.guard)
        with torch.no_grad():self.encoder.conv.weight.add_(1)
        with self.assertRaises(ValueError):
            encode_feature_batch(self.encoder,self.images,self.keys,CAMERAS,cache=cache,guard=self.guard)
        self.encoder.conv.weight.requires_grad_(True)
        with self.assertRaises(ValueError):FrozenEncoderGuard(self.encoder,namespace().encoder_sha256)
    def test_dropout_batchnorm_and_missing_guard_rejected(self):
        for stochastic in (torch.nn.Dropout(.1),torch.nn.BatchNorm2d(3),torch.nn.RReLU()):
            with self.assertRaises(ValueError):FrozenEncoderGuard(stochastic,namespace().encoder_sha256)
        with self.assertRaises(ValueError):
            encode_feature_batch(self.encoder,self.images,self.keys,CAMERAS,cache=FrozenFeatureCache(namespace()))
    def test_same_value_buffer_publication_allowed_actual_change_rejected(self):
        encoder=TinyFrozenEncoder()
        encoder.register_buffer("normalization",torch.tensor([.5,.5,.5]))
        guard=FrozenEncoderGuard(encoder,namespace().encoder_sha256)
        self.assertEqual(guard.buffer_snapshot_bytes,12)
        with torch.no_grad():encoder.normalization.copy_(torch.tensor([.5,.5,.5]))
        guard.check(namespace())
        with torch.no_grad():encoder.normalization.add_(.1)
        with self.assertRaises(ValueError):guard.check(namespace())
    def test_large_buffers_fail_closed_and_registration_mutation_rejected(self):
        encoder=TinyFrozenEncoder();encoder.register_buffer("large",torch.zeros(10))
        guard=FrozenEncoderGuard(encoder,namespace().encoder_sha256,max_buffer_snapshot_bytes=4)
        self.assertEqual(guard.buffer_snapshot_bytes,0)
        with torch.no_grad():encoder.large.copy_(torch.zeros(10))
        with self.assertRaises(ValueError):guard.check()
        other=TinyFrozenEncoder();guard=FrozenEncoderGuard(other,namespace().encoder_sha256)
        other.conv=torch.nn.Conv2d(3,4,1).requires_grad_(False)
        with self.assertRaises(ValueError):guard.check()
    def test_camera_mismatch_rejected(self):
        keys=copy.deepcopy(self.keys);keys[0][0].reverse()
        with self.assertRaises(ValueError):
            encode_feature_batch(self.encoder,self.images,keys,CAMERAS,cache=FrozenFeatureCache(namespace()),guard=self.guard)
    def test_no_encoder_or_gpu_reference_stored_in_cache(self):
        cache=FrozenFeatureCache(namespace())
        encode_feature_batch(self.encoder,self.images,self.keys,CAMERAS,cache=cache,guard=self.guard)
        for key,value,size in cache._entries.values():
            self.assertIsInstance(key,FrameFeatureKey)
            self.assertEqual(value.device.type,"cpu")
            self.assertFalse(value.requires_grad)
            self.assertIsNone(value.grad_fn)
            self.assertGreaterEqual(size,value.numel()*value.element_size())
        self.assertFalse(any(isinstance(value,torch.nn.Module) for value in vars(cache).values()))
    def test_pinned_encoder_mode_not_stochastic_sample(self):
        # Static evidence for the production encoder, separate from toy CPU
        # equality. Both upstream encoder families use mode, never sample.
        upstream=pinned_encoder_source()
        if not upstream.is_file():
            self.skipTest("Pinned upstream source is not present in this isolated test bundle")
        text=upstream.read_text(encoding="utf-8")
        sd3=text[text.index("class SD3Encoder"):text.index("def build_image_encoder") if "def build_image_encoder" in text else len(text)]
        self.assertIn("self.vae.encode(x).latent_dist.mode()",sd3)
        self.assertNotIn("latent_dist.sample()",sd3)

    def test_actual_sd3_encoder_forward_class_with_cpu_vae_double(self):
        # Execute the real pinned SD3Encoder class, substituting only a tiny
        # frozen VAE; this does not load SD3 weights or claim CUDA equality.
        import typing
        from types import SimpleNamespace
        path=pinned_encoder_source()
        if not path.is_file():self.skipTest("Pinned encoder source not shipped in isolated bundle")
        tree=ast.parse(path.read_text(encoding="utf-8"))
        node=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=="SD3Encoder")
        class TinyVAE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("basis",torch.linspace(.01,.2,16).reshape(1,16,1,1))
                self.config=SimpleNamespace(latent_channels=16,scaling_factor=1.5305,shift_factor=.0609)
            def encode(self,x):
                x=torch.nn.functional.avg_pool2d(x.mean(dim=1,keepdim=True),8)
                return SimpleNamespace(latent_dist=Posterior(x*self.basis))
        class Loader:
            @staticmethod
            def from_pretrained(*args,**kwargs):return TinyVAE()
        def rearrange(x,pattern,**values):
            # Only the four exact reshapes in the pinned SD3 encode forward;
            # the host test environment need not install the full VAE stack.
            if pattern=="b t c h w -> (b t) c h w":return x.flatten(0,1)
            if pattern=="b d (h h1) (w w1) -> b (d h1 w1) h w":
                b,d,hh,ww=x.shape;h1,w1=values['h1'],values['w1']
                return x.reshape(b,d,hh//h1,h1,ww//w1,w1).permute(0,1,3,5,2,4).reshape(b,d*h1*w1,hh//h1,ww//w1)
            if pattern=="b d h w -> b (h w) d":return x.flatten(2).transpose(1,2)
            if pattern=="(b t) n d -> b t n d":return x.reshape(values['b'],-1,*x.shape[1:])
            raise AssertionError('Unexpected production encoder rearrange: '+pattern)
        env={"torch":torch,"nn":torch.nn,"Tuple":typing.Tuple,"AutoencoderKL":Loader,"rearrange":rearrange}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),"exec"),env)
        encoder=env["SD3Encoder"]("local_cpu_double",image_size=(32,48),spatial_size=2,device="cpu")
        guard=FrozenEncoderGuard(encoder,namespace().encoder_sha256)
        raw=torch.linspace(0,1,2*40*2*3*32*48).reshape(2,40,2,3,32,48)
        state=torch.random.get_rng_state().clone()
        all_features=encode_feature_batch(encoder,raw,None,CAMERAS,guard=guard)
        selected=encode_feature_batch(encoder,raw[:,self.plan["selected_indices"]],None,CAMERAS,guard=guard)
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
        for key in all_features:
            self.assertEqual(tuple(selected[key].shape),(2,16,6,64))
            torch.testing.assert_close(selected[key],all_features[key][:,self.plan["selected_indices"]],rtol=0,atol=0)


if __name__ == "__main__":
    unittest.main()
