"""Behavioral comparison with the immutable pre-JIT parent, not a toy rewrite."""
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import unittest
import numpy as np

from fresh_config import FreshConfig
from fresh_policy_replay import SegmentReplay
from test_fresh_policy_replay_cpu import segment

ROOT=Path(__file__).resolve().parent
import os
PARENT=Path(os.environ.get('ACCERL_TEST_WORKSPACE',str(ROOT.parent)))/'output/weaver_execution_scheduler_20261102/source_v1/weaver_integrated_v1'


class ParentTests(unittest.TestCase):
    def test_all_parent_files_remain_frozen(self):
        manifest=json.loads((ROOT/'parent_source_manifest.json').read_text('utf8'))
        self.assertEqual(len(manifest['files']),228)
        for name,digest in manifest['files'].items():
            with self.subTest(file=name):self.assertEqual(hashlib.sha256((PARENT/name).read_bytes()).hexdigest(),digest)

    def test_jit_off_keeps_parent_sampling_targets_masks_age_and_use(self):
        spec=importlib.util.spec_from_file_location('jit_frozen_parent_replay',PARENT/'fresh_policy_replay.py')
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        config=FreshConfig(refresh_gae=True,replay_age_limits=True,real_policy_replay=True)
        current=SegmentReplay(80,config,rng=random.Random(71))
        parent=module.SegmentReplay(80,config,rng=random.Random(71))
        for name,source in [('i','imagined'),('r','real')]:
            item=segment(name,source,count=16)
            current.add(item);parent.add(module.PolicySegment(**item.__dict__))
        for version in range(5,11):
            current.set_versions(version,1);parent.set_versions(version,1)
            ta,a=current.reserve(8);tb,b=parent.reserve(8)
            self.assertEqual(a,b)
            refreshed={name:np.arange(seg.count+1)*.125 for name,seg in current.segments.items()}
            ca,cr=current.materialize(a,refreshed);pa,pr=parent.materialize(b,refreshed)
            for x,y in zip(ca[1:],pa[1:]):np.testing.assert_array_equal(x,y)
            self.assertEqual(cr,pr)
            for x,y in zip(ca[0],pa[0]):
                np.testing.assert_array_equal(x['_fresh_action_mask'],y['_fresh_action_mask'])
                self.assertEqual(x['value'],y['value'])
            current.commit(ta,a);parent.commit(tb,b)
            self.assertEqual(current.samples,parent.samples)


if __name__=='__main__':unittest.main()
