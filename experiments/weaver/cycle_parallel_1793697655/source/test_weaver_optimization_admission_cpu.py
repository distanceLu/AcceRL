from __future__ import annotations
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from test_weaver_feature_cache_cpu import SCHEMA,sample
from weaver_feature_cache import fingerprint
from weaver_optimization_admission import admit_config,admit_fixture,file_sha256,main


def config():
    return json.loads((Path(__file__).parent/"examples/weaver_integrated_internal.json").read_text(encoding="utf-8"))


def fixture():
    return {"protocol":"weaver_optimization_fixture_v1","source_proof":{
        "label_source":"libero_env_step","no_imagined_frames":True,"observations_source":"episode_frame_store",
        "observed_ns":1,"run_id":"cpu_structural_fixture_only","schema_sha256":fingerprint(SCHEMA)},
        "windows":[sample()]}


def tensor_wire(value):
    if isinstance(value,np.ndarray):return torch.from_numpy(value.copy())
    if isinstance(value,dict):return {key:tensor_wire(item) for key,item in value.items()}
    if isinstance(value,list):return [tensor_wire(item) for item in value]
    return value


class CPUAdmissionTests(unittest.TestCase):
    def test_actual_integrated_config_without_model_load(self):
        result=admit_config(config())
        self.assertEqual(result["source_optimizer_steps"],3200)
        self.assertEqual(len(result["selection"]["selected_indices"]),16)
        self.assertEqual(result["max_gpu_budget_seconds"],1200)
    def test_provenance_and_canonical_sample_identity_required(self):
        item=fixture()
        self.assertEqual(admit_fixture(item,SCHEMA)["window_count"],1)
        item["source_proof"]["no_imagined_frames"]=False
        with self.assertRaises(ValueError):admit_fixture(item,SCHEMA)
        item=fixture();item["windows"][0]["sample_id"]="fake-counter"
        with self.assertRaises(ValueError):admit_fixture(item,SCHEMA)
    def test_repeated_window_is_compute_not_independent_quality(self):
        item=fixture();item["windows"]*=4
        with self.assertRaises(ValueError):admit_fixture(item,SCHEMA)
        item["fixture_policy"]={"allow_repeated_windows":True,"purpose":"compute_equivalence_only"}
        result=admit_fixture(item,SCHEMA)
        self.assertEqual(result["unique_sample_count"],1)
        self.assertEqual(result["repeated_window_count"],3)
        self.assertFalse(result["quality_evaluation"])
    def test_fixture_needs_real_success_and_no_padding(self):
        item=fixture();item["windows"][0]["success"]=np.zeros(40,dtype=np.float32)
        with self.assertRaises(ValueError):admit_fixture(item,SCHEMA)
        item=fixture();item["windows"][0]["mask"][20]=False
        with self.assertRaises(ValueError):admit_fixture(item,SCHEMA)
    def test_main_default_cpu_has_no_execution_flag(self):
        with mock.patch("torch.cuda.init",side_effect=AssertionError("CUDA forbidden")),contextlib.redirect_stdout(io.StringIO()):
            result=main(["--config",str(Path(__file__).parent/"examples/weaver_integrated_internal.json")])
        self.assertTrue(result["cpu_admission_passed"])
        self.assertFalse(result["gpu_execution_available"])
        self.assertIsNone(result["comparison_passed"])
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            main(["--config","unused.json","--execute"])
    def test_real_torch_weights_only_fixture_loading_and_no_cuda(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);data=fixture()
            fixture_path=root/"actual_wire_fixture.pt"
            torch.save(tensor_wire(data),fixture_path)
            cfg=config();cfg["schema"]=dict(SCHEMA,n_history=2,n_memory_frames=6,t_memory=5)
            data["source_proof"]["schema_sha256"]=fingerprint(cfg["schema"])
            for row in data["windows"]:row["schema"]=cfg["schema"]
            torch.save(tensor_wire(data),fixture_path)
            cfgpath=root/"config.json";cfgpath.write_text(json.dumps(cfg),encoding="utf-8")
            output=root/"admission.json"
            with mock.patch("torch.cuda.init",side_effect=AssertionError("CUDA forbidden")),contextlib.redirect_stdout(io.StringIO()):
                result=main(["--config",str(cfgpath),"--fixture",str(fixture_path),
                    "--fixture-sha256",file_sha256(fixture_path),"--source-run-id",data["source_proof"]["run_id"],
                    "--output",str(output)])
            self.assertTrue(result["cpu_admission_passed"])
            self.assertEqual(result["fixture"]["window_count"],1)
            self.assertFalse(result["model_weights_loaded"])
            self.assertTrue(output.is_file())


if __name__=="__main__":unittest.main()
