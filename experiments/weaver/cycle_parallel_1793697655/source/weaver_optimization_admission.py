"""CPU-only admission for a future explicit WEAVER optimization comparison.

This is not a GPU benchmark and exposes no GPU execution flag. It never loads
model weights, creates a claim, initializes CUDA, or launches a child process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from weaver_feature_cache import (FeatureNamespace, build_frame_keys, fingerprint,
    training_frame_selection, validate_cpu_window)


MAX_FIXTURE_BYTES = 128 * 1024 * 1024
GPU4_UUID = "GPU-c43d6671-0979-b38d-9eae-4fe0a9d71c8c"


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def admit_config(config):
    if (config.get("backend")!="weaver" or config.get("world_trainable") is not True or
            config.get("reward_provider")!="world_head" or
            config.get("reward_semantics")!="delta_native_score"):
        raise ValueError("Comparison requires the actual integrated online WEAVER configuration")
    options=config["backend_options"]
    source=config["checkpoint_metadata"]["training_provenance"]
    if source.get("optimizer_steps")!=3200:
        raise ValueError("Comparison baseline must be the admitted native3200 bundle")
    if options.get("master_dtype")!="float32" or options.get("autocast_dtype")!="bfloat16":
        raise ValueError("Comparison must preserve actual FP32/BF16 execution precision")
    schema=config["schema"]
    if (schema.get("n_history")!=2 or schema.get("n_memory_frames")!=6 or
            schema.get("t_memory")!=5 or options.get("horizon")!=8):
        raise ValueError("Comparison geometry differs from native40/32+8")
    selection=training_frame_selection(40,2,8,6,5)
    return {"source_optimizer_steps":3200,"geometry":{"window":40,"context":32,"horizon":8},
            "selection":selection,"single_gpu_uuid":GPU4_UUID,
            "max_gpu_budget_seconds":1200,"precision":{"parameter":"float32","autocast":"bfloat16"}}


def admit_fixture(fixture,schema,*,expected_source_run_id=None):
    import numpy as np
    if not isinstance(fixture,dict) or fixture.get("protocol")!="weaver_optimization_fixture_v1":
        raise ValueError("Fixture protocol must explicitly identify a real-window comparison input")
    proof=fixture.get("source_proof",{})
    if (proof.get("label_source")!="libero_env_step" or proof.get("no_imagined_frames") is not True or
            proof.get("observations_source")!="episode_frame_store" or
            type(proof.get("observed_ns")) is not int or proof["observed_ns"]<=0 or
            type(proof.get("run_id")) is not str or not proof["run_id"]):
        raise ValueError("Fixture needs explicit actual simulator/frame-store provenance")
    if expected_source_run_id is not None and proof["run_id"]!=expected_source_run_id:
        raise ValueError("Fixture was collected from a different training run")
    if proof.get("schema_sha256")!=fingerprint(schema):
        raise ValueError("Fixture source schema differs from the admitted model")
    windows=fixture.get("windows")
    if not isinstance(windows,list) or not 1<=len(windows)<=4:
        raise ValueError("Finite comparison accepts one to four actual40-frame windows")
    samples=[];positive=negative=0
    for row in windows:
        checked=validate_cpu_window(row,schema,40,True,require_real_labels=True)
        if checked["frame_count"]!=40 or checked["observations"].dtype!=np.uint8:
            raise ValueError("Fixture images must be exact40-frame canonical uint8")
        labels=np.asarray(row.get("success"))
        if hasattr(row.get("success"),"numpy"):
            labels=row["success"].detach().numpy()
        if labels.shape!=(40,) or labels.dtype!=np.bool_:
            raise ValueError("Fixture labels must be actual boolean per-frame simulator success")
        # This checks the true episode/start/camera identity as well as bytes,
        # without claiming independently verified external provenance.
        ns=FeatureNamespace(proof["run_id"],fingerprint({"fixture":"admission_encoder"}),
                            fingerprint({"schema":schema}))
        build_frame_keys(row,ns,list(range(40)),require_window=True)
        samples.append(row["sample_id"])
        positive+=int(labels[30:40].sum());negative+=10-int(labels[30:40].sum())
    unique=len(set(samples))
    policy=fixture.get("fixture_policy",{})
    if unique<len(samples) and (policy.get("allow_repeated_windows") is not True or
            policy.get("purpose")!="compute_equivalence_only"):
        raise ValueError("Repeated windows must be explicit compute-equivalence inputs, never independent quality samples")
    return {"source_run_id":proof["run_id"],"window_count":len(windows),"unique_sample_count":unique,
            "repeated_window_count":len(windows)-unique,"sample_ids":samples,
            "selected_real_labels":{"positive":positive,"negative":negative},
            "provenance_scope":"fixture claims checked structurally; external source receipt still requires operator review",
            "quality_evaluation":False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--fixture",type=Path)
    parser.add_argument("--fixture-sha256")
    parser.add_argument("--source-run-id")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args(argv)
    if args.config.stat().st_size>2*1024*1024:
        raise ValueError("Configuration document is unexpectedly large")
    report={"protocol":"weaver_optimization_cpu_admission_v1","observed_ns":time.time_ns(),
            "cpu_admission_passed":False,"gpu_comparisons_run":False,"gpu_execution_available":False,
            "model_weights_loaded":False,"cuda_initialized_by_admission":False,
            "config_sha256":file_sha256(args.config)}
    config=json.loads(args.config.read_text(encoding="utf-8"))
    report["plan"]=admit_config(config)
    if args.fixture is not None:
        if args.fixture.stat().st_size>MAX_FIXTURE_BYTES:
            raise ValueError("Fixture exceeds the bounded CPU-admission input budget")
        if not args.fixture_sha256 or file_sha256(args.fixture)!=args.fixture_sha256:
            raise ValueError("Actual real-window fixture must match the caller's explicit SHA256")
        import torch
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU-admission caller already initialized CUDA")
        fixture=torch.load(args.fixture,map_location="cpu",weights_only=True,mmap=True)
        report["fixture"]=admit_fixture(fixture,config["schema"],expected_source_run_id=args.source_run_id)
        report["fixture_sha256"]=args.fixture_sha256
        report["torch_version"]=torch.__version__
        if torch.cuda.is_initialized():raise RuntimeError("CPU fixture admission unexpectedly initialized CUDA")
    report["cpu_admission_passed"]=True
    report["comparison_passed"]=None
    report["limitations"]=["No native model/GPU numerical or performance comparison was executed",
        "No new GPU claim or deadline was created", "Model/input provenance still needs the explicit execution gate"]
    text=json.dumps(report,indent=2,sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open("x",encoding="utf-8") as stream:stream.write(text+"\n")
    print(text)
    return report


if __name__=="__main__":main()
