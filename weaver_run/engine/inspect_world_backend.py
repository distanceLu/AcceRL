"""CPU-only configuration/asset admission report. Never creates model actors."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from world_backends import WorldBackendConfig, build_service_plan
from world_backends.libero import validate_online_schema


def interpreter_entry(path):
    """Identify a venv entry path without collapsing its symlink to base Python.

    Virtual environments commonly share one binary but discover distinct
    pyvenv.cfg/site-packages from the path through which Python is invoked.
    """
    return os.path.normcase(os.path.abspath(os.path.expanduser(os.fspath(path))))


def inspect(config_path, check_local=False):
    config = WorldBackendConfig.from_json(config_path)
    plan = build_service_plan(config)
    report = {"configuration": str(Path(config_path).resolve()), "service_plan": plan.to_dict(),
              "configuration_admitted": False, "assets_checked": False,
              "real_model_loaded": False, "gpu_validated": False, "errors": []}
    try:
        config.validate_admission()
        validate_online_schema(config)
        plan.validate_runtime()
        report["configuration_admitted"] = True
    except (ValueError, NotImplementedError) as error:
        report["errors"].append(str(error))
    if check_local and config.backend == "weaver":
        from weaver_checkpoint import verify_source, choose_checkpoint, environment_report
        options = config.backend_options
        report["controller_environment"] = environment_report()
        report["environment"] = report["controller_environment"]
        try:
            inference_python = options.get('inference_python')
            if inference_python:
                if not Path(inference_python).is_absolute() or not Path(inference_python).is_file():
                    raise ValueError('inference_python must be an existing absolute interpreter path')
                source = str(Path(__file__).resolve().parent)
                code = ('import sys,json;sys.path.insert(0,' + repr(source) + ');'
                        'from weaver_checkpoint import environment_report;'
                        'print(json.dumps(environment_report()))')
                env = {key: value for key, value in os.environ.items()
                       if not key.startswith(('NSYS', 'NVTX', 'CUDA_INJECTION')) and key != 'LD_PRELOAD'}
                env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
                process = subprocess.run([inference_python, '-c', code], env=env,
                                         capture_output=True, text=True, check=True, timeout=30)
                report['environment'] = json.loads(process.stdout)
                report['inference_python'] = inference_python
                report['controller_python_entry'] = interpreter_entry(sys.executable)
                report['inference_python_entry'] = interpreter_entry(inference_python)
                report['separate_inference_interpreter'] = (report['inference_python_entry']
                                                            != report['controller_python_entry'])
            report["source_path"] = str(verify_source(options["source_path"]))
            report["checkpoint_file"] = str(choose_checkpoint(Path(options["checkpoint_dir"]), options.get("checkpoint_file")))
            for key in ("image", "task"):
                location = options.get("encoder_paths", {}).get(key)
                if not location or not Path(location).is_dir():
                    raise ValueError("Missing local pretrained " + key + " encoder directory")
            if not config.world_trainable:
                from world_backend_audit import frozen_world_asset_report
                report['frozen_world'] = frozen_world_asset_report(config)
            elif options.get('transport') == 'isolated_online_ddp':
                training_python = options.get('training_python')
                if not training_python or not Path(training_python).is_absolute() or not Path(training_python).is_file():
                    raise ValueError('training_python must be an existing absolute interpreter path')
                source = str(Path(__file__).resolve().parent)
                code = ('import sys,json;sys.path.insert(0,' + repr(source) + ');'
                    'from weaver_checkpoint import environment_report;import torch;'
                    'r=environment_report();r.update(python=sys.executable,torch=torch.__version__,'
                    'sys_prefix=sys.prefix,distributed_available=torch.distributed.is_available(),'
                    'nccl_available=torch.distributed.is_nccl_available(),'
                    'cuda_initialized=torch.cuda.is_initialized(),gpu_access_performed=False);'
                    'r["compatible"]=bool(r["compatible"] and r["distributed_available"] and '
                    'r["nccl_available"] and not r["cuda_initialized"]);print(json.dumps(r))')
                env = {key: value for key, value in os.environ.items()
                       if not key.startswith(('NSYS', 'NVTX', 'CUDA_INJECTION')) and key != 'LD_PRELOAD'}
                env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
                process = subprocess.run([training_python, '-c', code], env=env,
                    capture_output=True, text=True, check=True, timeout=45)
                report['online_training_environment'] = json.loads(process.stdout)
                if not report['online_training_environment'].get('compatible'):
                    raise ValueError('Native WEAVER training interpreter lacks its pinned Torch/NCCL CPU admission')
            report["assets_checked"] = True
        except (ValueError, KeyError, OSError, RuntimeError, subprocess.SubprocessError) as error:
            report["errors"].append(str(error))
        if not report["environment"]["compatible"]:
            report["errors"].append("WEAVER dependencies do not match the pinned source environment")
    report["status"] = "blocked" if report["errors"] else "cpu_configuration_checked_only"
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--check-local", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inspect(args.config, args.check_local)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    raise SystemExit(2 if result["errors"] else 0)
