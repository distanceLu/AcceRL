"""Fail-closed admission and strict tensor audits for the pinned WEAVER source.

This module has no torch import, performs no downloads and never installs packages.
Official DROID release metadata is deliberately insufficient for LIBERO admission.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys

PINNED_COMMIT = "3e41e90738fea9a7b142db6b2fd3d9ae72b7a82e"
SOURCE_SHA256 = {
    "weaver/__init__.py": "731acc5dfc310720456d0cf61df985f631bc6748dc5d2000a6bfed943b197b82",
    "weaver/wm/__init__.py": "b9f3eb3b6a4ef37390574ab39263e651b32784b6dc482867740115adaa4fedfc",
    "weaver/utils/__init__.py": "996d20e890d86f01db50ab38ea27c9c2ec5f92af755d79fd19668503dd2d02d7",
    "weaver/config.yaml": "2388b4bb6a907e48e651061ba5deecccb028645f578c196291f4a0e30ebf4b94",
    "weaver/wm/decoders.py": "8e8888ac71afcc3475176e0831ca192fe8972d2deb5f01f240ba2c6044c2e26d",
    "weaver/utils/config.py": "ab7fe3fa0e214bb4d9038218e8eb126abea7f7dd4abe4f7a45a3585287ee1eea",
    "weaver/wm/model.py": "0bbd5b131679629bc8e10ed003248749a508d20a8f4440733747dfbea57a10b5",
    "weaver/wm/encoders.py": "257a93940b35cc2fd83c563e0ae50d120949853fecdb43dcd09c7126c5dfd2d6",
    "weaver/utils/wm_eval.py": "fc7c41fb3606f144845842616d203952773c0d5444d3c50013520ce64ee6ea34",
    "weaver/utils/tools.py": "95d82bdcfa58f468e4cc61b17c31d767bf72e30b07a09ae0cf72cd679a37d42a",
    "weaver/wm/nets.py": "6ca87526e12d40cb8cff0fcf279186c4a6e87a5f3228b1cc1ce6ea7c97c5de8e",
}


class WeaverAdmissionError(ValueError):
    """The source, checkpoint or trajectory does not satisfy a declared contract."""


def verify_source(source_path):
    root = Path(source_path).resolve()
    failures = []
    for name, expected in SOURCE_SHA256.items():
        path = root / name
        if not path.is_file():
            failures.append(f"missing {name}")
        elif hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != expected:
            failures.append(f"changed {name}")
    if failures:
        raise WeaverAdmissionError("WEAVER source must match audited commit " + PINNED_COMMIT + ": " + "; ".join(failures))
    return root


def environment_report():
    required = {"torch": "2.7.0", "torchvision": "0.22.0", "diffusers": "0.35.2",
                "transformers": "4.57.3", "einops": "0.8.1", "PyYAML": "6.0.3",
                "numpy": "1.26.4", "imageio": "2.37.2"}
    versions, missing, incompatible = {}, [], []
    for package, minimum in required.items():
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            missing.append(package)
            continue
        versions[package] = version
        def parts(value):
            return tuple(int(x) for x in value.split("+")[0].split(".")[:3] if x.isdigit())
        if (package == "torch" and parts(version) != parts(minimum)) or parts(version) < parts(minimum):
            incompatible.append(f"{package}={version}; upstream requires {'==' if package == 'torch' else '>='}{minimum}")
    if sys.version_info < (3, 11):
        incompatible.append("Python >=3.11 required")
    return {"versions": versions, "missing": missing, "incompatible": incompatible,
            "compatible": not missing and not incompatible, "environment_modified": False}


def clean_state_dict(state):
    cleaned = {}
    for name, value in state.items():
        original = name
        while name.startswith(("module.", "_orig_mod.")):
            name = name.split(".", 1)[1]
        if name in cleaned:
            raise WeaverAdmissionError(f"Checkpoint prefix collision: {original} -> {name}")
        cleaned[name] = value
    return cleaned


def audit_state_dict(expected, actual):
    actual = clean_state_dict(actual)
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    shapes = [{"name": name, "expected": list(expected[name].shape), "actual": list(actual[name].shape)}
              for name in sorted(set(expected) & set(actual))
              if tuple(expected[name].shape) != tuple(actual[name].shape)]
    report = {"missing_keys": missing, "unexpected_keys": unexpected, "shape_mismatches": shapes,
              "strict": True, "tensor_count": len(actual)}
    if missing or unexpected or shapes:
        raise WeaverAdmissionError("Strict WEAVER checkpoint audit failed: " + json.dumps(report, sort_keys=True))
    return actual, report


def validate_normalization(stats, schema):
    stats = stats.get("norm_stats", stats)
    for key, dim_name in (("state", "state_dim"), ("actions", "action_dim")):
        item = stats.get(key, {})
        for field in ("mean", "std"):
            values = item.get(field, [])
            if len(values) != int(schema[dim_name]) or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
                raise WeaverAdmissionError(f"Normalization {key}.{field} must contain {schema[dim_name]} finite values")
            if field == "std" and any(v <= 0 for v in values):
                raise WeaverAdmissionError(f"Normalization {key}.std must be positive")
    return stats


def admit_metadata(config, metadata, upstream_config=None):
    schema = config.get("schema", {})
    declared = metadata.get("schema", metadata)
    required = ("domain", "action_schema_id", "state_schema_id", "encoder_id", "normalizer_id",
                "action_dim", "state_dim", "alignment", "image_keys", "image_size", "image_value_range")
    errors = []
    for key in required:
        if key not in schema or key not in declared:
            errors.append(f"schema and checkpoint must declare {key}")
        elif schema[key] != declared[key]:
            errors.append(f"{key}: request={schema[key]!r}, checkpoint={declared[key]!r}")
    if schema.get("alignment") != "action_into_frame":
        errors.append("WEAVER currently requires action_into_frame alignment; provide real aligned actions, never padded dimensions")
    if schema.get("image_value_range") not in ("zero_one", "minus_one_one"):
        errors.append("image_value_range must be zero_one or minus_one_one")
    if metadata.get("code_revision", metadata.get("source_commit")) != PINNED_COMMIT:
        errors.append("checkpoint code_revision/source_commit must equal the audited WEAVER commit")
    if upstream_config is not None:
        dataset = upstream_config.get("dataset", {})
        domain = str(dataset.get("name", "")).lower()
        if domain != str(schema.get("domain", "")).lower():
            errors.append(f"checkpoint config dataset.name={domain!r} does not match declared domain; official DROID weights cannot be relabeled LIBERO")
        for field, schema_key in (("n_actions", "action_dim"), ("n_states", "state_dim"),
                                  ("img_keys", "image_keys"), ("image_size", "image_size")):
            actual = dataset.get(field, 8 if field in ("n_actions", "n_states") else None)
            if actual != schema.get(schema_key):
                errors.append(f"checkpoint dataset.{field}={actual!r} differs from schema.{schema_key}")
        for key in ("n_history", "n_memory_frames", "t_memory"):
            if upstream_config.get(key) != schema.get(key):
                errors.append(f"checkpoint {key} differs from declared schema")
    native_score=config.get('reward_semantics')=='delta_native_score'
    if native_score:
        from weaver_native_reward import validate_specification
        try:
            validate_specification(config.get('native_reward_head',{}))
        except (ValueError,TypeError) as error:
            errors.append(str(error))
        if (config.get('reward_provider')!='world_head' or config.get('terminal_rule')!='horizon_only'
                or config.get('train_reward_head') is not True or config.get('world_trainable') is not True):
            errors.append('Native reward score requires explicitly trainable joint World/head with horizon-only continuation')
        if any(metadata.get(name,{}).get('calibrated') is not False
               for name in ('success_head','terminal_head')):
            errors.append('Native reward head must retain uncalibrated success/terminal provenance')
        report=metadata.get('load_report',{})
        if (report.get('strict') is not True or report.get('tensor_count')!=866
                or report.get('checkpoint_sha256')!=config.get('initial_checkpoint_sha256')
                or report.get('missing_keys')!=[] or report.get('unexpected_keys')!=[]
                or report.get('shape_mismatches')!=[]):
            errors.append('Joint native head must start from the exact strict866 evaluated initial bundle')
    elif config.get("reward_provider", "external_vlm") == "world_head":
        for head_name, semantics in (("success_head", "environment_success"), ("terminal_head", "environment_terminal")):
            head = metadata.get(head_name, {})
            if head.get("calibrated") is not True or head.get("label_semantics") != semantics or not head.get("calibration_id"):
                errors.append(f"{head_name} requires calibrated {semantics} evidence and calibration_id; native progress regression is not a probability")
        head = metadata.get("success_head", {})
        if head.get("source") != "rm" or head.get("output") not in ("probability", "logit"):
            errors.append("success_head must declare source='rm' and output='probability' or 'logit'")
        terminal = metadata.get("terminal_head", {})
        if terminal.get("source") not in ("same_success", "terminal_head"):
            errors.append("terminal_head must declare calibrated source='same_success' or 'terminal_head'")
        if terminal.get("rule") != config.get("terminal_rule", "argmax"):
            errors.append("terminal_head.rule must equal configured terminal_rule")
    if errors:
        raise WeaverAdmissionError("WEAVER checkpoint admission rejected: " + "; ".join(errors))
    return metadata


def choose_checkpoint(directory, filename=None):
    root = Path(directory)
    if filename is not None:
        target = (root / filename).resolve()
        if not target.is_relative_to(root.resolve()):
            raise WeaverAdmissionError("checkpoint_file must stay inside checkpoint_dir")
        if not target.is_file():
            raise FileNotFoundError(target)
        return target
    if (root / "checkpoint.pt").is_file():
        return root / "checkpoint.pt"
    candidates = [p for p in root.glob("checkpoint_*.pt") if p.stem.rsplit("_", 1)[-1].isdigit()]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint.pt/checkpoint_<step>.pt in {root}; metadata files are not weights")
    return max(candidates, key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
