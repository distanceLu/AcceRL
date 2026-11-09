"""Finite, resumable, paired LIBERO policy evaluation of immutable snapshots.

Only the runtime entry point imports torch, LIBERO or the policy implementation.
The queue launches this foreground process after all training actors have exited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time


PROTOCOL = "accerl_frozen_policy_paired_v1"
CHECKPOINT_FORMAT = "accerl_parallel_weights_v1"
APPROVED_GPUS = (
    "GPU-c43d6671-0979-b38d-9eae-4fe0a9d71c8c",
    "GPU-fe0edfe4-b193-1af7-6a98-cbff40214f74",
    "GPU-8c9bc264-7f09-989e-1ec7-cecd6cde782c",
    "GPU-07ae0bbc-8923-bd0c-199e-c726821e655d",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def check_deadline(deadline, reserve=0):
    if time.time() + reserve >= deadline:
        raise TimeoutError("Fixed policy evaluation deadline reached")


def file_sha(path, deadline=math.inf):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            check_deadline(deadline)
            value.update(block)
    return value.hexdigest()


def _nonnegative_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(label + " must be a nonnegative integer")
    return value


def validate_checkpoint_payload(payload):
    """Reject missing recipe or trained state before constructing a GPU model."""
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Expected accerl_parallel_weights_v1 checkpoint")
    manifest = payload.get("manifest", {})
    if manifest.get("format") != CHECKPOINT_FORMAT or manifest.get("weights_only") is not True:
        raise ValueError("Checkpoint embedded manifest is invalid")
    policy_step = _nonnegative_int(manifest.get("steps", {}).get("policy"), "Policy step")
    metadata = manifest.get("metadata", {})
    args = metadata.get("args")
    required = ("pretrained_checkpoint", "checkpoint2", "benchmark", "num_images_in_input",
                "use_proprio", "use_bf16")
    if not isinstance(args, dict) or any(key not in args for key in required):
        raise ValueError("Checkpoint lacks the complete saved Policy recipe")
    if not isinstance(args["pretrained_checkpoint"], str) or not args["pretrained_checkpoint"]:
        raise ValueError("Policy base checkpoint is missing")
    if not isinstance(args["benchmark"], str) or not args["benchmark"]:
        raise ValueError("Policy benchmark is missing")
    if args["num_images_in_input"] not in (1, 2) or isinstance(args["num_images_in_input"], bool):
        raise ValueError("Unsupported Policy camera count")
    if not isinstance(args["use_proprio"], bool) or not isinstance(args["use_bf16"], bool):
        raise ValueError("Policy dtype/proprio metadata is not explicit")
    if args["checkpoint2"] is not None and not isinstance(args["checkpoint2"], str):
        raise ValueError("Invalid Policy checkpoint2")
    if not isinstance(metadata.get("run_id"), str) or not metadata["run_id"]:
        raise ValueError("Checkpoint has no training run identity")
    saved = payload.get("models", {}).get("policy", {})
    state, names = saved.get("state_dict"), saved.get("trainable_names")
    if (not isinstance(state, dict) or not state or not isinstance(names, list) or not names
            or any(not isinstance(name, str) for name in names) or len(set(names)) != len(names)
            or not set(names) <= set(state) or not isinstance(saved.get("full_state"), bool)):
        raise ValueError("Checkpoint is missing trained Policy tensors/names")
    for family in ("lora_", "lm_head", "attn_pool", "value_head"):
        if not any(family in name for name in names):
            raise ValueError("Checkpoint lacks trained Policy component: " + family)
    if args["use_proprio"] and not any("proprio_projector" in name for name in names):
        raise ValueError("Checkpoint lacks the trained proprio projector")
    count = manifest.get("tensor_counts", {}).get("policy")
    if count != len(state):
        raise ValueError("Policy tensor count differs from embedded manifest")
    return {"training_run_id": metadata["run_id"], "policy_step": policy_step,
            "args": {key: args[key] for key in required}, "tensor_count": len(state),
            "trainable_names_sha256": digest(sorted(names)),
            "manifest_sha256": digest(manifest)}


def make_plan(checkpoints, *, initial_state_count=50, seed=20261026,
              modes=("greedy", "stochastic"), repeats=1):
    if initial_state_count != 50:
        raise ValueError("Formal evaluation requires all 50 initial states")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
        raise ValueError("Evaluation seed must be an explicit 31-bit integer")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("Evaluation repeats must be positive")
    modes = list(modes)
    if not modes or len(set(modes)) != len(modes) or set(modes) - {"greedy", "stochastic"}:
        raise ValueError("Modes must be unique greedy/stochastic entries")
    if (not isinstance(checkpoints, list) or len(checkpoints) != 2
            or [item.get("name") for item in checkpoints] != ["baseline", "final"]):
        raise ValueError("Paired evaluation requires explicit baseline and final snapshots")
    for item in checkpoints:
        if (len(item.get("sha256", "")) != 64 or not item.get("path")
                or not isinstance(item.get("metadata"), dict)):
            raise ValueError("Snapshot fingerprint/metadata is missing")
    # Both checkpoints receive exactly the same state, environment and action seeds.
    cases = []
    for repeat in range(repeats):
        order = list(range(initial_state_count))
        random.Random(seed + repeat).shuffle(order)
        for initial_id in order:
            pair_seed = int(digest([seed, repeat, initial_id, "environment"])[:8], 16) % (2**31)
            action_seed = int(digest([seed, repeat, initial_id, "actions"])[:8], 16) % (2**31)
            for mode in modes:
                cases.append({"pair_id": f"r{repeat}:s{initial_id}:{mode}", "repeat": repeat,
                              "initial_state_id": initial_id, "env_seed": pair_seed,
                              "action_seed": action_seed, "mode": mode,
                              "deterministic": mode == "greedy"})
    plan = {"protocol": PROTOCOL, "checkpoints": checkpoints, "seed": seed,
            "initial_state_count": initial_state_count, "task_id": 0,
            "modes": modes, "repeats": repeats, "cases": cases,
            "horizon_action_steps": 220, "reset_settling_steps": 10,
            "policy_updates_during_evaluation": False, "inference_batch_size": 1,
            "expected_episode_records": len(cases) * len(checkpoints)}
    plan["sha256"] = digest(plan)
    return plan


def row_key(row):
    return row["checkpoint"], row["pair_id"]


def validate_row(row, plan):
    checkpoints = {item["name"]: item for item in plan["checkpoints"]}
    cases = {item["pair_id"]: item for item in plan["cases"]}
    if (row.get("protocol") != PROTOCOL or row.get("plan_sha256") != plan["sha256"]
            or row.get("checkpoint") not in checkpoints or row.get("pair_id") not in cases):
        raise ValueError("Episode does not belong to this immutable paired plan")
    case, checkpoint = cases[row["pair_id"]], checkpoints[row["checkpoint"]]
    if row.get("checkpoint_sha256") != checkpoint["sha256"]:
        raise ValueError("Episode checkpoint fingerprint differs")
    for field in ("initial_state_id", "repeat", "env_seed", "action_seed", "mode", "deterministic"):
        if row.get(field) != case[field]:
            raise ValueError("Episode pairing differs: " + field)
    if (not isinstance(row.get("success"), bool) or isinstance(row.get("steps"), bool)
            or not isinstance(row.get("steps"), int) or not 1 <= row["steps"] <= 220
            or row.get("termination") not in ("success", "horizon", "terminated_unsuccessful")
            or not isinstance(row.get("initial_state_hash"), str)
            or len(row["initial_state_hash"]) != 64
            or row.get("policy_version") != checkpoint["metadata"]["policy_step"]):
        raise ValueError("Malformed completed episode result")
    if row["success"] != (row["termination"] == "success"):
        raise ValueError("Episode success/termination disagree")
    value = row.get("episode_return")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Episode return is invalid")
    return row


def read_rows(path, plan, *, repair_partial_tail=False):
    path = Path(path)
    if not path.exists():
        return []
    contents = path.read_bytes()
    lines = contents.splitlines(keepends=True)
    rows, seen, complete_bytes = [], set(), 0
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if repair_partial_tail and index == len(lines) - 1 and not line.endswith(b"\n"):
                with path.open("r+b") as handle:
                    handle.truncate(complete_bytes)
                break
            raise ValueError("Corrupt episode record; only an interrupted final line is recoverable")
        validate_row(row, plan)
        key = row_key(row)
        if key in seen:
            raise ValueError("Duplicate completed episode record")
        seen.add(key)
        rows.append(row)
        complete_bytes += len(line)
    # A complete JSON record without its final newline must not merge with the next append.
    if contents and rows and not contents.endswith(b"\n") and complete_bytes == len(contents):
        if repair_partial_tail:
            with path.open("ab") as handle:
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
    return rows


def append_row(path, row, plan, completed_keys):
    validate_row(row, plan)
    key = row_key(row)
    if key in completed_keys:
        raise ValueError("Refusing to append a duplicate episode")
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    completed_keys.add(key)


def summarize(plan, rows, *, errors=(), finished_ns=None):
    keys = set()
    for row in rows:
        validate_row(row, plan)
        if row_key(row) in keys:
            raise ValueError("Duplicate result in summary")
        keys.add(row_key(row))
    groups, paired = {}, {}
    for checkpoint in plan["checkpoints"]:
        for mode in plan["modes"]:
            selected = [row for row in rows if row["checkpoint"] == checkpoint["name"] and row["mode"] == mode]
            covered = sorted({row["initial_state_id"] for row in selected})
            n = len(selected)
            groups[checkpoint["name"] + ":" + mode] = {
                "episodes": n, "expected_episodes": 50 * plan["repeats"],
                "successes": sum(row["success"] for row in selected),
                "success_rate": sum(row["success"] for row in selected) / n if n else None,
                "mean_steps": sum(row["steps"] for row in selected) / n if n else None,
                "mean_return": sum(row["episode_return"] for row in selected) / n if n else None,
                "initial_state_ids": covered, "missing_initial_state_ids": sorted(set(range(50)) - set(covered)),
            }
    for mode in plan["modes"]:
        left = {r["pair_id"]: r for r in rows if r["checkpoint"] == "baseline" and r["mode"] == mode}
        right = {r["pair_id"]: r for r in rows if r["checkpoint"] == "final" and r["mode"] == mode}
        matching = sorted(set(left) & set(right))
        for key in matching:
            if left[key]["initial_state_hash"] != right[key]["initial_state_hash"]:
                raise ValueError("Paired episodes refer to different serialized initial states")
        changes = [int(right[key]["success"]) - int(left[key]["success"]) for key in matching]
        paired[mode] = {"paired_episodes": len(matching), "expected_pairs": 50 * plan["repeats"],
                        "final_minus_baseline_success_rate": sum(changes) / len(changes) if changes else None,
                        "final_only_successes": sum(value == 1 for value in changes),
                        "baseline_only_successes": sum(value == -1 for value in changes),
                        "both_successes": sum(left[key]["success"] and right[key]["success"] for key in matching),
                        "both_failures": sum(not left[key]["success"] and not right[key]["success"] for key in matching),
                        "unpaired_baseline_episodes": len(left) - len(matching),
                        "unpaired_final_episodes": len(right) - len(matching)}
    complete = len(keys) == plan["expected_episode_records"] and not errors
    return {"protocol": PROTOCOL, "plan_sha256": plan["sha256"], "complete": complete,
            "finished_ns": finished_ns or time.time_ns(), "completed_episode_records": len(keys),
            "expected_episode_records": plan["expected_episode_records"], "checkpoints": plan["checkpoints"],
            "groups": groups, "paired": paired, "errors": list(errors),
            "limitations": ["All 50 states belong to training task 0; this is not an unseen-task test.",
                            "One stochastic repeat per state is the default; uncertainty needs more repeats.",
                            "Identical seeds do not guarantee bitwise identical GPU kernels across hardware/software.",
                            "Partial results compare only matched completed pairs and report missing coverage."]}


def load_snapshot(path, deadline):
    import torch
    path = Path(path).resolve(strict=True)
    before = path.stat()
    sha = file_sha(path, deadline)
    # mmap retains an open immutable inode if the training publisher were replaced.
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    metadata = validate_checkpoint_payload(payload)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("Checkpoint changed while being admitted")
    return payload, {"path": str(path), "sha256": sha, "bytes": after.st_size, "metadata": metadata}


def restore_policy(payload, deadline):
    import torch
    from experiments.robot.libero.libero_utils import GenerateConfig
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK
    from rl.actor_critic_model_discrete import ActorCritic
    metadata = validate_checkpoint_payload(payload)
    args = metadata["args"]
    if not Path(args["pretrained_checkpoint"]).is_dir():
        raise FileNotFoundError("Saved base Policy checkpoint is unavailable")
    if args["checkpoint2"] and not Path(args["checkpoint2"]).is_file():
        raise FileNotFoundError("Saved Policy checkpoint2 is unavailable")
    dtype = torch.bfloat16 if args["use_bf16"] else torch.float32
    cfg = GenerateConfig(pretrained_checkpoint=args["pretrained_checkpoint"],
                         use_l1_regression=False, use_diffusion=False, use_film=False,
                         num_images_in_input=args["num_images_in_input"], use_proprio=args["use_proprio"],
                         load_in_8bit=False, load_in_4bit=False, center_crop=True,
                         num_open_loop_steps=NUM_ACTIONS_CHUNK,
                         unnorm_key=args["benchmark"] + "_no_noops", checkpoint2=args["checkpoint2"])
    # Match the training recipe before overwriting every saved trainable tensor.
    torch.manual_seed(0)
    model = ActorCritic(cfg, torch_dtype=dtype)
    model.cuda()
    state = payload["models"]["policy"]
    actual_trainables = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if actual_trainables != set(state["trainable_names"]):
        raise ValueError("Saved trainable parameter names differ from the constructed Policy")
    destinations = model.state_dict()
    for name, tensor in state["state_dict"].items():
        check_deadline(deadline, 5)
        if (name not in destinations or not torch.is_tensor(tensor)
                or tensor.shape != destinations[name].shape
                or tensor.dtype != destinations[name].dtype):
            raise ValueError("Policy checkpoint tensor signature differs: " + name)
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError("Policy checkpoint has nonfinite tensors: " + name)
    persistent_buffers = {name for name, _ in model.named_buffers() if name in destinations}
    if not persistent_buffers <= set(state["state_dict"]):
        raise ValueError("Policy snapshot omits persistent buffers")
    loaded = model.load_state_dict(state["state_dict"], strict=state["full_state"])
    if loaded.unexpected_keys or actual_trainables.intersection(loaded.missing_keys):
        raise ValueError("Policy checkpoint does not cover all trained components")
    model.eval()
    model.requires_grad_(False)
    if cfg.unnorm_key not in model.vla.norm_stats:
        raise ValueError("Saved base model lacks action normalization statistics")
    return model, cfg, dtype


def verify_single_gpu():
    import subprocess
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if "," in visible or not visible:
        raise ValueError("Evaluator requires exactly one explicitly admitted GPU4-7 device")
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True)
    mapping = dict(line.strip().split(", ", 1) for line in inventory.splitlines() if line.strip())
    selected = mapping.get(visible, visible)
    if selected not in APPROVED_GPUS:
        raise ValueError("Evaluator GPU is outside the approved GPU4-7 UUIDs")
    import torch
    if torch.cuda.device_count() != 1:
        raise ValueError("Evaluator sees more or fewer than one GPU")
    return selected


def run_episode(model, cfg, dtype, env, case, checkpoint, plan, deadline):
    import visualization as visual_preview
    import numpy as np
    import torch
    from initial_state_env import reset_with_initial_state
    from rl.utils import prepare_one_obs
    started_ns = time.time_ns()
    observation, initial_info = reset_with_initial_state(env, case["initial_state_id"], case["env_seed"])
    if env.max_episode_steps != 220 or env.num_steps_wait != 10:
        raise ValueError("Formal task-0 evaluation horizon/reset protocol differs")
    # Action sampling only uses torch. Do not overwrite the simulator's NumPy
    # state with the action seed after its explicitly seeded reset.
    torch.manual_seed(case["action_seed"])
    torch.cuda.manual_seed_all(case["action_seed"])
    steps, total_return, chunks = 0, 0.0, 0
    action_hash = hashlib.sha256()
    preview = visual_preview.begin('evaluation', 'frozen-'+checkpoint['name']+'-'+str(os.getpid()),
        str(case), task=env.task_description, deterministic=case['deterministic'],
        initial_state_id=case['initial_state_id'], initial_state_hash=initial_info['initial_state_hash'],
        checkpoint=checkpoint['name'], checkpoint_sha256=checkpoint['sha256'],
        policy_version=checkpoint['metadata']['policy_step'], evaluation_kind='frozen_checkpoint')
    preview.add(observation, 0)
    while True:
        check_deadline(deadline, 5)
        inputs = prepare_one_obs(cfg, model.processor, observation, env.task_description, dtype)
        with torch.inference_mode():
            batch = model.prepare_inputs_batch([inputs])
            logits, _ = model(batch)
            _, tokens, actions = model.post_process(logits, deterministic=[case["deterministic"]])
            actions_env = model.vla._unnormalize_actions(actions[0], cfg.unnorm_key).astype(np.float32)
            action_hash.update(tokens.detach().cpu().contiguous().numpy().tobytes())
        chunks += 1
        for action in actions_env:
            observation, reward, terminated, truncated, info = env.step(action)
            steps += 1
            total_return += float(reward)
            preview.add(observation, steps, force=bool(terminated or truncated), reward=float(reward))
            if terminated or truncated:
                success = bool(info.get("is_success", False))
                termination = "success" if success else "horizon" if truncated else "terminated_unsuccessful"
                preview.finish(success=success, action_steps=steps, episode_return=total_return,
                    termination=termination, terminated=bool(terminated), truncated=bool(truncated))
                return {"protocol": PROTOCOL, "plan_sha256": plan["sha256"],
                        "checkpoint": checkpoint["name"], "checkpoint_sha256": checkpoint["sha256"],
                        "policy_version": checkpoint["metadata"]["policy_step"], **case,
                        "initial_state_hash": initial_info["initial_state_hash"],
                        "reset_env_steps": initial_info["reset_env_steps"], "success": success,
                        "steps": steps, "episode_return": total_return, "termination": termination,
                        "action_chunks": chunks, "action_tokens_sha256": action_hash.hexdigest(),
                        "started_ns": started_ns, "completed_ns": time.time_ns()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=7200)
    parser.add_argument("--deadline-unix", type=float)
    parser.add_argument("--initial-state-count", type=int, default=50)
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--final-checkpoint", type=Path)
    parser.add_argument("--modes", default="greedy,stochastic")
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if not 0 < args.max_seconds <= 14400:
        raise ValueError("Evaluation budget must be positive and at most four hours")
    deadline = min(time.time() + args.max_seconds, args.deadline_unix or math.inf)
    check_deadline(deadline, 10)
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    repository = Path(os.environ.get("ACCERL_REPO_ROOT", "/mnt/data/lcx1/dong/AcceRL-scheduler"))
    sys.path.insert(0, str(repository))
    args.output_root.mkdir(parents=True, exist_ok=True)
    payloads, checkpoints = {}, []
    paths = {"baseline": args.baseline_checkpoint or args.training_root / "training/checkpoints_initial/latest_weights.pt",
             "final": args.final_checkpoint or args.training_root / "training/checkpoints/latest_weights.pt"}
    for name, path in paths.items():
        payloads[name], item = load_snapshot(path, deadline)
        checkpoints.append(dict(item, name=name))
    if checkpoints[0]["metadata"]["args"] != checkpoints[1]["metadata"]["args"]:
        raise ValueError("Baseline/final Policy recipes differ; do not mix model or environment definitions")
    if checkpoints[0]["metadata"]["args"]["benchmark"] != "libero_spatial":
        raise ValueError("This formal paired plan is fixed to libero_spatial task0")
    plan = make_plan(checkpoints, initial_state_count=args.initial_state_count, seed=args.seed,
                     modes=args.modes.split(","), repeats=args.repeats)
    plan_path = args.output_root / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Existing evaluation output belongs to a different frozen plan")
    else:
        atomic_json(plan_path, plan)
    if args.validate_only:
        print(json.dumps({"validated": True, "plan_sha256": plan["sha256"],
                          "expected_episode_records": plan["expected_episode_records"],
                          "scope": "CPU checkpoint metadata/plan only; GPU restoration not yet tested"}))
        return 0
    rows_path = args.output_root / "episodes.jsonl"
    rows = read_rows(rows_path, plan, repair_partial_tail=True)
    completed_keys = {row_key(row) for row in rows}
    errors = []
    selected_gpu = verify_single_gpu()
    started_path = args.output_root / "started.json"
    if not started_path.exists():
        atomic_json(started_path, {"plan_sha256": plan["sha256"], "started_ns": time.time_ns(),
                                  "deadline_unix": deadline, "gpu_uuid": selected_gpu})
    else:
        original_deadline = json.loads(started_path.read_text())["deadline_unix"]
        deadline = min(deadline, original_deadline)
        check_deadline(deadline, 10)
    def interrupted(number, frame):
        raise InterruptedError("Evaluation stopped by supervisor signal " + str(number))
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, interrupted)
    env = model = None
    try:
        from rl.libero_env import LiberoEnvWrapper
        from initial_state_env import initial_state_pool_metadata
        import torch
        env = LiberoEnvWrapper(benchmark_name="libero_spatial", task_id=0, image_size=224, render_mode="rgb_array")
        pool = initial_state_pool_metadata(env)
        if pool["count"] != 50:
            raise ValueError("Live initial-state pool differs from the formal 50-state plan")
        for payload in payloads.values():
            saved_pool = payload['manifest']['metadata'].get('initial_state_queue', {}).get('pool')
            if saved_pool is not None and any(saved_pool.get(key) != pool.get(key)
                                               for key in ('count', 'shape', 'sha256')):
                raise ValueError('Formal evaluation initial-state asset differs from the training checkpoint')
        pool_path = args.output_root / "initial_state_pool.json"
        if pool_path.exists() and json.loads(pool_path.read_text()) != pool:
            raise ValueError("Initial-state asset changed across evaluation resume")
        atomic_json(pool_path, pool)
        for checkpoint in checkpoints:
            pending = [case for case in plan["cases"] if (checkpoint["name"], case["pair_id"]) not in completed_keys]
            if not pending:
                continue
            check_deadline(deadline, 30)
            model, cfg, dtype = restore_policy(payloads[checkpoint["name"]], deadline)
            for case in pending:
                row = run_episode(model, cfg, dtype, env, case, checkpoint, plan, deadline)
                append_row(rows_path, row, plan, completed_keys)
                rows.append(row)
                atomic_json(args.output_root / "progress.json", {
                    "plan_sha256": plan["sha256"], "completed_episode_records": len(rows),
                    "expected_episode_records": plan["expected_episode_records"],
                    "last_checkpoint": checkpoint["name"], "last_pair_id": case["pair_id"],
                    "updated_ns": time.time_ns(), "deadline_unix": deadline})
            del model
            model = None
            torch.cuda.empty_cache()
    except BaseException as exc:
        import traceback
        errors.append({"type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                errors.append({"type": type(exc).__name__, "error": "Environment close failed: " + str(exc)})
        summary = summarize(plan, rows, errors=errors)
        atomic_json(args.output_root / "summary.json", summary)
        atomic_json(args.output_root / "completed.json", {
                    **{key: summary[key] for key in ("protocol", "plan_sha256", "complete", "finished_ns",
                       "completed_episode_records", "expected_episode_records", "errors")},
                    "initial_state_count": plan["initial_state_count"], "modes": plan["modes"],
                    "checkpoint_sha256": {item["name"]: item["sha256"] for item in checkpoints}})
    print(json.dumps({"complete": summary["complete"], "episodes": len(rows), "errors": errors}, ensure_ascii=False))
    return 0 if summary["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
