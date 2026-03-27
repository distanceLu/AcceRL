#!/usr/bin/env python3
import argparse
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterator, List, Optional, Tuple

from tensorboard.compat.proto import event_pb2
from tensorboard.compat.proto import tensor_pb2
from tensorboard.compat.proto import types_pb2


RUN_NAME_RE = re.compile(
    r"simple_seed(?P<seed>\d+)_(?P<algo>ppo|sapo|gipo)"
    r"(?:_sigma(?P<sigma>\d+p\d+))?_",
    re.IGNORECASE,
)


@dataclass
class RunInfo:
    run_dir: Path
    seed: Optional[int]
    algo: str
    sigma: Optional[float]
    label: str
    metric_step: Optional[int]
    metric_value: Optional[float]
    max_scalar_step: int
    event_files: List[Path]


@dataclass
class FilteredRun:
    run_dir: Path
    label: str
    seed: Optional[int]
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "统计 MetaWorldSimple 实验中不同算法(含 gipo 不同 sigma)在多 seed 下的 "
            "最终分数 Mean ± Std，并过滤未达到指定迭代的数据。"
        )
    )
    parser.add_argument(
        "--task-root",
        type=Path,
        default=Path("runs/MetaWorldSimple/assembly-v3"),
        help="任务根目录，例如 runs/MetaWorldSimple/assembly-v3",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="Rollout/ReturnMean",
        help="用于统计的 TensorBoard 标量 tag，默认 Rollout/ReturnMean，可选 Eval/return_mean",
    )
    parser.add_argument(
        "--min-iter",
        type=int,
        default=90000,
        help="只保留最大迭代步达到该阈值的 run，默认 90000",
    )
    parser.add_argument(
        "--expected-seeds",
        type=int,
        default=5,
        help="期望的 seed 数量，用于报告完整性（不作为硬过滤条件）",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="",
        help=(
            "结果文件名（仅文件名，不含目录）。为空时自动命名并保存到 --task-root 目录下，"
            "如 summary_Rollout_ReturnMean_min90000.txt"
        ),
    )
    return parser.parse_args()


def iter_tfrecord_records(path: Path) -> Iterator[bytes]:
    with path.open("rb") as f:
        while True:
            header = f.read(8)
            if not header:
                break
            if len(header) < 8:
                break
            (length,) = struct.unpack("<Q", header)
            _ = f.read(4)  # length crc
            payload = f.read(length)
            _ = f.read(4)  # data crc
            if len(payload) != length:
                break
            yield payload


def tensor_scalar_to_float(tensor: tensor_pb2.TensorProto) -> Optional[float]:
    if tensor.dtype == types_pb2.DT_FLOAT:
        if tensor.float_val:
            return float(tensor.float_val[0])
        if tensor.tensor_content:
            return float(struct.unpack("<f", tensor.tensor_content[:4])[0])
    if tensor.dtype == types_pb2.DT_DOUBLE:
        if tensor.double_val:
            return float(tensor.double_val[0])
        if tensor.tensor_content:
            return float(struct.unpack("<d", tensor.tensor_content[:8])[0])
    if tensor.dtype in (types_pb2.DT_INT32, types_pb2.DT_INT16, types_pb2.DT_INT8):
        if tensor.int_val:
            return float(tensor.int_val[0])
    if tensor.dtype in (types_pb2.DT_INT64,):
        if tensor.int64_val:
            return float(tensor.int64_val[0])
    return None


def extract_run_identity(run_dir: Path) -> Tuple[str, Optional[int], Optional[float], str]:
    args_path = run_dir / "args.json"
    if args_path.exists():
        try:
            cfg = json.loads(args_path.read_text())
            clip_mode = str(cfg.get("clip_mode", "")).lower()
            seed = int(cfg["seed"]) if "seed" in cfg else None
            sigma = float(cfg["sigma"]) if "sigma" in cfg and cfg.get("sigma") is not None else None
            if clip_mode in {"ppo", "sapo"}:
                return clip_mode, seed, None, clip_mode
            if clip_mode == "gipo":
                if sigma is None:
                    return "gipo", seed, None, "gipo_sigma_unknown"
                return "gipo", seed, sigma, f"gipo_sigma{sigma:.1f}"
        except Exception:
            pass

    m = RUN_NAME_RE.search(run_dir.name)
    if m:
        algo = m.group("algo").lower()
        seed = int(m.group("seed"))
        sigma_token = m.group("sigma")
        if algo != "gipo":
            return algo, seed, None, algo
        sigma = None
        if sigma_token:
            sigma = float(sigma_token.replace("p", "."))
        return "gipo", seed, sigma, f"gipo_sigma{sigma:.1f}" if sigma is not None else "gipo_sigma_unknown"

    return "unknown", None, None, "unknown"


def parse_run_metric(run_dir: Path, metric_tag: str) -> Tuple[Optional[int], Optional[float], int, List[Path]]:
    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    best_metric_step: Optional[int] = None
    best_metric_value: Optional[float] = None
    max_scalar_step = -1

    for event_file in event_files:
        for payload in iter_tfrecord_records(event_file):
            ev = event_pb2.Event()
            try:
                ev.ParseFromString(payload)
            except Exception:
                continue
            if not ev.summary.value:
                continue

            step = int(ev.step)
            if step > max_scalar_step:
                max_scalar_step = step

            for v in ev.summary.value:
                if v.tag != metric_tag:
                    continue
                value: Optional[float] = None
                if v.HasField("simple_value"):
                    value = float(v.simple_value)
                elif v.HasField("tensor"):
                    value = tensor_scalar_to_float(v.tensor)
                if value is None:
                    continue
                if best_metric_step is None or step >= best_metric_step:
                    best_metric_step = step
                    best_metric_value = value

    return best_metric_step, best_metric_value, max_scalar_step, event_files


def collect_runs(task_root: Path, metric_tag: str) -> List[RunInfo]:
    run_dirs = sorted({p.parent for p in task_root.rglob("events.out.tfevents.*")})
    runs: List[RunInfo] = []
    for run_dir in run_dirs:
        algo, seed, sigma, label = extract_run_identity(run_dir)
        metric_step, metric_value, max_scalar_step, event_files = parse_run_metric(run_dir, metric_tag)
        runs.append(
            RunInfo(
                run_dir=run_dir,
                seed=seed,
                algo=algo,
                sigma=sigma,
                label=label,
                metric_step=metric_step,
                metric_value=metric_value,
                max_scalar_step=max_scalar_step,
                event_files=event_files,
            )
        )
    return runs


def sort_label(label: str) -> Tuple[int, float]:
    if label == "ppo":
        return (0, 0.0)
    if label == "sapo":
        return (1, 0.0)
    if label.startswith("gipo_sigma"):
        v = label.replace("gipo_sigma", "")
        try:
            return (2, float(v))
        except ValueError:
            return (2, 1e9)
    return (3, 0.0)


def pretty_label(label: str) -> str:
    if label == "ppo":
        return "ppo"
    if label == "sapo":
        return "sapo"
    if label.startswith("gipo_sigma"):
        return f"gipo {label.replace('gipo_sigma', '')}"
    return label


def make_output_filename(metric: str, min_iter: int) -> str:
    safe_metric = re.sub(r"[^A-Za-z0-9]+", "_", metric).strip("_")
    if not safe_metric:
        safe_metric = "metric"
    return f"summary_{safe_metric}_min{min_iter}.txt"


def main() -> None:
    args = parse_args()
    task_root = args.task_root.resolve()
    if not task_root.exists():
        raise FileNotFoundError(f"任务目录不存在: {task_root}")

    runs = collect_runs(task_root, args.metric)
    if not runs:
        print(f"[Error] 未发现事件文件: {task_root}")
        return

    filtered: List[FilteredRun] = []
    kept_by_label: Dict[str, List[RunInfo]] = {}

    for run in runs:
        if not run.event_files:
            filtered.append(
                FilteredRun(
                    run_dir=run.run_dir,
                    label=run.label,
                    seed=run.seed,
                    reason="没有 events.out.tfevents.* 文件",
                )
            )
            continue
        if run.max_scalar_step < args.min_iter:
            filtered.append(
                FilteredRun(
                    run_dir=run.run_dir,
                    label=run.label,
                    seed=run.seed,
                    reason=f"最大标量 step={run.max_scalar_step} < min_iter={args.min_iter}",
                )
            )
            continue
        if run.metric_step is None or run.metric_value is None:
            filtered.append(
                FilteredRun(
                    run_dir=run.run_dir,
                    label=run.label,
                    seed=run.seed,
                    reason=f"缺失指标 {args.metric}",
                )
            )
            continue
        if run.metric_step < args.min_iter:
            filtered.append(
                FilteredRun(
                    run_dir=run.run_dir,
                    label=run.label,
                    seed=run.seed,
                    reason=f"指标 {args.metric} 的最新 step={run.metric_step} < min_iter={args.min_iter}",
                )
            )
            continue
        kept_by_label.setdefault(run.label, []).append(run)

    report_lines: List[str] = []
    report_lines.append("=" * 90)
    report_lines.append(f"Task root     : {task_root}")
    report_lines.append(f"Metric tag    : {args.metric}")
    report_lines.append(f"Min iteration : {args.min_iter}")
    report_lines.append(f"Expected seeds: {args.expected_seeds}")
    report_lines.append(f"Total runs    : {len(runs)}")
    report_lines.append(f"Kept runs     : {sum(len(v) for v in kept_by_label.values())}")
    report_lines.append(f"Filtered runs : {len(filtered)}")
    report_lines.append("=" * 90)
    report_lines.append("")
    report_lines.append("### 聚合结果 (Mean ± Std)")
    if not kept_by_label:
        report_lines.append("没有满足条件的数据。")
    else:
        for label in sorted(kept_by_label, key=sort_label):
            items = kept_by_label[label]
            values = [x.metric_value for x in items if x.metric_value is not None]
            seeds = sorted([x.seed for x in items if x.seed is not None])
            if not values:
                continue
            m = mean(values)
            s = stdev(values) if len(values) >= 2 else 0.0
            report_lines.append(
                f"{pretty_label(label):<10s} : "
                f"{m:.4f} ± {s:.4f} "
                f"(n={len(values)}/{args.expected_seeds}, seeds={seeds})"
            )

            for item in sorted(items, key=lambda x: (x.seed is None, x.seed)):
                report_lines.append(
                    f"    - seed={item.seed} value={item.metric_value:.4f} "
                    f"(metric_step={item.metric_step}, max_step={item.max_scalar_step})"
                )
    report_lines.append("")

    report_lines.append("### 被过滤数据与原因")
    if not filtered:
        report_lines.append("无。")
    else:
        for fr in sorted(filtered, key=lambda x: (sort_label(x.label), x.run_dir.name)):
            report_lines.append(
                f"- [{pretty_label(fr.label)}] seed={fr.seed} "
                f"run={fr.run_dir.name} | reason: {fr.reason}"
            )

    report_text = "\n".join(report_lines)
    print(report_text)

    output_name = args.output_file.strip() if args.output_file.strip() else make_output_filename(args.metric, args.min_iter)
    output_path = task_root / output_name
    output_path.write_text(report_text + "\n")
    print(f"\n[Saved] {output_path}")


if __name__ == "__main__":
    main()
