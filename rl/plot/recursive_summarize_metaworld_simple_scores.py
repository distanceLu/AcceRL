#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Sequence, Set, Tuple

from aggregate_summary_tables import (
    build_markdown_content,
    collect_algo_labels,
    compute_elo_rankings,
    compute_mean_elo_rankings,
    filter_algo_labels,
    gather_entries,
    parse_summary_file,
    write_data_csv,
    write_elo_csv,
    write_meta_csv,
)
from summarize_metaworld_simple_scores import (
    FilteredRun,
    RunInfo,
    collect_runs,
    make_output_filename,
    pretty_label,
    sort_label,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "递归统计 runs/MetaWorldSimple 下每个 group 的分数。"
            "如果 group 目录下已存在对应的 summary 文件则直接复用，"
            "否则调用单 group 统计逻辑重新生成，最后再聚合到搜索根目录。"
        )
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        default=Path("runs/MetaWorldSimple"),
        help="递归搜索 group 的根目录，默认 runs/MetaWorldSimple",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="Rollout/ReturnMean",
        help="用于统计的 TensorBoard 标量 tag，默认 Rollout/ReturnMean",
    )
    parser.add_argument(
        "--min-iter",
        type=int,
        default=900,
        help="只保留最大迭代步达到该阈值的 run，默认 900",
    )
    parser.add_argument(
        "--expected-seeds",
        type=int,
        default=5,
        help="期望的 seed 数量，用于报告完整性，默认 5",
    )
    parser.add_argument(
        "--summary-file",
        type=str,
        default="",
        help=(
            "group 内 summary 文件名。为空时自动根据 metric/min-iter 生成，"
            "如 summary_Rollout_ReturnMean_min900.txt"
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="",
        help=(
            "最终聚合结果文件名前缀。为空时自动命名为 "
            "aggregated_summary_table_<metric>_min<iter>"
        ),
    )
    return parser.parse_args()


def make_aggregate_output_prefix(metric: str, min_iter: int) -> str:
    safe_metric = re.sub(r"[^A-Za-z0-9]+", "_", metric).strip("_")
    if not safe_metric:
        safe_metric = "metric"
    return f"aggregated_summary_table_{safe_metric}_min{min_iter}"


def discover_group_dirs(search_root: Path, summary_name: str) -> List[Path]:
    group_dirs: Set[Path] = set()

    for summary_path in search_root.rglob(summary_name):
        if summary_path.is_file():
            group_dirs.add(summary_path.parent.resolve())

    for event_path in search_root.rglob("events.out.tfevents.*"):
        run_dir = event_path.parent
        if run_dir.name == "checkpoints":
            run_dir = run_dir.parent
        group_dir = run_dir.parent.resolve()
        if group_dir == search_root or search_root in group_dir.parents:
            group_dirs.add(group_dir)

    return sorted(group_dirs, key=lambda path: str(path.relative_to(search_root)))


def build_group_report(
    task_root: Path,
    metric: str,
    min_iter: int,
    expected_seeds: int,
) -> str:
    runs = collect_runs(task_root, metric)
    if not runs:
        raise RuntimeError(f"未发现事件文件: {task_root}")

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
        if run.max_scalar_step < min_iter:
            filtered.append(
                FilteredRun(
                    run_dir=run.run_dir,
                    label=run.label,
                    seed=run.seed,
                    reason=f"最大标量 step={run.max_scalar_step} < min_iter={min_iter}",
                )
            )
            continue
        if run.metric_step is None or run.metric_value is None:
            filtered.append(
                FilteredRun(
                    run_dir=run.run_dir,
                    label=run.label,
                    seed=run.seed,
                    reason=f"缺失指标 {metric}",
                )
            )
            continue
        if run.metric_step < min_iter:
            filtered.append(
                FilteredRun(
                    run_dir=run.run_dir,
                    label=run.label,
                    seed=run.seed,
                    reason=f"指标 {metric} 的最新 step={run.metric_step} < min_iter={min_iter}",
                )
            )
            continue
        kept_by_label.setdefault(run.label, []).append(run)

    report_lines: List[str] = []
    report_lines.append("=" * 90)
    report_lines.append(f"Task root     : {task_root}")
    report_lines.append(f"Metric tag    : {metric}")
    report_lines.append(f"Min iteration : {min_iter}")
    report_lines.append(f"Expected seeds: {expected_seeds}")
    report_lines.append(f"Total runs    : {len(runs)}")
    report_lines.append(f"Kept runs     : {sum(len(items) for items in kept_by_label.values())}")
    report_lines.append(f"Filtered runs : {len(filtered)}")
    report_lines.append("=" * 90)
    report_lines.append("")
    report_lines.append("### 聚合结果 (Mean ± Std)")
    if not kept_by_label:
        report_lines.append("没有满足条件的数据。")
    else:
        for label in sorted(kept_by_label, key=sort_label):
            items = kept_by_label[label]
            values = [item.metric_value for item in items if item.metric_value is not None]
            seeds = sorted(item.seed for item in items if item.seed is not None)
            if not values:
                continue
            avg = mean(values)
            std = stdev(values) if len(values) >= 2 else 0.0
            report_lines.append(
                f"{pretty_label(label):<10s} : "
                f"{avg:.4f} ± {std:.4f} "
                f"(n={len(values)}/{expected_seeds}, seeds={seeds})"
            )
            for item in sorted(items, key=lambda current: (current.seed is None, current.seed)):
                report_lines.append(
                    f"    - seed={item.seed} value={item.metric_value:.4f} "
                    f"(metric_step={item.metric_step}, max_step={item.max_scalar_step})"
                )
    report_lines.append("")

    report_lines.append("### 被过滤数据与原因")
    if not filtered:
        report_lines.append("无。")
    else:
        for item in sorted(filtered, key=lambda current: (sort_label(current.label), current.run_dir.name)):
            report_lines.append(
                f"- [{pretty_label(item.label)}] seed={item.seed} "
                f"run={item.run_dir.name} | reason: {item.reason}"
            )
    return "\n".join(report_lines)


def summarize_one_group(
    group_dir: Path,
    summary_name: str,
    metric: str,
    min_iter: int,
    expected_seeds: int,
) -> Tuple[str, Path, bool]:
    summary_path = group_dir / summary_name
    if summary_path.exists() and parse_summary_file(summary_path) is not None:
        return summary_path.read_text(encoding="utf-8").rstrip(), summary_path, True

    report_text = build_group_report(
        task_root=group_dir.resolve(),
        metric=metric,
        min_iter=min_iter,
        expected_seeds=expected_seeds,
    )
    summary_path.write_text(report_text + "\n", encoding="utf-8")
    return report_text, summary_path, False


def print_group_result(
    index: int,
    total: int,
    search_root: Path,
    group_dir: Path,
    report_text: str,
    summary_path: Path,
    reused: bool,
) -> None:
    rel_group = group_dir.relative_to(search_root)
    rel_summary = summary_path.relative_to(search_root)
    action = "Reused" if reused else "Saved"
    print("")
    print("#" * 110)
    print(f"[{index}/{total}] {rel_group}")
    print("#" * 110)
    print(report_text)
    print(f"\n[{action}] {rel_summary}")


def write_aggregate_outputs(
    search_root: Path,
    summary_name: str,
    output_prefix: str,
) -> int:
    entries = gather_entries(search_root, summary_name)
    if not entries:
        print(f"[Info] 在 {search_root} 下未找到可解析的 summary 文件（glob={summary_name}）")
        return 0

    entries = sorted(entries, key=lambda item: (item.task_name, str(item.task_root), str(item.summary_file)))
    algo_labels = collect_algo_labels(entries)
    filtered_algo_labels = filter_algo_labels(algo_labels)
    elo_standings = compute_elo_rankings(entries, algo_labels)
    filtered_elo_standings = compute_elo_rankings(entries, filtered_algo_labels)
    mean_elo_standings = compute_mean_elo_rankings(entries, algo_labels)
    filtered_mean_elo_standings = compute_mean_elo_rankings(entries, filtered_algo_labels)
    markdown_content = build_markdown_content(entries, algo_labels, filtered_algo_labels)

    output_md = search_root / f"{output_prefix}.md"
    output_csv = search_root / f"{output_prefix}.csv"
    output_meta_csv = search_root / f"{output_prefix}_meta.csv"
    output_elo_csv = search_root / f"{output_prefix}_elo.csv"
    output_filtered_elo_csv = search_root / f"{output_prefix}_elo_filtered.csv"
    output_mean_elo_csv = search_root / f"{output_prefix}_elo_mean.csv"
    output_filtered_mean_elo_csv = search_root / f"{output_prefix}_elo_mean_filtered.csv"
    output_md.write_text(markdown_content, encoding="utf-8")
    write_data_csv(entries, algo_labels, output_csv)
    write_meta_csv(entries, output_meta_csv)
    write_elo_csv(elo_standings, output_elo_csv)
    write_elo_csv(filtered_elo_standings, output_filtered_elo_csv)
    write_elo_csv(mean_elo_standings, output_mean_elo_csv)
    write_elo_csv(filtered_mean_elo_standings, output_filtered_mean_elo_csv)

    print("")
    print("=" * 110)
    print("### 最终汇总")
    print("=" * 110)
    print(markdown_content)
    print(f"[Saved] markdown: {output_md}")
    print(f"[Saved] csv     : {output_csv}")
    print(f"[Saved] meta csv: {output_meta_csv}")
    print(f"[Saved] elo csv(seed): {output_elo_csv}")
    print(f"[Saved] elo csv(seed, filtered): {output_filtered_elo_csv}")
    print(f"[Saved] elo csv(mean): {output_mean_elo_csv}")
    print(f"[Saved] elo csv(mean, filtered): {output_filtered_mean_elo_csv}")
    print(f"[Done] entries={len(entries)} algos={len(algo_labels)}")
    return len(entries)


def main() -> None:
    args = parse_args()
    search_root = args.search_root.resolve()
    if not search_root.exists():
        raise FileNotFoundError(f"搜索目录不存在: {search_root}")

    summary_name = args.summary_file.strip() if args.summary_file.strip() else make_output_filename(args.metric, args.min_iter)
    output_prefix = (
        args.output_prefix.strip()
        if args.output_prefix.strip()
        else make_aggregate_output_prefix(args.metric, args.min_iter)
    )

    group_dirs = discover_group_dirs(search_root, summary_name)
    if not group_dirs:
        print(f"[Info] 在 {search_root} 下没有发现可统计的 group。")
        return

    reused_count = 0
    built_count = 0
    failed: List[Tuple[Path, str]] = []

    for index, group_dir in enumerate(group_dirs, start=1):
        try:
            report_text, summary_path, reused = summarize_one_group(
                group_dir=group_dir,
                summary_name=summary_name,
                metric=args.metric,
                min_iter=args.min_iter,
                expected_seeds=args.expected_seeds,
            )
            print_group_result(
                index=index,
                total=len(group_dirs),
                search_root=search_root,
                group_dir=group_dir,
                report_text=report_text,
                summary_path=summary_path,
                reused=reused,
            )
            if reused:
                reused_count += 1
            else:
                built_count += 1
        except Exception as exc:
            failed.append((group_dir, str(exc)))
            rel_group = group_dir.relative_to(search_root)
            print(f"[Error] {rel_group}: {exc}")

    entry_count = write_aggregate_outputs(
        search_root=search_root,
        summary_name=summary_name,
        output_prefix=output_prefix,
    )

    print("")
    print(
        f"[Summary] groups={len(group_dirs)} reused={reused_count} "
        f"built={built_count} failed={len(failed)} aggregated_entries={entry_count}"
    )
    if failed:
        print("### 失败的 group")
        for group_dir, reason in failed:
            print(f"- {group_dir.relative_to(search_root)} | {reason}")


if __name__ == "__main__":
    main()
