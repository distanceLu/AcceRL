#!/usr/bin/env python3
import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


TASK_ROOT_RE = re.compile(r"^Task root\s*:\s*(.+)\s*$")
METRIC_TAG_RE = re.compile(r"^Metric tag\s*:\s*(.+)\s*$")
MIN_ITER_RE = re.compile(r"^Min iteration\s*:\s*(\d+)\s*$")
ROW_RE = re.compile(
    r"^(?P<label>[^:]+?)\s*:\s*"
    r"(?P<mean>[-+0-9.eE]+)\s*±\s*(?P<std>[-+0-9.eE]+)\s*"
    r"\(n=(?P<n_kept>\d+)/(?P<n_expected>\d+),\s*seeds=\[(?P<seeds>[^\]]*)\]\)\s*$"
)


@dataclass
class SummaryEntry:
    summary_file: Path
    task_root: Path
    task_name: str
    metric_tag: str
    min_iter: int
    algo_stats: Dict[str, Dict[str, str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "递归搜索指定路径下的 summary_*.txt，并汇总为统一表格。"
            "默认输出 Markdown 和 CSV 到搜索根目录。"
        )
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        default=Path("runs/MetaWorldSimple"),
        help="递归搜索 summary 文件的根目录，默认 runs/MetaWorldSimple",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="summary_*.txt",
        help="summary 文件匹配模式，默认 summary_*.txt",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="aggregated_summary_table",
        help="输出文件名前缀（不含扩展名），默认 aggregated_summary_table",
    )
    return parser.parse_args()


def algo_sort_key(label: str) -> Tuple[int, float]:
    v = label.strip().lower()
    if v == "ppo":
        return (0, 0.0)
    if v == "sapo":
        return (1, 0.0)
    if v.startswith("gipo"):
        m = re.search(r"([-+]?\d*\.?\d+)", v)
        if m:
            try:
                return (2, float(m.group(1)))
            except ValueError:
                return (2, 1e9)
        return (2, 1e9)
    return (3, 0.0)


def infer_task_name(task_root: Path) -> str:
    parts = list(task_root.parts)
    if "MetaWorldSimple" in parts:
        idx = parts.index("MetaWorldSimple")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return task_root.parent.name if task_root.parent.name else task_root.name


def infer_regime(task_root: Path, summary_file: Path) -> str:
    text = f"{task_root} {summary_file}".lower()
    if "fresh" in text:
        return "fresh"
    if "stale" in text:
        return "stale"
    return "stale"


def parse_summary_file(summary_file: Path) -> Optional[SummaryEntry]:
    lines = summary_file.read_text().splitlines()
    task_root: Optional[Path] = None
    metric_tag: Optional[str] = None
    min_iter: Optional[int] = None
    algo_stats: Dict[str, Dict[str, str]] = {}

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        m_task = TASK_ROOT_RE.match(line)
        if m_task:
            task_root = Path(m_task.group(1).strip())
            continue

        m_metric = METRIC_TAG_RE.match(line)
        if m_metric:
            metric_tag = m_metric.group(1).strip()
            continue

        m_min_iter = MIN_ITER_RE.match(line)
        if m_min_iter:
            min_iter = int(m_min_iter.group(1))
            continue

        m_row = ROW_RE.match(line)
        if m_row:
            label = m_row.group("label").strip()
            algo_stats[label] = {
                "mean": f"{float(m_row.group('mean')):.4f}",
                "std": f"{float(m_row.group('std')):.4f}",
                "n_kept": m_row.group("n_kept"),
                "n_expected": m_row.group("n_expected"),
                "seeds": m_row.group("seeds").strip(),
            }

    if task_root is None or metric_tag is None or min_iter is None or not algo_stats:
        return None

    return SummaryEntry(
        summary_file=summary_file,
        task_root=task_root,
        task_name=infer_task_name(task_root),
        metric_tag=metric_tag,
        min_iter=min_iter,
        algo_stats=algo_stats,
    )


def gather_entries(search_root: Path, pattern: str) -> List[SummaryEntry]:
    summary_files = sorted(search_root.rglob(pattern))
    entries: List[SummaryEntry] = []
    for file_path in summary_files:
        parsed = parse_summary_file(file_path)
        if parsed is not None:
            entries.append(parsed)
    return entries


def make_data_markdown_table(entries: Sequence[SummaryEntry], algo_labels: Sequence[str]) -> str:
    headers = ["ID", "Task", "Regime", "Metric", "MinIter", *algo_labels]
    align = ["---"] * len(headers)
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(align) + " |"]

    for idx, e in enumerate(entries, start=1):
        best_mean: Optional[float] = None
        best_labels = set()
        for label, stat in e.algo_stats.items():
            try:
                m = float(stat["mean"])
            except (KeyError, ValueError):
                continue
            if best_mean is None or m > best_mean:
                best_mean = m
                best_labels = {label}
            elif m == best_mean:
                best_labels.add(label)

        row = [
            str(idx),
            e.task_name,
            infer_regime(e.task_root, e.summary_file),
            e.metric_tag,
            str(e.min_iter),
        ]
        for label in algo_labels:
            if label in e.algo_stats:
                s = e.algo_stats[label]
                cell = f"{s['mean']} ± {s['std']} (n={s['n_kept']}/{s['n_expected']})"
                if label in best_labels:
                    cell = f"**{cell}**"
                row.append(cell)
            else:
                row.append("-")
        rows.append("| " + " | ".join(row) + " |")
    return "\n".join(rows) + "\n"


def make_meta_markdown_table(entries: Sequence[SummaryEntry]) -> str:
    headers = ["ID", "Task", "TaskRoot", "SummaryFile"]
    align = ["---"] * len(headers)
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(align) + " |"]
    for idx, e in enumerate(entries, start=1):
        rows.append(
            "| "
            + " | ".join([str(idx), e.task_name, str(e.task_root), str(e.summary_file)])
            + " |"
        )
    return "\n".join(rows) + "\n"


def write_data_csv(entries: Sequence[SummaryEntry], algo_labels: Sequence[str], path: Path) -> None:
    fieldnames: List[str] = [
        "id",
        "task",
        "regime",
        "metric",
        "min_iter",
    ]
    for label in algo_labels:
        safe = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")
        fieldnames.extend(
            [
                f"{safe}_mean",
                f"{safe}_std",
                f"{safe}_n_kept",
                f"{safe}_n_expected",
                f"{safe}_seeds",
            ]
        )

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, e in enumerate(entries, start=1):
            row: Dict[str, str] = {
                "id": str(idx),
                "task": e.task_name,
                "regime": infer_regime(e.task_root, e.summary_file),
                "metric": e.metric_tag,
                "min_iter": str(e.min_iter),
            }
            for label in algo_labels:
                safe = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")
                if label in e.algo_stats:
                    s = e.algo_stats[label]
                    row[f"{safe}_mean"] = s["mean"]
                    row[f"{safe}_std"] = s["std"]
                    row[f"{safe}_n_kept"] = s["n_kept"]
                    row[f"{safe}_n_expected"] = s["n_expected"]
                    row[f"{safe}_seeds"] = s["seeds"]
                else:
                    row[f"{safe}_mean"] = ""
                    row[f"{safe}_std"] = ""
                    row[f"{safe}_n_kept"] = ""
                    row[f"{safe}_n_expected"] = ""
                    row[f"{safe}_seeds"] = ""
            writer.writerow(row)


def write_meta_csv(entries: Sequence[SummaryEntry], path: Path) -> None:
    fieldnames = ["id", "task", "task_root", "summary_file"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, e in enumerate(entries, start=1):
            writer.writerow(
                {
                    "id": str(idx),
                    "task": e.task_name,
                    "task_root": str(e.task_root),
                    "summary_file": str(e.summary_file),
                }
            )


def main() -> None:
    args = parse_args()
    search_root = args.search_root.resolve()
    if not search_root.exists():
        raise FileNotFoundError(f"搜索目录不存在: {search_root}")

    entries = gather_entries(search_root, args.glob)
    if not entries:
        print(f"[Info] 在 {search_root} 下未找到可解析的 summary 文件（glob={args.glob}）")
        return

    entries = sorted(entries, key=lambda x: (x.task_name, str(x.task_root), str(x.summary_file)))
    algo_labels = sorted({k for e in entries for k in e.algo_stats.keys()}, key=algo_sort_key)

    data_md = make_data_markdown_table(entries, algo_labels)
    meta_md = make_meta_markdown_table(entries)
    markdown_content = (
        "## 数据表\n\n"
        + data_md
        + "\n## 元信息表（TaskRoot / SummaryFile）\n\n"
        + meta_md
    )
    print(markdown_content)

    output_md = search_root / f"{args.output_prefix}.md"
    output_csv = search_root / f"{args.output_prefix}.csv"
    output_meta_csv = search_root / f"{args.output_prefix}_meta.csv"
    output_md.write_text(markdown_content)
    write_data_csv(entries, algo_labels, output_csv)
    write_meta_csv(entries, output_meta_csv)

    print(f"[Saved] markdown: {output_md}")
    print(f"[Saved] csv     : {output_csv}")
    print(f"[Saved] meta csv: {output_meta_csv}")
    print(f"[Done] entries={len(entries)} algos={len(algo_labels)}")


if __name__ == "__main__":
    main()
