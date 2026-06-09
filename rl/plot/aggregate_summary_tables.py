#!/usr/bin/env python3
import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


TASK_ROOT_RE = re.compile(r"^Task root\s*:\s*(.+)\s*$")
REGIME_RE = re.compile(r"^Regime\s*:\s*(.+)\s*$")
METRIC_TAG_RE = re.compile(r"^Metric tag\s*:\s*(.+)\s*$")
MIN_ITER_RE = re.compile(r"^Min iteration\s*:\s*(\d+)\s*$")
ROW_RE = re.compile(
    r"^(?P<label>[^:]+?)\s*:\s*"
    r"(?P<mean>[-+0-9.eE]+)\s*±\s*(?P<std>[-+0-9.eE]+)\s*"
    r"\(n=(?P<n_kept>\d+)/(?P<n_expected>\d+),\s*seeds=\[(?P<seeds>[^\]]*)\]\)\s*$"
)
SEED_VALUE_RE = re.compile(
    r"^- seed=(?P<seed>\d+)\s+value=(?P<value>[-+0-9.eE]+)\s+\(.*\)\s*$"
)
FILTERED_LABEL_RE = re.compile(r"^- \[(?P<label>[^\]]+)\]\s+seed=(?P<seed>\d+)\s+run=.*$")
GIPO_INTERNAL_LABEL_RE = re.compile(
    r"^gipo_sigma(?P<sigma>unknown|[-+]?\d*\.?\d+)(?:_neg(?P<sigma_neg_ratio>[-+]?\d*\.?\d+))?$",
    re.IGNORECASE,
)
GIPO_PRETTY_LABEL_RE = re.compile(
    r"^gipo\s+(?P<sigma>unknown|[-+]?\d*\.?\d+)(?:\s+neg\s+(?P<sigma_neg_ratio>[-+]?\d*\.?\d+))?$",
    re.IGNORECASE,
)

ELO_INITIAL_RATING = 1000.0
ELO_SCALE = 400.0
ELO_ITERATIONS = 10000
ELO_TOLERANCE = 1e-12
ELO_EPSILON = 1e-12


@dataclass
class SummaryEntry:
    summary_file: Path
    task_root: Path
    task_name: str
    metric_tag: str
    min_iter: int
    algo_stats: Dict[str, Dict[str, str]]
    algo_seed_values: Dict[str, Dict[int, float]]
    all_algo_labels: List[str]
    regime: str


@dataclass(frozen=True)
class PairwiseMatch:
    task_name: str
    task_root: Path
    summary_file: Path
    metric_tag: str
    min_iter: int
    seed: Optional[int]
    algo_a: str
    algo_b: str
    value_a: float
    value_b: float


@dataclass
class EloStanding:
    label: str
    rating: float = ELO_INITIAL_RATING
    matches: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0

    @property
    def win_rate(self) -> float:
        if self.matches == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.matches


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


def parse_float_token(raw_value: Optional[str]) -> Optional[float]:
    if raw_value is None:
        return None
    try:
        return float(raw_value)
    except ValueError:
        return None


def parse_gipo_label_components(label: str) -> Optional[Tuple[Optional[float], float]]:
    raw_label = label.strip().lower()
    internal_match = GIPO_INTERNAL_LABEL_RE.fullmatch(raw_label)
    if internal_match:
        sigma_raw = internal_match.group("sigma")
        sigma = None if sigma_raw == "unknown" else parse_float_token(sigma_raw)
        sigma_neg_ratio = parse_float_token(internal_match.group("sigma_neg_ratio"))
        return sigma, 1.0 if sigma_neg_ratio is None else sigma_neg_ratio

    normalized_label = " ".join(raw_label.replace("_unknown", " unknown").split())
    pretty_match = GIPO_PRETTY_LABEL_RE.fullmatch(normalized_label)
    if pretty_match:
        sigma_raw = pretty_match.group("sigma")
        sigma = None if sigma_raw == "unknown" else parse_float_token(sigma_raw)
        sigma_neg_ratio = parse_float_token(pretty_match.group("sigma_neg_ratio"))
        return sigma, 1.0 if sigma_neg_ratio is None else sigma_neg_ratio

    return None


def algo_sort_key(label: str) -> Tuple[int, float, float]:
    v = label.strip().lower()
    if v == "ppo":
        return (0, 0.0, 0.0)
    if v == "sapo":
        return (1, 0.0, 0.0)
    parsed_gipo = parse_gipo_label_components(label)
    if parsed_gipo is not None:
        sigma, sigma_neg_ratio = parsed_gipo
        return (2, sigma if sigma is not None else 1e9, sigma_neg_ratio)
    return (3, 0.0, 0.0)


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


def parse_summary_file(summary_file: Path) -> List[SummaryEntry]:
    lines = summary_file.read_text(encoding="utf-8").splitlines()
    entries = []
    
    task_root: Optional[Path] = None
    metric_tag: Optional[str] = None
    min_iter: Optional[int] = None
    regime: Optional[str] = None
    algo_stats: Dict[str, Dict[str, str]] = {}
    algo_seed_values: Dict[str, Dict[int, float]] = {}
    all_algo_labels = set()
    current_label: Optional[str] = None

    def save_entry():
        nonlocal task_root, metric_tag, min_iter, regime, algo_stats, algo_seed_values, all_algo_labels, current_label
        if task_root is not None and metric_tag is not None and min_iter is not None and algo_stats:
            if regime is None:
                regime = infer_regime(task_root, summary_file)
            entries.append(SummaryEntry(
                summary_file=summary_file,
                task_root=task_root,
                task_name=infer_task_name(task_root),
                metric_tag=metric_tag,
                min_iter=min_iter,
                algo_stats=algo_stats,
                algo_seed_values=algo_seed_values,
                all_algo_labels=sorted(all_algo_labels, key=algo_sort_key),
                regime=regime,
            ))
        task_root = None
        metric_tag = None
        min_iter = None
        regime = None
        algo_stats = {}
        algo_seed_values = {}
        all_algo_labels = set()
        current_label = None

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        m_task = TASK_ROOT_RE.match(line)
        if m_task:
            save_entry()
            task_root = Path(m_task.group(1).strip())
            continue

        m_regime = REGIME_RE.match(line)
        if m_regime:
            regime = m_regime.group(1).strip()
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
            algo_seed_values.setdefault(label, {})
            all_algo_labels.add(label)
            current_label = label
            continue

        m_seed_value = SEED_VALUE_RE.match(line)
        if current_label is not None and m_seed_value:
            seed = int(m_seed_value.group("seed"))
            value = float(m_seed_value.group("value"))
            algo_seed_values.setdefault(current_label, {})[seed] = value
            continue

        m_filtered_label = FILTERED_LABEL_RE.match(line)
        if m_filtered_label:
            all_algo_labels.add(m_filtered_label.group("label").strip())
            continue

        if line.startswith("### "):
            current_label = None

    save_entry()
    return entries


def gather_entries(search_root: Path, pattern: str) -> List[SummaryEntry]:
    summary_files = sorted(search_root.rglob(pattern))
    entries: List[SummaryEntry] = []
    for file_path in summary_files:
        parsed_entries = parse_summary_file(file_path)
        entries.extend(parsed_entries)
    return entries


def should_exclude_from_filtered_table(label: str) -> bool:
    parsed_gipo = parse_gipo_label_components(label)
    if parsed_gipo is None:
        return False
    sigma, _ = parsed_gipo
    return sigma in {0.1, 2.0}


def collect_algo_labels(entries: Sequence[SummaryEntry]) -> List[str]:
    return sorted(
        {
            label
            for entry in entries
            for label in entry.algo_stats.keys()
            if label != "unknown"
        },
        key=algo_sort_key,
    )


def filter_algo_labels(algo_labels: Sequence[str]) -> List[str]:
    return [label for label in algo_labels if not should_exclude_from_filtered_table(label)]


def compute_iqm(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    sorted_values = sorted(values)
    n = len(sorted_values)
    trim = int(n * 0.25)
    if n - 2 * trim <= 0:
        return sum(sorted_values) / n
    central = sorted_values[trim : n - trim]
    return sum(central) / len(central)


def safe_normalize_score(value: float, score_min: float, score_max: float) -> float:
    if score_max <= score_min:
        return 0.5
    return (value - score_min) / (score_max - score_min)


def make_data_markdown_table(entries: Sequence[SummaryEntry], algo_labels: Sequence[str]) -> str:
    headers = ["ID", "Task", "Regime", "Metric", "MinIter", *algo_labels]
    align = ["---"] * len(headers)
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(align) + " |"]
    displayed_algo_labels = set(algo_labels)

    for idx, e in enumerate(entries, start=1):
        best_mean: Optional[float] = None
        best_labels = set()
        row_means: Dict[str, float] = {}
        for label, stat in e.algo_stats.items():
            if label not in displayed_algo_labels:
                continue
            try:
                m = float(stat["mean"])
            except (KeyError, ValueError):
                continue
            row_means[label] = m
            if best_mean is None or m > best_mean:
                best_mean = m
                best_labels = {label}
            elif m == best_mean:
                best_labels.add(label)

        score_min = min(row_means.values()) if row_means else 0.0
        score_max = max(row_means.values()) if row_means else 1.0

        row = [
            str(idx),
            e.task_name,
            e.regime,
            e.metric_tag,
            str(e.min_iter),
        ]
        for label in algo_labels:
            if label in e.algo_stats:
                s = e.algo_stats[label]
                mean_value = row_means.get(label)
                seed_values = list(e.algo_seed_values.get(label, {}).values())
                iqm_value = compute_iqm(seed_values)
                norm_value = (
                    safe_normalize_score(mean_value, score_min, score_max)
                    if mean_value is not None
                    else None
                )
                iqm_text = "-" if iqm_value is None else f"{iqm_value:.4f}"
                norm_text = "-" if norm_value is None else f"{norm_value:.4f}"
                cell = (
                    f"{s['mean']} ± {s['std']} (n={s['n_kept']}/{s['n_expected']}, "
                    f"IQM={iqm_text}, Norm={norm_text})"
                )
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


def make_pairwise_matches(entries: Sequence[SummaryEntry], algo_labels: Sequence[str]) -> List[PairwiseMatch]:
    matches: List[PairwiseMatch] = []

    for entry in entries:
        entry_labels = [
            label
            for label in algo_labels
            if label in entry.algo_seed_values and entry.algo_seed_values[label]
        ]
        for idx, algo_a in enumerate(entry_labels):
            seeds_a = entry.algo_seed_values[algo_a]
            for algo_b in entry_labels[idx + 1 :]:
                seeds_b = entry.algo_seed_values[algo_b]
                shared_seeds = sorted(set(seeds_a) & set(seeds_b))
                for seed in shared_seeds:
                    matches.append(
                        PairwiseMatch(
                            task_name=entry.task_name,
                            task_root=entry.task_root,
                            summary_file=entry.summary_file,
                            metric_tag=entry.metric_tag,
                            min_iter=entry.min_iter,
                            seed=seed,
                            algo_a=algo_a,
                            algo_b=algo_b,
                            value_a=seeds_a[seed],
                            value_b=seeds_b[seed],
                        )
                    )

    return sorted(
        matches,
        key=lambda match: (
            match.task_name,
            str(match.task_root),
            str(match.summary_file),
            match.metric_tag,
            match.min_iter,
            match.seed is None,
            match.seed if match.seed is not None else -1,
            algo_sort_key(match.algo_a),
            algo_sort_key(match.algo_b),
        ),
    )


def make_mean_pairwise_matches(entries: Sequence[SummaryEntry], algo_labels: Sequence[str]) -> List[PairwiseMatch]:
    matches: List[PairwiseMatch] = []

    for entry in entries:
        entry_means: Dict[str, float] = {}
        for label in algo_labels:
            stat = entry.algo_stats.get(label)
            if not stat:
                continue
            try:
                entry_means[label] = float(stat["mean"])
            except (KeyError, ValueError):
                continue

        entry_labels = [label for label in algo_labels if label in entry_means]
        for idx, algo_a in enumerate(entry_labels):
            for algo_b in entry_labels[idx + 1 :]:
                matches.append(
                    PairwiseMatch(
                        task_name=entry.task_name,
                        task_root=entry.task_root,
                        summary_file=entry.summary_file,
                        metric_tag=entry.metric_tag,
                        min_iter=entry.min_iter,
                        seed=None,
                        algo_a=algo_a,
                        algo_b=algo_b,
                        value_a=entry_means[algo_a],
                        value_b=entry_means[algo_b],
                    )
                )

    return sorted(
        matches,
        key=lambda match: (
            match.task_name,
            str(match.task_root),
            str(match.summary_file),
            match.metric_tag,
            match.min_iter,
            algo_sort_key(match.algo_a),
            algo_sort_key(match.algo_b),
        ),
    )


def summarize_pairwise_matches(matches: Sequence[PairwiseMatch]) -> Dict[Tuple[str, str], Dict[str, float]]:
    pair_stats: Dict[Tuple[str, str], Dict[str, float]] = {}

    for match in matches:
        pair = tuple(sorted((match.algo_a, match.algo_b), key=algo_sort_key))
        if pair not in pair_stats:
            pair_stats[pair] = {"matches": 0.0, pair[0]: 0.0, pair[1]: 0.0}

        stat = pair_stats[pair]
        stat["matches"] += 1.0
        if match.value_a > match.value_b:
            stat[match.algo_a] += 1.0
        elif match.value_a < match.value_b:
            stat[match.algo_b] += 1.0
        else:
            stat[match.algo_a] += 0.5
            stat[match.algo_b] += 0.5

    return pair_stats


def compute_elo_rankings_from_matches(
    matches: Sequence[PairwiseMatch], algo_labels: Sequence[str]
) -> List[EloStanding]:
    standings = {label: EloStanding(label=label) for label in algo_labels}
    pair_stats = summarize_pairwise_matches(matches)

    for match in matches:
        standing_a = standings[match.algo_a]
        standing_b = standings[match.algo_b]

        if match.value_a > match.value_b:
            standing_a.wins += 1
            standing_b.losses += 1
        elif match.value_a < match.value_b:
            standing_a.losses += 1
            standing_b.wins += 1
        else:
            standing_a.draws += 1
            standing_b.draws += 1

        standing_a.matches += 1
        standing_b.matches += 1

    active_labels = [label for label, standing in standings.items() if standing.matches > 0]
    if active_labels:
        strengths = {label: 1.0 for label in active_labels}

        for _ in range(ELO_ITERATIONS):
            updated_strengths = dict(strengths)
            max_change = 0.0

            for label in active_labels:
                observed_score = standings[label].wins + 0.5 * standings[label].draws
                denom = 0.0
                for pair, stat in pair_stats.items():
                    if label not in pair:
                        continue
                    other = pair[1] if pair[0] == label else pair[0]
                    denom += stat["matches"] / max(strengths[label] + strengths[other], ELO_EPSILON)

                if denom <= 0.0:
                    updated = strengths[label]
                else:
                    updated = max(observed_score / denom, ELO_EPSILON)

                updated_strengths[label] = updated
                max_change = max(max_change, abs(updated - strengths[label]))

            mean_strength = sum(updated_strengths.values()) / len(active_labels)
            if mean_strength > 0.0:
                for label in active_labels:
                    updated_strengths[label] /= mean_strength

            strengths = updated_strengths
            if max_change < ELO_TOLERANCE:
                break

        raw_ratings = {
            label: ELO_SCALE * math.log10(max(strengths[label], ELO_EPSILON))
            for label in active_labels
        }
        mean_raw_rating = sum(raw_ratings.values()) / len(active_labels)
        for label in active_labels:
            standings[label].rating = ELO_INITIAL_RATING + raw_ratings[label] - mean_raw_rating

    return sorted(standings.values(), key=lambda standing: (-standing.rating, algo_sort_key(standing.label)))


def compute_elo_rankings(entries: Sequence[SummaryEntry], algo_labels: Sequence[str]) -> List[EloStanding]:
    return compute_elo_rankings_from_matches(make_pairwise_matches(entries, algo_labels), algo_labels)


def compute_mean_elo_rankings(entries: Sequence[SummaryEntry], algo_labels: Sequence[str]) -> List[EloStanding]:
    return compute_elo_rankings_from_matches(make_mean_pairwise_matches(entries, algo_labels), algo_labels)


def make_elo_markdown_table_from_standings(standings: Sequence[EloStanding]) -> str:
    total_matches = sum(standing.matches for standing in standings)
    if not standings or total_matches == 0:
        return "无可用于 ELO 的配对数据。\n"

    headers = ["Rank", "Algo", "ELO", "Matches", "Wins", "Draws", "Losses", "WinRate"]
    align = ["---"] * len(headers)
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(align) + " |"]

    for rank, standing in enumerate(standings, start=1):
        rows.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    standing.label,
                    f"{standing.rating:.2f}",
                    str(standing.matches),
                    str(standing.wins),
                    str(standing.draws),
                    str(standing.losses),
                    f"{standing.win_rate * 100:.2f}%",
                ]
            )
            + " |"
        )
    return "\n".join(rows) + "\n"


def make_elo_markdown_table(entries: Sequence[SummaryEntry], algo_labels: Sequence[str]) -> str:
    return make_elo_markdown_table_from_standings(compute_elo_rankings(entries, algo_labels))


def make_mean_elo_markdown_table(entries: Sequence[SummaryEntry], algo_labels: Sequence[str]) -> str:
    return make_elo_markdown_table_from_standings(compute_mean_elo_rankings(entries, algo_labels))


def build_markdown_content(
    entries: Sequence[SummaryEntry],
    algo_labels: Sequence[str],
    filtered_algo_labels: Sequence[str],
) -> str:
    data_md = make_data_markdown_table(entries, algo_labels)
    filtered_data_md = make_data_markdown_table(entries, filtered_algo_labels)
    elo_md = make_elo_markdown_table(entries, algo_labels)
    filtered_elo_md = make_elo_markdown_table(entries, filtered_algo_labels)
    mean_elo_md = make_mean_elo_markdown_table(entries, algo_labels)
    filtered_mean_elo_md = make_mean_elo_markdown_table(entries, filtered_algo_labels)
    meta_md = make_meta_markdown_table(entries)
    return (
        "## 数据表\n\n"
        + data_md
        + "\n## 数据表（过滤 gipo sigma 0.1 / 2.0）\n\n"
        + filtered_data_md
        + "\n## ELO 排名（按共同 seed）\n\n"
        + elo_md
        + "\n## ELO 排名（按共同 seed，过滤 gipo sigma 0.1 / 2.0）\n\n"
        + filtered_elo_md
        + "\n## ELO 排名（按多 seed 平均分）\n\n"
        + mean_elo_md
        + "\n## ELO 排名（按多 seed 平均分，过滤 gipo sigma 0.1 / 2.0）\n\n"
        + filtered_mean_elo_md
        + "\n## 元信息表（TaskRoot / SummaryFile）\n\n"
        + meta_md
    )


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
                f"{safe}_iqm",
                f"{safe}_norm",
            ]
        )

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, e in enumerate(entries, start=1):
            row_means: Dict[str, float] = {}
            for label in algo_labels:
                stat = e.algo_stats.get(label)
                if not stat:
                    continue
                try:
                    row_means[label] = float(stat["mean"])
                except (KeyError, ValueError):
                    continue
            score_min = min(row_means.values()) if row_means else 0.0
            score_max = max(row_means.values()) if row_means else 1.0

            row: Dict[str, str] = {
                "id": str(idx),
                "task": e.task_name,
                "regime": e.regime,
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
                    mean_value = row_means.get(label)
                    seed_values = list(e.algo_seed_values.get(label, {}).values())
                    iqm_value = compute_iqm(seed_values)
                    norm_value = (
                        safe_normalize_score(mean_value, score_min, score_max)
                        if mean_value is not None
                        else None
                    )
                    row[f"{safe}_iqm"] = "" if iqm_value is None else f"{iqm_value:.4f}"
                    row[f"{safe}_norm"] = "" if norm_value is None else f"{norm_value:.4f}"
                else:
                    row[f"{safe}_mean"] = ""
                    row[f"{safe}_std"] = ""
                    row[f"{safe}_n_kept"] = ""
                    row[f"{safe}_n_expected"] = ""
                    row[f"{safe}_seeds"] = ""
                    row[f"{safe}_iqm"] = ""
                    row[f"{safe}_norm"] = ""
            writer.writerow(row)


def write_elo_csv(standings: Sequence[EloStanding], path: Path) -> None:
    fieldnames = ["rank", "algo", "elo", "matches", "wins", "draws", "losses", "win_rate"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, standing in enumerate(standings, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "algo": standing.label,
                    "elo": f"{standing.rating:.2f}",
                    "matches": standing.matches,
                    "wins": standing.wins,
                    "draws": standing.draws,
                    "losses": standing.losses,
                    "win_rate": f"{standing.win_rate:.4f}",
                }
            )


def write_meta_csv(entries: Sequence[SummaryEntry], path: Path) -> None:
    fieldnames = ["id", "task", "task_root", "summary_file"]
    with path.open("w", newline="", encoding="utf-8") as f:
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

    entries_by_regime: Dict[str, List[SummaryEntry]] = {}
    for entry in entries:
        entries_by_regime.setdefault(entry.regime, []).append(entry)

    all_markdown_blocks = []
    
    for regime in sorted(entries_by_regime.keys()):
        regime_entries = sorted(entries_by_regime[regime], key=lambda item: (item.task_name, str(item.task_root), str(item.summary_file)))
        algo_labels = collect_algo_labels(regime_entries)
        filtered_algo_labels = filter_algo_labels(algo_labels)
        
        elo_standings = compute_elo_rankings(regime_entries, algo_labels)
        filtered_elo_standings = compute_elo_rankings(regime_entries, filtered_algo_labels)
        mean_elo_standings = compute_mean_elo_rankings(regime_entries, algo_labels)
        filtered_mean_elo_standings = compute_mean_elo_rankings(regime_entries, filtered_algo_labels)
        
        markdown_content = build_markdown_content(regime_entries, algo_labels, filtered_algo_labels)
        all_markdown_blocks.append(f"# Regime: {regime}\n\n{markdown_content}")
        
        regime_prefix = f"{args.output_prefix}_{regime}"
        write_elo_csv(elo_standings, search_root / f"{regime_prefix}_elo.csv")
        write_elo_csv(filtered_elo_standings, search_root / f"{regime_prefix}_elo_filtered.csv")
        write_elo_csv(mean_elo_standings, search_root / f"{regime_prefix}_elo_mean.csv")
        write_elo_csv(filtered_mean_elo_standings, search_root / f"{regime_prefix}_elo_mean_filtered.csv")

    final_markdown = "\n\n".join(all_markdown_blocks)
    
    output_md = search_root / f"{args.output_prefix}.md"
    output_csv = search_root / f"{args.output_prefix}.csv"
    output_meta_csv = search_root / f"{args.output_prefix}_meta.csv"
    
    output_md.write_text(final_markdown, encoding="utf-8")
    
    all_entries = sorted(entries, key=lambda item: (item.task_name, str(item.task_root), str(item.summary_file)))
    all_algo_labels = collect_algo_labels(all_entries)
    write_data_csv(all_entries, all_algo_labels, output_csv)
    write_meta_csv(all_entries, output_meta_csv)

    print("")
    print("=" * 110)
    print("### 最终汇总")
    print("=" * 110)
    print(final_markdown)
    print(f"[Saved] markdown: {output_md}")
    print(f"[Saved] csv     : {output_csv}")
    print(f"[Saved] meta csv: {output_meta_csv}")
    print(f"[Saved] elo csvs saved with regime suffixes")
    print(f"[Done] entries={len(entries)}")


if __name__ == "__main__":
    main()
