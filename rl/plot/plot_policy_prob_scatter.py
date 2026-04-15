import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot scatter of old_pi (x) vs new_pi (y) from policy_prob_pairs_latest.csv."
    )
    parser.add_argument(
        "input_csv",
        type=str,
        help="Path to policy_prob_pairs_latest.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path. Default: <input_stem>_scatter.png",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=200000,
        help="Max points to plot (uniformly subsampled if exceeded).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        help="Point alpha transparency.",
    )
    parser.add_argument(
        "--size",
        type=float,
        default=2.0,
        help="Point size.",
    )
    return parser.parse_args()


def _parse_line(line: str, expected_prefix: str) -> np.ndarray:
    if not line.startswith(expected_prefix + ","):
        raise ValueError(f"Expected line starts with '{expected_prefix},'")
    payload = line[len(expected_prefix) + 1 :].strip()
    if not payload:
        raise ValueError(f"No values found after '{expected_prefix},'")
    return np.fromstring(payload, sep=",", dtype=np.float64)


def load_policy_prob_pairs(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with open(csv_path, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]

    if len(lines) < 2:
        raise ValueError("CSV format invalid: expected at least two non-empty lines.")

    old_pi = _parse_line(lines[0], "old_pi")
    new_pi = _parse_line(lines[1], "new_pi")
    if old_pi.size != new_pi.size:
        raise ValueError(
            f"Length mismatch: old_pi={old_pi.size}, new_pi={new_pi.size}."
        )
    if old_pi.size == 0:
        raise ValueError("No data points found.")

    finite_mask = np.isfinite(old_pi) & np.isfinite(new_pi)
    old_pi = old_pi[finite_mask]
    new_pi = new_pi[finite_mask]
    if old_pi.size == 0:
        raise ValueError("All points are non-finite.")
    return old_pi, new_pi


def maybe_subsample(
    old_pi: np.ndarray, new_pi: np.ndarray, max_points: int
) -> Tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or old_pi.size <= max_points:
        return old_pi, new_pi
    indices = np.linspace(0, old_pi.size - 1, num=max_points, dtype=np.int64)
    return old_pi[indices], new_pi[indices]


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    output_path = (
        Path(args.output)
        if args.output
        else input_csv.with_name(f"{input_csv.stem}_scatter.png")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    old_pi, new_pi = load_policy_prob_pairs(input_csv)
    raw_points = old_pi.size
    old_pi, new_pi = maybe_subsample(old_pi, new_pi, args.max_points)
    plotted_points = old_pi.size

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(old_pi, new_pi, s=args.size, alpha=args.alpha, linewidths=0)

    lower = float(min(np.min(old_pi), np.min(new_pi)))
    upper = float(max(np.max(old_pi), np.max(new_pi)))
    if upper > lower:
        ax.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1.2, color="red")
        margin = (upper - lower) * 0.05
        ax.set_xlim(lower - margin, upper + margin)
        ax.set_ylim(lower - margin, upper + margin)

    ax.set_xlabel("old pi")
    ax.set_ylabel("new pi")
    ax.set_title("Policy Probability Scatter")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    print(f"input={input_csv}")
    print(f"output={output_path}")
    print(f"points_raw={raw_points}")
    print(f"points_plotted={plotted_points}")


if __name__ == "__main__":
    main()
