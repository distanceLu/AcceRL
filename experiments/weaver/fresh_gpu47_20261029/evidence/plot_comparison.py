"""Plot saved online success curves. Requires matplotlib; no server/model access."""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter

HERE = Path(__file__).resolve().parent


def main():
    with (HERE / "baseline_curves.csv").open(encoding="utf-8", newline="") as f:
        old = [r for r in csv.DictReader(f) if r["tag"] == "Eval/_Global/Average_Return"]
    with (HERE / "current_curves.csv").open(encoding="utf-8", newline="") as f:
        new = list(csv.DictReader(f))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2), layout="constrained")
    settings = [
        ("elapsed_hours", "hours_from_first_tb_event", "Hours since first TensorBoard event", (0, 160), "Online Eval: first crossing of 95%"),
        ("elapsed_hours", "hours_from_first_tb_event", "Hours since first TensorBoard event", (0, 24), "First 24 hours"),
        ("step", "policy_update", "Completed Policy updates", (0, 2500), "Learning progress per update"),
        ("real_completed_action_steps", "real_completed_action_steps", "Completed real training action steps", (0, 180000), "Real data used (Eval steps excluded)"),
    ]
    for ax, (ok, nk, xlabel, xlim, title) in zip(axes.flat, settings):
        ax.plot([float(r[ok]) for r in old], [float(r["value"]) for r in old], color="#a05a26", lw=1.4, label="Historical CTRL run (Oct 7)")
        ax.plot([float(r[nk]) for r in new], [float(r["eval_success"]) for r in new], color="#126f99", lw=1.8, label="Running WEAVER FRESH (Oct 29)")
        ax.axhline(.95, color="#555555", ls="--", lw=.9)
        ax.set(xlim=xlim, ylim=(.70, 1.005), xlabel=xlabel, ylabel="Online rolling Eval success", title=title)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.18)
    axes[0, 0].annotate("FRESH: 17.62 h / P349", (17.6183, .95), xytext=(30, .86), arrowprops={"arrowstyle": "->", "color": "#126f99"}, color="#126f99")
    axes[0, 0].annotate("Historical: 151.56 h / P2410", (151.5589, .953), xytext=(76, .80), arrowprops={"arrowstyle": "->", "color": "#a05a26"}, color="#a05a26")
    axes[1, 1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    axes[0, 1].legend(loc="lower right", fontsize=9)
    fig.suptitle("Same logged metric, different training recipes and online evaluation protocols", fontsize=14)
    fig.supxlabel("Task 0 only; up to 1,000 completed episodes per rolling window. Repeated states and changing policies; not fixed-checkpoint generalization.", fontsize=9)
    fig.savefig(HERE / "eval_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
