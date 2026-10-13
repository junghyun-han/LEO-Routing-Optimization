"""Strict session completion versus requested holding time on the frozen 3,600-request test set.

PPO-GCN is the mean of the three seeds' per-X completion; the band is one
standard deviation across seeds.

    python scripts/plot_completion_by_holding_time.py --out docs/figures/completion_by_holding_time.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from make_llm_table import ORDER, baseline_rows  # noqa: E402

STYLE = {"Dijkstra": ("#7f7f7f", "o"), "PRP": ("#ff7f0e", "^"), "Cost-Greedy": ("#9467bd", "v"),
         "Adapted ISASR": ("#2ca02c", "D"), "Lifetime": ("#1f77b4", "s"), "PPO-GCN": ("#d62728", "*")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "docs/figures/completion_by_holding_time.png")
    args = ap.parse_args()
    rows = baseline_rows()
    rows["seed"] = rows["seed"].fillna(-1)
    per_seed = 100 * rows.groupby(["method", "seed", "duration_s"]).completion.mean()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for method in ORDER:
        s = per_seed.loc[method].unstack("duration_s")
        mean, std = s.mean(), s.std(ddof=1) if len(s) > 1 else None
        color, marker = STYLE[method]
        ax.plot(mean.index, mean.values, marker=marker, color=color, label=method,
                linewidth=2.4 if method == "PPO-GCN" else 1.5, markersize=9 if method == "PPO-GCN" else 6)
        if std is not None:
            ax.fill_between(mean.index, mean - std, mean + std, color=color, alpha=0.2, linewidth=0)
    ax.set_xscale("log")
    ax.set_xticks([20, 60, 180, 300, 600])
    ax.set_xticklabels(["20", "60", "180", "300", "600"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("Requested holding time X (s)")
    ax.set_ylabel("Strict session completion (%)")
    ax.set_ylim(25, 102)
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=8, loc="lower left")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
