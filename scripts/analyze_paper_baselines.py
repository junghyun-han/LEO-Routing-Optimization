#!/usr/bin/env python3
"""Validate baseline CSVs and create paper tables/figures with a PPO placeholder."""
from __future__ import annotations

import argparse
import json
import math
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from leo_session import ContactTrace, GroundContactGenerator, WalkerDeltaConstellation  # noqa: E402


ONLINE = ["raw", "lifetime", "prp", "cost_greedy", "remaining"]
SENSITIVITY_ONLINE = ["raw", "lifetime", "prp", "remaining"]
ORDER = ONLINE + ["PPO-GCN", "physical_upper"]
LABEL = {
    "raw": "Raw Dijkstra",
    "lifetime": "Lifetime-Dijkstra",
    "prp": "PRP",
    "cost_greedy": "Local Cost-Greedy (shared reward)",
    "remaining": "Adapted ISASR",
    "PPO-GCN": "GCN-PPO (Ours)",
    "physical_upper": "Physical upper bound",
}
COLORS = {
    "raw": "#6b7280",
    "lifetime": "#2563eb",
    "prp": "#d97706",
    "cost_greedy": "#7c3aed",
    "remaining": "#059669",
    "physical_upper": "#111827",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-dir", type=Path, required=True)
    parser.add_argument(
        "--cost-greedy-dir",
        type=Path,
        default=ROOT / "results_paper/cost_greedy_current_reward_frozen3600",
    )
    parser.add_argument("--sensitivity-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-offsets", type=int, default=240)
    return parser.parse_args()


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


def validate(df: pd.DataFrame, expected_offsets: int) -> None:
    required = {"method", "request_id", "pair", "duration_s", "completion", "setup_success"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if df.duplicated(["method", "request_id"]).any():
        raise ValueError("duplicate method/request rows")
    expected_methods = set(ONLINE + ["physical_upper"])
    if set(df.method) != expected_methods:
        raise ValueError(f"unexpected methods: {sorted(set(df.method))}")
    per_request = df.groupby("request_id").method.nunique()
    if not (per_request == len(expected_methods)).all():
        raise ValueError("not every request has every method")
    expected_requests = 3 * 5 * expected_offsets
    if df.request_id.nunique() != expected_requests:
        raise ValueError(
            f"expected {expected_requests} requests, got {df.request_id.nunique()}"
        )
    if len(df) != expected_requests * len(expected_methods):
        raise ValueError("row count mismatch")
    completion = df.pivot(index="request_id", columns="method", values="completion")
    for method in ONLINE:
        if (completion[method].astype(int) > completion["physical_upper"].astype(int)).any():
            raise ValueError(f"{method} exceeds the declared physical upper bound")
    successful = df[df.setup_success]
    if successful[["setup_delay_s", "initial_hops"]].isna().any().any():
        raise ValueError("successful setup row has missing core metrics")


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, duration), g in df.groupby(["method", "duration_s"], sort=True):
        n = len(g)
        comp = int(g.completion.sum())
        setup = int(g.setup_success.sum())
        clo, chi = wilson(comp, n)
        slo, shi = wilson(setup, n)
        established = g[g.setup_success]
        rows.append({
            "method": method,
            "duration_s": duration,
            "n": n,
            "setup_success_rate": setup / n,
            "setup_ci_low": slo,
            "setup_ci_high": shi,
            "completion_rate": comp / n,
            "completion_ci_low": clo,
            "completion_ci_high": chi,
            "availability": g.availability.mean(),
            "setup_delay_s": established.setup_delay_s.mean(),
            "interruption_s": g.interruption_s.mean(),
            "reconfigurations": g.reconfigurations.mean(),
            "dynamic_setups": g.dynamic_setups.mean(),
            "initial_hops": established.initial_hops.mean(),
            "initial_path_km": established.initial_path_km.mean(),
            "initial_bottleneck_s": established.initial_bottleneck_s.mean(),
            "route_mean_hops": established.mean_hops.mean(),
            "session_max_hops": established.max_hops.mean(),
            "route_mean_path_km": established.mean_path_km.mean(),
            "session_max_path_km": established.max_path_km.mean(),
            "mean_path_stretch_excess": established.mean_path_stretch_excess.mean(),
            "mean_planned_bottleneck_s": established.mean_planned_bottleneck_s.mean(),
            "mean_min_planned_bottleneck_s": established.min_planned_bottleneck_s.mean(),
            "route_decisions": established.route_decisions.mean(),
            "total_pat_time_s": g.total_pat_time_s.mean(),
        })
    return pd.DataFrame(rows)


def paired_tests(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(20260831)
    for duration in sorted(df.duration_s.unique()):
        pivot = df[df.duration_s == duration].pivot(
            index="request_id", columns="method", values="completion"
        ).astype(float)
        comparisons = [
            ("remaining", "raw"),
            ("remaining", "lifetime"),
            ("remaining", "prp"),
            ("cost_greedy", "raw"),
            ("cost_greedy", "lifetime"),
            ("cost_greedy", "prp"),
            ("cost_greedy", "remaining"),
        ]
        for left, right in comparisons:
            a = pivot[left].to_numpy()
            b = pivot[right].to_numpy()
            wins = int(np.sum((a == 1) & (b == 0)))
            losses = int(np.sum((a == 0) & (b == 1)))
            discordant = wins + losses
            pvalue = (
                binomtest(min(wins, losses), discordant, 0.5, alternative="two-sided").pvalue
                if discordant else 1.0
            )
            idx = rng.integers(0, len(a), size=(10000, len(a)))
            delta_samples = (a[idx] - b[idx]).mean(axis=1)
            rows.append({
                "duration_s": duration,
                "comparison": f"{left} - {right}",
                "n": len(a),
                "completion_delta": float(np.mean(a - b)),
                "bootstrap_ci95_low": float(np.quantile(delta_samples, 0.025)),
                "bootstrap_ci95_high": float(np.quantile(delta_samples, 0.975)),
                "left_only_success": wins,
                "right_only_success": losses,
                "mcnemar_exact_p": pvalue,
            })
    result = pd.DataFrame(rows)
    result.to_csv(out / "paired_completion_tests.csv", index=False)
    return result


def sampling_convergence(
    df: pd.DataFrame, out: Path, expected_offsets: int
) -> pd.DataFrame:
    """Compare orbit-spanning interleaved subsets of the frozen offsets."""
    ranked = df.copy()
    ranked["offset_rank"] = ranked.groupby(
        ["method", "pair", "duration_s"]
    )["request_time_s"].rank(method="first").astype(int) - 1
    rows = []
    levels = sorted(set([max(1, expected_offsets // 4), max(1, expected_offsets // 2), expected_offsets]))
    for sample_n in levels:
        stride = expected_offsets // sample_n
        subset = ranked[ranked.offset_rank % stride == 0]
        for (method, duration), g in subset.groupby(["method", "duration_s"]):
            rows.append({
                "offsets_per_pair": sample_n,
                "method": method,
                "duration_s": duration,
                "n": len(g),
                "completion_rate": g.completion.mean(),
                "setup_success_rate": g.setup_success.mean(),
                "availability": g.availability.mean(),
            })
    result = pd.DataFrame(rows)
    result.to_csv(out / "sampling_convergence.csv", index=False)
    return result


def plot_duration(agg: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for method in ONLINE + ["physical_upper"]:
        g = agg[agg.method == method].sort_values("duration_s")
        y = g.completion_rate.to_numpy()
        lo = np.maximum(0.0, y - g.completion_ci_low.to_numpy())
        hi = np.maximum(0.0, g.completion_ci_high.to_numpy() - y)
        ax.errorbar(
            g.duration_s, y, yerr=np.vstack([lo, hi]), marker="o", capsize=3,
            linewidth=1.8, color=COLORS[method], label=LABEL[method],
            linestyle="--" if method == "physical_upper" else "-",
        )
    ax.set(xlabel="Required session duration X (s)", ylabel="Session completion rate", ylim=(-0.03, 1.05))
    ax.set_xticks(sorted(agg.duration_s.unique()))
    ax.grid(axis="y", color="#d1d5db", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "fig_completion_vs_x.png", dpi=300)
    fig.savefig(out / "fig_completion_vs_x.pdf")
    plt.close(fig)


def plot_tradeoff(agg: pd.DataFrame, out: Path, duration: float = 300.0) -> None:
    g = agg[(agg.duration_s == duration) & agg.method.isin(ONLINE)].copy()
    fig, ax = plt.subplots(figsize=(6.1, 4.1))
    for _, row in g.iterrows():
        m = row.method
        ax.scatter(row.initial_path_km, row.completion_rate, s=75, color=COLORS[m], zorder=3)
        ax.annotate(LABEL[m], (row.initial_path_km, row.completion_rate), xytext=(6, 5),
                    textcoords="offset points", fontsize=8)
    ax.set(xlabel="Mean initial path length (km)", ylabel=f"Completion rate (X={duration:g} s)", ylim=(-0.03, 1.05))
    ax.grid(color="#e5e7eb", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_reliability_efficiency_tradeoff.png", dpi=300)
    fig.savefig(out / "fig_reliability_efficiency_tradeoff.pdf")
    plt.close(fig)


def plot_pair_heatmap(df: pd.DataFrame, out: Path) -> None:
    g = (df[df.method.isin(ONLINE)]
         .groupby(["method", "pair", "duration_s"]).completion.mean().reset_index())
    durations = sorted(g.duration_s.unique())
    pairs = ["short", "medium", "long"]
    fig, axes = plt.subplots(
        1, 3, figsize=(10.5, 3.5), sharey=True, layout="constrained"
    )
    image = None
    for ax, pair in zip(axes, pairs):
        mat = np.asarray([
            [g[(g.method == m) & (g.pair == pair) & (g.duration_s == x)].completion.iloc[0]
             for x in durations] for m in ONLINE
        ])
        image = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_title(pair.capitalize())
        ax.set_xticks(range(len(durations)), [f"{x:g}" for x in durations])
        ax.set_xlabel("X (s)")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        color="white" if mat[i,j] < 0.65 else "black", fontsize=7)
    axes[0].set_yticks(range(len(ONLINE)), [LABEL[x] for x in ONLINE])
    fig.colorbar(image, ax=axes, fraction=0.022, pad=0.035, label="Completion rate")
    fig.savefig(out / "fig_completion_by_pair.png", dpi=300)
    fig.savefig(out / "fig_completion_by_pair.pdf")
    plt.close(fig)


def load_sensitivity(root: Path | None) -> pd.DataFrame:
    if root is None or not root.exists():
        return pd.DataFrame()
    frames = []
    for path in sorted(root.glob("setup_*/sessions.csv")):
        frame = pd.read_csv(path)
        frame["setup_delay_factor_s"] = float(path.parent.name.split("_")[1])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_sensitivity(df: pd.DataFrame, out: Path) -> None:
    if df.empty:
        return
    g = (df[df.method.isin(SENSITIVITY_ONLINE)]
         .groupby(["method", "setup_delay_factor_s", "duration_s"])
         .completion.mean().reset_index())
    durations = sorted(g.duration_s.unique())
    fig, axes = plt.subplots(1, len(durations), figsize=(7.6, 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, duration in zip(axes, durations):
        for method in SENSITIVITY_ONLINE:
            s = g[(g.method == method) & (g.duration_s == duration)].sort_values("setup_delay_factor_s")
            ax.plot(s.setup_delay_factor_s, s.completion, marker="o", color=COLORS[method], label=LABEL[method])
        ax.set_title(f"X={duration:g} s")
        ax.set_xlabel("PAT/setup delay (s)")
        ax.grid(axis="y", color="#d1d5db", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Completion rate")
    axes[0].set_ylim(-0.03, 1.05)
    axes[-1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_setup_sensitivity.png", dpi=300)
    fig.savefig(out / "fig_setup_sensitivity.pdf")
    plt.close(fig)


def contact_stats(out: Path) -> dict:
    constellation = WalkerDeltaConstellation()
    trace = ContactTrace.load(ROOT / "cache/contact_trace_1584_24_11.npz")
    ground = GroundContactGenerator(constellation).generate(constellation.period_s + 650.0)
    stats = {}
    for name, values in {"isl": trace.duration_samples(), "gsl": ground.duration_samples()}.items():
        stats[name] = {
            "n": int(len(values)), "min": float(np.min(values)), "max": float(np.max(values)),
            **{f"p{q}": float(np.percentile(values, q)) for q in (10, 25, 50, 75, 90, 95)},
        }
    (out / "contact_statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def fmt_rate(x: float) -> str:
    if pd.isna(x):
        return "--"
    return str(Decimal(str(100.0 * x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def write_tables(agg: pd.DataFrame, out: Path, duration: float = 300.0) -> None:
    g = agg[agg.duration_s == duration].set_index("method")
    rows = []
    for method in ORDER:
        if method == "PPO-GCN":
            rows.append([LABEL[method], "TBD", "TBD", "TBD", "TBD", "TBD", "TBD", "TBD"])
            continue
        r = g.loc[method]
        completion_ci = (
            f"[{100*r.completion_ci_low:.1f}, {100*r.completion_ci_high:.1f}]"
        )
        if method == "physical_upper":
            rows.append([LABEL[method], fmt_rate(r.setup_success_rate), fmt_rate(r.completion_rate), completion_ci, "--", "--", "--", "--"])
        else:
            rows.append([
                LABEL[method], fmt_rate(r.setup_success_rate), fmt_rate(r.completion_rate), completion_ci,
                f"{100*r.availability:.1f}", f"{r.setup_delay_s:.1f}",
                f"{r.interruption_s:.1f}", f"{r.reconfigurations:.2f}",
            ])
    header = ["Method", "Setup success (%)", "Completion (%)", "Completion 95% CI (%)", "Availability (%)", "Setup delay (s)", "Interruption (s)", "Reconfigs"]
    md = [f"# Main baseline table (X={duration:g} s)\n", "| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"]*(len(header)-1)) + "|"]
    md += ["| " + " | ".join(row) + " |" for row in rows]
    (out / "TABLE_MAIN.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    pd.DataFrame(rows, columns=header).to_csv(out / "table_main.csv", index=False)

    eff_rows = []
    for method in ONLINE + ["PPO-GCN"]:
        if method == "PPO-GCN":
            eff_rows.append([LABEL[method], "TBD", "TBD", "TBD", "TBD", "TBD", "TBD"])
        else:
            r = g.loc[method]
            eff_rows.append([
                LABEL[method],
                f"{r.initial_hops:.2f}",
                f"{r.route_mean_hops:.2f}",
                f"{r.session_max_hops:.2f}",
                f"{r.initial_path_km:.1f}",
                f"{r.route_mean_path_km:.1f}",
                f"{r.mean_path_stretch_excess:.3f}",
            ])
    h2 = [
        "Method",
        "Initial hops",
        "Route-mean hops",
        "Mean session-max hops",
        "Initial path (km)",
        "Route-mean path (km)",
        "Mean stretch excess",
    ]
    md2 = [f"# Route-efficiency table (X={duration:g} s)\n", "| " + " | ".join(h2) + " |", "|" + "|".join(["---"] + ["---:"]*(len(h2)-1)) + "|"]
    md2 += ["| " + " | ".join(row) + " |" for row in eff_rows]
    (out / "TABLE_EFFICIENCY.md").write_text("\n".join(md2) + "\n", encoding="utf-8")
    pd.DataFrame(eff_rows, columns=h2).to_csv(out / "table_efficiency.csv", index=False)

    overhead_rows = []
    for method in ONLINE + ["PPO-GCN"]:
        if method == "PPO-GCN":
            overhead_rows.append([LABEL[method]] + ["TBD"] * 8)
        else:
            r = g.loc[method]
            overhead_rows.append([
                LABEL[method],
                f"{r.setup_delay_s:.1f}",
                f"{r.total_pat_time_s:.1f}",
                f"{r.interruption_s:.1f}",
                f"{r.reconfigurations:.2f}",
                f"{r.dynamic_setups:.2f}",
                f"{r.route_decisions:.2f}",
                f"{r.mean_planned_bottleneck_s:.1f}",
                f"{r.mean_min_planned_bottleneck_s:.1f}",
            ])
    h3 = [
        "Method",
        "Initial setup delay (s)",
        "Total PAT time (s)",
        "Interruption (s)",
        "Reconfigs",
        "Dynamic setups",
        "Route decisions",
        "Mean bottleneck (s)",
        "Mean session-min bottleneck (s)",
    ]
    md3 = [
        f"# Control-overhead and stability table (X={duration:g} s)\n",
        "| " + " | ".join(h3) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(h3) - 1)) + "|",
    ]
    md3 += ["| " + " | ".join(row) + " |" for row in overhead_rows]
    (out / "TABLE_CONTROL_OVERHEAD.md").write_text("\n".join(md3) + "\n", encoding="utf-8")
    pd.DataFrame(overhead_rows, columns=h3).to_csv(
        out / "table_control_overhead.csv", index=False
    )


def write_report(
    df: pd.DataFrame,
    agg: pd.DataFrame,
    paired: pd.DataFrame,
    convergence: pd.DataFrame,
    stats: dict,
    sensitivity: pd.DataFrame,
    out: Path,
) -> None:
    x = 300.0
    g = agg[agg.duration_s == x].set_index("method")
    r = g.loc["remaining"]
    wide = convergence.pivot_table(
        index=["method", "duration_s"], columns="offsets_per_pair", values="completion_rate"
    )
    levels = sorted(wide.columns)
    previous_n, full_n = levels[-2], levels[-1]
    max_delta_previous_full = float((wide[previous_n] - wide[full_n]).abs().max())
    strongest_method = g.loc[ONLINE, "completion_rate"].idxmax()
    strongest = g.loc[strongest_method]
    lines = [
        "# CPU baseline results",
        "",
        "This report contains the frozen baseline experiment only. The GCN-PPO row is intentionally left TBD.",
        "",
        "## Frozen model",
        "",
        "Walker 1584/24/11, 550 km, 53 deg, 1,500 km LISL range, 1,000 km GSL range, two persistent plus one dynamic terminal, and 20 s break-before-make PAT.",
        "",
        "## Contact-scale check",
        "",
        f"The generated ISL contact median is {stats['isl']['p50']:.1f} s; the GSL contact median and maximum are {stats['gsl']['p50']:.1f} s and {stats['gsl']['max']:.1f} s. Thus X=300 and 600 s necessarily exercise access handover rather than only one static path.",
        "",
        f"## Representative result at X={x:g} s",
        "",
        f"The strongest achievable CPU baseline is {LABEL[strongest_method]}: setup success {100*strongest.setup_success_rate:.1f}%, strict completion {100*strongest.completion_rate:.1f}%, mean availability {100*strongest.availability:.1f}%, and {strongest.initial_hops:.2f} initial hops.",
        f"Adapted ISASR obtains {100*r.completion_rate:.1f}% strict completion at this operating point.",
        "",
        "## Completion over the frozen X sweep",
        "",
        "| Method | X=20 s | X=60 s | X=180 s | X=300 s | X=600 s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in ONLINE + ["physical_upper"]:
        values = agg[agg.method == method].sort_values("duration_s").completion_rate
        lines.append(
            "| " + LABEL[method] + " | "
            + " | ".join(f"{fmt_rate(v)}%" for v in values) + " |"
        )
    lines += [
        "",
        "## Paired completion comparison at X=300 s",
        "",
        "The following differences use identical request IDs; positive values favor the method on the left.",
        "",
        "| Comparison | Difference (percentage points) | Paired bootstrap 95% CI | Exact McNemar p |",
        "|---|---:|---:|---:|",
    ]
    paired_x = paired[paired.duration_s == x]
    for row in paired_x.itertuples(index=False):
        lines.append(
            f"| {row.comparison} | {100*row.completion_delta:+.1f} | "
            f"[{100*row.bootstrap_ci95_low:+.1f}, {100*row.bootstrap_ci95_high:+.1f}] | "
            f"{row.mcnemar_exact_p:.3g} |"
        )
    if not sensitivity.empty:
        sens = (
            sensitivity[sensitivity.method.isin(SENSITIVITY_ONLINE)]
            .groupby(["method", "setup_delay_factor_s", "duration_s"])
            .completion.mean()
            .reset_index()
        )
        lines += [
            "",
            "## PAT/setup sensitivity",
            "",
            "| Method | X (s) | PAT=0 s | PAT=5 s | PAT=10 s | PAT=20 s |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for method in SENSITIVITY_ONLINE:
            for duration in sorted(sens.duration_s.unique()):
                s = sens[
                    (sens.method == method) & (sens.duration_s == duration)
                ].sort_values("setup_delay_factor_s")
                lines.append(
                    f"| {LABEL[method]} | {duration:g} | "
                    + " | ".join(f"{fmt_rate(v)}%" for v in s.completion)
                    + " |"
                )
        lines += [
            "",
            "Adapted ISASR is policy-adaptive in this sweep: its new-link activeness term is normalized by the known PAT value. Therefore its curve need not be monotone; at larger PAT it can choose a more conservative route that offsets part of the added physical setup cost. The raw values are retained rather than smoothed.",
        ]
    lines += [
        "",
        "## Sampling check",
        "",
        f"Using an orbit-spanning {previous_n}-offset interleaved subset instead of all {full_n} offsets changes any method/X completion rate by at most {100*max_delta_previous_full:.1f} percentage points. Full values are in sampling_convergence.csv.",
        "",
        "See TABLE_MAIN.md, TABLE_EFFICIENCY.md, the all-X figures, and paired_completion_tests.csv. Claims about superiority must be completed only after the GCN-PPO row is evaluated on the identical request IDs.",
        "",
        "## Interpretation boundary",
        "",
        "The experiment supports claims about route establishment, continuity, PAT-related interruption, and path efficiency. It does not support throughput, congestion, queueing, energy, or commercial Starlink hardware claims.",
        "",
        "## Failure accounting",
        "",
    ]
    failures = (df.groupby(["method", "reason"]).size().reset_index(name="count"))
    failures["reason"] = failures.reason.replace({
        "completed with interruption": "session ended with interruption (strict failure)"
    })
    lines += ["| Method | Reason | Count |", "|---|---|---:|"]
    lines += [
        f"| {row.method} | {row.reason} | {int(row.count)} |"
        for row in failures.itertuples(index=False)
    ]
    lines += ["", "## GCN-PPO remaining work", "", "Use the same contact trace, request table, common access rule, initial-topology cache key, terminal/PAT state transition, and CSV schema. Do not resample test requests or tune reward coefficients on these baseline results."]
    (out / "RESULTS_BASELINES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.main_dir / "sessions.csv")
    cost = pd.read_csv(args.cost_greedy_dir / "sessions.csv")
    if set(cost.method) != {"cost_greedy"}:
        raise ValueError("cost-greedy result directory has an unexpected method")
    df = pd.concat([df, cost], ignore_index=True)
    validate(df, args.expected_offsets)
    agg = aggregate(df)
    agg.to_csv(args.output_dir / "aggregate_all_pairs.csv", index=False)
    paired = paired_tests(df, args.output_dir)
    convergence = sampling_convergence(df, args.output_dir, args.expected_offsets)
    stats = contact_stats(args.output_dir)
    sensitivity = load_sensitivity(args.sensitivity_root)
    if not sensitivity.empty:
        sensitivity.to_csv(args.output_dir / "sensitivity_sessions_combined.csv", index=False)
    write_tables(agg, args.output_dir)
    plot_duration(agg, args.output_dir)
    plot_tradeoff(agg, args.output_dir)
    plot_pair_heatmap(df, args.output_dir)
    plot_sensitivity(sensitivity, args.output_dir)
    requests = (
        df[["request_id", "pair", "source", "destination", "request_time_s", "duration_s"]]
        .drop_duplicates("request_id")
        .sort_values("request_id")
    )
    requests.to_csv(args.output_dir / "evaluation_requests.csv", index=False)
    write_report(df, agg, paired, convergence, stats, sensitivity, args.output_dir)
    print(f"validated {len(df)} rows and wrote paper artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
