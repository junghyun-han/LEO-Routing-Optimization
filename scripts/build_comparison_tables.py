#!/usr/bin/env python3
"""Build factual frozen-test tables for the v3 handover.

All rows are joined by the exact frozen request specification.  The script
fails instead of silently comparing different requests.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean


ROOT = Path(__file__).resolve().parents[1]
METHOD_ORDER = (
    "raw",
    "lifetime",
    "prp",
    "remaining",
    "cost_greedy",
    "PPO-GCN",
    "physical_upper",
)
METHOD_LABEL = {
    "raw": "Raw Dijkstra",
    "lifetime": "Lifetime",
    "prp": "PRP",
    "remaining": "ISASR/Remaining",
    "cost_greedy": "Current-reward Cost-Greedy",
    "PPO-GCN": "PPO-GCN 3L/128 1M",
    "physical_upper": "Relaxed physical upper bound",
}
BOOL_TRUE = {"true", "1", "yes"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-sessions",
        type=Path,
        default=ROOT / "results_paper/main_240_metrics/sessions.csv",
    )
    parser.add_argument(
        "--cost-greedy-sessions",
        type=Path,
        default=ROOT
        / "results_paper/cost_greedy_current_reward_frozen3600/sessions.csv",
    )
    parser.add_argument(
        "--ppo-sessions",
        type=Path,
        default=ROOT
        / "results_ppo_gcn/long_run_2026-09-04/gcn3_h128_v2_exact_route_reward_seed0_1m_mb64_gpu2/eval_full_test/sessions.csv",
    )
    parser.add_argument(
        "--request-file",
        type=Path,
        default=ROOT / "results_paper/paper_artifacts/evaluation_requests.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "results_ppo_gcn/long_run_2026-09-04/gcn3_h128_v2_exact_route_reward_seed0_1m_mb64_gpu2/eval_full_test/comparison_tables",
    )
    parser.add_argument("--ppo-label", default="PPO-GCN")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def request_key(row: dict[str, str]) -> tuple[object, ...]:
    return (
        int(row["request_id"]),
        row["pair"],
        row["source"],
        row["destination"],
        float(row["request_time_s"]),
        float(row["duration_s"]),
    )


def truth(value: str) -> bool:
    return value.strip().lower() in BOOL_TRUE


def number(row: dict[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError):
        return math.nan
    return value


def mean_finite(rows: list[dict[str, str]], field: str) -> float:
    values = [number(row, field) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    return fmean(values) if values else math.nan


def aggregate(method: str, rows: list[dict[str, str]], duration: float | None) -> dict[str, object]:
    selected = rows if duration is None else [row for row in rows if float(row["duration_s"]) == duration]
    established = [row for row in selected if truth(row["setup_success"])]
    n = len(selected)
    setup_n = sum(truth(row["setup_success"]) for row in selected)
    completion_n = sum(truth(row["completion"]) for row in selected)
    dynamic = mean_finite(selected, "dynamic_setups")
    return {
        "method": method,
        "method_label": METHOD_LABEL[method],
        "duration_s": "all" if duration is None else int(duration),
        "n": n,
        "setup_n": setup_n,
        "setup_rate": setup_n / n,
        "completion_n": completion_n,
        "completion_rate": completion_n / n,
        "mean_availability": mean_finite(selected, "availability"),
        "mean_interruption_s": mean_finite(selected, "interruption_s"),
        "mean_reconfigurations": mean_finite(selected, "reconfigurations"),
        "mean_dynamic_isl_setups": dynamic,
        "mean_pat_link_end_participations": 2.0 * dynamic if math.isfinite(dynamic) else math.nan,
        "mean_total_pat_time_s": mean_finite(selected, "total_pat_time_s"),
        "mean_route_hops_established": mean_finite(established, "mean_hops"),
        "mean_route_path_km_established": mean_finite(established, "mean_path_km"),
        "mean_path_stretch_excess_established": mean_finite(established, "mean_path_stretch_excess"),
        "mean_planned_bottleneck_s_established": mean_finite(established, "mean_planned_bottleneck_s"),
    }


def fmt(value: object, digits: int = 2) -> str:
    if isinstance(value, (float, int)) and math.isfinite(float(value)):
        return f"{float(value):.{digits}f}"
    return "N/A"


def markdown_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| Method | N | Setup | Strict completion | Availability | Interruption (s) | Reconfigs | New dynamic ISLs | Route hops | Route km |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        setup = f"{row['setup_n']}/{row['n']} ({100*float(row['setup_rate']):.2f}%)"
        completion = f"{row['completion_n']}/{row['n']} ({100*float(row['completion_rate']):.2f}%)"
        lines.append(
            "| {method_label} | {n} | {setup} | {completion} | {availability}% | {interruption} | {reconfigs} | {dynamic} | {hops} | {path} |".format(
                method_label=row["method_label"],
                n=row["n"],
                setup=setup,
                completion=completion,
                availability=fmt(100.0 * float(row["mean_availability"])),
                interruption=fmt(row["mean_interruption_s"]),
                reconfigs=fmt(row["mean_reconfigurations"], 3),
                dynamic=fmt(row["mean_dynamic_isl_setups"], 3),
                hops=fmt(row["mean_route_hops_established"]),
                path=fmt(row["mean_route_path_km_established"]),
            )
        )
    return "\n".join(lines)


def exact_mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    METHOD_LABEL["PPO-GCN"] = args.ppo_label
    frozen = read_csv(args.request_file)
    if len(frozen) != 3600 or len({request_key(row) for row in frozen}) != 3600:
        raise SystemExit("frozen request file must contain exactly 3,600 unique requests")
    frozen_keys = {request_key(row) for row in frozen}

    baseline_rows = read_csv(args.baseline_sessions)
    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in baseline_rows:
        if row["method"] in METHOD_ORDER:
            by_method[row["method"]].append(row)
    by_method["cost_greedy"] = read_csv(args.cost_greedy_sessions)
    by_method["PPO-GCN"] = read_csv(args.ppo_sessions)

    verification = {}
    indexed: dict[str, dict[tuple[object, ...], dict[str, str]]] = {}
    for method in METHOD_ORDER:
        rows = by_method[method]
        keys = [request_key(row) for row in rows]
        exact = len(rows) == 3600 and len(set(keys)) == 3600 and set(keys) == frozen_keys
        verification[method] = {
            "rows": len(rows),
            "unique_request_specs": len(set(keys)),
            "exact_frozen_request_match": exact,
        }
        if not exact:
            raise SystemExit(f"{method} is not an exact 3,600-request match")
        indexed[method] = {request_key(row): row for row in rows}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    durations = sorted({float(row["duration_s"]) for row in frozen})
    all_aggregates = [aggregate(method, by_method[method], None) for method in METHOD_ORDER]
    by_x = [aggregate(method, by_method[method], duration) for duration in durations for method in METHOD_ORDER]
    write_csv(args.output_dir / "overall_all_metrics.csv", all_aggregates)
    write_csv(args.output_dir / "by_x_all_metrics.csv", by_x)

    (args.output_dir / "OVERALL.md").write_text(
        "# Frozen 3,600-request overall results\n\n" + markdown_table(all_aggregates) + "\n",
        encoding="utf-8",
    )
    by_x_lines = ["# Frozen test results by requested duration X", ""]
    for duration in durations:
        rows = [row for row in by_x if row["duration_s"] == int(duration)]
        write_csv(args.output_dir / f"X_{int(duration)}s_all_metrics.csv", rows)
        text = f"# X = {int(duration)} seconds\n\n" + markdown_table(rows) + "\n"
        (args.output_dir / f"X_{int(duration)}s.md").write_text(text, encoding="utf-8")
        by_x_lines.extend([f"## X = {int(duration)} seconds", "", markdown_table(rows), ""])
    (args.output_dir / "RESULTS_BY_X.md").write_text("\n".join(by_x_lines), encoding="utf-8")

    paired = []
    ppo = indexed["PPO-GCN"]
    for method in METHOD_ORDER:
        if method == "PPO-GCN":
            continue
        ppo_only = baseline_only = both = neither = 0
        for key in frozen_keys:
            p = truth(ppo[key]["completion"])
            b = truth(indexed[method][key]["completion"])
            if p and b:
                both += 1
            elif p:
                ppo_only += 1
            elif b:
                baseline_only += 1
            else:
                neither += 1
        paired.append(
            {
                "comparison": f"PPO-GCN vs {METHOD_LABEL[method]}",
                "both_complete": both,
                "ppo_only_complete": ppo_only,
                "other_only_complete": baseline_only,
                "neither_complete": neither,
                "paired_completion_delta_pp": 100.0 * (ppo_only - baseline_only) / 3600,
                "exact_mcnemar_two_sided_p": exact_mcnemar(ppo_only, baseline_only),
            }
        )
    write_csv(args.output_dir / "paired_completion_comparisons.csv", paired)

    reasons = []
    for method in METHOD_ORDER:
        counts = Counter(row["reason"] for row in by_method[method])
        for reason, count in sorted(counts.items()):
            reasons.append({"method": method, "method_label": METHOD_LABEL[method], "reason": reason, "count": count})
    write_csv(args.output_dir / "outcome_reason_counts.csv", reasons)

    verification_payload = {
        "frozen_request_file": str(args.request_file.resolve()),
        "frozen_rows": len(frozen),
        "durations_s": durations,
        "requests_per_duration": {str(int(x)): sum(float(row["duration_s"]) == x for row in frozen) for x in durations},
        "methods": verification,
        "source_files": {
            "baselines_and_upper": str(args.baseline_sessions.resolve()),
            "current_reward_cost_greedy": str(args.cost_greedy_sessions.resolve()),
            "ppo_gcn_current": str(args.ppo_sessions.resolve()),
        },
    }
    (args.output_dir / "REQUEST_SET_VERIFICATION.json").write_text(
        json.dumps(verification_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(verification_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
