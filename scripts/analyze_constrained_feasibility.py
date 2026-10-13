#!/usr/bin/env python3
"""Build a conservative feasibility/conditional-completion report.

Completion by any method that obeys the v3 physical environment is a concrete
feasibility witness.  Non-completion is not called physical infeasibility
unless an exhaustive solver supplies a certificate.  This prevents an
omniscient heuristic failure from being mislabeled as a physical impossibility.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


TRUE = {"true", "1", "yes"}
RELAXED = {"physical_upper"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-file", type=Path, required=True)
    parser.add_argument("--constrained-sessions", type=Path, required=True)
    parser.add_argument("--baseline-sessions", type=Path, required=True)
    parser.add_argument("--cost-greedy-sessions", type=Path, required=True)
    parser.add_argument("--ppo-sessions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def key(row: dict[str, str]) -> tuple[object, ...]:
    return (
        int(row["request_id"]),
        row["pair"],
        row["source"],
        row["destination"],
        float(row["request_time_s"]),
        float(row["duration_s"]),
    )


def truth(value: str) -> bool:
    return value.strip().lower() in TRUE


def main() -> None:
    args = parse_args()
    frozen = read(args.request_file)
    frozen_keys = {key(row) for row in frozen}
    if len(frozen) != 3600 or len(frozen_keys) != 3600:
        raise SystemExit("request file is not the exact 3,600-request frozen set")

    sources = {
        "constrained_oracle": read(args.constrained_sessions),
        "baselines": read(args.baseline_sessions),
        "cost_greedy": read(args.cost_greedy_sessions),
        "ppo_gcn": read(args.ppo_sessions),
    }
    methods: dict[str, dict[tuple[object, ...], dict[str, str]]] = {}
    source_verification = {}
    for source_name, rows in sources.items():
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            if row["method"] not in RELAXED:
                grouped[row["method"]].append(row)
        for method, method_rows in grouped.items():
            indexed = {key(row): row for row in method_rows}
            exact = len(method_rows) == 3600 and set(indexed) == frozen_keys
            source_verification[f"{source_name}:{method}"] = {
                "rows": len(method_rows),
                "unique": len(indexed),
                "exact_frozen_match": exact,
            }
            if not exact:
                raise SystemExit(f"{source_name}:{method} is not an exact frozen match")
            methods[method] = indexed

    if "constrained_oracle" not in methods:
        raise SystemExit("constrained_oracle rows are missing")

    # Union of all physically constrained successful executions.  Each member
    # has a concrete route/topology/PAT execution in SessionEnvironment.
    witnessed = {
        request_key
        for request_key in frozen_keys
        if any(truth(rows[request_key]["completion"]) for rows in methods.values())
    }
    oracle_witnessed = {
        request_key
        for request_key, row in methods["constrained_oracle"].items()
        if truth(row["completion"])
    }
    unknown = frozen_keys - witnessed
    rows_out = []
    for method, indexed in sorted(methods.items()):
        completed = {k for k, row in indexed.items() if truth(row["completion"])}
        rows_out.append(
            {
                "method": method,
                "completed_all": len(completed),
                "completion_rate_all": len(completed) / 3600,
                "completed_witness_feasible": len(completed & witnessed),
                "witness_feasible_denominator": len(witnessed),
                "conditional_completion_rate": (
                    len(completed & witnessed) / len(witnessed) if witnessed else 0.0
                ),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "conditional_completion.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)

    fully_certified = len(witnessed) == 3600
    payload = {
        "frozen_requests": 3600,
        "oracle_completed_witnesses": len(oracle_witnessed),
        "union_completed_witnesses": len(witnessed),
        "feasibility_rate_lower_bound": len(witnessed) / 3600,
        "unknown_not_proven_infeasible": len(unknown),
        "binary_feasibility_rate_exact": 1.0 if fully_certified else None,
        "all_requests_feasible_certified": fully_certified,
        "infeasible_requests_certified": 0,
        "interpretation": (
            "Every frozen request has a physically constrained completion witness."
            if fully_certified
            else "Unwitnessed requests remain unknown; they are not labeled physically infeasible."
        ),
        "relaxed_physical_upper_excluded_from_witnesses": True,
        "source_verification": source_verification,
    }
    (args.output_dir / "feasibility_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
