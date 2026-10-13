#!/usr/bin/env python3
"""Evaluate trained PPO-GCN checkpoints on the frozen paired request file."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import fmean, stdev

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from leo_session import (  # noqa: E402
    ContactTrace,
    GroundContactGenerator,
    SessionEnvironment,
    SessionRequest,
    RewardConfig,
    WalkerDeltaConstellation,
)
from ppo_gcn.model import PPOAgent, PPOConfig  # noqa: E402
from ppo_gcn.trainer import collect_episode  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--request-file", type=Path, default=ROOT / "results_paper/paper_artifacts/evaluation_requests.csv")
    parser.add_argument("--trace", type=Path, default=ROOT / "cache/contact_trace_1584_24_11.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results_paper/ppo_gcn")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-rct", action="store_true")
    parser.add_argument(
        "--no-pat",
        action="store_true",
        help=(
            "label/evaluate the no-PAT-reward ablation; physical PAT remains enabled"
        ),
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return center - half, center + half


def main() -> None:
    args = parse_args()
    if args.no_rct and args.no_pat:
        raise SystemExit("evaluate one controlled ablation at a time")
    with args.request_file.open(newline="", encoding="utf-8") as stream:
        specs = list(csv.DictReader(stream))
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("shard-index must be in [0, num-shards)")
    specs = specs[args.shard_index :: args.num_shards]
    if not specs:
        raise SystemExit("request file is empty")
    constellation = WalkerDeltaConstellation()
    trace = ContactTrace.load(args.trace)
    horizon = max(float(row["request_time_s"]) + float(row["duration_s"]) for row in specs)
    horizon += 2.0 * constellation.config.dynamic_setup_s
    ground = GroundContactGenerator(constellation).generate(horizon)
    reward_config = RewardConfig(relay_pat_weight=0.0) if args.no_pat else RewardConfig()
    environment = SessionEnvironment(
        constellation,
        trace,
        ground,
        reward_config=reward_config,
    )
    rows = []
    method = (
        "PPO-GCN-noRCT"
        if args.no_rct
        else "PPO-GCN-noPAT"
        if args.no_pat
        else "PPO-GCN"
    )
    for seed, model_path in enumerate(args.models):
        checkpoint = torch.load(model_path, map_location=args.device)
        saved_config = dict(checkpoint["config"])
        saved_config.setdefault("current_route_state", False)
        agent = PPOAgent(PPOConfig(**saved_config), device=args.device)
        metadata = agent.load(str(model_path))
        actual_seed = int(metadata.get("seed", seed))
        for request_index, spec in enumerate(specs, 1):
            request = SessionRequest(
                str(spec["source"]),
                str(spec["destination"]),
                float(spec["request_time_s"]),
                float(spec["duration_s"]),
            )
            result, _trajectory, _reward = collect_episode(
                environment,
                agent,
                request,
                greedy=True,
                include_remaining_contact=not args.no_rct,
                record=False,
            )
            row = asdict(result)
            row.update(
                method=method,
                seed=actual_seed,
                request_id=int(spec["request_id"]),
                pair=str(spec["pair"]),
                source=str(spec["source"]),
                destination=str(spec["destination"]),
                request_time_s=float(spec["request_time_s"]),
                duration_s=float(spec["duration_s"]),
                availability=(
                    max(0.0, 1.0 - result.interruption_s / request.duration_s)
                    if result.setup_success
                    else 0.0
                ),
            )
            rows.append(row)
            if request_index % 100 == 0 or request_index == len(specs):
                print(
                    f"shard {args.shard_index + 1}/{args.num_shards}: "
                    f"{request_index}/{len(specs)} requests",
                    flush=True,
                )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "sessions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["seed"], row["pair"], row["duration_s"])].append(row)
    summary = []
    for (seed, pair, duration), items in sorted(grouped.items()):
        n = len(items)
        established = [item for item in items if bool(item["setup_success"])]
        setup = sum(bool(item["setup_success"]) for item in items)
        completion = sum(bool(item["completion"]) for item in items)
        completion_low, completion_high = wilson(completion, n)
        summary.append(
            {
                "method": method,
                "seed": seed,
                "pair": pair,
                "duration_s": duration,
                "n": n,
                "setup_success_rate": setup / n,
                "completion_rate": completion / n,
                "completion_ci95_low": completion_low,
                "completion_ci95_high": completion_high,
                "mean_availability": fmean(float(item["availability"]) for item in items),
                "mean_interruption_s": fmean(float(item["interruption_s"]) for item in items),
                "mean_reconfigurations": fmean(float(item["reconfigurations"]) for item in items),
                "mean_dynamic_setups": fmean(float(item["dynamic_setups"]) for item in items),
                "mean_total_pat_time_s": fmean(float(item["total_pat_time_s"]) for item in items),
                "mean_route_hops": fmean(float(item["mean_hops"]) for item in established) if established else math.nan,
                "mean_session_max_hops": fmean(float(item["max_hops"]) for item in established) if established else math.nan,
                "mean_route_path_km": fmean(float(item["mean_path_km"]) for item in established) if established else math.nan,
                "mean_session_max_path_km": fmean(float(item["max_path_km"]) for item in established) if established else math.nan,
                "mean_path_stretch_excess": fmean(float(item["mean_path_stretch_excess"]) for item in established) if established else math.nan,
                "mean_planned_bottleneck_s": fmean(float(item["mean_planned_bottleneck_s"]) for item in established) if established else math.nan,
                "mean_session_min_bottleneck_s": fmean(float(item["min_planned_bottleneck_s"]) for item in established) if established else math.nan,
            }
        )
    with (args.output_dir / "summary_by_seed.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    pooled_groups = defaultdict(list)
    for row in rows:
        pooled_groups[(row["seed"], row["duration_s"])].append(row)
    pooled = []
    for (seed, duration), items in sorted(pooled_groups.items()):
        established = [item for item in items if bool(item["setup_success"])]
        pooled.append(
            {
                "method": method,
                "seed": seed,
                "duration_s": duration,
                "n": len(items),
                "setup_success_rate": fmean(
                    bool(item["setup_success"]) for item in items
                ),
                "completion_rate": fmean(bool(item["completion"]) for item in items),
                "mean_availability": fmean(float(item["availability"]) for item in items),
                "mean_interruption_s": fmean(float(item["interruption_s"]) for item in items),
                "mean_reconfigurations": fmean(float(item["reconfigurations"]) for item in items),
                "mean_dynamic_setups": fmean(float(item["dynamic_setups"]) for item in items),
                "mean_initial_hops": fmean(
                    float(item["initial_hops"])
                    for item in established
                ) if established else math.nan,
                "mean_initial_path_km": fmean(
                    float(item["initial_path_km"])
                    for item in established
                ) if established else math.nan,
                "mean_route_hops": fmean(
                    float(item["mean_hops"]) for item in established
                ) if established else math.nan,
                "mean_session_max_hops": fmean(
                    float(item["max_hops"]) for item in established
                ) if established else math.nan,
                "mean_route_path_km": fmean(
                    float(item["mean_path_km"]) for item in established
                ) if established else math.nan,
                "mean_session_max_path_km": fmean(
                    float(item["max_path_km"]) for item in established
                ) if established else math.nan,
                "mean_path_stretch_excess": fmean(
                    float(item["mean_path_stretch_excess"]) for item in established
                ) if established else math.nan,
                "mean_total_pat_time_s": fmean(
                    float(item["total_pat_time_s"]) for item in items
                ),
                "mean_route_decisions": fmean(
                    float(item["route_decisions"]) for item in established
                ) if established else math.nan,
                "mean_planned_bottleneck_s": fmean(
                    float(item["mean_planned_bottleneck_s"]) for item in established
                ) if established else math.nan,
                "mean_session_min_bottleneck_s": fmean(
                    float(item["min_planned_bottleneck_s"]) for item in established
                ) if established else math.nan,
            }
        )
    with (args.output_dir / "summary_pooled_by_seed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pooled[0]))
        writer.writeheader()
        writer.writerows(pooled)

    seed_groups = defaultdict(list)
    for row in pooled:
        seed_groups[row["duration_s"]].append(row)
    seed_summary = []
    for duration, items in sorted(seed_groups.items()):
        def mean_std(field: str) -> tuple[float, float]:
            values = [float(item[field]) for item in items]
            return fmean(values), stdev(values) if len(values) > 1 else 0.0

        completion_mean, completion_std = mean_std("completion_rate")
        setup_mean, setup_std = mean_std("setup_success_rate")
        availability_mean, availability_std = mean_std("mean_availability")
        seed_summary.append(
            {
                "method": method,
                "duration_s": duration,
                "seeds": len(items),
                "completion_mean": completion_mean,
                "completion_std": completion_std,
                "setup_mean": setup_mean,
                "setup_std": setup_std,
                "availability_mean": availability_mean,
                "availability_std": availability_std,
            }
        )
    with (args.output_dir / "summary_across_seeds.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(seed_summary[0]))
        writer.writeheader()
        writer.writerows(seed_summary)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "models": [str(path.resolve()) for path in args.models],
                "request_file": str(args.request_file.resolve()),
                "trace": str(args.trace.resolve()),
                "method": method,
                "shared_reward": asdict(environment.reward_config),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
