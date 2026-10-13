#!/usr/bin/env python3
"""Qualitative fixed-request action audit for a PPO-GCN checkpoint.

This is a diagnostic gate, not paper-performance evaluation.  It records the
actual next-satellite choices on a balanced 30-request slice (two requests per
pair and duration) and reports whether the policy constructs routes without a
degenerate masked-action or 128-action failure mode.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from leo_session import (  # noqa: E402
    ContactTrace,
    GroundContactGenerator,
    RewardConfig,
    SessionEnvironment,
    SessionRequest,
    WalkerDeltaConstellation,
)
from ppo_gcn.model import PPOAgent, PPOConfig  # noqa: E402
from ppo_gcn.observation import build_policy_observation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--request-file",
        type=Path,
        default=ROOT / "results_paper/paper_artifacts/evaluation_requests.csv",
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=ROOT / "cache/contact_trace_1584_24_11.npz",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-rct", action="store_true")
    parser.add_argument("--no-pat", action="store_true")
    return parser.parse_args()


def balanced_slice(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, float], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["pair"], float(row["duration_s"]))].append(row)
    selected: list[dict[str, str]] = []
    for key in sorted(grouped):
        group = grouped[key]
        for fraction in (1.0 / 3.0, 2.0 / 3.0):
            index = min(len(group) - 1, int(fraction * len(group)))
            selected.append(group[index])
    return selected


class AuditCollector:
    def __init__(self, environment: SessionEnvironment, agent: PPOAgent, *, no_rct: bool):
        self.environment = environment
        self.agent = agent
        self.no_rct = no_rct
        self.route_id = 0
        self._route_static_observation = None
        self.actions: list[dict[str, object]] = []

    def record_route_outcome(self, route_id: int, **outcome: object) -> None:
        return None

    def __call__(self, decision):
        if len(decision.state.satellites) == 1:
            self.route_id += 1
            self._route_static_observation = None
        observation = build_policy_observation(
            self.environment.planner,
            decision,
            include_remaining_contact=not self.no_rct,
            include_route_history=self.agent.config.route_history,
            include_current_route_state=self.agent.config.current_route_state,
            static_observation=self._route_static_observation,
        )
        if self._route_static_observation is None:
            self._route_static_observation = observation
        action, log_prob, value = self.agent.act(observation, greedy=True)
        by_neighbor = {item.neighbor: item for item in decision.candidates}
        chosen = by_neighbor[action]
        ranked = sorted(
            decision.candidates,
            key=lambda item: (
                -item.immediate_reward(self.environment.reward_config), item.neighbor
            ),
        )
        reward_rank = next(i for i, item in enumerate(ranked, 1) if item.neighbor == action)
        breakdown = chosen.reward_breakdown(self.environment.reward_config)
        self.actions.append(
            {
                "route_id": self.route_id,
                "hop_index": len(decision.state.satellites) - 1,
                "current": decision.state.current,
                "chosen": action,
                "egress": decision.destination_satellite,
                "candidate_count": len(decision.candidates),
                "chosen_local_reward_rank": reward_rank,
                "chosen_local_reward": breakdown.reward,
                "normalized_progress": chosen.normalized_progress,
                "normalized_link_length": chosen.normalized_link_length,
                "lifetime_shortfall": chosen.lifetime_shortfall,
                "normalized_pat_cost": chosen.normalized_pat_cost,
                "is_dynamic": int(chosen.is_dynamic),
                "is_active": int(chosen.is_active),
                "is_new_dynamic": int(chosen.is_new_dynamic),
                "remaining_s": chosen.remaining_s,
                "required_s": decision.required_s,
                "log_prob": log_prob,
                "value": value,
            }
        )
        return action


def main() -> None:
    args = parse_args()
    if args.no_rct and args.no_pat:
        raise SystemExit("audit one controlled ablation at a time")
    with args.request_file.open(newline="", encoding="utf-8") as stream:
        specs = balanced_slice(list(csv.DictReader(stream)))
    constellation = WalkerDeltaConstellation()
    trace = ContactTrace.load(args.trace)
    horizon = max(
        float(row["request_time_s"]) + float(row["duration_s"]) for row in specs
    ) + 2.0 * constellation.config.dynamic_setup_s
    ground = GroundContactGenerator(constellation).generate(horizon)
    reward = RewardConfig(relay_pat_weight=0.0) if args.no_pat else RewardConfig()
    environment = SessionEnvironment(
        constellation, trace, ground, reward_config=reward
    )
    checkpoint = torch.load(args.model, map_location=args.device)
    saved_config = dict(checkpoint["config"])
    saved_config.setdefault("current_route_state", False)
    agent = PPOAgent(PPOConfig(**saved_config), device=args.device)
    metadata = agent.load(str(args.model))

    action_rows: list[dict[str, object]] = []
    request_rows: list[dict[str, object]] = []
    for spec in specs:
        request = SessionRequest(
            spec["source"],
            spec["destination"],
            float(spec["request_time_s"]),
            float(spec["duration_s"]),
        )
        collector = AuditCollector(environment, agent, no_rct=args.no_rct)
        result = environment.run_policy(request, collector)
        for row in collector.actions:
            row.update(
                request_id=int(spec["request_id"]),
                pair=spec["pair"],
                duration_s=float(spec["duration_s"]),
            )
            action_rows.append(row)
        request_rows.append(
            {
                "request_id": int(spec["request_id"]),
                "pair": spec["pair"],
                "duration_s": float(spec["duration_s"]),
                **asdict(result),
                "relay_actions": len(collector.actions),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("actions.csv", action_rows), ("requests.csv", request_rows)):
        with (args.output_dir / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    setup = sum(bool(row["setup_success"]) for row in request_rows)
    completion = sum(bool(row["completion"]) for row in request_rows)
    capped = sum(str(row["reason"]) == "no route at request time" and int(row["relay_actions"]) >= 128 for row in request_rows)
    finite_policy_values = all(
        math.isfinite(float(row["log_prob"])) and math.isfinite(float(row["value"]))
        for row in action_rows
    )
    # This gate checks implementation/action health, not whether an ablation is
    # competitive.  An intentionally weakened ablation may have low completion.
    gate_pass = bool(action_rows) and setup > 0 and capped < len(request_rows) and finite_policy_values
    summary = {
        "scope": "balanced 30-request qualitative diagnostic; not paper evidence",
        "model": str(args.model.resolve()),
        "seed": int(metadata.get("seed", -1)),
        "requests": len(request_rows),
        "actions": len(action_rows),
        "setup_successes": setup,
        "completions": completion,
        "max_action_failures": capped,
        "distinct_chosen_satellites": len({int(row["chosen"]) for row in action_rows}),
        "finite_policy_values": finite_policy_values,
        "gate_pass": gate_pass,
        "gate_definition": "nonempty actions, at least one setup, not all requests hit 128-action failure, finite policy outputs",
        "no_rct": args.no_rct,
        "no_pat_reward": args.no_pat,
        "physical_pat_enabled": True,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if not gate_pass:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
