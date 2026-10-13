"""Run the five baselines and PPO-GCN on the 180-request set, optionally under a constraint scenario.

These methods cannot read operator text; a scenario only changes the world
(e.g. unannounced satellite outages), exactly as it does for the LLM runs.
Without ``--scenario`` the rows must equal the frozen test-set rows.

    python scripts/run_constraint_baselines.py --out results_llm/baselines_none
    python scripts/run_constraint_baselines.py --scenario results_llm/scenarios/maintenance.json --out results_llm/baselines_maintenance
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_llm_routing import build_environment, subset_requests  # noqa: E402

from leo_session import SessionRequest  # noqa: E402

BASELINES = ("raw", "prp", "cost_greedy", "remaining", "lifetime")


def ppo_agents(device: str):
    import torch
    from ppo_gcn.model import PPOAgent, PPOConfig
    agents = []
    with open(ROOT / "docs/CHECKPOINT_INDEX.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["role"] != "main":
                continue
            checkpoint = torch.load(ROOT / row["path"], map_location=device)
            config = dict(checkpoint["config"])
            config.setdefault("current_route_state", False)
            agent = PPOAgent(PPOConfig(**config), device=device)
            agent.load(str(ROOT / row["path"]))
            agents.append((int(row["seed"]), agent))
    return agents


def apply_scenario(env, scenario, request_id):
    entry = scenario.get(str(request_id), {}) if scenario else {}
    env.set_outages({int(k): float(v) for k, v in entry.get("outages", {}).items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=Path)
    ap.add_argument("--methods", nargs="+", default=list(BASELINES) + ["ppo"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    scenario = json.loads(args.scenario.read_text()) if args.scenario else None
    all_rows, subset = subset_requests("eval")
    _, env = build_environment(all_rows)
    agents = ppo_agents(args.device) if "ppo" in args.methods else []
    rows, route_logs = [], []
    for spec in subset[: args.limit]:
        rid = int(spec["request_id"])
        request = SessionRequest(spec["source"], spec["destination"], float(spec["request_time_s"]), float(spec["duration_s"]))
        runs = [(m, None, lambda m=m: env.run(m, request)) for m in args.methods if m != "ppo"]
        if agents:
            from ppo_gcn.trainer import collect_episode
            runs += [("PPO-GCN", seed, lambda a=agent: collect_episode(env, a, request, greedy=True, record=False)[0])
                     for seed, agent in agents]
        for method, seed, go in runs:
            apply_scenario(env, scenario, rid)
            env.route_log = []
            result = go()
            row = asdict(result)
            row.update(method=method, seed=seed, request_id=rid, pair=spec["pair"], source=spec["source"],
                       destination=spec["destination"], request_time_s=float(spec["request_time_s"]),
                       duration_s=float(spec["duration_s"]))
            row["availability"] = max(0.0, 1.0 - result.interruption_s / row["duration_s"]) if result.setup_success else 0.0
            rows.append(row)
            route_logs.append({"request_id": rid, "method": method, "seed": seed, "source": row["source"],
                               "destination": row["destination"], "established_time_s": row["established_time_s"],
                               "duration_s": row["duration_s"], "dynamic_setups": row["dynamic_setups"],
                               "routes": env.route_log})
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "sessions.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(args.out / "routes.jsonl", "w") as handle:
        for r in route_logs:
            handle.write(json.dumps(r) + "\n")
    print(f"{len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
