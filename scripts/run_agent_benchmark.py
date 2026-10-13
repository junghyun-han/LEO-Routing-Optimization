"""Benchmark of the LLM routing agent on operator scenarios.

    python scripts/run_agent_benchmark.py scenarios          # build 30 scenarios on test decision times
    python scripts/run_agent_benchmark.py oracle             # exact and fast solutions with the true conditions
    python scripts/run_agent_benchmark.py run --provider openai --model gpt-4o --modes oneshot agent
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from leo_session.ground import GROUND_STATIONS  # noqa: E402
from multisession.agent.runner import evaluate, run, spec_accuracy  # noqa: E402
from multisession.agent.scenarios import make_scenarios  # noqa: E402
from multisession.agent.tools import Workspace  # noqa: E402
from multisession.build import build_snapshot, decision_times, ground_horizon, load_environment, make_instance  # noqa: E402
from multisession.conditions import Conditions, prepare, satellite_latlon  # noqa: E402
from multisession.exact import solve_exact  # noqa: E402
from multisession.fast import solve_fast  # noqa: E402
from multisession.simulate import replay  # noqa: E402

OUT = ROOT / "results_multisession/agent"


def environment():
    times = [t for t, s in decision_times(ROOT / "results_paper/paper_artifacts/evaluation_requests.csv") if s == "test"]
    env = load_environment(ROOT / "cache/contact_trace_1584_24_11.npz", ground_horizon(times))
    return times, env


def base_instance(env, sc):
    snap = build_snapshot(*env, sc["t0"], GROUND_STATIONS)
    inst = make_instance(snap, sc["per_destination"], sc["duration_s"], sc["antennas"], sc["capacity"])
    return inst, satellite_latlon(env[0], sc["t0"])


def cmd_scenarios(args):
    times, env = environment()
    instances, latlons = [], []
    for i in range(args.n):
        t0 = times[i % len(times)]
        snap = build_snapshot(*env, t0, GROUND_STATIONS)
        instances.append(make_instance(snap, 4, 60.0, 2 if i % 2 == 0 else 4, 1 if (i // 2) % 2 == 0 else 2))
        latlons.append(satellite_latlon(env[0], t0))
    scenarios = make_scenarios(instances, latlons, n=args.n, seed=args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scenarios.json").write_text(json.dumps(scenarios, ensure_ascii=False, indent=1))
    print(f"{len(scenarios)} scenarios -> {OUT / 'scenarios.json'}")


def cmd_oracle(args):
    _, env = environment()
    scenarios = json.loads((OUT / "scenarios.json").read_text())
    path = OUT / "oracle.jsonl"
    done = {json.loads(l)["id"] for l in path.read_text().splitlines()} if path.exists() else set()
    with open(path, "a") as handle:
        for sc in scenarios:
            if sc["id"] in done:
                continue
            inst, ll = base_instance(env, sc)
            truth = Conditions.from_dict(sc["truth"])
            world = prepare(inst, truth, ll)
            exact = solve_exact(inst, time_limit_s=60.0, hops_time_limit_s=10.0, prepared=world)
            row = {"id": sc["id"], "exact_status": exact.status, "exact_stages": exact.stages,
                   "exact": _brief(replay(world, exact.solution, *env)),
                   "fast_truth": _brief(replay(world, solve_fast(world), *env)),
                   "no_conditions": _brief(replay(world, solve_fast(prepare(inst, Conditions(), ll)), *env))}
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            print(sc["id"], exact.status, row["exact"]["completed"], row["fast_truth"]["completed"], flush=True)


def cmd_run(args):
    from multisession.agent.llm import ChatModel
    _, env = environment()
    scenarios = json.loads((OUT / "scenarios.json").read_text())
    model = ChatModel(args.provider, args.model)
    for mode in args.modes:
        path = OUT / "runs" / f"{args.provider}_{args.model.replace(':', '-').replace('/', '-')}_{mode}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        done = {json.loads(l)["id"] for l in path.read_text().splitlines()} if path.exists() else set()
        with open(path, "a") as handle:
            for sc in scenarios[: args.limit]:
                if sc["id"] in done:
                    continue
                inst, ll = base_instance(env, sc)
                ws = Workspace(inst, ll, env)
                out = run(model, sc["text"], ws, mode)
                truth = Conditions.from_dict(sc["truth"])
                row = {"id": sc["id"], "model": model.name, "mode": mode, "categories": sc["categories"],
                       "spec": out["spec"], "accuracy": spec_accuracy(out["spec"], truth),
                       "outcome": _brief(evaluate(out["spec"], truth, inst, ll, env)), "usage": out["usage"],
                       "error": out["error"], "explanation": out["explanation"], "transcript": out["transcript"]}
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                handle.flush()
                print(mode, sc["id"], "all_correct" if row["accuracy"].get("all_correct") else "wrong",
                      row["outcome"]["completed"], out["error"] or "", flush=True)


def _brief(r):
    return {k: v for k, v in r.items() if k != "sessions"}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scenarios")
    s.add_argument("--n", type=int, default=30)
    s.add_argument("--seed", type=int, default=0)
    sub.add_parser("oracle")
    r = sub.add_parser("run")
    r.add_argument("--provider", required=True, choices=["openai", "ollama"])
    r.add_argument("--model", required=True)
    r.add_argument("--modes", nargs="+", default=["oneshot", "agent"])
    r.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    {"scenarios": cmd_scenarios, "oracle": cmd_oracle, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
