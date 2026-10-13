"""Results table for the multi-constraint experiment (docs/LLM_ROUTING.md, section 15).

Every method is graded the same way from its logged routes:
* at the request time, the first established route against every condition
  (llm_routing.formulation.conditions_met) and its hop count against the
  author's CP-SAT model (re-solved here with a 60 s limit; status recorded);
* over the session, region violation, time over the delay bound, and the
  new-link budget (llm_routing.compliance), plus completion.
Methods that cannot read text are taken from a constraint run whose world is
unchanged (results_llm/baselines_avoid), restricted to the scenario's requests.

    python scripts/make_multi_table.py --out results_llm/table_multi.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from make_llm_table import NAMES, ORDER  # noqa: E402
from run_llm_routing import build_environment, subset_requests  # noqa: E402

from leo_session import WalkerDeltaConstellation  # noqa: E402
from leo_session.config import DEFAULT_CONFIG  # noqa: E402
from llm_routing.compliance import evaluate  # noqa: E402
from llm_routing.formulation import conditions_met, decision_data, reference_route  # noqa: E402

RUNS = [("Reference CP-SAT (author, true conditions)", "reference_multi"),
        ("Route editing, GPT-4o mini + tools", "gpt-4o-mini_multi_tools"),
        ("LLM writes the CP-SAT model, GPT-4o", "gpt-4o_multi_formulation"),
        ("LLM fills the verified CP-SAT model, GPT-4o", "gpt-4o_multi_extract"),
        ("LLM fills the verified CP-SAT model, GPT-4o mini", "gpt-4o-mini_multi_extract")]
PRICE = {"gpt-4o-mini": (0.15, 0.6), "gpt-4o": (2.5, 10.0)}


def request_data(scenario, cache: Path):
    """Request-time data and the reference optimum (60 s) per scenario request; cached."""
    if cache.exists():
        return {int(k): v for k, v in json.loads(cache.read_text()).items()}
    all_rows, subset = subset_requests("eval")
    _, env = build_environment(all_rows)
    out = {}
    for spec in subset:
        rid = int(spec["request_id"])
        if str(rid) not in scenario:
            continue
        c = scenario[str(rid)]
        t0, x = float(spec["request_time_s"]), float(spec["duration_s"])
        data = decision_data(env.planner, env.planner._graph(t0, env._initial_topology(t0)), spec["source"], spec["destination"], t0, x)
        ref = reference_route(data, c.get("avoid_region"), c.get("max_delay_ms"), c.get("max_new_links"), time_limit_s=60.0)
        out[rid] = {"data": data, "ref_hops": ref.get("hops"), "ref_status": ref["status"]}
    cache.write_text(json.dumps(out))
    return out


def grade(entries, scenario, req, con, label):
    rows = []
    for e in entries:
        rid = int(e["request_id"])
        if str(rid) not in scenario:
            continue
        c = scenario[str(rid)]
        first = e["routes"][0]["satellites"] if e["routes"] else None
        met = conditions_met(req[rid]["data"], first, c) if first else {}
        all_met = bool(first) and all(met.values())
        r = {"Method": label, "request_id": rid, "seed": e.get("seed"),
             "all_met_t0": float(all_met),
             "optimal_t0": float(all_met and len(first) - 1 <= req[rid]["ref_hops"])}
        ev = evaluate(e, c, con)
        if c.get("avoid_region"):
            r["avoid_violation"] = ev.get("avoid_violation", np.nan)
        if c.get("max_delay_ms") is not None:
            r["delay_over"] = ev.get("delay_violation_share", np.nan)
        if c.get("max_new_links") is not None:
            r["budget_violation"] = float(e["dynamic_setups"] > c["max_new_links"])
        r["mean_delay_ms"] = ev.get("mean_delay_ms", np.nan)
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=Path, default=ROOT / "results_llm/scenarios/multi20.json")
    ap.add_argument("--baselines", type=Path, default=ROOT / "results_llm/baselines_avoid")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    scenario = json.loads(args.scenario.read_text())
    req = request_data(scenario, ROOT / "results_llm/scenarios/multi_reference_t0.json")
    con = WalkerDeltaConstellation(DEFAULT_CONFIG)
    ids = {int(k) for k in scenario}
    graded, sessions = [], []
    base = [json.loads(l) for l in open(args.baselines / "routes.jsonl")]
    for m in ORDER:
        ents = [e for e in base if NAMES.get(e["method"], e["method"]) == m and int(e["request_id"]) in ids]
        graded += grade(ents, scenario, req, con, m)
    b = pd.read_csv(args.baselines / "sessions.csv")
    b["Method"] = b.method.map(lambda x: NAMES.get(x, x))
    sessions.append(b[b.request_id.isin(ids)])
    stats = []
    for label, name in RUNS:
        d = ROOT / "results_llm" / name
        if not (d / "sessions.csv").exists():
            continue
        graded += grade([json.loads(l) for l in open(d / "routes.jsonl")], scenario, req, con, label)
        run = pd.read_csv(d / "sessions.csv")
        sessions.append(run[run.request_id.isin(ids)].assign(Method=label))
        model = "gpt-4o-mini" if "mini" in name else ("gpt-4o" if "gpt-4o" in name else None)
        if (d / "codes.jsonl").exists() and model and "extract" in name:
            codes = [c for c in map(json.loads, open(d / "codes.jsonl")) if c["request_id"] in ids]
            cost = sum(c["prompt_tokens"] * PRICE[model][0] + c["completion_tokens"] * PRICE[model][1] for c in codes) / 1e6
            stats.append({"Run": label, "Sessions": len(codes),
                          "Conditions read correctly (%)": 100 * np.mean([all(c["correct"].values()) for c in codes]),
                          "Cost (USD)": cost})
        elif (d / "codes.jsonl").exists() and model:
            codes = [c for c in map(json.loads, open(d / "codes.jsonl")) if c["request_id"] in ids]
            cost = sum(c["prompt_tokens"] * PRICE[model][0] + c["completion_tokens"] * PRICE[model][1] for c in codes) / 1e6
            stats.append({"Run": label, "Sessions": len(codes), "Code accepted (%)": 100 * np.mean([c["accepted"] for c in codes]),
                          "Answers per session": np.mean([len(c["attempts"]) for c in codes]),
                          "Uses CP-SAT (%)": 100 * np.mean([bool(c["attempts"][-1]["code"]) and "cp_model" in (c["attempts"][-1]["code"] or "")
                                                            and ".Solve(" in (c["attempts"][-1]["code"] or "") for c in codes]),
                          "Cost (USD)": cost})
        elif (d / "decisions.jsonl").exists() and model:
            dec = [x for x in map(json.loads, open(d / "decisions.jsonl")) if x["request_id"] in ids]
            cost = sum(x.get("prompt_tokens", 0) * PRICE[model][0] + x.get("completion_tokens", 0) * PRICE[model][1] for x in dec) / 1e6
            stats.append({"Run": label, "Sessions": len({x["request_id"] for x in dec}), "Valid decisions (%)": 100 * np.mean([x["valid"] for x in dec]),
                          "Cost (USD)": cost})
    g = pd.DataFrame(graded)
    s = pd.concat(sessions)
    order = list(dict.fromkeys(g.Method))
    table = pd.DataFrame({
        "n": s.groupby("Method").size(),
        "Compl. (%)": 100 * s.groupby("Method").completion.mean(),
        "All conditions at t0 (%)": 100 * g.groupby("Method").all_met_t0.mean(),
        "Optimal at t0 (%)": 100 * g.groupby("Method").optimal_t0.mean(),
        "Region violation (%)": 100 * g.groupby("Method").avoid_violation.mean(),
        "Delay over bound (%)": 100 * g.groupby("Method").delay_over.mean(),
        "Budget violation (%)": 100 * g.groupby("Method").budget_violation.mean(),
        "Mean delay (ms)": g.groupby("Method").mean_delay_ms.mean(),
        "New ISLs": s.groupby("Method").dynamic_setups.mean(),
    }).loc[order].round(1).reset_index().rename(columns={"index": "Method"})
    status = pd.Series([v["ref_status"] for v in req.values()]).value_counts().to_dict()
    n_kind = {k: sum(1 for v in scenario.values() if k in v) for k in ("avoid_region", "max_delay_ms", "max_new_links")}
    text = (f"Multi-constraint requests: {len(scenario)} (region {n_kind['avoid_region']}, delay {n_kind['max_delay_ms']}, "
            f"new links {n_kind['max_new_links']}; every request also asks for links lasting the session). "
            f"Reference optimum at t0 (CP-SAT, 60 s): {status}. PPO-GCN rows pool 3 seeds.\n\n"
            + table.to_markdown(index=False))
    if stats:
        text += "\n\n" + pd.DataFrame(stats).round(2).to_markdown(index=False)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
