"""Multi-constraint routing by formulation (docs/LLM_ROUTING.md, section 15).

Per session the LLM reads the operator text once and writes ``solve(data)``,
an OR-Tools CP-SAT model of the conditions.  The code is run on the data at
the request time; if it fails to run or returns no rule-abiding route, the
error goes back to the model (at most ``--attempts`` answers in all).  The
final code is then used at every route decision of the simulated session.
``--reference`` runs the author's model (llm_routing.formulation.reference_route)
with the scenario's conditions instead of an LLM.

    python scripts/run_formulation.py --reference --out results_llm/reference_multi
    python scripts/run_formulation.py --provider openai --model gpt-4o --max-usd 2.5 --out results_llm/gpt-4o_multi_formulation
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_llm_routing import build_environment, subset_requests  # noqa: E402

from leo_session import SessionRequest  # noqa: E402
from leo_session.topology import canonical_edge  # noqa: E402
from llm_routing.formulation import DATA_DOC, FormulationPlanner, decision_data, reference_route  # noqa: E402
from llm_routing.planner import QuotaExhausted, _reraise_quota  # noqa: E402
from llm_routing.route_edit import to_route  # noqa: E402
from llm_routing.sandbox import ALLOWED_IMPORTS, run_code  # noqa: E402
from multisession.agent.scenarios import KO_REGION  # noqa: E402
from multisession.conditions import REGIONS  # noqa: E402

# Generic CP-SAT usage notes (prompt version 2, docs/LLM_ROUTING.md change log 8): API pitfalls both models hit
# on the dev requests, and the network size.  They say nothing about how to model this routing problem.
CP_SAT_NOTES = (
    "CP-SAT notes: (1) Coefficients and bounds must be integers; scale real numbers (e.g. km x 10) and round. "
    "(2) Variables and linear expressions cannot be used in Python if/and/or/not; add constraints with model.Add(...), "
    "and express logic with model.AddBoolOr, model.AddBoolAnd, model.AddImplication or .OnlyEnforceIf(literal). "
    "(3) After solver.Solve(model), read values with solver.Value(var), and check the status (cp_model.OPTIMAL or "
    "cp_model.FEASIBLE). (4) The network has about 1,600 satellites and 6,000 to 7,000 usable links; build a model whose "
    "size grows linearly with the number of links so that it builds and solves within the time limit. (5) The route must "
    "come from the CP-SAT solution, not from a separate search. (6) The terminal rule means a route may contain many "
    "inter-plane links, but two inter-plane links may never be consecutive on the route."
)

SYSTEM = (
    "You are an engineer in a LEO satellite network operations team (Walker-Delta, 1584 satellites; each satellite has "
    "in-plane laser links and one inter-plane laser terminal). For a ground-to-ground session, turn the operator's "
    "conditions into an OR-Tools CP-SAT model and solve it.\n\n"
    "Write one function solve(data) that builds the CP-SAT model, solves it, and returns the route as a list of "
    "satellite ids from data[\"ingress\"] to data[\"egress\"], or None if no route meets the conditions. Among routes "
    "that meet every condition, choose one with the fewest hops (links). solve(data) is called at the start of the "
    "session and again at every reroute, each time with fresh data. A route meeting every condition exists at the "
    "start of the session.\n\n"
    f"{DATA_DOC}\n\n"
    f"Use only {', '.join(sorted(ALLOWED_IMPORTS))}; no file or network access. Set "
    "solver.parameters.max_time_in_seconds = 20 and solver.parameters.num_workers = 8. "
    "Reply with a single ```python code block.\n\n"
    f"{CP_SAT_NOTES}"
)


EXTRACT_SYSTEM = (
    "You are an engineer in a LEO satellite network operations team. The team routes ground-to-ground sessions with a "
    "verified CP-SAT model. Read the operator's conditions for one session and fill in the model's parameters. Reply "
    "with one JSON object only: {\"avoid_region\": region key or null, \"max_delay_ms\": number or null, "
    "\"max_new_links\": integer or null, \"links_last_session\": true or false}. avoid_region: a region the route must "
    "never pass over (a key of the region table). max_delay_ms: bound on the one-way delay. max_new_links: most links "
    "that may be switched on (PAT re-pointings) in the whole session. links_last_session: true if every link must last "
    "until the session ends. Use null or false for conditions the operator did not give."
)
EXTRACT_KEYS = ("avoid_region", "max_delay_ms", "max_new_links", "links_last_session")


def extract_conditions(model, text: str, region_names: dict):
    """One call: operator text -> parameters of the reference model (or an error)."""
    user = f"Operator conditions:\n{text}\n\nRegion table (key: Korean name): {json.dumps(region_names, ensure_ascii=False)}"
    msg, usage = model.chat([{"role": "system", "content": EXTRACT_SYSTEM}, {"role": "user", "content": user}])
    found = re.findall(r"\{.*\}", msg["content"] or "", re.S)
    try:
        out = json.loads(found[-1])
        out = {k: out.get(k) for k in EXTRACT_KEYS}
        if out["avoid_region"] is not None and out["avoid_region"] not in region_names:
            return out, f"unknown region {out['avoid_region']}", usage
        return out, "", usage
    except (IndexError, ValueError, AttributeError) as exc:
        return None, f"no JSON parameters: {exc}", usage


def extraction_correct(ext, cond) -> dict:
    truth = {"avoid_region": cond.get("avoid_region"), "max_delay_ms": cond.get("max_delay_ms"),
             "max_new_links": cond.get("max_new_links"), "links_last_session": True}
    if ext is None:
        return {k: False for k in truth}
    def same(a, b):
        if a is None or b is None:
            return a is None and b is None
        return a == b if isinstance(b, (bool, str)) else abs(float(a) - float(b)) < 1e-9
    return {k: same(ext.get(k), v) for k, v in truth.items()}


def extract_code(text: str):
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text or "", re.S)
    return blocks[-1] if blocks else None


def write_code(model, text: str, data: dict, check, attempts: int, timeout_s: float):
    """Ask for solve(data), run it on the request-time data, and feed errors back."""
    example = data["links"][:3]
    user = (f"Operator conditions:\n{text}\n\nRegion names in data: {json.dumps(data['region_names'], ensure_ascii=False)}\n"
            f"At the start of the session data has {len(data['links'])} links, for example {json.dumps(example)}, "
            f"ingress {data['ingress']}, egress {data['egress']}, required_s {data['required_s']:.0f}.")
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    log, usage, code = [], {"prompt_tokens": 0, "completion_tokens": 0}, None
    for _ in range(attempts):
        msg, u = model.chat(messages)
        for k in usage:
            usage[k] += u.get(k, 0)
        messages.append({"role": "assistant", "content": msg["content"]})
        code = extract_code(msg["content"])
        if code is None:
            error, route, secs = "no ```python code block in the answer", None, 0.0
        else:
            route, error, secs = run_code(code, data, timeout_s)
            if not error:
                error = check(route)
        log.append({"code": code, "error": error, "route": route, "seconds": round(secs, 1)})
        if not error:
            break
        messages.append({"role": "user", "content": f"Running solve(data) on the data at the start of the session "
                                                    f"failed:\n{error}\nSend the corrected complete code."})
    return code, log, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--extract", action="store_true", help="LLM extracts the parameters; the reference model solves")
    ap.add_argument("--provider", choices=["openai", "ollama"])
    ap.add_argument("--model")
    ap.add_argument("--scenario", type=Path, default=ROOT / "results_llm/scenarios/multi.json")
    ap.add_argument("--split", choices=["eval", "dev"], default="eval")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=45.0, help="wall-clock limit per run of solve(data), s")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-usd", type=float)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    scenario = json.loads(args.scenario.read_text())
    all_rows, subset = subset_requests(args.split)
    _, env = build_environment(all_rows)
    base = env.planner
    model = None
    if not args.reference:
        from multisession.agent.llm import ChatModel
        model = ChatModel(args.provider, args.model, max_tokens=2048)
    price = {"gpt-4o": (2.5, 10.0), "gpt-4o-mini": (0.15, 0.6)}.get(args.model or "", (0.0, 0.0))
    args.out.mkdir(parents=True, exist_ok=True)
    rows, route_logs, decisions, codes, incomplete = [], [], [], [], None
    specs = [s for s in subset if str(s["request_id"]) in scenario][: args.limit]
    for spec in specs:
        spent = sum(c["prompt_tokens"] * price[0] + c["completion_tokens"] * price[1] for c in codes) / 1e6
        if args.max_usd is not None and spent > args.max_usd:
            incomplete = f"cost cap ${args.max_usd} reached after {len(rows)} requests"
            break
        rid, cond = int(spec["request_id"]), scenario[str(spec["request_id"])]
        t0, x = float(spec["request_time_s"]), float(spec["duration_s"])
        if args.reference:
            def solve(data, c=cond):
                r = reference_route(data, c.get("avoid_region"), c.get("max_delay_ms"), c.get("max_new_links"))
                return r["route"], ("" if r["route"] else f"reference {r['status']}")
            codes.append({"request_id": rid, "prompt_tokens": 0, "completion_tokens": 0})
        elif args.extract:
            try:
                ext, err, usage = extract_conditions(model, cond["text"], {r: KO_REGION[r] for r in sorted(REGIONS)})
            except QuotaExhausted as exc:
                incomplete = f"API credit exhausted after {len(rows)} requests: {exc}"
                break
            except Exception as exc:
                _reraise_quota(exc)
                ext, err, usage = None, f"{type(exc).__name__}: {exc}", {}
            codes.append({"request_id": rid, "extracted": ext, "error": err, "correct": extraction_correct(ext, cond),
                          "prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0)})

            def solve(data, e=ext, err=err):
                if e is None or err:
                    return None, err or "no parameters"
                r = reference_route(data, e["avoid_region"], e["max_delay_ms"], e["max_new_links"],
                                    require_lifetime=bool(e["links_last_session"]))
                return r["route"], ("" if r["route"] else f"model {r['status']}")
        else:
            topo = env._initial_topology(t0)
            graph = base._graph(t0, topo)
            data0 = decision_data(base, graph, spec["source"], spec["destination"], t0, x)
            active = {canonical_edge(*e) for e in topo.dynamic}

            def check(route):
                if route is None:
                    return "solve(data) returned None, but a route meeting every condition exists at the start of the session."
                ok, err = to_route(base, graph, route, spec["source"], spec["destination"], t0, active)
                return "" if ok is not None else f"the returned route breaks a routing rule: {err}"
            try:
                try:
                    code, log, usage = write_code(model, cond["text"], data0, check, args.attempts, args.timeout)
                except QuotaExhausted:
                    raise
                except Exception as exc:
                    _reraise_quota(exc)
                    code, log, usage = None, [{"code": None, "error": f"{type(exc).__name__}: {exc}"}], {}
            except QuotaExhausted as exc:
                incomplete = f"API credit exhausted after {len(rows)} requests: {exc}"
                break
            codes.append({"request_id": rid, "attempts": log, "accepted": not log[-1]["error"],
                          "prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0)})

            def solve(data, code=code):
                if code is None:
                    return None, "no code"
                route, err, _ = run_code(code, data, args.timeout)
                return route, err
        planner = FormulationPlanner(base, solve)
        planner.request_id, planner.condition = rid, cond
        env.planner = planner
        env.set_outages({})
        env.route_log = []
        result = env.run("llm", SessionRequest(spec["source"], spec["destination"], t0, x))
        decisions += planner.decisions
        row = asdict(result)
        row.update(request_id=rid, pair=spec["pair"], source=spec["source"], destination=spec["destination"],
                   request_time_s=t0, duration_s=x)
        row["availability"] = max(0.0, 1.0 - result.interruption_s / x) if result.setup_success else 0.0
        rows.append(row)
        route_logs.append({"request_id": rid, "source": row["source"], "destination": row["destination"],
                           "established_time_s": row["established_time_s"], "duration_s": x,
                           "dynamic_setups": row["dynamic_setups"], "routes": env.route_log})
        print(f"{rid}: first decision met {planner.decisions[0]['met'] if planner.decisions else None}", flush=True)
    env.planner = base
    if rows:
        with open(args.out / "sessions.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for name, items in (("routes.jsonl", route_logs), ("decisions.jsonl", decisions), ("codes.jsonl", codes)):
        with open(args.out / name, "w") as handle:
            for r in items:
                handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    if incomplete:
        (args.out / "INCOMPLETE").write_text(incomplete + "\n")
        print("INCOMPLETE:", incomplete)
    print(f"{len(rows)} requests, {len(decisions)} decisions, completion "
          f"{np.mean([r['completion'] for r in rows]) if rows else float('nan'):.4f} -> {args.out}")


if __name__ == "__main__":
    main()
