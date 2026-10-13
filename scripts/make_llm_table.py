"""Session-performance table in the WCNC paper's format (Table II) for one request set.

Metric definitions follow the paper: means over requests of setup success,
strict completion, availability, interruption time, reconfigurations, newly
established dynamic ISLs, and path length (over requests with a path).
PPO-GCN pools its three seeds.  ``--check-paper`` recomputes the full 3,600
set and must reproduce Table II.

    python scripts/make_llm_table.py --check-paper
    python scripts/make_llm_table.py --llm results_llm/gpt-4o_none results_llm/gpt-4o-mini_none --out results_llm/table_none.md
"""
from __future__ import annotations

import argparse
import math
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

from run_llm_routing import subset_requests  # noqa: E402

NAMES = {"raw": "Dijkstra", "lifetime": "Lifetime", "prp": "PRP", "remaining": "Adapted ISASR",
         "cost_greedy": "Cost-Greedy", "PPO-GCN": "PPO-GCN"}
ORDER = ["Dijkstra", "PRP", "Cost-Greedy", "Adapted ISASR", "Lifetime", "PPO-GCN"]


def baseline_rows() -> pd.DataFrame:
    base = pd.read_csv(ROOT / "results_paper/main_240_metrics/sessions.csv")
    cg = pd.read_csv(ROOT / "results_paper/cost_greedy_current_reward_frozen3600/sessions.csv")
    ppo = pd.read_csv(ROOT / "results_ppo_gcn/priority_evidence_2026-09-04/main_3l_h128_seeds0_1_2_1m/eval_full_test/sessions.csv")
    frames = [base[base.method.isin(["raw", "lifetime", "prp", "remaining"])], cg, ppo.assign(method="PPO-GCN")]
    out = pd.concat(frames, ignore_index=True)
    out["method"] = out.method.map(NAMES)
    return out


def wilson(k: float, n: int, z: float = 1.96):
    if n == 0:
        return math.nan, math.nan
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (centre - half), 100 * (centre + half)


METRICS = {"Setup (%)": ("setup_success", 100), "Compl. (%)": ("completion", 100), "Avail. (%)": ("availability", 100),
           "Int. (s)": ("interruption_s", 1), "Reconf.": ("reconfigurations", 1), "New ISLs": ("dynamic_setups", 1),
           "Path (km)": ("mean_path_km", 1)}
DIGITS = {"Hit by outage (%)": 1, "Setup (%)": 2, "Compl. (%)": 2, "Compl. (%), all 3,600": 2, "Avail. (%)": 2, "Int. (s)": 2, "Reconf.": 3, "New ISLs": 2, "Path (km)": 0}


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """Means per method; with several seeds, the mean of the per-seed means (as in the paper)."""
    out = []
    for method, g in rows.groupby("method", sort=False):
        k, n = float(g.completion.sum()), len(g)
        lo, hi = wilson(k, n)
        seeds = [s for _, s in g.groupby("seed")] if "seed" in g and g.seed.notna().all() else [g]
        row = {"Method": method, "n": n}
        for name, (col, scale) in METRICS.items():
            row[name] = scale * sum(float(s[col].astype(float).mean()) for s in seeds) / len(seeds)
        if "outage_hits" in g:
            row["Hit by outage (%)"] = 100 * float((g.outage_hits.fillna(0) > 0).mean())
        row["Compl. 95% CI"] = f"{lo:.1f}-{hi:.1f}"
        out.append(row)
    return pd.DataFrame(out)


def fmt(table: pd.DataFrame) -> pd.DataFrame:
    """Round half up like the paper (12.385 -> 12.39)."""
    t = table.copy()
    for name, d in DIGITS.items():
        if name in t:
            t[name] = [f"{Decimal(repr(v)).quantize(Decimal(1).scaleb(-d), ROUND_HALF_UP):,}" if pd.notna(v) else "" for v in t[name]]
    return t


PRICES = {"gpt-4o": (2.50, 10.00), "gpt-4o-mini": (0.15, 0.60)}  # USD per 1M input/output tokens, OpenAI standard tier


def agent_stats(path: Path, dijkstra: pd.DataFrame, scenario=None) -> dict:
    """Decision validity, edits, cost, and a paired exact McNemar test of completion against Dijkstra."""
    import json
    from collections import Counter
    from math import comb
    d = [json.loads(line) for line in (path / "decisions.jsonl").read_text().splitlines()]
    s = pd.read_csv(path / "sessions.csv").set_index("request_id")
    r = dijkstra.loc[s.index]
    only_llm = int((s.completion & ~r.completion.astype(bool)).sum())
    only_dij = int((~s.completion & r.completion.astype(bool)).sum())
    n, k = only_llm + only_dij, min(only_llm, only_dij)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    model = path.name.split("_")[0]
    tin, tout = sum(x["prompt_tokens"] for x in d), sum(x["completion_tokens"] for x in d)
    price = PRICES.get(model, (float("nan"), float("nan")))
    extra = {}
    if scenario and all("satellite" in v for v in scenario.values()):  # maintenance: announced satellite
        first = {}
        for x in d:
            first.setdefault(x["request_id"], x)
        sat = {int(k): v["satellite"] for k, v in scenario.items()}
        extra["First route avoids announced satellite (%)"] = round(
            100 * sum(sat[r] not in x["route"] for r, x in first.items()) / len(first), 1)
        invalid = [x for x in d if not x["valid"] and x.get("proposal")]
        if invalid:
            extra["Invalid proposals that tried to avoid it (%)"] = round(
                100 * sum(sat[x["request_id"]] not in x["proposal"] for x in invalid) / len(invalid), 1)
    if any("tool_calls" in x for x in d):
        extra["Tool calls per decision"] = round(sum(len(x.get("tool_calls", [])) for x in d) / len(d), 1)
    return {"Run": path.name, **extra, "Decisions": len(d), "Valid (%)": round(100 * sum(x["valid"] for x in d) / len(d), 1),
            "Changed": sum(x["changed"] for x in d),
            "Top invalid reasons": "; ".join(f"{e} ({c})" for e, c in Counter(x["error"][:40] for x in d if not x["valid"]).most_common(3)),
            "Completed only by LLM / only by Dijkstra": f"{only_llm} / {only_dij}", "McNemar p": f"{p:.4f}",
            "Tokens in/out per decision": f"{tin // len(d)} / {tout // len(d)}", "Cost (USD)": round(tin / 1e6 * price[0] + tout / 1e6 * price[1], 2),
            "s per decision": round(sum(x["seconds"] for x in d) / len(d), 1)}


CONDITION_KEYS = ("hotspot_region", "avoid_region", "max_delay_ms", "max_new_links")


def scenario_conditions(path):
    if not path:
        return False
    data = __import__("json").loads(path.read_text())
    return any(k in v for v in data.values() for k in CONDITION_KEYS)


def compliance_table(scenario_path, baselines_dir, llm_dirs) -> pd.DataFrame:
    """Per-method compliance from routes.jsonl, followed over the whole session."""
    import json
    from leo_session import WalkerDeltaConstellation
    from leo_session.config import DEFAULT_CONFIG
    from llm_routing.compliance import evaluate
    sys.path.insert(0, str(ROOT))
    con = WalkerDeltaConstellation(DEFAULT_CONFIG)
    cond = {int(k): v for k, v in json.loads(scenario_path.read_text()).items()}
    sources = [(baselines_dir / "routes.jsonl", None)] if baselines_dir else []
    sources += [(d / "routes.jsonl", f"LLM ({d.name})") for d in llm_dirs]
    rows = []
    for path, label in sources:
        for line in path.read_text().splitlines():
            e = json.loads(line)
            c = cond.get(int(e["request_id"]), {})
            r = evaluate(e, c, con)
            r["Method"] = label or NAMES.get(e["method"], e["method"])
            if "max_new_links" in c:
                r["budget_violation"] = float(e["dynamic_setups"] > c["max_new_links"])
            rows.append(r)
    df = pd.DataFrame(rows)
    agg = {"Mean delay (ms)": ("mean_delay_ms", 1)}
    for col, (src, scale) in {"Hotspot time (%)": ("hotspot_time_share", 100), "Hotspot relays": ("hotspot_satellites_mean", 1),
                              "Avoid-region violation (%)": ("avoid_violation", 100), "Avoid-region time (%)": ("avoid_time_share", 100),
                              "Delay over bound (%)": ("delay_violation_share", 100), "Budget violation (%)": ("budget_violation", 100)}.items():
        if src in df:
            agg[col] = (src, scale)
    out = df.groupby("Method", sort=False).agg(**{k: (v[0], "mean") for k, v in agg.items()}).reset_index()
    for k, (_, scale) in agg.items():
        out[k] = out[k] * scale
        DIGITS.setdefault(k, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-paper", action="store_true")
    ap.add_argument("--llm", nargs="*", type=Path, default=[])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--baselines", type=Path, help="rerun baselines/PPO-GCN under a scenario (run_constraint_baselines.py)")
    ap.add_argument("--scenario", type=Path, help="scenario file, for avoidance statistics")
    args = ap.parse_args()
    rows = baseline_rows()
    if args.check_paper:
        table = fmt(summarize(rows)).set_index("Method").loc[ORDER]
        print(table.to_string())
        return
    ids = {int(r["request_id"]) for r in subset_requests("eval")[1]}
    if args.baselines:
        rows = pd.read_csv(args.baselines / "sessions.csv")
        rows["method"] = rows.method.map(lambda m: NAMES.get(m, m))
    rows = rows[rows.request_id.isin(ids)]
    full = None if args.baselines else summarize(baseline_rows()).set_index("Method")["Compl. (%)"]
    for path in args.llm:
        llm = pd.read_csv(path / "sessions.csv")
        rows = pd.concat([rows, llm.assign(method=f"LLM ({path.name})")], ignore_index=True)
    table = summarize(rows)
    if full is not None:
        table["Compl. (%), all 3,600"] = table.Method.map(full)
    if scenario_conditions(args.scenario):
        comp = compliance_table(args.scenario, args.baselines, args.llm)
        table = table.merge(comp, on="Method", how="left")
    order = ORDER + [m for m in table.Method if m not in ORDER]
    table = table.set_index("Method").loc[order].reset_index()
    text = fmt(table).to_markdown(index=False)
    if args.llm:
        base = pd.read_csv((args.baselines or ROOT / "results_paper/main_240_metrics") / "sessions.csv")
        dij = base[base.method == "raw"].set_index("request_id")
        scenario = __import__("json").loads(args.scenario.read_text()) if args.scenario else None
        stats = pd.DataFrame([agent_stats(p, dij, scenario) for p in args.llm])
        text += "\n\nLLM decisions\n\n" + stats.to_markdown(index=False)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
