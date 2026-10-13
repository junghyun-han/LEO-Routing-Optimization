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
DIGITS = {"Setup (%)": 2, "Compl. (%)": 2, "Compl. (%), all 3,600": 2, "Avail. (%)": 2, "Int. (s)": 2, "Reconf.": 3, "New ISLs": 2, "Path (km)": 0}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-paper", action="store_true")
    ap.add_argument("--llm", nargs="*", type=Path, default=[])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    rows = baseline_rows()
    if args.check_paper:
        table = fmt(summarize(rows)).set_index("Method").loc[ORDER]
        print(table.to_string())
        return
    ids = {int(r["request_id"]) for r in subset_requests("eval")[1]}
    rows = rows[rows.request_id.isin(ids)]
    full = summarize(baseline_rows()).set_index("Method")["Compl. (%)"]
    for path in args.llm:
        llm = pd.read_csv(path / "sessions.csv")
        rows = pd.concat([rows, llm.assign(method=f"LLM ({path.name})")], ignore_index=True)
    table = summarize(rows)
    table["Compl. (%), all 3,600"] = table.Method.map(full)
    order = ORDER + [m for m in table.Method if m not in ORDER]
    table = table.set_index("Method").loc[order].reset_index()
    text = fmt(table).to_markdown(index=False)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
