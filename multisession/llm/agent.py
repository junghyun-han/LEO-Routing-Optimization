"""LLM heuristic-design loop with solver and simulator in the loop.

Each iteration the LLM sees the problem brief, the best heuristic so far, its
validation score, and feedback; it returns a new module.  The candidate runs in
a child process on the validation instances, is verified, and replaces the best
one only if it is lexicographically better (admitted, then new links, then hops).

Feedback variants (the controlled comparison):
* ``score``: aggregate validation score only (what score-driven heuristic search
  such as FunSearch/EoH uses).
* ``optimal``: the same score plus, for the instances with the largest gap, a
  comparison with the exact CP-SAT solution (per-destination admissions, new
  links and where they are, use of the source antenna links, hops).

Test instances are never touched here.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SYSTEM = ("You are an operations-research engineer who designs fast, correct heuristics "
          "for combinatorial optimization problems in Python.")


# ---------------------------------------------------------------- LLM backends
def claude_code(prompt: str, model: str = "sonnet", timeout_s: float = 600.0) -> Tuple[str, Dict]:
    """Claude through the Claude Code CLI (no tools, no settings, no MCP)."""
    cmd = ["claude", "-p", "--model", model, "--tools", "", "--no-session-persistence",
           "--strict-mcp-config", "--setting-sources", "", "--system-prompt", SYSTEM,
           "--output-format", "json"]
    out = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout_s,
                         cwd=tempfile.gettempdir())
    data = json.loads(out.stdout)
    usage = data.get("usage", {})
    return data.get("result", ""), {
        "input_tokens": usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0)
        + usage.get("cache_read_input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "model": next(iter(data.get("modelUsage", {})), model),
    }


BACKENDS = {"claude-code": claude_code}


# ---------------------------------------------------------------- evaluation
def run_candidate(code: str, keys: List[Dict], per_instance_s: float = 2.0, timeout_s: float = 1800.0) -> Dict:
    with tempfile.TemporaryDirectory() as tmp:
        code_path, keys_path, out_path = Path(tmp, "h.py"), Path(tmp, "keys.json"), Path(tmp, "out.json")
        code_path.write_text(code)
        keys_path.write_text(json.dumps(keys))
        try:
            subprocess.run([sys.executable, "-m", "multisession.llm.candidate", "--code", str(code_path),
                            "--keys", str(keys_path), "--out", str(out_path),
                            "--per-instance-s", str(per_instance_s)],
                           cwd=ROOT, timeout=timeout_s, capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            return {"error": "whole evaluation timed out", "rows": []}
        if not out_path.exists():
            return {"error": "candidate process crashed", "rows": []}
        return json.loads(out_path.read_text())


def score(result: Dict) -> Tuple[int, int, int]:
    rows = result.get("rows", [])
    return (sum(int(r.get("admitted", 0)) for r in rows),
            -sum(int(r.get("new_links", 0)) for r in rows if "error" not in r),
            -sum(int(r.get("total_hops", 0)) for r in rows if "error" not in r))


def summary(result: Dict, reference: Dict[str, Dict]) -> str:
    rows = result.get("rows", [])
    if result.get("error"):
        return f"The module failed: {result['error']}"
    adm = sum(int(r.get("admitted", 0)) for r in rows)
    opt = sum(reference[_key(r)]["exact"]["admitted"] for r in rows)
    greedy = sum(reference[_key(r)]["greedy"]["admitted"] for r in rows)
    at_opt = sum(int(r.get("admitted", 0)) >= reference[_key(r)]["exact"]["admitted"] for r in rows)
    ok = [r for r in rows if "error" not in r]
    lines = [
        f"Validation ({len(rows)} instances): admitted {adm} sessions; the exact optimum admits {opt} "
        f"({100.0 * adm / max(1, opt):.1f}%); the greedy baseline admits {greedy}.",
        f"Instances at the optimum: {at_opt}/{len(rows)}. New links: {sum(r['new_links'] for r in ok)}. "
        f"Mean hops per session: {sum(r['total_hops'] for r in ok) / max(1, sum(r['admitted'] for r in ok)):.2f}. "
        f"Max runtime: {max((r['runtime_s'] for r in ok), default=0):.2f} s.",
    ]
    if result.get("n_errors"):
        lines.append(f"{result['n_errors']} instances failed, e.g.: " + " | ".join(result.get("errors", [])))
    return "\n".join(lines)


def optimal_diff(result: Dict, reference: Dict[str, Dict], instances: Dict[str, object], worst: int = 5) -> str:
    """Compare with the exact solutions on the instances with the largest gap."""
    from ..model import route_edges
    gaps = []
    for r in result.get("rows", []):
        ref = reference[_key(r)]
        gap = ref["exact"]["admitted"] - int(r.get("admitted", 0))
        if gap > 0:
            gaps.append((gap, r))
    if not gaps:
        return "Your heuristic matches the optimum on every validation instance."
    gaps.sort(key=lambda g: -g[0])
    lines = [f"You lose sessions on {len(gaps)} instances. The {min(worst, len(gaps))} largest gaps, "
             f"compared with the exact CP-SAT solution:"]
    for gap, r in gaps[:worst]:
        inst = instances[_key(r)]
        ref = reference[_key(r)]
        sol = ref["solution"]
        dest_of = {s.index: s.destination for s in inst.sessions}
        mine = Counter(dest_of[int(i)] for i in r.get("routes", {}))
        best = Counter(dest_of[int(i)] for i in sol["routes"])
        initial = {tuple(e) for e in inst.snapshot.initial_dynamic}
        opt_new = {tuple(e) for e in sol["final_dynamic"]} - initial
        antennas = {s: st for st in ("new_york", "london", "istanbul", "hanoi") for s, _ in inst.antenna_sats(st)}
        near = Counter(antennas.get(u) or antennas.get(v) or "elsewhere" for u, v in opt_new)
        src = {s for s, _ in inst.antenna_sats("new_york")}

        def first_hops(routes):
            return len({route_edges(p)[0] for p in routes.values() if len(p) > 1 and p[0] in src})

        def hops(routes):
            n = [len(p) - 1 for p in routes.values()]
            return sum(n) / max(1, len(n))

        lines.append(
            f"- A={inst.antennas}, capacity={inst.capacity}, {len(inst.sessions)} sessions, X={inst.sessions[0].duration_s:.0f} s: "
            f"you admit {r.get('admitted', 0)}, optimum {ref['exact']['admitted']}. "
            f"Per destination you/opt: " + ", ".join(f"{d} {mine[d]}/{best[d]}" for d in ("london", "istanbul", "hanoi"))
            + f". Optimum switches on {len(opt_new)} new links ({', '.join(f'{k}: {v}' for k, v in near.items()) or 'none'}); "
            f"you switch on {r.get('new_links', 0)}. Distinct first-hop links out of New York antennas: you "
            f"{first_hops(r.get('routes', {}))}, optimum {first_hops(sol['routes'])}. Mean hops you/opt: "
            f"{hops(r.get('routes', {})):.1f}/{hops(sol['routes']):.1f}.")
    return "\n".join(lines)


def _key(row: Dict) -> str:
    return f"t{float(row['t0']):.3f}_A{int(row['antennas'])}_k{int(row['capacity'])}_K{int(row['per_destination'])}_X{float(row['duration_s']):.0f}"


def load_reference(exact_dir: Path) -> Tuple[List[Dict], Dict[str, Dict]]:
    import csv
    keys, ref = [], {}
    with open(exact_dir / "instances.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            k = _key(row)
            entry = ref.setdefault(k, {})
            entry[row["method"]] = {"admitted": int(row["admitted"]), "new_links": int(row["new_links"]),
                                    "total_hops": int(row["total_hops"])}
            if row["method"] == "exact":
                keys.append({"t0": float(row["t0"]), "antennas": int(row["antennas"]), "capacity": int(row["capacity"]),
                             "per_destination": int(row["per_destination"]), "duration_s": float(row["duration_s"])})
                entry["solution"] = json.loads((exact_dir / "solutions" / f"{k}.json").read_text())
    return keys, ref


def extract_code(text: str) -> Optional[str]:
    blocks = re.findall(r"```python\s*\n(.*?)```", text, re.S)
    return max(blocks, key=len) if blocks else None


# ---------------------------------------------------------------- loop
def design(variant: str, out_dir: Path, exact_dir: Path, iterations: int = 10, backend: str = "claude-code",
           model: str = "sonnet") -> Dict:
    from .candidate import load_instances
    out_dir.mkdir(parents=True, exist_ok=True)
    keys, reference = load_reference(exact_dir)
    instances = {_key(k): inst for k, inst in zip(keys, load_instances(keys))} if variant == "optimal" else {}
    brief = (HERE / "brief.md").read_text()
    best_code = (HERE / "seed.py").read_text()
    best = run_candidate(best_code, keys)
    best_score = score(best)
    history: List[str] = []
    log = out_dir / "log.jsonl"
    (out_dir / "iter_00.py").write_text(best_code)
    with open(log, "a") as handle:
        handle.write(json.dumps({"iteration": 0, "score": best_score, "accepted": True, "seed": True}) + "\n")
    for it in range(1, iterations + 1):
        feedback = summary(best, reference)
        if variant == "optimal":
            feedback += "\n\n" + optimal_diff(best, reference, instances)
        recent = "\n".join(history[-4:]) or "(none yet)"
        prompt = (f"{brief}\n\n## Current best heuristic\n```python\n{best_code}```\n\n## Its validation result\n"
                  f"{feedback}\n\n## Recent attempts (iteration: admitted, accepted?, idea)\n{recent}\n\n"
                  "Write an improved heuristic. Keep what works; change the strategy where sessions are lost.")
        tick = time.perf_counter()
        try:
            text, usage = BACKENDS[backend](prompt, model=model)
        except Exception as exc:  # backend failure: record and continue
            text, usage = "", {"error": repr(exc)}
        llm_s = time.perf_counter() - tick
        code = extract_code(text)
        if code is None:
            result, cand_score = {"error": "no python code block in the reply", "rows": []}, (0, 0, 0)
        else:
            result = run_candidate(code, keys)
            cand_score = score(result)
            (out_dir / f"iter_{it:02d}.py").write_text(code)
        accepted = code is not None and not result.get("error") and cand_score > best_score
        idea = (code or "").strip().splitlines()[0][:160] if code else "(no code)"
        history.append(f"{it}: {cand_score[0]}, {'accepted' if accepted else 'rejected'}, {idea}")
        if accepted:
            best_code, best, best_score = code, result, cand_score
        with open(log, "a") as handle:
            handle.write(json.dumps({"iteration": it, "score": cand_score, "accepted": accepted, "llm_s": llm_s,
                                     "usage": usage, "error": result.get("error"),
                                     "n_errors": result.get("n_errors", 0), "idea": idea}) + "\n")
        print(f"[{variant}] iter {it}: score={cand_score} accepted={accepted} best={best_score}", flush=True)
    (out_dir / "best.py").write_text(best_code)
    return {"best_score": best_score}
