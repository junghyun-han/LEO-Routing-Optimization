"""LLM heuristic design on validation instances: score feedback vs optimal-solution feedback.

Runs are interleaved (score r1, optimal r1, score r2, ...) so time-dependent
effects hit both variants equally.  Test instances are not used here.

    .venv/bin/python scripts/run_llm_design.py --reps 3 --iterations 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multisession.llm.agent import design  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["score", "optimal"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--exact", type=Path, default=ROOT / "results_multisession/exact_validation")
    ap.add_argument("--out", type=Path, default=ROOT / "results_multisession/llm_design")
    args = ap.parse_args()
    for rep in range(1, args.reps + 1):
        for variant in args.variants:
            run_dir = args.out / f"{variant}_r{rep}"
            if (run_dir / "best.py").exists():
                continue  # finished earlier
            design(variant, run_dir, args.exact, iterations=args.iterations, model=args.model)


if __name__ == "__main__":
    main()
