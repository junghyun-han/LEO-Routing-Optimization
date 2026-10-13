#!/usr/bin/env python3
"""Regenerate the periodic v3 inter-plane contact trace from geometry."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from leo_session import WalkerDeltaConstellation  # noqa: E402
from leo_session.config import DEFAULT_CONFIG  # noqa: E402
from leo_session.contacts import ContactTraceGenerator  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "cache/contact_trace_1584_24_11.npz",
    )
    parser.add_argument("--scan-step-s", type=float, default=5.0)
    args = parser.parse_args()
    if args.scan_step_s <= 0.0:
        raise SystemExit("scan-step-s must be positive")
    config = replace(DEFAULT_CONFIG, contact_scan_step_s=args.scan_step_s)
    constellation = WalkerDeltaConstellation(config)
    trace = ContactTraceGenerator(constellation, config).generate()
    metadata = {
        "generator": "v3",
        "scan_step_s": args.scan_step_s,
        "root_refined": True,
    }
    trace.save(args.output, metadata=metadata)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "period_s": trace.period_s,
                "intervals": len(trace.intervals),
                "metadata": metadata,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
