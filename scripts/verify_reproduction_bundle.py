#!/usr/bin/env python3
"""Verify the retained v3 inputs, checkpoints, and frozen-test outputs."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ppo_gcn.model import PPOAgent, PPOConfig  # noqa: E402


EXPECTED_FILES = {
    "cache/contact_trace_1584_24_11.npz": "e6baf0a0ad85a17533e2a0b06ed3ec959b85ad10152069377c7de92cf3df1637",
    "results_paper/paper_artifacts/evaluation_requests.csv": "acf51a3b3de8313a8f729e6f9188f71d34784f5e853fc3814bd4639d853951d4",
}

CHECKPOINTS = {
    "main_seed0": (
        "results_ppo_gcn/long_run_2026-09-04/gcn3_h128_v2_exact_route_reward_seed0_1m_mb64_gpu2/best_seed0.pt",
        "466875a5d8041b23510bb8595c87c7c9742c98300ffdb537bb780f41e26e1ef8",
    ),
    "main_seed1": (
        "results_ppo_gcn/priority_evidence_2026-09-04/main_3l_h128_seeds1_2_1m_mb64_gpu2/best_seed1.pt",
        "146bdd4eea24f4813155130dd9735af16f6a60d7d0dc8b91dfaca2375ea2b53e",
    ),
    "main_seed2": (
        "results_ppo_gcn/priority_evidence_2026-09-04/main_3l_h128_seeds1_2_1m_mb64_gpu2/best_seed2.pt",
        "f5502a3c37ffce00337a765c69b184cab7c6d82371d2a22c624a955f429e9905",
    ),
    "no_rct_seed0": (
        "results_ppo_gcn/priority_evidence_2026-09-04/no_rct_3l_h128_seed0_1m_mb64_gpu2/best_seed0.pt",
        "4166e151594719b9e163c7ff3d9e8a60fcc756f3ed419c1b0d7bf198bbeab62d",
    ),
    "no_pat_seed0": (
        "results_ppo_gcn/priority_evidence_2026-09-04/no_pat_3l_h128_seed0_1m_mb64_gpu2/best_seed0.pt",
        "a9fc5938bccaa83c089411d66dc282c5b8070b3ffb5899e9b0c56c4814f95702",
    ),
    "gcn5_seed0_secondary": (
        "results_ppo_gcn/long_run_2026-09-04/gcn5_h128_v2_exact_route_reward_seed0_1m_mb64_gpu2/best_seed0.pt",
        "1e5aca8f4fcbd8432b2bf690ce0add20b09f90bd6ee930a51bbbaaa6cf711ccc",
    ),
}

RESULTS = {
    "main_seed0": (
        "results_ppo_gcn/long_run_2026-09-04/gcn3_h128_v2_exact_route_reward_seed0_1m_mb64_gpu2/eval_full_test/sessions.csv",
        3400,
    ),
    "no_rct_seed0": (
        "results_ppo_gcn/priority_evidence_2026-09-04/no_rct_3l_h128_seed0_1m_mb64_gpu2/eval_full_test/sessions.csv",
        2816,
    ),
    "no_pat_seed0": (
        "results_ppo_gcn/priority_evidence_2026-09-04/no_pat_3l_h128_seed0_1m_mb64_gpu2/eval_full_test/sessions.csv",
        3011,
    ),
    "gcn5_seed0_secondary": (
        "results_ppo_gcn/long_run_2026-09-04/gcn5_h128_v2_exact_route_reward_seed0_1m_mb64_gpu2/eval_full_test/sessions.csv",
        3237,
    ),
    "constrained_witness": (
        "results_paper/constrained_oracle_frozen3600_2026-09-04/sessions.csv",
        3600,
    ),
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def completed(rows: list[dict[str, str]]) -> int:
    return sum(row["completion"].strip().lower() == "true" for row in rows)


def main() -> None:
    report: dict[str, object] = {"root": str(ROOT), "files": {}, "checkpoints": {}, "results": {}}
    for relative, expected in EXPECTED_FILES.items():
        actual = digest(ROOT / relative)
        if actual != expected:
            raise SystemExit(f"checksum mismatch: {relative}")
        report["files"][relative] = {"sha256": actual, "ok": True}

    for label, (relative, expected) in CHECKPOINTS.items():
        path = ROOT / relative
        actual = digest(path)
        if actual != expected:
            raise SystemExit(f"checkpoint checksum mismatch: {label}")
        raw = torch.load(path, map_location="cpu")
        config = dict(raw["config"])
        config.setdefault("current_route_state", False)
        agent = PPOAgent(PPOConfig(**config), device="cpu")
        metadata = agent.load(str(path))
        report["checkpoints"][label] = {
            "path": relative,
            "sha256": actual,
            "seed": int(metadata["seed"]),
            "relay_steps": int(metadata["relay_steps"]),
            "gcn_layers": int(config["gcn_layers"]),
            "hidden_dim": int(config["hidden_dim"]),
            "ok": True,
        }

    for label, (relative, expected_completion) in RESULTS.items():
        rows = read_rows(ROOT / relative)
        identities = {int(row["request_id"]) for row in rows}
        actual_completion = completed(rows)
        if len(rows) != 3600 or len(identities) != 3600 or actual_completion != expected_completion:
            raise SystemExit(f"frozen result mismatch: {label}")
        report["results"][label] = {
            "rows": len(rows),
            "unique_request_ids": len(identities),
            "completed": actual_completion,
            "ok": True,
        }

    multi_path = ROOT / "results_ppo_gcn/priority_evidence_2026-09-04/main_3l_h128_seeds0_1_2_1m/eval_full_test/sessions.csv"
    multi = read_rows(multi_path)
    expected_by_seed = {0: 3400, 1: 3308, 2: 3288}
    actual_by_seed = {}
    for seed, expected in expected_by_seed.items():
        rows = [row for row in multi if int(row["seed"]) == seed]
        if len(rows) != 3600 or len({int(row["request_id"]) for row in rows}) != 3600:
            raise SystemExit(f"multi-seed frozen identity mismatch: seed {seed}")
        actual = completed(rows)
        if actual != expected:
            raise SystemExit(f"multi-seed completion mismatch: seed {seed}")
        actual_by_seed[str(seed)] = actual
    report["results"]["main_three_seed"] = {
        "rows": len(multi),
        "completion_by_seed": actual_by_seed,
        "ok": True,
    }
    report["status"] = "PASS"
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
