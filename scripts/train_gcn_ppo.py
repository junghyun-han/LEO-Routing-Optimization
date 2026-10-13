#!/usr/bin/env python3
"""Train v3 PPO-GCN. This script is prepared but is not run by baseline jobs."""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from leo_session import (  # noqa: E402
    GROUND_PAIRS,
    ContactTrace,
    GroundContactGenerator,
    RewardConfig,
    SessionEnvironment,
    WalkerDeltaConstellation,
)
from ppo_gcn.model import PPOAgent, PPOConfig  # noqa: E402
from ppo_gcn.trainer import TrainingConfig, train_agent  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=ROOT / "cache/contact_trace_1584_24_11.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results_ppo_gcn")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--total-relay-steps", type=int, default=1_000_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--gcn-layers", type=int, default=3)
    parser.add_argument("--architecture", choices=("gcn", "mlp"), default="gcn")
    parser.add_argument("--learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--value-clip", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--validation-offsets", type=int, default=0)
    parser.add_argument("--snapshot-interval-updates", type=int, default=45)
    parser.add_argument("--checkpoint-interval-updates", type=int, default=45)
    parser.add_argument("--no-rct", action="store_true", help="observation ablation only; reward and environment remain unchanged")
    parser.add_argument(
        "--no-pat",
        action="store_true",
        help=(
            "reward ablation only: set the relay PAT-cost weight to zero; "
            "the 20 s physical setup process and all observation features remain unchanged"
        ),
    )
    parser.add_argument(
        "--route-history",
        action="store_true",
        help="include ordered route history, previous satellite, and hop progress",
    )
    parser.add_argument(
        "--no-current-route-state",
        action="store_true",
        help="ablate current-session-route reuse and break-before-make inputs",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    args = parse_args()
    if args.no_rct and args.no_pat:
        raise SystemExit("train one controlled ablation at a time")
    if (
        args.total_relay_steps < 1
        or args.rollout_steps < 1
        or args.minibatch_size < 1
        or args.validation_offsets < 0
        or args.snapshot_interval_updates < 1
        or args.checkpoint_interval_updates < 1
    ):
        raise SystemExit("training/batch sizes must be positive and validation offsets non-negative")
    device = resolve_device(args.device)
    constellation = WalkerDeltaConstellation()
    trace = ContactTrace.load(args.trace)
    horizon = constellation.period_s + 600.0 + 2.0 * constellation.config.dynamic_setup_s
    ground = GroundContactGenerator(constellation).generate(horizon)
    reward_config = RewardConfig(relay_pat_weight=0.0) if args.no_pat else RewardConfig()
    environment = SessionEnvironment(
        constellation,
        trace,
        ground,
        reward_config=reward_config,
    )
    ppo_config = PPOConfig(
        architecture=args.architecture,
        hidden_dim=args.hidden_dim,
        gcn_layers=args.gcn_layers,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        value_coefficient=args.value_coefficient,
        entropy_coefficient=args.entropy_coefficient,
        value_clip=args.value_clip,
        max_grad_norm=args.max_grad_norm,
        update_epochs=args.update_epochs,
        target_kl=args.target_kl,
        rollout_steps=args.rollout_steps,
        minibatch_size=args.minibatch_size,
        route_history=args.route_history,
        current_route_state=not args.no_current_route_state,
    )
    training_config = TrainingConfig(
        total_relay_steps=args.total_relay_steps,
        validation_offsets=args.validation_offsets,
        snapshot_interval_updates=args.snapshot_interval_updates,
        checkpoint_interval_updates=args.checkpoint_interval_updates,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for seed in args.seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        agent = PPOAgent(ppo_config, device=device)
        summary = train_agent(
            environment,
            agent,
            GROUND_PAIRS,
            args.output_dir,
            seed,
            training_config,
            include_remaining_contact=not args.no_rct,
        )
        summaries.append(summary)
    metadata = {
        "device": device,
        "trace": str(args.trace.resolve()),
        "seeds": args.seeds,
        "ppo_config": asdict(ppo_config),
        "training_config": asdict(training_config),
        "include_remaining_contact": not args.no_rct,
        "pat_reward_enabled": not args.no_pat,
        "ablation": (
            "no_rct" if args.no_rct else "no_pat" if args.no_pat else "none"
        ),
        "shared_reward": asdict(environment.reward_config),
        "route_progress_boundary_correction": False,
        "summaries": summaries,
    }
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
