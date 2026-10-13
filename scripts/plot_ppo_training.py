#!/usr/bin/env python3
"""Plot episode, PPO-update, and validation curves from a training CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def rolling(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return values.copy()
    out = np.convolve(values, np.ones(window) / window, mode="valid")
    return np.concatenate([np.full(window - 1, np.nan), out])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rolling-episodes", type=int, default=100)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.csv.open(newline="", encoding="utf-8")))
    steps = np.asarray([float(row["relay_steps"]) for row in rows])
    reward = np.asarray([float(row["reward"]) for row in rows])
    setup = np.asarray([float(row["setup"]) for row in rows])
    completion = np.asarray([float(row["completion"]) for row in rows])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    w = args.rolling_episodes
    axes[0].plot(steps, rolling(reward, w), label=f"reward ({w}-episode mean)")
    axes[0].set_ylabel("Episode return")
    axes[0].grid(alpha=0.25)
    axes[1].plot(steps, rolling(setup, w), label="training setup")
    axes[1].plot(steps, rolling(completion, w), label="training completion")
    axes[1].set(xlabel="Relay steps", ylabel="Rate", ylim=(-0.03, 1.03))
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "training_episode_curves.png", dpi=200)
    fig.savefig(args.output_dir / "training_episode_curves.pdf")
    plt.close(fig)

    update_rows = [row for row in rows if row.get("ppo_policy_loss", "")]
    update_steps = np.asarray([float(row["relay_steps"]) for row in update_rows])
    fields = [
        ("ppo_policy_loss", "Policy loss"),
        ("ppo_value_loss", "Value loss"),
        ("ppo_entropy", "Entropy"),
        ("ppo_kl", "Approx. KL"),
        ("ppo_clip_fraction", "Clip fraction"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(10, 10), sharex=True)
    for ax, (field, label) in zip(axes.flat, fields):
        ax.plot(update_steps, [float(row[field]) for row in update_rows])
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes.flat[-1].axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("Relay steps")
    fig.tight_layout()
    fig.savefig(args.output_dir / "training_ppo_curves.png", dpi=200)
    fig.savefig(args.output_dir / "training_ppo_curves.pdf")
    plt.close(fig)

    validation = [row for row in rows if row.get("validation_completion_rate", "")]
    vsteps = [float(row["relay_steps"]) for row in validation]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(vsteps, [float(row["validation_setup_rate"]) for row in validation], marker="o", label="setup")
    ax.plot(vsteps, [float(row["validation_completion_rate"]) for row in validation], marker="o", label="completion")
    ax.plot(vsteps, [float(row["validation_availability"]) for row in validation], marker="o", label="availability")
    ax.set(xlabel="Relay steps", ylabel="Validation metric", ylim=(-0.03, 1.03))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "validation_curves.png", dpi=200)
    fig.savefig(args.output_dir / "validation_curves.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
