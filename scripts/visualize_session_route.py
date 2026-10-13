#!/usr/bin/env python3
"""Visualize one actual v3 initial topology and planned ground-to-ground route."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from leo_session import ContactTrace, GroundContactGenerator, SessionEnvironment, WalkerDeltaConstellation  # noqa: E402
from leo_session.ground import GROUND_STATIONS  # noqa: E402
from leo_session.topology import build_initial_topology, canonical_edge  # noqa: E402


COLORS = {
    "persistent": "#d97706",
    "initial_dynamic": "#0891b2",
    "new_dynamic": "#db2777",
    "gsl": "#16a34a",
    "satellite": "#64748b",
    "earth": "#dbeafe",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=float, default=300.0)
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("--source", default="new_york")
    parser.add_argument("--destination", default="london")
    parser.add_argument("--method", choices=["raw", "lifetime", "prp", "remaining"], default="lifetime")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results_paper/paper_artifacts/fig_initial_topology_route.png",
    )
    return parser.parse_args()


def equal_3d(ax, radius: float) -> None:
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_zlim(-radius, radius)
    ax.set_box_aspect((1, 1, 1))


def edge_style(edge, topology):
    edge = canonical_edge(*edge)
    if edge in topology.persistent:
        return "persistent", "-"
    if edge in topology.dynamic:
        return "initial_dynamic", "-"
    return "new_dynamic", "--"


def main() -> None:
    args = parse_args()
    constellation = WalkerDeltaConstellation()
    trace = ContactTrace.load(ROOT / "cache/contact_trace_1584_24_11.npz")
    ground = GroundContactGenerator(constellation).generate(
        constellation.period_s + args.duration + 40.0
    )
    environment = SessionEnvironment(constellation, trace, ground)
    topology = build_initial_topology(constellation, trace, args.time)
    route = environment.planner.plan(
        args.method,
        args.source,
        args.destination,
        args.time,
        args.duration,
        topology,
    )
    if route is None:
        raise SystemExit("no route for the requested visualization case")

    positions = constellation.positions(args.time)
    source_pos = GROUND_STATIONS[args.source].eci(
        args.time, constellation.config.earth_radius_km
    )
    destination_pos = GROUND_STATIONS[args.destination].eci(
        args.time, constellation.config.earth_radius_km
    )

    fig = plt.figure(figsize=(13.0, 6.1))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    chain = fig.add_subplot(1, 2, 2)

    # Earth surface.
    theta = np.linspace(0, 2 * np.pi, 72)
    phi = np.linspace(0, np.pi, 36)
    x = constellation.config.earth_radius_km * np.outer(np.cos(theta), np.sin(phi))
    y = constellation.config.earth_radius_km * np.outer(np.sin(theta), np.sin(phi))
    z = constellation.config.earth_radius_km * np.outer(np.ones_like(theta), np.cos(phi))
    ax.plot_surface(x, y, z, color=COLORS["earth"], alpha=0.18, linewidth=0)

    # All satellites and the already-ACTIVE topology.
    ax.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2],
        s=2.0, color=COLORS["satellite"], alpha=0.34, depthshade=False,
    )
    for u, v in topology.persistent:
        p = positions[[u, v]]
        ax.plot(*p.T, color=COLORS["persistent"], alpha=0.035, linewidth=0.35)
    for u, v in topology.dynamic:
        p = positions[[u, v]]
        ax.plot(*p.T, color=COLORS["initial_dynamic"], alpha=0.16, linewidth=0.55)

    # Selected session route.
    route_nodes = list(route.satellites)
    for u, v in zip(route_nodes[:-1], route_nodes[1:]):
        kind, linestyle = edge_style((u, v), topology)
        p = positions[[u, v]]
        ax.plot(*p.T, color=COLORS[kind], linewidth=3.0, linestyle=linestyle)
    ax.plot(
        *np.vstack([source_pos, positions[route.ingress]]).T,
        color=COLORS["gsl"], linewidth=2.5,
    )
    ax.plot(
        *np.vstack([positions[route.egress], destination_pos]).T,
        color=COLORS["gsl"], linewidth=2.5,
    )
    ax.scatter(
        positions[route_nodes, 0], positions[route_nodes, 1], positions[route_nodes, 2],
        s=22, color="#111827", depthshade=False,
    )
    ax.scatter(*source_pos, s=55, marker="^", color=COLORS["gsl"], depthshade=False)
    ax.scatter(*destination_pos, s=55, marker="^", color=COLORS["gsl"], depthshade=False)

    radius = constellation.config.orbital_radius_km + 250.0
    equal_3d(ax, radius)
    ax.view_init(elev=22, azim=-62)
    ax.set_axis_off()
    ax.set_title(
        f"Initial ACTIVE topology and {args.method} route\n"
        f"t₀={args.time:g} s, X={args.duration:g} s",
        pad=8,
    )

    proxies = [
        Line2D([0], [0], color=COLORS["persistent"], lw=2, label="Persistent intra-plane ISL"),
        Line2D([0], [0], color=COLORS["initial_dynamic"], lw=2, label="Initial ACTIVE dynamic ISL"),
        Line2D([0], [0], color=COLORS["new_dynamic"], lw=2.5, ls="--", label="New dynamic ISL for route"),
        Line2D([0], [0], color=COLORS["gsl"], lw=2.5, label="Ground access link"),
    ]
    ax.legend(handles=proxies, loc="lower left", frameon=False, fontsize=8)

    # Route-chain panel: exact node ids and edge roles are readable here.
    labels = [GROUND_STATIONS[args.source].name] + [f"S{node}" for node in route_nodes] + [GROUND_STATIONS[args.destination].name]
    xs = np.arange(len(labels), dtype=float)
    for i in range(len(labels) - 1):
        if i == 0 or i == len(labels) - 2:
            kind, linestyle = "gsl", "-"
        else:
            kind, linestyle = edge_style((route_nodes[i - 1], route_nodes[i]), topology)
        chain.plot(xs[i:i+2], [0, 0], color=COLORS[kind], linewidth=4, linestyle=linestyle, solid_capstyle="round")

    chain.scatter(xs[1:-1], np.zeros(len(route_nodes)), s=70, color="#111827", zorder=3)
    chain.scatter([xs[0], xs[-1]], [0, 0], s=100, marker="^", color=COLORS["gsl"], zorder=3)
    for i, label in enumerate(labels):
        chain.text(xs[i], 0.12, label, ha="center", va="bottom", fontsize=8, rotation=35 if 0 < i < len(labels)-1 else 0)
    for i, node in enumerate(route_nodes, 1):
        address = constellation.address(node)
        chain.text(xs[i], -0.12, f"P{address.plane}, Q{address.slot}", ha="center", va="top", fontsize=7, color="#475569")

    chain.set_xlim(-0.5, len(labels) - 0.5)
    chain.set_ylim(-0.5, 0.5)
    chain.set_yticks([])
    chain.set_xticks([])
    chain.spines[["top", "right", "left", "bottom"]].set_visible(False)
    chain.set_title(
        f"Ground-to-ground route chain\n"
        f"{route.hops} hops incl. GSL · {len(route.new_dynamic_edges)} new dynamic ISLs · "
        f"initial topology: {len(topology.dynamic)} dynamic links",
        pad=14,
    )

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=260, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(
        f"wrote {args.output}; persistent={len(topology.persistent)}, "
        f"initial_dynamic={len(topology.dynamic)}, route={route.satellites}, "
        f"new_dynamic={route.new_dynamic_edges}"
    )


if __name__ == "__main__":
    main()
