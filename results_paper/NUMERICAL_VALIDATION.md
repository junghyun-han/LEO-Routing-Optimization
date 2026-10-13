# Numerical validation record

These are numerical-resolution checks, not additional physical assumptions.

## Contact trace scan

The adjacent-plane ISL contact trace was independently regenerated with scan
steps of 5.0 s and 2.5 s. Both produced 14,256 contact intervals. After the
bracketed crossings were root-refined, the maximum absolute start/end boundary
difference was `3.95e-7 s` (99th percentile `3.30e-7 s`).

## Relaxed physical upper bound

The relaxed graph was checked on a 2.5 s grid over one 5,738.99 s orbit for all
three ground pairs. None of the 6,888 `(time, pair)` checks was disconnected.
This independently supports the 100% relaxed feasibility result obtained with
the 5 s evaluation sampling.

## Request-time sampling

For every method and X, the orbit-spanning 120-offset subset was compared with
the final 240-offset result. The worst absolute completion-rate difference was
1.25 percentage points (Raw Dijkstra, X=300 s). Detailed values are in
`paper_artifacts/sampling_convergence.csv`.

## Interpretation

These checks support the selected contact scan, upper-bound sampling, and
request-time density. The physical upper bound still ignores PAT and dynamic
terminal occupancy, so it remains a relaxed feasibility ceiling rather than an
exact constrained oracle.
