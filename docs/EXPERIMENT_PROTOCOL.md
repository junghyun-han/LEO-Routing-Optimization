# Frozen CPU baseline experiment protocol

This protocol is fixed before the paper-scale baseline run. It is not chosen
from the final performance ranking.

## Scope

The simulator has the same abstraction level as the v2 study and the cited
on-demand-routing literature: moving satellites, geometry-derived time-varying
contacts, and network-level route/setup events. It does not model queues,
traffic load, packet transmission, or a detailed optical link budget.

## Main experiment

- constellation: Walker Delta 1584/24/11, altitude 550 km, inclination 53 deg
- LISL range: 1,500 km
- GSL range: 1,000 km
- topology: two persistent intra-plane plus one dynamic inter-plane terminal
- PAT/setup: deterministic 20 s, break-before-make
- ground pairs:
  - New York-London (short)
  - New York-Istanbul (medium)
  - New York-Hanoi (long)
- session durations: `X = {20, 60, 180, 300, 600} s`
- request times: 240 deterministic midpoints of equal bins over one orbit
- paired design: every method receives the same pair, `t0`, `X`, contact trace,
  and deterministic initial-topology rule
- common ground access: latest-ending currently visible GSL, selected before
  ISL routing and shared by all methods

There are 3,600 requests and 18,000 method-runs including the relaxed physical
upper bound.

The representative main table is fixed at `X=300 s`, the first tested duration
that exceeds the maximum observed single GSL contact. The duration-sweep figure
reports every X and is the primary guard against a single favorable operating
point.

## Why these X values

- 20 s: the reference PAT interval and the old study's short horizon
- 60 s: short-session condition from the PRP literature's one-minute case
- 180 s: the published PRP `p=0.9` example threshold
- 300 s and 600 s: require one or more access-link handovers because the
  generated 1,000 km GSL contact duration has median about 213 s and maximum
  about 239 s

`X` is a controlled factor. No real traffic-duration distribution is claimed.

## Baselines

1. Raw minimum-hop Dijkstra
2. Known-X lifetime-filtered minimum-hop Dijkstra
3. PRP (`p=0.9`, `T_tr=180 s`; raw routing on subsequent handover)
4. Adapted ISASR remaining-contact/activeness cost
5. Strong sequential Cost-Greedy using the exact relay reward shared with PPO
6. Relaxed physical feasibility upper bound

The upper bound ignores PAT and terminal occupancy and is used only for
physical-feasibility screening. It is not compared as an achievable method for
path length, setup delay, or reconfiguration count.

## Sensitivity experiment

- setup delay: `{0, 5, 10, 20} s`
- duration: `X = 180 and 300 s`
- request times: 60 deterministic mid-bin offsets per pair
- all four achievable CPU baselines; the setup-independent upper bound is not
  redundantly repeated

These setup values are controlled stress conditions bounded by the 20 s main
model. They are not claimed as measurements of four specific terminals.

## Metrics

- relaxed physical feasibility rate
- path-setup success rate and Wilson 95% confidence interval
- strict session-completion rate and Wilson 95% confidence interval
- availability during the requested interval
- setup delay
- interruption time
- route reconfiguration count
- number of dynamically established ISLs
- initial hop count and physical path length
- initial route bottleneck remaining contact time

## Acceptance checks

- all topology tests pass;
- no method receives a different request or initial topology;
- initial maximum active degree is at most three;
- all completed routes obey the one-dynamic-terminal route constraint;
- no result row is missing or duplicated;
- 120- and 240-offset aggregate conclusions are compared;
- failure reasons are reported, not discarded;
- the full CSV, metadata, tables, and plotting code are retained.

Protocol amendment: the first 120-offset run was retained, but its
orbit-spanning 60-offset convergence check showed a worst-case 7.2 percentage
point completion-rate difference. The final run was therefore expanded to 240
offsets before adding or evaluating GCN-PPO. No method, X value, or model rule
was changed in response to the 120-offset performance ranking.

Second protocol amendment: before GCN-PPO integration, Cost-Greedy was added
to validate the shared sequential relay action semantics. Its shaping weights
were selected on a separate 300-request midpoint grid with zero overlap with
`paper_artifacts/evaluation_requests.csv`, then frozen and run once on those
3,600 requests. Relay actions compose a centralized route at one event time and
do not advance the simulation clock. See `COST_GREEDY.md` and
`results_paper/cost_greedy_current_reward_frozen3600/` for the retained final
result. Superseded sweep output was removed from the live final workspace.

Withdrawn amendments: an intermediate reward used an X-feasible BFS
residual-hop potential. Audit showed that this gives PPO a global shortest-path
answer inside the reward and action observation, so the design and all of its
test results were withdrawn before PPO training and removed from the live final
workspace. They must not be cited as paper results.

Current reward amendment: the shared relay reward is purely local,
`rho_d - 0.05*l - 3*q_X - 0.15*q_P`. It contains egress progress, normalized
link length, the current candidate link's PAT-adjusted lifetime shortfall, and
its new-PAT cost. The original lifetime/PAT sweep and later link-term diagnostic
were diagnostic-only and have been removed. The current four-term Cost-Greedy
was rerun on all 3,600 frozen requests; its X=300 strict completion is 75.14%
and overall completion is 80.64%. See `REWARD_DESIGN.md`, `COST_GREEDY.md`, and the current
combined tables under the current successful 1M run's
`eval_full_test/comparison_tables/` directory.
