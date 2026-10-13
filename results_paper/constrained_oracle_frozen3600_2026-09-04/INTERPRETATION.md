# Constrained feasibility result

## Result

- Frozen request set: exactly 3,600 unique requests
- Constructive completion witnesses: 3,600/3,600 (100%)
- Per-duration witnesses: 720/720 for each of X = 20, 60, 180, 300, and 600 s
- Setup success: 3,600/3,600
- Interruption: 0 s for every request
- Unknown/unproven requests: 0

Every execution used the same `SessionEnvironment` as the baselines and
PPO-GCN: the same session-independent initial ACTIVE topology, two persistent
intra-plane ISLs plus one dynamic terminal per satellite, 20 s setup for every
new dynamic ISL, break-before-make, deterministic GSL access, and strict X-second
uninterrupted completion.

Therefore all 3,600 frozen requests are physically feasible under the modeled
constraints.  Conditional completion has the same denominator as unconditional
completion on this test set.  In particular, PPO-GCN's 200 non-completions are
policy failures rather than physically impossible requests.

## Naming limitation

The implemented planner is a constructive feasibility witness.  At each route
event it uses exact contact-end information and globally selects a path with the
largest contact bottleneck, then lets `SessionEnvironment` execute and validate
the route transition.  It completed all requests, which certifies the binary
feasibility result and reaches the 100% binary upper bound.

It is not an exhaustive time-expanded optimizer over every possible future route
sequence.  It must not be claimed to optimize path length, PAT work, or another
secondary cost, and a failure from this planner would not by itself prove
physical infeasibility.

## Diagnostic implication

The witness used only dynamic ISLs already ACTIVE in the session-independent
initial topology: mean new dynamic setups = 0 and mean elapsed PAT time = 0 s.
It traded a somewhat longer global route for stability.  This is direct evidence
that the current PPO-GCN's remaining gap is route selection, not infeasibility.
