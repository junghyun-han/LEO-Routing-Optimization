# v3 system design

## Meaning of episode time zero

Episode `t=0` is the arrival of a new session request, not the birth of an empty
satellite network. Let `t_req` denote the physical time inside the periodic
contact trace. Immediately before the request:

- the forward and backward intra-plane ISLs are `ACTIVE`;
- a session-independent matching has already activated at most one dynamic
  inter-plane ISL per satellite;
- other physically visible inter-plane contacts are only `FEASIBLE`;
- an initially ACTIVE dynamic link must have been feasible throughout
  `[t_req - 20 s, t_req]`, so its pre-request acquisition is physically possible.

The initial matching never uses the episode source, destination, requested
duration `X`, or future remaining contact time.

## Fixed terminal model

Each satellite has two persistent intra-plane terminals and one reconfigurable
inter-plane terminal. Consequently, the maximum ACTIVE degree is three. A
dynamic terminal changes peers using break-before-make:

1. release its old dynamic link;
2. start acquisition of the new feasible peer;
3. wait 20 seconds;
4. activate the new link only if contact and both terminals remain feasible.

When one route needs several new dynamic ISLs, their acquisitions run in
parallel because the route search guarantees distinct dynamic terminals. Thus
the route setup cost is 20 s, not `20 s x number of links`. This is a
centralized network-level abstraction and is stated as a model assumption.

This is the professor-suggested research model, not a claimed commercial
Starlink terminal specification.

## Physical time

Route-construction operations do not advance physical time. Physical time only
advances to setup completion, contact start/end, reconfiguration, or session
completion events. The 5-second contact scan is only a numerical root bracket;
contact boundaries are refined to continuous times.

## Session

A request is `(s, d, X)`, where `X` is known at request arrival. Setup succeeds
when an ACTIVE end-to-end path first exists. Setup delay is the elapsed physical
time from request to that event. Completion succeeds only when end-to-end
connectivity remains uninterrupted for `X` seconds after establishment. The
specific path may change if ACTIVE connectivity remains continuous.

## Ground access

At every route decision, each ground endpoint uses the currently visible GSL
whose contact ends latest. This deterministic rule is shared by every method;
ISL algorithms do not get to choose different ingress/egress satellites. It
keeps access handover in the experiment while isolating the claimed effect of
inter-satellite relay selection. The rule is an explicit study design, not a
commercial gateway policy.

## Initial dynamic matching

For every adjacent-plane interface, even slot indices on the left plane are
matched to odd slot indices on the right plane. Interfaces are therefore
balanced and terminals cannot conflict. The assignment first maximizes the
number of physically valid links and then minimizes total current link length.
This deterministic rule is a model initialization, not a literature-derived
Starlink operations policy.

## Baselines to be implemented

- `Dijkstra-Raw`: minimum-hop path on the currently feasible graph, with no
  remaining-contact filtering.
- `Dijkstra-Lifetime`: remove links unable to cover setup plus known `X`, then
  run the same minimum-hop Dijkstra.
- `PRP`: reproduce the probability-based lifetime filter with `p=0.9`, clearly
  labeling the adaptation from random call holding time to this session model.
- `RemainingContact`: published stability/activeness-style edge cost rather
  than an uncited hand-tuned reward. The adapted ISASR cost uses one unit per
  hop, `setup/remaining-contact` as its stability term, and one extra setup
  unit for a feasible but inactive dynamic edge at the 20 s reference PAT.
  In sensitivity runs this activeness term scales linearly as `setup/20` and
  becomes zero when setup is zero. No coefficient is fitted to the evaluation
  results.
- `Local Cost-Greedy`: sequential one-relay policy using the exact reward
  shared with PPO-GCN. It maximizes egress progress minus the current candidate
  link's lifetime shortfall and new-PAT cost. It does not run BFS/Dijkstra or
  receive a residual-hop target.
- `Physical-Upper`: all physically feasible contacts are made available
  instantly, ignoring terminal occupancy and PAT. This is a deliberately
  relaxed feasibility ceiling, not an achievable baseline and not the final
  constrained oracle.

The current code does **not** claim to contain an exhaustive time-expanded
optimizer.  It does contain a constrained constructive feasibility witness
(`constrained_oracle`) that uses the same 2+1 terminals, initial topology, 20 s
setup, and break-before-make rules.  Because it actually completes all 3,600
frozen requests, binary feasibility is certified for this test set.  It must
not be claimed to optimize path length, PAT work, or another secondary cost.
