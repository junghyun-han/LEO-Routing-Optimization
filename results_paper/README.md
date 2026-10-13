# Frozen paper baseline package

This directory contains the retained frozen CPU baselines, current-reward
Cost-Greedy result, frozen request specification, and sensitivity results.

## Result map

- `main_240_metrics/`: frozen baseline run with route and PAT metrics
- `cost_greedy_current_reward_frozen3600/`: current four-term Cost-Greedy on all 3,600 requests
- `sensitivity/setup_{0,5,10,20}/`: PAT/setup sensitivity raw results
- `paper_artifacts/evaluation_requests.csv`: frozen 3,600-request GCN-PPO test set
- `constrained_oracle_frozen3600_2026-09-04/`: physically constrained constructive
  completion witnesses and conditional-completion analysis
- `ROUTE_COVERAGE.md`: constellation, ingress, topology, and route coverage

## Reproduce the final CPU run

Run from the repository root:

```bash
.venv/bin/python -m pytest -q tests
.venv/bin/python scripts/run_cpu_baselines.py \
  --sessions-per-pair 240 \
  --durations 20 60 180 300 600 \
  --output-dir results_paper/main_240_metrics \
  --methods raw lifetime prp remaining physical_upper

.venv/bin/python scripts/run_cpu_baselines.py \
  --request-file results_paper/paper_artifacts/evaluation_requests.csv \
  --output-dir results_paper/cost_greedy_current_reward_frozen3600 \
  --methods cost_greedy

.venv/bin/python scripts/analyze_paper_baselines.py \
  --main-dir results_paper/main_240_metrics \
  --cost-greedy-dir results_paper/cost_greedy_current_reward_frozen3600 \
  --sensitivity-root results_paper/sensitivity \
  --output-dir results_paper/paper_artifacts \
  --expected-offsets 240
```

The analysis command fails on missing/duplicate method-request rows, an
unexpected method set, an incorrect request count, missing core metrics, or an
online method exceeding the declared relaxed physical upper bound.

## Frozen comparison target

All paper comparisons must use the exact 3,600 IDs in
`evaluation_requests.csv`. The retained PPO-GCN 3L/128 1M result reaches
94.44% strict completion overall (3,400/3,600), versus Lifetime 83.53% and
ISASR/Remaining 82.11% under the same requests and environment. Seeds 1 and 2
reach 91.89% and 91.33%; the three-seed mean is 92.56% with sample SD 1.66
percentage points. The constrained witness completes 3,600/3,600, so every
frozen request has an executable completion witness under the modeled constraints.
