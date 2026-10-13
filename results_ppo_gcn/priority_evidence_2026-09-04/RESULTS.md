# Priority evidence results (frozen 3,600 requests)

All paper-facing results below use the exact frozen 3,600-request set.  The
balanced 30-request action audits were used only as qualitative 10k diagnostic
gates and are not performance evidence.

## Controlled ablations

| Model | Setup | Strict completion | Availability | Interruption (s) | Reconfigs | New dynamic ISLs | Route hops | Route km |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PPO-GCN main, seed 0 | 3559/3600 (98.86%) | 3400/3600 (94.44%) | 97.98% | 4.13 | 0.893 | 8.811 | 24.23 | 19,239.30 |
| PPO-GCN No-RCT, seed 0 | 3584/3600 (99.56%) | 2816/3600 (78.22%) | 97.08% | 6.87 | 1.072 | 16.608 | 23.70 | 19,354.20 |
| PPO-GCN No-PAT, seed 0 | 3577/3600 (99.36%) | 3011/3600 (83.64%) | 96.13% | 14.69 | 0.828 | 19.149 | 25.04 | 20,325.28 |
| Lifetime baseline | 3600/3600 (100.00%) | 3007/3600 (83.53%) | 98.97% | 4.32 | 0.924 | 9.416 | 16.72 | 14,884.88 |
| ISASR/Remaining baseline | 3580/3600 (99.44%) | 2956/3600 (82.11%) | 98.01% | 5.68 | 1.085 | 4.594 | 18.35 | 14,586.03 |

Removing remaining-contact observations reduces strict completion by 584
requests (16.22 percentage points) relative to the same-seed main model.
Paired discordances are 696 main-only versus 112 No-RCT-only; exact two-sided
McNemar p = 9.40e-104.  No-RCT falls below both required baselines.

Removing only the PAT reward cost reduces strict completion by 389 requests
(10.81 percentage points).  Paired discordances are 493 main-only versus 104
No-PAT-only; exact two-sided McNemar p = 1.71e-61.  It also increases mean new
dynamic-ISL setups from 8.811 to 19.149 and mean interruption from 4.13 s to
14.69 s.  Physical 20 s PAT remains enabled in this ablation; only the explicit
PAT reward term is removed.

## Three-seed main PPO-GCN

| Seed | Strict completion | Completion rate |
|---:|---:|---:|
| 0 | 3400/3600 | 94.44% |
| 1 | 3308/3600 | 91.89% |
| 2 | 3288/3600 | 91.33% |
| Mean ± sample SD | — | 92.56% ± 1.66 percentage points |

Every seed exceeds Lifetime (83.53%, 3007/3600) and ISASR/Remaining (82.11%,
2956/3600).  The across-seed mean exceeds them by 9.03 and 10.44 percentage
points, respectively.  Per-X mean and standard deviation are in
`main_3l_h128_seeds0_1_2_1m/eval_full_test/summary_across_seeds.csv`.

## Constrained feasibility

A physically constrained constructive witness completed 3600/3600 requests,
including 720/720 at each X.  Hence all frozen requests have an executable
completion witness, physical infeasibility is zero on this request set, and
conditional completion uses the same denominator of 3,600.

The witness reaches the exact binary completion upper bound, but it is not an
exhaustive time-expanded optimizer for secondary costs such as route length or
PAT work.  See
`../../results_paper/constrained_oracle_frozen3600_2026-09-04/INTERPRETATION.md`.

## Completion status

- GPU restriction: physical GPU 2 only (`CUDA_VISIBLE_DEVICES=2`)
- Chain exit status: 0
- Finished: 2026-09-05 04:36:35 Asia/Seoul
- No training or evaluation process remains active.
