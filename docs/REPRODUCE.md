# Exact reproduction guide

Run every command below from the repository root.

## 1. Runtime environment

The retained results were produced on:

- Ubuntu 24.04.4 LTS, x86-64
- Python 3.14.7
- NVIDIA RTX A6000, driver 580.173.02
- PyTorch 2.13.0
- NumPy 2.5.2, SciPy 1.18.1, pandas 3.0.5
- NetworkX 3.6.1, Matplotlib 3.11.1, pytest 9.1.1

The exact Python package versions are in `requirements.txt`.  A virtual
environment is deliberately not included because it contains machine-specific
absolute paths and CUDA binaries.

```bash
cd project
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
export PYTHONPATH="$PWD"
```

On the original multi-GPU server, every v3 GPU command must expose physical GPU
2 only.  CUDA then reports it inside Python as `cuda:0`.

```bash
export CUDA_VISIBLE_DEVICES=2
.venv/bin/python -c 'import torch; assert torch.cuda.device_count() == 1; print(torch.cuda.get_device_name(0))'
```

## 2. Verify the supplied package before rerunning

From the extracted package root, verify the ZIP-independent internal manifest:

```bash
sha256sum -c CHECKSUMS_SHA256.txt
```

From the repository root, verify the exact frozen requests, cached contact trace, all six
retained checkpoints, their metadata, and every headline result:

```bash
.venv/bin/python scripts/verify_reproduction_bundle.py
.venv/bin/python -m pytest -q tests
```

Expected test status: 29 passed.  The verification script must print
`"status": "PASS"`.

## 3. Environment reconstruction

The exact paper run should use the supplied cached trace:

`cache/contact_trace_1584_24_11.npz`

Its SHA-256 is
`e6baf0a0ad85a17533e2a0b06ed3ec959b85ad10152069377c7de92cf3df1637`.
It contains 14,256 root-refined contact intervals for the 1,584/24/11 Walker
constellation.  To reconstruct it from the geometry code instead:

```bash
.venv/bin/python scripts/build_contact_trace.py \
  --scan-step-s 5 \
  --output cache/rebuilt_contact_trace_1584_24_11.npz
```

The regenerated trace must contain 14,256 intervals.  Root boundaries can
differ at floating-point roundoff scale across numerical-library builds (the
current same-machine regeneration differed by at most 4.01e-11 s), so it need
not have the cached NPZ's byte-level SHA.  Use the supplied cached trace for the
exact retained evaluation and `results_paper/NUMERICAL_VALIDATION.md` for the
independent 2.5 s scan convergence check.

The environment is event-driven.  Relay decisions consume zero simulated time;
time advances only on PAT completion, link/GSL expiry, rerouting, and session
completion.  `leo_session/config.py` is the numerical source of truth,
`leo_session/environment.py` is the event loop, and
`leo_session/topology.py` constructs the session-independent initial ACTIVE
matching.

## 4. Reproduce CPU baselines and constrained feasibility witness

All paper comparisons use exactly the 3,600 rows in
`results_paper/paper_artifacts/evaluation_requests.csv`.

```bash
.venv/bin/python scripts/run_cpu_baselines.py \
  --request-file results_paper/paper_artifacts/evaluation_requests.csv \
  --methods raw lifetime prp remaining physical_upper \
  --output-dir results_paper/reproduction_cpu

.venv/bin/python scripts/run_cpu_baselines.py \
  --request-file results_paper/paper_artifacts/evaluation_requests.csv \
  --methods cost_greedy \
  --output-dir results_paper/reproduction_cost_greedy

.venv/bin/python scripts/run_cpu_baselines.py \
  --request-file results_paper/paper_artifacts/evaluation_requests.csv \
  --methods constrained_oracle \
  --output-dir results_paper/reproduction_constrained_witness
```

`constrained_oracle` is a constructive feasibility witness: it globally chooses
the currently feasible path with maximum contact bottleneck using exact contact
end times, while `SessionEnvironment` still applies the same terminal occupancy,
PAT, break-before-make, and continuity rules.  Its 3,600/3,600 completion proves
binary feasibility for this frozen set.  It is not an exhaustive optimizer of
path length or PAT work.

## 5. Reproduce the three main PPO-GCN training runs

The following command trains seeds 0, 1, and 2 sequentially with the final
3-layer/128 configuration.  It performs no validation-based checkpoint
selection; `best_seed*.pt` is the final network at approximately 1M relay steps.

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/train_gcn_ppo.py \
  --seeds 0 1 2 \
  --total-relay-steps 1000000 \
  --device cuda:0 \
  --hidden-dim 128 --gcn-layers 3 \
  --minibatch-size 64 --rollout-steps 1024 --update-epochs 4 \
  --learning-rate 0.003 --gamma 0.99 --gae-lambda 0.95 \
  --value-clip 1.0 --validation-offsets 0 \
  --checkpoint-interval-updates 1000000 \
  --output-dir results_ppo_gcn/reproduction_main_seeds012
```

Evaluate every seed on the full frozen set:

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/evaluate_gcn_ppo.py \
  --models \
    results_ppo_gcn/reproduction_main_seeds012/best_seed0.pt \
    results_ppo_gcn/reproduction_main_seeds012/best_seed1.pt \
    results_ppo_gcn/reproduction_main_seeds012/best_seed2.pt \
  --request-file results_paper/paper_artifacts/evaluation_requests.csv \
  --device cuda:0 \
  --output-dir results_ppo_gcn/reproduction_main_seeds012/eval_full_test
```

The retained checkpoints produced completion counts 3400, 3308, and 3288 out
of 3,600 for seeds 0, 1, and 2.  Mean ± sample SD is 92.56% ± 1.66 percentage
points.  GPU kernels and library changes can prevent bit-for-bit retraining,
so compare the full metrics and checkpoint metadata rather than assuming binary
weight identity across machines.

## 6. Reproduce the controlled ablations

No-RCT removes remaining-contact values only from the observation.  The model,
environment, and reward stay unchanged.

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/train_gcn_ppo.py \
  --seeds 0 --total-relay-steps 1000000 --device cuda:0 \
  --hidden-dim 128 --gcn-layers 3 --minibatch-size 64 \
  --rollout-steps 1024 --update-epochs 4 --learning-rate 0.003 \
  --gamma 0.99 --gae-lambda 0.95 --value-clip 1.0 \
  --validation-offsets 0 --checkpoint-interval-updates 1000000 \
  --no-rct --output-dir results_ppo_gcn/reproduction_no_rct

CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/evaluate_gcn_ppo.py \
  --models results_ppo_gcn/reproduction_no_rct/best_seed0.pt \
  --request-file results_paper/paper_artifacts/evaluation_requests.csv \
  --device cuda:0 --no-rct \
  --output-dir results_ppo_gcn/reproduction_no_rct/eval_full_test
```

No-PAT removes only the explicit relay PAT reward cost.  Physical PAT remains
20 s and all PAT/ACTIVE observation fields remain present.

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/train_gcn_ppo.py \
  --seeds 0 --total-relay-steps 1000000 --device cuda:0 \
  --hidden-dim 128 --gcn-layers 3 --minibatch-size 64 \
  --rollout-steps 1024 --update-epochs 4 --learning-rate 0.003 \
  --gamma 0.99 --gae-lambda 0.95 --value-clip 1.0 \
  --validation-offsets 0 --checkpoint-interval-updates 1000000 \
  --no-pat --output-dir results_ppo_gcn/reproduction_no_pat

CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/evaluate_gcn_ppo.py \
  --models results_ppo_gcn/reproduction_no_pat/best_seed0.pt \
  --request-file results_paper/paper_artifacts/evaluation_requests.csv \
  --device cuda:0 --no-pat \
  --output-dir results_ppo_gcn/reproduction_no_pat/eval_full_test
```

Before any new reward, feature, or architecture long run, use at most 10k relay
steps and inspect actual actions with `scripts/audit_gcn_actions.py`.

## 7. Rebuild comparison tables

The table builder verifies an exact one-to-one match with all 3,600 frozen
request specifications before aggregating.

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/evaluate_gcn_ppo.py \
  --models results_ppo_gcn/reproduction_main_seeds012/best_seed0.pt \
  --request-file results_paper/paper_artifacts/evaluation_requests.csv \
  --device cuda:0 \
  --output-dir results_ppo_gcn/reproduction_main_seed0_eval

.venv/bin/python scripts/build_comparison_tables.py \
  --ppo-sessions results_ppo_gcn/reproduction_main_seed0_eval/sessions.csv \
  --output-dir results_ppo_gcn/reproduction_tables
```

For a multi-seed `sessions.csv`, build a paper table for one seed at a time or
use the supplied `summary_by_seed.csv`, `summary_pooled_by_seed.csv`, and
`summary_across_seeds.csv`.  The retained seed-0 comparison tables and all X
slices are already included in the handover.
