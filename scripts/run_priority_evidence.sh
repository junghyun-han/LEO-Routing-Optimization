#!/usr/bin/env bash
set -euo pipefail

# Priority evidence chain requested on 2026-09-04:
#   1. controlled No-RCT and No-PAT diagnostics + long runs + frozen tests
#   2. final 3-layer model seeds 1 and 2 + three-seed frozen evaluation
# Physical GPU 2 is mandatory.  CUDA remaps it to cuda:0 inside Python.
export CUDA_VISIBLE_DEVICES=2

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN="$ROOT/.venv/bin/python"
OUT="$ROOT/results_ppo_gcn/priority_evidence_2026-09-04"
REQUESTS="$ROOT/results_paper/paper_artifacts/evaluation_requests.csv"
BASELINES="$ROOT/results_paper/main_240_metrics/sessions.csv"
COST_GREEDY="$ROOT/results_paper/cost_greedy_current_reward_frozen3600/sessions.csv"
SEED0="$ROOT/results_ppo_gcn/long_run_2026-09-04/gcn3_h128_v2_exact_route_reward_seed0_1m_mb64_gpu2/best_seed0.pt"
STATUS="$OUT/STATUS.tsv"

mkdir -p "$OUT"
cd "$ROOT"
export PYTHONPATH="$ROOT"

stamp() {
    printf '%s\t%s\n' "$(date --iso-8601=seconds)" "$1" | tee -a "$STATUS"
}

verify_gpu() {
    "$PYTHON_BIN" -c 'import os, torch; assert os.environ.get("CUDA_VISIBLE_DEVICES") == "2"; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; print("CUDA restriction verified: physical GPU 2 only; Python device=cuda:0;", torch.cuda.get_device_name(0), flush=True)'
}

train_common() {
    local seed="$1"
    local steps="$2"
    local output="$3"
    shift 3
    "$PYTHON_BIN" scripts/train_gcn_ppo.py \
        --seeds "$seed" \
        --total-relay-steps "$steps" \
        --device cuda:0 \
        --hidden-dim 128 \
        --gcn-layers 3 \
        --minibatch-size 64 \
        --rollout-steps 1024 \
        --update-epochs 4 \
        --learning-rate 0.003 \
        --gamma 0.99 \
        --gae-lambda 0.95 \
        --value-clip 1.0 \
        --validation-offsets 0 \
        --checkpoint-interval-updates 1000000 \
        --output-dir "$output" \
        "$@"
}

run_ablation() {
    local name="$1"
    local label="$2"
    local flag="$3"
    local diagnostic="$OUT/${name}_diagnostic_10k"
    local long="$OUT/${name}_3l_h128_seed0_1m_mb64_gpu2"

    stamp "DIAGNOSTIC_START variant=$name steps=10000"
    train_common 0 10000 "$diagnostic" "$flag"
    "$PYTHON_BIN" scripts/audit_gcn_actions.py \
        --model "$diagnostic/best_seed0.pt" \
        --request-file "$REQUESTS" \
        --device cuda:0 \
        --output-dir "$diagnostic/action_audit" \
        "$flag"
    find "$diagnostic" -maxdepth 1 -type f -name '*.pt' -delete
    stamp "DIAGNOSTIC_PASS variant=$name action_audit=$diagnostic/action_audit/summary.json"

    stamp "TRAIN_START variant=$name steps=1000000 seed=0"
    train_common 0 1000000 "$long" "$flag"
    stamp "TRAIN_DONE variant=$name steps=1000000 seed=0"
    stamp "FROZEN_EVAL_START variant=$name requests=3600"
    "$PYTHON_BIN" scripts/evaluate_gcn_ppo.py \
        --models "$long/best_seed0.pt" \
        --request-file "$REQUESTS" \
        --device cuda:0 \
        --output-dir "$long/eval_full_test" \
        "$flag"
    "$PYTHON_BIN" scripts/build_comparison_tables.py \
        --baseline-sessions "$BASELINES" \
        --cost-greedy-sessions "$COST_GREEDY" \
        --ppo-sessions "$long/eval_full_test/sessions.csv" \
        --request-file "$REQUESTS" \
        --ppo-label "$label" \
        --output-dir "$long/eval_full_test/comparison_tables"
    "$PYTHON_BIN" - "$long/eval_full_test/sessions.csv" "$name" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], newline='', encoding='utf-8')))
assert len(rows) == 3600
assert len({int(row['request_id']) for row in rows}) == 3600
completed = sum(row['completion'].lower() == 'true' for row in rows)
print(f"FROZEN_RESULT variant={sys.argv[2]} completion={completed}/3600 ({100*completed/3600:.2f}%)", flush=True)
PY
    stamp "FROZEN_EVAL_DONE variant=$name requests=3600"
}

verify_gpu
stamp "CHAIN_START physical_gpu=2"
run_ablation "no_rct" "PPO-GCN No-RCT 3L/128 1M" "--no-rct"
run_ablation "no_pat" "PPO-GCN No-PAT 3L/128 1M" "--no-pat"

MULTI="$OUT/main_3l_h128_seeds1_2_1m_mb64_gpu2"
stamp "MULTISEED_TRAIN_START seeds=1,2 steps_per_seed=1000000"
"$PYTHON_BIN" scripts/train_gcn_ppo.py \
    --seeds 1 2 \
    --total-relay-steps 1000000 \
    --device cuda:0 \
    --hidden-dim 128 \
    --gcn-layers 3 \
    --minibatch-size 64 \
    --rollout-steps 1024 \
    --update-epochs 4 \
    --learning-rate 0.003 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --value-clip 1.0 \
    --validation-offsets 0 \
    --checkpoint-interval-updates 1000000 \
    --output-dir "$MULTI"
stamp "MULTISEED_TRAIN_DONE seeds=1,2 steps_per_seed=1000000"

MULTI_EVAL="$OUT/main_3l_h128_seeds0_1_2_1m/eval_full_test"
stamp "MULTISEED_FROZEN_EVAL_START seeds=0,1,2 requests_per_seed=3600"
"$PYTHON_BIN" scripts/evaluate_gcn_ppo.py \
    --models "$SEED0" "$MULTI/best_seed1.pt" "$MULTI/best_seed2.pt" \
    --request-file "$REQUESTS" \
    --device cuda:0 \
    --output-dir "$MULTI_EVAL"
"$PYTHON_BIN" - "$MULTI_EVAL/sessions.csv" <<'PY'
import csv, collections, sys
rows = list(csv.DictReader(open(sys.argv[1], newline='', encoding='utf-8')))
counts = collections.Counter(int(row['seed']) for row in rows)
assert counts == {0: 3600, 1: 3600, 2: 3600}, counts
for seed in sorted(counts):
    selected = [row for row in rows if int(row['seed']) == seed]
    completed = sum(row['completion'].lower() == 'true' for row in selected)
    print(f"FROZEN_RESULT seed={seed} completion={completed}/3600 ({100*completed/3600:.2f}%)", flush=True)
PY
stamp "MULTISEED_FROZEN_EVAL_DONE seeds=0,1,2 requests_per_seed=3600"
stamp "CHAIN_DONE"
