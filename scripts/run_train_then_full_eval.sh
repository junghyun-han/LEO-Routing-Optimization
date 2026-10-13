#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 GCN_LAYERS RELAY_STEPS OUTPUT_DIR" >&2
    exit 2
fi

GCN_LAYERS="$1"
RELAY_STEPS="$2"
OUTPUT_DIR="$3"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN="$ROOT/.venv/bin/python"

cd "$ROOT"
export PYTHONPATH="$ROOT"

"$PYTHON_BIN" scripts/train_gcn_ppo.py \
    --seeds 0 \
    --total-relay-steps "$RELAY_STEPS" \
    --device cuda:0 \
    --hidden-dim 128 \
    --gcn-layers "$GCN_LAYERS" \
    --minibatch-size 64 \
    --rollout-steps 1024 \
    --update-epochs 4 \
    --learning-rate 0.003 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --value-clip 1.0 \
    --validation-offsets 0 \
    --checkpoint-interval-updates 49 \
    --output-dir "$OUTPUT_DIR"

"$PYTHON_BIN" scripts/evaluate_gcn_ppo.py \
    --models "$OUTPUT_DIR/best_seed0.pt" \
    --request-file results_paper/paper_artifacts/evaluation_requests.csv \
    --device cuda:0 \
    --output-dir "$OUTPUT_DIR/eval_full_test"
