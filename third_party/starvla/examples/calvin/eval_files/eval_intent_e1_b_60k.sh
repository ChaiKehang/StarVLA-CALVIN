#!/usr/bin/env bash
# Offline labeled Intent evaluation for the E1-B 60k checkpoint.

set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/liuchang/miniconda3/envs/starvla-e0/bin/python}"
CHECKPOINT="${CHECKPOINT:-${STARVLA_DIR}/playground/Pretrained_models/kehang-StarVLA/checkpoints/calvin/e1_b_abc_rel_scaled_intent125_h8/checkpoints/steps_60000_pytorch_model.pt}"
DATA_ROOT="${DATA_ROOT:-/home/data/datasets/kehang-CALVIN/calvin/lerobot}"
MAX_SAMPLES="${MAX_SAMPLES:-10000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
GPU_ID="${GPU_ID:-2}"
OUTPUT="${OUTPUT:-/home/liuchang/kehang/488project/eval_logs/e1_b/steps_60000_intent_offline.json}"

for required in "${STARVLA_DIR}" "${CHECKPOINT}" "${DATA_ROOT}"; do
    if [[ ! -e "${required}" ]]; then
        echo "[ERROR] Required path does not exist: ${required}" >&2
        exit 2
    fi
done

mkdir -p "$(dirname "${OUTPUT}")"
cd "${STARVLA_DIR}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"

echo "Checkpoint=${CHECKPOINT}"
echo "CUDA_VISIBLE_DEVICES=${GPU_ID}"
echo "Offline labeled samples=${MAX_SAMPLES}; output=${OUTPUT}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" \
    examples/calvin/eval_files/eval_intent_checkpoint.py \
    --checkpoint "${CHECKPOINT}" \
    --data-root "${DATA_ROOT}" \
    --max-samples "${MAX_SAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --output "${OUTPUT}"
