#!/usr/bin/env bash
# Factorized 255-D bounded sparse Query-FiLM main run: S1 15k + S2 75k.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
STARVLA_DIR="${STARVLA_DIR:-${PROJECT_ROOT}/third_party/starvla}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/home/data/models/kehang-StarVLA/checkpoints/calvin}"

export CONFIG_YAML="${CONFIG_YAML:-${STARVLA_DIR}/examples/calvin/train_files/e1_factorized_sparse_query_s1_15k_s2_75k.yaml}"
export S0_STEPS=40000
export S0_RUN_ID="${S0_RUN_ID:-e1_factorized_spatial_intent_s0_50k}"
export S0_INTENT_CHECKPOINT="${S0_INTENT_CHECKPOINT:-${CHECKPOINT_ROOT}/${S0_RUN_ID}/checkpoints/best_intent_pytorch_model.pt}"
export STAGE1_STEPS=15000
export MAIN_MAX_STEPS=90000
export ACTION_STEP_OFFSET=0
export RUN_ID="${RUN_ID:-e1_factorized_sparse_query_s1_15k_s2_75k}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
export WANDB_PROJECT="${WANDB_PROJECT:-starVLA_Calvin_E1_Factorized_Main}"

exec bash "${SCRIPT_DIR}/train_e1_spatial_intent_s1_s2_90k.sh"
