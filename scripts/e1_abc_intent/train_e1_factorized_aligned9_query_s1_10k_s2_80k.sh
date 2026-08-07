#!/usr/bin/env bash
# Nine-layer depth-aligned factorized Query-FiLM main run:
# S1 10k + S2 80k, router LR 2e-6, bound 0.1, condition dropout 0.2.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
STARVLA_DIR="${STARVLA_DIR:-${PROJECT_ROOT}/third_party/starvla}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/home/data/models/kehang-StarVLA/checkpoints/calvin}"

export CONFIG_YAML="${CONFIG_YAML:-${STARVLA_DIR}/examples/calvin/train_files/e1_factorized_aligned9_query_s1_10k_s2_80k.yaml}"
export ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-${PROJECT_ROOT}/scripts/e1_abc_intent/deepspeed/deepspeed_zero2_grad_accum2.yaml}"
export S0_STEPS="${S0_STEPS:-50000}"
export S0_RUN_ID="${S0_RUN_ID:-e1_factorized_aligned9_spatial_intent_s0_50k}"
export S0_INTENT_CHECKPOINT="${S0_INTENT_CHECKPOINT:-${CHECKPOINT_ROOT}/${S0_RUN_ID}/checkpoints/best_intent_pytorch_model.pt}"
export STAGE1_STEPS="${STAGE1_STEPS:-10000}"
export MAIN_MAX_STEPS="${MAIN_MAX_STEPS:-90000}"
export ACTION_STEP_OFFSET="${ACTION_STEP_OFFSET:-0}"
export RUN_ID="${RUN_ID:-e1_factorized_aligned9_query_s1_10k_s2_80k}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
export WANDB_PROJECT="${WANDB_PROJECT:-starVLA_Calvin_E1_Factorized_Aligned9_Main}"

exec bash "${SCRIPT_DIR}/train_e1_spatial_intent_s1_s2_90k.sh"
