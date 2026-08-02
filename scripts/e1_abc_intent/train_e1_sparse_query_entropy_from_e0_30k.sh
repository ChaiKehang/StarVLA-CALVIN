#!/usr/bin/env bash
# E0 30k -> S1 10k -> S2 50k, using the pretrained S0 v2 Intent head.
# Run this inside an existing one- or two-GPU Slurm allocation/tmux shell.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
STARVLA_DIR="${STARVLA_DIR:-${PROJECT_ROOT}/third_party/starvla}"
MODEL_ROOT="${MODEL_ROOT:-/home/data/models/kehang-StarVLA}"

export CONFIG_YAML="${CONFIG_YAML:-${STARVLA_DIR}/examples/calvin/train_files/e1_spatial_intent_sparse_query_entropy_from_e0_30k.yaml}"
export PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-${MODEL_ROOT}/checkpoints/calvin/e0_abc_rel/checkpoints/steps_30000_pytorch_model.pt}"
export S0_INTENT_CHECKPOINT="${S0_INTENT_CHECKPOINT:-${MODEL_ROOT}/checkpoints/calvin/e1_spatial_intent_query_ffn_v2_s0_restart40k_20k/checkpoints/steps_20000_pytorch_model.pt}"

export ACTION_STEP_OFFSET=30000
export STAGE1_STEPS=10000
export MAIN_MAX_STEPS=60000
export RUN_ID="${RUN_ID:-e1_sparse_query_11_23_35_linear_entropy_from_e0_30k_s1_10k_s2_50k}"

export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
export ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-${SCRIPT_DIR}/deepspeed/deepspeed_zero2_grad_accum2.yaml}"
export WANDB_PROJECT="${WANDB_PROJECT:-starVLA_Calvin_E1_Sparse_Query_Entropy}"

exec bash "${SCRIPT_DIR}/train_e1_spatial_intent_s1_s2_90k.sh"
