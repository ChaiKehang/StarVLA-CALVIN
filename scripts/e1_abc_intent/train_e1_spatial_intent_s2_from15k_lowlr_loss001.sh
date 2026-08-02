#!/usr/bin/env bash
# Start an independent S2 branch from the existing 15k model checkpoint.
# This intentionally creates a new local run and a new W&B run because both
# Intent learning rates and the auxiliary-loss weight differ from the parent.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
STARVLA_DIR="${STARVLA_DIR:-${PROJECT_ROOT}/third_party/starvla}"

export CONFIG_YAML="${CONFIG_YAML:-${STARVLA_DIR}/examples/calvin/train_files/e1_spatial_intent_s2_from15k_lowlr_loss001.yaml}"
export RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-/home/data/models/kehang-StarVLA/checkpoints/calvin/e1_spatial_intent_query_ffn_v2_s1_15k_s2_75k_90k_bs4_ga2_restart/checkpoints/steps_15000_pytorch_model.pt}"
export RESUME_STEP=15000
export STAGE1_STEPS=15000
export MAIN_MAX_STEPS=90000
export RUN_ID="${RUN_ID:-e1_spatial_intent_query_ffn_v2_s2_from15k_lowlr_loss001}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
export ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-${SCRIPT_DIR}/deepspeed/deepspeed_zero2_grad_accum2.yaml}"
export WANDB_PROJECT="${WANDB_PROJECT:-starVLA_Calvin_E1_Spatial_Intent_Main_V2}"

# A changed hyperparameter branch must not append to the parent W&B history.
unset WANDB_RUN_ID
unset WANDB_RESUME

exec bash "${SCRIPT_DIR}/train_e1_spatial_intent_s1_s2_90k.sh"
