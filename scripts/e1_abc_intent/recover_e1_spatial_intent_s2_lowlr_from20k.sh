#!/usr/bin/env bash
# Recover the interrupted low-LR S2 run in place from its saved 20k model
# checkpoint. W&B resumes the original run but suppresses replayed 20k-21.52k
# history so its global step remains monotonic.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
STARVLA_DIR="${STARVLA_DIR:-${PROJECT_ROOT}/third_party/starvla}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/home/data/models/kehang-StarVLA/checkpoints/calvin}"

export CONFIG_YAML="${CONFIG_YAML:-${STARVLA_DIR}/examples/calvin/train_files/e1_spatial_intent_s2_from15k_lowlr_loss001.yaml}"
export RUN_ID=e1_spatial_intent_query_ffn_v2_s2_from15k_lowlr_loss001
export RESUME_CHECKPOINT="${CHECKPOINT_ROOT}/${RUN_ID}/checkpoints/steps_20000_pytorch_model.pt"
export RESUME_STEP=20000
export RESUME_IN_PLACE=true
export STAGE1_STEPS=15000
export MAIN_MAX_STEPS=90000
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
export ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-${SCRIPT_DIR}/deepspeed/deepspeed_zero2_grad_accum2.yaml}"
export WANDB_PROJECT="${WANDB_PROJECT:-starVLA_Calvin_E1_Spatial_Intent_Main_V2}"

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_RUN_ID=weihdyzg
export WANDB_RESUME=must
export WANDB_LOG_AFTER_STEP_OVERRIDE=21520
export RECOVERED_FROM_PHASE_STEP_OVERRIDE=20000
export TRAINING_PHASE_ID_OVERRIDE=1

exec bash "${SCRIPT_DIR}/train_e1_spatial_intent_s1_s2_90k.sh"
