#!/usr/bin/env bash
# Restart from the completed V2 S0 40k model for an independent low-LR 20k
# experiment. It uses a new local output directory and a new W&B run, leaving
# the earlier interrupted 30k-schedule run untouched.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
STARVLA_DIR="${STARVLA_DIR:-${PROJECT_ROOT}/third_party/starvla}"

export CONFIG_YAML="${STARVLA_DIR}/examples/calvin/train_files/e1_spatial_intent_s0_query_ffn_v2_cont_40k_to_50k.yaml"
export PRETRAINED_CHECKPOINT="/home/liuchang/kehang/488project/code/starvla/playground/Pretrained_models/kehang-StarVLA/checkpoints/calvin/e1_spatial_intent_query_ffn_v2_s0_40000/final_model/pytorch_model.pt"
export S0_MAX_STEPS=20000
export S0_WARMUP_STEPS=0
export TRAINER_IS_RESUME=false
export RUN_ID=e1_spatial_intent_query_ffn_v2_s0_restart40k_20k
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-32}"
export WANDB_NAME_OVERRIDE=e1_spatial_intent_query_ffn_v2_s0_restart40k_20k

# Do not inherit recovery-only overrides if this launcher is invoked from a
# shell that previously exported them.
unset TRAINING_PHASE_ID_OVERRIDE
unset WANDB_LOG_AFTER_STEP_OVERRIDE
unset RECOVERED_FROM_PHASE_STEP_OVERRIDE

export WANDB_MODE=online
export WANDB_ENTITY=chaikehang-sjtu-hpc-center
export WANDB_PROJECT=starVLA_Calvin_E1_Spatial_Intent_S0_V2
export WANDB_RUN_ID=hxtpxb3j
export WANDB_RESUME=never
export WANDB_REQUIRED=true

echo "Experiment=V2 S0 restart from 40k, independent 20k LR schedule"
echo "Local run=${RUN_ID}"
echo "New W&B=${WANDB_ENTITY}/${WANDB_PROJECT}/${WANDB_RUN_ID}, resume=${WANDB_RESUME}"
echo "W&B displayed steps=40000..60000; local phase steps=0..20000"
echo "Original W&B run 1atoivj7 is not modified by this launcher."

exec bash "${SCRIPT_DIR}/train_e1_spatial_intent_s0.sh"
