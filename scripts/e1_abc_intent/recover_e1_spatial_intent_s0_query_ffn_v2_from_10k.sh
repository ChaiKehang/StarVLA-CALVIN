#!/usr/bin/env bash
# Recover the interrupted V2 S0 continuation from its latest complete local
# checkpoint (currently phase step 10k) and continue the original 30k phase.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
STARVLA_DIR="${STARVLA_DIR:-${PROJECT_ROOT}/third_party/starvla}"

export CONFIG_YAML="${STARVLA_DIR}/examples/calvin/train_files/e1_spatial_intent_s0_query_ffn_v2_cont_40k_to_50k.yaml"
export PRETRAINED_CHECKPOINT="/home/liuchang/kehang/488project/code/starvla/playground/Pretrained_models/kehang-StarVLA/checkpoints/calvin/e1_spatial_intent_query_ffn_v2_s0_40000/final_model/pytorch_model.pt"
export RUN_ID=e1_spatial_intent_query_ffn_v2_s0_cont_40k_to_50k
export S0_MAX_STEPS=30000
export S0_WARMUP_STEPS=0
export PER_DEVICE_BATCH_SIZE=32
export TRAINER_IS_RESUME=true

# The first attempt uploaded through W&B global step 53780 but only persisted
# model phase step 10000. Replayed phase steps 10000..13780 are intentionally
# not sent out of order; uploading resumes at phase step 13800/global 53800.
export TRAINING_PHASE_ID_OVERRIDE=2
export RECOVERED_FROM_PHASE_STEP_OVERRIDE=10000
export WANDB_LOG_AFTER_STEP_OVERRIDE=53780

export WANDB_MODE=online
export WANDB_ENTITY=chaikehang-sjtu-hpc-center
export WANDB_PROJECT=starVLA_Calvin_E1_Spatial_Intent_S0_V2
export WANDB_RUN_ID=1atoivj7
export WANDB_RESUME=must

CHECKPOINT_DIR="/home/data/models/kehang-StarVLA/checkpoints/calvin/${RUN_ID}/checkpoints"
EXPECTED_CHECKPOINT="${CHECKPOINT_DIR}/steps_10000_pytorch_model.pt"
if [[ ! -s "${EXPECTED_CHECKPOINT}" ]]; then
  echo "[ERROR] Required recovery checkpoint is missing or empty: ${EXPECTED_CHECKPOINT}" >&2
  exit 2
fi

echo "Recovery=local phase step 10000 -> 30000"
echo "Checkpoint discovery directory=${CHECKPOINT_DIR}"
echo "W&B=${WANDB_ENTITY}/${WANDB_PROJECT}/${WANDB_RUN_ID}"
echo "W&B upload resumes after global step ${WANDB_LOG_AFTER_STEP_OVERRIDE}"
echo "Replayed local steps 10000..13780 remain terminal-only to avoid out-of-order W&B history."

RECOVERY_CONSOLE_LOG="/home/data/models/kehang-StarVLA/checkpoints/calvin/${RUN_ID}/recovery_console.log"
echo "Recovery console log=${RECOVERY_CONSOLE_LOG}"

set +e
bash "${SCRIPT_DIR}/train_e1_spatial_intent_s0.sh" 2>&1 | tee -a "${RECOVERY_CONSOLE_LOG}"
TRAINING_STATUS=${PIPESTATUS[0]}
set -e
exit "${TRAINING_STATUS}"
