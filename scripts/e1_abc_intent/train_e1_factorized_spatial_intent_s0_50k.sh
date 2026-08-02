#!/usr/bin/env bash
# Factorized XYZ/RPY/gripper Intent-only S0. This wrapper does not create or
# overwrite the derived dataset; run build_factorized_intent_dataset.py once.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
STARVLA_DIR="${STARVLA_DIR:-${PROJECT_ROOT}/third_party/starvla}"
DERIVED_DATASET="${DERIVED_DATASET:-/home/data/datasets/kehang-CALVIN/calvin/lerobot/sixpigs1_calvin2lerobotV21_ABC_D_scnet_rel_calvin_scaled_intent_factorized_h8}"

if [[ ! -f "${DERIVED_DATASET}/meta/factorized_intent_config.json" ]]; then
  echo "[ERROR] Factorized dataset is not prepared: ${DERIVED_DATASET}" >&2
  echo "Run:" >&2
  echo "/home/liuchang/miniconda3/envs/starvla-e0/bin/python ${SCRIPT_DIR}/build_factorized_intent_dataset.py" >&2
  exit 2
fi

export CONFIG_YAML="${CONFIG_YAML:-${STARVLA_DIR}/examples/calvin/train_files/e1_factorized_spatial_intent_s0_50k.yaml}"
export S0_MAX_STEPS=40000
export S0_WARMUP_STEPS=4000
export RUN_ID="${RUN_ID:-e1_factorized_spatial_intent_s0_50k}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-32}"
export WANDB_PROJECT="${WANDB_PROJECT:-starVLA_Calvin_E1_Factorized_Intent_S0}"

exec bash "${SCRIPT_DIR}/train_e1_spatial_intent_s0.sh"
