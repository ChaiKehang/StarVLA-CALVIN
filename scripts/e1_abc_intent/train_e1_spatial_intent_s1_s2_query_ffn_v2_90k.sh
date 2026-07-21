#!/usr/bin/env bash
# Spatial Intent Aggregator v2 S1/S2 launcher. S1+S2 remain exactly 90k Action
# optimizer steps. The delegated launcher supplies the watchdog and <=2-GPU
# validation used by the existing E1 training workflow.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
STARVLA_DIR="${STARVLA_DIR:-${PROJECT_ROOT}/third_party/starvla}"

export CONFIG_YAML="${CONFIG_YAML:-${STARVLA_DIR}/examples/calvin/train_files/e1_spatial_intent_s1_s2_query_ffn_v2_90k.yaml}"
export S0_STEPS="${S0_STEPS:-60000}"
export S0_RUN_ID="${S0_RUN_ID:-e1_spatial_intent_query_ffn_v2_s0_${S0_STEPS}}"
export STAGE1_STEPS="${STAGE1_STEPS:-10000}"
export MAIN_MAX_STEPS=90000
export RUN_ID="${RUN_ID:-e1_spatial_intent_query_ffn_v2_s1_s2_90k}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
export WANDB_PROJECT="${WANDB_PROJECT:-starVLA_Calvin_E1_Spatial_Intent_Main_V2}"

if (( S0_STEPS < 10000 || S0_STEPS > 70000 )); then
  echo "[ERROR] v2 S0_STEPS must be within [10000, 70000], got ${S0_STEPS}." >&2
  exit 2
fi

exec bash "${SCRIPT_DIR}/train_e1_spatial_intent_s1_s2_90k.sh"
