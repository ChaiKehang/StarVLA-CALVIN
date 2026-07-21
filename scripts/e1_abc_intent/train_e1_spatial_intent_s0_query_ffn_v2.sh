#!/usr/bin/env bash
# Spatial Intent Aggregator v2 S0 launcher. The delegated launcher provides
# the RAM/GPU watchdog and enforces a maximum of two visible GPUs/processes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
STARVLA_DIR="${STARVLA_DIR:-${PROJECT_ROOT}/third_party/starvla}"

export CONFIG_YAML="${CONFIG_YAML:-${STARVLA_DIR}/examples/calvin/train_files/e1_spatial_intent_s0_query_ffn_v2.yaml}"
export S0_MAX_STEPS="${S0_MAX_STEPS:-60000}"
export S0_WARMUP_STEPS="${S0_WARMUP_STEPS:-5000}"
export RUN_ID="${RUN_ID:-e1_spatial_intent_query_ffn_v2_s0_${S0_MAX_STEPS}}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-32}"
export WANDB_PROJECT="${WANDB_PROJECT:-starVLA_Calvin_E1_Spatial_Intent_S0_V2}"

if (( S0_MAX_STEPS < 10000 || S0_MAX_STEPS > 70000 )); then
  echo "[ERROR] v2 S0_MAX_STEPS must be within [10000, 70000], got ${S0_MAX_STEPS}." >&2
  exit 2
fi

exec bash "${SCRIPT_DIR}/train_e1_spatial_intent_s0.sh"
