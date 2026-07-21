#!/usr/bin/env bash
# Run E1-B inside a Slurm allocation created by a direct `srun ... bash SCRIPT`.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/liuchang/kehang/488project}"
LOG_DIR="${PROJECT_ROOT}/logs/e1_abc_intent"
RUN_ID="${RUN_ID:-e1_b_abc_rel_scaled_intent125_h8}"
LOG_FILE="${LOG_DIR}/${RUN_ID}_srun_${SLURM_JOB_ID:-no_slurm}_$(date +%Y%m%d_%H%M%S).log"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "[ERROR] Invoke this script through srun so Slurm allocates the GPUs." >&2
  exit 40
fi

ALLOCATED_GPUS="${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
if [[ -z "${ALLOCATED_GPUS}" ]]; then
  echo "[ERROR] Slurm did not expose the allocated GPU list." >&2
  exit 41
fi
if tr ',' '\n' <<<"${ALLOCATED_GPUS}" | sed 's/^gpu://; s/^[[:space:]]*//; s/[[:space:]]*$//' | grep -qx '0'; then
  echo "[ERROR] Allocation contains physical GPU 0; refusing to train." >&2
  exit 42
fi

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}"
echo "Log file: ${LOG_FILE}"

source /home/liuchang/miniconda3/bin/activate starvla-e0

export PROJECT_ROOT
export STARVLA_DIR="${STARVLA_DIR:-${PROJECT_ROOT}/third_party/starvla}"
export MODEL_ROOT="${MODEL_ROOT:-/home/data/models/kehang-StarVLA}"
export LEROBOT_ROOT="${LEROBOT_ROOT:-/home/data/datasets/kehang-CALVIN/calvin/lerobot}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/home/data/models/kehang-StarVLA/checkpoints/calvin}"
export CACHE_ROOT="${CACHE_ROOT:-/home/data/datasets/kehang-CALVIN/.cache}"
export CONFIG_YAML="${CONFIG_YAML:-${STARVLA_DIR}/examples/calvin/train_files/e1_b_abc_rel_calvin_scaled_intent125.yaml}"
export RUN_ID NUM_PROCESSES="${NUM_PROCESSES:-2}" PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
export USE_INTENT_AUX_LOSS="${USE_INTENT_AUX_LOSS:-true}"
export ADD_INTENT_TO_TIMESTEP="${ADD_INTENT_TO_TIMESTEP:-true}"
export USE_FFN_INTENT_FILM="${USE_FFN_INTENT_FILM:-false}"
export WANDB_MODE="${WANDB_MODE:-online}"
export E1_WANDB_PROJECT="${E1_WANDB_PROJECT:-starVLA_Calvin_E1_B}"
export E1_WANDB_ENTITY="${E1_WANDB_ENTITY:-chaikehang-sjtu-hpc-center}"

watchdog() {
  while sleep 5; do
    available_gb="$(free -g | awk '/Mem:/ {print $7}')"
    if [[ "${available_gb:-999}" -lt 12 ]]; then
      echo "[WATCHDOG] Available RAM ${available_gb}GiB < 12GiB; cancelling ${SLURM_JOB_ID}." >&2
      scancel "${SLURM_JOB_ID}" || true
      return 99
    fi

    while IFS= read -r gpu; do
      gpu="$(xargs <<<"${gpu}")"
      [[ "${gpu}" =~ ^[0-9]+$ ]] || continue
      used="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | head -1)"
      if [[ "${used:-0}" -gt 49000 ]]; then
        echo "[WATCHDOG] GPU ${gpu} uses ${used}MiB; cancelling ${SLURM_JOB_ID}." >&2
        scancel "${SLURM_JOB_ID}" || true
        return 99
      fi
    done < <(tr ',' '\n' <<<"${ALLOCATED_GPUS}" | sed 's/^gpu://')
  done
}

watchdog &
WATCHDOG_PID=$!
trap 'kill "${WATCHDOG_PID}" 2>/dev/null || true' EXIT

bash "${PROJECT_ROOT}/scripts/e1_abc_intent/train_e1_b_90k_generic.sh"
