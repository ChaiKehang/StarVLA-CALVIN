#!/usr/bin/env bash
# DIDIS Lab wrapper for train_e0_90k_generic.sh.
# Submit only after `squeue`/`sinfo` confirm the node is healthy and the
# allocated physical GPU list does not contain GPU 0.

#SBATCH --job-name=e0_abc_90k
#SBATCH --partition=compute
#SBATCH --gres=gpu:A6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/liuchang/kehang/488project/logs/e0_abc_rel/%x_%j.out
#SBATCH --error=/home/liuchang/kehang/488project/logs/e0_abc_rel/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/home/liuchang/kehang/488project
MODEL_ROOT=/home/data/models/kehang-StarVLA
LEROBOT_ROOT=/home/data/datasets/kehang-CALVIN/calvin/lerobot
CHECKPOINT_ROOT=/home/data/models/kehang-StarVLA/checkpoints/calvin
CACHE_ROOT=/home/data/datasets/kehang-CALVIN/.cache

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "[ERROR] Run this file with sbatch, not bash." >&2
  exit 40
fi

# GPU 0 is reserved for the desktop/remote graphics service.  Preserve the
# mapping created by Slurm; never replace it manually.  Abort if the allocation
# explicitly contains physical GPU 0.
ALLOCATED_GPUS="${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
if [[ -z "${ALLOCATED_GPUS}" ]]; then
  echo "[ERROR] Slurm did not expose the allocated GPU list." >&2
  exit 41
fi
if tr ',' '\n' <<<"${ALLOCATED_GPUS}" | sed 's/^gpu://; s/^[[:space:]]*//; s/[[:space:]]*$//' | grep -qx '0'; then
  echo "[ERROR] Allocation contains physical GPU 0; refusing to train." >&2
  exit 42
fi

source /home/liuchang/miniconda3/bin/activate starvla-e0

export PROJECT_ROOT MODEL_ROOT LEROBOT_ROOT CHECKPOINT_ROOT CACHE_ROOT
export STARVLA_DIR="${PROJECT_ROOT}/third_party/starvla"
export CONFIG_YAML="${PROJECT_ROOT}/configs/starvla/e0_abc_rel_calvin_scaled_90k.yaml"
export NUM_PROCESSES=2
export PER_DEVICE_BATCH_SIZE=8
export RUN_ID=e0_abc_rel

# Reproduction defaults to no external telemetry.  To enable W&B, submit with
# --export=ALL,WANDB_MODE=online,E0_WANDB_ENTITY=<your-entity>.
export WANDB_MODE="${WANDB_MODE:-disabled}"
export E0_WANDB_PROJECT="${E0_WANDB_PROJECT:-starVLA_Calvin_E0}"
export E0_WANDB_ENTITY="${E0_WANDB_ENTITY:-disabled}"

mkdir -p "${PROJECT_ROOT}/logs/e0_abc_rel" "${CACHE_ROOT}"

watchdog() {
  while sleep 5; do
    available_gb="$(free -g | awk '/Mem:/ {print $7}')"
    if [[ "${available_gb:-999}" -lt 12 ]]; then
      echo "[WATCHDOG] Available host RAM ${available_gb}GiB < 12GiB; cancelling ${SLURM_JOB_ID}." >&2
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

bash "${PROJECT_ROOT}/scripts/e0_abc_rel/train_e0_90k_generic.sh"
