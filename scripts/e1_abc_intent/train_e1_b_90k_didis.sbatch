#!/usr/bin/env bash
# DIDIS Lab Slurm wrapper for the E1-B 90k launcher.

#SBATCH --job-name=e1_b_abc_90k
#SBATCH --partition=compute
#SBATCH --gres=gpu:A6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/liuchang/kehang/488project/logs/e1_abc_intent/%x_%j.out
#SBATCH --error=/home/liuchang/kehang/488project/logs/e1_abc_intent/%x_%j.err

set -euo pipefail

export PROJECT_ROOT=/home/liuchang/kehang/488project
export STARVLA_DIR="${PROJECT_ROOT}/third_party/starvla"
export MODEL_ROOT=/home/data/models/kehang-StarVLA
export LEROBOT_ROOT=/home/data/datasets/kehang-CALVIN/calvin/lerobot
export CHECKPOINT_ROOT=/home/data/models/kehang-StarVLA/checkpoints/calvin
export CACHE_ROOT=/home/data/datasets/kehang-CALVIN/.cache
export CONFIG_YAML="${CONFIG_YAML:-${STARVLA_DIR}/examples/calvin/train_files/e1_b_abc_rel_calvin_scaled_intent125.yaml}"
export RUN_ID="${RUN_ID:-e1_b_abc_rel_scaled_intent125_h8}"
export NUM_PROCESSES="${NUM_PROCESSES:-2}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
export USE_INTENT_AUX_LOSS="${USE_INTENT_AUX_LOSS:-true}"
export ADD_INTENT_TO_TIMESTEP="${ADD_INTENT_TO_TIMESTEP:-true}"
export USE_FFN_INTENT_FILM="${USE_FFN_INTENT_FILM:-false}"
export WANDB_MODE="${WANDB_MODE:-online}"
export E1_WANDB_PROJECT="${E1_WANDB_PROJECT:-starVLA_Calvin_E1_B}"
export E1_WANDB_ENTITY="${E1_WANDB_ENTITY:-chaikehang-sjtu-hpc-center}"

mkdir -p "${PROJECT_ROOT}/logs/e1_abc_intent" "${CACHE_ROOT}"

bash "${PROJECT_ROOT}/scripts/e1_abc_intent/train_e1_b_90k_srun.sh"
