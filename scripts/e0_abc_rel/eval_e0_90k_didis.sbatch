#!/usr/bin/env bash
# DIDIS Lab one-GPU CALVIN policy-server + evaluator job.

#SBATCH --job-name=e0_eval_90k
#SBATCH --partition=compute
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/liuchang/kehang/488project/logs/e0_abc_rel/%x_%j.out
#SBATCH --error=/home/liuchang/kehang/488project/logs/e0_abc_rel/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/home/liuchang/kehang/488project
ALLOCATED_GPUS="${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${ALLOCATED_GPUS}" ]]; then
  echo "[ERROR] This script requires an sbatch GPU allocation." >&2
  exit 40
fi
if tr ',' '\n' <<<"${ALLOCATED_GPUS}" | sed 's/^gpu://; s/^[[:space:]]*//; s/[[:space:]]*$//' | grep -qx '0'; then
  echo "[ERROR] Allocation contains physical GPU 0; refusing to run." >&2
  exit 42
fi

export PROJECT_ROOT
export STARVLA_DIR="${PROJECT_ROOT}/code/starvla"
export CALVIN_DIR="${PROJECT_ROOT}/code/calvin"
export EVO_DIR="${PROJECT_ROOT}/scripts/reference/Evo-1_sixpigs"
export STARVLA_PYTHON=/home/liuchang/miniconda3/envs/starvla-e0/bin/python
export CALVIN_PYTHON=/home/liuchang/miniconda3/envs/calvin-eval/bin/python
export CKPT90K=/home/data/models/kehang-StarVLA/checkpoints/calvin/e0_abc_rel/checkpoints/steps_90000_pytorch_model.pt
export PORT="${PORT:-5694}"
export UNNORM_KEY=franka
export NUM_SEQUENCES="${NUM_SEQUENCES:-10}"
export REPLAN_STEPS="${REPLAN_STEPS:-5}"
export DEBUG="${DEBUG:-true}"
export EVAL_LOG_DIR="${EVAL_LOG_DIR:-${PROJECT_ROOT}/eval_logs/e0_abc_rel/repro_${NUM_SEQUENCES}_${SLURM_JOB_ID}}"

export HF_HOME=/home/data/datasets/kehang-CALVIN/.cache/hf
export TORCH_HOME=/home/data/datasets/kehang-CALVIN/.cache/torch
export XDG_CACHE_HOME=/home/data/datasets/kehang-CALVIN/.cache/xdg
mkdir -p "${PROJECT_ROOT}/logs/e0_abc_rel" "${EVAL_LOG_DIR}" "${XDG_CACHE_HOME}"

bash "${PROJECT_ROOT}/scripts/e0_abc_rel/eval_e0_90k_generic.sh"
