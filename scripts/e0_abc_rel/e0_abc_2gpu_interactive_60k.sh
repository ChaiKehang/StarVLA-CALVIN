#!/usr/bin/env bash
# Run the independent 90k -> 180k low-LR retraining phase inside an existing
# two-GPU Slurm interactive allocation, e.g.
#   srun --partition=compute --gres=gpu:A6000:2 -c 16 --mem=80G --time=2-00:00:00 --pty bash
#
# This is NOT an sbatch script.  It expects SLURM_JOB_ID to already exist.

set -euo pipefail

PROJECT=/home/liuchang/kehang/488project
LOG_DIR="$PROJECT/logs/e0_abc_rel"
RETRAIN_RUN_ID=e0_abc_rel_retrain_from_90k_90k
RUN_ROOT=/home/data/models/kehang-StarVLA/checkpoints/calvin
CONFIG_YAML_REL=examples/calvin/train_files/e0_abc_rel_calvin_scaled_retrain_30000steps.yaml
SOURCE_CHECKPOINT="$RUN_ROOT/e0_abc_rel/checkpoints/steps_90000_pytorch_model.pt"
TARGET_RUN_DIR="$RUN_ROOT/$RETRAIN_RUN_ID"
LOG_FILE="$LOG_DIR/${RETRAIN_RUN_ID}_interactive_${SLURM_JOB_ID:-no_slurm}_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR" "$RUN_ROOT"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[$(date '+%F %T')] Starting E0 ABC-D 90k -> 180k low-LR retraining phase"
echo "Log file: $LOG_FILE"

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "[ERROR] SLURM_JOB_ID is unset. This script should be run inside an srun --pty bash interactive allocation." >&2
  echo "Example:" >&2
  echo "  srun --partition=compute --gres=gpu:A6000:2 -c 16 --mem=80G --time=2-00:00:00 --pty bash" >&2
  exit 2
fi

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "Original CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# DIDIS Lab guide: avoid GPU 0.  This two-GPU run uses physical GPU 2,3 by default.
# Override only if you have coordinated a different allocation:
#   E0_CUDA_VISIBLE_DEVICES=1,2 bash e0_abc_2gpu_interactive_60k.sh
export CUDA_VISIBLE_DEVICES="${E0_CUDA_VISIBLE_DEVICES:-2,3}"
echo "Forced CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

source "$PROJECT/scripts/e0_abc_rel/env.sh"
source /home/liuchang/miniconda3/bin/activate starvla-e0

# env.sh defines the old E0 RUN_ID for general project paths.  This script uses
# a distinct run id so the retraining checkpoints can never mix with the 90k/
# 120k experiment checkpoints.
export RUN_ID="$RETRAIN_RUN_ID"
CONFIG_YAML="$STARVLA_DIR/$CONFIG_YAML_REL"

export WANDB_MODE=online
export WANDB_PROJECT=starVLA_Calvin_E0
export WANDB_ENTITY=chaikehang-sjtu-hpc-center
export WANDB_CACHE_DIR=/home/data/datasets/kehang-CALVIN/.cache/wandb
export WANDB_DIR=/home/data/datasets/kehang-CALVIN/.cache/wandb
export HF_HOME=/home/data/datasets/kehang-CALVIN/.cache/hf
export HUGGINGFACE_HUB_CACHE=/home/data/datasets/kehang-CALVIN/.cache/hf
export TORCH_HOME=/home/data/datasets/kehang-CALVIN/.cache/torch
export TRITON_CACHE_DIR=/home/data/datasets/kehang-CALVIN/.cache/triton
export XDG_CACHE_HOME=/home/data/datasets/kehang-CALVIN/.cache/xdg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=4

mkdir -p "$HF_HOME" "$TORCH_HOME" "$WANDB_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"

cd "$STARVLA_DIR"

echo "Working directory: $(pwd)"
echo "Conda env: ${CONDA_DEFAULT_ENV:-unset}"
echo "Python: $(which python)"
echo "Retraining config: ${CONFIG_YAML}"
echo "Target run id: ${RETRAIN_RUN_ID}"
echo "Source 90k checkpoint: ${SOURCE_CHECKPOINT}"
if [ ! -f "$CONFIG_YAML" ]; then
  echo "[ERROR] Retraining YAML is missing: $CONFIG_YAML" >&2
  exit 44
fi
if [ ! -f "$SOURCE_CHECKPOINT" ]; then
  echo "[ERROR] Source 90k checkpoint is missing: $SOURCE_CHECKPOINT" >&2
  exit 44
fi
CONFIG_RUN_ID="$(python -c 'from omegaconf import OmegaConf; import sys; print(OmegaConf.load(sys.argv[1]).run_id)' "$CONFIG_YAML")"
CONFIG_SOURCE_CHECKPOINT="$(python -c 'from omegaconf import OmegaConf; import sys; print(OmegaConf.load(sys.argv[1]).trainer.pretrained_checkpoint)' "$CONFIG_YAML")"
if [ "$CONFIG_RUN_ID" != "$RETRAIN_RUN_ID" ]; then
  echo "[ERROR] YAML run_id ($CONFIG_RUN_ID) does not match script target ($RETRAIN_RUN_ID)." >&2
  exit 46
fi
if [ "$CONFIG_SOURCE_CHECKPOINT" != "$SOURCE_CHECKPOINT" ]; then
  echo "[ERROR] YAML pretrained_checkpoint does not match the intended 90k source checkpoint." >&2
  echo "        YAML: $CONFIG_SOURCE_CHECKPOINT" >&2
  echo "        Script: $SOURCE_CHECKPOINT" >&2
  exit 47
fi
if compgen -G "$TARGET_RUN_DIR/checkpoints/steps_*_pytorch_model.pt" > /dev/null; then
  echo "[ERROR] Target run directory already contains checkpoints: $TARGET_RUN_DIR" >&2
  echo "        Refusing to overwrite a retraining phase. Use a new run_id/YAML to start a separate run." >&2
  exit 45
fi
echo "Checking Slurm allocation:"
squeue -j "$SLURM_JOB_ID" || true
echo "Checking GPU visibility:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

IFS=',' read -r -a GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"

# Refuse to start if selected physical GPUs already have large memory usage.
while IFS=',' read -r idx used total; do
  idx="$(echo "$idx" | xargs)"
  used="$(echo "$used" | xargs | sed 's/ MiB//')"
  for want in "${GPU_LIST[@]}"; do
    want="$(echo "$want" | xargs)"
    if [ "$idx" = "$want" ] && [ "$used" -gt 1000 ]; then
      echo "[PRECHECK] GPU ${idx} already uses ${used}MiB; refusing to start." >&2
      exit 43
    fi
  done
done < <(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader)

watchdog() {
  while true; do
    avail_gb="$(free -g | awk '/Mem:/ {print $7}')"
    if [ "${avail_gb:-999}" -lt 12 ]; then
      echo "[WATCHDOG] Available RAM ${avail_gb}GB < 12GB; cancelling interactive job ${SLURM_JOB_ID}" >&2
      scancel "$SLURM_JOB_ID" || true
      exit 99
    fi

    while IFS=',' read -r idx used; do
      idx="$(echo "$idx" | xargs)"
      used="$(echo "$used" | xargs)"
      for want in "${GPU_LIST[@]}"; do
        want="$(echo "$want" | xargs)"
        if [ "$idx" = "$want" ] && [ "$used" -gt 49000 ]; then
          echo "[WATCHDOG] GPU ${idx} memory ${used}MiB > 49000MiB; cancelling interactive job ${SLURM_JOB_ID}" >&2
          scancel "$SLURM_JOB_ID" || true
          exit 99
        fi
      done
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)

    sleep 2
  done
}

watchdog &
WATCHDOG_PID=$!
trap 'kill "$WATCHDOG_PID" 2>/dev/null || true' EXIT

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 2 \
  starVLA/training/train_starvla.py \
  --config_yaml "$CONFIG_YAML_REL"

echo "[$(date '+%F %T')] Training command finished"
