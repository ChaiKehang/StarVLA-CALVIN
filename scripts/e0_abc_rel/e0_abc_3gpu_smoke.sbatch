#!/usr/bin/env bash
# E0 ABC->D CALVIN-scaled relative-action 3-GPU smoke run.
#
# This script intentionally overrides the heavier defaults in
# examples/calvin/train_files/e0_abc_rel_calvin_scaled.yaml and runs only a
# 10-step smoke test.  Do not use it for 2k/30k training without changing
# run_id, max_train_steps, save_interval and batch/accumulation deliberately.

#SBATCH --job-name=e0_abc_3gpu_smoke
#SBATCH --partition=compute
#SBATCH --gres=gpu:A6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/liuchang/kehang/488project/logs/e0_abc_rel/%x_%j.out
#SBATCH --error=/home/liuchang/kehang/488project/logs/e0_abc_rel/%x_%j.err

set -euo pipefail

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "Original CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# DIDIS Lab guide: GPU 0 is reserved for graphics/remote desktop.  Use GPU 1,2,3.
# This server's Slurm does not reliably expose a physical-GPU exclusion feature,
# so we force the compute-visible devices according to the local server guide.
export CUDA_VISIBLE_DEVICES=2,3
echo "Forced CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

source /home/liuchang/kehang/488project/scripts/e0_abc_rel/env.sh
source /home/liuchang/miniconda3/bin/activate starvla-e0

export WANDB_MODE=disabled
export HF_HOME=/home/data/datasets/kehang-CALVIN/.cache/hf
export HUGGINGFACE_HUB_CACHE=/home/data/datasets/kehang-CALVIN/.cache/hf
export TORCH_HOME=/home/data/datasets/kehang-CALVIN/.cache/torch
export WANDB_CACHE_DIR=/home/data/datasets/kehang-CALVIN/.cache/wandb
export TRITON_CACHE_DIR=/home/data/datasets/kehang-CALVIN/.cache/triton
export XDG_CACHE_HOME=/home/data/datasets/kehang-CALVIN/.cache/xdg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=4

mkdir -p \
  /home/liuchang/kehang/488project/logs/e0_abc_rel \
  /home/data/models/kehang-StarVLA/checkpoints/calvin \
  "$HF_HOME" "$TORCH_HOME" "$WANDB_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"

cd "$STARVLA_DIR"

echo "Working directory: $(pwd)"
echo "Config: examples/calvin/train_files/e0_abc_rel_calvin_scaled.yaml"
echo "Checking GPU visibility..."
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

# Refuse to start if another compute process is already occupying GPU 1/2/3.
# A little display/Xorg memory is fine; >1000 MiB means somebody is likely using it.
while IFS=',' read -r idx used total; do
  idx="$(echo "$idx" | xargs)"
  used="$(echo "$used" | xargs | sed 's/ MiB//')"
  if [[ "$idx" =~ ^[123]$ ]] && [ "$used" -gt 1000 ]; then
    echo "[PRECHECK] GPU ${idx} already uses ${used}MiB; refusing to start." >&2
    exit 43
  fi
done < <(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader)

watchdog() {
  while true; do
    avail_gb="$(free -g | awk '/Mem:/ {print $7}')"
    if [ "${avail_gb:-999}" -lt 12 ]; then
      echo "[WATCHDOG] Available RAM ${avail_gb}GB < 12GB; cancelling job ${SLURM_JOB_ID:-unknown}" >&2
      scancel "${SLURM_JOB_ID:-}" || true
      exit 99
    fi

    while IFS=',' read -r idx used; do
      idx="$(echo "$idx" | xargs)"
      used="$(echo "$used" | xargs)"
      if [[ "$idx" =~ ^[123]$ ]] && [ "$used" -gt 49000 ]; then
        echo "[WATCHDOG] GPU ${idx} memory ${used}MiB > 49000MiB; cancelling job ${SLURM_JOB_ID:-unknown}" >&2
        scancel "${SLURM_JOB_ID:-}" || true
        exit 99
      fi
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
  --config_yaml examples/calvin/train_files/e0_abc_rel_calvin_scaled.yaml \
  --framework.qwenvl.vl_hidden_dim 2560 \
  --framework.qwenvl.num_vl_layers 36 \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen3-VL-4B-Instruct \
  --datasets.vla_data.data_root_dir /home/data/datasets/kehang-CALVIN/calvin/lerobot \
  --datasets.vla_data.data_mix calvin_abc_d_sixpigs_rel_scaled \
  --datasets.vla_data.action_mode abs \
  --datasets.vla_data.per_device_batch_size 8 \
  --datasets.vla_data.video_backend torchvision_av \
  --datasets.vlm_data.per_device_batch_size 1 \
  --trainer.pretrained_checkpoint playground/Pretrained_models/starvla_qwenpi_pretrain_qwen3_4b_bridge-rt_1/checkpoints/steps_50000_pytorch_model.pt \
  --trainer.freeze_modules qwen_vl_interface \
  --trainer.loss_scale.vla 1.0 \
  --trainer.loss_scale.vlm 0.0 \
  --trainer.max_train_steps 60000 \
  --trainer.num_warmup_steps 5000 \
  --trainer.save_interval 6000 \
  --trainer.logging_frequency 20 \
  --trainer.eval_interval 1000 \
  --trainer.gradient_accumulation_steps 1 \
  --run_root_dir /home/data/models/kehang-StarVLA/checkpoints/calvin \
  --run_id e0_abc_rel_scaled_bridge_rt1_2gpu_b8_60k \
  --wandb_project starVLA_Calvin_E0 \
  --wandb_entity disabled
