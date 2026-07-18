#!/usr/bin/env bash
#SBATCH --job-name=libero_3gpu_smoke
#SBATCH --partition=compute
#SBATCH --gres=gpu:A6000:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/home/liuchang/kehang/488project/logs/libero_smoke/%x_%j.out
#SBATCH --error=/home/liuchang/kehang/488project/logs/libero_smoke/%x_%j.err

set -euo pipefail

echo "Original SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "Original CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# DIDIS Lab guide: GPU 0 is reserved for graphics rendering; training must use GPU 1,2,3.
export CUDA_VISIBLE_DEVICES=1,2,3
echo "Forced CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES according to DIDIS guide"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

source /home/liuchang/kehang/488project/scripts/e0_abc_rel/env.sh
source /home/liuchang/miniconda3/bin/activate starvla-e0

export WANDB_MODE=disabled
export HF_HOME=/home/data/datasets/kehang-CALVIN/.cache/hf
export HUGGINGFACE_HUB_CACHE=/home/data/datasets/kehang-CALVIN/.cache/hf
export TORCH_HOME=/home/data/datasets/kehang-CALVIN/.cache/torch
export WANDB_CACHE_DIR=/home/data/datasets/kehang-CALVIN/.cache/wandb
export TRITON_CACHE_DIR=/home/data/datasets/kehang-CALVIN/.cache/triton
export XDG_CACHE_HOME=/home/data/datasets/kehang-CALVIN/.cache/xdg
mkdir -p /home/liuchang/kehang/488project/logs/libero_smoke \
  /home/liuchang/kehang/488project/runs/libero_smoke \
  "$HF_HOME" "$TORCH_HOME" "$WANDB_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"

cd "$STARVLA_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=4

watchdog() {
  while true; do
    avail_gb=$(free -g | awk '/Mem:/ {print $7}')

    if [ "${avail_gb:-999}" -lt 12 ]; then
      echo "[WATCHDOG] Available RAM ${avail_gb}GB < 12GB; cancelling job ${SLURM_JOB_ID:-unknown}" >&2
      scancel "${SLURM_JOB_ID}"
      exit 99
    fi

    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | while IFS=',' read -r idx used; do
      idx=$(echo "$idx" | xargs)
      used=$(echo "$used" | xargs)

      if [ "$idx" != "0" ] && [ "$used" -gt 49000 ]; then
        echo "[WATCHDOG] GPU ${idx} memory ${used}MiB > 48000MiB; cancelling job ${SLURM_JOB_ID:-unknown}" >&2
        scancel "${SLURM_JOB_ID}"
        exit 99
      fi
    done

    sleep 1
  done
}

watchdog &
WATCHDOG_PID=$!
trap 'kill "$WATCHDOG_PID" 2>/dev/null || true' EXIT

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 3 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/LIBERO/train_files/starvla_cotrain_libero.yaml \
  --framework.action_model.diffusion_model_cfg.output_dim 1024 \
  --framework.name QwenPI_v3 \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen3-VL-4B-Instruct \
  --trainer.pretrained_checkpoint playground/Pretrained_models/starvla_qwenpi_pretrain_qwen3_4b_bridge-rt_1/checkpoints/steps_50000_pytorch_model.pt \
  --datasets.vla_data.data_root_dir playground/Datasets/LEROBOT_LIBERO_DATA \
  --datasets.vla_data.data_mix libero_all \
  --datasets.vla_data.per_device_batch_size 12 \
  --datasets.vlm_data.per_device_batch_size 3 \
  --datasets.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules "qwen_vl_interface" \
  --trainer.max_train_steps 20 \
  --trainer.save_interval 200 \
  --trainer.logging_frequency 1 \
  --trainer.eval_interval 100 \
  --trainer.gradient_accumulation_steps 3 \
  --run_root_dir /home/liuchang/kehang/488project/runs/libero_smoke \
  --run_id starvla_qwenpi_3gpu_20step \
  --wandb_project starVLA_Libero \
  --wandb_entity disabled
