#!/usr/bin/env bash
# Canonical fresh run for factorized aligned-9 S1 (10k) + S2 (80k).
# This launcher intentionally has no resume, step-offset, or W&B-continuation path.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
STARVLA_DIR="${STARVLA_DIR:-${PROJECT_ROOT}/third_party/starvla}"
CONFIG_YAML="${CONFIG_YAML:-${STARVLA_DIR}/examples/calvin/train_files/e1_factorized_aligned9_query_s1_10k_s2_80k.yaml}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/home/liuchang/miniconda3/envs/starvla-e0/bin/accelerate}"
ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-${PROJECT_ROOT}/scripts/e1_abc_intent/deepspeed/deepspeed_zero2_grad_accum2.yaml}"

MODEL_ROOT="${MODEL_ROOT:-/home/data/models/kehang-StarVLA}"
LEROBOT_ROOT="${LEROBOT_ROOT:-/home/data/datasets/kehang-CALVIN/calvin/lerobot}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${MODEL_ROOT}/checkpoints/calvin}"
BASE_VLM="${BASE_VLM:-${MODEL_ROOT}/Qwen3-VL-4B-Instruct}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-${MODEL_ROOT}/pretrained/starvla_qwenpi_pretrain_qwen3_4B_bridge-rt_1/checkpoints/steps_50000_pytorch_model.pt}"
S0_INTENT_CHECKPOINT="${S0_INTENT_CHECKPOINT:-${CHECKPOINT_ROOT}/e1_factorized_aligned9_spatial_intent_s0_50k/checkpoints/best_intent_pytorch_model.pt}"
S0_BEST_METRICS="${S0_BEST_METRICS:-${CHECKPOINT_ROOT}/e1_factorized_aligned9_spatial_intent_s0_50k/checkpoints/best_intent_metrics.json}"
DERIVED_DATASET="${DERIVED_DATASET:-${LEROBOT_ROOT}/sixpigs1_calvin2lerobotV21_ABC_D_scnet_rel_calvin_scaled_intent_factorized_h8}"

RUN_ID="${RUN_ID:-e1_factorized_aligned9_query_s1_10k_s2_80k}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-$((20000 + $$ % 20000))}"
CACHE_ROOT="${CACHE_ROOT:-${PROJECT_ROOT}/.cache}"
CHECK_ONLY="${CHECK_ONLY:-false}"

WATCHDOG_INTERVAL_SECONDS="${WATCHDOG_INTERVAL_SECONDS:-5}"
MIN_AVAILABLE_RAM_GB="${MIN_AVAILABLE_RAM_GB:-12}"
MAX_GPU_MEMORY_MIB="${MAX_GPU_MEMORY_MIB:-49000}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
IFS=',' read -r -a VISIBLE_GPUS <<<"${CUDA_VISIBLE_DEVICES}"
if (( ${#VISIBLE_GPUS[@]} < NUM_PROCESSES || NUM_PROCESSES != 2 )); then
  echo "[ERROR] Canonical S1/S2 requires exactly two processes and two visible GPUs; got devices=${CUDA_VISIBLE_DEVICES}, processes=${NUM_PROCESSES}." >&2
  exit 2
fi
if (( PER_DEVICE_BATCH_SIZE != 4 || GRADIENT_ACCUMULATION_STEPS != 2 )); then
  echo "[ERROR] Canonical global batch must remain 4 x 2 GPUs x accumulation 2 = 16." >&2
  exit 2
fi
if [[ "${CHECK_ONLY}" != "true" && "${CHECK_ONLY}" != "false" ]]; then
  echo "[ERROR] CHECK_ONLY must be true or false." >&2
  exit 2
fi

REQUIRED_PATHS=(
  "${STARVLA_DIR}"
  "${CONFIG_YAML}"
  "${ACCELERATE_BIN}"
  "${ACCELERATE_CONFIG_FILE}"
  "${BASE_VLM}"
  "${PRETRAINED_CHECKPOINT}"
  "${S0_INTENT_CHECKPOINT}"
  "${S0_BEST_METRICS}"
  "${DERIVED_DATASET}/meta/factorized_intent_config.json"
  "${DERIVED_DATASET}/meta/splits/factorized_seed42_train.json"
)
for required in "${REQUIRED_PATHS[@]}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[ERROR] Required path does not exist: ${required}" >&2
    exit 2
  fi
done

TARGET_RUN_DIR="${CHECKPOINT_ROOT}/${RUN_ID}"
if [[ -d "${TARGET_RUN_DIR}" ]] && [[ -n "$(find "${TARGET_RUN_DIR}" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "[ERROR] Fresh-only S1/S2 refuses to reuse non-empty run directory: ${TARGET_RUN_DIR}" >&2
  echo "Choose a new RUN_ID; do not resume or overwrite the canonical run." >&2
  exit 3
fi

export WANDB_MODE="${WANDB_MODE:-online}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${CACHE_ROOT}/wandb}"
export WANDB_DIR="${WANDB_DIR:-${CACHE_ROOT}/wandb}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

WATCHDOG_GPUS="${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES}}"
stop_training() {
  local train_pid="$1"
  if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v scancel >/dev/null 2>&1; then
    scancel "${SLURM_JOB_ID}" || true
  else
    kill -TERM "${train_pid}" 2>/dev/null || true
  fi
}
watchdog() {
  local train_pid="$1"
  while kill -0 "${train_pid}" 2>/dev/null; do
    sleep "${WATCHDOG_INTERVAL_SECONDS}"
    kill -0 "${train_pid}" 2>/dev/null || return 0
    local available_gb
    available_gb="$(free -g | awk '/Mem:/ {print $7}')"
    if [[ "${available_gb:-999}" -lt "${MIN_AVAILABLE_RAM_GB}" ]]; then
      echo "[WATCHDOG] Available RAM ${available_gb}GiB is below ${MIN_AVAILABLE_RAM_GB}GiB." >&2
      stop_training "${train_pid}"
      return 99
    fi
    while IFS= read -r gpu; do
      gpu="$(xargs <<<"${gpu}")"
      gpu="${gpu#gpu:}"
      [[ "${gpu}" =~ ^[0-9]+$ ]] || continue
      local used
      used="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || true)"
      if [[ "${used:-0}" -gt "${MAX_GPU_MEMORY_MIB}" ]]; then
        echo "[WATCHDOG] GPU ${gpu} uses ${used}MiB, above ${MAX_GPU_MEMORY_MIB}MiB." >&2
        stop_training "${train_pid}"
        return 99
      fi
    done < <(tr ',' '\n' <<<"${WATCHDOG_GPUS}")
  done
}

echo "Canonical stage: S1 10k + S2 80k continuous aligned-9 main run"
echo "Config=${CONFIG_YAML}"
echo "Action initialization=${PRETRAINED_CHECKPOINT}"
echo "Intent initialization=${S0_INTENT_CHECKPOINT}"
echo "Output=${TARGET_RUN_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, processes=${NUM_PROCESSES}"
echo "Steps=90000, S1=10000, S2=80000, warmup=5000"
echo "Batch=${PER_DEVICE_BATCH_SIZE} x ${NUM_PROCESSES} GPUs x accumulation ${GRADIENT_ACCUMULATION_STEPS} = 16"
echo "Resume=disabled; step offsets=0; W&B continuation=disabled"

if [[ "${CHECK_ONLY}" == "true" ]]; then
  echo "CHECK_ONLY=true: canonical S1/S2 configuration validation passed."
  exit 0
fi

mkdir -p "${CHECKPOINT_ROOT}" "${HF_HOME}" "${TORCH_HOME}" \
  "${WANDB_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${XDG_CACHE_HOME}"
cd "${STARVLA_DIR}"
"${ACCELERATE_BIN}" launch \
  --config_file "${ACCELERATE_CONFIG_FILE}" \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --framework.qwenvl.base_vlm "${BASE_VLM}" \
  --framework.intent.stage1_steps 10000 \
  --datasets.vla_data.data_root_dir "${LEROBOT_ROOT}" \
  --datasets.vla_data.per_device_batch_size 4 \
  --trainer.pretrained_checkpoint "${PRETRAINED_CHECKPOINT}" \
  --trainer.intent_pretrained_checkpoint "${S0_INTENT_CHECKPOINT}" \
  --trainer.is_resume false \
  --trainer.max_train_steps 90000 \
  --trainer.scheduler_total_steps 90000 \
  --trainer.scheduler_step_offset 0 \
  --trainer.wandb_step_offset 0 \
  --trainer.num_warmup_steps 5000 \
  --trainer.gradient_accumulation_steps 2 \
  --trainer.save_interval 5000 \
  --trainer.eval_interval 1000 \
  --trainer.repeated_diffusion_steps 16 \
  --run_root_dir "${CHECKPOINT_ROOT}" \
  --run_id "${RUN_ID}" \
  --wandb_project "${WANDB_PROJECT:-starVLA_Calvin_E1_Factorized_Aligned9_Main}" \
  --wandb_entity "${WANDB_ENTITY:-chaikehang-sjtu-hpc-center}" &

TRAINING_PID=$!
watchdog "${TRAINING_PID}" &
WATCHDOG_PID=$!
cleanup() {
  kill "${WATCHDOG_PID}" 2>/dev/null || true
  if kill -0 "${TRAINING_PID}" 2>/dev/null; then
    kill -TERM "${TRAINING_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM
set +e
wait "${TRAINING_PID}"
TRAINING_STATUS=$?
set -e
kill "${WATCHDOG_PID}" 2>/dev/null || true
wait "${WATCHDOG_PID}" 2>/dev/null || true
exit "${TRAINING_STATUS}"
