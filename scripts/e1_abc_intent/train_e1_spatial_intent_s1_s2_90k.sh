#!/usr/bin/env bash
# Run inside an already allocated one- or two-GPU node/tmux shell.
# S1 (default 10k) + S2 (remaining steps) share one continuous 90k run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
STARVLA_DIR="${STARVLA_DIR:-${PROJECT_ROOT}/third_party/starvla}"
CONFIG_YAML="${CONFIG_YAML:-${STARVLA_DIR}/examples/calvin/train_files/e1_spatial_intent_s1_s2_90k.yaml}"

MODEL_ROOT="${MODEL_ROOT:-/home/data/models/kehang-StarVLA}"
LEROBOT_ROOT="${LEROBOT_ROOT:-/home/data/datasets/kehang-CALVIN/calvin/lerobot}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${MODEL_ROOT}/checkpoints/calvin}"
BASE_VLM="${BASE_VLM:-${MODEL_ROOT}/Qwen3-VL-4B-Instruct}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-${MODEL_ROOT}/pretrained/starvla_qwenpi_pretrain_qwen3_4B_bridge-rt_1/checkpoints/steps_50000_pytorch_model.pt}"

S0_STEPS="${S0_STEPS:-30000}"
S0_RUN_ID="${S0_RUN_ID:-e1_spatial_intent_s0_${S0_STEPS}}"
S0_INTENT_CHECKPOINT="${S0_INTENT_CHECKPOINT:-${CHECKPOINT_ROOT}/${S0_RUN_ID}/checkpoints/steps_${S0_STEPS}_pytorch_model.pt}"
STAGE1_STEPS="${STAGE1_STEPS:-10000}"
MAIN_MAX_STEPS="${MAIN_MAX_STEPS:-90000}"
RUN_ID="${RUN_ID:-e1_spatial_intent_query_ffn_s1_s2_90k}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
CACHE_ROOT="${CACHE_ROOT:-${PROJECT_ROOT}/.cache}"
WATCHDOG_INTERVAL_SECONDS="${WATCHDOG_INTERVAL_SECONDS:-5}"
MIN_AVAILABLE_RAM_GB="${MIN_AVAILABLE_RAM_GB:-12}"
MAX_GPU_MEMORY_MIB="${MAX_GPU_MEMORY_MIB:-49000}"

if [[ "${MAIN_MAX_STEPS}" != "90000" ]]; then
  echo "[ERROR] MAIN_MAX_STEPS must remain 90000 for the matched E0 comparison." >&2
  exit 2
fi
if (( STAGE1_STEPS <= 0 || STAGE1_STEPS >= MAIN_MAX_STEPS )); then
  echo "[ERROR] STAGE1_STEPS must be within (0, ${MAIN_MAX_STEPS}), got ${STAGE1_STEPS}." >&2
  exit 2
fi
# Respect Slurm's visibility when present. Outside Slurm, default to physical 2,3.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
IFS=',' read -r -a VISIBLE_GPUS <<<"${CUDA_VISIBLE_DEVICES}"
VISIBLE_GPU_COUNT=${#VISIBLE_GPUS[@]}
if (( VISIBLE_GPU_COUNT < 1 || VISIBLE_GPU_COUNT > 2 )); then
  echo "[ERROR] One or two visible GPUs are allowed, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}." >&2
  exit 2
fi
NUM_PROCESSES="${NUM_PROCESSES:-${VISIBLE_GPU_COUNT}}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-$((20000 + $$ % 20000))}"
if (( NUM_PROCESSES < 1 || NUM_PROCESSES > 2 || NUM_PROCESSES > VISIBLE_GPU_COUNT )); then
  echo "[ERROR] NUM_PROCESSES must be 1 or 2 and cannot exceed ${VISIBLE_GPU_COUNT} visible GPU(s); got ${NUM_PROCESSES}." >&2
  exit 2
fi

# Prefer physical IDs reported by Slurm for nvidia-smi; otherwise use the
# caller-selected CUDA-visible IDs. Non-numeric UUID entries are skipped.
WATCHDOG_GPUS="${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES}}"

stop_training() {
  local train_pid="$1"
  if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v scancel >/dev/null 2>&1; then
    echo "[WATCHDOG] Cancelling Slurm job ${SLURM_JOB_ID}." >&2
    scancel "${SLURM_JOB_ID}" || true
  else
    echo "[WATCHDOG] No Slurm job ID; terminating launcher PID ${train_pid}." >&2
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
      echo "[WATCHDOG] Available RAM ${available_gb}GiB < ${MIN_AVAILABLE_RAM_GB}GiB." >&2
      stop_training "${train_pid}"
      return 99
    fi

    while IFS= read -r gpu; do
      gpu="$(xargs <<<"${gpu}")"
      gpu="${gpu#gpu:}"
      [[ "${gpu}" =~ ^[0-9]+$ ]] || continue
      local used
      if ! used="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)"; then
        echo "[WATCHDOG] Could not query GPU ${gpu}; retrying on the next interval." >&2
        continue
      fi
      if [[ "${used:-0}" -gt "${MAX_GPU_MEMORY_MIB}" ]]; then
        echo "[WATCHDOG] GPU ${gpu} uses ${used}MiB > ${MAX_GPU_MEMORY_MIB}MiB." >&2
        stop_training "${train_pid}"
        return 99
      fi
    done < <(tr ',' '\n' <<<"${WATCHDOG_GPUS}")
  done
}

for required in "${STARVLA_DIR}" "${CONFIG_YAML}" "${LEROBOT_ROOT}" "${BASE_VLM}" \
  "${PRETRAINED_CHECKPOINT}" "${S0_INTENT_CHECKPOINT}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[ERROR] Required path does not exist: ${required}" >&2
    if [[ "${required}" == "${S0_INTENT_CHECKPOINT}" ]]; then
      echo "Set S0_INTENT_CHECKPOINT to the chosen 10k-30k S0 checkpoint." >&2
    fi
    exit 2
  fi
done

TARGET_RUN_DIR="${CHECKPOINT_ROOT}/${RUN_ID}"
if [[ -d "${TARGET_RUN_DIR}" ]] && [[ -n "$(find "${TARGET_RUN_DIR}" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "[ERROR] ${TARGET_RUN_DIR} already exists and is not empty; choose a new RUN_ID." >&2
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

mkdir -p "${CHECKPOINT_ROOT}" "${HF_HOME}" "${TORCH_HOME}" \
  "${WANDB_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${XDG_CACHE_HOME}"

echo "Stage=S1+S2 continuous main run"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Processes=${NUM_PROCESSES} (maximum 2), watchdog GPUs=${WATCHDOG_GPUS}"
echo "Accelerate main process port=${MAIN_PROCESS_PORT}"
echo "Config=${CONFIG_YAML}"
echo "Action initialization=${PRETRAINED_CHECKPOINT}"
echo "Intent overlay=${S0_INTENT_CHECKPOINT}"
echo "Output=${TARGET_RUN_DIR}"
echo "S1=${STAGE1_STEPS}, S2=$((MAIN_MAX_STEPS - STAGE1_STEPS)), total=${MAIN_MAX_STEPS}"
echo "Conditioning=timestep:false, query_film:true, ffn_film:true"

cd "${STARVLA_DIR}"
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --framework.qwenvl.base_vlm "${BASE_VLM}" \
  --framework.intent.stage1_steps "${STAGE1_STEPS}" \
  --datasets.vla_data.data_root_dir "${LEROBOT_ROOT}" \
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --trainer.pretrained_checkpoint "${PRETRAINED_CHECKPOINT}" \
  --trainer.intent_pretrained_checkpoint "${S0_INTENT_CHECKPOINT}" \
  --trainer.is_resume false \
  --trainer.max_train_steps "${MAIN_MAX_STEPS}" \
  --trainer.num_warmup_steps 5000 \
  --trainer.save_interval 5000 \
  --trainer.eval_interval 1000 \
  --trainer.repeated_diffusion_steps 16 \
  --run_root_dir "${CHECKPOINT_ROOT}" \
  --run_id "${RUN_ID}" \
  --wandb_project "${WANDB_PROJECT:-starVLA_Calvin_E1_Spatial_Intent_Main}" \
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
