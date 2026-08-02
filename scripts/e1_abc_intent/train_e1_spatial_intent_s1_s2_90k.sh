#!/usr/bin/env bash
# Run inside an already allocated one- or two-GPU node/tmux shell.
# S1 (default 15k) + S2 (remaining steps) share one continuous 90k run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
STARVLA_DIR="${STARVLA_DIR:-${PROJECT_ROOT}/third_party/starvla}"
CONFIG_YAML="${CONFIG_YAML:-${STARVLA_DIR}/examples/calvin/train_files/e1_spatial_intent_s1_s2_90k.yaml}"
ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-${STARVLA_DIR}/starVLA/config/deepseeds/deepspeed_zero2.yaml}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/home/liuchang/miniconda3/envs/starvla-e0/bin/accelerate}"

MODEL_ROOT="${MODEL_ROOT:-/home/data/models/kehang-StarVLA}"
LEROBOT_ROOT="${LEROBOT_ROOT:-/home/data/datasets/kehang-CALVIN/calvin/lerobot}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${MODEL_ROOT}/checkpoints/calvin}"
BASE_VLM="${BASE_VLM:-${MODEL_ROOT}/Qwen3-VL-4B-Instruct}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-${MODEL_ROOT}/pretrained/starvla_qwenpi_pretrain_qwen3_4B_bridge-rt_1/checkpoints/steps_50000_pytorch_model.pt}"

S0_STEPS="${S0_STEPS:-30000}"
S0_RUN_ID="${S0_RUN_ID:-e1_spatial_intent_s0_${S0_STEPS}}"
S0_INTENT_CHECKPOINT="${S0_INTENT_CHECKPOINT:-${CHECKPOINT_ROOT}/${S0_RUN_ID}/checkpoints/steps_${S0_STEPS}_pytorch_model.pt}"
STAGE1_STEPS="${STAGE1_STEPS:-15000}"
MAIN_MAX_STEPS="${MAIN_MAX_STEPS:-90000}"
ACTION_STEP_OFFSET="${ACTION_STEP_OFFSET:-0}"
RUN_ID="${RUN_ID:-e1_spatial_intent_query_ffn_s1_s2_90k}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
RESUME_STEP="${RESUME_STEP:-}"
RESUME_TRAINING_STATE="${RESUME_TRAINING_STATE:-}"
RESUME_IN_PLACE="${RESUME_IN_PLACE:-false}"
SCHEDULER_TOTAL_STEPS="${SCHEDULER_TOTAL_STEPS:-$((ACTION_STEP_OFFSET + MAIN_MAX_STEPS))}"
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-5000}"
WANDB_LOG_AFTER_STEP_OVERRIDE="${WANDB_LOG_AFTER_STEP_OVERRIDE:-}"
RECOVERED_FROM_PHASE_STEP_OVERRIDE="${RECOVERED_FROM_PHASE_STEP_OVERRIDE:-}"
TRAINING_PHASE_ID_OVERRIDE="${TRAINING_PHASE_ID_OVERRIDE:-}"
CHECK_ONLY="${CHECK_ONLY:-false}"
CACHE_ROOT="${CACHE_ROOT:-${PROJECT_ROOT}/.cache}"
WATCHDOG_INTERVAL_SECONDS="${WATCHDOG_INTERVAL_SECONDS:-5}"
MIN_AVAILABLE_RAM_GB="${MIN_AVAILABLE_RAM_GB:-12}"
MAX_GPU_MEMORY_MIB="${MAX_GPU_MEMORY_MIB:-49000}"

if (( ACTION_STEP_OFFSET < 0 )); then
  echo "[ERROR] ACTION_STEP_OFFSET must be non-negative, got ${ACTION_STEP_OFFSET}." >&2
  exit 2
fi
if (( MAIN_MAX_STEPS + ACTION_STEP_OFFSET != 90000 )); then
  echo "[ERROR] ACTION_STEP_OFFSET + MAIN_MAX_STEPS must equal 90000 for the matched E0 comparison; got ${ACTION_STEP_OFFSET} + ${MAIN_MAX_STEPS}." >&2
  exit 2
fi
if (( STAGE1_STEPS <= 0 || STAGE1_STEPS >= MAIN_MAX_STEPS )); then
  echo "[ERROR] STAGE1_STEPS must be within (0, ${MAIN_MAX_STEPS}), got ${STAGE1_STEPS}." >&2
  exit 2
fi
if (( PER_DEVICE_BATCH_SIZE <= 0 || GRADIENT_ACCUMULATION_STEPS <= 0 )); then
  echo "[ERROR] Batch size and gradient accumulation must be positive; got batch=${PER_DEVICE_BATCH_SIZE}, accumulation=${GRADIENT_ACCUMULATION_STEPS}." >&2
  exit 2
fi
if (( SCHEDULER_TOTAL_STEPS <= 0 || NUM_WARMUP_STEPS < 0 )); then
  echo "[ERROR] Scheduler steps must be positive and warmup non-negative; got total=${SCHEDULER_TOTAL_STEPS}, warmup=${NUM_WARMUP_STEPS}." >&2
  exit 2
fi
if [[ -n "${RESUME_CHECKPOINT}" || -n "${RESUME_STEP}" ]]; then
  if [[ -z "${RESUME_CHECKPOINT}" || -z "${RESUME_STEP}" ]]; then
    echo "[ERROR] RESUME_CHECKPOINT and RESUME_STEP must be set together." >&2
    exit 2
  fi
  if [[ ! "${RESUME_STEP}" =~ ^[0-9]+$ ]] || (( RESUME_STEP <= 0 || RESUME_STEP >= MAIN_MAX_STEPS )); then
    echo "[ERROR] RESUME_STEP must be an integer within (0, ${MAIN_MAX_STEPS}), got ${RESUME_STEP}." >&2
    exit 2
  fi
  if [[ ! -f "${RESUME_CHECKPOINT}" ]]; then
    echo "[ERROR] Resume checkpoint does not exist: ${RESUME_CHECKPOINT}" >&2
    exit 2
  fi
  LEGACY_WEIGHT_RESUME=true
else
  LEGACY_WEIGHT_RESUME=false
fi
if [[ -n "${RESUME_TRAINING_STATE}" ]]; then
  if [[ ! -d "${RESUME_TRAINING_STATE}" || ! -f "${RESUME_TRAINING_STATE}/trainer_state.json" ]]; then
    echo "[ERROR] Full training state is incomplete or missing: ${RESUME_TRAINING_STATE}" >&2
    exit 2
  fi
  FULL_STATE_RESUME=true
else
  FULL_STATE_RESUME=false
fi
if [[ "${LEGACY_WEIGHT_RESUME}" == "true" && "${FULL_STATE_RESUME}" == "true" ]]; then
  echo "[ERROR] Use either RESUME_CHECKPOINT/RESUME_STEP or RESUME_TRAINING_STATE, not both." >&2
  exit 2
fi
if [[ "${RESUME_IN_PLACE}" != "true" && "${RESUME_IN_PLACE}" != "false" ]]; then
  echo "[ERROR] RESUME_IN_PLACE must be true or false, got ${RESUME_IN_PLACE}." >&2
  exit 2
fi
if [[ "${CHECK_ONLY}" != "true" && "${CHECK_ONLY}" != "false" ]]; then
  echo "[ERROR] CHECK_ONLY must be true or false, got ${CHECK_ONLY}." >&2
  exit 2
fi
if [[ "${RESUME_IN_PLACE}" == "true" || "${LEGACY_WEIGHT_RESUME}" == "true" || "${FULL_STATE_RESUME}" == "true" ]]; then
  IS_RESUME=true
else
  IS_RESUME=false
fi
for integer_override in \
  "${WANDB_LOG_AFTER_STEP_OVERRIDE}" \
  "${RECOVERED_FROM_PHASE_STEP_OVERRIDE}" \
  "${TRAINING_PHASE_ID_OVERRIDE}"; do
  if [[ -n "${integer_override}" && ! "${integer_override}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Recovery step/phase overrides must be non-negative integers." >&2
    exit 2
  fi
done
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

REQUIRED_PATHS=(
  "${STARVLA_DIR}"
  "${CONFIG_YAML}"
  "${ACCELERATE_CONFIG_FILE}"
  "${ACCELERATE_BIN}"
  "${LEROBOT_ROOT}"
  "${BASE_VLM}"
)
if [[ "${IS_RESUME}" == "false" ]]; then
  REQUIRED_PATHS+=("${PRETRAINED_CHECKPOINT}" "${S0_INTENT_CHECKPOINT}")
fi

for required in "${REQUIRED_PATHS[@]}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[ERROR] Required path does not exist: ${required}" >&2
    if [[ "${required}" == "${S0_INTENT_CHECKPOINT}" ]]; then
      echo "Set S0_INTENT_CHECKPOINT to the chosen S0 checkpoint." >&2
    fi
    exit 2
  fi
done

TARGET_RUN_DIR="${CHECKPOINT_ROOT}/${RUN_ID}"
TARGET_RUN_NONEMPTY=false
if [[ -d "${TARGET_RUN_DIR}" ]] && [[ -n "$(find "${TARGET_RUN_DIR}" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  TARGET_RUN_NONEMPTY=true
fi
if [[ "${TARGET_RUN_NONEMPTY}" == "true" && "${RESUME_IN_PLACE}" != "true" ]]; then
  echo "[ERROR] ${TARGET_RUN_DIR} already exists and is not empty; choose a new RUN_ID." >&2
  exit 3
fi
if [[ "${RESUME_IN_PLACE}" == "true" && "${LEGACY_WEIGHT_RESUME}" == "false" && "${FULL_STATE_RESUME}" == "false" ]]; then
  shopt -s nullglob
  FULL_STATE_MARKERS=(
    "${TARGET_RUN_DIR}"/checkpoints/steps_*_training_state/trainer_state.json
  )
  shopt -u nullglob
  if (( ${#FULL_STATE_MARKERS[@]} == 0 )); then
    echo "[ERROR] No complete full training state was found under ${TARGET_RUN_DIR}/checkpoints." >&2
    echo "A full-state resume requires a steps_*_training_state/trainer_state.json marker." >&2
    exit 3
  fi
fi
if [[ "${RESUME_IN_PLACE}" == "true" && "${LEGACY_WEIGHT_RESUME}" == "true" ]]; then
  EXPECTED_RESUME_CHECKPOINT="${TARGET_RUN_DIR}/checkpoints/steps_${RESUME_STEP}_pytorch_model.pt"
  if [[ ! -f "${EXPECTED_RESUME_CHECKPOINT}" ]]; then
    echo "[ERROR] In-place resume checkpoint is missing: ${EXPECTED_RESUME_CHECKPOINT}" >&2
    exit 3
  fi
  if [[ "$(readlink -f "${EXPECTED_RESUME_CHECKPOINT}")" != "$(readlink -f "${RESUME_CHECKPOINT}")" ]]; then
    echo "[ERROR] RESUME_CHECKPOINT does not match the selected checkpoint inside ${TARGET_RUN_DIR}." >&2
    exit 3
  fi
elif [[ "${LEGACY_WEIGHT_RESUME}" == "true" ]]; then
  RESUME_LINK="${TARGET_RUN_DIR}/checkpoints/steps_${RESUME_STEP}_pytorch_model.pt"
  mkdir -p "$(dirname "${RESUME_LINK}")"
  ln -s "${RESUME_CHECKPOINT}" "${RESUME_LINK}"
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
echo "Accelerate config=${ACCELERATE_CONFIG_FILE}"
echo "Action initialization=${PRETRAINED_CHECKPOINT}"
echo "Intent overlay=${S0_INTENT_CHECKPOINT}"
echo "Output=${TARGET_RUN_DIR}"
echo "Prior E0 Action steps=${ACTION_STEP_OFFSET}"
echo "S1=${STAGE1_STEPS}, S2=$((MAIN_MAX_STEPS - STAGE1_STEPS)), new steps=${MAIN_MAX_STEPS}, equivalent total=$((ACTION_STEP_OFFSET + MAIN_MAX_STEPS))"
echo "Scheduler local steps=${SCHEDULER_TOTAL_STEPS}, warmup=${NUM_WARMUP_STEPS}"
if [[ "${IS_RESUME}" == "true" ]]; then
  if [[ "${LEGACY_WEIGHT_RESUME}" == "true" ]]; then
    echo "Resume=legacy model weights at global step ${RESUME_STEP} from ${RESUME_CHECKPOINT}"
    echo "Resume note=no full state was supplied; AdamW moments start fresh unless a newer complete training-state directory is found"
  elif [[ "${FULL_STATE_RESUME}" == "true" ]]; then
    echo "Resume=external full model/AdamW/scheduler/RNG state from ${RESUME_TRAINING_STATE}"
  else
    echo "Resume=latest complete full model/AdamW/scheduler/RNG state in ${TARGET_RUN_DIR}/checkpoints"
  fi
  echo "Resume in place=${RESUME_IN_PLACE}"
fi
echo "Per-device batch=${PER_DEVICE_BATCH_SIZE}, accumulation=${GRADIENT_ACCUMULATION_STEPS}, global batch=$((PER_DEVICE_BATCH_SIZE * NUM_PROCESSES * GRADIENT_ACCUMULATION_STEPS))"
echo "Conditioning=see ${CONFIG_YAML}"

TRAINER_OVERRIDES=()
if [[ "${FULL_STATE_RESUME}" == "true" ]]; then
  TRAINER_OVERRIDES+=(--trainer.resume_training_state "${RESUME_TRAINING_STATE}")
fi
if [[ -n "${WANDB_LOG_AFTER_STEP_OVERRIDE}" ]]; then
  TRAINER_OVERRIDES+=(--trainer.wandb_log_after_step "${WANDB_LOG_AFTER_STEP_OVERRIDE}")
fi
if [[ -n "${RECOVERED_FROM_PHASE_STEP_OVERRIDE}" ]]; then
  TRAINER_OVERRIDES+=(--trainer.recovered_from_phase_step "${RECOVERED_FROM_PHASE_STEP_OVERRIDE}")
fi
if [[ -n "${TRAINING_PHASE_ID_OVERRIDE}" ]]; then
  TRAINER_OVERRIDES+=(--trainer.training_phase_id "${TRAINING_PHASE_ID_OVERRIDE}")
fi

if [[ "${CHECK_ONLY}" == "true" ]]; then
  echo "CHECK_ONLY=true: configuration and required-path validation passed; training was not launched."
  exit 0
fi

cd "${STARVLA_DIR}"
"${ACCELERATE_BIN}" launch \
  --config_file "${ACCELERATE_CONFIG_FILE}" \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --framework.qwenvl.base_vlm "${BASE_VLM}" \
  --framework.intent.stage1_steps "${STAGE1_STEPS}" \
  --datasets.vla_data.data_root_dir "${LEROBOT_ROOT}" \
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --trainer.pretrained_checkpoint "${PRETRAINED_CHECKPOINT}" \
  --trainer.intent_pretrained_checkpoint "${S0_INTENT_CHECKPOINT}" \
  --trainer.is_resume "${IS_RESUME}" \
  --trainer.max_train_steps "${MAIN_MAX_STEPS}" \
  --trainer.scheduler_total_steps "${SCHEDULER_TOTAL_STEPS}" \
  --trainer.scheduler_step_offset "${ACTION_STEP_OFFSET}" \
  --trainer.wandb_step_offset "${ACTION_STEP_OFFSET}" \
  --trainer.num_warmup_steps "${NUM_WARMUP_STEPS}" \
  --trainer.gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --trainer.save_interval 5000 \
  --trainer.eval_interval 1000 \
  --trainer.repeated_diffusion_steps 16 \
  --run_root_dir "${CHECKPOINT_ROOT}" \
  --run_id "${RUN_ID}" \
  --wandb_project "${WANDB_PROJECT:-starVLA_Calvin_E1_Spatial_Intent_Main}" \
  --wandb_entity "${WANDB_ENTITY:-chaikehang-sjtu-hpc-center}" \
  "${TRAINER_OVERRIDES[@]}" &

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
