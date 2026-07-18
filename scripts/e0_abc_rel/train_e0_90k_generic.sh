#!/usr/bin/env bash
# Portable launcher for the successful CALVIN ABC->D 90k baseline.
#
# It deliberately contains no Slurm directives and never chooses physical
# GPUs.  Set CUDA_VISIBLE_DEVICES yourself or let your scheduler set it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
STARVLA_DIR="${STARVLA_DIR:-${PROJECT_ROOT}/third_party/starvla}"
CONFIG_YAML="${CONFIG_YAML:-${PROJECT_ROOT}/configs/starvla/e0_abc_rel_calvin_scaled_90k.yaml}"

: "${LEROBOT_ROOT:?Set LEROBOT_ROOT to the directory containing the derived LeRobot dataset}"
: "${MODEL_ROOT:?Set MODEL_ROOT to the directory containing the VLM and pretrained checkpoint}"
: "${CHECKPOINT_ROOT:?Set CHECKPOINT_ROOT to the output parent directory}"

BASE_VLM="${BASE_VLM:-${MODEL_ROOT}/Qwen3-VL-4B-Instruct}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-${MODEL_ROOT}/pretrained/starvla_qwenpi_pretrain_qwen3_4B_bridge-rt_1/checkpoints/steps_50000_pytorch_model.pt}"
RUN_ID="${RUN_ID:-e0_abc_rel}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
WANDB_MODE="${WANDB_MODE:-disabled}"
E0_WANDB_PROJECT="${E0_WANDB_PROJECT:-starVLA_Calvin_E0}"
E0_WANDB_ENTITY="${E0_WANDB_ENTITY:-disabled}"
CACHE_ROOT="${CACHE_ROOT:-${PROJECT_ROOT}/.cache}"

for required in "${STARVLA_DIR}" "${CONFIG_YAML}" "${LEROBOT_ROOT}" "${BASE_VLM}" "${PRETRAINED_CHECKPOINT}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[ERROR] Required path does not exist: ${required}" >&2
    exit 2
  fi
done

DERIVED_DATASET="${LEROBOT_ROOT}/sixpigs1_calvin2lerobotV21_ABC_D_scnet_rel_calvin_scaled"
for metadata in info.json modality.json stats_gr00t.json action_conversion.json; do
  if [[ ! -f "${DERIVED_DATASET}/meta/${metadata}" ]]; then
    echo "[ERROR] Derived dataset metadata is missing: ${DERIVED_DATASET}/meta/${metadata}" >&2
    exit 2
  fi
done

PRETRAINED_RUN_DIR="$(dirname "$(dirname "${PRETRAINED_CHECKPOINT}")")"
for sidecar in config.yaml dataset_statistics.json; do
  if [[ ! -f "${PRETRAINED_RUN_DIR}/${sidecar}" ]]; then
    echo "[ERROR] Pretrained checkpoint sidecar is missing: ${PRETRAINED_RUN_DIR}/${sidecar}" >&2
    exit 2
  fi
done

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a VISIBLE_GPUS <<<"${CUDA_VISIBLE_DEVICES}"
  if (( ${#VISIBLE_GPUS[@]} < NUM_PROCESSES )); then
    echo "[ERROR] CUDA_VISIBLE_DEVICES exposes ${#VISIBLE_GPUS[@]} device(s), but NUM_PROCESSES=${NUM_PROCESSES}." >&2
    exit 2
  fi
fi

TARGET_RUN_DIR="${CHECKPOINT_ROOT}/${RUN_ID}"
if [[ -d "${TARGET_RUN_DIR}" ]] && [[ -n "$(find "${TARGET_RUN_DIR}" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "[ERROR] ${TARGET_RUN_DIR} already exists and is not empty." >&2
  echo "Use a new RUN_ID; this baseline launcher will not reuse, overwrite or resume a run." >&2
  exit 3
fi

export WANDB_MODE
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

echo "Project root: ${PROJECT_ROOT}"
echo "StarVLA: ${STARVLA_DIR}"
echo "Config: ${CONFIG_YAML}"
echo "Data: ${LEROBOT_ROOT}"
echo "Base VLM: ${BASE_VLM}"
echo "Initialization checkpoint: ${PRETRAINED_CHECKPOINT}"
echo "Output: ${TARGET_RUN_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Processes=${NUM_PROCESSES}, per-device batch=${PER_DEVICE_BATCH_SIZE}"

cd "${STARVLA_DIR}"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --framework.qwenvl.base_vlm "${BASE_VLM}" \
  --datasets.vla_data.data_root_dir "${LEROBOT_ROOT}" \
  --datasets.vla_data.data_mix calvin_abc_d_sixpigs_rel_scaled \
  --datasets.vla_data.action_mode abs \
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --trainer.pretrained_checkpoint "${PRETRAINED_CHECKPOINT}" \
  --trainer.is_resume false \
  --trainer.max_train_steps 90000 \
  --trainer.num_warmup_steps 5000 \
  --trainer.save_interval 6000 \
  --trainer.eval_interval 1000 \
  --trainer.logging_frequency 20 \
  --trainer.freeze_modules qwen_vl_interface \
  --trainer.repeated_diffusion_steps 16 \
  --run_root_dir "${CHECKPOINT_ROOT}" \
  --run_id "${RUN_ID}" \
  --wandb_project "${E0_WANDB_PROJECT}" \
  --wandb_entity "${E0_WANDB_ENTITY}"
