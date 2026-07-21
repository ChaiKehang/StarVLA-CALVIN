#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
star_vla_python="${star_vla_python:-/home/liuchang/miniconda3/envs/starvla-e0/bin/python}"
your_ckpt="${your_ckpt:-/home/liuchang/kehang/488project/code/starvla/playground/Pretrained_models/kehang-StarVLA/checkpoints/calvin/e1_b_abc_rel_scaled_intent125_h8/checkpoints/steps_90000_pytorch_model.pt}"
gpu_id="${gpu_id:-2}"
port="${port:-5697}"
USE_BF16="${USE_BF16:-1}"

if [[ ! -f "${your_ckpt}" ]]; then
  echo "[ERROR] Checkpoint does not exist: ${your_ckpt}" >&2
  exit 2
fi

cd "${STARVLA_DIR}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"

CMD=(
  "${star_vla_python}" deployment/model_server/server_policy.py
  --ckpt_path "${your_ckpt}"
  --port "${port}"
)

if [[ "${USE_BF16}" == "1" ]]; then
  CMD+=(--use_bf16)
fi

CUDA_VISIBLE_DEVICES="${gpu_id}" "${CMD[@]}"
