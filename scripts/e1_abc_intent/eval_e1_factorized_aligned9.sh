#!/usr/bin/env bash
# Canonical 500-sequence CALVIN-D evaluation for aligned-9 checkpoints.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
STARVLA_DIR="${STARVLA_DIR:-${PROJECT_ROOT}/third_party/starvla}"
CALVIN_DIR="${CALVIN_DIR:-${PROJECT_ROOT}/third_party/calvin}"
MODEL_ROOT="${MODEL_ROOT:-/home/data/models/kehang-StarVLA}"
RUN_DIR="${RUN_DIR:-${MODEL_ROOT}/checkpoints/calvin/e1_factorized_aligned9_query_s1_10k_s2_80k}"

CHECKPOINT_STEP="${CHECKPOINT_STEP:-90000}"
case "${CHECKPOINT_STEP}" in
  60000|90000) ;;
  *) echo "[ERROR] CHECKPOINT_STEP must be 60000 or 90000." >&2; exit 2 ;;
esac
CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/checkpoints/steps_${CHECKPOINT_STEP}_pytorch_model.pt}"

STARVLA_PYTHON="${STARVLA_PYTHON:-/home/liuchang/miniconda3/envs/starvla-e0/bin/python}"
CALVIN_PYTHON="${CALVIN_PYTHON:-/home/liuchang/miniconda3/envs/calvin-eval/bin/python}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5701}"
NUM_SEQUENCES="${NUM_SEQUENCES:-500}"
REPLAN_STEPS="${REPLAN_STEPS:-5}"
INFERENCE_SEED="${INFERENCE_SEED:-42}"
DEBUG="${DEBUG:-true}"
UNNORM_KEY="${UNNORM_KEY:-franka}"
EVAL_DATASET="${EVAL_DATASET:-${PROJECT_ROOT}/scripts/reference/Evo-1_sixpigs/CALVIN_evaluation/ABC_D_validation}"
CALVIN_CONFIG_PATH="${CALVIN_CONFIG_PATH:-${CALVIN_DIR}/calvin_models/conf}"
EVAL_SEQUENCES="${EVAL_SEQUENCES:-${STARVLA_DIR}/examples/calvin/eval_files/eval_sequences.json}"
EVAL_LOG_DIR="${EVAL_LOG_DIR:-${PROJECT_ROOT}/eval_logs/e1_factorized_intent/aligned9_steps${CHECKPOINT_STEP}_calvin${NUM_SEQUENCES}_intent_on_$(date +%Y%m%d_%H%M%S)}"

for required in "${STARVLA_DIR}" "${CALVIN_DIR}" "${CHECKPOINT}" \
  "${RUN_DIR}/config.yaml" "${RUN_DIR}/dataset_statistics.json" \
  "${EVAL_DATASET}/validation/.hydra/merged_config.yaml" \
  "${CALVIN_CONFIG_PATH}" "${EVAL_SEQUENCES}" \
  "${STARVLA_PYTHON}" "${CALVIN_PYTHON}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[ERROR] Required evaluation path does not exist: ${required}" >&2
    exit 2
  fi
done
if [[ -d "${EVAL_LOG_DIR}" ]] && [[ -n "$(find "${EVAL_LOG_DIR}" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "[ERROR] Evaluation output is not empty: ${EVAL_LOG_DIR}" >&2
  exit 3
fi

case "${DEBUG}" in
  true|True|TRUE|1|yes|YES|on|ON) DEBUG_ARG=--args.debug ;;
  false|False|FALSE|0|no|NO|off|OFF) DEBUG_ARG=--args.no-debug ;;
  *) echo "[ERROR] DEBUG must be true or false." >&2; exit 2 ;;
esac

mkdir -p "${EVAL_LOG_DIR}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
POLICY_LOG="${EVAL_LOG_DIR}/policy_server.log"
EVAL_LOG="${EVAL_LOG_DIR}/eval_client.log"

cleanup() {
  if [[ -n "${POLICY_PID:-}" ]] && kill -0 "${POLICY_PID}" 2>/dev/null; then
    kill "${POLICY_PID}" 2>/dev/null || true
    wait "${POLICY_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "${STARVLA_DIR}"
"${STARVLA_PYTHON}" deployment/model_server/server_policy.py \
  --ckpt_path "${CHECKPOINT}" \
  --port "${PORT}" \
  --use_bf16 >"${POLICY_LOG}" 2>&1 &
POLICY_PID=$!

"${CALVIN_PYTHON}" - "${HOST}" "${PORT}" "${POLICY_PID}" <<'PY'
import os
import socket
import sys
import time

host, port, pid = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
deadline = time.time() + 300
while time.time() < deadline:
    try:
        os.kill(pid, 0)
    except OSError as exc:
        raise SystemExit(f"policy server exited before opening the port: {exc}")
    try:
        with socket.create_connection((host, port), timeout=2):
            break
    except OSError:
        time.sleep(2)
else:
    raise SystemExit(f"timed out waiting for {host}:{port}")
PY

set -o pipefail
"${CALVIN_PYTHON}" examples/calvin/eval_files/eval_calvin.py \
  --args.pretrained-path "${CHECKPOINT}" \
  --args.unnorm-key "${UNNORM_KEY}" \
  --args.host "${HOST}" \
  --args.port "${PORT}" \
  --args.dataset_path "${EVAL_DATASET}" \
  --args.calvin_config_path "${CALVIN_CONFIG_PATH}" \
  --args.eval_sequences_path "${EVAL_SEQUENCES}" \
  --args.num_sequences "${NUM_SEQUENCES}" \
  --args.replan_steps "${REPLAN_STEPS}" \
  --args.inference-seed "${INFERENCE_SEED}" \
  "${DEBUG_ARG}" \
  --args.eval_log_dir "${EVAL_LOG_DIR}" 2>&1 | tee "${EVAL_LOG}"
