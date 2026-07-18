#!/usr/bin/env bash
# Start the 90k policy server and run CALVIN evaluation in one allocation.
# The caller controls CUDA visibility; this script never selects a physical GPU.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
STARVLA_DIR="${STARVLA_DIR:-${PROJECT_ROOT}/code/starvla}"
CALVIN_DIR="${CALVIN_DIR:-${PROJECT_ROOT}/code/calvin}"
EVO_DIR="${EVO_DIR:-${PROJECT_ROOT}/scripts/reference/Evo-1_sixpigs}"

: "${CKPT90K:?Set CKPT90K to steps_90000_pytorch_model.pt}"

STARVLA_PYTHON="${STARVLA_PYTHON:-python}"
CALVIN_PYTHON="${CALVIN_PYTHON:-python}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5694}"
UNNORM_KEY="${UNNORM_KEY:-franka}"
NUM_SEQUENCES="${NUM_SEQUENCES:-10}"
REPLAN_STEPS="${REPLAN_STEPS:-5}"
DEBUG="${DEBUG:-true}"
EVAL_DATASET="${EVAL_DATASET:-${EVO_DIR}/CALVIN_evaluation/ABC_D_validation}"
CALVIN_CONFIG_PATH="${CALVIN_CONFIG_PATH:-${CALVIN_DIR}/calvin_models/conf}"
EVAL_SEQUENCES="${EVAL_SEQUENCES:-${STARVLA_DIR}/examples/calvin/eval_files/eval_sequences.json}"
EVAL_LOG_DIR="${EVAL_LOG_DIR:-${PROJECT_ROOT}/eval_logs/e0_abc_rel/repro_${NUM_SEQUENCES}_$(date +%Y%m%d_%H%M%S)}"

for required in "${STARVLA_DIR}" "${CALVIN_DIR}" "${CKPT90K}" \
  "${EVAL_DATASET}/validation/.hydra/merged_config.yaml" \
  "${CALVIN_CONFIG_PATH}" "${EVAL_SEQUENCES}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[ERROR] Required evaluation path does not exist: ${required}" >&2
    exit 2
  fi
done

RUN_DIR="$(dirname "$(dirname "${CKPT90K}")")"
for sidecar in "${RUN_DIR}/config.yaml" "${RUN_DIR}/dataset_statistics.json"; do
  if [[ ! -f "${sidecar}" ]]; then
    echo "[ERROR] Checkpoint sidecar is missing: ${sidecar}" >&2
    exit 3
  fi
done

case "${DEBUG}" in
  true|True|TRUE|1|yes|YES|on|ON) DEBUG_ARG=--args.debug ;;
  false|False|FALSE|0|no|NO|off|OFF) DEBUG_ARG=--args.no-debug ;;
  *) echo "[ERROR] DEBUG must be true or false." >&2; exit 4 ;;
esac

if [[ -d "${EVAL_LOG_DIR}" ]] && [[ -n "$(find "${EVAL_LOG_DIR}" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "[ERROR] Evaluation directory is not empty: ${EVAL_LOG_DIR}" >&2
  echo "Choose a new EVAL_LOG_DIR so existing results/videos are not overwritten." >&2
  exit 5
fi
mkdir -p "${EVAL_LOG_DIR}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

POLICY_LOG="${EVAL_LOG_DIR}/policy_server.log"
EVAL_STDOUT="${EVAL_LOG_DIR}/eval_stdout.log"

cleanup() {
  if [[ -n "${POLICY_PID:-}" ]] && kill -0 "${POLICY_PID}" 2>/dev/null; then
    kill "${POLICY_PID}" 2>/dev/null || true
    wait "${POLICY_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "${STARVLA_DIR}"
"${STARVLA_PYTHON}" deployment/model_server/server_policy.py \
  --ckpt_path "${CKPT90K}" \
  --port "${PORT}" \
  --use_bf16 >"${POLICY_LOG}" 2>&1 &
POLICY_PID=$!

echo "Policy server PID=${POLICY_PID}; log=${POLICY_LOG}"
echo "Waiting for ${HOST}:${PORT} ..."
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
  --args.pretrained-path "${CKPT90K}" \
  --args.unnorm-key "${UNNORM_KEY}" \
  --args.host "${HOST}" \
  --args.port "${PORT}" \
  --args.dataset_path "${EVAL_DATASET}" \
  --args.calvin_config_path "${CALVIN_CONFIG_PATH}" \
  --args.eval_sequences_path "${EVAL_SEQUENCES}" \
  --args.num_sequences "${NUM_SEQUENCES}" \
  --args.replan_steps "${REPLAN_STEPS}" \
  "${DEBUG_ARG}" \
  --args.eval_log_dir "${EVAL_LOG_DIR}" 2>&1 | tee "${EVAL_STDOUT}"
