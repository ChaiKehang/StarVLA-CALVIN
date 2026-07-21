#!/usr/bin/env bash
set -euo pipefail

###########################################################################################
# === Please modify the following paths according to your environment ===
STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}" # let Calvin client find websocket tools from main repo
export calvin_python="${calvin_python:-/home/liuchang/miniconda3/envs/calvin-eval/bin/python}"

host="${host:-127.0.0.1}"
base_port="${base_port:-5697}"
unnorm_key="${unnorm_key:-franka}"
your_ckpt="${your_ckpt:-/home/liuchang/kehang/488project/code/starvla/playground/Pretrained_models/kehang-StarVLA/checkpoints/calvin/e1_b_abc_rel_scaled_intent125_h8/checkpoints/steps_90000_pytorch_model.pt}"
dataset_path="${dataset_path:-/home/liuchang/kehang/488project/scripts/reference/Evo-1_sixpigs/CALVIN_evaluation/ABC_D_validation}"
calvin_config_path="${calvin_config_path:-/home/liuchang/kehang/488project/code/calvin/calvin_models/conf}"
eval_sequences_path="${eval_sequences_path:-/home/liuchang/kehang/488project/code/starvla/examples/calvin/eval_files/eval_sequences.json}"
num_sequences="${num_sequences:-1}"
replan_steps="${replan_steps:-5}"
inference_seed="${inference_seed:-42}"
debug="${debug:-true}"
DISABLE_INTENT_CONDITIONING="${DISABLE_INTENT_CONDITIONING:-false}"

if [[ ! -f "${your_ckpt}" ]]; then
    echo "[ERROR] Checkpoint does not exist: ${your_ckpt}" >&2
    exit 2
fi

DEBUG_ARGS=()
case "${debug}" in
    true|True|TRUE|1|yes|YES|on|ON)
        DEBUG_ARGS+=(--args.debug)
        ;;
    false|False|FALSE|0|no|NO|off|OFF)
        DEBUG_ARGS+=(--args.no-debug)
        ;;
    *)
        echo "Invalid debug='${debug}'. Use true/false." >&2
        exit 2
        ;;
esac

INTENT_ABLATION_ARGS=()
case "${DISABLE_INTENT_CONDITIONING}" in
    true|True|TRUE|1|yes|YES|on|ON)
        INTENT_ABLATION_ARGS+=(--args.disable-intent-conditioning)
        condition_tag="intent_off"
        ;;
    false|False|FALSE|0|no|NO|off|OFF)
        condition_tag="intent_on"
        ;;
    *)
        echo "Invalid DISABLE_INTENT_CONDITIONING='${DISABLE_INTENT_CONDITIONING}'. Use true/false." >&2
        exit 2
        ;;
esac

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
# === End of environment variable configuration ===
###########################################################################################

LOG_DIR="${LOG_DIR:-/home/liuchang/kehang/488project/eval_logs/e1_b/${folder_name}_${condition_tag}_$(date +"%Y%m%d_%H%M%S")}"
mkdir -p ${LOG_DIR}

cd "${STARVLA_DIR}"

"${calvin_python}" ./examples/calvin/eval_files/eval_calvin.py \
    --args.pretrained-path "${your_ckpt}" \
    --args.unnorm-key "${unnorm_key}" \
    --args.host "$host" \
    --args.port "${base_port}" \
    --args.dataset_path "${dataset_path}" \
    --args.calvin_config_path "${calvin_config_path}" \
    --args.eval_sequences_path "${eval_sequences_path}" \
    --args.num_sequences "${num_sequences}" \
    --args.replan_steps "${replan_steps}" \
    --args.inference-seed "${inference_seed}" \
    "${DEBUG_ARGS[@]}" \
    "${INTENT_ABLATION_ARGS[@]}" \
    --args.eval_log_dir "${LOG_DIR}"
