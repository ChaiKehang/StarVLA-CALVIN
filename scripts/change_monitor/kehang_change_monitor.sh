#!/usr/bin/env bash

#改動摘要

set -euo pipefail

INTERVAL_SECONDS="${INTERVAL_SECONDS:-1800}"
WORKSPACE="${WORKSPACE:-/home/liuchang/kehang}"
PROJECT="${PROJECT:-/home/liuchang/kehang/488project}"
CHANGELOG="${CHANGELOG:-/home/liuchang/kehang/改动日志.md}"
STATE_DIR="${STATE_DIR:-$PROJECT/manifests/change_monitor}"
LOG_DIR="${LOG_DIR:-$PROJECT/logs/change_monitor}"
DATA_DIR="${DATA_DIR:-/home/data/datasets/kehang-CALVIN}"
MODEL_DIR="${MODEL_DIR:-/home/data/models/kehang-StarVLA}"
CONDA_BIN="${CONDA_BIN:-/home/liuchang/miniconda3/bin/conda}"

mkdir -p "$STATE_DIR" "$LOG_DIR"
touch "$CHANGELOG"

CURRENT_FILES="$STATE_DIR/current_files.tsv"
PREV_FILES="$STATE_DIR/prev_files.tsv"
CURRENT_SUMMARY="$STATE_DIR/current_summary.txt"
PREV_SUMMARY="$STATE_DIR/prev_summary.txt"
DIFF_OUT="$STATE_DIR/last_diff.txt"
PID_FILE="$STATE_DIR/monitor.pid"
STOP_FILE="$STATE_DIR/stop"

echo "$$" > "$PID_FILE"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %z'
}

collect_files() {
  local root="$1"
  if [[ ! -d "$root" ]]; then
    return 0
  fi

  find "$root" \
    \( -path '*/.git/*' \
       -o -path '*/__pycache__/*' \
       -o -path '*/.pytest_cache/*' \
       -o -path '*/.mypy_cache/*' \
       -o -path '*/.ruff_cache/*' \
       -o -path '*/node_modules/*' \
       -o -path '*/wandb/*' \
       -o -path '*/runs/*/checkpoints/*' \
       -o -path '*/logs/change_monitor/*' \
       -o -path '*/manifests/change_monitor/*' \) -prune \
    -o -type f -printf '%T@	%s	%p\n' 2>/dev/null \
    | sort -k3,3
}

safe_du() {
  local path="$1"
  if [[ -e "$path" ]]; then
    du -sh "$path" 2>/dev/null | awk '{print $1 " " $2}'
  else
    echo "missing $path"
  fi
}

git_summary() {
  local repo="$1"
  if [[ -d "$repo/.git" ]]; then
    {
      echo "[git] $repo"
      git -C "$repo" rev-parse --abbrev-ref HEAD 2>/dev/null || true
      git -C "$repo" rev-parse --short HEAD 2>/dev/null || true
      git -C "$repo" status --short 2>/dev/null | sed -n '1,80p' || true
    } | sed 's/[[:cntrl:]]//g'
  fi
}

collect_summary() {
  {
    echo "timestamp=$(timestamp)"
    echo "workspace=$WORKSPACE"
    echo
    echo "## file_inventory"
    collect_files "$WORKSPACE"
    echo
    echo "## shared_sizes"
    safe_du "$DATA_DIR"
    safe_du "$MODEL_DIR"
    echo
    echo "## conda_envs"
    if [[ -x "$CONDA_BIN" ]]; then
      "$CONDA_BIN" env list 2>/dev/null | sed -n '1,80p' || true
    else
      echo "conda_missing=$CONDA_BIN"
    fi
    echo
    echo "## slurm_jobs"
    squeue -u "${USER:-liuchang}" -o '%.18i %.12P %.24j %.8u %.2t %.10M %.3D %R' 2>/dev/null || true
    echo
    echo "## git_status"
    git_summary "$PROJECT/code/starvla"
    git_summary "$PROJECT/code/calvin"
    git_summary "$WORKSPACE"
  } > "$CURRENT_SUMMARY"
}

append_change() {
  local now="$1"
  local diff_file="$2"
  {
    echo
    echo "### $now 自动监控：服务器改动摘要"
    echo
    echo "- 范围：$WORKSPACE；共享目录摘要：$DATA_DIR、$MODEL_DIR。"
    echo "- 方式：每 ${INTERVAL_SECONDS}s 比较文件路径/mtime/大小、git 状态、conda env 列表、当前 Slurm 队列；不记录文件内容或凭据。"
    echo "- 结果：检测到相对上次快照有变化，详见下方摘要。"
    echo
    echo '```text'
    sed -n '1,160p' "$diff_file"
    echo '```'
  } >> "$CHANGELOG"
}

collect_summary
if [[ ! -f "$PREV_SUMMARY" ]]; then
  cp "$CURRENT_SUMMARY" "$PREV_SUMMARY"
  {
    echo
    echo "### $(timestamp) 自动监控：已建立 baseline"
    echo
    echo "- 范围：$WORKSPACE；共享目录摘要：$DATA_DIR、$MODEL_DIR。"
    echo "- 周期：每 ${INTERVAL_SECONDS}s 检查一次。"
    echo "- 说明：仅记录摘要和路径级变化，不记录文件内容或凭据。"
  } >> "$CHANGELOG"
fi

while true; do
  sleep "$INTERVAL_SECONDS"
  if [[ -f "$STOP_FILE" ]]; then
    {
      echo
      echo "### $(timestamp) 自动监控：已停止"
      echo
      echo "- 检测到停止文件：$STOP_FILE。"
      echo "- 未删除任何监控快照；如需重启，删除该文件后重新运行脚本。"
    } >> "$CHANGELOG"
    exit 0
  fi
  collect_summary
  if ! cmp -s "$PREV_SUMMARY" "$CURRENT_SUMMARY"; then
    diff -u "$PREV_SUMMARY" "$CURRENT_SUMMARY" > "$DIFF_OUT" || true
    append_change "$(timestamp)" "$DIFF_OUT"
    cp "$CURRENT_SUMMARY" "$PREV_SUMMARY"
  fi
done
