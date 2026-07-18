#!/usr/bin/env bash
# Project-local environment for StarVLA + CALVIN ABC->D E0.
# Usage:
#   source /home/liuchang/kehang/488project/scripts/e0_abc_rel/env.sh
#
# Keep large Hugging Face / Torch caches on the shared data disk. Do not put
# tokens or other credentials in this file.

export PROJECT=/home/liuchang/kehang/488project
export STARVLA_DIR="$PROJECT/third_party/starvla"
export CALVIN_DIR="$PROJECT/third_party/calvin"

export RUN_ID=e0_abc_rel
export RUN_DIR="$PROJECT/runs/$RUN_ID"
export LOG_DIR="$PROJECT/logs/$RUN_ID"
export MANIFEST_DIR="$PROJECT/manifests/$RUN_ID"
export AUDIT_DIR="$PROJECT/audits/$RUN_ID"

export DATA_ROOT=/home/data/datasets/kehang-CALVIN/calvin
export CALVIN_ORIGINAL="$DATA_ROOT/original"
export CALVIN_LEROBOT="$DATA_ROOT/lerobot"

export LEROBOT_SOURCE_REPO=sixpigs1/calvin2lerobotV21_ABC_D_scnet
export LEROBOT_SOURCE_REV=78faab6c533506cea5c526ea606f2afdffd43dac
export LEROBOT_RAW_DIR="$CALVIN_LEROBOT/sixpigs1_calvin2lerobotV21_ABC_D_scnet_raw"
export LEROBOT_E0_DIR="$CALVIN_LEROBOT/sixpigs1_calvin2lerobotV21_ABC_D_scnet_e0_rel"

export MODEL_ROOT=/home/data/models/kehang-StarVLA
export QWEN3_VL_MODEL_DIR="$MODEL_ROOT/Qwen3-VL-4B-Instruct"
export MODEL_DIR="$MODEL_ROOT/Qwen3-VL-4B-Instruct-Action"

export HF_HOME=/home/data/datasets/kehang-CALVIN/.cache/hf
export HUGGINGFACE_HUB_CACHE=/home/data/datasets/kehang-CALVIN/.cache/hf
export TORCH_HOME=/home/data/datasets/kehang-CALVIN/.cache/torch
export WANDB_CACHE_DIR=/home/data/datasets/kehang-CALVIN/.cache/wandb
export TRITON_CACHE_DIR=/home/data/datasets/kehang-CALVIN/.cache/triton
export XDG_CACHE_HOME=/home/data/datasets/kehang-CALVIN/.cache/xdg

export CONDA=/home/liuchang/miniconda3/bin/conda
