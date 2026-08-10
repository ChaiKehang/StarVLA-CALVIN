# Canonical aligned-9 Factorized Intent pipeline

This directory retains one supported E1 path only:

1. build the factorized CALVIN labels and trajectory split;
2. train S0 Intent-only for 50,000 optimizer steps;
3. load the best S0 validation checkpoint;
4. train one continuous S1 10k + S2 80k run;
5. evaluate the retained 60k and 90k main checkpoints.

The launchers are intentionally fresh-only. They do not support weight-only
resume, step offsets, learning-rate restart patches, or W&B continuation.

## 1. Prepare the factorized dataset

```bash
/home/liuchang/miniconda3/envs/starvla-e0/bin/python \
  /home/liuchang/kehang/488project/scripts/e1_abc_intent/build_factorized_intent_dataset.py
```

## 2. Train S0 (50k)

```bash
CUDA_VISIBLE_DEVICES=2,3 \
NUM_PROCESSES=2 \
WANDB_MODE=online \
bash /home/liuchang/kehang/488project/scripts/e1_abc_intent/train_e1_factorized_aligned9_spatial_intent_s0_50k.sh
```

S0 loads the original Bridge-RT1 50k checkpoint, freezes Qwen and the action
path, and selects `best_intent_pytorch_model.pt` using validation metrics.

## 3. Train S1 10k + S2 80k

```bash
CUDA_VISIBLE_DEVICES=2,3 \
NUM_PROCESSES=2 \
WANDB_MODE=online \
S0_INTENT_CHECKPOINT=/home/data/models/kehang-StarVLA/checkpoints/calvin/e1_factorized_aligned9_spatial_intent_s0_50k/checkpoints/best_intent_pytorch_model.pt \
bash /home/liuchang/kehang/488project/scripts/e1_abc_intent/train_e1_factorized_aligned9_query_s1_10k_s2_80k.sh
```

The main run always starts the action path from the original Bridge-RT1 50k
checkpoint. S1 freezes Intent for 10k steps; S2 unfreezes it for the remaining
80k. The cosine schedule is continuous over all 90k steps.

## 4. Evaluate

```bash
CUDA_VISIBLE_DEVICES=1 CHECKPOINT_STEP=60000 \
bash /home/liuchang/kehang/488project/scripts/e1_abc_intent/eval_e1_factorized_aligned9.sh

CUDA_VISIBLE_DEVICES=1 CHECKPOINT_STEP=90000 \
bash /home/liuchang/kehang/488project/scripts/e1_abc_intent/eval_e1_factorized_aligned9.sh
```

The canonical evaluation uses 500 CALVIN-D sequences, seed 42, replanning
every five environment steps, and Intent conditioning enabled.
