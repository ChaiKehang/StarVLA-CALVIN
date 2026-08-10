# 488project cleanup manifest — 2026-08-10

## Authorized filesystem scope

- `/home/liuchang/kehang`
- `/home/data/datasets/kehang-CALVIN` (resolved dataset symlink target)
- `/home/data/models/kehang-StarVLA` (resolved model/checkpoint symlink target)

No path outside these roots is part of this cleanup.

## Canonical experiments retained

### E0 baseline

- Config: `configs/starvla/e0_abc_rel_calvin_scaled_90k.yaml`
- Checkpoints: `e0_abc_rel/checkpoints/steps_60000_pytorch_model.pt` and
  `steps_90000_pytorch_model.pt`
- Full 500-sequence evaluation directories are retained for both checkpoints.

The historical `e0_abc_rel` run directory was later extended to 120k and its
sidecar config reports 120k. The retained project config above is the canonical
90k recipe; checkpoints later than 90k are deliberately removed.

### E1 factorized aligned-9

- S0: `e1_factorized_aligned9_spatial_intent_s0_50k`
- Main: `e1_factorized_aligned9_query_s1_10k_s2_80k`
- VLM source and Query-FiLM layers: `3,7,11,15,19,23,27,31,35`
- Main run: S1 10k + S2 80k, continuous 90k cosine schedule
- Router LR: `2e-6`; modulation bound: `0.1`; condition dropout: `0.2`
- S0 best, S0 50k, Main 60k, Main 90k, and Main 90k full training state are retained.
- Full 500-sequence evaluation directories are retained for Main 60k and 90k.

## Evaluation result mapping

| Artifact | Avg. chain | Success@1–5 |
|---|---:|---|
| E0 directory named `baseline_60000steps_1.37epoch_500` | 2.714 | 88.4, 66.8, 50.2, 38.2, 27.8 |
| E0 directory named `baseline_90000steps_1.37epoch_500` | 3.214 | 92.4, 78.0, 61.8, 50.6, 38.6 |
| aligned-9 Main 60k | 3.240 | 93.0, 78.8, 61.8, 50.6, 39.8 |
| aligned-9 Main 90k | 3.130 | 90.2, 76.2, 59.6, 48.4, 38.6 |

Directory names are preserved rather than silently relabeled.

## SHA-256 of retained checkpoints

```text
7f59a5d0fa9c167fabd941bca8e606bdf5597bfb4f99ca83e345672dd9c345ed  pretrained/starvla_qwenpi_pretrain_qwen3_4B_bridge-rt_1/checkpoints/steps_50000_pytorch_model.pt
a75f7429956335ab96036a2616293a74cdc93a3ac1915d53f2a0bc188099c261  checkpoints/calvin/e0_abc_rel/checkpoints/steps_60000_pytorch_model.pt
0db203c7e314b4453a47f8d2e9b46d7cee3d53871e0634ea300e72de435cf848  checkpoints/calvin/e0_abc_rel/checkpoints/steps_90000_pytorch_model.pt
306d4061e05cb711f30ebf179bc377deb48ca686e972ec110dc9ea2e451d36c3  checkpoints/calvin/e1_factorized_aligned9_spatial_intent_s0_50k/checkpoints/best_intent_pytorch_model.pt
b01c8d8902f76dcd98d7a53a5d6be5690564f73b65fa4e8c6f0527a83cffc497  checkpoints/calvin/e1_factorized_aligned9_spatial_intent_s0_50k/checkpoints/steps_50000_pytorch_model.pt
7d2e4597e530620afecdb7f42593b962bcae3da5002b256db61cca274f1e5de4  checkpoints/calvin/e1_factorized_aligned9_query_s1_10k_s2_80k/checkpoints/steps_60000_pytorch_model.pt
903bc539a284d83fca3b391530c057a0aa3f4f4390e2345bf927c030af1d3631  checkpoints/calvin/e1_factorized_aligned9_query_s1_10k_s2_80k/checkpoints/steps_90000_pytorch_model.pt
```

## SHA-256 of retained evaluation summaries

```text
a067772a5829e119be1f154d449b0dba94394e561413e06d080b37a44180c602  e0 baseline 60k results.json
e674c70c2bd29962c8de300e895833b9843bb75c40314234fd1999d87c66065a  e0 baseline 90k results.json
63cf2ce0bea4ba61e589a485e4b88bedc50f0528656a0eee19d2f7f55b423d50  aligned-9 60k results.json
6a3b51fb3197dc269d76d62b6f891611d55ec2ca9bb96ae2d267bb0f3e47b813  aligned-9 90k results.json
```

## Canonical fresh-run scripts

- `scripts/e1_abc_intent/train_e1_factorized_aligned9_spatial_intent_s0_50k.sh`
- `scripts/e1_abc_intent/train_e1_factorized_aligned9_query_s1_10k_s2_80k.sh`
- `scripts/e1_abc_intent/eval_e1_factorized_aligned9.sh`
- `scripts/e1_abc_intent/README_ALIGNED9_PIPELINE.md`

These launchers reject non-empty output directories and expose no legacy
weight-only resume, step-offset, LR-restart, or W&B-continuation path.

## Post-clean validation

- Active checkpoint root contains only `e0_abc_rel`,
  `e1_factorized_aligned9_spatial_intent_s0_50k`, and
  `e1_factorized_aligned9_query_s1_10k_s2_80k`.
- Active dataset root contains only the scaled E0 dataset and factorized-h8
  E1 dataset under `calvin/lerobot`.
- Factorized split contains 16,080 training trajectories and 1,790 validation
  trajectories with zero trajectory-ID overlap.
- All canonical shell launchers pass `bash -n`; both E1 launchers pass their
  `CHECK_ONLY=true` configuration and path validation.
- Factorized Intent head smoke tests pass, and the DiT Intent-conditioning and
  Intent-evaluation suites pass all 15 unit tests.
- The four retained 500-sequence evaluation directories each contain a
  non-empty `results.json` and their original rollout artifacts.
- `artifacts/datasets`, `artifacts/models`, and `artifacts/checkpoints` are the
  only project-level links to external data/model storage; all three resolve.

## Cleanup completion

Retired files were first moved within their original filesystems to exact
quarantine roots so the active tree could be validated:

- `/home/liuchang/kehang/.cleanup_quarantine_488project_20260810` (about 15 GB)
- `/home/data/models/kehang-StarVLA/.cleanup_quarantine_20260810` (about 1.5 TB)
- `/home/data/datasets/kehang-CALVIN/.cleanup_quarantine_20260810` (about 14 GB)

All three quarantine roots were permanently removed after validation. The
cleanup reduced the active model root to about 97 GB and the active dataset
root to about 772 MB; `/home/data` increased to about 6.7 TB free. Retired
contents are no longer recoverable from this workspace. Git-tracked source
deletions remain visible in `git status` until the cleanup commit is created.
