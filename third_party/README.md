# Vendored upstream source

This directory contains source snapshots rather than Git submodules or nested
Git repositories. It lets a normal clone of `StarVLA-CALVIN` include the exact
StarVLA and CALVIN code used by the E0 baseline.

## Versions

| Directory | Upstream | Commit |
|---|---|---|
| `starvla/` | <https://github.com/starVLA/starVLA> (`starVLA_dev`) | `6dc01d0781a817c007f74927a75bf63d89d521e2` |
| `calvin/` | <https://github.com/mees/calvin> | `fa03f01f19c65920e18cf37398a9ce859274af76` |
| `calvin/calvin_env/` | <https://github.com/mees/calvin_env> | `1431a46bd36bde5903fb6345e68b5ccc30def666` |
| `calvin/calvin_env/tacto/` | <https://github.com/lukashermann/tacto> | `dd53360d9a8c186f0d6439372ec0be0fa5e21731` |

The original license files are retained in their corresponding directories.

## Local E0 changes applied to StarVLA

The StarVLA snapshot contains only the local changes required by this baseline:

- `deployment/model_server/tools/websocket_policy_client.py`: compatibility
  with the evaluator's websocket dependency;
- `examples/calvin/eval_files/eval_calvin.py`: current `ModelClient` API and
  `num_sequences` slicing;
- `starVLA/dataloader/gr00t_lerobot/mixtures.py`: registration of
  `calvin_abc_d_sixpigs_rel_scaled`.

## Local E1 Intent extensions applied to StarVLA

The snapshot also contains the configurable E1 Intent family used by this
project: E1-B timestep conditioning, E1-C FFN-FiLM, and the Spatial Intent v2
hierarchical Query Transformer with Query/FFN-FiLM. The implementation,
training/evaluation configs, and CPU tests live inside `third_party/starvla`;
the server launch and dataset-label utilities live in
`scripts/e1_abc_intent/` at the parent-repository level.

Machine-specific shell defaults, downloaded models, datasets, checkpoints,
playground symlinks, caches and unrelated dirty working-tree files were not
copied into this snapshot.

## Snapshot semantics

These directories intentionally contain no `.git` metadata. Consequently,
`git -C third_party/starvla pull` and similar commands do not apply. To update
an upstream version, export a fresh pinned commit, reapply the reviewed E0
changes, keep the license/provenance record current, and commit the resulting
snapshot in the parent repository.
