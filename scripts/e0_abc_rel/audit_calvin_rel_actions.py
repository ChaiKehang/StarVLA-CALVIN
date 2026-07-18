#!/usr/bin/env python3
"""Audit sixpigs-derived actions against original CALVIN rel_actions.

The original CALVIN dataset stores one `episode_XXXXXXX.npz` per timestep.  The
sixpigs LeRobot conversion stores one parquet per trajectory and keeps the
original timestep id in the raw dataset's `index` column.  This script samples
trajectory/timestep pairs from the derived LeRobot dataset, maps them back to
the raw sixpigs `index`, loads the corresponding original CALVIN npz, and
compares action semantics.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_ORIGINAL = Path(
    "/home/data/datasets/kehang-CALVIN/calvin/original/task_ABC_D/training"
)
DEFAULT_RAW_SIXPIGS = Path(
    "/home/data/datasets/kehang-CALVIN/calvin/lerobot/"
    "sixpigs1_calvin2lerobotV21_ABC_D_scnet_raw"
)
DEFAULT_DERIVED = Path(
    "/home/data/datasets/kehang-CALVIN/calvin/lerobot/"
    "sixpigs1_calvin2lerobotV21_ABC_D_scnet_rel"
)
DEFAULT_OUTPUT = Path(
    "/home/liuchang/kehang/488project/audits/e0_abc_rel/"
    "calvin_rel_action_audit.json"
)

CALVIN_REL_SCALE = np.array([50.0, 50.0, 50.0, 20.0, 20.0, 20.0, 1.0], dtype=np.float64)


def jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def load_info(dataset: Path) -> dict[str, Any]:
    with (dataset / "meta/info.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def npz_path(original_dir: Path, source_index: int) -> Path:
    return original_dir / f"episode_{source_index:07d}.npz"


def summarize_errors(errors: np.ndarray) -> dict[str, Any]:
    if errors.size == 0:
        return {}
    return {
        "mean_abs_error_per_dim": errors.mean(axis=0),
        "max_abs_error_per_dim": errors.max(axis=0),
        "mean_abs_error_all": float(errors.mean()),
        "max_abs_error_all": float(errors.max()),
    }


def discover_original_status(original_dir: Path) -> dict[str, Any]:
    candidates = []
    root = Path("/home/data/datasets/kehang-CALVIN/calvin/original")
    if root.exists():
        for p in sorted(root.glob("**/episode_*.npz"))[:10]:
            candidates.append(str(p))
    return {
        "expected_original_split_dir": str(original_dir),
        "exists": original_dir.exists(),
        "sample_npz_count_in_expected_dir": len(list(original_dir.glob("episode_*.npz"))) if original_dir.exists() else 0,
        "nearby_npz_examples": candidates,
    }


def infer_calvin_rel_scale_from_debug() -> dict[str, Any]:
    debug = Path("/home/data/datasets/kehang-CALVIN/calvin/original/calvin_debug_dataset/validation")
    files = sorted(debug.glob("episode_*.npz"))[:100] if debug.exists() else []
    ratios = []
    examples = []
    for f in files:
        d = np.load(f)
        delta = d["actions"][:6].astype(np.float64) - d["robot_obs"][:6].astype(np.float64)
        rel = d["rel_actions"][:6].astype(np.float64)
        good = np.abs(delta) > 1e-9
        ratio = np.full(6, np.nan, dtype=np.float64)
        ratio[good] = rel[good] / delta[good]
        ratios.append(ratio)
        if len(examples) < 3:
            examples.append(
                {
                    "file": str(f),
                    "delta_actions_minus_robot_obs_first6": delta,
                    "rel_actions_first6": rel,
                    "ratio_first6": ratio,
                    "gripper_action": float(d["actions"][6]),
                    "gripper_rel_action": float(d["rel_actions"][6]),
                }
            )
    if not ratios:
        return {"available": False}
    ratios_np = np.array(ratios, dtype=np.float64)
    return {
        "available": True,
        "debug_dir": str(debug),
        "num_debug_samples": len(files),
        "median_scale_first6": np.nanmedian(ratios_np, axis=0),
        "min_scale_first6": np.nanmin(ratios_np, axis=0),
        "max_scale_first6": np.nanmax(ratios_np, axis=0),
        "expected_scale_used_by_audit": CALVIN_REL_SCALE,
        "examples": examples,
    }


def sample_pairs(episodes: list[dict[str, Any]], samples: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    valid = [(int(ep["episode_index"]), int(ep["length"])) for ep in episodes if int(ep["length"]) > 0]
    pairs: list[tuple[int, int]] = []
    for _ in range(samples):
        ep, length = rng.choice(valid)
        frame = rng.randrange(length)
        pairs.append((ep, frame))
    return pairs


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "started",
        "original_split_dir": str(args.original_dir),
        "raw_sixpigs_dir": str(args.raw_sixpigs_dir),
        "derived_rel_dir": str(args.derived_rel_dir),
        "samples_requested": args.samples,
        "seed": args.seed,
        "calvin_rel_scale_candidate": CALVIN_REL_SCALE,
    }

    if not args.derived_rel_dir.exists():
        report["status"] = "blocked_missing_derived_rel_dataset"
        return report
    if not args.raw_sixpigs_dir.exists():
        report["status"] = "blocked_missing_raw_sixpigs_dataset"
        return report

    report["original_status"] = discover_original_status(args.original_dir)
    report["debug_scale_diagnostic"] = infer_calvin_rel_scale_from_debug()
    if not args.original_dir.exists() or not list(args.original_dir.glob("episode_*.npz")):
        report["status"] = "blocked_missing_full_original_calvin_abc_d_training_npz"
        report["reason"] = (
            "Full original CALVIN task_ABC_D/training npz files are required. "
            "Only debug validation npz files were found locally, which do not align "
            "with sixpigs ABC-D training episode/timestep ids."
        )
        return report

    info = load_info(args.derived_rel_dir)
    chunks_size = int(info.get("chunks_size", 1000))
    episodes = read_jsonl(args.derived_rel_dir / "meta/episodes.jsonl")
    pairs = sample_pairs(episodes, args.samples, args.seed)

    records = []
    unscaled_errors = []
    scaled_errors = []
    abs_action_errors = []
    state_errors = []
    missing = []

    for ep, frame in pairs:
        chunk = ep // chunks_size
        raw_pq = args.raw_sixpigs_dir / f"data/chunk-{chunk:03d}/episode_{ep:06d}.parquet"
        rel_pq = args.derived_rel_dir / f"data/chunk-{chunk:03d}/episode_{ep:06d}.parquet"
        if not raw_pq.exists() or not rel_pq.exists():
            missing.append({"episode_index": ep, "frame_index": frame, "missing": "parquet"})
            continue
        raw_df = pd.read_parquet(raw_pq)
        rel_df = pd.read_parquet(rel_pq)
        if frame >= len(raw_df) or frame >= len(rel_df):
            missing.append({"episode_index": ep, "frame_index": frame, "missing": "frame"})
            continue

        source_index = int(raw_df["index"].iloc[frame])
        original_npz = npz_path(args.original_dir, source_index)
        if not original_npz.exists():
            missing.append(
                {
                    "episode_index": ep,
                    "frame_index": frame,
                    "source_index": source_index,
                    "missing": str(original_npz),
                }
            )
            continue

        npz = np.load(original_npz)
        converted = np.asarray(rel_df["action"].iloc[frame], dtype=np.float64)
        converted_scaled = converted * CALVIN_REL_SCALE
        original_rel = np.asarray(npz["rel_actions"], dtype=np.float64)
        original_abs = np.asarray(npz["actions"], dtype=np.float64)
        original_state = np.asarray(npz["robot_obs"][:7], dtype=np.float64)
        raw_abs = np.asarray(raw_df["action"].iloc[frame], dtype=np.float64)
        raw_state = np.asarray(raw_df["observation.state"].iloc[frame], dtype=np.float64)

        ue = np.abs(converted - original_rel)
        se = np.abs(converted_scaled - original_rel)
        ae = np.abs(raw_abs - original_abs)
        ste = np.abs(raw_state - original_state)
        unscaled_errors.append(ue)
        scaled_errors.append(se)
        abs_action_errors.append(ae)
        state_errors.append(ste)
        records.append(
            {
                "episode_index": ep,
                "frame_index": frame,
                "source_index": source_index,
                "original_npz": str(original_npz),
                "converted_action": converted,
                "converted_action_scaled_candidate": converted_scaled,
                "original_rel_actions": original_rel,
                "abs_error_unscaled": ue,
                "abs_error_scaled_candidate": se,
                "raw_abs_action_vs_original_actions_abs_error": ae,
                "raw_state_vs_original_robot_obs_abs_error": ste,
            }
        )

    unscaled = np.array(unscaled_errors, dtype=np.float64)
    scaled = np.array(scaled_errors, dtype=np.float64)
    abs_err = np.array(abs_action_errors, dtype=np.float64)
    st_err = np.array(state_errors, dtype=np.float64)

    report.update(
        {
            "status": "completed" if len(records) == args.samples else "completed_with_missing_samples",
            "samples_compared": len(records),
            "missing_samples": missing[:50],
            "missing_samples_count": len(missing),
            "unscaled_vs_original_rel_actions": summarize_errors(unscaled),
            "scaled_candidate_vs_original_rel_actions": summarize_errors(scaled),
            "raw_sixpigs_abs_action_vs_original_actions": summarize_errors(abs_err),
            "raw_sixpigs_state_vs_original_robot_obs": summarize_errors(st_err),
            "pass_unscaled_tol": bool(unscaled.size and float(unscaled.max()) <= args.tolerance),
            "pass_scaled_candidate_tol": bool(scaled.size and float(scaled.max()) <= args.tolerance),
            "tolerance": args.tolerance,
            "records_preview": records[: min(10, len(records))],
        }
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-dir", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--raw-sixpigs-dir", type=Path, default=DEFAULT_RAW_SIXPIGS)
    parser.add_argument("--derived-rel-dir", type=Path, default=DEFAULT_DERIVED)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(jsonable(report), f, ensure_ascii=False, indent=2)
    print(json.dumps(jsonable(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
