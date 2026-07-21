#!/usr/bin/env python3
"""Derive horizon-aware spatial intent labels for the E1 CALVIN dataset.

The intent target is the net XYZ displacement over the next ``horizon``
relative actions.  Labels are computed from the unscaled relative-action
dataset, while all training columns are copied from the CALVIN-scaled E0
dataset.  Neither source dataset is modified.

For the default 5x5x5 target, each axis is assigned to one of five symmetric
bins using pooled absolute-displacement quantiles q20 and q60.  The joint class
ID is ``25 * bx + 5 * by + bz`` and therefore lies in [0, 124].
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DATASET_ROOT = Path("/home/data/datasets/kehang-CALVIN/calvin/lerobot")
DEFAULT_REL_SRC = DATASET_ROOT / "sixpigs1_calvin2lerobotV21_ABC_D_scnet_rel"
DEFAULT_SCALED_SRC = (
    DATASET_ROOT / "sixpigs1_calvin2lerobotV21_ABC_D_scnet_rel_calvin_scaled"
)
DEFAULT_DST = (
    DATASET_ROOT
    / "sixpigs1_calvin2lerobotV21_ABC_D_scnet_rel_calvin_scaled_intent125_h8"
)

ACTION_KEY = "action"
INTENT_CLASS_KEY = "intent.class_id"
INTENT_DISPLACEMENT_KEY = "intent.displacement_xyz"
ALIGNMENT_KEYS = ("episode_index", "frame_index", "index")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(value), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stack_vector_column(series: pd.Series, dtype: np.dtype = np.float32) -> np.ndarray:
    return np.stack(series.to_numpy()).astype(dtype, copy=False)


def net_displacement(relative_actions: np.ndarray, horizon: int) -> np.ndarray:
    """Return truncated horizon sums, equivalent to zero-padding episode tails."""
    xyz = np.asarray(relative_actions[:, :3], dtype=np.float64)
    prefix = np.concatenate(
        [np.zeros((1, 3), dtype=np.float64), np.cumsum(xyz, axis=0)], axis=0
    )
    starts = np.arange(len(xyz), dtype=np.int64)
    ends = np.minimum(starts + horizon, len(xyz))
    return (prefix[ends] - prefix[starts]).astype(np.float32)


def axis_bins(displacement: np.ndarray, q20: float, q60: float) -> np.ndarray:
    """Map XYZ displacement values to symmetric five-way bins in [0, 4]."""
    bins = np.full(displacement.shape, 2, dtype=np.uint8)
    bins[displacement < -q60] = 0
    bins[(displacement >= -q60) & (displacement < -q20)] = 1
    bins[(displacement >= q20) & (displacement < q60)] = 3
    bins[displacement >= q60] = 4
    return bins


def joint_class_ids(displacement: np.ndarray, q20: float, q60: float) -> np.ndarray:
    bins = axis_bins(displacement, q20, q60).astype(np.uint16)
    class_ids = 25 * bins[:, 0] + 5 * bins[:, 1] + bins[:, 2]
    if class_ids.size and (class_ids.min() < 0 or class_ids.max() > 124):
        raise AssertionError("intent class IDs must be in [0, 124]")
    return class_ids.astype(np.uint8)


def summarize_array(array: np.ndarray) -> dict[str, Any]:
    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
        "count": [int(len(values))],
    }


def episode_file(root: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunk_size = int(info.get("chunks_size", 1000))
    return root / info["data_path"].format(
        episode_chunk=episode_index // chunk_size,
        episode_index=episode_index,
    )


def validate_sources(
    rel_src: Path, scaled_src: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    required = ["meta/info.json", "meta/episodes.jsonl", "meta/tasks.jsonl", "data"]
    for root in (rel_src, scaled_src):
        missing = [item for item in required if not (root / item).exists()]
        if missing:
            raise FileNotFoundError(f"{root} is missing required entries: {missing}")

    rel_info = read_json(rel_src / "meta/info.json")
    scaled_info = read_json(scaled_src / "meta/info.json")
    for key in ("total_episodes", "total_frames", "chunks_size", "data_path"):
        if rel_info.get(key) != scaled_info.get(key):
            raise ValueError(
                f"source metadata mismatch for {key}: "
                f"relative={rel_info.get(key)!r}, scaled={scaled_info.get(key)!r}"
            )

    rel_episodes_path = rel_src / "meta/episodes.jsonl"
    scaled_episodes_path = scaled_src / "meta/episodes.jsonl"
    if sha256_file(rel_episodes_path) != sha256_file(scaled_episodes_path):
        raise ValueError("relative and scaled episodes.jsonl files are not identical")

    episodes = read_jsonl(scaled_episodes_path)
    if len(episodes) != int(scaled_info["total_episodes"]):
        raise ValueError("episodes.jsonl count does not match info.json")
    if sum(int(row["length"]) for row in episodes) != int(scaled_info["total_frames"]):
        raise ValueError("episode lengths do not sum to info.json total_frames")
    return scaled_info, episodes


def prepare_destination(dst: Path, overwrite: bool, overwrite_smoke: bool) -> None:
    if dst.exists():
        may_remove = overwrite or (overwrite_smoke and dst.name.endswith("_smoke"))
        if not may_remove:
            raise SystemExit(
                f"Destination exists: {dst}\n"
                "Use --overwrite, or --overwrite-smoke for a *_smoke destination."
            )
        shutil.rmtree(dst)
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    (dst / "data").mkdir(parents=True, exist_ok=True)


def collect_displacements(
    rel_src: Path,
    info: dict[str, Any],
    episodes: list[dict[str, Any]],
    horizon: int,
    progress_every: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for ordinal, episode in enumerate(episodes, start=1):
        episode_index = int(episode["episode_index"])
        path = episode_file(rel_src, info, episode_index)
        frame = pd.read_parquet(path, columns=[ACTION_KEY])
        if len(frame) != int(episode["length"]):
            raise ValueError(
                f"relative episode {episode_index} length mismatch: "
                f"parquet={len(frame)}, metadata={episode['length']}"
            )
        actions = stack_vector_column(frame[ACTION_KEY])
        if actions.ndim != 2 or actions.shape[1] < 3:
            raise ValueError(f"invalid action shape in {path}: {actions.shape}")
        chunks.append(net_displacement(actions, horizon))
        if progress_every and (
            ordinal % progress_every == 0 or ordinal == len(episodes)
        ):
            frames = sum(len(item) for item in chunks)
            print(
                f"[threshold-pass] {ordinal}/{len(episodes)} episodes, "
                f"{frames} displacement labels",
                flush=True,
            )
    return np.concatenate(chunks, axis=0)


def copy_base_metadata(
    scaled_src: Path,
    dst: Path,
    selected_episodes: list[dict[str, Any]],
) -> None:
    shutil.copy2(scaled_src / "meta/tasks.jsonl", dst / "meta/tasks.jsonl")
    source_conversion = scaled_src / "meta/action_conversion.json"
    if source_conversion.exists():
        shutil.copy2(
            source_conversion, dst / "meta/source_action_conversion.json"
        )
    attributes = scaled_src / ".gitattributes"
    if attributes.exists():
        shutil.copy2(attributes, dst / ".gitattributes")
    write_jsonl(dst / "meta/episodes.jsonl", selected_episodes)

    videos = scaled_src / "videos"
    if videos.exists() or videos.is_symlink():
        os.symlink(videos, dst / "videos", target_is_directory=True)


def write_steps_index(dst: Path, episodes: list[dict[str, Any]]) -> None:
    """Write a step cache that exactly matches the selected episodes."""
    steps = [
        (int(episode["episode_index"]), frame_index)
        for episode in episodes
        for frame_index in range(int(episode["length"]))
    ]
    cache = {
        "config_key": "derived_e1_intent",
        "steps": steps,
        "num_trajectories": len(episodes),
        "total_steps": len(steps),
        "computed_timestamp": datetime.now(timezone.utc).isoformat(),
        "delete_pause_frame": False,
    }
    with (dst / "meta/steps_data_index.pkl").open("wb") as handle:
        pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)


def update_info(
    source_info: dict[str, Any], selected_episodes: list[dict[str, Any]]
) -> dict[str, Any]:
    info = json.loads(json.dumps(source_info))
    info["total_episodes"] = len(selected_episodes)
    info["total_frames"] = sum(int(row["length"]) for row in selected_episodes)
    info["splits"] = {"train": f"0:{len(selected_episodes)}"}
    features = info.setdefault("features", {})
    features[INTENT_CLASS_KEY] = {
        "dtype": "uint8",
        "shape": [1],
        "names": None,
    }
    features[INTENT_DISPLACEMENT_KEY] = {
        "dtype": "float32",
        "shape": [3],
        "names": ["x", "y", "z"],
    }
    return info


def update_modality(source: dict[str, Any]) -> dict[str, Any]:
    modality = json.loads(json.dumps(source))
    annotation = modality.setdefault("annotation", {})
    annotation["intent.spatial.class_id"] = {"original_key": INTENT_CLASS_KEY}
    annotation["intent.spatial.displacement_xyz"] = {
        "original_key": INTENT_DISPLACEMENT_KEY
    }
    return modality


def update_stats_gr00t(
    source: dict[str, Any], class_ids: np.ndarray, displacements: np.ndarray
) -> dict[str, Any]:
    stats = json.loads(json.dumps(source))
    statistics = stats.setdefault("statistics", {})
    statistics[INTENT_CLASS_KEY] = summarize_array(class_ids)
    statistics[INTENT_DISPLACEMENT_KEY] = summarize_array(displacements)
    return stats


def update_episode_stats(
    scaled_src: Path,
    dst: Path,
    selected_indices: set[int],
    intent_stats: dict[int, dict[str, Any]],
) -> None:
    source_path = scaled_src / "meta/episodes_stats.jsonl"
    target_path = dst / "meta/episodes_stats.jsonl"
    seen: set[int] = set()
    with source_path.open("r", encoding="utf-8") as source, target_path.open(
        "w", encoding="utf-8"
    ) as target:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            episode_index = int(row["episode_index"])
            if episode_index not in selected_indices:
                continue
            row.setdefault("stats", {}).update(intent_stats[episode_index])
            target.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")
            seen.add(episode_index)
    missing = selected_indices - seen
    if missing:
        raise ValueError(f"episodes_stats.jsonl is missing episodes: {sorted(missing)[:10]}")


def make_class_mapping() -> list[dict[str, Any]]:
    names = [
        "strong_negative",
        "weak_negative",
        "near_zero",
        "weak_positive",
        "strong_positive",
    ]
    mapping: list[dict[str, Any]] = []
    for bx in range(5):
        for by in range(5):
            for bz in range(5):
                mapping.append(
                    {
                        "class_id": 25 * bx + 5 * by + bz,
                        "axis_bins": [bx, by, bz],
                        "axis_names": [names[bx], names[by], names[bz]],
                    }
                )
    return mapping


def normalized_entropy(counts: np.ndarray) -> float:
    probabilities = counts[counts > 0].astype(np.float64)
    probabilities /= probabilities.sum()
    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(entropy / math.log(len(counts)))


def write_labeled_data(
    rel_src: Path,
    scaled_src: Path,
    dst: Path,
    info: dict[str, Any],
    episodes: list[dict[str, Any]],
    horizon: int,
    q20: float,
    q60: float,
    progress_every: int,
) -> tuple[np.ndarray, np.ndarray, dict[int, dict[str, Any]]]:
    all_displacements: list[np.ndarray] = []
    all_class_ids: list[np.ndarray] = []
    episode_stats: dict[int, dict[str, Any]] = {}

    for ordinal, episode in enumerate(episodes, start=1):
        episode_index = int(episode["episode_index"])
        rel_path = episode_file(rel_src, info, episode_index)
        scaled_path = episode_file(scaled_src, info, episode_index)
        dst_path = episode_file(dst, info, episode_index)

        rel_frame = pd.read_parquet(
            rel_path, columns=[ACTION_KEY, *ALIGNMENT_KEYS]
        )
        scaled_frame = pd.read_parquet(scaled_path)
        expected_length = int(episode["length"])
        if len(rel_frame) != expected_length or len(scaled_frame) != expected_length:
            raise ValueError(
                f"episode {episode_index} length mismatch: relative={len(rel_frame)}, "
                f"scaled={len(scaled_frame)}, metadata={expected_length}"
            )
        for key in ALIGNMENT_KEYS:
            if not np.array_equal(rel_frame[key].to_numpy(), scaled_frame[key].to_numpy()):
                raise ValueError(f"episode {episode_index} is misaligned at column {key}")

        relative_actions = stack_vector_column(rel_frame[ACTION_KEY])
        displacement = net_displacement(relative_actions, horizon)
        class_ids = joint_class_ids(displacement, q20, q60)

        out_frame = scaled_frame.copy()
        out_frame[INTENT_CLASS_KEY] = class_ids
        out_frame[INTENT_DISPLACEMENT_KEY] = [row for row in displacement]
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        out_frame.to_parquet(dst_path, index=False)

        all_displacements.append(displacement)
        all_class_ids.append(class_ids)
        episode_stats[episode_index] = {
            INTENT_CLASS_KEY: summarize_array(class_ids),
            INTENT_DISPLACEMENT_KEY: summarize_array(displacement),
        }

        if progress_every and (
            ordinal % progress_every == 0 or ordinal == len(episodes)
        ):
            frames = sum(len(item) for item in all_class_ids)
            print(
                f"[write-pass] {ordinal}/{len(episodes)} episodes, "
                f"{frames} labeled frames -> {dst}",
                flush=True,
            )

    return (
        np.concatenate(all_class_ids, axis=0),
        np.concatenate(all_displacements, axis=0),
        episode_stats,
    )


def audit_output(
    rel_src: Path,
    scaled_src: Path,
    dst: Path,
    info: dict[str, Any],
    episodes: list[dict[str, Any]],
    horizon: int,
    q20: float,
    q60: float,
    sample_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    total_frames = sum(int(row["length"]) for row in episodes)
    count = min(sample_count, total_frames)
    rng = np.random.default_rng(seed)
    flat_offsets = sorted(rng.choice(total_frames, size=count, replace=False).tolist())

    audits: list[dict[str, Any]] = []
    offset_cursor = 0
    sample_cursor = 0
    for episode in episodes:
        if sample_cursor >= len(flat_offsets):
            break
        length = int(episode["length"])
        end = offset_cursor + length
        local_rows: list[int] = []
        while sample_cursor < len(flat_offsets) and flat_offsets[sample_cursor] < end:
            local_rows.append(flat_offsets[sample_cursor] - offset_cursor)
            sample_cursor += 1
        if local_rows:
            episode_index = int(episode["episode_index"])
            rel_frame = pd.read_parquet(
                episode_file(rel_src, info, episode_index), columns=[ACTION_KEY]
            )
            scaled_frame = pd.read_parquet(
                episode_file(scaled_src, info, episode_index)
            )
            output_frame = pd.read_parquet(episode_file(dst, info, episode_index))
            displacement = net_displacement(
                stack_vector_column(rel_frame[ACTION_KEY]), horizon
            )
            expected_ids = joint_class_ids(displacement, q20, q60)
            for local_row in local_rows:
                stored_displacement = np.asarray(
                    output_frame.iloc[local_row][INTENT_DISPLACEMENT_KEY],
                    dtype=np.float32,
                )
                stored_id = int(output_frame.iloc[local_row][INTENT_CLASS_KEY])
                if not np.allclose(
                    stored_displacement, displacement[local_row], atol=1e-7, rtol=0
                ):
                    raise AssertionError(
                        f"stored displacement audit failed: episode={episode_index}, "
                        f"frame={local_row}"
                    )
                if stored_id != int(expected_ids[local_row]):
                    raise AssertionError(
                        f"stored class audit failed: episode={episode_index}, "
                        f"frame={local_row}"
                    )
                for key in scaled_frame.columns:
                    left = scaled_frame.iloc[local_row][key]
                    right = output_frame.iloc[local_row][key]
                    if isinstance(left, np.ndarray):
                        equal = np.array_equal(left, right)
                    else:
                        equal = left == right
                    if not bool(equal):
                        raise AssertionError(
                            f"source preservation audit failed: episode={episode_index}, "
                            f"frame={local_row}, column={key}"
                        )
                audits.append(
                    {
                        "episode_index": episode_index,
                        "frame_index": int(output_frame.iloc[local_row]["frame_index"]),
                        "displacement_xyz": displacement[local_row].tolist(),
                        "axis_bins": axis_bins(
                            displacement[local_row : local_row + 1], q20, q60
                        )[0].tolist(),
                        "class_id": stored_id,
                    }
                )
        offset_cursor = end
    if len(audits) != count:
        raise AssertionError(f"expected {count} audit samples, produced {len(audits)}")
    return audits


def derive_dataset(args: argparse.Namespace) -> dict[str, Any]:
    info, all_episodes = validate_sources(args.rel_src, args.scaled_src)
    selected_episodes = all_episodes
    if args.max_episodes is not None:
        selected_episodes = all_episodes[: args.max_episodes]
    if not selected_episodes:
        raise ValueError("no episodes selected")

    if (args.q20 is None) != (args.q60 is None):
        raise ValueError("--q20 and --q60 must be supplied together")

    if args.q20 is None:
        threshold_displacements = collect_displacements(
            args.rel_src,
            info,
            all_episodes,
            args.horizon,
            args.progress_every,
        )
        pooled_absolute = np.abs(threshold_displacements).reshape(-1)
        q20, q60 = np.quantile(pooled_absolute, [0.2, 0.6]).tolist()
        threshold_source = "computed_from_all_training_episodes"
        threshold_frame_count = len(threshold_displacements)
    else:
        q20, q60 = float(args.q20), float(args.q60)
        threshold_source = "provided_by_cli"
        threshold_frame_count = None
    if not (0 <= q20 < q60):
        raise ValueError(f"expected 0 <= q20 < q60, got q20={q20}, q60={q60}")

    prepare_destination(args.dst, args.overwrite, args.overwrite_smoke)
    copy_base_metadata(args.scaled_src, args.dst, selected_episodes)
    write_steps_index(args.dst, selected_episodes)

    class_ids, displacements, per_episode_stats = write_labeled_data(
        args.rel_src,
        args.scaled_src,
        args.dst,
        info,
        selected_episodes,
        args.horizon,
        q20,
        q60,
        args.progress_every,
    )

    output_info = update_info(info, selected_episodes)
    write_json(args.dst / "meta/info.json", output_info)
    source_modality = read_json(args.scaled_src / "meta/modality.json")
    write_json(args.dst / "meta/modality.json", update_modality(source_modality))
    source_stats = read_json(args.scaled_src / "meta/stats_gr00t.json")
    write_json(
        args.dst / "meta/stats_gr00t.json",
        update_stats_gr00t(source_stats, class_ids, displacements),
    )
    update_episode_stats(
        args.scaled_src,
        args.dst,
        {int(row["episode_index"]) for row in selected_episodes},
        per_episode_stats,
    )

    audits = audit_output(
        args.rel_src,
        args.scaled_src,
        args.dst,
        info,
        selected_episodes,
        args.horizon,
        q20,
        q60,
        args.audit_samples,
        args.seed,
    )

    counts = np.bincount(class_ids.astype(np.int64), minlength=125)
    axis = axis_bins(displacements, q20, q60)
    axis_counts = {
        name: np.bincount(axis[:, index], minlength=5).tolist()
        for index, name in enumerate(("x", "y", "z"))
    }
    metadata_hashes: dict[str, dict[str, str]] = {}
    for label, root in (("relative", args.rel_src), ("scaled", args.scaled_src)):
        metadata_hashes[label] = {
            name: sha256_file(root / "meta" / name)
            for name in ("info.json", "episodes.jsonl", "tasks.jsonl")
        }

    config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "format_version": 1,
        "source_datasets": {
            "unscaled_relative_actions": str(args.rel_src.resolve()),
            "scaled_training_samples": str(args.scaled_src.resolve()),
        },
        "derived_dataset": str(args.dst.resolve()),
        "source_metadata_sha256": metadata_hashes,
        "label_definition": {
            "name": "future_net_translation_intent",
            "source_column": ACTION_KEY,
            "source_action_semantics": "unscaled one-step relative XYZ displacement in meters",
            "formula": "sum(action[t:t+horizon, 0:3])",
            "horizon": args.horizon,
            "episode_tail_policy": "zero_pad_equivalent_truncated_sum",
            "output_displacement_column": INTENT_DISPLACEMENT_KEY,
            "output_class_column": INTENT_CLASS_KEY,
        },
        "binning": {
            "scheme": "pooled_absolute_zero_symmetric_quantiles",
            "quantiles": [0.2, 0.6],
            "q20_meters": q20,
            "q60_meters": q60,
            "threshold_source": threshold_source,
            "threshold_frame_count": threshold_frame_count,
            "axis_bin_names": [
                "strong_negative",
                "weak_negative",
                "near_zero",
                "weak_positive",
                "strong_positive",
            ],
            "joint_class_formula": "25 * bx + 5 * by + bz",
            "num_classes": 125,
            "class_mapping": make_class_mapping(),
        },
        "statistics": {
            "total_episodes": len(selected_episodes),
            "total_frames": len(class_ids),
            "class_counts": counts.tolist(),
            "class_fractions": (counts / counts.sum()).tolist(),
            "occupied_classes": int(np.count_nonzero(counts)),
            "min_nonzero_class_count": int(counts[counts > 0].min()),
            "max_class_count": int(counts.max()),
            "normalized_class_entropy": normalized_entropy(counts),
            "axis_bin_counts": axis_counts,
            "displacement_xyz": summarize_array(displacements),
        },
        "storage": {
            "parquet_policy": "copy_scaled_training_columns_and_append_intent_columns",
            "videos_policy": "symlink_to_scaled_source_videos",
            "videos_link_target": os.readlink(args.dst / "videos")
            if (args.dst / "videos").is_symlink()
            else None,
        },
        "audit": {
            "seed": args.seed,
            "requested_samples": args.audit_samples,
            "validated_samples": len(audits),
            "checks": [
                "stored displacement equals recomputed horizon sum",
                "stored class ID equals recomputed symmetric-quantile label",
                "all original scaled parquet columns are unchanged",
            ],
            "samples": audits,
        },
    }
    write_json(args.dst / "meta/intent_label_config.json", config)
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add horizon-8 5x5x5 spatial intent labels to the E0 dataset."
    )
    parser.add_argument("--rel-src", type=Path, default=DEFAULT_REL_SRC)
    parser.add_argument("--scaled-src", type=Path, default=DEFAULT_SCALED_SRC)
    parser.add_argument("--dst", type=Path, default=None)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--q20", type=float, default=None)
    parser.add_argument("--q60", type=float, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--audit-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-smoke", action="store_true")
    args = parser.parse_args()

    if args.horizon <= 0:
        parser.error("--horizon must be positive")
    if args.max_episodes is not None and args.max_episodes <= 0:
        parser.error("--max-episodes must be positive")
    if args.audit_samples < 0:
        parser.error("--audit-samples must be non-negative")

    if args.dst is None:
        args.dst = (
            Path(str(DEFAULT_DST) + "_smoke")
            if args.max_episodes is not None
            else DEFAULT_DST
        )
    return args


def main() -> None:
    args = parse_args()
    report = derive_dataset(args)
    summary = {
        "derived_dataset": report["derived_dataset"],
        "total_episodes": report["statistics"]["total_episodes"],
        "total_frames": report["statistics"]["total_frames"],
        "q20_meters": report["binning"]["q20_meters"],
        "q60_meters": report["binning"]["q60_meters"],
        "occupied_classes": report["statistics"]["occupied_classes"],
        "normalized_class_entropy": report["statistics"][
            "normalized_class_entropy"
        ],
        "validated_samples": report["audit"]["validated_samples"],
    }
    print(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
