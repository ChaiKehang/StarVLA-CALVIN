#!/usr/bin/env python3
"""Build XYZ/RPY/gripper Intent labels and a leakage-free ABC split.

The source datasets are read-only.  Scaled training parquet files are copied
to a new derived dataset and augmented with three scalar label columns.  Train
and validation manifests are trajectory-level, task-stratified, and use only
the train episodes when estimating XYZ/RPY bin thresholds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


ROOT = Path("/home/data/datasets/kehang-CALVIN/calvin/lerobot")
DEFAULT_REL = ROOT / "sixpigs1_calvin2lerobotV21_ABC_D_scnet_rel"
DEFAULT_RAW = ROOT / "sixpigs1_calvin2lerobotV21_ABC_D_scnet_raw"
DEFAULT_SCALED = ROOT / "sixpigs1_calvin2lerobotV21_ABC_D_scnet_rel_calvin_scaled"
DEFAULT_DST = (
    ROOT
    / "sixpigs1_calvin2lerobotV21_ABC_D_scnet_rel_calvin_scaled_intent_factorized_h8"
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def episode_path(root: Path, info: dict, episode_id: int) -> Path:
    return root / info["data_path"].format(
        episode_chunk=episode_id // int(info.get("chunks_size", 1000)),
        episode_index=episode_id,
    )


def stack(series: pd.Series) -> np.ndarray:
    return np.stack(series.to_numpy()).astype(np.float64, copy=False)


def horizon_sum(values: np.ndarray, horizon: int) -> np.ndarray:
    prefix = np.concatenate(
        [np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)], axis=0
    )
    start = np.arange(len(values))
    end = np.minimum(start + horizon, len(values))
    return prefix[end] - prefix[start]


def horizon_mean(values: np.ndarray, horizon: int) -> np.ndarray:
    prefix = np.concatenate([[0.0], np.cumsum(values)])
    start = np.arange(len(values))
    end = np.minimum(start + horizon, len(values))
    return (prefix[end] - prefix[start]) / np.maximum(end - start, 1)


def relative_rotation_vectors(
    states: np.ndarray, absolute_actions: np.ndarray, horizon: int
) -> np.ndarray:
    end = np.minimum(np.arange(len(states)) + horizon - 1, len(states) - 1)
    current = Rotation.from_euler("xyz", states[:, 3:6])
    target = Rotation.from_euler("xyz", absolute_actions[end, 3:6])
    return (current.inv() * target).as_rotvec()


def thresholds(values: np.ndarray) -> tuple[float, float]:
    q20, q60 = np.quantile(np.abs(values).reshape(-1), [0.2, 0.6])
    if not 0 <= q20 < q60:
        raise ValueError(f"invalid thresholds q20={q20}, q60={q60}")
    return float(q20), float(q60)


def class125(values: np.ndarray, q20: float, q60: float) -> np.ndarray:
    bins = np.full(values.shape, 2, dtype=np.uint8)
    bins[values < -q60] = 0
    bins[(values >= -q60) & (values < -q20)] = 1
    bins[(values >= q20) & (values < q60)] = 3
    bins[values >= q60] = 4
    return (25 * bins[:, 0] + 5 * bins[:, 1] + bins[:, 2]).astype(np.uint8)


def gripper_class5(values: np.ndarray) -> np.ndarray:
    return np.digitize(values, [-0.6, -0.2, 0.2, 0.6]).astype(np.uint8)


def make_split(episodes: list[dict], seed: int, val_fraction: float):
    groups = defaultdict(list)
    for episode in episodes:
        groups[tuple(episode.get("tasks", []))].append(int(episode["episode_index"]))
    rng = np.random.default_rng(seed)
    train, val = [], []
    for task in sorted(groups, key=str):
        ids = np.asarray(sorted(groups[task]), dtype=np.int64)
        rng.shuffle(ids)
        count = 0 if len(ids) == 1 else max(
            1, int(round(len(ids) * val_fraction))
        )
        count = min(count, max(len(ids) - 1, 0))
        val.extend(ids[:count].tolist())
        train.extend(ids[count:].tolist())
    if set(train) & set(val):
        raise AssertionError("trajectory split overlap")
    if set(train) | set(val) != {
        int(row["episode_index"]) for row in episodes
    }:
        raise AssertionError("trajectory split does not cover every episode")
    return sorted(train), sorted(val)


def load_raw_targets(
    rel_root: Path,
    raw_root: Path,
    info: dict,
    episode_id: int,
    horizon: int,
):
    rel = pd.read_parquet(
        episode_path(rel_root, info, episode_id),
        columns=["action", "frame_index"],
    )
    raw = pd.read_parquet(
        episode_path(raw_root, info, episode_id),
        columns=["observation.state", "action", "frame_index"],
    )
    raw = raw.set_index("frame_index").loc[rel["frame_index"].tolist()]
    relative_actions = stack(rel["action"])
    states = stack(raw["observation.state"])
    absolute_actions = stack(raw["action"])
    if not (len(relative_actions) == len(states) == len(absolute_actions)):
        raise ValueError(f"episode {episode_id} source length mismatch")
    xyz = horizon_sum(relative_actions[:, :3], horizon)
    rpy = relative_rotation_vectors(states, absolute_actions, horizon)
    gripper = horizon_mean(absolute_actions[:, 6], horizon)
    return xyz, rpy, gripper


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rel-src", type=Path, default=DEFAULT_REL)
    parser.add_argument("--raw-src", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--scaled-src", type=Path, default=DEFAULT_SCALED)
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=250)
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be in (0,1)")
    if args.dst.exists():
        if not args.overwrite:
            raise SystemExit(f"Destination already exists: {args.dst}")
        shutil.rmtree(args.dst)

    infos = [read_json(root / "meta/info.json") for root in (args.rel_src, args.raw_src, args.scaled_src)]
    for key in ("total_episodes", "data_path", "chunks_size"):
        if len({json.dumps(info.get(key), sort_keys=True) for info in infos}) != 1:
            raise ValueError(f"source metadata mismatch for {key}")
    if infos[0]["total_frames"] != infos[2]["total_frames"]:
        raise ValueError("relative and scaled source frame counts do not match")
    info = infos[-1]
    episodes = read_jsonl(args.scaled_src / "meta/episodes.jsonl")
    train_ids, val_ids = make_split(episodes, args.seed, args.val_fraction)
    train_set = set(train_ids)

    xyz_train, rpy_train = [], []
    for ordinal, episode_id in enumerate(train_ids, start=1):
        xyz, rpy, _ = load_raw_targets(
            args.rel_src, args.raw_src, info, episode_id, args.horizon
        )
        xyz_train.append(xyz)
        rpy_train.append(rpy)
        if args.progress_every and ordinal % args.progress_every == 0:
            print(f"[thresholds] {ordinal}/{len(train_ids)}", flush=True)
    xyz_q20, xyz_q60 = thresholds(np.concatenate(xyz_train))
    rpy_q20, rpy_q60 = thresholds(np.concatenate(rpy_train))
    del xyz_train, rpy_train

    shutil.copytree(args.scaled_src / "meta", args.dst / "meta")
    videos = args.scaled_src / "videos"
    if videos.exists():
        os.symlink(videos, args.dst / "videos", target_is_directory=True)

    counts = {
        "xyz": np.zeros(125, dtype=np.int64),
        "rpy": np.zeros(125, dtype=np.int64),
        "gripper": np.zeros(5, dtype=np.int64),
    }
    train_gripper_counts = np.zeros(5, dtype=np.int64)
    for ordinal, episode in enumerate(episodes, start=1):
        episode_id = int(episode["episode_index"])
        xyz, rpy, gripper = load_raw_targets(
            args.rel_src, args.raw_src, info, episode_id, args.horizon
        )
        xyz_ids = class125(xyz, xyz_q20, xyz_q60)
        rpy_ids = class125(rpy, rpy_q20, rpy_q60)
        gripper_ids = gripper_class5(gripper)
        frame = pd.read_parquet(episode_path(args.scaled_src, info, episode_id))
        if len(frame) != len(xyz_ids):
            raise ValueError(f"episode {episode_id} scaled-source length mismatch")
        frame["intent.xyz_class_id"] = xyz_ids
        frame["intent.rpy_class_id"] = rpy_ids
        frame["intent.gripper_class_id"] = gripper_ids
        out = episode_path(args.dst, info, episode_id)
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(out, index=False)
        counts["xyz"] += np.bincount(xyz_ids, minlength=125)
        counts["rpy"] += np.bincount(rpy_ids, minlength=125)
        counts["gripper"] += np.bincount(gripper_ids, minlength=5)
        if episode_id in train_set:
            train_gripper_counts += np.bincount(gripper_ids, minlength=5)
        if args.progress_every and ordinal % args.progress_every == 0:
            print(f"[write] {ordinal}/{len(episodes)}", flush=True)

    output_info = read_json(args.dst / "meta/info.json")
    output_info.setdefault("features", {}).update(
        {
            "intent.xyz_class_id": {"dtype": "uint8", "shape": [1], "names": None},
            "intent.rpy_class_id": {"dtype": "uint8", "shape": [1], "names": None},
            "intent.gripper_class_id": {"dtype": "uint8", "shape": [1], "names": None},
        }
    )
    write_json(args.dst / "meta/info.json", output_info)

    split_dir = args.dst / "meta/splits"
    split_common = {
        "seed": args.seed,
        "strategy": "task_stratified_trajectory",
        "val_fraction": args.val_fraction,
        "source_episodes_sha256": sha256(args.scaled_src / "meta/episodes.jsonl"),
    }
    write_json(split_dir / f"factorized_seed{args.seed}_train.json", {
        **split_common, "split": "train", "episode_ids": train_ids,
    })
    write_json(split_dir / f"factorized_seed{args.seed}_val.json", {
        **split_common, "split": "validation", "episode_ids": val_ids,
    })

    inverse_sqrt = 1.0 / np.sqrt(np.maximum(train_gripper_counts, 1))
    inverse_sqrt /= np.sum(
        inverse_sqrt * train_gripper_counts
    ) / np.maximum(train_gripper_counts.sum(), 1)
    inverse_sqrt = np.clip(inverse_sqrt, 0.5, 3.0)
    write_json(args.dst / "meta/factorized_intent_config.json", {
        "format_version": 1,
        "horizon": args.horizon,
        "rotation": "R_current.inv() * R_future_target; scipy Rotation.as_rotvec",
        "xyz_thresholds": {"q20": xyz_q20, "q60": xyz_q60},
        "rpy_thresholds": {"q20": rpy_q20, "q60": rpy_q60},
        "gripper_thresholds": [-0.6, -0.2, 0.2, 0.6],
        "gripper_train_class_counts": train_gripper_counts.tolist(),
        "gripper_loss_weights": inverse_sqrt.tolist(),
        "class_counts_all": {name: value.tolist() for name, value in counts.items()},
        "train_episodes": len(train_ids),
        "validation_episodes": len(val_ids),
    })
    print(f"Derived factorized dataset: {args.dst}")
    print(f"train={len(train_ids)}, validation={len(val_ids)}")


if __name__ == "__main__":
    main()
