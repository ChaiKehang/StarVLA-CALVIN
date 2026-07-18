#!/usr/bin/env python3
"""Create a relative-action derivative of sixpigs CALVIN LeRobot v2.1.

The source dataset stores `action[t]` as an absolute next end-effector target
that is approximately equal to `observation.state[t + 1]`.  This script creates
a new dataset where `action` is converted to:

    [delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw, gripper_cmd]

The source dataset is never modified.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_SRC = Path(
    "/home/data/datasets/kehang-CALVIN/calvin/lerobot/"
    "sixpigs1_calvin2lerobotV21_ABC_D_scnet_raw"
)
DEFAULT_DST = Path(
    "/home/data/datasets/kehang-CALVIN/calvin/lerobot/"
    "sixpigs1_calvin2lerobotV21_ABC_D_scnet_rel"
)
DEFAULT_DST_CALVIN_SCALED = Path(
    "/home/data/datasets/kehang-CALVIN/calvin/lerobot/"
    "sixpigs1_calvin2lerobotV21_ABC_D_scnet_rel_calvin_scaled"
)

STATE_STARVLA_KEY = "observation.state_starvla"
SOURCE_STATE_KEY = "observation.state"
ACTION_KEY = "action"


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")


def to_jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def stack_vector_column(series: pd.Series, dtype: np.dtype = np.float32) -> np.ndarray:
    return np.stack(series.to_numpy()).astype(dtype, copy=False)


def column_matrix(values: pd.Series) -> np.ndarray:
    first = values.iloc[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        return np.stack(values.to_numpy()).astype(np.float64, copy=False)
    return values.to_numpy(dtype=np.float64).reshape(-1, 1)


def summarize_array(arr: np.ndarray) -> dict[str, list[float] | list[int]]:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return {
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "q01": np.quantile(arr, 0.01, axis=0).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).tolist(),
    }


def summarize_for_episodes_stats(df: pd.DataFrame, source_image_stats: dict[str, Any] | None) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for col in [
        SOURCE_STATE_KEY,
        STATE_STARVLA_KEY,
        ACTION_KEY,
        "timestamp",
        "episode_index",
        "frame_index",
        "index",
        "task_index",
    ]:
        if col in df.columns and len(df):
            col_stats = summarize_array(column_matrix(df[col]))
            col_stats["count"] = [int(len(df))]
            stats[col] = col_stats

    # Video statistics are expensive to recompute over ~1M frames.  They are not
    # consumed by StarVLA's stats_gr00t path, but keeping the keys makes the
    # metadata closer to the source LeRobot dataset.  Record the policy in
    # action_conversion.json.
    if source_image_stats:
        for col in ["observation.images.rgb_static", "observation.images.rgb_gripper"]:
            if col in source_image_stats:
                copied = json.loads(json.dumps(source_image_stats[col]))
                if isinstance(copied, dict) and "count" in copied:
                    copied["count"] = [int(len(df))]
                stats[col] = copied
    return stats


def make_stats_gr00t(arrays: dict[str, list[np.ndarray]]) -> dict[str, Any]:
    statistics: dict[str, Any] = {}
    for key, chunks in arrays.items():
        if not chunks:
            continue
        arr = np.concatenate(chunks, axis=0)
        statistics[key] = summarize_array(arr)
    return {
        "__format_version": 2,
        "__cache_config": {"mode": "abs"},
        "statistics": statistics,
    }


def make_modality_json() -> dict[str, Any]:
    def span(original_key: str, start: int, end: int, *, absolute: bool, rotation_type: str | None = None) -> dict[str, Any]:
        item: dict[str, Any] = {
            "start": start,
            "end": end,
            "original_key": original_key,
            "absolute": absolute,
        }
        if rotation_type is not None:
            item["rotation_type"] = rotation_type
        return item

    return {
        "state": {
            "x": span(STATE_STARVLA_KEY, 0, 1, absolute=True),
            "y": span(STATE_STARVLA_KEY, 1, 2, absolute=True),
            "z": span(STATE_STARVLA_KEY, 2, 3, absolute=True),
            "roll": span(STATE_STARVLA_KEY, 3, 4, absolute=True, rotation_type="euler_angles_rpy"),
            "pitch": span(STATE_STARVLA_KEY, 4, 5, absolute=True, rotation_type="euler_angles_rpy"),
            "yaw": span(STATE_STARVLA_KEY, 5, 6, absolute=True, rotation_type="euler_angles_rpy"),
            "pad": span(STATE_STARVLA_KEY, 6, 7, absolute=True),
            "gripper": span(STATE_STARVLA_KEY, 7, 8, absolute=True),
        },
        "action": {
            "x": span(ACTION_KEY, 0, 1, absolute=False),
            "y": span(ACTION_KEY, 1, 2, absolute=False),
            "z": span(ACTION_KEY, 2, 3, absolute=False),
            "roll": span(ACTION_KEY, 3, 4, absolute=False, rotation_type="euler_angles_rpy"),
            "pitch": span(ACTION_KEY, 4, 5, absolute=False, rotation_type="euler_angles_rpy"),
            "yaw": span(ACTION_KEY, 5, 6, absolute=False, rotation_type="euler_angles_rpy"),
            "gripper": span(ACTION_KEY, 6, 7, absolute=True),
        },
        "video": {
            "primary_image": {"original_key": "observation.images.rgb_static"},
            "wrist_image": {"original_key": "observation.images.rgb_gripper"},
        },
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"},
        },
    }


def update_info(info: dict[str, Any], total_episodes: int, total_frames: int) -> dict[str, Any]:
    out = json.loads(json.dumps(info))
    out["total_episodes"] = int(total_episodes)
    out["total_frames"] = int(total_frames)
    out["splits"] = {"train": f"0:{int(total_episodes)}"}
    features = out.setdefault("features", {})
    source_state = json.loads(json.dumps(features[SOURCE_STATE_KEY]))
    source_state["shape"] = [8]
    source_state["names"] = {
        "motors": ["x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"]
    }
    features[STATE_STARVLA_KEY] = source_state
    features[ACTION_KEY]["names"] = {
        "motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    }
    return out


def prepare_dst(dst: Path, overwrite: bool, overwrite_smoke: bool) -> None:
    if dst.exists():
        can_remove = overwrite or (overwrite_smoke and dst.name.endswith("_smoke"))
        if not can_remove:
            raise SystemExit(
                f"Destination exists: {dst}\n"
                "Use --overwrite for the full target or --overwrite-smoke for *_smoke targets."
            )
        shutil.rmtree(dst)
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    (dst / "data").mkdir(parents=True, exist_ok=True)


def convert_dataset(args: argparse.Namespace) -> dict[str, Any]:
    src: Path = args.src
    dst: Path = args.dst
    info = json.load((src / "meta/info.json").open("r", encoding="utf-8"))
    chunks_size = int(info.get("chunks_size", 1000))
    episodes = read_jsonl(src / "meta/episodes.jsonl")
    source_episode_stats = {
        int(row["episode_index"]): row.get("stats", {})
        for row in read_jsonl(src / "meta/episodes_stats.jsonl")
    }

    if args.max_episodes is not None:
        episodes = episodes[: int(args.max_episodes)]

    prepare_dst(dst, args.overwrite, args.overwrite_smoke)

    shutil.copy2(src / "meta/tasks.jsonl", dst / "meta/tasks.jsonl")
    if (src / ".gitattributes").exists():
        shutil.copy2(src / ".gitattributes", dst / ".gitattributes")

    videos_src = src / "videos"
    videos_dst = dst / "videos"
    if videos_src.exists():
        os.symlink(videos_src, videos_dst, target_is_directory=True)

    new_episodes: list[dict[str, Any]] = []
    new_episode_stats: list[dict[str, Any]] = []
    global_arrays: dict[str, list[np.ndarray]] = {
        SOURCE_STATE_KEY: [],
        STATE_STARVLA_KEY: [],
        ACTION_KEY: [],
        "timestamp": [],
        "episode_index": [],
        "frame_index": [],
        "index": [],
        "task_index": [],
    }

    conversion_errors: list[float] = []
    action_abs_ranges: list[np.ndarray] = []
    action_rel_ranges: list[np.ndarray] = []
    total_frames = 0

    for n, episode in enumerate(episodes, start=1):
        episode_index = int(episode["episode_index"])
        chunk = episode_index // chunks_size
        src_file = src / info["data_path"].format(
            episode_chunk=chunk, episode_index=episode_index
        )
        if not src_file.exists():
            raise FileNotFoundError(src_file)

        df = pd.read_parquet(src_file)
        if len(df) < 2:
            raise ValueError(f"Episode {episode_index} has fewer than 2 frames")

        state = stack_vector_column(df[SOURCE_STATE_KEY])
        action_abs = stack_vector_column(df[ACTION_KEY])
        action_rel = action_abs.copy()
        action_rel[:, 0:3] = action_abs[:, 0:3] - state[:, 0:3]
        action_rel[:, 3:6] = wrap_to_pi(action_abs[:, 3:6] - state[:, 3:6])
        action_rel[:, 6] = action_abs[:, 6]
        if args.calvin_scaled:
            action_rel[:, 0:3] *= 50.0
            action_rel[:, 3:6] *= 20.0
            action_rel[:, 0:6] = np.clip(action_rel[:, 0:6], -1.0, 1.0)

        # Drop the last frame so every target has a real t -> t+1 transition.
        keep = len(df) - 1
        out_df = df.iloc[:keep].copy()
        out_rel = action_rel[:keep].astype(np.float32, copy=False)
        out_state = state[:keep].astype(np.float32, copy=False)
        state_starvla = np.concatenate(
            [
                out_state[:, 0:6],
                np.zeros((keep, 1), dtype=np.float32),
                out_state[:, 6:7],
            ],
            axis=1,
        )

        out_df[ACTION_KEY] = [row for row in out_rel]
        out_df[STATE_STARVLA_KEY] = [row for row in state_starvla]
        out_df["frame_index"] = np.arange(keep, dtype=np.int64)
        out_df["index"] = np.arange(total_frames, total_frames + keep, dtype=np.int64)

        dst_file = dst / info["data_path"].format(
            episode_chunk=chunk, episode_index=episode_index
        )
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(dst_file, index=False)

        old_tasks = episode.get("tasks", [])
        new_episodes.append(
            {
                "episode_index": episode_index,
                "tasks": old_tasks,
                "length": int(keep),
            }
        )
        new_episode_stats.append(
            {
                "episode_index": episode_index,
                "stats": summarize_for_episodes_stats(
                    out_df,
                    source_episode_stats.get(episode_index, {}),
                ),
            }
        )

        for key in global_arrays:
            global_arrays[key].append(column_matrix(out_df[key]))

        compare_n = min(keep, len(state) - 1)
        next_err = np.abs(action_abs[:compare_n, :6] - state[1 : compare_n + 1, :6])
        conversion_errors.append(float(next_err.mean()))
        action_abs_ranges.append(
            np.stack([action_abs[:keep].min(axis=0), action_abs[:keep].max(axis=0)])
        )
        action_rel_ranges.append(
            np.stack([out_rel.min(axis=0), out_rel.max(axis=0)])
        )

        total_frames += keep
        if args.progress_every and (n % args.progress_every == 0 or n == len(episodes)):
            print(f"[convert] {n}/{len(episodes)} episodes, {total_frames} frames -> {dst}")

    new_info = update_info(info, len(new_episodes), total_frames)
    json.dump(to_jsonable(new_info), (dst / "meta/info.json").open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
    write_jsonl(dst / "meta/episodes.jsonl", new_episodes)
    write_jsonl(dst / "meta/episodes_stats.jsonl", new_episode_stats)
    json.dump(make_modality_json(), (dst / "meta/modality.json").open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(make_stats_gr00t(global_arrays), (dst / "meta/stats_gr00t.json").open("w", encoding="utf-8"), ensure_ascii=False, indent=2)

    abs_range = np.stack(action_abs_ranges)
    rel_range = np.stack(action_rel_ranges)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset_dir": str(src),
        "derived_dataset_dir": str(dst),
        "source_codebase_version": info.get("codebase_version"),
        "source_action_semantics": "absolute next end-effector target; action[t] is approximately observation.state[t+1]",
        "target_action_semantics": (
            "CALVIN-scaled relative action: clipped(delta_xyz * 50, delta_rpy * 20) plus gripper command"
            if args.calvin_scaled
            else "relative delta xyz/rpy plus gripper command"
        ),
        "formula": {
            "xyz": (
                "clip((action_abs[:3] - observation.state[:3]) * 50, -1, 1)"
                if args.calvin_scaled
                else "action_abs[:3] - observation.state[:3]"
            ),
            "rpy": (
                "clip(wrap_to_pi(action_abs[3:6] - observation.state[3:6]) * 20, -1, 1)"
                if args.calvin_scaled
                else "wrap_to_pi(action_abs[3:6] - observation.state[3:6])"
            ),
            "gripper": "action_abs[6]",
        },
        "calvin_scaled": bool(args.calvin_scaled),
        "calvin_scale_factors": [50.0, 50.0, 50.0, 20.0, 20.0, 20.0, 1.0]
        if args.calvin_scaled
        else None,
        "clip_range_first6": [-1.0, 1.0] if args.calvin_scaled else None,
        "last_frame_policy": "drop_last_frame_per_episode",
        "video_policy": "symlink_to_source_videos",
        "episode_image_stats_policy": "copied_from_source_with_count_adjusted; StarVLA uses stats_gr00t for low-dimensional stats",
        "total_episodes": len(new_episodes),
        "source_total_frames_selected": int(sum(int(ep["length"]) for ep in episodes)),
        "derived_total_frames": int(total_frames),
        "frames_removed": int(len(new_episodes)),
        "mean_abs_action_vs_next_state_error_first6": float(np.mean(conversion_errors)),
        "action_abs_min": abs_range[:, 0, :].min(axis=0).tolist(),
        "action_abs_max": abs_range[:, 1, :].max(axis=0).tolist(),
        "action_rel_min": rel_range[:, 0, :].min(axis=0).tolist(),
        "action_rel_max": rel_range[:, 1, :].max(axis=0).tolist(),
    }
    json.dump(to_jsonable(report), (dst / "meta/action_conversion.json").open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--dst", type=Path, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--calvin-scaled",
        action="store_true",
        help="Convert to official CALVIN rel_actions scale: xyz*50, rpy*20, clip first 6 dims to [-1, 1].",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-smoke", action="store_true")
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()
    if args.dst is None:
        default_dst = DEFAULT_DST_CALVIN_SCALED if args.calvin_scaled else DEFAULT_DST
        args.dst = Path(str(default_dst) + "_smoke") if args.max_episodes is not None else default_dst
    return args


def main() -> None:
    args = parse_args()
    report = convert_dataset(args)
    print(json.dumps(to_jsonable(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
