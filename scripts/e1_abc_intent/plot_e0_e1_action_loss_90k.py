#!/usr/bin/env python3
"""Build a presentation-ready E0/E1 Action DiT loss comparison from local logs.

The script is read-only with respect to training artifacts.  It reconstructs:

* E0: the original 0--90k run from the complete terminal log.
* E1: 0--15k from the S1 run, 15--20k from the first S2 process, and
  20--90k from the recovered S2 process.

Only ``action_dit_loss`` is compared because E1 ``total_loss`` additionally
contains the auxiliary Intent CE term and is therefore not directly comparable
with the E0 scalar loss.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


E0_LOG = Path(
    "/home/liuchang/kehang/488project/logs/e0_abc_rel/"
    "e0_abc_rel_scaled_bridge_rt1_2gpu_b8_60k_interactive_460_20260702_162921.log"
)
E1_S1_LOG = Path(
    "/home/liuchang/kehang/488project/code/starvla/playground/Pretrained_models/"
    "kehang-StarVLA/checkpoints/calvin/"
    "e1_spatial_intent_query_ffn_v2_s1_15k_s2_75k_90k_bs4_ga2_restart/"
    "wandb/wandb/run-20260723_111224-qzgyetnj/files/output.log"
)
E1_S1_WANDB = E1_S1_LOG.parents[1] / "run-qzgyetnj.wandb"
E1_S2_15K_TO_20K_LOG = Path(
    "/home/liuchang/kehang/488project/code/starvla/playground/Pretrained_models/"
    "kehang-StarVLA/checkpoints/calvin/"
    "e1_spatial_intent_query_ffn_v2_s2_from15k_lowlr_loss001/"
    "wandb/wandb/run-20260723_224246-weihdyzg/files/output.log"
)
E1_S2_15K_TO_20K_WANDB = (
    E1_S2_15K_TO_20K_LOG.parents[1] / "run-weihdyzg.wandb"
)
E1_S2_20K_TO_90K_LOG = Path(
    "/home/liuchang/kehang/488project/code/starvla/playground/Pretrained_models/"
    "kehang-StarVLA/checkpoints/calvin/"
    "e1_spatial_intent_query_ffn_v2_s2_from15k_lowlr_loss001/"
    "wandb/wandb/run-20260724_021443-weihdyzg/files/output.log"
)
E1_S2_20K_TO_90K_WANDB = (
    E1_S2_20K_TO_90K_LOG.parents[1] / "run-weihdyzg.wandb"
)
DEFAULT_OUT_DIR = Path(
    "/home/liuchang/kehang/488project/plan/presentation/assets/"
    "e0_e1_action_loss_90k"
)

CSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
OSC_RE = re.compile(r"\x1b\].*?(?:\x07|\x1b\\)", re.S)
SOURCE_LOC_RE = re.compile(r"\s+[A-Za-z_][\w.-]*\.py:\d+")
STEP_RE = re.compile(r"(?:Phase\s+step|Step)\s+(\d+)\s*,", re.I)
LOSS_RE = re.compile(r"'action_dit_loss'\s*:\s*([-+0-9.eE]+)", re.S)


@dataclass(frozen=True)
class Point:
    step: int
    loss: float
    source: str


def clean_terminal_text(text: str) -> str:
    text = OSC_RE.sub("", text)
    text = CSI_RE.sub("", text)
    # Rich may insert a source-location column between a metric key and value,
    # e.g. ``'action_dit_loss': train_starvla.py:305 0.69``.
    text = SOURCE_LOC_RE.sub("", text)
    return text.replace("\r", "\n")


def parse_action_loss(path: Path, source: str) -> list[Point]:
    text = clean_terminal_text(path.read_text(errors="replace"))
    matches = list(STEP_RE.finditer(text))
    points: dict[int, Point] = {}
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        chunk = text[start:end]
        # Rich may wrap inside both the metric key and numeric literal when the
        # terminal width changes.  Whitespace removal reconstructs, for example,
        # ``'action_dit_loss\\n': 0.074757814407348\\n63`` losslessly.
        compact_chunk = re.sub(r"\s+", "", chunk)
        loss_match = LOSS_RE.search(compact_chunk)
        if not loss_match:
            continue
        step = int(match.group(1))
        loss = float(loss_match.group(1))
        if np.isfinite(loss):
            points[step] = Point(step=step, loss=loss, source=source)
    return [points[key] for key in sorted(points)]


def parse_wandb_action_loss(path: Path, source: str) -> list[Point]:
    """Read scalar history directly from a local W&B binary event store."""
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    store = DataStore()
    store.open_for_scan(str(path))
    points: dict[int, Point] = {}
    while True:
        data = store.scan_data()
        if data is None:
            break
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        if record.WhichOneof("record_type") != "history":
            continue
        values = {}
        for item in record.history.item:
            key = item.key or "/".join(item.nested_key)
            if not key:
                continue
            try:
                values[key] = json.loads(item.value_json)
            except (TypeError, json.JSONDecodeError):
                continue
        step = values.get("training/phase_step", values.get("_step"))
        loss = values.get("action_dit_loss")
        if step is None or loss is None:
            continue
        step = int(step)
        loss = float(loss)
        if np.isfinite(loss):
            points[step] = Point(step=step, loss=loss, source=source)
    store.close()
    return [points[key] for key in sorted(points)]


def select(points: list[Point], low_exclusive: int, high_inclusive: int) -> list[Point]:
    return [point for point in points if low_exclusive < point.step <= high_inclusive]


def merge_points(*point_sets: list[Point]) -> list[Point]:
    merged: dict[int, Point] = {}
    for point_set in point_sets:
        for point in point_set:
            merged[point.step] = point
    return [merged[key] for key in sorted(merged)]


def merge_e1() -> list[Point]:
    s1 = select(
        merge_points(
            parse_action_loss(E1_S1_LOG, "E1 S1 terminal-log fallback"),
            parse_wandb_action_loss(E1_S1_WANDB, "E1 S1 local W&B history"),
        ),
        -1,
        15_000,
    )
    s2a = select(
        merge_points(
            parse_action_loss(
                E1_S2_15K_TO_20K_LOG,
                "E1 S2 pre-interruption terminal-log fallback",
            ),
            parse_wandb_action_loss(
                E1_S2_15K_TO_20K_WANDB,
                "E1 S2 pre-interruption local W&B history",
            ),
        ),
        15_000,
        20_000,
    )
    s2b = select(
        merge_points(
            parse_action_loss(
                E1_S2_20K_TO_90K_LOG,
                "E1 S2 recovered terminal-log fallback",
            ),
            parse_wandb_action_loss(
                E1_S2_20K_TO_90K_WANDB,
                "E1 S2 recovered local W&B history",
            ),
        ),
        20_000,
        90_000,
    )
    merged: dict[int, Point] = {}
    for point in s1 + s2a + s2b:
        merged[point.step] = point
    return [merged[key] for key in sorted(merged)]


def moving_average(values: np.ndarray, window_points: int) -> np.ndarray:
    """Centered moving average with edge padding and no curve fabrication."""
    if len(values) == 0:
        return values
    window_points = max(1, min(int(window_points), len(values)))
    left = window_points // 2
    right = window_points - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    kernel = np.ones(window_points, dtype=np.float64) / window_points
    return np.convolve(padded, kernel, mode="valid")


def infer_log_interval(points: list[Point]) -> int:
    diffs = np.diff([point.step for point in points])
    positive = diffs[diffs > 0]
    return int(np.median(positive)) if len(positive) else 20


def write_csv(path: Path, e0: list[Point], e1: list[Point], window_steps: int):
    e0_interval = infer_log_interval(e0)
    e1_interval = infer_log_interval(e1)
    e0_smooth = moving_average(
        np.asarray([point.loss for point in e0]), round(window_steps / e0_interval)
    )
    e1_smooth = moving_average(
        np.asarray([point.loss for point in e1]), round(window_steps / e1_interval)
    )
    rows = []
    rows.extend(
        {
            "run": "E0",
            "step": point.step,
            "action_dit_loss_raw": point.loss,
            "action_dit_loss_smooth": float(smoothed),
            "source": point.source,
        }
        for point, smoothed in zip(e0, e0_smooth)
    )
    rows.extend(
        {
            "run": "E1",
            "step": point.step,
            "action_dit_loss_raw": point.loss,
            "action_dit_loss_smooth": float(smoothed),
            "source": point.source,
        }
        for point, smoothed in zip(e1, e1_smooth)
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return e0_smooth, e1_smooth


def plot(
    png_path: Path,
    svg_path: Path,
    e0: list[Point],
    e1: list[Point],
    e0_smooth: np.ndarray,
    e1_smooth: np.ndarray,
    window_steps: int,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    background = "#10243A"
    foreground = "#D7E5F2"
    muted_text = "#AFC1D2"
    teal = "#22C7C9"
    baseline_blue = "#AFC3D8"
    grid = "#35506B"
    stage_orange = "#F5A623"

    fig, ax = plt.subplots(figsize=(12.8, 6.4), dpi=180)
    fig.patch.set_facecolor(background)
    ax.set_facecolor(background)

    e0_steps = np.asarray([point.step for point in e0])
    e1_steps = np.asarray([point.step for point in e1])
    ax.plot(
        e0_steps,
        e0_smooth,
        color=baseline_blue,
        linewidth=3.2,
        label="E0 baseline · 90k",
        solid_capstyle="round",
    )
    ax.plot(
        e1_steps,
        e1_smooth,
        color=teal,
        linewidth=3.5,
        label="E1 spatial intent · merged 90k",
        solid_capstyle="round",
    )

    ax.axvline(15_000, color=stage_orange, linewidth=1.8, linestyle=(0, (4, 4)))
    ax.text(
        15_000,
        0.985,
        "S1 → S2",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        color=stage_orange,
        fontsize=12,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": background,
            "edgecolor": "none",
        },
    )

    ax.set_xlim(0, 90_000)
    all_smooth = np.concatenate([e0_smooth, e1_smooth])
    # Keep the first-stage transient visible without letting isolated raw spikes
    # determine the scale.
    uncapped_ymax = float(np.quantile(all_smooth, 0.997) * 1.08)
    ymax = min(0.24, uncapped_ymax)
    ymin = max(0.0, float(np.quantile(all_smooth, 0.005) * 0.88))
    if ymax - ymin < 0.04:
        ymax = ymin + 0.04
    ax.set_ylim(ymin, ymax)
    if uncapped_ymax > ymax:
        ax.text(
            0.012,
            0.955,
            "initial warm-up transient exceeds the displayed y-range",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color=muted_text,
        )

    ax.set_title(
        "Action DiT Training Loss: E0 vs. Spatial-Intent E1",
        fontsize=23,
        fontweight="bold",
        color=foreground,
        pad=18,
    )
    ax.text(
        0.5,
        1.005,
        f"Centered {window_steps // 1000}k-step moving average · local logs · raw per-step noise omitted",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11.5,
        color=muted_text,
    )
    ax.set_xlabel("training step", fontsize=14, color=foreground, labelpad=10)
    ax.set_ylabel("action_dit_loss", fontsize=14, color=foreground, labelpad=10)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value / 1000)}k"))
    ax.grid(True, axis="both", color=grid, linewidth=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors=muted_text, labelsize=12)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(muted_text)
    ax.spines["bottom"].set_color(muted_text)
    legend = ax.legend(loc="upper right", frameon=False, fontsize=12.5)
    for text in legend.get_texts():
        text.set_color(foreground)

    fig.tight_layout(pad=1.6)
    fig.savefig(png_path, dpi=180, facecolor=background)
    fig.savefig(svg_path, facecolor=background)
    plt.close(fig)


def validate_sequence(name: str, points: list[Point], expected_end: int):
    if not points:
        raise RuntimeError(f"{name}: no points parsed")
    if points[-1].step != expected_end:
        raise RuntimeError(
            f"{name}: expected final logged step {expected_end}, got {points[-1].step}"
        )
    if any(right.step <= left.step for left, right in zip(points, points[1:])):
        raise RuntimeError(f"{name}: steps are not strictly increasing")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--window-steps",
        type=int,
        default=3_000,
        help="Centered moving-average width in training steps",
    )
    args = parser.parse_args()

    for source in (
        E0_LOG,
        E1_S1_LOG,
        E1_S1_WANDB,
        E1_S2_15K_TO_20K_LOG,
        E1_S2_15K_TO_20K_WANDB,
        E1_S2_20K_TO_90K_LOG,
        E1_S2_20K_TO_90K_WANDB,
    ):
        if not source.exists():
            raise FileNotFoundError(source)

    e0 = select(parse_action_loss(E0_LOG, "E0 original terminal log"), -1, 90_000)
    e1 = merge_e1()
    validate_sequence("E0", e0, 90_000)
    validate_sequence("E1", e1, 90_000)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "e0_e1_action_dit_loss_90k.csv"
    png_path = args.out_dir / "e0_e1_action_dit_loss_90k_smoothed_ppt_dark.png"
    svg_path = args.out_dir / "e0_e1_action_dit_loss_90k_smoothed_ppt_dark.svg"
    metadata_path = args.out_dir / "e0_e1_action_dit_loss_90k_metadata.json"

    e0_smooth, e1_smooth = write_csv(
        csv_path, e0, e1, window_steps=args.window_steps
    )
    plot(
        png_path,
        svg_path,
        e0,
        e1,
        e0_smooth,
        e1_smooth,
        window_steps=args.window_steps,
    )

    metadata = {
        "metric": "action_dit_loss",
        "smoothing": {
            "method": "centered moving average with edge padding",
            "window_steps": args.window_steps,
        },
        "E0": {
            "points": len(e0),
            "first_step": e0[0].step,
            "last_step": e0[-1].step,
            "source": str(E0_LOG),
            "raw_first": e0[0].loss,
            "raw_last": e0[-1].loss,
            "smoothed_last": float(e0_smooth[-1]),
        },
        "E1": {
            "points": len(e1),
            "first_step": e1[0].step,
            "last_step": e1[-1].step,
            "sources": [
                str(E1_S1_WANDB),
                str(E1_S2_15K_TO_20K_WANDB),
                str(E1_S2_20K_TO_90K_WANDB),
            ],
            "terminal_log_fallbacks": [
                str(E1_S1_LOG),
                str(E1_S2_15K_TO_20K_LOG),
                str(E1_S2_20K_TO_90K_LOG),
            ],
            "splice": {
                "S1": "0 < step <= 15000",
                "S2_before_interruption": "15000 < step <= 20000",
                "S2_recovered": "20000 < step <= 90000",
            },
            "raw_first": e1[0].loss,
            "raw_last": e1[-1].loss,
            "smoothed_last": float(e1_smooth[-1]),
        },
        "comparability_note": (
            "The chart compares action_dit_loss only. E1 total_loss includes "
            "auxiliary Intent CE and is not directly comparable with E0."
        ),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"PNG: {png_path}")
    print(f"SVG: {svg_path}")
    print(f"CSV: {csv_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
