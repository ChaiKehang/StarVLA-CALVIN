#!/usr/bin/env python3
"""Parse StarVLA training logs and generate local visualization files.

This script is intentionally read-only with respect to the training process:
it only reads a `.log` file produced by the training wrapper and writes small
CSV/HTML/PNG artifacts for inspection.

Example:
  python /home/liuchang/kehang/488project/scripts/e0_abc_rel/plot_e0_training_log.py

  python plot_e0_training_log.py \
    --log /path/to/train.log \
    --out-dir /path/to/output_dir
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


DEFAULT_LOG = Path(
    "/home/liuchang/kehang/488project/logs/e0_abc_rel/"
    "e0_abc_rel_scaled_bridge_rt1_2gpu_b8_60k_interactive_460_20260702_162921.log"
)
DEFAULT_OUT_DIR = Path("/home/liuchang/kehang/488project/logs/e0_abc_rel/visualization")


ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
STEP_RE = re.compile(r"Step\s+(\d+),\s+Loss:")


@dataclass
class MetricRow:
    step: int
    action_dit_loss: float | None = None
    timing_data: float | None = None
    timing_model: float | None = None
    lr_action_model: float | None = None
    lr_base: float | None = None
    epoch: float | None = None


def clean_text(text: str) -> str:
    """Remove terminal control codes and normalize progress-bar carriage returns."""
    text = ANSI_RE.sub("", text)
    text = text.replace("\r", "\n")
    return text


def extract_float(pattern: str, chunk: str) -> float | None:
    match = re.search(pattern, chunk, flags=re.S)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def parse_metrics(log_path: Path) -> list[MetricRow]:
    text = clean_text(log_path.read_text(errors="replace"))
    matches = list(STEP_RE.finditer(text))
    rows: list[MetricRow] = []

    for i, match in enumerate(matches):
        step = int(match.group(1))
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else min(len(text), match.start() + 4000)
        chunk = text[start:end]

        row = MetricRow(
            step=step,
            action_dit_loss=extract_float(r"'action_dit_loss'\s*:\s*([-+0-9.eE]+)", chunk),
            timing_data=extract_float(r"'timing/data'\s*:\s*([-+0-9.eE]+)", chunk),
            timing_model=extract_float(r"'timing/model'\s*:\s*([-+0-9.eE]+)", chunk),
            lr_action_model=extract_float(r"'learning_rate/action_model\s*'\s*:\s*([-+0-9.eE]+)", chunk),
            lr_base=extract_float(r"'learning_rate/base'\s*:\s*([-+0-9.eE]+)", chunk),
            epoch=extract_float(r"'epoch'\s*:\s*([-+0-9.eE]+)", chunk),
        )
        rows.append(row)

    # Keep the last record for duplicated steps, if any.
    dedup: dict[int, MetricRow] = {}
    for row in rows:
        dedup[row.step] = row
    return [dedup[k] for k in sorted(dedup)]


def write_csv(rows: Iterable[MetricRow], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(MetricRow(step=0)).keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def finite_pairs(rows: list[MetricRow], attr: str) -> list[tuple[float, float]]:
    pairs = []
    for row in rows:
        value = getattr(row, attr)
        if value is not None and math.isfinite(value):
            pairs.append((float(row.step), float(value)))
    return pairs


def polyline(points: list[tuple[float, float]], width: int, height: int, pad: int) -> tuple[str, dict[str, float]]:
    if not points:
        return "", {}
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax == xmin:
        xmax = xmin + 1.0
    if ymax == ymin:
        ymax = ymin + 1.0

    coords = []
    for x, y in points:
        sx = pad + (x - xmin) / (xmax - xmin) * (width - 2 * pad)
        sy = height - pad - (y - ymin) / (ymax - ymin) * (height - 2 * pad)
        coords.append(f"{sx:.1f},{sy:.1f}")
    return " ".join(coords), {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}


def svg_chart(rows: list[MetricRow], attr: str, title: str, color: str) -> str:
    width, height, pad = 960, 360, 54
    points = finite_pairs(rows, attr)
    line, meta = polyline(points, width, height, pad)
    if not line:
        return f"<section><h2>{html.escape(title)}</h2><p>No data.</p></section>"

    last_x, last_y = points[-1]
    return f"""
<section class="card">
  <h2>{html.escape(title)}</h2>
  <svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
    <rect x="0" y="0" width="{width}" height="{height}" fill="white"/>
    <line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#999"/>
    <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#999"/>
    <polyline fill="none" stroke="{color}" stroke-width="2.5" points="{line}"/>
    <text x="{pad}" y="{height-16}" fill="#555">step {meta['xmin']:.0f}</text>
    <text x="{width-pad-120}" y="{height-16}" fill="#555">step {meta['xmax']:.0f}</text>
    <text x="10" y="{pad+4}" fill="#555">{meta['ymax']:.4g}</text>
    <text x="10" y="{height-pad+4}" fill="#555">{meta['ymin']:.4g}</text>
  </svg>
  <p class="small">Latest: step {last_x:.0f}, {html.escape(attr)} = {last_y:.6g}</p>
</section>
"""


def write_html(rows: list[MetricRow], html_path: Path, csv_path: Path, log_path: Path) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    latest = asdict(rows[-1]) if rows else {}
    content = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <title>E0 ABC→D Training Metrics</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; background: #f7f7fb; color: #1f2933; }}
    h1 {{ margin-bottom: 0.2rem; }}
    .meta, .small {{ color: #667085; font-size: 0.92rem; }}
    .card {{ background: white; border: 1px solid #e4e7ec; border-radius: 14px; padding: 18px; margin: 18px 0; box-shadow: 0 1px 2px rgba(16,24,40,.04); }}
    code {{ background: #eef2f6; padding: 2px 5px; border-radius: 5px; }}
    table {{ border-collapse: collapse; background: white; }}
    td, th {{ border: 1px solid #e4e7ec; padding: 6px 10px; text-align: right; }}
    th {{ background: #f2f4f7; }}
  </style>
</head>
<body>
  <h1>E0 ABC→D Training Metrics</h1>
  <p class="meta">Parsed from <code>{html.escape(str(log_path))}</code></p>
  <p class="meta">CSV: <code>{html.escape(str(csv_path))}</code></p>
  <section class="card">
    <h2>Latest parsed row</h2>
    <pre>{html.escape(json.dumps(latest, indent=2, ensure_ascii=False))}</pre>
  </section>
  {svg_chart(rows, "action_dit_loss", "action_dit_loss", "#2563eb")}
  {svg_chart(rows, "timing_model", "timing/model seconds per step", "#16a34a")}
  {svg_chart(rows, "lr_action_model", "learning_rate/action_model", "#dc2626")}
</body>
</html>
"""
    html_path.write_text(content, encoding="utf-8")


def try_write_png(rows: list[MetricRow], out_dir: Path) -> list[Path]:
    """Write PNG charts if matplotlib is available. HTML/CSV are always produced."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    written: list[Path] = []
    charts = [
        ("action_dit_loss", "Action DiT Loss", "e0_abc_action_dit_loss.png"),
        ("timing_model", "Model Time / Step", "e0_abc_step_time.png"),
        ("lr_action_model", "Action Model Learning Rate", "e0_abc_lr_action_model.png"),
    ]
    for attr, title, filename in charts:
        pairs = finite_pairs(rows, attr)
        if not pairs:
            continue
        xs, ys = zip(*pairs)
        plt.figure(figsize=(10, 4.8))
        plt.plot(xs, ys, linewidth=1.8)
        plt.title(title)
        plt.xlabel("step")
        plt.ylabel(attr)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        path = out_dir / filename
        plt.savefig(path, dpi=160)
        plt.close()
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG, help="Training log path")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
    args = parser.parse_args()

    if not args.log.exists():
        raise SystemExit(f"Log file not found: {args.log}")

    rows = parse_metrics(args.log)
    if not rows:
        raise SystemExit(f"No metric rows parsed from: {args.log}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "e0_abc_train_metrics.csv"
    html_path = args.out_dir / "e0_abc_train_metrics.html"

    write_csv(rows, csv_path)
    write_html(rows, html_path, csv_path, args.log)
    png_paths = try_write_png(rows, args.out_dir)

    print(f"Parsed rows: {len(rows)}")
    print(f"Latest step: {rows[-1].step}")
    print(f"CSV: {csv_path}")
    print(f"HTML: {html_path}")
    if png_paths:
        print("PNG:")
        for path in png_paths:
            print(f"  {path}")
    else:
        print("PNG: skipped because matplotlib is unavailable; use the HTML chart.")


if __name__ == "__main__":
    main()
