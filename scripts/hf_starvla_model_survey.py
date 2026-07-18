#!/usr/bin/env python3
"""Generate a small-file-only Hugging Face StarVLA model survey HTML.

This script intentionally avoids downloading model weights/videos. It reads
Hugging Face API metadata plus small text/config artifacts when available.
"""

from __future__ import annotations

import datetime as dt
import html
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import HfApi, hf_hub_download


QUERY_URL = "https://huggingface.co/models?sort=created&search=starvla"
OUTPUT = Path("/home/liuchang/kehang/488project/plan/hf_starvla_model_survey.html")
MAX_SMALL_FILE_BYTES = 2 * 1024 * 1024

SKIP_SUFFIXES = (
    ".pt",
    ".pth",
    ".safetensors",
    ".bin",
    ".ckpt",
    ".mp4",
    ".avi",
    ".mov",
    ".log",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".zip",
    ".tar",
    ".gz",
)

SMALL_ROOT_PATTERNS = (
    re.compile(r"^README\.md$", re.I),
    re.compile(r"^config.*\.ya?ml$", re.I),
    re.compile(r"^run.*\.sh$", re.I),
    re.compile(r"^dataset_statistics\.json$", re.I),
    re.compile(r"^modality\.json$", re.I),
)


def e(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def truncate(value: str, n: int = 6000) -> str:
    value = value or ""
    return value if len(value) <= n else value[:n] + "\n…[truncated]"


def card_to_dict(card: Any) -> dict[str, Any]:
    if card is None:
        return {}
    if isinstance(card, dict):
        return card
    for attr in ("to_dict", "data"):
        obj = getattr(card, attr, None)
        if callable(obj):
            try:
                result = obj()
                if isinstance(result, dict):
                    return result
            except Exception:
                pass
        elif isinstance(obj, dict):
            return obj
    try:
        parsed = yaml.safe_load(str(card))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def sibling_size(sibling: Any) -> int | None:
    for attr in ("size", "lfs_size", "blob_size"):
        value = getattr(sibling, attr, None)
        if isinstance(value, int):
            return value
    return None


def should_fetch(rfilename: str, size: int | None) -> bool:
    base = rfilename.split("/")[-1]
    if rfilename.lower().endswith(SKIP_SUFFIXES):
        return False
    if "/" in rfilename and not (
        rfilename.startswith("docs/") or rfilename.startswith("examples/")
    ):
        return False
    if not any(p.match(base) for p in SMALL_ROOT_PATTERNS):
        return False
    if size is not None and size > MAX_SMALL_FILE_BYTES:
        return False
    return True


def safe_download_text(model_id: str, filename: str) -> tuple[str | None, str | None]:
    try:
        path = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            repo_type="model",
            etag_timeout=20,
            resume_download=True,
        )
        p = Path(path)
        if p.stat().st_size > MAX_SMALL_FILE_BYTES:
            return None, f"skipped after download: {p.stat().st_size} bytes"
        return p.read_text(encoding="utf-8", errors="replace"), None
    except Exception as exc:  # keep report best-effort
        return None, f"{type(exc).__name__}: {exc}"


def parse_yaml_documents(files: dict[str, str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for name, text in files.items():
        if not name.lower().endswith((".yaml", ".yml")):
            continue
        try:
            obj = yaml.safe_load(text)
            if isinstance(obj, dict):
                parsed[name] = obj
        except Exception:
            continue
    return parsed


def deep_get(obj: Any, path: list[str], default: Any = None) -> Any:
    cur = obj
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def first_framework_config(configs: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    if "config.yaml" in configs and isinstance(configs["config.yaml"], dict):
        return "config.yaml", configs["config.yaml"]
    for name, cfg in configs.items():
        if isinstance(cfg, dict) and isinstance(cfg.get("framework"), dict):
            return name, cfg
    return None, {}


def normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value]
    return [str(value)]


def find_vlm(text: str, config: dict[str, Any], card: dict[str, Any]) -> tuple[str, str]:
    base_vlm = deep_get(config, ["framework", "qwenvl", "base_vlm"])
    if base_vlm:
        return str(base_vlm).split("/")[-1], "config"
    base_models = normalize_list(card.get("base_model"))
    if base_models:
        return ", ".join(base_models), "model_card"

    patterns = [
        (r"Qwen2\.5[-_]?VL[-_]?3B", "Qwen2.5-VL-3B"),
        (r"Qwen3[-_]?VL[-_]?8B", "Qwen3-VL-8B"),
        (r"Qwen3[-_]?VL[-_]?4B|Qwen3VL[-_]?4B", "Qwen3-VL-4B"),
        (r"Qwen3[-_]?VL[-_]?2B|Qwen3VL2B|Qwen3[-_]?VL[-_]?2B", "Qwen3-VL-2B"),
        (r"Qwen3\.5[-_]?9B", "Qwen3.5-9B"),
        (r"Qwen3\.5[-_]?4B", "Qwen3.5-4B"),
        (r"Qwen3\.5[-_]?2B", "Qwen3.5-2B"),
        (r"qwen35[-_]?08b|Qwen3\.5[-_]?0\.?8B", "Qwen3.5-0.8B"),
        (r"InternVL3\.5[-_]?1B", "InternVL3.5-1B"),
        (r"MiniCPM[-_]?V[-_]?4\.6", "MiniCPM-V-4.6"),
        (r"Gemma4|Gemma[-_]?4", "Gemma4"),
    ]
    for pat, label in patterns:
        if re.search(pat, text, flags=re.I):
            return label, "filename_or_tags_inferred"
    return "unknown", "unknown"


def find_size(vlm: str, text: str) -> tuple[str, str]:
    combined = f"{vlm} {text}"
    if re.search(r"qwen35[-_]?08b", combined, flags=re.I):
        return "0.8B", "filename_inferred"
    m = re.search(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*B(?![A-Za-z])", combined, flags=re.I)
    if m:
        return f"{m.group(1)}B", "filename_or_model_inferred"
    if "MiniCPM-V-4.6" in combined:
        return "1.3B", "known_model_family"
    return "unknown", "unknown"


def find_action_head(text: str, config: dict[str, Any]) -> tuple[str, str]:
    name = deep_get(config, ["framework", "name"])
    if name:
        return str(name), "config"
    checks = [
        (r"QwenPI[_-]?v3|PI[_-]?v3|qwenpi_v3", "QwenPI_v3"),
        (r"QwenPI|qwenpi", "QwenPI"),
        (r"QwenGR00T|GR00T[_-]?v2|qwengr00t|gr00t", "QwenGR00T"),
        (r"OFT", "QwenOFT"),
        (r"FAST", "QwenFAST"),
    ]
    for pat, label in checks:
        if re.search(pat, text, flags=re.I):
            return label, "filename_or_tags_inferred"
    return "unknown", "unknown"


def action_head_config(config: dict[str, Any], readme: str) -> tuple[str, str]:
    action = deep_get(config, ["framework", "action_model"], {})
    if isinstance(action, dict):
        dcfg = action.get("diffusion_model_cfg") or {}
        parts = []
        for key in (
            "action_dit_hidden_dim",
            "input_embedding_dim",
            "cross_attention_dim",
            "output_dim",
            "num_layers",
            "num_attention_heads",
            "attention_head_dim",
        ):
            if isinstance(dcfg, dict) and key in dcfg:
                parts.append(f"{key}={dcfg[key]}")
        for key in ("action_model_type", "action_horizon", "action_dim", "state_dim"):
            if key in action:
                parts.append(f"{key}={action[key]}")
        if parts:
            return "; ".join(parts), "config"
    m = re.search(r"Action head\s*[:：]\s*([^\n]+)", readme, flags=re.I)
    if m:
        return m.group(1).strip(), "model_card"
    return "unknown", "unknown"


def find_datasets(text: str, config: dict[str, Any], card: dict[str, Any]) -> tuple[str, str]:
    datasets = normalize_list(card.get("datasets"))
    if datasets:
        return ", ".join(datasets), "model_card"
    data_mix = deep_get(config, ["datasets", "vla_data", "data_mix"])
    data_root = deep_get(config, ["datasets", "vla_data", "data_root_dir"])
    if data_mix or data_root:
        return ", ".join(str(x) for x in (data_mix, data_root) if x), "config"
    checks = [
        (r"Bridge[-_]?RT[_-]?1|Bridge.*RT-?1|RT[_-]?1|OXE", "Bridge + RT-1 / OXE"),
        (r"Calvin[_-]?D[_-]?D|CALVIN[_-]?D[_-]?D", "CALVIN D-D"),
        (r"Calvin|CALVIN", "CALVIN"),
        (r"LIBERO|libero", "LIBERO"),
        (r"PickOrange|pick-orange|leisaac-pick-orange", "LightwheelAI/leisaac-pick-orange"),
        (r"VLN[-_]?CE", "VLN-CE"),
        (r"RoboPro|robopro", "RoboPro"),
    ]
    found = []
    for pat, label in checks:
        if re.search(pat, text, flags=re.I):
            found.append(label)
    if found:
        return ", ".join(dict.fromkeys(found)), "filename_or_tags_inferred"
    return "unknown", "unknown"


def training_stage(text: str, config: dict[str, Any]) -> tuple[str, str]:
    if re.search(r"pretrain|stage-1", text, flags=re.I):
        return "pretrain checkpoint", "model_card_or_filename"
    if re.search(r"Calvin|LIBERO|PickOrange|Bridge|RT[_-]?1|VLN", text, flags=re.I):
        return "finetune / task checkpoint", "filename_or_tags_inferred"
    if deep_get(config, ["trainer", "pretrained_checkpoint"]):
        return "finetune from checkpoint", "config"
    return "unknown", "unknown"


def checkpoint_steps(siblings: list[Any]) -> str:
    steps = []
    for sib in siblings:
        name = getattr(sib, "rfilename", "")
        m = re.search(r"steps[_-](\d+).*?(?:pytorch_model|model)", name)
        if m:
            steps.append(int(m.group(1)))
    if not steps:
        return "unknown"
    uniq = sorted(set(steps))
    if len(uniq) > 8:
        return f"{uniq[0]}–{uniq[-1]} ({len(uniq)} checkpoints)"
    return ", ".join(str(x) for x in uniq)


def source_badge(source: str) -> str:
    cls = "unknown" if source == "unknown" else ("inferred" if "inferred" in source else "direct")
    return f'<span class="src {cls}">{e(source)}</span>'


def confidence(sources: list[str]) -> str:
    direct = sum(1 for s in sources if s in {"config", "model_card"})
    inferred = sum(1 for s in sources if "inferred" in s)
    unknown = sum(1 for s in sources if s == "unknown")
    if direct >= 3 and unknown == 0:
        return "high"
    if direct >= 1 or inferred >= 2:
        return "medium"
    return "low"


def summarize_readme(readme: str) -> str:
    lines = []
    for line in readme.splitlines():
        s = line.strip()
        if not s or s.startswith("---"):
            continue
        if any(k.lower() in s.lower() for k in ["base vlm", "action head", "dataset", "pretrain", "finetune", "calvin", "libero", "bridge", "rt-1", "recommended", "success"]):
            lines.append(s)
        if len(lines) >= 10:
            break
    return "\n".join(lines)


def compact_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "framework": cfg.get("framework"),
        "datasets": cfg.get("datasets"),
        "trainer": {
            k: v
            for k, v in (cfg.get("trainer") or {}).items()
            if k in {"max_train_steps", "freeze_modules", "pretrained_checkpoint", "learning_rate", "save_interval"}
        }
        if isinstance(cfg.get("trainer"), dict)
        else None,
        "run_id": cfg.get("run_id"),
    }


def fetch_model_record(api: HfApi, model: Any, index: int, total: int) -> dict[str, Any]:
    model_id = model.modelId
    print(f"[{index:02d}/{total}] {model_id}", flush=True)
    errors = []
    try:
        info = api.model_info(model_id, files_metadata=True)
    except Exception as exc:
        errors.append(f"model_info failed: {type(exc).__name__}: {exc}")
        info = model

    siblings = list(getattr(info, "siblings", []) or [])
    tags = list(getattr(info, "tags", []) or [])
    card = card_to_dict(getattr(info, "cardData", None))
    files: dict[str, str] = {}

    for sib in siblings:
        name = getattr(sib, "rfilename", "")
        size = sibling_size(sib)
        if should_fetch(name, size):
            text, err = safe_download_text(model_id, name)
            if text is not None:
                files[name] = text
            elif err:
                errors.append(f"{name}: {err}")
            time.sleep(0.05)

    configs = parse_yaml_documents(files)
    cfg_name, cfg = first_framework_config(configs)
    readme = files.get("README.md", "")
    card_text = yaml.safe_dump(card, allow_unicode=True, sort_keys=False) if card else ""
    all_text = "\n".join([model_id, " ".join(tags), readme, card_text, "\n".join(files.keys())])

    vlm, vlm_src = find_vlm(all_text, cfg, card)
    vlm_size, vlm_size_src = find_size(vlm, all_text)
    action, action_src = find_action_head(all_text, cfg)
    action_size, action_size_src = action_head_config(cfg, readme)
    ds, ds_src = find_datasets(all_text, cfg, card)
    stage, stage_src = training_stage(all_text, cfg)
    steps = checkpoint_steps(siblings)

    sources = [vlm_src, vlm_size_src, action_src, action_size_src, ds_src, stage_src]
    return {
        "model_id": model_id,
        "url": f"https://huggingface.co/{model_id}",
        "author": model_id.split("/")[0] if "/" in model_id else "",
        "created": str(getattr(info, "created_at", getattr(model, "created_at", "")) or ""),
        "updated": str(getattr(info, "last_modified", getattr(model, "last_modified", "")) or ""),
        "pipeline": getattr(info, "pipeline_tag", None) or "",
        "library": getattr(info, "library_name", None) or card.get("library_name", "") or "",
        "license": card.get("license", ""),
        "downloads": getattr(info, "downloads", None),
        "likes": getattr(info, "likes", None),
        "tags": tags,
        "card": card,
        "small_files": sorted(files.keys()),
        "config_file": cfg_name,
        "config_summary": compact_config(cfg) if cfg else {},
        "readme_key_lines": summarize_readme(readme),
        "vlm": vlm,
        "vlm_src": vlm_src,
        "vlm_size": vlm_size,
        "vlm_size_src": vlm_size_src,
        "action_head": action,
        "action_head_src": action_src,
        "action_head_size": action_size,
        "action_head_size_src": action_size_src,
        "dataset": ds,
        "dataset_src": ds_src,
        "stage": stage,
        "stage_src": stage_src,
        "checkpoint_steps": steps,
        "confidence": confidence(sources),
        "errors": errors,
    }


def render_html(records: list[dict[str, Any]], failures: list[str]) -> str:
    generated = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    total = len(records)
    vlm_known = sum(r["vlm"] != "unknown" for r in records)
    action_known = sum(r["action_head"] != "unknown" for r in records)
    unknown_cells = sum(
        1
        for r in records
        for k in ("vlm", "vlm_size", "action_head", "action_head_size", "dataset", "stage")
        if r[k] == "unknown"
    )
    dataset_counter = Counter(r["dataset"] for r in records if r["dataset"] != "unknown")
    action_counter = Counter(r["action_head"] for r in records if r["action_head"] != "unknown")

    def row(r: dict[str, Any]) -> str:
        detail = {
            "tags": r["tags"],
            "cardData": r["card"],
            "small_files_read": r["small_files"],
            "config_file": r["config_file"],
            "config_summary": r["config_summary"],
            "readme_key_lines": r["readme_key_lines"],
            "errors": r["errors"],
        }
        details_html = e(json.dumps(detail, ensure_ascii=False, indent=2))
        badges = " ".join(
            [
                source_badge(r["vlm_src"]),
                source_badge(r["action_head_src"]),
                source_badge(r["dataset_src"]),
            ]
        )
        conf_cls = e(r["confidence"])
        return f"""
        <tr class="conf-{conf_cls}">
          <td><a href="{e(r['url'])}" target="_blank" rel="noreferrer">{e(r['model_id'])}</a><br><span class="muted">{e(r['author'])}</span></td>
          <td>{e(r['vlm'])}<br>{source_badge(r['vlm_src'])}</td>
          <td>{e(r['vlm_size'])}<br>{source_badge(r['vlm_size_src'])}</td>
          <td>{e(r['action_head'])}<br>{source_badge(r['action_head_src'])}</td>
          <td class="small">{e(r['action_head_size'])}<br>{source_badge(r['action_head_size_src'])}</td>
          <td>{e(r['stage'])}<br>{source_badge(r['stage_src'])}</td>
          <td class="small">{e(r['dataset'])}<br>{source_badge(r['dataset_src'])}</td>
          <td>{e(r['checkpoint_steps'])}</td>
          <td><span class="conf">{e(r['confidence'])}</span><br>{badges}</td>
          <td class="small">{e(r['updated'])}</td>
          <td><details><summary>details</summary><pre>{details_html}</pre></details></td>
        </tr>
        """

    dataset_summary = "".join(f"<li>{e(k)}: {v}</li>" for k, v in dataset_counter.most_common(12))
    action_summary = "".join(f"<li>{e(k)}: {v}</li>" for k, v in action_counter.most_common())
    failure_html = "".join(f"<li>{e(x)}</li>" for x in failures)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Hugging Face StarVLA Model Survey</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 24px; color: #1f2937; }}
    h1 {{ margin-bottom: 0.2rem; }}
    .muted {{ color: #6b7280; font-size: 0.88em; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 18px 0; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px; background: #f9fafb; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 8px; vertical-align: top; }}
    th {{ position: sticky; top: 0; background: #111827; color: #fff; z-index: 2; }}
    tr:nth-child(even) {{ background: #fafafa; }}
    tr.conf-low {{ background: #fff7ed; }}
    tr.conf-medium {{ background: #f8fafc; }}
    a {{ color: #2563eb; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .src {{ display: inline-block; margin-top: 2px; padding: 1px 5px; border-radius: 999px; font-size: 11px; }}
    .src.direct {{ background: #dcfce7; color: #166534; }}
    .src.inferred {{ background: #fef3c7; color: #92400e; }}
    .src.unknown {{ background: #fee2e2; color: #991b1b; }}
    .conf {{ font-weight: 600; }}
    .small {{ max-width: 260px; overflow-wrap: anywhere; }}
    pre {{ white-space: pre-wrap; max-width: 720px; max-height: 480px; overflow: auto; background: #0b1020; color: #d1d5db; padding: 10px; border-radius: 8px; }}
    .note {{ background: #eff6ff; border-left: 4px solid #3b82f6; padding: 10px 12px; }}
  </style>
</head>
<body>
  <h1>Hugging Face StarVLA Model Survey</h1>
  <p class="muted">Generated: {e(generated)} · Query: <a href="{e(QUERY_URL)}">{e(QUERY_URL)}</a></p>
  <div class="note">This report reads Hugging Face API metadata and small text/config files only. It does not download model weights or videos. Inferred fields are explicitly marked and should be re-audited before training.</div>
  <section class="summary">
    <div class="card"><strong>Total models</strong><br>{total}</div>
    <div class="card"><strong>VLM parsed</strong><br>{vlm_known}/{total}</div>
    <div class="card"><strong>Action head parsed</strong><br>{action_known}/{total}</div>
    <div class="card"><strong>Unknown cells</strong><br>{unknown_cells}</div>
  </section>
  <section class="summary">
    <div class="card"><strong>Action heads</strong><ul>{action_summary}</ul></div>
    <div class="card"><strong>Top datasets</strong><ul>{dataset_summary}</ul></div>
    <div class="card"><strong>Failures</strong><ul>{failure_html or '<li>None</li>'}</ul></div>
  </section>
  <table>
    <thead>
      <tr>
        <th>Model</th><th>VLM</th><th>VLM Size</th><th>Action Head</th><th>Action Head Size / Config</th>
        <th>Pretrain / Fine-tune</th><th>Dataset</th><th>Checkpoint Steps</th><th>Confidence</th><th>Updated</th><th>Trace</th>
      </tr>
    </thead>
    <tbody>
      {''.join(row(r) for r in records)}
    </tbody>
  </table>
</body>
</html>
"""


def main() -> int:
    api = HfApi()
    failures: list[str] = []
    models = list(api.list_models(search="starvla", sort="created_at", direction=-1, limit=100, full=True))
    total = len(models)
    records: list[dict[str, Any]] = []
    for idx, model in enumerate(models, start=1):
        try:
            records.append(fetch_model_record(api, model, idx, total))
        except Exception as exc:
            msg = f"{model.modelId}: {type(exc).__name__}: {exc}"
            print("ERROR", msg, file=sys.stderr, flush=True)
            failures.append(msg)
        time.sleep(0.1)

    html_text = render_html(records, failures)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(html_text, encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(f"Models: {len(records)} / API returned {total}")
    print(f"Failures: {len(failures)}")
    return 0 if len(records) == total else 1


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    raise SystemExit(main())
