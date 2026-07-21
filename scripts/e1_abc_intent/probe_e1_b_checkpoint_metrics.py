#!/usr/bin/env python3
"""Read-only E1-B checkpoint probe for condition-scale and gradient ratios.

This script is intentionally independent from the training entry point.  It:

* never attaches to or mutates a running training process;
* never creates an optimizer or calls ``backward``/``optimizer.step``;
* loads a completed model checkpoint read-only;
* replays fixed dataset batches and uses ``torch.autograd.grad`` only on
  ``project_layers[-1]``;
* refuses to start on a GPU that already has a compute process.

The two reported core metrics are:

1. Per-sample condition scale

       ||c_intent||_2 / (||e_time||_2 + eps)

   measured immediately before the DiT adds the two 1024-D conditions.

2. Weighted auxiliary/action gradient ratio on the last projector

       ||grad(lambda * L_intent)||_2 / (||grad(L_action)||_2 + eps)

The second metric is meaningful for E1-B because Action loss can reach the
last projector through the predicted-intent conditioning branch.  If that
path is disabled or disconnected, the script reports an undefined ratio
instead of silently dividing by zero.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STARVLA_ROOT = PROJECT_ROOT / "third_party" / "starvla"
if str(STARVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(STARVLA_ROOT))

DEFAULT_CHECKPOINT = (
    STARVLA_ROOT
    / "playground"
    / "Pretrained_models"
    / "kehang-StarVLA"
    / "checkpoints"
    / "calvin"
    / "e1_b_abc_rel_scaled_intent125_h8"
    / "checkpoints"
    / "steps_42000_pytorch_model.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only E1-B checkpoint probe for condition/timestep and "
            "weighted-Intent/action gradient ratios."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Defaults to <checkpoint run dir>/config.full.yaml.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Explicit logical CUDA device, for example cuda:0. "
            "CUDA_VISIBLE_DEVICES must also be set."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--skip-batches", type=int, default=0)
    parser.add_argument(
        "--diffusion-repeats",
        type=int,
        default=1,
        help=(
            "Noise/timestep samples inside each action-loss forward. Training "
            "used 16; use 1 first for safety and increase only if memory allows."
        ),
    )
    parser.add_argument(
        "--noise-seeds-per-batch",
        type=int,
        default=2,
        help="Independent downstream dropout/noise samples for each fixed data batch.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-mode",
        choices=("train", "eval"),
        default="train",
        help="train matches dropout behavior; eval gives a deterministic representation probe.",
    )
    parser.add_argument(
        "--min-free-gib",
        type=float,
        default=24.0,
        help="Refuse to start unless the selected GPU has at least this much free memory.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional result path. By default results are printed only.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run lightweight math tests without loading a checkpoint, dataset, or GPU.",
    )
    return parser.parse_args()


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _infer_config_path(checkpoint: Path) -> Path:
    # <run>/checkpoints/steps_x.pt -> <run>/config.full.yaml
    return checkpoint.resolve().parent.parent / "config.full.yaml"


def _logical_cuda_index(device: str) -> int:
    parsed = torch.device(device)
    if parsed.type != "cuda" or parsed.index is None:
        raise ValueError("--device must be explicit, for example --device cuda:0")
    return int(parsed.index)


def _visible_gpu_target(logical_index: int) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES is unset. Refusing automatic GPU selection; "
            "expose exactly the GPU(s) assigned by the scheduler first."
        )
    entries = [entry.strip() for entry in visible.split(",") if entry.strip()]
    if logical_index >= len(entries):
        raise RuntimeError(
            f"Logical cuda:{logical_index} is not exposed by CUDA_VISIBLE_DEVICES={visible!r}"
        )
    return entries[logical_index]


def _query_compute_processes(gpu_target: str) -> list[str]:
    command = [
        "nvidia-smi",
        "-i",
        gpu_target,
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            "Could not verify whether the selected GPU is idle; refusing to run."
        ) from exc
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def assert_gpu_is_safe(device: str, min_free_gib: float) -> dict[str, object]:
    logical_index = _logical_cuda_index(device)
    gpu_target = _visible_gpu_target(logical_index)
    processes = _query_compute_processes(gpu_target)
    if processes:
        formatted = "\n  ".join(processes)
        raise RuntimeError(
            "Selected GPU already has compute process(es); refusing to interfere:\n"
            f"  {formatted}"
        )

    torch.cuda.set_device(logical_index)
    free_bytes, total_bytes = torch.cuda.mem_get_info(logical_index)
    free_gib = free_bytes / 2**30
    total_gib = total_bytes / 2**30
    if free_gib < min_free_gib:
        raise RuntimeError(
            f"Selected GPU has only {free_gib:.2f} GiB free, below "
            f"--min-free-gib={min_free_gib:.2f}; refusing to run."
        )
    return {
        "logical_device": f"cuda:{logical_index}",
        "visible_target": gpu_target,
        "name": torch.cuda.get_device_name(logical_index),
        "free_gib_before_load": free_gib,
        "total_gib": total_gib,
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _tensor_l2(grads: Sequence[torch.Tensor | None]) -> torch.Tensor:
    terms = [grad.float().square().sum() for grad in grads if grad is not None]
    if not terms:
        return torch.zeros((), dtype=torch.float32)
    total = terms[0]
    for term in terms[1:]:
        total = total + term
    return total.sqrt()


def _gradient_dot(
    left: Sequence[torch.Tensor | None], right: Sequence[torch.Tensor | None]
) -> torch.Tensor:
    terms = [
        a.float().mul(b.float()).sum()
        for a, b in zip(left, right)
        if a is not None and b is not None
    ]
    if not terms:
        return torch.zeros((), dtype=torch.float32)
    total = terms[0]
    for term in terms[1:]:
        total = total + term
    return total


def _describe(values: Iterable[float]) -> dict[str, float | int | None]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "p90": None,
            "max": None,
        }
    ordered = sorted(clean)
    tensor = torch.tensor(ordered, dtype=torch.float64)
    return {
        "count": len(clean),
        "mean": statistics.fmean(clean),
        "median": statistics.median(clean),
        "std": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "min": ordered[0],
        "p90": torch.quantile(tensor, 0.9).item(),
        "max": ordered[-1],
    }


def _condition_assessment(value: float | None) -> str:
    if value is None:
        return "missing"
    if value < 0.05:
        return "very_weak_below_5pct"
    if value < 0.30:
        return "moderate_5_to_30pct"
    if value < 0.50:
        return "caution_30_to_50pct"
    if value < 1.00:
        return "high_50_to_100pct"
    return "dominant_at_least_100pct"


def _gradient_assessment(value: float | None) -> str:
    if value is None:
        return "undefined_action_gradient_zero_or_disconnected"
    if value < 0.10:
        return "below_plan_target"
    if value <= 0.30:
        return "inside_plan_target_0p1_to_0p3"
    if value <= 0.50:
        return "above_target_caution"
    return "high_auxiliary_gradient"


def _safe_ratio(numerator: float, denominator: float, eps: float = 1e-12) -> float | None:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return None
    if denominator <= eps:
        return None
    return numerator / denominator


def run_self_test() -> None:
    left = [torch.tensor([3.0, 4.0]), torch.tensor([0.0])]
    right = [torch.tensor([0.0, 5.0]), torch.tensor([12.0])]
    assert math.isclose(_tensor_l2(left).item(), 5.0)
    assert math.isclose(_tensor_l2(right).item(), 13.0)
    assert math.isclose(_gradient_dot(left, right).item(), 20.0)
    assert _condition_assessment(0.20) == "moderate_5_to_30pct"
    assert _gradient_assessment(0.20) == "inside_plan_target_0p1_to_0p3"
    assert _safe_ratio(1.0, 0.0) is None
    stats = _describe([1.0, 2.0, 3.0])
    assert stats["mean"] == 2.0 and stats["median"] == 2.0
    print("Self-test passed.")


def _load_config(config_path: Path, batch_size: int, diffusion_repeats: int):
    from starVLA.model.framework.share_tools import apply_config_compat

    cfg = apply_config_compat(OmegaConf.load(config_path))
    cfg.datasets.vla_data.per_device_batch_size = batch_size
    cfg.datasets.vla_data.num_workers = 0
    cfg.datasets.vla_data.pin_memory = False
    cfg.datasets.vla_data.persistent_workers = False
    cfg.trainer.repeated_diffusion_steps = diffusion_repeats
    return cfg


def _build_dataset_loader(cfg, batch_size: int) -> DataLoader:
    # Import the dataset factory directly.  build_dataloader() is deliberately
    # avoided because its training path writes dataset statistics to output_dir.
    from starVLA.dataloader.lerobot_datasets import collate_fn, get_vla_dataset

    dataset = get_vla_dataset(
        data_cfg=cfg.datasets.vla_data,
        seed=int(cfg.get("seed", 42)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
    )


def _build_and_load_model(cfg, checkpoint: Path, device: torch.device):
    from starVLA.model.framework.base_framework import build_framework

    model = build_framework(cfg)
    model.to(dtype=torch.bfloat16)

    print(f"Loading checkpoint read-only: {checkpoint}", flush=True)
    state_dict = torch.load(
        checkpoint,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    try:
        incompatible = model.load_state_dict(state_dict, strict=True, assign=True)
    except TypeError:
        incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    del state_dict
    gc.collect()

    model.to(device=device, dtype=torch.bfloat16)
    torch.cuda.empty_cache()
    return model


def _freeze_except_last_projector(model) -> tuple[torch.nn.Parameter, ...]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    target_module = model.project_layers[-1]
    for parameter in target_module.parameters():
        parameter.requires_grad_(True)
    target_params = tuple(target_module.parameters())
    if not target_params:
        raise RuntimeError("project_layers[-1] has no parameters")
    return target_params


@torch.no_grad()
def _encode_and_cache_static_features(model, examples: list[dict]):
    batch_images = [example["image"] for example in examples]
    instructions = [example["lang"] for example in examples]
    state = [example["state"] for example in examples] if "state" in examples[0] else None
    if state is not None:
        instructions = model.add_discretized_state_to_instruction(instructions, state)

    qwen_inputs = model.qwen_vl_interface.build_qwenvl_inputs(
        images=batch_images,
        instructions=instructions,
    )
    attention_mask = qwen_inputs.get("attention_mask", None)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model.qwen_vl_interface(
            **qwen_inputs,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        raw_layers = list(outputs.hidden_states[-model.num_action_dit_layers :])
        static_projected = [
            projector(hidden).detach()
            for projector, hidden in zip(model.project_layers[:-1], raw_layers[:-1])
        ]
        last_raw_hidden = raw_layers[-1].detach()

    del outputs, raw_layers, qwen_inputs
    actions = torch.as_tensor(
        np.array([example["action"] for example in examples]),
        device=last_raw_hidden.device,
        dtype=last_raw_hidden.dtype,
    )[:, -model.action_horizon :, :]
    return static_projected, last_raw_hidden, attention_mask, actions


def _probe_one_seed(
    *,
    model,
    examples: list[dict],
    static_projected: list[torch.Tensor],
    last_raw_hidden: torch.Tensor,
    attention_mask: torch.Tensor | None,
    actions_target: torch.Tensor,
    target_params: tuple[torch.nn.Parameter, ...],
    repeat_count: int,
    seed: int,
) -> tuple[dict[str, object], list[float]]:
    from starVLA.model.modules.intent_head import compute_intent_auxiliary_loss

    _set_seed(seed)
    captured_timestep_embeddings: list[torch.Tensor] = []

    def capture_timestep_embedding(_module, _inputs, output):
        captured_timestep_embeddings.append(output)

    hook = model.action_model.model.timestep_encoder.register_forward_hook(
        capture_timestep_embedding
    )
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            last_projected = model.project_layers[-1](last_raw_hidden)
            vl_embs_list = [*static_projected, last_projected]
            intent_logits, probabilities = model._predict_intent(
                last_projected, attention_mask
            )
            intent_condition = model.intent_to_timestep(probabilities)

            repeated_actions = actions_target.repeat(repeat_count, 1, 1)
            repeated_vl = [hidden.repeat(repeat_count, 1, 1) for hidden in vl_embs_list]
            repeated_mask = (
                attention_mask.repeat(repeat_count, 1).to(dtype=torch.bool)
                if attention_mask is not None
                else None
            )
            repeated_condition = intent_condition.repeat(repeat_count, 1)
            repeated_ffn_probabilities = (
                probabilities.repeat(repeat_count, 1)
                if model.use_ffn_intent_film
                else None
            )

            action_loss, condition_diagnostics = model.action_model(
                repeated_vl,
                repeated_actions,
                None,
                encoder_attention_mask=repeated_mask,
                intent_condition=repeated_condition,
                ffn_intent_probabilities=repeated_ffn_probabilities,
                return_condition_diagnostics=True,
            )
            targets = model._intent_targets(examples, last_projected.device)
            intent_loss_output = compute_intent_auxiliary_loss(
                intent_logits,
                targets,
                model.intent_loss_weight,
            )
            weighted_intent_loss = intent_loss_output.weighted_loss

        action_grads = torch.autograd.grad(
            action_loss,
            target_params,
            retain_graph=True,
            allow_unused=True,
        )
        intent_grads = torch.autograd.grad(
            weighted_intent_loss,
            target_params,
            retain_graph=False,
            allow_unused=True,
        )
    finally:
        hook.remove()

    if len(captured_timestep_embeddings) != 1:
        raise RuntimeError(
            "Expected exactly one DiT timestep-encoder call, got "
            f"{len(captured_timestep_embeddings)}"
        )
    timestep_embedding = captured_timestep_embeddings[0].detach().float()
    repeated_condition_fp32 = repeated_condition.detach().float()
    timestep_l2 = timestep_embedding.norm(dim=-1)
    intent_l2 = repeated_condition_fp32.norm(dim=-1)
    if timestep_l2.shape != intent_l2.shape:
        raise RuntimeError(
            f"Condition/timestep shape mismatch: {intent_l2.shape} vs {timestep_l2.shape}"
        )
    sample_ratios = (
        intent_l2 / timestep_l2.clamp_min(torch.finfo(torch.float32).eps)
    )

    action_grad_norm = _tensor_l2(action_grads).item()
    weighted_intent_grad_norm = _tensor_l2(intent_grads).item()
    gradient_ratio = _safe_ratio(weighted_intent_grad_norm, action_grad_norm)
    grad_cosine = None
    if action_grad_norm > 1e-12 and weighted_intent_grad_norm > 1e-12:
        grad_cosine = (
            _gradient_dot(action_grads, intent_grads)
            / (action_grad_norm * weighted_intent_grad_norm)
        ).item()

    diagnostic_ratio = float(
        condition_diagnostics["intent_condition_to_timestep_l2_ratio"].item()
    )
    computed_ratio = float(sample_ratios.mean().item())
    if not math.isclose(diagnostic_ratio, computed_ratio, rel_tol=5e-3, abs_tol=5e-5):
        raise RuntimeError(
            "Hook-computed condition ratio disagrees with DiT diagnostic: "
            f"{computed_ratio} vs {diagnostic_ratio}"
        )

    result = {
        "seed": seed,
        "num_original_samples": len(examples),
        "num_diffusion_samples": int(sample_ratios.numel()),
        "action_loss": float(action_loss.detach().float().item()),
        "intent_ce": float(intent_loss_output.loss.detach().float().item()),
        "weighted_intent_loss": float(weighted_intent_loss.detach().float().item()),
        "intent_top1_accuracy": float(intent_loss_output.top1_accuracy.item()),
        "condition_ratio": _describe(sample_ratios.cpu().tolist()),
        "intent_condition_l2_mean": float(intent_l2.mean().item()),
        "timestep_embedding_l2_mean": float(timestep_l2.mean().item()),
        "joint_condition_l2_mean": float(
            condition_diagnostics["joint_timestep_condition_l2_mean"].item()
        ),
        "action_grad_norm_on_last_projector": action_grad_norm,
        "weighted_intent_grad_norm_on_last_projector": weighted_intent_grad_norm,
        "weighted_intent_to_action_grad_ratio": gradient_ratio,
        "action_intent_gradient_cosine": grad_cosine,
        "gradient_ratio_status": _gradient_assessment(gradient_ratio),
    }
    return result, sample_ratios.cpu().tolist()


def _aggregate(records: list[dict[str, object]], all_condition_ratios: list[float]):
    action_losses = [float(record["action_loss"]) for record in records]
    intent_losses = [float(record["intent_ce"]) for record in records]
    weighted_intent_losses = [
        float(record["weighted_intent_loss"]) for record in records
    ]
    action_grad_norms = [
        float(record["action_grad_norm_on_last_projector"]) for record in records
    ]
    weighted_intent_grad_norms = [
        float(record["weighted_intent_grad_norm_on_last_projector"])
        for record in records
    ]
    gradient_ratios = [
        float(record["weighted_intent_to_action_grad_ratio"])
        for record in records
        if record["weighted_intent_to_action_grad_ratio"] is not None
    ]
    grad_cosines = [
        float(record["action_intent_gradient_cosine"])
        for record in records
        if record["action_intent_gradient_cosine"] is not None
    ]
    condition_stats = _describe(all_condition_ratios)
    gradient_stats = _describe(gradient_ratios)
    condition_reference = condition_stats["mean"]
    gradient_reference = gradient_stats["median"]
    suggested_lambda = None
    if gradient_reference is not None and gradient_reference > 0:
        current_lambda = float(records[0]["intent_loss_weight"])
        suggested_lambda = min(
            0.30,
            max(0.01, current_lambda * 0.20 / float(gradient_reference)),
        )
    return {
        "action_loss_across_probe_runs": _describe(action_losses),
        "intent_ce_across_probe_runs": _describe(intent_losses),
        "weighted_intent_loss_across_probe_runs": _describe(weighted_intent_losses),
        "condition_ratio_all_samples": condition_stats,
        "action_grad_norm_on_last_projector": _describe(action_grad_norms),
        "weighted_intent_grad_norm_on_last_projector": _describe(
            weighted_intent_grad_norms
        ),
        "gradient_ratio_across_probe_runs": gradient_stats,
        "gradient_cosine_across_probe_runs": _describe(grad_cosines),
        "condition_ratio_assessment": _condition_assessment(
            float(condition_reference) if condition_reference is not None else None
        ),
        "gradient_ratio_assessment": _gradient_assessment(
            float(gradient_reference) if gradient_reference is not None else None
        ),
        "lambda_suggested_by_plan_formula": suggested_lambda,
        "assessment_note": (
            "Threshold labels are engineering diagnostics, not proof of causal "
            "policy improvement. Compare multiple fixed batches/checkpoints."
        ),
    }


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return

    _require_positive("--batch-size", args.batch_size)
    _require_positive("--num-batches", args.num_batches)
    _require_positive("--diffusion-repeats", args.diffusion_repeats)
    _require_positive("--noise-seeds-per-batch", args.noise_seeds_per_batch)
    if args.skip_batches < 0:
        raise ValueError("--skip-batches must be non-negative")
    if args.device is None:
        raise ValueError("--device is required unless --self-test is used")

    checkpoint = args.checkpoint.resolve()
    config_path = (args.config or _infer_config_path(checkpoint)).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    gpu_info = assert_gpu_is_safe(args.device, args.min_free_gib)
    device = torch.device(args.device)
    _set_seed(args.seed)

    cfg = _load_config(config_path, args.batch_size, args.diffusion_repeats)
    if str(cfg.framework.name) != "QwenPIIntent_v3":
        raise RuntimeError(
            f"Expected QwenPIIntent_v3, got framework.name={cfg.framework.name!r}"
        )
    if not bool(cfg.framework.intent.add_to_timestep_embedding):
        raise RuntimeError(
            "This checkpoint/config has add_to_timestep_embedding=false; "
            "the E1-B Action gradient path is disabled."
        )

    model = _build_and_load_model(cfg, checkpoint, device)
    if args.model_mode == "train":
        model.train()
    else:
        model.eval()
    target_params = _freeze_except_last_projector(model)
    loader = _build_dataset_loader(cfg, args.batch_size)

    records: list[dict[str, object]] = []
    all_condition_ratios: list[float] = []
    selected_batches = 0
    for batch_index, examples in enumerate(loader):
        if batch_index < args.skip_batches:
            continue
        if selected_batches >= args.num_batches:
            break

        batch_seed = args.seed + batch_index * 100_003
        _set_seed(batch_seed)
        static_projected, last_raw_hidden, attention_mask, actions_target = (
            _encode_and_cache_static_features(model, examples)
        )

        for noise_index in range(args.noise_seeds_per_batch):
            noise_seed = batch_seed + noise_index * 1_009
            record, sample_ratios = _probe_one_seed(
                model=model,
                examples=examples,
                static_projected=static_projected,
                last_raw_hidden=last_raw_hidden,
                attention_mask=attention_mask,
                actions_target=actions_target,
                target_params=target_params,
                repeat_count=args.diffusion_repeats,
                seed=noise_seed,
            )
            record["batch_index"] = batch_index
            record["noise_index"] = noise_index
            record["intent_loss_weight"] = float(model.intent_loss_weight)
            records.append(record)
            all_condition_ratios.extend(sample_ratios)
            print(
                f"batch={batch_index} noise={noise_index} "
                f"condition_ratio={record['condition_ratio']['mean']:.6g} "
                f"grad_ratio={record['weighted_intent_to_action_grad_ratio']} "
                f"grad_cos={record['action_intent_gradient_cosine']}",
                flush=True,
            )

        del static_projected, last_raw_hidden, attention_mask, actions_target
        gc.collect()
        torch.cuda.empty_cache()
        selected_batches += 1

    if selected_batches != args.num_batches:
        raise RuntimeError(
            f"Dataset ended after {selected_batches} selected batches; "
            f"requested {args.num_batches}."
        )

    result = {
        "metadata": {
            "checkpoint": str(checkpoint),
            "config": str(config_path),
            "device": gpu_info,
            "model_mode": args.model_mode,
            "batch_size": args.batch_size,
            "num_batches": args.num_batches,
            "skip_batches": args.skip_batches,
            "diffusion_repeats": args.diffusion_repeats,
            "training_diffusion_repeats": int(
                OmegaConf.load(config_path).trainer.get("repeated_diffusion_steps", 16)
            ),
            "noise_seeds_per_batch": args.noise_seeds_per_batch,
            "seed": args.seed,
            "gradient_target": "project_layers[-1]",
            "gradient_target_parameter_count": sum(
                parameter.numel() for parameter in target_params
            ),
            "intent_loss_weight": float(model.intent_loss_weight),
            "add_to_timestep_embedding": bool(
                model.add_intent_to_timestep_embedding
            ),
            "read_only": True,
            "optimizer_created": False,
            "optimizer_step_called": False,
        },
        "aggregate": _aggregate(records, all_condition_ratios),
        "records": records,
    }

    rendered = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    print(rendered)
    if args.output_json is not None:
        output_path = args.output_json.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote diagnostic result: {output_path}")


if __name__ == "__main__":
    main()
