"""Offline evaluation for the 125-way CALVIN spatial Intent head.

This evaluates classification on a labeled LeRobot dataset. It is separate
from CALVIN rollout success: simulator rollouts do not provide expert future
actions, so they cannot provide ground-truth Intent class accuracy.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from starVLA.dataloader.lerobot_datasets import collate_fn, get_vla_dataset
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import read_mode_config


LOGGER = logging.getLogger(__name__)


def decode_classes(class_ids: np.ndarray) -> np.ndarray:
    class_ids = np.asarray(class_ids, dtype=np.int64)
    return np.stack(
        (class_ids // 25, (class_ids % 25) // 5, class_ids % 5), axis=-1
    )


def classification_metrics(
    probabilities: np.ndarray, targets: np.ndarray
) -> dict:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    predictions = probabilities.argmax(axis=-1)
    top5 = np.argpartition(probabilities, -5, axis=-1)[:, -5:]
    eps = np.finfo(np.float64).tiny

    confusion = np.zeros((125, 125), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)
    support = confusion.sum(axis=1)
    predicted_count = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    supported = support > 0
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=supported,
    )
    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=predicted_count > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )

    target_bins = decode_classes(targets)
    predicted_bins = decode_classes(predictions)
    axis_confusions = []
    axis_accuracies = []
    axis_mean_distances = []
    for axis in range(3):
        matrix = np.zeros((5, 5), dtype=np.int64)
        np.add.at(matrix, (target_bins[:, axis], predicted_bins[:, axis]), 1)
        axis_confusions.append(matrix.tolist())
        axis_accuracies.append(
            float(np.mean(target_bins[:, axis] == predicted_bins[:, axis]))
        )
        axis_mean_distances.append(
            float(np.mean(np.abs(target_bins[:, axis] - predicted_bins[:, axis])))
        )

    entropy = -(probabilities * np.log(np.clip(probabilities, eps, None))).sum(
        axis=-1
    )
    return {
        "num_samples": int(len(targets)),
        "cross_entropy": float(
            -np.log(np.clip(probabilities[np.arange(len(targets)), targets], eps, None)).mean()
        ),
        "top1_accuracy": float(np.mean(predictions == targets)),
        "top5_accuracy": float(np.mean(np.any(top5 == targets[:, None], axis=1))),
        "balanced_accuracy_supported_classes": float(recall[supported].mean()),
        "macro_f1_supported_classes": float(f1[supported].mean()),
        "occupied_target_classes": int(supported.sum()),
        "mean_bin_manhattan_distance": float(
            np.abs(target_bins - predicted_bins).sum(axis=-1).mean()
        ),
        "per_axis_accuracy": dict(zip(("x", "y", "z"), axis_accuracies)),
        "per_axis_mean_bin_distance": dict(
            zip(("x", "y", "z"), axis_mean_distances)
        ),
        "mean_entropy": float(entropy.mean()),
        "mean_max_probability": float(probabilities.max(axis=-1).mean()),
        "target_class_counts": support.tolist(),
        "predicted_class_counts": predicted_count.tolist(),
        "per_axis_confusion_matrices": dict(
            zip(("x", "y", "z"), axis_confusions)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--data-mix", default=None)
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    model_config, _ = read_mode_config(str(checkpoint))
    cfg = OmegaConf.create(model_config)
    data_cfg = cfg.datasets.vla_data
    if args.data_root is not None:
        data_cfg.data_root_dir = args.data_root
    if args.data_mix is not None:
        data_cfg.data_mix = args.data_mix
    data_cfg.include_intent = True
    data_cfg.intent_class_column = "intent.class_id"
    data_cfg.intent_num_classes = 125
    data_cfg.video_backend = data_cfg.get("video_backend", "torchvision_av")

    mixture = get_vla_dataset(
        data_cfg=data_cfg,
        mode="test",
        balance_dataset_weights=False,
        balance_trajectory_weights=False,
        seed=args.seed,
    )
    dataset = mixture.datasets[0] if len(mixture.datasets) == 1 else mixture
    total_samples = len(dataset)
    requested = total_samples if args.max_samples <= 0 else min(
        args.max_samples, total_samples
    )
    rng = np.random.default_rng(args.seed)
    indices = np.sort(rng.choice(total_samples, size=requested, replace=False))
    dataloader = DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    model = baseframework.from_pretrained(str(checkpoint))
    if not hasattr(model, "predict_intent"):
        raise TypeError(
            f"{type(model).__name__} does not expose predict_intent; this is not an E1 Intent checkpoint"
        )
    if not args.no_bf16:
        model = model.to(torch.bfloat16)
    model = model.to(args.device).eval()

    all_probabilities = []
    all_targets = []
    for examples in tqdm(dataloader, desc="Intent evaluation"):
        output = model.predict_intent(examples=examples)
        all_probabilities.append(np.asarray(output["probabilities"]))
        all_targets.append(
            np.asarray([example["intent_class_id"] for example in examples])
        )

    probabilities = np.concatenate(all_probabilities, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    report = {
        "checkpoint": str(checkpoint),
        "data_root": str(data_cfg.data_root_dir),
        "data_mix": str(data_cfg.data_mix),
        "sample_seed": args.seed,
        "evaluation_scope": (
            "labeled offline dataset; the default data_mix is the training dataset, "
            "so these metrics measure fit rather than held-out generalization"
        ),
        "metrics": classification_metrics(probabilities, targets),
    }

    output_path = Path(args.output) if args.output else checkpoint.parent.parent / (
        f"intent_eval_{checkpoint.stem}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    LOGGER.info("Intent evaluation written to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
