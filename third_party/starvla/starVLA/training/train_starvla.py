# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).
"""

# Standard Library
import argparse
import copy
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Tuple

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist

# NPU support: import torch_npu and enable automatic CUDA→NPU mapping.
# On GPU-only environments this is a no-op (ImportError is silently ignored).
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    pass

import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.dataloader import build_dataloader
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups, setup_optimizer_and_scheduler, normalize_dotlist_args

deepspeed_plugin = DeepSpeedPlugin()
# The explicit training loop owns scheduler.step().  Keeping this disabled is
# essential when split_batches=False: AcceleratedScheduler would otherwise
# advance once per process (twice for the standard two-GPU run).
accelerator = Accelerator(
    deepspeed_plugin=deepspeed_plugin,
    step_scheduler_with_optimizer=False,
)
accelerator.print(accelerator.state)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize logger
logger = get_logger(__name__)


def load_fast_tokenizer():
    return AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def prepare_data(cfg, accelerator, output_dir) -> DataLoader:
    """Prepare VLA training data."""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(
        cfg=cfg,
        dataset_py=cfg.datasets.vla_data.dataset_py,
        mode="train",
    )
    vla_validation_dataloader = None
    validation_manifest = cfg.datasets.vla_data.get(
        "validation_trajectory_manifest", None
    )
    if validation_manifest:
        raw_cfg = cfg.unwrap() if isinstance(cfg, AccessTrackedConfig) else cfg
        validation_cfg = OmegaConf.create(
            OmegaConf.to_container(raw_cfg, resolve=True)
        )
        validation_cfg.datasets.vla_data.trajectory_manifest = validation_manifest
        validation_cfg.datasets.vla_data.persistent_workers = False
        vla_validation_dataloader = build_dataloader(
            cfg=validation_cfg,
            dataset_py=validation_cfg.datasets.vla_data.dataset_py,
            mode="val",
        )

    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()
    return vla_train_dataloader, vla_validation_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Set optimizer and scheduler."""
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
        fused=True,
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # Strip keys unknown to transformers' get_scheduler before passing kwargs.
    sched_kwargs = {k: v for k, v in cfg.trainer.scheduler_specific_kwargs.items()}
    scheduler_total_steps = int(
        cfg.trainer.get("scheduler_total_steps", cfg.trainer.max_train_steps)
    )
    if scheduler_total_steps <= 0:
        raise ValueError("trainer.scheduler_total_steps must be positive")
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=scheduler_total_steps,
        scheduler_specific_kwargs=sched_kwargs,
    )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(
        self,
        cfg,
        model,
        vla_train_dataloader,
        optimizer,
        lr_scheduler,
        accelerator,
        vla_validation_dataloader=None,
    ):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.vla_validation_dataloader = vla_validation_dataloader
        self.best_intent_validation_score = float("-inf")
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self._initial_lr_scheduler_state = copy.deepcopy(
            lr_scheduler.state_dict()
        )
        self._scheduler_limit_reported = False

        self.completed_steps = 0
        self._resume_training_state_dir = None
        self._full_training_state_loaded = False
        self.total_batch_size = self._calculate_total_batch_size()

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # Save config snapshots upfront so that even if a later setup step
        # (ckpt load / DeepSpeed init / dataloader build) crashes, the
        # produced run dir is still introspectable / from_pretrained-able.
        self._save_initial_configs()

        self._init_checkpointing()

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)

        # The scheduler must be prepared as well, otherwise
        # Accelerator.save_state() cannot persist/restore its exact position.
        components = [
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
            self.lr_scheduler,
        ]
        if self.vla_validation_dataloader is not None:
            components.append(self.vla_validation_dataloader)
        prepared = self.setup_distributed_training(self.accelerator, *components)
        (
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
            self.lr_scheduler,
        ) = prepared[:4]
        if self.vla_validation_dataloader is not None:
            self.vla_validation_dataloader = prepared[4]

        self._restore_full_training_state()
        self._adjust_lr_scheduler_for_resume()
        self._init_wandb()

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    @staticmethod
    def _module_grad_norm(module: torch.nn.Module):
        """Return a pre-clipping L2 gradient norm, or None when no gradient exists."""

        return VLATrainer._parameters_grad_norm(module.parameters())

    @staticmethod
    def _parameters_grad_norm(parameters):
        """Return an L2 norm for a parameter iterable without re-registering modules."""

        squared_norm = None
        for parameter in parameters:
            if parameter.grad is None:
                continue
            contribution = parameter.grad.detach().float().square().sum()
            squared_norm = contribution if squared_norm is None else squared_norm + contribution
        return squared_norm.sqrt() if squared_norm is not None else None

    def _collect_intent_gradient_metrics(self, model):
        """Collect E1 pre-clipping norms only on steps that will be logged."""

        gradient_metrics = {}
        monitored_modules = {
            "action_model/grad_norm": getattr(model, "action_model", None),
            "intent_head/grad_norm": getattr(model, "intent_head", None),
            "intent_to_timestep/grad_norm": getattr(model, "intent_to_timestep", None),
        }
        intent_head = getattr(model, "intent_head", None)
        if intent_head is not None:
            monitored_modules.update(
                {
                    "intent_project_layers/grad_norm": getattr(
                        intent_head, "project_layers", None
                    ),
                    "intent_token_attention/grad_norm": getattr(
                        intent_head, "token_attention", None
                    ),
                    "intent_token_query_attention/grad_norm": getattr(
                        getattr(intent_head, "token_query_block", None),
                        "attention",
                        None,
                    ),
                    "intent_token_query_ffn/grad_norm": getattr(
                        getattr(intent_head, "token_query_block", None),
                        "ffn",
                        None,
                    ),
                    "intent_layer_attention/grad_norm": getattr(
                        intent_head, "layer_attention", None
                    ),
                    "intent_layer_query_attention/grad_norm": getattr(
                        getattr(intent_head, "layer_query_block", None),
                        "attention",
                        None,
                    ),
                    "intent_layer_query_ffn/grad_norm": getattr(
                        getattr(intent_head, "layer_query_block", None),
                        "ffn",
                        None,
                    ),
                    "intent_classifier/grad_norm": getattr(
                        intent_head, "classifier", None
                    ),
                    "intent_layer_query_blocks/grad_norm": getattr(
                        intent_head, "layer_query_blocks", None
                    ),
                    "intent_xyz_classifier/grad_norm": getattr(
                        intent_head, "xyz_classifier", None
                    ),
                    "intent_rpy_classifier/grad_norm": getattr(
                        intent_head, "rpy_classifier", None
                    ),
                    "intent_gripper_classifier/grad_norm": getattr(
                        intent_head, "gripper_classifier", None
                    ),
                }
            )
        action_dit = getattr(getattr(model, "action_model", None), "model", None)
        if action_dit is not None:
            transformer_blocks = getattr(action_dit, "transformer_blocks", [])
            film_parameter_groups = {
                "query_film/grad_norm": (
                    parameter
                    for block in transformer_blocks
                    for module in [getattr(block, "cross_attn_query_intent_film", None)]
                    if module is not None
                    for parameter in module.parameters()
                ),
                "ffn_film/grad_norm": (
                    parameter
                    for block in transformer_blocks
                    for module in [getattr(block, "ffn_intent_film", None)]
                    if module is not None
                    for parameter in module.parameters()
                ),
            }
            for metric_name, parameters in film_parameter_groups.items():
                grad_norm = self._parameters_grad_norm(parameters)
                if grad_norm is not None:
                    gradient_metrics[metric_name] = grad_norm.detach().item()
        for metric_name, module in monitored_modules.items():
            if module is None:
                continue
            grad_norm = self._module_grad_norm(module)
            if grad_norm is not None:
                gradient_metrics[metric_name] = grad_norm.detach().item()
        return gradient_metrics

    def _metric_scalar(self, value, *, aggregate: bool) -> float:
        """Convert a scalar to float, averaging ranks on logging steps."""

        if isinstance(value, torch.Tensor):
            scalar = value.detach().float().reshape(1)
        else:
            scalar = torch.tensor(
                [float(value)], device=self.accelerator.device, dtype=torch.float32
            )
        if aggregate and self.accelerator.num_processes > 1:
            scalar = self.accelerator.gather_for_metrics(scalar).mean().reshape(1)
        return scalar.item()

    def _init_wandb(self):
        """Initialize W&B, with strict failure for an explicitly required resume."""
        self._wandb_enabled = False
        if os.environ.get("WANDB_MODE") == "disabled" or os.environ.get("WANDB_DISABLED", "").lower() in {
            "1",
            "true",
            "yes",
        }:
            self.accelerator.wait_for_everyone()
            return
        if self.accelerator.is_main_process:
            try:
                wandb_init_kwargs = {}
                if os.environ.get("WANDB_RUN_ID"):
                    wandb_init_kwargs["id"] = os.environ["WANDB_RUN_ID"]
                if os.environ.get("WANDB_RESUME"):
                    wandb_init_kwargs["resume"] = os.environ["WANDB_RESUME"]
                wandb.init(
                    name=self.config.get("wandb_name", self.config.run_id),
                    dir=os.path.join(self.config.output_dir, "wandb"),
                    project=self.config.wandb_project,
                    entity=self.config.wandb_entity,
                    group="vla-train",
                    **wandb_init_kwargs,
                )
                self._wandb_enabled = True
            except Exception as exc:
                wandb_required = os.environ.get(
                    "WANDB_REQUIRED", ""
                ).lower() in {"1", "true", "yes"}
                if (
                    os.environ.get("WANDB_RESUME", "").lower() == "must"
                    or wandb_required
                ):
                    logger.error(f"Required W&B initialization failed: {exc}")
                    raise
                logger.warning(f"W&B init failed; continuing without W&B: {exc}")
                self._wandb_enabled = False
        # Rendezvous after rank-0 W&B init. Otherwise a slow or failing init on
        # rank 0 lets the other ranks reach the first collective alone and
        # eventually hit an NCCL watchdog timeout.
        self.accelerator.wait_for_everyone()

    def _save_initial_configs(self):
        """Save full config and training script at the very start of training."""
        if not self.accelerator.is_main_process:
            return

        output_dir = Path(self.config.output_dir)

        # 1. Save config.full.yaml — the complete merged config (all parameters)
        if isinstance(self.config, AccessTrackedConfig):
            full_cfg = self.config.unwrap()
        else:
            full_cfg = self.config
        full_yaml_path = output_dir / "config.full.yaml"
        OmegaConf.save(full_cfg, full_yaml_path, resolve=True)
        logger.info(f"📝 Full config saved at {full_yaml_path}")

        # 2. Save config.yaml — accessed-only snapshot (will be updated at checkpoints)
        if isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
            logger.info(f"📊 Accessed config snapshot saved at {output_dir / 'config.yaml'}")

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        best_metrics_path = Path(self.checkpoint_dir) / "best_intent_metrics.json"
        if best_metrics_path.exists():
            best_metrics = json.loads(best_metrics_path.read_text(encoding="utf-8"))
            self.best_intent_validation_score = float(
                best_metrics.get("selection_score", float("-inf"))
            )

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint

        if is_resume:
            requested_training_state = self.config.trainer.get(
                "resume_training_state", None
            )
            if requested_training_state:
                resume_state, resume_step = self._validate_training_state_dir(
                    requested_training_state
                )
            else:
                resume_state, resume_step = self._get_latest_training_state(
                    self.checkpoint_dir
                )
            if resume_state:
                self._resume_training_state_dir = resume_state
                self.completed_steps = resume_step
                logger.info(
                    "Found complete model/optimizer/scheduler training state: "
                    f"{resume_state}, steps: {resume_step}"
                )
                return

            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                logger.info(
                    f"Resuming training from checkpoint: {self.resume_from_checkpoint}, steps: {self.completed_steps}"
                )
                return

            logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
            self.completed_steps = 0

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            initialize_intent_projectors = bool(
                self.config.framework.get("intent", {}).get(
                    "initialize_projectors_from_action", False
                )
            )
            if initialize_intent_projectors and hasattr(
                self.model, "initialize_intent_projectors_from_action"
            ):
                self.model.initialize_intent_projectors_from_action()
                logger.info("Initialized independent Intent projectors from Action projectors")

            intent_pretrained_checkpoint = self.config.trainer.get(
                "intent_pretrained_checkpoint", None
            )
            if intent_pretrained_checkpoint:
                self.model = self.load_pretrained_backbones(
                    self.model,
                    intent_pretrained_checkpoint,
                    reload_modules="intent_head",
                )
                logger.info(
                    "Overlayed pretrained Intent branch from "
                    f"{intent_pretrained_checkpoint}"
                )
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

    @staticmethod
    def _validate_training_state_dir(training_state_dir):
        """Validate a completed full-state checkpoint and return its saved step."""

        path = Path(training_state_dir).expanduser().resolve()
        metadata_path = path / "trainer_state.json"
        if not path.is_dir():
            raise FileNotFoundError(
                f"Training-state checkpoint directory does not exist: {path}"
            )
        if not metadata_path.is_file():
            raise RuntimeError(
                "Training-state checkpoint is incomplete (missing "
                f"{metadata_path.name}): {path}"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        completed_steps = int(metadata["completed_steps"])
        match = re.fullmatch(r"steps_(\d+)_training_state", path.name)
        if match is None or int(match.group(1)) != completed_steps:
            raise RuntimeError(
                "Training-state directory name and metadata step disagree: "
                f"{path}, completed_steps={completed_steps}"
            )
        return str(path), completed_steps

    def _get_latest_training_state(self, checkpoint_dir):
        """Return the newest complete Accelerator/DeepSpeed training state."""

        root = Path(checkpoint_dir)
        if not root.is_dir():
            return None, 0
        candidates = []
        for path in root.iterdir():
            match = re.fullmatch(r"steps_(\d+)_training_state", path.name)
            if not path.is_dir() or match is None:
                continue
            try:
                validated_path, completed_steps = (
                    self._validate_training_state_dir(path)
                )
            except (FileNotFoundError, KeyError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
                logger.warning(f"Ignoring incomplete training state {path}: {exc}")
                continue
            candidates.append((completed_steps, validated_path))
        if not candidates:
            return None, 0
        completed_steps, path = max(candidates, key=lambda item: item[0])
        return path, completed_steps

    def _restore_full_training_state(self):
        """Restore model, AdamW, scheduler, dataloader and RNG after prepare()."""

        if self._resume_training_state_dir is None:
            return
        metadata_path = (
            Path(self._resume_training_state_dir) / "trainer_state.json"
        )
        saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.accelerator.load_state(self._resume_training_state_dir)
        self._full_training_state_loaded = True
        current_schedule_id = str(
            self.config.trainer.get("lr_schedule_id", "default")
        )
        saved_schedule_id = str(
            saved_metadata.get("lr_schedule_id", "default")
        )
        reset_on_mismatch = bool(
            self.config.trainer.get(
                "reset_scheduler_on_schedule_mismatch", False
            )
        )
        if reset_on_mismatch and saved_schedule_id != current_schedule_id:
            expected_origin = int(
                self.config.trainer.get(
                    "scheduler_restart_origin_step", self.completed_steps
                )
            )
            if self.completed_steps != expected_origin:
                raise RuntimeError(
                    "Refusing to reset an LR schedule at an unexpected step: "
                    f"checkpoint={self.completed_steps}, "
                    f"expected={expected_origin}"
                )
            self._reset_lr_scheduler_for_new_schedule()
            logger.warning(
                "Reset LR scheduler after schedule-id change: "
                f"{saved_schedule_id!r} -> {current_schedule_id!r}. "
                "Model and AdamW moments remain restored."
            )
        self.accelerator.print(
            "✅ Restored full training state at step "
            f"{self.completed_steps} from {self._resume_training_state_dir}"
        )

    @staticmethod
    def _unwrap_lr_scheduler(lr_scheduler):
        """Return the underlying PyTorch scheduler from Accelerate wrappers."""

        return getattr(lr_scheduler, "scheduler", lr_scheduler)

    def _reset_lr_scheduler_for_new_schedule(self):
        """Start the configured recovery schedule without resetting AdamW."""

        scheduler = self._unwrap_lr_scheduler(self.lr_scheduler)
        scheduler.load_state_dict(
            copy.deepcopy(self._initial_lr_scheduler_state)
        )
        current_lrs = list(scheduler.get_last_lr())
        if len(current_lrs) != len(self.optimizer.param_groups):
            raise RuntimeError(
                "Scheduler/optimizer parameter-group count changed during resume"
            )
        for optimizer_group, lr in zip(
            self.optimizer.param_groups, current_lrs
        ):
            optimizer_group["lr"] = float(lr)
        if scheduler.optimizer is not self.optimizer:
            for optimizer_group, lr in zip(
                scheduler.optimizer.param_groups, current_lrs
            ):
                optimizer_group["lr"] = float(lr)
        logger.info(
            "Started replacement LR schedule at local scheduler step %s with "
            "group LRs %s",
            scheduler.last_epoch,
            current_lrs,
        )

    def _step_lr_scheduler(self):
        """Advance exactly once and clamp permanently at the schedule endpoint."""

        scheduler = self._unwrap_lr_scheduler(self.lr_scheduler)
        scheduler_total_steps = int(
            self.config.trainer.get(
                "scheduler_total_steps", self.config.trainer.max_train_steps
            )
        )
        if int(scheduler.last_epoch) >= scheduler_total_steps:
            if not self._scheduler_limit_reported:
                logger.warning(
                    "LR scheduler reached its endpoint (%s); holding the "
                    "final LR instead of entering another cosine half-cycle.",
                    scheduler_total_steps,
                )
                self._scheduler_limit_reported = True
            return
        self.lr_scheduler.step()

    def _adjust_lr_scheduler_for_resume(self):
        """Advance only legacy weight-only resumes; full-state resumes load it."""

        if self._full_training_state_loaded:
            logger.info(
                "LR scheduler restored from full training state; "
                f"current LR: {self.lr_scheduler.get_last_lr()}"
            )
            return
        scheduler_step_offset = int(
            self.config.trainer.get("scheduler_step_offset", 0)
        )
        if scheduler_step_offset < 0:
            raise ValueError("trainer.scheduler_step_offset must be non-negative")
        scheduler_total_steps = int(
            self.config.trainer.get(
                "scheduler_total_steps", self.config.trainer.max_train_steps
            )
        )
        scheduler_steps = min(
            scheduler_step_offset + self.completed_steps,
            scheduler_total_steps,
        )
        if scheduler_steps > 0:
            logger.info(
                "Adjusting LR scheduler to global step "
                f"{scheduler_steps} (offset={scheduler_step_offset}, "
                f"phase_step={self.completed_steps})"
            )
            for _ in range(scheduler_steps):
                self._step_lr_scheduler()
            logger.info(
                f"LR scheduler adjusted to step {scheduler_steps}, "
                f"current LR: {self.lr_scheduler.get_last_lr()}"
            )

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _training_state_metadata(self):
        """Describe the exact next optimization step represented by a save."""

        scheduler_step_offset = int(
            self.config.trainer.get("scheduler_step_offset", 0)
        )
        stage1_steps = int(
            self.config.framework.get("intent", {}).get("stage1_steps", 0)
        )
        next_stage = (
            "s1"
            if self.config.framework.get("intent", {}).get("training_stage")
            == "s1_s2"
            and self.completed_steps < stage1_steps
            else "s2"
        )
        return {
            "format_version": 1,
            "completed_steps": int(self.completed_steps),
            "global_scheduler_step": int(
                scheduler_step_offset + self.completed_steps
            ),
            "next_training_stage": next_stage,
            "stage1_steps": stage1_steps,
            "optimizer": "AdamW",
            "lr_schedule_id": str(
                self.config.trainer.get("lr_schedule_id", "default")
            ),
            "scheduler_last_epoch": int(
                self._unwrap_lr_scheduler(self.lr_scheduler).last_epoch
            ),
            "parameter_group_names": [
                str(group.get("name", index))
                for index, group in enumerate(self.optimizer.param_groups)
            ],
        }

    def _prune_old_training_states(self):
        """Keep recent full states while retaining every lightweight model file."""

        keep_last = int(
            self.config.trainer.get("training_state_keep_last", 0)
        )
        if keep_last <= 0:
            return
        states = []
        for path in Path(self.checkpoint_dir).iterdir():
            match = re.fullmatch(r"steps_(\d+)_training_state", path.name)
            if path.is_dir() and match is not None:
                states.append((int(match.group(1)), path))
        states.sort(key=lambda item: item[0])
        for _, path in states[:-keep_last]:
            shutil.rmtree(path)
            logger.info(
                "Removed old full training state to limit disk use: "
                f"{path}. The corresponding model-only checkpoint is retained."
            )

    def _save_checkpoint(self):
        """Save current training state."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")

            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, checkpoint_path + "_model.safetensors")
            elif save_format == "pt":
                torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

            summary_data = {"steps": self.completed_steps}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")

            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()

        if bool(self.config.trainer.get("save_training_state", False)):
            training_state_dir = os.path.join(
                self.checkpoint_dir,
                f"steps_{self.completed_steps}_training_state",
            )
            if self.accelerator.is_main_process:
                state_path = Path(training_state_dir)
                if state_path.exists():
                    marker = state_path / "trainer_state.json"
                    if marker.exists():
                        raise FileExistsError(
                            "Refusing to overwrite a completed training state: "
                            f"{training_state_dir}"
                        )
                    shutil.rmtree(state_path)
                    logger.warning(
                        "Removed an incomplete state directory left by an "
                        f"interrupted save: {training_state_dir}"
                    )
            self.accelerator.wait_for_everyone()

            # All ranks must participate because DeepSpeed ZeRO stores sharded
            # model/optimizer state. The scheduler was included in prepare().
            self.accelerator.save_state(training_state_dir)
            self.accelerator.wait_for_everyone()

            if self.accelerator.is_main_process:
                metadata_path = Path(training_state_dir) / "trainer_state.json"
                metadata_tmp = metadata_path.with_suffix(".json.tmp")
                metadata_tmp.write_text(
                    json.dumps(
                        self._training_state_metadata(),
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                os.replace(metadata_tmp, metadata_path)
                self._prune_old_training_states()
                self.accelerator.print(
                    "✅ Full model/AdamW/scheduler/RNG state saved at "
                    f"{training_state_dir}"
                )
            self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """Record training metrics."""
        if self.completed_steps % self.config.trainer.logging_frequency == 0 and dist.get_rank() == 0:
            phase_step = self.completed_steps
            wandb_step_offset = int(
                self.config.trainer.get("wandb_step_offset", 0)
            )
            if wandb_step_offset < 0:
                raise ValueError("trainer.wandb_step_offset must be non-negative")
            global_completed_steps = wandb_step_offset + phase_step
            last_lrs = self.lr_scheduler.get_last_lr()
            for i, group in enumerate(self.optimizer.param_groups):
                group_name = group.get("name", str(i))
                metrics[f"learning_rate/{group_name}"] = last_lrs[i] if i < len(last_lrs) else last_lrs[-1]
            metrics["training/phase_step"] = phase_step
            metrics["training/phase_id"] = int(
                self.config.trainer.get("training_phase_id", 0)
            )
            recovered_from_step = self.config.trainer.get(
                "recovered_from_phase_step", None
            )
            if recovered_from_step is not None:
                metrics["training/recovered_from_phase_step"] = int(
                    recovered_from_step
                )
            metrics["epoch"] = round(
                global_completed_steps / len(self.vla_train_dataloader), 2
            )
            wandb_log_after_step = int(
                self.config.trainer.get("wandb_log_after_step", -1)
            )
            should_log_to_wandb = global_completed_steps > wandb_log_after_step
            if getattr(self, "_wandb_enabled", False) and should_log_to_wandb:
                try:
                    wandb.log(metrics, step=global_completed_steps)
                except Exception as exc:
                    self._wandb_enabled = False
                    logger.warning(f"W&B log failed; disabling W&B: {exc}")
            logger.info(
                f"Phase step {phase_step}, global logging step "
                f"{global_completed_steps}, Loss: {metrics})"
            )

    def _create_data_iterators(self):
        """Create data iterators."""
        self.vla_iter = iter(self.vla_train_dataloader)

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)."""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def train(self):
        """Execute training loop."""
        self._log_training_config()
        self._create_data_iterators()
        progress_bar = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            optimizer_stepped = step_metrics.pop(
                "_optimizer_step", self.accelerator.sync_gradients
            )
            if optimizer_stepped:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            if (
                optimizer_stepped
                and bool(self.config.trainer.get("enable_action_eval", True))
                and self.completed_steps % self.config.trainer.eval_interval == 0
            ):
                step_metrics = self.eval_action_model(step_metrics)

            validation_interval = int(
                self.config.trainer.get("intent_validation_interval", 0)
            )
            if (
                optimizer_stepped
                and self.vla_validation_dataloader is not None
                and validation_interval > 0
                and self.completed_steps % validation_interval == 0
            ):
                step_metrics.update(self.eval_factorized_intent())

            step_metrics["timing/data"] = t_end_data - t_start_data
            step_metrics["timing/model"] = t_end_model - t_start_model
            if optimizer_stepped:
                self._log_metrics(step_metrics)

            if (
                optimizer_stepped
                and self.completed_steps % self.config.trainer.save_interval == 0
                and self.completed_steps > 0
            ):
                self._save_checkpoint()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    @staticmethod
    def _classification_metrics_from_confusion(
        confusion: torch.Tensor,
    ) -> tuple[float, float]:
        confusion = confusion.double()
        tp = confusion.diag()
        support = confusion.sum(dim=1)
        predicted = confusion.sum(dim=0)
        recall = tp / support.clamp_min(1)
        precision = tp / predicted.clamp_min(1)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1.0e-12)
        occupied = support > 0
        return (
            f1[occupied].mean().item() if occupied.any() else 0.0,
            recall[occupied].mean().item() if occupied.any() else 0.0,
        )

    @torch.inference_mode()
    def eval_factorized_intent(self) -> dict[str, float]:
        """Evaluate the fixed trajectory-level validation split and save best S0."""

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        if not getattr(unwrapped_model, "use_factorized_intent", False):
            return {}
        was_training = self.model.training
        self.model.eval()
        full_interval = int(
            self.config.trainer.get("intent_validation_full_interval", 5000)
        )
        run_full = full_interval > 0 and self.completed_steps % full_interval == 0
        subset_batches = int(
            self.config.trainer.get("intent_validation_subset_batches", 128)
        )
        max_batches = len(self.vla_validation_dataloader) if run_full else subset_batches
        max_batches = min(max_batches, len(self.vla_validation_dataloader))

        confusions = {
            "xyz": torch.zeros(125, 125, device=self.accelerator.device, dtype=torch.long),
            "rpy": torch.zeros(125, 125, device=self.accelerator.device, dtype=torch.long),
            "gripper": torch.zeros(5, 5, device=self.accelerator.device, dtype=torch.long),
        }
        loss_sums = {
            name: torch.zeros((), device=self.accelerator.device)
            for name in ("xyz", "rpy", "gripper")
        }
        top5_correct = {
            name: torch.zeros((), device=self.accelerator.device, dtype=torch.long)
            for name in ("xyz", "rpy", "gripper")
        }
        entropy_sums = {
            name: torch.zeros((), device=self.accelerator.device)
            for name in ("xyz", "rpy", "gripper")
        }
        axis_correct = {
            name: torch.zeros(3, device=self.accelerator.device)
            for name in ("xyz", "rpy")
        }
        axis_distance = {
            name: torch.zeros(3, device=self.accelerator.device)
            for name in ("xyz", "rpy")
        }
        within_chebyshev_one = {
            name: torch.zeros((), device=self.accelerator.device)
            for name in ("xyz", "rpy")
        }
        sample_count = torch.zeros(
            (), device=self.accelerator.device, dtype=torch.long
        )
        for batch_index, examples in enumerate(self.vla_validation_dataloader):
            if batch_index >= max_batches:
                break
            output = self.model(examples)
            batch_size = len(examples)
            sample_count += batch_size
            for name, classes in (("xyz", 125), ("rpy", 125), ("gripper", 5)):
                logits = output[f"intent_{name}_logits"]
                targets = output[f"intent_{name}_targets"]
                predictions = logits.argmax(dim=-1)
                probabilities = torch.softmax(logits.float(), dim=-1)
                entropy_sums[name] += (
                    -(
                        probabilities
                        * probabilities.clamp_min(
                            torch.finfo(torch.float32).tiny
                        ).log()
                    ).sum(dim=-1)
                ).sum()
                top5_correct[name] += (
                    logits.topk(min(5, classes), dim=-1)
                    .indices.eq(targets[:, None])
                    .any(dim=-1)
                    .sum()
                )
                flat = targets * classes + predictions
                confusions[name] += torch.bincount(
                    flat, minlength=classes * classes
                ).reshape(classes, classes)
                loss_sums[name] += (
                    torch.nn.functional.cross_entropy(
                        logits.float(), targets, reduction="sum"
                    )
                )
                if name in axis_correct:
                    predicted_bins = torch.stack(
                        [
                            predictions // 25,
                            (predictions % 25) // 5,
                            predictions % 5,
                        ],
                        dim=-1,
                    )
                    target_bins = torch.stack(
                        [
                            targets // 25,
                            (targets % 25) // 5,
                            targets % 5,
                        ],
                        dim=-1,
                    )
                    distance = (predicted_bins - target_bins).abs().float()
                    axis_correct[name] += (distance == 0).sum(dim=0)
                    axis_distance[name] += distance.sum(dim=0)
                    within_chebyshev_one[name] += (
                        distance.max(dim=-1).values <= 1
                    ).sum()

        sample_count = self.accelerator.reduce(sample_count, reduction="sum")
        for name in confusions:
            confusions[name] = self.accelerator.reduce(
                confusions[name], reduction="sum"
            )
            loss_sums[name] = self.accelerator.reduce(
                loss_sums[name], reduction="sum"
            )
            top5_correct[name] = self.accelerator.reduce(
                top5_correct[name], reduction="sum"
            )
            entropy_sums[name] = self.accelerator.reduce(
                entropy_sums[name], reduction="sum"
            )
            if name in axis_correct:
                axis_correct[name] = self.accelerator.reduce(
                    axis_correct[name], reduction="sum"
                )
                axis_distance[name] = self.accelerator.reduce(
                    axis_distance[name], reduction="sum"
                )
                within_chebyshev_one[name] = self.accelerator.reduce(
                    within_chebyshev_one[name], reduction="sum"
                )
        count = max(int(sample_count.item()), 1)
        metrics = {
            "validation/num_samples": float(sample_count.item()),
            "validation/is_full": float(run_full),
        }
        macro_f1 = {}
        for name in ("xyz", "rpy", "gripper"):
            f1, balanced_accuracy = self._classification_metrics_from_confusion(
                confusions[name]
            )
            macro_f1[name] = f1
            metrics[f"validation/{name}_macro_f1"] = f1
            metrics[f"validation/{name}_balanced_accuracy"] = balanced_accuracy
            metrics[f"validation/{name}_ce"] = loss_sums[name].item() / count
            metrics[f"validation/{name}_top1_accuracy"] = (
                confusions[name].diag().sum().item() / count
            )
            metrics[f"validation/{name}_top5_accuracy"] = (
                top5_correct[name].item() / count
            )
            metrics[f"validation/{name}_probability_entropy"] = (
                entropy_sums[name].item() / count
            )
            if name in axis_correct:
                for axis_index, axis_name in enumerate(("x", "y", "z")):
                    metrics[
                        f"validation/{name}_{axis_name}_accuracy"
                    ] = axis_correct[name][axis_index].item() / count
                    metrics[
                        f"validation/{name}_{axis_name}_mean_bin_distance"
                    ] = axis_distance[name][axis_index].item() / count
                metrics[
                    f"validation/{name}_within_chebyshev_1_accuracy"
                ] = within_chebyshev_one[name].item() / count
        score = (
            0.4 * macro_f1["xyz"]
            + 0.4 * macro_f1["rpy"]
            + 0.2 * macro_f1["gripper"]
        )
        metrics["validation/selection_score"] = score
        if run_full and score > self.best_intent_validation_score:
            self.best_intent_validation_score = score
            self._save_best_intent_checkpoint(metrics)
        if was_training:
            self.model.train()
        return metrics

    def _save_best_intent_checkpoint(self, metrics: dict[str, float]) -> None:
        """Persist only the best validation-selected Intent branch."""

        if self.accelerator.is_main_process:
            intent_head = getattr(
                self.accelerator.unwrap_model(self.model), "intent_head"
            )
            intent_state = {
                f"intent_head.{name}": value.detach().cpu()
                for name, value in intent_head.state_dict().items()
            }
            path = Path(self.checkpoint_dir) / "best_intent_pytorch_model.pt"
            torch.save(intent_state, path)
            write_path = Path(self.checkpoint_dir) / "best_intent_metrics.json"
            write_path.write_text(
                json.dumps(
                    {
                        "step": self.completed_steps,
                        "selection_score": self.best_intent_validation_score,
                        **metrics,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            logger.info(
                "Saved best validation Intent checkpoint at step %s: score=%.6f",
                self.completed_steps,
                self.best_intent_validation_score,
            )
        self.accelerator.wait_for_everyone()

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """Run simple action-eval on current batch and attach score to metrics."""
        examples = self._get_next_batch()
        actions = [example["action"] for example in examples]
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        output_dict = unwrapped_model.predict_action(
            examples=examples, use_ddim=True, num_ddim_steps=20
        )

        enable_intent_probe = bool(
            self.config.trainer.get("enable_intent_condition_probe", False)
        )
        probe_metrics = {}
        if enable_intent_probe and hasattr(unwrapped_model, "diagnose_intent_condition"):
            probe_seed = int(
                self.config.trainer.get(
                    "intent_condition_probe_seed",
                    getattr(self.config, "seed", 42),
                )
            )
            probe_metrics = unwrapped_model.diagnose_intent_condition(
                examples=examples,
                probe_seed=probe_seed,
            )

        if self.accelerator.is_main_process:
            normalized_actions = output_dict["normalized_actions"]
            actions = np.array(actions)
            num_pots = np.prod(actions.shape)
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            step_metrics["mse_score"] = score / num_pots
            for metric_name, value in probe_metrics.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    step_metrics[metric_name] = value.detach().item()

        del examples
        dist.barrier()
        return step_metrics

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_vla, batch_vlm=None):
        """Execute single training step."""
        will_log = (
            (self.completed_steps + 1)
            % int(self.config.trainer.logging_frequency)
            == 0
        )
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        if hasattr(unwrapped_model, "set_training_step"):
            unwrapped_model.set_training_step(self.completed_steps)

        # Under ZeRO-2/3, DeepSpeed owns gradient-accumulation boundaries.
        # Accelerate's accumulate() context uses no_sync(), which DeepSpeed
        # explicitly disallows with partitioned gradients when accumulation > 1.
        is_deepspeed_engine = hasattr(
            self.model, "is_gradient_accumulation_boundary"
        ) and hasattr(self.model, "backward")
        if is_deepspeed_engine:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
                # E1 frameworks return the already-composed 4.1 objective.
                # Existing frameworks keep their original action-only behavior.
                total_loss = output_dict.get("total_loss", action_loss)

            self.model.backward(total_loss)
            optimizer_stepped = bool(
                self.model.is_gradient_accumulation_boundary()
            )

            # Capture pre-clipping gradients only for the micro-step that will
            # produce the logged optimizer update. DeepSpeed applies the
            # configured global clipping inside engine.step().
            gradient_metrics = {}
            if will_log and optimizer_stepped:
                gradient_metrics = self._collect_intent_gradient_metrics(
                    unwrapped_model
                )

            self.model.step()
            if optimizer_stepped:
                self._step_lr_scheduler()
        else:
            with self.accelerator.accumulate(self.model):
                self.optimizer.zero_grad()

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output_dict = self.model.forward(batch_vla)
                    action_loss = output_dict["action_loss"]
                    total_loss = output_dict.get("total_loss", action_loss)

                self.accelerator.backward(total_loss)

                gradient_metrics = {}
                if will_log:
                    gradient_metrics = self._collect_intent_gradient_metrics(
                        unwrapped_model
                    )

                if self.config.trainer.gradient_clipping is not None:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.trainer.gradient_clipping,
                    )

                self.optimizer.step()
                optimizer_stepped = self.accelerator.sync_gradients
                if optimizer_stepped:
                    self._step_lr_scheduler()

        metrics = {
            "action_dit_loss": self._metric_scalar(action_loss, aggregate=will_log),
            "total_loss": self._metric_scalar(total_loss, aggregate=will_log),
            "loss/intent_contribution": self._metric_scalar(
                total_loss.detach() - action_loss.detach(), aggregate=will_log
            ),
        }
        metrics.update(
            {
                name: self._metric_scalar(value, aggregate=will_log)
                for name, value in gradient_metrics.items()
            }
        )
        metrics["_optimizer_step"] = optimizer_stepped
        optional_metric_names = {
            "intent_loss": "intent_ce",
            "weighted_intent_loss": "weighted_intent_loss",
            "intent_top1_accuracy": "intent_top1_accuracy",
            "intent_top5_accuracy": "intent_top5_accuracy",
            "intent_probability_entropy": "intent/probability_entropy",
            "intent_max_probability": "intent/max_probability",
            "intent_film_confidence_mean": "intent/film_confidence_mean",
            "intent_condition_l2_mean": "intent_condition/l2_mean",
            "intent_condition_abs_mean": "intent_condition/abs_mean",
            "intent_to_timestep_weight_norm": "intent_to_timestep/weight_norm",
            "timestep_embedding_l2_mean": "timestep_embedding/l2_mean",
            "joint_timestep_condition_l2_mean": "joint_timestep_condition/l2_mean",
            "intent_condition_to_timestep_l2_ratio": "intent_condition/to_timestep_l2_ratio",
            "intent_training_stage_id": "intent/training_stage_id",
            "intent_global_feature_l2_mean": "intent/global_feature_l2_mean",
            "intent_token_attention_residual_ratio": "intent/token_attention_residual_ratio",
            "intent_token_ffn_residual_ratio": "intent/token_ffn_residual_ratio",
            "intent_layer_attention_residual_ratio": "intent/layer_attention_residual_ratio",
            "intent_layer_ffn_residual_ratio": "intent/layer_ffn_residual_ratio",
            "intent_token_summary_pre_norm_l2_mean": "intent/token_summary_pre_norm_l2_mean",
            "intent_token_summary_post_norm_l2_mean": "intent/token_summary_post_norm_l2_mean",
            "intent_global_feature_pre_norm_l2_mean": "intent/global_feature_pre_norm_l2_mean",
            "intent_global_feature_post_norm_l2_mean": "intent/global_feature_post_norm_l2_mean",
            "intent_layer_attention_entropy": "intent/layer_attention_entropy",
            "query_film_delta_gamma_rms_mean": "query_film/delta_gamma_rms_mean",
            "query_film_delta_beta_rms_mean": "query_film/delta_beta_rms_mean",
            "query_film_modulation_to_query_l2_ratio_mean": "query_film/modulation_to_query_l2_ratio_mean",
            "ffn_film_delta_gamma_rms_mean": "ffn_film/delta_gamma_rms_mean",
            "ffn_film_delta_beta_rms_mean": "ffn_film/delta_beta_rms_mean",
            "ffn_film_modulation_to_input_l2_ratio_mean": "ffn_film/modulation_to_input_l2_ratio_mean",
            "query_film_raw_delta_gamma_abs_max_mean": "query_film/raw_delta_gamma_abs_max",
            "query_film_raw_delta_beta_abs_max_mean": "query_film/raw_delta_beta_abs_max",
            "query_film_bounded_delta_gamma_abs_max_mean": "query_film/bounded_delta_gamma_abs_max",
            "query_film_bounded_delta_beta_abs_max_mean": "query_film/bounded_delta_beta_abs_max",
        }
        for head in ("xyz", "rpy", "gripper"):
            optional_metric_names.update(
                {
                    f"intent_{head}_loss": f"intent/{head}_ce",
                    f"intent_{head}_top1_accuracy": f"intent/{head}_top1_accuracy",
                    f"intent_{head}_top5_accuracy": f"intent/{head}_top5_accuracy",
                    f"intent_{head}_probability_entropy": f"intent/{head}_probability_entropy",
                    f"intent_{head}_max_probability": f"intent/{head}_max_probability",
                    f"intent_{head}_feature_l2_mean": f"intent/{head}_feature_l2_mean",
                    f"intent_{head}_layer_attention_entropy": f"intent/{head}_layer_attention_entropy",
                }
            )
        for output_name, metric_name in optional_metric_names.items():
            value = output_dict.get(output_name)
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                metrics[metric_name] = self._metric_scalar(
                    value, aggregate=will_log
                )
        for output_name, value in output_dict.items():
            if output_name.startswith("intent_layer_attention_weight_"):
                layer = output_name.rsplit("_", 1)[-1]
                metrics[f"intent/layer_attention_weight_{layer}"] = self._metric_scalar(
                    value, aggregate=will_log
                )
            elif output_name.startswith("intent_token_attention_entropy_"):
                layer = output_name.rsplit("_", 1)[-1]
                metrics[f"intent/token_attention_entropy_{layer}"] = self._metric_scalar(
                    value, aggregate=will_log
                )
            elif (
                output_name.startswith("intent_xyz_layer_attention_weight_")
                or output_name.startswith("intent_rpy_layer_attention_weight_")
                or output_name.startswith("intent_gripper_layer_attention_weight_")
            ):
                prefix, layer = output_name.rsplit("_", 1)
                head = prefix.split("_")[1]
                metrics[f"intent/{head}_layer_attention_weight_{layer}"] = (
                    self._metric_scalar(value, aggregate=will_log)
                )
        return metrics

    def _finalize_training(self):
        """Training end processing."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, os.path.join(final_checkpoint, "model.safetensors"))
            elif save_format == "pt":
                torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        if self.accelerator.is_main_process and getattr(self, "_wandb_enabled", False):
            try:
                wandb.finish()
            except Exception:
                pass

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")

    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    output_dir = setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    vla_train_dataloader, vla_validation_dataloader = prepare_data(
        cfg=cfg, accelerator=accelerator, output_dir=output_dir
    )
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        vla_validation_dataloader=vla_validation_dataloader,
    )

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # Normalise legacy YAML keys into the current `version_id == "0.21"` schema.
    # This is idempotent and does not modify framework class signatures.
    # See bar/config_收紧.md for the rationale.
    cfg = apply_config_compat(cfg)

    # Store source config path for later copying to output dir
    cfg.config_yaml = args.config_yaml

    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
