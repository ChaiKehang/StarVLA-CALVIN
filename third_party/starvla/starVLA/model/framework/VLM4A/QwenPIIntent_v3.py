"""QwenPI_v3 with the optional E1 intent auxiliary loss and DiT condition."""

from contextlib import nullcontext
from typing import List, Optional

import numpy as np
import torch

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenPI_v3 import Qwen_PI_v3
from starVLA.model.modules.intent_head import (
    IntentClassificationHead,
    IntentHeadConfig,
    MultiLayerIntentClassificationHead,
    build_zero_initialized_intent_projection,
    compute_intent_auxiliary_loss,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@FRAMEWORK_REGISTRY.register("QwenPIIntent_v3")
class Qwen_PI_Intent_v3(Qwen_PI_v3):
    """E1-A/E1-B framework built on the unchanged QwenPI_v3 action path.

    ``framework.intent.use_aux_loss`` controls the 4.1 CE term and
    ``framework.intent.add_to_timestep_embedding`` controls whether the
    predicted soft intent distribution is projected into the DiT AdaLN
    timestep-conditioning stream. ``framework.intent.use_ffn_film`` adds the
    independent E1-C per-block FFN FiLM. Keeping these switches independent
    makes E1-A, E1-B, E1-C, and conditioning-only ablations selectable from
    YAML/CLI.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)

        intent_cfg = self.config.framework.get("intent", {})
        self.use_intent_aux_loss = bool(intent_cfg.get("use_aux_loss", True))
        self.add_intent_to_timestep_embedding = bool(
            intent_cfg.get("add_to_timestep_embedding", True)
        )
        self.use_ffn_intent_film = bool(intent_cfg.get("use_ffn_film", False))
        self.use_cross_attn_query_film = bool(
            intent_cfg.get("use_cross_attn_query_film", False)
        )
        self.use_multilayer_intent = bool(
            intent_cfg.get("use_multilayer_aggregator", False)
        )
        self.intent_action_condition_source = str(
            intent_cfg.get("action_condition_source", "probabilities")
        )
        if self.intent_action_condition_source != "probabilities":
            raise ValueError(
                "Only the controlled 125-way soft-probability Action interface is "
                "implemented; framework.intent.action_condition_source must be "
                f"'probabilities', got {self.intent_action_condition_source!r}"
            )
        self.intent_training_stage = str(intent_cfg.get("training_stage", "joint"))
        if self.intent_training_stage not in {"joint", "s0", "s1_s2"}:
            raise ValueError(
                "framework.intent.training_stage must be one of joint/s0/s1_s2, "
                f"got {self.intent_training_stage!r}"
            )
        self.intent_stage1_steps = int(intent_cfg.get("stage1_steps", 10000))
        if self.intent_stage1_steps < 0:
            raise ValueError("framework.intent.stage1_steps must be non-negative")
        self._training_step = 0
        self.intent_loss_weight = float(intent_cfg.get("loss_weight", 0.1))
        if self.intent_loss_weight < 0:
            raise ValueError("framework.intent.loss_weight must be non-negative")

        intent_hidden_size = int(intent_cfg.get("hidden_size", self.action_dit_hidden_dim))
        if intent_hidden_size != self.action_dit_hidden_dim:
            raise ValueError(
                "Intent hidden_size must match action_dit_hidden_dim: "
                f"got {intent_hidden_size} and {self.action_dit_hidden_dim}"
            )

        num_classes = int(intent_cfg.get("num_classes", 125))
        head_config = IntentHeadConfig(
            hidden_size=intent_hidden_size,
            num_attention_heads=int(intent_cfg.get("num_attention_heads", 16)),
            classifier_hidden_size=int(intent_cfg.get("classifier_hidden_size", 512)),
            num_classes=num_classes,
            dropout=float(intent_cfg.get("dropout", 0.1)),
            pooler_type=str(intent_cfg.get("pooler_type", "learned_query")),
            num_queries=1,
        )
        source_layers = tuple(
            int(layer)
            for layer in intent_cfg.get(
                "source_layers", [4, 8, 12, 16, 20, 24, 28, 32, 36]
            )
        )
        if any(layer > self.num_action_dit_layers for layer in source_layers):
            raise ValueError(
                "Intent source layer exceeds the available VLM layers: "
                f"source_layers={source_layers}, available={self.num_action_dit_layers}"
            )
        self.intent_source_layers = source_layers
        if self.use_multilayer_intent:
            vlm_hidden_size = int(self.config.framework.qwenvl.vl_hidden_dim)
            self.intent_head = MultiLayerIntentClassificationHead(
                input_hidden_size=vlm_hidden_size,
                source_layers=source_layers,
                config=head_config,
                use_layer_position_embedding=bool(
                    intent_cfg.get("use_layer_position_embedding", True)
                ),
                aggregator_block_type=str(
                    intent_cfg.get("aggregator_block_type", "legacy")
                ),
                token_query_use_ffn=bool(
                    intent_cfg.get("token_query_use_ffn", True)
                ),
                layer_query_use_ffn=bool(
                    intent_cfg.get("layer_query_use_ffn", True)
                ),
                query_ffn_multiplier=float(
                    intent_cfg.get("query_ffn_multiplier", 2.0)
                ),
                query_ffn_dropout=float(
                    intent_cfg.get("query_ffn_dropout", 0.1)
                ),
                query_attention_dropout=float(
                    intent_cfg.get("query_attention_dropout", 0.0)
                ),
                query_norm_eps=float(intent_cfg.get("query_norm_eps", 1.0e-5)),
            )
        else:
            self.intent_head = IntentClassificationHead(head_config)
        self.intent_to_timestep = build_zero_initialized_intent_projection(
            num_classes, self.action_dit_hidden_dim
        )
        self.intent_num_classes = num_classes
        if self.use_ffn_intent_film:
            self.action_model.model.enable_ffn_intent_film(num_classes)
        if self.use_cross_attn_query_film:
            self.action_model.model.enable_cross_attn_query_intent_film(num_classes)

    @property
    def intent_project_layers(self):
        """Expose dedicated Intent projectors without registering a second alias."""

        return getattr(self.intent_head, "project_layers", None)

    def initialize_intent_projectors_from_action(self) -> None:
        """Copy matching Action-projector weights into independent Intent projectors."""

        if not self.use_multilayer_intent:
            return
        for intent_index, source_layer in enumerate(self.intent_source_layers):
            action_projector = self.project_layers[source_layer - 1]
            intent_projector = self.intent_head.project_layers[intent_index]
            try:
                intent_projector.load_state_dict(action_projector.state_dict(), strict=True)
            except RuntimeError as exc:
                raise RuntimeError(
                    "Cannot initialize Intent projector from Action projector for "
                    f"1-based VLM layer {source_layer}: {exc}"
                ) from exc
            action_parameters = list(action_projector.parameters())
            intent_parameters = list(intent_projector.parameters())
            if len(action_parameters) != len(intent_parameters):
                raise RuntimeError("Action and Intent projector parameter layouts differ")
            if any(a.data_ptr() == b.data_ptr() for a, b in zip(action_parameters, intent_parameters)):
                raise RuntimeError("Intent and Action projectors must not share parameter storage")

    def set_training_step(self, step: int) -> None:
        """Receive the main-run optimizer step from the trainer."""

        self._training_step = int(step)

    def _current_intent_stage(self) -> str:
        if self.intent_training_stage == "s0":
            return "s0"
        if self.intent_training_stage == "s1_s2":
            return "s1" if self._training_step < self.intent_stage1_steps else "s2"
        return "joint"

    def _valid_attention_mask(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if attention_mask is None:
            return torch.ones(
                hidden_states.shape[:2],
                device=hidden_states.device,
                dtype=torch.bool,
            )
        return attention_mask.to(device=hidden_states.device, dtype=torch.bool)

    def _predict_intent(
        self,
        intent_hidden_states,
        attention_mask: Optional[torch.Tensor],
        *,
        return_attention_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        reference_hidden = (
            intent_hidden_states[0]
            if isinstance(intent_hidden_states, (list, tuple))
            else intent_hidden_states
        )
        mask = self._valid_attention_mask(reference_hidden, attention_mask)
        # Training already has an outer bf16 autocast context; inference does
        # not.  Entering it here on CUDA keeps projected bf16 VLM features and
        # fp32 module weights compatible in both paths.
        autocast_context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if reference_hidden.device.type == "cuda"
            else nullcontext()
        )
        with autocast_context:
            intent_output = self.intent_head(
                intent_hidden_states,
                mask,
                return_attention_weights=return_attention_weights,
            )
        # Compute softmax in fp32 for numerical stability, then cast to the
        # projection weight dtype without detaching the intent-head graph.
        probabilities = torch.softmax(intent_output.logits.float(), dim=-1).to(
            dtype=self.intent_to_timestep.weight.dtype
        )
        diagnostics = {
            "intent_global_feature_l2_mean": intent_output.pooled_features.float()
            .norm(dim=-1)
            .mean()
            .detach()
        }
        scalar_diagnostics = {
            "intent_token_attention_residual_ratio": (
                intent_output.token_attention_residual_ratio
            ),
            "intent_token_ffn_residual_ratio": intent_output.token_ffn_residual_ratio,
            "intent_layer_attention_residual_ratio": (
                intent_output.layer_attention_residual_ratio
            ),
            "intent_layer_ffn_residual_ratio": intent_output.layer_ffn_residual_ratio,
            "intent_token_summary_pre_norm_l2_mean": (
                intent_output.token_summary_pre_norm_l2_mean
            ),
            "intent_token_summary_post_norm_l2_mean": (
                intent_output.token_summary_post_norm_l2_mean
            ),
            "intent_global_feature_pre_norm_l2_mean": (
                intent_output.global_feature_pre_norm_l2_mean
            ),
            "intent_global_feature_post_norm_l2_mean": (
                intent_output.global_feature_post_norm_l2_mean
            ),
        }
        diagnostics.update(
            {
                name: value.detach()
                for name, value in scalar_diagnostics.items()
                if value is not None
            }
        )
        if intent_output.layer_attention_weights is not None:
            layer_weights = intent_output.layer_attention_weights.float()
            safe_weights = layer_weights.clamp_min(torch.finfo(torch.float32).tiny)
            diagnostics["intent_layer_attention_entropy"] = (
                -(layer_weights * safe_weights.log()).sum(dim=-1).mean().detach()
            )
            for index, source_layer in enumerate(self.intent_source_layers):
                diagnostics[f"intent_layer_attention_weight_{source_layer}"] = (
                    layer_weights[:, index].mean().detach()
                )
        if intent_output.token_attention_weights is not None:
            token_weights = intent_output.token_attention_weights.float()
            safe_weights = token_weights.clamp_min(torch.finfo(torch.float32).tiny)
            token_entropy = -(token_weights * safe_weights.log()).sum(dim=-1).mean(dim=0)
            for index, source_layer in enumerate(self.intent_source_layers):
                diagnostics[f"intent_token_attention_entropy_{source_layer}"] = (
                    token_entropy[index].detach()
                )
        return intent_output.logits, probabilities, diagnostics

    def _select_intent_hidden_states(
        self,
        raw_vl_embs_list: List[torch.Tensor],
        projected_vl_embs_list: Optional[List[torch.Tensor]],
    ):
        if self.use_multilayer_intent:
            return [raw_vl_embs_list[layer - 1] for layer in self.intent_source_layers]
        if projected_vl_embs_list is None:
            raise ValueError("The legacy Intent head requires projected VLM hidden states")
        return projected_vl_embs_list[-1]

    def _intent_targets(self, examples: List[dict], device: torch.device) -> torch.Tensor:
        missing = [index for index, example in enumerate(examples) if "intent_class_id" not in example]
        if missing:
            raise KeyError(
                "framework.intent.use_aux_loss=True requires intent_class_id in every sample; "
                f"missing at batch indices {missing[:8]}"
            )
        targets = torch.as_tensor(
            [int(example["intent_class_id"]) for example in examples],
            device=device,
            dtype=torch.long,
        )
        if torch.any((targets < 0) | (targets >= self.intent_num_classes)):
            bad = targets[(targets < 0) | (targets >= self.intent_num_classes)][0].item()
            raise ValueError(
                f"intent_class_id must be in [0, {self.intent_num_classes - 1}], got {bad}"
            )
        return targets

    def _repeat_count(self) -> int:
        trainer_cfg = self.config.trainer if self.config and self.config.trainer else None
        if trainer_cfg is not None and trainer_cfg.get("repeated_diffusion_steps", None) is not None:
            repeat_count = int(trainer_cfg.repeated_diffusion_steps)
        else:
            repeat_count = int(self.config.framework.action_model.get("repeated_diffusion_steps", 4))
        if repeat_count <= 0:
            raise ValueError("repeated_diffusion_steps must be positive")
        return repeat_count

    @staticmethod
    def _intent_probability_metrics(probabilities: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return detached, low-cost diagnostics for the predicted intent distribution."""

        probabilities_fp32 = probabilities.float()
        safe_probabilities = probabilities_fp32.clamp_min(torch.finfo(torch.float32).tiny)
        entropy = -(probabilities_fp32 * safe_probabilities.log()).sum(dim=-1).mean()
        return {
            "intent_probability_entropy": entropy.detach(),
            "intent_max_probability": probabilities_fp32.max(dim=-1).values.mean().detach(),
        }

    @staticmethod
    def _decode_intent_classes(class_ids: torch.Tensor) -> torch.Tensor:
        """Decode ``25*bx + 5*by + bz`` class IDs into XYZ bins."""

        class_ids = class_ids.to(dtype=torch.long)
        return torch.stack(
            (
                class_ids // 25,
                (class_ids % 25) // 5,
                class_ids % 5,
            ),
            dim=-1,
        )

    def _format_intent_predictions(
        self, probabilities: torch.Tensor
    ) -> dict[str, np.ndarray]:
        """Create serializable per-sample Intent outputs for evaluation."""

        probabilities_fp32 = probabilities.float()
        safe_probabilities = probabilities_fp32.clamp_min(
            torch.finfo(torch.float32).tiny
        )
        top_k = min(5, probabilities_fp32.shape[-1])
        top_probabilities, top_class_ids = probabilities_fp32.topk(top_k, dim=-1)
        predicted_class_ids = top_class_ids[:, 0]
        return {
            "probabilities": probabilities_fp32.detach().cpu().numpy(),
            "predicted_class_id": predicted_class_ids.detach().cpu().numpy(),
            "predicted_axis_bins": self._decode_intent_classes(
                predicted_class_ids
            ).detach().cpu().numpy(),
            "top5_class_ids": top_class_ids.detach().cpu().numpy(),
            "top5_probabilities": top_probabilities.detach().cpu().numpy(),
            "entropy": (
                -(probabilities_fp32 * safe_probabilities.log()).sum(dim=-1)
            ).detach().cpu().numpy(),
            "max_probability": top_probabilities[:, 0].detach().cpu().numpy(),
        }

    def _run_action_loss_with_fixed_rng(
        self,
        *,
        vl_embs_list: List[torch.Tensor],
        actions_target: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        intent_probabilities: torch.Tensor,
        probe_seed: int,
    ) -> torch.Tensor:
        """Evaluate one condition with reproducible flow noise, timestep, and dropout RNG."""

        intent_condition = (
            self.intent_to_timestep(intent_probabilities)
            if self.add_intent_to_timestep_embedding
            else None
        )
        ffn_intent_probabilities = (
            intent_probabilities if self.use_ffn_intent_film else None
        )
        cross_attn_query_intent_probabilities = (
            intent_probabilities if self.use_cross_attn_query_film else None
        )
        cuda_devices = []
        if actions_target.device.type == "cuda":
            cuda_devices = [
                actions_target.device.index
                if actions_target.device.index is not None
                else torch.cuda.current_device()
            ]
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(probe_seed)
            if actions_target.device.type == "cuda":
                torch.cuda.manual_seed(probe_seed)
            return self.action_model(
                vl_embs_list,
                actions_target,
                None,
                encoder_attention_mask=attention_mask,
                intent_condition=intent_condition,
                ffn_intent_probabilities=ffn_intent_probabilities,
                cross_attn_query_intent_probabilities=cross_attn_query_intent_probabilities,
            )

    @torch.inference_mode()
    def diagnose_intent_condition(
        self,
        examples: List[dict],
        probe_seed: int = 42,
    ) -> dict[str, torch.Tensor]:
        """Compare true, zero, and batch-shuffled intent under identical flow RNG.

        Positive ``gain_vs_zero`` / ``gain_vs_shuffle`` means that the correctly
        aligned condition produces a smaller action loss than the corresponding
        ablation on this probe batch.  This is a diagnostic, not a training loss.
        """

        if not (
            self.add_intent_to_timestep_embedding
            or self.use_ffn_intent_film
            or self.use_cross_attn_query_film
        ):
            return {}

        was_training = self.training
        self.eval()
        try:
            batch_images = [example["image"] for example in examples]
            instructions = [example["lang"] for example in examples]
            actions = [example["action"] for example in examples]
            state = [example["state"] for example in examples] if "state" in examples[0] else None
            instructions = (
                self.add_discretized_state_to_instruction(instructions, state)
                if state is not None
                else instructions
            )

            vl_embs_list, attention_mask, raw_vl_embs_list = self._encode_vl_hidden_states(
                batch_images, instructions, return_unprojected=True
            )
            base_hidden = vl_embs_list[-1]
            if attention_mask is not None:
                attention_mask = attention_mask.to(dtype=torch.bool)
            intent_hidden_states = self._select_intent_hidden_states(
                raw_vl_embs_list, vl_embs_list
            )
            _, probabilities, _ = self._predict_intent(
                intent_hidden_states, attention_mask
            )
            actions_target = torch.as_tensor(
                np.array(actions),
                device=base_hidden.device,
                dtype=base_hidden.dtype,
            )[:, -self.action_horizon :, :]

            true_loss = self._run_action_loss_with_fixed_rng(
                vl_embs_list=vl_embs_list,
                actions_target=actions_target,
                attention_mask=attention_mask,
                intent_probabilities=probabilities,
                probe_seed=probe_seed,
            )
            zero_loss = self._run_action_loss_with_fixed_rng(
                vl_embs_list=vl_embs_list,
                actions_target=actions_target,
                attention_mask=attention_mask,
                intent_probabilities=torch.zeros_like(probabilities),
                probe_seed=probe_seed,
            )
            output = {
                "intent_probe/loss_true": true_loss.detach(),
                "intent_probe/loss_zero": zero_loss.detach(),
                "intent_probe/gain_vs_zero": (zero_loss - true_loss).detach(),
            }
            if probabilities.shape[0] > 1:
                shuffled_probabilities = probabilities.roll(shifts=1, dims=0)
                shuffled_loss = self._run_action_loss_with_fixed_rng(
                    vl_embs_list=vl_embs_list,
                    actions_target=actions_target,
                    attention_mask=attention_mask,
                    intent_probabilities=shuffled_probabilities,
                    probe_seed=probe_seed,
                )
                output.update(
                    {
                        "intent_probe/loss_shuffled": shuffled_loss.detach(),
                        "intent_probe/gain_vs_shuffle": (shuffled_loss - true_loss).detach(),
                    }
                )
            return output
        finally:
            self.train(was_training)

    def forward(self, examples: List[dict] = None, **kwargs) -> dict[str, torch.Tensor]:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        instructions = (
            self.add_discretized_state_to_instruction(instructions, state)
            if state is not None
            else instructions
        )
        state = None

        current_stage = self._current_intent_stage()
        project_for_action = current_stage != "s0"
        vl_embs_list, backbone_attention_mask, raw_vl_embs_list = self._encode_vl_hidden_states(
            batch_images,
            instructions,
            return_unprojected=True,
            project_for_action=project_for_action,
        )
        base_hidden = (
            vl_embs_list[-1] if vl_embs_list is not None else raw_vl_embs_list[-1]
        )
        intent_hidden_states = self._select_intent_hidden_states(
            raw_vl_embs_list, vl_embs_list
        )

        need_intent = (
            self.use_intent_aux_loss
            or self.add_intent_to_timestep_embedding
            or self.use_ffn_intent_film
            or self.use_cross_attn_query_film
            or current_stage == "s0"
        )
        intent_logits = None
        intent_probabilities = None
        intent_condition = None
        intent_diagnostics = {}
        if need_intent:
            attention_diagnostics_interval = int(
                self.config.framework.intent.get("attention_diagnostics_interval", 100)
            )
            if attention_diagnostics_interval <= 0:
                raise ValueError("attention_diagnostics_interval must be positive")
            return_attention_weights = (
                bool(
                    self.config.framework.intent.get(
                        "return_attention_diagnostics", False
                    )
                )
                # Metrics from optimizer step N are logged after the forward
                # that starts with completed_steps=N-1.
                and (self._training_step + 1) % attention_diagnostics_interval == 0
            )
            if current_stage == "s1":
                self.intent_head.eval()
                with torch.no_grad():
                    intent_logits, intent_probabilities, intent_diagnostics = self._predict_intent(
                        intent_hidden_states,
                        backbone_attention_mask,
                        return_attention_weights=return_attention_weights,
                    )
            else:
                self.intent_head.train(self.training)
                intent_logits, intent_probabilities, intent_diagnostics = self._predict_intent(
                    intent_hidden_states,
                    backbone_attention_mask,
                    return_attention_weights=return_attention_weights,
                )
            if self.add_intent_to_timestep_embedding:
                intent_condition = self.intent_to_timestep(intent_probabilities)

        if current_stage == "s0":
            targets = self._intent_targets(examples, base_hidden.device)
            intent_loss_output = compute_intent_auxiliary_loss(
                intent_logits, targets, loss_weight=1.0
            )
            zero_action_loss = intent_logits.sum() * 0.0
            output = {
                "action_loss": zero_action_loss,
                "total_loss": intent_loss_output.loss,
                "intent_loss": intent_loss_output.loss,
                "weighted_intent_loss": intent_loss_output.loss,
                "intent_top1_accuracy": intent_loss_output.top1_accuracy,
                "intent_top5_accuracy": (
                    intent_logits.topk(k=min(5, self.intent_num_classes), dim=-1).indices
                    == targets[:, None]
                ).any(dim=-1).float().mean().detach(),
                "intent_training_stage_id": torch.zeros(
                    (), device=base_hidden.device, dtype=torch.float32
                ),
            }
            output.update(self._intent_probability_metrics(intent_probabilities))
            output.update(intent_diagnostics)
            return output

        actions = [example["action"] for example in examples]

        with torch.autocast("cuda", dtype=torch.float32):
            actions_tensor = torch.as_tensor(
                np.array(actions),
                device=base_hidden.device,
                dtype=base_hidden.dtype,
            )
            actions_target = actions_tensor[:, -self.action_horizon :, :]
            repeat_count = self._repeat_count()

            actions_target_repeated = actions_target.repeat(repeat_count, 1, 1)
            vl_embs_list_repeated = [h.repeat(repeat_count, 1, 1) for h in vl_embs_list]
            repeated_attention_mask = None
            if backbone_attention_mask is not None:
                repeated_attention_mask = backbone_attention_mask.repeat(repeat_count, 1).to(
                    dtype=torch.bool
                )
            repeated_intent_condition = (
                intent_condition.repeat(repeat_count, 1)
                if intent_condition is not None
                else None
            )
            repeated_ffn_intent_probabilities = (
                intent_probabilities.repeat(repeat_count, 1)
                if self.use_ffn_intent_film
                else None
            )
            repeated_cross_attn_query_intent_probabilities = (
                intent_probabilities.repeat(repeat_count, 1)
                if self.use_cross_attn_query_film
                else None
            )

            action_output = self.action_model(
                vl_embs_list_repeated,
                actions_target_repeated,
                None,
                encoder_attention_mask=repeated_attention_mask,
                intent_condition=repeated_intent_condition,
                ffn_intent_probabilities=repeated_ffn_intent_probabilities,
                cross_attn_query_intent_probabilities=(
                    repeated_cross_attn_query_intent_probabilities
                ),
                return_condition_diagnostics=bool(
                    repeated_intent_condition is not None
                    or repeated_ffn_intent_probabilities is not None
                    or repeated_cross_attn_query_intent_probabilities is not None
                ),
            )
            condition_diagnostics = {}
            if (
                repeated_intent_condition is not None
                or repeated_ffn_intent_probabilities is not None
                or repeated_cross_attn_query_intent_probabilities is not None
            ):
                action_loss, condition_diagnostics = action_output
            else:
                action_loss = action_output

        output = {
            "action_loss": action_loss,
            "total_loss": action_loss,
        }
        output.update(condition_diagnostics)
        output.update(intent_diagnostics)
        stage_id = 1.0 if current_stage == "s1" else 2.0
        output["intent_training_stage_id"] = torch.tensor(
            stage_id, device=base_hidden.device, dtype=torch.float32
        )
        if intent_probabilities is not None:
            output.update(self._intent_probability_metrics(intent_probabilities))
        if intent_condition is not None:
            output.update(
                {
                    "intent_condition_l2_mean": intent_condition.float().norm(dim=-1).mean().detach(),
                    "intent_condition_abs_mean": intent_condition.float().abs().mean().detach(),
                    "intent_to_timestep_weight_norm": self.intent_to_timestep.weight.float().norm().detach(),
                }
            )
        if self.use_intent_aux_loss and intent_logits is not None:
            targets = self._intent_targets(examples, base_hidden.device)
            intent_loss_output = compute_intent_auxiliary_loss(
                intent_logits,
                targets,
                self.intent_loss_weight,
            )
            weighted_loss = (
                torch.zeros_like(intent_loss_output.weighted_loss)
                if current_stage == "s1"
                else intent_loss_output.weighted_loss
            )
            output.update(
                {
                    "intent_loss": intent_loss_output.loss,
                    "weighted_intent_loss": weighted_loss,
                    "total_loss": action_loss + weighted_loss,
                    "intent_top1_accuracy": intent_loss_output.top1_accuracy,
                    "intent_top5_accuracy": (
                        intent_logits.topk(k=min(5, self.intent_num_classes), dim=-1).indices
                        == targets[:, None]
                    ).any(dim=-1).float().mean().detach(),
                }
            )
        return output

    @torch.inference_mode()
    def predict_intent(
        self, examples: List[dict] = None, **kwargs: str
    ) -> dict[str, np.ndarray]:
        """Predict the 125-way spatial Intent without running the Action DiT."""

        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        instructions = (
            self.add_discretized_state_to_instruction(instructions, state)
            if state is not None
            else instructions
        )

        train_obs_image_size = getattr(
            self.config.datasets.vla_data, "obs_image_size", None
        )
        if train_obs_image_size:
            batch_images = resize_images(
                batch_images, target_size=train_obs_image_size
            )

        projected, attention_mask, raw_hidden_states = self._encode_vl_hidden_states(
            batch_images,
            instructions,
            return_unprojected=True,
            project_for_action=not self.use_multilayer_intent,
        )
        intent_hidden_states = self._select_intent_hidden_states(
            raw_hidden_states, projected
        )
        _, probabilities, _ = self._predict_intent(
            intent_hidden_states, attention_mask
        )
        return self._format_intent_predictions(probabilities)

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs: str) -> dict[str, np.ndarray]:
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        instructions = (
            self.add_discretized_state_to_instruction(instructions, state)
            if state is not None
            else instructions
        )
        state = None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        vl_embs_list, backbone_attention_mask, raw_vl_embs_list = self._encode_vl_hidden_states(
            batch_images, instructions, return_unprojected=True
        )
        base_hidden = vl_embs_list[-1]
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)

        intent_condition = None
        intent_probabilities = None
        disable_intent_conditioning = bool(
            kwargs.get("disable_intent_conditioning", False)
        )
        if (
            self.add_intent_to_timestep_embedding
            or self.use_ffn_intent_film
            or self.use_cross_attn_query_film
        ):
            intent_hidden_states = self._select_intent_hidden_states(
                raw_vl_embs_list, vl_embs_list
            )
            _, intent_probabilities, _ = self._predict_intent(
                intent_hidden_states, backbone_attention_mask
            )
        if self.add_intent_to_timestep_embedding and not disable_intent_conditioning:
            intent_condition = self.intent_to_timestep(intent_probabilities)

        inference_seed = kwargs.get("inference_seed", None)
        cuda_devices = []
        if base_hidden.device.type == "cuda":
            cuda_devices = [
                base_hidden.device.index
                if base_hidden.device.index is not None
                else torch.cuda.current_device()
            ]
        rng_context = (
            torch.random.fork_rng(devices=cuda_devices)
            if inference_seed is not None
            else nullcontext()
        )
        with rng_context:
            if inference_seed is not None:
                torch.manual_seed(int(inference_seed))
                if base_hidden.device.type == "cuda":
                    torch.cuda.manual_seed(int(inference_seed))
            with torch.autocast("cuda", dtype=torch.float32):
                pred_actions = self.action_model.predict_action(
                    vl_embs_list,
                    None,
                    encoder_attention_mask=backbone_attention_mask,
                    intent_condition=intent_condition,
                    ffn_intent_probabilities=(
                        intent_probabilities
                        if self.use_ffn_intent_film and not disable_intent_conditioning
                        else None
                    ),
                    cross_attn_query_intent_probabilities=(
                        intent_probabilities
                        if self.use_cross_attn_query_film and not disable_intent_conditioning
                        else None
                    ),
                )

        output = {"normalized_actions": pred_actions.detach().cpu().numpy()}
        if intent_probabilities is not None:
            intent_predictions = self._format_intent_predictions(
                intent_probabilities
            )
            intent_predictions["conditioning_applied"] = np.full(
                intent_probabilities.shape[0],
                not disable_intent_conditioning,
                dtype=np.bool_,
            )
            output["intent_predictions"] = intent_predictions
        return output
