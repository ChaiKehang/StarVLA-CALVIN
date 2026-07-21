# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import (
    SinusoidalPositionalEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from torch import nn


class TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim, compute_dtype=torch.float32):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps):
        dtype = next(self.parameters()).dtype
        timesteps_proj = self.time_proj(timesteps).to(dtype)
        timesteps_emb = self.timestep_embedder(timesteps_proj)  # (N, D)
        return timesteps_emb


class AdaLayerNorm(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ):
        super().__init__()
        self.chunk_dim = chunk_dim
        output_dim = embedding_dim * 2
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def forward(
        self,
        x: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        temb = self.linear(self.silu(temb))
        scale, shift = temb.chunk(2, dim=1)
        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.dropout = dropout
        self.cross_attention_dim = cross_attention_dim
        self.activation_fn = activation_fn
        self.attention_bias = attention_bias
        self.norm_elementwise_affine = norm_elementwise_affine
        self.positional_embeddings = positional_embeddings
        self.num_positional_embeddings = num_positional_embeddings
        self.norm_type = norm_type

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(dim, max_seq_length=num_positional_embeddings)
        else:
            self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
        self.ffn_intent_film: Optional[nn.Linear] = None
        self.cross_attn_query_intent_film: Optional[nn.Linear] = None
        self._last_intent_film_diagnostics = {}
        if final_dropout:
            self.final_dropout = nn.Dropout(dropout)
        else:
            self.final_dropout = None

    def enable_ffn_intent_film(self, num_intent_classes: int) -> None:
        """Attach the E1-C per-block FiLM projection without replacing FFN weights."""

        if num_intent_classes <= 1:
            raise ValueError("num_intent_classes must be greater than one")
        if self.ffn_intent_film is not None:
            if self.ffn_intent_film.in_features != num_intent_classes:
                raise ValueError(
                    "FFN Intent-FiLM is already configured for "
                    f"{self.ffn_intent_film.in_features} classes"
                )
            return

        self.ffn_intent_film = nn.Linear(num_intent_classes, 2 * self.dim, bias=False)
        nn.init.zeros_(self.ffn_intent_film.weight)

    def enable_cross_attn_query_intent_film(self, num_intent_classes: int) -> None:
        """Attach zero-initialized Intent Query-FiLM to a cross-attention block."""

        if self.cross_attention_dim is None:
            raise ValueError("Query-FiLM can only be enabled on a cross-attention block")
        if num_intent_classes <= 1:
            raise ValueError("num_intent_classes must be greater than one")
        if self.cross_attn_query_intent_film is not None:
            if self.cross_attn_query_intent_film.in_features != num_intent_classes:
                raise ValueError(
                    "Cross-attention Query-FiLM is already configured for "
                    f"{self.cross_attn_query_intent_film.in_features} classes"
                )
            return
        self.cross_attn_query_intent_film = nn.Linear(
            num_intent_classes, 2 * self.dim, bias=False
        )
        nn.init.zeros_(self.cross_attn_query_intent_film.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.LongTensor] = None,
        ffn_intent_probabilities: Optional[torch.Tensor] = None,
        cross_attn_query_intent_probabilities: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        self._last_intent_film_diagnostics = {}

        # 0. Self-Attention
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, temb)
        else:
            norm_hidden_states = self.norm1(hidden_states)

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        if self.cross_attn_query_intent_film is not None:
            if cross_attn_query_intent_probabilities is None:
                raise ValueError(
                    "Cross-attention Intent Query-FiLM is enabled but probabilities were not provided"
                )
            film_parameters = self.cross_attn_query_intent_film(
                cross_attn_query_intent_probabilities.to(
                    device=self.cross_attn_query_intent_film.weight.device,
                    dtype=self.cross_attn_query_intent_film.weight.dtype,
                )
            )
            delta_gamma, delta_beta = film_parameters.chunk(2, dim=-1)
            delta_gamma = delta_gamma[:, None].to(dtype=norm_hidden_states.dtype)
            delta_beta = delta_beta[:, None].to(dtype=norm_hidden_states.dtype)
            original_query = norm_hidden_states
            norm_hidden_states = (1 + delta_gamma) * original_query + delta_beta
            eps = torch.finfo(torch.float32).eps
            self._last_intent_film_diagnostics.update(
                {
                    "query_film_delta_gamma_rms": delta_gamma.float().square().mean().sqrt().detach(),
                    "query_film_delta_beta_rms": delta_beta.float().square().mean().sqrt().detach(),
                    "query_film_modulation_to_query_l2_ratio": (
                        (norm_hidden_states - original_query).float().norm(dim=-1)
                        / original_query.float().norm(dim=-1).clamp_min(eps)
                    ).mean().detach(),
                }
            )

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,  # @JinhuiYE original attention_mask=attention_mask
        )
        if self.final_dropout:
            attn_output = self.final_dropout(attn_output)

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 4. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        if self.ffn_intent_film is not None:
            if ffn_intent_probabilities is None:
                raise ValueError(
                    "FFN Intent-FiLM is enabled but ffn_intent_probabilities was not provided"
                )
            film_parameters = self.ffn_intent_film(
                ffn_intent_probabilities.to(
                    device=self.ffn_intent_film.weight.device,
                    dtype=self.ffn_intent_film.weight.dtype,
                )
            )
            delta_gamma, delta_beta = film_parameters.chunk(2, dim=-1)
            original_ffn_input = norm_hidden_states
            norm_hidden_states = (
                (1 + delta_gamma[:, None].to(dtype=norm_hidden_states.dtype))
                * original_ffn_input
                + delta_beta[:, None].to(dtype=norm_hidden_states.dtype)
            )
            eps = torch.finfo(torch.float32).eps
            self._last_intent_film_diagnostics.update(
                {
                    "ffn_film_delta_gamma_rms": delta_gamma.float().square().mean().sqrt().detach(),
                    "ffn_film_delta_beta_rms": delta_beta.float().square().mean().sqrt().detach(),
                    "ffn_film_modulation_to_input_l2_ratio": (
                        (norm_hidden_states - original_ffn_input).float().norm(dim=-1)
                        / original_ffn_input.float().norm(dim=-1).clamp_min(eps)
                    ).mean().detach(),
                }
            )
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states


class DiT(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    # register_to_config auto-registers constructor params into config, enabling access via self.config.xxx instead of self.xxx
    @register_to_config  # Registers passed params to config. TODO: replace with our singleton pattern, implement a mergeable @merge_param_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        max_num_positional_embeddings: int = 512,
        compute_dtype=torch.float32,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = "sinusoidal",
        interleave_self_attention=False,
        use_canonical_forward: bool = True,  # False restores the legacy all-cross-attention forward (old checkpoints)
        cross_attention_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        if not use_canonical_forward:
            logging.getLogger(__name__).warning(
                "use_canonical_forward=False: running the legacy all-cross-attention DiT forward. "
                "Old checkpoints may have degraded performance due to state-conditioning issues. "
                "Please retrain with the updated forward path when possible."
            )
        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.gradient_checkpointing = False

        # Timestep encoder
        #  self.config.compute_dtype may not exist, handle it in advance
        compute_dtype = getattr(self.config, "compute_dtype", torch.float32)
        self.timestep_encoder = (
            TimestepEncoder(  # TODO BUG: self.config.compute_dtype doesn't error during training but fails at eval
                embedding_dim=self.inner_dim, compute_dtype=compute_dtype
            )
        )

        all_blocks = []
        for idx in range(self.config.num_layers):

            use_self_attn = idx % 2 == 1 and interleave_self_attention
            curr_cross_attention_dim = cross_attention_dim if not use_self_attn else None

            all_blocks += [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=self.config.max_num_positional_embeddings,
                    final_dropout=final_dropout,
                    cross_attention_dim=curr_cross_attention_dim,
                )
            ]
        self.transformer_blocks = nn.ModuleList(all_blocks)

        # Output blocks
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(self.inner_dim, self.config.output_dim)
        print(
            "Total number of DiT parameters: ",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def enable_ffn_intent_film(self, num_intent_classes: int) -> None:
        """Enable an independent zero-initialized E1-C FiLM in every block."""

        for block in self.transformer_blocks:
            block.enable_ffn_intent_film(num_intent_classes)

    def enable_cross_attn_query_intent_film(self, num_intent_classes: int) -> None:
        """Enable Query-FiLM only in blocks that consume VLM cross-attention K/V."""

        for block in self.transformer_blocks:
            if block.cross_attention_dim is not None:
                block.enable_cross_attn_query_intent_film(num_intent_classes)

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (B, T, D)
        encoder_hidden_states: torch.Tensor,  # Shape: (B, S, D) or list of layer-wise tensors
        timestep: Optional[torch.LongTensor] = None,
        return_all_hidden_states: bool = False,
        encoder_attention_mask=None,
        return_pre_output: bool = False,
        intent_condition: Optional[torch.Tensor] = None,
        ffn_intent_probabilities: Optional[torch.Tensor] = None,
        cross_attn_query_intent_probabilities: Optional[torch.Tensor] = None,
        return_condition_diagnostics: bool = False,
    ):
        # Encode timesteps
        temb = self.timestep_encoder(timestep)
        condition_diagnostics = {}
        if intent_condition is not None:
            if intent_condition.ndim != 2:
                raise ValueError(
                    "intent_condition must have shape [B,D], "
                    f"got {tuple(intent_condition.shape)}"
                )
            if intent_condition.shape != temb.shape:
                raise ValueError(
                    "intent_condition shape must match timestep embedding: "
                    f"got {tuple(intent_condition.shape)} and {tuple(temb.shape)}"
                )
            aligned_intent_condition = intent_condition.to(device=temb.device, dtype=temb.dtype)
            # Measure the actual per-sample scale seen by AdaLN.  Computing the
            # ratio before addition distinguishes a weak auxiliary signal from
            # one large enough to dominate the pretrained timestep embedding.
            if return_condition_diagnostics:
                timestep_l2 = temb.float().norm(dim=-1)
                intent_l2 = aligned_intent_condition.float().norm(dim=-1)
                joint_l2 = (temb + aligned_intent_condition).float().norm(dim=-1)
                condition_diagnostics = {
                    "timestep_embedding_l2_mean": timestep_l2.mean().detach(),
                    "joint_timestep_condition_l2_mean": joint_l2.mean().detach(),
                    "intent_condition_to_timestep_l2_ratio": (
                        intent_l2 / timestep_l2.clamp_min(torch.finfo(torch.float32).eps)
                    ).mean().detach(),
                }

            # The E1-B condition joins the existing timestep stream before any
            # transformer block, so every actually-executed AdaLN sees it.
            temb = temb + aligned_intent_condition

        if ffn_intent_probabilities is not None:
            if ffn_intent_probabilities.ndim != 2:
                raise ValueError(
                    "ffn_intent_probabilities must have shape [B,C], "
                    f"got {tuple(ffn_intent_probabilities.shape)}"
                )
            if ffn_intent_probabilities.shape[0] != hidden_states.shape[0]:
                raise ValueError(
                    "FFN intent batch must match hidden_states batch: "
                    f"got {ffn_intent_probabilities.shape[0]} and {hidden_states.shape[0]}"
                )
        if cross_attn_query_intent_probabilities is not None:
            if cross_attn_query_intent_probabilities.ndim != 2:
                raise ValueError(
                    "cross_attn_query_intent_probabilities must have shape [B,C], "
                    f"got {tuple(cross_attn_query_intent_probabilities.shape)}"
                )
            if cross_attn_query_intent_probabilities.shape[0] != hidden_states.shape[0]:
                raise ValueError(
                    "Query-FiLM intent batch must match hidden_states batch: "
                    f"got {cross_attn_query_intent_probabilities.shape[0]} and {hidden_states.shape[0]}"
                )

        # Process through transformer blocks - single pass through the blocks
        hidden_states = hidden_states.contiguous()
        is_layerwise_encoder = isinstance(encoder_hidden_states, (list, tuple))
        encoder_hidden_states = (
            [state.contiguous() for state in encoder_hidden_states]
            if is_layerwise_encoder
            else encoder_hidden_states.contiguous()
        )

        all_hidden_states = [hidden_states]
        film_diagnostics = {}

        # Process through transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1 and self.config.interleave_self_attention and self.config.use_canonical_forward:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                    ffn_intent_probabilities=ffn_intent_probabilities,
                    cross_attn_query_intent_probabilities=cross_attn_query_intent_probabilities,
                )
            else:
                if is_layerwise_encoder:
                    block_encoder_hidden_states = encoder_hidden_states[idx]
                else:
                    block_encoder_hidden_states = encoder_hidden_states
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=block_encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    temb=temb,
                    ffn_intent_probabilities=ffn_intent_probabilities,
                    cross_attn_query_intent_probabilities=cross_attn_query_intent_probabilities,
                )
            if return_condition_diagnostics:
                for name, value in block._last_intent_film_diagnostics.items():
                    film_diagnostics.setdefault(name, []).append(value)
            all_hidden_states.append(hidden_states)

        if return_condition_diagnostics:
            for name, values in film_diagnostics.items():
                condition_diagnostics[f"{name}_mean"] = torch.stack(values).mean().detach()

        if return_pre_output:
            if return_all_hidden_states:
                output = (hidden_states, all_hidden_states)
            else:
                output = hidden_states
            if return_condition_diagnostics:
                return output, condition_diagnostics
            return output

        # Output processing
        conditioning = temb
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        if return_all_hidden_states:
            output = (self.proj_out_2(hidden_states), all_hidden_states)
        else:
            output = self.proj_out_2(hidden_states)
        if return_condition_diagnostics:
            return output, condition_diagnostics
        return output


class SelfAttentionTransformer(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        max_num_positional_embeddings: int = 512,
        compute_dtype=torch.float32,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = "sinusoidal",
        interleave_self_attention=False,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.gradient_checkpointing = False

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=self.config.max_num_positional_embeddings,
                    final_dropout=final_dropout,
                )
                for _ in range(self.config.num_layers)
            ]
        )
        print(
            "Total number of SelfAttentionTransformer parameters: ",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (B, T, D)
        return_all_hidden_states: bool = False,
    ):
        # Process through transformer blocks - single pass through the blocks
        hidden_states = hidden_states.contiguous()
        all_hidden_states = [hidden_states]

        # Process through transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            hidden_states = block(hidden_states)
            all_hidden_states.append(hidden_states)

        if return_all_hidden_states:
            return hidden_states, all_hidden_states
        else:
            return hidden_states
