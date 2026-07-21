"""Intent pooling and classification modules for the E1 CALVIN experiment.

This module intentionally does not depend on a particular StarVLA framework.
It consumes the last projected VLM hidden state ``[B, L, D]`` and its token
attention mask.  Framework/loss integration is left to the E1-A/E1-B stage so
that importing this module cannot change the existing E0 action path.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class IntentHeadConfig:
    """Configuration shared by the intent-pooling comparison.

    The defaults are the fixed E1 configuration.  Smaller dimensions remain
    configurable for CPU tests and future controlled ablations.
    """

    hidden_size: int = 1024
    num_attention_heads: int = 16
    classifier_hidden_size: int = 512
    num_classes: int = 125
    dropout: float = 0.1
    pooler_type: str = "learned_query"
    num_queries: int = 1

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads: "
                f"got {self.hidden_size} and {self.num_attention_heads}"
            )
        if self.classifier_hidden_size <= 0:
            raise ValueError("classifier_hidden_size must be positive")
        if self.num_classes <= 1:
            raise ValueError("num_classes must be greater than one")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.num_queries != 1:
            raise ValueError("E1 predicts one global intent, so num_queries must be 1")
        if self.pooler_type not in {"learned_query", "last_token", "masked_mean"}:
            raise ValueError(f"Unsupported pooler_type: {self.pooler_type}")

    @property
    def attention_head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


@dataclass
class IntentHeadOutput:
    """Outputs needed by intent loss, diagnostics, and later conditioning."""

    logits: torch.Tensor
    pooled_features: torch.Tensor
    attention_weights: Optional[torch.Tensor] = None
    token_attention_weights: Optional[torch.Tensor] = None
    layer_attention_weights: Optional[torch.Tensor] = None
    token_attention_residual_ratio: Optional[torch.Tensor] = None
    token_ffn_residual_ratio: Optional[torch.Tensor] = None
    layer_attention_residual_ratio: Optional[torch.Tensor] = None
    layer_ffn_residual_ratio: Optional[torch.Tensor] = None
    token_summary_pre_norm_l2_mean: Optional[torch.Tensor] = None
    token_summary_post_norm_l2_mean: Optional[torch.Tensor] = None
    global_feature_pre_norm_l2_mean: Optional[torch.Tensor] = None
    global_feature_post_norm_l2_mean: Optional[torch.Tensor] = None


@dataclass
class QueryCrossAttentionBlockOutput:
    """Output and low-cost diagnostics for one learned-query block."""

    hidden_states: torch.Tensor
    attention_weights: Optional[torch.Tensor]
    attention_residual_ratio: torch.Tensor
    ffn_residual_ratio: Optional[torch.Tensor]
    pre_norm_l2_mean: torch.Tensor
    post_norm_l2_mean: torch.Tensor


@dataclass
class IntentAuxiliaryLossOutput:
    """Unweighted/weighted E1 intent loss and its top-1 diagnostic."""

    loss: torch.Tensor
    weighted_loss: torch.Tensor
    top1_accuracy: torch.Tensor


def compute_intent_auxiliary_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_weight: float,
) -> IntentAuxiliaryLossOutput:
    """Compute the once-per-observation 4.1 intent CE objective."""

    if logits.ndim != 2:
        raise ValueError(f"intent logits must have shape [B,C], got {tuple(logits.shape)}")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError(
            "intent targets must have shape [B] matching logits, "
            f"got {tuple(targets.shape)} and {tuple(logits.shape)}"
        )
    if loss_weight < 0:
        raise ValueError("intent loss_weight must be non-negative")
    if targets.dtype != torch.long:
        raise ValueError(f"intent targets must use torch.long, got {targets.dtype}")
    if torch.any((targets < 0) | (targets >= logits.shape[-1])):
        raise ValueError(f"intent targets must be in [0, {logits.shape[-1] - 1}]")

    loss = F.cross_entropy(logits.float(), targets)
    return IntentAuxiliaryLossOutput(
        loss=loss,
        weighted_loss=loss * loss_weight,
        top1_accuracy=(logits.argmax(dim=-1) == targets).float().mean().detach(),
    )


def build_zero_initialized_intent_projection(
    num_classes: int,
    hidden_size: int,
) -> nn.Linear:
    """Build the E1-B ``W_intent`` projection with exact E0 initialization."""

    if num_classes <= 1 or hidden_size <= 0:
        raise ValueError("num_classes must be >1 and hidden_size must be positive")
    projection = nn.Linear(num_classes, hidden_size, bias=False)
    nn.init.zeros_(projection.weight)
    return projection


def _validate_and_convert_mask(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    expected_hidden_size: int,
) -> torch.Tensor:
    """Validate ``[B,L,D]`` features and return a boolean valid-token mask."""

    if hidden_states.ndim != 3:
        raise ValueError(f"hidden_states must have shape [B,L,D], got {tuple(hidden_states.shape)}")
    if hidden_states.shape[-1] != expected_hidden_size:
        raise ValueError(
            f"Expected hidden size {expected_hidden_size}, got {hidden_states.shape[-1]}"
        )
    if attention_mask.ndim != 2:
        raise ValueError(f"attention_mask must have shape [B,L], got {tuple(attention_mask.shape)}")
    if attention_mask.shape != hidden_states.shape[:2]:
        raise ValueError(
            "attention_mask shape must match hidden_states [B,L]: "
            f"got {tuple(attention_mask.shape)} and {tuple(hidden_states.shape[:2])}"
        )

    valid_mask = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
    if not torch.all(valid_mask.any(dim=1)):
        raise ValueError("Every sample must contain at least one valid token")
    return valid_mask


class LastValidTokenPooling(nn.Module):
    """Select the last position marked valid by the attention mask."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        valid_mask = _validate_and_convert_mask(hidden_states, attention_mask, self.hidden_size)
        positions = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        last_indices = positions.unsqueeze(0).expand_as(valid_mask).masked_fill(~valid_mask, -1).max(dim=1).values
        batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        return hidden_states[batch_indices, last_indices]


class MaskedMeanPooling(nn.Module):
    """Mean-pool only positions marked valid by the attention mask."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        valid_mask = _validate_and_convert_mask(hidden_states, attention_mask, self.hidden_size)
        weights = valid_mask.to(dtype=hidden_states.dtype).unsqueeze(-1)
        return (hidden_states * weights).sum(dim=1) / weights.sum(dim=1)


class LearnedQueryAttentionPooling(nn.Module):
    """Pool a token sequence with one trainable query and masked MHA."""

    def __init__(self, hidden_size: int = 1024, num_attention_heads: int = 16) -> None:
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads: "
                f"got {hidden_size} and {num_attention_heads}"
            )

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intent_query = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=0.0,
            batch_first=True,
        )
        nn.init.normal_(self.intent_query, mean=0.0, std=hidden_size**-0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        valid_mask = _validate_and_convert_mask(hidden_states, attention_mask, self.hidden_size)
        query = self.intent_query.expand(hidden_states.shape[0], -1, -1)

        pooled, attention_weights = self.attention(
            query=query,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=~valid_mask,
            need_weights=return_attention_weights,
            average_attn_weights=True,
        )
        pooled = pooled[:, 0, :]

        if return_attention_weights:
            return pooled, attention_weights[:, 0, :]
        return pooled


def build_intent_pooler(config: IntentHeadConfig) -> nn.Module:
    """Build one of the three pooling candidates with a shared interface."""

    if config.pooler_type == "learned_query":
        return LearnedQueryAttentionPooling(config.hidden_size, config.num_attention_heads)
    if config.pooler_type == "last_token":
        return LastValidTokenPooling(config.hidden_size)
    if config.pooler_type == "masked_mean":
        return MaskedMeanPooling(config.hidden_size)
    raise ValueError(f"Unsupported pooler_type: {config.pooler_type}")


class IntentClassificationHead(nn.Module):
    """Pool projected VLM tokens and predict one of 125 intent classes.

    Main E1 path::

        LearnedQueryAttentionPooling
        -> LayerNorm(1024)
        -> Linear(1024, 512)
        -> GELU
        -> Dropout(0.1)
        -> Linear(512, 125)

    ``hidden_states`` is deliberately used without ``detach()``.  Once this
    module is connected to QwenPIIntent_v3, intent loss can therefore update
    the upstream trainable ``project_layers`` while the separately frozen Qwen
    backbone remains unchanged.
    """

    def __init__(self, config: Optional[IntentHeadConfig] = None) -> None:
        super().__init__()
        self.config = config or IntentHeadConfig()
        self.pooler = build_intent_pooler(self.config)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.config.hidden_size),
            nn.Linear(self.config.hidden_size, self.config.classifier_hidden_size),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.classifier_hidden_size, self.config.num_classes),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> IntentHeadOutput:
        attention_weights = None
        if isinstance(self.pooler, LearnedQueryAttentionPooling):
            pooled_output = self.pooler(
                hidden_states,
                attention_mask,
                return_attention_weights=return_attention_weights,
            )
            if return_attention_weights:
                pooled_features, attention_weights = pooled_output
            else:
                pooled_features = pooled_output
        else:
            pooled_features = self.pooler(hidden_states, attention_mask)

        logits = self.classifier(pooled_features)
        return IntentHeadOutput(
            logits=logits,
            pooled_features=pooled_features,
            attention_weights=attention_weights,
        )


class LearnedQueryCrossAttentionBlock(nn.Module):
    """A pre-norm Transformer cross-attention block for one learned query.

    The query attends to an external context, then passes through attention and
    FFN residual paths. Query self-attention is intentionally omitted because
    this experiment uses exactly one query per pooling operation.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        use_ffn: bool = True,
        ffn_multiplier: float = 2.0,
        ffn_dropout: float = 0.1,
        attention_dropout: float = 0.0,
        norm_eps: float = 1.0e-5,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if num_attention_heads <= 0 or hidden_size % num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads: "
                f"got {hidden_size} and {num_attention_heads}"
            )
        if ffn_multiplier <= 0:
            raise ValueError("ffn_multiplier must be positive")
        if not 0.0 <= ffn_dropout < 1.0:
            raise ValueError("ffn_dropout must be in [0, 1)")
        if not 0.0 <= attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1)")
        if norm_eps <= 0:
            raise ValueError("norm_eps must be positive")

        self.use_ffn = bool(use_ffn)
        self.query_norm = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.context_norm = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=norm_eps)
        if self.use_ffn:
            ffn_hidden_size = max(1, int(round(hidden_size * ffn_multiplier)))
            self.ffn = nn.Sequential(
                nn.Linear(hidden_size, ffn_hidden_size),
                nn.GELU(),
                nn.Dropout(ffn_dropout),
                nn.Linear(ffn_hidden_size, hidden_size),
                nn.Dropout(ffn_dropout),
            )
        else:
            self.ffn = None
        self.output_norm = nn.LayerNorm(hidden_size, eps=norm_eps)

    @staticmethod
    def _residual_ratio(delta: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(torch.float32).eps
        return (
            delta.detach().float().norm(dim=-1)
            / reference.detach().float().norm(dim=-1).clamp_min(eps)
        ).mean()

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> QueryCrossAttentionBlockOutput:
        if query.ndim != 3 or query.shape[1] != 1:
            raise ValueError(
                "query must have shape [B,1,D] for single-query pooling, "
                f"got {tuple(query.shape)}"
            )
        if context.ndim != 3:
            raise ValueError(f"context must have shape [B,L,D], got {tuple(context.shape)}")
        if query.shape[0] != context.shape[0] or query.shape[-1] != context.shape[-1]:
            raise ValueError(
                "query and context batch/hidden dimensions must match, "
                f"got {tuple(query.shape)} and {tuple(context.shape)}"
            )

        normalized_context = self.context_norm(context)
        attention_delta, attention_weights = self.attention(
            query=self.query_norm(query),
            key=normalized_context,
            value=normalized_context,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention_weights,
            average_attn_weights=True,
        )
        attention_ratio = self._residual_ratio(attention_delta, query)
        after_attention = query + self.attention_dropout(attention_delta)

        ffn_ratio = None
        pre_norm = after_attention
        if self.ffn is not None:
            ffn_delta = self.ffn(self.ffn_norm(after_attention))
            ffn_ratio = self._residual_ratio(ffn_delta, after_attention)
            pre_norm = after_attention + ffn_delta

        hidden_states = self.output_norm(pre_norm)
        return QueryCrossAttentionBlockOutput(
            hidden_states=hidden_states,
            attention_weights=attention_weights,
            attention_residual_ratio=attention_ratio,
            ffn_residual_ratio=ffn_ratio,
            pre_norm_l2_mean=pre_norm.detach().float().norm(dim=-1).mean(),
            post_norm_l2_mean=hidden_states.detach().float().norm(dim=-1).mean(),
        )


class MultiLayerIntentClassificationHead(nn.Module):
    """Hierarchically pool selected raw VLM layers into one Intent feature.

    Each source layer has an independent ``LayerNorm + Linear`` projector into
    the Action-DiT-sized Intent space. A shared token-pooling module uses a
    distinct learned query for every source layer, then a second learned query
    attends over the layer summaries. ``legacy`` reproduces the original bare
    MHA readouts; ``query_ffn_v2`` wraps both readouts in pre-norm attention and
    FFN residual paths with an output LayerNorm. The final classifier keeps the
    existing E1 ``LN -> Linear -> GELU -> Dropout -> Linear`` layout.
    """

    def __init__(
        self,
        *,
        input_hidden_size: int,
        source_layers: Sequence[int],
        config: Optional[IntentHeadConfig] = None,
        use_layer_position_embedding: bool = True,
        aggregator_block_type: str = "legacy",
        token_query_use_ffn: bool = True,
        layer_query_use_ffn: bool = True,
        query_ffn_multiplier: float = 2.0,
        query_ffn_dropout: float = 0.1,
        query_attention_dropout: float = 0.0,
        query_norm_eps: float = 1.0e-5,
    ) -> None:
        super().__init__()
        self.config = config or IntentHeadConfig()
        self.input_hidden_size = int(input_hidden_size)
        self.source_layers = tuple(int(layer) for layer in source_layers)
        if self.input_hidden_size <= 0:
            raise ValueError("input_hidden_size must be positive")
        if not self.source_layers:
            raise ValueError("source_layers must not be empty")
        if any(layer <= 0 for layer in self.source_layers):
            raise ValueError("source_layers use 1-based positive layer numbers")
        if len(set(self.source_layers)) != len(self.source_layers):
            raise ValueError("source_layers must not contain duplicates")
        self.aggregator_block_type = str(aggregator_block_type)
        if self.aggregator_block_type not in {"legacy", "query_ffn_v2"}:
            raise ValueError(
                "aggregator_block_type must be legacy or query_ffn_v2, "
                f"got {self.aggregator_block_type!r}"
            )

        hidden_size = self.config.hidden_size
        num_layers = len(self.source_layers)
        self.project_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.input_hidden_size),
                    nn.Linear(self.input_hidden_size, hidden_size),
                )
                for _ in self.source_layers
            ]
        )
        self.token_queries = nn.Parameter(torch.empty(num_layers, 1, hidden_size))
        if self.aggregator_block_type == "legacy":
            self.token_attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=self.config.num_attention_heads,
                dropout=0.0,
                batch_first=True,
            )
            self.token_query_block = None
        else:
            self.token_attention = None
            self.token_query_block = LearnedQueryCrossAttentionBlock(
                hidden_size=hidden_size,
                num_attention_heads=self.config.num_attention_heads,
                use_ffn=token_query_use_ffn,
                ffn_multiplier=query_ffn_multiplier,
                ffn_dropout=query_ffn_dropout,
                attention_dropout=query_attention_dropout,
                norm_eps=query_norm_eps,
            )
        self.use_layer_position_embedding = bool(use_layer_position_embedding)
        if self.use_layer_position_embedding:
            self.layer_position_embedding = nn.Parameter(torch.empty(num_layers, hidden_size))
        else:
            self.register_parameter("layer_position_embedding", None)
        self.layer_query = nn.Parameter(torch.empty(1, 1, hidden_size))
        if self.aggregator_block_type == "legacy":
            self.layer_attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=self.config.num_attention_heads,
                dropout=0.0,
                batch_first=True,
            )
            self.layer_query_block = None
        else:
            self.layer_attention = None
            self.layer_query_block = LearnedQueryCrossAttentionBlock(
                hidden_size=hidden_size,
                num_attention_heads=self.config.num_attention_heads,
                use_ffn=layer_query_use_ffn,
                ffn_multiplier=query_ffn_multiplier,
                ffn_dropout=query_ffn_dropout,
                attention_dropout=query_attention_dropout,
                norm_eps=query_norm_eps,
            )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, self.config.classifier_hidden_size),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.classifier_hidden_size, self.config.num_classes),
        )

        query_std = hidden_size**-0.5
        nn.init.normal_(self.token_queries, mean=0.0, std=query_std)
        nn.init.normal_(self.layer_query, mean=0.0, std=query_std)
        if self.layer_position_embedding is not None:
            nn.init.normal_(self.layer_position_embedding, mean=0.0, std=query_std)

    def forward(
        self,
        hidden_states_by_layer: Sequence[torch.Tensor],
        attention_mask: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> IntentHeadOutput:
        if len(hidden_states_by_layer) != len(self.source_layers):
            raise ValueError(
                "Expected one hidden-state tensor per configured Intent source layer: "
                f"got {len(hidden_states_by_layer)} and {len(self.source_layers)}"
            )

        summaries = []
        token_weights = []
        token_attention_ratios = []
        token_ffn_ratios = []
        token_pre_norms = []
        token_post_norms = []
        for layer_index, (hidden_states, projector) in enumerate(
            zip(hidden_states_by_layer, self.project_layers)
        ):
            valid_mask = _validate_and_convert_mask(
                hidden_states, attention_mask, self.input_hidden_size
            )
            projected = projector(hidden_states)
            query = self.token_queries[layer_index : layer_index + 1].expand(
                projected.shape[0], -1, -1
            )
            if self.aggregator_block_type == "legacy":
                summary, weights = self.token_attention(
                    query=query,
                    key=projected,
                    value=projected,
                    key_padding_mask=~valid_mask,
                    need_weights=return_attention_weights,
                    average_attn_weights=True,
                )
                summaries.append(summary[:, 0, :])
                if return_attention_weights:
                    token_weights.append(weights[:, 0, :])
            else:
                block_output = self.token_query_block(
                    query,
                    projected,
                    key_padding_mask=~valid_mask,
                    return_attention_weights=return_attention_weights,
                )
                summaries.append(block_output.hidden_states[:, 0, :])
                token_attention_ratios.append(block_output.attention_residual_ratio)
                if block_output.ffn_residual_ratio is not None:
                    token_ffn_ratios.append(block_output.ffn_residual_ratio)
                token_pre_norms.append(block_output.pre_norm_l2_mean)
                token_post_norms.append(block_output.post_norm_l2_mean)
                if return_attention_weights:
                    token_weights.append(block_output.attention_weights[:, 0, :])

        layer_summaries = torch.stack(summaries, dim=1)
        if self.layer_position_embedding is not None:
            layer_summaries = layer_summaries + self.layer_position_embedding[None].to(
                dtype=layer_summaries.dtype
            )
        layer_query = self.layer_query.expand(layer_summaries.shape[0], -1, -1)
        layer_block_output = None
        if self.aggregator_block_type == "legacy":
            pooled, layer_weights = self.layer_attention(
                query=layer_query,
                key=layer_summaries,
                value=layer_summaries,
                need_weights=return_attention_weights,
                average_attn_weights=True,
            )
        else:
            layer_block_output = self.layer_query_block(
                layer_query,
                layer_summaries,
                return_attention_weights=return_attention_weights,
            )
            pooled = layer_block_output.hidden_states
            layer_weights = layer_block_output.attention_weights
        pooled = pooled[:, 0, :]
        logits = self.classifier(pooled)
        layer_attention_weights = (
            layer_weights[:, 0, :] if return_attention_weights else None
        )
        return IntentHeadOutput(
            logits=logits,
            pooled_features=pooled,
            attention_weights=layer_attention_weights,
            token_attention_weights=(
                torch.stack(token_weights, dim=1) if return_attention_weights else None
            ),
            layer_attention_weights=layer_attention_weights,
            token_attention_residual_ratio=(
                torch.stack(token_attention_ratios).mean()
                if token_attention_ratios
                else None
            ),
            token_ffn_residual_ratio=(
                torch.stack(token_ffn_ratios).mean() if token_ffn_ratios else None
            ),
            layer_attention_residual_ratio=(
                layer_block_output.attention_residual_ratio
                if layer_block_output is not None
                else None
            ),
            layer_ffn_residual_ratio=(
                layer_block_output.ffn_residual_ratio
                if layer_block_output is not None
                else None
            ),
            token_summary_pre_norm_l2_mean=(
                torch.stack(token_pre_norms).mean() if token_pre_norms else None
            ),
            token_summary_post_norm_l2_mean=(
                torch.stack(token_post_norms).mean() if token_post_norms else None
            ),
            global_feature_pre_norm_l2_mean=(
                layer_block_output.pre_norm_l2_mean
                if layer_block_output is not None
                else None
            ),
            global_feature_post_norm_l2_mean=(
                layer_block_output.post_norm_l2_mean
                if layer_block_output is not None
                else None
            ),
        )
