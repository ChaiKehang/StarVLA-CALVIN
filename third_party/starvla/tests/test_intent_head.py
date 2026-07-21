"""CPU tests for E1 intent pooling and classification modules."""

import unittest

import torch
import torch.nn as nn

from starVLA.model.modules.intent_head import (
    IntentClassificationHead,
    IntentHeadConfig,
    LastValidTokenPooling,
    LearnedQueryCrossAttentionBlock,
    LearnedQueryAttentionPooling,
    MaskedMeanPooling,
    MultiLayerIntentClassificationHead,
    build_zero_initialized_intent_projection,
    compute_intent_auxiliary_loss,
)


class IntentHeadTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.small_config = IntentHeadConfig(
            hidden_size=32,
            num_attention_heads=4,
            classifier_hidden_size=16,
            num_classes=5,
            dropout=0.1,
        )

    def test_fixed_e1_defaults(self):
        config = IntentHeadConfig()
        self.assertEqual(config.hidden_size, 1024)
        self.assertEqual(config.num_attention_heads, 16)
        self.assertEqual(config.attention_head_dim, 64)
        self.assertEqual(config.classifier_hidden_size, 512)
        self.assertEqual(config.num_classes, 125)
        self.assertEqual(config.dropout, 0.1)
        self.assertEqual(config.num_queries, 1)
        self.assertEqual(config.pooler_type, "learned_query")

    def test_main_head_shapes_and_classifier_layout(self):
        head = IntentClassificationHead(self.small_config)
        hidden_states = torch.randn(3, 6, 32)
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1],
            ]
        )
        output = head(hidden_states, attention_mask, return_attention_weights=True)

        self.assertEqual(output.logits.shape, (3, 5))
        self.assertEqual(output.pooled_features.shape, (3, 32))
        self.assertEqual(output.attention_weights.shape, (3, 6))
        self.assertIsInstance(head.classifier[0], nn.LayerNorm)
        self.assertIsInstance(head.classifier[1], nn.Linear)
        self.assertIsInstance(head.classifier[2], nn.GELU)
        self.assertIsInstance(head.classifier[3], nn.Dropout)
        self.assertIsInstance(head.classifier[4], nn.Linear)

    def test_learned_query_ignores_padding(self):
        pooler = LearnedQueryAttentionPooling(hidden_size=32, num_attention_heads=4).eval()
        attention_mask = torch.tensor([[1, 1, 1, 0, 0]])
        original = torch.randn(1, 5, 32)
        changed_padding = original.clone()
        changed_padding[:, 3:] = 10_000.0

        pooled_a, weights = pooler(original, attention_mask, return_attention_weights=True)
        pooled_b = pooler(changed_padding, attention_mask)

        torch.testing.assert_close(pooled_a, pooled_b)
        torch.testing.assert_close(weights[:, 3:], torch.zeros_like(weights[:, 3:]))

    def test_last_valid_token_supports_noncontiguous_mask(self):
        pooler = LastValidTokenPooling(hidden_size=2)
        hidden_states = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]])
        attention_mask = torch.tensor([[1, 0, 1, 0]])
        torch.testing.assert_close(pooler(hidden_states, attention_mask), torch.tensor([[3.0, 3.0]]))

    def test_masked_mean_ignores_padding(self):
        pooler = MaskedMeanPooling(hidden_size=2)
        hidden_states = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]]])
        attention_mask = torch.tensor([[1, 1, 0]])
        torch.testing.assert_close(pooler(hidden_states, attention_mask), torch.tensor([[2.0, 4.0]]))

    def test_empty_sequence_is_rejected(self):
        head = IntentClassificationHead(self.small_config)
        with self.assertRaisesRegex(ValueError, "at least one valid token"):
            head(torch.randn(1, 3, 32), torch.zeros(1, 3))

    def test_intent_gradient_reaches_upstream_projector(self):
        upstream_projector = nn.Linear(12, 32)
        head = IntentClassificationHead(self.small_config)
        source_features = torch.randn(2, 4, 12)
        projected_features = upstream_projector(source_features)
        output = head(projected_features, torch.ones(2, 4))

        output.logits.square().mean().backward()

        self.assertIsNotNone(upstream_projector.weight.grad)
        self.assertGreater(upstream_projector.weight.grad.abs().sum().item(), 0.0)

    def test_intent_projection_is_exactly_zero_initialized(self):
        projection = build_zero_initialized_intent_projection(num_classes=5, hidden_size=32)
        probabilities = torch.softmax(torch.randn(3, 5), dim=-1)

        torch.testing.assert_close(projection.weight, torch.zeros_like(projection.weight))
        torch.testing.assert_close(projection(probabilities), torch.zeros(3, 32))

    def test_multilayer_head_projects_then_pools_tokens_and_layers(self):
        head = MultiLayerIntentClassificationHead(
            input_hidden_size=12,
            source_layers=[1, 3, 5],
            config=self.small_config,
        )
        hidden_states = [torch.randn(2, 6, 12) for _ in range(3)]
        attention_mask = torch.tensor(
            [[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]]
        )
        output = head(
            hidden_states,
            attention_mask,
            return_attention_weights=True,
        )

        self.assertEqual(output.logits.shape, (2, 5))
        self.assertEqual(output.pooled_features.shape, (2, 32))
        self.assertEqual(output.token_attention_weights.shape, (2, 3, 6))
        self.assertEqual(output.layer_attention_weights.shape, (2, 3))
        torch.testing.assert_close(
            output.layer_attention_weights.sum(dim=-1), torch.ones(2)
        )
        self.assertEqual(head.project_layers[0][1].in_features, 12)
        self.assertEqual(head.project_layers[0][1].out_features, 32)

    def test_multilayer_head_ignores_padding_in_every_source_layer(self):
        head = MultiLayerIntentClassificationHead(
            input_hidden_size=12,
            source_layers=[1, 2],
            config=self.small_config,
        ).eval()
        attention_mask = torch.tensor([[1, 1, 1, 0, 0]])
        original = [torch.randn(1, 5, 12) for _ in range(2)]
        changed = [hidden.clone() for hidden in original]
        for hidden in changed:
            hidden[:, 3:] = 10000.0

        output_a = head(original, attention_mask)
        output_b = head(changed, attention_mask)
        torch.testing.assert_close(output_a.logits, output_b.logits)

    def test_multilayer_head_validates_source_count(self):
        head = MultiLayerIntentClassificationHead(
            input_hidden_size=12,
            source_layers=[1, 2],
            config=self.small_config,
        )
        with self.assertRaisesRegex(ValueError, "one hidden-state tensor"):
            head([torch.randn(1, 3, 12)], torch.ones(1, 3))

    def test_query_ffn_v2_has_transformer_residuals_and_normalized_output(self):
        head = MultiLayerIntentClassificationHead(
            input_hidden_size=12,
            source_layers=[1, 2, 3],
            config=self.small_config,
            aggregator_block_type="query_ffn_v2",
            query_ffn_multiplier=2.0,
            query_ffn_dropout=0.0,
        ).eval()
        hidden_states = [torch.randn(2, 5, 12) for _ in range(3)]
        output = head(hidden_states, torch.ones(2, 5), return_attention_weights=True)

        self.assertIsInstance(head.token_query_block, LearnedQueryCrossAttentionBlock)
        self.assertIsInstance(head.layer_query_block, LearnedQueryCrossAttentionBlock)
        self.assertIsNone(head.token_attention)
        self.assertIsNone(head.layer_attention)
        self.assertIsNotNone(output.token_attention_residual_ratio)
        self.assertIsNotNone(output.token_ffn_residual_ratio)
        self.assertIsNotNone(output.layer_attention_residual_ratio)
        self.assertIsNotNone(output.layer_ffn_residual_ratio)
        self.assertIsNotNone(output.global_feature_pre_norm_l2_mean)
        self.assertIsNotNone(output.global_feature_post_norm_l2_mean)
        torch.testing.assert_close(
            output.pooled_features.norm(dim=-1),
            torch.full((2,), self.small_config.hidden_size**0.5),
            rtol=2.0e-3,
            atol=2.0e-3,
        )

    def test_query_ffn_v2_ignores_padding_and_backpropagates_through_both_ffns(self):
        head = MultiLayerIntentClassificationHead(
            input_hidden_size=12,
            source_layers=[1, 2],
            config=self.small_config,
            aggregator_block_type="query_ffn_v2",
            query_ffn_dropout=0.0,
        ).eval()
        attention_mask = torch.tensor([[1, 1, 1, 0, 0]])
        original = [torch.randn(1, 5, 12) for _ in range(2)]
        changed = [hidden.clone() for hidden in original]
        for hidden in changed:
            hidden[:, 3:] = 10000.0

        output_a = head(original, attention_mask)
        output_b = head(changed, attention_mask)
        torch.testing.assert_close(output_a.logits, output_b.logits)

        output_a.logits.square().mean().backward()
        self.assertGreater(
            head.token_query_block.ffn[0].weight.grad.abs().sum().item(), 0.0
        )
        self.assertGreater(
            head.layer_query_block.ffn[0].weight.grad.abs().sum().item(), 0.0
        )

    def test_query_ffn_v2_supports_attention_only_ablation(self):
        head = MultiLayerIntentClassificationHead(
            input_hidden_size=12,
            source_layers=[1, 2],
            config=self.small_config,
            aggregator_block_type="query_ffn_v2",
            token_query_use_ffn=False,
            layer_query_use_ffn=False,
        )
        output = head([torch.randn(2, 4, 12) for _ in range(2)], torch.ones(2, 4))
        self.assertIsNone(head.token_query_block.ffn)
        self.assertIsNone(head.layer_query_block.ffn)
        self.assertIsNone(output.token_ffn_residual_ratio)
        self.assertIsNone(output.layer_ffn_residual_ratio)

    def test_multilayer_head_rejects_unknown_aggregator_block(self):
        with self.assertRaisesRegex(ValueError, "aggregator_block_type"):
            MultiLayerIntentClassificationHead(
                input_hidden_size=12,
                source_layers=[1, 2],
                config=self.small_config,
                aggregator_block_type="unknown",
            )

    def test_auxiliary_loss_matches_section_4_1(self):
        logits = torch.tensor([[3.0, 0.0, -1.0], [0.0, 2.0, 1.0]], requires_grad=True)
        targets = torch.tensor([0, 2], dtype=torch.long)
        output = compute_intent_auxiliary_loss(logits, targets, loss_weight=0.1)

        expected = nn.functional.cross_entropy(logits, targets)
        torch.testing.assert_close(output.loss, expected)
        torch.testing.assert_close(output.weighted_loss, 0.1 * expected)
        self.assertEqual(output.top1_accuracy.item(), 0.5)

        output.weighted_loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_auxiliary_loss_rejects_invalid_targets(self):
        with self.assertRaisesRegex(ValueError, "must be in"):
            compute_intent_auxiliary_loss(
                torch.randn(2, 5),
                torch.tensor([0, 5], dtype=torch.long),
                loss_weight=0.1,
            )


if __name__ == "__main__":
    unittest.main()
