"""CPU tests for the optional E1-B DiT conditioning input."""

import unittest

import torch

from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import (
    DiT,
    IntentFiLMMLP,
)


class DiTIntentConditioningTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.model = DiT(
            num_attention_heads=2,
            attention_head_dim=4,
            num_layers=2,
            dropout=0.0,
            final_dropout=False,
            positional_embeddings=None,
            interleave_self_attention=True,
            cross_attention_dim=8,
            output_dim=8,
            norm_type="ada_norm",
        ).eval()
        self.hidden_states = torch.randn(2, 4, 8)
        self.encoder_hidden_states = [torch.randn(2, 3, 8) for _ in range(2)]
        self.timesteps = torch.tensor([10, 20], dtype=torch.long)

    def _forward(self, intent_condition=None):
        return self.model(
            hidden_states=self.hidden_states,
            encoder_hidden_states=self.encoder_hidden_states,
            timestep=self.timesteps,
            return_pre_output=True,
            intent_condition=intent_condition,
        )

    def test_zero_condition_preserves_existing_dit_output(self):
        output_without_intent = self._forward()
        output_with_zero_intent = self._forward(torch.zeros(2, 8))
        torch.testing.assert_close(output_without_intent, output_with_zero_intent)

    def test_nonzero_condition_changes_executed_adaln_path(self):
        output_without_intent = self._forward()
        output_with_intent = self._forward(torch.randn(2, 8))
        self.assertFalse(torch.allclose(output_without_intent, output_with_intent))

    def test_invalid_condition_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "shape must match"):
            self._forward(torch.zeros(2, 7))

    def test_condition_diagnostics_report_actual_relative_scale(self):
        intent_condition = torch.randn(2, 8)
        output, diagnostics = self.model(
            hidden_states=self.hidden_states,
            encoder_hidden_states=self.encoder_hidden_states,
            timestep=self.timesteps,
            return_pre_output=True,
            intent_condition=intent_condition,
            return_condition_diagnostics=True,
        )

        timestep_embedding = self.model.timestep_encoder(self.timesteps)
        timestep_l2 = timestep_embedding.float().norm(dim=-1)
        intent_l2 = intent_condition.float().norm(dim=-1)
        expected_joint_l2 = (timestep_embedding + intent_condition).float().norm(dim=-1).mean()
        expected_ratio = (
            intent_l2 / timestep_l2.clamp_min(torch.finfo(torch.float32).eps)
        ).mean()

        self.assertEqual(output.shape, self.hidden_states.shape)
        torch.testing.assert_close(
            diagnostics["timestep_embedding_l2_mean"], timestep_l2.mean()
        )
        torch.testing.assert_close(
            diagnostics["joint_timestep_condition_l2_mean"], expected_joint_l2
        )
        torch.testing.assert_close(
            diagnostics["intent_condition_to_timestep_l2_ratio"], expected_ratio
        )

    def test_ffn_intent_film_is_opt_in_and_zero_initialized(self):
        norm3_ids = [id(block.norm3) for block in self.model.transformer_blocks]
        ffn_ids = [id(block.ff) for block in self.model.transformer_blocks]
        output_before_enable = self._forward()

        self.model.enable_ffn_intent_film(num_intent_classes=5)
        probabilities = torch.softmax(torch.randn(2, 5), dim=-1)
        output_after_enable = self.model(
            hidden_states=self.hidden_states,
            encoder_hidden_states=self.encoder_hidden_states,
            timestep=self.timesteps,
            return_pre_output=True,
            ffn_intent_probabilities=probabilities,
        )

        for index, block in enumerate(self.model.transformer_blocks):
            self.assertEqual(id(block.norm3), norm3_ids[index])
            self.assertEqual(id(block.ff), ffn_ids[index])
            self.assertIsNotNone(block.ffn_intent_film)
            self.assertIsNone(block.ffn_intent_film.bias)
            torch.testing.assert_close(
                block.ffn_intent_film.weight,
                torch.zeros_like(block.ffn_intent_film.weight),
            )
        torch.testing.assert_close(output_before_enable, output_after_enable)

    def test_nonzero_ffn_intent_film_changes_output(self):
        self.model.enable_ffn_intent_film(num_intent_classes=5)
        probabilities = torch.softmax(torch.randn(2, 5), dim=-1)
        output_at_zero = self.model(
            hidden_states=self.hidden_states,
            encoder_hidden_states=self.encoder_hidden_states,
            timestep=self.timesteps,
            return_pre_output=True,
            ffn_intent_probabilities=probabilities,
        )
        torch.nn.init.normal_(self.model.transformer_blocks[0].ffn_intent_film.weight)
        output_after_update = self.model(
            hidden_states=self.hidden_states,
            encoder_hidden_states=self.encoder_hidden_states,
            timestep=self.timesteps,
            return_pre_output=True,
            ffn_intent_probabilities=probabilities,
        )
        self.assertFalse(torch.allclose(output_at_zero, output_after_update))

    def test_enabled_ffn_intent_film_requires_probabilities(self):
        self.model.enable_ffn_intent_film(num_intent_classes=5)
        with self.assertRaisesRegex(ValueError, "was not provided"):
            self._forward()

    def test_query_film_is_only_added_to_cross_attention_blocks_and_starts_zero(self):
        output_before_enable = self._forward()
        self.model.enable_cross_attn_query_intent_film(num_intent_classes=5)
        probabilities = torch.softmax(torch.randn(2, 5), dim=-1)
        output_after_enable = self.model(
            hidden_states=self.hidden_states,
            encoder_hidden_states=self.encoder_hidden_states,
            timestep=self.timesteps,
            return_pre_output=True,
            cross_attn_query_intent_probabilities=probabilities,
        )

        self.assertIsNotNone(
            self.model.transformer_blocks[0].cross_attn_query_intent_film
        )
        self.assertIsNone(
            self.model.transformer_blocks[1].cross_attn_query_intent_film
        )
        projection = self.model.transformer_blocks[0].cross_attn_query_intent_film
        self.assertIsNone(projection.bias)
        torch.testing.assert_close(projection.weight, torch.zeros_like(projection.weight))
        torch.testing.assert_close(output_before_enable, output_after_enable)

    def test_nonzero_query_film_changes_cross_attention_output(self):
        output_without_query_film = self._forward()
        self.model.enable_cross_attn_query_intent_film(num_intent_classes=5)
        probabilities = torch.softmax(torch.randn(2, 5), dim=-1)
        projection = self.model.transformer_blocks[0].cross_attn_query_intent_film
        torch.nn.init.normal_(projection.weight)
        output_with_query_film = self.model(
            hidden_states=self.hidden_states,
            encoder_hidden_states=self.encoder_hidden_states,
            timestep=self.timesteps,
            return_pre_output=True,
            cross_attn_query_intent_probabilities=probabilities,
        )
        self.assertFalse(torch.allclose(output_without_query_film, output_with_query_film))

    def test_enabled_query_film_requires_probabilities(self):
        self.model.enable_cross_attn_query_intent_film(num_intent_classes=5)
        with self.assertRaisesRegex(ValueError, "probabilities were not provided"):
            self._forward()

    def test_mlp_film_uses_independent_256d_zero_initialized_projectors(self):
        output_before_enable = self._forward()
        self.model.enable_ffn_intent_film(
            num_intent_classes=5,
            projector_type="mlp",
            hidden_dim=256,
        )
        self.model.enable_cross_attn_query_intent_film(
            num_intent_classes=5,
            projector_type="mlp",
            hidden_dim=256,
        )
        probabilities = torch.softmax(torch.randn(2, 5), dim=-1)
        output_at_zero = self.model(
            hidden_states=self.hidden_states,
            encoder_hidden_states=self.encoder_hidden_states,
            timestep=self.timesteps,
            return_pre_output=True,
            ffn_intent_probabilities=probabilities,
            cross_attn_query_intent_probabilities=probabilities,
        )

        ffn_projectors = [
            block.ffn_intent_film for block in self.model.transformer_blocks
        ]
        self.assertEqual(len({id(module) for module in ffn_projectors}), 2)
        for module in ffn_projectors:
            self.assertIsInstance(module, IntentFiLMMLP)
            self.assertEqual(module.in_features, 5)
            self.assertEqual(module.hidden_features, 256)
            self.assertEqual(module.out_features, 16)
            self.assertIsInstance(module.activation, torch.nn.SiLU)
            self.assertIsInstance(module.norm, torch.nn.LayerNorm)
            self.assertIsNone(module.output_proj.bias)
            torch.testing.assert_close(
                module.output_proj.weight,
                torch.zeros_like(module.output_proj.weight),
            )

        query_projector = self.model.transformer_blocks[0].cross_attn_query_intent_film
        self.assertIsInstance(query_projector, IntentFiLMMLP)
        self.assertEqual(query_projector.hidden_features, 256)
        self.assertIsNone(
            self.model.transformer_blocks[1].cross_attn_query_intent_film
        )
        self.assertNotEqual(id(query_projector), id(ffn_projectors[0]))
        torch.testing.assert_close(output_before_enable, output_at_zero)

        torch.nn.init.normal_(ffn_projectors[0].output_proj.weight)
        output_after_update = self.model(
            hidden_states=self.hidden_states,
            encoder_hidden_states=self.encoder_hidden_states,
            timestep=self.timesteps,
            return_pre_output=True,
            ffn_intent_probabilities=probabilities,
            cross_attn_query_intent_probabilities=probabilities,
        )
        self.assertFalse(torch.allclose(output_at_zero, output_after_update))

    def test_factorized_router_is_zero_initialized_bounded_and_maskable(self):
        output_before = self._forward()
        self.model.enable_cross_attn_query_intent_film(
            num_intent_classes=255,
            projector_type="mlp",
            hidden_dim=512,
            layer_numbers=[1],
            output_bias=True,
            modulation_bound=0.2,
        )
        probabilities = torch.cat(
            [
                torch.softmax(torch.randn(2, 125), dim=-1),
                torch.softmax(torch.randn(2, 125), dim=-1),
                torch.softmax(torch.randn(2, 5), dim=-1),
            ],
            dim=-1,
        )
        router = self.model.transformer_blocks[0].cross_attn_query_intent_film
        self.assertEqual(router.in_features, 255)
        self.assertEqual(router.hidden_features, 512)
        self.assertIsNotNone(router.output_proj.bias)
        torch.testing.assert_close(
            router.output_proj.weight, torch.zeros_like(router.output_proj.weight)
        )
        torch.testing.assert_close(
            router.output_proj.bias, torch.zeros_like(router.output_proj.bias)
        )
        output_at_zero = self.model(
            hidden_states=self.hidden_states,
            encoder_hidden_states=self.encoder_hidden_states,
            timestep=self.timesteps,
            return_pre_output=True,
            cross_attn_query_intent_probabilities=probabilities,
        )
        torch.testing.assert_close(output_before, output_at_zero)

        torch.nn.init.normal_(router.output_proj.weight, std=10.0)
        output_masked, diagnostics = self.model(
            hidden_states=self.hidden_states,
            encoder_hidden_states=self.encoder_hidden_states,
            timestep=self.timesteps,
            return_pre_output=True,
            cross_attn_query_intent_probabilities=probabilities,
            intent_film_sample_mask=torch.zeros(2),
            return_condition_diagnostics=True,
        )
        torch.testing.assert_close(output_before, output_masked)
        self.assertLessEqual(
            diagnostics["query_film_bounded_delta_gamma_abs_max_mean"].item(),
            0.2 + 1.0e-6,
        )
        self.assertLessEqual(
            diagnostics["query_film_bounded_delta_beta_abs_max_mean"].item(),
            0.2 + 1.0e-6,
        )


if __name__ == "__main__":
    unittest.main()
