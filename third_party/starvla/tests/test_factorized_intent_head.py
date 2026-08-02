"""Shape and parameter-isolation tests for the factorized Intent head."""

import torch

from starVLA.model.modules.intent_head import (
    FactorizedMultiLayerIntentClassificationHead,
)


def test_factorized_intent_output_shapes_and_independent_layer_queries():
    torch.manual_seed(7)
    head = FactorizedMultiLayerIntentClassificationHead(
        input_hidden_size=16,
        source_layers=[1, 2, 3],
        hidden_size=8,
        num_attention_heads=2,
        xyz_rpy_classifier_hidden_size=6,
        gripper_classifier_hidden_size=4,
        query_ffn_dropout=0.0,
        query_attention_dropout=0.0,
        dropout=0.0,
    ).eval()
    hidden = [torch.randn(2, 5, 16) for _ in range(3)]
    mask = torch.ones(2, 5, dtype=torch.bool)
    output = head(hidden, mask, return_attention_weights=True)

    assert output.xyz_logits.shape == (2, 125)
    assert output.rpy_logits.shape == (2, 125)
    assert output.gripper_logits.shape == (2, 5)
    assert output.xyz_features.shape == (2, 8)
    assert output.token_attention_weights.shape == (2, 3, 5)
    assert set(output.layer_attention_weights) == {"xyz", "rpy", "gripper"}
    assert len({id(block) for block in head.layer_query_blocks.values()}) == 3
    assert len({query.data_ptr() for query in head.layer_queries.values()}) == 3
