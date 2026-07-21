"""CPU tests for E1 stage switching and independent projector warm-start."""

import unittest

import torch
import torch.nn as nn

from starVLA.model.framework.VLM4A.QwenPIIntent_v3 import Qwen_PI_Intent_v3
from starVLA.model.modules.intent_head import (
    IntentHeadConfig,
    MultiLayerIntentClassificationHead,
)


class MultiLayerIntentFrameworkTest(unittest.TestCase):
    def test_intent_class_decode_matches_label_formula(self):
        decoded = Qwen_PI_Intent_v3._decode_intent_classes(
            torch.tensor([0, 62, 124])
        )
        torch.testing.assert_close(
            decoded, torch.tensor([[0, 0, 0], [2, 2, 2], [4, 4, 4]])
        )

    def test_s1_s2_boundary_uses_main_run_step(self):
        model = Qwen_PI_Intent_v3.__new__(Qwen_PI_Intent_v3)
        model.intent_training_stage = "s1_s2"
        model.intent_stage1_steps = 10000

        expected = {0: "s1", 9999: "s1", 10000: "s2", 89999: "s2"}
        for step, stage in expected.items():
            model._training_step = step
            self.assertEqual(model._current_intent_stage(), stage)

    def test_intent_projectors_copy_values_without_sharing_storage(self):
        model = Qwen_PI_Intent_v3.__new__(Qwen_PI_Intent_v3)
        nn.Module.__init__(model)
        model.use_multilayer_intent = True
        model.intent_source_layers = (1, 2)
        model.project_layers = nn.ModuleList(
            [
                nn.Sequential(nn.LayerNorm(12), nn.Linear(12, 8)),
                nn.Sequential(nn.LayerNorm(12), nn.Linear(12, 8)),
            ]
        )
        model.intent_head = MultiLayerIntentClassificationHead(
            input_hidden_size=12,
            source_layers=(1, 2),
            config=IntentHeadConfig(
                hidden_size=8,
                num_attention_heads=2,
                classifier_hidden_size=4,
                num_classes=3,
            ),
        )
        with torch.no_grad():
            model.project_layers[0][1].weight.fill_(0.25)

        model.initialize_intent_projectors_from_action()

        action_weight = model.project_layers[0][1].weight
        intent_weight = model.intent_head.project_layers[0][1].weight
        torch.testing.assert_close(action_weight, intent_weight)
        self.assertNotEqual(action_weight.data_ptr(), intent_weight.data_ptr())

        with torch.no_grad():
            intent_weight.add_(1.0)
        self.assertFalse(torch.equal(action_weight, intent_weight))


if __name__ == "__main__":
    unittest.main()
