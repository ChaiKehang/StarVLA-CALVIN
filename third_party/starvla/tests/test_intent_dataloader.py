"""CPU tests for opt-in intent label passthrough in the LeRobot loader."""

import unittest

import numpy as np
import pandas as pd

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset


class IntentDataloaderTest(unittest.TestCase):
    def _stub_dataset(self, frame: pd.DataFrame, data_cfg: dict) -> LeRobotSingleDataset:
        dataset = LeRobotSingleDataset.__new__(LeRobotSingleDataset)
        dataset.data_cfg = data_cfg
        dataset._dataset_name = "intent-unit-test"
        dataset._modality_keys = {}
        dataset.curr_traj_data = None
        dataset.get_trajectory_data = lambda trajectory_id: frame
        dataset._apply_action_mode = lambda data: data
        return dataset

    def test_get_step_data_reads_scalar_intent_column(self):
        frame = pd.DataFrame({"intent.class_id": np.asarray([17, 42], dtype=np.uint8)})
        dataset = self._stub_dataset(
            frame,
            {
                "include_intent": True,
                "intent_class_column": "intent.class_id",
                "intent_num_classes": 125,
            },
        )

        data = dataset.get_step_data(trajectory_id=0, base_index=1)
        self.assertEqual(data["intent_class_id"], 42)

    def test_get_step_data_rejects_missing_intent_column(self):
        dataset = self._stub_dataset(pd.DataFrame({"action": [0]}), {"include_intent": True})
        with self.assertRaisesRegex(KeyError, "parquet column"):
            dataset.get_step_data(trajectory_id=0, base_index=0)

    def test_get_step_data_does_not_require_intent_when_disabled(self):
        dataset = self._stub_dataset(pd.DataFrame({"action": [0]}), {"include_intent": False})
        self.assertNotIn(
            "intent_class_id",
            dataset.get_step_data(trajectory_id=0, base_index=0),
        )


if __name__ == "__main__":
    unittest.main()
