import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from driving_model import (
    DRIVING_LABELS,
    ProfessorSafetyCNN,
    load_driving_model,
    predict_label,
    preprocess_bgr_frame,
)


class DrivingModelTest(unittest.TestCase):
    def test_loads_preprocess_safety_cnn_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "safety_cnn.pt"
            model = ProfessorSafetyCNN(
                image_height=224,
                image_width=224,
                conv_hidden_channels=(8, 16, 32, 64, 128),
                fc_hidden_nodes=32,
                num_classes=2,
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": list(DRIVING_LABELS),
                    "model_config": {
                        "image_height": 224,
                        "image_width": 224,
                        "conv_hidden_channels": [8, 16, 32, 64, 128],
                        "fc_hidden_nodes": 32,
                        "preprocessing": "lab_table_mask",
                    },
                },
                path,
            )

            loaded, metadata = load_driving_model(path, torch.device("cpu"))

        self.assertIsInstance(loaded, ProfessorSafetyCNN)
        self.assertEqual(metadata["preprocessing"], "lab_table_mask")
        self.assertEqual(metadata["labels"], list(DRIVING_LABELS))

    def test_preprocess_mode_returns_three_channel_batch(self):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        frame[:, :, :] = (145, 195, 220)

        tensor = preprocess_bgr_frame(
            frame,
            image_size=64,
            device=torch.device("cpu"),
            preprocessing="lab_table_mask",
        )

        self.assertEqual(tuple(tensor.shape), (1, 3, 64, 64))
        self.assertGreaterEqual(float(tensor.min()), 0.0)
        self.assertLessEqual(float(tensor.max()), 1.0)

    def test_predict_label_uses_loaded_metadata(self):
        model = ProfessorSafetyCNN(
            image_height=64,
            image_width=64,
            conv_hidden_channels=(8, 16, 32),
            fc_hidden_nodes=16,
            num_classes=2,
        )
        frame = np.zeros((64, 64, 3), dtype=np.uint8)

        label, confidence, probs = predict_label(
            model,
            frame,
            device=torch.device("cpu"),
            labels=list(DRIVING_LABELS),
            image_size=64,
            preprocessing="lab_table_mask",
        )

        self.assertIn(label, DRIVING_LABELS)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertEqual(len(probs), 2)


if __name__ == "__main__":
    unittest.main()
