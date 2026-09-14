"""Check point batching against the actual supplied checkpoints' full forward pass."""
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.export_supervised_full_disk import load_model, predict


class SupervisedExportTest(unittest.TestCase):
    def test_supplied_checkpoints_match_normal_forward(self):
        torch.set_num_threads(2)
        rng = np.random.default_rng(4)
        aia = rng.uniform(-2, 500, (6, 3, 5)).astype(np.float32)
        checkpoint_dir = Path(__file__).resolve().parents[1] / 'eval_and_enet_specs/checkpoints'
        for filename in ('model_best_bp.pth', 'model_best_en.pth'):
            with self.subTest(checkpoint=filename):
                model, _ = load_model(checkpoint_dir / filename, torch.device('cpu'))
                with torch.inference_mode():
                    expected = model(torch.from_numpy(aia)[None])[0][0, :18].numpy()
                actual = predict(model, aia, pixel_batch=7)
                np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-4)
                self.assertEqual(actual.shape, (18, 3, 5))

    def test_spatial_convolution_is_rejected(self):
        with self.assertRaisesRegex(ValueError, '1x1'):
            predict(torch.nn.Conv2d(6, 18, 3), np.zeros((6, 5, 5), dtype=np.float32))


if __name__ == '__main__':
    unittest.main()
