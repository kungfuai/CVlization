"""Unit tests for Mean structured average prediction.
"""
import numpy as np
from cvlization.torch.metrics.msap import msAP


def test_msap():
    line_pred = np.array([[[0, 5]] * 1000])  # shape = (batch, seq_length, 2)
    line_gt = np.array([[[0, 5]] * 3])  # shape = (batch, num_gt, 2)
    print(line_pred.shape)
    print(line_gt.shape)
    threshold = 1
    assert msAP(line_pred, line_gt, threshold) == 1
