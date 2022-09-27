import numpy as np
from cvlization.data.mock_dataset import RandomImageClassificationDataset


def test_random_image_classification_dataset():
    ds = RandomImageClassificationDataset(
        height=32, width=20, num_classes=5, multilabel=False, sample_size=10)
    assert len(ds) == 10
    example = ds[0]
    image, label = example
    assert image.shape == (3, 32, 20)
    assert image.dtype == np.float32
    assert isinstance(label, int)

    # Multilabel
    ds = RandomImageClassificationDataset(
        height=32, width=20, num_classes=5, multilabel=True, sample_size=10)
    example = ds[1]
    label = example[1]
    assert label.shape == (5,)
    assert label.max() == 1
    assert label.dtype == np.float32
