import numpy as np
import torch
from torch.utils.data import DataLoader
from cvlization.torch.data.torchvision_dataset_builder import TorchvisionDatasetBuilder


class MockDataset:
    def __getitem__(self, index):
        image = np.zeros((3, 32, 32))
        return [image], 1

    def __len__(self):
        return 3


def test_torchvision_dataset_works_with_loader():
    dl = DataLoader(MockDataset(), batch_size=2, collate_fn=None)
    batch = next(iter(dl))
    inputs, targets = batch
    assert inputs[0].shape == (2, 3, 32, 32)
    assert isinstance(inputs[0], torch.Tensor)
    assert isinstance(targets, torch.Tensor)

    ds = TorchvisionDatasetBuilder(dataset_classname="CIFAR10").training_dataset()
    dl = DataLoader(ds, batch_size=2, collate_fn=None)
    batch = next(iter(dl))
    inputs, targets = batch
    assert type(inputs) == list
    assert inputs[0].shape == (2, 3, 32, 32)
    assert isinstance(inputs[0], torch.Tensor)
    assert isinstance(targets, list)
    assert isinstance(targets[0], torch.Tensor)
