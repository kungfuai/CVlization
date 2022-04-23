import pytest
from cvlization.lab.datasets import TorchVisionDataset, TFDSImageDataset


def test_torchvision_dataset_can_get_correct_classname():
    ds = TorchVisionDataset("random_torchvision")
    assert ds.get_dataset_classname() == "RANDOM"
    with pytest.raises(ValueError):
        ds.training_dataset()

    with pytest.raises(ValueError):
        ds.validation_dataset()

    ds = TorchVisionDataset("cifar10")
    assert ds.get_dataset_classname() == "CIFAR10"
    ds = TorchVisionDataset("cifar10_torchvision")
    assert ds.get_dataset_classname() == "CIFAR10"
    ds = TorchVisionDataset("cifar10_data")
    assert ds.get_dataset_classname() == "CIFAR10_DATA"


def test_torchvision_dataset_tensor_shape():
    import torch

    ds = TorchVisionDataset("mnist_torchvision")
    train_examples = ds.training_dataset(batch_size=None)
    assert len(train_examples) == 60000
    for example in train_examples:
        assert type(example) == list
        assert len(example) == 2
        assert example[0].shape == (1, 28, 28)
        assert type(example[1]) == int
        break
    train_batches = ds.training_dataset(batch_size=2)
    assert len(train_batches) == 30000
    for batch in train_batches:
        assert type(batch) == list
        assert len(batch) == 2
        assert batch[0].shape == (2, 1, 28, 28)
        assert batch[1].shape == (2,)
        assert batch[1].shape == torch.Size([2])
        break


def test_tfds_dataset_for_mnist_tensor_shape():
    ds = TFDSImageDataset("mnist_tfds")
    assert ds._tfds_dataset_name == "mnist"
    train_data = ds.training_dataset(batch_size=None)
    assert len(train_data) == 60000
    for x in train_data:
        assert isinstance(x, tuple)
        assert len(x) == 2
        inputs, targets = x
        assert inputs.shape == (28, 28, 1)
        assert targets.shape == []
        break

    train_data = ds.training_dataset(batch_size=2)
    assert len(train_data) == 30000
    for x in train_data:
        assert isinstance(x, tuple)
        assert len(x) == 2
        inputs, targets = x
        assert inputs.shape == (2, 28, 28, 1)
        assert targets.shape == [2]
        break
