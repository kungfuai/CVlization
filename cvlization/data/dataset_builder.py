from collections.abc import Iterable
import enum
from typing import Union, List, runtime_checkable, Callable
from ..utils import Protocol


@runtime_checkable
class MapStyleDataset(Protocol):
    """An abstract interface for Dataset.

    TODO: consider using a stronger type for the return value of __getitem__.
    Datasets for supervised learning, variational inference,
    causal inference (e.g. potential outcomes), reinforcement learning can have
    different structures in individual data examples. Whether to use sample_weight
    is also a potential source of ambiguity.

    TODO: For large scale data, sequential-access datasets may be preferrable for
    their high performance. Fot example, TFRecordDataset supports __iter__ but not
    __getitem__. Another Protocol class may be needed to support this. We have a
    workaround for TFRecord where each training example has a large size: have a label
    file that can be loaded into a
    random-access Dataset; store only one tf.Example in a tfrecord file; construct
    a TFRecordDataset with all tfrecord file paths; implement a generator to yield
    data from TFRecordDataset; use `tf.data.Dataset.from_generator` to do parsing,
    mappping, batching, prefetching.
    """

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError


Dataset = Union[MapStyleDataset, Iterable]


class DatasetProvider(str, enum.Enum):
    TORCHVISION = "torchvision"
    TENSORFLOW_DATASETS = "tensorflow_datasets"
    HUGGINGFACE = "huggingface"
    KAGGLE = "kaggle"
    CVLIZATION = "cvlization"


# TODO: include an example of writing a custom dataset for
#   an object detection dataset (potentially multi-task).
#
# TODO: include an example of writing a custom tabular dataset,
#   where the size is too big to fit in the memory.


@runtime_checkable
class DatasetBuilder(Protocol):
    @property
    def dataset_provider(self):
        ...

    def training_dataset(self) -> Dataset:
        ...

    def validation_dataset(self) -> Union[Dataset, List[Dataset]]:
        # For some use cases, more than one validation datasets are returned.
        ...


class BaseDatasetBuilder:
    """Random seed is optional, as most public datasets are already splitted."""

    @property
    def dataset_provider(self):
        return None

    def training_dataset(self) -> Dataset:
        raise NotImplementedError

    def validation_dataset(self) -> Union[Dataset, List[Dataset]]:
        raise NotImplementedError

    def test_dataset(self) -> Union[Dataset, List[Dataset]]:
        return None


class TransformedMapStyleDataset:
    def __init__(
        self,
        base_dataset: MapStyleDataset,
        transform: Callable,
    ):
        # TODO: consider allowing shuffle.
        self.base_dataset = base_dataset
        self.transform = transform
        assert isinstance(
            base_dataset, MapStyleDataset
        ), f"{base_dataset} is not a MapStyleDataset: {type(base_dataset)}"

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        return self.transform(self.base_dataset[index])


class TransformedIterableDataset:
    def __init__(
        self,
        base_dataset: Dataset,
        transform: Callable,
    ):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterable:
        if isinstance(self.base_dataset, MapStyleDataset):
            indices = range(len(self.base_dataset))
            for i in indices:
                yield self.transform(self.base_dataset[i])
        else:
            for item in self.base_dataset:
                yield self.transform(item)
