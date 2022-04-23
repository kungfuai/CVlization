import logging
import numpy as np
from typing import Iterable
from torch.utils.data import Dataset, IterableDataset
from ..data import MLDataset


LOGGER = logging.getLogger(__name__)


class MapDataset(Dataset):
    def __init__(self, ml_dataset: MLDataset):
        self.ml_dataset = ml_dataset

    def __len__(self):
        return len(self.ml_dataset)

    def __getitem__(self, index):
        return self.ml_dataset[index]


class GeneratorDataset(IterableDataset):
    # TODO: need to use an unbatched version of tf.data.Dataset or iterable.
    def __init__(self, dataset: Iterable):
        super().__init__()
        self._iterable = dataset
        self._iterated = self._iter()

    def __next__(self):
        return next(self._iterated)

    def __iter__(self):
        return self._iter()

    def _iter(self):
        for k, batch in enumerate(self._iterable):
            LOGGER.info(f" .. loading batch {k}")
            result_batch = []
            for i, part in enumerate(batch):
                LOGGER.info(f" .. loading part {i}")

                if hasattr(part, "numpy"):
                    LOGGER.info(f"{part.shape}")
                    result_batch.append(part.numpy())
                elif isinstance(part, list) or isinstance(part, tuple):
                    result_part = []
                    for j, tensor in enumerate(part):
                        if hasattr(tensor, "numpy"):
                            LOGGER.info(f"{i}.{j}: {tensor.shape}")
                            result_part.append(tensor.numpy())
                        elif isinstance(tensor, np.ndarray):
                            result_part.append(tensor)
                        else:
                            raise ValueError(
                                f"Unsupported type: {type(tensor)}, in part {i}.{j} of a batch"
                            )
                    if isinstance(part, tuple):
                        result_part = tuple(result_part)
                    result_batch.append(result_part)
                elif isinstance(part, np.ndarray):
                    result_batch.append(part)
                else:
                    raise ValueError(
                        f"Unsupported type in part {i} of a batch: {type(part)}"
                    )

            LOGGER.info(f"result batch: {len(result_batch)}")
            if isinstance(batch, tuple):
                result_batch = tuple(result_batch)
            yield result_batch


if __name__ == "__main__":
    import tensorflow as tf

    def gen():
        while True:
            yield np.random.rand(10, 10), np.random.rand(10, 10)

    tds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32))
    ds = GeneratorDataset(tds)
    for i, batch in enumerate(ds):
        print(i)
        if i > 10:
            break
