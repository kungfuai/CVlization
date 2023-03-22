from typing import Callable


class TransformedDataset:
    def __init__(self, source_dataset, image_transform: Callable):
        self._source_dataset = source_dataset
        self._transform = image_transform

    def __getitem__(self, index):
        example = self._source_dataset[index]
        image, label = example
        transformed_example = self._transform(image), label
        return transformed_example

    def __len__(self):
        return len(self._source_dataset)

    @property
    def source_dataset(self):
        return self._source_dataset