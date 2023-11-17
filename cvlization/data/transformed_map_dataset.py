# TODO: deprecated
from typing import Callable


class TransformedMapDataset:
    def __init__(self, source_dataset,
                 image_transform: Callable=None,
                 input_and_target_transform: Callable=None,
                 ):
        # TODO: support chaining of transforms.
        #   If the source dataset is a TransformedMapDataset, then we can chain the transforms.
        self._source_dataset = source_dataset
        if image_transform is None and input_and_target_transform is None:
            raise ValueError("Either image_transform or input_and_target_transform must be provided.")
        self._image_transform = image_transform
        self._input_and_target_transform = input_and_target_transform

    def __getitem__(self, index):
        example = self._source_dataset[index]
        if isinstance(example, tuple) and len(example) == 2:
            # For torchvision datasets.
            image, label = example
        else:
            # For HuggingFace datasets.
            image, label = example["image"], example.get("label", 0)
        if self._input_and_target_transform is not None:
            transformed_example = self._input_and_target_transform(example)
        else:
            transformed_example = self._image_transform(image), label
        return transformed_example

    def __len__(self):
        return len(self._source_dataset)

    @property
    def source_dataset(self):
        return self._source_dataset