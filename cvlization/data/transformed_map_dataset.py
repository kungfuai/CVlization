from typing import Callable


class TransformedMapDataset:
    def __init__(self, source_dataset, image_transform: Callable):
        # TODO: support joint transform of image and label
        self._source_dataset = source_dataset
        self._transform = image_transform

    def __getitem__(self, index):
        example = self._source_dataset[index]
        if isinstance(example, tuple) and len(example) == 2:
            # For torchvision datasets.
            image, label = example
        else:
            # For HuggingFace datasets.
            image, label = example["image"], example.get("label", 0)
        transformed_example = self._transform(image), label
        return transformed_example

    def __len__(self):
        return len(self._source_dataset)

    @property
    def source_dataset(self):
        return self._source_dataset