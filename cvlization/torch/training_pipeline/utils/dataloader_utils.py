from dataclasses import dataclass
import logging
import torch
from typing import Union, Callable, Optional
from ....data.dataset_builder import DatasetBuilder, DatasetProvider
from ....specs.data_column import DataColumnType
from ....specs import ModelSpec, MLFramework
from ....specs import ensure_dataset_shapes_and_types

LOGGER = logging.getLogger(__name__)


@dataclass
class DataLoaderUtils:

    collate_method: Optional[Union[str, Callable]] = "zip"
    train_batch_size: int = 32
    val_batch_size: int = 32
    num_workers: int = 0
    model_spec: ModelSpec = None
    
    def create_collate_fn(self) -> Optional[Callable]:
        if self.collate_method == "zip":

            def collate_fn(batch):
                return tuple(zip(*batch))

            return collate_fn
        elif callable(self.collate_method):
            return self.collate_method
        else:
            return None

    def check_data_type_and_shape(self, dataloader, dataset_builder):

        # Data type and shape checks.
        batch = next(iter(dataloader))

        def ensure_list(x):
            if isinstance(x, list):
                return x
            if isinstance(x, tuple):
                return list(x)
            return [x]

        if self.collate_method != "zip":
            batch = tuple([ensure_list(x) for x in batch])

            if isinstance(batch[0], list):
                if self.get_model_inputs() is not None:
                    assert len(self.get_model_inputs()) == len(
                        batch[0]
                    ), f"{len(self.get_model_inputs())} model inputs expected, {len(batch[0])} actual arrays."
                else:
                    LOGGER.warning(
                        f"Saw {len(batch[0])} input arrays, please check if this is expected."
                        " If model spec is provided, the check can be done automatically."
                    )
            else:
                if self.get_model_inputs() is not None:
                    assert (
                        len(self.get_model_inputs()) == 1
                    ), f"Input arrays is not a list, indicating it is probably a single input. But {len(self.get_model_inputs())} model inputs expected."
                else:
                    LOGGER.warning(
                        f"Input arrays is not a list, indicating it is probably a single input."
                        " Please check if this is expected."
                    )
            if self.get_model_inputs() is not None:
                # Some asserts about ml framework, column types. To be refactored.
                for model_input, array in zip(self.get_model_inputs(), batch[0]):
                    if model_input.column_type == DataColumnType.IMAGE:
                        assert (
                            len(array.shape) == 4
                        ), f"Image batch has shape {array.shape}. Training dataset is {self.train_data} with batch size {self.train_data.batch_size}"
                        dataset_provider = self._get_dataset_provider(dataset_builder)
                        if dataset_provider is None:
                            if self.ml_framework == MLFramework.PYTORCH:
                                assert array.shape[1] in [
                                    1,
                                    3,
                                ], f"image batch has shape {array.shape}. Expect channels_first format when dataset_provider is None"

                model_spec = ModelSpec(
                    model_inputs=self.get_model_inputs(),
                    model_targets=self.get_model_targets(),
                )
                ensure_dataset_shapes_and_types(
                    model_spec=model_spec, dataset=dataset_builder.training_dataset()
                )

    def create_training_dataloader(self, dataset_builder):
        dataset_provider = self._get_dataset_provider(dataset_builder)
        if dataset_provider in [
            DatasetProvider.TORCHVISION,
            DatasetProvider.CVLIZATION,
            None,
        ]:
            train_ds = dataset_builder.training_dataset()
            LOGGER.info(f"Training data: {len(train_ds)} examples")
            dl = torch.utils.data.DataLoader(
                train_ds,
                batch_size=self.train_batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=self.create_collate_fn(),
            )
            self.check_data_type_and_shape(dl, dataset_builder)
            return dl
        elif dataset_provider == DatasetProvider.TENSORFLOW_DATASETS:
            # TODO: consider dealing with tfds inside the dataset_builder
            training_dataset = dataset_builder.training_dataset()
            training_dataset = self.convert_tf_dataset_to_iterable_dataset(
                training_dataset
            )
            dl = torch.utils.data.DataLoader(
                training_dataset,
                batch_size=self.train_batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=self.create_collate_fn(),
            )
            self.check_data_type_and_shape(dl, dataset_builder)
        else:
            raise ValueError(f"Unknown dataset provider: {dataset_provider}")
    
    def create_validation_dataloader(self, dataset_builder):
        dataset_provider = self._get_dataset_provider(dataset_builder)
        
        if dataset_provider in [
            DatasetProvider.TORCHVISION,
            DatasetProvider.CVLIZATION,
            None,
        ]:
            val_ds = dataset_builder.validation_dataset()
            LOGGER.info(f"Validation data: {len(val_ds)} examples")
            dl = torch.utils.data.DataLoader(
                val_ds,
                batch_size=self.val_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self.create_collate_fn(),
            )
            return dl
        elif dataset_provider == DatasetProvider.TENSORFLOW_DATASETS:
            validation_dataset = dataset_builder.validation_dataset()
            validation_dataset = self.convert_tf_dataset_to_iterable_dataset(
                validation_dataset
            )
            return torch.utils.data.DataLoader(
                dataset_builder.validation_dataset(),
                batch_size=self.val_batch_size,
            )
        else:
            raise ValueError(f"Unknown dataset provider: {dataset_provider}")
   
    def create_test_dataloader(self, dataset_builder):
        raise NotImplementedError

    def convert_tf_dataset_to_iterable_dataset(self, tf_dataset):
        # TODO: allow using imgaug as a default
        # TODO: transforms should be in a separate class
        # https://github.com/kungfuai/mtrx_2/blob/1b5ff963f4b732883e95e1f86dfbecbb95a7a9ff/src/data/transforms.py#L31

        class IterableImageDataset(torch.utils.data.IterableDataset):
            def __iter__(self):
                for image, label in tf_dataset:
                    image = image.numpy() / 255
                    label = label.numpy()
                    image = torch.cat(
                        [torch.unsqueeze(image[i], 0) for i in range(len(image))]
                    )
                    yield image, label

        return IterableImageDataset()

    def _get_dataset_provider(
        self, dataset_builder: DatasetBuilder
    ) -> Union[DatasetProvider, None]:
        if hasattr(dataset_builder, "dataset_provider"):
            return dataset_builder.dataset_provider
        else:
            return None
    
    def get_model_inputs(self):
        if self.model_spec:
            return self.model_spec.get_model_inputs()

    def get_model_targets(self):
        if self.model_spec:
            return self.model_spec.get_model_targets()

