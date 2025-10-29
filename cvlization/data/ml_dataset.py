from typing import List, Optional
import random
import sys
from dataclasses import dataclass
import logging
import numpy as np

from cvlization.specs.type_checks import MapLike, SelfCheckable, CheckedWithModelSpec

from ..data.utils import one_hot
from ..specs import (
    DataColumnType,
    ModelInput,
    ModelTarget,
    ModelSpec,
    ensure_dataset_shapes_and_types,
)
from .data_rows import DataRows

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


@dataclass
class MLDataset(MapLike, SelfCheckable, CheckedWithModelSpec):
    """A MLDataset is a dataset that is potentially trainable.

    It manages the list of inputs and the list of targets.
    which in turn manage the parameters of feature transforms.

    How to use it:
    1. Declare model inputs and targets, together with feature transforms.
    2. Instruct how each training example is fetched, and returned as a dict in the __getitem__ method.
    3. Convert MLDataset to a Keras Sequence, or torch Dataset using provided utilities.
    4. Create a model that matches the above model inputs and model targets.
    5. Train the model.
    """

    # TODO: handle image augmentation.
    # TODO: shuffle is off by default. Consider turning it on for training.

    data_rows: DataRows
    model_inputs: List[ModelInput]
    model_targets: List[ModelTarget]
    batch_size: Optional[int] = 2
    shuffle: Optional[bool] = False

    use_one_hot_for_categorical: bool = True
    name: Optional[str] = ""

    def __post_init__(self):
        self._index_map = list(range(len(self.data_rows)))
        if self.shuffle:
            random.shuffle(self._index_map)

    def get_row(self, i: int):
        j = self._index_map[i]
        try:
            data_row = self.data_rows[j]
            for model_target in self.model_targets:
                val = data_row.get(model_target.key)
                LOGGER.debug(
                    f"key={model_target.key}, val={val}, {type(val)}, neg_class_weight={model_target.negative_class_weight}"
                )
                if val is None:
                    msg = f"None value for target {model_target.key}!"
                    LOGGER.info(data_row)
                    LOGGER.error(msg)
                    raise ValueError(msg)
                if int(val) == 0:
                    sample_weight = model_target.negative_class_weight
                    # TODO: should not set sample weight to the dictionary!
                    # It should be immutable.
                    if data_row.get("sample_weight") is None:
                        data_row["sample_weight"] = sample_weight
            return data_row
        except KeyboardInterrupt:
            sys.exit(1)
        except Exception:
            raise

    def __len__(self):
        return len(self.data_rows)

    def __getitem__(self, i):
        """Want to return transformed inputs."""
        inputs = []
        row_dict = self.get_row(i)
        for model_input in self.model_inputs:
            raw_value = row_dict[model_input.key]
            x = raw_value
            for t in model_input.transforms or []:
                x = t.transform([x])[0]
            if x is None:
                msg = f"None value for model input: {model_input.key}. Using data from {self.data_rows}"
                LOGGER.error(msg)
                raise IOError(msg)
            if type(x) in [int, float]:
                x = [x]
            x = np.array(x)
            # if len(x.shape) == 0:
            #     x = x.reshape(1)
            if model_input.column_type == DataColumnType.IMAGE:
                if model_input.raw_shape:
                    actual_shape = x.shape
                    assert len(actual_shape) == len(
                        model_input.raw_shape
                    ), f"{actual_shape} != {model_input.raw_shape}"
                    for raw_shape_value, actual_shape_value in zip(
                        model_input.raw_shape, actual_shape
                    ):
                        if raw_shape_value is not None:
                            assert (
                                raw_shape_value == actual_shape_value
                            ), f"{model_input.raw_shape} != {actual_shape}"
            elif model_input.column_type in [
                DataColumnType.CATEGORICAL,
                DataColumnType.BOOLEAN,
            ]:
                x = x.astype(int)
            inputs.append(x)
            if model_input.column_type == DataColumnType.IMAGE:
                assert (
                    len(x.shape) >= 3
                ), f"Expect image shape to be at least 3. Got {len(x.shape)}"
            # TODO: consider if we should cache the transformed value
            # The transformation class could handle caching.
            # Fixed transform should have caching, but not random augmentations.
        targets = []
        for model_target in self.model_targets:
            raw_value = row_dict[model_target.key]
            x = raw_value
            for t in model_target.transforms or []:
                x = t.transform([x])[0]
            if type(x) in [int, float]:
                x = [x]
            x = np.array(x)
            # if len(x.shape) == 0:
            #     x = x.reshape(1)
            if model_target.column_type in [
                DataColumnType.CATEGORICAL,
                DataColumnType.BOOLEAN,
            ]:
                x = x.astype(int)
            targets.append(x)
        LOGGER.debug(f"prepared inputs and targets for example {i}")
        LOGGER.debug(f"inputs: {len(inputs)}, targets: {len(targets)}")
        for j, t in enumerate(inputs):
            LOGGER.debug(f"input {j}: {t.shape}, {t.dtype}")
        for j, t in enumerate(targets):
            LOGGER.debug(f"target {j}: {t.shape}, {t.dtype}")
        return inputs, targets, row_dict.get("sample_weight", 1)

    def get_model_spec(self):
        return ModelSpec(self.model_inputs, self.model_targets)

    def check(self) -> None:
        ensure_dataset_shapes_and_types(dataset=self, model_spec=self.get_model_spec())

    # TODO: get_batch_from_range() can be a helper function like collate().
    # But it should be called in the DataLoader class or a tf.data.Dataset class.
    def get_batch_from_range(self, begin_idx: int, end_idx: int):
        inputs_batch = [list() for _ in range(len(self.model_inputs))]
        targets_batch = [list() for _ in range(len(self.model_targets))]
        sample_weight_batch = []
        for j in range(begin_idx, end_idx):
            try:
                # TODO: add a test that raises exception in __getitem__.
                inputs_one_example, targets_one_example, sample_weight = self[j]
            except IOError as e:
                LOGGER.error(
                    f"Missing data for example {j}. Using data from {self.data_rows}"
                )
                LOGGER.error(e)
                continue
            assert len(inputs_one_example) == len(self.model_inputs)
            assert len(targets_one_example) == len(self.model_targets)

            sample_weight_batch.append(sample_weight)
            for input_idx, array in enumerate(inputs_one_example):
                # Standardize the array.
                array = np.array(array)
                if self.model_inputs[input_idx].column_type == DataColumnType.BOOLEAN:
                    array = array.astype(int)
                if np.size(array) == 1:
                    array = array.reshape((1,))
                inputs_batch[input_idx].append(array)
                assert (
                    len(inputs_batch[input_idx]) <= (j - begin_idx) + 1
                ), f"{len(inputs_batch[input_idx])}, j={j}, begin_idx={begin_idx}, input_idx={input_idx}"
            for target_idx, array in enumerate(targets_one_example):
                array = np.array(array)
                if self.model_targets[target_idx].column_type == DataColumnType.BOOLEAN:
                    array = array.astype(int)
                if np.size(array) == 1:
                    array = array.reshape((1,))
                if (
                    self.model_targets[target_idx].column_type
                    == DataColumnType.CATEGORICAL
                ):
                    # TODO: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
                    # For pytorch, it is recommended to use class index instead of class probs.
                    if self.use_one_hot_for_categorical:
                        array = one_hot(
                            array[0], self.model_targets[target_idx].n_categories
                        )
                targets_batch[target_idx].append(array)
        assert len(inputs_batch) == len(self.model_inputs)
        if len(sample_weight_batch) == 0:
            return None, None, None
        inputs_batch = [np.stack(arrays) for arrays in inputs_batch]
        targets_batch = [np.stack(arrays) for arrays in targets_batch]
        sample_weight_batch = np.array(sample_weight_batch, dtype=np.float32)
        return inputs_batch, targets_batch, sample_weight_batch

    def fit(self):
        """Calling individual fit functions of input transforms.

        e.g. sklearn preprocessors
        """
        pass

    def transform(self):
        pass

    def to_keras_sequence(self):
        raise NotImplementedError

    def to_torch_dataset(self):
        raise NotImplementedError
