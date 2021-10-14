from typing import List, Optional
from dataclasses import dataclass, field as dataclass_field
import logging
import numpy as np

from ..data.utils import one_hot
from .data_column import DataColumnType
from .model_input import ModelInput
from .model_target import ModelTarget
from .data_rows import DataRows

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


@dataclass
class MLDataset:
    """A MLDataset is a dataset that is potentially trainable.

    It manages the list of inputs and the list of targets.
    which in turn manage the parameters of feature transforms.

    How to use it:
    1. Declare model inputs and targets, together with feature transforms.
    2. Instruct how each training example is fetched, and returned as a dict.
    3. Create a model_fn that computes output tensors from input tensors.
    4. KerasTrainer creates a custom keras model based on model inputs, targets,
        transforms, and arbitrary neural net architecture options.
    """

    data_rows: DataRows
    model_inputs: List[ModelInput]
    model_targets: List[ModelTarget]
    batch_size: Optional[int] = 2

    def get_row(self, i: int):
        return self.data_rows[i]

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
            inputs.append(x)
            # TODO: consider if we should cache the transformed value
            # The transformation class could handle caching.
            # Fixed transform should have caching, but not random augmentations.
        targets = []
        for model_target in self.model_targets:
            raw_value = row_dict[model_target.key]
            x = raw_value
            for t in model_target.transforms or []:
                x = t.transform([x])[0]
            targets.append(x)
        return inputs, targets

    def get_batch_from_range(self, begin_idx: int, end_idx: int):
        inputs_batch = [list() for _ in range(len(self.model_inputs))]
        targets_batch = [list() for _ in range(len(self.model_targets))]
        for j in range(begin_idx, end_idx):
            inputs_one_example, targets_one_example = self[j]
            assert len(inputs_one_example) == len(self.model_inputs)
            assert len(targets_one_example) == len(self.model_targets)

            for input_idx, array in enumerate(inputs_one_example):
                # Standardize the array.
                array = np.array(array)
                if self.model_inputs[input_idx].column_type == DataColumnType.BOOLEAN:
                    array = array.astype(int)
                if np.size(array) == 1:
                    array = array.reshape((1,))
                inputs_batch[input_idx].append(array)
                assert (
                    len(inputs_batch[input_idx]) == (j - begin_idx) + 1
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
                    array = one_hot(
                        array[0], self.model_targets[target_idx].n_categories
                    )
                targets_batch[target_idx].append(array)
        assert len(inputs_batch) == len(self.model_inputs)
        inputs_batch = [np.stack(arrays) for arrays in inputs_batch]
        targets_batch = [np.stack(arrays) for arrays in targets_batch]
        return inputs_batch, targets_batch

    def fit(
        self,
    ):
        """Calling individual fit functions of input transforms.

        e.g. sklearn preprocessors
        """
        pass

    def transform(self):
        pass


class TrainingLoop:
    pass
