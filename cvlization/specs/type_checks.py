from collections.abc import Iterable, Sequence
import logging
import numpy as np
from typing import List, Optional, Union, Dict, Tuple
from cvlization.specs import ModelSpec, ModelInput, ModelTarget
from cvlization.specs.data_column import DataColumn, DataColumnType

try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable


LOGGER = logging.getLogger(__name__)


@runtime_checkable
class MapLike(Protocol):
    def __getitem__(self, key):
        ...

    def __len__(self):
        ...


@runtime_checkable
class SelfCheckable(Protocol):
    def check(self):
        ...


@runtime_checkable
class CheckedWithModelSpec(Protocol):
    def get_model_spec(self):
        ...

    def check(self):
        ensure_dataset_shapes_and_types(dataset=self, model_spec=self.get_model_spec())


def ensure_array_shape_and_type(array, data_column: DataColumn):
    array = np.array(array)
    if data_column.column_type == DataColumnType.BOOLEAN:
        if not data_column.sequence:
            assert (
                array.size == 1
            ), f"Expected boolean array to have size 1, but got {array.size}"
        return

    if len(data_column.raw_shape) == len(array.shape):
        LOGGER.warning(
            f"Shape of array is {array.shape}, raw_shape of data column {data_column.key} is {data_column.raw_shape} (not including batch axis)"
        )
    else:
        assert len(data_column.raw_shape) == len(array.shape) - 1
        for expected_size, actual_size in zip(
            data_column.raw_shape[::-1], array.shape[::-1]
        ):
            if (
                expected_size is not None
                and expected_size > 0
                and expected_size != actual_size
            ):
                if actual_size is None and expected_size == 1:
                    # (batch, 1) vs. (batch,)
                    # TODO: is this really the right thing to do?
                    pass
                else:
                    raise TypeError(
                        f"Data column: {data_column.key}. Expected {expected_size} but got {actual_size}. "
                        f"Expected shape (ignoring batch axis): {data_column.raw_shape}, actual shape (with batch axis): {array.shape}"
                    )


def _gather_sequence_groups(
    example: tuple, model_spec: ModelSpec
) -> Dict[str, List[np.ndarray]]:
    sequence_groups = {}
    target_arrays = example[1]
    for target_array, model_target in zip(target_arrays, model_spec.model_targets):
        if model_target.sequence:
            group_key = str(model_target.sequence)
            sequence_groups.setdefault(group_key, []).append(
                {"array": target_array, "model_target": model_target}
            )
    return sequence_groups


def ensure_sequence_arrays_have_the_same_length(example: tuple, model_spec: ModelSpec):
    sequence_groups = _gather_sequence_groups(example, model_spec)
    for sequence_key, sequence_group in sequence_groups.items():
        # -2 is the sequence axis
        sequence_lengths = [item["array"].shape[-2] for item in sequence_group]
        for l in sequence_lengths:
            assert (
                l == sequence_lengths[0]
            ), f"Sequences do not have the same length. Shapes: {[(item['model_target'].key, item['array'].shape) for item in sequence_group]}"


def _cast_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, np.ndarray):
        return [x]
    if hasattr(x, "numpy"):
        return [x]
    raise TypeError(
        f"Expected list, tuple, numpy array, or tensor (numpy-able), but got {type(x)}"
    )


def ensure_example_shapes_and_types(example: tuple, model_spec: ModelSpec):
    """Ensure a training example or mini-batch has correct shapes and types."""
    if isinstance(example, list):
        example = tuple(example)
    assert isinstance(
        example, tuple
    ), f"Expected example to be a tuple, but got {type(example)}."
    # Either (inputs, targets) or (inputs, targets, sample_weights)
    assert len(example) == 2 or len(example) == 3
    if len(example) == 2:
        inputs, targets = example
        inputs, targets = (
            _cast_list(inputs),
            _cast_list(targets),
        )
        sample_weights = None
    else:
        inputs, targets, sample_weights = example
        inputs, targets, sample_weights = (
            _cast_list(inputs),
            _cast_list(targets),
            _cast_list(sample_weights),
        )
    assert len(inputs) == len(
        model_spec.model_inputs
    ), f"Expected {len(model_spec.model_inputs)} inputs, but got {len(inputs)}"
    assert len(targets) == len(
        model_spec.model_targets
    ), f"Expected {len(model_spec.model_targets)} targets, but got {len(targets)}"
    if sample_weights is not None:
        assert len(sample_weights) == len(model_spec.model_targets)
    for i, (input_array, data_column) in enumerate(
        zip(inputs, model_spec.model_inputs)
    ):
        ensure_array_shape_and_type(array=input_array, data_column=data_column)
    for target_array, data_column in zip(targets, model_spec.model_targets):
        ensure_array_shape_and_type(array=target_array, data_column=data_column)

    ensure_sequence_arrays_have_the_same_length(example=example, model_spec=model_spec)


def ensure_dataset_shapes_and_types(
    dataset: Union[Iterable, MapLike], model_spec: ModelSpec
):
    if isinstance(dataset, MapLike):
        n = len(dataset)
        assert n > 0
        idx = 0
        example = dataset[idx]
    elif isinstance(dataset, Iterable):
        example = next(iter(dataset))
    else:
        raise TypeError(
            f"Expected dataset to be an Iterable or a MapStyleDataset, but got {type(dataset)}. dir: {dir(dataset)}"
        )
    ensure_example_shapes_and_types(example, model_spec)
