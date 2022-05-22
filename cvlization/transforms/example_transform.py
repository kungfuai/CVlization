import numpy as np
from typing import Callable, List, Union

try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable

from ..specs.data_column import DataColumn, DataColumnType
from ..specs import ModelInput, ModelTarget


@runtime_checkable
class AugmenterForImageAndTargets(Protocol):
    def transform_image_and_targets(
        self,
        image: np.ndarray,
        bboxes: np.ndarray = None,
        mask: np.ndarray = None,
        keypoints: np.ndarray = None,
    ) -> dict:
        ...


class ExampleTransform:
    """
    Wraps an image augmentation function, and applies it to inputs and targets which
    can have flexible structures.

    Main method: `transform_example` retrains the structure of input arrays.

    It assumes `transform_image_and_targets` returns a dict with the following keys:
    "image", "bboxes", "mask", "keypoints".

    The `column_type_to_string_key` method defines the key names for the
    different data column types. The default implementation is:

    ```
        if column_type == DataColumnType.IMAGE:
            return "image"
        elif column_type == DataColumnType.BOUNDING_BOXES:
            return "bboxes"
        elif column_type == DataColumnType.MASKS:
            return "mask"
        elif column_type == DataColumnType.KEYPOINTS:
            return "keypoints"
    ```
    """

    def __init__(
        self,
        model_inputs: List[ModelInput],
        model_targets: List[ModelTarget],
        image_augmentation: AugmenterForImageAndTargets = None,
    ):
        self._model_inputs = model_inputs
        self._model_targets = model_targets
        self._image_augmentation = image_augmentation

    def __call__(self, args, **kwargs):
        self.transform_example(*args, **kwargs)

    def transform_image_and_targets(
        self, image, bboxes=None, mask=None, keypoints=None
    ):
        assert (
            self._image_augmentation is not None
        ), f"{self} has no image augmentation set"
        return self._image_augmentation.transform_image_and_targets(
            image, bboxes=bboxes, mask=mask, keypoints=keypoints
        )

    def noop_transform_image_and_targets(
        self,
        image,
        bboxes=None,
        mask=None,
        keypoints=None,
    ) -> dict:
        output = {"image": image}
        if bboxes is not None:
            output["bboxes"] = bboxes
        if mask is not None:
            output["mask"] = mask
        if keypoints is not None:
            output["keypoints"] = keypoints
        return output

    def get_model_inputs(self) -> List[ModelInput]:
        return self._model_inputs

    def get_model_targets(self) -> List[ModelTarget]:
        return self._model_targets

    def get_data_columns(self):
        ...

    def transform_example(self, example):
        """
        `example` is the output from a map style or iterable dataset.
        It can be of various types.

        It is expected to be a single example rather than a mini-batch.

        The same geometric augmentation should be applied to both the image
        and the target, keeping them consistent.

        Returns:
            `transformed_example`, with the same type as `example`.
        """
        self._assert_type_and_length(example)
        return self._first_not_empty_result(
            example,
            transform_funcs=[
                self._transform_example_tuple_or_list,
                self._transform_example_dict,
            ],
        )

    def _first_not_empty_result(self, x, transform_funcs: List[Callable]):
        for func in transform_funcs:
            result = func(x)
            if result is not None:
                return result

    def _assert_type_and_length(self, example):
        if isinstance(example, tuple):
            assert len(example) in [
                2,
                3,
            ], f"example tuple must have length 2 or 3, not {len(example)}"
        elif isinstance(example, list):
            assert len(example) in [
                2,
                3,
            ], f"example list must have length 2 or 3, not {len(example)}"
        else:
            assert isinstance(
                example, dict
            ), f"example must be a tuple, list or dict, not {type(example)}"

    def column_type_to_string_key(self, column_type: DataColumnType) -> str:
        if column_type == DataColumnType.IMAGE:
            return "image"
        elif column_type == DataColumnType.BOUNDING_BOXES:
            return "bboxes"
        elif column_type == DataColumnType.MASKS:
            return "mask"
        elif column_type == DataColumnType.KEYPOINTS:
            return "keypoints"
        return None

    def _inputs_targets_to_dict(
        self, inputs: Union[list, tuple], targets: Union[list, tuple]
    ) -> dict:
        model_inputs = self.get_model_inputs()
        model_targets = self.get_model_targets()
        transform_input_dict = {}
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            pass
        else:
            assert (
                hasattr(inputs, "numpy")
                or isinstance(inputs, np.ndarray)
                or isinstance(inputs, int)
                or isinstance(inputs, float)
            ), f"targets is not a list or tuple, so a single target tensor is expected. Found {type(inputs)}"
            inputs = [np.array(inputs)]

        if isinstance(targets, list) or isinstance(targets, tuple):
            pass
        else:
            assert (
                hasattr(targets, "numpy")
                or isinstance(targets, np.ndarray)
                or isinstance(targets, int)
                or isinstance(targets, float)
            ), f"targets is not a list or tuple, so a single target tensor is expected. Found {type(targets)}"
            targets = [np.array(targets)]

        assert len(inputs) == len(
            model_inputs
        ), f"inputs length mismatch. Found {len(inputs)}, expecting {len(model_inputs)} according declared model_inputs spec."
        assert len(targets) == len(
            model_targets
        ), f"targets length mismatch. Found {len(targets)}, expecting {len(model_targets)} according declared model_targets spec."
        for data_column, array in zip(
            model_inputs + model_targets, list(inputs) + list(targets)
        ):
            key = self.column_type_to_string_key(data_column.column_type)
            if key:
                transform_input_dict[key] = array

        return transform_input_dict

    def _dict_to_inputs_targets(self, transform_output_dict: dict, inputs, targets):
        """Recover the original structure of inputs and targets."""
        model_inputs = self.get_model_inputs()
        model_targets = self.get_model_targets()

        if isinstance(inputs, list) or isinstance(inputs, tuple):
            transformed_inputs = []
            for data_column, array in zip(model_inputs, inputs):
                key = self.column_type_to_string_key(data_column.column_type)
                if key:
                    transformed_inputs.append(transform_output_dict[key])
                else:
                    transformed_inputs.append(array)
            transformed_inputs = type(inputs)(transformed_inputs)
        else:
            data_column = model_inputs[0]
            key = self.column_type_to_string_key(data_column.column_type)
            if key:
                transformed_inputs = transform_output_dict[key]
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            assert type(transformed_inputs) == type(
                inputs
            ), f"{type(transformed_inputs)} != {type(inputs)}"

        if isinstance(targets, list) or isinstance(targets, tuple):
            transformed_targets = []
            for data_column, array in zip(model_targets, targets):
                key = self.column_type_to_string_key(data_column.column_type)
                if key and (key in transform_output_dict):
                    transformed_targets.append(transform_output_dict[key])
                else:
                    transformed_targets.append(array)
            transformed_targets = type(targets)(transformed_targets)
        else:
            data_column = model_targets[0]
            key = self.column_type_to_string_key(data_column.column_type)
            if key:
                transformed_targets = transform_output_dict[key]
            else:
                transformed_targets = targets
        assert type(transformed_targets) == type(
            targets
        ), f"{type(transformed_targets)} != {type(targets)}"

        return transformed_inputs, transformed_targets

    def _transform_example_tuple_or_list(self, example):
        """inputs, targets, ..."""
        if (isinstance(example, tuple) or isinstance(example, list)) and len(
            example
        ) >= 2:
            inputs, targets = example[0], example[1]
            transform_input_dict = self._inputs_targets_to_dict(inputs, targets)
            try:
                transform_output_dict = self.transform_image_and_targets(
                    **transform_input_dict
                )
            except Exception as e:
                raise Exception(
                    f"Exception while transforming {list(transform_input_dict.keys())}. Exception: {e}"
                )
            inputs, targets = self._dict_to_inputs_targets(
                transform_output_dict=transform_output_dict,
                inputs=inputs,
                targets=targets,
            )
            return type(example)([inputs, targets]) + example[2:]

    def _transform_example_dict(self, example):
        if isinstance(example, dict):
            transform_input_dict = {}
            data_columns: List[DataColumn] = (
                self.get_model_inputs() + self.get_model_targets()
            )
            for data_column in data_columns:
                transform_input_key = self.column_type_to_string_key(
                    data_column.column_type
                )
                if transform_input_key:
                    transform_input_dict[transform_input_key] = example[data_column.key]
            transform_output_dict = self.transform_image_and_targets(
                **transform_input_dict
            )
            transformed_example = {k: v for k, v in example.items()}
            for data_column in data_columns:
                transform_output_key = self.column_type_to_string_key(
                    data_column.column_type
                )
                if transform_output_key:
                    transformed_example[data_column.key] = transform_output_dict[
                        transform_output_key
                    ]
            return transformed_example
