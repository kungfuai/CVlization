import numpy as np
import pytest
from cvlization.specs import ModelSpec, ModelInput, ModelTarget
from cvlization.specs import DataColumnType
from cvlization.specs import ensure_dataset_shapes_and_types, MapLike


@pytest.fixture(name="image_classification_dataset")
def mock_image_classification_dataset():
    class MockImageClassificationDataset:
        def __getitem__(self, index):
            image_input = np.zeros((100, 100, 3))
            inputs = [image_input]
            label_target = 1
            targets = [label_target]
            return inputs, targets

        def __len__(self):
            return 10

    ds = MockImageClassificationDataset()
    assert isinstance(ds, MapLike)
    return ds


@pytest.fixture()
def image_classfication_model_spec():
    return ModelSpec(
        model_inputs=[
            ModelInput(
                key="image", column_type=DataColumnType.IMAGE, raw_shape=[None, None, 3]
            )
        ],
        model_targets=[ModelTarget(key="label", column_type=DataColumnType.BOOLEAN)],
    )


@pytest.fixture(name="object_detection_dataset")
def mock_object_detection_dataset():
    class MockObjectDetectionDataset:
        def __getitem__(self, index):
            image_input = np.zeros((100, 100, 3))
            inputs = [image_input]
            label_target = np.ones((3, 1))
            bboxes_target = np.ones((3, 4))
            targets = [label_target, bboxes_target]
            return inputs, targets

        def __len__(self):
            return 10

    return MockObjectDetectionDataset()


@pytest.fixture(name="object_detection_dataset_wrong_sequence_size")
def mock_object_detection_dataset_with_wrong_sequence_size():
    class MockObjectDetectionDataset:
        def __getitem__(self, index):
            image_input = np.zeros((100, 100, 3))
            inputs = [image_input]
            label_target = np.ones((3, 1))
            bboxes_target = np.ones((11, 4))
            targets = [label_target, bboxes_target]
            return inputs, targets

        def __len__(self):
            return 20

    return MockObjectDetectionDataset()


@pytest.fixture()
def object_detection_model_spec():
    return ModelSpec(
        model_inputs=[
            ModelInput(
                key="image", column_type=DataColumnType.IMAGE, raw_shape=[None, None, 3]
            )
        ],
        model_targets=[
            ModelTarget(
                key="bbox_labels",
                column_type=DataColumnType.BOOLEAN,
                sequence=True,
                # The batch axis is not included in raw_shape.
                # The sequence axis is included in raw_shape.
                raw_shape=[None, 1],
            ),
            ModelTarget(
                key="bboxes",
                sequence=True,
                column_type=DataColumnType.BOUNDING_BOXES,
                raw_shape=[None, 4],
            ),
        ],
    )


def test_model_spec_can_typecheck_a_dataset(
    image_classfication_model_spec,
    image_classification_dataset,
    object_detection_dataset,
    object_detection_model_spec,
    object_detection_dataset_wrong_sequence_size,
):
    ensure_dataset_shapes_and_types(
        dataset=image_classification_dataset,
        model_spec=image_classfication_model_spec,
    )

    with pytest.raises(AssertionError) as excinfo:
        ensure_dataset_shapes_and_types(
            dataset=object_detection_dataset,
            model_spec=image_classfication_model_spec,
        )
    assert "Expected 1 targets, but got 2" == str(excinfo.value)

    ensure_dataset_shapes_and_types(
        dataset=object_detection_dataset,
        model_spec=object_detection_model_spec,
    )

    with pytest.raises(AssertionError) as excinfo:
        ensure_dataset_shapes_and_types(
            dataset=object_detection_dataset_wrong_sequence_size,
            model_spec=object_detection_model_spec,
        )
    assert (
        """Sequences do not have the same length. Shapes: [('bbox_labels', (3, 1)), ('bboxes', (11, 4))]"""
        == str(excinfo.value)
    )
