import numpy as np
import pytest
from cvlization.transforms.example_transform import ExampleTransform
from cvlization.specs import ModelInput, ModelTarget, ModelSpec, DataColumnType
from cvlization.specs.prediction_tasks import ObjectDetection


class MyTransform(ExampleTransform):
    def transform_image_and_targets(
        self, image, bboxes=None, mask=None, keypoints=None
    ):
        return self.noop_transform_image_and_targets(
            image, bboxes=bboxes, mask=mask, keypoints=keypoints
        )


@pytest.fixture
def obj_det_transform() -> ExampleTransform:
    task = ObjectDetection()

    return MyTransform(
        model_inputs=task.get_model_inputs(), model_targets=task.get_model_targets()
    )


def test_obj_det_transform(obj_det_transform):
    image = np.zeros((100, 100, 3))
    bboxes = np.zeros((7, 4))
    bbox_class = np.zeros((7,))
    inputs = image
    targets = (bboxes, bbox_class)
    example = (inputs, targets)
    aug_inputs, aug_targets = obj_det_transform.transform_example(example)
    assert aug_inputs.shape == (100, 100, 3)
    assert isinstance(aug_targets, tuple)
    assert aug_targets[0].shape == (7, 4)
    assert aug_targets[1].shape == (7,)

    image = np.zeros((100, 100, 3))
    bboxes = np.zeros((7, 4))
    bbox_class = np.zeros((7,))
    inputs = image
    targets = (bboxes, bbox_class)
    sample_weights = [[1], [1]]
    example = (inputs, targets, sample_weights)
    aug_inputs, aug_targets, sample_weights = obj_det_transform.transform_example(
        example
    )
    assert aug_inputs.shape == (100, 100, 3)
    assert isinstance(aug_targets, tuple)
    assert aug_targets[0].shape == (7, 4)
    assert aug_targets[1].shape == (7,)
    assert sample_weights == [[1], [1]]

    inputs = (image,)
    targets = (bboxes, bbox_class)
    example = (inputs, targets)
    aug_inputs, aug_targets = obj_det_transform.transform_example(example)
    assert isinstance(aug_inputs, tuple)
    assert aug_inputs[0].shape == (100, 100, 3)
    assert isinstance(aug_targets, tuple)
    assert aug_targets[0].shape == (7, 4)
    assert aug_targets[1].shape == (7,)

    inputs = image
    targets = [bboxes, bbox_class]
    example = (inputs, targets)
    aug_output = obj_det_transform.transform_example(example)
    assert isinstance(aug_output, tuple)
    aug_inputs, aug_targets = aug_output
    assert isinstance(aug_inputs, np.ndarray)
    assert aug_inputs.shape == (100, 100, 3)
    assert isinstance(aug_targets, list)
    assert aug_targets[0].shape == (7, 4)
    assert aug_targets[1].shape == (7,)

    inputs = [image]
    targets = [bboxes, bbox_class]
    example = [inputs, targets]
    aug_output = obj_det_transform.transform_example(example)
    assert isinstance(aug_output, list)
    [aug_inputs, aug_targets] = aug_output
    assert isinstance(aug_inputs, list)
    assert aug_inputs[0].shape == (100, 100, 3)
    assert isinstance(aug_targets, list)
    assert aug_targets[0].shape == (7, 4)
    assert aug_targets[1].shape == (7,)

    example = {"image": image, "bbox": bboxes, "bbox_class": bbox_class}
    aug_output = obj_det_transform.transform_example(example)
    assert isinstance(aug_output, dict)
    aug_image, aug_bbox = aug_output["image"], aug_output["bbox"]
    assert isinstance(aug_image, np.ndarray)
    assert aug_image.shape == (100, 100, 3)
    assert isinstance(aug_bbox, np.ndarray)
    assert aug_bbox.shape == (7, 4)


def test_obj_det_transform_mismatch_tensor_count(obj_det_transform):
    image = np.zeros((100, 100, 3))
    bboxes = np.zeros((7, 4))
    bbox_class = np.zeros((7,))

    image_class = np.zeros((1,))
    inputs = [image]
    targets = [bboxes, bbox_class, image_class]
    example = [inputs, targets]
    with pytest.raises(AssertionError):
        _ = obj_det_transform.transform_example(example)


def test_obj_det_transform_invalid_targets():
    adjusted_obj_det_transform = MyTransform(
        model_inputs=[
            ModelInput(
                key="image", column_type=DataColumnType.IMAGE, raw_shape=(100, 100, 3)
            )
        ],
        model_targets=[
            ModelTarget(
                key="bbox",
                column_type=DataColumnType.BOUNDING_BOXES,
                raw_shape=[None, 4],
            ),
            ModelTarget(
                key="bbox_class",
                column_type=DataColumnType.CATEGORICAL,
                n_categories=10,
                raw_shape=[None, 1],
            ),
            ModelTarget(
                key="image_class",
                column_type=DataColumnType.CATEGORICAL,
                n_categories=5,
                raw_shape=[1],
            ),
        ],
    )

    image = np.zeros((100, 100, 3))
    bboxes = np.zeros((7, 4))
    bbox_class = np.zeros((7,))
    image_class = np.zeros((1,))
    inputs = [image]
    targets = [bboxes, bbox_class, image_class]
    example = (inputs, targets)
    aug_output = adjusted_obj_det_transform.transform_example(example)
    assert isinstance(aug_output, tuple)
    aug_inputs, aug_targets = aug_output
    assert isinstance(aug_inputs, list)
    assert aug_inputs[0].shape == (100, 100, 3)
    assert isinstance(aug_targets, list)
    assert aug_targets[0].shape == (7, 4)
    assert aug_targets[1].shape == (7,)
    assert aug_targets[2].shape == (1,)

    # Change the order of arrays.
    adjusted_obj_det_transform = MyTransform(
        model_inputs=[
            ModelInput(
                key="image", column_type=DataColumnType.IMAGE, raw_shape=(100, 100, 3)
            )
        ],
        model_targets=[
            ModelTarget(
                key="image_class",
                column_type=DataColumnType.CATEGORICAL,
                n_categories=5,
                raw_shape=[1],
            ),
            ModelTarget(
                key="bbox",
                column_type=DataColumnType.BOUNDING_BOXES,
                raw_shape=[None, 4],
            ),
            ModelTarget(
                key="bbox_class",
                column_type=DataColumnType.CATEGORICAL,
                n_categories=10,
                raw_shape=[None, 1],
            ),
        ],
    )
    image = np.zeros((100, 100, 3))
    bboxes = np.zeros((7, 4))
    bbox_class = np.zeros((7,))
    image_class = np.zeros((1,))
    inputs = [image]
    targets = [image_class, bboxes, bbox_class]
    example = (inputs, targets)
    aug_output = adjusted_obj_det_transform.transform_example(example)
    assert isinstance(aug_output, tuple)
    aug_inputs, aug_targets = aug_output
    assert isinstance(aug_inputs, list)
    assert aug_inputs[0].shape == (100, 100, 3)
    assert isinstance(aug_targets, list)
    assert aug_targets[1].shape == (7, 4)
    assert aug_targets[2].shape == (7,)
    assert aug_targets[0].shape == (1,)
