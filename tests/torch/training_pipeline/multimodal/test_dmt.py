import numpy as np
import pytest
from cvlization.torch.training_pipeline.multimodal.dmt_net import (
    DMTNet,
    ModelInput,
    InvalidInputError,
    Example,
)


def test_inference_for_vision_tasks():
    model = DMTNet()
    img = np.random.rand(3, 224, 224)
    bad_input = ModelInput(images=None)
    with pytest.raises(InvalidInputError):
        model.image_classification(bad_input)
    x = ModelInput(images=[img])
    output = model.image_classification(x)  # similar interface as UnifiedIO
    assert output.category is not None

    output = model.image_tagging(x)
    assert output.tags is not None  # Multi-label classification

    output = model.object_detection(x)
    assert output.boxes is not None

    output = model.instance_segmentation(x)
    assert output.masks is not None

    output = model.semantic_segmentation(x)
    assert output.masks is not None

    output = model.image_captioning(x)
    assert output.captions is not None

    output = model.panoptic_segmentation(x)
    assert output.masks is not None

    # This should not work. Need to specify bounding boxes as queries.
    x_with_box = ModelInput(images=[img], boxes=[(0, 0, 100, 100)])
    output = model.keypoint_detection(x_with_box)
    assert output.keypoints is not None

    output = model.line_segment_detection(x)
    assert output.line_segments is not None


def test_inference_for_nlp_tasks():
    model = DMTNet()
    text = "This is a test."
    bad_input = ModelInput(text=None)
    with pytest.raises(InvalidInputError):
        model.text_classification(bad_input)

    x = ModelInput(text=text)
    output = model.text_classification(x)
    assert output.category is not None

    output = model.text_tagging(x)
    assert output.tags is not None

    output = model.text_summarization(x)
    assert output.summary is not None

    output = model.text_translation(x)
    assert output.translation is not None

    output = model.text_question_answering(x)
    assert output.answer is not None

    output = model.text_generation(x)
    assert output.text is not None

    output = model.text_sentiment_analysis(x)
    assert output.sentiment is not None

    output = model.text_style_transfer(x)
    assert output.text is not None

    output = model.text_topic_modeling(x)
    assert output.topics is not None

    output = model.text_entity_recognition(x)
    assert output.entities is not None

    output = model.text_entity_relation_extraction(x)
    assert output.relations is not None


def test_inference_with_language_vision_tasks():
    model = DMTNet()
    img = np.random.rand(3, 224, 224)
    text = "This is a test."
    x = ModelInput(images=[img], text=text)
    output = model.visual_question_answering(x)
    assert output.answer is not None


def test_inference_with_mixed_modalities():
    model = DMTNet()
    img = np.random.rand(3, 224, 224)
    text = "This is a test."
    x = ModelInput(images=[img])
    output = model.image_classification(x)
    assert output.category is not None

    x = ModelInput(text=text)
    output = model.text_classification(x)
    assert output.category is not None


def test_training_with_mixed_modalities():
    model = DMTNet()
    batch = [
        Example(
            input=ModelInput(images=[np.random.rand(3, 224, 224)]),
            target={"category": "cat"},
            task="image_classification",
        ),
        Example(
            input=ModelInput(text="This is a test."),
            target={"category": "news"},
            task="text_classification",
        ),
    ]

    # TODO: collate

    # Now train the model.
    # train_step(model, batch)
    loss = model.forward_train(batch)
