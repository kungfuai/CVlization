import torch
from cvlization.torch.evaluation.object_detection_evaluator.box_matcher import BoxMatcher
from tests.torch.object_detection_evaluator.box_matcher_features import (
    targets,
    detections,
    detections2,
    matched_output,
    unmatched_output,
    unmatched_output2,
    real_targets,
    real_detections,
)


def test_call():
    box_matcher = BoxMatcher()
    matched_list, unmatched_list = box_matcher(targets=targets, detections=detections)
    assert torch.isclose(torch.sum(matched_list[0]), torch.sum(matched_output[0]))
    assert torch.isclose(torch.sum(matched_list[1]), torch.sum(matched_output[1]))
    assert unmatched_list[0] == unmatched_output[0]
    assert unmatched_list[1] == unmatched_output[1]


def test_call2():
    box_matcher = BoxMatcher()
    matched_list, unmatched_list = box_matcher(targets=targets, detections=detections2)
    assert torch.isclose(torch.sum(matched_list[0]), torch.sum(matched_output[0]))
    assert torch.isclose(torch.sum(matched_list[1]), torch.sum(matched_output[1]))
    assert torch.isclose(torch.sum(unmatched_list[0]), torch.sum(unmatched_output2[0]))
    assert unmatched_list[1] == unmatched_output2[1]


def test_call_real():
    box_matcher = BoxMatcher()
    _, _ = box_matcher(targets=real_targets, detections=real_detections)
