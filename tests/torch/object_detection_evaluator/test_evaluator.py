from cvlization.torch.evaluation.object_detection_evaluator.evaluator import Evaluator
from tests.torch.object_detection_evaluator.fixtures import (
    detections2_plus_one,
    detections2_with_empty,
    detections2,
    real_detections,
    real_targets,
    targets_plus_one,
    targets_with_empty,
    targets,
)
from tests.torch.object_detection_evaluator.multi_class_fixtures import (
    detections2_plus_one_cls0,
    detections2_with_empty_cls1,
    targets_plus_one_cls1,
    targets_with_empty_cls0,
)


def test_init():
    evaluator = Evaluator()
    assert evaluator.metrics.f1_max == 0
    assert evaluator.metrics.score_at_f1_max == 0
    assert evaluator.metrics.mean_average_precision == 0


def test_calculate():
    """
    # TODO Calculate by hand
    """
    evaluator = Evaluator()
    evaluator.add(targets=targets, detections=detections2)
    evaluator.calculate()

    assert round(evaluator.metrics.f1_max, 3) == 1.000
    assert round(evaluator.metrics.mean_average_precision, 3) == 0.750  # 52

    # test adding more targets and dets
    evaluator.add(targets=targets, detections=detections2)
    evaluator.calculate()

    assert round(evaluator.metrics.f1_max, 3) == 1.000
    assert round(evaluator.metrics.mean_average_precision, 3) == 0.750  # 52


def test_reset():
    evaluator = Evaluator()
    evaluator.add(targets=targets, detections=detections2)
    evaluator.calculate()

    assert round(evaluator.metrics.f1_max, 3) == 1.00
    assert round(evaluator.metrics.mean_average_precision, 3) == 0.750

    # reset
    evaluator.reset()

    # add again
    evaluator.add(targets=targets, detections=detections2)
    evaluator.calculate()

    assert round(evaluator.metrics.f1_max, 3) == 1.000
    assert round(evaluator.metrics.mean_average_precision, 3) == 0.75


def test_real_example():
    """These values are real. The model was not trained yet so it spit out random detections.
    I just wanted to verify that water could flow through the pipes with real detections."""
    evaluator = Evaluator()
    evaluator.add(targets=real_targets, detections=real_detections)
    evaluator.calculate()
    f1_max = evaluator.metrics.f1_max
    assert round(f1_max, 3) == 0.000
    ap = evaluator.metrics.mean_average_precision
    assert ap == 0.000


def test_empty_targets():
    """
    # TODO Calculate by hand
    """
    evaluator = Evaluator()
    evaluator.add(targets=targets_with_empty, detections=detections2_plus_one)
    evaluator.calculate()
    f1_max = evaluator.metrics.f1_max
    score_at_f1_max = evaluator.metrics.score_at_f1_max
    assert round(score_at_f1_max, 2) == 1.00
    assert round(f1_max, 2) == 1.00
    map = evaluator.metrics.mean_average_precision
    assert round(map, 3) == 0.583


def test_empty_detections():
    """
    # TODO Calculate by hand
    """
    evaluator = Evaluator()
    evaluator.add(targets=targets_plus_one, detections=detections2_with_empty)
    evaluator.calculate()
    f1_max = evaluator.metrics.f1_max
    score_at_f1_max = evaluator.metrics.score_at_f1_max
    assert round(score_at_f1_max, 2) == 1.00
    assert round(f1_max, 3) == 1.000
    map = evaluator.metrics.mean_average_precision
    assert round(map, 3) == 0.75


def test_multi_class():
    # TODO Need to definitively test multi-class
    evaluator = Evaluator(num_classes=2)

    targets = targets_with_empty_cls0 + targets_plus_one_cls1
    detections = detections2_plus_one_cls0 + detections2_with_empty_cls1
    evaluator.add(targets=targets, detections=detections)
    evaluator.calculate()
    f1_max = evaluator.metrics.f1_max
    assert round(f1_max, 3) == 1.000
    map = evaluator.metrics.mean_average_precision
    assert round(map, 3) == 0.667


def test_voc_multiclass_dets():
    evaluator = Evaluator(num_classes=20)
    from tests.torch.object_detection_evaluator.real_multiclass_fixures import voc_dets

    evaluator.add(targets=voc_dets, detections=voc_dets)


def test_num_images():
    evaluator = Evaluator(num_classes=20)
    from tests.torch.object_detection_evaluator.real_multiclass_fixures import voc_dets

    evaluator.add(targets=voc_dets, detections=voc_dets)
    assert evaluator.metrics._num_images == 20
