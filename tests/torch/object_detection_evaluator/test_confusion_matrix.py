from cvlization.torch.evaluation.object_detection_evaluator.confusion_matrix import ConfusionMatrix


def test_init_empty():
    cm = ConfusionMatrix()
    assert cm.true_positives == 0
    assert cm.false_negatives == 0
    assert cm.true_positives == 0


def test_init_with_vals():
    cm = ConfusionMatrix(true_positives=5, false_negatives=1, false_positives=2)
    assert cm.true_positives == 5
    assert cm.false_negatives == 1
    assert cm.false_positives == 2


def test_precision_recall_properties():
    cm = ConfusionMatrix(true_positives=5, false_negatives=1, false_positives=2)
    assert cm.precision == 5 / (5 + 2)
    assert cm.recall == 5 / (5 + 1)
    assert round(cm.f1, 3) == 0.769
