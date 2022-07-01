from cvlization.torch.evaluation.object_detection_evaluator.confusion_matrix_maker import ConfusionMatrixMaker
from tests.torch.object_detection_evaluator.fixtures import matched_output, unmatched_output2


def test_call():
    confusion_matrix_maker = ConfusionMatrixMaker()
    confusion_matrix = confusion_matrix_maker(
        matched_list=matched_output, unmatched_list=unmatched_output2
    )
    assert confusion_matrix.true_positives == 4
    assert confusion_matrix.false_negatives == 2
    assert confusion_matrix.false_positives == 1
