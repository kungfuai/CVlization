from cvlization.torch.evaluation.object_detection_evaluator.holder import Holder


def test_init():
    holder = Holder(class_id=1)
    assert holder.class_id == 1


def test_add():
    holder = Holder(class_id=1)
    assert holder.matched_list == []
    assert holder.unmatched_list == []
    holder.add(matched=[1], unmatched=[1, 2])
    assert holder.matched_list == [1]
    assert holder.unmatched_list == [1, 2]
