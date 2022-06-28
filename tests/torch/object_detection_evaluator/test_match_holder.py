from cvlization.torch.evaluation.object_detection_evaluator.match_holder import MatchHolder
from cvlization.torch.evaluation.object_detection_evaluator.holder import Holder


def test_init():
    match_holder = MatchHolder(num_classes=4)
    assert len(match_holder.holder_list) == 4
    assert isinstance(match_holder.holder_list[0], Holder)


def test_add():
    match_holder = MatchHolder(num_classes=2)
    match_holder.add(matched=[], unmatched=[], class_id=0)


def test_reset():
    match_holder = MatchHolder(num_classes=1)
    match_holder.holder_list = [50]
    assert not isinstance(match_holder.holder_list[0], Holder)
    match_holder.reset()
    assert isinstance(match_holder.holder_list[0], Holder)
