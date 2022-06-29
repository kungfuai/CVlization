from torch import Tensor
from typing import List, Optional
from .holder import Holder


class MatchHolder:
    """
    Holds a list of Holder objects
    """

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.holder_list = self._make_holder_list(num_classes=self.num_classes)

    def _make_holder_list(self, num_classes: int) -> List[Holder]:
        holder_list = []
        for i in range(num_classes):
            holder_list.append(Holder(class_id=i))
        return holder_list

    def add(
        self, matched: List[Tensor], unmatched: List[Optional[Tensor]], class_id: int
    ) -> None:
        self.holder_list[class_id].add(matched=matched, unmatched=unmatched)

    def reset(self):
        self.holder_list = self._make_holder_list(num_classes=self.num_classes)
