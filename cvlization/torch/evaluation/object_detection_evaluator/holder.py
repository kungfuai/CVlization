from torch import Tensor
from typing import List, Optional

class Holder:
    """
    This object holds lists of matched and unmatched detections corresponding to a single class_id (label id)
    """

    def __init__(self, class_id: int) -> None:
        self.matched_list = []
        self.unmatched_list = []
        self.class_id = class_id

    def add(self, matched: List[Tensor], unmatched: List[Optional[Tensor]]) -> None:
        self.matched_list.extend(matched)
        self.unmatched_list.extend(unmatched)
