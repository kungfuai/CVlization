from dataclasses import dataclass
from typing import List

from ..data_column import DataColumn


@dataclass
class ModelInput(DataColumn):
    input_groups: List[str] = None

    @classmethod
    def DEFAULT_INPUT_GROUP(cls):
        return "default"

    def __post_init__(self):
        if self.input_groups is None:
            self.input_groups = [ModelInput.DEFAULT_INPUT_GROUP()]
        elif ModelInput.DEFAULT_INPUT_GROUP() not in self.input_groups:
            self.input_groups.append(ModelInput.DEFAULT_INPUT_GROUP())
