from typing import List
from .model_input import ModelInput
from .model_target import ModelTarget

"""
TODO: keep track of the re-usable parts of the model. This may have to be done
for specific frameworks (e.g. to keep track of individual layers).
TODO: add spec for multi-path model: multiple paths to get from
model inputs to model targets, besides the main path encoders->aggregators->heads.
"""


class ModelSpec:
    def get_model_inputs(self) -> List[ModelInput]:
        raise NotImplementedError

    def get_model_targets(self) -> List[ModelTarget]:
        raise NotImplementedError
