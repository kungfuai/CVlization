"""
ModelSpec serves as type annotations for machine learning models and machine
learning datasets.

It aims to maintain a stable set of options for data types, shapes, metrics and losses.
Different data types have different semantic meanings and are consumed differently. For example, 
if you want to visualize an array of type BOUNDING_BOX, it would be treated 
differently than an array of type CATEGORICAL.

ModelSpec does not concern itself with how exactly the type annotations will be
consumed in visualization, model training, etc.

TODO: keep track of the re-usable parts of the model. This may have to be done
for specific frameworks (e.g. to keep track of individual layers).
TODO: add spec for multi-path model: multiple paths to get from
model inputs to model targets, besides the main path encoders->aggregators->heads.
"""
from dataclasses import dataclass
from typing import List, Optional

from .model_input import ModelInput
from .model_target import ModelTarget


@dataclass
class ModelSpec:
    model_inputs: Optional[List[ModelInput]] = None
    model_targets: Optional[List[ModelTarget]] = None

    def get_model_inputs(self) -> List[ModelInput]:
        return self.model_inputs

    def get_model_targets(self) -> List[ModelTarget]:
        return self.model_targets
