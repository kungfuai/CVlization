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
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional

from .model_input import ModelInput
from .model_target import ModelTarget


class ModelProvider(str, Enum):
    KERAS_APPLICATIONS = "keras_applications"
    TF_HUB = "tf_hub"

    TORCHVISION = "torchvision"
    TORCH_HUB = "torch_hub"
    TIMM = "timm"

    HUGGINGFACE = "huggingface"

    CVLIZATION = "cvlization"


@dataclass
class ModelSpec:
    model_inputs: Optional[List[ModelInput]] = None
    model_targets: Optional[List[ModelTarget]] = None
    net: Optional[str] = None
    provider: Optional[str] = None

    # loss_function_included_in_model: bool = False
    image_backbone: str = None  # e.g. "resnet50"
    image_pool: str = "avg"  # "avg", "max", "flatten"
    dense_layer_sizes: List[int] = field(default_factory=list)
    input_shape: List[int] = field(
        default_factory=lambda: [None, None, 3]
    )  # e.g. [224, 224, 3]
    dropout: float = 0
    pretrained: bool = False
    permute_image: bool = False
    customize_conv1: bool = False

    def get_model_inputs(self) -> List[ModelInput]:
        return self.model_inputs

    def get_model_targets(self) -> List[ModelTarget]:
        return self.model_targets

    def __repr__(self) -> str:
        return f"""
        ModelSpec(
            model_inputs={self.model_inputs},
            model_targets={self.model_targets},
        )
        """
