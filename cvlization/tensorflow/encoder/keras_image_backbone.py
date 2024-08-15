import logging
import re
from tensorflow import keras
from typing import List

from ..net import convnext, simple_conv_net, resnet
from ..net import swin_transformer
try:
    from ..net import vit
except AttributeError:
    vit = None
    print("Warning: tf VIT models are not available.")


LOGGER = logging.getLogger(__name__)


def get_image_backbone_creators():
    # TODO: vit models assume 3 channels, so we need to add a check for 1 channel.

    return {
        "simple": lambda kwargs: simple_conv_net.SimpleConvNet(
            input_shape=kwargs["input_shape"],
            pooling=kwargs.get("pooling"),
        ),
        "resnet18v2d": lambda kwargs: resnet.ResNet18(
            input_shape=kwargs.get("input_shape", None),
            include_top=kwargs.get("include_top", False),
            final_pooling=kwargs.get("pooling"),
            residual_unit="v2d",
        ),
        "resnet18v2_smallimage": lambda kwargs: resnet.ResNet18(
            input_shape=kwargs.get("input_shape", None),
            include_top=kwargs.get("include_top", False),
            final_pooling=kwargs.get("pooling"),
            residual_unit="v2",
            initial_strides=(1, 1),
            initial_pooling=None,
        ),
        "resnet18v2": lambda kwargs: resnet.ResNet18(
            input_shape=kwargs.get("input_shape", None),
            include_top=kwargs.get("include_top", False),
            final_pooling=kwargs.get("pooling"),
            residual_unit="v2",
        ),
        "resnet18v1": lambda kwargs: resnet.ResNet18(
            input_shape=kwargs.get("input_shape", None),
            include_top=kwargs.get("include_top", False),
            final_pooling=kwargs.get("pooling"),
            residual_unit="v1",
        ),
        "resnet18v1d": lambda kwargs: resnet.ResNet18(
            input_shape=kwargs.get("input_shape", None),
            include_top=kwargs.get("include_top", False),
            final_pooling=kwargs.get("pooling"),
            residual_unit="v1d",
        ),
        "vit_b16": lambda kwargs: vit.vit_b16(
            image_size=tuple(kwargs.get("input_shape", [None, None, 3])[:2]),
            include_top=kwargs.get("include_top", False),
            pretrained=kwargs.get("pretrained", True),
            pretrained_top=False,
        ),
        "vit_b32": lambda kwargs: vit.vit_b32(
            image_size=tuple(kwargs.get("input_shape", [None, None, 3])[:2]),
            include_top=kwargs.get("include_top", False),
            pretrained=kwargs.get("pretrained", True),
            pretrained_top=False,
        ),
        "vit_l16": lambda kwargs: vit.vit_l16(
            image_size=tuple(kwargs.get("input_shape", [None, None, 3])[:2]),
            include_top=kwargs.get("include_top", False),
            pretrained=kwargs.get("pretrained", True),
            pretrained_top=False,
        ),
        "vit_l32": lambda kwargs: vit.vit_l32(
            image_size=tuple(kwargs.get("input_shape", [None, None, 3])[:2]),
            include_top=kwargs.get("include_top", False),
            pretrained=kwargs.get("pretrained", True),
            pretrained_top=False,
        ),
        "vit_custom_b": lambda kwargs: vit.vit_custom_b(
            image_size=tuple(kwargs.get("input_shape", [None, None, 3])[:2]),
            include_top=kwargs.get("include_top", False),
            pretrained=kwargs.get("pretrained", True),
            patch_size=kwargs.get("vit_patch_size", 32),
            num_channels=kwargs.get("in_chans"),
        ),
        "vit_custom_l": lambda kwargs: vit.vit_custom_l(
            image_size=tuple(kwargs.get("input_shape", [None, None, 3])[:2]),
            include_top=kwargs.get("include_top", False),
            pretrained=kwargs.get("pretrained", True),
            patch_size=kwargs.get("vit_patch_size", 32),
            num_channels=kwargs.get("in_chans"),
        ),
        "convnext_tiny_224": lambda kwargs: convnext.create_model(
            model_name="convnext_tiny_224",
            input_shape=tuple(kwargs.get("input_shape", [None, None, 3])[:2]),
            include_top=kwargs.get("include_top", False),
            pretrained=kwargs.get("pretrained", True),
        ),
        "convnext_small_224": lambda kwargs: convnext.create_model(
            model_name="convnext_small_224",
            input_shape=tuple(kwargs.get("input_shape", [None, None, 3])[:2]),
            include_top=kwargs.get("include_top", False),
            pretrained=kwargs.get("pretrained", True),
        ),
        "convnext_base_224": lambda kwargs: convnext.create_model(
            model_name="convnext_base_224",
            input_shape=tuple(kwargs.get("input_shape", [None, None, 3])[:2]),
            include_top=kwargs.get("include_top", False),
            pretrained=kwargs.get("pretrained", True),
        ),
        "convnext_base_384": lambda kwargs: convnext.create_model(
            model_name="convnext_base_384",
            input_shape=tuple(kwargs.get("input_shape", [None, None, 3])[:2]),
            include_top=kwargs.get("include_top", False),
            pretrained=kwargs.get("pretrained", True),
        ),
        "convnext_large_224": lambda kwargs: convnext.create_model(
            model_name="convnext_large_224",
            input_shape=tuple(kwargs.get("input_shape", [None, None, 3])[:2]),
            include_top=kwargs.get("include_top", False),
            pretrained=kwargs.get("pretrained", True),
        ),
        "convnext_large_384": lambda kwargs: convnext.create_model(
            model_name="convnext_large_384",
            input_shape=tuple(kwargs.get("input_shape", [None, None, 3])[:2]),
            include_top=kwargs.get("include_top", False),
            pretrained=kwargs.get("pretrained", True),
        ),
        "convnext_xlarge_224": lambda kwargs: convnext.create_model(
            model_name="convnext_xlarge_224",
            input_shape=tuple(kwargs.get("input_shape", [None, None, 3])[:2]),
            include_top=kwargs.get("include_top", False),
            pretrained=kwargs.get("pretrained", True),
        ),
        "convnext_xlarge_384": lambda kwargs: convnext.create_model(
            model_name="convnext_xlarge_384",
            input_shape=tuple(kwargs.get("input_shape", [None, None, 3])[:2]),
            include_top=kwargs.get("include_top", False),
            pretrained=kwargs.get("pretrained", True),
        ),
        "swin_tiny_224": lambda kwargs: swin_transformer.SwinTransformer(
            model_name="swin_tiny_224",
            include_top=kwargs.get("include_top", False),
            pretrained=kwargs.get("pretrained", True),
        ),
        "swin_small_224": lambda kwargs: swin_transformer.SwinTransformer(
            model_name="swin_small_224",
            include_top=kwargs.get("include_top", False),
            pretrained=kwargs.get("pretrained", True),
        ),
        "swin_base_224": lambda kwargs: swin_transformer.SwinTransformer(
            model_name="swin_base_224",
            include_top=kwargs.get("include_top", False),
            pretrained=kwargs.get("pretrained", True),
        ),
        "swin_base_384": lambda kwargs: swin_transformer.SwinTransformer(
            model_name="swin_base_384",
            include_top=kwargs.get("include_top", False),
            pretrained=kwargs.get("pretrained", True),
        ),
        "swin_large_224": lambda kwargs: swin_transformer.SwinTransformer(
            model_name="swin_large_224",
            include_top=kwargs.get("include_top", False),
            pretrained=kwargs.get("pretrained", True),
        ),
        "swin_large_384": lambda kwargs: swin_transformer.SwinTransformer(
            model_name="swin_large_384",
            include_top=kwargs.get("include_top", False),
            pretrained=kwargs.get("pretrained", True),
        ),
    }


def _create_model_keras_application(
    name: str,
    pretrained: bool = True,
    input_shape: List[int] = None,
    pooling: str = "avg",
    **kwargs,
) -> keras.models.Model:
    if hasattr(keras.applications, name):
        # Calling keras.applications API:
        # e.g. https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/ResNet50
        if len(input_shape or []) and (input_shape[-1] == 1):
            LOGGER.warning("Using single channel image with keras application.")
        create_model = getattr(keras.applications, name)
        weights = "imagenet" if pretrained else None
        if pooling == "flatten":
            pooling = None
        backbone = create_model(
            include_top=False, weights=weights, input_shape=input_shape, pooling=pooling
        )
        return backbone
    return None


def create_image_backbone(
    name: str,
    pretrained: bool = True,
    in_chans: int = 3,
    input_shape: List[int] = None,
    pooling: str = "avg",
    freeze: bool = False,
    vit_patch_size: int = None,
) -> keras.Model:

    input_shape = input_shape or [None, None, in_chans]

    if name.startswith("vit_"):
        if input_shape[-1] != 3:
            raise ValueError(f"VIT model requires 3 channels, got {input_shape[-1]}")

    kwargs = dict(
        name=name,
        pretrained=pretrained,
        input_shape=input_shape,
        pooling=pooling,
        vit_patch_size=vit_patch_size,
    )
    model_creator = get_image_backbone_creators().get(name)
    if model_creator is not None:
        backbone = model_creator(kwargs)
    else:
        backbone = _create_model_keras_application(**kwargs)
    if freeze:
        backbone.trainable = False
    return backbone


def _is_valid_keras_application_model_name(s: str) -> bool:
    if len(s) < 5:
        return False
    if re.match(r"[A-Z].*", s):
        return True
    return False


def image_backbone_names() -> List[str]:
    keras_application_model_names = [
        x for x in dir(keras.applications) if _is_valid_keras_application_model_name(x)
    ]
    return keras_application_model_names + list(get_image_backbone_creators().keys())


if __name__ == "__main__":
    print("Available image backbone models in keras:", image_backbone_names())
