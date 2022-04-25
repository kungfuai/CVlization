import logging
from torch import nn
import torch
from ..net.davidnet.dawn_utils import net, Network
from ..net.simple_conv import SimpleConv


LOGGER = logging.getLogger(__name__)


def load_dino_model(name):
    model = torch.hub.load("facebookresearch/dino:main", name)
    return model


def load_timm_model(name, pretrained: bool = True, in_chans=3):
    import timm

    # https://rwightman.github.io/pytorch-image-models/feature_extraction/
    model = timm.create_model(
        name, pretrained=pretrained, global_pool="", num_classes=0, in_chans=in_chans
    )
    # TODO:
    # For multi-scale feature maps, it will output a list of tensors.
    # e.g. timm.create_model('resnest26d', features_only=True, pretrained=True)
    return model


def create_image_backbone(
    name: str, pretrained: bool = True, in_chans=3, input_shape=None
) -> nn.Module:
    if name.startswith("dino_"):
        return load_dino_model(name)
    elif name == "simple_conv":
        return SimpleConv()
    elif name == "davidnet":
        model = Network(net())
        # print(model)
        # assert False

        class WrappedModel(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.sub_model = model

            def forward(self, x):
                if isinstance(x, tuple) or isinstance(x, list):
                    x = x[0]
                x_dict = {"input": x}
                outputs = self.sub_model(x_dict)
                # See dawn_utils.py.
                try:
                    return outputs["pool"]
                    # return outputs["layer3/residual/add"]
                except KeyError:
                    print(outputs.keys())
                    raise

        return WrappedModel()

    else:
        return load_timm_model(name.lower(), pretrained=pretrained, in_chans=in_chans)


def image_backbone_names():
    LOGGER.info(
        f"Listing models in CVlization and timm. Additional models available. Please check https://pytorch.org/hub/ for additional models."
    )
    model_names = ["simple_conv", "davidnet"]

    try:
        import timm

        model_names += timm.list_models()
    except Exception as e:
        LOGGER.warning(f"Failed to list timm models: {e}")

    try:
        model_names += torch.hub.list("facebookresearch/dino:main")
    except Exception as e:
        LOGGER.warning(f"Failed to list dino models: {e}")

    return model_names
