import logging
from torch import nn
import torch
import torchvision
from pytorch_lightning import LightningModule
from ...metrics.semantic_segmentation_confusion_matrix import (
    SemanticSegmentationConfusionMatrix,
)


LOGGER = logging.getLogger(__name__)


class TorchvisionSemanticSegmentationModelFactory:
    def __init__(
        self,
        num_classes: int = 3,
        net: str = "fcn_resnet50",
        pretrained_backbone: bool = True,
        lightning: bool = True,
        lr: float = 0.0001,
    ):
        """
        :param num_classes: Number of classes to detect, excluding the background.
        """
        self.num_classes = num_classes
        self.net = net
        self.lightning = lightning
        # TODO: consider not putting lr here.
        self.lr = lr  # applicable for lightining model only
        self.pretrained_backbone = pretrained_backbone

    def run(self):
        model = create_semantic_segmentation_model_with_torchvision(
            self.num_classes + 1,  # TODO: does num_classes include the background?
            net=self.net,
            pretrained=False,
            pretrained_backbone=self.pretrained_backbone,
        )
        if self.lightning:
            model = LitSegmentor(model, lr=self.lr, num_classes=self.num_classes + 1)
        return model

    @classmethod
    def model_names(cls):
        names = [
            x
            for x in torchvision.models.segmentation.__dict__.keys()
            if not x.startswith("_") and x[0] in "abcdefghijklmnopqrstuvwxyz"
        ]
        names = [x for x in names if callable(torchvision.models.segmentation.__dict__[x])]
        return names


class LitSegmentor(LightningModule):
    def __init__(self, model, num_classes, lr=0.001):
        super().__init__()
        self.model = model
        self.lr = lr
        self.num_classes = num_classes
        self.confmat = SemanticSegmentationConfusionMatrix(num_classes)

    def forward(self, *args, **kwargs):
        images = args[0]
        outputs = self.model.forward(images, **kwargs)
        if self.model.training:
            target = args[1]
            losses = self.criterion(outputs, target)
            return losses
        else:
            return outputs

    def criterion(self, outputs, target):
        """Only one target is expected."""
        losses = {}
        for name, x in outputs.items():
            target = target.long()
            losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)
        if "aux" in losses:
            losses["aux"] *= 0.5
        return losses

    def training_step(self, train_batch, batch_idx):
        # This assumes the model's forward method returns a dictionary of losses.
        loss_dict = self.forward(*train_batch)
        loss = sum(loss_dict.values())
        self.log("lr", self.lr, on_step=True, on_epoch=True, prog_bar=True)
        for k, v in loss_dict.items():
            self.log(k[:5], v.detach(), prog_bar=True)
        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):
        images, targets = val_batch
        predictions = self.forward(images)
        assert isinstance(
            predictions, dict
        ), f"predictions is not a dict: {predictions}"
        gt_segmentation = targets
        predicted_segmentation = predictions["out"]
        self.confmat.update(
            gt_segmentation.flatten(), predicted_segmentation.argmax(1).flatten()
        )

    def on_validation_epoch_end(self):
        print(self.confmat)
        self.log(
            "mean_iou",
            self.confmat.mean_iou,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        return self.confmat.mean_iou

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer


def get_resnet_backbone(backbone_name: str, pretrained: bool = True) -> nn.Module:
    if backbone_name == "resnet18":
        resnet_net = torchvision.models.resnet18(pretrained=pretrained)
        modules = list(resnet_net.children())[:-2]
        backbone = nn.Sequential(*modules)
        backbone.out_channels = 512
    elif backbone_name == "resnet34":
        resnet_net = torchvision.models.resnet34(pretrained=pretrained)
        modules = list(resnet_net.children())[:-2]
        backbone = nn.Sequential(*modules)
        backbone.out_channels = 512
    elif backbone_name == "resnet50":
        resnet_net = torchvision.models.resnet50(pretrained=pretrained)
        modules = list(resnet_net.children())[:-2]
        backbone = nn.Sequential(*modules)
        backbone.out_channels = 2048
    elif backbone_name == "resnet101":
        resnet_net = torchvision.models.resnet101(pretrained=pretrained)
        modules = list(resnet_net.children())[:-2]
        backbone = nn.Sequential(*modules)
        backbone.out_channels = 2048
    elif backbone_name == "resnet152":
        resnet_net = torchvision.models.resnet152(pretrained=pretrained)
        modules = list(resnet_net.children())[:-2]
        backbone = nn.Sequential(*modules)
        backbone.out_channels = 2048
    elif backbone_name == "resnet50_modified_stride_1":
        resnet_net = torchvision.models.resnet50(pretrained=pretrained)
        modules = list(resnet_net.children())[:-2]
        backbone = nn.Sequential(*modules)
        backbone.out_channels = 2048
    elif backbone_name == "resnext101_32x8d":
        resnet_net = torchvision.models.resnext101_32x8d(pretrained=pretrained)
        modules = list(resnet_net.children())[:-2]
        backbone = nn.Sequential(*modules)
        backbone.out_channels = 2048
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
    return backbone


def create_semantic_segmentation_model_with_torchvision(
    num_classes=3,
    net="fcn_resnet50",
    pretrained: bool = True,
    aux_loss=True,
    pretrained_backbone: bool = True,
):
    if net in torchvision.models.segmentation.__dict__:
        model_factory = torchvision.models.segmentation.__dict__[net]
        print(model_factory.__doc__)
        model = model_factory(
            pretrained=pretrained,
            aux_loss=aux_loss,
            pretrained_backbone=pretrained_backbone,
            num_classes=num_classes,
        )
    else:
        raise ValueError(
            f"Unknown network name for semantic segmentation in torchvision: {net}"
        )
    print("********************************** Model signature:")
    print(model.forward.__doc__)

    return model
