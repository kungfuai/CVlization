import logging
from torch import nn
import torch
import torchvision
from torchvision.models import detection
from torchvision.models.detection.rpn import AnchorGenerator
from pytorch_lightning.core.lightning import LightningModule
from torchmetrics.detection.map import MeanAveragePrecision


LOGGER = logging.getLogger(__name__)


# TODO: Decipher how to use torchvision's FPN module.
class TorchvisionDetectionModelFactory:
    def __init__(
        self,
        num_classes: int = 3,
        net: str = "fcos_resnet50_fpn",
        lightning: bool = True,
        lr: float = 0.0001,
        pretrained: bool = True,
    ):
        self.num_classes = num_classes
        self.net = net
        self.lightning = lightning
        # TODO: consider not putting lr here.
        self.lr = lr  # applicable for lightining model only
        self.pretrained = pretrained

    def run(self):
        model = create_detection_model_with_torchvision(
            self.num_classes, self.net, pretrained=self.pretrained
        )
        if self.lightning:
            model = LitDetector(model, lr=self.lr)
        return model

    @classmethod
    def model_names(cls):
        model_names = ["maskrcnn_resnet50_fpn"]
        return model_names


class LitDetector(LightningModule):
    def __init__(self, model, lr=0.001):
        super().__init__()
        self.model = model
        self.lr = lr
        self.val_mAP = MeanAveragePrecision()

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

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
        detections = self.forward(*val_batch)
        assert isinstance(detections, list), f"detections is not a list: {detections}"
        for det in detections:
            assert isinstance(det, dict), f"det is not a dict: {det}"
            assert len(det["labels"].shape) == 1

        for target in targets:
            assert isinstance(target, dict), f"target is not a dict: {target}"
            target["boxes"] = target["boxes"].to("cuda")
            target["labels"] = target["labels"].to("cuda")
            if len(target["labels"].shape) == 2:
                target["labels"] = target["labels"][:, 0]
                assert len(target["labels"].shape) == 1

        assert len(detections) == len(
            targets
        ), f"{len(detections)} detections but {len(targets)} targets. detections={detections}"
        self.val_mAP.update(preds=detections, target=targets)

    def validation_epoch_end(self, outputs):
        mAP = self.val_mAP.compute()
        self.log_dict(
            mAP,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.val_mAP.reset()
        LOGGER.info(f"\nValidation mAP: {float(mAP['map_50'].numpy())}")
        return mAP

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


def create_detection_model_with_torchvision(
    num_classes=3, net="fcos_resnet50_fpn", pretrained=True
):
    if net == "fcos_resnet50_fpn":
        model = detection.fcos_resnet50_fpn(
            num_classes=num_classes, pretrained_backbone=pretrained
        )
    elif net == "fcos_resnet50":
        backbone = get_resnet_backbone(
            "resnet50",
            pretrained=pretrained,
        )  # using torchvision's provided method
        # backbone = create_image_backbone(name="resnet50", pretrained=True) # using cvlization's method
        backbone.out_channels = 2048
        anchor_generator = AnchorGenerator(
            sizes=((32,),),
            aspect_ratios=((1.0,)),
        )
        model = detection.FCOS(
            backbone=backbone,
            num_classes=num_classes,
            anchor_generator=anchor_generator,
        )
    elif net == "retinanet_resnet50_fpn":
        model = detection.retinanet_resnet50_fpn(
            num_classes=num_classes, pretrained_backbone=pretrained
        )
    elif net == "fasterrcnn_resnet50_fpn":
        model = detection.fasterrcnn_resnet50_fpn(
            num_classes=num_classes, pretrained_backbone=pretrained
        )
    elif net == "retinanet_mobilenet":
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.33, 0.5, 1.0, 2.0, 3),)
        )
        backbone = torchvision.models.mobilenet_v2(pretrained=pretrained).features
        backbone.out_channels = 1280
        model = detection.RetinaNet(
            backbone=backbone,
            num_classes=num_classes,
            anchor_generator=anchor_generator,
        )
    elif net.startswith("retinanet_resnet"):
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.33, 0.5, 1.0, 2.0, 3),)
        )
        backbone = get_resnet_backbone(
            net.replace("retinanet_", ""), pretrained=pretrained
        )
        model = detection.RetinaNet(
            backbone=backbone,
            num_classes=num_classes,
            anchor_generator=anchor_generator,
        )
    else:
        raise ValueError(
            f"Unknown network name for object detection in torchvision: {net}"
        )
    print("********************************** Model signature:")
    print(model.forward.__doc__)

    return model
