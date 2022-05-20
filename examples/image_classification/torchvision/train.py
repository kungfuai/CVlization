import logging
import torch
from torch import nn
from pytorch_lightning.core import LightningModule
from torchmetrics.detection.map import MeanAveragePrecision
import torchvision
from torchvision.models import detection
from torchvision.models.detection.rpn import AnchorGenerator

from cvlization.specs.ml_framework import MLFramework
from cvlization.torch.data.torchvision_dataset_builder import TorchvisionDatasetBuilder
from cvlization.specs.prediction_tasks import ImageClassification
from cvlization.training_pipeline import TrainingPipeline, TrainingPipelineConfig
from cvlization.torch.encoder.torch_image_backbone import create_image_backbone
from cvlization.lab.experiment import Experiment


LOGGER = logging.getLogger(__name__)


def create_detection_model(num_classes=3, net="fcos_resnet50_fpn"):
    if net == "fcos_resnet50_fpn":
        model = detection.fcos_resnet50_fpn(
            num_classes=num_classes, pretrained_backbone=True
        )
    elif net == "fcos_resnet50":
        # backbone = get_resnet_backbone("resnet50")
        backbone = create_image_backbone(name="resnet50", pretrained=True)
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
            num_classes=num_classes, pretrained_backbone=True
        )
    elif net == "fasterrcnn_resnet50_fpn":
        model = detection.fasterrcnn_resnet50_fpn(
            num_classes=num_classes, pretrained_backbone=True
        )
    elif net == "retinanet_mobilenet":
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.33, 0.5, 1.0, 2.0, 3),)
        )
        backbone = torchvision.models.mobilenet_v2(pretrained=True).features
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
        backbone = get_resnet_backbone(net.replace("retinanet_", ""))
        model = detection.RetinaNet(
            backbone=backbone,
            num_classes=num_classes,
            anchor_generator=anchor_generator,
        )
    print("********************************** Model signature:")
    print(model.forward.__doc__)

    return model


def get_resnet_backbone(backbone_name: str) -> nn.Module:
    if backbone_name == "resnet18":
        resnet_net = torchvision.models.resnet18(pretrained=True)
        modules = list(resnet_net.children())[:-2]
        backbone = nn.Sequential(*modules)
        backbone.out_channels = 512
    elif backbone_name == "resnet34":
        resnet_net = torchvision.models.resnet34(pretrained=True)
        modules = list(resnet_net.children())[:-2]
        backbone = nn.Sequential(*modules)
        backbone.out_channels = 512
    elif backbone_name == "resnet50":
        resnet_net = torchvision.models.resnet50(pretrained=True)
        modules = list(resnet_net.children())[:-2]
        backbone = nn.Sequential(*modules)
        backbone.out_channels = 2048
    elif backbone_name == "resnet101":
        resnet_net = torchvision.models.resnet101(pretrained=True)
        modules = list(resnet_net.children())[:-2]
        backbone = nn.Sequential(*modules)
        backbone.out_channels = 2048
    elif backbone_name == "resnet152":
        resnet_net = torchvision.models.resnet152(pretrained=True)
        modules = list(resnet_net.children())[:-2]
        backbone = nn.Sequential(*modules)
        backbone.out_channels = 2048
    elif backbone_name == "resnet50_modified_stride_1":
        resnet_net = torchvision.models.resnet50(pretrained=True)
        modules = list(resnet_net.children())[:-2]
        backbone = nn.Sequential(*modules)
        backbone.out_channels = 2048
    elif backbone_name == "resnext101_32x8d":
        resnet_net = torchvision.models.resnext101_32x8d(pretrained=True)
        modules = list(resnet_net.children())[:-2]
        backbone = nn.Sequential(*modules)
        backbone.out_channels = 2048
    return backbone


class DetectorModel(LightningModule):
    def __init__(self, model, lr=0.001):
        super().__init__()
        self.model = model
        self.lr = lr
        self.val_mAP = MeanAveragePrecision()

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def training_step(self, train_batch, batch_idx):
        loss_dict = self.forward(*train_batch)
        loss = sum(loss_dict.values())
        self.log("lr", self.lr, on_step=True, on_epoch=True, prog_bar=True)
        for k, v in loss_dict.items():
            self.log(k[:5], v.detach(), prog_bar=True)
        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):
        images, targets = val_batch
        detections = self.forward(*val_batch)  # inputs=images)
        assert isinstance(detections, list), f"detections is not a list: {detections}"
        for det in detections:
            assert isinstance(det, dict), f"det is not a dict: {det}"
            assert len(det["labels"].shape) == 1
            # det["labels"] = torch.unsqueeze(det["labels"], -1)

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


if __name__ == "__main__":
    """
    python -m cvlization.lab.experiment_advanced
    """

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--track", action="store_true")

    args = parser.parse_args()

    training_pipeline = TrainingPipeline(
        ml_framework=MLFramework.PYTORCH,
        image_backbone="resnet18",
        loss_function_included_in_model=False,
        collate_method=None,
        pretrained=False,
        epochs=50,
        train_batch_size=8,
        val_batch_size=2,
        train_steps_per_epoch=500,
        val_steps_per_epoch=200,
        optimizer_name="Adam",
        lr=0.0001,
        n_gradients=1,
        dropout=0,
        experiment_tracker=None,
    )

    Experiment(
        # The interface (inputs and outputs) of the model.
        prediction_task=ImageClassification(
            n_classes=10,
            num_channels=3,
            image_height=32,
            image_width=32,
            channels_first=True,
        ),
        # Dataset and transforms.
        dataset_builder=TorchvisionDatasetBuilder(dataset_classname="CIFAR10"),
        # Model, optimizer and hyperparams.
        training_pipeline=training_pipeline,
    ).run()
