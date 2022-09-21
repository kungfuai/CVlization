import logging
import torch
import torchvision
from pytorch_lightning.core.lightning import LightningModule
try:
    from torchmetrics.detection.map import MeanAveragePrecision
except ImportError:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    # Tested on torchmetrics 0.7.*, 0.9.*
    # TODO: there seems to be a bug in torchmetrics mAP calculation,
    #   when FPs and TPs have the same score.
    #   https://github.com/Lightning-AI/metrics/issues/1184
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


LOGGER = logging.getLogger(__name__)


# TODO: Decipher how to use torchvision's FPN module.
class TorchvisionInstanceSegmentationModelFactory:
    def __init__(
        self,
        num_classes: int = 3,
        net: str = "maskrcnn_resnet50_fpn",
        lightning: bool = True,
        lr: float = 0.0001,
        pretrained: bool = True,
    ):
        """
        :param num_classes: Number of classes to detect, excluding the background.
        """
        self.num_classes = num_classes
        self.net = net
        self.lightning = lightning
        # TODO: consider not putting lr here.
        self.lr = lr  # applicable for lightining model only
        self.pretrained = pretrained

    def run(self):
        model = create_instance_segmentation_model_with_torchvision(
            self.num_classes, self.net, pretrained=self.pretrained
        )
        if self.lightning:
            model = LitDetector(model, lr=self.lr)
        return model

    @classmethod
    def model_names(cls):
        return ["maskrcnn_resnet50_fpn"]


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
            target["masks"] = target["masks"].to("cuda")
            if len(target["labels"].shape) == 2:
                target["labels"] = target["labels"][:, 0]
                assert len(target["labels"].shape) == 1

        assert len(detections) == len(
            targets
        ), f"{len(detections)} detections but {len(targets)} targets. detections={detections}"
        # TODO: add metric for masks!
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


def create_instance_segmentation_model_with_torchvision(
    num_classes=3, net="maskrcnn_resnet50_fpn", pretrained=True
):
    if net == "maskrcnn_resnet50_fpn":
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained=pretrained
        )

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        num_classes_including_background = num_classes + 1
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes_including_background
        )

        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes_including_background
        )
    else:
        raise ValueError(
            f"Unknown network for instance segmentation in torchvision: {net}"
        )

    print("********************************** Model signature:")
    print(model.forward.__doc__)

    return model
