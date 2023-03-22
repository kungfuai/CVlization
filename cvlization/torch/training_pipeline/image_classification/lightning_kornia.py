import os
import numpy as np
import pandas as pd
from typing import List
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy, F1Score
from torchmetrics import AveragePrecision
from torchmetrics.classification import (
    MultilabelRecallAtFixedPrecision,
    BinaryRecallAtFixedPrecision,
)
import kornia


class MultiLabelImageClassifier(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module, 
        num_classes: int = None,
        use_color_hist: bool = False,
        color_hist_num_bins: int = 32,
        color_hist_bandwidth: float = 0.001,
        lr: float = 0.0001,
        lr_scheduling: bool = False,
        pred_csv_filename: str = None,
    ):
        super().__init__()
        self.lr = lr
        self.lr_scheduling = lr_scheduling
        self.num_classes = num_classes
        self.net = model
        self.original_fc_in_features = self.net.fc.in_features
        self.use_color_hist = use_color_hist
        self.pred_csv_filename = pred_csv_filename
        if use_color_hist:
            # Turn self.net into a feature extractor.
            self.net.fc = nn.Identity()
            # for color histogram
            self.bins = torch.torch.linspace(0, 255, color_hist_num_bins) / 255.0
            self.bandwidth = torch.tensor(color_hist_bandwidth)
            in_features = self.original_fc_in_features + 3 * color_hist_num_bins
            self.new_fc = nn.Linear(in_features, self.num_classes)
        else:
            self.net.fc = nn.Linear(self.original_fc_in_features, num_classes)

        self.metrics = dict(
            train_acc=Accuracy(num_classes=num_classes),
            val_acc=Accuracy(num_classes=num_classes),
            val_acc_by_=Accuracy(num_classes=num_classes, average=None),
            train_f1=F1Score(num_classes=num_classes),
            val_f1=F1Score(num_classes=num_classes),
            val_f1_by_=F1Score(num_classes=num_classes, average=None),
            train_ap=AveragePrecision(num_classes=num_classes),
            val_ap=AveragePrecision(num_classes=num_classes),
            val_ap_by_=AveragePrecision(num_classes=num_classes, average=None),
            val_recall_at_precision_80_by_=self._create_recall_at_fixed_prediction_metric(
                self.num_classes, 0.8
            ),
            val_recall_at_precision_60_by_=self._create_recall_at_fixed_prediction_metric(
                self.num_classes, 0.6
            ),
            val_recall_at_precision_50_by_=self._create_recall_at_fixed_prediction_metric(
                self.num_classes, 0.5
            )
        )
        self.metrics = nn.ModuleDict(self.metrics)

    def _extract_color_histogram(self, x):
        original_fc_in_features = self.original_fc_in_features
        assert x.shape[1] == 3
        batch_size = x.shape[0]
        hsv = kornia.color.rgb_to_hsv(x)
        h_hist = kornia.enhance.histogram(
            hsv[:, 0, :, :].view(batch_size, -1),
            bins=self.bins,
            bandwidth=self.bandwidth,
        )
        s_hist = kornia.enhance.histogram(
            hsv[:, 1, :, :].view(batch_size, -1),
            bins=self.bins,
            bandwidth=self.bandwidth,
        )
        v_hist = kornia.enhance.histogram(
            hsv[:, 2, :, :].view(batch_size, -1),
            bins=self.bins,
            bandwidth=self.bandwidth,
        )
        hist = torch.cat([h_hist, s_hist, v_hist], dim=1)
        flattened_hist = hist.view(hist.shape[0], -1)
        return flattened_hist

    def forward(self, x):
        if self.use_color_hist:
            assert len(x.shape) == 4
            original_backbone_features = self.net(x)
            color_hist = self._extract_color_histogram(x)
            new_features = torch.cat([original_backbone_features, color_hist], dim=-1)
            y_hat = self.new_fc(new_features)
            return y_hat
        else:
            return self.net(x)

    def _create_recall_at_fixed_prediction_metric(
        self, num_classes: int, precision: float
    ):
        if num_classes > 1:
            return MultilabelRecallAtFixedPrecision(
                num_labels=num_classes, min_precision=precision
            )
        else:
            return BinaryRecallAtFixedPrecision(min_precision=precision)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.lr_scheduling:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "max", factor=0.5, patience=5, threshold=0.02, cooldown=5
            )
        return optimizer

    def loss(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)

    def training_step(self, batch, batch_idx=None):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.metrics["train_acc"].update(preds=y_hat, target=y.int())
        self.metrics["train_f1"].update(preds=y_hat, target=y.int())
        self.metrics["train_ap"].update(preds=y_hat, target=y.int())

        # Note: The return value can be None, a loss tensor, or a dictionary with a "loss" key.
        return {"loss": loss}

    def validation_step(self, batch, batch_idx=None):
        x, y = batch
        y_hat = self(x)
        self.metrics["val_acc"].update(preds=y_hat, target=y.int())
        self.metrics["val_f1"].update(preds=y_hat, target=y.int())
        self.metrics["val_acc_by_"].update(preds=y_hat, target=y.int())
        self.metrics["val_f1_by_"].update(preds=y_hat, target=y.int())
        self.metrics["val_ap"].update(preds=y_hat, target=y.int())
        self.metrics["val_ap_by_"].update(preds=y_hat, target=y.int())
        self.metrics["val_recall_at_precision_80_by_"].update(
            preds=y_hat, target=y.int()
        )
        self.metrics["val_recall_at_precision_60_by_"].update(
            preds=y_hat, target=y.int()
        )
        self.metrics["val_recall_at_precision_50_by_"].update(
            preds=y_hat, target=y.int()
        )

        val_loss = self.loss(y_hat, y)
        return {"loss": val_loss, "y_hat": y_hat, "y": y}

    def predict_step(self, batch, batch_idx=None):
        x, y = batch
        y_hat = self(x)
        return y_hat

class ImageClassifierCallback(Callback):
    def __init__(self, save_prediction_examples=True, class_labels=[], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_epoch = 0
        self.save_prediction_examples = save_prediction_examples
        self.class_labels = class_labels
        self.val_prediction_rows: List[dict] = []
        self.predict_rows = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        loss = outputs["loss"]
        self.log(name="loss", value=loss.item(), on_step=True)

    def on_train_epoch_end(self, trainer, pl_module, *args, **kwargs):
        super().on_train_epoch_end(trainer, pl_module, *args, **kwargs)
        for metric_name in ["train_acc", "train_f1", "train_ap"]:
            metric_value = pl_module.metrics[metric_name].compute()
            self.log(metric_name, metric_value, prog_bar=True)
            print(metric_name, metric_value)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
        loss = outputs["loss"]
        self.log(name="loss", value=loss.item(), on_step=True)
        self._insert_prediction_batch(outputs)
        # self.log(name="acc", value=pl_module.val_accuracy.compute(), on_step=True)
        return {"loss": loss}

    def _insert_prediction_batch(self, outputs: dict):
        y_hat = outputs["y_hat"]
        y = outputs["y"]
        if hasattr(y_hat, "cpu"):
            y_hat = y_hat.cpu()
        if hasattr(y, "cpu"):
            y = y.cpu()
        y_hat = y_hat.numpy()
        y = y.numpy()
        for i in range(len(y)):
            row = {"pred_" + l: pred for l, pred in zip(self.class_labels, y_hat[i])}
            row = {**row, **{l: true for l, true in zip(self.class_labels, y[i])}}
            self.val_prediction_rows.append(row)

    def _reset_predictions(self):
        self.val_prediction_rows = []

    def _get_metric_values(self, metric_value) -> list:
        if hasattr(metric_value, "cpu"):
            metric_value = metric_value.cpu()
        values = []
        if hasattr(metric_value, "numpy"):
            metric_value_np = metric_value.numpy()
            for i, _ in enumerate(self.class_labels):
                if len(metric_value_np.shape) >= 1:
                    class_metric = metric_value_np[i]
                else:
                    class_metric = float(metric_value_np)
                values.append(class_metric)
        elif isinstance(metric_value, list):
            values = metric_value
        else:
            raise TypeError(f"Unknown metric value type: {type(metric_value)}")
        return values

    def on_validation_epoch_end(self, trainer, pl_module, *args, **kwargs):
        super().on_validation_epoch_end(trainer, pl_module, *args, **kwargs)
        for metric_name in ["val_acc", "val_f1", "val_ap"]:
            metric_value = pl_module.metrics[metric_name].compute()
            self.log(metric_name, metric_value, prog_bar=True)
            print(metric_name, metric_value)

        # Thoughts: had to do a different thing for val_ap_by_ because it's a list of tensors of scores, not 1 tensor of all the scores
        for metric_name in ["val_acc_by_", "val_f1_by_"]:
            metric_value = pl_module.metrics[metric_name].compute()
            metric_value = self._get_metric_values(metric_value)
            for i, label in enumerate(self.class_labels):
                class_metric = metric_value[i]
                self.log(metric_name + label, class_metric, prog_bar=True)
                print(metric_name + label, class_metric)

        metric_value = pl_module.metrics["val_ap_by_"].compute()
        metric_value = self._get_metric_values(metric_value)
        for i, label in enumerate(self.class_labels):
            class_metric = metric_value[i]
            self.log("val_ap_by_" + label, class_metric, prog_bar=True)
            print("val_ap_by_" + label, class_metric)

        for precision_threshold in ["80", "60", "50"]:
            metric_name = "val_recall_at_precision_" + precision_threshold + "_by_"
            metric_value, _thresholds = pl_module.metrics[metric_name].compute()
            metric_value = self._get_metric_values(metric_value)
            for i, label in enumerate(self.class_labels):
                class_metric = metric_value[i]
                self.log(metric_name + label, class_metric, prog_bar=True)
                print(metric_name + label, class_metric)

        if self.save_prediction_examples:
            # TODO: hard coded tmp path
            pred_file_name = f"predictions_{self.current_epoch:03d}.csv"
            predictions_path = os.path.join("/tmp", pred_file_name)
            pd.DataFrame(self.val_prediction_rows).to_csv(predictions_path, index=False)
            trainer.logger.experiment.log_artifact(
                run_id=trainer.logger.run_id,
                local_path=predictions_path,
                artifact_path=f"predictions",
                # if we set artifact_path=f"predictions/predictions.csv"), then
                # it will create a folder predictions.csv/.
            )

        self._reset_predictions()

        if pl_module.lr_scheduling:
            pl_module.scheduler.step(pl_module.metrics["val_ap"].compute())

        self.current_epoch += 1

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_predict_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if self.predict_rows == None:
            self.predict_rows = []
        y_hat = outputs
        if hasattr(y_hat, "cpu"):
            y_hat = y_hat.cpu()
        y_hat = y_hat.numpy()
        self.predict_rows.append(y_hat)
    
    def on_predict_end(self, trainer, pl_module):
        super().on_predict_end(trainer, pl_module)
        pred_file_path = os.path.join(trainer.logger.log_dir, self.pred_csv_filename)
        preds = pd.DataFrame(np.concatenate(self.predict_rows), columns=self.class_labels)
        pd.DataFrame(preds).to_csv(pred_file_path, index=False)



