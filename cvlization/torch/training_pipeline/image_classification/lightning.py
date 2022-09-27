import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy


class ImageClassifier(pl.LightningModule):
    def __init__(self, model: nn.Module, num_classes: int = None, lr: float = 0.0001):
        super().__init__()
        self.net = model
        self.train_accuracy = Accuracy(num_classes=num_classes)
        self.val_accuracy = Accuracy(num_classes=num_classes)
        self.lr = lr

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def loss(self, y_hat, y):
        return F.cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx=None):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.train_accuracy.update(preds=y_hat, target=y.int())

        # Note: The return value can be None, a loss tensor, or a dictionary with a "loss" key.
        return {"loss": loss}

    def validation_step(self, batch, batch_idx=None):
        x, y = batch
        y_hat = self(x)
        self.val_accuracy.update(preds=y_hat, target=y.int())
        val_loss = self.loss(y_hat, y)
        return {"loss": val_loss}


class ImageClassifierCallback(Callback):

    def __init__(self, save_prediction_examples=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_epoch = 0
        self.save_prediction_examples = save_prediction_examples

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        loss = outputs["loss"]
        self.log(name="loss", value=loss.item(), on_step=True)

    def on_train_epoch_end(self, trainer, pl_module, *args, **kwargs):
        super().on_train_epoch_end(trainer, pl_module, *args, **kwargs)
        avg_acc = pl_module.train_accuracy.compute()
        self.log('train_acc', avg_acc, prog_bar=True)
        print("train_accuracy:", avg_acc)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_validation_batch_end(trainer, pl_module,
                                        outputs, batch, batch_idx, dataloader_idx)
        loss = outputs["loss"]
        self.log(name="loss", value=loss.item(), on_step=True)
        # self.log(name="acc", value=pl_module.val_accuracy.compute(), on_step=True)
        return {"loss": loss}

    def on_validation_epoch_end(self, trainer, pl_module, *args, **kwargs):
        super().on_validation_epoch_end(trainer, pl_module, *args, **kwargs)
        avg_acc = pl_module.val_accuracy.compute()
        self.log('val_acc', avg_acc, prog_bar=True)
        print("val_accuracy:", avg_acc)
        if self.save_prediction_examples:
            pass  # not implemented
        self.current_epoch += 1
