import logging
import torch
import re
import pytorch_lightning as pl
from torchmetrics import F1Score, Accuracy
from transformers import LayoutLMv3ForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput


LOGGER = logging.getLogger(__name__)


class LayoutLMv3SequenceClassifier(pl.LightningModule):
    """LayoutLMv3 Model with a sequence classification head on top (a linear layer on top of the final hidden state of the [CLS] token)
    for document image classification tasks.
    This model is a PyTorch torch.nn.Module sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all
    matter related to general usage and behavior.
    Parameters
    ----------
    config: dict
        A map of parameter names to values, e.g. dropout_rate
    id2label: Dict[int, str]
        A map of index to label name, e.g. 1 -> "A"
    label2id: Dict[str, int]
        A map of label name to index, e.g. "A" -> 1
    train_pattern: str
        Regular expression of layer names to train. Any layer that does not match the pattern will have its weights frozen. The default
        value (".*") will train the entire model. A value of "attention[0-3].+" would freeze any layer that matches that pattern (the
        first 4 attention layers).
    """

    def __init__(self, config: dict, id2label: dict, label2id: dict, train_pattern: str = r".*"):
        super().__init__()
        self.model = LayoutLMv3ForSequenceClassification.from_pretrained(
            "microsoft/layoutlmv3-base", id2label=id2label, label2id=label2id
        )
        self.model.classifier.dropout.p = config["dropout_rate"]
        self.base_lr = config["base_lr"]
        self.head_lr = config["head_lr"]
        self.train_f1 = F1Score(num_classes=len(id2label))
        self.eval_f1 = F1Score(num_classes=len(id2label))
        self.test_f1 = F1Score(num_classes=len(id2label))
        self.train_acc = Accuracy(num_classes=len(id2label))
        self.eval_acc = Accuracy(num_classes=len(id2label))
        self.test_acc = Accuracy(num_classes=len(id2label))
        self.id2label = id2label
        self.label2id = label2id
        # Freeze layers
        for name, param in self.model.named_parameters():
            if not re.match(train_pattern, name):
                LOGGER.debug("Freezing layer:", name)
                param.requires_grad = False

    def forward(self, **inputs) -> SequenceClassifierOutput:
        return self.model(**inputs)

    def training_step(self, batch, batch_idx) -> torch.FloatTensor:
        outputs: SequenceClassifierOutput = self(**batch)
        loss, logits = outputs.loss, outputs.logits
        self.train_f1(logits, batch["labels"])
        self.train_acc(logits, batch["labels"])
        self.log("metrics/train_loss", loss, on_epoch=True, on_step=False)
        self.log("metrics/train_f1", self.train_f1, on_epoch=True, on_step=False)
        self.log("metrics/train_accuracy", self.train_acc, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs: SequenceClassifierOutput = self(**batch)
        loss, logits = outputs.loss, outputs.logits
        self.eval_f1(logits, batch["labels"])
        self.eval_acc(logits, batch["labels"])
        self.log("metrics/val_loss", loss, on_epoch=True, on_step=False)
        self.log("metrics/val_f1", self.eval_f1, on_epoch=True, on_step=False)
        self.log("metrics/val_accuracy", self.eval_acc, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        outputs: SequenceClassifierOutput = self(**batch)
        loss, logits = outputs.loss, outputs.logits
        self.test_f1(logits, batch["labels"])
        self.test_acc(logits, batch["labels"])
        self.log("metrics/test_loss", loss, on_epoch=True, on_step=False)
        self.log("metrics/test_f1", self.test_f1, on_epoch=True, on_step=False)
        self.log("metrics/test_accuracy", self.test_acc, on_epoch=True, on_step=False)

    def predict_step(self, batch, batch_idx) -> torch.FloatTensor:
        outputs: SequenceClassifierOutput = self(**batch)
        # TODO: return logits here for confidence scores
        return outputs.logits.argmax(axis=-1)

    def configure_optimizers(self) -> torch.optim.Adam:
        optimizer = torch.optim.Adam(
            [
                {"params": self.model.layoutlmv3.parameters(), "lr": self.base_lr},
                {"params": self.model.classifier.parameters(), "lr": self.head_lr},
            ],
        )
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        return optimizer
