from typing import List
from .trainer import Trainer
from ..data.model_target import DataColumn


class IndexedDataset:
    model_inputs: List[DataColumn]
    model_outputs = List[DataColumn]

    def get_example(self, i: int) -> dict:
        raise NotImplementedError

    def __getitem__(self, i: int):
        ex = self.get_example(i)
        inputs = [i.get_value(ex) for i in self.model_inputs]
        targets = [i.get_value(ex) for i in self.model_targets]
        return inputs, targets


class IterableDataset:
    def next_example(self) -> dict:
        raise NotImplementedError


def multi_input_multi_targets_image_predictor():
    model_inputs = [
        DataColumn(name="left_cc_pixel"),
        DataColumn(name="right_cc_pixel"),
    ]
    model_targets = [
        DataColumn(name="5y_risk", transforms=None),
        DataColumn(name="density", transforms=None),
    ]
    train_dataset = IndexedDataset(
        model_inputs=model_inputs, model_targets=model_targets
    )
    val_dataset = IndexedDataset(model_inputs=model_inputs, model_targets=model_targets)
    Trainer(train_dataset=train_dataset, val_dataset=val_dataset)
