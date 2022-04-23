from typing import List, Iterable
from ..specs import ModelSpec as PredictionTask


class SplittedDataset:
    # TODO: store cross validation splits.
    def training_dataset(self) -> Iterable:
        raise NotImplementedError

    def validation_dataset(self) -> Iterable:
        raise NotImplementedError

    def additional_validation_datasets(self) -> List[Iterable]:
        raise NotImplementedError

    def supported_prediction_tasks(self) -> List[PredictionTask]:
        raise NotImplementedError
