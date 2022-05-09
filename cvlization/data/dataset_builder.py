from typing import Tuple, Union, List


class Dataset:
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError


class DatasetBuilder:
    def run(self) -> Tuple[Dataset, Union[Dataset, List[Dataset]]]:
        # Returns a training dataset and a validation dataset.
        # Sometimes, more than one validation datasets are returned.
        raise NotImplementedError
