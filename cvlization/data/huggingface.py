from dataclasses import dataclass
from datasets import load_dataset


@dataclass
class HuggingFaceDatasetBuilder:
    dataset_name: str
    train_split_name: str = "train"
    validation_split_name: str = "validation"

    def training_dataset(self):
        ds = load_dataset(self.dataset_name, split=self.train_split_name)
        return ds

    def validation_dataset(self):
        ds = load_dataset(self.dataset_name, split=self.validation_split_name)
        return ds
