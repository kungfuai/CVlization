from dataclasses import dataclass
import torch

from cvlization.training_pipeline import TrainingPipelineConfig


@dataclass
class MultiLabelImageClassificationPipelineConfig(TrainingPipelineConfig):
    pass


class MultiLabelImageClassificationPipeline:
    def __init__(self, config: MultiLabelImageClassificationPipelineConfig):
        self.config = config

    def fit(self, dataset_builder):
        pass


