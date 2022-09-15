import logging

from cvlization.specs.ml_framework import MLFramework
from cvlization.specs import ModelSpec
from cvlization.torch.data.torchvision_dataset_builder import TorchvisionDatasetBuilder
from cvlization.specs.prediction_tasks import ImageClassification
from cvlization.lab.experiment import Experiment
from cvlization.torch.encoder.torch_image_backbone import image_backbone_names
from cvlization.torch.training_pipeline.image_classification.pipeline import ImageClassificationTrainingPipeline


LOGGER = logging.getLogger(__name__)


class TrainingSession:
    def __init__(self, args):
        self.args = args

    def run(self):
        dataset_builder = TorchvisionDatasetBuilder(dataset_classname="CIFAR10")
        ImageClassificationTrainingPipeline().train(dataset_builder=dataset_builder)


if __name__ == "__main__":
    """
    python -m examples.image_classification.torchvision.train
    """

    from argparse import ArgumentParser

    options = image_backbone_names()
    parser = ArgumentParser(
        epilog=f"""
                options for net: {options} ({len(options)} of them).
            """
    )
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--net", default="resnet18")
    args = parser.parse_args()
    TrainingSession(args).run()
