import logging

from cvlization.torch.data.torchvision_dataset_builder import TorchvisionDatasetBuilder
from cvlization.torch.encoder.torch_image_backbone import image_backbone_names
from cvlization.torch.training_pipeline.image_classification.simple_pipeline import SimpleImageClassificationPipeline


LOGGER = logging.getLogger(__name__)


class TrainingSession:
    def __init__(self, args):
        self.args = args

    def run(self):
        dataset_builder = TorchvisionDatasetBuilder(dataset_classname="CIFAR10")
        SimpleImageClassificationPipeline(
            config=SimpleImageClassificationPipeline.Config(
                model_name=self.args.net,
                num_classes=dataset_builder.num_classes,
                # TODO: pass tracking args to the training pipeline
            )
        ).fit(dataset_builder=dataset_builder)


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
