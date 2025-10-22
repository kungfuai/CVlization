import logging

from cvlization.torch.data.torchvision_dataset_builder import TorchvisionDatasetBuilder
from cvlization.dataset.kitti_tiny import KittiTinyDatasetBuilder
from cvlization.dataset.penn_fudan_pedestrian import PennFudanPedestrianDatasetBuilder
from cvlization.torch.training_pipeline.object_detection.torchvision.pipeline import TorchvisionObjectDetection


LOGGER = logging.getLogger(__name__)


class TrainingSession:
    def __init__(self, args):
        self.args = args

    def run(self):
        dataset_builder = self.create_dataset()
        TorchvisionObjectDetection(
            net=self.args.net,
            epochs=30,
        ).fit(dataset_builder=dataset_builder)

    def create_dataset(self):
        LOGGER.info(
            f"Available dataset builders: {KittiTinyDatasetBuilder(), PennFudanPedestrianDatasetBuilder()}"
        )
        dataset_builder = KittiTinyDatasetBuilder(flavor="torchvision", label_offset=0)
        # dataset_builder = PennFudanPedestrianDatasetBuilder(
        #     flavor="torchvision", include_masks=False, label_offset=1
        # )
        return dataset_builder

    


if __name__ == "__main__":
    """
    python -m examples.object_detection.torchvision.train
    """

    from argparse import ArgumentParser

    options = TorchvisionObjectDetection.model_names()
    parser = ArgumentParser(
        epilog=f"""
            Options for net: {options} ({len(options)} of them). Though you probably want a fpn in it.
            
            """
    )
    parser.add_argument("--net", type=str, default="fcos_resnet50_fpn")
    # Common options:
    # net="fcos_resnet50_fpn"
    # net="retinanet_resnet50_fpn",
    # net="fasterrcnn_resnet50_fpn", and more
    parser.add_argument("--track", action="store_true")

    args = parser.parse_args()
    TrainingSession(args).run()
