from cvlization.lab.kitti_tiny import KittiTinyDatasetBuilder
from cvlization.lab.penn_fudan_pedestrian import PennFudanPedestrianDatasetBuilder
from cvlization.torch.training_pipeline.object_detection.mmdet.pipeline import MMDetObjectDetection

class TrainingSession:
    # TODO: add experiment tracker to the training pipeline.
    def __init__(self, args):
        self.args = args

    def run(self):
        self.dataset_builder_cls = KittiTinyDatasetBuilder
        dataset_builder = self.dataset_builder_cls(flavor=None, to_torch_tensor=False)
        tp = MMDetObjectDetection(net=self.args.net, config_override_fn=None)
        tp.train(dataset_builder)


if __name__ == "__main__":
    """
    python -m examples.object_detection.mmdet.train
    """

    from argparse import ArgumentParser

    options = MMDetObjectDetection.model_names()
    parser = ArgumentParser(
        epilog=f"""
            Options for net: {options} ({len(options)} of them).
            """
    )
    parser.add_argument("--net", type=str, default="fcos")
    # Alternative options:
    # net="deformable_detr",
    # net="dyhead",
    # net="retinanet_r18"
    parser.add_argument("--track", action="store_true")

    args = parser.parse_args()
    TrainingSession(args).run()
