from cvlization.dataset.cord_v2 import CordV2DatasetBuilder
from cvlization.torch.training_pipeline.doc_ai.huggingface.donut.pipeline import Donut
from cvlization.torch.training_pipeline.doc_ai.huggingface.donut.model import DonutPredictionTask

"""
Example based on:
https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Donut
"""

class TrainingSession:
    def __init__(self, args):
        self.args = args

    def run(self):
        dataset_builder = self.create_dataset()
        config = {
            "task": DonutPredictionTask.PARSE,
            "max_length": CordV2DatasetBuilder.max_length,
            "task_start_token": CordV2DatasetBuilder.task_start_token,
            "image_height": CordV2DatasetBuilder.image_height,
            "image_width": CordV2DatasetBuilder.image_width,
            "ignore_id": CordV2DatasetBuilder.ignore_id,
        }
        Donut(**config).train(dataset_builder=dataset_builder)

    def create_dataset(self):
        dataset_builder = CordV2DatasetBuilder()
        return dataset_builder


if __name__ == "__main__":
    """
    python -m examples.object_detection.torchvision.train
    """

    from argparse import ArgumentParser

    options = ["donut"]
    parser = ArgumentParser(
        epilog=f"""
            Options for net: {options} ({len(options)} of them).
            
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
