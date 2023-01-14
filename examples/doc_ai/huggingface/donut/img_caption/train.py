from cvlization.lab.conceptual_captions import ConceptualCaptionsDatasetBuilder
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
            "task": DonutPredictionTask.CAPTION,
            "max_length": ConceptualCaptionsDatasetBuilder.max_length,
            "task_start_token": ConceptualCaptionsDatasetBuilder.task_start_token,
            "image_height": ConceptualCaptionsDatasetBuilder.image_height,
            "image_width": ConceptualCaptionsDatasetBuilder.image_width,
            "ignore_id": ConceptualCaptionsDatasetBuilder.ignore_id,
        }
        # Donut(**config).train(dataset_builder=dataset_builder)

    def create_dataset(self):
        dataset_builder = ConceptualCaptionsDatasetBuilder()
        dataset_builder.load()
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
