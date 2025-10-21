from cvlization.dataset.rvl_cdip_tiny import RvlCdipTinyDatasetBuilder
from cvlization.torch.training_pipeline.doc_ai.huggingface.donut.pipeline import Donut


class TrainingSession:
    def __init__(self, args):
        self.args = args

    def run(self):
        dataset_builder = self.create_dataset()
        Donut().train(dataset_builder=dataset_builder)

    def create_dataset(self):
        dataset_builder = RvlCdipTinyDatasetBuilder()
        return dataset_builder


if __name__ == "__main__":
    """
    python -m examples.doc_ai.huggingface.donut.doc_classification.train --net donut
    """

    from argparse import ArgumentParser

    options = ["donut"]
    parser = ArgumentParser(
        epilog=f"""
            Options for net: {options} ({len(options)} of them).
            
            """
    )
    parser.add_argument("--net", type=str, default="donut")
    parser.add_argument("--track", action="store_true")

    args = parser.parse_args()
    TrainingSession(args).run()
