from mmdet.datasets import build_dataset
from cvlization.lab.penn_fudan_pedestrian import PennFudanPedestrianDatasetBuilder
from cvlization.torch.net.panoptic_segmentation.mmdet import (
    MMPanopticSegmentationModels,
    MMDatasetAdaptor,
    MMTrainer,
)


class TrainingSession:
    # TODO: integrate with Experiment interface.
    #   Create a MMDetTrainer class.

    def __init__(self, args):
        self.args = args

    def run(self):
        self.model, self.cfg = self.create_model(1)
        self.datasets = self.create_dataset(self.cfg)
        num_classes = len(self.datasets[0].CLASSES)
        # Create the model again.
        # TODO: separate create_config and create_model into functions.
        self.model, _ = self.create_model(num_classes)
        self.trainer = self.create_trainer(self.cfg, self.args.net)
        self.cfg = self.trainer.config
        print("batch size:", self.cfg.data.samples_per_gpu)
        self.trainer.fit(
            model=self.model,
            train_dataset=self.datasets[0],
            val_dataset=self.datasets[1],
        )

    def create_model(self, num_classes: int):
        model_registry = MMPanopticSegmentationModels(num_classes=num_classes)
        model_dict = model_registry[self.args.net]
        model, cfg = model_dict["model"], model_dict["config"]

        return model, cfg

    def create_dataset(self, config):
        # dsb = self.dataset_builder_cls(flavor=None, to_torch_tensor=False)
        MMDatasetAdaptor.set_dataset_info_in_config(config, image_dir=None)

        if hasattr(config.data.train, "dataset"):
            datasets = [
                build_dataset(config.data.train.dataset),
                build_dataset(config.data.val),
            ]
        else:
            datasets = [
                build_dataset(config.data.train),
            ]
            datasets.append(build_dataset(config.data.val))

        # print(config.pretty_text)
        print("\n***** Training data:", type(datasets[0]))
        print(datasets[0])

        print("\n***** Validation data:")
        print(datasets[1])

        print("\n----------------------------- first training example:")
        print(datasets[0][0])
        return datasets

    def create_trainer(self, cfg, net: str):
        return MMTrainer(cfg, net)


if __name__ == "__main__":
    """
    python -m examples.panoptic_segmentation.mmdet.train
    """

    from argparse import ArgumentParser

    options = MMPanopticSegmentationModels.model_names()
    parser = ArgumentParser(
        epilog=f"""
            Options for net: {options} ({len(options)} of them).
            """
    )
    parser.add_argument("--net", type=str, default="maskrcnn_r50")
    parser.add_argument("--track", action="store_true")

    args = parser.parse_args()
    TrainingSession(args).run()
