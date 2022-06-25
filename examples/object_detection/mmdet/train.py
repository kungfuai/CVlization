from mmdet.datasets import build_dataset
from cvlization.lab.kitti_tiny import KittiTinyDatasetBuilder
from cvlization.lab.penn_fudan_pedestrian import PennFudanPedestrianDatasetBuilder
from cvlization.torch.net.object_detection.mmdet import (
    MMDetectionModels,
    MMDatasetAdaptor,
    MMDetectionTrainer,
)

# pip install mmdet==2.24.1
# pip install -U mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.11.0/index.html


class TrainingSession:
    # TODO: integrate with Experiment interface.
    #   Create a MMDetTrainer class.

    def __init__(self, args):
        self.args = args

    def run(self):
        # Get num_classes from the dataset. This is needed to construct the model.
        self.dataset_builder_cls = KittiTinyDatasetBuilder
        # self.dataset_builder_cls = PennFudanPedestrianDatasetBuilder
        num_classes = self.dataset_builder_cls().num_classes
        self.model, self.cfg = self.create_model(num_classes)
        # self.cfg.data.samples_per_gpu = 2
        # self.cfg.optimizer.lr = 0.00001
        self.datasets = self.create_dataset(self.cfg)
        self.trainer = self.create_trainer(self.cfg, self.args.net)
        self.cfg = self.trainer.config
        # Additional customization of the config here. e.g.
        #   cfg.optimizer.lr = 0.0001
        print("batch size:", self.cfg.data.samples_per_gpu)
        print(self.datasets[0])
        self.trainer.fit(
            model=self.model,
            train_dataset=self.datasets[0],
            val_dataset=self.datasets[1],
        )

    def create_model(self, num_classes: int):
        model_registry = MMDetectionModels(num_classes=num_classes)
        model_dict = model_registry[self.args.net]
        model, cfg = model_dict["model"], model_dict["config"]
        return model, cfg

    def create_dataset(self, config):
        dsb = self.dataset_builder_cls(flavor=None, to_torch_tensor=False)
        dataset_classname = MMDatasetAdaptor.adapt_and_register_detection_dataset(dsb)
        print("registered:", dataset_classname)

        MMDatasetAdaptor.set_dataset_info_in_config(
            config, dataset_classname=dataset_classname, image_dir=dsb.image_dir
        )

        if hasattr(config.data.train, "dataset"):
            datasets = [
                build_dataset(config.data.train.dataset),
                build_dataset(config.data.val),
            ]
        else:
            datasets = [
                build_dataset(config.data.train),
                build_dataset(config.data.val),
            ]

        print("\n***** Training data:")
        print(datasets[0])

        print("\n***** Validation data:")
        print(datasets[1])
        return datasets

    def create_trainer(self, cfg, net: str):
        return MMDetectionTrainer(cfg, net)


if __name__ == "__main__":
    """
    python -m examples.object_detection.mmdet.train
    """

    from argparse import ArgumentParser

    options = MMDetectionModels.model_names()
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
