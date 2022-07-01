# pip install mmsegmentation==0.25.0
# pip install -U mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.11.0/index.html

import os
from mmseg.datasets import build_dataset
from cvlization.lab.stanford_background import StanfordBackgroundDatasetBuilder
from cvlization.torch.net.semantic_segmentation.mmseg import (
    MMDatasetAdaptor,
    MMSemanticSegmentationModels,
    MMTrainer,
)


class TrainingSession:
    def __init__(self, args):
        self.args = args

    def run(self):
        self.dataset_builder_cls = StanfordBackgroundDatasetBuilder
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
        model_registry = MMSemanticSegmentationModels(num_classes=num_classes)
        model_dict = model_registry[self.args.net]
        model, cfg = model_dict["model"], model_dict["config"]
        return model, cfg

    def create_dataset(self, config):
        dsb = self.dataset_builder_cls(flavor=None, to_torch_tensor=False)
        dataset_classname = MMDatasetAdaptor.adapt_and_register_detection_dataset(dsb)
        print("registered:", dataset_classname)

        MMDatasetAdaptor.set_dataset_info_in_config(
            config,
            dataset_classname=dataset_classname,
            dataset_dir=os.path.join(dsb.data_dir, dsb.dataset_folder),
            train_anno_file=dsb.train_ann_file,
            val_anno_file=dsb.val_ann_file,
            image_dir=dsb.img_folder,
            seg_dir=dsb.seg_folder,
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
        return MMTrainer(cfg, net)


if __name__ == "__main__":
    """
    python -m examples.semantic_segmentation.mmseg.train
    """

    from argparse import ArgumentParser

    options = MMSemanticSegmentationModels.model_names()
    parser = ArgumentParser(
        epilog=f"""
            *** Options for net: {options} ({len(options)} of them).
            """
    )
    parser.add_argument("--net", type=str, default="pspnet_r50")
    parser.add_argument("--track", action="store_true")

    args = parser.parse_args()
    TrainingSession(args).run()
