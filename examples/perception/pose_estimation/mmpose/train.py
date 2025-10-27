# Adapted from https://github.com/open-mmlab/mmpose/blob/master/demo/MMPose_Tutorial.ipynb
# pip install mmpose==0.27.0
# pip install -U mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.11.0/index.html

import os
from pathlib import Path
from mmpose.datasets.builder import build_dataset
from cvlization.dataset.coco_pose_tiny import CocoPoseTinyDatasetBuilder
from cvlization.torch.net.pose_estimation.mmpose import (
    MMDatasetAdaptor,
    MMPoseModels,
    MMTrainer,
)


def get_cache_dir() -> Path:
    """Get the CVlization cache directory for datasets."""
    cache_home = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    cache_dir = Path(cache_home) / "cvlization" / "data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


class TrainingSession:
    def __init__(self, args):
        self.args = args

    def run(self):
        self.dataset_builder_cls = CocoPoseTinyDatasetBuilder
        # num_classes = self.dataset_builder_cls().num_classes
        self.model, self.cfg = self.create_model()  # num_classes)
        # self.cfg.data.samples_per_gpu = 2
        # self.cfg.optimizer.lr = 0.00001
        self.datasets = self.create_dataset(self.cfg)

        self.trainer = self.create_trainer(self.cfg, self.args.net)

        self.cfg = self.trainer.config

        # Additional customization of the config here. e.g.
        #   cfg.optimizer.lr = 0.0001
        self.cfg.total_epochs = 30
        print("batch size:", self.cfg.data.samples_per_gpu)
        print(self.datasets[0])
        self.trainer.fit(
            model=self.model,
            train_dataset=self.datasets[0],
            val_dataset=self.datasets[1],
        )

    def create_model(self):
        model_registry = MMPoseModels()
        model_dict = model_registry[self.args.net]
        model, cfg = model_dict["model"], model_dict["config"]
        return model, cfg

    def create_dataset(self, config):
        cache_dir = str(get_cache_dir())
        print(f"Using cache directory: {cache_dir}")
        dsb = self.dataset_builder_cls(flavor=None, to_torch_tensor=False, data_dir=cache_dir)
        dataset_classname = MMDatasetAdaptor.adapt_and_register_detection_dataset(dsb)
        print("registered:", dataset_classname)

        MMDatasetAdaptor.set_dataset_info_in_config(
            config,
            dataset_classname=dataset_classname,
            dataset_dir=os.path.join(dsb.data_dir, dsb.dataset_folder),
            train_anno_file=dsb.train_ann_file,
            val_anno_file=dsb.val_ann_file,
            image_dir=dsb.img_folder,
        )

        if hasattr(config.data.train, "dataset"):
            datasets = [
                build_dataset(config.data.train.dataset),
                build_dataset(config.data.val.dataset),
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
        print(datasets[1][0])

        return datasets

    def create_trainer(self, cfg, net: str):
        return MMTrainer(cfg, net)


if __name__ == "__main__":
    """
    python -m examples.pose_estimation.mmpose.train
    """

    from argparse import ArgumentParser

    options = MMPoseModels.model_names()
    parser = ArgumentParser(
        epilog=f"""
            *** Options for net: {options} ({len(options)} of them).
            """
    )
    parser.add_argument("--net", type=str, default="pose_hrnet_w32")
    parser.add_argument("--track", action="store_true")

    args = parser.parse_args()
    TrainingSession(args).run()
