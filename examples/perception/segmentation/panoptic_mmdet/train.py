import os
from pathlib import Path
try:
    from mmdet.datasets import build_dataset
except ImportError:
    print("mmdet not installed")
    print("For torch 1.11.*:")
    print("pip install mmdet==2.24.1")
    print(
        "pip install -U mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.11.0/index.html"
    )
    print("For torch 1.12.*:")
    print("pip install mmdet==2.25.1")
    print(
        "pip install -U mmcv-full==1.6.1 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.12.0/index.html"
    )
    raise
from cvlization.dataset.coco_panoptic_tiny import CocoPanopticTinyDatasetBuilder
from cvlization.torch.net.panoptic_segmentation.mmdet import (
    MMPanopticSegmentationModels,
    MMDatasetAdaptor,
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
        # Use centralized cache directory for datasets
        cache_dir = str(get_cache_dir())
        print(f"Using cache directory: {cache_dir}")
        dsb = CocoPanopticTinyDatasetBuilder(preload=True, flavor=None, data_dir=cache_dir)
        self.model, self.cfg = self.create_model(
            num_things_classes=len(dsb.things_classes),
            num_stuff_classes=len(dsb.stuff_classes),
        )
        self.datasets = self.create_dataset(
            self.cfg,
            train_annotation_path=os.path.join(
                dsb.data_dir, dsb.dataset_folder, dsb.train_ann_file
            ),
            val_annotation_path=os.path.join(
                dsb.data_dir, dsb.dataset_folder, dsb.val_ann_file
            ),
            img_dir=os.path.join(dsb.data_dir, dsb.dataset_folder, dsb.img_folder),
            seg_dir=os.path.join(dsb.data_dir, dsb.dataset_folder, dsb.seg_folder),
            classes=dsb.classes,
        )
        self.trainer = self.create_trainer(self.cfg, self.args.net)
        self.cfg = self.trainer.config
        print("batch size:", self.cfg.data.samples_per_gpu)
        self.trainer.fit(
            model=self.model,
            train_dataset=self.datasets[0],
            val_dataset=self.datasets[1],
        )

    def create_model(self, num_things_classes: int, num_stuff_classes: int):
        model_registry = MMPanopticSegmentationModels(
            num_things_classes=num_things_classes, num_stuff_classes=num_stuff_classes
        )
        model_dict = model_registry[self.args.net]
        model, cfg = model_dict["model"], model_dict["config"]

        return model, cfg

    def create_dataset(
        self,
        config,
        train_annotation_path,
        val_annotation_path,
        img_dir,
        seg_dir,
        classes,
    ):
        MMDatasetAdaptor.set_dataset_info_in_config(
            config,
            train_anno_file=train_annotation_path,
            val_anno_file=val_annotation_path,
            image_dir=img_dir,
            seg_dir=seg_dir,
            classes=classes,
        )

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
