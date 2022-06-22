from dataclasses import dataclass
import json
from mmdet.datasets import build_dataset
from cvlization.lab.penn_fudan_pedestrian import PennFudanPedestrianDatasetBuilder
from cvlization.torch.net.panoptic_segmentation.mmdet import (
    MMPanopticSegmentationModels,
    MMDatasetAdaptor,
    MMTrainer,
)


@dataclass
class COCOPanopticDatasetBuilder:
    train_annotation_path: str = (
        "/home/ubuntu/zz/coco/annotations/panoptic_val2017_first500.json"
    )
    val_annotation_path: str = (
        "/home/ubuntu/zz/coco/annotations/panoptic_val2017_last200.json"
    )
    img_dir = "/home/ubuntu/zz/coco/val2017"
    seg_dir = "/home/ubuntu/zz/coco/annotations/panoptic_val2017"

    def __post_init__(self):
        a = json.load(open(self.val_annotation_path))
        self.things_classes = [c["name"] for c in a["categories"] if c["isthing"] == 1]
        self.stuff_classes = [c["name"] for c in a["categories"] if c["isthing"] == 0]
        self.classes = [c["name"] for c in a["categories"]]

    def download(self):
        pass


class TrainingSession:
    # TODO: integrate with Experiment interface.
    #   Create a MMDetTrainer class.

    def __init__(self, args):
        self.args = args

    def run(self):
        dsb = COCOPanopticDatasetBuilder()
        self.model, self.cfg = self.create_model(
            num_things_classes=len(dsb.things_classes),
            num_stuff_classes=len(dsb.stuff_classes),
        )
        self.datasets = self.create_dataset(
            self.cfg,
            train_annotation_path=dsb.train_annotation_path,
            val_annotation_path=dsb.val_annotation_path,
            img_dir=dsb.img_dir,
            seg_dir=dsb.seg_dir,
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
