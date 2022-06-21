from mmdet.datasets import build_dataset
import numpy as np
from pprint import pprint
from cvlization.lab.penn_fudan_pedestrian import PennFudanPedestrianDatasetBuilder
from cvlization.torch.net.instance_segmentation.mmdet import (
    MMInstanceSegmentationModels,
    MMDatasetAdaptor,
    MMTrainer,
)

# pip install mmdet==2.24.1 or 2.25.0
# pip install -U mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.11.0/index.html


class TrainingSession:
    # TODO: integrate with Experiment interface.
    #   Create a MMDetTrainer class.

    def __init__(self, args):
        self.args = args

    def run(self):
        # User: Need to adjust num_classes to match the dataset.
        self.dataset_builder_cls = PennFudanPedestrianDatasetBuilder
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
        model_registry = MMInstanceSegmentationModels(num_classes=num_classes)
        model_dict = model_registry[self.args.net]
        model, cfg = model_dict["model"], model_dict["config"]
        print("******************************")
        print(cfg.pretty_text)
        # forward_fn = model.forward_test

        # def new_forward(*args, **kwargs):
        #     outputs = forward_fn(*args, **kwargs)
        #     for i, x in enumerate(outputs):
        #         if isinstance(x, list) or isinstance(x, tuple):
        #             for j, xx in enumerate(x):
        #                 if isinstance(xx, list) or isinstance(xx, tuple):
        #                     for k, xxx in enumerate(xx):
        #                         print(i, j, k, np.array(xxx).shape)
        #                 else:
        #                     print(i, j, xx.shape)
        #         elif isinstance(x, np.ndarray):
        #             print(i, x.shape)
        #     raise ValueError("aaaa")
        #     return outputs

        # model.forward_test = new_forward

        return model, cfg

    def create_dataset(self, config):
        dsb = self.dataset_builder_cls(flavor=None, to_torch_tensor=False)
        dataset_classname = MMDatasetAdaptor.adapt_and_register_detection_dataset(dsb)
        print("registered:", dataset_classname)

        MMDatasetAdaptor.set_dataset_info_in_config(
            config, dataset_builder=dsb, image_dir=dsb.image_dir
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
    python -m examples.instance_segmentation.mmdet.train
    """

    from argparse import ArgumentParser

    options = MMInstanceSegmentationModels.model_names()
    parser = ArgumentParser(
        epilog=f"""
            Options for net: {options} ({len(options)} of them).
            """
    )
    parser.add_argument("--net", type=str, default="maskrcnn_r50")
    parser.add_argument("--track", action="store_true")

    args = parser.parse_args()
    TrainingSession(args).run()
