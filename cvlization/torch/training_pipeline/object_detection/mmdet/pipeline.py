from typing import Callable
try:
    from mmdet.datasets import build_dataset
except ImportError:
    print("mmdet not installed")
    print("For torch 1.11.*:")
    print("pip install mmdet==2.24.1")
    print("pip install -U mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.11.0/index.html")
    print("For torch 1.12.*:")
    print("pip install mmdet==2.25.1")
    print("pip install -U mmcv-full==1.6.1 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.12.0/index.html")
    raise
from .model import (
    MMDetectionModels,
    MMDatasetAdaptor,
    MMDetectionTrainer,
)


class MMDetObjectDetection:
    # TODO: consider subclassing from this class, to create a training pipelines for specific models.
    def __init__(self, net: str, config_override_fn: Callable = None):
        # TODO: allow additional customization of the config.
        self._net = net
        self._config_override_fn = config_override_fn
    
    def fit(self, dataset_builder):
        self.train(dataset_builder)
    
    def train(self, dataset_builder):
        self.model, self.cfg = self.create_model(num_classes=dataset_builder.num_classes, net=self._net)
        self.datasets = self.create_dataset(dataset_builder, self.cfg)
        self.trainer = MMDetectionTrainer(self.cfg, self._net)
        self.cfg = self.trainer.config
        if self._config_override_fn is not None:
            self.cfg = self._config_override_fn(self.cfg)
        # Additional customization of the config here. e.g.
        #   cfg.optimizer.lr = 0.0001
        #   cfg.data.samples_per_gpu = 2
        print("batch size:", self.cfg.data.samples_per_gpu)
        print(self.datasets[0])
        self.trainer.fit(
            model=self.model,
            train_dataset=self.datasets[0],
            val_dataset=self.datasets[1],
        )
    
    @classmethod
    def model_names(cls):
        return MMDetectionModels.model_names()
    
    def create_model(self, num_classes: int, net: str):
        model_registry = MMDetectionModels(num_classes=num_classes)
        model_dict = model_registry[net]
        model, cfg = model_dict["model"], model_dict["config"]
        return model, cfg
    
    def create_dataset(self, dataset_builder, config):
        dsb = dataset_builder
        # dsb = self.dataset_builder_cls(flavor=None, to_torch_tensor=False)
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

    


MMDetObjectDetection.fit = MMDetObjectDetection.train
