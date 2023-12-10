"""
Adapted from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial8/Deep_Energy_Models.html

Once the training starts, you start a tensorboard:

tensorboard --logdir lightning_logs/version_xxx

Example usage (and documentation):

https://colab.research.google.com/drive/1Cm3CITo9deSV_MrrIMpOn4pTLr2NihRm?authuser=1#scrollTo=Tpsi93eyHjs6

TODO: remove mnist-specific logic.
"""
from dataclasses import dataclass
import os
import torch.utils.data as data
from torchvision.datasets import MNIST
from torchvision import transforms
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from cvlization.data.dataset_builder import TransformedMapStyleDataset
from .lightning import (
    DeepEnergyModel,
    GenerateCallback,
    SamplerCallback,
    OutlierCallback,
)


DATASET_PATH = "./data/raw/MNIST"


@dataclass
class PipelineResult:
    """TODO: use this to as a return type for the pipeline.
    During fit(), the PipelineResult should be updated.
    """

    model_checkpoint_dir: str = None
    lightning_logs_path: str = None
    tensorboard_logs_dir: str = None
    instructions: str = """
    Once the training starts, you start a tensorboard:

    tensorboard --logdir lightning_logs/version_{pl_version}
    """

    @property
    def lightning_module(self):
        raise NotImplementedError("Need to load the lightning module ")


@dataclass
class TrainingPipeline:
    # Device
    device: str = "cuda"

    # Model
    backbone: str = "resnet18"  # TODO: not implemented! See lightning.py

    # Data
    img_shape: tuple = (1, 28, 28)
    train_batch_size: int = 128
    val_batch_size: int = 128

    # Optimizer
    lr: float = 1e-4
    epochs: int = 60

    # Persistence
    checkpoint_path: str = None
    name: str = "uva_energy"

    def fit(self, dataset_builder):
        train_loader, val_loader = self._create_dataloaders(dataset_builder)
        trainer = self._create_trainer()
        model = self._create_model()
        trainer.fit(model, train_loader, val_loader)
        # Finally, pick the best model
        # model = DeepEnergyModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    def _create_dataloaders(self, dataset_builder):
        train_raw = dataset_builder.training_dataset()
        val_raw = dataset_builder.validation_dataset()
        transform = transforms.Compose(
            [
                transforms.Resize(self.img_shape[1:]),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        def transform_example(example):
            image, target = example
            image = transform(image)
            return image, target

        train_set = TransformedMapStyleDataset(train_raw, transform=transform_example)
        test_set = TransformedMapStyleDataset(val_raw, transform=transform_example)
        # We define a set of data loaders that we can use for various purposes later.
        # Note that for actually training a model, we will use different data loaders
        # with a lower batch size.
        # print("first training example:", train_set[0])
        train_loader = data.DataLoader(
            train_set,
            batch_size=self.train_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
        )
        test_loader = data.DataLoader(
            test_set,
            batch_size=self.val_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4,
        )
        # print("train_loader first batch:", next(iter(train_loader)))
        return train_loader, test_loader

    def _create_trainer(self):
        trainer = pl.Trainer(
            # default_root_dir=os.path.join(self.checkpoint_path, "MNIST"),
            accelerator="gpu" if str(self.device).startswith("cuda") else "cpu",
            devices=1,
            max_epochs=self.epochs,
            # For debugging
            # limit_train_batches=10,
            # limit_val_batches=10,
            gradient_clip_val=0.1,
            logger=TensorBoardLogger("./", name=self.name),
            # logger=MLFlowLogger(experiment_name="MNIST_uva_energy"),
            callbacks=[
                ModelCheckpoint(
                    save_weights_only=True,
                    mode="min",
                    monitor="val_contrastive_divergence",
                ),
                GenerateCallback(every_n_epochs=5),
                SamplerCallback(every_n_epochs=1),
                OutlierCallback(),
                LearningRateMonitor("epoch"),
            ],
        )
        assert hasattr(
            trainer.logger.experiment, "add_image"
        ), f"Logger {trainer.logger} does not support adding images. Consider using one that can."
        return trainer

    def _create_model(self):
        # Check whether pretrained model exists. If yes, load it and skip training
        if self.checkpoint_path:
            pretrained_filename = os.path.join(self.checkpoint_path, "MNIST.ckpt")
            if os.path.isfile(pretrained_filename):
                print("Found pretrained model, loading...")
                model = DeepEnergyModel.load_from_checkpoint(pretrained_filename)
        else:
            pl.seed_everything(42)
            model = DeepEnergyModel(
                img_shape=self.img_shape,
                batch_size=self.train_batch_size,
                lr=self.lr,
                beta1=0.0,
            )
        return model


if __name__ == "__main__":

    class MNISTDatasetBuilder:
        def training_dataset(self):
            return MNIST(root=DATASET_PATH, train=True, download=True)

        def validation_dataset(self):
            return MNIST(root=DATASET_PATH, train=False, download=True)

    TrainingPipeline().fit(MNISTDatasetBuilder())
