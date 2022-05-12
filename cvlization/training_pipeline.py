from dataclasses import dataclass, field
import logging
import numpy as np
from typing import List, Callable, Union

from cvlization.specs.data_column import DataColumnType

from .data.splitted_dataset import SplittedDataset
from .data.dataset_builder import DatasetBuilder, DatasetProvider
from .tensorflow.aggregator.keras_aggregator import KerasAggregator
from .specs import ModelSpec, MLFramework
from .specs import ensure_dataset_shapes_and_types


LOGGER = logging.getLogger(__name__)


@dataclass
class TrainingPipelineConfig:
    # Model
    model: Callable = None  # If specified, the following parameters will be ignored.
    image_backbone: str = None  # e.g. "resnet50"
    image_pool: str = "avg"  # "avg", "max", "flatten"
    dense_layer_sizes: List[int] = field(default_factory=list)
    input_shape: List[int] = field(
        default_factory=lambda: [None, None, 3]
    )  # e.g. [224, 224, 3]
    dropout: float = 0
    pretrained: bool = False
    permute_image: bool = False
    customize_conv1: bool = False

    # Precision
    precision: str = "fp32"  # "fp16", "fp32"

    # Data
    num_workers: int = 0

    # Optimizer
    lr: float = 0.0001
    optimizer_name: str = "Adam"
    optimizer_kwargs: dict = None
    lr_scheduler_name: str = None
    lr_scheduler_kwargs: dict = None
    n_gradients: int = 1  # for gradient accumulation
    epochs: int = 10
    train_batch_size: int = 32
    train_steps_per_epoch: int = None
    val_batch_size: int = 32
    val_steps_per_epoch: int = None
    reduce_lr_patience: int = 5
    early_stop_patience: int = 10

    # Logging
    experiment_tracker: str = None
    experiment_name: str = "cvlab"
    run_name: str = None

    # Debugging
    data_only: bool = False  # No training, only run through data.


class TrainingPipeline:
    # TODO: Consider folding TrainingPipeline into Trainer.
    # TODO: Consider using Experiment to create_model, create_trainer, and TrainingSession to handle hyperparameter sweep on Experiments.
    def __init__(
        self,
        framework: MLFramework,
        config: TrainingPipelineConfig,
    ):
        # add more input args: model config, optimizer config, training_loop_config, validation_config, etc.
        # TODO: rename ModelSpec to PredictionSpec ?
        self.framework = framework
        self.config = config
        if self.config.data_only:
            self.config.train_batch_size = 1

    def create_model(self, model_spec: ModelSpec):
        self.model_inputs = model_spec.get_model_inputs()
        self.model_targets = model_spec.get_model_targets()
        if self.config.data_only:
            return self
        if self.framework == MLFramework.TENSORFLOW:
            self.model = self._create_keras_model()
        elif self.framework == MLFramework.PYTORCH:
            self.model = self._create_torch_model()
        return self

    def convert_tuple_iterator_to_tf_dataset(self, tuple_iterator, channel_first=True):
        # TODO: move this method to a data adaptor class under `cvlization.keras`.
        import tensorflow as tf

        def gen():
            for example in tuple_iterator:
                image = example[0]
                label = example[1]
                if hasattr(image, "numpy"):
                    image = image.numpy()
                if hasattr(label, "numpy"):
                    label = label.numpy()
                label = np.array(label).astype(np.float32)
                if channel_first:
                    image = image.transpose(1, 2, 0)
                yield (
                    tuple([tf.convert_to_tensor(image)]),
                    tuple([tf.convert_to_tensor(label)]),
                )

        output_signature = (
            tuple([tf.TensorSpec(shape=None, dtype=tf.float32)]),
            tuple([tf.TensorSpec(shape=None, dtype=tf.float32)]),
        )

        ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
        return ds

    def create_tf_training_dataloader(self, dataset_builder: DatasetBuilder):
        import tensorflow as tf
        from .tensorflow.transforms.image_augmentation import normalize

        training_dataset = dataset_builder.training_dataset()
        if dataset_builder.dataset_provider == DatasetProvider.TENSORFLOW_DATASETS:
            # Dataset is already batched.
            # TODO: we should expect an already augmented dataset when calling DatasetBuilder.training_dataset().
            # TODO: dataset builder should handle shuffling.
            if dataset_builder.shuffle_size:
                training_dataset = training_dataset.shuffle(
                    dataset_builder.shuffle_size
                )
            training_dataset = training_dataset.map(
                normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
            return training_dataset
        elif dataset_builder.dataset_provider in [DatasetProvider.TORCHVISION, None]:
            ds = self.convert_tuple_iterator_to_tf_dataset(training_dataset)
            return ds.batch(self.config.train_batch_size)
        else:
            raise ValueError(
                "Unknown dataset provider: {}".format(dataset_builder.dataset_provider)
            )

    def create_tf_validation_dataloader(self, dataset_builder: DatasetBuilder):
        ds = dataset_builder.validation_dataset()
        if dataset_builder.dataset_provider == DatasetProvider.TENSORFLOW_DATASETS:
            return ds.batch(self.config.val_batch_size)
        elif dataset_builder.dataset_provider in [DatasetProvider.TORCHVISION, None]:
            return self.convert_tuple_iterator_to_tf_dataset(ds).batch(
                self.config.val_batch_size
            )
        else:
            raise ValueError(
                "Unknown dataset provider: {}".format(dataset_builder.dataset_provider)
            )

    def convert_tf_dataset_to_iterable_dataset(self, tf_dataset):
        # TODO: allow using imgaug as a default
        # TODO: transforms should be in a separate class
        # https://github.com/kungfuai/mtrx_2/blob/1b5ff963f4b732883e95e1f86dfbecbb95a7a9ff/src/data/transforms.py#L31
        import torch

        class IterableImageDataset(torch.utils.data.IterableDataset):
            def __iter__(self):
                for image, label in tf_dataset:
                    image = image.numpy() / 255
                    label = label.numpy()
                    image = torch.cat(
                        [torch.unsqueeze(image[i], 0) for i in range(len(image))]
                    )
                    yield image, label

        return IterableImageDataset()

    def create_training_dataloader(self, dataset_builder: DatasetBuilder):
        """
        Dataloader is more closely coupled with the trainer than the dataset, and is dependent on
        the ML framework.
        """
        if (
            self.framework == MLFramework.TENSORFLOW
        ):  # This is the framework of the trainer, not the dataset.
            return self.create_tf_training_dataloader(dataset_builder)
        elif self.framework == MLFramework.PYTORCH:
            if dataset_builder.dataset_provider == DatasetProvider.TORCHVISION:
                import torch

                return torch.utils.data.DataLoader(
                    dataset_builder.training_dataset(),
                    batch_size=self.config.train_batch_size,
                    shuffle=True,
                    num_workers=self.config.num_workers,
                )
            elif (
                dataset_builder.dataset_provider == DatasetProvider.TENSORFLOW_DATASETS
            ):
                training_dataset = dataset_builder.training_dataset()
                training_dataset = self.convert_tf_dataset_to_iterable_dataset(
                    training_dataset
                )
                return torch.utils.data.DataLoader(
                    training_dataset,
                    batch_size=self.config.train_batch_size,
                    shuffle=True,
                    num_workers=self.config.num_workers,
                )
            else:
                raise ValueError(
                    f"Unknown dataset provider: {dataset_builder.dataset_provider}"
                )

        raise ValueError(f"Unknown ML framework: {self.framework}")

    def create_validation_dataloader(self, dataset_builder: DatasetBuilder):
        if self.framework == MLFramework.TENSORFLOW:
            return self.create_tf_validation_dataloader(dataset_builder)
        elif self.framework == MLFramework.PYTORCH:
            if dataset_builder.dataset_provider == DatasetProvider.TORCHVISION:
                import torch

                return torch.utils.data.DataLoader(
                    dataset_builder.validation_dataset(),
                    batch_size=self.config.val_batch_size,
                )
            elif (
                dataset_builder.dataset_provider == DatasetProvider.TENSORFLOW_DATASETS
            ):
                validation_dataset = dataset_builder.validation_dataset()
                validation_dataset = self.convert_tf_dataset_to_iterable_dataset(
                    validation_dataset
                )
                return torch.utils.data.DataLoader(
                    dataset_builder.validation_dataset(),
                    batch_size=self.config.val_batch_size,
                )

    def prepare_datasets(self, dataset_builder: Union[SplittedDataset, DatasetBuilder]):

        train_data = self.create_training_dataloader(dataset_builder)
        val_data = self.create_validation_dataloader(dataset_builder)

        if self.framework == MLFramework.TENSORFLOW:

            # batch = next(iter(train_data))
            # raise ValueError(str(batch[0][0].shape))
            if self.config.train_steps_per_epoch is not None:
                if hasattr(train_data, "repeat"):
                    train_data = train_data.repeat()
            if self.config.val_steps_per_epoch is not None:
                if hasattr(val_data, "repeat"):
                    val_data = val_data.repeat()
        elif self.framework == MLFramework.PYTORCH:
            if self.config.train_steps_per_epoch is not None:
                if hasattr(train_data, "repeat"):
                    train_data = train_data.repeat()
            if self.config.val_steps_per_epoch is not None:
                if hasattr(val_data, "repeat"):
                    val_data = val_data.repeat()
            if dataset_builder.dataset_key.endswith("tfds") and hasattr(
                dataset_builder, "transform_training_dataset_tf"
            ):
                train_data = dataset_builder.transform_training_dataset_tf(train_data)
                val_data = dataset_builder.transform_validation_dataset_tf(val_data)

        self.train_data = train_data
        self.val_data = val_data

        # Data type and shape checks.
        batch = next(iter(train_data))
        assert len(self.model_inputs) == len(
            batch[0]
        ), f"{len(self.model_inputs)} model inputs expected, {len(batch[0])} actual arrays"
        for model_input, array in zip(self.model_inputs, batch[0]):
            if model_input.column_type == DataColumnType.IMAGE:
                assert len(array.shape) == 4, f"image batch has shape {array.shape}"
                if dataset_builder.dataset_provider == None:
                    assert array.shape[1] in [
                        1,
                        3,
                    ], f"image batch has shape {array.shape}. Expect channel_first format when dataset_provider is None"

        model_spec = ModelSpec(
            model_inputs=self.model_inputs, model_targets=self.model_targets
        )
        ensure_dataset_shapes_and_types(model_spec=model_spec, dataset=train_data)

        return self

    def create_trainer(self):
        if self.config.data_only:
            return self
        if self.framework == MLFramework.TENSORFLOW:
            self.trainer = self._create_keras_trainer(
                self.model, self.train_data, self.val_data
            )
        elif self.framework == MLFramework.PYTORCH:
            self.trainer = self._create_torch_trainer(
                self.model, self.train_data, self.val_data
            )
        return self

    def run(self):
        if self.config.data_only:
            self._iterate_through_data()
        else:
            self.trainer.run()

    def _iterate_through_data(self):
        LOGGER.info("Running through data without model training.")

        def display_dict_or_list_of_tensors(tensors):
            if isinstance(tensors, dict):
                for k, v in tensors.items():
                    LOGGER.info(f"item {k}:")
                    display_dict_or_list_of_tensors(v)
            for j, tensor in enumerate(tensors):
                if hasattr(tensor, "shape"):
                    LOGGER.info(
                        f"  item {j} ({type(tensor)}) has shape: {tensor.shape}"
                    )
                else:
                    LOGGER.info(f"  item {j} ({type(tensor)}) has no shape.")

        prefix = "training"
        for i, batch in enumerate(self.train_data):
            if i >= 10:
                LOGGER.info("Showing first 10 examples.")
                break
            LOGGER.info(f"{prefix} example {i}: {len(batch)} items.")
            if isinstance(batch, dict) or hasattr(batch, "keys"):
                for k, v in batch.items():
                    LOGGER.info(f"  item {k}:")
                    display_dict_or_list_of_tensors(v)
            elif hasattr(batch, "shape"):
                LOGGER.info(f"  shape: {batch.shape}")
            elif isinstance(batch, list) or isinstance(batch, tuple):
                for j, item in enumerate(batch):
                    if hasattr(item, "shape"):
                        LOGGER.info(
                            f"  example {i} part {j} ({type(item)}) has shape: {item.shape}"
                        )
                    else:
                        LOGGER.info(f"  example {i} part {j}:")
                        display_dict_or_list_of_tensors(item)

    def _create_keras_model(self):
        from tensorflow import keras

        if self.config.model is not None and callable(self.config.model):
            if not isinstance(self.config.model, keras.Model):
                raise ValueError(
                    f"model must be a keras.Model, got {type(self.config.model)}"
                )
            return self.config.model
        elif self.config.model is not None:
            raise ValueError(f"model must be callable, got {type(self.config.model)}")

        from .tensorflow.keras_model_factory import KerasModelFactory
        from .tensorflow.encoder.keras_image_encoder import KerasImageEncoder
        from .tensorflow.encoder.keras_image_backbone import create_image_backbone

        model_inputs = self.model_inputs
        model_targets = self.model_targets
        # TODO: use model_config that is passed in during init.
        config = self.config

        if callable(config.image_backbone):
            backbone = config.image_backbone
        else:
            backbone = create_image_backbone(
                name=config.image_backbone,
                pretrained=config.pretrained,
                pooling=config.image_pool,
                input_shape=config.input_shape,
            )
        image_encoder = (
            KerasImageEncoder(
                backbone=backbone,
                dropout=self.config.dropout,
                pool_name=config.image_pool,
                dense_layer_sizes=config.dense_layer_sizes,
                permute_image=config.permute_image,
            )
            if config.image_backbone
            else None
        )
        aggregator = KerasAggregator(image_feature_pooling_method=config.image_pool)
        model_factory = KerasModelFactory(
            model_inputs=model_inputs,
            model_targets=model_targets,
            image_encoder=image_encoder,
            aggregator=aggregator,
            optimizer_name=config.optimizer_name,
            n_gradients=config.n_gradients,
            lr=config.lr,
            epochs=config.epochs,  # TODO: both model factory and trainer have this parameter!
        )
        ckpt = model_factory.create_model()
        model = ckpt.model

        return model

    def _create_keras_trainer(self, model, train_dataset, val_dataset):
        from .tensorflow.keras_trainer import KerasTrainer
        from tensorflow.keras import callbacks

        callbacks = [
            callbacks.ReduceLROnPlateau(patience=self.config.reduce_lr_patience),
            callbacks.EarlyStopping(patience=self.config.early_stop_patience),
        ]
        if self.config.experiment_tracker == "wandb":
            import wandb
            from wandb.keras import WandbCallback

            # TODO: move this out to experiment level.
            wandb.init(project="cvlab")
            if self.config.run_name:
                wandb.run.name = self.config.run_name
            callbacks.append(
                WandbCallback(
                    log_gradients=True,
                    log_weights=True,
                    training_data=train_dataset.take(10),
                    validation_data=val_dataset,
                    validation_steps=5,
                )
            )

        config = self.config
        trainer = KerasTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=config.epochs,
            train_steps_per_epoch=config.train_steps_per_epoch,
            val_steps_per_epoch=config.val_steps_per_epoch,
            callbacks=callbacks,
        )
        return trainer

    def _create_torch_trainer(self, model, train_dataset, val_dataset):
        from .torch.net.davidnet.dawn_utils import net, Network

        if self.config.optimizer_name == "SGD_david":
            import torch
            from .torch.torch_trainer import DavidTrainer

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = Network(net()).to(device).half()

            class WrappedModel(torch.nn.Module):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.sub_model = model

                def forward(self, x):
                    outputs = self.sub_model(x)
                    # See dawn_utils.py.
                    try:
                        # return outputs["pool"]
                        return outputs["logits"]
                        # return outputs["layer3/residual/add"]
                    except KeyError:
                        print(outputs.keys())
                        raise

            # model = WrappedModel().to(device).half()
            trainer = DavidTrainer(
                model=model,
                epochs=self.config.epochs,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
            )
            return trainer

        from .torch.torch_trainer import TorchTrainer

        config = self.config
        trainer = TorchTrainer(
            model=model,
            model_inputs=self.model_inputs,
            model_targets=self.model_targets,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=config.epochs,
            train_steps_per_epoch=config.train_steps_per_epoch,
            val_steps_per_epoch=config.val_steps_per_epoch,
            num_workers=config.num_workers,
            experiment_tracker=config.experiment_tracker,
            experiment_name=config.experiment_name,
            run_name=config.run_name,
            n_gradients=config.n_gradients,
            precision=config.precision,
        )
        # TODO: experiment tracker should be an experiment-level object.
        trainer.log_params(config.__dict__)
        trainer.log_params({"framework": self.framework.value})

        return trainer

    def _create_torch_model(self):
        from pytorch_lightning.core.lightning import LightningModule
        from torch import nn
        from .torch.torch_model import TorchModel

        if self.model is not None and callable(self.model):
            if isinstance(self.model, LightningModule):
                return self.model
            elif isinstance(self.model, nn.Module):
                return TorchModel(
                    config=TorchModel.TorchModelConfig(
                        model_inputs=self.model_inputs,
                        model_targets=self.model_targets,
                        model=self.model,
                        optimizer_name=self.config.optimizer_name,
                        optimizer_kwargs=self.config.optimizer_kwargs,
                        n_gradients=self.config.n_gradients,
                        lr=self.config.lr,
                        epochs=self.config.epochs,
                        lr_scheduler_name=self.config.lr_scheduler_name,
                        lr_scheduler_kwargs=self.config.lr_scheduler_kwargs,
                    ),
                )
            else:
                raise ValueError(
                    f"model must be a LightningModule or torch.nn.Module, got {type(self.model)}"
                )
        elif self.model is not None:
            raise ValueError(f"model must be callable, got {type(self.model)}")

        from .torch.torch_model_factory import TorchModelFactory
        from .torch.encoder.torch_image_encoder import TorchImageEncoder
        from .torch.encoder.torch_image_backbone import create_image_backbone

        model_inputs = self.model_inputs
        model_targets = self.model_targets
        image_encoder = TorchImageEncoder(
            backbone=create_image_backbone(
                self.config.image_backbone,
                pretrained=self.config.pretrained,
                in_chans=self.config.input_shape[-1],
            ),
            permute_image=self.config.permute_image,
            customize_conv1=self.config.customize_conv1,
            dense_layer_sizes=self.config.dense_layer_sizes,
        )
        ckpt = TorchModelFactory(
            model_inputs=model_inputs,
            model_targets=model_targets,
            image_encoder=image_encoder,
            optimizer_name=self.config.optimizer_name,
            optimizer_kwargs=self.config.optimizer_kwargs,
            n_gradients=self.config.n_gradients,
            lr=self.config.lr,
            epochs=self.config.epochs,
            lr_scheduler_name=self.config.lr_scheduler_name,
            lr_scheduler_kwargs=self.config.lr_scheduler_kwargs,
        ).create_model()
        model = ckpt.model
        return model
