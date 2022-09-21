from dataclasses import dataclass, field
import logging
import numpy as np
from typing import List, Callable, Union

from .specs.data_column import DataColumnType
from .data.splitted_dataset import SplittedDataset
from .data.dataset_builder import DatasetBuilder, DatasetProvider
from .tensorflow.aggregator.keras_aggregator import KerasAggregator
from .specs import ModelSpec, MLFramework
from .specs import ensure_dataset_shapes_and_types


LOGGER = logging.getLogger(__name__)



class DataReducer:
    """
    R. A. Fisher, 1921: "In its most concrete form, the object of statistical methods is the reduction of data."
    https://royalsocietypublishing.org/doi/pdf/10.1098/rsta.1922.0009

    A DataReducer implements this view, by having one and only method: _reduce, which takes
    data and no other arguments as input. No optimizer options, no model architecture options, no training loop options.
    By having this restriction, the DataReducer is encouraged to be self contained recipes, and ready
    to receive data. The user of DataReducers can then focus on getting the data ready.

    This is in contract to learning systems augmented with causal inference, where the reduce() method
    takes as input not only the data, but a causal diagram that encodes domain knowledge.
    """
    def _reduce(self,
        training_dataset=None, validation_dataset=None, test_dataset=None,
        dataset_builder=None,
        data_module=None,
    ):
        raise NotImplementedError


# TODO: Re-define TrainingPipeline as a DataReducer.

@dataclass
class TrainingPipeline:
    """
    A TrainingPipeline object manages the model and trainer.
    """

    ml_framework: MLFramework = MLFramework.PYTORCH

    # Model
    #   Can be a ModelSpec, nn.Module/LightningModule, keras.Model, a python function to transform tensors (for keras)
    model: Union[
        ModelSpec, Callable
    ] = None  # If specified, the following parameters will be ignored.
    prediction_task: ModelSpec = None
    loss_function_included_in_model: bool = False

    # Precision
    precision: str = "fp32"  # "fp16", "fp32"

    # Data
    num_workers: int = 0
    collate_method: Union[str, Callable] = None  # "zip", None

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
    check_val_every_n_epoch: int = 5
    reduce_lr_patience: int = 5
    early_stop_patience: int = 10

    # Logging
    experiment_tracker: str = None
    experiment_name: str = "cvlab"
    run_name: str = None

    # Debugging
    data_only: bool = False  # No training, only run through data.

    # TODO: Consider using Experiment to create_model, create_trainer, and TrainingSession to handle hyperparameter sweep on Experiments.
    def __post_init__(self):
        # add more input args: model config, optimizer config, training_loop_config, validation_config, etc.
        self._adjust_batch_size_if_doing_data_only_debugging()
        self._populate_model_spec_based_on_user_provided_model()

    def run(self):
        """Run the training pipeline.

        Call this method after create_dataloaders(), create_model(), create_trainer() have been called."""
        if self.data_only:
            LOGGER.info("Running in data-only mode for debugging.")
            self._iterate_through_data()
        else:
            LOGGER.info(f"Running the trainer: {self.trainer}")
            self.trainer.run()

    def create_dataloaders(
        self, dataset_builder: Union[SplittedDataset, DatasetBuilder]
    ):
        train_data = self.create_training_dataloader(dataset_builder)
        val_data = self.create_validation_dataloader(dataset_builder)

        if self.ml_framework == MLFramework.TENSORFLOW:
            if self.train_steps_per_epoch is not None:
                if hasattr(train_data, "repeat"):
                    train_data = train_data.repeat()
            if self.val_steps_per_epoch is not None:
                if hasattr(val_data, "repeat"):
                    val_data = val_data.repeat()
        elif self.ml_framework == MLFramework.PYTORCH:
            if self.train_steps_per_epoch is not None:
                if hasattr(train_data, "repeat"):
                    train_data = train_data.repeat()
            if self.val_steps_per_epoch is not None:
                if hasattr(val_data, "repeat"):
                    val_data = val_data.repeat()
            # TODO: remove
            # if dataset_builder.dataset_key.endswith("tfds") and hasattr(
            #     dataset_builder, "transform_training_dataset_tf"
            # ):
            #     train_data = dataset_builder.transform_training_dataset_tf(train_data)
            #     val_data = dataset_builder.transform_validation_dataset_tf(val_data)

        self.train_data = train_data
        self.val_data = val_data

        # Data type and shape checks.
        # TODO: need to use type and shape checks applicable to data loader.
        batch = next(iter(train_data))

        def ensure_list(x):
            if isinstance(x, list):
                return x
            if isinstance(x, tuple):
                return list(x)
            return [x]

        if self.collate_method != "zip":
            batch = tuple([ensure_list(x) for x in batch])

            if isinstance(batch[0], list):
                if self.get_model_inputs() is not None:
                    assert len(self.get_model_inputs()) == len(
                        batch[0]
                    ), f"{len(self.get_model_inputs())} model inputs expected, {len(batch[0])} actual arrays."
                else:
                    LOGGER.warning(
                        f"Saw {len(batch[0])} input arrays, please check if this is expected."
                        " If model spec is provided, the check can be done automatically."
                    )
            else:
                if self.get_model_inputs() is not None:
                    assert (
                        len(self.get_model_inputs()) == 1
                    ), f"Input arrays is not a list, indicating it is probably a single input. But {len(self.get_model_inputs())} model inputs expected."
                else:
                    LOGGER.warning(
                        f"Input arrays is not a list, indicating it is probably a single input."
                        " Please check if this is expected."
                    )
            if self.get_model_inputs() is not None:
                # Some asserts about ml framework, column types. To be refactored.
                for model_input, array in zip(self.get_model_inputs(), batch[0]):
                    if model_input.column_type == DataColumnType.IMAGE:
                        assert (
                            len(array.shape) == 4
                        ), f"Image batch has shape {array.shape}. Training dataset is {self.train_data} with batch size {self.train_data.batch_size}"
                        dataset_provider = self._get_dataset_provider(dataset_builder)
                        if dataset_provider is None:
                            if self.ml_framework == MLFramework.PYTORCH:
                                assert array.shape[1] in [
                                    1,
                                    3,
                                ], f"image batch has shape {array.shape}. Expect channels_first format when dataset_provider is None"

                model_spec = ModelSpec(
                    model_inputs=self.get_model_inputs(),
                    model_targets=self.get_model_targets(),
                )
                ensure_dataset_shapes_and_types(
                    model_spec=model_spec, dataset=train_data
                )

        return self

    def create_trainer(self):
        if self.data_only:
            return self
        if self.ml_framework == MLFramework.TENSORFLOW:
            self.trainer = self._create_keras_trainer(
                self.model, self.train_data, self.val_data
            )
        elif self.ml_framework == MLFramework.PYTORCH:
            from pytorch_lightning.core import LightningModule

            assert isinstance(self.model, LightningModule)
            self.trainer = self._create_torch_trainer(
                self.model,
                self.train_data,
                self.val_data,
            )
        return self

    def create_model(self):
        if self.data_only:
            # Do nothing.
            return self
        if self._model_spec_is_provided():
            self.model = self.create_model_from_spec()
            LOGGER.info(f"Model created from spec: {type(self.model)}")
            return self
        elif self._model_is_provided():
            self._check_the_user_providered_model()
            LOGGER.info(f"Using the model provided by the user: {type(self.model)}")
            return self
        else:
            raise ValueError(
                f"model must be a ModelSpec or a Callable object, but got {self.model}"
            )

    def _model_spec_is_provided(self):
        return isinstance(self.model, ModelSpec)

    def _model_is_provided(self):
        return callable(self.model)

    def _check_the_user_providered_model(self):
        if self.ml_framework == MLFramework.TENSORFLOW:
            LOGGER.info(f"Using the tensorflow model passed in: {self.model}")
            self.model = self._ensure_keras_model()
        elif self.ml_framework == MLFramework.PYTORCH:
            LOGGER.info(f"Using the torch model passed in: {self.model}")
            self.model = self._ensure_torch_model()
        return self

    def _adjust_batch_size_if_doing_data_only_debugging(self):
        if self.data_only:
            self.train_batch_size = 1

    def _populate_model_spec_based_on_user_provided_model(self):
        if isinstance(self.model, ModelSpec):
            self.model_spec = self.model
        elif isinstance(self.prediction_task, ModelSpec):
            self.model_spec = self.prediction_task
        else:
            self.model_spec = None

    def create_model_from_spec(self):
        """
        Build the neural network model architecture and initialize the weights,
        according to model_spec.
        """
        LOGGER.info("Creating model from spec.")
        LOGGER.info(str(self.model_spec))

        if self.ml_framework == MLFramework.TENSORFLOW:
            self.model = self._create_keras_model_from_spec()
        elif self.ml_framework == MLFramework.PYTORCH:
            self.model = self._create_torch_model_from_spec()
        return self.model

    def convert_data_rows_to_tf_dataset(self, tuple_iterator):
        import tensorflow as tf

        assert tuple_iterator is not None, f"tuple_iterator is None"

        def gen():
            # TODO: the output of gen() is not flexible.
            for example in tuple_iterator:
                inputs = example[0]
                targets = example[1]
                yield (
                    tuple([tf.convert_to_tensor(x) for x in inputs]),
                    tuple([tf.convert_to_tensor(x) for x in targets]),
                )

        output_signature = (
            tuple(
                [tf.TensorSpec(shape=None, dtype=tf.float32)]
                * len(self.prediction_task.model_inputs)
            ),
            tuple(
                [tf.TensorSpec(shape=None, dtype=tf.float32)]
                * len(self.prediction_task.model_targets)
            ),
        )
        return tf.data.Dataset.from_generator(
            gen,
            output_signature=output_signature,
        )

    def convert_tuple_iterator_to_tf_dataset(
        self, tuple_iterator, source_image_is_channels_first=True
    ):
        # TODO: move this method to a data adaptor class under `cvlization.keras`.
        import tensorflow as tf

        def gen():
            # TODO: the output of gen() is not flexible.
            for example in tuple_iterator:
                inputs = example[0]
                if isinstance(inputs, list):
                    image = inputs[0]
                else:
                    image = inputs
                targets = example[1]
                if isinstance(targets, list):
                    label = targets[0]
                else:
                    label = targets
                if hasattr(image, "numpy"):
                    image = image.numpy()
                if hasattr(label, "numpy"):
                    label = label.numpy()
                label = np.array(label).astype(np.float32)
                if source_image_is_channels_first:
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
        dataset_provider = self._get_dataset_provider(dataset_builder)
        if dataset_provider == DatasetProvider.TENSORFLOW_DATASETS:
            # Dataset is already batched.
            # TODO: we should expect an already augmented dataset when calling DatasetBuilder.training_dataset().
            # TODO: dataset builder should handle shuffling.
            if (
                hasattr(dataset_builder, "shuffle_size")
                and dataset_builder.shuffle_size
            ):
                training_dataset = training_dataset.shuffle(
                    dataset_builder.shuffle_size
                )
            training_dataset = training_dataset.map(
                normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
            return training_dataset
        elif dataset_provider in [DatasetProvider.TORCHVISION, None]:
            ds = self.convert_tuple_iterator_to_tf_dataset(
                training_dataset, source_image_is_channels_first=True
            )
            return ds.batch(self.train_batch_size)
        elif dataset_provider == DatasetProvider.CVLIZATION:
            ds = self.convert_data_rows_to_tf_dataset(training_dataset)
            return ds.batch(self.train_batch_size)
        else:
            raise ValueError("Unknown dataset provider: {}".format(dataset_provider))

    def create_tf_validation_dataloader(self, dataset_builder: DatasetBuilder):
        ds = dataset_builder.validation_dataset()
        dataset_provider = self._get_dataset_provider(dataset_builder)
        if dataset_provider == DatasetProvider.TENSORFLOW_DATASETS:
            return ds.batch(self.val_batch_size)
        elif dataset_provider in [DatasetProvider.TORCHVISION, None]:
            return self.convert_tuple_iterator_to_tf_dataset(ds).batch(
                self.val_batch_size
            )
        elif dataset_provider == DatasetProvider.CVLIZATION:
            ds = self.convert_data_rows_to_tf_dataset(ds)
            return ds.batch(self.val_batch_size)
        else:
            raise ValueError("Unknown dataset provider: {}".format(dataset_provider))

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
            self.ml_framework == MLFramework.TENSORFLOW
        ):  # This is the framework of the trainer, not the dataset.
            return self.create_tf_training_dataloader(dataset_builder)
        elif self.ml_framework == MLFramework.PYTORCH:
            import torch

            dataset_provider = self._get_dataset_provider(dataset_builder)
            if dataset_provider in [
                DatasetProvider.TORCHVISION,
                DatasetProvider.CVLIZATION,
                None,
            ]:
                train_ds = dataset_builder.training_dataset()
                LOGGER.info(f"Training data: {len(train_ds)} examples")
                dl = torch.utils.data.DataLoader(
                    train_ds,
                    batch_size=self.train_batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                    collate_fn=self.create_collate_fn(),
                )
                return dl
            elif dataset_provider == DatasetProvider.TENSORFLOW_DATASETS:
                training_dataset = dataset_builder.training_dataset()
                training_dataset = self.convert_tf_dataset_to_iterable_dataset(
                    training_dataset
                )
                return torch.utils.data.DataLoader(
                    training_dataset,
                    batch_size=self.train_batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                    collate_fn=self.create_collate_fn(),
                )
            else:
                raise ValueError(f"Unknown dataset provider: {dataset_provider}")

        raise ValueError(f"Unknown ML framework: {self.ml_framework}")

    def _get_dataset_provider(
        self, dataset_builder: DatasetBuilder
    ) -> Union[DatasetProvider, None]:
        if hasattr(dataset_builder, "dataset_provider"):
            return dataset_builder.dataset_provider
        else:
            return None

    def create_validation_dataloader(self, dataset_builder: DatasetBuilder):
        dataset_provider = self._get_dataset_provider(dataset_builder)
        if self.ml_framework == MLFramework.TENSORFLOW:
            return self.create_tf_validation_dataloader(dataset_builder)
        elif self.ml_framework == MLFramework.PYTORCH:
            if dataset_provider in [
                DatasetProvider.TORCHVISION,
                DatasetProvider.CVLIZATION,
                None,
            ]:
                import torch

                val_ds = dataset_builder.validation_dataset()
                LOGGER.info(f"Validation data: {len(val_ds)} examples")
                dl = torch.utils.data.DataLoader(
                    val_ds,
                    batch_size=self.val_batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    collate_fn=self.create_collate_fn(),
                )
                return dl
            elif dataset_provider == DatasetProvider.TENSORFLOW_DATASETS:
                validation_dataset = dataset_builder.validation_dataset()
                validation_dataset = self.convert_tf_dataset_to_iterable_dataset(
                    validation_dataset
                )
                return torch.utils.data.DataLoader(
                    dataset_builder.validation_dataset(),
                    batch_size=self.val_batch_size,
                )
            else:
                raise ValueError(f"Unknown dataset provider: {dataset_provider}")

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

    def _ensure_keras_model(self):
        from tensorflow import keras

        if self.model is not None and callable(self.model):
            if not isinstance(self.model, keras.Model):
                raise ValueError(f"model must be a keras.Model, got {type(self.model)}")
            return self.model
        elif self.model is not None:
            raise ValueError(f"model must be callable, got {type(self.model)}")
        return self.model

    def _create_keras_model_from_spec(self):

        from .tensorflow.keras_model_factory import KerasModelFactory
        from .tensorflow.encoder.keras_image_encoder import KerasImageEncoder
        from .tensorflow.encoder.keras_image_backbone import create_image_backbone

        model_spec = self.model_spec
        model_inputs = model_spec.get_model_inputs()
        model_targets = model_spec.get_model_targets()

        if callable(self.model_spec.image_backbone):
            backbone = self.model_spec.image_backbone
        else:
            backbone = create_image_backbone(
                name=model_spec.image_backbone,
                pretrained=model_spec.pretrained,
                pooling=model_spec.image_pool,
                input_shape=model_spec.input_shape,
            )
        image_encoder = (
            KerasImageEncoder(
                backbone=backbone,
                dropout=model_spec.dropout,
                pool_name=model_spec.image_pool,
                dense_layer_sizes=model_spec.dense_layer_sizes,
                permute_image=model_spec.permute_image,
            )
            if model_spec.image_backbone
            else None
        )
        aggregator = KerasAggregator(image_feature_pooling_method=model_spec.image_pool)
        model_factory = KerasModelFactory(
            model_inputs=model_inputs,
            model_targets=model_targets,
            image_encoder=image_encoder,
            aggregator=aggregator,
            optimizer_name=self.optimizer_name,
            n_gradients=self.n_gradients,
            lr=self.lr,
            epochs=self.epochs,  # TODO: both model factory and trainer have this parameter!
        )
        ckpt = model_factory.create_model()
        model = ckpt.model

        return model

    def _create_keras_trainer(self, model, train_dataset, val_dataset):
        from .tensorflow.keras_trainer import KerasTrainer
        from tensorflow import keras

        callbacks = [
            keras.callbacks.ReduceLROnPlateau(patience=self.reduce_lr_patience),
            keras.callbacks.EarlyStopping(patience=self.early_stop_patience),
        ]
        if self.experiment_tracker == "wandb":
            import wandb
            from wandb.keras import WandbCallback

            # TODO: move this out to experiment level.
            wandb.init(project="cvlab")
            if self.run_name:
                wandb.run.name = self.run_name
            callbacks.append(
                WandbCallback(
                    log_gradients=True,
                    log_weights=True,
                    training_data=train_dataset.take(10),
                    validation_data=val_dataset,
                    validation_steps=5,
                )
            )

        config = self
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
        from .torch.net.image_classification.davidnet.dawn_utils import net, Network

        if self.optimizer_name == "SGD_david":
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
                epochs=self.epochs,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
            )
            return trainer

        from .torch.torch_trainer import TorchTrainer

        config = self
        trainer = TorchTrainer(
            model=model,
            model_inputs=self.get_model_inputs(),
            model_targets=self.get_model_targets(),
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_batch_size=config.train_batch_size,
            val_batch_size=config.val_batch_size,
            epochs=config.epochs,
            train_steps_per_epoch=config.train_steps_per_epoch,
            val_steps_per_epoch=config.val_steps_per_epoch,
            num_workers=config.num_workers,
            experiment_tracker=config.experiment_tracker,
            experiment_name=config.experiment_name,
            run_name=config.run_name,
            n_gradients=config.n_gradients,
            precision=config.precision,
            loss_function_included_in_model=config.loss_function_included_in_model,
            collate_method=config.collate_method,
            check_val_every_n_epoch=config.check_val_every_n_epoch,
        )
        # TODO: experiment tracker should be an experiment-level object.
        trainer.log_params(config.__dict__)

        return trainer

    def _ensure_torch_model(self):
        from pytorch_lightning.core.lightning import LightningModule
        from torch import nn
        from .torch.torch_model import TorchLitModel

        if self.model is not None and callable(self.model):
            if isinstance(self.model, LightningModule):
                if hasattr(self.model, "lr"):
                    self.model.lr = self.lr
                else:
                    raise ValueError("Model does not have a lr attribute.")
                return self.model
            elif isinstance(self.model, nn.Module):
                return TorchLitModel(
                    config=TorchLitModel.TorchModelConfig(
                        model_inputs=self.get_model_inputs(),
                        model_targets=self.get_model_targets(),
                        model=self.model,
                        optimizer_name=self.optimizer_name,
                        optimizer_kwargs=self.optimizer_kwargs,
                        n_gradients=self.n_gradients,
                        lr=self.lr,
                        epochs=self.epochs,
                        lr_scheduler_name=self.lr_scheduler_name,
                        lr_scheduler_kwargs=self.lr_scheduler_kwargs,
                    ),
                )
            else:
                raise ValueError(
                    f"model must be a LightningModule or torch.nn.Module, got {type(self.model)}"
                )
        elif self.model is not None:
            raise ValueError(f"model must be callable, got {type(self.model)}")
        return self.model

    def _create_torch_model_from_spec(self):
        from .torch.torch_model_factory import TorchModelFactory
        from .torch.encoder.torch_image_encoder import TorchImageEncoder
        from .torch.encoder.torch_image_backbone import create_image_backbone
        from pytorch_lightning.core import LightningModule
        from .torch.torch_model import TorchLitModel

        model_inputs = self.get_model_inputs()
        model_targets = self.get_model_targets()
        image_encoder = TorchImageEncoder(
            backbone=create_image_backbone(
                self.model_spec.image_backbone,
                pretrained=self.model_spec.pretrained,
                in_chans=self.model_spec.input_shape[-1],
            ),
            permute_image=self.model_spec.permute_image,
            customize_conv1=self.model_spec.customize_conv1,
            dense_layer_sizes=self.model_spec.dense_layer_sizes,
        )
        ckpt = TorchModelFactory(
            model_inputs=model_inputs,
            model_targets=model_targets,
            image_encoder=image_encoder,
            optimizer_name=self.optimizer_name,
            optimizer_kwargs=self.optimizer_kwargs,
            n_gradients=self.n_gradients,
            lr=self.lr,
            epochs=self.epochs,
            lr_scheduler_name=self.lr_scheduler_name,
            lr_scheduler_kwargs=self.lr_scheduler_kwargs,
        ).create_model()
        model = ckpt.model

        if not isinstance(model, LightningModule):
            model = TorchLitModel(
                config=TorchLitModel.TorchModelConfig(
                    model_inputs=self.get_model_inputs(),
                    model_targets=self.get_model_targets(),
                    model=self.model,
                    optimizer_name=self.optimizer_name,
                    optimizer_kwargs=self.optimizer_kwargs,
                    n_gradients=self.n_gradients,
                    lr=self.lr,
                    epochs=self.epochs,
                    lr_scheduler_name=self.lr_scheduler_name,
                    lr_scheduler_kwargs=self.lr_scheduler_kwargs,
                ),
            )
        assert isinstance(model, LightningModule)
        return model

    def create_collate_fn(self):
        if self.collate_method == "zip":

            def collate_fn(batch):
                return tuple(zip(*batch))

            return collate_fn
        elif callable(self.collate_method):
            return self.collate_method
        else:
            return None

    def get_model_inputs(self):
        if self.model_spec:
            return self.model_spec.get_model_inputs()

    def get_model_targets(self):
        if self.model_spec:
            return self.model_spec.get_model_targets()
