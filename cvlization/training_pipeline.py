from dataclasses import dataclass, field
import logging
from typing import List, Callable

from .data.splitted_dataset import SplittedDataset
from .keras.aggregator.keras_aggregator import KerasAggregator
from .specs import ModelSpec, MLFramework


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
    # TODO: perhaps make it friendly for parameter sweep.
    def __init__(
        self,
        framework: MLFramework,
        config: TrainingPipelineConfig,
    ):
        # add more input args: model config, optimizer config, training_loop_config, validation_config, etc.
        # TODO: rename ModelSpec to PredictionSpec ?
        self.framework = framework
        self.config = config

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

    def prepare_datasets(self, dataset: SplittedDataset):
        train_data = dataset.training_dataset(batch_size=self.config.train_batch_size)
        # TODO: the dataset need to know about model inputs and model targets.
        # if isinstance(train_data, RichDataFrame):
        #     train_data = MLDataset(
        #         model_inputs=model_inputs,
        #         model_targets=model_targets,
        #         data_rows=train_data,
        #     )
        val_data = dataset.validation_dataset(batch_size=self.config.val_batch_size)
        # if isinstance(val_data, RichDataFrame):
        #     val_data = MLDataset(
        #         model_inputs=model_inputs,
        #         model_targets=model_targets,
        #         data_rows=val_data,
        #     )

        if self.framework == MLFramework.TENSORFLOW:
            # Use tf.data API.
            train_data = dataset.transform_training_dataset_tf(train_data)
            val_data = dataset.transform_validation_dataset_tf(val_data)
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
            if dataset.dataset_key.endswith("tfds") and hasattr(
                dataset, "transform_training_dataset_tf"
            ):
                train_data = dataset.transform_training_dataset_tf(train_data)
                val_data = dataset.transform_validation_dataset_tf(val_data)

        self.train_data = train_data
        self.val_data = val_data
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
            self._run_through_data()
        self.trainer.run()

    def _run_through_data(self):
        LOGGER.info("Running through data without model training.")

    def _create_keras_model(self):
        from tensorflow import keras

        if self.model is not None and callable(self.model):
            if not isinstance(self.model, keras.Model):
                raise ValueError(f"model must be a keras.Model, got {type(self.model)}")
            return self.model
        elif self.model is not None:
            raise ValueError(f"model must be callable, got {type(self.model)} instead")

        from .keras.keras_model_factory import KerasModelFactory
        from .keras.encoder.keras_image_encoder import KerasImageEncoder
        from .keras.encoder.keras_image_backbone import create_image_backbone

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
        from .keras.keras_trainer import KerasTrainer
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
