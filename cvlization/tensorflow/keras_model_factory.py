from glob import glob
from typing import List, Callable, Optional, Union, Dict, Tuple
from dataclasses import dataclass
import logging
import pandas as pd
from os import path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses

try:
    # TODO: use importlib?
    from tensorflow_addons.optimizers import LAMB, AdamW
except ImportError:
    pass

from ..tensorflow.eager_model import EagerModel
from ..tensorflow.model import Model
from ..specs import DataColumnType, ModelInput, ModelTarget, EnsembleModelTarget
from ..tensorflow.metrics.multiclass_auc import MulticlassAUC
from .encoder.keras_image_encoder import KerasImageEncoder
from .encoder.keras_mlp_encoder import KerasMLPEncoder
from .aggregator.keras_aggregator import KerasAggregator
from .head.keras_head import KerasHead


LOGGER = logging.getLogger(__name__)


def get_default_custom_objects() -> dict:
    return {
        "MulticlassAUC": MulticlassAUC,
        "EagerModel": EagerModel,
    }


@dataclass
class KerasModelCheckpoint:
    model: keras.Model
    epochs_done: int


@dataclass
class KerasModelFactory:
    """Create customized keras functional API models.

    model_fn: a function that takes in keras input tensors or layers,
        and returns output tensors.
    """

    # TODO: implement task specific defaults for model_fn.

    model_inputs: List[ModelInput]
    model_targets: List[ModelTarget]
    # Can be a folder, a path or a wildcard.
    model_checkpoint_path: str = None
    recompile_checkpointed_model: bool = True
    reuse_full_model: bool = True
    reuse_image_encoder: bool = False
    freeze_image_encoder: bool = False
    model_fn: Optional[Callable] = None
    eager: Optional[bool] = False
    optimizer_name: Optional[str] = "Adam"
    lr: Optional[float] = 0.001
    n_gradients: Optional[int] = 1
    epochs: Optional[int] = None
    steps_per_epoch: Optional[int] = None
    weight_decay: Optional[float] = 0.0001
    min_decay: Optional[float] = 0.01

    # image_encoder is by default shared by all image inputs.
    # Specify either image_encoder or image_encoder_kwargs.
    # image_encoder_kwargs won't have effect if image_encoder is given.
    # TODO: consider removing image_encoder field, if image_encoder_kwargs is a preferred input.
    image_encoder: Optional[
        Union[KerasImageEncoder, Dict[str, KerasImageEncoder]]
    ] = None
    image_backbone_layer_name: str = None
    image_encoder_kwargs: Optional[Dict] = None
    share_image_encoder: Optional[bool] = True

    # mlp_encoder applies on concatenated tabular data by default.
    mlp_encoder: Optional[KerasMLPEncoder] = None
    mlp_encoder_kwargs: Optional[Dict] = None
    aggregator: Optional[KerasAggregator] = None
    aggregator_kwargs: Optional[Dict] = None

    # For keras model loading.
    custom_objects: Optional[Dict] = None

    def create_model(self) -> KerasModelCheckpoint:
        return self()

    def _load_model_checkpoint_and_recompile_if_needed(
        self, loss_functions, loss_weights, metric_functions
    ):
        model_checkpoint = self.load_model_checkpoint()
        if model_checkpoint is not None:
            LOGGER.warning("Loading the model checkpoint.")
            # TODO: print a high level description about the model. Perhaps the subclassed keras model
            #   can contain some model spec info (model inputs and model targets).
            LOGGER.warning("Skipping contruction of a new model.")
            if self.recompile_checkpointed_model:
                LOGGER.warning(
                    "Recompile the model with an optimizer, losses and metrics."
                )
                model_checkpoint.model.compile(
                    loss=loss_functions,
                    loss_weights=loss_weights,
                    metrics=metric_functions,
                    optimizer=self._create_optimizer(),
                    run_eagerly=self.eager,
                )
                model_checkpoint.model._input_aware_metric_map = getattr(
                    self, "_input_aware_metric_map", {}
                ).copy()

            else:
                model_checkpoint.model._input_aware_metric_map = getattr(
                    self, "_input_aware_metric_map", {}
                ).copy()

            return model_checkpoint

    def get_losses_and_metrics(self):
        loss_functions = []
        loss_weights = []
        metric_functions = []
        self._input_aware_metric_map = {}

        for idx, head in enumerate(self._heads):
            loss_functions.append(head.loss_function)
            loss_weights.append(head.loss_weight)
            metrics_for_head = []
            for metric in head.metrics:
                if hasattr(metric, "update_state_with_inputs_and_outputs"):
                    metric_name = metric.name
                    if metric_name in self._input_aware_metric_map:
                        # ensure stable unique metric names per target
                        unique_name = f"{metric_name}_{head.layer.name}"
                        if hasattr(metric, "_set_name"):
                            metric._set_name(unique_name)
                        else:
                            metric._name = unique_name  # fallback
                        metric_name = unique_name
                    self._input_aware_metric_map.setdefault(metric_name, []).append(idx)
                metrics_for_head.append(metric)
            metric_functions.append(metrics_for_head)

        return loss_functions, loss_weights, metric_functions

    def _extract_image_backbone(self, model: keras.Model):
        LOGGER.info("***** Extracting image backbone.")
        if self.image_backbone_layer_name:
            try:
                return model.get_layer(self.image_backbone_layer_name)
            except ValueError:
                LOGGER.error(
                    f"Could not find image backbone layer {self.image_backbone_layer_name}."
                )
                LOGGER.error(f"Available layers: {[(l.name, l) for l in model.layers]}")
                raise

        for layer in model.layers:
            LOGGER.info(f"layer: {layer}")
            if layer.__class__.__name__ == "Functional":
                # TODO: want to check for functional layer. Is this the way to do this?
                return layer

    def __call__(self) -> KerasModelCheckpoint:
        (
            loss_functions,
            loss_weights,
            metric_functions,
        ) = self.get_losses_and_metrics()

        model_checkpoint = self._load_model_checkpoint_and_recompile_if_needed(
            loss_functions, loss_weights, metric_functions
        )
        if not model_checkpoint:
            LOGGER.info("No model checkpoint found. A new model will be created.")
        else:
            if self.reuse_full_model:
                LOGGER.info(
                    "Re-using the full model. Parameters `reuse_image_encoder` and `freeze_image_encoder` has no effect."
                )
                return model_checkpoint
            elif self.reuse_image_encoder:
                # TODO: KerasImageEncoder should keep track of both backbone and non-backbone layers. It should save
                #   these layer names to a file, so that when loading the full model, we can load the full image
                #   encoder, rather than just the backbone.
                LOGGER.info("To reuse the image encoder.")
                if self.image_encoder is not None:
                    LOGGER.warning(
                        "Reusing image encoder from checkpoint. Overiding the image encoder passed in."
                    )
                image_backbone = self._extract_image_backbone(model_checkpoint.model)
                if self.freeze_image_encoder:
                    LOGGER.info(
                        "Freezing the image backbone model in the image encoder."
                    )
                    image_backbone.trainable = False
                if self.image_encoder:
                    # Replace image backbone.
                    self.image_encoder.backbone = image_backbone
                else:
                    self.image_encoder = KerasImageEncoder(
                        backbone=image_backbone,
                        **(self.image_encoder_kwargs or {}),
                    )
                LOGGER.info(f"Reusing image_encoder: {self.image_encoder}")
            else:
                LOGGER.info("Not reusing the checkpointed model.")

        LOGGER.info("Creating model ...")
        self._prepare_encoder_models()
        self._create_aggregator_if_needed()
        self._create_mlp_encoder_if_needed()
        inputs = self.create_keras_inputs()
        LOGGER.info(f"Model inputs: {inputs}")
        outputs = self.model_fn(inputs)
        if self.eager:
            model = EagerModel(
                inputs=inputs, outputs=outputs, n_gradients=self.n_gradients
            )
        else:
            # Lazy model.
            if self.n_gradients is None or self.n_gradients <= 1:
                model = keras.Model(inputs=inputs, outputs=outputs)
                LOGGER.warning(
                    "No gradient accumulation. Using the default Keras Model class."
                )
            else:
                model = Model(
                    inputs=inputs, outputs=outputs, n_gradients=self.n_gradients
                )

        model.compile(
            loss=loss_functions,
            loss_weights=loss_weights,
            metrics=metric_functions,
            optimizer=self._create_optimizer(),
            run_eagerly=self.eager,
        )
        model._input_aware_metric_map = getattr(
            self, "_input_aware_metric_map", {}
        ).copy()
        LOGGER.info("model constructed in model factory")
        model.summary()
        # model.summary(expand_nested=True)

        # Fail early
        model.to_json()

        return KerasModelCheckpoint(model, epochs_done=0)

    def __post_init__(self):
        if self.model_fn is None:
            self.model_fn = self.default_model_fn
        self._heads = [
            KerasHead.from_model_target(model_target)
            for model_target in self.model_targets
        ]

    def load_functional_model_and_convert_to_custom_model(self, model_path):
        custom_objects = self.custom_objects or get_default_custom_objects()
        LOGGER.info(f"custom_objects: {custom_objects}")
        # model loaded is a Functional API model.
        model = keras.models.load_model(
            model_path, custom_objects=custom_objects, compile=False
        )
        custom_model = Model.from_functional_model(model, n_gradients=self.n_gradients)
        return custom_model

    def load_model_checkpoint(self) -> KerasModelCheckpoint:
        LOGGER.info(
            f"Trying to load model checkpoint from {self.model_checkpoint_path}"
        )
        if self.model_checkpoint_path is None or len(self.model_checkpoint_path) == 0:
            LOGGER.warning(
                f"Empty model checkpoint path: '{self.model_checkpoint_path}'."
            )
            return None
        if self.model_checkpoint_path.endswith(".keras") and path.isfile(
            self.model_checkpoint_path
        ):
            model = self.load_functional_model_and_convert_to_custom_model(
                self.model_checkpoint_path
            )
            return KerasModelCheckpoint(model=model, epochs_done=0)
        else:
            history_csv_path = path.join(self.model_checkpoint_path, "history.csv")
            if path.isfile(history_csv_path):
                try:
                    df = pd.read_csv(history_csv_path)
                    epochs_done = len(df)
                except Exception:
                    LOGGER.warning("History CSV is empty.")
                    epochs_done = 0
            else:
                epochs_done = 0
            files = glob(self.model_checkpoint_path)
            if len(files) == 0:
                raise FileNotFoundError(
                    f"No checkpoint files found at {self.model_checkpoint_path}"
                )

            if len(files) == 1 and path.isdir(files[0]):
                h5_files = glob(path.join(files[0], "*.h5"))
                if len(h5_files) > 0:
                    model_path = h5_files[-1]
                    LOGGER.info(f"Found existing model at: {model_path}")
                    model = self.load_functional_model_and_convert_to_custom_model(
                        model_path
                    )
                    LOGGER.info(f"loaded model's attributes: {dir(model)}")
                    return KerasModelCheckpoint(model=model, epochs_done=epochs_done)
                pb_files = glob(path.join(files[0], "*.pb")) + glob(
                    path.join(files[0], "*.pbtxt")
                )
                if len(pb_files) > 0:
                    LOGGER.info(f"Found existing model at: {files[0]}")
                    model = self.load_functional_model_and_convert_to_custom_model(
                        files[0]
                    )
                    return KerasModelCheckpoint(model=model, epochs_done=epochs_done)
            elif len(files) >= 1 and path.isfile(files[-1]):
                LOGGER.info(f"Found existing model at: {files[-1]}")
                if files[-1].endswith(".h5"):
                    model = self.load_functional_model_and_convert_to_custom_model(
                        files[-1]
                    )
                    return KerasModelCheckpoint(model=model, epochs_done=epochs_done)
                else:
                    raise NotImplementedError

    def create_keras_inputs(self):
        # Use ModelInput.shape
        layer_names = {}

        def _next_layer_name(layer_name: str):
            parts = layer_name.split("_")
            try:
                idx = int(parts[-1])
                return "_".join(parts[:-1]) + "_" + str(idx + 1)
            except ValueError:
                return layer_name + "_" + str(1)

        def _get_layer_name(model_input):
            layer_name = model_input.key
            count = 0
            while layer_name in layer_names:
                layer_name = _next_layer_name(layer_name)
                count += 1
                if count > 100:
                    raise ValueError(
                        f"Could not create a unique layer name for {model_input.key}, after many attempts."
                    )
            layer_names[layer_name] = True
            return layer_name

        input_layers = [
            layers.Input(shape=model_input.shape, name=_get_layer_name(model_input))
            for model_input in self.model_inputs
        ]
        return input_layers

    def _apply_encoders(self, keras_inputs) -> List[Tuple[str, tf.Tensor]]:
        groups_and_encoded_tensors = []
        tensors_not_encoded = []
        for input_layer, encoder_model, model_input in zip(
            keras_inputs, self._encoder_models, self.model_inputs
        ):
            if encoder_model is not None:
                encoded = encoder_model(input_layer)
                for input_group in model_input.input_groups or []:
                    groups_and_encoded_tensors.append((input_group, encoded))
            else:
                tensors_not_encoded.append(input_layer)

        # All tabular features are concatenated and fed into the mlp encoder.
        if len(tensors_not_encoded) > 1:
            mlp_input_tensor = layers.Concatenate()(tensors_not_encoded)
        elif len(tensors_not_encoded) == 1:
            mlp_input_tensor = tensors_not_encoded[0]
        else:
            mlp_input_tensor = None

        if mlp_input_tensor is not None:
            encoded_by_mlp = self.mlp_encoder(mlp_input_tensor)
            groups_and_encoded_tensors.append(
                (ModelInput.DEFAULT_INPUT_GROUP(), encoded_by_mlp)
            )
        LOGGER.info(f"Encoded tensors: {groups_and_encoded_tensors}")
        return groups_and_encoded_tensors

    def _get_aggregator(self, name: str):
        if name == ModelInput.DEFAULT_INPUT_GROUP():
            return self.aggregator
        return self.aggregator

    def _apply_aggregators(
        self, groups_and_encoded_tensors: List[Tuple[str, tf.Tensor]]
    ) -> dict:
        grouped_tensors = {}
        aggregated_tensors = {}
        for group, tensor in groups_and_encoded_tensors:
            grouped_tensors.setdefault(group, []).append(tensor)
        for group, tensors in grouped_tensors.items():
            if len(tensors) == 1:
                aggregated_tensors[group] = tensors[0]
            else:
                aggregator_fn = self._get_aggregator(group)
                aggregated_tensors[group] = aggregator_fn(tensors)
        return aggregated_tensors

    def _apply_heads(self, aggregated_tensors: dict) -> list:
        output_tensors = []
        for model_target, head in zip(self.model_targets, self._heads):
            if isinstance(model_target, EnsembleModelTarget):
                output_tensor = None
            elif model_target.input_group is None or model_target.input_group in [
                "",
                ModelInput.DEFAULT_INPUT_GROUP(),
            ]:
                input_group = ModelInput.DEFAULT_INPUT_GROUP()
                output_tensor = head.layer(aggregated_tensors[input_group])
            else:
                input_group = model_target.input_group
                output_tensor = head.layer(aggregated_tensors[input_group])
            output_tensors.append(output_tensor)
        LOGGER.info(f"Output tensors after applying heads: {output_tensors}")
        return output_tensors

    def _apply_ensemble_heads(self, output_tensors: list) -> list:
        for i, (model_target, head) in enumerate(zip(self.model_targets, self._heads)):
            if isinstance(model_target, EnsembleModelTarget):
                assert (
                    output_tensors[i] is None
                ), f"Expecting output tensor {i} to be None."
                target_group = model_target.target_group_to_ensemble
                output_tensors_to_ensemble = [
                    tensor
                    for child_target, tensor in zip(self.model_targets, output_tensors)
                    if target_group
                    in (
                        child_target.target_groups
                        + [ModelTarget.DEFAULT_TARGET_GROUP()]
                    )
                ]
                output_tensor = head.ensemble_fn(output_tensors_to_ensemble)
                output_tensors[i] = output_tensor
        LOGGER.info(f"Output tensors after applying ensemble heads: {output_tensors}")
        return output_tensors

    def default_model_fn(self, keras_inputs: List[layers.Input]) -> List[tf.Tensor]:
        """The default model function that produces the output tensors from keras input layers.

        Args:
            keras_inputs (List[layers.Input]): Keras input layers

        Returns:
            List[tf.Tensor]: Output tensors that should correspond to losses and metrics.
        """
        tensors_encoded = self._apply_encoders(keras_inputs)
        tensors_aggregated = self._apply_aggregators(tensors_encoded)
        output_tensors = self._apply_heads(tensors_aggregated)
        output_tensors = self._apply_ensemble_heads(output_tensors)
        for i, output_tensor in enumerate(output_tensors):
            LOGGER.info(f"Output tensor {i}: {output_tensor.shape}")
        return output_tensors

    def _create_optimizer(self) -> keras.optimizers.Optimizer:
        LOGGER.info(f"Creating optimizer: {self.optimizer_name}")
        if self.optimizer_name == "LAMB":
            optimizer_class = LAMB
        elif self.optimizer_name == "LAMB-Cosine":
            optimizer_class = LAMB
            # TODO investigate these hypers, esp. the last one
            lr_decayed_fn = keras.optimizers.schedules.CosineDecay(
                self.lr, self.epochs * self.steps_per_epoch, self.min_decay
            )
            return optimizer_class(learning_rate=lr_decayed_fn)
        elif self.optimizer_name == "AdamW":
            return AdamW(learning_rate=self.lr, weight_decay=self.weight_decay)
        elif hasattr(keras.optimizers, self.optimizer_name):
            optimizer_class = getattr(keras.optimizers, self.optimizer_name)
        else:
            raise ValueError(f"Optimizer name not supported: {self.optimizer_name}")

        return optimizer_class(learning_rate=self.lr)

    def _create_image_encoder_if_needed(self):
        if self.image_encoder is None:
            kwargs = self.image_encoder_kwargs or {}
            self.image_encoder = KerasImageEncoder(**kwargs)

    def _create_mlp_encoder_if_needed(self):
        if self.mlp_encoder is None:
            kwargs = self.mlp_encoder_kwargs or {}
            self.mlp_encoder = KerasMLPEncoder(**kwargs)

    def _create_aggregator_if_needed(self):
        if self.aggregator is None:
            kwargs = self.mlp_encoder_kwargs or {}
            self.aggregator = KerasAggregator(**kwargs)

    def _create_binary_classifier(self, name_prefix=None):
        if not name_prefix:
            name = "binary_classifier"
        else:
            name = name_prefix + "_binary_classifier"
        return layers.Dense(1, activation="linear", name=name)

    def _create_multiclass_classifier(self, n_classes: int, name_prefix=None):
        if not name_prefix:
            name = "classifier"
        else:
            name = name_prefix + "_classifier"
        return layers.Dense(n_classes, activation="linear", name=name)

    def _create_regressor(self, name_prefix=None):
        if not name_prefix:
            name = "regressor"
        else:
            name = name_prefix + "_regressor"
        return layers.Dense(1, name=name)

    def _prepare_encoder_models(self):
        self._encoder_models = []
        if self.share_image_encoder:
            self._shared_image_encoder = None
        for model_input in self.model_inputs:
            if model_input.column_type == DataColumnType.IMAGE:
                if self.share_image_encoder:
                    self._create_image_encoder_if_needed()
                    self._encoder_models.append(self.image_encoder)
                else:
                    kwargs = self.image_encoder_kwargs or {}
                    image_encoder = KerasImageEncoder(**kwargs)
                    self._encoder_models.append(image_encoder)
            elif model_input.column_type in [
                DataColumnType.NUMERICAL,
                DataColumnType.CATEGORICAL,
                DataColumnType.BOOLEAN,
            ]:
                self._encoder_models.append(None)
            else:
                raise NotImplementedError(
                    f"Data column type {model_input.column_type} not supported."
                )

    def _get_model_summary(self, model):
        if not model:
            return ""
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        return "\n".join(stringlist)
