from typing import List, Callable, Optional, Union, Dict
from dataclasses import dataclass
import logging

from tensorflow import keras
from tensorflow.keras import layers, losses

from ..keras.eager_model import EagerModel
from ..data.data_column import DataColumnType
from ..data.model_input import ModelInput
from ..data.model_target import ModelTarget
from ..losses.loss_type import LossType
from .encoder.keras_image_encoder import KerasImageEncoder
from .encoder.keras_mlp_encoder import KerasMLPEncoder
from .aggregator.keras_aggregator import KerasAggregator
from ..keras.model import Model


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


@dataclass
class KerasModelFactory:
    """Create customized keras functional API models."""

    # TODO: where is the triplet model path ???
    # TripletDataRowsFactory(data_rows, model): handle hard example mining here or in the trainer?
    # TripletDataRowsFactory[i] -> Tuple[Row, Row, Row]
    # def generate_mined_hard_examples()
    #
    # Model target: example-level target
    # batch x hidden_dim, batch x n_refs x hidden_dim
    # head 1: adjusted_prices: batch x n_refs
    # head 2: similarities: batch x n_refs, compared to abs(true prices - adjusted prices)
    #
    # TODO: allow intermediate encoding outputs in the "model spec". e.g. Just want image encoding.

    # TODO: implement task specific defaults for model_fn.

    model_inputs: List[ModelInput]
    model_targets: List[ModelTarget]
    model_fn: Optional[Callable] = None
    eager: Optional[bool] = True
    optimizer_name: Optional[str] = "Adam"
    lr: Optional[float] = 0.001
    n_gradients: Optional[int] = 5

    # image_encoder is by default shared by all image inputs.
    image_encoder: Optional[
        Union[KerasImageEncoder, Dict[str, KerasImageEncoder]]
    ] = None
    image_encoder_kwargs: Optional[Dict] = None
    share_image_encoder: Optional[bool] = True
    # mlp_encoder applies on concatenated tabular data by default.
    mlp_encoder: Optional[KerasMLPEncoder] = None
    mlp_encoder_kwargs: Optional[Dict] = None
    aggregator: Optional[KerasAggregator] = None
    aggregator_kwargs: Optional[Dict] = None

    def __call__(self) -> keras.Model:
        LOGGER.info("Creating model ...")
        inputs = self.create_keras_inputs()
        LOGGER.info(f"Model inputs: {inputs}")
        outputs = self.model_fn(inputs)
        if self.eager:
            model = EagerModel(
                inputs=inputs, outputs=outputs, n_gradients=self.n_gradients
            )
        else:
            # Lazy model.
            model = Model(inputs=inputs, outputs=outputs, n_gradients=self.n_gradients)

        loss_functions = []
        loss_weights = []
        metric_functions = []
        for j, model_target in enumerate(self.model_targets):
            loss_weights.append(model_target.loss_weight)
            if model_target.loss == LossType.CATEGORICAL_CROSSENTROPY:
                loss_fn = losses.CategoricalCrossentropy()
            elif model_target.loss == LossType.BINARY_CROSSENTROPY:
                loss_fn = losses.BinaryCrossentropy()
            elif model_target.loss == LossType.MSE:
                loss_fn = losses.MSE()
            elif model_target.loss == LossType.MAE:
                loss_fn = losses.MAE()
            else:
                raise NotImplementedError
            loss_functions.append(loss_fn)

            # TODO: create keras metrics from generic metric enums
            if model_target.metrics:
                LOGGER.info(f"{model.outputs[j]}")
                # metric_functions[model.outputs[j].name] = model_target.metrics
                metric_functions.append(model_target.metrics)
            else:
                metric_functions.append(None)

        LOGGER.info("losses and weights:")
        for loss_function, loss_weight in zip(loss_functions, loss_weights):
            LOGGER.info(f"{loss_weight}: {loss_function}")
        # metric_functions = {"digit_classifier": ["accuracy", "categorical_accuracy"]}
        model.compile(
            loss=loss_functions,
            loss_weights=loss_weights,
            metrics=metric_functions,
            optimizer=self._create_optimizer(),
            run_eagerly=self.eager,
        )
        model.summary()
        return model

    def __post_init__(self):
        if self.model_fn is None:
            self.model_fn = self.default_model_fn
        self._prepare_encoder_models()
        self._create_aggregator_if_needed()

    def create_keras_inputs(self):
        # Use ModelInput.shape
        return [
            layers.Input(shape=model_input.shape, name=model_input.key)
            for model_input in self.model_inputs
        ]

    def default_model_fn(self, keras_inputs: List[layers.Input]):
        tensors_encoded = []
        tensors_not_encoded = []
        for input_layer, encoder_model in zip(keras_inputs, self._encoder_models):
            if encoder_model is not None:
                encoded = encoder_model(input_layer)
                tensors_encoded.append(encoded)
            else:
                tensors_not_encoded.append(input_layer)

        if len(tensors_not_encoded) > 1:
            mlp_input_tensor = layers.Concatenate()(tensors_not_encoded)
        elif len(tensors_not_encoded) == 1:
            mlp_input_tensor = tensors_not_encoded[0]
        else:
            mlp_input_tensor = None

        if mlp_input_tensor is not None:
            self._create_mlp_encoder_if_needed()
            encoded_by_mlp = self.mlp_encoder(mlp_input_tensor)
            tensors_encoded.append(encoded_by_mlp)

        encoded_agg = self.aggregator(tensors_encoded)

        target_tensors = []
        # TODO: allow passing in custom heads?
        for model_target in self.model_targets:
            if model_target.column_type == DataColumnType.BOOLEAN:
                binary_classifier = self._create_binary_classifier(
                    name_prefix=model_target.key
                )
                target_tensor = binary_classifier(encoded_agg)
                target_tensors.append(target_tensor)
            elif model_target.column_type == DataColumnType.NUMERICAL:
                regressor = self._create_regressor(name_prefix=model_target.key)
                target_tensor = regressor(encoded_agg)
                target_tensors.append(target_tensor)
            elif model_target.column_type == DataColumnType.CATEGORICAL:
                classifier = self._create_multiclass_classifier(
                    n_classes=model_target.n_categories, name_prefix=model_target.key
                )
                target_tensor = classifier(encoded_agg)
                target_tensors.append(target_tensor)
        return target_tensors

    def _create_optimizer(self) -> keras.optimizers.Optimizer:
        optimizer_class = getattr(keras.optimizers, self.optimizer_name)
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
        return layers.Dense(1, activation="sigmoid", name=name)

    def _create_multiclass_classifier(self, n_classes: int, name_prefix=None):
        if not name_prefix:
            name = "classifier"
        else:
            name = name_prefix + "_classifier"
        return layers.Dense(n_classes, activation="softmax", name=name)

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


"""
TODO: provide an easier way to create a keras/torch model according to
a model plan that is composed of submodels, with intermediate encodings.

Model plan:
encoders: [
    {
        ops: [
            {key: left_cc_pixel, op: {net: resnet50, pool: avg, hidden: 128, dropout: 0.2}}
            {key: right_cc_pixel, op: {net: resnet50, pool: avg, hidden: 128, dropout: 0.2}}
            {key: age, op: {net: mlp, layer_sizes: [16, 8, 8], dropout: 0.2}}
            {key: device, op: {net: mlp, layer_sizes: [32, 8, 8], dropout: 0.2}}
        ]
    }
]
"""
