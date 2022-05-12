from dataclasses import dataclass, field
from pyexpat import model
from typing import List
import logging
import tensorflow as tf
from tensorflow.keras import layers, losses, metrics
from ...specs import (
    ModelTarget,
    EnsembleModelTarget,
    DataColumnType,
    LossType,
    MetricType,
)


LOGGER = logging.getLogger(__name__)


@dataclass
class KerasHead:
    layer: layers.Layer
    ensemble_fn: layers.Layer = None
    loss_function: layers.Layer = None
    loss_weight: float = 1
    metrics: List = field(default_factory=list)

    @classmethod
    def from_model_target(cls, model_target: ModelTarget):
        layer = cls._get_head_layer(model_target)
        loss_weight = cls._get_loss_weight(model_target)
        loss_fn = cls._get_loss_function(model_target)
        LOGGER.info(f"adding loss for {model_target.key}: {loss_fn}")
        metric_functions = cls._get_metric_functions(model_target)
        LOGGER.info(f"adding metrics for {model_target.key}: {metric_functions}")
        ensemble_fn = None
        if isinstance(model_target, EnsembleModelTarget):
            layer_name = f"{model_target.key}_ensemble"
            if model_target.aggregation_method == "avg":
                ensemble_fn = layers.Average(name=layer_name)
            else:
                ensemble_fn = layers.Maximum(name=layer_name)
        return cls(
            layer=layer,
            loss_function=loss_fn,
            loss_weight=loss_weight,
            metrics=metric_functions,
            ensemble_fn=ensemble_fn,
        )

    @classmethod
    def _get_head_layer(cls, model_target: ModelTarget):
        key_name = (
            model_target.key.lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace(".", "")
        )
        name_prefix = key_name
        if model_target.input_group:
            name_prefix = f"{model_target.input_group}_{key_name}"
        if model_target.column_type == DataColumnType.BOOLEAN:
            layer = cls._create_binary_classifier(
                name_prefix=name_prefix,
                prefer_logits=model_target.prefer_logits,
            )
        elif model_target.column_type == DataColumnType.NUMERICAL:
            layer = cls._create_regressor(name_prefix=name_prefix)
        elif model_target.column_type == DataColumnType.CATEGORICAL:
            layer = cls._create_multiclass_classifier(
                n_classes=model_target.n_categories,
                name_prefix=name_prefix,
                prefer_logits=model_target.prefer_logits,
            )
        else:
            raise ValueError(
                f"Unsupported column type for classifier/regressor: {model_target.column_type}"
            )
        return layer

    @classmethod
    def _get_loss_weight(cls, model_target: ModelTarget):
        loss_weight = (
            1 if model_target.loss_weight is None else model_target.loss_weight
        )
        return loss_weight

    @classmethod
    def _get_loss_function(cls, model_target: ModelTarget):
        if model_target.loss == LossType.CATEGORICAL_CROSSENTROPY:
            loss_fn = losses.CategoricalCrossentropy(
                from_logits=model_target.prefer_logits
            )
        elif model_target.loss == LossType.SPARSE_CATEGORICAL_CROSSENTROPY:
            loss_fn = losses.SparseCategoricalCrossentropy(
                from_logits=model_target.prefer_logits
            )
        elif model_target.loss == LossType.BINARY_CROSSENTROPY:
            loss_fn = losses.BinaryCrossentropy(from_logits=model_target.prefer_logits)

        elif model_target.loss == LossType.MSE:
            loss_fn = losses.MeanSquaredError()
        elif model_target.loss == LossType.MAE:
            loss_fn = losses.MAE()
        else:
            raise NotImplementedError(f"Loss type unsupported: {model_target.loss}")
        return loss_fn

    @classmethod
    def _get_metric_functions(cls, model_target: ModelTarget):
        metric_functions = []
        for metric_type in model_target.metrics or []:
            if metric_type == MetricType.MULTICLASS_AUC:
                # TODO: add one metric for each
                metric_fn = MulticlassAUC()
            elif metric_type == MetricType.ACCURACY:
                if model_target.column_type == DataColumnType.BOOLEAN:
                    metric_fn = metrics.BinaryAccuracy()
                elif model_target.column_type == DataColumnType.CATEGORICAL:
                    if model_target.loss == LossType.CATEGORICAL_CROSSENTROPY:
                        metric_fn = metrics.CategoricalAccuracy()
                    elif model_target.loss == LossType.SPARSE_CATEGORICAL_CROSSENTROPY:
                        metric_fn = metrics.SparseCategoricalAccuracy()
                    else:
                        raise NotImplementedError(
                            f"Metric type unsupported: {metric_type} when loss is {model_target.loss}"
                        )
                else:
                    raise ValueError(
                        f"Accuracy metric is not supported for {model_target.column_type}"
                    )
            elif metric_type == MetricType.AUROC:
                metric_fn = metrics.AUC(
                    name="auc",
                    num_thresholds=2000,
                    from_logits=model_target.prefer_logits,
                )
            elif metric_type == MetricType.MAP:
                metric_fn = metrics.AUC(
                    curve="pr",
                    name="prauc",
                    num_thresholds=2000,
                    from_logits=model_target.prefer_logits,
                )
            elif metric_type == MetricType.MSE:
                metric_fn = metrics.MeanSquaredError()
            else:
                metric_fn = metric_type
                assert isinstance(
                    metric_type, metrics.Metric
                ), f"Metric type unsupported: {metric_type}"
            if metric_fn is not None:
                metric_functions.append(metric_fn)
        return metric_functions

    @classmethod
    def _create_binary_classifier(cls, name_prefix=None, n=1, prefer_logits=False):
        if not name_prefix:
            name = "binary_classifier"
        else:
            name = name_prefix + "_binary_classifier"
        activation = "linear" if prefer_logits else "sigmoid"
        return layers.Dense(n, activation=activation, name=name)

    @classmethod
    def _create_multiclass_classifier(
        cls, n_classes: int, name_prefix=None, prefer_logits=False
    ):
        if not name_prefix:
            name = "classifier"
        else:
            name = name_prefix + "_classifier"
        activation = "linear" if prefer_logits else "softmax"
        return layers.Dense(n_classes, activation=activation, name=name)

    @classmethod
    def _create_regressor(cls, name_prefix=None):
        if not name_prefix:
            name = "regressor"
        else:
            name = name_prefix + "_regressor"
        return layers.Dense(1, name=name)
