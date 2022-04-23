import enum


class MetricType(enum.Enum):
    """
    Enum class for metric types.
    """

    # Classification metrics.
    ACCURACY = "accuracy"
    AUROC = "auroc"
    MULTICLASS_AUC = "multiclass_auc"
    MAP = "map"
    # Regression metrics.
    MSE = "mse"
    MAPE = "mape"
    SMAPE = "smape"
    # Diagnostic and visualization metrics.
    TOP_MISTAKES = "top_mistakes"
    BEST_PREDICTIONS = "best_predictions"
    HEATMAP = "heatmap"
    SHAP = "shap"
