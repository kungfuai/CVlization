"""Backward-compatible import for the legacy training pipeline."""
import warnings

from .cross_framework_training_pipeline import CrossFrameworkTrainingPipeline

warnings.warn(
    "`cvlization.legacy_training_pipeline.LegacyTrainingPipeline` is deprecated; "
    "use `CrossFrameworkTrainingPipeline` from `cvlization.cross_framework_training_pipeline` instead.",
    DeprecationWarning,
    stacklevel=2,
)

LegacyTrainingPipeline = CrossFrameworkTrainingPipeline

__all__ = ["LegacyTrainingPipeline", "CrossFrameworkTrainingPipeline"]
