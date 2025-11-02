"""Compatibility helpers for Lightning imports.

This module wraps the modern ``lightning.pytorch`` API while providing a
fallback to the legacy ``pytorch_lightning`` package. By centralising imports
here we can migrate the rest of the codebase to the new package without having
multiple conditional imports scattered around.
"""

from __future__ import annotations

import importlib

try:  # Prefer the modern Lightning package.
    import lightning.pytorch as pl  # type: ignore
    from lightning.pytorch import seed_everything  # type: ignore
    from lightning.pytorch.callbacks import Callback, LearningRateMonitor  # type: ignore
    from lightning.pytorch.loggers import MLFlowLogger, WandbLogger  # type: ignore
    CallbacksModule = importlib.import_module("lightning.pytorch.callbacks")  # type: ignore
    LoggersModule = importlib.import_module("lightning.pytorch.loggers")  # type: ignore
except ImportError:  # pragma: no cover - legacy fallback
    import pytorch_lightning as pl  # type: ignore
    from pytorch_lightning import seed_everything  # type: ignore
    from pytorch_lightning.callbacks import Callback, LearningRateMonitor  # type: ignore
    from pytorch_lightning.loggers import MLFlowLogger, WandbLogger  # type: ignore
    CallbacksModule = importlib.import_module("pytorch_lightning.callbacks")  # type: ignore
    LoggersModule = importlib.import_module("pytorch_lightning.loggers")  # type: ignore

LightningModule = pl.LightningModule
Trainer = pl.Trainer
callbacks = CallbacksModule
loggers = LoggersModule

__all__ = [
    "pl",
    "seed_everything",
    "Callback",
    "LearningRateMonitor",
    "MLFlowLogger",
    "WandbLogger",
    "LightningModule",
    "Trainer",
    "callbacks",
    "loggers",
]
