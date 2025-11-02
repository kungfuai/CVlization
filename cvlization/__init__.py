"""
Copyright (c) 2022 KUNGFU.AI.
All rights reserved.
"""

from .base_trainer import BaseTrainer as Trainer
from .data.ml_dataset import MLDataset
from .data.data_rows import DataRows as RichDataFrame
from .specs import ModelSpec


def configure_logging(level=None):
    """Configure default logging for consumers of the library."""
    import logging

    if level is None:
        level = logging.INFO
    # Avoid duplicate handlers if caller already configured logging.
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)",
        )
    else:
        logging.getLogger().setLevel(level)


configure_logging()
