from typing import List, Optional
from dataclasses import dataclass
import logging

from ..specs import ModelInput, ModelTarget

LOGGER = logging.getLogger(__name__)


@dataclass
class MLIterableDataset:
    model_inputs: List[ModelInput]
    model_targets: List[ModelTarget]
    batch_size: Optional[int] = 2
    tfrecord_paths: Optional[List[str]] = None

    def __next__(self):
        pass

    def to_tf_dataset(self):
        raise NotImplementedError
