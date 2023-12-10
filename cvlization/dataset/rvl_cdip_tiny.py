from dataclasses import dataclass, field
import json
import logging
import random
from typing import Any, List, Tuple
from datasets import load_dataset
from ..data.dataset_builder import Dataset, DatasetProvider


LOGGER = logging.getLogger(__name__)


@dataclass
class RvlCdipTinyDatasetBuilder:
    max_length: int = 8
    # TODO: image_size is not used?
    image_size: List[int] = field(default_factory=lambda: [2560, 1920])
    ignore_id: int = -100
    task_start_token: str = "<s>"
    prompt_end_token: str = None
    sort_json_key: bool = True

    @property
    def dataset_provider(self):
        return DatasetProvider.HUGGINGFACE
    
    def training_dataset(self) -> Dataset:
        hf_ds = load_dataset("nielsr/rvl_cdip_10_examples_per_class_donut")["train"]
        return hf_ds
    
    def validation_dataset(self) -> Dataset:
        hf_ds = load_dataset("nielsr/rvl_cdip_10_examples_per_class_donut")["test"]
        return hf_ds
    
