from dataclasses import dataclass, field
import json
import logging
import random
from typing import Any, List, Tuple
from datasets import load_dataset
from ..data.dataset_builder import Dataset, DatasetProvider


LOGGER = logging.getLogger(__name__)


@dataclass
class CordV2DatasetBuilder:
    max_length: int = 768
    image_height: int = 2560
    image_width: int = 1920
    ignore_id: int = -100
    task_start_token: str = "<s_cord-v2>"
    prompt_end_token: str = None
    sort_json_key: bool = True

    @property
    def dataset_provider(self):
        return DatasetProvider.HUGGINGFACE
    
    def training_dataset(self) -> Dataset:
        hf_ds = load_dataset("naver-clova-ix/cord-v2")["train"]
        return hf_ds
    
    def validation_dataset(self) -> Dataset:
        hf_ds = load_dataset("naver-clova-ix/cord-v2")["test"]
        return hf_ds
