from dataclasses import dataclass, field
import json
import logging
import random
from typing import Any, List, Tuple
from datasets import load_dataset, load_from_disk
from ..data.dataset_builder import Dataset, DatasetProvider


LOGGER = logging.getLogger(__name__)

# FIXME: This only works on my disk!
dataset_path = "/datasets/flickr30k/processed_as_huggingface_for_donutdocai"

@dataclass
class FlickrDatasetBuilder:
    max_length: int = 768
    image_height: int = 500
    image_width: int = 500
    ignore_id: int = -100
    task_start_token: str = "<s_cord-v2>" # FIXME: Need to change this and retrain
    prompt_end_token: str = None
    sort_json_key: bool = True

    @property
    def dataset_provider(self):
        return DatasetProvider.HUGGINGFACE
    
    def training_dataset(self) -> Dataset:
        hf_ds = load_from_disk(dataset_path)["train"]
        return hf_ds
    
    def validation_dataset(self) -> Dataset:
        hf_ds = load_from_disk(dataset_path)["test"]
        return hf_ds
