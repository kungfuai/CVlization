"""
A huggingface interface for Google's Conceptual Captions dataset.
"""

from dataclasses import dataclass, field
import json
import logging
import random
from typing import Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import io
import urllib
import PIL.Image

from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent

from ..data.dataset_builder import Dataset, DatasetProvider


LOGGER = logging.getLogger(__name__)


# https://huggingface.co/datasets/conceptual_captions
USER_AGENT = get_datasets_user_agent()

def fetch_single_image(image_url, timeout=None, retries=0):
    for _ in range(retries + 1):
        try:
            request = urllib.request.Request(
                image_url,
                data=None,
                headers={"user-agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=timeout) as req:
                image = PIL.Image.open(io.BytesIO(req.read()))
            break
        except Exception:
            image = None
    return image


def fetch_images(batch, num_threads, timeout=None, retries=0):
    fetch_single_image_with_args = partial(fetch_single_image, timeout=timeout, retries=retries)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        batch["image"] = list(executor.map(fetch_single_image_with_args, batch["image_url"]))
    return batch

@dataclass
class ConceptualCaptionsDatasetBuilder:
    num_threads: int = 40
    desired_images: int = 100000
    cache_path: str = "/datasets/conceptual_captions_100k"
    max_length: int = 768
    image_height: int = 500
    image_width: int = 500
    ignore_id: int = -100
    task_start_token: str = "<s_caption>" # FIXME: Need to change this and retrain
    prompt_end_token: str = None
    sort_json_key: bool = True

    @property
    def dataset_provider(self):
        return DatasetProvider.HUGGINGFACE

    def load(self):
        dset = load_dataset("conceptual_captions")
        # About 300k images
        frac = self.desired_images / len(dset["train"])
        subsampled_dset = dset["train"].train_test_split(test_size=frac, shuffle=True)["test"]
        dset = subsampled_dset.train_test_split(test_size=0.05, shuffle=True)
        print(dset)
        # return
        dset = dset.map(fetch_images, batched=False, fn_kwargs={"num_threads": self.num_threads})
        dset.save_to_disk(self.cache_path)
        self.hf_ds = dset

    def training_dataset(self) -> Dataset:
        return self.hf_ds["train"]
    
    def validation_dataset(self) -> Dataset:
        return self.hf_ds["test"]
