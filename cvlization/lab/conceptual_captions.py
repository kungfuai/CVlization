"""
A huggingface interface for Google's Conceptual Captions dataset.
"""

from pathlib import Path
import json
from dataclasses import dataclass
import logging
import io
import urllib
import PIL.Image

import datasets
from datasets.utils.file_utils import get_datasets_user_agent

from ..data.dataset_builder import Dataset, DatasetProvider


LOGGER = logging.getLogger(__name__)


# https://huggingface.co/datasets/conceptual_captions
USER_AGENT = get_datasets_user_agent()

class Util:
    @staticmethod
    def fetch_image(image_url, timeout=None, retries=0):
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


@dataclass
class ConceptualCaptionsDatasetBuilder:
    num_proc: int = 32
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
        if not Path(self.cache_path).exists():
            dset = datasets.load_dataset("conceptual_captions")
            frac = self.desired_images / len(dset["train"])
            subsampled_dset = dset["train"].train_test_split(test_size=frac, shuffle=True)["test"]
            dset = subsampled_dset.train_test_split(test_size=0.05, shuffle=True)
            print("Creating dataset:", dset)
            # return
            dset = dset.map(self.fetch_image_and_format, num_proc=self.num_proc) \
                .filter(lambda x: x["image"] != None) \
                .cast_column("image", datasets.Image())
            print("Saving to disk")
            dset.save_to_disk(self.cache_path)
        else:
            dset = datasets.load_from_disk(self.cache_path)
        self.hf_ds = dset

    def training_dataset(self) -> Dataset:
        return self.hf_ds["train"]
    
    def validation_dataset(self) -> Dataset:
        return self.hf_ds["test"]

    def fetch_image_and_format(self, sample):
        pil_image = Util.fetch_image(sample["image_url"], timeout=5, retries=1)
        sample["image"] = pil_image.resize((self.image_width, self.image_height)) \
            if pil_image is not None else None
        sample["ground_truth"] = json.dumps({
            "gt_parse": {
                "caption": sample["caption"],
            }
        })
        return sample
