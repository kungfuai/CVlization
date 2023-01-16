"""
A huggingface interface for Google's Conceptual Captions dataset.
"""

from pathlib import Path
import json
import os
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
    def download_image(image_url, image_path, resize=None, timeout=None, retries=0):
        for _ in range(retries + 1):
            try:
                request = urllib.request.Request(
                    image_url,
                    data=None,
                    headers={"user-agent": USER_AGENT},
                )
                with urllib.request.urlopen(request, timeout=timeout) as req:
                    image = PIL.Image.open(io.BytesIO(req.read()))
                    if resize is not None:
                        image = image.resize(resize)
                    image = image.convert("RGB")
                    image.save(image_path, "png", optimize=True)
                    image.close()
                break
            except Exception:
                return False
        return True


@dataclass
class ConceptualCaptionsDatasetBuilder:
    # Networking seems to be the bottleneck, not CPU. Still want avoid timeouts though.
    # Image size effects CPU requirements b/c of conversion, saving & loading.
    num_proc: int = os.cpu_count() * 2
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
            shuffled_images = dset["train"].shuffle(seed=0)
            frac_images = shuffled_images.train_test_split(test_size=frac, shuffle=False)["test"]
            dset = frac_images.train_test_split(test_size=0.05, shuffle=False)
            print("Creating dataset:", dset)
            dset = dset.map(self.fetch_image_and_format, num_proc=self.num_proc, with_indices=True) \
                .filter(lambda x: x["image"] != None) \
                .cast_column("image", datasets.Image(decode=True))
            """
            My understanding of saving img file to disk (setting `["image"] = <path>`) then using `Image(decode=True)`
            as opposed to downloading the file to `["image"] = PIL.Image()`
            is, based on observation, that writing downloaded PIL images DIRECTLY to the dataset object
            keeps every image in RAM, crashing the program, whereas `Image(decode=True)` somehow
            plays nicely with RAM usage.
            This could be wrong, in which case writing all the images to disk is a waste of time and disk.
            """
            dset.save_to_disk(self.cache_path, num_proc=self.num_proc)
        else:
            dset = datasets.load_from_disk(self.cache_path)
        self.hf_ds = dset

    def training_dataset(self) -> Dataset:
        return self.hf_ds["train"]
    
    def validation_dataset(self) -> Dataset:
        return self.hf_ds["test"]

    def fetch_image_and_format(self, sample, idx):
        img_dir = Path(self.cache_path) / "img"
        img_dir.mkdir(exist_ok=True, parents=True)
        img_path = img_dir / f"{idx}.png"

        is_success = img_path.exists() \
            or Util.download_image(
                sample["image_url"],
                str(img_path),
                resize=(self.image_width, self.image_height),
                timeout=5, retries=1)

        sample["image"] = str(img_path) if is_success else None

        # Format label for donut
        sample["ground_truth"] = json.dumps({
            "gt_parse": {
                "caption": sample["caption"],
            }
        })

        return sample
