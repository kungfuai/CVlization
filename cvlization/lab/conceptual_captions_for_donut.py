"""
A huggingface interface for Google's Conceptual Captions dataset.
"""

from pathlib import Path
import os
from dataclasses import dataclass
import logging
import io
import json
import urllib
import PIL.Image

from torch.utils.data import IterableDataset
import datasets
from datasets.utils.file_utils import get_datasets_user_agent

from .postgres_image_iterable import PostgresImageIterable
from ..data.dataset_builder import Dataset, DatasetProvider

LOGGER = logging.getLogger(__name__)


# https://huggingface.co/datasets/conceptual_captions
USER_AGENT = get_datasets_user_agent()


@dataclass
class ConceptualCaptionsForDonutDatasetBuilder(IterableDataset):
    # Networking seems to be the bottleneck, not CPU. Still want avoid timeouts though.
    # Image size effects CPU requirements b/c of conversion, saving & loading.
    num_proc: int = os.cpu_count()
    desired_images: int = 200000
    batch_size: int = 1
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

    @property
    def postgres_connection_info(self):
        # FIXME: Don't hardcode
        return {
            "host": "172.17.0.1", # Docker postgres IP
            "database": "datasets",
            "user": "postgres",
            "password": "password",
        }

    def load(self):
        self.dataset = PostgresImageIterable(
            db_conn_info=self.postgres_connection_info,
            dataset_name="conceptual_captions",
            batch_size=self.batch_size,
        )

        if not self.dataset.check_exists():
            """
            Google's Conceptual Captions dataset has 3 million images.
            We're gonna choose a random subset from their "train" set.
            """
            # Randomly downsample dataset["train"]
            dset = datasets.load_dataset("conceptual_captions")
            frac = self.desired_images / len(dset["train"])
            shuffled_images = dset["train"].shuffle(seed=0)
            frac_images = shuffled_images.train_test_split(test_size=frac, shuffle=False)["test"]
            dset = frac_images.train_test_split(test_size=0.05, shuffle=False)
            # Save `dset` to Postgres
            ## Use huggingface's nice multi-core map function
            self.dataset.create()
            self.dataset.close_connection() # Prevent pickling error w/ db conn object
            dset.map(self._fetch_image_and_save, num_proc=self.num_proc, with_indices=True)

    def training_dataset(self) -> IterableDataset:
        """
        PSQL iterable -> will never leave this dataset,
        which streams existing & newly added samples.
        """
        return self

    def validation_dataset(self) -> IterableDataset:
        """
        PSQL iterable -> will never leave this dataset,
        which streams existing & newly added samples.
        """
        return self
        return datasets.load_from_disk("/datasets/conceptual_captions_100k")["test"]

    def is_iterable(self):
        return True

    def __iter__(self):
        self._iter = iter(self.dataset)
        return self

    def __next__(self):
        """
        Return the next batch in the IterableDataset.

        Need to transform labels to Donut-expected format,
        to be ingested by `cvlization.torch.training_pipeline.doc_ai.huggingface.donut.pipeline.ProcessedDataset,
        which will tokenize the PIL Image.
        """
        batch = next(self._iter)
        samples = [
            {
                "image": img,
                "ground_truth": json.dumps({
                    "gt_parse": {
                        "caption": meta["caption"],
                    },
                }) if "caption" in meta else None,
                "update_fxn": update_fxn,
            }
            for img, meta, update_fxn in batch
        ]
        return samples

    def _fetch_image_and_save(self, sample, idx):
        img_url = sample["image_url"]
        img_metadata = {
            "caption": sample["caption"],
        }

        timeout = 5
        retries = 0

        for _ in range(retries + 1):
            try:
                request = urllib.request.Request(
                    img_url,
                    data=None,
                    headers={"user-agent": USER_AGENT},
                )
                with urllib.request.urlopen(request, timeout=timeout) as req:
                    image = PIL.Image.open(io.BytesIO(req.read()))
                    image = image.convert("RGB")
                    self.dataset.add_sample(pil_image=image, metadata_dict=img_metadata, has_label=True)
                    image.close()
                break
            except Exception as ex:
                # LOGGER.warn(f"Failed image fetch attempt: {str(ex)}")
                pass

        return sample
