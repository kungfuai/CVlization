from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import requests
from tqdm import tqdm

from cvl.core.downloads import get_cache_dir
from ..data.dataset_builder import Dataset, DatasetProvider, MapStyleDataset


SLEEP_EDF_BASE_URL = "https://physionet.org/files/sleep-edfx/1.0.0"
DEFAULT_RECORDS = [
    "SC4001E0-PSG.edf",
    "SC4002E0-PSG.edf",
    "SC4011E0-PSG.edf",
]


def _default_data_dir() -> Path:
    cache_dir = get_cache_dir() / "data" / "sleep-edf"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@dataclass
class SleepEDFRecord:
    record_id: str
    psg_path: Path
    hypnogram_path: Optional[Path]


class SleepEDFRecordDataset(MapStyleDataset):
    def __init__(
        self,
        records: Iterable[SleepEDFRecord],
        load_signals: bool = False,
        channels: Optional[List[str]] = None,
    ):
        self.records = list(records)
        self.load_signals = load_signals
        self.channels = channels

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        example = {
            "record_id": record.record_id,
            "psg_path": str(record.psg_path),
            "hypnogram_path": str(record.hypnogram_path) if record.hypnogram_path else None,
        }
        if not self.load_signals:
            return example

        try:
            import mne  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Reading EDF signals requires `mne`. Install via `pip install mne`."
            ) from exc

        raw = mne.io.read_raw_edf(record.psg_path, preload=True, verbose=False)
        if self.channels:
            resolved: List[str] = []
            for ch in self.channels:
                if ch in raw.ch_names:
                    resolved.append(ch)
                else:
                    eeg_alias = f"EEG {ch}" if not ch.startswith("EEG ") else ch
                    if eeg_alias in raw.ch_names:
                        resolved.append(eeg_alias)
                    else:
                        raise ValueError(
                            f"Channel '{ch}' not found in record {record.record_id}; available: {raw.ch_names}"
                        )
            raw.pick(resolved)
        signals = raw.get_data()
        example["signals"] = signals.astype(np.float32)
        example["sampling_rate"] = float(raw.info["sfreq"])
        example["channel_names"] = list(raw.ch_names)

        if record.hypnogram_path and record.hypnogram_path.exists():
            annotations = mne.read_annotations(record.hypnogram_path)
            example["annotations"] = annotations.to_data_frame()
        return example


@dataclass
class SleepEDFBuilder:
    """Lightweight dataset builder for the Sleep-EDF Expanded collection (PhysioNet).

    Downloads a user-specified subset of EDF recordings plus optional hypnogram annotations.
    Data is cached under ``~/.cache/cvlization/data/sleep-edf`` by default.
    """

    subset: str = "sleep-cassette"
    records: Optional[List[str]] = None
    data_dir: Path = field(default_factory=_default_data_dir)
    download_annotations: bool = True
    load_signals: bool = False
    channels: Optional[List[str]] = None
    validation_fraction: float = 0.2

    _downloaded_records: Optional[List[SleepEDFRecord]] = field(
        init=False, default=None, repr=False
    )

    @property
    def dataset_provider(self):
        return DatasetProvider.CVLIZATION

    def _annotation_name(self, record_name: str) -> Optional[str]:
        if not self.download_annotations:
            return None
        if record_name.endswith("E0-PSG.edf"):
            return record_name.replace("E0-PSG.edf", "EC-Hypnogram.edf")
        if record_name.endswith("F0-PSG.edf"):
            return record_name.replace("F0-PSG.edf", "FC-Hypnogram.edf")
        return None

    def _download_file(self, url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            return
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            with dest.open("wb") as fh, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"Downloading {dest.name}",
            ) as bar:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        fh.write(chunk)
                        bar.update(len(chunk))

    def _ensure_downloaded(self) -> List[SleepEDFRecord]:
        if self._downloaded_records is not None:
            return self._downloaded_records

        subset_dir = self.data_dir / self.subset
        subset_dir.mkdir(parents=True, exist_ok=True)

        target_records = self.records or DEFAULT_RECORDS
        downloaded: List[SleepEDFRecord] = []
        for record_name in target_records:
            record_url = f"{SLEEP_EDF_BASE_URL}/{self.subset}/{record_name}"
            record_path = subset_dir / record_name
            self._download_file(record_url, record_path)

            annotation_name = self._annotation_name(record_name)
            annotation_path = None
            if annotation_name is not None:
                annotation_url = f"{SLEEP_EDF_BASE_URL}/{self.subset}/{annotation_name}"
                annotation_path = subset_dir / annotation_name
                self._download_file(annotation_url, annotation_path)

            downloaded.append(
                SleepEDFRecord(
                    record_id=record_name.replace("-PSG.edf", ""),
                    psg_path=record_path,
                    hypnogram_path=annotation_path,
                )
            )

        self._downloaded_records = downloaded
        return downloaded

    def _split_records(self) -> tuple[List[SleepEDFRecord], List[SleepEDFRecord]]:
        records = self._ensure_downloaded()
        if not records:
            return [], []
        split_idx = max(1, int(len(records) * self.validation_fraction))
        val = records[:split_idx]
        train = records[split_idx:] or records[split_idx - 1 :]
        return train, val

    def _build_dataset(self, records: List[SleepEDFRecord]) -> Dataset:
        return SleepEDFRecordDataset(
            records,
            load_signals=self.load_signals,
            channels=self.channels,
        )

    def training_dataset(self) -> Dataset:
        train_records, _ = self._split_records()
        return self._build_dataset(train_records)

    def validation_dataset(self) -> Dataset:
        _, val_records = self._split_records()
        return self._build_dataset(val_records)
