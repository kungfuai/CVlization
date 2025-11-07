from __future__ import annotations

import argparse
import json
from pathlib import Path

from cvlization.dataset.sleep_edf import SleepEDFBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and catalog Sleep-EDF records.")
    parser.add_argument(
        "--records",
        nargs="*",
        default=None,
        help="Specific EDF filenames to download (defaults to small curated list).",
    )
    parser.add_argument(
        "--subset",
        default="sleep-cassette",
        help="Sleep-EDF subset (sleep-cassette or sleep-telemetry).",
    )
    parser.add_argument(
        "--load-signals",
        action="store_true",
        help="Eagerly load signals to verify they can be read (requires mne).",
    )
    parser.add_argument(
        "--channels",
        nargs="*",
        default=None,
        help="Optional list of channel names to pick when loading signals.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builder = SleepEDFBuilder(
        subset=args.subset,
        records=args.records,
        load_signals=args.load_signals,
        channels=args.channels,
    )
    train_ds = builder.training_dataset()
    val_ds = builder.validation_dataset()

    manifest = {
        "training_records": [train_ds[i]["record_id"] for i in range(len(train_ds))],
        "validation_records": [val_ds[i]["record_id"] for i in range(len(val_ds))],
    }
    Path("outputs").mkdir(parents=True, exist_ok=True)
    (Path("outputs") / "sleep_edf_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Downloaded records:", manifest)


if __name__ == "__main__":
    main()
