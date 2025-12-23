#!/usr/bin/env python
import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import requests
from loguru import logger

from cvlization.paths import resolve_input_path, resolve_output_path

REPO_DIR = Path(__file__).resolve().parent
sys.path.append(str(REPO_DIR))
from lite_avatar import liteAvatar  # noqa: E402


WEIGHTS_EXPECTED = [
    Path("weights/model_1.onnx"),
    Path("weights/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/model.pb"),
    Path("weights/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/lm/lm.pb"),
]
SAMPLE_DATA_URL = os.environ.get(
    "LITE_AVATAR_SAMPLE_URL",
    "https://github.com/HumanAIGC/lite-avatar/raw/refs/heads/main/data/sample_data.zip",
)


def ensure_models(repo_root: Path) -> None:
    """Ensure the LiteAvatar weights are present, otherwise download them."""
    missing = [path for path in WEIGHTS_EXPECTED if not (repo_root / path).exists()]
    if not missing:
        return

    logger.info("Missing weight files detected: {}", [str(p) for p in missing])
    script_path = repo_root / "download_model.sh"
    if not script_path.exists():
        raise FileNotFoundError(f"download_model.sh not found at {script_path}")

    subprocess.run(["bash", str(script_path)], cwd=repo_root, check=True)


def ensure_sample_archive(sample_zip: Path) -> None:
    """Download the sample avatar archive if missing."""
    if sample_zip.exists():
        return
    sample_zip.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading sample avatar assets from {}", SAMPLE_DATA_URL)
    with requests.get(SAMPLE_DATA_URL, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(sample_zip, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)


def ensure_sample_data(data_dir: Path, sample_zip: Path) -> None:
    """Unpack the bundled sample data if the target directory is absent."""
    if data_dir.exists():
        return
    if not sample_zip.exists():
        raise FileNotFoundError(
            f"Sample data directory {data_dir} not found and archive {sample_zip} missing."
        )

    logger.info("Extracting sample data from {}", sample_zip)
    with zipfile.ZipFile(sample_zip, "r") as archive:
        archive.extractall(sample_zip.parent)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LiteAvatar inference.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to avatar asset directory (defaults to bundled sample data).",
    )
    parser.add_argument(
        "--audio-file",
        type=Path,
        default=None,
        help="Path to WAV audio file (defaults to bundled sample).",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where the generated MP4 will be written.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Enable GPU execution (CPU by default).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Output video frames-per-second.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = REPO_DIR

    ensure_models(repo_root)

    sample_zip = repo_root / "data" / "sample_data.zip"
    default_data_dir = repo_root / "data" / "preload"
    ensure_sample_archive(sample_zip)
    ensure_sample_data(default_data_dir, sample_zip)

    data_dir = Path(resolve_input_path(args.data_dir)) if args.data_dir else default_data_dir

    default_audio = (
        repo_root
        / "weights"
        / "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
        / "example"
        / "asr_example.wav"
    )
    audio_file = Path(resolve_input_path(args.audio_file)) if args.audio_file else default_audio

    if not data_dir.exists():
        raise FileNotFoundError(f"Avatar data directory not found: {data_dir}")
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    args.result_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running LiteAvatar inference")
    logger.info("Data directory: {}", data_dir)
    logger.info("Audio input: {}", audio_file)
    logger.info("Output directory: {}", args.result_dir)

    avatar = liteAvatar(
        data_dir=str(data_dir),
        num_threads=1,
        generate_offline=True,
        use_gpu=args.use_gpu,
        fps=args.fps,
    )
    avatar.handle(str(audio_file), str(args.result_dir))

    logger.info("Finished! Rendered video: {}", args.result_dir / "test_demo.mp4")


if __name__ == "__main__":
    main()
