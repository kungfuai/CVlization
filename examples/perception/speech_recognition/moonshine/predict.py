#!/usr/bin/env python3
"""Useful Sensors Moonshine ASR for CVLization.

Moonshine targets fast on-device speech-to-text: small models, low latency,
CPU-friendly. Two model sizes are publicly available — `moonshine/tiny` and
`moonshine/base` — both load via the `useful-moonshine` package.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("KERAS_BACKEND", "torch")
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger("moonshine_cvl")

try:
    from cvlization.paths import resolve_input_path, resolve_output_path
except ImportError:

    def resolve_input_path(path: str, input_dir: Optional[Path] = None) -> str:
        if path.startswith(("http://", "https://")) or path.startswith("/"):
            return path
        return str(Path(path).expanduser())

    def resolve_output_path(
        path: Optional[str] = None,
        output_dir: Optional[Path] = None,
        default_filename: str = "result.txt",
    ) -> str:
        output_root = output_dir or Path("outputs")
        output_root.mkdir(parents=True, exist_ok=True)
        path = path or default_filename
        return path if path.startswith("/") else str((output_root / path).resolve())


HF_DATA_REPO = "zzsi/cvl"
HF_SAMPLE_FILE = "livetalk/example.wav"
EXAMPLE_NAME = "moonshine"


def configure_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.ERROR
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s", force=True)
    LOGGER.setLevel(level)


def ensure_sample_audio(cache_root: Optional[Path] = None) -> Path:
    if cache_root is None:
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        cache_root = hf_home / "cvl_data" / EXAMPLE_NAME
    sample_path = cache_root / "example.wav"
    if sample_path.exists():
        return sample_path
    cache_root.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download
    print(f"Downloading sample audio from {HF_DATA_REPO}/{HF_SAMPLE_FILE}...")
    downloaded = hf_hub_download(
        repo_id=HF_DATA_REPO, filename=HF_SAMPLE_FILE, repo_type="dataset",
    )
    shutil.copy2(downloaded, sample_path)
    return sample_path


def resolve_audio_arg(audio: Optional[str]) -> str:
    if not audio or audio == "sample":
        return str(ensure_sample_audio())
    return resolve_input_path(audio)


def maybe_resample_to_16k(audio_path: str) -> str:
    """Moonshine expects 16 kHz mono PCM. Resample if needed."""
    import soundfile as sf
    info = sf.info(audio_path)
    if info.samplerate == 16000 and info.channels == 1:
        return audio_path
    import librosa
    print(f"Resampling {info.samplerate} Hz / {info.channels}ch -> 16000 Hz / mono")
    y, _ = librosa.load(audio_path, sr=16000, mono=True)
    out_path = str(Path(audio_path).with_suffix(".16k.wav"))
    sf.write(out_path, y, 16000)
    return out_path


def transcribe_audio(args: argparse.Namespace) -> Dict[str, Any]:
    import moonshine  # noqa: WPS433 - lazy so help text still works without deps

    audio_path = resolve_audio_arg(args.audio)
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    audio_path = maybe_resample_to_16k(audio_path)

    print(f"Audio: {audio_path}")
    print(f"Model: {args.model}")
    print(f"Keras backend: {os.environ.get('KERAS_BACKEND', 'default')}")

    raw = moonshine.transcribe(audio_path, args.model)
    # `moonshine.transcribe` returns a list of strings (one per utterance) in
    # recent versions; older versions returned a single string. Normalize.
    if isinstance(raw, (list, tuple)):
        text = " ".join(str(x).strip() for x in raw).strip()
    else:
        text = str(raw).strip()

    return {
        "text": text,
        "model": args.model,
        "audio": audio_path,
        "task": "transcribe",
    }


def save_result(result: Dict[str, Any], output_path: str, output_format: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        payload = dict(result)
        payload["created_at"] = datetime.now(timezone.utc).isoformat()
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    elif output_format == "txt":
        out.write_text(result["text"] + "\n", encoding="utf-8")
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    print(f"Saved {output_format} output to: {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fast on-device speech transcription with Useful Sensors Moonshine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--audio", default="sample",
                   help="Audio file path or 'sample' to download the default CVL sample.")
    p.add_argument("--model", default="moonshine/base", choices=["moonshine/tiny", "moonshine/base"],
                   help="moonshine/tiny (~27M params, smallest) or moonshine/base (~61M params).")
    p.add_argument("--output", default="moonshine_transcript.json")
    p.add_argument("--format", choices=["json", "txt"], default="json")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    output_path = resolve_output_path(args.output)
    result = transcribe_audio(args)

    print()
    print("Transcript:")
    print(result["text"] or "(empty)")
    print()

    save_result(result, output_path, args.format)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
