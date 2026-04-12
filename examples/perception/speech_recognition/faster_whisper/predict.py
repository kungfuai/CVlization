#!/usr/bin/env python3
"""Faster-Whisper speech recognition for CVLization."""

import argparse
import json
import logging
import os
import shutil
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger("faster_whisper_cvl")

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
EXAMPLE_NAME = "faster_whisper"


def configure_logging(verbose: bool) -> None:
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", force=True)
        LOGGER.setLevel(logging.INFO)
    else:
        logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s", force=True)
        LOGGER.setLevel(logging.ERROR)


def detect_device(requested: str) -> str:
    if requested != "auto":
        return requested

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass

    return "cpu"


def default_compute_type(device: str) -> str:
    return "float16" if device == "cuda" else "int8"


def ensure_sample_audio(cache_root: Optional[Path] = None) -> Path:
    """Download the default sample audio from the shared CVL dataset."""
    if cache_root is None:
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        cache_root = hf_home / "cvl_data" / EXAMPLE_NAME

    sample_path = cache_root / "example.wav"
    if sample_path.exists():
        return sample_path

    cache_root.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download the default sample. "
            "Install requirements or pass --audio /path/to/audio.wav."
        ) from exc

    print(f"Downloading sample audio from {HF_DATA_REPO}/{HF_SAMPLE_FILE}...")
    downloaded = hf_hub_download(
        repo_id=HF_DATA_REPO,
        filename=HF_SAMPLE_FILE,
        repo_type="dataset",
    )
    shutil.copy2(downloaded, sample_path)
    return sample_path


def resolve_audio_arg(audio: Optional[str]) -> str:
    if not audio or audio == "sample":
        return str(ensure_sample_audio())
    return resolve_input_path(audio)


def transcribe_audio(args: argparse.Namespace) -> Dict[str, Any]:
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    device = detect_device(args.device)
    compute_type = args.compute_type or default_compute_type(device)
    audio_path = resolve_audio_arg(args.audio)

    if not audio_path.startswith(("http://", "https://")) and not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"Audio: {audio_path}")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Compute type: {compute_type}")

    model = WhisperModel(
        args.model,
        device=device,
        compute_type=compute_type,
        cpu_threads=args.cpu_threads,
        num_workers=args.num_workers,
    )

    transcriber: Any = model
    transcribe_kwargs: Dict[str, Any] = {
        "beam_size": args.beam_size,
        "language": args.language,
        "task": args.task,
        "vad_filter": args.vad,
        "word_timestamps": args.word_timestamps,
        "condition_on_previous_text": args.condition_on_previous_text,
    }

    if args.vad:
        transcribe_kwargs["vad_parameters"] = {
            "min_silence_duration_ms": args.min_silence_duration_ms
        }

    if args.batch_size > 1:
        transcriber = BatchedInferencePipeline(model=model)
        transcribe_kwargs["batch_size"] = args.batch_size

    segments_iter, info = transcriber.transcribe(audio_path, **transcribe_kwargs)
    segments = [segment_to_dict(segment) for segment in segments_iter]

    text = "".join(segment["text"] for segment in segments).strip()
    return {
        "text": text,
        "segments": segments,
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": getattr(info, "duration", None),
        "model": args.model,
        "device": device,
        "compute_type": compute_type,
        "task": args.task,
        "audio": audio_path,
    }


def segment_to_dict(segment: Any) -> Dict[str, Any]:
    data = {
        "id": segment.id,
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
    }
    words = getattr(segment, "words", None)
    if words:
        data["words"] = [
            {
                "start": word.start,
                "end": word.end,
                "word": word.word,
                "probability": word.probability,
            }
            for word in words
        ]
    return data


def format_timestamp(seconds: Optional[float], decimal_marker: str = ".") -> str:
    seconds = max(float(seconds or 0), 0.0)
    milliseconds = round(seconds * 1000.0)
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    secs = milliseconds // 1000
    millis = milliseconds % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal_marker}{millis:03d}"


def render_srt(segments: List[Dict[str, Any]]) -> str:
    blocks = []
    for idx, segment in enumerate(segments, start=1):
        start = format_timestamp(segment["start"], decimal_marker=",")
        end = format_timestamp(segment["end"], decimal_marker=",")
        blocks.append(f"{idx}\n{start} --> {end}\n{segment['text'].strip()}\n")
    return "\n".join(blocks).strip() + "\n"


def render_vtt(segments: List[Dict[str, Any]]) -> str:
    blocks = ["WEBVTT\n"]
    for segment in segments:
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        blocks.append(f"{start} --> {end}\n{segment['text'].strip()}\n")
    return "\n".join(blocks).strip() + "\n"


def save_result(result: Dict[str, Any], output_path: str, output_format: str) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "json":
        payload = dict(result)
        payload["created_at"] = datetime.now(timezone.utc).isoformat()
        output_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    elif output_format == "txt":
        output_file.write_text(result["text"] + "\n", encoding="utf-8")
    elif output_format == "srt":
        output_file.write_text(render_srt(result["segments"]), encoding="utf-8")
    elif output_format == "vtt":
        output_file.write_text(render_vtt(result["segments"]), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    print(f"Saved {output_format} output to: {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe or translate speech audio with faster-whisper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--audio",
        default="sample",
        help="Audio file path, URL, or 'sample' to download the default CVL sample.",
    )
    parser.add_argument(
        "--model",
        default="tiny.en",
        help="Faster-Whisper model size/name or local CTranslate2 model path.",
    )
    parser.add_argument("--language", default=None, help="Optional language code, e.g. en, de, ja.")
    parser.add_argument("--task", choices=["transcribe", "translate"], default="transcribe")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--compute-type",
        default=None,
        help="CTranslate2 compute type. Defaults to float16 on CUDA and int8 on CPU.",
    )
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--vad", action="store_true", help="Enable Silero VAD filtering.")
    parser.add_argument(
        "--min-silence-duration-ms",
        type=int,
        default=500,
        help="Minimum silence duration used when --vad is enabled.",
    )
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Include word-level timestamps in JSON output.",
    )
    parser.add_argument(
        "--no-condition-on-previous-text",
        dest="condition_on_previous_text",
        action="store_false",
        help="Disable conditioning on previous text between windows.",
    )
    parser.set_defaults(condition_on_previous_text=True)
    parser.add_argument("--output", default="faster_whisper_transcript.json")
    parser.add_argument("--format", choices=["json", "txt", "srt", "vtt"], default="json")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser.parse_args()


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
