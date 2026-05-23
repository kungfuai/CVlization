#!/usr/bin/env python3
"""NVIDIA Parakeet-TDT ASR for CVLization, via NeMo.

Parakeet-TDT pairs a FastConformer encoder with a Token-and-Duration Transducer
decoder. The TDT decoder skips most blank predictions, giving the famous
>2000x RTFx on the Hugging Face Open-ASR leaderboard. That decoder lives in
NVIDIA NeMo; using the Hugging Face transformers adapter loses most of it,
so this preset defaults to NeMo.
"""

from __future__ import annotations

import argparse
import contextlib
import io
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
LOGGER = logging.getLogger("parakeet_tdt_cvl")

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
EXAMPLE_NAME = "parakeet_tdt"


def configure_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.ERROR
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s", force=True)
    LOGGER.setLevel(level)


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
    """Parakeet expects 16 kHz mono. If the input differs, resample to a
    sibling temp file. Returns the path to use for inference."""
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
    # NeMo prints ~hundreds of lines of import/config noise. Silence by default.
    quiet_ctx = contextlib.redirect_stdout(io.StringIO()) if not args.verbose else contextlib.nullcontext()
    with quiet_ctx:
        import nemo.collections.asr as nemo_asr  # noqa: WPS433
        import torch  # noqa: WPS433

    device = detect_device(args.device)
    audio_path = resolve_audio_arg(args.audio)
    if not audio_path.startswith(("http://", "https://")) and not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    audio_path = maybe_resample_to_16k(audio_path)

    print(f"Audio: {audio_path}")
    print(f"Model: {args.model}")
    print(f"Device: {device}")

    with quiet_ctx:
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=args.model)
        if device == "cuda":
            asr_model = asr_model.to("cuda")
        if hasattr(asr_model, "eval"):
            asr_model.eval()

    # NeMo's .transcribe() can return strings (older) or Hypothesis objects
    # (newer, with timestamps). We accept either.
    kwargs: Dict[str, Any] = {"batch_size": args.batch_size}
    if args.word_timestamps:
        kwargs["timestamps"] = True

    with torch.inference_mode() if hasattr(torch, "inference_mode") else contextlib.nullcontext():
        with quiet_ctx:
            transcriptions = asr_model.transcribe([audio_path], **kwargs)

    text, segments = _normalize_transcription(transcriptions[0])

    return {
        "text": text,
        "segments": segments,
        "model": args.model,
        "device": device,
        "audio": audio_path,
        "task": "transcribe",
    }


def _normalize_transcription(item: Any) -> tuple[str, List[Dict[str, Any]]]:
    if isinstance(item, str):
        return item.strip(), []
    text = getattr(item, "text", None) or str(item)
    segments: List[Dict[str, Any]] = []
    ts = getattr(item, "timestamp", None) or {}
    for kind in ("segment", "word"):
        for s in ts.get(kind, []) or []:
            segments.append({
                "kind": kind,
                "start": s.get("start"),
                "end": s.get("end"),
                "text": s.get("segment") or s.get("word") or "",
            })
    return text.strip(), segments


def format_timestamp(seconds: Optional[float], decimal_marker: str = ".") -> str:
    seconds = max(float(seconds or 0), 0.0)
    ms = round(seconds * 1000.0)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{decimal_marker}{ms:03d}"


def render_srt(segments: List[Dict[str, Any]]) -> str:
    blocks = []
    seg_only = [s for s in segments if s.get("kind", "segment") == "segment"] or segments
    for idx, seg in enumerate(seg_only, start=1):
        start = format_timestamp(seg["start"], decimal_marker=",")
        end = format_timestamp(seg["end"], decimal_marker=",")
        blocks.append(f"{idx}\n{start} --> {end}\n{(seg.get('text') or '').strip()}\n")
    return "\n".join(blocks).strip() + "\n"


def save_result(result: Dict[str, Any], output_path: str, output_format: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        payload = dict(result)
        payload["created_at"] = datetime.now(timezone.utc).isoformat()
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    elif output_format == "txt":
        out.write_text(result["text"] + "\n", encoding="utf-8")
    elif output_format == "srt":
        if not result["segments"]:
            raise ValueError("--format srt requires --word-timestamps and segment output from the model.")
        out.write_text(render_srt(result["segments"]), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    print(f"Saved {output_format} output to: {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Transcribe speech with NVIDIA NeMo Parakeet-TDT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--audio", default="sample",
                   help="Audio file path, URL, or 'sample' to download the default CVL sample.")
    p.add_argument("--model", default="nvidia/parakeet-tdt-1.1b",
                   help="NeMo ASR model name (e.g. nvidia/parakeet-tdt-1.1b, nvidia/parakeet-tdt-0.6b-v2).")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--word-timestamps", action="store_true",
                   help="Request segment / word timestamps (NeMo TDT decoder supports both).")
    p.add_argument("--output", default="parakeet_tdt_transcript.json")
    p.add_argument("--format", choices=["json", "txt", "srt"], default="json")
    p.add_argument("--verbose", action="store_true", help="Don't suppress NeMo's startup logs.")
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
