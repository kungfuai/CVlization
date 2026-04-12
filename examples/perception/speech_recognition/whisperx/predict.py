#!/usr/bin/env python3
"""WhisperX speech transcription, alignment, and optional diarization for CVLization."""

import argparse
import gc
import json
import logging
import os
import shutil
import sys
import urllib.request
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger("whisperx_cvl")

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
EXAMPLE_NAME = "whisperx"


def configure_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.ERROR
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s", force=True)
    LOGGER.setLevel(level)
    for logger_name in [
        "lightning",
        "pytorch_lightning",
        "pyannote",
        "torch",
        "torchaudio",
        "transformers",
        "whisperx",
    ]:
        logging.getLogger(logger_name).setLevel(level)


def configure_whisperx_logging(verbose: bool) -> None:
    """WhisperX installs its own logger on first import; reset it after import."""
    level_name = "info" if verbose else "error"
    try:
        from whisperx.log_utils import setup_logging

        setup_logging(level=level_name)
    except Exception:
        pass
    if not verbose:
        try:
            import lightning.pytorch.utilities.migration.utils as lightning_migration
            import lightning.pytorch.utilities.rank_zero as lightning_rank_zero
            import pytorch_lightning.utilities.migration.utils as pl_migration
            import pytorch_lightning.utilities.rank_zero as pl_rank_zero

            def _quiet_rank_zero_warn(*_args: Any, **_kwargs: Any) -> None:
                return None

            lightning_rank_zero.rank_zero_warn = _quiet_rank_zero_warn
            lightning_migration.rank_zero_warn = _quiet_rank_zero_warn
            lightning_migration._log.setLevel(logging.ERROR)
            lightning_migration._log.disabled = True
            pl_rank_zero.rank_zero_warn = _quiet_rank_zero_warn
            pl_migration.rank_zero_warn = _quiet_rank_zero_warn
            pl_migration._log.setLevel(logging.ERROR)
            pl_migration._log.disabled = True
        except Exception:
            pass
    configure_logging(verbose)


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


def get_cache_root() -> Path:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return hf_home / "cvl_data" / EXAMPLE_NAME


def ensure_sample_audio(cache_root: Optional[Path] = None) -> Path:
    """Download the default sample audio from the shared CVL dataset."""
    cache_root = cache_root or get_cache_root()
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


def download_url_audio(url: str, cache_root: Optional[Path] = None) -> Path:
    cache_root = cache_root or get_cache_root()
    cache_root.mkdir(parents=True, exist_ok=True)
    suffix = Path(url.split("?", 1)[0]).suffix or ".audio"
    target = cache_root / f"url_input{suffix}"
    print(f"Downloading audio URL to {target}...")
    with urllib.request.urlopen(url) as response, target.open("wb") as output:
        shutil.copyfileobj(response, output)
    return target


def resolve_audio_arg(audio: Optional[str]) -> str:
    if not audio or audio == "sample":
        return str(ensure_sample_audio())
    if audio.startswith(("http://", "https://")):
        return str(download_url_audio(audio))
    return resolve_input_path(audio)


def get_model_dir(args_model_dir: Optional[str]) -> str:
    model_dir = args_model_dir or os.environ.get("MODEL_BASE")
    if not model_dir:
        model_dir = str(Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "whisperx_models")
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    return model_dir


def normalize_language(language: Optional[str], model_name: str, detected: Optional[str]) -> str:
    if model_name.endswith(".en"):
        return "en"
    return language or detected or "en"


def transcribe_with_whisperx(args: argparse.Namespace) -> Dict[str, Any]:
    import torch
    import whisperx
    from whisperx.diarize import DiarizationPipeline

    configure_whisperx_logging(args.verbose)

    device = detect_device(args.device)
    compute_type = args.compute_type or default_compute_type(device)
    audio_path = resolve_audio_arg(args.audio)
    model_dir = get_model_dir(args.model_dir)
    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if args.diarize and not hf_token:
        raise RuntimeError(
            "--diarize requires --hf-token or HF_TOKEN/HUGGINGFACE_TOKEN for pyannote gated models."
        )

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    print(f"Audio: {audio_path}")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Compute type: {compute_type}")
    print(f"Alignment: {'off' if args.no_align or args.task == 'translate' else 'on'}")
    print(f"Diarization: {'on' if args.diarize else 'off'}")

    asr_options = {
        "beam_size": args.beam_size,
        "best_of": args.best_of,
        "patience": args.patience,
        "length_penalty": args.length_penalty,
        "temperatures": [args.temperature],
        "compression_ratio_threshold": args.compression_ratio_threshold,
        "log_prob_threshold": args.logprob_threshold,
        "no_speech_threshold": args.no_speech_threshold,
        "condition_on_previous_text": args.condition_on_previous_text,
        "initial_prompt": args.initial_prompt,
        "hotwords": args.hotwords,
        "suppress_tokens": [int(token) for token in args.suppress_tokens.split(",")],
        "suppress_numerals": args.suppress_numerals,
    }

    vad_options = {
        "chunk_size": args.chunk_size,
        "vad_onset": args.vad_onset,
        "vad_offset": args.vad_offset,
    }

    model = whisperx.load_model(
        args.model,
        device=device,
        device_index=args.device_index,
        compute_type=compute_type,
        language=args.language,
        asr_options=asr_options,
        vad_method=args.vad_method,
        vad_options=vad_options,
        task=args.task,
        download_root=model_dir,
        local_files_only=args.model_cache_only,
        threads=max(args.threads, 0) or 4,
        use_auth_token=hf_token,
    )

    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(
        audio,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        print_progress=args.verbose,
        verbose=args.verbose,
    )

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    language = normalize_language(args.language, args.model, result.get("language"))
    aligned = False

    if not args.no_align and args.task != "translate" and result.get("segments"):
        align_model, align_metadata = whisperx.load_align_model(
            language_code=language,
            device=device,
            model_name=args.align_model,
            model_dir=model_dir,
            model_cache_only=args.model_cache_only,
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            align_metadata,
            audio,
            device,
            interpolate_method=args.interpolate_method,
            return_char_alignments=args.return_char_alignments,
            print_progress=args.verbose,
        )
        language = align_metadata.get("language", language)
        aligned = True
        del align_model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    diarized = False
    if args.diarize:
        diarize_model = DiarizationPipeline(
            model_name=args.diarize_model,
            token=hf_token,
            device=device,
            cache_dir=model_dir,
        )
        diarize_segments = diarize_model(
            audio_path,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
        )
        result = whisperx.assign_word_speakers(diarize_segments, result)
        diarized = True

    result["language"] = language
    text = " ".join(segment.get("text", "").strip() for segment in result.get("segments", [])).strip()

    return {
        "text": text,
        "segments": result.get("segments", []),
        "word_segments": result.get("word_segments", []),
        "language": language,
        "model": args.model,
        "device": device,
        "compute_type": compute_type,
        "task": args.task,
        "audio": audio_path,
        "aligned": aligned,
        "diarized": diarized,
        "vad_method": args.vad_method,
        "model_dir": model_dir,
    }


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return value


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


def segment_words(segment: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    words = segment.get("words")
    return words if isinstance(words, list) else []


def render_text(result: Dict[str, Any]) -> str:
    lines = []
    for segment in result.get("segments", []):
        speaker = segment.get("speaker")
        text = segment.get("text", "").strip()
        if not text:
            continue
        if speaker:
            lines.append(f"[{speaker}]: {text}")
        else:
            lines.append(text)
    return "\n".join(lines).strip() + "\n"


def render_srt(result: Dict[str, Any]) -> str:
    blocks = []
    for idx, segment in enumerate(result.get("segments", []), start=1):
        start = segment.get("start")
        end = segment.get("end")
        words = list(segment_words(segment))
        if words:
            starts = [word["start"] for word in words if "start" in word]
            ends = [word["end"] for word in words if "end" in word]
            if starts and ends:
                start = min(starts)
                end = max(ends)
        text = segment.get("text", "").strip()
        if segment.get("speaker"):
            text = f"[{segment['speaker']}]: {text}"
        blocks.append(
            f"{idx}\n{format_timestamp(start, decimal_marker=',')} --> "
            f"{format_timestamp(end, decimal_marker=',')}\n{text}\n"
        )
    return "\n".join(blocks).strip() + "\n"


def render_vtt(result: Dict[str, Any]) -> str:
    blocks = ["WEBVTT\n"]
    for segment in result.get("segments", []):
        start = segment.get("start")
        end = segment.get("end")
        words = list(segment_words(segment))
        if words:
            starts = [word["start"] for word in words if "start" in word]
            ends = [word["end"] for word in words if "end" in word]
            if starts and ends:
                start = min(starts)
                end = max(ends)
        text = segment.get("text", "").strip()
        if segment.get("speaker"):
            text = f"[{segment['speaker']}]: {text}"
        blocks.append(f"{format_timestamp(start)} --> {format_timestamp(end)}\n{text}\n")
    return "\n".join(blocks).strip() + "\n"


def save_result(result: Dict[str, Any], output_path: str, output_format: str) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "json":
        payload = make_json_safe(dict(result))
        payload["created_at"] = datetime.now(timezone.utc).isoformat()
        output_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    elif output_format == "txt":
        output_file.write_text(render_text(result), encoding="utf-8")
    elif output_format == "srt":
        output_file.write_text(render_srt(result), encoding="utf-8")
    elif output_format == "vtt":
        output_file.write_text(render_vtt(result), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    print(f"Saved {output_format} output to: {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe speech audio with WhisperX alignment and optional diarization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--audio",
        default="sample",
        help="Audio file path, URL, or 'sample' to download the default CVL sample.",
    )
    parser.add_argument("--model", default="tiny.en", help="Whisper model size/name.")
    parser.add_argument("--model-dir", default=None, help="Directory for model downloads.")
    parser.add_argument("--model-cache-only", action="store_true", help="Only use cached models.")
    parser.add_argument("--language", default=None, help="Optional language code, e.g. en, de, ja.")
    parser.add_argument("--task", choices=["transcribe", "translate"], default="transcribe")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument(
        "--compute-type",
        default=None,
        help="CTranslate2 compute type. Defaults to float16 on CUDA and int8 on CPU.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--best-of", type=int, default=5)
    parser.add_argument("--patience", type=float, default=1.0)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--compression-ratio-threshold", type=float, default=2.4)
    parser.add_argument("--logprob-threshold", type=float, default=-1.0)
    parser.add_argument("--no-speech-threshold", type=float, default=0.6)
    parser.add_argument("--suppress-tokens", default="-1")
    parser.add_argument("--suppress-numerals", action="store_true")
    parser.add_argument("--initial-prompt", default=None)
    parser.add_argument("--hotwords", default=None)
    parser.add_argument(
        "--condition-on-previous-text",
        action="store_true",
        help="Condition each segment on prior recognized text.",
    )
    parser.add_argument("--vad-method", choices=["pyannote", "silero"], default="pyannote")
    parser.add_argument("--vad-onset", type=float, default=0.500)
    parser.add_argument("--vad-offset", type=float, default=0.363)
    parser.add_argument("--no-align", action="store_true", help="Skip forced alignment.")
    parser.add_argument("--align-model", default=None, help="Optional phoneme-level ASR model.")
    parser.add_argument(
        "--interpolate-method",
        choices=["nearest", "linear", "ignore"],
        default="nearest",
    )
    parser.add_argument("--return-char-alignments", action="store_true")
    parser.add_argument("--diarize", action="store_true", help="Enable speaker diarization.")
    parser.add_argument("--diarize-model", default="pyannote/speaker-diarization-community-1")
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--hf-token", default=None, help="HF token for diarization gated models.")
    parser.add_argument("--output", default="whisperx_transcript.json")
    parser.add_argument("--format", choices=["json", "txt", "srt", "vtt"], default="json")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    output_path = resolve_output_path(args.output)
    result = transcribe_with_whisperx(args)

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
