#!/usr/bin/env python3
"""MOSS-Transcribe-Diarize: joint transcription + speaker diarization for CVLization."""

import argparse
import json
import os
import re
import shutil
import sys
import urllib.request
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", category=UserWarning)

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
HF_SAMPLE_FILE = "moss_transcribe_diarize/multi_speaker_sample.wav"
EXAMPLE_NAME = "moss_transcribe_diarize"

DEFAULT_PROMPT = (
    "Transcribe the audio. For each segment, start with the timestamp "
    "and speaker ID ([S01], [S02], [S03], ...), then the spoken text, "
    "and end with the segment timestamp."
)


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    speaker: str
    text: str


def parse_transcript(raw_text: str) -> List[TranscriptSegment]:
    """Parse MOSS output format: [start][Sxx]text[end] into segments."""
    segments = []
    pattern = re.compile(
        r"\[(\d+(?:\.\d+)?)\]"   # start timestamp
        r"\[S(\d+)\]"            # speaker ID
        r"(.*?)"                 # text (non-greedy)
        r"\[(\d+(?:\.\d+)?)\]"   # end timestamp
    )
    for match in pattern.finditer(raw_text):
        start = float(match.group(1))
        speaker = f"S{match.group(2)}"
        text = match.group(3).strip()
        end = float(match.group(4))
        if text:
            segments.append(TranscriptSegment(start=start, end=end, speaker=speaker, text=text))
    return segments


def get_cache_root() -> Path:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return hf_home / "cvl_data" / EXAMPLE_NAME


def ensure_sample_audio(cache_root: Optional[Path] = None) -> Path:
    """Download the default multi-speaker sample from the shared CVL dataset."""
    cache_root = cache_root or get_cache_root()
    sample_path = cache_root / "multi_speaker_sample.wav"
    if sample_path.exists():
        return sample_path

    cache_root.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download the default sample. "
            "Install it or pass --audio /path/to/audio.wav."
        ) from exc

    print(f"Downloading sample audio from {HF_DATA_REPO}/{HF_SAMPLE_FILE}...", flush=True)
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
    suffix = Path(url.split("?", 1)[0]).suffix or ".wav"
    target = cache_root / f"url_input{suffix}"
    print(f"Downloading audio from URL to {target}...", flush=True)
    with urllib.request.urlopen(url) as response, target.open("wb") as output:
        shutil.copyfileobj(response, output)
    return target


def resolve_audio_arg(audio: Optional[str]) -> str:
    if not audio or audio == "sample":
        return str(ensure_sample_audio())
    if audio.startswith(("http://", "https://")):
        return str(download_url_audio(audio))
    return resolve_input_path(audio)


def detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def transcribe(args: argparse.Namespace) -> Dict[str, Any]:
    """Run MOSS-Transcribe-Diarize inference."""
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    device_str = args.device if args.device != "auto" else detect_device()
    device = torch.device(device_str)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    audio_path = resolve_audio_arg(args.audio)
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"Audio: {audio_path}", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Device: {device_str}", flush=True)
    print(f"Dtype: {dtype}", flush=True)
    print(f"Max new tokens: {args.max_new_tokens}", flush=True)

    print("Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device).eval()

    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    print("Model loaded.", flush=True)

    # Build messages in the expected multimodal chat format
    prompt = args.prompt or DEFAULT_PROMPT
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Use the model's inference utilities if available, otherwise do it manually
    try:
        from moss_transcribe_diarize.inference_utils import (
            generate_transcription,
        )
        print("Using moss_transcribe_diarize inference utilities.", flush=True)
        result = generate_transcription(
            model,
            processor,
            messages,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            device=device,
            dtype=dtype,
        )
        raw_text = result["text"]
    except ImportError:
        print("Using manual inference path.", flush=True)
        raw_text = _manual_inference(model, processor, messages, device, dtype, args.max_new_tokens)

    print(f"\nRaw model output:\n{raw_text}\n", flush=True)

    segments = parse_transcript(raw_text)
    speakers = sorted(set(seg.speaker for seg in segments))
    print(f"Parsed {len(segments)} segments from {len(speakers)} speaker(s): {speakers}", flush=True)

    return {
        "raw_text": raw_text,
        "segments": [asdict(seg) for seg in segments],
        "speakers": speakers,
        "num_segments": len(segments),
        "num_speakers": len(speakers),
        "model": args.model,
        "device": device_str,
        "audio": audio_path,
    }


def _manual_inference(
    model: Any,
    processor: Any,
    messages: List[Dict],
    device: Any,
    dtype: Any,
    max_new_tokens: int,
) -> str:
    """Manual inference path when moss_transcribe_diarize package is not installed."""
    import torch

    # Apply chat template
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Load audio
    audio_path = messages[0]["content"][0]["audio"]
    sampling_rate = getattr(processor, "sampling_rate", 16000)
    if hasattr(processor, "feature_extractor") and hasattr(processor.feature_extractor, "sampling_rate"):
        sampling_rate = processor.feature_extractor.sampling_rate

    audio_arrays = _load_audio(audio_path, sampling_rate)

    # Process inputs
    audio_kwargs = {"sampling_rate": sampling_rate}
    inputs = processor(
        text=text,
        audio=audio_arrays,
        audio_kwargs=audio_kwargs,
        return_tensors="pt",
    )

    # Move to device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    # Generate
    prompt_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Decode only new tokens
    generated_ids = output_ids[0, prompt_len:]
    text_output = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text_output


def _load_audio(path: str, sampling_rate: int) -> list:
    """Load audio file and return list of numpy arrays at target sampling rate."""
    import numpy as np

    # Try librosa first (handles resampling well)
    try:
        import librosa
        audio, _ = librosa.load(path, sr=sampling_rate, mono=True)
        return [audio]
    except ImportError:
        pass

    # Try soundfile + soxr for resampling
    try:
        import soundfile as sf
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != sampling_rate:
            try:
                import soxr
                audio = soxr.resample(audio, sr, sampling_rate)
            except ImportError:
                # Simple linear interpolation as fallback
                ratio = sampling_rate / sr
                new_len = int(len(audio) * ratio)
                indices = np.linspace(0, len(audio) - 1, new_len)
                audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
        return [audio]
    except ImportError:
        pass

    # Fallback: wave module (only works for WAV at 16kHz)
    import wave
    with wave.open(path, "rb") as wf:
        assert wf.getsampwidth() == 2, "Only 16-bit WAV supported"
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        sr = wf.getframerate()
        if sr != sampling_rate:
            ratio = sampling_rate / sr
            new_len = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_len)
            audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
    return [audio]


def save_result(result: Dict[str, Any], output_path: str) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    payload = dict(result)
    payload["created_at"] = datetime.now(timezone.utc).isoformat()
    output_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved JSON output to: {output_file}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe and diarize speech with MOSS-Transcribe-Diarize.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--audio",
        default="sample",
        help="Audio file path, URL, or 'sample' for the default multi-speaker CVL sample.",
    )
    parser.add_argument(
        "--model",
        default="OpenMOSS-Team/MOSS-Transcribe-Diarize",
        help="HuggingFace model ID or local path.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Compute device.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Custom transcription prompt. Defaults to English diarization prompt.",
    )
    parser.add_argument(
        "--output",
        default="moss_transcript.json",
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    output_path = resolve_output_path(args.output)
    result = transcribe(args)

    # Print summary
    print("\n--- Transcript Summary ---", flush=True)
    for seg in result["segments"]:
        print(f"[{seg['start']:.2f} - {seg['end']:.2f}] [{seg['speaker']}] {seg['text']}", flush=True)
    print(f"\nSpeakers detected: {result['num_speakers']} ({', '.join(result['speakers'])})", flush=True)
    print(f"Total segments: {result['num_segments']}", flush=True)

    save_result(result, output_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
