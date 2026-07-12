#!/usr/bin/env python3
"""Voxtral Mini 4B Realtime streaming ASR client for CVlization.

Connects to a running vLLM server's /v1/realtime WebSocket endpoint, streams
audio in chunks, and captures incremental transcription events. Supports 13
languages: Arabic, German, English, Spanish, French, Hindi, Italian, Dutch,
Portuguese, Chinese, Japanese, Korean, Russian.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger("voxtral_realtime_cvl")

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
HF_SAMPLE_FILE = "voxtral_realtime/example_en.wav"
EXAMPLE_NAME = "voxtral_realtime"
MODEL_ID = "mistralai/Voxtral-Mini-4B-Realtime-2602"
CHUNK_SIZE = 4096  # 4KB raw audio per WebSocket message


def configure_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.ERROR
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s", force=True)
    LOGGER.setLevel(level)


def ensure_sample_audio(cache_root: Optional[Path] = None) -> Path:
    """Download the canonical CVL sample audio from HuggingFace."""
    if cache_root is None:
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        cache_root = hf_home / "cvl_data" / EXAMPLE_NAME
    sample_path = cache_root / "example_en.wav"
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
    """Resolve the --audio argument to a file path."""
    if not audio or audio == "sample":
        return str(ensure_sample_audio())
    return resolve_input_path(audio)


def load_audio_pcm16(audio_path: str, target_sr: int = 16000) -> bytes:
    """Load an audio file and convert to PCM16 @ target sample rate."""
    import librosa
    import numpy as np

    audio, _ = librosa.load(audio_path, sr=target_sr, mono=True)
    pcm16 = (audio * 32767).astype(np.int16)
    return pcm16.tobytes()


async def stream_transcribe(
    audio_path: str,
    host: str,
    port: int,
    model: str,
    chunk_size: int = CHUNK_SIZE,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Connect to the vLLM realtime WebSocket and stream audio for transcription.

    Returns a dict with transcript text, incremental deltas, timing, and usage.
    """
    import base64

    import websockets

    uri = f"ws://{host}:{port}/v1/realtime"
    deltas: List[str] = []
    final_text = ""
    usage = {}
    t_start = time.time()

    print(f"Connecting to {uri} ...")
    async with websockets.connect(uri) as ws:
        # 1. Receive session.created
        response = json.loads(await ws.recv())
        if response.get("type") != "session.created":
            raise RuntimeError(f"Expected session.created, got: {response}")
        session_id = response.get("id", "unknown")
        if verbose:
            print(f"Session created: {session_id}")

        # 2. Validate model
        await ws.send(json.dumps({"type": "session.update", "model": model}))

        # 3. Signal ready
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        # 4. Load and stream audio
        print(f"Loading audio: {audio_path}")
        audio_bytes = load_audio_pcm16(audio_path)
        total_chunks = (len(audio_bytes) + chunk_size - 1) // chunk_size
        duration_sec = len(audio_bytes) / (16000 * 2)  # 16kHz, 16-bit
        print(f"Audio duration: {duration_sec:.1f}s ({total_chunks} chunks)")

        t_stream_start = time.time()
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i : i + chunk_size]
            await ws.send(
                json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("utf-8"),
                })
            )

        # 5. Signal end of audio
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
        t_stream_end = time.time()
        print(f"Audio streamed in {t_stream_end - t_stream_start:.2f}s. Waiting for transcription...\n")

        # 6. Receive transcription events
        print("Transcription: ", end="", flush=True)
        while True:
            response = json.loads(await ws.recv())
            msg_type = response.get("type", "")

            if msg_type == "transcription.delta":
                delta = response.get("delta", "")
                deltas.append(delta)
                print(delta, end="", flush=True)
            elif msg_type == "transcription.done":
                final_text = response.get("text", "")
                usage = response.get("usage", {})
                break
            elif msg_type == "error":
                error_msg = response.get("error", response)
                raise RuntimeError(f"Server error: {error_msg}")
            elif verbose:
                print(f"\n[event: {msg_type}]", end="", flush=True)

    t_end = time.time()
    print(f"\n\nTotal time: {t_end - t_start:.2f}s")

    return {
        "text": final_text,
        "deltas": deltas,
        "model": model,
        "audio": audio_path,
        "audio_duration_sec": round(duration_sec, 2),
        "total_chunks": total_chunks,
        "session_id": session_id,
        "timing": {
            "total_sec": round(t_end - t_start, 2),
            "stream_sec": round(t_stream_end - t_stream_start, 2),
            "transcription_sec": round(t_end - t_stream_end, 2),
        },
        "usage": usage,
    }


def save_result(result: Dict[str, Any], output_path: str, output_format: str) -> None:
    """Save the transcription result to disk."""
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
        description="Voxtral Mini 4B Realtime streaming ASR client.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--audio", default="sample",
        help="Audio file path or 'sample' to use the default CVL sample."
    )
    p.add_argument(
        "--model", default=MODEL_ID,
        help="Model ID served by the vLLM server."
    )
    p.add_argument(
        "--host", default=os.getenv("VOXTRAL_HOST", "localhost"),
        help="vLLM server host."
    )
    p.add_argument(
        "--port", type=int, default=int(os.getenv("VOXTRAL_PORT", "8000")),
        help="vLLM server port."
    )
    p.add_argument(
        "--chunk-size", type=int, default=CHUNK_SIZE,
        help="Audio chunk size in bytes for streaming."
    )
    p.add_argument("--output", default="voxtral_realtime_transcript.json")
    p.add_argument("--format", choices=["json", "txt"], default="json")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    audio_path = resolve_audio_arg(args.audio)
    if not Path(audio_path).exists():
        print(f"ERROR: Audio file not found: {audio_path}", file=sys.stderr)
        return 1

    output_path = resolve_output_path(args.output)

    result = asyncio.run(
        stream_transcribe(
            audio_path=audio_path,
            host=args.host,
            port=args.port,
            model=args.model,
            chunk_size=args.chunk_size,
            verbose=args.verbose,
        )
    )

    print()
    print("Final transcript:")
    print(result["text"] or "(empty)")
    print()

    if not result["text"]:
        print("WARNING: Empty transcript received", file=sys.stderr)
        return 1

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
