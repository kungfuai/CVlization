#!/usr/bin/env python3
"""Voxtral Mini 4B Realtime streaming ASR client for CVlization.

Connects to a running vLLM server's /v1/realtime WebSocket endpoint, streams
audio in chunks, and captures incremental transcription events. Supports 13
languages: Arabic, German, English, Spanish, French, Hindi, Italian, Dutch,
Portuguese, Chinese, Japanese, Korean, Russian.

Two streaming modes:
  --mode fast    Send audio as fast as possible (file-mode; default for batch).
  --mode realtime  Pace audio at true playback rate with concurrent receive.
                   Records first-delta latency and per-delta timestamps.
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


def _writable_cache_root() -> Path:
    """Return a writable cache directory for sample data."""
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    preferred = hf_home / "cvl_data" / EXAMPLE_NAME
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        return preferred
    except PermissionError:
        fallback = Path("/tmp") / "cvl_data" / EXAMPLE_NAME
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def ensure_sample_audio(cache_root: Optional[Path] = None) -> Path:
    """Download the canonical CVL sample audio from HuggingFace."""
    if cache_root is None:
        cache_root = _writable_cache_root()
    sample_path = cache_root / "example_en.wav"
    if sample_path.exists():
        return sample_path

    from huggingface_hub import hf_hub_download

    print(f"Downloading sample audio from {HF_DATA_REPO}/{HF_SAMPLE_FILE}...")
    try:
        downloaded = hf_hub_download(
            repo_id=HF_DATA_REPO, filename=HF_SAMPLE_FILE, repo_type="dataset",
        )
    except PermissionError:
        # HF default cache not writable; download to /tmp
        downloaded = hf_hub_download(
            repo_id=HF_DATA_REPO, filename=HF_SAMPLE_FILE, repo_type="dataset",
            cache_dir="/tmp/hf_cache",
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


async def stream_transcribe_fast(
    audio_path: str,
    host: str,
    port: int,
    model: str,
    chunk_size: int = CHUNK_SIZE,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Fast file-mode: send all audio as quickly as possible, then receive.

    Best for throughput when latency measurement is not needed.
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
        print(f"Audio streamed in {t_stream_end - t_stream_start:.2f}s (fast mode). Waiting for transcription...\n")

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
        "mode": "fast",
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


async def stream_transcribe_realtime(
    audio_path: str,
    host: str,
    port: int,
    model: str,
    chunk_size: int = CHUNK_SIZE,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Realtime-paced mode: send audio at playback speed with concurrent receive.

    Sends chunks paced to match real audio duration and concurrently receives
    transcription events. Records:
      - first_event_latency_sec: time to first protocol delta (may be empty)
      - first_text_latency_sec: time to first non-empty text delta (user-visible)
      - delta_events: list of {delta, wall_clock_sec, audio_sent_sec} per event
    """
    import base64

    import websockets

    uri = f"ws://{host}:{port}/v1/realtime"
    delta_events: List[Dict[str, Any]] = []
    final_text = ""
    usage = {}
    session_id = "unknown"
    first_event_time: Optional[float] = None
    first_text_time: Optional[float] = None
    audio_sent_sec_shared: List[float] = [0.0]  # mutable shared state for sender progress
    send_done = asyncio.Event()

    print(f"Connecting to {uri} ...")
    ws = await websockets.connect(uri)

    try:
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

        # 4. Load audio
        print(f"Loading audio: {audio_path}")
        audio_bytes = load_audio_pcm16(audio_path)
        total_chunks = (len(audio_bytes) + chunk_size - 1) // chunk_size
        duration_sec = len(audio_bytes) / (16000 * 2)  # 16kHz, 16-bit
        # Seconds of audio per chunk
        chunk_duration_sec = chunk_size / (16000 * 2)
        print(f"Audio duration: {duration_sec:.1f}s ({total_chunks} chunks)")
        print(f"Mode: REALTIME (pacing at {chunk_duration_sec*1000:.0f}ms per chunk)\n")

        t_start = time.time()

        async def sender():
            """Send audio chunks paced at realtime speed."""
            nonlocal send_done
            t_send_start = time.time()
            chunks_sent = 0
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i : i + chunk_size]
                await ws.send(
                    json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode("utf-8"),
                    })
                )
                chunks_sent += 1
                audio_sent_sec_shared[0] = chunks_sent * chunk_duration_sec
                # Pace: wait until the wall clock matches the audio time sent so far
                target_elapsed = chunks_sent * chunk_duration_sec
                actual_elapsed = time.time() - t_send_start
                sleep_time = target_elapsed - actual_elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            # Signal end of audio
            await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
            send_done.set()
            t_send_end = time.time()
            if verbose:
                print(f"\n[sender done: {t_send_end - t_send_start:.2f}s for {duration_sec:.1f}s audio]")

        async def receiver():
            """Receive transcription events concurrently."""
            nonlocal final_text, usage, first_event_time, first_text_time
            print("Transcription: ", end="", flush=True)
            while True:
                response = json.loads(await ws.recv())
                msg_type = response.get("type", "")
                wall_sec = time.time() - t_start

                if msg_type == "transcription.delta":
                    delta = response.get("delta", "")
                    if first_event_time is None:
                        first_event_time = wall_sec
                    if first_text_time is None and delta:
                        first_text_time = wall_sec
                    delta_events.append({
                        "delta": delta,
                        "wall_clock_sec": round(wall_sec, 3),
                        "audio_sent_sec": round(audio_sent_sec_shared[0], 3),
                    })
                    print(delta, end="", flush=True)
                elif msg_type == "transcription.done":
                    final_text = response.get("text", "")
                    usage = response.get("usage", {})
                    break
                elif msg_type == "error":
                    error_msg = response.get("error", response)
                    raise RuntimeError(f"Server error: {error_msg}")
                elif verbose:
                    print(f"\n[event: {msg_type} @ {wall_sec:.2f}s]", end="", flush=True)

        # Run sender and receiver concurrently
        await asyncio.gather(sender(), receiver())

    finally:
        await ws.close()

    t_end = time.time()
    total_sec = t_end - t_start
    first_event_latency = first_event_time if first_event_time is not None else total_sec
    first_text_latency = first_text_time if first_text_time is not None else total_sec

    print(f"\n\nTotal time: {total_sec:.2f}s")
    print(f"First event latency: {first_event_latency:.3f}s (protocol delta, may be empty)")
    print(f"First text latency:  {first_text_latency:.3f}s (first non-empty text)")
    print(f"Audio duration: {duration_sec:.1f}s")

    return {
        "mode": "realtime",
        "text": final_text,
        "deltas": [e["delta"] for e in delta_events],
        "delta_events": delta_events,
        "first_event_latency_sec": round(first_event_latency, 3),
        "first_text_latency_sec": round(first_text_latency, 3),
        "model": model,
        "audio": audio_path,
        "audio_duration_sec": round(duration_sec, 2),
        "total_chunks": total_chunks,
        "session_id": session_id,
        "timing": {
            "total_sec": round(total_sec, 2),
            "audio_paced_sec": round(duration_sec, 2),
            "first_event_sec": round(first_event_latency, 3),
            "first_text_sec": round(first_text_latency, 3),
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
    p.add_argument(
        "--mode", choices=["fast", "realtime"], default="fast",
        help="Streaming mode: 'fast' sends as quickly as possible; "
             "'realtime' paces audio at playback speed with concurrent receive."
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

    if args.mode == "realtime":
        result = asyncio.run(
            stream_transcribe_realtime(
                audio_path=audio_path,
                host=args.host,
                port=args.port,
                model=args.model,
                chunk_size=args.chunk_size,
                verbose=args.verbose,
            )
        )
    else:
        result = asyncio.run(
            stream_transcribe_fast(
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
