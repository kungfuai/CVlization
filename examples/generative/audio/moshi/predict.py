#!/usr/bin/env python3
"""Kyutai Moshi batch-dialogue wrapper for CVLization.

Moshi is a full-duplex speech-to-speech dialogue model. The natural way to
demo it live is via the WebSocket server (`serve.sh`) + a browser/terminal
client. For a one-shot CVL `predict` preset we use the upstream
`moshi-inference` CLI (== `python -m moshi.run_inference`) which:
  - takes ONE input audio file (the user turn),
  - generates Moshi's spoken response,
  - writes it to an output wav.

This wrapper resolves the input/output paths through cvlization.paths (so
the output lands in the host cwd under `cvl run`), then shells out to the
upstream CLI.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
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
        default_filename: str = "result.wav",
    ) -> str:
        output_root = output_dir or Path("outputs")
        output_root.mkdir(parents=True, exist_ok=True)
        path = path or default_filename
        return path if path.startswith("/") else str((output_root / path).resolve())


HF_DATA_REPO = "zzsi/cvl"
HF_SAMPLE_FILE = "livetalk/example.wav"
EXAMPLE_NAME = "moshi"
DEFAULT_HF_REPO = "kyutai/moshiko-pytorch-bf16"


def _env(name: str, default: str) -> str:
    """Return env value, treating empty string the same as unset."""
    v = os.getenv(name, "").strip()
    return v if v else default


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


def maybe_pad_with_silence(audio_path: str, target_seconds: float) -> str:
    """Pad `audio_path` at the end with silence so total length >= target_seconds.

    Moshi's reply duration tracks input duration — once the user-turn audio ends,
    Moshi stops generating. Padding the input with silence gives Moshi a window
    to keep talking, so the reply doesn't get cut off mid-sentence.

    Returns the path actually used for inference (a sibling `.padded.wav` if
    padding was applied, otherwise the original path unchanged).
    """
    if target_seconds <= 0:
        return audio_path
    import soundfile as sf
    import numpy as np
    info = sf.info(audio_path)
    current_seconds = info.frames / float(info.samplerate)
    if current_seconds >= target_seconds:
        return audio_path
    data, sr = sf.read(audio_path, always_2d=False)
    needed = int(round((target_seconds - current_seconds) * sr))
    if data.ndim == 1:
        silence = np.zeros(needed, dtype=data.dtype)
    else:
        silence = np.zeros((needed, data.shape[1]), dtype=data.dtype)
    padded = np.concatenate([data, silence], axis=0)
    out_path = str(Path(audio_path).with_suffix(".padded.wav"))
    sf.write(out_path, padded, sr)
    print(f"Padded input {current_seconds:.1f}s -> {target_seconds:.1f}s with silence -> {out_path}")
    return out_path


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def extract_moshi_text(combined: str) -> str:
    """Strip Moshi's text reply out of moshi-inference's stdout/stderr.

    moshi-inference streams text tokens (via printer.print_token) interleaved
    with two log streams (`Info:` and ANSI-coloured `[Info]`). The reply text
    sits between the "loading input file" log line and the "[Info] writing"
    line just before the wav is dumped. Strip ANSI codes first so the regex
    doesn't have to deal with them.
    """
    no_ansi = _ANSI_RE.sub("", combined)
    match = re.search(
        r"loading input file.*?\n(?P<text>.*?)(?:\[Info\]\s*writing|Info: processed)",
        no_ansi,
        re.DOTALL,
    )
    if not match:
        return ""
    raw = match.group("text")
    # Drop any "[Info]/Info:/Warning" log lines that interleave with the tokens.
    cleaned_lines = [
        line for line in raw.splitlines()
        if "Info:" not in line and "[Info]" not in line and "Warning" not in line
    ]
    return "\n".join(cleaned_lines).strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch spoken dialogue with Kyutai Moshi (user wav in -> Moshi wav out).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--audio", default="sample",
                   help="Input user-turn audio. Path / URL / 'sample' for the bundled CVL clip.")
    p.add_argument("--output", default="moshi_response.wav",
                   help="Output wav where Moshi's response is written.")
    p.add_argument("--hf-repo", default=_env("MOSHI_HF_REPO", DEFAULT_HF_REPO),
                   help="Moshi checkpoint repo on HuggingFace (Moshiko male / Moshika female; pytorch-bf16 default).")
    p.add_argument("--device", default=_env("MOSHI_DEVICE", "cuda"),
                   choices=["cuda", "cpu"])
    p.add_argument("--cfg-coef", type=float, default=float(_env("MOSHI_CFG_COEF", "1.0")),
                   help="Classifier-free guidance coefficient (1.0 = off).")
    p.add_argument("--batch-size", type=int, default=int(_env("MOSHI_BATCH_SIZE", "1")),
                   help="moshi-inference writes one wav per batch entry (suffixed -N.wav). "
                        "Default 1 yields a single response wav.")
    p.add_argument("--pad-input-to", type=float, default=float(_env("MOSHI_PAD_INPUT_TO", "15.0")),
                   help="Pad input audio at the end with silence to reach at least this many "
                        "seconds before inference. Moshi's reply duration tracks input duration, "
                        "so a short user turn truncates the response. 0 disables.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    input_path = resolve_audio_arg(args.audio)
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input audio not found: {input_path}")
    input_path = maybe_pad_with_silence(input_path, args.pad_input_to)

    output_path = resolve_output_path(args.output)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "moshi.run_inference",
        "--hf-repo", args.hf_repo,
        "--device", args.device,
        "--cfg-coef", str(args.cfg_coef),
        "--batch-size", str(args.batch_size),
        input_path, output_path,
    ]
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Model:  {args.hf_repo} (device: {args.device})")
    print("Launching moshi-inference:")
    print(" ", " ".join(shlex.quote(a) for a in cmd))

    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    # Forward upstream's stdout/stderr verbatim so users still see progress logs.
    if proc.stdout:
        sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        sys.stderr.write(f"\nmoshi-inference exited {proc.returncode}\n")
        return proc.returncode

    # Print Moshi's text reply on its own, distinct from the noisy logs above.
    combined = (proc.stdout or "") + (proc.stderr or "")
    reply_text = extract_moshi_text(combined)
    if reply_text:
        print("\n" + "=" * 60)
        print("Moshi reply (text):")
        print(reply_text)
        print("=" * 60)

    # moshi-inference writes <stem>-N.wav (one per batch entry). For batch_size=1
    # the single output is <stem>-0.wav — rename it to the user-requested path.
    out = Path(output_path)
    suffixed_zero = out.with_name(f"{out.stem}-0{out.suffix}")
    if not out.exists() and suffixed_zero.exists():
        suffixed_zero.rename(out)
    if not out.exists() or out.stat().st_size == 0:
        # If batch_size > 1, list the per-entry files instead.
        siblings = sorted(out.parent.glob(f"{out.stem}-*{out.suffix}"))
        if siblings:
            print("Moshi wrote per-batch wavs:")
            for s in siblings:
                print(f"  {s} ({s.stat().st_size:,} bytes)")
            return 0
        sys.stderr.write(f"Expected output at {output_path} but it's missing or empty.\n")
        return 1
    print(f"Saved Moshi response to: {output_path} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
