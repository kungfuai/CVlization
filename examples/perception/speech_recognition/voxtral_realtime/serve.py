#!/usr/bin/env python3
"""vLLM server launcher for Voxtral Mini 4B Realtime.

Starts the vLLM OpenAI-compatible server with the Voxtral realtime model,
exposing the /v1/realtime WebSocket endpoint for streaming transcription.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys


def main():
    model_id = os.getenv("MODEL_ID", "mistralai/Voxtral-Mini-4B-Realtime-2602")
    host = os.getenv("HOST", "0.0.0.0")
    port = os.getenv("PORT", "8000")
    max_model_len = os.getenv("VLLM_MAX_MODEL_LEN", "45000")
    extra_args = os.getenv("VLLM_EXTRA_ARGS", "")

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model", model_id,
        "--host", host,
        "--port", port,
        "--max-model-len", max_model_len,
        "--dtype", "bfloat16",
        "--trust-remote-code",
        "--enforce-eager",
    ]

    if extra_args:
        cmd.extend(shlex.split(extra_args))

    print(f"Starting vLLM server for {model_id}")
    print(f"  Host: {host}:{port}")
    print(f"  Max model len: {max_model_len}")
    print(f"  Command: {' '.join(shlex.quote(c) for c in cmd)}")
    sys.stdout.flush()

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
