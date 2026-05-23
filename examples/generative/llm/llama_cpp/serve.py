#!/usr/bin/env python3
"""Auto-tuned llama.cpp OpenAI-compatible server launcher.

Wraps `llama-server` with sensible defaults sourced from env vars:
  MODEL_ID            -- HF repo + quant tag for `-hf`, e.g. Qwen/Qwen3-8B-GGUF:Q4_K_M
                        (or a local .gguf path via MODEL_PATH)
  MODEL_PATH          -- absolute path to a local .gguf file (overrides MODEL_ID)
  HOST, PORT          -- bind address + port (default 0.0.0.0:8080)
  LLAMA_CONTEXT_LENGTH -- `-c` context size (auto-tuned by VRAM if unset)
  LLAMA_GPU_LAYERS    -- `-ngl` layers to offload (auto: 999 if GPU else 0)
  LLAMA_PARALLEL_SLOTS -- `-np` parallel decoding slots (default 1)
  LLAMA_REASONING_FORMAT -- `--reasoning-format` value (default `auto`)
  LLAMA_USE_JINJA     -- if "1" (default), pass `--jinja` for HF chat templates
  LLAMA_FLASH_ATTN    -- if "1" (default), pass `-fa` for flash attention
  LLAMA_API_KEY       -- if set, served with `--api-key` (else open)
  LLAMA_EXTRA_ARGS    -- raw extra args appended to the llama-server command
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys

from gpu_utils import detect_gpu, recommend_context_length, recommend_gpu_layers


DEFAULT_MODEL = os.getenv("MODEL_ID", "Qwen/Qwen3-8B-GGUF:Q4_K_M")


def env_str(name: str, default: str) -> str:
    """Return env value, treating empty string the same as unset."""
    v = os.getenv(name, "").strip()
    return v if v else default


def env_int(name: str, default: int) -> int:
    return int(env_str(name, str(default)))


def env_bool(name: str, default: str = "1") -> bool:
    return env_str(name, default).lower() in ("1", "true", "yes")


def build_cmd() -> list[str]:
    gpu_count, min_vram_gb = detect_gpu()

    ctx = env_int("LLAMA_CONTEXT_LENGTH", recommend_context_length(min_vram_gb))
    ngl = env_int("LLAMA_GPU_LAYERS", recommend_gpu_layers(min_vram_gb))
    parallel = env_int("LLAMA_PARALLEL_SLOTS", 1)
    host = env_str("HOST", "0.0.0.0")
    port = env_int("PORT", 8080)
    reasoning_fmt = env_str("LLAMA_REASONING_FORMAT", "auto")

    cmd: list[str] = ["llama-server", "--host", host, "--port", str(port),
                      "-c", str(ctx), "-ngl", str(ngl), "-np", str(parallel),
                      "--reasoning-format", reasoning_fmt]

    if env_bool("LLAMA_USE_JINJA"):
        cmd.append("--jinja")
    if env_bool("LLAMA_FLASH_ATTN"):
        cmd.extend(["-fa", "on"])

    model_path = os.getenv("MODEL_PATH")
    if model_path:
        cmd.extend(["-m", model_path])
    else:
        cmd.extend(["-hf", DEFAULT_MODEL])

    api_key = os.getenv("LLAMA_API_KEY")
    if api_key:
        cmd.extend(["--api-key", api_key])

    extra = os.getenv("LLAMA_EXTRA_ARGS", "").strip()
    if extra:
        cmd.extend(shlex.split(extra))

    print(f"llama.cpp auto-config: gpu_count={gpu_count} min_vram_gb={min_vram_gb:.1f} "
          f"ctx={ctx} ngl={ngl} parallel={parallel}")
    print("Launching llama-server:")
    print(" ", " ".join(shlex.quote(a) for a in cmd))
    return cmd


def main() -> int:
    return subprocess.run(build_cmd(), check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
