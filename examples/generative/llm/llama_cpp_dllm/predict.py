#!/usr/bin/env python3
"""Run a diffusion language model (LLaDA, Dream) via llama.cpp's
`llama-diffusion-cli`.

Diffusion LMs don't fit llama.cpp's OpenAI-compatible llama-server path
(it crashes with "the current context does not logits computation"), so
this preset shells out to the dedicated `llama-diffusion-cli` binary
shipped in the same image as our `llama_cpp` example.

The CLI prints diffusion-step progress to stdout/stderr and then the
final generation. We capture the full output, extract the text after the
final "total time:" marker, and save that.

Env / args:
  MODEL_ID                 HF repo:quant (default mradermacher/LLaDA-8B-Instruct-GGUF:Q4_K_M)
  MODEL_PATH               local .gguf overrides MODEL_ID
  LLAMA_GPU_LAYERS         `-ngl` (default 999 if GPU else 0)
  LLAMA_DIFFUSION_STEPS    `--diffusion-steps` (default 128 = upstream default)
  LLAMA_DIFFUSION_ALGORITHM `--diffusion-algorithm` (default 4 = confidence-based)
  LLAMA_DIFFUSION_BLOCK_LENGTH  LLaDA only; default 32 for LLaDA, 0 otherwise
  LLAMA_DIFFUSION_EPS      Dream only; default 1e-3 for Dream, 0 otherwise
  LLAMA_DIFFUSION_CFG_SCALE LLaDA classifier-free guidance (default 0)
  LLAMA_EXTRA_ARGS         raw extra args appended to the CLI command
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

from cvlization.paths import resolve_output_path
from gpu_utils import detect_gpu, recommend_gpu_layers


DEFAULT_MODEL = os.getenv("MODEL_ID", "mradermacher/LLaDA-8B-Instruct-GGUF:Q4_K_M")


def env_str(name: str, default: str) -> str:
    v = os.getenv(name, "").strip()
    return v if v else default


def env_int(name: str, default: int) -> int:
    return int(env_str(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(env_str(name, str(default)))


def infer_family(model_id: str) -> str:
    """Return 'llada' or 'dream' (defaults to 'llada')."""
    lower = model_id.lower()
    if "dream" in lower:
        return "dream"
    return "llada"


def build_cmd(args) -> list[str]:
    _, min_vram_gb = detect_gpu()
    ngl = env_int("LLAMA_GPU_LAYERS", recommend_gpu_layers(min_vram_gb))
    steps = env_int("LLAMA_DIFFUSION_STEPS", 128)
    algorithm = env_int("LLAMA_DIFFUSION_ALGORITHM", 4)
    cfg_scale = env_float("LLAMA_DIFFUSION_CFG_SCALE", 0.0)

    family = infer_family(args.model)
    # llama-diffusion-cli asserts (eps == 0) XOR (block_length == 0).
    if family == "dream":
        block_length = env_int("LLAMA_DIFFUSION_BLOCK_LENGTH", 0)
        eps = env_float("LLAMA_DIFFUSION_EPS", 1e-3)
    else:
        block_length = env_int("LLAMA_DIFFUSION_BLOCK_LENGTH", 32)
        eps = env_float("LLAMA_DIFFUSION_EPS", 0.0)

    cmd: list[str] = ["llama-diffusion-cli",
                      "-p", args.prompt,
                      "-n", str(args.max_tokens),
                      "-c", str(args.context_length),
                      "-ngl", str(ngl),
                      "--diffusion-steps", str(steps),
                      "--diffusion-algorithm", str(algorithm),
                      "--diffusion-block-length", str(block_length),
                      "--diffusion-eps", str(eps),
                      "--diffusion-cfg-scale", str(cfg_scale)]

    model_path = os.getenv("MODEL_PATH", "").strip()
    if model_path:
        cmd.extend(["-m", model_path])
    else:
        cmd.extend(["-hf", args.model])

    extra = env_str("LLAMA_EXTRA_ARGS", "")
    if extra:
        cmd.extend(shlex.split(extra))

    print(f"diffusion auto-config: family={family} ngl={ngl} steps={steps} "
          f"algorithm={algorithm} block_length={block_length} eps={eps} cfg_scale={cfg_scale}")
    print("Launching llama-diffusion-cli:")
    print(" ", " ".join(shlex.quote(a) for a in cmd))
    return cmd


_GEN_MARKER = re.compile(r"^.*total time:.*$")
# llama.cpp log lines look like "0.08.454.373 I ..." (timestamp + log-level letter).
_LOG_LINE = re.compile(r"^\d+(?:\.\d+)+\s+[IWED]\s.*$|^\d+(?:\.\d+)+\s+[IWED]\s*$")


def extract_generation(raw: str) -> str:
    """Return everything after the final "total time:" line, stripped of log noise."""
    lines = raw.splitlines()
    last_marker = -1
    for i, line in enumerate(lines):
        if _GEN_MARKER.match(line):
            last_marker = i
    if last_marker < 0:
        # Fallback: return the whole captured output.
        return raw.strip()
    tail = lines[last_marker + 1:]
    # Drop leading empty lines + llama.cpp log lines (timestamped or bare "I ...").
    while tail and (not tail[0].strip()
                    or _LOG_LINE.match(tail[0])
                    or tail[0].lstrip().startswith("I ")):
        tail.pop(0)
    return "\n".join(tail).strip()


def save_output(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n" if text else "", encoding="utf-8")
    print(f"Saved output to {path}")


def parse_args():
    p = argparse.ArgumentParser(description="llama.cpp diffusion-LM inference (LLaDA, Dream).")
    p.add_argument("--prompt", default="Say hello in one sentence.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-tokens", type=int, default=env_int("LLAMA_N_PREDICT", 256))
    p.add_argument("--context-length", type=int, default=env_int("LLAMA_CONTEXT_LENGTH", 4096))
    p.add_argument("--output", type=Path, default=Path("outputs/result.txt"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cmd = build_cmd(args)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        sys.stderr.write(f"\nllama-diffusion-cli exited {proc.returncode}\n")
        return proc.returncode
    # diffusion-cli interleaves info on stdout AND stderr; combine for parsing.
    combined = (proc.stdout or "") + (proc.stderr or "")
    generation = extract_generation(combined)
    print("Generation:\n")
    print(generation if generation else "(empty)")
    save_output(generation, Path(resolve_output_path(str(args.output))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
