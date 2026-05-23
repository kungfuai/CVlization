"""GPU detection helpers for llama.cpp auto-tuning.

llama-server is simpler than vLLM/SGLang on GPU selection: a single
`-ngl <n>` knob controls how many transformer layers to offload to GPU.
For modern GPUs with enough VRAM, `-ngl 999` (offload all) is the right
default; we only need to know whether a GPU is present and roughly how
much memory to plan for context.

We avoid importing torch — the llama.cpp upstream image doesn't ship it.
Instead we shell out to `nvidia-smi`, which the CUDA image always has.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Tuple


def detect_gpu() -> Tuple[int, float]:
    """Return (gpu_count, min_vram_gb). (0, 0.0) on CPU-only hosts."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return 0, 0.0
    try:
        out = subprocess.run(
            [smi, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    except Exception:
        return 0, 0.0
    if not out:
        return 0, 0.0
    mems_mib = [int(x.strip()) for x in out.splitlines() if x.strip()]
    if not mems_mib:
        return 0, 0.0
    return len(mems_mib), min(mems_mib) / 1024.0


def recommend_context_length(min_vram_gb: float) -> int:
    """Pick `-c` (context length) using the same brackets as the sglang preset."""
    if min_vram_gb >= 80:
        return 65536
    if min_vram_gb >= 48:
        return 32768
    if min_vram_gb >= 24:
        return 16384
    if min_vram_gb >= 16:
        return 8192
    return 4096


def recommend_gpu_layers(min_vram_gb: float) -> int:
    """`-ngl` default. Full offload (999) if any GPU is present, else 0 (CPU)."""
    return 999 if min_vram_gb > 0 else 0
