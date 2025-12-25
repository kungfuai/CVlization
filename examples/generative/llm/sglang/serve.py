#!/usr/bin/env python3
"""
Auto-tuned SGLang server launcher (OpenAI-compatible HTTP).
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional

import torch

from gpu_utils import get_optimal_attention_backend


DEFAULT_MODEL = os.getenv("MODEL_ID", "allenai/Olmo-3-7B-Instruct")


@dataclass
class ServeConfig:
    model_id: str
    host: str
    port: int
    tp_size: int
    context_length: int
    dtype: str
    mem_fraction_static: float
    trust_remote_code: bool
    tokenizer_path: Optional[str] = None
    extra_args: str = ""


def detect_gpu_info():
    if not torch.cuda.is_available():
        return {"gpus": 0, "min_mem_gb": 0.0, "names": []}
    count = torch.cuda.device_count()
    mem_gb = []
    names = []
    for idx in range(count):
        props = torch.cuda.get_device_properties(idx)
        mem_gb.append(props.total_memory / (1024**3))
        names.append(props.name)
    return {"gpus": count, "min_mem_gb": min(mem_gb), "names": names}


def recommend_settings(user_tp: Optional[int], user_ctx: Optional[int], user_dtype: Optional[str],
                       user_mem_frac: Optional[float]) -> ServeConfig:
    info = detect_gpu_info()
    gpus = info["gpus"]
    min_mem = info["min_mem_gb"]

    if gpus == 0:
        tp = 1
        context = user_ctx or 4096
        dtype = user_dtype or "float32"
        mem_frac = user_mem_frac or 0.8
    else:
        tp = user_tp or gpus
        tp = max(1, min(tp, gpus))
        if user_ctx:
            context = user_ctx
        elif min_mem >= 80:
            context = 65536
        elif min_mem >= 48:
            context = 32768
        elif min_mem >= 24:
            context = 16384
        elif min_mem >= 16:
            context = 8192
        else:
            context = 4096
        dtype = user_dtype or "bfloat16"
        mem_frac = user_mem_frac or (0.92 if min_mem >= 32 else 0.9)

    return ServeConfig(
        model_id=DEFAULT_MODEL,
        host="0.0.0.0",
        port=30000,
        tp_size=tp,
        context_length=context,
        dtype=dtype,
        mem_fraction_static=mem_frac,
        trust_remote_code=True,
    )


def build_command(cfg: ServeConfig) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        cfg.model_id,
        "--tokenizer-path",
        cfg.tokenizer_path or cfg.model_id,
        "--host",
        cfg.host,
        "--port",
        str(cfg.port),
        "--tensor-parallel-size",
        str(cfg.tp_size),
        "--context-length",
        str(cfg.context_length),
        "--dtype",
        cfg.dtype,
        "--mem-fraction-static",
        str(cfg.mem_fraction_static),
    ]

    if cfg.trust_remote_code:
        cmd.append("--trust-remote-code")
    if cfg.extra_args:
        cmd.extend(shlex.split(cfg.extra_args))
    # Auto-detect attention backend if not explicitly specified
    if "--attention-backend" not in " ".join(cmd):
        backend = get_optimal_attention_backend()
        cmd.extend(["--attention-backend", backend])
        print(f"Auto-selected attention backend: {backend}")
    return cmd


def parse_args():
    parser = argparse.ArgumentParser(description="Auto-configured SGLang server")
    parser.add_argument("--model-id", default=DEFAULT_MODEL, help="HF model to serve")
    parser.add_argument("--tokenizer-path", default=os.getenv("TOKENIZER_PATH"))
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "30000")))
    parser.add_argument("--trust-remote-code", action="store_true",
                        default=os.getenv("TRUST_REMOTE_CODE", "1") != "0")

    parser.add_argument("--tp-size", type=int, default=None,
                        help="Tensor parallel size (env: SGLANG_TP_SIZE)")
    parser.add_argument("--context-length", type=int, default=None,
                        help="Override max context (env: SGLANG_CONTEXT_LENGTH)")
    parser.add_argument("--dtype", default=None, help="Override dtype, e.g. bfloat16 (env: SGLANG_DTYPE)")
    parser.add_argument("--mem-fraction-static", type=float, default=None,
                        help="Fraction of memory for weights+KV (env: SGLANG_MEM_FRACTION_STATIC)")
    parser.add_argument("--extra-args", default=os.getenv("SGLANG_EXTRA_ARGS", ""),
                        help="Additional args passed verbatim to sglang.launch_server")
    return parser.parse_args()


def main():
    args = parse_args()
    user_tp = args.tp_size or os.getenv("SGLANG_TP_SIZE")
    user_tp = int(user_tp) if user_tp else None

    user_ctx = args.context_length or os.getenv("SGLANG_CONTEXT_LENGTH")
    user_ctx = int(user_ctx) if user_ctx else None

    user_dtype = args.dtype or os.getenv("SGLANG_DTYPE")

    user_mem = args.mem_fraction_static or os.getenv("SGLANG_MEM_FRACTION_STATIC")
    user_mem = float(user_mem) if user_mem else None

    cfg = recommend_settings(user_tp, user_ctx, user_dtype, user_mem)
    cfg.model_id = args.model_id
    cfg.tokenizer_path = args.tokenizer_path or args.model_id
    cfg.host = args.host
    cfg.port = args.port
    cfg.trust_remote_code = args.trust_remote_code
    cfg.extra_args = args.extra_args

    print("SGLang auto-configuration:")
    print(f"  GPUs detected: {detect_gpu_info()}")
    print(f"  model: {cfg.model_id}")
    print(f"  tokenizer: {cfg.tokenizer_path}")
    print(f"  tp_size: {cfg.tp_size}")
    print(f"  context_length: {cfg.context_length}")
    print(f"  dtype: {cfg.dtype}")
    print(f"  mem_fraction_static: {cfg.mem_fraction_static}")
    if cfg.extra_args:
        print(f"  extra_args: {cfg.extra_args}")

    cmd = build_command(cfg)
    print("\nLaunching SGLang:")
    print(" ", " ".join(shlex.quote(arg) for arg in cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
