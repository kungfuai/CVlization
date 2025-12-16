#!/usr/bin/env python3
"""
Auto-tuned vLLM server launcher.

Heuristics pick tensor parallelism, dtype, max context, and GPU memory
utilization based on detected hardware. Any setting can be overridden via
CLI flags or environment variables.
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


DEFAULT_MODEL = os.getenv("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")


@dataclass
class ServeConfig:
    model_id: str
    host: str
    port: int
    tp_size: int
    max_model_len: int
    dtype: str
    gpu_memory_utilization: float
    trust_remote_code: bool
    served_model_name: Optional[str] = None
    extra_args: str = ""


def detect_gpu_info():
    if not torch.cuda.is_available():
        return {"gpus": 0, "min_mem_gb": 0.0, "names": []}
    count = torch.cuda.device_count()
    mem_gb = []
    names = []
    for idx in range(count):
        props = torch.cuda.get_device_properties(idx)
        mem_gb.append(props.total_memory / (1024 ** 3))
        names.append(props.name)
    return {"gpus": count, "min_mem_gb": min(mem_gb), "names": names}


def recommend_settings(user_tp: Optional[int], user_max_len: Optional[int], user_dtype: Optional[str],
                       user_gpu_mem_util: Optional[float]) -> ServeConfig:
    info = detect_gpu_info()
    gpus = info["gpus"]
    min_mem = info["min_mem_gb"]

    if gpus == 0:
        # CPU-only fallback; warn and use conservative defaults.
        tp = 1
        max_len = user_max_len or 4096
        dtype = user_dtype or "float32"
        gpu_mem_util = user_gpu_mem_util or 0.85
    else:
        tp = user_tp or gpus
        # Cap TP to available GPUs
        tp = max(1, min(tp, gpus))

        # Max context heuristics based on smallest GPU in the node.
        if user_max_len:
            max_len = user_max_len
        elif min_mem >= 80:
            max_len = 131072
        elif min_mem >= 48:
            max_len = 65536
        elif min_mem >= 24:
            max_len = 16384
        elif min_mem >= 16:
            max_len = 8192
        else:
            max_len = 4096

        dtype = user_dtype or "bfloat16"
        gpu_mem_util = user_gpu_mem_util or (0.92 if min_mem >= 32 else 0.9)

    return ServeConfig(
        model_id="",
        host="0.0.0.0",
        port=8000,
        tp_size=tp,
        max_model_len=max_len,
        dtype=dtype,
        gpu_memory_utilization=gpu_mem_util,
        trust_remote_code=True,
    )


def build_command(cfg: ServeConfig) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        cfg.model_id,
        "--host",
        cfg.host,
        "--port",
        str(cfg.port),
        "--tensor-parallel-size",
        str(cfg.tp_size),
        "--max-model-len",
        str(cfg.max_model_len),
        "--dtype",
        cfg.dtype,
        "--gpu-memory-utilization",
        str(cfg.gpu_memory_utilization),
    ]

    served_name = cfg.served_model_name or cfg.model_id
    cmd.extend(["--served-model-name", served_name])

    if cfg.trust_remote_code:
        cmd.append("--trust-remote-code")

    if cfg.extra_args:
        cmd.extend(shlex.split(cfg.extra_args))

    return cmd


def parse_args():
    parser = argparse.ArgumentParser(description="Auto-configured vLLM OpenAI server")
    parser.add_argument("--model-id", default=DEFAULT_MODEL, help="HF model to serve")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument("--served-model-name", default=os.getenv("SERVED_MODEL_NAME"))
    parser.add_argument("--trust-remote-code", action="store_true",
                        default=os.getenv("TRUST_REMOTE_CODE", "1") != "0")

    parser.add_argument("--tp-size", type=int, default=None,
                        help="Tensor parallel size (env: VLLM_TP_SIZE)")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Override max context (env: VLLM_MAX_MODEL_LEN)")
    parser.add_argument("--dtype", default=None, help="Override dtype, e.g. bfloat16 (env: VLLM_DTYPE)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=None,
                        help="Override GPU mem util (env: VLLM_GPU_MEMORY_UTILIZATION)")
    parser.add_argument("--extra-args", default=os.getenv("VLLM_EXTRA_ARGS", ""),
                        help="Additional args passed verbatim to vLLM")
    return parser.parse_args()


def main():
    args = parse_args()
    user_tp = args.tp_size or os.getenv("VLLM_TP_SIZE")
    user_tp = int(user_tp) if user_tp else None

    user_max_len = args.max_model_len or os.getenv("VLLM_MAX_MODEL_LEN")
    user_max_len = int(user_max_len) if user_max_len else None

    user_dtype = args.dtype or os.getenv("VLLM_DTYPE")

    user_gpu_mem_util = args.gpu_memory_utilization or os.getenv("VLLM_GPU_MEMORY_UTILIZATION")
    user_gpu_mem_util = float(user_gpu_mem_util) if user_gpu_mem_util else None

    cfg = recommend_settings(user_tp, user_max_len, user_dtype, user_gpu_mem_util)
    cfg.model_id = args.model_id
    cfg.host = args.host
    cfg.port = args.port
    cfg.served_model_name = args.served_model_name
    cfg.trust_remote_code = args.trust_remote_code
    cfg.extra_args = args.extra_args

    print("vLLM auto-configuration:")
    print(f"  GPUs detected: {detect_gpu_info()}")
    print(f"  model: {cfg.model_id}")
    print(f"  served name: {cfg.served_model_name or cfg.model_id}")
    print(f"  tp_size: {cfg.tp_size}")
    print(f"  max_model_len: {cfg.max_model_len}")
    print(f"  dtype: {cfg.dtype}")
    print(f"  gpu_memory_utilization: {cfg.gpu_memory_utilization}")
    if cfg.extra_args:
        print(f"  extra_args: {cfg.extra_args}")

    cmd = build_command(cfg)
    print("\nLaunching vLLM:")
    print(" ", " ".join(shlex.quote(arg) for arg in cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
