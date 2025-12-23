#!/usr/bin/env python3
"""
Kandinsky 5.0 - Video and Image Generation

A lightweight 2B parameter model from Sber AI for T2V, I2V, and T2I generation.
Runs on 12GB+ VRAM with optimizations.

License: MIT
"""
import argparse
import os
import sys
import time
import warnings
import logging

import torch

from cvlization.paths import resolve_input_path, resolve_output_path

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger("torch").setLevel(logging.ERROR)

# Config mapping
CONFIG_MAP = {
    "t2v_sft_5s": "configs/k5_lite_t2v_5s_sft_sd.yaml",
    "t2v_sft_10s": "configs/k5_lite_t2v_10s_sft_sd.yaml",
    "t2v_distilled_5s": "configs/k5_lite_t2v_5s_distil_sd.yaml",
    "t2v_distilled_10s": "configs/k5_lite_t2v_10s_distil_sd.yaml",
    "t2v_nocfg_5s": "configs/k5_lite_t2v_5s_nocfg_sd.yaml",
    "t2v_nocfg_10s": "configs/k5_lite_t2v_10s_nocfg_sd.yaml",
    "i2v_5s": "configs/k5_lite_i2v_5s_sft_sd.yaml",
    "t2i": "configs/k5_lite_t2i_sft_hd.yaml",
}


def main():
    parser = argparse.ArgumentParser(description="Kandinsky 5.0 generation")

    parser.add_argument("--prompt", "-p", type=str, required=True,
                        help="Text prompt for generation")
    parser.add_argument("--image", type=str, default=None,
                        help="Input image for I2V mode")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file path")
    parser.add_argument("--mode", type=str, default="t2v",
                        choices=["t2v", "i2v", "t2i"],
                        help="Generation mode")
    parser.add_argument("--config", type=str, default="sft",
                        choices=["sft", "distilled", "nocfg"],
                        help="Model variant")
    parser.add_argument("--duration", type=int, default=5,
                        choices=[5, 10],
                        help="Video duration in seconds")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)

    # Optimization flags
    parser.add_argument("--offload", action="store_true",
                        help="Enable CPU offloading for low VRAM")
    parser.add_argument("--qwen_quantization", action="store_true",
                        help="Use 4-bit quantized Qwen encoder")
    parser.add_argument("--magcache", action="store_true",
                        help="Enable MagCache for faster inference")

    args = parser.parse_args()

    # Resolve input paths
    if args.image:
        args.image = resolve_input_path(args.image)

    # Determine config path
    if args.mode == "t2i":
        config_key = "t2i"
    elif args.mode == "i2v":
        config_key = "i2v_5s"
    else:
        config_key = f"t2v_{args.config}_{args.duration}s"

    config_path = CONFIG_MAP.get(config_key)
    if not config_path:
        print(f"Error: Unknown config {config_key}")
        sys.exit(1)

    # Set output path
    if args.output is None:
        ext = "png" if args.mode == "t2i" else "mp4"
        args.output = f"outputs/output.{ext}"

    # Resolve output path for CVL mode
    output_path = resolve_output_path(args.output)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Print info
    print("=" * 60)
    print("Kandinsky 5.0 Generation")
    print("=" * 60)
    print(f"Mode: {args.mode.upper()}")
    print(f"Config: {config_key}")
    print(f"Prompt: {args.prompt[:60]}...")
    print(f"Output: {output_path}")
    if args.offload:
        print("Offloading: Enabled")
    if args.qwen_quantization:
        print("Qwen Quantization: Enabled (4-bit)")
    print("=" * 60)

    # Import pipelines and ensure models are downloaded
    from kandinsky import get_T2V_pipeline, get_I2V_pipeline, get_T2I_pipeline, ensure_models_downloaded

    # Download models if needed
    ensure_models_downloaded(config_path)

    device_map = {
        "dit": torch.device("cuda:0"),
        "vae": torch.device("cuda:0"),
        "text_embedder": torch.device("cuda:0"),
    }

    # Load pipeline
    start_time = time.perf_counter()

    if args.mode == "t2i":
        pipe = get_T2I_pipeline(
            device_map=device_map,
            conf_path=config_path,
            offload=args.offload,
            magcache=args.magcache,
            quantized_qwen=args.qwen_quantization,
            attention_engine="sdpa",
        )
        result = pipe(
            args.prompt,
            width=args.width,
            height=args.height,
            save_path=output_path,
            seed=args.seed,
        )
    elif args.mode == "i2v":
        if not args.image:
            print("Error: --image required for I2V mode")
            sys.exit(1)
        pipe = get_I2V_pipeline(
            device_map=device_map,
            conf_path=config_path,
            offload=args.offload,
            magcache=args.magcache,
            quantized_qwen=args.qwen_quantization,
            attention_engine="sdpa",
        )
        result = pipe(
            args.prompt,
            image=args.image,
            time_length=args.duration,
            save_path=output_path,
            seed=args.seed,
        )
    else:  # t2v
        pipe = get_T2V_pipeline(
            device_map=device_map,
            conf_path=config_path,
            offload=args.offload,
            magcache=args.magcache,
            quantized_qwen=args.qwen_quantization,
            attention_engine="sdpa",
        )
        result = pipe(
            args.prompt,
            time_length=args.duration,
            width=args.width,
            height=args.height,
            save_path=output_path,
            seed=args.seed,
        )

    elapsed = time.perf_counter() - start_time
    print(f"\nGeneration complete in {elapsed:.1f}s")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
