#!/usr/bin/env python
"""Mixtral 8x7B inference with HQQ quantization and layer offloading.

Uses the mixtral-offloading library to fit Mixtral 8x7B into 10-20GB VRAM
by offloading expert layers between GPU and CPU/disk.
"""

import argparse
import logging
import sys
from pathlib import Path
from glob import glob

import torch
from hqq.core.quantize import BaseQuantizeConfig
from transformers import AutoConfig, AutoTokenizer, TextStreamer

sys.path.append("/opt/mixtral-offloading")
from src.build_model import OffloadConfig, QuantConfig, build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Mixtral 8x7B with offloading")
    parser.add_argument("--prompt", default="Explain what a mixture-of-experts model is.", help="Prompt text")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--output_file", type=Path, default=Path("outputs/mixtral-output.txt"))
    parser.add_argument("--offload_per_layer", type=int, default=4,
                        help="Experts to offload per layer (0=20.4GB, 1=18GB, 2=16.9GB, 3=14.1GB, 4=12.2GB, 5=10.2GB)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Download and setup model paths
    model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    quantized_model_name = "lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo"

    LOGGER.info(f"Downloading pre-quantized model: {quantized_model_name}")
    from subprocess import run
    run(f"huggingface-cli download {quantized_model_name}".split())

    # Find downloaded model path
    pretrain_path = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{quantized_model_name.replace('/', '--')}"
    if not pretrain_path.exists():
        raise FileNotFoundError(f"Model not found at {pretrain_path}")

    snapshots = glob(str(pretrain_path / "snapshots" / "*" / "config.json"))
    if not snapshots:
        raise FileNotFoundError(f"No model snapshots found in {pretrain_path}")

    local_model_dir = Path(snapshots[0]).parent
    config = AutoConfig.from_pretrained(local_model_dir)

    # Setup offloading configuration
    device = torch.device("cuda:0")
    num_experts = config.num_local_experts

    LOGGER.info(f"Configuring offloading: {args.offload_per_layer} experts per layer")
    LOGGER.info(f"Expected VRAM usage: ~{[20.4, 18, 16.9, 14.1, 12.2, 10.2][min(args.offload_per_layer, 5)]}GB")

    offload_config = OffloadConfig(
        main_size=config.num_hidden_layers * (num_experts - args.offload_per_layer),
        offload_size=config.num_hidden_layers * args.offload_per_layer,
        buffer_size=4,
        offload_per_layer=args.offload_per_layer,
    )

    # Setup quantization: 4-bit attention, 2-bit FFN
    attn_config = BaseQuantizeConfig(
        nbits=4,
        group_size=64,
        quant_zero=True,
        quant_scale=True,
    )
    attn_config["scale_quant_params"]["group_size"] = 256

    ffn_config = BaseQuantizeConfig(
        nbits=2,
        group_size=16,
        quant_zero=True,
        quant_scale=True,
    )
    quant_config = QuantConfig(ffn_config=ffn_config, attn_config=attn_config)

    # Build model with offloading
    LOGGER.info("Building model with HQQ quantization and offloading...")
    model = build_model(
        device=device,
        quant_config=quant_config,
        offload_config=offload_config,
        state_path=local_model_dir,
    )

    # Load tokenizer
    LOGGER.info(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # Generate
    LOGGER.info(f"Generating text with prompt: {args.prompt[:50]}...")
    user_entry = dict(role="user", content=args.prompt)
    input_ids = tokenizer.apply_chat_template([user_entry], return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    result = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        streamer=streamer,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Save output
    output_text = tokenizer.decode(result[0], skip_special_tokens=True)
    # Remove the prompt from output
    output_text = output_text[len(args.prompt):].strip()

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(output_text + "\n")
    LOGGER.info(f"Saved output to {args.output_file}")

    print("\n=== Prompt ===")
    print(args.prompt)
    print("\n=== Output ===")
    print(output_text)


if __name__ == "__main__":
    main()
