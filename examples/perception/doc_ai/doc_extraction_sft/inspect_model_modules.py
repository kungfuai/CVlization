import argparse

import torch
import yaml
from transformers import BitsAndBytesConfig

from train import load_base_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--contains", action="append", default=[])
    parser.add_argument("--limit", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    model_config = config["model"]
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=model_config.get("load_in_4bit", True),
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model_load_kwargs = {
        "quantization_config": quantization_config,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if model_config.get("attn_implementation"):
        model_load_kwargs["attn_implementation"] = model_config["attn_implementation"]
    model = load_base_model(model_config, model_load_kwargs)

    needles = args.contains or [""]
    count = 0
    for name, module in model.named_modules():
        if not any(needle in name for needle in needles):
            continue
        print(name, type(module).__name__)
        count += 1
        if count >= args.limit:
            break


if __name__ == "__main__":
    main()
