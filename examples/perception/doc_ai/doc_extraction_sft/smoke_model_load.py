import argparse
from typing import Any

import torch
import yaml
from transformers import BitsAndBytesConfig

from train import apply_chat_template, load_base_model, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Training config to validate")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_config = config["model"]
    model_id = model_config["name"]
    print(f"config_ok model={model_id} max_seq_length={model_config.get('max_seq_length')}")

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

    tokenizer = load_tokenizer(model_config)
    print(f"tokenizer_ok class={tokenizer.__class__.__name__} eos={tokenizer.eos_token_id}")

    model = load_base_model(model_config, model_load_kwargs)
    model.config.use_cache = True
    model.eval()
    print(
        f"model_ok class={model.__class__.__name__} "
        f"device={next(model.parameters()).device} type={getattr(model.config, 'model_type', None)}"
    )

    messages = [{"role": "user", "content": 'Return JSON only: {"ok": true}'}]
    prompt = apply_chat_template(
        tokenizer,
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    ).strip()
    print(f"generated={generated}")


if __name__ == "__main__":
    main()
