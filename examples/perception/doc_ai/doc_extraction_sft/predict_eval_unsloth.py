import argparse
import json
from pathlib import Path
from typing import Any

import unsloth  # noqa: F401  # Import first so Unsloth patches transformers before use.
import torch
import yaml
from transformers import StoppingCriteriaList
from unsloth import FastLanguageModel, FastModel

from predict_eval import BalancedJsonStoppingCriteria, json_is_balanced, prepare_eval_rows
from train import (
    apply_dataset_overrides,
    apply_chat_template,
    build_prompt_messages,
    build_target_response,
    model_supports_logits_to_keep,
)


def text_tokenizer(processor_or_tokenizer):
    return getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Training/eval config")
    parser.add_argument("--dataset-path", help="Override dataset.path from config, e.g. json or a HF dataset")
    parser.add_argument("--data-files", help="Override dataset.data_files from config for local JSON/JSONL files")
    parser.add_argument("--data-dir", help="Override dataset.data_dir from config")
    parser.add_argument("--adapter", required=True, help="LoRA adapter directory")
    parser.add_argument("--output", required=True, help="JSONL output path")
    parser.add_argument("--max-samples", type=int, default=64, help="Max eval rows to generate")
    parser.add_argument("--max-new-tokens", type=int, default=8192, help="Generation output budget")
    parser.add_argument("--max-target-tokens", type=int)
    parser.add_argument(
        "--sample-strategy",
        choices=["first", "document_id"],
        default="first",
        help="How to choose eval rows",
    )
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--no-stop-on-balanced-json", action="store_true")
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        help="Left-truncate prompt input IDs to this many tokens before generation",
    )
    parser.add_argument(
        "--disable-cache",
        action="store_true",
        help="Pass use_cache=False during generation",
    )
    parser.add_argument(
        "--attn-implementation",
        help="Optional attention implementation to pass to Unsloth model loading",
    )
    parser.add_argument(
        "--no-for-inference",
        action="store_true",
        help="Do not call FastLanguageModel.for_inference(model)",
    )
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        help="Use a manual greedy generation loop with chunked prompt prefill",
    )
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def forward_with_optional_logits_to_keep(model, **kwargs):
    if model_supports_logits_to_keep(model):
        kwargs["logits_to_keep"] = 1
    return model(**kwargs)


def generate_without_cache(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    stop_on_balanced_json: bool,
) -> torch.Tensor:
    current_ids = input_ids
    generated_ids = []
    eos_token_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        attention_mask = torch.ones_like(current_ids)
        outputs = forward_with_optional_logits_to_keep(
            model,
            input_ids=current_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated_ids.append(next_token)
        current_ids = torch.cat([current_ids, next_token], dim=-1)

        if eos_token_id is not None and int(next_token.item()) == eos_token_id:
            break

        if stop_on_balanced_json and len(generated_ids) % 16 == 0:
            text = tokenizer.decode(
                torch.cat(generated_ids, dim=-1)[0],
                skip_special_tokens=True,
            )
            if len(generated_ids) >= 64 and json_is_balanced(text):
                break

    if not generated_ids:
        return input_ids[:, :0]
    return torch.cat(generated_ids, dim=-1)


def generate_with_chunked_prefill(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    chunk_size: int,
    stop_on_balanced_json: bool,
) -> torch.Tensor:
    past_key_values = None
    outputs = None
    total_prompt_tokens = input_ids.shape[-1]

    for end in range(chunk_size, total_prompt_tokens + chunk_size, chunk_size):
        start = end - chunk_size
        end = min(end, total_prompt_tokens)
        chunk_ids = input_ids[:, start:end]
        outputs = forward_with_optional_logits_to_keep(
            model,
            input_ids=chunk_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        next_past_key_values = getattr(outputs, "past_key_values", None)
        if next_past_key_values is None:
            print(
                "Model did not return past_key_values; falling back to no-cache greedy generation.",
                flush=True,
            )
            return generate_without_cache(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                stop_on_balanced_json=stop_on_balanced_json,
            )
        past_key_values = next_past_key_values
        if end == total_prompt_tokens:
            break

    if outputs is None:
        raise ValueError("Cannot generate from an empty prompt")

    generated_ids = []
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    eos_token_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        generated_ids.append(next_token)
        if eos_token_id is not None and int(next_token.item()) == eos_token_id:
            break

        if stop_on_balanced_json and len(generated_ids) % 16 == 0:
            text = tokenizer.decode(
                torch.cat(generated_ids, dim=-1)[0],
                skip_special_tokens=True,
            )
            if len(generated_ids) >= 64 and json_is_balanced(text):
                break

        outputs = forward_with_optional_logits_to_keep(
            model,
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

    if not generated_ids:
        return input_ids[:, :0]
    return torch.cat(generated_ids, dim=-1)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = apply_dataset_overrides(config, args)
    model_config = config["model"]

    print(f"Loading Unsloth adapter for inference: {args.adapter}", flush=True)
    loader_kwargs = {
        "model_name": args.adapter,
        "max_seq_length": model_config["max_seq_length"],
        "dtype": None,
        "load_in_4bit": model_config.get("load_in_4bit", True),
    }
    if "trust_remote_code" in model_config:
        loader_kwargs["trust_remote_code"] = model_config["trust_remote_code"]
    if args.attn_implementation:
        loader_kwargs["attn_implementation"] = args.attn_implementation
    elif "attn_implementation" in model_config:
        loader_kwargs["attn_implementation"] = model_config["attn_implementation"]
    if "unsloth_force_compile" in model_config:
        loader_kwargs["unsloth_force_compile"] = model_config["unsloth_force_compile"]
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(**loader_kwargs)
        inference_cls = FastLanguageModel
    except Exception as first_error:
        print(
            "FastLanguageModel.from_pretrained failed; retrying with FastModel. "
            f"First error: {type(first_error).__name__}: {first_error}",
            flush=True,
        )
        model, tokenizer = FastModel.from_pretrained(**loader_kwargs)
        inference_cls = FastModel
    plain_tokenizer = text_tokenizer(tokenizer)
    if plain_tokenizer.pad_token is None:
        plain_tokenizer.pad_token = plain_tokenizer.eos_token
    model.config.use_cache = not args.disable_cache
    if not args.no_for_inference:
        inference_cls.for_inference(model)
    model.eval()

    rows = prepare_eval_rows(config, plain_tokenizer, args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = model_device(model)

    with output_path.open("w") as f:
        for index, row in enumerate(rows):
            messages = build_prompt_messages(row, config["dataset"])
            prompt = apply_chat_template(
                tokenizer,
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = plain_tokenizer(prompt, return_tensors="pt").to(device)
            original_prompt_tokens = inputs["input_ids"].shape[-1]
            if args.max_prompt_tokens and original_prompt_tokens > args.max_prompt_tokens:
                keep = args.max_prompt_tokens
                inputs = {
                    key: value[:, -keep:].contiguous()
                    for key, value in inputs.items()
                }
            target = build_target_response(row, config["dataset"])
            target_token_count = len(plain_tokenizer(target).input_ids)
            print(
                "Generating "
                f"{index + 1}/{len(rows)}: "
                f"prompt_tokens={inputs['input_ids'].shape[-1]} "
                f"original_prompt_tokens={original_prompt_tokens} "
                f"target_tokens={target_token_count} "
                f"max_new_tokens={args.max_new_tokens}",
                flush=True,
            )
            stopping_criteria = None
            if not args.no_stop_on_balanced_json:
                stopping_criteria = StoppingCriteriaList(
                    [
                        BalancedJsonStoppingCriteria(
                            tokenizer=plain_tokenizer,
                            prompt_length=inputs["input_ids"].shape[-1],
                        )
                    ]
                )
            with torch.no_grad():
                if args.prefill_chunk_size:
                    generated = generate_with_chunked_prefill(
                        model=model,
                        tokenizer=plain_tokenizer,
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=args.max_new_tokens,
                        chunk_size=args.prefill_chunk_size,
                        stop_on_balanced_json=not args.no_stop_on_balanced_json,
                    )[0]
                else:
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        eos_token_id=plain_tokenizer.eos_token_id,
                        pad_token_id=plain_tokenizer.eos_token_id,
                        stopping_criteria=stopping_criteria,
                        use_cache=not args.disable_cache,
                    )
                    generated = outputs[0][inputs["input_ids"].shape[-1] :]
            prediction = plain_tokenizer.decode(generated, skip_special_tokens=True).strip()
            record = {
                "index": index,
                "row_id": row.get("row_id") or row.get("id") or index,
                "target_source": row.get("target_source"),
                "prediction": prediction,
                "target": target,
                "prompt_tokens": int(inputs["input_ids"].shape[-1]),
                "original_prompt_tokens": int(original_prompt_tokens),
                "target_tokens": int(target_token_count),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print(
                f"Wrote prediction {index + 1}/{len(rows)}: "
                f"generated_tokens={generated.shape[-1]}",
                flush=True,
            )


if __name__ == "__main__":
    main()
