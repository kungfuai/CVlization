import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import PeftModel
from transformers import (
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

from train import (
    apply_dataset_overrides,
    apply_chat_template,
    build_prompt_messages,
    build_target_response,
    filter_dataset,
    load_base_model,
    load_tokenizer,
    load_training_dataset,
    split_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Training config used for the run")
    parser.add_argument("--dataset-path", help="Override dataset.path from config, e.g. json or a HF dataset")
    parser.add_argument("--data-files", help="Override dataset.data_files from config for local JSON/JSONL files")
    parser.add_argument("--data-dir", help="Override dataset.data_dir from config")
    parser.add_argument(
        "--model-name",
        help=(
            "Override model.name from the config. Use this for a base model, "
            "a full merged checkpoint directory, or a Hugging Face model id."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        help=(
            "Alias for --model-name. This is for base/full checkpoints. "
            "Use --adapter for a LoRA adapter on top of the config/base model."
        ),
    )
    parser.add_argument(
        "--adapter",
        help="Optional LoRA adapter directory. Omit to evaluate the base model from the config.",
    )
    parser.add_argument("--attn-implementation", help="Override model.attn_implementation")
    parser.add_argument("--max-seq-length", type=int, help="Override model.max_seq_length")
    quantization_group = parser.add_mutually_exclusive_group()
    quantization_group.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Force 4-bit loading regardless of config.",
    )
    quantization_group.add_argument(
        "--no-load-in-4bit",
        action="store_true",
        help="Disable 4-bit loading regardless of config.",
    )
    parser.add_argument("--output", required=True, help="JSONL output path")
    parser.add_argument("--max-samples", type=int, default=64, help="Max eval rows to generate")
    parser.add_argument("--max-new-tokens", type=int, default=8192, help="Generation output budget")
    parser.add_argument(
        "--max-target-tokens",
        type=int,
        help="Only evaluate rows whose target token count is at or below this value",
    )
    parser.add_argument(
        "--sample-strategy",
        choices=["first", "document_id"],
        default="first",
        help="How to choose eval rows",
    )
    parser.add_argument(
        "--sample-offset",
        type=int,
        default=0,
        help="Start offset within the selected eval-row order before applying --max-samples",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=1,
        help="Stride within the selected eval-row order. Useful for multi-GPU eval sharding.",
    )
    parser.add_argument(
        "--no-stop-on-balanced-json",
        action="store_true",
        help="Disable early stopping when a balanced JSON object/array is generated",
    )
    parser.add_argument(
        "--no-truncate-prompt",
        action="store_true",
        help=(
            "Disable left-truncation of prompts to model.max_seq_length - max_new_tokens. "
            "By default eval mirrors assistant-only training, which keeps the suffix of long prompts."
        ),
    )
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_model_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    model_config = dict(config["model"])
    model_name = args.checkpoint or args.model_name
    if model_name:
        model_config["name"] = model_name
    if args.attn_implementation:
        model_config["attn_implementation"] = args.attn_implementation
    if args.max_seq_length:
        model_config["max_seq_length"] = args.max_seq_length
    if args.load_in_4bit:
        model_config["load_in_4bit"] = True
    if args.no_load_in_4bit:
        model_config["load_in_4bit"] = False
    config = dict(config)
    config["model"] = model_config
    return config


def maybe_json_loads(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def json_is_balanced(text: str) -> bool:
    text = text.strip()
    if not text or text[0] not in "[{":
        return False

    stack = []
    in_string = False
    escape = False
    opening = {"{": "}", "[": "]"}
    closing = {"}", "]"}

    for char in text:
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char in opening:
            stack.append(opening[char])
        elif char in closing:
            if not stack or stack.pop() != char:
                return False

    return not stack and not in_string


class BalancedJsonStoppingCriteria(StoppingCriteria):
    def __init__(
        self,
        tokenizer,
        prompt_length: int,
        min_new_tokens: int = 64,
        check_every_tokens: int = 16,
    ):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.min_new_tokens = min_new_tokens
        self.check_every_tokens = check_every_tokens

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        generated_length = input_ids.shape[-1] - self.prompt_length
        if generated_length < self.min_new_tokens:
            return False
        if generated_length % self.check_every_tokens != 0:
            return False

        text = self.tokenizer.decode(
            input_ids[0, self.prompt_length :],
            skip_special_tokens=True,
        )
        return json_is_balanced(text)


def prepare_eval_rows(config: dict[str, Any], tokenizer, args: argparse.Namespace):
    dataset_config = config["dataset"]
    training_config = config["training"]

    dataset = load_training_dataset(dataset_config)
    dataset = filter_dataset(dataset, dataset_config)
    _, eval_dataset = split_dataset(dataset, dataset_config, {**training_config, "do_eval": True})

    if args.max_target_tokens:
        kept_indices = []
        for index, row in enumerate(eval_dataset):
            target = build_target_response(row, dataset_config)
            if len(tokenizer(target).input_ids) <= args.max_target_tokens:
                kept_indices.append(index)
        print(
            f"Filtered eval rows by target_tokens <= {args.max_target_tokens}: "
            f"{len(kept_indices)}/{len(eval_dataset)} rows"
        )
        eval_dataset = eval_dataset.select(kept_indices)

    if args.sample_strategy == "document_id" and "document_id" in eval_dataset.column_names:
        seen_document_ids = set()
        selected_indices = []
        for index, document_id in enumerate(eval_dataset["document_id"]):
            document_id = str(document_id)
            if document_id in seen_document_ids:
                continue
            seen_document_ids.add(document_id)
            selected_indices.append(index)

        selected_indices = selected_indices[args.sample_offset :: args.sample_stride][
            : args.max_samples
        ]
        print(
            f"Selected {len(selected_indices)} eval rows from distinct document_id groups "
            f"(offset={args.sample_offset}, stride={args.sample_stride})"
        )
        return eval_dataset.select(selected_indices)

    selected_indices = list(range(len(eval_dataset)))
    selected_indices = selected_indices[args.sample_offset :: args.sample_stride][: args.max_samples]
    return eval_dataset.select(selected_indices)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = apply_dataset_overrides(config, args)
    config = apply_model_overrides(config, args)
    model_config = config["model"]

    model_load_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if model_config.get("load_in_4bit", True):
        model_load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    if model_config.get("attn_implementation"):
        model_load_kwargs["attn_implementation"] = model_config["attn_implementation"]

    print(f"Loading base/full model checkpoint: {model_config['name']}", flush=True)
    tokenizer = load_tokenizer(model_config)

    model = load_base_model(model_config, model_load_kwargs)
    if args.adapter:
        print(f"Loading LoRA adapter: {args.adapter}", flush=True)
        model = PeftModel.from_pretrained(model, args.adapter)
    else:
        print("No LoRA adapter provided; evaluating the base model.", flush=True)
    model.config.use_cache = True
    model.eval()

    rows = prepare_eval_rows(config, tokenizer, args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        for index, row in enumerate(rows):
            messages = build_prompt_messages(row, config["dataset"])
            prompt = apply_chat_template(
                tokenizer,
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            target = build_target_response(row, config["dataset"])
            target_token_count = len(tokenizer(target).input_ids)
            original_prompt_tokens = inputs["input_ids"].shape[-1]
            kept_prompt_tokens = original_prompt_tokens
            max_seq_length = model_config.get("max_seq_length")
            if max_seq_length and not args.no_truncate_prompt:
                prompt_budget = max_seq_length - args.max_new_tokens
                if prompt_budget <= 0:
                    raise ValueError(
                        "model.max_seq_length must be larger than --max-new-tokens "
                        f"for prompt truncation: {max_seq_length=} "
                        f"max_new_tokens={args.max_new_tokens}"
                    )
                if original_prompt_tokens > prompt_budget:
                    inputs = {
                        key: value[:, -prompt_budget:].contiguous()
                        for key, value in inputs.items()
                    }
                    kept_prompt_tokens = prompt_budget
            print(
                "Generating "
                f"{index + 1}/{len(rows)}: "
                f"prompt_tokens={original_prompt_tokens} "
                f"kept_prompt_tokens={kept_prompt_tokens} "
                f"target_tokens={target_token_count} "
                f"max_new_tokens={args.max_new_tokens}",
                flush=True,
            )
            with torch.no_grad():
                stopping_criteria = None
                if not args.no_stop_on_balanced_json:
                    stopping_criteria = StoppingCriteriaList(
                        [
                            BalancedJsonStoppingCriteria(
                                tokenizer=tokenizer,
                                prompt_length=inputs["input_ids"].shape[-1],
                            )
                        ]
                    )
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria=stopping_criteria,
                )

            generated = outputs[0][inputs["input_ids"].shape[-1] :]
            prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()

            record = {
                "index": index,
                "row_id": row.get("row_id") or row.get("id") or index,
                "target_source": row.get("target_source"),
                "prediction": prediction,
                "target": target,
                "prompt_tokens": original_prompt_tokens,
                "kept_prompt_tokens": kept_prompt_tokens,
                "target_tokens": target_token_count,
                "max_new_tokens": args.max_new_tokens,
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
