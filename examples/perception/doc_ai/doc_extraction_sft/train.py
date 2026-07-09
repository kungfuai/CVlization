import argparse
import inspect
import json
import os
import random
from typing import Any, TypedDict

import torch
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune an instruction LLM for document extraction SFT.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Config skeleton:
  dataset:
    path: "json"
    data_files: "/path/to/train.jsonl"
    split: "train"
    target_column: "target_json"
    target_sources: ["ground_truth", "form_identification"]
    split_strategy: "document_id"
  model:
    name: "Qwen/Qwen3-8B"
    max_seq_length: 32768
    max_target_length: 8192
  lora:
    r: 32
    alpha: 64
    dropout: 0.05
    target_modules: "all-linear"
  training:
    output_dir: "outputs/qwen3_8b"
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 8
    assistant_only_loss: true

You may keep dataset.data_files in the config, override it with --data-files,
or use DOC_EXTRACTION_SFT_TRAIN_JSONL in shared configs.
""",
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--dataset-path", help="Override dataset.path from config, e.g. json or a HF dataset")
    parser.add_argument("--data-files", help="Override dataset.data_files from config for local JSON/JSONL files")
    parser.add_argument("--data-dir", help="Override dataset.data_dir from config")
    parser.add_argument(
        "--resume-from-checkpoint",
        help="Optional Trainer checkpoint path to resume optimizer/scheduler/trainer state",
    )
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    print(f"Loading configuration from {path}...")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_dataset_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    dataset_config = dict(config["dataset"])
    if getattr(args, "dataset_path", None):
        dataset_config["path"] = args.dataset_path
    if getattr(args, "data_files", None):
        dataset_config["data_files"] = args.data_files
    if getattr(args, "data_dir", None):
        dataset_config["data_dir"] = args.data_dir
    config = dict(config)
    config["dataset"] = dataset_config
    return config


def expand_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if "$" in expanded:
            raise ValueError(
                f"Unexpanded environment variable in config value {value!r}. "
                "Set the variable or replace it with an explicit value."
            )
        return expanded
    if isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_env_vars(item) for key, item in value.items()}
    return value


def load_training_dataset(dataset_config: dict[str, Any]):
    dataset_path = dataset_config["path"]
    split = dataset_config.get("split", "train")
    load_kwargs: dict[str, Any] = {}
    for key in ("name", "data_files", "data_dir", "revision"):
        if key in dataset_config:
            load_kwargs[key] = expand_env_vars(dataset_config[key])

    print(f"Loading dataset: {dataset_path}...")
    return load_dataset(dataset_path, split=split, **load_kwargs)


def filter_supported_kwargs(callable_obj, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(callable_obj)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def apply_chat_template(tokenizer, messages: list[dict[str, str]], **kwargs) -> str:
    template_kwargs = {"enable_thinking": False, **kwargs}
    try:
        return tokenizer.apply_chat_template(messages, **template_kwargs)
    except TypeError:
        template_kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **template_kwargs)


def load_tokenizer(model_config: dict[str, Any]):
    tokenizer_kwargs = {"trust_remote_code": True}
    if "tokenizer_use_fast" in model_config:
        tokenizer_kwargs["use_fast"] = model_config["tokenizer_use_fast"]
    if "tokenizer_extra_special_tokens" in model_config:
        tokenizer_kwargs["extra_special_tokens"] = model_config["tokenizer_extra_special_tokens"]

    tokenizer = AutoTokenizer.from_pretrained(model_config["name"], **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def auto_model_class(model_config: dict[str, Any]):
    class_name = model_config.get("auto_model_class", "causal_lm")
    if class_name == "causal_lm":
        return AutoModelForCausalLM
    if class_name == "image_text_to_text":
        import transformers

        return transformers.AutoModelForImageTextToText
    raise ValueError(f"Unsupported model.auto_model_class={class_name!r}")


def load_base_model(model_config: dict[str, Any], model_load_kwargs: dict[str, Any]):
    patch_transformers_remote_code_compat()
    return auto_model_class(model_config).from_pretrained(
        model_config["name"],
        **model_load_kwargs,
    )


def patch_transformers_remote_code_compat() -> None:
    import transformers.utils as transformers_utils

    try:
        getattr(transformers_utils, "LossKwargs")
    except (AttributeError, NameError):
        class LossKwargs(TypedDict, total=False):
            labels: Any

        transformers_utils.LossKwargs = LossKwargs


def model_supports_logits_to_keep(model) -> bool:
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    try:
        return "logits_to_keep" in inspect.signature(base_model.forward).parameters
    except (TypeError, ValueError):
        return False


def torch_dtype_from_config(value: str | None):
    if value is None:
        return None
    if value == "auto":
        return "auto"
    if value == "bfloat16":
        return torch.bfloat16
    if value == "float16":
        return torch.float16
    if value == "float32":
        return torch.float32
    raise ValueError(f"Unsupported model.torch_dtype={value!r}")


def build_quantization_config(model_config: dict[str, Any]):
    quantization = model_config.get("quantization")
    if quantization == "mxfp4":
        from transformers import Mxfp4Config

        return Mxfp4Config(dequantize=model_config.get("mxfp4_dequantize", True))

    if quantization not in (None, "bnb_4bit"):
        raise ValueError(f"Unsupported model.quantization={quantization!r}")

    if not model_config.get("load_in_4bit", True):
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def chunked_cross_entropy(
    logits: torch.Tensor,
    shift_labels: torch.Tensor,
    ignore_index: int = -100,
    chunk_tokens: int = 512,
) -> torch.Tensor:
    vocab_size = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab_size)
    flat_labels = shift_labels.to(logits.device).reshape(-1)
    token_count = flat_labels.ne(ignore_index).sum().clamp_min(1)
    loss_sum = torch.zeros((), device=logits.device, dtype=torch.float32)

    for start in range(0, flat_logits.shape[0], chunk_tokens):
        end = min(start + chunk_tokens, flat_logits.shape[0])
        labels_chunk = flat_labels[start:end]
        if labels_chunk.ne(ignore_index).any():
            loss_sum = loss_sum + F.cross_entropy(
                flat_logits[start:end].float(),
                labels_chunk,
                ignore_index=ignore_index,
                reduction="sum",
            )

    return loss_sum / token_count


class AssistantOnlyTrainer(Trainer):
    def __init__(self, *args, use_suffix_logits: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_suffix_logits = use_suffix_logits and model_supports_logits_to_keep(self.model)
        if self.use_suffix_logits:
            print("Using suffix-only logits for assistant-only loss.")
        elif use_suffix_logits:
            print("Model does not expose logits_to_keep; using standard full-sequence logits.")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        if not self.use_suffix_logits or labels is None:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        supervised = labels.ne(-100)
        if not supervised.any():
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        first_supervised = torch.argmax(supervised.int(), dim=1)
        start = max(0, int(first_supervised.min().item()) - 1)
        logits_to_keep = labels.shape[-1] - start

        shift_labels = F.pad(labels, (0, 1), value=-100)[..., 1:]
        shift_labels = shift_labels[:, -logits_to_keep:].contiguous()

        model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
        outputs = model(**model_inputs, logits_to_keep=logits_to_keep)
        loss = chunked_cross_entropy(
            outputs.logits,
            shift_labels,
            chunk_tokens=512,
        )

        return (loss, outputs) if return_outputs else loss


def maybe_json_loads(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def clean_messages(messages: Any) -> list[dict[str, str]]:
    parsed = maybe_json_loads(messages)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected messages to be a list, got {type(parsed).__name__}")

    cleaned = []
    for message in parsed:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role in {"system", "user"} and content:
            cleaned.append({"role": role, "content": str(content)})
    if not cleaned:
        raise ValueError("No system/user messages found in row")
    return cleaned


def format_json_response(value: Any, pretty: bool) -> str:
    parsed = maybe_json_loads(value)
    if isinstance(parsed, (dict, list)):
        if pretty:
            return json.dumps(parsed, ensure_ascii=False, indent=2, sort_keys=True)
        return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return str(value)


FIELD_METADATA_KEYS = {"comparator", "description", "normalizer", "page", "path"}


def strip_ground_truth_field_metadata(value: Any) -> Any:
    if isinstance(value, list):
        return [strip_ground_truth_field_metadata(item) for item in value]

    if not isinstance(value, dict):
        return value

    if "value" in value and FIELD_METADATA_KEYS.intersection(value):
        return strip_ground_truth_field_metadata(value["value"])

    return {
        key: strip_ground_truth_field_metadata(child)
        for key, child in value.items()
        if key not in FIELD_METADATA_KEYS
    }


def strip_ground_truth_schema_metadata(value: Any) -> Any:
    if isinstance(value, list):
        return [strip_ground_truth_schema_metadata(item) for item in value]

    if not isinstance(value, dict):
        return value

    properties = value.get("properties")
    if isinstance(properties, dict) and "value" in properties and FIELD_METADATA_KEYS.intersection(properties):
        value_schema = strip_ground_truth_schema_metadata(properties["value"])
        if isinstance(value_schema, dict):
            return value_schema
        return {"type": ["string", "null"]}

    return {key: strip_ground_truth_schema_metadata(child) for key, child in value.items()}


def replace_schema_block(content: str, schema_text: str) -> str:
    xml_start_token = "<Schema>"
    xml_end_token = "</Schema>"
    xml_start = content.find(xml_start_token)
    xml_end = content.find(xml_end_token, xml_start + len(xml_start_token))
    if xml_start != -1 and xml_end != -1:
        schema_start = xml_start + len(xml_start_token)
        return (
            content[:schema_start]
            + "\n"
            + schema_text
            + "\n"
            + content[xml_end:]
        )

    marker = "SCHEMA:"
    marker_index = content.find(marker)
    if marker_index == -1:
        return content

    schema_start = marker_index + len(marker)
    end_candidates = [
        pos
        for token in ("\n\nOCR TEXT:", "\n\nOCR:", "\nOCR TEXT:", "\nOCR:")
        if (pos := content.find(token, schema_start)) != -1
    ]
    if not end_candidates:
        return content

    schema_end = min(end_candidates)
    return content[:schema_start] + "\n" + schema_text + content[schema_end:]


def build_prompt_messages(row: dict[str, Any], dataset_config: dict[str, Any] | None = None) -> list[dict[str, str]]:
    dataset_config = dataset_config or {}
    messages = clean_messages(row["messages"])
    ground_truth_target_format = dataset_config.get("ground_truth_target_format", "value_only")
    transform_schema = dataset_config.get("transform_ground_truth_schema", True)
    if (
        row.get("target_source") != "ground_truth"
        or ground_truth_target_format != "value_only"
        or not transform_schema
    ):
        return messages

    schema = maybe_json_loads(row.get("schema"))
    if not isinstance(schema, dict):
        return messages

    schema = strip_ground_truth_schema_metadata(schema)
    schema_text = json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True)
    return [
        {**message, "content": replace_schema_block(message["content"], schema_text)}
        for message in messages
    ]


def resolve_target_value(row: dict[str, Any], dataset_config: dict[str, Any]) -> Any:
    target_column = dataset_config.get("target_column", "target_json")
    if target_column in row and row[target_column] is not None:
        return row[target_column]
    if target_column != "target_json" and row.get("target_json") is not None:
        return row["target_json"]
    if row.get("target_text") is not None:
        return row["target_text"]
    raise ValueError(
        f"Missing target column {target_column!r}. "
        "Set dataset.target_column or provide target_json/target_text."
    )


def build_target_response(row: dict[str, Any], dataset_config: dict[str, Any]) -> str:
    pretty_json = dataset_config.get("pretty_json", False)
    parsed = maybe_json_loads(resolve_target_value(row, dataset_config))

    ground_truth_target_format = dataset_config.get("ground_truth_target_format", "value_only")
    if row.get("target_source") == "ground_truth" and ground_truth_target_format == "value_only":
        parsed = strip_ground_truth_field_metadata(parsed)

    return format_json_response(parsed, pretty=pretty_json)


def teacher_trace_is_clean(row: dict[str, Any], teacher_target_sources: set[str]) -> bool:
    if row.get("target_source") not in teacher_target_sources:
        return True

    metadata = maybe_json_loads(row.get("metadata", "{}"))
    if not isinstance(metadata, dict):
        return False

    trace_quality = metadata.get("trace_quality", {})
    issues = trace_quality.get("issues", []) if isinstance(trace_quality, dict) else []
    return not issues


def filter_dataset(dataset, dataset_config: dict[str, Any]):
    target_sources = set(dataset_config.get("target_sources") or ["ground_truth"])
    keep_only_clean_teacher_traces = dataset_config.get("keep_only_clean_teacher_traces", True)
    teacher_target_sources = set(dataset_config.get("teacher_target_sources") or ["teacher"])

    print(f"Filtering target_source in {sorted(target_sources)}...")
    dataset = dataset.filter(lambda row: row.get("target_source") in target_sources)

    if keep_only_clean_teacher_traces:
        print("Filtering low-quality teacher traces...")
        dataset = dataset.filter(
            lambda row: teacher_trace_is_clean(row, teacher_target_sources)
        )

    if "max_samples" in dataset_config:
        max_samples = dataset_config["max_samples"]
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"Limited dataset to {len(dataset)} samples")

    return dataset


def split_dataset(dataset, dataset_config: dict[str, Any], training_config: dict[str, Any]):
    if not training_config.get("do_eval", False):
        return dataset, None

    split_strategy = dataset_config.get("split_strategy", "document_id")
    eval_split_ratio = training_config.get("eval_split_ratio", 0.1)
    seed = training_config["seed"]

    if split_strategy == "random":
        split = dataset.train_test_split(test_size=eval_split_ratio, seed=seed)
        return split["train"], split["test"]

    if split_strategy not in dataset.column_names:
        raise ValueError(
            f"split_strategy={split_strategy!r} is not a dataset column. "
            f"Available columns: {dataset.column_names}"
        )

    group_ids = sorted({str(value) for value in dataset[split_strategy]})
    rng = random.Random(seed)
    rng.shuffle(group_ids)
    eval_group_count = max(1, round(len(group_ids) * eval_split_ratio))
    eval_groups = set(group_ids[:eval_group_count])

    train_dataset = dataset.filter(lambda row: str(row[split_strategy]) not in eval_groups)
    eval_dataset = dataset.filter(lambda row: str(row[split_strategy]) in eval_groups)
    print(
        f"Split by {split_strategy}: {len(train_dataset)} train rows, "
        f"{len(eval_dataset)} eval rows, {eval_group_count}/{len(group_ids)} eval groups"
    )
    return train_dataset, eval_dataset


def prepare_text_dataset(dataset, tokenizer, dataset_config: dict[str, Any]):
    def format_batch(examples):
        texts = []
        for i in range(len(examples["messages"])):
            row = {key: values[i] for key, values in examples.items()}
            messages = build_prompt_messages(row, dataset_config)
            messages.append(
                {
                    "role": "assistant",
                    "content": build_target_response(row, dataset_config),
                }
            )
            texts.append(
                apply_chat_template(
                    tokenizer,
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
        return {"text": texts}

    print("Formatting chat SFT examples...")
    return dataset.map(format_batch, batched=True, remove_columns=dataset.column_names)


def tokenize_assistant_only_dataset(
    dataset,
    tokenizer,
    dataset_config: dict[str, Any],
    max_seq_length: int,
    max_target_length: int | None = None,
):
    def tokenize_batch(examples):
        input_ids_batch = []
        attention_mask_batch = []
        labels_batch = []
        prompt_token_counts = []
        target_token_counts = []
        kept_prompt_token_counts = []
        kept_target_token_counts = []
        target_truncated = []

        for i in range(len(examples["messages"])):
            row = {key: values[i] for key, values in examples.items()}
            messages = build_prompt_messages(row, dataset_config)
            target = build_target_response(row, dataset_config)

            prompt_text = apply_chat_template(
                tokenizer,
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = apply_chat_template(
                tokenizer,
                messages + [{"role": "assistant", "content": target}],
                tokenize=False,
                add_generation_prompt=False,
            )

            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
            if full_ids[: len(prompt_ids)] == prompt_ids:
                target_ids = full_ids[len(prompt_ids) :]
            else:
                target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
                if tokenizer.eos_token_id is not None:
                    target_ids = target_ids + [tokenizer.eos_token_id]

            original_target_len = len(target_ids)
            if max_target_length and len(target_ids) > max_target_length:
                target_ids = target_ids[:max_target_length]

            if max_seq_length and len(prompt_ids) + len(target_ids) > max_seq_length:
                if len(target_ids) >= max_seq_length:
                    kept_prompt_ids = []
                    kept_target_ids = target_ids[:max_seq_length]
                else:
                    prompt_budget = max_seq_length - len(target_ids)
                    kept_prompt_ids = prompt_ids[-prompt_budget:]
                    kept_target_ids = target_ids
            else:
                kept_prompt_ids = prompt_ids
                kept_target_ids = target_ids

            input_ids = kept_prompt_ids + kept_target_ids
            labels = [-100] * len(kept_prompt_ids) + kept_target_ids

            input_ids_batch.append(input_ids)
            attention_mask_batch.append([1] * len(input_ids))
            labels_batch.append(labels)
            prompt_token_counts.append(len(prompt_ids))
            target_token_counts.append(original_target_len)
            kept_prompt_token_counts.append(len(kept_prompt_ids))
            kept_target_token_counts.append(len(kept_target_ids))
            target_truncated.append(len(kept_target_ids) < original_target_len)

        return {
            "input_ids": input_ids_batch,
            "attention_mask": attention_mask_batch,
            "labels": labels_batch,
            "prompt_token_count": prompt_token_counts,
            "target_token_count": target_token_counts,
            "kept_prompt_token_count": kept_prompt_token_counts,
            "kept_target_token_count": kept_target_token_counts,
            "target_truncated": target_truncated,
        }

    tokenized = dataset.map(tokenize_batch, batched=True, remove_columns=dataset.column_names)
    prompt_counts = tokenized["prompt_token_count"]
    target_counts = tokenized["target_token_count"]
    kept_prompt_counts = tokenized["kept_prompt_token_count"]
    kept_target_counts = tokenized["kept_target_token_count"]
    target_truncated_count = sum(tokenized["target_truncated"])
    print(
        "Tokenized assistant-only examples: "
        f"prompt median={sorted(prompt_counts)[len(prompt_counts)//2]}, "
        f"target median={sorted(target_counts)[len(target_counts)//2]}, "
        f"kept_prompt median={sorted(kept_prompt_counts)[len(kept_prompt_counts)//2]}, "
        f"kept_target median={sorted(kept_target_counts)[len(kept_target_counts)//2]}, "
        f"target_truncated={target_truncated_count}/{len(tokenized)}"
    )
    return tokenized.remove_columns(
        [
            "prompt_token_count",
            "target_token_count",
            "kept_prompt_token_count",
            "kept_target_token_count",
            "target_truncated",
        ]
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = apply_dataset_overrides(config, args)
    dataset_config = config["dataset"]
    model_config = config["model"]
    lora_config = config["lora"]
    training_config = config["training"]

    print(f"Loading model: {model_config['name']}...")
    quantization_config = build_quantization_config(model_config)
    model_load_kwargs = {
        "device_map": model_config.get("device_map", "auto"),
        "trust_remote_code": True,
    }
    if model_config.get("max_memory_per_gpu"):
        visible_gpu_count = torch.cuda.device_count()
        model_load_kwargs["max_memory"] = {
            index: model_config["max_memory_per_gpu"] for index in range(visible_gpu_count)
        }
    if quantization_config is not None:
        model_load_kwargs["quantization_config"] = quantization_config
    if model_config.get("attn_implementation"):
        model_load_kwargs["attn_implementation"] = model_config["attn_implementation"]
    if "use_cache" in model_config:
        model_load_kwargs["use_cache"] = model_config["use_cache"]
    torch_dtype = torch_dtype_from_config(model_config.get("torch_dtype"))
    if torch_dtype is not None:
        model_load_kwargs["torch_dtype"] = torch_dtype

    model = load_base_model(model_config, model_load_kwargs)
    tokenizer = load_tokenizer(model_config)
    tokenizer.padding_side = model_config.get("tokenizer_padding_side", "right")

    dataset = load_training_dataset(dataset_config)
    dataset = filter_dataset(dataset, dataset_config)
    if training_config.get("holdout_split") and not training_config.get("do_eval", False):
        train_dataset, heldout_dataset = split_dataset(
            dataset,
            dataset_config,
            {**training_config, "do_eval": True},
        )
        print(
            "Holding out "
            f"{len(heldout_dataset)} examples for generation eval; Trainer eval disabled."
        )
        eval_dataset = None
    else:
        train_dataset, eval_dataset = split_dataset(dataset, dataset_config, training_config)
    print(f"Prepared {len(dataset)} filtered examples")

    print("Setting up LoRA...")
    peft_config_kwargs = {
        "r": lora_config["r"],
        "lora_alpha": lora_config["alpha"],
        "lora_dropout": lora_config["dropout"],
        "target_modules": lora_config["target_modules"],
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    if lora_config.get("target_parameters"):
        peft_config_kwargs["target_parameters"] = lora_config["target_parameters"]
    peft_config = LoraConfig(**peft_config_kwargs)

    max_steps = training_config.get("max_steps", -1)
    num_train_epochs = 1 if max_steps != -1 else training_config.get("num_epochs", 1)

    loss_mode = training_config.get("loss", "assistant_only")
    max_seq_length = model_config.get("max_seq_length")
    max_target_length = model_config.get("max_target_length")

    if loss_mode == "assistant_only":
        if not max_seq_length:
            raise ValueError("model.max_seq_length is required for assistant_only loss")
        print(
            "Tokenizing datasets with assistant-only labels "
            f"max_seq_length={max_seq_length}, max_target_length={max_target_length}..."
        )
        train_dataset = tokenize_assistant_only_dataset(
            train_dataset,
            tokenizer,
            dataset_config,
            max_seq_length=max_seq_length,
            max_target_length=max_target_length,
        )
        if eval_dataset is not None:
            eval_dataset = tokenize_assistant_only_dataset(
                eval_dataset,
                tokenizer,
                dataset_config,
                max_seq_length=max_seq_length,
                max_target_length=max_target_length,
            )

        if model_config.get("prepare_model_for_kbit_training", True):
            print("Preparing quantized model for LoRA training...")
            model = prepare_model_for_kbit_training(
                model,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        training_args_kwargs = {
            "output_dir": training_config["output_dir"],
            "max_steps": max_steps,
            "num_train_epochs": num_train_epochs,
            "per_device_train_batch_size": training_config["per_device_train_batch_size"],
            "per_device_eval_batch_size": training_config.get(
                "per_device_eval_batch_size", training_config["per_device_train_batch_size"]
            ),
            "gradient_accumulation_steps": training_config["gradient_accumulation_steps"],
            "learning_rate": training_config["learning_rate"],
            "warmup_steps": training_config["warmup_steps"],
            "lr_scheduler_type": training_config["lr_scheduler_type"],
            "optim": training_config["optim"],
            "weight_decay": training_config["weight_decay"],
            "logging_steps": training_config["logging_steps"],
            "save_steps": training_config["save_steps"],
            "save_total_limit": training_config.get("save_total_limit"),
            "seed": training_config["seed"],
            "fp16": not torch.cuda.is_bf16_supported(),
            "bf16": torch.cuda.is_bf16_supported(),
            "eval_strategy": "steps" if eval_dataset is not None else "no",
            "evaluation_strategy": "steps" if eval_dataset is not None else "no",
            "eval_steps": training_config.get("eval_steps", training_config["save_steps"])
            if eval_dataset is not None
            else None,
            "do_eval": eval_dataset is not None,
            "remove_unused_columns": True,
            "report_to": [],
            "gradient_checkpointing": training_config.get("gradient_checkpointing", False),
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
        }
        training_args = TrainingArguments(
            **filter_supported_kwargs(TrainingArguments.__init__, training_args_kwargs)
        )
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        )
        trainer = AssistantOnlyTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            use_suffix_logits=training_config.get("use_suffix_logits", True),
        )
    elif loss_mode == "full_text":
        train_dataset = prepare_text_dataset(train_dataset, tokenizer, dataset_config)
        if eval_dataset is not None:
            eval_dataset = prepare_text_dataset(eval_dataset, tokenizer, dataset_config)

        print("Setting up trainer...")
        sft_config_kwargs = {
            "output_dir": training_config["output_dir"],
            "max_steps": max_steps,
            "num_train_epochs": num_train_epochs,
            "per_device_train_batch_size": training_config["per_device_train_batch_size"],
            "per_device_eval_batch_size": training_config.get(
                "per_device_eval_batch_size", training_config["per_device_train_batch_size"]
            ),
            "gradient_accumulation_steps": training_config["gradient_accumulation_steps"],
            "learning_rate": training_config["learning_rate"],
            "warmup_steps": training_config["warmup_steps"],
            "lr_scheduler_type": training_config["lr_scheduler_type"],
            "optim": training_config["optim"],
            "weight_decay": training_config["weight_decay"],
            "logging_steps": training_config["logging_steps"],
            "save_steps": training_config["save_steps"],
            "save_total_limit": training_config.get("save_total_limit"),
            "seed": training_config["seed"],
            "fp16": not torch.cuda.is_bf16_supported(),
            "bf16": torch.cuda.is_bf16_supported(),
            "dataset_text_field": "text",
            "packing": False,
            "eval_strategy": "steps" if eval_dataset is not None else "no",
            "evaluation_strategy": "steps" if eval_dataset is not None else "no",
            "eval_steps": training_config.get("eval_steps", training_config["save_steps"])
            if eval_dataset is not None
            else None,
            "do_eval": eval_dataset is not None,
        }

        if max_seq_length:
            sft_config_kwargs["max_seq_length"] = max_seq_length
            sft_config_kwargs["max_length"] = max_seq_length

        training_args = SFTConfig(**filter_supported_kwargs(SFTConfig.__init__, sft_config_kwargs))

        trainer_kwargs = {
            "model": model,
            "processing_class": tokenizer,
            "tokenizer": tokenizer,
            "args": training_args,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "peft_config": peft_config,
        }
        trainer = SFTTrainer(**filter_supported_kwargs(SFTTrainer.__init__, trainer_kwargs))
    else:
        raise ValueError(f"Unsupported training.loss: {loss_mode}")

    resume_from_checkpoint = args.resume_from_checkpoint or training_config.get("resume_from_checkpoint")
    if resume_from_checkpoint:
        print(f"Starting training from checkpoint: {resume_from_checkpoint}")
    else:
        print("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    final_model_dir = f"{training_config['output_dir']}/final_model"
    print(f"Saving model to {final_model_dir}...")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print("Training complete.")


if __name__ == "__main__":
    main()
