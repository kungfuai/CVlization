import argparse
import inspect
from typing import Any, TypedDict

from unsloth import FastLanguageModel, FastModel
import torch
import yaml
from transformers import DataCollatorForSeq2Seq, TrainingArguments
from trl import SFTTrainer

from train import (
    AssistantOnlyTrainer,
    apply_dataset_overrides,
    apply_chat_template,
    build_prompt_messages,
    build_target_response,
    filter_dataset,
    load_training_dataset,
    split_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Unsloth experiment config")
    parser.add_argument("--dataset-path", help="Override dataset.path from config, e.g. json or a HF dataset")
    parser.add_argument("--data-files", help="Override dataset.data_files from config for local JSON/JSONL files")
    parser.add_argument("--data-dir", help="Override dataset.data_dir from config")
    parser.add_argument(
        "--resume-from-checkpoint",
        help="Optional Trainer checkpoint path to resume optimizer/scheduler/trainer state",
    )
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def filter_supported_kwargs(callable_obj, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(callable_obj)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def text_tokenizer(processor_or_tokenizer):
    return getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)


def encode_text(processor_or_tokenizer, text: str, add_special_tokens: bool = False):
    tokenizer = text_tokenizer(processor_or_tokenizer)
    return tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"]


def patch_transformers_remote_code_compat() -> None:
    import transformers.utils as transformers_utils

    try:
        getattr(transformers_utils, "LossKwargs")
    except (AttributeError, NameError):
        class LossKwargs(TypedDict, total=False):
            labels: Any

        transformers_utils.LossKwargs = LossKwargs


def load_unsloth_model(model_config: dict[str, Any]):
    patch_transformers_remote_code_compat()
    model_name = model_config["name"]
    loader_kwargs = {
        "model_name": model_name,
        "max_seq_length": model_config["max_seq_length"],
        "dtype": None,
        "load_in_4bit": model_config.get("load_in_4bit", True),
        "full_finetuning": model_config.get("full_finetuning", False),
    }
    if "trust_remote_code" in model_config:
        loader_kwargs["trust_remote_code"] = model_config["trust_remote_code"]
    if "attn_implementation" in model_config:
        loader_kwargs["attn_implementation"] = model_config["attn_implementation"]
    if "unsloth_force_compile" in model_config:
        loader_kwargs["unsloth_force_compile"] = model_config["unsloth_force_compile"]
    if token := model_config.get("token"):
        loader_kwargs["token"] = token

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(**loader_kwargs)
        peft_cls = FastLanguageModel
    except Exception as first_error:
        print(
            "FastLanguageModel.from_pretrained failed; retrying with FastModel. "
            f"First error: {type(first_error).__name__}: {first_error}"
        )
        model, tokenizer = FastModel.from_pretrained(**loader_kwargs)
        peft_cls = FastModel

    plain_tokenizer = text_tokenizer(tokenizer)
    if plain_tokenizer.pad_token is None:
        plain_tokenizer.pad_token = plain_tokenizer.eos_token
    plain_tokenizer.padding_side = "right"

    lora_config = model_config["lora"]
    target_modules = lora_config["target_modules"]
    peft_kwargs = {
        "r": lora_config["r"],
        "target_modules": target_modules,
        "lora_alpha": lora_config["alpha"],
        "lora_dropout": lora_config.get("dropout", 0.0),
        "bias": "none",
        "use_gradient_checkpointing": lora_config.get("use_gradient_checkpointing", "unsloth"),
        "random_state": model_config.get("seed", 3407),
        "use_rslora": lora_config.get("use_rslora", False),
        "loftq_config": None,
    }
    model = peft_cls.get_peft_model(model, **peft_kwargs)
    return model, tokenizer


def tokenize_assistant_only_dataset(
    dataset,
    tokenizer,
    dataset_config: dict[str, Any],
    max_seq_length: int,
    max_target_length: int | None,
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

            plain_tokenizer = text_tokenizer(tokenizer)
            prompt_text = apply_chat_template(
                plain_tokenizer,
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = apply_chat_template(
                plain_tokenizer,
                messages + [{"role": "assistant", "content": target}],
                tokenize=False,
                add_generation_prompt=False,
            )

            prompt_ids = encode_text(tokenizer, prompt_text, add_special_tokens=False)
            full_ids = encode_text(tokenizer, full_text, add_special_tokens=False)
            if full_ids[: len(prompt_ids)] == prompt_ids:
                target_ids = full_ids[len(prompt_ids) :]
            else:
                target_ids = encode_text(tokenizer, target, add_special_tokens=False)
                eos_token_id = text_tokenizer(tokenizer).eos_token_id
                if eos_token_id is not None:
                    target_ids = target_ids + [eos_token_id]

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
    training_config = config["training"]
    model_config["lora"] = config["lora"]
    model_config["seed"] = training_config["seed"]

    print(f"Loading Unsloth model: {model_config['name']}")
    model, tokenizer = load_unsloth_model(model_config)

    dataset = load_training_dataset(dataset_config)
    dataset = filter_dataset(dataset, dataset_config)
    split_training_config = dict(training_config)
    if training_config.get("holdout_split", training_config.get("do_eval", False)):
        split_training_config["do_eval"] = True
    train_dataset, eval_dataset = split_dataset(dataset, dataset_config, split_training_config)
    print(f"Prepared {len(dataset)} filtered examples")

    train_dataset = tokenize_assistant_only_dataset(
        train_dataset,
        tokenizer,
        dataset_config,
        max_seq_length=model_config["max_seq_length"],
        max_target_length=model_config.get("max_target_length"),
    )
    run_trainer_eval = training_config.get("run_trainer_eval", training_config.get("do_eval", False))
    if eval_dataset is not None and run_trainer_eval:
        eval_dataset = tokenize_assistant_only_dataset(
            eval_dataset,
            tokenizer,
            dataset_config,
            max_seq_length=model_config["max_seq_length"],
            max_target_length=model_config.get("max_target_length"),
        )
    else:
        eval_dataset = None

    max_steps = training_config.get("max_steps", -1)
    num_train_epochs = 1 if max_steps != -1 else training_config.get("num_epochs", 1)
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
        "max_grad_norm": training_config.get("max_grad_norm", 1.0),
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
    }

    training_args = TrainingArguments(
        **filter_supported_kwargs(TrainingArguments.__init__, training_args_kwargs)
    )
    plain_tokenizer = text_tokenizer(tokenizer)
    trainer = AssistantOnlyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=plain_tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        ),
        use_suffix_logits=training_config.get("use_suffix_logits", True),
    )

    resume_from_checkpoint = (
        args.resume_from_checkpoint or training_config.get("resume_from_checkpoint")
    )
    if resume_from_checkpoint:
        print(f"Starting Unsloth training from checkpoint: {resume_from_checkpoint}")
    else:
        print("Starting Unsloth training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    final_output_dir = f"{training_config['output_dir'].rstrip('/')}/final_model"
    print(f"Saving model to {final_output_dir}...")
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    print("Unsloth training complete.")


if __name__ == "__main__":
    main()
