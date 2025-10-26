#!/usr/bin/env python
"""Utility script for fine-tuning instruction datasets on top of Mistral 7B (or compatible) using LoRA.

The original example was hard-coded for long training runs.  This rewrite exposes
command-line switches so you can run quick smoke tests (e.g. a single step on a
few samples) or scale up to the full 7B workflow.  Invoke `python train.py --help`
for the complete list of options.
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(levelname)s: %(message)s")

PROMPT_TEMPLATE = """Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values.
This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute'].
The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier'].

### Target sentence:
{target}

### Meaning representation:
{meaning}
"""


DEFAULT_VIGGO_FILES = {
    "train": "https://huggingface.co/datasets/gem/viggo/resolve/main/data/train.json?download=1",
    "validation": "https://huggingface.co/datasets/gem/viggo/resolve/main/data/validation.json?download=1",
    "test": "https://huggingface.co/datasets/gem/viggo/resolve/main/data/test.json?download=1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Mistral-style models with LoRA")
    parser.add_argument("--model_id", default="mistralai/Mistral-7B-v0.1", help="Base model to load")
    parser.add_argument(
        "--dataset_name", default="gem/viggo", help="Dataset name or local path passed to `datasets.load_dataset`"
    )
    parser.add_argument("--train_split", default="train", help="Dataset split for training")
    parser.add_argument("--eval_split", default="validation", help="Dataset split for evaluation")
    parser.add_argument(
        "--data_format",
        default="auto",
        choices=["auto", "json", "parquet", "csv", "text", "dataset"],
        help="Data loader to use when train/eval files are provided",
    )
    parser.add_argument("--train_file", default=None, help="Optional path/URL to training file")
    parser.add_argument("--eval_file", default=None, help="Optional path/URL to eval file")
    parser.add_argument("--test_file", default=None, help="Optional path/URL to test file")
    parser.add_argument(
        "--dataset_trust_remote_code",
        dest="dataset_trust_remote_code",
        action="store_true",
        help="Forward trust_remote_code=True to datasets.load_dataset",
    )
    parser.add_argument("--no_dataset_trust_remote_code", dest="dataset_trust_remote_code", action="store_false")
    parser.set_defaults(dataset_trust_remote_code=True)
    parser.add_argument("--max_train_samples", type=int, default=None, help="Limit number of training examples")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Limit number of eval examples")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Tokenization max length")

    parser.add_argument("--output_dir", default="./mistral-viggo", help="Where to store checkpoints and logs")
    parser.add_argument("--max_steps", type=int, default=1000, help="Total optimisation steps")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2.5e-5)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_strategy", default="steps")
    parser.add_argument("--evaluation_strategy", default="steps")
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument("--report_to", default="none", help="Reporting destination (wandb|tensorboard|none)")
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--load_best_model_at_end", action="store_true", help="Enable Trainer's best-model loading")
    parser.add_argument("--no_load_best_model_at_end", dest="load_best_model_at_end", action="store_false")
    parser.set_defaults(load_best_model_at_end=False)

    parser.add_argument("--bf16", dest="bf16", action="store_true", help="Enable bf16")
    parser.add_argument("--no_bf16", dest="bf16", action="store_false")
    parser.set_defaults(bf16=True)

    parser.add_argument("--load_in_4bit", dest="load_in_4bit", action="store_true")
    parser.add_argument("--no_load_in_4bit", dest="load_in_4bit", action="store_false")
    parser.set_defaults(load_in_4bit=True)

    parser.add_argument("--gradient_checkpointing", dest="gradient_checkpointing", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.set_defaults(gradient_checkpointing=True)

    parser.add_argument("--trust_remote_code", dest="trust_remote_code", action="store_true")
    parser.add_argument("--no_trust_remote_code", dest="trust_remote_code", action="store_false")
    parser.set_defaults(trust_remote_code=False)

    parser.add_argument("--do_eval", dest="do_eval", action="store_true")
    parser.add_argument("--no_eval", dest="do_eval", action="store_false")
    parser.set_defaults(do_eval=True)

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head",
        help="Comma separated list of modules to LoRA-ise",
    )

    parser.add_argument("--max_new_tokens_eval", type=int, default=128, help="Optional sample generation length")
    parser.add_argument("--sample_prompt", default=None, help="When provided, generate text after training")

    return parser.parse_args()


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_datasets(args: argparse.Namespace, tokenizer) -> tuple[Dataset, Optional[Dataset]]:
    data_format = args.data_format
    data_files = {}
    if args.train_file:
        data_files[args.train_split] = args.train_file
    if args.eval_file:
        data_files[args.eval_split] = args.eval_file
    if args.test_file:
        data_files["test"] = args.test_file

    if data_format == "auto":
        if data_files:
            sample_path = next(iter(data_files.values()))
            ext = Path(sample_path).suffix.lower().lstrip(".")
            guess_map = {"json": "json", "jsonl": "json", "parquet": "parquet", "csv": "csv", "txt": "text"}
            data_format = guess_map.get(ext, "json")
        elif args.dataset_name == "gem/viggo":
            data_format = "json"
            data_files = DEFAULT_VIGGO_FILES.copy()
        else:
            data_format = "dataset"

    LOGGER.info("Loading dataset using format=%s", data_format)

    if data_format == "dataset":
        train_dataset = load_dataset(
            args.dataset_name,
            split=args.train_split,
            trust_remote_code=args.dataset_trust_remote_code,
        )
        eval_dataset = None
        if args.do_eval:
            eval_dataset = load_dataset(
                args.dataset_name,
                split=args.eval_split,
                trust_remote_code=args.dataset_trust_remote_code,
            )
    else:
        if not data_files:
            raise ValueError(
                "No data files supplied. Provide --train_file/--eval_file or use a dataset name with --data_format dataset."
            )
        dataset_dict = load_dataset(data_format, data_files=data_files)
        if args.train_split not in dataset_dict:
            raise ValueError(f"Train split '{args.train_split}' not available in {list(dataset_dict.keys())}")
        train_dataset = dataset_dict[args.train_split]
        eval_dataset = None
        if args.do_eval and args.eval_split in dataset_dict:
            eval_dataset = dataset_dict[args.eval_split]

    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
    if args.do_eval and args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))

    def build_prompts(batch):
        prompts = [
            PROMPT_TEMPLATE.format(target=target, meaning=meaning)
            for target, meaning in zip(batch["target"], batch["meaning_representation"])
        ]
        tokenized = tokenizer(
            prompts,
            truncation=True,
            max_length=args.max_seq_length,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    remove_columns = train_dataset.column_names
    tokenized_train = train_dataset.map(
        build_prompts,
        batched=True,
        remove_columns=remove_columns,
    )

    tokenized_eval = None
    if args.do_eval and eval_dataset is not None:
        tokenized_eval = eval_dataset.map(
            build_prompts,
            batched=True,
            remove_columns=eval_dataset.column_names,
        )

    LOGGER.info(
        "Prepared datasets - train: %d examples%s",
        len(tokenized_train),
        f", eval: {len(tokenized_eval)}" if tokenized_eval is not None else "",
    )
    return tokenized_train, tokenized_eval


def load_model_and_tokenizer(args: argparse.Namespace):
    LOGGER.info("Loading tokenizer from %s", args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        padding_side="left",
        add_eos_token=True,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    LOGGER.info("Loading base model (%s-bit=%s)", "4" if args.load_in_4bit else "16", args.load_in_4bit)
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            quantization_config=quant_config,
            trust_remote_code=args.trust_remote_code,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=dtype,
            trust_remote_code=args.trust_remote_code,
        )

    target_modules: List[str] = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False
    return model, tokenizer


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args)
    train_dataset, eval_dataset = prepare_datasets(args, tokenizer)

    current_time = datetime.utcnow().strftime("%Y-%m-%d-%H-%M")
    run_name = f"mistral-viggo-{current_time}"
    output_dir = args.output_dir

    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        logging_steps=max(1, args.logging_steps),
        save_strategy=args.save_strategy,
        save_steps=max(1, args.save_steps),
        save_total_limit=args.save_total_limit,
        evaluation_strategy="no" if not args.do_eval else args.evaluation_strategy,
        eval_steps=max(1, args.eval_steps),
        bf16=args.bf16 and torch.cuda.is_available(),
        optim="paged_adamw_8bit",
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=[args.report_to] if args.report_to else ["none"],
        do_eval=args.do_eval,
        load_best_model_at_end=args.load_best_model_at_end and args.do_eval,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if args.do_eval else None,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    LOGGER.info("Starting training (max_steps=%d)", args.max_steps)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model()

    if args.sample_prompt:
        LOGGER.info("Generating sample using prompt: %s", args.sample_prompt)
        input_ids = tokenizer(args.sample_prompt, return_tensors="pt").to(trainer.model.device)
        with torch.no_grad():
            generated = trainer.model.generate(
                **input_ids,
                max_new_tokens=args.max_new_tokens_eval,
                do_sample=True,
                top_p=0.9,
                temperature=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        LOGGER.info("Sample generation:\n%s", text)


if __name__ == "__main__":
    main()
