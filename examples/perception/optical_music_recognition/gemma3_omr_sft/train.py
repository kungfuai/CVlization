#!/usr/bin/env python3
"""
Gemma-3 Vision fine-tuning for Optical Music Recognition (OMR).

Fine-tunes Gemma-3 on the zzsi/openscore pages_transcribed dataset:
  input  — full-page sheet music image (LilyPond render)
  output — per-page MusicXML (bars on that page)

Based on the gemma3_vision_sft example; adapted for OMR with the openscore dataset.

Usage:
  python train.py                        # uses config.yaml defaults (smoke test)
  python train.py --corpus lieder quartets  # override corpus filter
  python train.py --epochs 2             # full training run
"""

import argparse
import yaml
import torch
from datasets import load_dataset
from unsloth import FastVisionModel, get_chat_template
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

INSTRUCTION = "Transcribe this sheet music page to MusicXML."


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--corpus", nargs="+", default=None,
                        help="Override corpus filter (e.g. lieder quartets orchestra)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override num_train_epochs (disables max_steps)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Override max_samples")
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading configuration...")
    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset_config  = config["dataset"]
    model_config    = config["model"]
    lora_config     = config["lora"]
    training_config = config["training"]

    # CLI overrides
    corpora = args.corpus or dataset_config.get("corpora", None)
    if args.epochs is not None:
        training_config["num_train_epochs"] = args.epochs
        training_config["max_steps"] = -1
    if args.max_samples is not None:
        dataset_config["max_samples"] = args.max_samples

    # ── Model ──────────────────────────────────────────────────────────────────
    print(f"Loading model: {model_config['name']} ...")
    model, processor = FastVisionModel.from_pretrained(
        model_config["name"],
        load_in_4bit=model_config["load_in_4bit"],
        use_gradient_checkpointing=model_config.get("use_gradient_checkpointing", "unsloth"),
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=lora_config["finetune_vision_layers"],
        finetune_language_layers=lora_config["finetune_language_layers"],
        finetune_attention_modules=lora_config["finetune_attention_modules"],
        finetune_mlp_modules=lora_config["finetune_mlp_modules"],
        r=lora_config["r"],
        lora_alpha=lora_config["alpha"],
        lora_dropout=lora_config["dropout"],
        bias="none",
        random_state=training_config["seed"],
        use_rslora=False,
        loftq_config=None,
        target_modules="all-linear",
    )

    # ── Dataset ────────────────────────────────────────────────────────────────
    repo   = dataset_config["repo"]
    cfg    = dataset_config["config"]
    split  = dataset_config.get("split", "train")
    print(f"Loading dataset: {repo} (config={cfg}, split={split}) ...")
    dataset = load_dataset(repo, cfg, split=split)

    if corpora:
        print(f"Filtering to corpora: {corpora} ...")
        corpus_set = set(corpora)
        dataset = dataset.filter(lambda r: r["corpus"] in corpus_set)
        print(f"  {len(dataset)} rows after filter")

    if "max_samples" in dataset_config:
        n = dataset_config["max_samples"]
        dataset = dataset.select(range(min(n, len(dataset))))
        print(f"  Limited to {len(dataset)} samples")

    print(f"Dataset size: {len(dataset)}")
    print(f"Sample — score_id={dataset[0]['score_id']} "
          f"corpus={dataset[0]['corpus']} "
          f"page={dataset[0]['page']}/{dataset[0]['n_pages']} "
          f"bars={dataset[0]['bar_start']}-{dataset[0]['bar_end']}")
    print(f"MusicXML snippet: {dataset[0]['musicxml'][:80].strip()} ...")

    # ── Chat format ────────────────────────────────────────────────────────────
    def convert_to_conversation(sample):
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text",  "text": INSTRUCTION},
                        {"type": "image", "image": sample["image"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": sample["musicxml"]}],
                },
            ]
        }

    print("Converting to conversation format ...")
    converted = [convert_to_conversation(s) for s in dataset]

    val_ratio = dataset_config.get("val_split_ratio", 0.0)
    if val_ratio > 0:
        split_idx = int(len(converted) * (1 - val_ratio))
        train_data = converted[:split_idx]
        val_data   = converted[split_idx:]
        print(f"Split: {len(train_data)} train, {len(val_data)} val")
    else:
        train_data = converted
        val_data   = None

    chat_template = model_config.get("chat_template", "gemma-3")
    print(f"Setting up {chat_template} chat template ...")
    processor = get_chat_template(processor, chat_template)

    # ── Training ───────────────────────────────────────────────────────────────
    FastVisionModel.for_training(model)

    training_args = SFTConfig(
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=training_config.get("max_grad_norm", 0.3),
        warmup_ratio=training_config.get("warmup_ratio", 0.03),
        max_steps=training_config.get("max_steps", -1),
        num_train_epochs=(
            training_config.get("num_train_epochs", 1)
            if training_config.get("max_steps", -1) == -1 else 1
        ),
        learning_rate=training_config["learning_rate"],
        logging_steps=training_config["logging_steps"],
        save_strategy="steps",
        save_steps=training_config["save_steps"],
        eval_strategy="steps" if val_data else "no",
        eval_steps=(
            training_config.get("val_steps", training_config["save_steps"])
            if val_data else None
        ),
        optim=training_config["optim"],
        weight_decay=training_config["weight_decay"],
        lr_scheduler_type=training_config["lr_scheduler_type"],
        seed=training_config["seed"],
        output_dir=training_config["output_dir"],
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=model_config.get("max_length", 4096),
    )

    gpu_stats = torch.cuda.get_device_properties(0)
    start_mem = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3)
    max_mem   = round(gpu_stats.total_memory / 1024 ** 3, 3)
    print(f"\nGPU: {gpu_stats.name}  |  {max_mem} GB total  |  {start_mem} GB reserved")

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=processor.tokenizer,
        data_collator=UnslothVisionDataCollator(model, processor),
        args=training_args,
    )

    print("\nStarting training ...")
    stats = trainer.train()

    used_mem = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3)
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"  Time   : {round(stats.metrics['train_runtime'] / 60, 2)} min")
    print(f"  Memory : {used_mem} GB peak ({round(used_mem / max_mem * 100, 1)}%)")
    print('='*70)

    # ── Save ───────────────────────────────────────────────────────────────────
    output_dir    = training_config["output_dir"]
    final_dir     = f"{output_dir}/final_model"
    print(f"\nSaving model to {final_dir} ...")
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print("Model saved.")

    # ── Post-training inference test ───────────────────────────────────────────
    if training_config.get("test_after_training", True) and len(dataset) >= 2:
        print("\nPost-training inference test ...")
        FastVisionModel.for_inference(model)

        sample  = dataset[min(10, len(dataset) - 1)]
        image   = sample["image"]
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": INSTRUCTION}],
        }]
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            image, input_text, add_special_tokens=False, return_tensors="pt"
        ).to("cuda")

        from transformers import TextStreamer
        streamer = TextStreamer(processor, skip_prompt=True)
        print(f"\n--- Sample (score_id={sample['score_id']}, page={sample['page']}) ---")
        print(f"Reference (first 200 chars): {sample['musicxml'][:200].strip()}")
        print("Prediction:")
        model.generate(**inputs, streamer=streamer, max_new_tokens=512,
                       use_cache=True, temperature=1.0, top_p=0.95)
        print("\n--- End ---")


if __name__ == "__main__":
    main()
