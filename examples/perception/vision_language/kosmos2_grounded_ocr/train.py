#!/usr/bin/env python3
"""
Finetune Kosmos2.5 on grounded OCR annotations (DocLayNet).
Converted from https://huggingface.co/merve/smol-vision/blob/main/Grounded_Fine_tuning.ipynb
"""
import argparse
import random
import re
from dataclasses import dataclass
from typing import List, Tuple

import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    Kosmos2_5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    set_seed,
)


PROMPT = "<ocr>"


def preprocess_batch(pdf_cells_list, img_sizes, target_sizes) -> List[str]:
    outputs = []
    for pdf_cells, img_size, target_size in zip(pdf_cells_list, img_sizes, target_sizes):
        img_w, img_h = img_size
        target_w, target_h = target_size

        if not pdf_cells:
            cells = []
        elif isinstance(pdf_cells[0], list):
            cells = [cell for group in pdf_cells if group for cell in group]
        else:
            cells = pdf_cells

        scale_x = target_w / float(img_w)
        scale_y = target_h / float(img_h)

        lines = []
        for cell in cells:
            if "bbox" not in cell:
                continue
            x, y, w, h = map(float, cell["bbox"])
            x0 = max(0.0, x)
            y0 = max(0.0, y)
            x1 = min(img_w, x + w)
            y1 = min(img_h, y + h)
            if (x1 - x0) < 1.0 or (y1 - y0) < 1.0:
                continue

            px0 = int(round(x0 * scale_x))
            py0 = int(round(y0 * scale_y))
            px1 = int(round(x1 * scale_x))
            py1 = int(round(y1 * scale_y))
            if not (px1 > px0 and py1 > py0):
                continue

            text = re.sub(r"\s+", " ", cell.get("text", "").replace("\r", " ")).strip()
            text = text.replace("<", "‹").replace(">", "›")
            lines.append(f"<bbox><x_{px0}><y_{py0}><x_{px1}><y_{py1}></bbox>{text}")

        outputs.append("\n".join(lines))
    return outputs


@dataclass
class KosmosCollator:
    processor: AutoProcessor
    device: torch.device

    def __call__(self, examples):
        images = [ex["image"].convert("RGB") for ex in examples]
        pdf_cells_list = [ex["pdf_cells"] for ex in examples]
        img_sizes = [img.size for img in images]

        target_sizes = []
        for img in images:
            inp = self.processor(images=img, return_tensors="pt")
            target_sizes.append((int(inp["width"]), int(inp["height"])))

        targets = preprocess_batch(pdf_cells_list, img_sizes, target_sizes)
        full_texts = [PROMPT + t for t in targets]

        inputs = self.processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = inputs.to(self.device)

        labels = inputs["input_ids"].clone()
        prompt_ids_batch = self.processor.tokenizer([PROMPT] * len(full_texts), add_special_tokens=True).input_ids

        for idx, prompt_ids in enumerate(prompt_ids_batch):
            labels[idx, : len(prompt_ids)] = -100

        inputs["labels"] = labels
        return inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finetune Kosmos2.5 on grounded OCR")
    parser.add_argument("--model-id", default="microsoft/kosmos-2.5")
    parser.add_argument("--dataset-name", default="merve/doclaynet-small")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--eval-ratio", type=float, default=0.1, help="Fraction reserved for eval split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", dest="eval_batch_size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--output-dir", default="kosmos2_5_grounded")
    parser.add_argument("--report-to", default="trackio", help='Comma separated list for Trainer report_to (use "none" to disable)')
    parser.add_argument("--push-to-hub", action="store_true", help="Enable push to Hub (requires HF token)")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    raw_ds = load_dataset(args.dataset_name)[args.dataset_split]
    split_ds = raw_ds.train_test_split(test_size=args.eval_ratio, seed=args.seed)
    train_ds = split_ds["train"]
    eval_ds = split_ds["test"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Kosmos2_5ForConditionalGeneration.from_pretrained(
        args.model_id,
        device_map="auto" if device.type == "cuda" else None,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    processor = AutoProcessor.from_pretrained(args.model_id)

    collate_fn = KosmosCollator(processor=processor, device=model.device)

    report_to = None if args.report_to.lower() == "none" else [t.strip() for t in args.report_to.split(",")]

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        remove_unused_columns=False,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=args.warmup_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        adam_beta2=0.999,
        logging_steps=args.logging_steps,
        optim="adamw_torch",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=1,
        bf16=device.type == "cuda",
        report_to=report_to,
        dataloader_pin_memory=False,
        push_to_hub=args.push_to_hub,
        eval_strategy="steps",
        eval_steps=args.save_steps,
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate_fn,
        args=training_args,
    )

    trainer.train()


if __name__ == "__main__":
    main()
