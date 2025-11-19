#!/usr/bin/env python3
"""
Finetune Microsoft Florence-2 on the DocumentVQA dataset.
Converted from https://huggingface.co/merve/smol-vision/blob/main/Fine_tune_Florence_2.ipynb
"""
import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AdamW,
    AutoModelForCausalLM,
    AutoProcessor,
    get_scheduler,
    set_seed,
)


class DocVQADataset(Dataset):
    def __init__(self, dataset_split, prompt_prefix: str = "<DocVQA>"):
        self.data = dataset_split
        self.prompt_prefix = prompt_prefix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        question = self.prompt_prefix + example["question"]
        answer = example["answers"][0]
        image = example["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        return question, answer, image


@dataclass
class FlorenceCollator:
    processor: AutoProcessor
    device: torch.device

    def __call__(self, batch):
        questions, answers, images = zip(*batch)
        inputs = self.processor(
            text=list(questions),
            images=list(images),
            return_tensors="pt",
            padding=True,
        )
        labels = self.processor.tokenizer(
            text=list(answers),
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        ).input_ids

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        labels = labels.to(self.device)
        inputs["labels"] = labels
        return inputs


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune Florence-2 on DocVQA")
    parser.add_argument("--model-id", default="microsoft/Florence-2-base-ft")
    parser.add_argument("--revision", default="refs/pr/6", help="Model revision/tag/commit")
    parser.add_argument("--dataset-name", default="HuggingFaceM4/DocumentVQA")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="validation")
    parser.add_argument("--output-dir", default="outputs/florence2_docvqa")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-freeze-vision", action="store_true", help="Do not freeze the vision tower")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default=None, help="Target repo name when pushing to hub")
    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)

    dataset = load_dataset(args.dataset_name)
    train_split = dataset[args.train_split]
    val_split = dataset[args.val_split]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        revision=args.revision,
        trust_remote_code=True,
    )

    if not args.no_freeze_vision:
        for param in model.vision_tower.parameters():
            param.requires_grad = False

    train_ds = DocVQADataset(train_split)
    val_ds = DocVQADataset(val_split)

    collator = FlorenceCollator(processor=processor, device=model.device)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr)
    num_training_steps = args.epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            labels = batch.pop("labels")
            outputs = model(**batch, labels=labels)
            loss = outputs.loss / args.grad_accum
            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            running_loss += loss.item() * args.grad_accum

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} - training loss {avg_loss:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                labels = batch.pop("labels")
                outputs = model(**batch, labels=labels)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} - validation loss {avg_val_loss:.4f}")

        ckpt_dir = Path(args.checkpoint_dir) / f"epoch_{epoch+1}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        processor.save_pretrained(ckpt_dir)

    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

    if args.push_to_hub:
        repo_id = args.hub_model_id or Path(args.output_dir).name
        print(f"Pushing model to hub repo: {repo_id}")
        model.push_to_hub(repo_id)
        processor.push_to_hub(repo_id)


if __name__ == "__main__":
    main()
