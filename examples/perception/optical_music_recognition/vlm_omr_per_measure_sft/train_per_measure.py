#!/usr/bin/env python3
"""Per-measure SFT: continue from `safckylj` at the per-measure scale.

Each training sample = (one measure-cell image, that measure's MXC2
slice). Output is short (~30-400 tokens) -- much smaller than whole-page.

Run via train.sh (mounts paths + image + data dir).
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch

# Make vlm_omr_sft and ourselves importable
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))


INSTRUCTION = ("Transcribe this single measure of sheet music to MXC2 "
               "(compact MusicXML). Output only this measure, including "
               "every part's content for it.")


def _load_jsonl(path: Path) -> list[dict]:
    """Load and filter samples. Drops cells with extreme aspect ratios
    (unsloth's vision util refuses aspect > 200; >20 is musically
    implausible for a measure)."""
    out, dropped = [], 0
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            w, h = r.get("width", 0), r.get("height", 0)
            if w <= 0 or h <= 0 or max(w/max(h,1), h/max(w,1)) > 20:
                dropped += 1
                continue
            out.append(r)
    if dropped:
        print(f"  dropped {dropped} extreme-aspect cells from {path.name}",
              flush=True)
    return out


def _open_pil(rec: dict, root: Path):
    from PIL import Image
    return Image.open(root / rec["image"]).convert("RGB")


def _convert_to_conversation(rec: dict, root: Path) -> dict:
    img = _open_pil(rec, root)
    return {
        "messages": [
            {"role": "user", "content": [
                {"type": "text",  "text": INSTRUCTION},
                {"type": "image", "image": img},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": rec["mxc2"]},
            ]},
        ],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, type=Path,
                   help="dir from labels/build_per_measure_dataset.py")
    p.add_argument("--vlm-ckpt",
                   default="/vlm_sft/outputs/safckylj/final_model",
                   help="base VLM checkpoint to SFT off")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--max-train", type=int, default=None,
                   help="cap train samples for quick smoke runs")
    p.add_argument("--max-eval", type=int, default=200)
    p.add_argument("--seed", type=int, default=3407)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # Unsloth + TRL imports (slow; do them after argparse)
    from unsloth import FastVisionModel, is_bf16_supported  # type: ignore
    from unsloth.trainer import UnslothVisionDataCollator  # type: ignore
    from trl import SFTTrainer, SFTConfig  # type: ignore

    print(f"Loading base VLM from {args.vlm_ckpt} ...", flush=True)
    model, processor = FastVisionModel.from_pretrained(
        args.vlm_ckpt, load_in_4bit=True)
    # safckylj's `final_model` is unmerged -- it carries adapter_model.safetensors
    # on top of the unsloth qwen3-vl-8b base. The adapters are already
    # attached after from_pretrained, so we *don't* re-call get_peft_model
    # (that errors with "You already added LoRA adapters to your model!").
    # We just continue training the existing adapters.
    print("Reusing existing LoRA adapters from checkpoint", flush=True)

    print(f"Loading per-measure JSONL from {args.data} ...", flush=True)
    train_recs = _load_jsonl(args.data / "labels_train.jsonl")
    dev_recs = _load_jsonl(args.data / "labels_dev.jsonl")
    if args.max_train and len(train_recs) > args.max_train:
        import random
        random.Random(args.seed).shuffle(train_recs)
        train_recs = train_recs[:args.max_train]
    if args.max_eval and len(dev_recs) > args.max_eval:
        import random
        random.Random(args.seed).shuffle(dev_recs)
        dev_recs = dev_recs[:args.max_eval]
    print(f"  train: {len(train_recs)}, dev: {len(dev_recs)}", flush=True)

    print("Converting to conversation format ...", flush=True)
    t0 = time.time()
    train_data = [_convert_to_conversation(r, args.data) for r in train_recs]
    val_data = [_convert_to_conversation(r, args.data) for r in dev_recs]
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    FastVisionModel.for_training(model)

    cfg = SFTConfig(
        output_dir=str(args.output),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.03,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=args.seed,
        report_to="none",
        # Multimodal-aware SFT settings
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=args.max_seq_length,
        # Periodic eval
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
    )

    processing_class = getattr(processor, "tokenizer", processor)
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=processing_class,
        data_collator=UnslothVisionDataCollator(model, processor),
        args=cfg,
    )
    print("\nStarting training ...", flush=True)
    stats = trainer.train()
    print(f"\nTraining done. Saving final model -> {args.output}/final_model",
          flush=True)
    trainer.save_model(str(args.output / "final_model"))


if __name__ == "__main__":
    main()
