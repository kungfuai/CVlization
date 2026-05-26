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


_PART_DECL_RE = re.compile(r"^(P\d+)\s+(.+)$")
_M_HEADER_RE = re.compile(
    r"^M\s+(\d+)"
    r"(?:\s+key=(-?\d+))?"
    r"(?:\s+time=(\S+))?"
    r"(?:\s+clef=(\S+))?"
    r"(?:\s+clef2=(\S+))?"
    r"(?:\s+staves=(\d+))?"
)


def _context_from_mxc2(mxc2: str, attrs: bool = True) -> str:
    """Extract per-part header info from this measure's MXC2.

    Two modes:
      attrs=True  (legacy v4):  full header per part
        P1 Voice: key=-4 time=4/4 clef=G2
      attrs=False (v5+):        part declarations only
        P1 Voice
        P2 Piano
    The attrs=False mode forces the model to infer key/time/clef from
    the image, which removes the inference-time problem of supplying
    correct defaults for those attributes.
    """
    parts: dict[str, str] = {}    # P1 -> 'Voice'
    headers: dict[str, str] = {}  # P1 -> 'key=-4 time=4/4 clef=G2,F4'
    current_part = None
    for line in mxc2.splitlines():
        line = line.rstrip()
        if line == "---":
            continue
        m = _PART_DECL_RE.match(line)
        if m and m.group(1) not in parts:
            parts[m.group(1)] = m.group(2)
            continue
        if line.startswith(("P1", "P2", "P3", "P4", "P5")) and "\t" not in line and "=" not in line:
            # Part-switch line inside the measure body: "P1"
            tok = line.split()[0]
            current_part = tok
            continue
        if line.startswith("M ") and current_part:
            mh = _M_HEADER_RE.match(line)
            if mh:
                key = mh.group(2)
                time_s = mh.group(3)
                clef = mh.group(4)
                clef2 = mh.group(5)
                bits = []
                if key is not None: bits.append(f"key={key}")
                if time_s:          bits.append(f"time={time_s}")
                if clef:
                    if clef2:
                        bits.append(f"clef={clef},{clef2}")
                    else:
                        bits.append(f"clef={clef}")
                headers.setdefault(current_part, " ".join(bits))
    if not headers and not parts:
        return ""
    lines = []
    if attrs:
        for p_id in sorted(headers):
            name = parts.get(p_id, "")
            lines.append(f"{p_id} {name}: {headers[p_id]}".strip())
        return "Active header per part:\n" + "\n".join(lines)
    # attrs=False: names-only
    for p_id in sorted(parts or headers):
        name = parts.get(p_id, "")
        lines.append(f"{p_id} {name}".strip())
    return "Parts:\n" + "\n".join(lines)


def _make_prompt(mxc2: str, attrs: bool = True) -> str:
    ctx = _context_from_mxc2(mxc2, attrs=attrs)
    if ctx:
        return INSTRUCTION + "\n\n" + ctx
    return INSTRUCTION


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


def _convert_to_conversation(rec: dict, root: Path,
                              attrs_in_prompt: bool = True) -> dict:
    img = _open_pil(rec, root)
    prompt = _make_prompt(rec["mxc2"], attrs=attrs_in_prompt)
    return {
        "messages": [
            {"role": "user", "content": [
                {"type": "text",  "text": prompt},
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
    p.add_argument("--fresh", action="store_true",
                   help="Start from base Qwen3-VL-8B with fresh LoRA "
                        "adapters (instead of continuing safckylj's).")
    p.add_argument("--lora-r", type=int, default=16,
                   help="LoRA rank when --fresh.")
    p.add_argument("--lora-alpha", type=int, default=16,
                   help="LoRA alpha when --fresh.")
    p.add_argument("--no-attr-header", action="store_true",
                   help="Drop key/time/clef from the prompt context for "
                        "EVERY measure. Strictest mode — model infers them "
                        "from image always.")
    p.add_argument("--two-mode", action="store_true",
                   help="Drop attr header only for first-of-page measures "
                        "(where the page visually shows clef/key/time). "
                        "Subsequent measures still get full header context "
                        "in the prompt. Matches two-pass inference. "
                        "Recommended for v5.")
    p.add_argument("--first-oversample", type=int, default=1,
                   help="Replicate first-of-page records this many times "
                        "in train (default 1 = no oversample). 3-5 gives "
                        "the no-header task more practice.")
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

    base = ("unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
            if args.fresh else args.vlm_ckpt)
    print(f"Loading base VLM from {base} ...", flush=True)
    model, processor = FastVisionModel.from_pretrained(base, load_in_4bit=True)
    if args.fresh:
        # Fresh LoRA adapters on the unmodified base.
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=True,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0,
            bias="none",
            random_state=args.seed,
            use_rslora=False,
            loftq_config=None,
        )
        print(f"Attached fresh LoRA r={args.lora_r} alpha={args.lora_alpha}", flush=True)
    else:
        # safckylj's `final_model` already carries adapters; skip
        # get_peft_model and continue training those.
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
    def _is_first_of_page_map(records: list[dict]) -> dict[int, bool]:
        # records keyed by id() -> True iff lowest 'measure' for its
        # (source, score_id, page) group.
        first_m: dict[tuple, int] = {}
        for r in records:
            key = (r.get("source"), r.get("score_id"), r.get("page"))
            m = r.get("measure")
            if m is None:
                continue
            if key not in first_m or m < first_m[key]:
                first_m[key] = m
        out: dict[int, bool] = {}
        for r in records:
            key = (r.get("source"), r.get("score_id"), r.get("page"))
            out[id(r)] = (r.get("measure") == first_m.get(key))
        return out

    def _attrs_for(rec, is_first):
        if args.no_attr_header:
            return False
        if args.two_mode and is_first:
            return False
        return True

    train_first = _is_first_of_page_map(train_recs)
    dev_first = _is_first_of_page_map(dev_recs)

    if args.two_mode and args.first_oversample > 1:
        oversampled = []
        for r in train_recs:
            oversampled.append(r)
            if train_first[id(r)]:
                for _ in range(args.first_oversample - 1):
                    oversampled.append(r)
                    train_first[id(r)] = True  # same record, same id
        n_first = sum(1 for r in train_recs if train_first[id(r)])
        print(f"  oversampled first-of-page: {n_first} -> "
              f"{n_first * args.first_oversample} (total train={len(oversampled)})",
              flush=True)
        train_recs = oversampled

    print(f"  no_attr_header={args.no_attr_header}  two_mode={args.two_mode}  "
          f"first_oversample={args.first_oversample}", flush=True)
    train_data = [_convert_to_conversation(r, args.data,
                                            _attrs_for(r, train_first[id(r)]))
                  for r in train_recs]
    val_data = [_convert_to_conversation(r, args.data,
                                          _attrs_for(r, dev_first[id(r)]))
                for r in dev_recs]
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
