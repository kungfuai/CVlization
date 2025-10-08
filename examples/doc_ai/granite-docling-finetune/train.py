"""
Minimal Single‑GPU TRL SFT example for Granite‑Docling (Vision‑Language)

• Model: ibm-granite/granite-docling-258M (VLM)
• Trainer: TRL SFTTrainer + PEFT (LoRA / DoRA)
• Hardware: Works on a single 24 GB GPU (A10/RTX 6000 Ada). For T4 (16 GB), lower max_seq_len, r, and image sizes.
• Data: Expects a JSON/JSONL Dataset with fields: {image_path, prompt, doctags}
         - prompt: the instruction to the model (e.g., "Extract text + layout as DocTags")
         - doctags: ground-truth DocTags string (XML-ish)

Tip: If you see the notorious "!!!!!!!" generations on older GPUs (no bf16), run inference with dtype=float32.

This script:
1) Loads model + processor correctly (no manual image transforms)
2) Builds Granite chat-formatted samples via processor.apply_chat_template
3) Masks non‑assistant tokens (only trains on the DocTags span)
4) Trains with LoRA (text tower only) and saves adapters
5) Includes a quick smoke test and a vLLM serving note

Install deps (Python ≥3.10):
    pip install -U transformers accelerate peft trl datasets bitsandbytes pillow
    # Optional: wandb

Run:
    python train.py
"""
from __future__ import annotations
import os, json, math, random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image

from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# -----------------------------
# Config
# -----------------------------
MODEL_ID = os.environ.get("MODEL_ID", "ibm-granite/granite-docling-258M")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./outputs/granite_docling_sft")
TRAIN_DATA = os.environ.get("TRAIN_DATA", "ds4sd/docling-dpbench")  # "ds4sd/docling-dpbench" or path to JSON/JSONL
TRAIN_SPLIT = os.environ.get("TRAIN_SPLIT", "test")  # dpbench only has 'test' split
VAL_SPLIT = os.environ.get("VAL_SPLIT", None)  # e.g., "validation" or None
MAX_TRAIN_SAMPLES = int(os.environ.get("MAX_TRAIN_SAMPLES", 0)) or None  # Limit training samples (0 = use all)

# Training hyperparams (safe-ish defaults for 24 GB)
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 1))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", 8))
NUM_EPOCHS = float(os.environ.get("NUM_EPOCHS", 2))
LR = float(os.environ.get("LR", 1e-4))
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", 2048))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 512))
BF16 = os.environ.get("BF16", "true").lower() == "true"

# LoRA/DoRA config
LORA_R = int(os.environ.get("LORA_R", 16))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", 32))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", 0.05))
USE_DORA = os.environ.get("USE_DORA", "true").lower() == "true"
TARGET_MODULES = os.environ.get(
    "TARGET_MODULES",
    "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
).split(",")

# 4-bit (preferred for single GPU). If you switch to 8-bit, remove 4-bit flags.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if BF16 else torch.float16,
    bnb_4bit_use_double_quant=True,
)

device_map = "auto"

torch.backends.cuda.matmul.allow_tf32 = True

# -----------------------------
# Data
# -----------------------------
class DoclingJSONDataset(Dataset):
    """Generic JSON/JSONL or HF dataset loader that yields dicts:
       {image_path: str, prompt: str, doctags: str}
    """
    def __init__(self, source: str, split: str | None = None, max_items: int | None = None):
        self.samples: List[Dict[str, Any]] = []
        if os.path.exists(source):
            # A local JSON or JSONL file
            if source.endswith(".jsonl"):
                with open(source, "r", encoding="utf-8") as f:
                    for line in f:
                        ex = json.loads(line)
                        self.samples.append(ex)
            else:
                with open(source, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    assert isinstance(data, list)
                    self.samples = data
        else:
            # A named HF dataset
            ds = load_dataset(source)
            subset = ds[split] if split else ds[list(ds.keys())[0]]

            # Special handling for ds4sd/docling-dpbench
            if source == "ds4sd/docling-dpbench":
                import io
                for ex in subset:
                    # Get ground truth image (first page)
                    if ex.get("GroundTruthPageImages") and len(ex["GroundTruthPageImages"]) > 0:
                        image_bytes = ex["GroundTruthPageImages"][0]["bytes"]
                        image = Image.open(io.BytesIO(image_bytes))

                        # Get ground truth document and extract text
                        gt_doc = json.loads(ex["GroundTruthDocument"]) if isinstance(ex["GroundTruthDocument"], str) else ex["GroundTruthDocument"]

                        # Extract text from the structured format
                        doctags = ""
                        if isinstance(gt_doc, dict):
                            # Extract from 'texts' field (list of text elements)
                            texts = gt_doc.get("texts", [])
                            if texts:
                                # Concatenate all text content
                                text_parts = [t.get("text", "") for t in texts if isinstance(t, dict) and t.get("text")]
                                doctags = "\n".join(text_parts)

                        if not doctags:
                            # Fallback: convert entire structure to string (for debugging)
                            doctags = str(gt_doc)[:2048]

                        if image and doctags:
                            self.samples.append({
                                "image": image,  # PIL Image directly
                                "prompt": "Extract text and layout from this document.",
                                "doctags": doctags[:2048],  # Limit length
                            })
            else:
                # Generic HF dataset
                for ex in subset:
                    image_path = ex.get("image") or ex.get("image_path") or ex.get("image_file")
                    prompt = ex.get("prompt") or "Extract text and layout as DocTags."
                    doctags = ex.get("doctags") or ex.get("ground_truth") or ex.get("labels")
                    if image_path and doctags:
                        self.samples.append({
                            "image_path": image_path,
                            "prompt": prompt,
                            "doctags": doctags,
                        })
        if max_items:
            self.samples = self.samples[:max_items]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.samples[idx]
        # Load PIL image lazily per sample; no manual resize/normalize.
        if "image" in ex:
            # Already a PIL Image (from dpbench)
            image = ex["image"]
        else:
            # Load from path
            image = Image.open(ex["image_path"]).convert("RGB")
        return {
            "image": image,
            "prompt": ex["prompt"],
            "doctags": ex["doctags"],
        }

# -----------------------------
# Chat formatting & masking helpers
# -----------------------------

def build_messages(prompt: str, doctags: str) -> List[Dict[str, Any]]:
    """Granite-style conversation: system + user(image+text) + assistant(ground truth).
       Image will be injected later via processor; here we just craft roles/contents.
    """
    system = {
        "role": "system",
        "content": "You are a precise document extraction model. Output valid DocTags only.",
    }
    user = {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            # image will be passed through the processor call; keep structure consistent
        ],
    }
    assistant = {"role": "assistant", "content": doctags}
    return [system, user, assistant]


def find_subseq_positions(haystack: torch.Tensor, needle: List[int]) -> int:
    """Return start index of the last occurrence of `needle` inside 1D `haystack`.
       If not found, returns 0. Assumes needle length is small (role header length).
    """
    if len(needle) == 0:
        return 0
    h = haystack.tolist()
    n = len(needle)
    start = 0
    found = 0
    while True:
        try:
            i = h.index(needle[0], start)
        except ValueError:
            break
        if h[i : i + n] == needle:
            found = i
        start = i + 1
    return found


@dataclass
class Collator:
    processor: Any
    tokenizer: Any
    eos_id: int
    assistant_header_ids: List[int]
    max_seq_len: int

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Build role-formatted texts via chat template, but pass images via processor below.
        texts: List[str] = []
        images: List[List[Image.Image]] = []  # each sample may contain multiple images; here we use one
        for ex in batch:
            msgs = build_messages(ex["prompt"], ex["doctags"])
            txt = self.processor.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(txt)
            images.append([ex["image"]])

        # Tokenize text; process images together to keep sizes consistent
        toks = self.processor.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
        )
        vis = self.processor(
            text=None, images=images, return_tensors="pt", padding=True
        )  # yields pixel_values

        input_ids = toks["input_ids"]
        labels = input_ids.clone()

        # Mask everything up to the *last* assistant header occurrence per row
        for i in range(input_ids.size(0)):
            start_idx = find_subseq_positions(input_ids[i], self.assistant_header_ids)
            labels[i, :start_idx] = -100
            # optional: stop loss at first eos after start_idx
            eos_positions = (input_ids[i] == self.eos_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                first_eos_after = next((int(p) for p in eos_positions if p >= start_idx), None)
                if first_eos_after is not None:
                    labels[i, first_eos_after + 1 :] = -100

        return {
            **toks,
            **{k: v for k, v in vis.items() if k == "pixel_values"},
            "labels": labels,
        }

# -----------------------------
# Load model, processor, tokenizer
# -----------------------------
print(f"Loading model: {MODEL_ID}")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map=device_map,
)

tokenizer = processor.tokenizer

# Freeze the vision tower explicitly (we adapt text tower only)
if hasattr(model, "vision_tower"):
    for p in model.vision_tower.parameters():
        p.requires_grad_(False)

# Build assistant header ids by rendering an empty assistant turn
assistant_messages = [{"role": "assistant", "content": ""}]
assistant_text = processor.apply_chat_template(
    assistant_messages,
    tokenize=False,
    add_generation_prompt=False,
)
assistant_tokens = tokenizer(assistant_text, add_special_tokens=False)
assistant_header_ids = assistant_tokens["input_ids"]
if isinstance(assistant_header_ids, list) and len(assistant_header_ids) > 0 and isinstance(assistant_header_ids[0], list):
    assistant_header_ids = assistant_header_ids[0]

eos_id = tokenizer.eos_token_id or tokenizer.convert_tokens_to_ids("<|end_of_text|>")

# Attach LoRA adapters to the text tower
peft_cfg = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES,
    use_dora=USE_DORA,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_cfg)
model.print_trainable_parameters()

# -----------------------------
# Datasets
# -----------------------------
print(f"Loading dataset: {TRAIN_DATA}, split: {TRAIN_SPLIT}")
train_ds = DoclingJSONDataset(TRAIN_DATA, split=TRAIN_SPLIT, max_items=MAX_TRAIN_SAMPLES)
eval_ds = DoclingJSONDataset(TRAIN_DATA, split=VAL_SPLIT) if VAL_SPLIT else None

print(f"Loaded {len(train_ds)} training samples")
if eval_ds:
    print(f"Loaded {len(eval_ds)} eval samples")

collate = Collator(
    processor=processor,
    tokenizer=tokenizer,
    eos_id=eos_id,
    assistant_header_ids=assistant_header_ids,
    max_seq_len=MAX_SEQ_LEN,
)

# -----------------------------
# Trainer
# -----------------------------
train_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    eval_strategy="no" if eval_ds is None else "steps",
    eval_steps=1000 if eval_ds else None,
    bf16=BF16,
    fp16=not BF16,
    remove_unused_columns=False,  # IMPORTANT for pixel_values to pass through
    # max_seq_length parameter moved to SFTTrainer
    dataset_text_field="",  # We use custom collator, not text field
    packing=False,
)

trainer = SFTTrainer(
    model=model,
    processing_class=processor,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    args=train_args,
    data_collator=collate,
    # max_seq_length handled by our custom collator
)

trainer.train()

# -----------------------------
# Save adapters
# -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.model.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapters"))
processor.save_pretrained(OUTPUT_DIR)
print(f"Saved LoRA adapters to {os.path.join(OUTPUT_DIR, 'lora_adapters')}")

# -----------------------------
# Quick smoke test (greedy)
# -----------------------------
@torch.inference_mode()
def quick_test(sample: Dict[str, Any]):
    model.eval()
    msgs = build_messages(sample["prompt"], "")  # generation prompt (no ground truth)
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text], images=[[sample["image"]]], return_tensors="pt", padding=True
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    decoded = tokenizer.decode(out[0], skip_special_tokens=False)
    print("\n=== GENERATION ===\n", decoded)

# Example (first training sample)
try:
    ex = train_ds[0]
    quick_test(ex)
except Exception as e:
    print("Smoke test skipped:", e)

# -----------------------------
# vLLM serving note (for inference only)
# -----------------------------
print(
    "\n[Note] For vLLM serving, consider: \n"
    "  vllm serve ibm-granite/granite-docling-258M --revision untied --dtype float32\n"
    "(On older GPUs without bf16, float32 avoids the '!!!!' outputs.)\n"
)
